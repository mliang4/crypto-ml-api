"""
Test suite for the ML API
Run with: pytest tests/
"""

import pytest
import sys
from pathlib import Path
from fastapi.testclient import TestClient
import numpy as np
import joblib

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Import the API
try:
    from api_production import app, load_latest_model
    API_AVAILABLE = True
except ImportError:
    API_AVAILABLE = False
    app = None


@pytest.fixture
def client():
    """Create test client"""
    if not API_AVAILABLE:
        pytest.skip("API not available")
    return TestClient(app)


@pytest.fixture
def sample_request():
    """Sample valid prediction request"""
    return {
        "price": 45000.0,
        "price_change": 0.02,
        "price_ma_7": 44500.0,
        "price_ma_30": 43000.0,
        "price_std_7": 1200.0,
        "volume_ma_7": 25000000000.0,
        "price_lag_1": 44000.0,
        "price_lag_3": 43500.0,
        "price_lag_7": 42000.0
    }


class TestHealthEndpoints:
    """Test health and status endpoints"""
    
    def test_root_endpoint(self, client):
        """Test root endpoint returns 200"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
    
    def test_health_check(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "checks" in data
    
    def test_metrics_endpoint(self, client):
        """Test metrics endpoint"""
        response = client.get("/metrics")
        assert response.status_code == 200


class TestPredictionEndpoint:
    """Test prediction functionality"""
    
    def test_predict_valid_request(self, client, sample_request):
        """Test prediction with valid request"""
        response = client.post("/predict", json=sample_request)
        assert response.status_code == 200
        
        data = response.json()
        assert "prediction" in data
        assert "confidence" in data
        assert "message" in data
        assert "model_version" in data
        assert "timestamp" in data
        
        # Validate prediction value
        assert data["prediction"] in [0, 1]
        assert 0 <= data["confidence"] <= 1
    
    def test_predict_invalid_price(self, client, sample_request):
        """Test prediction with invalid price"""
        sample_request["price"] = -1000  # Invalid negative price
        response = client.post("/predict", json=sample_request)
        assert response.status_code == 422  # Validation error
    
    def test_predict_missing_field(self, client, sample_request):
        """Test prediction with missing required field"""
        del sample_request["price_change"]
        response = client.post("/predict", json=sample_request)
        assert response.status_code == 422
    
    def test_predict_invalid_types(self, client, sample_request):
        """Test prediction with invalid data types"""
        sample_request["price"] = "invalid"
        response = client.post("/predict", json=sample_request)
        assert response.status_code == 422
    
    def test_predict_response_time(self, client, sample_request):
        """Test that prediction response time is tracked"""
        response = client.post("/predict", json=sample_request)
        assert response.status_code == 200
        
        data = response.json()
        assert "response_time_ms" in data
        assert data["response_time_ms"] > 0


class TestModelEndpoints:
    """Test model management endpoints"""
    
    def test_list_models(self, client):
        """Test model listing endpoint"""
        response = client.get("/models/list")
        # May return 503 if registry not available
        assert response.status_code in [200, 503]
    
    def test_model_registry(self, client):
        """Test model registry endpoint"""
        response = client.get("/models/registry")
        # May return 503 if registry not available
        assert response.status_code in [200, 503]


class TestInputValidation:
    """Test input validation and edge cases"""
    
    def test_zero_values(self, client, sample_request):
        """Test with zero values where appropriate"""
        sample_request["price_change"] = 0.0
        response = client.post("/predict", json=sample_request)
        assert response.status_code == 200
    
    def test_extreme_values(self, client, sample_request):
        """Test with extreme but valid values"""
        sample_request["price"] = 1000000.0  # Very high price
        sample_request["volume_ma_7"] = 100000000000.0  # Very high volume
        response = client.post("/predict", json=sample_request)
        assert response.status_code == 200
    
    def test_negative_change(self, client, sample_request):
        """Test with negative price change"""
        sample_request["price_change"] = -0.05
        response = client.post("/predict", json=sample_request)
        assert response.status_code == 200


class TestMonitoring:
    """Test monitoring functionality"""
    
    def test_metrics_after_predictions(self, client, sample_request):
        """Test that metrics are updated after predictions"""
        # Make several predictions
        for _ in range(5):
            client.post("/predict", json=sample_request)
        
        # Check metrics
        response = client.get("/metrics")
        assert response.status_code == 200
        
        data = response.json()
        if "total_predictions" in data:
            assert data["total_predictions"] >= 5


# Unit tests for individual functions
class TestModelFunctions:
    """Test model-related functions"""
    
    def test_load_model(self):
        """Test model loading"""
        try:
            model = load_latest_model()
            assert model is not None
        except FileNotFoundError:
            pytest.skip("No model file available for testing")
    
    def test_model_prediction_shape(self):
        """Test that model produces correct output shape"""
        try:
            model = load_latest_model()
            
            # Create sample input
            sample = np.array([[0.02, 44500, 43000, 1200, 2.5e10, 44000, 43500, 42000]])
            
            prediction = model.predict(sample)
            assert prediction.shape == (1,)
            assert prediction[0] in [0, 1]
            
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(sample)
                assert proba.shape == (1, 2)
                assert np.allclose(proba.sum(), 1.0)
        except FileNotFoundError:
            pytest.skip("No model file available for testing")


# Integration tests
class TestIntegration:
    """Integration tests"""
    
    def test_full_prediction_workflow(self, client, sample_request):
        """Test complete prediction workflow"""
        # 1. Check health
        health_response = client.get("/health")
        assert health_response.status_code == 200
        
        # 2. Make prediction
        pred_response = client.post("/predict", json=sample_request)
        assert pred_response.status_code == 200
        
        # 3. Check metrics updated
        metrics_response = client.get("/metrics")
        assert metrics_response.status_code == 200


# Performance tests
class TestPerformance:
    """Performance and load tests"""
    
    def test_concurrent_requests(self, client, sample_request):
        """Test handling multiple concurrent requests"""
        import concurrent.futures
        
        def make_request():
            return client.post("/predict", json=sample_request)
        
        # Make 10 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            responses = [f.result() for f in futures]
        
        # All should succeed
        assert all(r.status_code == 200 for r in responses)
    
    def test_response_time_threshold(self, client, sample_request):
        """Test that response time is within acceptable threshold"""
        import time
        
        start = time.time()
        response = client.post("/predict", json=sample_request)
        end = time.time()
        
        response_time = (end - start) * 1000  # Convert to ms
        
        assert response.status_code == 200
        assert response_time < 1000  # Should respond within 1 second


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
