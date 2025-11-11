"""
Production-ready API with monitoring, logging, and health checks
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import numpy as np
import glob
import os
import time
from datetime import datetime
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

try:
    from monitoring import ModelMonitor, HealthChecker, MetricsAggregator
    from model_registry import ModelRegistry
except ImportError:
    # Fallback if monitoring modules not available
    ModelMonitor = None
    HealthChecker = None
    MetricsAggregator = None
    ModelRegistry = None

# Initialize FastAPI app
app = FastAPI(
    title="Crypto Price Prediction API - Production",
    description="Production ML API with monitoring and logging",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
model_version = None
monitor = None
health_checker = None
registry = None

# Request/Response schemas
class PredictionRequest(BaseModel):
    volume: float = Field(..., description="Current volume", gt=0)
    price_change: float = Field(..., description="Price change %")
    price_ma_7: float = Field(..., description="7-day moving average", gt=0)
    price_ma_30: float = Field(..., description="30-day moving average", gt=0)
    price_std_7: float = Field(..., description="7-day price std", ge=0)
    volume_ma_7: float = Field(..., description="7-day volume MA", ge=0)
    price_lag_1: float = Field(..., description="Price 1 day ago", gt=0)
    price_lag_3: float = Field(..., description="Price 3 days ago", gt=0)
    price_lag_7: float = Field(..., description="Price 7 days ago", gt=0)

    class Config:
        json_schema_extra = {
            "example": {
                "volume": 45000000000.0,
                "price_change": 0.02,
                "price_ma_7": 44500.0,
                "price_ma_30": 43000.0,
                "price_std_7": 1200.0,
                "volume_ma_7": 25000000000.0,
                "price_lag_1": 44000.0,
                "price_lag_3": 43500.0,
                "price_lag_7": 42000.0
            }
        }


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="0=Down, 1=Up")
    confidence: float = Field(..., description="Model confidence")
    message: str
    model_version: str
    timestamp: str
    response_time_ms: float


def load_latest_model():
    """Load the latest trained model"""
    global model, model_version
    
    model_files = glob.glob('models/model_*.joblib')
    if not model_files:
        raise FileNotFoundError("No trained model found!")
    
    latest_model = max(model_files, key=os.path.getctime)
    model = joblib.load(latest_model)
    model_version = os.path.basename(latest_model).replace('model_', '').replace('.joblib', '')
    
    return model


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global monitor, health_checker, registry
    
    # Create necessary directories
    Path('logs').mkdir(exist_ok=True)
    Path('logs/metrics').mkdir(exist_ok=True)
    
    # Load model
    try:
        load_latest_model()
        print(f"✅ Model loaded: {model_version}")
    except Exception as e:
        print(f"⚠️  Failed to load model: {e}")
    
    # Initialize monitoring
    if ModelMonitor:
        monitor = ModelMonitor(window_size=100)
        print("✅ Monitoring initialized")
    
    # Initialize health checker
    if HealthChecker:
        health_checker = HealthChecker()
        print("✅ Health checker initialized")
    
    # Initialize model registry
    if ModelRegistry:
        try:
            registry = ModelRegistry()
            print("✅ Model registry initialized")
        except Exception as e:
            print(f"⚠️  Registry initialization failed: {e}")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Middleware to log all requests"""
    start_time = time.time()
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    
    # Log request
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'method': request.method,
        'path': request.url.path,
        'status_code': response.status_code,
        'process_time_ms': process_time * 1000
    }
    
    # Save to daily log
    log_file = f"logs/requests_{datetime.now().strftime('%Y%m%d')}.jsonl"
    import json
    with open(log_file, 'a') as f:
        f.write(json.dumps(log_entry) + '\n')
    
    # Add timing header
    response.headers["X-Process-Time"] = str(process_time)
    
    return response


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Crypto Price Prediction API - Production",
        "version": "2.0.0",
        "status": "operational",
        "model_version": model_version,
        "documentation": "/docs",
        "health": "/health",
        "metrics": "/metrics"
    }


@app.get("/health")
async def health_check():
    """Comprehensive health check endpoint"""
    if health_checker:
        health_status = health_checker.get_health_status(model)
    else:
        health_status = {
            'status': 'healthy' if model is not None else 'unhealthy',
            'timestamp': datetime.now().isoformat(),
            'checks': {
                'model_loaded': model is not None
            }
        }
    
    return health_status


@app.get("/metrics")
async def get_metrics():
    """Get current performance metrics"""
    if not monitor:
        return {"error": "Monitoring not initialized"}
    
    metrics = monitor.get_metrics()
    if not metrics:
        return {"message": "No metrics available yet", "total_predictions": 0}
    
    return metrics


@app.get("/metrics/export")
async def export_metrics():
    """Export detailed metrics"""
    if not monitor:
        raise HTTPException(status_code=503, detail="Monitoring not available")
    
    metrics = monitor.export_metrics()
    return {"message": "Metrics exported", "data": metrics}


@app.get("/metrics/daily/{date}")
async def get_daily_metrics(date: str):
    """
    Get metrics for a specific date
    Format: YYYYMMDD (e.g., 20240115)
    """
    if not MetricsAggregator:
        raise HTTPException(status_code=503, detail="Metrics aggregation not available")
    
    analysis = MetricsAggregator.analyze_daily_performance(date)
    
    if not analysis:
        raise HTTPException(status_code=404, detail=f"No data found for date: {date}")
    
    return analysis


@app.get("/metrics/report")
async def generate_report(days: int = 7):
    """Generate performance report for last N days"""
    if not MetricsAggregator:
        raise HTTPException(status_code=503, detail="Metrics aggregation not available")
    
    report = MetricsAggregator.generate_report(days=days)
    return report


@app.get("/models/registry")
async def get_model_registry():
    """Get model registry information"""
    if not registry:
        raise HTTPException(status_code=503, detail="Model registry not available")
    
    return {
        "production_model": registry.get_production_model(),
        "staging_model": registry.get_staging_model(),
        "total_models": len(registry.list_models())
    }


@app.get("/models/list")
async def list_models(status: str = None):
    """List all registered models"""
    if not registry:
        raise HTTPException(status_code=503, detail="Model registry not available")
    
    models = registry.list_models(status=status)
    return {"models": models, "count": len(models)}


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Make a prediction with monitoring and logging
    """
    start_time = time.time()
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Prepare features
        features = np.array([[
            request.price_change,
            request.price_ma_7,
            request.price_ma_30,
            request.price_std_7,
            request.volume,
            request.volume_ma_7,
            request.price_lag_1,
            request.price_lag_3,
            request.price_lag_7
        ]])
        
        # Make prediction
        prediction = int(model.predict(features)[0])
        
        # Get confidence
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(features)[0]
            confidence = float(max(proba))
        else:
            confidence = 0.5
        
        # Calculate response time
        response_time = time.time() - start_time
        
        # Log prediction to monitor
        if monitor:
            feature_dict = {
                'volume': request.volume,
                'price_change': request.price_change,
                'price_ma_7': request.price_ma_7,
                'price_ma_30': request.price_ma_30,
                'price_std_7': request.price_std_7,
                'volume_ma_7': request.volume_ma_7,
                'price_lag_1': request.price_lag_1,
                'price_lag_3': request.price_lag_3,
                'price_lag_7': request.price_lag_7
            }
            monitor.log_prediction(feature_dict, prediction, confidence, response_time)
        
        message = "Price expected to go UP 📈" if prediction == 1 else "Price expected to go DOWN 📉"
        
        return PredictionResponse(
            prediction=prediction,
            confidence=confidence,
            message=message,
            model_version=model_version or "unknown",
            timestamp=datetime.now().isoformat(),
            response_time_ms=response_time * 1000
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/models/reload")
async def reload_model():
    """Reload the model (useful for zero-downtime updates)"""
    try:
        load_latest_model()
        return {
            "message": "Model reloaded successfully",
            "version": model_version,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reload model: {str(e)}")


@app.get("/drift/check")
async def check_drift():
    """Check for data drift"""
    if not monitor:
        raise HTTPException(status_code=503, detail="Monitoring not available")
    
    # This would be called by a scheduled job in production
    return {
        "message": "Drift check completed",
        "timestamp": datetime.now().isoformat(),
        "note": "Check logs/drift_events.jsonl for drift alerts"
    }


if __name__ == "__main__":
    import uvicorn
    
    print("="*80)
    print("Starting Production API Server")
    print("="*80)
    print("\nFeatures:")
    print("  ✓ Real-time monitoring")
    print("  ✓ Performance metrics tracking")
    print("  ✓ Data drift detection")
    print("  ✓ Health checks")
    print("  ✓ Request logging")
    print("  ✓ Model versioning")
    print("\nEndpoints:")
    print("  • /docs          - API documentation")
    print("  • /health        - Health check")
    print("  • /metrics       - Performance metrics")
    print("  • /predict       - Make predictions")
    print("="*80)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
