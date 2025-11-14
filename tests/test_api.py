import requests

API_URL = "https://crypto-ml-api-686x.onrender.com"

def test_health():
    response = requests.get(f"{API_URL}/health")
    print("Health Check:", response.json())

def test_prediction():
    payload = {
        "price_change": -200000,
        "price_ma_7": 44500.0,
        "price_ma_30": 43000.0,
        "price_std_7": 1200.0,
        "volume": 300000.0,
        "volume_ma_7": 2500.0,
        "price_lag_1": 44000.0,
        "price_lag_3": 43500.0,
        "price_lag_7": 42000.0
    }
    
    response = requests.post(f"{API_URL}/predict", json=payload)
    result = response.json()
    
    print("\nPrediction Result:")
    print(f"  Prediction: {'UP' if result['prediction'] == 1 else 'DOWN'}")
    print(f"  Confidence: {result['confidence']:.2%}")
    print(f"  Message: {result['message']}")

if __name__ == "__main__":
    test_health()
    test_prediction()