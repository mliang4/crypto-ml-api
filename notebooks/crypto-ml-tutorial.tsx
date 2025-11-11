# Train and Deploy ML Model for Crypto Price Prediction

A complete, practical tutorial designed for ML engineers preparing for deployment interviews. Build a real ML system from scratch in one week.

---

## Overview

**Timeline:** Complete in 5-7 days  
**Interview Ready:** All essential ML deployment skills  
**Tech Stack:** Python, FastAPI, Docker, AWS/Render  

### What You'll Learn
- Train ML models with proper experiment tracking
- Build REST APIs to serve predictions
- Containerize with Docker
- Deploy to cloud (AWS/Render)
- Implement monitoring and versioning

---

## Setup and Prerequisites

### Required Software
- **Python 3.10+** - Check with `python --version`
- **Docker Desktop** - Download from docker.com
- **Git** - For version control
- **Code Editor** - VS Code recommended

### Project Setup

```bash
# Create project directory
mkdir crypto-ml-api
cd crypto-ml-api

# Create virtual environment
conda create -n crypto-ml python=3.10 -y
conda activate crypto-ml

# Install core dependencies
pip install pandas numpy scikit-learn xgboost
pip install fastapi uvicorn requests python-dotenv
pip install matplotlib seaborn joblib

# Create project structure
mkdir -p data models notebooks src tests
echo. > src/__init__.py
echo. > src/data.py 
echo. > src/train.py 
echo. > src/api.py
echo. > requirements.txt 
echo. > README.md
```

### Save Requirements

Create `requirements.txt`:

```
pip freeze > requirements.txt
pip freeze | findstr /V "file:///" > requirements.txt
```

---

## Part 1: Data and Model Training (Days 1-3)

### Learning Goals
- Fetch and process cryptocurrency data
- Engineer features for price prediction
- Train and evaluate multiple models
- Save the best model for deployment

### Step 1: Data Collection

Create `src/data.py`:

```python
import pandas as pd
import requests
from datetime import datetime, timedelta

def fetch_crypto_data(symbol='BTC', days=365):
    """Fetch historical crypto data from CoinGecko API (free)"""
    url = f'https://api.coingecko.com/api/v3/coins/{symbol.lower()}/market_chart'
    params = {
        'vs_currency': 'usd',
        'days': days,
        'interval': 'daily'
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    # Convert to DataFrame
    df = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['volume'] = [v[1] for v in data['total_volumes']]
    
    return df

def create_features(df):
    """Engineer features for prediction"""
    df = df.copy()
    
    # Price-based features
    df['price_change'] = df['price'].pct_change()
    df['price_ma_7'] = df['price'].rolling(window=7).mean()
    df['price_ma_30'] = df['price'].rolling(window=30).mean()
    df['price_std_7'] = df['price'].rolling(window=7).std()
    
    # Volume features
    df['volume_ma_7'] = df['volume'].rolling(window=7).mean()
    
    # Lag features
    for lag in [1, 3, 7]:
        df[f'price_lag_{lag}'] = df['price'].shift(lag)
    
    # Target: next day price movement (1 = up, 0 = down)
    df['target'] = (df['price'].shift(-1) > df['price']).astype(int)
    
    # Drop NaN values
    df = df.dropna()
    
    return df

if __name__ == '__main__':
    # Test the functions
    print("Fetching Bitcoin data...")
    df = fetch_crypto_data('bitcoin', days=365)
    print(f"Downloaded {len(df)} days of data")
    
    print("\nCreating features...")
    df_features = create_features(df)
    print(f"Created {len(df_features.columns)} features")
    
    # Save to disk
    df_features.to_csv('data/bitcoin_features.csv', index=False)
    print("\nData saved to data/bitcoin_features.csv")
```

**Run it:** `python src/data.py`

### Step 2: Model Training

Create `src/train.py`:

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
import joblib
import json
from datetime import datetime

def load_data(filepath='data/bitcoin_features.csv'):
    """Load preprocessed data"""
    df = pd.read_csv(filepath)
    
    # Feature columns (exclude timestamp and target)
    feature_cols = [col for col in df.columns 
                   if col not in ['timestamp', 'target', 'price']]
    
    X = df[feature_cols]
    y = df['target']
    
    return X, y, feature_cols

def train_models(X, y):
    """Train multiple models and return the best one"""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )
    
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        print(classification_report(y_test, y_pred))
    
    # Select best model
    best_name = max(results.keys(), key=lambda k: results[k]['accuracy'])
    best_model = results[best_name]['model']
    
    print(f"\nBest model: {best_name}")
    
    return best_model, results, best_name

def save_model(model, feature_cols, metrics, model_name):
    """Save model and metadata"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save model
    model_path = f'models/model_{timestamp}.joblib'
    joblib.dump(model, model_path)
    
    # Save metadata
    metadata = {
        'model_name': model_name,
        'timestamp': timestamp,
        'features': feature_cols,
        'metrics': {k: v for k, v in metrics.items() if k != 'model'},
        'model_path': model_path
    }
    
    with open(f'models/metadata_{timestamp}.json', 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    print(f"\nModel saved to {model_path}")
    return model_path

if __name__ == '__main__':
    print("Loading data...")
    X, y, feature_cols = load_data()
    
    print(f"Dataset shape: {X.shape}")
    print(f"Features: {feature_cols}")
    
    print("\nTraining models...")
    best_model, results, model_name = train_models(X, y)
    
    print("\nSaving model...")
    model_path = save_model(
        best_model, 
        feature_cols, 
        results[model_name],
        model_name
    )
    
    print("\nTraining complete!")
```

**Run it:** `python src/train.py`

---

## Part 2: Build REST API (Days 4-5)

### Learning Goals
- Build a FastAPI endpoint for predictions
- Handle request validation with Pydantic
- Implement health checks and monitoring
- Containerize with Docker

### Step 1: Create API

Create `src/api.py`:

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import glob
import os

# Initialize FastAPI app
app = FastAPI(
    title="Crypto Price Prediction API",
    description="Predict crypto price movement (up/down)",
    version="1.0.0"
)

# Load the latest model
def load_latest_model():
    model_files = glob.glob('models/model_*.joblib')
    if not model_files:
        raise FileNotFoundError("No trained model found!")
    latest_model = max(model_files, key=os.path.getctime)
    return joblib.load(latest_model)

model = load_latest_model()

# Request schema
class PredictionRequest(BaseModel):
    price: float = Field(..., description="Current price")
    price_change: float = Field(..., description="Price change %")
    price_ma_7: float = Field(..., description="7-day moving average")
    price_ma_30: float = Field(..., description="30-day moving average")
    price_std_7: float = Field(..., description="7-day price std")
    volume: float = Field(..., description="Current volume")
    volume_ma_7: float = Field(..., description="7-day volume MA")
    price_lag_1: float = Field(..., description="Price 1 day ago")
    price_lag_3: float = Field(..., description="Price 3 days ago")
    price_lag_7: float = Field(..., description="Price 7 days ago")

    class Config:
        json_schema_extra = {
            "example": {
                "price": 45000.0,
                "price_change": 0.02,
                "price_ma_7": 44500.0,
                "price_ma_30": 43000.0,
                "price_std_7": 1200.0,
                "volume": 30000000000.0,
                "volume_ma_7": 25000000000.0,
                "price_lag_1": 44000.0,
                "price_lag_3": 43500.0,
                "price_lag_7": 42000.0
            }
        }

# Response schema
class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="0=Down, 1=Up")
    confidence: float = Field(..., description="Model confidence")
    message: str

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Crypto Price Prediction API",
        "status": "healthy",
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict crypto price movement
    Returns 1 for UP, 0 for DOWN
    """
    try:
        # Prepare features in correct order
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
        
        # Get confidence (probability)
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(features)[0]
            confidence = float(max(proba))
        else:
            confidence = 0.0
        
        message = "Price expected to go UP" if prediction == 1 else "Price expected to go DOWN"
        
        return PredictionResponse(
            prediction=prediction,
            confidence=confidence,
            message=message
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**Run locally:** `python src/api.py`  
**Test at:** http://localhost:8000/docs

### Step 2: Dockerize

Create `Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY models/ ./models/

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Build and run:**

```bash
docker build -t crypto-ml-api .
docker run -p 8000:8000 crypto-ml-api
curl http://localhost:8000/health
```

---

## Part 3: Deploy to Cloud (Days 6-7)

### Option A: Deploy to Render (Easiest)

**1. Push to GitHub:**

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin <your-repo-url>
git push -u origin main
```

**2. Deploy on Render:**
- Go to render.com and sign up
- Click "New +" then "Web Service"
- Connect your GitHub repo
- Configure: Name (crypto-ml-api), Environment (Docker), Instance Type (Free)
- Click "Create Web Service"
- Wait 5-10 minutes
- Get your URL

### Option B: Deploy to AWS EC2

```bash
# Launch EC2 instance (Ubuntu 22.04, t2.micro)
# SSH into instance
ssh -i your-key.pem ubuntu@your-ec2-ip

# Install Docker
sudo apt update
sudo apt install -y docker.io
sudo systemctl start docker
sudo usermod -aG docker ubuntu

# Clone and run
git clone <your-repo-url>
cd crypto-ml-api
docker build -t crypto-ml-api .
docker run -d -p 80:8000 --name api crypto-ml-api
```

### Test Deployment

Create `test_api.py`:

```python
import requests

API_URL = "https://crypto-ml-api.onrender.com"

def test_health():
    response = requests.get(f"{API_URL}/health")
    print("Health Check:", response.json())

def test_prediction():
    payload = {
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
    
    response = requests.post(f"{API_URL}/predict", json=payload)
    result = response.json()
    
    print("\nPrediction Result:")
    print(f"  Prediction: {'UP' if result['prediction'] == 1 else 'DOWN'}")
    print(f"  Confidence: {result['confidence']:.2%}")
    print(f"  Message: {result['message']}")

if __name__ == "__main__":
    test_health()
    test_prediction()
```

---

## Congratulations

### What You Built
- Complete ML pipeline (data to training to deployment)
- Production-ready REST API with FastAPI
- Dockerized application
- Cloud-deployed service

### Interview Talking Points
- Feature engineering for time-series data
- Model selection and evaluation
- REST API design patterns
- Containerization benefits
- Cloud deployment strategies
- Monitoring and health checks

### Enhancements
- Add model versioning with MLflow
- Implement A/B testing
- Add monitoring with Prometheus
- Set up CI/CD with GitHub Actions
- Add authentication (API keys)
- Implement rate limiting

### Common Interview Questions You Can Answer

1. **How do you deploy an ML model to production?**
   - Talk through your pipeline: training to API to Docker to cloud

2. **What is the difference between batch and real-time inference?**
   - Your API does real-time, explain trade-offs

3. **How do you monitor model performance?**
   - Discuss health checks, logging, metrics tracking

4. **Why use Docker for ML deployment?**
   - Consistency, portability, dependency management

5. **How do you handle model versioning?**
   - Show your timestamp-based approach, mention MLflow

6. **What is your deployment pipeline?**
   - Walk through: code to Docker to cloud platform

Good luck with your interview!