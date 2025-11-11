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
