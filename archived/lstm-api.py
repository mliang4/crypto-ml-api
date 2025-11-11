from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import torch
import torch.nn as nn
import joblib
import numpy as np
from typing import List
import glob
import os
import json

# Initialize FastAPI app
app = FastAPI(
    title="Crypto Price Prediction API - LSTM",
    description="Predict crypto price movement using LSTM (up/down)",
    version="2.0.0"
)


class CryptoLSTM(nn.Module):
    """LSTM model for crypto price prediction (same as in training)"""
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
        super(CryptoLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 2)
        
    def forward(self, x):
        lstm_out, (hidden, cell) = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        out = self.fc1(last_output)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out


def load_latest_lstm_model():
    """Load the latest LSTM model, scaler, and metadata"""
    # Find latest model files
    model_files = glob.glob('models/lstm_model_*.pth')
    if not model_files:
        raise FileNotFoundError("No LSTM model found! Train one first.")
    
    latest_model_file = max(model_files, key=os.path.getctime)
    timestamp = latest_model_file.split('_')[-1].replace('.pth', '')
    
    # Load metadata
    metadata_file = f'models/lstm_metadata_{timestamp}.json'
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    # Load scaler
    scaler_file = f'models/lstm_scaler_{timestamp}.joblib'
    scaler = joblib.load(scaler_file)
    
    # Load model
    checkpoint = torch.load(latest_model_file, map_location=torch.device('cpu'))
    arch = checkpoint['model_architecture']
    
    model = CryptoLSTM(
        input_size=arch['input_size'],
        hidden_size=arch['hidden_size'],
        num_layers=arch['num_layers'],
        dropout=arch['dropout']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded LSTM model from {latest_model_file}")
    print(f"Model validation accuracy: {metadata['metrics']['best_val_acc']:.4f}")
    
    return model, scaler, metadata


# Load model at startup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, scaler, metadata = load_latest_lstm_model()
model = model.to(device)
sequence_length = metadata['sequence_length']
feature_names = metadata['features']


# Request schema - now requires a sequence of data
class SequenceDataPoint(BaseModel):
    """Single time step of data"""
    price_change: float
    price_ma_7: float
    price_ma_30: float
    price_std_7: float
    volume_ma_7: float
    price_lag_1: float
    price_lag_3: float
    price_lag_7: float


class PredictionRequest(BaseModel):
    """Request containing a sequence of time steps"""
    sequence: List[SequenceDataPoint] = Field(
        ..., 
        description=f"Sequence of {sequence_length} time steps",
        min_length=1
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "sequence": [
                    {
                        "price_change": 0.02,
                        "price_ma_7": 44500.0,
                        "price_ma_30": 43000.0,
                        "price_std_7": 1200.0,
                        "volume_ma_7": 25000000000.0,
                        "price_lag_1": 44000.0,
                        "price_lag_3": 43500.0,
                        "price_lag_7": 42000.0
                    }
                    # ... repeat for sequence_length time steps
                ]
            }
        }


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="0=Down, 1=Up")
    confidence: float = Field(..., description="Model confidence (probability)")
    probabilities: dict = Field(..., description="Probabilities for each class")
    message: str
    sequence_length_received: int
    sequence_length_expected: int


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Crypto Price Prediction API - LSTM",
        "status": "healthy",
        "model": "LSTM",
        "framework": "PyTorch",
        "sequence_length": sequence_length,
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_type": "LSTM",
        "device": str(device),
        "validation_accuracy": metadata['metrics']['best_val_acc']
    }


@app.get("/model-info")
async def model_info():
    """Get detailed model information"""
    return {
        "model_name": metadata['model_name'],
        "timestamp": metadata['timestamp'],
        "framework": metadata['framework'],
        "sequence_length": sequence_length,
        "features": feature_names,
        "metrics": metadata['metrics'],
        "total_parameters": sum(p.numel() for p in model.parameters())
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict crypto price movement using LSTM
    Requires a sequence of historical data points
    Returns 1 for UP, 0 for DOWN
    """
    try:
        # Validate sequence length
        if len(request.sequence) != sequence_length:
            # If user provides fewer data points, we can pad or reject
            if len(request.sequence) < sequence_length:
                raise HTTPException(
                    status_code=400,
                    detail=f"Sequence too short. Expected {sequence_length} time steps, got {len(request.sequence)}"
                )
            # If too long, take the last N points
            request.sequence = request.sequence[-sequence_length:]
        
        # Convert sequence to numpy array
        sequence_data = []
        for point in request.sequence:
            sequence_data.append([
                point.price_change,
                point.price_ma_7,
                point.price_ma_30,
                point.price_std_7,
                point.volume_ma_7,
                point.price_lag_1,
                point.price_lag_3,
                point.price_lag_7
            ])
        
        sequence_array = np.array(sequence_data)
        
        # Scale the sequence
        sequence_scaled = scaler.transform(sequence_array)
        
        # Convert to tensor and add batch dimension
        sequence_tensor = torch.FloatTensor(sequence_scaled).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(sequence_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][prediction].item()
            
            probs_dict = {
                "down": float(probabilities[0][0]),
                "up": float(probabilities[0][1])
            }
        
        message = "Price expected to go UP 📈" if prediction == 1 else "Price expected to go DOWN 📉"
        
        return PredictionResponse(
            prediction=prediction,
            confidence=confidence,
            probabilities=probs_dict,
            message=message,
            sequence_length_received=len(request.sequence),
            sequence_length_expected=sequence_length
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict-simple")
async def predict_simple(
    price_change: float,
    price_ma_7: float,
    price_ma_30: float,
    price_std_7: float,
    volume_ma_7: float,
    price_lag_1: float,
    price_lag_3: float,
    price_lag_7: float
):
    """
    Simplified prediction endpoint that repeats single data point
    WARNING: This is not ideal for LSTM but provided for convenience
    For best results, use /predict with actual historical sequence
    """
    # Create a sequence by repeating the same values
    # This is NOT recommended but allows single-point inference
    data_point = SequenceDataPoint(
        price_change=price_change,
        price_ma_7=price_ma_7,
        price_ma_30=price_ma_30,
        price_std_7=price_std_7,
        volume_ma_7=volume_ma_7,
        price_lag_1=price_lag_1,
        price_lag_3=price_lag_3,
        price_lag_7=price_lag_7
    )
    
    # Repeat to create sequence
    sequence = [data_point] * sequence_length
    request = PredictionRequest(sequence=sequence)
    
    result = await predict(request)
    result.message += " (Warning: Using repeated values, not a real sequence)"
    
    return result


if __name__ == "__main__":
    import uvicorn
    print(f"Starting LSTM API...")
    print(f"Model expects sequences of length: {sequence_length}")
    print(f"Features: {feature_names}")
    uvicorn.run(app, host="0.0.0.0", port=8000)
