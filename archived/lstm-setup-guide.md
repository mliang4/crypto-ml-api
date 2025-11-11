# LSTM Model Setup Guide

## Installation

### 1. Install PyTorch

```bash
# Activate your environment
conda activate crypto-ml

# Install PyTorch (CPU version)
pip install torch torchvision torchaudio

# Or for GPU (if you have CUDA)
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. Install Additional Dependencies

```bash
pip install matplotlib
```

### 3. Update requirements.txt

Add these lines to your `requirements.txt`:

```
torch>=2.0.0
matplotlib>=3.7.0
```

## Usage

### Step 1: Train LSTM Model

```bash
# Make sure you have your data ready
python src/data.py  # If not already done

# Train LSTM model
python src/train_lstm.py
```

**Expected Output:**
- Training progress with loss and accuracy per epoch
- Final validation accuracy (should be > 60%, hopefully 65-75%)
- Saved model files in `models/` directory
- Training plot showing learning curves

**Training Time:**
- CPU: 2-5 minutes
- GPU: 30-60 seconds

### Step 2: Test the LSTM API

```bash
# Start the API
python src/api_lstm.py
```

Then open browser: http://localhost:8000/docs

### Step 3: Make Predictions

#### Option A: Using the /predict endpoint (recommended)

You need to provide a sequence of 30 time steps. Here's a Python example:

```python
import requests
import json

url = "http://localhost:8000/predict"

# Create a sequence of 30 data points
# In practice, you'd get this from your recent data
sequence = []
for i in range(30):
    sequence.append({
        "price_change": 0.02 + (i * 0.001),
        "price_ma_7": 44500.0,
        "price_ma_30": 43000.0,
        "price_std_7": 1200.0,
        "volume_ma_7": 25000000000.0,
        "price_lag_1": 44000.0,
        "price_lag_3": 43500.0,
        "price_lag_7": 42000.0
    })

payload = {"sequence": sequence}

response = requests.post(url, json=payload)
print(json.dumps(response.json(), indent=2))
```

#### Option B: Using /predict-simple (quick test)

```bash
curl -X POST "http://localhost:8000/predict-simple?price_change=0.02&price_ma_7=44500&price_ma_30=43000&price_std_7=1200&volume_ma_7=25000000000&price_lag_1=44000&price_lag_3=43500&price_lag_7=42000"
```

## Understanding the Results

### What Changed from sklearn models?

**1. Input Format:**
- **Old (sklearn):** Single data point
- **New (LSTM):** Sequence of 30 data points
- **Why:** LSTM learns patterns over time, not just from a single moment

**2. Performance:**
- **Old:** ~50% accuracy (basically random)
- **Expected LSTM:** 60-75% accuracy
- **Why:** LSTM captures temporal dependencies that sklearn models miss

**3. Model Complexity:**
- **Old:** Few hundred parameters
- **LSTM:** 50,000+ parameters
- **Why:** Deep learning can learn more complex patterns

### Why LSTM Should Perform Better

1. **Temporal patterns:** Sees trends over 30 days, not just current values
2. **Sequential learning:** Understands that today's price depends on yesterday's
3. **Long-term dependencies:** Can remember patterns from weeks ago
4. **Non-linear relationships:** Captures complex interactions between features

## Model Architecture

```
Input: [batch_size, 30, 8]  # 30 time steps, 8 features
  ↓
LSTM Layer 1: 64 hidden units
  ↓
LSTM Layer 2: 64 hidden units
  ↓
Dropout: 30%
  ↓
Fully Connected: 64 → 32
  ↓
ReLU + Dropout
  ↓
Fully Connected: 32 → 2 (Up/Down)
  ↓
Softmax (probabilities)
```

**Total Parameters:** ~50,000

## Hyperparameter Tuning

Want to improve performance? Try adjusting these in `train_lstm.py`:

```python
model, scaler, history, feature_cols, sequence_length = train_lstm_model(
    df,
    sequence_length=30,    # Try: 20, 40, 60
    epochs=50,             # Try: 100 (with early stopping)
    batch_size=32,         # Try: 16, 64
    learning_rate=0.001    # Try: 0.0001, 0.01
)
```

**In the model architecture (CryptoLSTM class):**

```python
model = CryptoLSTM(
    input_size=n_features,
    hidden_size=64,        # Try: 32, 128, 256
    num_layers=2,          # Try: 1, 3, 4
    dropout=0.3            # Try: 0.2, 0.4, 0.5
)
```

## Troubleshooting

### Issue: "No LSTM model found!"
**Solution:** Run `python src/train_lstm.py` first

### Issue: Low accuracy (still around 50%)
**Possible causes:**
1. Crypto prices are highly random (this is expected)
2. Need more features (technical indicators, sentiment)
3. Need more data (try more than 1 year)
4. Need different sequence length
5. Class imbalance (add class weights)

**Try this fix for class imbalance:**

```python
# In train_lstm.py, replace criterion with:
# Calculate class weights
class_counts = np.bincount(y_train)
class_weights = torch.FloatTensor([1.0/c for c in class_counts]).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
```

### Issue: Training is slow
**Solutions:**
- Reduce sequence_length (30 → 20)
- Reduce batch_size (32 → 16)
- Reduce hidden_size (64 → 32)
- Use GPU if available

### Issue: Overfitting (train acc high, val acc low)
**Solutions:**
- Increase dropout (0.3 → 0.5)
- Add L2 regularization
- Use less complex model (fewer layers/units)
- Get more data

## Deploy LSTM Model

### Update Dockerfile

Replace your `Dockerfile` with:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install PyTorch (CPU version for smaller image)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY src/ ./src/
COPY models/ ./models/

EXPOSE 8000

# Use LSTM API
CMD ["uvicorn", "src.api_lstm:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Build and run:

```bash
docker build -t crypto-ml-lstm .
docker run -p 8000:8000 crypto-ml-lstm
```

## Interview Talking Points

### Why LSTM over sklearn?

**1. Temporal Nature:**
- "Crypto prices are time-series data. sklearn models treat each sample independently, but LSTM considers the sequence and temporal relationships."

**2. Pattern Recognition:**
- "LSTM can learn that certain price patterns over multiple days predict future movements, while sklearn only sees a snapshot."

**3. Memory:**
- "The LSTM's hidden state acts as memory, allowing it to remember important information from earlier in the sequence."

**4. Non-linearity:**
- "Deep learning naturally captures complex non-linear relationships without manual feature engineering."

### Architecture Decisions

**Why 2 layers?**
- "Two layers provide good capacity without overfitting. One layer is often too simple, three+ risks overfitting with limited data."

**Why dropout?**
- "Prevents overfitting by randomly dropping neurons during training. Essential for deep learning with financial data."

**Why sequence_length=30?**
- "One month of daily data captures monthly patterns while keeping computational cost reasonable."

### Trade-offs

**LSTM Advantages:**
- Better for sequential data
- Learns temporal patterns
- More flexible

**LSTM Disadvantages:**
- Slower inference (needs full sequence)
- More complex to deploy
- Harder to interpret
- Requires more data

## Next Steps

### 1. Add More Features
```python
# In data.py, add technical indicators:
df['RSI'] = calculate_rsi(df['price'])
df['MACD'] = calculate_macd(df['price'])
df['Bollinger_upper'] = calculate_bollinger_bands(df['price'])
```

### 2. Try Bidirectional LSTM
```python
# In CryptoLSTM class:
self.lstm = nn.LSTM(..., bidirectional=True)
# Then update fc1 input size: hidden_size * 2
```

### 3. Add Attention Mechanism
```python
# Add attention layer after LSTM
self.attention = nn.MultiheadAttention(hidden_size, num_heads=4)
```

### 4. Implement Model Ensemble
```python
# Train multiple models and average predictions
predictions = []
for model in models:
    pred = model(sequence)
    predictions.append(pred)
final_pred = torch.mean(torch.stack(predictions), dim=0)
```

Good luck with your interview! 🚀
