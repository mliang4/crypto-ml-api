import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
from sklearn.utils.class_weight import compute_class_weight
import json
from datetime import datetime
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class CryptoDataset(Dataset):
    """PyTorch Dataset for crypto time series"""
    def __init__(self, sequences, labels):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


class CryptoLSTM(nn.Module):
    """LSTM model for crypto price prediction"""
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
        super(CryptoLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 2)  # Binary classification
        
    def forward(self, x):
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use the last output
        last_output = lstm_out[:, -1, :]
        
        # Fully connected layers
        out = self.fc1(last_output)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out


def create_sequences(df, sequence_length=30):
    """
    Create sequences for LSTM training
    
    Args:
        df: DataFrame with features and target
        sequence_length: Number of time steps to look back
    
    Returns:
        sequences: numpy array of shape (n_samples, sequence_length, n_features)
        labels: numpy array of shape (n_samples,)
    """
    # Feature columns (exclude timestamp and target)
    feature_cols = [col for col in df.columns 
                   if col not in ['timestamp', 'target', 'price']]
    
    sequences = []
    labels = []
    
    # Create sequences
    for i in range(sequence_length, len(df)):
        # Get sequence of features
        seq = df[feature_cols].iloc[i-sequence_length:i].values
        sequences.append(seq)
        
        # Get label (target at time i)
        label = df['target'].iloc[i]
        labels.append(label)
    
    return np.array(sequences), np.array(labels), feature_cols


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for sequences, labels in train_loader:
        sequences = sequences.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total
    return avg_loss, accuracy


def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for sequences, labels in val_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            total_loss += loss.item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total
    return avg_loss, accuracy, all_preds, all_labels


def train_lstm_model(df, sequence_length=30, epochs=50, batch_size=32, learning_rate=0.001):
    """
    Train LSTM model
    
    Args:
        df: DataFrame with features
        sequence_length: Number of time steps to look back
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
    
    Returns:
        model: Trained PyTorch model
        scaler: Fitted StandardScaler
        history: Training history
    """
    print("Creating sequences...")
    sequences, labels, feature_cols = create_sequences(df, sequence_length)
    print(f"Created {len(sequences)} sequences with shape {sequences.shape}")

    # Ensure labels are integer type (CrossEntropy expects integer class labels)
    labels = labels.astype(np.int64)

    # Split data first (time-series split - no shuffle) to avoid data leakage when scaling
    split_idx = int(len(sequences) * 0.8)
    X_train = sequences[:split_idx]
    y_train = labels[:split_idx]
    X_val = sequences[split_idx:]
    y_val = labels[split_idx:]

    print("\nScaling features (fit only on training set)...")
    scaler = StandardScaler()
    # Fit scaler on training set only
    n_train, n_timesteps, n_features = X_train.shape
    X_train_reshaped = X_train.reshape(-1, n_features)
    X_train_scaled = scaler.fit_transform(X_train_reshaped)
    X_train = X_train_scaled.reshape(n_train, n_timesteps, n_features)

    # Transform validation set using the same scaler
    n_val = X_val.shape[0]
    if n_val > 0:
        X_val_reshaped = X_val.reshape(-1, n_features)
        X_val_scaled = scaler.transform(X_val_reshaped)
        X_val = X_val_scaled.reshape(n_val, n_timesteps, n_features)
    
    print(f"\nTrain size: {len(X_train)}, Validation size: {len(X_val)}")
    # safe bincount prints
    print(f"Class distribution - Train: {np.bincount(y_train) if len(y_train)>0 else 'empty'}, Val: {np.bincount(y_val) if len(y_val)>0 else 'empty'}")
    
    # Create datasets and dataloaders
    train_dataset = CryptoDataset(X_train, y_train)
    val_dataset = CryptoDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    model = CryptoLSTM(
        input_size=n_features,
        hidden_size=64,
        num_layers=2,
        dropout=0.3
    ).to(device)
    
    print(f"\nModel architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # initialize best model state to current model to ensure variable exists
    best_val_acc = 0
    best_model_state = model.state_dict().copy()
    patience = 10
    patience_counter = 0
    
    print("\nStarting training...")
    print("=" * 80)
    
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_preds, val_labels = validate(model, val_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Save best model
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Final evaluation
    print("\n" + "=" * 80)
    print("Final Evaluation on Validation Set:")
    val_loss, val_acc, val_preds, val_labels = validate(model, val_loader, criterion, device)
    print(f"Validation Accuracy: {val_acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(val_labels, val_preds, target_names=['Down', 'Up']))
    
    return model, scaler, history, feature_cols, sequence_length


def save_pytorch_model(model, scaler, history, feature_cols, sequence_length, metrics):
    """Save PyTorch model and metadata"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save model state dict
    model_path = f'models/lstm_model_{timestamp}.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_architecture': {
            'input_size': len(feature_cols),
            'hidden_size': 64,
            'num_layers': 2,
            'dropout': 0.3
        }
    }, model_path)
    
    # Save scaler
    scaler_path = f'models/lstm_scaler_{timestamp}.joblib'
    joblib.dump(scaler, scaler_path)
    
    # Save metadata
    metadata = {
        'model_name': 'LSTM',
        'timestamp': timestamp,
        'features': feature_cols,
        'sequence_length': sequence_length,
        'metrics': metrics,
        'model_path': model_path,
        'scaler_path': scaler_path,
        'framework': 'pytorch'
    }
    
    metadata_path = f'models/lstm_metadata_{timestamp}.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    
    plt.tight_layout()
    plot_path = f'models/lstm_training_{timestamp}.png'
    plt.savefig(plot_path)
    print(f"\nTraining plot saved to {plot_path}")
    
    print(f"\n✅ Model saved to {model_path}")
    print(f"✅ Scaler saved to {scaler_path}")
    print(f"✅ Metadata saved to {metadata_path}")
    
    return model_path, scaler_path, metadata_path


if __name__ == '__main__':
    print("=" * 80)
    print("LSTM Model Training for Crypto Price Prediction")
    print("=" * 80)
    
    # Load data
    print("\nLoading data...")
    df = pd.read_csv('data/bitcoin_features.csv')
    print(f"Loaded {len(df)} samples")
    
    # Train model
    model, scaler, history, feature_cols, sequence_length = train_lstm_model(
        df,
        sequence_length=30,  # Look back 30 days
        epochs=50,
        batch_size=32,
        learning_rate=0.0001
    )
    
    # Save model
    print("\n" + "=" * 80)
    print("Saving model...")
    metrics = {
        'final_train_acc': history['train_acc'][-1],
        'final_val_acc': history['val_acc'][-1],
        'best_val_acc': max(history['val_acc'])
    }
    
    model_path, scaler_path, metadata_path = save_pytorch_model(
        model, scaler, history, feature_cols, sequence_length, metrics
    )
    
    print("\n" + "=" * 80)
    print("✅ LSTM Training Complete!")
    print(f"Best Validation Accuracy: {max(history['val_acc']):.4f}")
    print("=" * 80)
