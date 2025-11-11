import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import math
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import joblib
import json
from datetime import datetime
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)

class CryptoDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


class PositionalEncoding(nn.Module):
    """Add positional information to input embeddings"""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class CryptoTransformer(nn.Module):
    """
    Transformer model for crypto price prediction
    Uses self-attention to learn temporal patterns
    """
    def __init__(self, input_size, d_model=64, nhead=4, num_layers=2, dropout=0.3):
        super(CryptoTransformer, self).__init__()
        
        self.d_model = d_model
        
        # Project input features to d_model dimensions
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(d_model, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 2)
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, input_size]
        
        # Project to d_model dimensions
        x = self.input_projection(x)  # [batch, seq_len, d_model]
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Apply transformer
        x = self.transformer_encoder(x)  # [batch, seq_len, d_model]
        
        # Use last timestep for prediction (or could use mean/max pooling)
        x = x[:, -1, :]  # [batch, d_model]
        
        # Classification layers
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class AttentionVisualizationTransformer(CryptoTransformer):
    """
    Extended version that captures attention weights for visualization
    Useful for understanding what the model focuses on
    """
    def forward_with_attention(self, x):
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        
        # Get attention weights from first layer
        attention_weights = []
        
        for layer in self.transformer_encoder.layers:
            # Store attention weights
            attn_output, attn_weights = layer.self_attn(x, x, x, need_weights=True)
            attention_weights.append(attn_weights)
            
            # Continue forward pass
            x = layer(x)
        
        x = x[:, -1, :]
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x, attention_weights


def create_sequences(df, sequence_length=30):
    feature_cols = [col for col in df.columns 
                   if col not in ['timestamp', 'target', 'price']]
    
    sequences = []
    labels = []
    
    for i in range(sequence_length, len(df)):
        seq = df[feature_cols].iloc[i-sequence_length:i].values
        sequences.append(seq)
        label = df['target'].iloc[i]
        labels.append(label)
    
    return np.array(sequences), np.array(labels), feature_cols


def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for sequences, labels in train_loader:
        sequences = sequences.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        total_loss += loss.item()
    
    return total_loss / len(train_loader), correct / total


def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for sequences, labels in val_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            total_loss += loss.item()
    
    accuracy = accuracy_score(all_labels, all_preds)
    
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.5
    
    return total_loss / len(val_loader), accuracy, auc, all_preds, all_labels


def train_transformer_model(df, sequence_length=30, epochs=100, batch_size=32, learning_rate=0.0001):
    """
    Train Transformer model for crypto prediction
    """
    print("="*80)
    print("TRANSFORMER MODEL TRAINING")
    print("="*80)
    
    print("\nCreating sequences...")
    sequences, labels, feature_cols = create_sequences(df, sequence_length)
    print(f"Created {len(sequences)} sequences with shape {sequences.shape}")
    
    # Scale features
    print("\nScaling features...")
    scaler = StandardScaler()
    n_samples, n_timesteps, n_features = sequences.shape
    sequences_reshaped = sequences.reshape(-1, n_features)
    sequences_scaled = scaler.fit_transform(sequences_reshaped)
    sequences = sequences_scaled.reshape(n_samples, n_timesteps, n_features)
    
    # Split data
    split_idx = int(len(sequences) * 0.75)
    X_train = sequences[:split_idx]
    y_train = labels[:split_idx]
    X_val = sequences[split_idx:]
    y_val = labels[split_idx:]
    
    print(f"\nTrain size: {len(X_train)}, Validation size: {len(X_val)}")
    print(f"Class distribution:")
    print(f"  Train - Down: {(y_train == 0).sum()} Up: {(y_train == 1).sum()}")
    print(f"  Val   - Down: {(y_val == 0).sum()} Up: {(y_val == 1).sum()}")
    
    # Handle class imbalance
    class_counts = np.bincount(y_train)
    class_weights = len(y_train) / (len(class_counts) * class_counts)
    class_weights = torch.FloatTensor(class_weights)
    
    # Create data loaders
    train_dataset = CryptoDataset(X_train, y_train)
    val_dataset = CryptoDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    model = CryptoTransformer(
        input_size=n_features,
        d_model=64,      # Model dimension
        nhead=4,         # Number of attention heads
        num_layers=2,    # Number of transformer layers
        dropout=0.3
    ).to(device)
    
    class_weights = class_weights.to(device)
    
    print(f"\nModel architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=7
    )
    
    # Training loop
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_auc': []
    }
    
    best_val_acc = 0
    best_val_auc = 0
    patience = 15
    patience_counter = 0
    
    print("\n" + "="*80)
    print("TRAINING")
    print("="*80)
    
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_auc, val_preds, val_labels = validate(model, val_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_auc'].append(val_auc)
        
        scheduler.step(val_acc)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            gap = (train_acc - val_acc) * 100
            print(f"Epoch [{epoch+1:3d}/{epochs}] "
                  f"Train: Loss={train_loss:.4f} Acc={train_acc:.4f} | "
                  f"Val: Loss={val_loss:.4f} Acc={val_acc:.4f} AUC={val_auc:.4f} | "
                  f"Gap: {gap:+.1f}%")
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_auc = val_auc
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    val_loss, val_acc, val_auc, val_preds, val_labels = validate(model, val_loader, criterion, device)
    
    print(f"\nBest Validation Accuracy: {best_val_acc:.4f} ({best_val_acc*100:.1f}%)")
    print(f"Best Validation AUC: {best_val_auc:.4f}")
    
    baseline_acc = max(np.mean(val_labels), 1 - np.mean(val_labels))
    improvement = (best_val_acc - baseline_acc) * 100
    print(f"\nBaseline (majority class): {baseline_acc:.4f}")
    print(f"Improvement over baseline: {improvement:+.2f}%")
    
    print(f"\nClassification Report:")
    print(classification_report(val_labels, val_preds, target_names=['Down', 'Up']))
    
    # Performance assessment
    print("\n" + "="*80)
    if best_val_acc >= 0.58:
        print("🎉 EXCELLENT! Strong predictive signal detected")
        print("The Transformer found meaningful patterns in the data")
    elif best_val_acc >= 0.55:
        print("✓ GOOD! Moderate predictive signal")
        print("Model shows promising results")
    elif best_val_acc >= 0.52:
        print("~ WEAK signal detected")
        print("Model slightly better than random - patterns exist but are subtle")
    else:
        print("⚠️  No significant signal")
        print("Data appears random at this timeframe - consider:")
        print("  • Different prediction horizon (weekly instead of daily)")
        print("  • External features (sentiment, news, on-chain data)")
        print("  • Different problem formulation (volatility, price range)")
    print("="*80)
    
    return model, scaler, history, feature_cols, sequence_length


def save_transformer_model(model, scaler, history, feature_cols, sequence_length, metrics):
    """Save transformer model and metadata"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save model
    model_path = f'models/transformer_model_{timestamp}.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_architecture': {
            'input_size': len(feature_cols),
            'd_model': 64,
            'nhead': 4,
            'num_layers': 2,
            'dropout': 0.3
        }
    }, model_path)
    
    # Save scaler
    scaler_path = f'models/transformer_scaler_{timestamp}.joblib'
    joblib.dump(scaler, scaler_path)
    
    # Save metadata
    metadata = {
        'model_name': 'Transformer',
        'timestamp': timestamp,
        'features': feature_cols,
        'sequence_length': sequence_length,
        'metrics': metrics,
        'model_path': model_path,
        'scaler_path': scaler_path,
        'framework': 'pytorch'
    }
    
    with open(f'models/transformer_metadata_{timestamp}.json', 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    # Create comprehensive plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train', alpha=0.8)
    axes[0, 0].plot(history['val_loss'], label='Val', alpha=0.8)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].set_title('Training vs Validation Loss')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[0, 1].plot(history['train_acc'], label='Train', alpha=0.8)
    axes[0, 1].plot(history['val_acc'], label='Val', alpha=0.8)
    axes[0, 1].axhline(y=0.5, color='r', linestyle='--', label='Random', alpha=0.5)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].set_title('Training vs Validation Accuracy')
    axes[0, 1].grid(True, alpha=0.3)
    
    # AUC
    axes[0, 2].plot(history['val_auc'], color='purple', alpha=0.8)
    axes[0, 2].axhline(y=0.5, color='r', linestyle='--', label='Random', alpha=0.5)
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('AUC')
    axes[0, 2].legend()
    axes[0, 2].set_title('Validation AUC Score')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Overfitting gap
    gap = [t - v for t, v in zip(history['train_acc'], history['val_acc'])]
    axes[1, 0].plot(gap, color='orange', alpha=0.8)
    axes[1, 0].axhline(y=0, color='g', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Train - Val Accuracy')
    axes[1, 0].set_title('Overfitting Gap')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Accuracy distribution
    axes[1, 1].hist(history['val_acc'], bins=20, alpha=0.7, color='blue', edgecolor='black')
    axes[1, 1].axvline(x=0.5, color='r', linestyle='--', label='Random')
    axes[1, 1].axvline(x=max(history['val_acc']), color='g', linestyle='--', label='Best')
    axes[1, 1].set_xlabel('Validation Accuracy')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].legend()
    axes[1, 1].set_title('Validation Accuracy Distribution')
    
    # Summary stats
    axes[1, 2].axis('off')
    summary_text = f"""
    TRANSFORMER MODEL SUMMARY
    
    Best Val Accuracy: {max(history['val_acc']):.4f}
    Best Val AUC: {max(history['val_auc']):.4f}
    Final Train Acc: {history['train_acc'][-1]:.4f}
    Final Val Acc: {history['val_acc'][-1]:.4f}
    
    Overfitting Gap: {gap[-1]:.4f}
    Total Epochs: {len(history['train_acc'])}
    
    Model: Transformer
    Parameters: {sum(p.numel() for p in model.parameters()):,}
    Sequence Length: {sequence_length}
    Features: {len(feature_cols)}
    """
    axes[1, 2].text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                    verticalalignment='center')
    
    plt.tight_layout()
    plot_path = f'models/transformer_training_{timestamp}.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    
    print(f"\n✅ Model saved to {model_path}")
    print(f"✅ Scaler saved to {scaler_path}")
    print(f"✅ Metadata saved")
    print(f"✅ Training plot saved to {plot_path}")
    
    return model_path


if __name__ == '__main__':
    print("="*80)
    print("TRANSFORMER MODEL FOR CRYPTO PRICE PREDICTION")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    try:
        df = pd.read_csv('data/bitcoin_features_enhanced.csv')
        print("✓ Using enhanced features")
    except:
        df = pd.read_csv('data/bitcoin_features.csv')
        print("✓ Using basic features")
    
    print(f"Loaded {len(df)} samples")
    
    # Train model
    print("\nStarting Transformer training...")
    print("This may take longer than LSTM but should give better results!\n")
    
    model, scaler, history, feature_cols, sequence_length = train_transformer_model(
        df,
        sequence_length=30,
        epochs=100,
        batch_size=32,
        learning_rate=0.00001
    )
    
    # Save model
    metrics = {
        'final_train_acc': history['train_acc'][-1],
        'final_val_acc': history['val_acc'][-1],
        'best_val_acc': max(history['val_acc']),
        'best_val_auc': max(history['val_auc']),
        'overfitting_gap': history['train_acc'][-1] - history['val_acc'][-1]
    }
    
    save_transformer_model(model, scaler, history, feature_cols, sequence_length, metrics)
    
    print("\n" + "="*80)
    print("✅ TRANSFORMER TRAINING COMPLETE!")
    print("="*80)
    print(f"\nBest performance: {max(history['val_acc'])*100:.1f}% accuracy")
    print("\nCompare with your LSTM results to see if Transformer performs better!")
