import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
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


class SimpleLSTM(nn.Module):
    """Simplified LSTM to prevent overfitting"""
    def __init__(self, input_size, hidden_size=32):
        super(SimpleLSTM, self).__init__()
        
        # Single LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0
        )
        
        # Heavy dropout before output
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_size, 2)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        out = self.dropout(last_output)
        out = self.fc(out)
        return out


def create_sequences(df, sequence_length=30):
    """Create sequences with proper validation"""
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
        
        # Gradient clipping to prevent exploding gradients
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
    
    # Calculate AUC if possible
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.5
    
    return total_loss / len(val_loader), accuracy, auc, all_preds, all_labels


def select_top_features(df, n_features=15):
    """Select most correlated features with target"""
    feature_cols = [col for col in df.columns 
                   if col not in ['timestamp', 'target', 'price']]
    
    # Calculate correlations
    correlations = df[feature_cols].corrwith(df['target']).abs()
    top_features = correlations.nlargest(n_features).index.tolist()
    
    print(f"\nSelected top {n_features} features:")
    for i, feat in enumerate(top_features, 1):
        print(f"  {i}. {feat}: {correlations[feat]:.4f}")
    
    return top_features


def train_with_strategies(df, sequence_length=20, epochs=100, batch_size=64, learning_rate=0.0001):
    """
    Train with anti-overfitting strategies:
    - Smaller model
    - More dropout
    - L2 regularization
    - Gradient clipping
    - Feature selection
    - Longer sequences
    - Larger batches
    """
    
    print("="*80)
    print("ANTI-OVERFITTING TRAINING STRATEGY")
    print("="*80)
    
    # Strategy 1: Feature selection
    print("\nStrategy 1: Selecting most predictive features...")
    top_features = select_top_features(df, n_features=15)
    
    # Keep only selected features plus target
    selected_cols = ['timestamp', 'price'] + top_features + ['target']
    df_selected = df[selected_cols].copy()
    
    print(f"\nOriginal features: {len([c for c in df.columns if c not in ['timestamp', 'target', 'price']])}")
    print(f"Selected features: {len(top_features)}")
    
    # Strategy 2: Create sequences
    print(f"\nStrategy 2: Using sequence length of {sequence_length} (shorter = less overfitting)")
    sequences, labels, feature_cols = create_sequences(df_selected, sequence_length)
    print(f"Created {len(sequences)} sequences")
    
    # Check class balance
    print(f"\nClass distribution:")
    print(f"  Down (0): {(labels == 0).sum()} ({(labels == 0).mean()*100:.1f}%)")
    print(f"  Up (1): {(labels == 1).sum()} ({(labels == 1).mean()*100:.1f}%)")
    
    # Strategy 3: Scale features
    print("\nStrategy 3: Scaling features...")
    scaler = StandardScaler()
    n_samples, n_timesteps, n_features = sequences.shape
    sequences_reshaped = sequences.reshape(-1, n_features)
    sequences_scaled = scaler.fit_transform(sequences_reshaped)
    sequences = sequences_scaled.reshape(n_samples, n_timesteps, n_features)
    
    # Strategy 4: Larger validation set for better generalization estimate
    print("\nStrategy 4: Using 30% validation set (larger = better estimate)")
    split_idx = int(len(sequences) * 0.7)
    X_train = sequences[:split_idx]
    y_train = labels[:split_idx]
    X_val = sequences[split_idx:]
    y_val = labels[split_idx:]
    
    print(f"Train size: {len(X_train)}, Validation size: {len(X_val)}")
    
    # Strategy 5: Handle class imbalance
    print("\nStrategy 5: Adding class weights for imbalance...")
    class_counts = np.bincount(y_train)
    class_weights = len(y_train) / (len(class_counts) * class_counts)
    class_weights = torch.FloatTensor(class_weights)
    
    # Create datasets with larger batches
    train_dataset = CryptoDataset(X_train, y_train)
    val_dataset = CryptoDataset(X_val, y_val)
    
    print(f"\nStrategy 6: Using batch size {batch_size} (larger = more stable)")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Strategy 7: Smaller model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nStrategy 7: Using simplified model architecture")
    print(f"Device: {device}")
    
    model = SimpleLSTM(input_size=n_features, hidden_size=32).to(device)
    class_weights = class_weights.to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Strategy 8: Lower learning rate + L2 regularization
    print(f"\nStrategy 8: Low learning rate ({learning_rate}) + L2 regularization (0.01)")
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Strategy 9: Learning rate scheduler
    print("\nStrategy 9: Learning rate decay")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    # Training loop
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_auc': []}
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
        
        # Update learning rate based on validation accuracy
        scheduler.step(val_acc)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:3d}/{epochs}] "
                  f"Train: Loss={train_loss:.4f} Acc={train_acc:.4f} | "
                  f"Val: Loss={val_loss:.4f} Acc={val_acc:.4f} AUC={val_auc:.4f} | "
                  f"Gap: {(train_acc - val_acc)*100:+.1f}%")
        
        # Save best model
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
    print(f"\nBest Validation Accuracy: {best_val_acc:.4f}")
    print(f"Best Validation AUC: {best_val_auc:.4f}")
    print(f"\nFinal Classification Report:")
    print(classification_report(val_labels, val_preds, target_names=['Down', 'Up']))
    
    # Calculate baseline
    baseline_acc = max(np.mean(val_labels), 1 - np.mean(val_labels))
    improvement = (best_val_acc - baseline_acc) * 100
    print(f"\nBaseline (always predict majority): {baseline_acc:.4f}")
    print(f"Model improvement over baseline: {improvement:+.2f}%")
    
    if best_val_acc < 0.52:
        print("\n⚠️  WARNING: Model barely beats random chance!")
        print("This suggests:")
        print("  1. Crypto prices at this timeframe are highly random")
        print("  2. Current features don't capture predictive patterns")
        print("  3. Consider: different timeframe, external data, or problem reframing")
    elif best_val_acc < 0.55:
        print("\n✓ Model shows weak signal (52-55% range)")
        print("This is typical for short-term crypto prediction")
    else:
        print("\n✓✓ Model shows meaningful signal (>55%)")
        print("Good performance for this problem!")
    
    return model, scaler, history, feature_cols, sequence_length, top_features


def save_model(model, scaler, history, feature_cols, sequence_length, metrics):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    model_path = f'models/lstm_anti_overfit_{timestamp}.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_architecture': {'input_size': len(feature_cols), 'hidden_size': 32}
    }, model_path)
    
    scaler_path = f'models/lstm_scaler_{timestamp}.joblib'
    joblib.dump(scaler, scaler_path)
    
    metadata = {
        'model_name': 'LSTM_AntiOverfit',
        'timestamp': timestamp,
        'features': feature_cols,
        'sequence_length': sequence_length,
        'metrics': metrics,
        'model_path': model_path,
        'scaler_path': scaler_path,
        'framework': 'pytorch'
    }
    
    with open(f'models/lstm_metadata_{timestamp}.json', 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train', alpha=0.7)
    axes[0, 0].plot(history['val_loss'], label='Val', alpha=0.7)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].set_title('Training vs Validation Loss')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[0, 1].plot(history['train_acc'], label='Train', alpha=0.7)
    axes[0, 1].plot(history['val_acc'], label='Val', alpha=0.7)
    axes[0, 1].axhline(y=0.5, color='r', linestyle='--', label='Random', alpha=0.5)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].set_title('Training vs Validation Accuracy')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Overfitting gap
    gap = [t - v for t, v in zip(history['train_acc'], history['val_acc'])]
    axes[1, 0].plot(gap, color='orange', alpha=0.7)
    axes[1, 0].axhline(y=0, color='g', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Train - Val Accuracy')
    axes[1, 0].set_title('Overfitting Gap (lower is better)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # AUC
    axes[1, 1].plot(history['val_auc'], color='purple', alpha=0.7)
    axes[1, 1].axhline(y=0.5, color='r', linestyle='--', label='Random', alpha=0.5)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Validation AUC')
    axes[1, 1].legend()
    axes[1, 1].set_title('Validation AUC Score')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = f'models/lstm_training_{timestamp}.png'
    plt.savefig(plot_path, dpi=150)
    
    print(f"\n✅ Model saved to {model_path}")
    print(f"✅ Plot saved to {plot_path}")
    
    return model_path


if __name__ == '__main__':
    # Load data
    print("Loading data...")
    try:
        df = pd.read_csv('data/bitcoin_features_enhanced.csv')
        print("✓ Using enhanced features")
    except:
        df = pd.read_csv('data/bitcoin_features.csv')
        print("✓ Using basic features")
    
    print(f"Loaded {len(df)} samples\n")
    
    # Train with anti-overfitting strategies
    model, scaler, history, feature_cols, sequence_length, top_features = train_with_strategies(
        df,
        sequence_length=20,     # Shorter sequences
        epochs=100,             # More epochs with early stopping
        batch_size=64,          # Larger batches
        learning_rate=0.0001    # Lower learning rate
    )
    
    # Save
    metrics = {
        'final_train_acc': history['train_acc'][-1],
        'final_val_acc': history['val_acc'][-1],
        'best_val_acc': max(history['val_acc']),
        'best_val_auc': max(history['val_auc']),
        'overfitting_gap': history['train_acc'][-1] - history['val_acc'][-1]
    }
    
    save_model(model, scaler, history, top_features, sequence_length, metrics)
    
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE")
    print("="*80)
