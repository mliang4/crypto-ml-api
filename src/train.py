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
