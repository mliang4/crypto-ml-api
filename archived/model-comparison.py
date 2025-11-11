import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import glob
import os

def load_model_metadata(model_type):
    """Load all metadata files for a given model type"""
    pattern = f'models/{model_type}_metadata_*.json'
    files = glob.glob(pattern)
    
    if not files:
        return None
    
    # Get latest file
    latest_file = max(files, key=os.path.getctime)
    
    with open(latest_file, 'r') as f:
        metadata = json.load(f)
    
    return metadata

def compare_all_models():
    """Compare all trained models"""
    print("="*80)
    print("MODEL COMPARISON")
    print("="*80)
    
    models = {}
    
    # Load LSTM models
    lstm_meta = load_model_metadata('lstm')
    if lstm_meta:
        models['LSTM'] = lstm_meta
        print(f"\n✓ Found LSTM model")
    
    # Load Transformer models
    transformer_meta = load_model_metadata('transformer')
    if transformer_meta:
        models['Transformer'] = transformer_meta
        print(f"✓ Found Transformer model")
    
    # Load anti-overfit LSTM
    lstm_anti_meta = load_model_metadata('lstm_anti_overfit')
    if lstm_anti_meta:
        models['LSTM (Anti-Overfit)'] = lstm_anti_meta
        print(f"✓ Found LSTM Anti-Overfit model")
    
    if not models:
        print("\n❌ No models found! Train some models first.")
        return
    
    print(f"\nTotal models found: {len(models)}")
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON")
    print("="*80)
    
    # Create comparison table
    comparison_data = []
    
    for name, meta in models.items():
        metrics = meta['metrics']
        comparison_data.append({
            'Model': name,
            'Train Acc': f"{metrics.get('final_train_acc', 0):.4f}",
            'Val Acc': f"{metrics.get('final_val_acc', 0):.4f}",
            'Best Val Acc': f"{metrics.get('best_val_acc', 0):.4f}",
            'AUC': f"{metrics.get('best_val_auc', 0):.4f}",
            'Overfit Gap': f"{metrics.get('overfitting_gap', 0):.4f}",
            'Timestamp': meta['timestamp']
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    df_comparison = df_comparison.sort_values('Best Val Acc', ascending=False)
    
    print("\n")
    print(df_comparison.to_string(index=False))
    
    # Find best model
    best_idx = df_comparison['Best Val Acc'].astype(float).idxmax()
    best_model = df_comparison.iloc[best_idx]['Model']
    best_acc = float(df_comparison.iloc[best_idx]['Best Val Acc'])
    
    print("\n" + "="*80)
    print(f"🏆 BEST MODEL: {best_model}")
    print(f"   Validation Accuracy: {best_acc:.4f} ({best_acc*100:.1f}%)")
    print("="*80)
    
    # Visualize comparison
    create_comparison_plots(models)
    
    # Provide recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    if best_acc >= 0.58:
        print("\n✓✓ EXCELLENT performance!")
        print("Your best model found strong predictive patterns.")
        print("This is production-ready for crypto prediction at this timeframe.")
    elif best_acc >= 0.55:
        print("\n✓ GOOD performance!")
        print("Your model shows meaningful signal above random.")
        print("Consider:")
        print("  • Ensemble multiple models for better stability")
        print("  • Add more features (sentiment, on-chain data)")
        print("  • Fine-tune hyperparameters further")
    elif best_acc >= 0.52:
        print("\n~ WEAK signal detected")
        print("Models are learning but patterns are subtle.")
        print("Consider:")
        print("  • Different prediction horizon (weekly vs daily)")
        print("  • External data sources")
        print("  • Problem reframing (volatility prediction)")
    else:
        print("\n⚠️  No significant signal")
        print("Short-term crypto prices may be too random at this scale.")
        print("This is a VALID finding - market efficiency!")
        print("\nFor your interview, explain:")
        print("  • Tried multiple architectures (LSTM, Transformer)")
        print("  • Applied proper ML practices (regularization, validation)")
        print("  • Results suggest data is largely random")
        print("  • Would explore: different timeframes, external features, ensemble methods")
    
    # Model-specific insights
    print("\n" + "="*80)
    print("MODEL-SPECIFIC INSIGHTS")
    print("="*80)
    
    if 'LSTM' in models and 'Transformer' in models:
        lstm_acc = float(models['LSTM']['metrics']['best_val_acc'])
        trans_acc = float(models['Transformer']['metrics']['best_val_acc'])
        diff = (trans_acc - lstm_acc) * 100
        
        print(f"\nTransformer vs LSTM:")
        if diff > 2:
            print(f"  ✓ Transformer is {diff:.1f}% better!")
            print("    → Attention mechanism helps with crypto patterns")
            print("    → Use Transformer for production")
        elif diff > 0:
            print(f"  ~ Transformer is slightly better (+{diff:.1f}%)")
            print("    → Marginal improvement, either could work")
        else:
            print(f"  ○ LSTM performs similarly (diff: {diff:.1f}%)")
            print("    → LSTM is simpler and faster, prefer it")
    
    print("\n" + "="*80)

def create_comparison_plots(models):
    """Create visual comparison of models"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    model_names = list(models.keys())
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    
    # Plot 1: Validation Accuracy Comparison
    val_accs = [models[name]['metrics']['best_val_acc'] for name in model_names]
    bars1 = axes[0].bar(range(len(model_names)), val_accs, color=colors[:len(model_names)], alpha=0.7)
    axes[0].axhline(y=0.5, color='r', linestyle='--', label='Random', linewidth=2)
    axes[0].set_xticks(range(len(model_names)))
    axes[0].set_xticklabels(model_names, rotation=45, ha='right')
    axes[0].set_ylabel('Validation Accuracy')
    axes[0].set_title('Best Validation Accuracy Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot 2: Overfitting Gap Comparison
    gaps = [abs(models[name]['metrics'].get('overfitting_gap', 0)) for name in model_names]
    bars2 = axes[1].bar(range(len(model_names)), gaps, color=colors[:len(model_names)], alpha=0.7)
    axes[1].set_xticks(range(len(model_names)))
    axes[1].set_xticklabels(model_names, rotation=45, ha='right')
    axes[1].set_ylabel('Overfitting Gap (abs)')
    axes[1].set_title('Overfitting Gap (Lower is Better)')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars2:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot 3: AUC Score Comparison
    aucs = [models[name]['metrics'].get('best_val_auc', 0.5) for name in model_names]
    bars3 = axes[2].bar(range(len(model_names)), aucs, color=colors[:len(model_names)], alpha=0.7)
    axes[2].axhline(y=0.5, color='r', linestyle='--', label='Random', linewidth=2)
    axes[2].set_xticks(range(len(model_names)))
    axes[2].set_xticklabels(model_names, rotation=45, ha='right')
    axes[2].set_ylabel('AUC Score')
    axes[2].set_title('AUC Score Comparison')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars3:
        height = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plot_path = 'models/model_comparison.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Comparison plot saved to {plot_path}")
    plt.close()

def generate_interview_summary():
    """Generate a summary for interview discussions"""
    print("\n" + "="*80)
    print("INTERVIEW TALKING POINTS")
    print("="*80)
    
    print("""
    1. PROBLEM APPROACH:
       "I approached crypto price prediction as a time-series classification problem,
       predicting next-day price direction (up/down). I implemented multiple deep
       learning architectures to compare performance."
    
    2. MODELS TRIED:
       • LSTM: Sequential processing, good for time-series
       • Transformer: Self-attention mechanism, captures long-range dependencies
       • Applied regularization techniques to prevent overfitting
    
    3. CHALLENGES:
       • Class imbalance → Used weighted loss function
       • Overfitting → Dropout, L2 regularization, early stopping
       • Weak signal → Feature engineering, hyperparameter tuning
    
    4. RESULTS & INSIGHTS:
       • Achieved X% validation accuracy (compare to 50% baseline)
       • Transformer [better/similar] to LSTM because [attention/sequential]
       • Key learning: Short-term crypto prices have limited predictability,
         suggesting market efficiency at this timeframe
    
    5. PRODUCTION CONSIDERATIONS:
       • Model monitoring for data drift
       • Ensemble methods for robustness
       • A/B testing for deployment
       • Regular retraining schedule
    
    6. NEXT STEPS IF MORE TIME:
       • External features (sentiment, news, on-chain metrics)
       • Different prediction horizons (weekly vs daily)
       • Multi-task learning (predict volatility + direction)
       • Attention visualization to understand model decisions
    """)
    
    print("="*80)

if __name__ == '__main__':
    compare_all_models()
    generate_interview_summary()
    
    print("\n✅ Comparison complete!")
    print("Check 'models/model_comparison.png' for visual comparison")
