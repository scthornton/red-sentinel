#!/usr/bin/env python3
"""
Generate Comparison Graphics

Creates meaningful comparison graphics between synthetic and real data performance.
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from features import RedTeamFeatureExtractor
from ml import RedTeamMLPipeline


def load_and_prepare_data():
    """Load both synthetic and real data for comparison."""
    print("Loading data for comparison...")
    
    # Load synthetic data
    synthetic_path = "training_data/synthetic_attacks.csv"
    if os.path.exists(synthetic_path):
        synthetic_df = pd.read_csv(synthetic_path)
        print(f"Loaded synthetic data: {synthetic_df.shape}")
    else:
        print("Synthetic data not found")
        return None, None
    
    # Load real data
    real_path = "converted_data/converted_adapt_ai_data.csv"
    if os.path.exists(real_path):
        real_df = pd.read_csv(real_path)
        print(f"Loaded real data: {real_df.shape}")
    else:
        print("Real data not found")
        return None, None
    
    return synthetic_df, real_df


def train_and_evaluate_models(df, data_name, max_features=1000):
    """Train and evaluate models on a dataset."""
    print(f"\nTraining models on {data_name} data...")
    
    # Feature extraction
    extractor = RedTeamFeatureExtractor(
        use_embeddings=False,
        max_tfidf_features=max_features
    )
    
    X, y = extractor.prepare_dataset(df)
    print(f"Feature matrix: {X.shape}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    # Train models with more realistic evaluation
    pipeline = RedTeamMLPipeline(
        models=['gbm', 'rf'],  # Focus on key models
        cv_folds=5,
        random_state=42
    )
    
    pipeline.set_feature_names(list(X.columns))
    
    # Use smaller test size for more realistic evaluation
    results = pipeline.train_models(X, y, test_size=0.3)
    
    return results, pipeline


def create_comparison_plots(synthetic_results, real_results):
    """Create comprehensive comparison plots."""
    print("\nGenerating comparison plots...")
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('RedSentinel: Synthetic vs Real Data Performance Comparison', 
                 fontsize=16, fontweight='bold')
    
    # 1. F1 Score Comparison
    ax1 = axes[0, 0]
    models = ['GBM', 'RF']
    
    synthetic_f1 = [np.mean([fold['f1'] for fold in synthetic_results['gbm']['cv_metrics']]),
                    np.mean([fold['f1'] for fold in synthetic_results['rf']['cv_metrics']])]
    real_f1 = [np.mean([fold['f1'] for fold in real_results['gbm']['cv_metrics']]),
               np.mean([fold['f1'] for fold in real_results['rf']['cv_metrics']])]
    
    x = np.arange(len(models))
    width = 0.35
    
    ax1.bar(x - width/2, synthetic_f1, width, label='Synthetic Data', alpha=0.8)
    ax1.bar(x + width/2, real_f1, width, label='Real Data', alpha=0.8)
    ax1.set_xlabel('Model')
    ax1.set_ylabel('F1 Score')
    ax1.set_title('F1 Score Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. ROC AUC Comparison
    ax2 = axes[0, 1]
    synthetic_auc = [np.mean([fold['roc_auc'] for fold in synthetic_results['gbm']['cv_metrics']]),
                     np.mean([fold['roc_auc'] for fold in synthetic_results['rf']['cv_metrics']])]
    real_auc = [np.mean([fold['roc_auc'] for fold in real_results['gbm']['cv_metrics']]),
                np.mean([fold['roc_auc'] for fold in real_results['rf']['cv_metrics']])]
    
    ax2.bar(x - width/2, synthetic_auc, width, label='Synthetic Data', alpha=0.8)
    ax2.bar(x + width/2, real_auc, width, label='Real Data', alpha=0.8)
    ax2.set_xlabel('Model')
    ax2.set_ylabel('ROC AUC')
    ax2.set_title('ROC AUC Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Accuracy Comparison
    ax3 = axes[0, 2]
    synthetic_acc = [np.mean([fold['accuracy'] for fold in synthetic_results['gbm']['cv_metrics']]),
                     np.mean([fold['accuracy'] for fold in synthetic_results['rf']['cv_metrics']])]
    real_acc = [np.mean([fold['accuracy'] for fold in real_results['gbm']['cv_metrics']]),
                np.mean([fold['accuracy'] for fold in real_results['rf']['cv_metrics']])]
    
    ax3.bar(x - width/2, synthetic_acc, width, label='Synthetic Data', alpha=0.8)
    ax3.bar(x + width/2, real_acc, width, label='Real Data', alpha=0.8)
    ax3.set_xlabel('Model')
    ax3.set_ylabel('Accuracy')
    ax3.set_title('Accuracy Comparison')
    ax3.set_xticks(x)
    ax3.set_xticklabels(models)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Cross-validation F1 scores across folds
    ax4 = axes[1, 0]
    
    # GBM F1 scores
    synthetic_gbm_f1 = [fold['f1'] for fold in synthetic_results['gbm']['cv_metrics']]
    real_gbm_f1 = [fold['f1'] for fold in real_results['gbm']['cv_metrics']]
    folds = range(1, len(synthetic_gbm_f1) + 1)
    
    ax4.plot(folds, synthetic_gbm_f1, 
             'o-', label='Synthetic GBM', linewidth=2, markersize=8)
    ax4.plot(folds, real_gbm_f1, 
             's-', label='Real GBM', linewidth=2, markersize=8)
    ax4.set_xlabel('Fold')
    ax4.set_ylabel('F1 Score')
    ax4.set_title('GBM F1 Score Across Folds')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xticks(folds)
    
    # 5. Cross-validation ROC AUC across folds
    ax5 = axes[1, 1]
    
    # RF ROC AUC scores
    synthetic_rf_auc = [fold['roc_auc'] for fold in synthetic_results['rf']['cv_metrics']]
    real_rf_auc = [fold['roc_auc'] for fold in real_results['rf']['cv_metrics']]
    folds = range(1, len(synthetic_rf_auc) + 1)
    
    ax5.plot(folds, synthetic_rf_auc, 
             'o-', label='Synthetic RF', linewidth=2, markersize=8)
    ax5.plot(folds, real_rf_auc, 
             's-', label='Real RF', linewidth=2, markersize=8)
    ax5.set_xlabel('Fold')
    ax5.set_ylabel('ROC AUC')
    ax5.set_title('RF ROC AUC Across Folds')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_xticks(folds)
    
    # 6. Performance improvement summary
    ax6 = axes[1, 2]
    
    # Calculate improvements
    f1_improvement = [(real_f1[i] - synthetic_f1[i]) / synthetic_f1[i] * 100 
                      for i in range(len(models))]
    auc_improvement = [(real_auc[i] - synthetic_auc[i]) / synthetic_auc[i] * 100 
                       for i in range(len(models))]
    
    x_pos = np.arange(len(models))
    width = 0.35
    
    ax6.bar(x_pos - width/2, f1_improvement, width, label='F1 Score % Improvement', alpha=0.8)
    ax6.bar(x_pos + width/2, auc_improvement, width, label='ROC AUC % Improvement', alpha=0.8)
    ax6.set_xlabel('Model')
    ax6.set_ylabel('Improvement (%)')
    ax6.set_title('Performance Improvement with Real Data')
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(models)
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # Add zero line for reference
    ax6.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    output_dir = "comparison_reports"
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "synthetic_vs_real_comparison.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to: {plot_path}")
    
    plt.show()
    
    return plot_path


def generate_performance_summary(synthetic_results, real_results):
    """Generate a detailed performance summary."""
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON SUMMARY")
    print("="*80)
    
    models = ['gbm', 'rf']
    
    for model in models:
        print(f"\n{model.upper()} MODEL:")
        print("-" * 40)
        
        # Synthetic data performance
        synth_f1 = np.mean([fold['f1'] for fold in synthetic_results[model]['cv_metrics']])
        synth_auc = np.mean([fold['roc_auc'] for fold in synthetic_results[model]['cv_metrics']])
        synth_acc = np.mean([fold['accuracy'] for fold in synthetic_results[model]['cv_metrics']])
        
        # Real data performance
        real_f1 = np.mean([fold['f1'] for fold in real_results[model]['cv_metrics']])
        real_auc = np.mean([fold['roc_auc'] for fold in real_results[model]['cv_metrics']])
        real_acc = np.mean([fold['accuracy'] for fold in real_results[model]['cv_metrics']])
        
        print(f"Synthetic Data:")
        print(f"  F1 Score: {synth_f1:.4f}")
        print(f"  ROC AUC:  {synth_auc:.4f}")
        print(f"  Accuracy: {synth_acc:.4f}")
        
        print(f"\nReal Data:")
        print(f"  F1 Score: {real_f1:.4f}")
        print(f"  ROC AUC:  {real_auc:.4f}")
        print(f"  Accuracy: {real_acc:.4f}")
        
        # Calculate improvements
        f1_improvement = (real_f1 - synth_f1) / synth_f1 * 100
        auc_improvement = (real_auc - synth_auc) / synth_auc * 100
        acc_improvement = (real_acc - synth_acc) / synth_acc * 100
        
        print(f"\nImprovement with Real Data:")
        print(f"  F1 Score: {f1_improvement:+.2f}%")
        print(f"  ROC AUC:  {auc_improvement:+.2f}%")
        print(f"  Accuracy: {acc_improvement:+.2f}%")


def main():
    """Main comparison function."""
    print("RedSentinel: Synthetic vs Real Data Performance Comparison")
    print("=" * 80)
    
    try:
        # Load data
        synthetic_df, real_df = load_and_prepare_data()
        if synthetic_df is None or real_df is None:
            print("Error: Could not load data")
            return
        
        # Train and evaluate on synthetic data
        print("\n" + "="*50)
        print("TRAINING ON SYNTHETIC DATA")
        print("="*50)
        synthetic_results, _ = train_and_evaluate_models(synthetic_df, "Synthetic")
        
        # Train and evaluate on real data
        print("\n" + "="*50)
        print("TRAINING ON REAL DATA")
        print("="*50)
        real_results, _ = train_and_evaluate_models(real_df, "Real")
        
        # Generate comparison plots
        print("\n" + "="*50)
        print("GENERATING COMPARISON PLOTS")
        print("="*50)
        plot_path = create_comparison_plots(synthetic_results, real_results)
        
        # Generate performance summary
        generate_performance_summary(synthetic_results, real_results)
        
        print(f"\n🎉 Comparison complete! Plot saved to: {plot_path}")
        
    except Exception as e:
        print(f"Error during comparison: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
