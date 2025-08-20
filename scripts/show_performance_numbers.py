#!/usr/bin/env python3
"""
Show Performance Numbers

Extract and display the specific performance metrics from the comparison.
"""

from ml import RedTeamMLPipeline
from features import RedTeamFeatureExtractor
import sys
import os
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))


def extract_performance_metrics():
    """Extract performance metrics from both datasets."""
    print("=" * 80)
    print("REDSENTINEL PERFORMANCE METRICS EXTRACTION")
    print("=" * 80)

    # Load synthetic data
    print("\n📊 SYNTHETIC DATA PERFORMANCE")
    print("-" * 50)
    synthetic_path = "training_data/synthetic_attacks.csv"
    if os.path.exists(synthetic_path):
        synthetic_df = pd.read_csv(synthetic_path)
        print(f"Dataset size: {len(synthetic_df):,} attacks")

        # Train and evaluate on synthetic data
        extractor = RedTeamFeatureExtractor(
            use_embeddings=False, max_tfidf_features=500)
        X_synth, y_synth = extractor.prepare_dataset(synthetic_df)

        pipeline = RedTeamMLPipeline(
            models=['gbm', 'rf'], cv_folds=5, random_state=42)
        pipeline.set_feature_names(list(X_synth.columns))

        synthetic_results = pipeline.train_models(
            X_synth, y_synth, test_size=0.3)

        # Extract metrics
        for model_name in ['gbm', 'rf']:
            if model_name in synthetic_results:
                cv_metrics = synthetic_results[model_name]['cv_metrics']
                f1_scores = [fold['f1'] for fold in cv_metrics]
                auc_scores = [fold['roc_auc'] for fold in cv_metrics]
                acc_scores = [fold['accuracy'] for fold in cv_metrics]

                print(f"\n{model_name.upper()} Model:")
                print(
                    f"  F1 Score: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
                print(
                    f"  ROC AUC:  {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")
                print(
                    f"  Accuracy: {np.mean(acc_scores):.4f} ± {np.std(acc_scores):.4f}")
    else:
        print("Synthetic data not found")

    # Load real data
    print("\n🎯 REAL DATA PERFORMANCE")
    print("-" * 50)
    real_path = "converted_data/converted_adapt_ai_data.csv"
    if os.path.exists(real_path):
        real_df = pd.read_csv(real_path)
        print(f"Dataset size: {len(real_df):,} attacks")

        # Train and evaluate on real data
        extractor = RedTeamFeatureExtractor(
            use_embeddings=False, max_tfidf_features=500)
        X_real, y_real = extractor.prepare_dataset(real_df)

        pipeline = RedTeamMLPipeline(
            models=['gbm', 'rf'], cv_folds=5, random_state=42)
        pipeline.set_feature_names(list(X_real.columns))

        real_results = pipeline.train_models(X_real, y_real, test_size=0.3)

        # Extract metrics
        for model_name in ['gbm', 'rf']:
            if model_name in real_results:
                cv_metrics = real_results[model_name]['cv_metrics']
                f1_scores = [fold['f1'] for fold in cv_metrics]
                auc_scores = [fold['roc_auc'] for fold in cv_metrics]
                acc_scores = [fold['accuracy'] for fold in cv_metrics]

                print(f"\n{model_name.upper()} Model:")
                print(
                    f"  F1 Score: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
                print(
                    f"  ROC AUC:  {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")
                print(
                    f"  Accuracy: {np.mean(acc_scores):.4f} ± {np.std(acc_scores):.4f}")
    else:
        print("Real data not found")

    # Calculate improvements
    if 'synthetic_results' in locals() and 'real_results' in locals():
        print("\n🚀 PERFORMANCE IMPROVEMENT ANALYSIS")
        print("-" * 50)

        for model_name in ['gbm', 'rf']:
            if model_name in synthetic_results and model_name in real_results:
                # Synthetic metrics
                synth_cv = synthetic_results[model_name]['cv_metrics']
                synth_f1 = np.mean([fold['f1'] for fold in synth_cv])
                synth_auc = np.mean([fold['roc_auc'] for fold in synth_cv])
                synth_acc = np.mean([fold['accuracy'] for fold in synth_cv])

                # Real metrics
                real_cv = real_results[model_name]['cv_metrics']
                real_f1 = np.mean([fold['f1'] for fold in real_cv])
                real_auc = np.mean([fold['roc_auc'] for fold in real_cv])
                real_acc = np.mean([fold['accuracy'] for fold in real_cv])

                # Calculate improvements
                f1_improvement = (real_f1 - synth_f1) / synth_f1 * 100
                auc_improvement = (real_auc - synth_auc) / synth_auc * 100
                acc_improvement = (real_acc - synth_acc) / synth_acc * 100

                print(f"\n{model_name.upper()} Model Improvements:")
                print(
                    f"  F1 Score: {f1_improvement:+.2f}% ({synth_f1:.4f} → {real_f1:.4f})")
                print(
                    f"  ROC AUC:  {auc_improvement:+.2f}% ({synth_auc:.4f} → {real_auc:.4f})")
                print(
                    f"  Accuracy: {acc_improvement:+.2f}% ({synth_acc:.4f} → {real_acc:.4f})")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)


def main():
    """Main function."""
    try:
        extract_performance_metrics()
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
