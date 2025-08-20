#!/usr/bin/env python3
"""
RedSentinel ML Pipeline Runner

Runs the complete machine learning pipeline on training data.
Trains multiple models, performs cross-validation, and generates reports.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from ml import RedTeamMLPipeline
from features import RedTeamFeatureExtractor
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def load_training_data(data_path: str) -> pd.DataFrame:
    """Load training data from CSV file."""
    print(f"Loading training data from {data_path}")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Training data file not found: {data_path}")

    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} records with {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)}")

    return df


def run_feature_extraction(df: pd.DataFrame,
                           use_embeddings: bool = False,
                           max_tfidf_features: int = 5000) -> tuple:
    """Run feature extraction on the training data."""
    print("\n" + "="*50)
    print("FEATURE EXTRACTION")
    print("="*50)

    extractor = RedTeamFeatureExtractor(
        use_embeddings=use_embeddings,
        max_tfidf_features=max_tfidf_features
    )

    # Prepare dataset
    X, y = extractor.prepare_dataset(df)

    print(f"\nFeature extraction complete!")
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target distribution: {y.value_counts().to_dict()}")

    # Save transformers for later use
    os.makedirs("models", exist_ok=True)
    extractor.save_transformers("models/feature_transformers.joblib")

    return X, y, extractor


def run_ml_training(X: pd.DataFrame,
                    y: pd.Series,
                    models: list = None,
                    cv_folds: int = 5) -> RedTeamMLPipeline:
    """Run ML training pipeline."""
    print("\n" + "="*50)
    print("MACHINE LEARNING TRAINING")
    print("="*50)

    if models is None:
        models = ['gbm', 'rf', 'xgb']

    # Initialize pipeline
    pipeline = RedTeamMLPipeline(
        models=models,
        cv_folds=cv_folds,
        random_state=42
    )

    # Set feature names for plotting
    pipeline.set_feature_names(list(X.columns))

    # Train models
    results = pipeline.train_models(X, y, test_size=0.2)

    print(f"\nTraining complete for {len(results)} models!")

    # Show best model
    best_model_name, best_model = pipeline.get_best_model(metric='f1')
    print(f"Best model: {best_model_name.upper()}")

    return pipeline


def generate_reports(pipeline: RedTeamMLPipeline,
                     extractor: RedTeamFeatureExtractor,
                     output_dir: str = "ml_reports"):
    """Generate comprehensive ML reports."""
    print("\n" + "="*50)
    print("GENERATING REPORTS")
    print("="*50)

    # Generate ML pipeline report
    pipeline.generate_report(output_dir)

    # Save best model
    best_model_name, _ = pipeline.get_best_model(metric='f1')
    pipeline.save_model(
        best_model_name, f"models/best_model_{best_model_name}.joblib")

    print(f"Reports generated in {output_dir}")
    print(f"Best model saved as models/best_model_{best_model_name}.joblib")


def analyze_results(pipeline: RedTeamMLPipeline, X: pd.DataFrame, y: pd.Series):
    """Analyze and display training results."""
    print("\n" + "="*50)
    print("RESULTS ANALYSIS")
    print("="*50)

    # Show CV results summary
    print("\nCross-Validation Results Summary:")
    print("-" * 40)

    for model_name, cv_metrics in pipeline.cv_results.items():
        avg_f1 = np.mean([fold['f1'] for fold in cv_metrics])
        avg_roc = np.mean([fold['roc_auc'] for fold in cv_metrics])
        avg_acc = np.mean([fold['accuracy'] for fold in cv_metrics])

        print(
            f"{model_name.upper():<10} | F1: {avg_f1:.4f} | ROC: {avg_roc:.4f} | Acc: {avg_acc:.4f}")

    # Show best model details
    best_model_name, best_model = pipeline.get_best_model(metric='f1')
    print(f"\nBest Model: {best_model_name.upper()}")

    # Evaluate on full dataset
    full_metrics = pipeline.evaluate_model(best_model_name, X, y)

    print(f"\nFull Dataset Performance:")
    print(f"Accuracy:  {full_metrics['accuracy']:.4f}")
    print(f"Precision: {full_metrics['precision']:.4f}")
    print(f"Recall:    {full_metrics['recall']:.4f}")
    print(f"F1 Score:  {full_metrics['f1']:.4f}")
    print(f"ROC AUC:   {full_metrics['roc_auc']:.4f}")

    # Show confusion matrix
    print(f"\nConfusion Matrix:")
    print(full_metrics['confusion_matrix'])


def main():
    """Main function to run the ML pipeline."""
    parser = argparse.ArgumentParser(description="Run RedSentinel ML Pipeline")
    parser.add_argument("--data", "-d",
                        default="training_data/synthetic_attacks.csv",
                        help="Path to training data CSV file")
    parser.add_argument("--models", "-m",
                        nargs="+",
                        default=["gbm", "rf", "xgb"],
                        help="Models to train (gbm, rf, xgb, lgb, lr)")
    parser.add_argument("--cv-folds", "-cv",
                        type=int,
                        default=5,
                        help="Number of cross-validation folds")
    parser.add_argument("--max-features", "-f",
                        type=int,
                        default=5000,
                        help="Maximum TF-IDF features")
    parser.add_argument("--use-embeddings", "-e",
                        action="store_true",
                        help="Use embeddings instead of TF-IDF")
    parser.add_argument("--output-dir", "-o",
                        default="ml_reports",
                        help="Output directory for reports")

    args = parser.parse_args()

    print("RedSentinel ML Pipeline Runner")
    print("=" * 50)
    print(f"Training data: {args.data}")
    print(f"Models: {args.models}")
    print(f"CV folds: {args.cv_folds}")
    print(f"Max TF-IDF features: {args.max_features}")
    print(f"Use embeddings: {args.use_embeddings}")
    print(f"Output directory: {args.output_dir}")

    try:
        # Load data
        df = load_training_data(args.data)

        # Feature extraction
        X, y, extractor = run_feature_extraction(
            df,
            use_embeddings=args.use_embeddings,
            max_tfidf_features=args.max_features
        )

        # ML training
        pipeline = run_ml_training(X, y, args.models, args.cv_folds)

        # Generate reports
        generate_reports(pipeline, extractor, args.output_dir)

        # Analyze results
        analyze_results(pipeline, X, y)

        print("\n" + "="*50)
        print("PIPELINE COMPLETE!")
        print("="*50)
        print(f"Reports saved in: {args.output_dir}")
        print(f"Models saved in: models/")

    except Exception as e:
        print(f"\nError running pipeline: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
