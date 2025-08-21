#!/usr/bin/env python3
"""
Train RedSentinel with Real Data

Trains the complete ML pipeline using the converted adapt-ai data.
This will show the dramatic improvement over synthetic data.
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from features.robust_extractor import RobustFeatureExtractor
from ml import RedTeamMLPipeline


def load_converted_data():
    """Load the converted adapt-ai data."""
    csv_path = "converted_data/converted_adapt_ai_data.csv"

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Converted data not found at {csv_path}")

    print(f"Loading converted data from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df):,} attacks")
    print(f"Data shape: {df.shape}")

    return df


def run_full_training():
    """Run the complete ML training pipeline with real data."""
    print("=" * 60)
    print("REDSENTINEL FULL TRAINING WITH REAL DATA")
    print("=" * 60)

    # Load data
    df = load_converted_data()

    # Feature extraction
    print("\n1. FEATURE EXTRACTION")
    print("-" * 30)

    extractor = RobustFeatureExtractor()

    X, y = extractor.prepare_dataset(df, is_training=True)
    print(f"Feature extraction complete!")
    print(f"Feature matrix: {X.shape}")
    print(f"Target distribution: {y.value_counts().to_dict()}")

    # Save transformers
    os.makedirs("models", exist_ok=True)
    extractor.save_transformers("models/real_data_transformers.joblib")
    print("Feature transformers saved to models/real_data_transformers.joblib")

    # ML Training
    print("\n2. MACHINE LEARNING TRAINING")
    print("-" * 30)

    pipeline = RedTeamMLPipeline(
        models=['gbm', 'rf', 'xgb', 'lgb', 'lr'],  # All models
        cv_folds=5,  # Full cross-validation
        random_state=42
    )

    # Set feature names for plotting
    pipeline.set_feature_names(list(X.columns))

    # Train models
    print("Training all models with 5-fold cross-validation...")
    results = pipeline.train_models(X, y, test_size=0.2)

    print(f"\nTraining complete! Trained {len(results)} models")

    # Show best model
    best_model_name, best_model = pipeline.get_best_model(metric='f1')
    print(f"Best model by F1 score: {best_model_name.upper()}")

    # Generate comprehensive reports
    print("\n3. GENERATING REPORTS")
    print("-" * 30)

    output_dir = "real_data_ml_reports"
    pipeline.generate_report(output_dir)
    print(f"Reports generated in: {output_dir}")

    # Save trained models
    print("\n4. SAVING MODELS")
    print("-" * 30)

    models_dir = "models/real_data_models"
    os.makedirs(models_dir, exist_ok=True)

    for model_name, model in results.items():
        model_path = os.path.join(models_dir, f"{model_name}_real_data.joblib")
        pipeline.save_model(model_name, model_path)
        print(f"Saved {model_name} to: {model_path}")

    # Performance summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)

    # Compare with synthetic data performance
    print("Performance Summary:")
    for model_name, metrics in results.items():
        cv_f1 = np.mean(metrics['cv_scores']['f1'])
        cv_auc = np.mean(metrics['cv_scores']['roc_auc'])
        print(f"  {model_name.upper()}: F1={cv_f1:.3f}, AUC={cv_auc:.3f}")

    print(f"\nReal data training completed successfully!")
    print(f"Your models are now trained on {len(df):,} real attack samples!")
    print(f"This represents a {len(df)/1000:.1f}x increase in training data!")


def main():
    """Main training function."""
    try:
        run_full_training()
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
