#!/usr/bin/env python3
"""
Test Converted Data Integration

Tests the converted adapt-ai data with the RedSentinel pipeline
to ensure full compatibility and functionality.
"""

from ml import RedTeamMLPipeline
from features import RedTeamFeatureExtractor
from core import PromptEvaluator, AttackLogger
import sys
import os
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))


def test_data_compatibility():
    """Test if converted data is compatible with RedSentinel components."""
    print("=" * 60)
    print("TESTING CONVERTED DATA COMPATIBILITY")
    print("=" * 60)

    # Load converted data
    csv_path = "converted_data/converted_adapt_ai_data.csv"
    if not os.path.exists(csv_path):
        print(f"Error: Converted data not found at {csv_path}")
        return False

    df = pd.read_csv(csv_path)
    print(f"Loaded converted data: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # Test 1: PromptEvaluator compatibility
    print("\n1. Testing PromptEvaluator compatibility...")
    evaluator = PromptEvaluator()

    # Test a few samples
    sample_responses = df['response'].head(5).tolist()
    for i, response in enumerate(sample_responses):
        result = evaluator.evaluate_response(response)
        print(
            f"  Sample {i+1}: {result['label']} (confidence: {result['confidence']:.2f})")

    # Test 2: Feature extraction compatibility
    print("\n2. Testing Feature Extractor compatibility...")
    try:
        extractor = RedTeamFeatureExtractor(
            use_embeddings=False,
            max_tfidf_features=1000  # Reduced for testing
        )

        # Prepare dataset
        X, y = extractor.prepare_dataset(df)
        print(f"  Feature extraction successful!")
        print(f"  Feature matrix shape: {X.shape}")
        print(f"  Target distribution: {y.value_counts().to_dict()}")

    except Exception as e:
        print(f"  Feature extraction failed: {e}")
        return False

    # Test 3: ML Pipeline compatibility
    print("\n3. Testing ML Pipeline compatibility...")
    try:
        # Ensure balanced sampling for testing
        success_samples = df[df['final_label'] == 'success'].head(500)
        failure_samples = df[df['final_label'] == 'failure'].head(500)
        balanced_df = pd.concat([success_samples, failure_samples])

        # Extract features for balanced dataset
        X_balanced, y_balanced = extractor.prepare_dataset(balanced_df)
        print(f"  Balanced dataset: {X_balanced.shape}")
        print(
            f"  Balanced target distribution: {y_balanced.value_counts().to_dict()}")

        pipeline = RedTeamMLPipeline(
            models=['gbm', 'rf'],  # Test with fewer models
            cv_folds=3,  # Reduced for testing
            random_state=42
        )

        # Set feature names
        pipeline.set_feature_names(list(X_balanced.columns))

        # Train models
        results = pipeline.train_models(X_balanced, y_balanced, test_size=0.2)
        print(f"  ML training successful!")
        print(f"  Trained {len(results)} models")

        # Show best model
        best_model_name, best_model = pipeline.get_best_model(metric='f1')
        print(f"  Best model: {best_model_name.upper()}")

    except Exception as e:
        print(f"  ML training failed: {e}")
        return False

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED! Data is fully compatible.")
    print("=" * 60)
    return True


def analyze_data_quality():
    """Analyze the quality and characteristics of converted data."""
    print("\n" + "=" * 60)
    print("DATA QUALITY ANALYSIS")
    print("=" * 60)

    csv_path = "converted_data/converted_adapt_ai_data.csv"
    df = pd.read_csv(csv_path)

    # Basic statistics
    print(f"Total attacks: {len(df):,}")
    print(f"Success rate: {(df['final_label'] == 'success').mean():.2%}")

    # Model coverage
    print(f"\nModel coverage:")
    model_counts = df['model_name'].value_counts()
    for model, count in model_counts.items():
        print(f"  {model}: {count:,} attacks")

    # Technique distribution
    print(f"\nTechnique distribution:")
    tech_counts = df['technique_category'].value_counts()
    for tech, count in tech_counts.items():
        print(f"  {tech}: {count:,} attacks")

    # Parameter analysis
    print(f"\nParameter analysis:")
    print(
        f"  Temperature range: {df['temperature'].min():.3f} - {df['temperature'].max():.3f}")
    print(f"  Top_p range: {df['top_p'].min():.3f} - {df['top_p'].max():.3f}")
    print(
        f"  Max tokens range: {df['max_tokens'].min()} - {df['max_tokens'].max()}")

    # Response length analysis
    df['response_length'] = df['response'].str.len()
    print(
        f"  Response length range: {df['response_length'].min()} - {df['response_length'].max()} characters")

    # Prompt length analysis
    df['prompt_length'] = df['prompt'].str.len()
    print(
        f"  Prompt length range: {df['prompt_length'].min()} - {df['prompt_length'].max()} characters")

    # Success rate by technique
    print(f"\nSuccess rate by technique:")
    for tech in df['technique_category'].unique():
        tech_data = df[df['technique_category'] == tech]
        success_rate = (tech_data['final_label'] == 'success').mean()
        print(f"  {tech}: {success_rate:.2%} ({len(tech_data):,} attacks)")

    # Success rate by model
    print(f"\nSuccess rate by model:")
    for model in df['model_name'].unique():
        model_data = df[df['model_name'] == model]
        success_rate = (model_data['final_label'] == 'success').mean()
        print(f"  {model}: {success_rate:.2%} ({len(model_data):,} attacks)")


def main():
    """Main test function."""
    print("RedSentinel Converted Data Integration Test")
    print("=" * 60)

    # Test compatibility
    if test_data_compatibility():
        # Analyze data quality
        analyze_data_quality()

        print("\n" + "=" * 60)
        print("INTEGRATION SUCCESSFUL!")
        print("=" * 60)
        print("Your converted adapt-ai data is fully compatible with RedSentinel.")
        print("You can now use this data to train superior ML models!")

    else:
        print("\nIntegration failed. Please check the errors above.")


if __name__ == "__main__":
    main()
