#!/usr/bin/env python3
"""
Test Simplified Feature Extractor
================================

This script tests the simplified feature extractor to see if it addresses
the overfitting issues by reducing feature count and improving generalization.
"""

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from features.simple_extractor import SimpleFeatureExtractor
import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))


def test_simplified_features():
    """
    Test the simplified feature extractor against the original.
    """
    print("=" * 60)
    print("TESTING SIMPLIFIED FEATURE EXTRACTOR")
    print("=" * 60)
    print()

    # Check if converted data exists
    data_path = "converted_data/converted_adapt_ai_data.csv"
    if not os.path.exists(data_path):
        print("❌ Converted data not found. Please run convert_adapt_ai_data.py first.")
        return

    # Load data
    print("📊 Loading converted data...")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} attack records")
    print(f"Data shape: {df.shape}")
    print(f"Label distribution: {df['final_label'].value_counts().to_dict()}")
    print()

    # Test simple feature extractor
    print("🔧 Testing Simple Feature Extractor...")
    simple_extractor = SimpleFeatureExtractor()

    try:
        # Extract simple features
        X_simple, y_simple = simple_extractor.prepare_dataset(df)

        print(f"✅ Simple features extracted successfully!")
        print(f"   Feature matrix shape: {X_simple.shape}")
        print(f"   Target distribution: {y_simple.value_counts().to_dict()}")
        print()

        # Compare with original feature count
        original_count = 5048
        simple_count = X_simple.shape[1]
        reduction = ((original_count - simple_count) / original_count) * 100

        print(f"📊 FEATURE REDUCTION ACHIEVED:")
        print(f"   Original features: {original_count}")
        print(f"   Simple features: {simple_count}")
        print(f"   Reduction: {reduction:.1f}%")
        print()

        # Test ML performance with simple features
        print("🤖 Testing ML Performance with Simple Features...")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_simple, y_simple, test_size=0.3, random_state=42, stratify=y_simple
        )

        print(f"   Training set: {X_train.shape[0]} samples")
        print(f"   Test set: {X_test.shape[0]} samples")
        print()

        # Train a simple Random Forest
        rf = RandomForestClassifier(
            n_estimators=100, random_state=42, max_depth=10)
        rf.fit(X_train, y_train)

        # Predict on test set
        y_pred = rf.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)

        print(f"📈 SIMPLE FEATURES PERFORMANCE:")
        print(f"   Accuracy: {accuracy:.3f}")
        print(f"   F1 Score: {f1:.3f}")
        print(f"   Precision: {precision:.3f}")
        print(f"   Recall: {recall:.3f}")
        print()

        # Assess if overfitting is reduced
        if f1 > 0.95:
            print("⚠️  WARNING: Still getting suspiciously high performance")
            print("   This may indicate the simplification wasn't sufficient")
        elif f1 > 0.90:
            print("✅ Good performance - reasonable for this task")
            print("   Overfitting may be reduced")
        elif f1 > 0.80:
            print("✅ Solid performance - shows real ML capability")
            print("   Overfitting appears to be addressed")
        else:
            print("📊 Room for improvement - realistic assessment")
            print("   Overfitting is likely addressed")

        print()

        # Feature importance analysis
        feature_importance = pd.DataFrame({
            'feature': X_simple.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)

        print("🔍 TOP 10 MOST IMPORTANT FEATURES:")
        for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
            print(f"   {i+1:2d}. {row['feature']}: {row['importance']:.4f}")

        print()

        # Save simple features for comparison
        output_path = "simple_features_test.csv"
        X_simple.to_csv(output_path, index=False)
        print(f"💾 Simple features saved to: {output_path}")

    except Exception as e:
        print(f"❌ Simplified feature extraction failed: {e}")
        print("This shows the system still has implementation issues.")
        return

    print("=" * 60)
    print("SIMPLIFIED FEATURE TEST COMPLETE")
    print("=" * 60)
    print()
    print("Key Findings:")
    print(f"✅ Feature count reduced from {original_count} to {simple_count}")
    print(f"✅ ML performance: F1 = {f1:.3f}")
    print(
        f"✅ Overfitting assessment: {'Reduced' if f1 < 0.95 else 'Still Present'}")
    print()
    print("Next Steps:")
    print("1. Analyze if performance is now realistic")
    print("2. Test against different attack patterns")
    print("3. Document the improvement process")
    print("4. Continue refining the approach")


if __name__ == "__main__":
    test_simplified_features()
