#!/usr/bin/env python3
"""
Save Robust Model for Production
===============================

Simple script to save our robust model for production testing.
"""

import sys
import os
import joblib
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from features.robust_extractor import RobustFeatureExtractor
from sklearn.ensemble import RandomForestClassifier

def save_robust_model():
    """Save a robust model for production testing."""
    print("Saving robust model for production...")
    
    # Load the converted data
    data_path = "converted_data/converted_adapt_ai_data.csv"
    if not Path(data_path).exists():
        print(f"❌ Data not found: {data_path}")
        return
    
    import pandas as pd
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} samples")
    
    # Initialize robust extractor
    extractor = RobustFeatureExtractor()
    
    # Extract features
    X, y = extractor.prepare_dataset(df, is_training=True)
    print(f"Features extracted: {X.shape}")
    
    # Train a simple but robust model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    # Train the model
    print("Training robust model...")
    model.fit(X, y)
    
    # Evaluate
    y_pred = model.predict(X)
    from sklearn.metrics import accuracy_score, f1_score
    accuracy = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    
    print(f"Model performance:")
    print(f"  Accuracy: {accuracy:.3f}")
    print(f"  F1 Score: {f1:.3f}")
    
    # Save the model
    os.makedirs("models", exist_ok=True)
    model_path = "models/robust_model.joblib"
    joblib.dump(model, model_path)
    print(f"✅ Model saved to: {model_path}")
    
    # Save transformers
    transformers_path = "models/robust_model_transformers.joblib"
    extractor.save_transformers(transformers_path)
    print(f"✅ Transformers saved to: {transformers_path}")
    
    print("\n🎯 Robust model ready for production!")

if __name__ == "__main__":
    save_robust_model()
