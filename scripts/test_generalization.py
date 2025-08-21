#!/usr/bin/env python3
"""
Test Generalization Capability
==============================

This script tests if the simplified feature extractors can generalize
to completely different data, which is the true test of overfitting.
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from features.simple_extractor import SimpleFeatureExtractor
from features.ultra_minimal_extractor import UltraMinimalFeatureExtractor
from features.robust_extractor import RobustFeatureExtractor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def generate_synthetic_test_data(n_samples: int = 1000) -> pd.DataFrame:
    """
    Generate completely different synthetic data to test generalization.
    This data has different patterns than the training data.
    """
    print(f"🔧 Generating {n_samples} synthetic test samples...")
    
    # Different model names
    synthetic_models = [
        'synthetic-gpt-5', 'synthetic-claude-4', 'synthetic-gemini-2',
        'test-model-alpha', 'test-model-beta', 'experimental-llm'
    ]
    
    # Different technique categories
    synthetic_techniques = [
        'synthetic_override', 'test_manipulation', 'fake_injection',
        'mock_attack', 'simulated_bypass', 'dummy_technique'
    ]
    
    # Different parameter ranges
    synthetic_temps = np.random.uniform(0.1, 0.9, n_samples)
    synthetic_top_p = np.random.uniform(0.1, 0.9, n_samples)
    synthetic_max_tokens = np.random.randint(100, 2000, n_samples)
    
    # Different responses (completely different text)
    synthetic_responses = [
        f"This is synthetic response {i} with different patterns than training data"
        for i in range(n_samples)
    ]
    
    # Create synthetic DataFrame
    synthetic_data = pd.DataFrame({
        'attack_id': [f'synthetic_{i}' for i in range(n_samples)],
        'model_name': np.random.choice(synthetic_models, n_samples),
        'model_family': np.random.choice(['synthetic_gpt', 'synthetic_claude', 'synthetic_gemini'], n_samples),
        'technique_category': np.random.choice(synthetic_techniques, n_samples),
        'temperature': synthetic_temps,
        'top_p': synthetic_top_p,
        'max_tokens': synthetic_max_tokens,
        'response': synthetic_responses,
        'final_label': np.random.choice(['success', 'failure'], n_samples, p=[0.7, 0.3])  # Different distribution
    })
    
    print(f"✅ Generated synthetic data with {len(synthetic_data)} samples")
    print(f"   Models: {synthetic_data['model_name'].nunique()} unique")
    print(f"   Techniques: {synthetic_data['technique_category'].nunique()} unique")
    print(f"   Label distribution: {synthetic_data['final_label'].value_counts().to_dict()}")
    
    return synthetic_data

def test_generalization():
    """
    Test if the simplified feature extractors can generalize to new data.
    """
    print("=" * 70)
    print("TESTING GENERALIZATION CAPABILITY")
    print("=" * 70)
    print()
    
    # Check if converted data exists
    data_path = "converted_data/converted_adapt_ai_data.csv"
    if not os.path.exists(data_path):
        print("❌ Converted data not found. Please run convert_adapt_ai_data.py first.")
        return
    
    # Load real training data
    print("📊 Loading real training data...")
    real_df = pd.read_csv(data_path)
    print(f"Loaded {len(real_df)} real attack records")
    print()
    
    # Generate synthetic test data
    synthetic_df = generate_synthetic_test_data(1000)
    print()
    
    # Test all feature extractors
    extractors = {
        'Simple': SimpleFeatureExtractor(),
        'Ultra-Minimal': UltraMinimalFeatureExtractor(),
        'Robust': RobustFeatureExtractor()
    }
    
    results = {}
    
    for name, extractor in extractors.items():
        print(f"🔧 Testing {name} Feature Extractor...")
        print("-" * 50)
        
        try:
            # Extract features from real data (training)
            X_real, y_real = extractor.prepare_dataset(real_df, is_training=True)
            print(f"   Real data features: {X_real.shape}")
            
            # Extract features from synthetic data (testing)
            X_synthetic, y_synthetic = extractor.prepare_dataset(synthetic_df, is_training=False)
            print(f"   Synthetic data features: {X_synthetic.shape}")
            
            # Train on real data
            print("   Training on real data...")
            X_train, X_val, y_train, y_val = train_test_split(
                X_real, y_real, test_size=0.2, random_state=42, stratify=y_real
            )
            
            rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
            rf.fit(X_train, y_train)
            
            # Test on real validation data
            y_val_pred = rf.predict(X_val)
            val_f1 = f1_score(y_val, y_val_pred, zero_division=0)
            val_acc = accuracy_score(y_val, y_val_pred)
            
            print(f"   Real data validation - F1: {val_f1:.3f}, Accuracy: {val_acc:.3f}")
            
            # Test on synthetic data (generalization test)
            y_synthetic_pred = rf.predict(X_synthetic)
            synthetic_f1 = f1_score(y_synthetic, y_synthetic_pred, zero_division=0)
            synthetic_acc = accuracy_score(y_synthetic, y_synthetic_pred)
            
            print(f"   Synthetic data test - F1: {synthetic_f1:.3f}, Accuracy: {synthetic_acc:.3f}")
            
            # Calculate generalization gap
            f1_gap = val_f1 - synthetic_f1
            acc_gap = val_acc - synthetic_acc
            
            print(f"   Generalization gap - F1: {f1_gap:+.3f}, Accuracy: {acc_gap:+.3f}")
            
            # Assess generalization quality
            if f1_gap < 0.1:
                generalization_quality = "Excellent"
            elif f1_gap < 0.2:
                generalization_quality = "Good"
            elif f1_gap < 0.3:
                generalization_quality = "Fair"
            else:
                generalization_quality = "Poor"
            
            print(f"   Generalization quality: {generalization_quality}")
            
            # Store results
            results[name] = {
                'real_f1': val_f1,
                'real_acc': val_acc,
                'synthetic_f1': synthetic_f1,
                'synthetic_acc': synthetic_acc,
                'f1_gap': f1_gap,
                'acc_gap': acc_gap,
                'generalization_quality': generalization_quality,
                'feature_count': X_real.shape[1]
            }
            
        except Exception as e:
            print(f"   ❌ {name} extractor failed: {e}")
            results[name] = {'error': str(e)}
        
        print()
    
    # Summary and analysis
    print("=" * 70)
    print("GENERALIZATION TEST RESULTS")
    print("=" * 70)
    print()
    
    for name, result in results.items():
        if 'error' in result:
            print(f"❌ {name}: Failed - {result['error']}")
            continue
            
        print(f"📊 {name} Feature Extractor:")
        print(f"   Features: {result['feature_count']}")
        print(f"   Real data performance: F1={result['real_f1']:.3f}, Acc={result['real_acc']:.3f}")
        print(f"   Synthetic data performance: F1={result['synthetic_f1']:.3f}, Acc={result['synthetic_acc']:.3f}")
        print(f"   Generalization gap: F1={result['f1_gap']:+.3f}, Acc={result['acc_gap']:+.3f}")
        print(f"   Quality: {result['generalization_quality']}")
        print()
    
    # Overall assessment
    print("🎯 OVERALL ASSESSMENT:")
    
    if all('error' not in result for result in results.values()):
        best_extractor = min(results.keys(), key=lambda x: results[x]['f1_gap'])
        best_gap = results[best_extractor]['f1_gap']
        
        if best_gap < 0.1:
            print("✅ Excellent generalization - overfitting likely addressed!")
        elif best_gap < 0.2:
            print("✅ Good generalization - overfitting significantly reduced")
        elif best_gap < 0.3:
            print("⚠️  Fair generalization - some overfitting remains")
        else:
            print("❌ Poor generalization - overfitting still significant")
        
        print(f"   Best extractor: {best_extractor} (gap: {best_gap:.3f})")
    else:
        print("❌ Some extractors failed - system needs debugging")
    
    print()
    print("Next Steps:")
    print("1. If generalization is good, document the success")
    print("2. If generalization is poor, continue refining")
    print("3. Test against more diverse real-world data")
    print("4. Document the learning process")

if __name__ == "__main__":
    test_generalization()
