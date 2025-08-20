#!/usr/bin/env python3
"""
Debug Results Structure

Examine the actual structure of ML pipeline results to fix the comparison script.
"""

import sys
import os
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from features import RedTeamFeatureExtractor
from ml import RedTeamMLPipeline


def debug_results_structure():
    """Debug the structure of ML pipeline results."""
    print("Debugging ML pipeline results structure...")
    
    # Load a small dataset for testing
    from pathlib import Path
    import pandas as pd
    
    # Load synthetic data
    synthetic_path = "training_data/synthetic_attacks.csv"
    if os.path.exists(synthetic_path):
        synthetic_df = pd.read_csv(synthetic_path)
        print(f"Loaded synthetic data: {synthetic_df.shape}")
        
        # Use a small subset for debugging
        test_df = synthetic_df.head(100)
        
        # Feature extraction
        extractor = RedTeamFeatureExtractor(
            use_embeddings=False,
            max_tfidf_features=100
        )
        
        X, y = extractor.prepare_dataset(test_df)
        print(f"Feature matrix: {X.shape}")
        
        # Train a simple model
        pipeline = RedTeamMLPipeline(
            models=['gbm'],
            cv_folds=3,
            random_state=42
        )
        
        pipeline.set_feature_names(list(X.columns))
        
        # Train and examine results
        results = pipeline.train_models(X, y, test_size=0.3)
        
        print("\nResults structure:")
        print(f"Type: {type(results)}")
        print(f"Keys: {list(results.keys())}")
        
        if 'gbm' in results:
            gbm_result = results['gbm']
            print(f"\nGBM result type: {type(gbm_result)}")
            print(f"GBM result keys: {list(gbm_result.keys()) if hasattr(gbm_result, 'keys') else 'No keys'}")
            
            # Print the full structure
            print(f"\nFull GBM result:")
            print(json.dumps(gbm_result, indent=2, default=str))
        
    else:
        print("Synthetic data not found")


if __name__ == "__main__":
    debug_results_structure()
