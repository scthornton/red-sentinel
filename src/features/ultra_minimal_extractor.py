"""
Ultra-Minimal RedSentinel Feature Extractor
==========================================

This version eliminates ALL text features to prevent memorization.
Uses only structural features: model type, technique, parameters.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from typing import Tuple, List


class UltraMinimalFeatureExtractor:
    """
    Ultra-minimal feature extractor - NO text features to prevent memorization.
    Focuses only on structural patterns that should generalize.
    """

    def __init__(self):
        """Initialize the ultra-minimal feature extractor."""
        self.ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.scaler = StandardScaler()
        self.is_fitted = False

    def prepare_dataset(self, df: pd.DataFrame, target_col: str = 'final_label') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Convert attack logs to ML-ready dataset with ONLY structural features.
        """
        print("Preparing ultra-minimal dataset (NO text features)...")
        print(f"Input data shape: {df.shape}")

        # Prepare target variable
        if target_col not in df.columns:
            print(f"Warning: Target column '{target_col}' not found. Using 'success' as target.")
            y = (df['final_label'] == 'success').astype(int)
        else:
            # Map labels to numeric
            label_mapping = {
                'success': 1,
                'partial_success': 1,
                'failure': 0,
                'unknown': 0
            }
            y = df[target_col].map(label_mapping).fillna(0)

        # Extract ONLY structural features
        X = self._extract_structural_features(df)

        # Mark as fitted
        self.is_fitted = True

        print(f"Final dataset shape: {X.shape}")
        print(f"Target distribution: {y.value_counts().to_dict()}")
        print(f"Feature types: Structural only (no text)")

        return X, y

    def _extract_structural_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract ONLY structural features - no text, no memorization.
        """
        feature_dfs = []

        # 1. Categorical features (model identity)
        cat_cols = [col for col in ['model_name', 'model_family', 'technique_category']
                    if col in df.columns]
        if cat_cols:
            X_cat = self.fit_transform_categorical(df, cat_cols)
            feature_dfs.append(X_cat)
            print(f"Added {X_cat.shape[1]} categorical features")

        # 2. Numeric features (only essential parameters)
        num_cols = [col for col in ['temperature', 'top_p', 'max_tokens']
                    if col in df.columns]
        if num_cols:
            X_num = self.fit_transform_numeric(df, num_cols)
            feature_dfs.append(X_num)
            print(f"Added {X_num.shape[1]} numeric features")

        # 3. Derived structural features (attack patterns)
        X_derived = self._create_derived_features(df)
        if not X_derived.empty:
            feature_dfs.append(X_derived)
            print(f"Added {X_derived.shape[1]} derived structural features")

        # Combine all features
        if feature_dfs:
            X = pd.concat(feature_dfs, axis=1)
        else:
            X = pd.DataFrame(index=df.index)

        return X

    def _create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create derived features that capture attack patterns without memorization.
        """
        derived_features = pd.DataFrame(index=df.index)

        # 1. Model family encoding (numeric)
        if 'model_family' in df.columns:
            family_mapping = {
                'gpt_like': 1,
                'claude_like': 2,
                'gemini_like': 3,
                'other': 4
            }
            derived_features['model_family_encoded'] = df['model_family'].map(
                family_mapping).fillna(4)

        # 2. Technique complexity (based on category)
        if 'technique_category' in df.columns:
            technique_complexity = {
                'direct_override': 1,
                'role_playing': 2,
                'context_manipulation': 3,
                'multi_step': 4,
                'other': 1
            }
            derived_features['technique_complexity'] = df['technique_category'].map(
                technique_complexity).fillna(1)

        # 3. Parameter ranges (normalized)
        if 'temperature' in df.columns:
            derived_features['temp_normalized'] = (df['temperature'] - 0.5) / 0.5  # Center around 0.5
        if 'top_p' in df.columns:
            derived_features['top_p_normalized'] = (df['top_p'] - 0.5) / 0.5

        # 4. Attack success patterns (if available)
        if 'final_label' in df.columns:
            # Create a feature that captures the pattern of success/failure
            # without directly using the target
            success_rate = df['final_label'].value_counts(normalize=True)
            if 'success' in success_rate:
                derived_features['success_rate_context'] = success_rate['success']
            else:
                derived_features['success_rate_context'] = 0.5

        return derived_features

    def fit_transform_categorical(self, df: pd.DataFrame, cat_cols: List[str]) -> pd.DataFrame:
        """Fit and transform categorical features."""
        if not cat_cols:
            return pd.DataFrame(index=df.index)

        # Handle missing values
        df_clean = df[cat_cols].fillna('unknown')
        
        cat_features = self.ohe.fit_transform(df_clean)
        cat_feature_names = self.ohe.get_feature_names_out(cat_cols)

        return pd.DataFrame(cat_features, columns=cat_feature_names, index=df.index)

    def fit_transform_numeric(self, df: pd.DataFrame, num_cols: List[str]) -> pd.DataFrame:
        """Fit and transform numeric features."""
        if not num_cols:
            return pd.DataFrame(index=df.index)

        # Handle missing values
        df_clean = df[num_cols].fillna(df[num_cols].median())

        scaled = self.scaler.fit_transform(df_clean)

        return pd.DataFrame(scaled, columns=num_cols, index=df.index)

    def get_feature_names(self) -> List[str]:
        """Get names of all extracted features."""
        if not self.is_fitted:
            return []

        feature_names = []

        if hasattr(self.ohe, 'get_feature_names_out'):
            feature_names.extend(self.ohe.get_feature_names_out())

        if hasattr(self.scaler, 'get_feature_names_out'):
            feature_names.extend(self.scaler.get_feature_names_out())

        return feature_names

    def get_feature_summary(self) -> dict:
        """Get a summary of the features extracted."""
        if not self.is_fitted:
            return {}

        summary = {
            'total_features': len(self.get_feature_names()),
            'categorical_features': len(self.ohe.get_feature_names_out()) if hasattr(self.ohe, 'get_feature_names_out') else 0,
            'numeric_features': len(self.scaler.get_feature_names_out()) if hasattr(self.scaler, 'get_feature_names_out') else 0,
            'text_features': 0,  # Always 0 for ultra-minimal
            'feature_types': ['structural_only', 'no_text', 'no_memorization']
        }

        return summary
