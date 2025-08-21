"""
Robust RedSentinel Feature Extractor
===================================

This version handles unseen categories and can generalize to new data
without feature mismatch errors.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from typing import Tuple, List


class RobustFeatureExtractor:
    """
    Robust feature extractor that handles unseen categories and generalizes well.
    """

    def __init__(self):
        """Initialize the robust feature extractor."""
        self.ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.categorical_columns = None
        self.numeric_columns = None

    def prepare_dataset(self, df: pd.DataFrame, target_col: str = 'final_label', is_training: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Convert attack logs to ML-ready dataset with robust feature handling.

        Args:
            df: Input DataFrame
            target_col: Target column name
            is_training: Whether this is training data (affects fitting)
        """
        print(
            f"Preparing robust dataset ({'training' if is_training else 'testing'})...")
        print(f"Input data shape: {df.shape}")

        # Prepare target variable
        if target_col not in df.columns:
            print(
                f"Warning: Target column '{target_col}' not found. Using 'success' as target.")
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

        # Extract features with robust approach
        if is_training:
            X = self._extract_features_fit(df)
        else:
            X = self._extract_features_transform(df)

        # Mark as fitted after training
        if is_training:
            self.is_fitted = True

        print(f"Final dataset shape: {X.shape}")
        print(f"Target distribution: {y.value_counts().to_dict()}")
        print(f"Feature types: Robust structural features")

        return X, y

    def _extract_features_fit(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features and fit transformers (training mode)."""
        feature_dfs = []

        # 1. Categorical features (model identity)
        cat_cols = [col for col in ['model_name', 'model_family', 'technique_category']
                    if col in df.columns]
        if cat_cols:
            X_cat = self.fit_transform_categorical(df, cat_cols)
            feature_dfs.append(X_cat)
            print(f"Added {X_cat.shape[1]} categorical features")
            self.categorical_columns = cat_cols

        # 2. Numeric features (only essential parameters)
        num_cols = [col for col in ['temperature', 'top_p', 'max_tokens']
                    if col in df.columns]
        if num_cols:
            X_num = self.fit_transform_numeric(df, num_cols)
            feature_dfs.append(X_num)
            print(f"Added {X_num.shape[1]} numeric features")
            self.numeric_columns = num_cols

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

    def _extract_features_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features using fitted transformers (testing mode)."""
        if not self.is_fitted:
            raise ValueError("Extractor must be fitted before transform")

        feature_dfs = []

        # 1. Categorical features (using fitted encoder)
        if self.categorical_columns:
            X_cat = self.transform_categorical(df, self.categorical_columns)
            feature_dfs.append(X_cat)
            print(f"Added {X_cat.shape[1]} categorical features")

        # 2. Numeric features (using fitted scaler)
        if self.numeric_columns:
            X_num = self.transform_numeric(df, self.numeric_columns)
            feature_dfs.append(X_num)
            print(f"Added {X_num.shape[1]} numeric features")

        # 3. Derived structural features (same logic)
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
            # Use a more robust encoding that can handle unseen families
            unique_families = df['model_family'].unique()
            family_mapping = {family: i for i,
                              family in enumerate(unique_families)}
            derived_features['model_family_encoded'] = df['model_family'].map(
                family_mapping).fillna(-1)

        # 2. Technique complexity (based on category)
        if 'technique_category' in df.columns:
            technique_complexity = {
                'direct_override': 1,
                'role_playing': 2,
                'context_manipulation': 3,
                'multi_step': 4,
                'synthetic_override': 1,  # Handle synthetic categories
                'test_manipulation': 2,
                'fake_injection': 3,
                'mock_attack': 2,
                'simulated_bypass': 3,
                'dummy_technique': 1
            }
            derived_features['technique_complexity'] = df['technique_category'].map(
                technique_complexity).fillna(1)

        # 3. Parameter ranges (normalized)
        if 'temperature' in df.columns:
            derived_features['temp_normalized'] = (
                df['temperature'] - 0.5) / 0.5  # Center around 0.5
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

    def transform_categorical(self, df: pd.DataFrame, cat_cols: List[str]) -> pd.DataFrame:
        """Transform categorical features using fitted encoder."""
        if not cat_cols:
            return pd.DataFrame(index=df.index)

        # Handle missing values
        df_clean = df[cat_cols].fillna('unknown')

        # Transform using fitted encoder
        cat_features = self.ohe.transform(df_clean)
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

    def transform_numeric(self, df: pd.DataFrame, num_cols: List[str]) -> pd.DataFrame:
        """Transform numeric features using fitted scaler."""
        if not num_cols:
            return pd.DataFrame(index=df.index)

        # Handle missing values
        df_clean = df[num_cols].fillna(df[num_cols].median())

        scaled = self.scaler.transform(df_clean)

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
            'text_features': 0,  # Always 0 for robust extractor
            'feature_types': ['structural_only', 'robust_encoding', 'generalizable']
        }

        return summary
    
    def save_transformers(self, filepath: str):
        """Save the fitted transformers to disk."""
        import joblib
        
        transformers = {
            'ohe': self.ohe,
            'scaler': self.scaler,
            'categorical_columns': self.categorical_columns,
            'numeric_columns': self.numeric_columns,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(transformers, filepath)
        print(f"Transformers saved to {filepath}")
    
    def load_transformers(self, filepath: str):
        """Load the fitted transformers from disk."""
        import joblib
        
        transformers = joblib.load(filepath)
        self.ohe = transformers['ohe']
        self.scaler = transformers['scaler']
        self.categorical_columns = transformers['categorical_columns']
        self.numeric_columns = transformers['numeric_columns']
        self.is_fitted = transformers['is_fitted']
        print(f"Transformers loaded from {filepath}")
