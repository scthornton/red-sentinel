"""
Simplified RedSentinel Feature Extractor
========================================

This version addresses overfitting by:
1. Reducing feature count from 5,048 to ~200-300
2. Focusing on most important features
3. Avoiding text memorization
4. Creating more generalizable patterns
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Tuple, Optional, List
import json


class SimplifiedFeatureExtractor:
    """
    Simplified feature extractor to address overfitting issues.
    Focuses on core features rather than memorizing specific patterns.
    """

    def __init__(self,
                 max_tfidf_features: int = 100,  # Reduced from 5000
                 # Reduced from (1,2)
                 text_ngram_range: Tuple[int, int] = (1, 1)):
        """
        Initialize the simplified feature extractor.

        Args:
            max_tfidf_features: Maximum TF-IDF vocabulary size (reduced)
            text_ngram_range: N-gram range for text features (simplified)
        """
        self.max_tfidf_features = max_tfidf_features
        self.text_ngram_range = text_ngram_range

        # Initialize transformers
        self.ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.scaler = StandardScaler()
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.max_tfidf_features,
            ngram_range=text_ngram_range,
            stop_words='english',
            min_df=5,  # Only include terms that appear in at least 5 documents
            max_df=0.8  # Exclude terms that appear in more than 80% of documents
        )

        # Store fitted state
        self.is_fitted = False
        self.categorical_columns = None
        self.numeric_columns = None

    def aggregate_multi_step(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Simplified multi-step aggregation focusing on key metrics.
        """
        if 'step_number' not in df.columns:
            print("Warning: No step_number column found. Returning original DataFrame.")
            return df

        # Only aggregate essential columns to reduce feature explosion
        essential_cols = ['attack_id', 'step_number',
                          'step_label', 'step_confidence']
        existing_cols = [col for col in essential_cols if col in df.columns]

        if not existing_cols:
            print("Warning: No aggregatable columns found. Returning original DataFrame.")
            return df

        df_agg = df.groupby('attack_id').agg(existing_cols)

        # Flatten column names
        df_agg.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col
                          for col in df_agg.columns.values]

        # Add only essential derived features
        df_agg['total_steps'] = df.groupby('attack_id')['step_number'].max()
        df_agg['any_success'] = df.groupby('attack_id')['step_label'].apply(
            lambda x: int(any(l == 'success' for l in x))
        )
        df_agg['success_ratio'] = df.groupby('attack_id')['step_label'].apply(
            lambda x: sum(1 for l in x if l == 'success') /
            len(x) if len(x) > 0 else 0
        )

        return df_agg.reset_index()

    def prepare_dataset(self, df: pd.DataFrame, target_col: str = 'final_label') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Convert attack logs to ML-ready dataset with simplified features.
        """
        print("Preparing simplified dataset...")

        # Aggregate multi-step features
        df_agg = self.aggregate_multi_step(df)
        print(f"Aggregated {len(df)} rows to {len(df_agg)} unique attacks")

        # Merge aggregated features back for other columns
        first_rows = df.groupby('attack_id').first().reset_index()
        merge_cols = ['attack_id', 'model_name', 'model_family', 'technique_category',
                      'temperature', 'top_p', 'max_tokens', 'presence_penalty',
                      'frequency_penalty', 'prompt', 'response', 'final_label']

        # Only merge columns that exist
        existing_merge_cols = [
            col for col in merge_cols if col in first_rows.columns]

        df_merged = df_agg.merge(
            first_rows[existing_merge_cols],
            on='attack_id',
            how='left'
        )

        # Prepare target variable
        if target_col not in df_merged.columns:
            print(
                f"Warning: Target column '{target_col}' not found. Using 'any_success' as target.")
            y = df_merged['any_success']
        else:
            # Map labels to numeric
            label_mapping = {
                'success': 1,
                'partial_success': 1,
                'failure': 0,
                'unknown': 0
            }
            y = df_merged[target_col].map(label_mapping).fillna(0)

        # Prepare features with simplified approach
        X = self._extract_simplified_features(df_merged)

        # Mark as fitted
        self.is_fitted = True

        print(f"Final dataset shape: {X.shape}")
        print(f"Target distribution: {y.value_counts().to_dict()}")

        return X, y

    def _extract_simplified_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract simplified features focusing on core patterns.
        """
        feature_dfs = []

        # 1. Categorical features (most important)
        cat_cols = [col for col in ['model_name', 'model_family', 'technique_category']
                    if col in df.columns]
        if cat_cols:
            X_cat = self.fit_transform_categorical(df, cat_cols)
            feature_dfs.append(X_cat)
            print(f"Added {X_cat.shape[1]} categorical features")

        # 2. Numeric features (core parameters)
        num_cols = [col for col in ['temperature', 'top_p', 'max_tokens', 'total_steps', 'any_success', 'success_ratio']
                    if col in df.columns]
        if num_cols:
            X_num = self.fit_transform_numeric(df, num_cols)
            feature_dfs.append(X_num)
            print(f"Added {X_num.shape[1]} numeric features")

        # 3. Simplified text features (reduced vocabulary)
        text_cols = [col for col in [
            'prompt', 'response'] if col in df.columns]
        if text_cols:
            # Create simplified text features
            df['text_feature'] = df[text_cols].fillna(
                '').astype(str).agg(' '.join, axis=1)

            # Extract only the most important text patterns
            X_text = self.fit_transform_simplified_text(df['text_feature'])
            feature_dfs.append(X_text)
            print(f"Added {X_text.shape[1]} simplified text features")

        # 4. Basic aggregate features (limited set)
        if 'step_confidence' in df.columns:
            # Only basic confidence statistics - handle the groupby properly
            try:
                df['avg_confidence'] = df.groupby(
                    'attack_id')['step_confidence'].transform('mean')
                df['max_confidence'] = df.groupby(
                    'attack_id')['step_confidence'].transform('max')

                confidence_features = df[[
                    'avg_confidence', 'max_confidence']].fillna(0)
                feature_dfs.append(confidence_features)
                print(
                    f"Added {confidence_features.shape[1]} confidence features")
            except Exception as e:
                print(f"Warning: Could not create confidence features: {e}")
                # Skip confidence features if there's an issue

        # Combine all features
        if feature_dfs:
            X = pd.concat(feature_dfs, axis=1)
        else:
            X = pd.DataFrame(index=df.index)

        return X

    def fit_transform_categorical(self, df: pd.DataFrame, cat_cols: List[str]) -> pd.DataFrame:
        """Fit and transform categorical features."""
        if not cat_cols:
            return pd.DataFrame(index=df.index)

        cat_features = self.ohe.fit_transform(df[cat_cols])
        cat_feature_names = self.ohe.get_feature_names_out(cat_cols)
        self.categorical_columns = cat_cols

        return pd.DataFrame(cat_features, columns=cat_feature_names, index=df.index)

    def fit_transform_numeric(self, df: pd.DataFrame, num_cols: List[str]) -> pd.DataFrame:
        """Fit and transform numeric features."""
        if not num_cols:
            return pd.DataFrame(index=df.index)

        # Handle missing values
        df_clean = df[num_cols].fillna(df[num_cols].median())

        scaled = self.scaler.fit_transform(df_clean)
        self.numeric_columns = num_cols

        return pd.DataFrame(scaled, columns=num_cols, index=df.index)

    def fit_transform_simplified_text(self, text_series: pd.Series) -> pd.DataFrame:
        """Fit and transform text features with simplified approach."""
        if text_series.empty:
            return pd.DataFrame(index=text_series.index)

        # Use simplified TF-IDF with reduced features
        text_features = self.tfidf_vectorizer.fit_transform(text_series)
        feature_names = self.tfidf_vectorizer.get_feature_names_out()

        return pd.DataFrame(text_features.toarray(), columns=feature_names, index=text_series.index)

    def get_feature_names(self) -> List[str]:
        """Get names of all extracted features."""
        if not self.is_fitted:
            return []

        feature_names = []

        if self.categorical_columns:
            feature_names.extend(
                self.ohe.get_feature_names_out(self.categorical_columns))

        if self.numeric_columns:
            feature_names.extend(self.numeric_columns)

        if hasattr(self.tfidf_vectorizer, 'get_feature_names_out'):
            feature_names.extend(self.tfidf_vectorizer.get_feature_names_out())

        return feature_names

    def save_transformers(self, filepath: str):
        """Save fitted transformers for later use."""
        import joblib

        transformers = {
            'ohe': self.ohe,
            'scaler': self.scaler,
            'tfidf': self.tfidf_vectorizer,
            'categorical_columns': self.categorical_columns,
            'numeric_columns': self.numeric_columns,
            'is_fitted': self.is_fitted
        }

        joblib.dump(transformers, filepath)
        print(f"Simplified transformers saved to {filepath}")

    def load_transformers(self, filepath: str):
        """Load fitted transformers."""
        import joblib

        transformers = joblib.load(filepath)

        self.ohe = transformers['ohe']
        self.scaler = transformers['scaler']
        self.tfidf_vectorizer = transformers['tfidf']
        self.categorical_columns = transformers['categorical_columns']
        self.numeric_columns = transformers['numeric_columns']
        self.is_fitted = transformers['is_fitted']

        print(f"Simplified transformers loaded from {filepath}")
