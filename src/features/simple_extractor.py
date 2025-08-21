"""
Simple RedSentinel Feature Extractor
===================================

A minimal feature extractor that works with the actual data structure
and focuses on core features to avoid overfitting.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Tuple, List


class SimpleFeatureExtractor:
    """
    Simple feature extractor focusing on core features only.
    """

    def __init__(self, max_tfidf_features: int = 50):
        """
        Initialize the simple feature extractor.

        Args:
            max_tfidf_features: Maximum TF-IDF vocabulary size (very limited)
        """
        self.max_tfidf_features = max_tfidf_features

        # Initialize transformers
        self.ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.scaler = StandardScaler()
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.max_tfidf_features,
            ngram_range=(1, 1),  # Only unigrams
            stop_words='english',
            min_df=10,  # Only include terms that appear in at least 10 documents
            max_df=0.7  # Exclude terms that appear in more than 70% of documents
        )

        # Store fitted state
        self.is_fitted = False

    def prepare_dataset(self, df: pd.DataFrame, target_col: str = 'final_label') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Convert attack logs to ML-ready dataset with minimal features.
        """
        print("Preparing simple dataset...")
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

        # Prepare features with minimal approach
        X = self._extract_simple_features(df)

        # Mark as fitted
        self.is_fitted = True

        print(f"Final dataset shape: {X.shape}")
        print(f"Target distribution: {y.value_counts().to_dict()}")

        return X, y

    def _extract_simple_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract only the most essential features.
        """
        feature_dfs = []

        # 1. Categorical features (model info)
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

        # 3. Very limited text features (only most common patterns)
        if 'response' in df.columns:
            # Only use response text, not prompts (to avoid memorization)
            X_text = self.fit_transform_simple_text(df['response'])
            feature_dfs.append(X_text)
            print(f"Added {X_text.shape[1]} simple text features")

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

    def fit_transform_simple_text(self, text_series: pd.Series) -> pd.DataFrame:
        """Fit and transform text features with very limited vocabulary."""
        if text_series.empty:
            return pd.DataFrame(index=text_series.index)

        # Clean text data
        text_clean = text_series.fillna('').astype(str)

        # Use very limited TF-IDF
        text_features = self.tfidf_vectorizer.fit_transform(text_clean)
        feature_names = self.tfidf_vectorizer.get_feature_names_out()

        return pd.DataFrame(text_features.toarray(), columns=feature_names, index=text_series.index)

    def get_feature_names(self) -> List[str]:
        """Get names of all extracted features."""
        if not self.is_fitted:
            return []

        feature_names = []

        if hasattr(self.ohe, 'get_feature_names_out'):
            feature_names.extend(self.ohe.get_feature_names_out())

        if hasattr(self.scaler, 'get_feature_names_out'):
            feature_names.extend(self.scaler.get_feature_names_out())

        if hasattr(self.tfidf_vectorizer, 'get_feature_names_out'):
            feature_names.extend(self.tfidf_vectorizer.get_feature_names_out())

        return feature_names
