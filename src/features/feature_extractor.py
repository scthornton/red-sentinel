"""
RedSentinel Feature Extraction Module

Converts attack logs into ML-ready datasets with comprehensive
feature engineering for multi-step attacks and model parameters.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Tuple, Optional, List
import json


class RedTeamFeatureExtractor:
    """
    Extracts features from attack logs for ML training.
    Handles numeric, categorical, text, and multi-step features.
    """

    def __init__(self,
                 use_embeddings: bool = False,
                 max_tfidf_features: int = 5000,
                 text_ngram_range: Tuple[int, int] = (1, 2)):
        """
        Initialize the feature extractor.

        Args:
            use_embeddings: If True, use embeddings instead of TF-IDF
            max_tfidf_features: Maximum TF-IDF vocabulary size
            text_ngram_range: N-gram range for text features
        """
        self.use_embeddings = use_embeddings
        self.max_tfidf_features = max_tfidf_features
        self.text_ngram_range = text_ngram_range

        # Initialize transformers
        self.ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.scaler = StandardScaler()
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.max_tfidf_features,
            ngram_range=text_ngram_range,
            stop_words='english'
        )

        # Store fitted state
        self.is_fitted = False
        self.categorical_columns = None
        self.numeric_columns = None

    def fit_transform_categorical(self, df: pd.DataFrame, cat_cols: List[str]) -> pd.DataFrame:
        """Fit and transform categorical features."""
        if not cat_cols:
            return pd.DataFrame(index=df.index)

        cat_features = self.ohe.fit_transform(df[cat_cols])
        cat_feature_names = self.ohe.get_feature_names_out(cat_cols)
        self.categorical_columns = cat_cols

        return pd.DataFrame(cat_features, columns=cat_feature_names, index=df.index)

    def transform_categorical(self, df: pd.DataFrame, cat_cols: List[str]) -> pd.DataFrame:
        """Transform categorical features using fitted encoder."""
        if not cat_cols or not self.is_fitted:
            return pd.DataFrame(index=df.index)

        cat_features = self.ohe.transform(df[cat_cols])
        cat_feature_names = self.ohe.get_feature_names_out(cat_cols)

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

    def transform_numeric(self, df: pd.DataFrame, num_cols: List[str]) -> pd.DataFrame:
        """Transform numeric features using fitted scaler."""
        if not num_cols or not self.is_fitted:
            return pd.DataFrame(index=df.index)

        # Handle missing values
        df_clean = df[num_cols].fillna(df[num_cols].median())

        scaled = self.scaler.transform(df_clean)
        return pd.DataFrame(scaled, columns=num_cols, index=df.index)

    def fit_transform_text(self, text_series: pd.Series) -> pd.DataFrame:
        """Fit and transform text features using TF-IDF."""
        if text_series.empty:
            return pd.DataFrame(index=text_series.index)

        # Clean text
        text_clean = text_series.fillna("").astype(str)

        tfidf = self.tfidf_vectorizer.fit_transform(text_clean)
        feature_names = self.tfidf_vectorizer.get_feature_names_out()

        return pd.DataFrame(tfidf.toarray(), columns=feature_names, index=text_series.index)

    def transform_text(self, text_series: pd.Series) -> pd.DataFrame:
        """Transform text features using fitted TF-IDF vectorizer."""
        if text_series.empty or not self.is_fitted:
            return pd.DataFrame(index=text_series.index)

        # Clean text
        text_clean = text_series.fillna("").astype(str)

        tfidf = self.tfidf_vectorizer.transform(text_clean)
        feature_names = self.tfidf_vectorizer.get_feature_names_out()

        return pd.DataFrame(tfidf.toarray(), columns=feature_names, index=text_series.index)

    def aggregate_multi_step(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate multi-step attack features.

        Args:
            df: DataFrame with attack logs

        Returns:
            Aggregated DataFrame with one row per attack_id
        """
        if 'attack_id' not in df.columns:
            print("Warning: No attack_id column found. Returning original DataFrame.")
            return df

        # Group by attack_id and aggregate
        agg_funcs = {
            'step_confidence': ['max', 'mean', 'sum', 'std'],
            'step_number': ['max', 'count'],
            'step_label': lambda x: int(any(l == 'success' for l in x)),
            'step_reason': lambda x: list(set(x))  # Unique reasons
        }

        # Only aggregate columns that exist
        existing_cols = {col: agg_funcs[col]
                         for col in agg_funcs.keys() if col in df.columns}

        if not existing_cols:
            print("Warning: No aggregatable columns found. Returning original DataFrame.")
            return df

        df_agg = df.groupby('attack_id').agg(existing_cols)

        # Flatten column names
        df_agg.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col
                          for col in df_agg.columns.values]

        # Add derived features
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
        Convert attack logs to ML-ready dataset.

        Args:
            df: DataFrame with attack logs
            target_col: Column name for target variable

        Returns:
            Tuple of (features, target)
        """
        print("Preparing dataset...")

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

        # Prepare features
        X = self._extract_features(df_merged)

        # Mark as fitted
        self.is_fitted = True

        print(f"Final dataset shape: {X.shape}")
        print(f"Target distribution: {y.value_counts().to_dict()}")

        return X, y

    def _extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract all feature types from the merged DataFrame."""
        feature_dfs = []

        # Categorical features
        cat_cols = [col for col in ['model_name', 'model_family', 'technique_category']
                    if col in df.columns]
        if cat_cols:
            X_cat = self.fit_transform_categorical(df, cat_cols)
            feature_dfs.append(X_cat)
            print(f"Added {X_cat.shape[1]} categorical features")

        # Numeric features
        num_cols = [col for col in ['temperature', 'top_p', 'max_tokens', 'presence_penalty',
                                    'frequency_penalty', 'total_steps', 'any_success', 'success_ratio']
                    if col in df.columns]
        if num_cols:
            X_num = self.fit_transform_numeric(df, num_cols)
            feature_dfs.append(X_num)
            print(f"Added {X_num.shape[1]} numeric features")

        # Multi-step aggregate features
        agg_cols = [col for col in df.columns if any(prefix in col for prefix in
                                                     ['step_confidence_', 'step_number_', 'step_reason_'])]
        if agg_cols:
            # Convert step_reason lists to string for processing
            for col in agg_cols:
                if col.startswith('step_reason_'):
                    df[col] = df[col].apply(lambda x: ' '.join(
                        x) if isinstance(x, list) else str(x))

            X_agg = self.fit_transform_text(
                df[agg_cols].astype(str).agg(' '.join, axis=1))
            feature_dfs.append(X_agg)
            print(f"Added {X_agg.shape[1]} aggregate features")

        # Text features (prompt + response)
        text_cols = [col for col in [
            'prompt', 'response'] if col in df.columns]
        if text_cols:
            df['text_feature'] = df[text_cols].fillna(
                '').astype(str).agg(' '.join, axis=1)
            X_text = self.fit_transform_text(df['text_feature'])
            feature_dfs.append(X_text)
            print(f"Added {X_text.shape[1]} text features")

        # Combine all features
        if feature_dfs:
            X = pd.concat(feature_dfs, axis=1)
        else:
            X = pd.DataFrame(index=df.index)

        return X

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
        print(f"Transformers saved to {filepath}")

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

        print(f"Transformers loaded from {filepath}")
