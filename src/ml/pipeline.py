"""
RedSentinel ML Pipeline

Complete machine learning pipeline for training models on red team data.
Supports multiple algorithms, cross-validation, and comprehensive evaluation.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb
import lightgbm as lgb
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os


class RedTeamMLPipeline:
    """
    Complete ML pipeline for training models on red team attack data.
    """

    def __init__(self,
                 models: Optional[List[str]] = None,
                 cv_folds: int = 5,
                 random_state: int = 42):
        """
        Initialize the ML pipeline.

        Args:
            models: List of model types to train ('gbm', 'rf', 'xgb', 'lgb', 'lr')
            cv_folds: Number of cross-validation folds
            random_state: Random seed for reproducibility
        """
        if models is None:
            models = ['gbm', 'rf', 'xgb']

        self.models = models
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.trained_models = {}
        self.cv_results = {}
        self.feature_importance = {}

        # Initialize model configurations
        self.model_configs = {
            'gbm': {
                'class': GradientBoostingClassifier,
                'params': {
                    'n_estimators': 200,
                    'learning_rate': 0.1,
                    'max_depth': 5,
                    'random_state': random_state
                }
            },
            'rf': {
                'class': RandomForestClassifier,
                'params': {
                    'n_estimators': 200,
                    'max_depth': 10,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'random_state': random_state
                }
            },
            'xgb': {
                'class': xgb.XGBClassifier,
                'params': {
                    'n_estimators': 200,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'random_state': random_state,
                    'eval_metric': 'logloss'
                }
            },
            'lgb': {
                'class': lgb.LGBMClassifier,
                'params': {
                    'n_estimators': 200,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'random_state': random_state,
                    'verbose': -1
                }
            },
            'lr': {
                'class': LogisticRegression,
                'params': {
                    'max_iter': 1000,
                    'random_state': random_state,
                    'solver': 'liblinear'
                }
            }
        }

    def train_models(self,
                     X: pd.DataFrame,
                     y: pd.Series,
                     test_size: float = 0.2) -> Dict[str, Any]:
        """
        Train all specified models with cross-validation.

        Args:
            X: Feature matrix
            y: Target variable
            test_size: Fraction of data to use for testing

        Returns:
            Dictionary with training results
        """
        print(
            f"Training {len(self.models)} models on dataset of shape {X.shape}")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )

        results = {}

        for model_name in self.models:
            if model_name not in self.model_configs:
                print(f"Warning: Unknown model type '{model_name}'. Skipping.")
                continue

            print(f"\nTraining {model_name.upper()}...")

            try:
                # Train model
                model_result = self._train_single_model(
                    model_name, X_train, y_train, X_test, y_test
                )

                results[model_name] = model_result
                self.trained_models[model_name] = model_result['model']
                self.cv_results[model_name] = model_result['cv_metrics']

                # Feature importance
                if hasattr(model_result['model'], 'feature_importances_'):
                    self.feature_importance[model_name] = model_result['model'].feature_importances_

                print(f"{model_name.upper()} training completed successfully!")

            except Exception as e:
                print(f"Error training {model_name}: {e}")
                continue

        return results

    def _train_single_model(self,
                            model_name: str,
                            X_train: pd.DataFrame,
                            y_train: pd.Series,
                            X_test: pd.DataFrame,
                            y_test: pd.Series) -> Dict[str, Any]:
        """Train a single model with cross-validation."""

        config = self.model_configs[model_name]
        model_class = config['class']
        model_params = config['params']

        # Initialize model
        model = model_class(**model_params)

        # Cross-validation
        cv = StratifiedKFold(n_splits=self.cv_folds,
                             shuffle=True, random_state=self.random_state)
        cv_metrics = []

        for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
            X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            # Train on fold
            model_fold = model_class(**model_params)
            model_fold.fit(X_fold_train, y_fold_train)

            # Predict on validation fold
            y_pred = model_fold.predict(X_fold_val)
            y_proba = model_fold.predict_proba(X_fold_val)[:, 1]

            # Calculate metrics
            fold_metrics = {
                'fold': fold + 1,
                'accuracy': accuracy_score(y_fold_val, y_pred),
                'precision': precision_score(y_fold_val, y_pred, zero_division=0),
                'recall': recall_score(y_fold_val, y_pred, zero_division=0),
                'f1': f1_score(y_fold_val, y_pred, zero_division=0),
                'roc_auc': roc_auc_score(y_fold_val, y_proba)
            }
            cv_metrics.append(fold_metrics)

        # Train final model on full training data
        final_model = model_class(**model_params)
        final_model.fit(X_train, y_train)

        # Evaluate on test set
        y_pred_test = final_model.predict(X_test)
        y_proba_test = final_model.predict_proba(X_test)[:, 1]

        test_metrics = {
            'accuracy': accuracy_score(y_test, y_pred_test),
            'precision': precision_score(y_test, y_pred_test, zero_division=0),
            'recall': recall_score(y_test, y_pred_test, zero_division=0),
            'f1': f1_score(y_test, y_pred_test, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_proba_test)
        }

        return {
            'model': final_model,
            'cv_metrics': cv_metrics,
            'test_metrics': test_metrics,
            'predictions': y_pred_test,
            'probabilities': y_proba_test
        }

    def get_best_model(self, metric: str = 'f1') -> Tuple[str, Any]:
        """Get the best performing model based on specified metric."""
        if not self.cv_results:
            raise ValueError("No models have been trained yet.")

        # Calculate average CV performance for each model
        model_performance = {}
        for model_name, cv_metrics in self.cv_results.items():
            avg_metric = np.mean([fold[metric] for fold in cv_metrics])
            model_performance[model_name] = avg_metric

        # Find best model
        best_model = max(model_performance.items(), key=lambda x: x[1])

        return best_model[0], self.trained_models[best_model[0]]

    def evaluate_model(self, model_name: str, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Evaluate a specific trained model."""
        if model_name not in self.trained_models:
            raise ValueError(
                f"Model '{model_name}' not found. Available models: {list(self.trained_models.keys())}")

        model = self.trained_models[model_name]
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)[:, 1]

        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1': f1_score(y, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y, y_proba),
            'confusion_matrix': confusion_matrix(y, y_pred),
            'classification_report': classification_report(y, y_pred)
        }

        return metrics

    def plot_cv_results(self, save_path: Optional[str] = None):
        """Plot cross-validation results for all models."""
        if not self.cv_results:
            print("No CV results to plot.")
            return

        # Prepare data for plotting
        plot_data = []
        for model_name, cv_metrics in self.cv_results.items():
            for fold_metrics in cv_metrics:
                plot_data.append({
                    'Model': model_name.upper(),
                    'Fold': fold_metrics['fold'],
                    'F1 Score': fold_metrics['f1'],
                    'ROC AUC': fold_metrics['roc_auc'],
                    'Accuracy': fold_metrics['accuracy']
                })

        df_plot = pd.DataFrame(plot_data)

        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Cross-Validation Results Across Models', fontsize=16)

        # F1 Score
        sns.boxplot(data=df_plot, x='Model', y='F1 Score', ax=axes[0, 0])
        axes[0, 0].set_title('F1 Score Distribution')
        axes[0, 0].tick_params(axis='x', rotation=45)

        # ROC AUC
        sns.boxplot(data=df_plot, x='Model', y='ROC AUC', ax=axes[0, 1])
        axes[0, 1].set_title('ROC AUC Distribution')
        axes[0, 1].tick_params(axis='x', rotation=45)

        # Accuracy
        sns.boxplot(data=df_plot, x='Model', y='Accuracy', ax=axes[1, 0])
        axes[1, 0].set_title('Accuracy Distribution')
        axes[1, 0].tick_params(axis='x', rotation=45)

        # Line plot of F1 scores across folds
        sns.lineplot(data=df_plot, x='Fold', y='F1 Score',
                     hue='Model', ax=axes[1, 1])
        axes[1, 1].set_title('F1 Score Across Folds')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")

        plt.show()

    def plot_feature_importance(self,
                                model_name: str,
                                feature_names: List[str],
                                top_n: int = 20,
                                save_path: Optional[str] = None):
        """Plot feature importance for a specific model."""
        if model_name not in self.feature_importance:
            print(f"No feature importance available for model '{model_name}'")
            return

        importance = self.feature_importance[model_name]

        # Create DataFrame for plotting
        df_importance = pd.DataFrame({
            'feature': feature_names[:len(importance)],
            'importance': importance
        }).sort_values('importance', ascending=True).tail(top_n)

        plt.figure(figsize=(10, 8))
        plt.barh(range(len(df_importance)), df_importance['importance'])
        plt.yticks(range(len(df_importance)), df_importance['feature'])
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Features - {model_name.upper()}')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature importance plot saved to {save_path}")

        plt.show()

    def save_model(self, model_name: str, filepath: str):
        """Save a trained model to disk."""
        if model_name not in self.trained_models:
            raise ValueError(f"Model '{model_name}' not found.")

        model_data = {
            'model': self.trained_models[model_name],
            'cv_results': self.cv_results.get(model_name, []),
            'feature_importance': self.feature_importance.get(model_name, None)
        }

        joblib.dump(model_data, filepath)
        print(f"Model '{model_name}' saved to {filepath}")

    def load_model(self, filepath: str) -> str:
        """Load a saved model from disk."""
        model_data = joblib.load(filepath)

        # Determine model type
        model = model_data['model']
        if isinstance(model, GradientBoostingClassifier):
            model_name = 'gbm'
        elif isinstance(model, RandomForestClassifier):
            model_name = 'rf'
        elif isinstance(model, xgb.XGBClassifier):
            model_name = 'xgb'
        elif isinstance(model, lgb.LGBMClassifier):
            model_name = 'lgb'
        elif isinstance(model, LogisticRegression):
            model_name = 'lr'
        else:
            model_name = 'unknown'

        # Store loaded model
        self.trained_models[model_name] = model
        self.cv_results[model_name] = model_data.get('cv_results', [])
        if model_data.get('feature_importance') is not None:
            self.feature_importance[model_name] = model_data['feature_importance']

        print(f"Model loaded from {filepath}")
        return model_name

    def generate_report(self, output_dir: str = "ml_reports"):
        """Generate comprehensive ML training report."""
        os.makedirs(output_dir, exist_ok=True)

        if not self.cv_results:
            print("No training results to report.")
            return

        # Summary statistics
        summary = []
        for model_name, cv_metrics in self.cv_results.items():
            avg_f1 = np.mean([fold['f1'] for fold in cv_metrics])
            avg_roc = np.mean([fold['roc_auc'] for fold in cv_metrics])
            avg_acc = np.mean([fold['accuracy'] for fold in cv_metrics])

            summary.append({
                'Model': model_name.upper(),
                'Avg F1': f"{avg_f1:.4f}",
                'Avg ROC AUC': f"{avg_roc:.4f}",
                'Avg Accuracy': f"{avg_acc:.4f}"
            })

        df_summary = pd.DataFrame(summary)
        df_summary.to_csv(os.path.join(
            output_dir, 'model_summary.csv'), index=False)

        # Save CV results
        for model_name, cv_metrics in self.cv_results.items():
            df_cv = pd.DataFrame(cv_metrics)
            df_cv.to_csv(os.path.join(
                output_dir, f'{model_name}_cv_results.csv'), index=False)

        # Generate plots
        self.plot_cv_results(os.path.join(output_dir, 'cv_results.png'))

        # Feature importance plots
        if hasattr(self, 'feature_names'):
            for model_name in self.feature_importance.keys():
                self.plot_feature_importance(
                    model_name,
                    self.feature_names,
                    save_path=os.path.join(
                        output_dir, f'{model_name}_feature_importance.png')
                )

        print(f"ML report generated in {output_dir}")

    def set_feature_names(self, feature_names: List[str]):
        """Set feature names for plotting and analysis."""
        self.feature_names = feature_names
