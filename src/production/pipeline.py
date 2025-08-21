"""
RedSentinel Production Pipeline
==============================

Production-ready system for real-time attack detection, monitoring,
and continuous learning from new attack patterns.
"""

import pandas as pd
import numpy as np
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import joblib

# Import our feature extractor
from features.robust_extractor import RobustFeatureExtractor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RedSentinelProductionPipeline:
    """
    Production pipeline for real-time attack detection and monitoring.
    """

    def __init__(self,
                 model_path: str = "models/robust_model.joblib",
                 config_path: str = "config/production_config.yaml",
                 alert_threshold: float = 0.8):
        """
        Initialize the production pipeline.

        Args:
            model_path: Path to trained model
            config_path: Path to production configuration
            alert_threshold: Confidence threshold for alerts
        """
        self.model_path = model_path
        self.config_path = config_path
        self.alert_threshold = alert_threshold

        # Initialize components
        self.feature_extractor = RobustFeatureExtractor()
        self.model = None
        self.is_loaded = False

        # Performance tracking
        self.performance_metrics = {
            'total_predictions': 0,
            'correct_predictions': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'start_time': datetime.now(),
            'last_alert': None
        }

        # Attack pattern tracking
        self.attack_patterns = {
            'detected_attacks': [],
            'new_patterns': [],
            'model_performance': []
        }

        # Load model if available
        self._load_model()

    def _load_model(self):
        """Load the trained model and feature extractor."""
        try:
            if Path(self.model_path).exists():
                self.model = joblib.load(self.model_path)
                logger.info(f"Model loaded from {self.model_path}")

                # Load feature extractor transformers
                transformers_path = self.model_path.replace(
                    '.joblib', '_transformers.joblib')
                if Path(transformers_path).exists():
                    self.feature_extractor.load_transformers(transformers_path)
                    logger.info("Feature extractor transformers loaded")

                self.is_loaded = True
            else:
                logger.warning(f"Model not found at {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")

    def detect_attack(self,
                      prompt: str,
                      response: str,
                      model_name: str,
                      model_family: str,
                      technique_category: str,
                      temperature: float = 0.7,
                      top_p: float = 0.9,
                      max_tokens: int = 1000) -> Dict:
        """
        Detect if an attack attempt is successful.

        Args:
            prompt: The attack prompt
            response: The LLM response
            model_name: Name of the target model
            model_family: Family of the target model
            technique_category: Category of attack technique
            temperature: Model temperature parameter
            top_p: Model top_p parameter
            max_tokens: Model max_tokens parameter

        Returns:
            Dictionary with detection results and confidence
        """
        if not self.is_loaded:
            return {
                'attack_detected': False,
                'confidence': 0.0,
                'error': 'Model not loaded',
                'timestamp': datetime.now().isoformat()
            }

        try:
            # Create attack record
            attack_record = pd.DataFrame([{
                'attack_id': f"prod_{int(time.time())}",
                'timestamp': datetime.now(),
                'prompt': prompt,
                'response': response,
                'model_name': model_name,
                'model_family': model_family,
                'technique_category': technique_category,
                'temperature': temperature,
                'top_p': top_p,
                'max_tokens': max_tokens,
                'final_label': 'unknown'  # Will be predicted
            }])

            # Extract features
            X, _ = self.feature_extractor.prepare_dataset(
                attack_record, is_training=False)

            # Make prediction
            prediction = self.model.predict(X)[0]
            confidence = self.model.predict_proba(X)[0].max()

            # Determine attack status
            attack_detected = prediction == 1
            attack_status = 'success' if attack_detected else 'failure'

            # Update performance tracking
            self._update_performance_tracking(attack_detected, confidence)

            # Check for alerts
            alert_triggered = self._check_alert_conditions(
                attack_detected, confidence)

            # Store attack pattern
            self._store_attack_pattern(
                attack_record.iloc[0], attack_status, confidence)

            result = {
                'attack_detected': attack_detected,
                'attack_status': attack_status,
                'confidence': float(confidence),
                'model_prediction': int(prediction),
                'alert_triggered': alert_triggered,
                'timestamp': datetime.now().isoformat(),
                'attack_id': attack_record.iloc[0]['attack_id']
            }

            logger.info(
                f"Attack detection completed: {attack_status} (confidence: {confidence:.3f})")
            return result

        except Exception as e:
            logger.error(f"Attack detection failed: {e}")
            return {
                'attack_detected': False,
                'confidence': 0.0,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def _update_performance_tracking(self, attack_detected: bool, confidence: float):
        """Update performance tracking metrics."""
        self.performance_metrics['total_predictions'] += 1

        # Note: In production, we don't have ground truth labels
        # This is for monitoring prediction confidence and patterns
        if confidence > self.alert_threshold:
            self.performance_metrics['last_alert'] = datetime.now()

    def _check_alert_conditions(self, attack_detected: bool, confidence: float) -> bool:
        """Check if alert conditions are met."""
        if attack_detected and confidence > self.alert_threshold:
            return True
        return False

    def _store_attack_pattern(self, attack_record: pd.Series, status: str, confidence: float):
        """Store attack pattern for analysis."""
        pattern = {
            'timestamp': attack_record['timestamp'],
            'model_name': attack_record['model_name'],
            'technique_category': attack_record['technique_category'],
            'status': status,
            'confidence': confidence,
            'prompt_length': len(str(attack_record['prompt'])),
            'response_length': len(str(attack_record['response']))
        }

        self.attack_patterns['detected_attacks'].append(pattern)

        # Keep only last 1000 patterns to prevent memory issues
        if len(self.attack_patterns['detected_attacks']) > 1000:
            self.attack_patterns['detected_attacks'] = self.attack_patterns['detected_attacks'][-1000:]

    def get_performance_summary(self) -> Dict:
        """Get current performance summary."""
        uptime = datetime.now() - self.performance_metrics['start_time']

        summary = {
            'uptime_hours': uptime.total_seconds() / 3600,
            'total_predictions': self.performance_metrics['total_predictions'],
            'alerts_triggered': len([p for p in self.attack_patterns['detected_attacks']
                                     if p['confidence'] > self.alert_threshold]),
            'last_alert': self.performance_metrics['last_alert'].isoformat() if self.performance_metrics['last_alert'] else None,
            'attack_patterns_detected': len(self.attack_patterns['detected_attacks']),
            'model_loaded': self.is_loaded,
            'current_time': datetime.now().isoformat()
        }

        return summary

    def get_attack_patterns(self, hours: int = 24) -> List[Dict]:
        """Get attack patterns from the last N hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        recent_patterns = [
            p for p in self.attack_patterns['detected_attacks']
            if p['timestamp'] > cutoff_time
        ]

        return recent_patterns

    def export_performance_data(self, output_path: str):
        """Export performance data for analysis."""
        data = {
            'performance_metrics': self.performance_metrics,
            'attack_patterns': self.attack_patterns,
            'export_timestamp': datetime.now().isoformat()
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"Performance data exported to {output_path}")

    def retrain_model(self, new_data_path: str, output_model_path: str = None):
        """
        Retrain the model with new data.

        Args:
            new_data_path: Path to new training data
            output_model_path: Path to save retrained model
        """
        try:
            logger.info("Starting model retraining...")

            # Load new data
            new_data = pd.read_csv(new_data_path)
            logger.info(f"Loaded {len(new_data)} new training samples")

            # Extract features
            X, y = self.feature_extractor.prepare_dataset(
                new_data, is_training=True)

            # Train new model
            new_model = RandomForestClassifier(
                n_estimators=100, random_state=42, max_depth=10)
            new_model.fit(X, y)

            # Evaluate new model
            y_pred = new_model.predict(X)
            f1 = f1_score(y, y_pred, zero_division=0)
            accuracy = accuracy_score(y, y_pred)

            logger.info(
                f"New model performance - F1: {f1:.3f}, Accuracy: {accuracy:.3f}")

            # Save new model
            if output_model_path is None:
                output_model_path = self.model_path.replace(
                    '.joblib', '_retrained.joblib')

            joblib.dump(new_model, output_model_path)

            # Save feature extractor transformers
            transformers_path = output_model_path.replace(
                '.joblib', '_transformers.joblib')
            self.feature_extractor.save_transformers(transformers_path)

            logger.info(f"Retrained model saved to {output_model_path}")

            return {
                'success': True,
                'new_model_path': output_model_path,
                'performance': {'f1': f1, 'accuracy': accuracy}
            }

        except Exception as e:
            logger.error(f"Model retraining failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
