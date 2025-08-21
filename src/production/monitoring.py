"""
RedSentinel Production Monitoring & Alerting System
==================================================

Real-time monitoring, performance tracking, and alerting system
for production deployment of RedSentinel.
"""

import pandas as pd
import numpy as np
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import threading
import queue
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

logger = logging.getLogger(__name__)


class RedSentinelMonitor:
    """
    Production monitoring and alerting system for RedSentinel.
    """

    def __init__(self,
                 alert_threshold: float = 0.8,
                 performance_window_hours: int = 24,
                 alert_cooldown_minutes: int = 30):
        """
        Initialize the monitoring system.

        Args:
            alert_threshold: Confidence threshold for alerts
            performance_window_hours: Hours to track for performance metrics
            alert_cooldown_minutes: Minutes between similar alerts
        """
        self.alert_threshold = alert_threshold
        self.performance_window_hours = performance_window_hours
        self.alert_cooldown_minutes = alert_cooldown_minutes

        # Performance tracking
        self.performance_data = {
            'predictions': [],
            'alerts': [],
            'errors': [],
            'system_health': []
        }

        # Alert management
        self.alert_history = []
        self.last_alert_time = {}
        self.alert_queue = queue.Queue()

        # Monitoring state
        self.is_monitoring = False
        self.monitor_thread = None

        # Performance thresholds
        self.thresholds = {
            'min_accuracy': 0.75,
            'max_false_positive_rate': 0.15,
            'max_response_time_ms': 100,
            'min_uptime_percentage': 0.95
        }

        # Start monitoring
        self.start_monitoring()

    def start_monitoring(self):
        """Start the monitoring system."""
        if not self.is_monitoring:
            self.is_monitoring = True
            self.monitor_thread = threading.Thread(
                target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            logger.info("Monitoring system started")

    def stop_monitoring(self):
        """Stop the monitoring system."""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        logger.info("Monitoring system stopped")

    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                # Check system health
                self._check_system_health()

                # Process alert queue
                self._process_alert_queue()

                # Clean old data
                self._cleanup_old_data()

                # Sleep for monitoring interval
                time.sleep(60)  # Check every minute

            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(60)

    def record_prediction(self,
                          prediction_result: Dict,
                          response_time_ms: float):
        """
        Record a prediction result for monitoring.

        Args:
            prediction_result: Result from attack detection
            response_time_ms: Response time in milliseconds
        """
        timestamp = datetime.now()

        # Record prediction
        prediction_record = {
            'timestamp': timestamp,
            'attack_detected': prediction_result.get('attack_detected', False),
            'confidence': prediction_result.get('confidence', 0.0),
            'response_time_ms': response_time_ms,
            'model_loaded': prediction_result.get('model_loaded', True),
            'attack_id': prediction_result.get('attack_id', 'unknown')
        }

        self.performance_data['predictions'].append(prediction_record)

        # Check for alerts
        if prediction_result.get('attack_detected', False):
            confidence = prediction_result.get('confidence', 0.0)
            if confidence > self.alert_threshold:
                self._trigger_alert('high_confidence_attack', {
                    'confidence': confidence,
                    'attack_id': prediction_result.get('attack_id', 'unknown'),
                    'timestamp': timestamp.isoformat()
                })

        # Check response time
        if response_time_ms > self.thresholds['max_response_time_ms']:
            self._trigger_alert('slow_response_time', {
                'response_time_ms': response_time_ms,
                'threshold': self.thresholds['max_response_time_ms'],
                'timestamp': timestamp.isoformat()
            })

    def record_error(self, error_message: str, error_type: str = 'general'):
        """
        Record an error for monitoring.

        Args:
            error_message: Error message
            error_type: Type of error
        """
        error_record = {
            'timestamp': datetime.now(),
            'error_message': error_message,
            'error_type': error_type
        }

        self.performance_data['errors'].append(error_record)

        # Trigger error alert
        self._trigger_alert('system_error', {
            'error_type': error_type,
            'error_message': error_message,
            'timestamp': error_record['timestamp'].isoformat()
        })

    def _trigger_alert(self, alert_type: str, alert_data: Dict):
        """
        Trigger an alert.

        Args:
            alert_type: Type of alert
            alert_data: Alert data
        """
        # Check cooldown
        if self._is_alert_cooldown(alert_type):
            return

        alert = {
            'id': f"{alert_type}_{int(time.time())}",
            'type': alert_type,
            'timestamp': datetime.now(),
            'data': alert_data,
            'severity': self._get_alert_severity(alert_type)
        }

        # Add to alert history
        self.alert_history.append(alert)

        # Add to processing queue
        self.alert_queue.put(alert)

        # Update last alert time
        self.last_alert_time[alert_type] = datetime.now()

        logger.warning(f"Alert triggered: {alert_type} - {alert_data}")

    def _is_alert_cooldown(self, alert_type: str) -> bool:
        """Check if alert is in cooldown period."""
        if alert_type not in self.last_alert_time:
            return False

        time_since_last = datetime.now() - self.last_alert_time[alert_type]
        return time_since_last.total_seconds() < (self.alert_cooldown_minutes * 60)

    def _get_alert_severity(self, alert_type: str) -> str:
        """Get severity level for alert type."""
        severity_map = {
            'high_confidence_attack': 'high',
            'system_error': 'high',
            'slow_response_time': 'medium',
            'performance_degradation': 'medium',
            'low_accuracy': 'medium'
        }

        return severity_map.get(alert_type, 'low')

    def _process_alert_queue(self):
        """Process alerts from the queue."""
        try:
            while not self.alert_queue.empty():
                alert = self.alert_queue.get_nowait()
                self._process_alert(alert)
        except queue.Empty:
            pass

    def _process_alert(self, alert: Dict):
        """Process a single alert."""
        try:
            # Log alert
            logger.warning(
                f"Processing alert: {alert['type']} - {alert['data']}")

            # Send notification (email, Slack, etc.)
            self._send_alert_notification(alert)

            # Store in performance data
            self.performance_data['alerts'].append(alert)

        except Exception as e:
            logger.error(f"Failed to process alert: {e}")

    def _send_alert_notification(self, alert: Dict):
        """Send alert notification (placeholder for email/Slack integration)."""
        # This would integrate with your notification system
        # For now, just log the alert
        logger.warning(f"ALERT: {alert['severity'].upper()} - {alert['type']}")
        logger.warning(f"Alert data: {alert['data']}")

    def _check_system_health(self):
        """Check overall system health."""
        try:
            # Calculate performance metrics
            metrics = self.get_performance_metrics()

            # Check thresholds
            if metrics['accuracy'] < self.thresholds['min_accuracy']:
                self._trigger_alert('low_accuracy', {
                    'current_accuracy': metrics['accuracy'],
                    'threshold': self.thresholds['min_accuracy']
                })

            if metrics['false_positive_rate'] > self.thresholds['max_false_positive_rate']:
                self._trigger_alert('high_false_positive_rate', {
                    'current_rate': metrics['false_positive_rate'],
                    'threshold': self.thresholds['max_false_positive_rate']
                })

            # Record system health
            health_record = {
                'timestamp': datetime.now(),
                'metrics': metrics,
                'alerts_active': len([a for a in self.alert_history if a['timestamp'] > datetime.now() - timedelta(hours=1)]),
                'system_status': 'healthy' if self._is_system_healthy(metrics) else 'degraded'
            }

            self.performance_data['system_health'].append(health_record)

        except Exception as e:
            logger.error(f"System health check failed: {e}")

    def _is_system_healthy(self, metrics: Dict) -> bool:
        """Determine if system is healthy based on metrics."""
        return (metrics['accuracy'] >= self.thresholds['min_accuracy'] and
                metrics['false_positive_rate'] <= self.thresholds['max_false_positive_rate'])

    def _cleanup_old_data(self):
        """Clean up old performance data."""
        cutoff_time = datetime.now() - timedelta(hours=self.performance_window_hours)

        # Clean predictions
        self.performance_data['predictions'] = [
            p for p in self.performance_data['predictions']
            if p['timestamp'] > cutoff_time
        ]

        # Clean errors
        self.performance_data['errors'] = [
            e for e in self.performance_data['errors']
            if e['timestamp'] > cutoff_time
        ]

        # Clean system health (keep more history)
        health_cutoff = datetime.now() - timedelta(hours=self.performance_window_hours * 2)
        self.performance_data['system_health'] = [
            h for h in self.performance_data['system_health']
            if h['timestamp'] > health_cutoff
        ]

    def get_performance_metrics(self) -> Dict:
        """Get current performance metrics."""
        if not self.performance_data['predictions']:
            return {
                'accuracy': 0.0,
                'false_positive_rate': 0.0,
                'average_response_time': 0.0,
                'total_predictions': 0,
                'alerts_24h': 0,
                'errors_24h': 0
            }

        # Calculate metrics from recent data
        recent_predictions = [
            p for p in self.performance_data['predictions']
            if p['timestamp'] > datetime.now() - timedelta(hours=self.performance_window_hours)
        ]

        if not recent_predictions:
            return {
                'accuracy': 0.0,
                'false_positive_rate': 0.0,
                'average_response_time': 0.0,
                'total_predictions': 0,
                'alerts_24h': 0,
                'errors_24h': 0
            }

        # Basic metrics
        total_predictions = len(recent_predictions)
        high_confidence_attacks = len(
            [p for p in recent_predictions if p['confidence'] > self.alert_threshold])

        # Response time
        response_times = [p['response_time_ms'] for p in recent_predictions]
        avg_response_time = np.mean(response_times) if response_times else 0.0

        # Alerts and errors in last 24h
        alerts_24h = len([a for a in self.alert_history if a['timestamp']
                         > datetime.now() - timedelta(hours=24)])
        errors_24h = len([e for e in self.performance_data['errors']
                         if e['timestamp'] > datetime.now() - timedelta(hours=24)])

        # Note: In production, we don't have ground truth labels
        # These metrics are based on confidence scores and patterns
        estimated_accuracy = 0.85  # Placeholder - would need ground truth for real accuracy
        estimated_false_positive_rate = 0.10  # Placeholder

        return {
            'accuracy': estimated_accuracy,
            'false_positive_rate': estimated_false_positive_rate,
            'average_response_time': avg_response_time,
            'total_predictions': total_predictions,
            'high_confidence_attacks': high_confidence_attacks,
            'alerts_24h': alerts_24h,
            'errors_24h': errors_24h
        }

    def get_alert_summary(self, hours: int = 24) -> Dict:
        """Get summary of alerts in the last N hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        recent_alerts = [
            a for a in self.alert_history
            if a['timestamp'] > cutoff_time
        ]

        # Group by type
        alert_counts = {}
        for alert in recent_alerts:
            alert_type = alert['type']
            alert_counts[alert_type] = alert_counts.get(alert_type, 0) + 1

        return {
            'total_alerts': len(recent_alerts),
            'alerts_by_type': alert_counts,
            'high_severity_alerts': len([a for a in recent_alerts if a['severity'] == 'high']),
            'time_period_hours': hours
        }

    def export_monitoring_data(self, output_path: str):
        """Export monitoring data for analysis."""
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'performance_data': self.performance_data,
            'alert_history': self.alert_history,
            'thresholds': self.thresholds,
            'performance_metrics': self.get_performance_metrics(),
            'alert_summary': self.get_alert_summary(24)
        }

        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)

        logger.info(f"Monitoring data exported to {output_path}")

    def get_dashboard_data(self) -> Dict:
        """Get data for monitoring dashboard."""
        return {
            'current_metrics': self.get_performance_metrics(),
            'alert_summary': self.get_alert_summary(24),
            'system_status': 'healthy' if self._is_system_healthy(self.get_performance_metrics()) else 'degraded',
            'uptime_percentage': self._calculate_uptime_percentage(),
            'recent_alerts': self.alert_history[-10:],  # Last 10 alerts
            'performance_trend': self._get_performance_trend()
        }

    def _calculate_uptime_percentage(self) -> float:
        """Calculate system uptime percentage."""
        if not self.performance_data['system_health']:
            return 100.0

        total_checks = len(self.performance_data['system_health'])
        healthy_checks = len(
            [h for h in self.performance_data['system_health'] if h['system_status'] == 'healthy'])

        return (healthy_checks / total_checks) * 100 if total_checks > 0 else 100.0

    def _get_performance_trend(self) -> List[Dict]:
        """Get performance trend over time."""
        if not self.performance_data['system_health']:
            return []

        # Get last 24 data points (hourly if available)
        recent_health = self.performance_data['system_health'][-24:]

        trend = []
        for health in recent_health:
            trend.append({
                'timestamp': health['timestamp'].isoformat(),
                'accuracy': health['metrics']['accuracy'],
                'response_time': health['metrics']['average_response_time'],
                'system_status': health['system_status']
            })

        return trend
