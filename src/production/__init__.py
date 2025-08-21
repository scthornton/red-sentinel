"""
RedSentinel Production Module
============================

Production-ready components for RedSentinel deployment including:
- Production pipeline for real-time attack detection
- Real-world testing framework
- Monitoring and alerting system
- Performance tracking and optimization
"""

from .pipeline import RedSentinelProductionPipeline
from .real_world_tester import RealWorldTester
from .monitoring import RedSentinelMonitor

__all__ = [
    'RedSentinelProductionPipeline',
    'RealWorldTester',
    'RedSentinelMonitor'
]

__version__ = "1.0.0"
__author__ = "RedSentinel Team"
__description__ = "Production-ready AI security monitoring system"
