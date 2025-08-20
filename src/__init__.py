"""
RedSentinel - Red Team LLM Security Testing Tool

A comprehensive tool for testing LLM guardrails and detecting system prompt extraction.
"""

__version__ = "1.0.0"
__author__ = "Scott Thornton / perfecXion.ai"

from .core import PromptEvaluator, AttackLogger
from .features import RedTeamFeatureExtractor
from .ml import RedTeamMLPipeline

__all__ = [
    'PromptEvaluator',
    'AttackLogger',
    'RedTeamFeatureExtractor',
    'RedTeamMLPipeline'
]
