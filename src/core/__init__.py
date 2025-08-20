"""
RedSentinel Core Module

Core components for the RedSentinel red team tool:
- PromptEvaluator: Evaluates LLM responses for system prompt extraction
- AttackLogger: Logs attack attempts with comprehensive metadata
"""

from .prompt_evaluator import PromptEvaluator
from .attack_logger import AttackLogger

__all__ = ['PromptEvaluator', 'AttackLogger']
