"""Optional LLM evaluation package.

The LLM NEVER decides hard eligibility - it only optionally refines
project-quality assessment during scoring. On failure the deterministic
score is used and the failure is recorded.
"""
from .base import BaseLLMEvaluator, ProjectQualityVerdict
from .adapter import OpenAICompatEvaluator

__all__ = ["BaseLLMEvaluator", "ProjectQualityVerdict", "OpenAICompatEvaluator"]
