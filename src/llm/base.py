"""LLM evaluator interface - structured Pydantic output, never eligibility."""
from typing import Optional

from pydantic import BaseModel, Field


class ProjectQualityVerdict(BaseModel):
    """Structured result an LLM evaluator must return."""

    quality: str = Field(description="'strong' | 'moderate' | 'shallow'")
    penalty_points: int = Field(default=0, ge=0, le=15)
    penalty_reason: str = ""


class BaseLLMEvaluator:
    """Contract for optional semantic project-quality evaluators."""

    name: str = "base"

    def evaluate(self, project_text: str) -> ProjectQualityVerdict:
        """Return a ProjectQualityVerdict. Implementations must raise on
        failure so scoring can fall back to the deterministic path."""
        raise NotImplementedError
