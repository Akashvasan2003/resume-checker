"""OpenAI-compatible evaluator - optional, only used when configured."""
import os
from typing import Optional

from .base import BaseLLMEvaluator, ProjectQualityVerdict

SYSTEM_PROMPT = (
    "You assess the implementation depth of an AI project described in a "
    "resume. Reply ONLY with JSON: {\"quality\": \"strong|moderate|shallow\", "
    "\"penalty_points\": 0-15, \"penalty_reason\": \"...\"}. Give penalty "
    "points when the project is a thin LLM/API wrapper, tutorial-only, or "
    "lacks meaningful workflow, retrieval, state management, backend logic "
    "or evaluation. Never decide eligibility."
)


class OpenAICompatEvaluator(BaseLLMEvaluator):
    """Works with OpenAI API or any OpenAI-compatible endpoint."""

    name = "openai-compat"

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.model = model
        if not self.api_key:
            raise RuntimeError("no OPENAI_API_KEY configured - LLM evaluation disabled")

    def evaluate(self, project_text: str) -> ProjectQualityVerdict:
        try:
            from openai import OpenAI

            client = OpenAI(api_key=self.api_key)
            resp = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": project_text[:4000]},
                ],
                response_format={"type": "json_object"},
                temperature=0,
            )
            return ProjectQualityVerdict.model_validate_json(
                resp.choices[0].message.content
            )
        except Exception as exc:
            raise RuntimeError(f"LLM evaluation failed: {exc}") from exc
