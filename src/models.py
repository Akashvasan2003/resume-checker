"""Pydantic models for candidate/resume data."""
from typing import List, Optional

from pydantic import BaseModel, Field

__all__ = [
    "Candidate", "ResumeRecord", "BatchResult",
    "EligibilityResult", "ScoreResult",
]


class Candidate(BaseModel):
    """Structured data extracted from one resume."""

    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    github_url: Optional[str] = None
    skills: List[str] = Field(default_factory=list)
    projects: List[str] = Field(default_factory=list)
    education: List[str] = Field(default_factory=list)
    experience: List[str] = Field(default_factory=list)
    raw_text: str = ""


class ResumeRecord(BaseModel):
    """One processed file - success or failure, never raises on bad input."""

    filename: str
    candidate: Optional[Candidate] = None
    error: Optional[str] = None
    is_duplicate: bool = False
    eligibility: Optional["EligibilityResult"] = None
    score: Optional["ScoreResult"] = None
    github: Optional[dict] = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.candidate is not None


class BatchResult(BaseModel):
    """Aggregated result of processing a whole folder."""

    total_files: int
    parsed: int
    failed: int
    duplicates: int
    eligible: int = 0
    records: List[ResumeRecord]


class ScoreResult(BaseModel):
    """100-point evidence-based score (deterministic + optional LLM assist)."""

    ai_project_depth: int = 0
    python_backend: int = 0
    cloud_fullstack: int = 0
    github: int = 0
    engineering_depth: int = 0
    total_score: int = 0
    rank: Optional[int] = None
    reasons: List[str] = Field(default_factory=list)
    penalties: List[str] = Field(default_factory=list)
    llm_used: bool = False
    llm_error: Optional[str] = None


class EligibilityResult(BaseModel):
    """Deterministic hard-eligibility decision (Python + AI evidence)."""

    eligible: bool
    rejection_reasons: List[str] = Field(default_factory=list)
    python_evidence: List[str] = Field(default_factory=list)
    ai_evidence: List[str] = Field(default_factory=list)
    matched_skills: List[str] = Field(default_factory=list)
