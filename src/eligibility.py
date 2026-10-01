"""Hard eligibility filtering - deterministic, no LLM involved.

A candidate is eligible only with BOTH:
  1. Python evidence
  2. Meaningful AI/LLM/agentic evidence

False positives are avoided by:
  - Context-aware matching: bare "AI"/"Python" mentions only count when they
    appear in implementation contexts (projects / experience / concrete
    frameworks), not in interest statements.
  - A negative-pattern blocklist for phrases like "interested in AI",
    "AI enthusiast", "Python course", "learning Python", "Python certification".
  - Bare "AI" in a skills list alone is NOT accepted (anyone can write "AI"
    in skills); only concrete AI technologies (LangChain, RAG, OpenAI API,
    PyTorch, transformers...) count as skill-level evidence.
"""
import re
from typing import List, Optional

from .models import Candidate, EligibilityResult

# Concrete AI technologies - acceptable even in a skills list
AI_TECH_KEYWORDS = [
    "llm", "rag", "langchain", "langgraph", "google adk", "adk",
    "llamaindex", "llama index", "embeddings", "vector search",
    "vector database", "vectordb", "faiss", "pinecone", "chromadb",
    "tool calling", "function calling", "agent", "multi-agent",
    "ai pipeline", "eval", "evaluation pipeline", "fine-tuning", "finetuning",
    "transformers", "hugging face", "huggingface", "openai api", "gemini api",
    "claude api", "anthropic", "pytorch", "tensorflow", "keras",
    "scikit-learn", "sklearn", "nlp", "genai", "gen ai", "generative ai",
]

# Bare terms that only count with implementation context
AI_BARE_TERMS = ["ai", "artificial intelligence", "machine learning", "ml", "deep learning"]

# Implementation verbs/objects that make a bare AI/ML mention meaningful
IMPLEMENTATION_HINTS = [
    "built", "developed", "implemented", "created", "deployed", "designed",
    "engineered", "integrated", "trained", "fine-tuned", "shipped",
    "chatbot", "pipeline", "application", "app", "system", "service",
    "project", "agent", "tool", "assistant", "api", "dashboard", "model",
]

# Phrases that indicate interest/learning ONLY - never evidence
NEGATIVE_PATTERNS = [
    r"interested in (?:ai|artificial intelligence|machine learning|python)",
    r"(?:ai|ml|python)[- ](?:enthusiast|curious)",
    r"(?:ai|ml|python)[- ](?:course|tutorial|certification|bootcamp)",
    r"learning (?:ai|ml|python)",
    r"(?:eager|passionate|excited|keen) about (?:ai|ml|python)",
    r"familiar with (?:ai|ml|python)\b",
]

NEGATIVE_RE = re.compile("|".join(NEGATIVE_PATTERNS), re.IGNORECASE)


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def _is_negated(segment: str) -> bool:
    return bool(NEGATIVE_RE.search(segment))


def _find_ai_tech(haystack: str) -> List[str]:
    low = haystack.lower()
    return sorted({kw for kw in AI_TECH_KEYWORDS if kw in low}, key=lambda k: -len(k))


def _find_bare_ai(haystack: str) -> List[str]:
    """Bare AI/ML terms mentioned WITH implementation context, no negation."""
    hits = []
    for line in haystack.splitlines():
        low = _clean(line)
        if _is_negated(low):
            continue
        for term in AI_BARE_TERMS:
            if re.search(rf"\b{re.escape(term)}\b", low) and any(
                hint in low for hint in IMPLEMENTATION_HINTS
            ):
                hits.append(term)
                break
    return sorted(set(hits))


def _find_python_evidence(candidate: Candidate) -> List[str]:
    """Python evidence from skills, projects, experience or raw text."""
    evidence: List[str] = []

    if any(_clean(s) == "python" for s in candidate.skills):
        evidence.append("listed skill: python")

    for line in candidate.projects:
        if re.search(r"\bpython\b", line, re.IGNORECASE) and not _is_negated(line):
            evidence.append(f"project: {_clean(line)[:100]}")

    for line in candidate.experience:
        if re.search(r"\bpython\b", line, re.IGNORECASE) and not _is_negated(line):
            evidence.append(f"experience: {_clean(line)[:100]}")

    # Raw-text fallback: python mentioned alongside an implementation hint
    if not evidence:
        for line in candidate.raw_text.splitlines():
            low = _clean(line)
            if re.search(r"\bpython\b", low) and not _is_negated(low) and any(
                h in low for h in IMPLEMENTATION_HINTS
            ):
                evidence.append(f"text: {low[:100]}")
    return evidence


def _find_ai_evidence(candidate: Candidate) -> List[str]:
    evidence: List[str] = []

    # Concrete AI techs mentioned anywhere (skills list is acceptable here)
    techs = _find_ai_tech(candidate.raw_text + " " + " ".join(candidate.skills))
    for t in techs:
        evidence.append(f"technology: {t}")

    # Bare AI/ML terms need implementation context
    for line in candidate.projects + candidate.experience:
        low = _clean(line)
        if _is_negated(low):
            continue
        for term in _find_bare_ai(line):
            evidence.append(f"{'project' if line in candidate.projects else 'experience'}: {low[:100]}")

    # Exclude technologies that appear only in negated statements
    evidence = list(dict.fromkeys(evidence))
    return evidence


def check_eligibility(candidate: Candidate) -> EligibilityResult:
    python_evidence = _find_python_evidence(candidate)
    ai_evidence = _find_ai_evidence(candidate)

    reasons: List[str] = []
    if not python_evidence:
        reasons.append("no meaningful Python evidence")
    if not ai_evidence:
        reasons.append("no meaningful AI/LLM/agentic evidence")

    matched = sorted(set(python_evidence) | set(ai_evidence))
    return EligibilityResult(
        eligible=not reasons,
        rejection_reasons=reasons,
        python_evidence=python_evidence,
        ai_evidence=ai_evidence,
        matched_skills=matched,
    )


def eligible_candidates(records) -> List:
    """Records whose candidate parsed and passed hard eligibility."""
    return [
        r for r in records
        if r.ok and r.eligibility and r.eligibility.eligible
    ]
