"""Direct tests for hard eligibility business rules.

Every test here maps to a concrete requirement from the assignment:
  - BOTH Python evidence AND meaningful AI/LLM/agentic evidence are required.
  - Weak statements (interested in AI, AI course, bare 'AI' skill) are rejected.
  - JavaScript / Java / React / Next.js do NOT cause rejection when the two
    core requirements are satisfied.
"""
import pytest

from src.eligibility import check_eligibility
from src.models import Candidate


def _cand(**kw) -> Candidate:
    base = dict(skills=[], projects=[], education=[], experience=[], raw_text="")
    base.update(kw)
    return Candidate(**base)


# ── helpers ──────────────────────────────────────────────────────────────────

def _eligible(cand: Candidate) -> bool:
    return check_eligibility(cand).eligible


def _reasons(cand: Candidate):
    return check_eligibility(cand).rejection_reasons


# ══════════════════════════════════════════════════════════════════════════════
# Cases that MUST be rejected
# ══════════════════════════════════════════════════════════════════════════════

def test_java_react_only_rejected():
    cand = _cand(
        skills=["java", "react"],
        projects=["Built a React app with Java Spring Boot backend"],
        raw_text="java react spring boot rest api",
        experience=["Java developer at Acme"],
    )
    assert not _eligible(cand)
    reasons = _reasons(cand)
    assert any("python" in r.lower() for r in reasons)


def test_python_only_no_ai_rejected():
    cand = _cand(
        skills=["python", "fastapi"],
        projects=["Built a FastAPI REST service with PostgreSQL"],
        raw_text="python fastapi postgresql rest api backend",
        experience=["Python backend developer"],
    )
    assert not _eligible(cand)
    assert any("ai" in r.lower() for r in _reasons(cand))


def test_ai_only_no_python_rejected():
    cand = _cand(
        skills=["langchain", "rag", "openai api"],
        projects=["Built a RAG chatbot with LangChain, embeddings and vector search"],
        raw_text="langchain rag embeddings vector search openai api",
        experience=["AI engineer"],
    )
    assert not _eligible(cand)
    assert any("python" in r.lower() for r in _reasons(cand))


def test_python_interested_in_ai_rejected():
    """'Interested in AI' must not count as meaningful AI evidence."""
    cand = _cand(
        skills=["python"],
        raw_text="python interested in AI AI enthusiast learning AI",
    )
    assert not _eligible(cand)


def test_python_ai_course_only_rejected():
    """Completing an AI course is not implementation evidence."""
    cand = _cand(
        skills=["python"],
        raw_text="python completed AI course AI certification bootcamp",
    )
    assert not _eligible(cand)


def test_python_bare_ai_skill_only_rejected():
    """Listing 'AI' in skills without any concrete technology is not enough."""
    cand = _cand(
        skills=["python", "ai"],
        raw_text="python ai",
    )
    assert not _eligible(cand)


# ══════════════════════════════════════════════════════════════════════════════
# Cases that MUST be accepted
# ══════════════════════════════════════════════════════════════════════════════

def test_python_rag_langchain_eligible():
    cand = _cand(
        skills=["python", "langchain", "rag"],
        projects=["Built a RAG pipeline with LangChain, embeddings and vector search"],
        raw_text="python langchain rag embeddings vector search",
    )
    assert _eligible(cand)


def test_python_langgraph_agent_eligible():
    cand = _cand(
        skills=["python", "langgraph"],
        projects=["Built a multi-agent orchestration system with LangGraph and state management"],
        raw_text="python langgraph multi-agent orchestration state management",
    )
    assert _eligible(cand)


def test_python_react_rag_eligible():
    """React / JS presence must NOT cause rejection when Python + AI are present."""
    cand = _cand(
        skills=["python", "javascript", "react", "langchain"],
        projects=["Built a RAG system with Python backend, React frontend, embeddings and vector search"],
        raw_text="python javascript react langchain rag embeddings vector search",
    )
    assert _eligible(cand)


def test_python_nextjs_rag_eligible():
    """Next.js presence must NOT cause rejection when Python + AI are present."""
    cand = _cand(
        skills=["python", "next.js", "langchain"],
        projects=["Built a RAG chatbot with LangChain, embeddings; Next.js frontend"],
        raw_text="python next.js langchain rag embeddings vector search",
    )
    assert _eligible(cand)


def test_java_python_ai_eligible():
    """Java + Python + meaningful AI must be eligible (Java doesn't block)."""
    cand = _cand(
        skills=["java", "python", "langchain"],
        projects=["Built a RAG pipeline with Python, LangChain and embeddings"],
        raw_text="java python langchain rag embeddings vector search",
    )
    assert _eligible(cand)


def test_java_react_nextjs_python_and_meaningful_ai_eligible():
    cand = _cand(
        skills=["java", "react", "next.js", "python", "langgraph"],
        projects=[
            "Built a Python LangGraph agent with retrieval and tool calling"
        ],
        raw_text="java react next.js python langgraph retrieval tool calling",
    )
    assert _eligible(cand)


# ══════════════════════════════════════════════════════════════════════════════
# Evidence fields are populated correctly
# ══════════════════════════════════════════════════════════════════════════════

def test_python_evidence_populated():
    cand = _cand(
        skills=["python", "langchain"],
        projects=["Built a RAG pipeline with LangChain and embeddings"],
        raw_text="python langchain rag embeddings",
    )
    result = check_eligibility(cand)
    assert result.python_evidence, "python_evidence must not be empty"


def test_ai_evidence_populated():
    cand = _cand(
        skills=["python", "langchain", "rag"],
        projects=["Built a RAG pipeline with LangChain, embeddings and vector search"],
        raw_text="python langchain rag embeddings vector search",
    )
    result = check_eligibility(cand)
    assert result.ai_evidence, "ai_evidence must not be empty"


def test_rejection_reasons_populated_when_ineligible():
    cand = _cand(skills=["java"], raw_text="java spring boot")
    result = check_eligibility(cand)
    assert not result.eligible
    assert result.rejection_reasons
