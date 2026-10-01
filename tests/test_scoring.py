"""Tests for evidence-based scoring (Part 3)."""
from src.eligibility import check_eligibility
from src.models import Candidate
from src.scoring import WEIGHTS, _ai_score, _penalties, score_candidate


def _cand(**kw) -> Candidate:
    base = dict(skills=[], projects=[], education=[], experience=[], raw_text="")
    base.update(kw)
    return Candidate(**base)


def _eligible(cand):
    result = check_eligibility(cand)
    assert result.eligible, f"test candidate not eligible: {result.rejection_reasons}"
    return result


def test_strong_rag_project_scores_high():
    cand = _cand(
        skills=["python", "langchain", "rag"],
        projects=["Built a RAG Q&A system with embeddings, vector search and "
                  "retrieval over company docs using LangChain and FAISS"],
        experience=["SDE Intern: developed backend REST API in Python/FastAPI"],
        raw_text="python langchain rag embeddings vector search faiss fastapi",
    )
    result = score_candidate(cand, _eligible(cand))
    assert result.ai_project_depth >= 25
    assert result.total_score <= 100


def test_shallow_llm_wrapper_penalized():
    cand = _cand(
        skills=["python", "openai"],
        projects=["Simple chatbot using the OpenAI API"],
        raw_text="python openai api chatbot",
    )
    result = score_candidate(cand, _eligible(cand))
    assert result.ai_project_depth < 20
    assert any("wrapper" in p or "chatbot" in p for p in result.penalties)


def test_strong_python_backend():
    cand = _cand(
        skills=["python", "fastapi", "postgresql", "redis", "langchain"],
        projects=["Built an async FastAPI backend with PostgreSQL database "
                  "design, Redis caching and Celery background workers"],
        raw_text="python fastapi async postgresql redis celery backend rest api",
    )
    result = score_candidate(cand, check_eligibility(cand))
    assert result.python_backend >= 20


def test_cloud_deployment_evidence():
    cand = _cand(
        skills=["python", "docker", "aws", "llm"],
        projects=["Deployed a FastAPI service with Docker on AWS with CI/CD "
                  "via GitHub Actions; React frontend for a full-stack app"],
        raw_text="python docker aws ci/cd github actions deployed react full-stack",
    )
    result = score_candidate(cand, check_eligibility(cand))
    assert result.cloud_fullstack > 0


def test_engineering_depth_signals():
    cand = _cand(
        skills=["python", "pytest", "kafka", "llm"],
        projects=["Built a service with unit tests (pytest), Kafka message "
                  "queue, caching, monitoring and error handling"],
        raw_text="pytest kafka caching monitoring error handling concurrency",
    )
    result = score_candidate(cand, check_eligibility(cand))
    assert 0 < result.engineering_depth <= WEIGHTS["engineering_depth"]


def test_total_never_exceeds_100():
    cand = _cand(
        skills=["python", "rag", "langchain", "docker", "aws", "fastapi",
                "pytest", "react"],
        projects=["RAG with embeddings, vector search, multi-agent "
                  "orchestration, evaluation, fine-tuning, deployed with "
                  "Docker on AWS via CI/CD, FastAPI backend, Kafka, "
                  "monitoring, testing, React frontend, full-stack"],
        raw_text=" ".join(["rag embeddings vector search multi-agent agent "
                           "tool calling orchestration evaluation fine-tuning "
                           "langgraph llamaindex faiss pinecone chromadb"] * 3),
    )
    result = score_candidate(cand, _eligible(cand))
    assert result.total_score <= 100


def test_total_never_below_zero():
    cand = _cand(
        skills=["python", "openai api"],
        projects=["Tutorial chatbot: simple basic thin OpenAI API wrapper"],
        raw_text="python openai simple chatbot tutorial wrapper",
    )
    result = score_candidate(cand, _eligible(cand))
    assert result.total_score >= 0
    assert result.ai_project_depth >= 0


def test_skills_only_mentions_earn_little():
    """Framework name in skills without project detail must not give full points."""
    cand = _cand(
        skills=["python", "rag", "langchain", "embeddings", "vector search",
                "agent", "multi-agent", "fine-tuning", "pytorch"],
        raw_text="python rag langchain embeddings vector search agent "
                 "multi-agent fine-tuning pytorch",
    )
    pts, reason = _ai_score(cand)
    assert pts < WEIGHTS["ai_project_depth"]


def test_llm_failure_falls_back_deterministic():
    class Broken:
        def evaluate(self, text):
            raise RuntimeError("api down")

    cand = _cand(
        skills=["python", "langchain", "rag"],
        projects=["Built a RAG pipeline with embeddings, vector search and "
                  "retrieval using LangChain"],
        raw_text="python langchain rag embeddings vector search retrieval",
    )
    result = score_candidate(cand, _eligible(cand), llm_evaluator=Broken())
    assert result.llm_used is False
    assert result.llm_error and "api down" in result.llm_error
    assert result.ai_project_depth > 0  # deterministic score kept


# ── Fix 2: double-penalty tests ───────────────────────────────────────────────

def test_wrapper_penalty_fires_at_most_once():
    """A single thin-wrapper project must not accumulate more than -15 from
    the wrapper-hint logic.  The total wrapper deduction must be exactly -15."""
    penalties = _penalties(
        "Simple OpenAI API wrapper around GPT-4",
        "python openai",
    )
    wrapper_deductions = sum(pts for pts, _ in penalties if pts == -15)
    assert wrapper_deductions == -15, (
        f"Expected exactly one -15 wrapper penalty, got: {penalties}"
    )


def test_simple_openai_wrapper_has_no_overlapping_penalties():
    penalties = _penalties("Simple OpenAI API wrapper", "python openai")
    assert sum(points for points, _ in penalties) == -15


def test_wrapper_and_shallow_chatbot_are_not_double_counted():
    penalties = _penalties(
        "Simple chatbot using the OpenAI API only", "python openai"
    )
    assert sum(points for points, _ in penalties) == -15


def test_independent_tutorial_penalty_stacks_with_wrapper():
    penalties = _penalties(
        "Tutorial project: simple OpenAI API wrapper", "python openai"
    )
    assert sum(points for points, _ in penalties) == -25
    assert any("tutorial-style" in reason for _, reason in penalties)


def test_strong_rag_no_wrapper_penalty():
    """A genuine RAG project must receive zero wrapper penalties."""
    penalties = _penalties(
        "Built a RAG system with embeddings, vector search and LangChain retrieval",
        "python langchain rag",
    )
    assert not any(pts == -15 for pts, _ in penalties), (
        f"Strong RAG project should not be penalised: {penalties}"
    )


def test_total_wrapper_penalty_capped():
    """Even with both a simple-chatbot and a wrapper pattern, total penalty
    on ai_project_depth must not push it below 0."""
    cand = _cand(
        skills=["python", "openai api"],
        projects=["Simple chatbot: a wrapper around the OpenAI API"],
        raw_text="python openai api simple chatbot wrapper",
    )
    result = score_candidate(cand, _eligible(cand))
    assert result.ai_project_depth >= 0
    assert result.total_score >= 0


# ── Fix 3: no-section-header AI scoring tests ─────────────────────────────────

def test_raw_text_rag_scores_above_zero():
    """A resume with strong AI evidence in plain prose (no section headers)
    must receive ai_project_depth > 0."""
    cand = _cand(
        skills=["python", "langchain"],
        # No projects or experience — only raw_text
        raw_text="python langchain built a RAG system using embeddings and vector search",
    )
    pts, reason = _ai_score(cand)
    assert pts > 0, f"Expected ai_score > 0 for raw-text RAG evidence, got {pts}. Reason: {reason}"


def test_raw_text_agentic_scores_above_zero():
    """LangGraph / agentic workflow in raw text must score > 0."""
    cand = _cand(
        skills=["python", "langgraph"],
        raw_text="python langgraph built an agentic workflow using LangGraph with tool calling",
    )
    pts, _ = _ai_score(cand)
    assert pts > 0


def test_exact_plain_text_rag_scores_above_zero():
    cand = _cand(raw_text="Built a RAG system using embeddings and vector search.")
    pts, _ = _ai_score(cand)
    assert pts > 0


def test_raw_text_interested_in_ai_scores_zero():
    """'Interested in AI' with no implementation keywords must score 0."""
    cand = _cand(
        skills=["python"],
        raw_text="python interested in AI AI enthusiast learning AI",
    )
    pts, _ = _ai_score(cand)
    assert pts == 0


def test_raw_text_cap_below_structured():
    """Raw-text-only AI score must be strictly lower than the same evidence
    placed in a structured projects section."""
    keywords = "rag embeddings vector search retrieval langchain"
    raw_only = _cand(
        skills=["python", "langchain"],
        raw_text=f"python {keywords}",
    )
    structured = _cand(
        skills=["python", "langchain"],
        projects=[f"Built a RAG system with {keywords}"],
        raw_text=f"python {keywords}",
    )
    pts_raw, _ = _ai_score(raw_only)
    pts_structured, _ = _ai_score(structured)
    assert pts_raw < pts_structured, (
        f"Raw-text score ({pts_raw}) should be lower than structured score ({pts_structured})"
    )
