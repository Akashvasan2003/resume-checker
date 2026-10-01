"""Evidence-based 100-point scoring (deterministic core + optional LLM assist).

Weights (assignment baseline):
  AI/Agentic/RAG project depth ... 40
  Python & backend engineering ... 30
  Cloud / deployment / fullstack . 15
  GitHub activity ................ 10  (filled in Part 4)
  Engineering depth signals ...... 5

Skills-list mentions alone earn little; points come from implementation
evidence in projects/experience text. Shallow AI projects are penalized.
"""
import re
from typing import List, Optional, Tuple

from .models import Candidate, EligibilityResult, ScoreResult

WEIGHTS = {
    "ai_project_depth": 40,
    "python_backend": 30,
    "cloud_fullstack": 15,
    "github": 10,
    "engineering_depth": 5,
}

# ---- keyword tables (each hit earns partial credit, capped per category) ----

# (keyword, points) - strong AI implementation signals
AI_STRONG: List[Tuple[str, int]] = [
    ("rag", 10), ("retrieval", 8), ("retrieval-augmented", 10),
    ("embeddings", 8), ("vector search", 8), ("vector database", 8), ("faiss", 6),
    ("pinecone", 6), ("chromadb", 6), ("milvus", 6),
    ("multi-agent", 10), ("agent", 7), ("tool calling", 8), ("function calling", 8),
    ("orchestration", 7), ("state management", 5), ("memory", 4),
    ("langgraph", 9), ("langchain", 6), ("llamaindex", 6), ("google adk", 8),
    ("fine-tun", 8), ("lora", 5), ("qlora", 6),
    ("evaluation", 6), ("eval harness", 8), ("benchmark", 5),
]
AI_MODERATE: List[Tuple[str, int]] = [
    ("llm", 5), ("openai api", 4), ("gemini api", 4), ("claude api", 4),
    ("hugging face", 4), ("transformers", 4), ("pytorch", 4), ("tensorflow", 3),
]

PYTHON_CORE: List[Tuple[str, int]] = [
    ("python", 10), ("fastapi", 7), ("flask", 6), ("django", 7),
    ("async", 5), ("await", 3), ("postgresql", 5), ("mysql", 3), ("sqlite", 2),
    ("redis", 4), ("celery", 5), ("background worker", 6),
    ("database design", 6), ("schema design", 5), ("rest api", 6),
    ("backend", 5), ("crud", 3), ("sqlalchemy", 4), ("orm", 3),
]

CLOUD: List[Tuple[str, int]] = [
    ("docker", 5), ("kubernetes", 4), ("aws", 5), ("gcp", 4), ("azure", 4),
    ("ci/cd", 5), ("github actions", 4), ("deployment", 4), ("deployed", 4),
    ("heroku", 2), ("vercel", 3), ("render", 2), ("terraform", 3),
]
FULLSTACK: List[Tuple[str, int]] = [
    ("full-stack", 5), ("fullstack", 5), ("react", 4), ("next.js", 4),
    ("frontend", 2),
]

DEPTH: List[Tuple[str, int]] = [
    ("unit test", 2), ("pytest", 2), ("testing", 2), ("test coverage", 2),
    ("architecture", 2), ("system design", 2), ("caching", 2),
    ("message queue", 2), ("kafka", 2), ("rabbitmq", 2), ("sqs", 2),
    ("observability", 2), ("monitoring", 2), ("logging", 1),
    ("concurrency", 2), ("thread", 1), ("error handling", 2),
    ("scalab", 2), ("load balanc", 2), ("docker compose", 1),
]

# Shallow-project penalties
# A wrapper replaces overlapping shallow-chatbot penalties; tutorial penalties
# remain independent and may still stack with it.
SHALLOW_PATTERNS = [
    (r"simple chatbot|basic chatbot", -10, "simple chatbot with no real workflow"),
    # Removed the duplicate thin-wrapper entry here; WRAPPER_HINTS handles it below.
    (r"tutorial|followed a (?:guide|tutorial)|todo app", -10,
     "tutorial-style project with no implementation evidence"),
    (r"chatbot using (?:the )?openai api(?: only)?\.?$", -12,
     "chatbot that only calls the OpenAI API"),
    (r"ai project(?! (?:that|with|using|involving))", -8,
     "'AI project' with no implementation details"),
]
# WRAPPER_HINTS fires at most once and is checked separately from SHALLOW_PATTERNS.
WRAPPER_HINTS = [
    r"wrapper (?:around|for) (?:the )?(?:openai|llm|gpt|gemini|claude)",
    r"(?:simple|basic|thin) (?:openai|llm|gpt) (?:api )?wrapper",
    r"using (?:the )?(?:openai|llm|gpt|gemini|claude) api (?:only|to (?:generate|answer|get))",
]

MAX_HITS_PER_CATEGORY = 6  # cap keyword accumulation


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def _impl_text(candidate: Candidate) -> str:
    """Text where implementation evidence lives: projects + experience."""
    return "\n".join(candidate.projects + candidate.experience)


def _score_keywords(
    text: str, table: List[Tuple[str, int]], max_points: int,
    cap_hits: int = MAX_HITS_PER_CATEGORY,
) -> Tuple[int, List[str]]:
    """Sum keyword points found in implementation text (deduped, capped)."""
    low = _clean(text)
    hits: List[str] = []
    total = 0
    seen = set()
    for kw, pts in table:
        if kw in low and kw not in seen:
            seen.add(kw)
            hits.append(kw)
            total += pts
            if len(seen) >= cap_hits:
                break
    return min(total, max_points), hits


def _penalties(impl_text: str, skills_text: str) -> List[Tuple[int, str]]:
    """Assignment project-quality penalties.

    A thin wrapper gets one -15 penalty. Tutorial-only evidence remains
    independent; generic chatbot/API shallowness is not stacked on that same
    wrapper offense.
    """
    found: List[Tuple[int, str]] = []
    low_impl = _clean(impl_text)
    has_wrapper = any(
        re.search(pattern, low_impl, re.IGNORECASE)
        for pattern in WRAPPER_HINTS
    )
    for pattern, pts, reason in SHALLOW_PATTERNS:
        if not re.search(pattern, low_impl, re.IGNORECASE):
            continue
        if has_wrapper and "tutorial-style" not in reason:
            continue
        found.append((pts, reason))
    if has_wrapper:
        found.append((-15, "thin LLM/API wrapper with no meaningful "
                           "workflow, retrieval, processing, state "
                           "management, backend logic or evaluation"))
    return found


# Cap for raw-text-only AI evidence (no structured project/experience sections).
# Lower than the full 40 to still reward structured resumes more.
_RAW_TEXT_AI_CAP = 20


def _ai_score(candidate: Candidate) -> Tuple[int, str]:
    text = _impl_text(candidate)
    if not _clean(text):
        # No structured project/experience sections — scan raw_text with both
        # strong and moderate tables but apply a lower cap so that random
        # keyword mentions in an unstructured resume don't score as high as
        # a candidate who clearly described their implementation.
        pts, hits = _score_keywords(
            candidate.raw_text, AI_STRONG + AI_MODERATE, _RAW_TEXT_AI_CAP
        )
        return pts, (
            f"AI depth {pts}/40 (raw-text scan, cap {_RAW_TEXT_AI_CAP}): "
            f"evidence: {', '.join(hits) if hits else 'none'} — "
            "no structured project/experience sections found"
        )
    pts, hits = _score_keywords(text, AI_STRONG + AI_MODERATE, WEIGHTS["ai_project_depth"])
    reasons = (
        f"AI project depth {pts}/40 - evidence: {', '.join(hits)} "
        "(rewarded for retrieval, agents, orchestration, evaluation and "
        "similar implementation signals rather than framework names in skills)"
    )
    return pts, reasons


def _python_score(candidate: Candidate) -> Tuple[int, str]:
    text = _impl_text(candidate) or candidate.raw_text
    pts, hits = _score_keywords(text, PYTHON_CORE, WEIGHTS["python_backend"])
    reasons = (
        f"Python & backend {pts}/30 - evidence: {', '.join(hits)}"
        if hits else "Python & backend 0/30 - no implementation evidence"
    )
    return pts, reasons


def _cloud_score(candidate: Candidate) -> Tuple[int, str]:
    text = _impl_text(candidate) or candidate.raw_text
    pts, hits = _score_keywords(text, CLOUD, WEIGHTS["cloud_fullstack"])
    fs_pts, fs_hits = _score_keywords(text, FULLSTACK, 5)
    total = min(pts + fs_pts, WEIGHTS["cloud_fullstack"])
    reasons = f"Cloud/fullstack {total}/15 - evidence: {', '.join(hits + fs_hits)}"
    return total, reasons


def _depth_score(candidate: Candidate) -> Tuple[int, str]:
    text = _impl_text(candidate) or candidate.raw_text
    pts, hits = _score_keywords(text, DEPTH, WEIGHTS["engineering_depth"])
    reasons = f"Engineering depth {pts}/5 - evidence: {', '.join(hits)}"
    return pts, reasons


def score_candidate(
    candidate: Candidate,
    eligibility: EligibilityResult,
    github_points: int = 0,
    llm_evaluator=None,
) -> ScoreResult:
    """Score an eligible candidate. Deterministic unless an LLM evaluator
    is passed (Part 4 may extend GitHub; LLM only refines AI project depth)."""
    assert eligibility.eligible, "score only eligible candidates"

    ai_pts, ai_reason = _ai_score(candidate)
    py_pts, py_reason = _python_score(candidate)
    cloud_pts, cloud_reason = _cloud_score(candidate)
    depth_pts, depth_reason = _depth_score(candidate)
    github_pts = max(0, min(int(github_points), WEIGHTS["github"]))
    github_reason = f"GitHub activity {github_pts}/10 (enrichment arrives in Part 4)"

    penalties: List[str] = []
    llm_error = None
    llm_used = False
    if llm_evaluator is not None:
        try:
            verdict = llm_evaluator.evaluate(
                _impl_text(candidate) or candidate.raw_text
            )
            llm_used = True
            if verdict.penalty_points:
                ai_pts = max(0, ai_pts - verdict.penalty_points)
                penalties.append(
                    f"LLM quality assessment: -{verdict.penalty_points} "
                    f"({verdict.penalty_reason})"
                )
        except Exception as exc:
            llm_error = str(exc)
            penalties.append(f"LLM evaluation failed, deterministic score kept: {exc}")

    for pts, reason in _penalties(_impl_text(candidate), candidate.raw_text):
        ai_pts = max(0, ai_pts + pts)
        penalties.append(f"{reason} ({pts:+d} AI points)")

    ai_pts = max(0, min(ai_pts, WEIGHTS["ai_project_depth"]))
    py_pts = max(0, min(py_pts, WEIGHTS["python_backend"]))
    cloud_pts = max(0, min(cloud_pts, WEIGHTS["cloud_fullstack"]))
    depth_pts = max(0, min(depth_pts, WEIGHTS["engineering_depth"]))

    total = max(0, min(
        ai_pts + py_pts + cloud_pts + github_pts + depth_pts, 100
    ))

    return ScoreResult(
        ai_project_depth=ai_pts,
        python_backend=py_pts,
        cloud_fullstack=cloud_pts,
        github=github_pts,
        engineering_depth=depth_pts,
        total_score=total,
        reasons=[ai_reason, py_reason, cloud_reason, github_reason, depth_reason],
        penalties=penalties,
        llm_used=llm_used,
        llm_error=llm_error,
    )


def rank_records(records, github_enricher=None, llm_evaluator=None) -> List:
    """Score eligible records (with optional GitHub enrichment and LLM evaluator)
    and rank them in-place by total_score descending."""
    eligible = []
    for r in records:
        if r.ok and r.eligibility and r.eligibility.eligible:
            github_points = 0
            github_result = None
            if github_enricher is not None and r.candidate.github_url:
                github_result = github_enricher.enrich(r.candidate.github_url)
                github_points = github_result.get("github_score", 0)
                r.github = github_result
            r.score = score_candidate(
                r.candidate, r.eligibility,
                github_points=github_points,
                llm_evaluator=llm_evaluator,
            )
            if github_result:
                r.score.reasons.append(
                    f"GitHub activity {github_points}/10 - "
                    f"{github_result.get('github_summary', '')}"
                )
            eligible.append(r)
    eligible.sort(key=lambda r: r.score.total_score, reverse=True)
    for i, r in enumerate(eligible, 1):
        r.score.rank = i
    return records
