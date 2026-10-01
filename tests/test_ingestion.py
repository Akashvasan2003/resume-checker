"""Tests for the ingestion pipeline."""
from pathlib import Path

import pymupdf
import pytest

from src.extractor import extract_candidate, extract_github, extract_name
from src.llm import OpenAICompatEvaluator, ProjectQualityVerdict
from src.models import Candidate
from src.parser import extract_text, extract_text_pdf
from src.pipeline import ingest_folder
import src.pipeline as pipeline


@pytest.fixture
def valid_pdf(tmp_path: Path) -> Path:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Jane Doe\njane.doe@example.com\n+1 415 555 0100\ngithub.com/janedoe\n\nSKILLS\nPython, PyTorch, FastAPI\n\nPROJECTS\nLLM agent playground\n\nEDUCATION\nB.Tech Computer Science\n\nEXPERIENCE\nSDE Intern at Acme - built REST APIs\n")
    path = tmp_path / "jane.pdf"
    doc.save(path)
    doc.close()
    return path


def test_valid_pdf_parsing(valid_pdf):
    text = extract_text_pdf(valid_pdf)
    assert "Jane Doe" in text
    cand = extract_candidate(text)
    assert cand.email == "jane.doe@example.com"
    assert cand.name == "Jane Doe"
    assert "python" in cand.skills
    assert "SDE Intern at Acme" in cand.experience[0]


def test_malformed_pdf(tmp_path):
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"%PDF-1.4 this is not a real pdf \x00\x01garbage")
    try:
        text = extract_text_pdf(bad)
    except ValueError:
        text = None
    if text is not None:
        assert isinstance(text, str)
    # and the batch must survive it
    good = tmp_path / "skip.txt"
    good.write_text("hello")
    result = ingest_folder(tmp_path)
    assert result.failed >= 1
    assert result.total_files == 2


def test_missing_fields(tmp_path):
    (tmp_path / "a.txt").write_text("no contact info at all")
    result = ingest_folder(tmp_path)
    cand = result.records[0].candidate
    assert isinstance(cand, Candidate)
    assert cand.email is None
    assert cand.phone is None


def test_github_url_extraction():
    text = "Check my work: https://github.com/janedoe/ml-projects"
    assert extract_github(text) == "https://github.com/janedoe/ml-projects"
    text2 = "github.com/foo-bar"
    assert extract_github(text2).endswith("github.com/foo-bar")


# ── Fix 4: name extraction with hyphens / apostrophes ────────────────────────

def test_extract_name_simple():
    assert extract_name("John Smith\njohn@example.com") == "John Smith"


def test_extract_name_skips_professional_summary_heading():
    assert extract_name(
        "Professional Summary\nJane Doe\njane@example.com"
    ) == "Jane Doe"


def test_extract_name_hyphenated():
    assert extract_name("Mary-Jane Watson\nmary@example.com") == "Mary-Jane Watson"


def test_extract_name_hyphenated_first():
    assert extract_name("Jean-Pierre Dupont\njp@example.com") == "Jean-Pierre Dupont"


def test_extract_name_accepts_accented_letters():
    assert extract_name("Élodie Martin\nelodie@example.com") == "Élodie Martin"


def test_extract_name_apostrophe():
    assert extract_name("O'Brien\nob@example.com") == "O'Brien"


def test_extract_name_apostrophe_with_second_name():
    assert extract_name("O'Brien Connor\nob@example.com") == "O'Brien Connor"


def test_extract_name_all_caps():
    assert extract_name("JOHN SMITH\njohn@example.com") == "John Smith"


def test_extract_name_all_lower():
    assert extract_name("john smith\njohn@example.com") == "John Smith"


# ── Fix 5: pipeline integration tests ────────────────────────────────────────

def _write(path: Path, name: str, content: str) -> Path:
    f = path / name
    f.write_text(content, encoding="utf-8")
    return f


def test_batch_math(tmp_path):
    """parsed + duplicates + failed == total_files for a mixed batch."""
    content = (
        "Jane Doe\njane@example.com\n\nSkills\nPython, LangChain\n\n"
        "Projects\nBuilt a RAG pipeline with embeddings and vector search"
    )
    _write(tmp_path, "r1.txt", content)          # eligible
    _write(tmp_path, "r2.txt", content)          # duplicate of r1
    _write(tmp_path, "r3.txt",                   # ineligible (python only)
           "Bob Smith\nbob@example.com\n\nSkills\nPython, FastAPI\n\n"
           "Projects\nBuilt a REST API with FastAPI")
    result = ingest_folder(tmp_path)
    assert result.total_files == 3
    assert result.parsed + result.duplicates + result.failed == result.total_files


def test_duplicate_detection(tmp_path):
    content = (
        "Jane Doe\njane@example.com\n\nSkills\nPython, LangChain\n\n"
        "Projects\nBuilt a RAG pipeline with embeddings and vector search"
    )
    _write(tmp_path, "a.txt", content)
    _write(tmp_path, "b.txt", content)  # exact duplicate
    result = ingest_folder(tmp_path)
    assert result.duplicates == 1
    dup_records = [r for r in result.records if r.is_duplicate]
    assert len(dup_records) == 1


def test_ranking_order(tmp_path):
    """Highest total_score must be rank 1; scores must be descending."""
    _write(tmp_path, "strong.txt",
           "Alice Smith\nalice@example.com\n\nSkills\nPython, LangChain, RAG\n\n"
           "Projects\nBuilt a RAG system with embeddings, vector search, retrieval "
           "and LangChain. Deployed on AWS with Docker and CI/CD.\n\n"
           "Experience\nSDE Intern: async FastAPI backend with PostgreSQL")
    _write(tmp_path, "weak.txt",
           "Bob Jones\nbob@example.com\n\nSkills\nPython, LangChain\n\n"
           "Projects\nBuilt a simple chatbot using the OpenAI API")
    result = ingest_folder(tmp_path)
    ranked = sorted(
        [r for r in result.records if r.score is not None],
        key=lambda r: r.score.rank,
    )
    assert len(ranked) >= 1
    scores = [r.score.total_score for r in ranked]
    assert scores == sorted(scores, reverse=True), "Scores must be descending by rank"
    assert ranked[0].score.rank == 1


def test_rejected_candidates_not_ranked(tmp_path):
    """Ineligible candidates must have score=None and no rank."""
    _write(tmp_path, "inelig.txt",
           "Bob Jones\nbob@example.com\n\nSkills\nJava, React\n\n"
           "Projects\nBuilt a React app with Java backend")
    result = ingest_folder(tmp_path)
    for r in result.records:
        if r.eligibility and not r.eligibility.eligible:
            assert r.score is None, "Ineligible candidate must not have a score"


def test_malformed_resume_does_not_crash_batch(tmp_path):
    """A completely corrupt file must not stop other files from being processed."""
    (tmp_path / "corrupt.pdf").write_bytes(b"not a pdf at all")
    _write(tmp_path, "good.txt",
           "Jane Doe\njane@example.com\n\nSkills\nPython, LangChain\n\n"
           "Projects\nBuilt a RAG pipeline with embeddings and vector search")
    result = ingest_folder(tmp_path)
    assert result.total_files == 2
    assert result.failed >= 1
    assert result.parsed >= 1


def test_pipeline_uses_configured_llm_evaluator(tmp_path, monkeypatch):
    class FakeEvaluator:
        def __init__(self):
            self.inputs = []

        def evaluate(self, project_text):
            self.inputs.append(project_text)
            return ProjectQualityVerdict(quality="strong", penalty_points=0)

    evaluator = FakeEvaluator()
    monkeypatch.setattr(pipeline, "_enricher", lambda: None)
    monkeypatch.setattr(pipeline, "_llm_evaluator", lambda: evaluator)
    _write(
        tmp_path,
        "candidate.txt",
        "Jane Doe\nSkills\nPython, LangChain\nProjects\n"
        "Built a RAG pipeline with embeddings and vector search",
    )

    result = ingest_folder(tmp_path)

    scored = result.records[0].score
    assert scored is not None and scored.llm_used
    assert len(evaluator.inputs) == 1
    assert "RAG pipeline" in evaluator.inputs[0]


def test_llm_factory_is_optional_and_environment_configured(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert pipeline._llm_evaluator() is None

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    assert isinstance(pipeline._llm_evaluator(), OpenAICompatEvaluator)
