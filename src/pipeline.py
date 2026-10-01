"""Ingestion pipeline - orchestrates parse -> extract -> dedupe."""
from pathlib import Path

from .config import OUTPUT_DIR
from .eligibility import check_eligibility
from .extractor import extract_candidate
from .models import BatchResult, Candidate, ResumeRecord
from .parser import extract_text
from .scoring import rank_records
from .utils import list_resume_files, logger, sha256_of


def _enricher():
    """Build a GitHub enricher unless enrichment is disabled via env."""
    import os
    from .github_client import GitHubEnricher
    if os.getenv("DISABLE_GITHUB_ENRICHMENT", "").lower() in ("1", "true", "yes"):
        return None
    return GitHubEnricher()


def _llm_evaluator():
    """Build an LLM evaluator when OPENAI_API_KEY is configured, else None.

    The LLM is entirely optional:
    - No OPENAI_API_KEY -> returns None, deterministic scoring is used.
    - Key present but API call fails at runtime -> score_candidate() catches
      the exception, records llm_error, and keeps the deterministic score.
    - Hard eligibility is never affected by the LLM.
    """
    import os
    from .llm import OpenAICompatEvaluator
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return None
    try:
        return OpenAICompatEvaluator(api_key=api_key)
    except Exception as exc:
        logger.warning("LLM evaluator could not be initialised: %s", exc)
        return None


def process_resume(path: Path) -> ResumeRecord:
    """Process a single resume file. Always returns a record, never raises."""
    try:
        text = extract_text(path)
        candidate = extract_candidate(text)
        record = ResumeRecord(filename=path.name, candidate=candidate)
        record.eligibility = check_eligibility(candidate)
        return record
    except Exception as exc:
        logger.error("Failed to parse %s: %s", path.name, exc)
        return ResumeRecord(filename=path.name, error=str(exc))


def ingest_folder(input_dir: Path) -> BatchResult:
    """Process every resume in a folder; one bad file never stops the batch."""
    files = list_resume_files(input_dir)
    records: list = []
    seen_hashes: set = set()

    for path in files:
        record = process_resume(path)

        # Exact duplicate detection by file content hash
        try:
            digest = sha256_of(path)
        except Exception as exc:
            logger.error("Could not hash %s: %s", path.name, exc)
            digest = None
        if digest and digest in seen_hashes:
            record.is_duplicate = True
            logger.info("Duplicate file detected: %s", path.name)
        elif digest:
            seen_hashes.add(digest)

        records.append(record)

    parsed = sum(1 for r in records if r.ok and not r.is_duplicate)
    duplicates = sum(1 for r in records if r.is_duplicate)
    failed = len(records) - parsed - duplicates
    eligible = sum(
        1 for r in records
        if r.ok and not r.is_duplicate and r.eligibility and r.eligibility.eligible
    )
    # Score eligible candidates only, then rank (0 stays 0 on failure).
    # Both GitHub enrichment and LLM evaluation are optional; failures are
    # caught inside rank_records / score_candidate and never crash the batch.
    try:
        rank_records(
            [r for r in records if r.ok and not r.is_duplicate],
            github_enricher=_enricher(),
            llm_evaluator=_llm_evaluator(),
        )
    except Exception as exc:
        logger.error("Ranking failed, continuing without scores: %s", exc)
    return BatchResult(
        total_files=len(files), parsed=parsed, failed=failed,
        duplicates=duplicates, eligible=eligible, records=records,
    )


def save_results(result: BatchResult, output_path: Path) -> Path:
    """Write the JSON output, creating parent dirs as needed."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    logger.info("Results written to %s", output_path)
    return output_path
