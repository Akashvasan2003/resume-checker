# AI Resume Screening & Ranking System (SDE Intern Assignment)

Batch-processes PDF / DOCX / TXT resumes, applies hard eligibility filtering,
scores candidates on a 100-point rubric, optionally enriches with GitHub
activity, and optionally refines AI project quality with an LLM.

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env          # fill in keys as needed (all optional)
python main.py --input ./resumes --output ./output/results.json
```

## Structure

| Module | Purpose |
|---|---|
| `src/config.py` | Paths & env-based configuration |
| `src/models.py` | Pydantic models (`Candidate`, `ResumeRecord`, `BatchResult`, …) |
| `src/parser.py` | PDF / DOCX / TXT text extraction (PyMuPDF) |
| `src/extractor.py` | Layout-agnostic field extraction (name, email, phone, skills, projects, experience, GitHub URL) |
| `src/eligibility.py` | **Deterministic** hard filter — requires Python evidence **and** meaningful AI/LLM/agentic evidence; no LLM involved |
| `src/scoring.py` | Evidence-based 100-point scoring (AI depth 40, Python/backend 30, cloud/full-stack 15, GitHub 10, engineering depth 5) |
| `src/llm/` | **Optional** LLM adapter — refines AI project-quality score only; structured Pydantic output; graceful fallback |
| `src/github_client.py` | GitHub public API enrichment (0–10 points); all failure modes handled gracefully |
| `src/pipeline.py` | Batch orchestration, duplicate detection, JSON output |

## Scoring weights

| Category | Max points |
|---|---|
| AI / Agentic / RAG project depth | 40 |
| Python & backend engineering | 30 |
| Cloud / deployment / full-stack | 15 |
| GitHub activity | 10 |
| Engineering depth signals | 5 |
| **Total** | **100** |

## LLM integration (optional)

The LLM is used **only** to refine the AI project-depth score for eligible
candidates.  It never decides hard eligibility.

**To enable:** set `OPENAI_API_KEY` in your `.env` file.

**When disabled** (no key set): the system runs fully deterministically with
no loss of core functionality.

**When an API call fails at runtime:** the error is recorded in
`score.llm_error`, the deterministic score is kept, and the batch continues.

## GitHub enrichment (optional)

Set `GITHUB_TOKEN` to raise the API rate limit from 60 to 5 000 req/hour.

Set `DISABLE_GITHUB_ENRICHMENT=1` to skip GitHub calls entirely (useful for
offline testing or CI).

## Behavior

- Every supported file in the input folder is processed; a malformed file is
  logged and skipped — it never crashes the batch.
- Exact duplicate resumes are detected via SHA-256 content hash and flagged
  (`is_duplicate: true`).
- Only candidates with **both** Python evidence **and** meaningful AI/LLM/
  agentic evidence are eligible for scoring and ranking.
- Output: `output/results.json` — full summary and per-file records.

## Run tests

```bash
pytest
```
