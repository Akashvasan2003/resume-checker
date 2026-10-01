"""Configuration module - keeps paths & settings separate from business logic."""
import os
from pathlib import Path
from dotenv import load_dotenv

# Defaults are relative to the project root (parent of src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

INPUT_DIR = Path(os.getenv("INPUT_DIR", str(PROJECT_ROOT / "resumes")))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", str(PROJECT_ROOT / "output")))

# Optional: used later for LLM scoring / GitHub enrichment (Parts 2-3)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


def ensure_dirs() -> None:
    """Make sure the standard folders exist."""
    for d in (INPUT_DIR, OUTPUT_DIR):
        d.mkdir(parents=True, exist_ok=True)
