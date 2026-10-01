"""Shared helpers - batching, hashing, safe file handling."""
import hashlib
import logging
from pathlib import Path
from typing import List

from .config import SUPPORTED_EXTENSIONS

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("resume-screener")


def sha256_of(path: Path, chunk_size: int = 65536) -> str:
    """Content hash of a file - used for exact duplicate detection."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def list_resume_files(input_dir: Path) -> List[Path]:
    """All supported resume files in a folder (non-recursive, sorted)."""
    if not input_dir.exists():
        logger.warning("Input folder does not exist: %s", input_dir)
        return []
    return sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def safe(fn, *args, default=None):
    """Run fn; never let one bad item kill the batch."""
    try:
        return fn(*args)
    except Exception as exc:
        logger.error("Error in %s: %s", getattr(fn, "__name__", fn), exc)
        return default
