"""CLI entry point for the AI Resume Screening & Ranking System (Part 1: ingestion)."""
import argparse
from pathlib import Path

from src.config import INPUT_DIR, OUTPUT_DIR, ensure_dirs
from src.pipeline import ingest_folder, save_results
from src.utils import logger


def main() -> None:
    ap = argparse.ArgumentParser(description="AI Resume Screening - ingestion & extraction")
    ap.add_argument("--input", default=None, help="Folder containing resumes")
    ap.add_argument("--output", default=None, help="Path for results JSON")
    args = ap.parse_args()

    ensure_dirs()
    input_dir = Path(args.input) if args.input else INPUT_DIR
    output_path = Path(args.output) if args.output else OUTPUT_DIR / "results.json"

    result = ingest_folder(input_dir)
    save_results(result, output_path)

    logger.info(
        "Done: %d files, %d parsed, %d duplicates, %d failed",
        result.total_files, result.parsed, result.duplicates, result.failed,
    )


if __name__ == "__main__":
    main()
