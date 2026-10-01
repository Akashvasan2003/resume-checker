"""File parsing - extracts raw text from PDF/DOCX/TXT files."""
from pathlib import Path

import pymupdf


def extract_text_pdf(path: Path) -> str:
    """Extract text from a PDF. Raises ValueError on unreadable files."""
    try:
        with pymupdf.open(path) as doc:
            text = "\n".join(page.get_text() for page in doc)
        if not text.strip():
            raise ValueError("no extractable text (may be a scanned image)")
        return text
    except Exception as exc:
        raise ValueError(f"unreadable PDF: {exc}") from exc


def extract_text_txt(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        raise ValueError(f"unreadable TXT: {exc}") from exc


def extract_text_docx(path: Path) -> str:
    """Optional DOCX support via python-docx."""
    try:
        from docx import Document

        doc = Document(str(path))
        return "\n".join(p.text for p in doc.paragraphs)
    except Exception as exc:
        raise ValueError(f"unreadable DOCX: {exc}") from exc


def extract_text(path: Path) -> str:
    """Dispatch by extension; raises ValueError on unsupported/unreadable files."""
    ext = path.suffix.lower()
    if ext == ".pdf":
        return extract_text_pdf(path)
    if ext == ".txt":
        return extract_text_txt(path)
    if ext == ".docx":
        return extract_text_docx(path)
    raise ValueError(f"unsupported file type: {ext}")
