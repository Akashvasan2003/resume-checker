"""Field extraction - turns raw resume text into a Candidate.

Heuristic-based and layout-agnostic: works by fuzzy section headers and
regex patterns rather than fixed positions.
"""
import re
from typing import List, Optional

from .models import Candidate

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE_RE = re.compile(r"(\+?\d[\d\s().-]{7,}\d)")
GITHUB_RE = re.compile(r"(?:https?://)?(?:www\.)?github\.com/[\w.-]+(?:/[\w./-]*)?", re.IGNORECASE)

SECTION_PATTERNS = {
    "skills": re.compile(r"^\s*(technical\s+)?skills?\b", re.IGNORECASE | re.MULTILINE),
    "projects": re.compile(r"^\s*(personal\s+)?projects?\b", re.IGNORECASE | re.MULTILINE),
    "education": re.compile(r"^\s*(academic\s+)?(education|qualifications?|academics?)\b", re.IGNORECASE | re.MULTILINE),
    "experience": re.compile(r"^\s*(work\s+|professional\s+|relevant\s+)?(experience|employment|internships?)\b", re.IGNORECASE | re.MULTILINE),
}

# Common skill keywords found anywhere in the text (layout-agnostic fallback)
SKILL_KEYWORDS = {
    "python", "java", "c++", "javascript", "typescript", "go", "rust", "sql",
    "html", "css", "react", "node.js", "django", "flask", "fastapi",
    "pytorch", "tensorflow", "keras", "scikit-learn", "pandas", "numpy",
    "langchain", "llama", "openai", "hugging face", "transformers",
    "docker", "kubernetes", "git", "aws", "azure", "gcp", "linux",
    "mongodb", "postgresql", "mysql", "redis", "rest", "graphql",
}


def _split_sections(text: str) -> dict:
    """Split text into sections using fuzzy header matching."""
    sections: dict = {}
    matches = sorted(
        (m for patterns in SECTION_PATTERNS.values() for m in [patterns.search(text)] if m),
        key=lambda m: m.start(),
    )
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[m.end():end].strip()
        for key, pattern in SECTION_PATTERNS.items():
            if pattern.pattern == m.re.pattern:
                sections.setdefault(key, body)
                break
    return sections


def extract_name(text: str) -> Optional[str]:
    """First non-empty line that looks like a person name.

    Accepts letters, hyphens, and apostrophes within words so that names
    like Mary-Jane Watson, Jean-Pierre Dupont, and O'Brien are handled.
    """
    for line in text.splitlines()[:10]:
        line = line.strip()
        if not line or "@" in line or any(ch.isdigit() for ch in line):
            continue
        if line.casefold() in {"professional summary", "career summary"}:
            continue
        words = line.split()
        valid_words = all(
            all(part.isalpha() for part in re.split(r"[-']", word))
            for word in words
        )
        has_name_shape = len(words) > 1 or any(mark in words[0] for mark in "-'")
        if 0 < len(words) <= 5 and valid_words and has_name_shape:
            return line.title()
    return None


def extract_email(text: str) -> Optional[str]:
    m = EMAIL_RE.search(text)
    return m.group(0).lower() if m else None


def extract_phone(text: str) -> Optional[str]:
    # Prefer phones near contact context to reduce false positives
    for line in text.splitlines()[:15]:
        m = PHONE_RE.search(line)
        if m and len(re.sub(r"\D", "", m.group(1))) >= 8:
            return re.sub(r"\s+", " ", m.group(1)).strip()
    return None


def extract_github(text: str) -> Optional[str]:
    m = GITHUB_RE.search(text)
    if m:
        url = m.group(0)
        return url if url.startswith("http") else f"https://{url}"
    return None


def extract_skills(text: str, sections: dict) -> List[str]:
    found: List[str] = []
    haystack = sections.get("skills", "") + "\n" + text
    low = haystack.lower()
    for skill in sorted(SKILL_KEYWORDS, key=len, reverse=True):
        if skill in low and skill not in found:
            found.append(skill)
    return found


def extract_lines(body: Optional[str], max_items: int = 10) -> List[str]:
    """Turn a section body into a cleaned list of lines."""
    if not body:
        return []
    lines = [re.sub(r"\s+", " ", ln).strip(" •-–|") for ln in body.splitlines()]
    return [ln for ln in lines if ln][:max_items]


def extract_candidate(raw_text: str) -> Candidate:
    sections = _split_sections(raw_text)
    return Candidate(
        name=extract_name(raw_text),
        email=extract_email(raw_text),
        phone=extract_phone(raw_text),
        github_url=extract_github(raw_text),
        skills=extract_skills(raw_text, sections),
        projects=extract_lines(sections.get("projects")),
        education=extract_lines(sections.get("education")),
        experience=extract_lines(sections.get("experience")),
        raw_text=raw_text,
    )
