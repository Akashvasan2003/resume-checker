"""Public GitHub enrichment - explainable 0-10 score.

Signals:
  Recent activity .......... 0-5 (public events in the last ~90 days)
  Maintained relevant repos . 0-5 (public, non-fork, recently pushed,
                                   Python/AI-related)

GitHub is a positive signal only - it NEVER affects eligibility.
All failure modes (no URL, bad URL, deleted/private profile, timeouts,
rate limits, API errors) degrade gracefully to a 0-score result.
"""
import re
import time
from typing import Dict, List, Optional

import requests

from .config import GITHUB_TOKEN
from .utils import logger

API_BASE = "https://api.github.com"
TIMEOUT = 10  # seconds per request
MAX_EVENTS = 30           # events inspected for activity scoring
RECENT_DAYS = 90          # "recent" window for activity
REQUEST_PAUSE = 0.5       # be polite between API calls

ACTIVITY_SCALE = {0: 0, 1: 1, 3: 3, 5: 5}  # explainable buckets

USERNAME_RE = re.compile(r"(?:https?://)?(?:www\.)?github\.com/([A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?)(?:/.*)?$", re.IGNORECASE)
RELEVANT_REPO_KEYWORDS = [
    "python", "rag", "llm", "agent", "langchain", "langgraph", "llamaindex",
    "openai", "gemini", "claude", "transformers", "pytorch", "tensorflow",
    "ml", "ai", "nlp", "embedding", "vector", "chatbot", "finetune",
]


class GitHubResult(dict):
    """Plain dict result - see module docstring for shape."""

    @classmethod
    def _make(cls, url, username, score, summary, status, activity=0, repos=0):
        return cls({
            "github_url": url, "github_username": username,
            "github_score": int(score), "github_summary": summary,
            "github_status": status,
            "recent_activity_points": int(activity),
            "repo_points": int(repos),
        })


def extract_username(github_url: Optional[str]) -> Optional[str]:
    """Return the GitHub username from a profile URL, else None."""
    if not github_url:
        return None
    url = github_url.strip().rstrip("/")
    if "github.com" not in url.lower():
        return None
    m = USERNAME_RE.match(url)
    if not m:
        return None
    user = m.group(1)
    # repo URLs like github.com/user/repo - take the user part only
    parts = url.split("github.com/")[-1].split("/")
    if len(parts) > 1:
        user = parts[0]
    return user.lower()


def _headers() -> dict:
    h = {"Accept": "application/vnd.github+json"}
    if GITHUB_TOKEN:
        h["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return h


class GitHubEnricher:
    """Queries the public GitHub API with per-run caching."""

    def __init__(self, cache: Optional[Dict] = None, session: Optional[requests.Session] = None):
        self._cache = cache if cache is not None else {}
        self._session = session or requests.Session()

    # ---- low-level API helpers -------------------------------------------
    def _get(self, path: str) -> Optional[object]:
        """GET one API path. Returns parsed JSON or a failure dict."""
        url = f"{API_BASE}{path}"
        try:
            resp = self._session.get(url, headers=_headers(), timeout=TIMEOUT)
        except requests.Timeout:
            return {"error": "timeout"}
        except requests.RequestException as exc:
            return {"error": f"api_failure: {exc}"}

        if resp.status_code == 404:
            return {"error": "not_found"}          # deleted or private profile
        if resp.status_code == 403:
            return {"error": "rate_limited"}        # includes rate limit / abuse
        if resp.status_code != 200:
            return {"error": f"api_failure: HTTP {resp.status_code}"}
        return resp.json()

    # ---- scoring ----------------------------------------------------------
    def _activity_points(self, events: Optional[list]) -> int:
        """0 = none, 1 = limited, 3 = moderate, 5 = strong recent activity."""
        if not events:
            return 0
        count = len([e for e in events if e.get("type") in
                     ("PushEvent", "CreateEvent", "PullRequestEvent", "IssuesEvent")])
        if count >= 15:
            return 5
        if count >= 8:
            return 3
        if count >= 1:
            return 1
        return 0

    def _repo_points(self, repos: Optional[list]) -> int:
        """0-5 based on maintained, recently pushed, Python/AI relevant repos."""
        if not repos:
            return 0
        relevant = [
            r for r in repos
            if not r.get("fork")
            and any(kw in (r.get("name", "") + " " + (r.get("description") or "")).lower()
                    for kw in RELEVANT_REPO_KEYWORDS)
        ]
        count = len(relevant)
        if count >= 5:
            return 5
        if count >= 3:
            return 4
        if count >= 2:
            return 3
        if count == 1:
            return 1
        return 0

    # ---- main entry -------------------------------------------------------
    def enrich(self, github_url: Optional[str]) -> GitHubResult:
        """Enrich one profile. Never raises; every failure yields a result."""
        username = extract_username(github_url)
        if not username:
            return GitHubResult._make(github_url, None, 0,
                                      "no valid GitHub profile URL found", "no_profile")

        if username in self._cache:
            return GitHubResult(self._cache[username])

        time.sleep(REQUEST_PAUSE)  # stay under unauthenticated limits

        user = self._get(f"/users/{username}")
        if isinstance(user, dict) and "error" in user:
            err = user["error"]
            summary = {
                "not_found": "GitHub profile not found (deleted or renamed account)",
                "rate_limited": "GitHub enrichment failed due to API rate limit",
                "timeout": "GitHub enrichment failed due to API timeout",
            }.get(err, f"GitHub enrichment failed due to API failure ({err})")
            status = "rate_limited" if err == "rate_limited" else (
                "not_found" if err == "not_found" else "error")
            result = GitHubResult._make(github_url, username, 0, summary, status)
            self._cache[username] = dict(result)
            return result

        events = self._get(f"/users/{username}/events/public?per_page={MAX_EVENTS}")
        repos = self._get(f"/users/{username}/repos?per_page=100&sort=pushed")

        act = self._activity_points(events if isinstance(events, list) else None)
        rep = self._repo_points(repos if isinstance(repos, list) else None)
        total = min(act + rep, 10)

        summary = (f"{'Strong' if act >= 5 else 'Moderate' if act >= 3 else 'Limited' if act >= 1 else 'No'} "
                   f"recent activity with "
                   f"{'multiple maintained Python/AI' if rep >= 3 else 'some' if rep >= 1 else 'no relevant maintained'} "
                   f"repositories.")
        result = GitHubResult._make(
            github_url, username, total, summary, "success",
            activity=act, repos=rep,
        )
        logger.info("GitHub %s: %d/10", username, total)
        self._cache[username] = dict(result)
        return result
