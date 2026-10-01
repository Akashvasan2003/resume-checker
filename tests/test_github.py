"""Tests for GitHub enrichment (Part 4) - fully mocked, no live API calls."""
import requests

from src.github_client import GitHubEnricher, extract_username
from src.models import Candidate
from src.scoring import WEIGHTS, score_candidate

USER = {
    "login": "janedoe", "public_repos": 6, "created_at": "2020-01-01T00:00:00Z",
}
EVENTS_ACTIVE = [
    {"type": "PushEvent"} for _ in range(20)
]
EVENTS_SPARSE = [{"type": "PushEvent"}, {"type": "WatchEvent"}]
REPOS_RELEVANT = [
    {"name": "rag-chatbot", "fork": False, "description": "RAG chatbot with langchain"},
    {"name": "llm-agents", "fork": False, "description": "multi-agent LLM pipeline"},
    {"name": "pytorch-finetune", "fork": False, "description": "fine-tuning transformers"},
    {"name": "vector-db-demo", "fork": False, "description": "embeddings vector search"},
    {"name": "python-utils", "fork": False, "description": "python helpers"},
    {"name": "fork-of-x", "fork": True, "description": "someone else's work"},
]
REPOS_NONE = [{"name": "setup", "fork": False, "description": "dotfiles"}]


class FakeSession:
    """Scriptable stand-in for requests.Session."""

    def __init__(self, responses):
        self.responses = responses  # path fragment -> response or exception

    def get(self, url, headers=None, timeout=None):
        for frag, val in sorted(self.responses.items(), key=lambda kv: -len(kv[0])):
            if frag in url:
                if isinstance(val, Exception):
                    raise val
                return val
        raise AssertionError(f"unexpected URL {url}")


class FakeResponse:
    def __init__(self, status, payload):
        self.status_code = status
        self._payload = payload

    def json(self):
        return self._payload


def _enricher(responses, cache=None):
    return GitHubEnricher(session=FakeSession(responses), cache=cache or {})


def test_extract_username():
    assert extract_username("https://github.com/janedoe") == "janedoe"
    assert extract_username("github.com/janedoe/") == "janedoe"
    assert extract_username("https://github.com/janedoe/rag-chatbot") == "janedoe"
    assert extract_username("https://gitlab.com/janedoe") is None
    assert extract_username(None) is None
    assert extract_username("https://github.com/") is None


def test_success_full_score():
    e = _enricher({
        "/users/janedoe": FakeResponse(200, USER),
        "/users/janedoe/events/public": FakeResponse(200, EVENTS_ACTIVE),
        "/users/janedoe/repos": FakeResponse(200, REPOS_RELEVANT),
    })
    res = e.enrich("https://github.com/janedoe")
    assert res["github_status"] == "success"
    assert res["github_score"] == 10
    assert res["recent_activity_points"] == 5
    assert res["repo_points"] == 5


def test_sparse_activity_lower_score():
    e = _enricher({
        "/users/janedoe": FakeResponse(200, USER),
        "/users/janedoe/events/public": FakeResponse(200, EVENTS_SPARSE),
        "/users/janedoe/repos": FakeResponse(200, REPOS_NONE),
    })
    res = e.enrich("https://github.com/janedoe")
    assert res["github_score"] <= 2


def test_deleted_account():
    e = _enricher({"/users/janedoe": FakeResponse(404, {"message": "Not Found"})})
    res = e.enrich("https://github.com/janedoe")
    assert res["github_status"] == "not_found"
    assert res["github_score"] == 0


def test_rate_limited():
    e = _enricher({"/users/janedoe": FakeResponse(403, {"message": "rate limit"})})
    res = e.enrich("https://github.com/janedoe")
    assert res["github_status"] == "rate_limited"
    assert "rate limit" in res["github_summary"]


def test_timeout():
    e = _enricher({"/users/janedoe": requests.Timeout("timed out")})
    res = e.enrich("https://github.com/janedoe")
    assert res["github_status"] == "error"
    assert res["github_score"] == 0


def test_no_or_invalid_url():
    e = _enricher({})
    assert e.enrich(None)["github_status"] == "no_profile"
    assert e.enrich("https://gitlab.com/janedoe")["github_status"] == "no_profile"


def test_caching_same_profile_not_queried_twice():
    calls = {"n": 0}

    class Counting(FakeSession):
        def get(self, url, headers=None, timeout=None):
            calls["n"] += 1
            return super().get(url, headers, timeout)

    session = Counting({
        "/users/janedoe": FakeResponse(200, USER),
        "/users/janedoe/events/public": FakeResponse(200, EVENTS_ACTIVE),
        "/users/janedoe/repos": FakeResponse(200, REPOS_RELEVANT),
    })
    e = GitHubEnricher(session=session)
    first = e.enrich("https://github.com/janedoe")
    second = e.enrich("https://github.com/janedoe")
    assert first == second
    assert calls["n"] == 3  # user + events + repos, exactly once


def test_github_points_fed_into_score():
    cand = Candidate(
        name="Jane", github_url="https://github.com/janedoe",
        skills=["python", "langchain"],
        projects=["Built a RAG system with embeddings and vector search"],
        raw_text="python rag langchain embeddings vector search",
    )
    from src.eligibility import check_eligibility
    el = check_eligibility(cand)
    base = score_candidate(cand, el, github_points=0)
    boosted = score_candidate(cand, el, github_points=8)
    assert boosted.github == 8
    assert boosted.total_score == base.total_score + 8
    assert boosted.total_score <= 100

