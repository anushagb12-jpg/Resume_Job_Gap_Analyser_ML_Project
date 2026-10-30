"""Tests for src/ai_assistant.py.

Ollama is NOT needed to run these tests. Every HTTP call is replaced with a
fake, so the tests check exactly what we send to Ollama and how each kind of
failure is turned into a friendly error - quickly, and on any machine.

The fakes record the arguments of every request they receive, which is what
lets us assert things like "the model name was llama3.2:3b" and "the context
rode along with the system message".
"""

from __future__ import annotations

import pandas as pd
import pytest
import requests

from src import ai_assistant
from src.ai_assistant import (
    MODEL_NAME,
    AssistantTimeoutError,
    AssistantUnavailableError,
    EmptyMessageError,
    EmptyReplyError,
    ModelNotInstalledError,
    ask_assistant,
    build_context,
    check_ollama,
)
from src.matcher import MatchResult

# ---------------------------------------------------------------------------
# Fakes for the HTTP layer
# ---------------------------------------------------------------------------


class FakeResponse:
    """Just enough of requests.Response for the code paths we exercise."""

    def __init__(self, status_code: int = 200, payload: dict | None = None):
        self.status_code = status_code
        self._payload = {} if payload is None else payload

    def json(self) -> dict:
        return self._payload


def patch_get(monkeypatch, response=None, error=None) -> list[dict]:
    """Replace requests.get inside ai_assistant with a controllable fake."""
    calls: list[dict] = []

    def fake_get(url, timeout=None):
        calls.append({"url": url, "timeout": timeout})
        if error is not None:
            raise error
        return response

    monkeypatch.setattr(ai_assistant.requests, "get", fake_get)
    return calls


def patch_post(monkeypatch, response=None, error=None) -> list[dict]:
    """Replace requests.post inside ai_assistant with a controllable fake."""
    calls: list[dict] = []

    def fake_post(url, json=None, timeout=None):
        calls.append({"url": url, "json": json, "timeout": timeout})
        if error is not None:
            raise error
        return response

    monkeypatch.setattr(ai_assistant.requests, "post", fake_post)
    return calls


def ok_reply(content: str = "Learn Docker first.") -> FakeResponse:
    """A well-formed Ollama /api/chat response."""
    return FakeResponse(
        payload={"model": MODEL_NAME, "message": {"role": "assistant", "content": content}}
    )


def make_result(**overrides) -> MatchResult:
    """A small analysis with two gaps, used to test build_context."""
    defaults = dict(
        resume_skills={"python", "tableau"},
        job_skills={"python", "docker", "aws"},
        matched_skills=["python"],
        missing_skills=["aws", "docker"],
        extra_skills=["tableau"],
        coverage=1 / 3,
        text_similarity=0.2,
        match_score=40,
        semantic_similarity=None,
        skill_importance={"docker": 1.0, "aws": 0.5, "python": 0.9},
        gap_table=pd.DataFrame(),
    )
    defaults.update(overrides)
    return MatchResult(**defaults)


# ---------------------------------------------------------------------------
# check_ollama: is the server up, is the model there?
# ---------------------------------------------------------------------------


def test_check_ollama_reports_server_down_on_connection_error(monkeypatch):
    patch_get(monkeypatch, error=requests.exceptions.ConnectionError("refused"))
    status = check_ollama()
    assert status.server_running is False
    assert status.model_installed is False


def test_check_ollama_reports_server_down_on_timeout(monkeypatch):
    patch_get(monkeypatch, error=requests.exceptions.Timeout("too slow"))
    status = check_ollama()
    assert status.server_running is False


def test_check_ollama_finds_our_model_when_installed(monkeypatch):
    response = FakeResponse(
        payload={"models": [{"name": MODEL_NAME}, {"name": "mistral:latest"}]}
    )
    calls = patch_get(monkeypatch, response=response)
    status = check_ollama()
    assert status.server_running is True
    assert status.model_installed is True
    # It asked the right endpoint, and only that.
    assert calls[0]["url"].endswith("/api/tags")


def test_check_ollama_reports_a_missing_model(monkeypatch):
    response = FakeResponse(payload={"models": [{"name": "mistral:latest"}]})
    patch_get(monkeypatch, response=response)
    status = check_ollama()
    assert status.server_running is True
    assert status.model_installed is False


def test_check_ollama_treats_an_http_error_as_not_running(monkeypatch):
    patch_get(monkeypatch, response=FakeResponse(status_code=500))
    status = check_ollama()
    assert status.server_running is False


# ---------------------------------------------------------------------------
# build_context: what the model is allowed to know
# ---------------------------------------------------------------------------


def test_build_context_lists_matched_missing_and_extra_skills():
    context = build_context(
        "resume with python", "job wants docker and aws", make_result()
    )
    assert "python" in context  # matched
    assert "docker" in context and "aws" in context  # missing
    assert "tableau" in context  # extra
    assert "40/100" in context  # the match score


def test_build_context_orders_missing_skills_by_importance():
    context = build_context("r", "j", make_result())
    # docker has importance 1.0, aws 0.5, so docker is listed first.
    assert context.index("docker") < context.index("aws")


def test_build_context_includes_both_texts():
    context = build_context("I know python deeply", "Wanted: a docker guru", make_result())
    assert "I know python deeply" in context
    assert "Wanted: a docker guru" in context


def test_build_context_truncates_very_long_texts():
    long_resume = "word " * 5000  # ~25,000 characters
    context = build_context(long_resume, "short job", make_result())
    assert "truncated" in context
    assert len(context) < len(long_resume)


def test_build_context_says_none_when_there_are_no_gaps():
    result = make_result(
        missing_skills=[], matched_skills=["aws", "docker", "python"]
    )
    context = build_context("r", "j", result)
    assert "- none" in context


# ---------------------------------------------------------------------------
# ask_assistant: the request we send and the reply we accept
# ---------------------------------------------------------------------------


def test_ask_assistant_returns_the_reply_text(monkeypatch):
    patch_post(monkeypatch, response=ok_reply("Learn Docker first."))
    reply = ask_assistant("What should I learn?", "CONTEXT")
    assert reply == "Learn Docker first."


def test_ask_assistant_sends_system_history_and_question_in_order(monkeypatch):
    calls = patch_post(monkeypatch, response=ok_reply())
    history = [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
    ]
    ask_assistant("second question", "CONTEXT", history=history)
    messages = calls[0]["json"]["messages"]
    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    # The context rides along with the system prompt, before anything else.
    assert messages[0]["content"].endswith("CONTEXT")
    assert messages[-1]["content"] == "second question"


def test_the_system_prompt_forbids_inventing_credentials():
    prompt = ai_assistant.SYSTEM_PROMPT.lower()
    assert "never invent" in prompt
    for word in ("qualification", "experience", "certification"):
        assert word in prompt


def test_ask_assistant_requests_the_right_model_without_streaming(monkeypatch):
    calls = patch_post(monkeypatch, response=ok_reply())
    ask_assistant("q", "c")
    payload = calls[0]["json"]
    assert payload["model"] == MODEL_NAME
    assert payload["stream"] is False


def test_a_low_temperature_is_requested(monkeypatch):
    # Low temperature = fewer invented facts.
    calls = patch_post(monkeypatch, response=ok_reply())
    ask_assistant("q", "c")
    assert calls[0]["json"]["options"]["temperature"] <= 0.3


def test_ask_assistant_uses_the_local_url_with_a_timeout(monkeypatch):
    calls = patch_post(monkeypatch, response=ok_reply())
    ask_assistant("q", "c")
    assert calls[0]["url"].startswith("http://localhost:11434")
    assert calls[0]["timeout"] == (
        ai_assistant.CONNECT_TIMEOUT,
        ai_assistant.READ_TIMEOUT,
    )


def test_the_base_url_can_be_overridden(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11500/")
    calls = patch_post(monkeypatch, response=ok_reply())
    ask_assistant("q", "c")
    assert calls[0]["url"].startswith("http://localhost:11500")


# ---------------------------------------------------------------------------
# ask_assistant: every way it can fail
# ---------------------------------------------------------------------------


def test_empty_questions_are_rejected_without_calling_ollama(monkeypatch):
    calls = patch_post(monkeypatch, response=ok_reply())
    with pytest.raises(EmptyMessageError):
        ask_assistant("   ", "CONTEXT")
    assert calls == []


def test_connection_refusal_becomes_a_friendly_error(monkeypatch):
    patch_post(monkeypatch, error=requests.exceptions.ConnectionError("refused"))
    with pytest.raises(AssistantUnavailableError) as info:
        ask_assistant("q", "c")
    # The message tells the user what to do, not what errno said.
    assert "ollama serve" in str(info.value)


def test_a_timeout_becomes_a_friendly_error(monkeypatch):
    patch_post(monkeypatch, error=requests.exceptions.Timeout("timed out"))
    with pytest.raises(AssistantTimeoutError):
        ask_assistant("q", "c")


def test_a_404_means_the_model_is_not_installed(monkeypatch):
    response = FakeResponse(
        status_code=404, payload={"error": "model 'llama3.2:3b' not found"}
    )
    patch_post(monkeypatch, response=response)
    with pytest.raises(ModelNotInstalledError) as info:
        ask_assistant("q", "c")
    assert "ollama pull" in str(info.value)


def test_an_empty_reply_is_rejected(monkeypatch):
    patch_post(monkeypatch, response=ok_reply("   "))
    with pytest.raises(EmptyReplyError):
        ask_assistant("q", "c")


def test_garbage_from_ollama_becomes_a_friendly_error(monkeypatch):
    class Garbage(FakeResponse):
        def json(self):
            raise ValueError("not JSON")

    patch_post(monkeypatch, response=Garbage())
    with pytest.raises(ai_assistant.AssistantError):
        ask_assistant("q", "c")


def test_a_bad_role_in_history_is_rejected(monkeypatch):
    patch_post(monkeypatch, response=ok_reply())
    with pytest.raises(ValueError):
        ask_assistant("q", "c", history=[{"role": "system", "content": "nope"}])
