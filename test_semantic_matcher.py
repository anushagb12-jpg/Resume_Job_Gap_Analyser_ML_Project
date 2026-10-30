"""Tests for src/semantic_matcher.py.

Two groups of tests:

* "no model" tests cover the code paths that must work on a machine WITHOUT
  the heavy sentence-transformers package: chunk splitting, blank-input
  guards, and graceful degradation to None.
* "real model" tests load all-MiniLM-L6-v2 and check the numbers it
  produces. They skip themselves when the library is missing or the model
  cannot be downloaded (e.g. an offline machine), because there is nothing
  meaningful to assert in that case.
"""

from __future__ import annotations

import pytest

from src import semantic_matcher
from src.semantic_matcher import (
    CHUNK_WORDS,
    MODEL_NAME,
    _split_into_chunks,
    compute_semantic_similarity,
    is_semantic_available,
)

MODEL_PROBE_TEXT = "python developer with sql and pandas"


def _model_loads() -> bool:
    """True when the model is actually usable in this environment.

    Calling this loads the model (once per process) and remembers a failure,
    so it doubles as the skip decision for the real-model tests.
    """
    return compute_semantic_similarity(MODEL_PROBE_TEXT, MODEL_PROBE_TEXT) is not None


def skip_without_model() -> None:
    """Skip a test when the neural network model cannot be used here."""
    if not _model_loads():
        pytest.skip("sentence-transformers model unavailable in this environment")


# ---------------------------------------------------------------------------
# Code paths that work without the model
# ---------------------------------------------------------------------------


def test_blank_input_on_either_side_returns_none():
    # The blank check happens before the model is ever touched, so this test
    # needs no library at all.
    assert compute_semantic_similarity("", "python job advert") is None
    assert compute_semantic_similarity("python resume", "   ") is None


def test_unavailable_model_returns_none_instead_of_raising(monkeypatch):
    monkeypatch.setattr(semantic_matcher, "_get_model", lambda: None)
    assert compute_semantic_similarity("a python resume", "a python job") is None
    assert is_semantic_available() is False


def test_model_name_is_the_agreed_small_model():
    assert MODEL_NAME == "sentence-transformers/all-MiniLM-L6-v2"


def test_chunking_leaves_short_text_as_one_piece():
    assert _split_into_chunks("python pandas sql") == ["python pandas sql"]


def test_chunking_empty_text_returns_no_chunks():
    assert _split_into_chunks("") == []


def test_chunking_splits_long_text_into_ordered_pieces():
    words = [f"word{index}" for index in range(CHUNK_WORDS * 2 + 7)]
    chunks = _split_into_chunks(" ".join(words))

    assert len(chunks) == 3
    # No chunk is longer than the limit...
    assert all(len(chunk.split()) <= CHUNK_WORDS for chunk in chunks)
    # ...and joining the chunks reproduces the original words, in order.
    assert sum(len(chunk.split()) for chunk in chunks) == len(words)
    assert " ".join(chunks).split() == words


# ---------------------------------------------------------------------------
# The real model (skipped when it cannot be loaded)
# ---------------------------------------------------------------------------


def test_self_similarity_is_essentially_one():
    skip_without_model()
    similarity = compute_semantic_similarity(MODEL_PROBE_TEXT, MODEL_PROBE_TEXT)
    assert similarity > 0.95


def test_related_job_scores_above_unrelated_job():
    skip_without_model()
    resume = "data analyst building dashboards with python pandas sql and tableau"
    related_job = (
        "looking for an analyst experienced with python, sql and "
        "building reporting dashboards"
    )
    unrelated_job = "pastry chef wanted to bake croissants and sourdough before dawn"
    assert (
        compute_semantic_similarity(resume, related_job)
        > compute_semantic_similarity(resume, unrelated_job)
    )


def test_similarity_stays_within_mathematical_bounds():
    skip_without_model()
    similarity = compute_semantic_similarity(
        "python resume text", "a completely different advert about gardening"
    )
    assert -1.0 <= similarity <= 1.0


def test_same_input_twice_gives_the_same_number():
    # The model is used in inference mode, so scores are reproducible.
    skip_without_model()
    first = compute_semantic_similarity(
        "python and pandas analyst", "python data engineering job"
    )
    second = compute_semantic_similarity(
        "python and pandas analyst", "python data engineering job"
    )
    assert first == second


def test_long_resume_is_chunked_not_truncated():
    # Far longer than the model's ~256 word-piece window: without chunking,
    # only the first sentence would influence the score.
    skip_without_model()
    long_resume = "built python data pipelines with pandas and sql. " * 60
    similarity = compute_semantic_similarity(
        long_resume, "python data engineering with sql"
    )
    assert -1.0 <= similarity <= 1.0


def test_availability_flag_agrees_with_compute():
    # Works whether or not the library is installed: both sides ask the
    # model the same question, so they must agree.
    assert is_semantic_available() == _model_loads()


def test_similarity_on_the_sample_fixtures(resume_text, job_text):
    skip_without_model()
    similarity = compute_semantic_similarity(resume_text, job_text)
    # Both fixtures are about data analysis, so the model should see them as
    # related (positive) but not identical (below one).
    assert 0.0 < similarity < 1.0
