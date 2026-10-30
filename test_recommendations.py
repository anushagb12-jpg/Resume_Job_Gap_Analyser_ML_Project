"""Tests for src/recommendations.py."""

from __future__ import annotations

import pytest

from src.matcher import analyze
from src.recommendations import (
    DEFAULT_PLAN,
    HIGH_PRIORITY_MIN,
    LEARNING_PLAN,
    MEDIUM_PRIORITY_MIN,
    build_recommendation,
    generate_recommendations,
    highlight_strengths,
    priority_for,
    summarize,
    to_dataframe,
    verdict_for,
)
from src.skill_extractor import SKILL_LEXICON

EXPECTED_COLUMNS = [
    "Skill",
    "Category",
    "Priority",
    "Importance",
    "Asked for directly",
    "What to do",
    "Where to learn",
]


@pytest.fixture
def result_with_gaps():
    """An analysis where the resume covers some skills and misses others."""
    resume = "python pandas sql git communication"
    job = (
        "Requirements: Python, pandas, SQL, scikit-learn, Docker, AWS, Kubernetes\n"
        "Nice to have: Airflow, dbt and Snowflake"
    )
    return analyze(resume, job)


@pytest.fixture
def result_without_gaps():
    text = "Requirements: python and sql"
    return analyze("python sql", text)


# ---------------------------------------------------------------------------
# Priority and verdict labels
# ---------------------------------------------------------------------------


def test_priority_bands():
    assert priority_for(1.0) == "High"
    assert priority_for(HIGH_PRIORITY_MIN) == "High"
    assert priority_for(MEDIUM_PRIORITY_MIN) == "Medium"
    assert priority_for(MEDIUM_PRIORITY_MIN - 0.01) == "Low"
    assert priority_for(0.0) == "Low"


def test_high_threshold_is_above_medium_threshold():
    assert HIGH_PRIORITY_MIN > MEDIUM_PRIORITY_MIN


def test_verdict_bands():
    assert verdict_for(100) == "Strong match"
    assert verdict_for(80) == "Strong match"
    assert verdict_for(79) == "Good match"
    assert verdict_for(60) == "Good match"
    assert verdict_for(45) == "Partial match"
    assert verdict_for(10) == "Early days"
    assert verdict_for(0) == "Early days"


# ---------------------------------------------------------------------------
# Building a single recommendation
# ---------------------------------------------------------------------------


def test_build_recommendation_fills_in_the_skill_name():
    item = build_recommendation("kubernetes", 0.9, listed_in_ad=True)
    assert item.skill == "kubernetes"
    assert item.priority == "High"
    assert item.listed_in_ad is True
    assert "kubernetes" in item.action
    assert "kubernetes" in item.where_to_learn


def test_build_recommendation_uses_the_category_plan():
    python_item = build_recommendation("python", 0.9, True)
    docker_item = build_recommendation("docker", 0.9, True)
    assert python_item.action != docker_item.action
    assert python_item.category == "Programming Languages"
    assert docker_item.category == "Cloud & DevOps"


def test_unknown_category_falls_back_to_the_default_plan():
    item = build_recommendation("quantum-flux", 0.5, False)
    assert item.category == "Other"
    assert item.action == DEFAULT_PLAN[0].format(skill="quantum-flux")
    assert item.listed_in_ad is False


def test_every_category_in_the_lexicon_has_a_learning_plan():
    missing = set(SKILL_LEXICON) - set(LEARNING_PLAN)
    assert not missing, f"categories with no learning plan: {missing}"


def test_every_plan_template_accepts_a_skill_placeholder():
    # A typo like "{skills}" would raise KeyError only when a user hits it, so
    # check all of them here.
    for category, (action, resource) in list(LEARNING_PLAN.items()) + [("Other", DEFAULT_PLAN)]:
        try:
            action.format(skill="example")
            resource.format(skill="example")
        except (KeyError, IndexError) as error:
            pytest.fail(f"bad template for {category!r}: {error}")


# ---------------------------------------------------------------------------
# Generating the ranked list
# ---------------------------------------------------------------------------


def test_recommendations_respect_the_limit(result_with_gaps):
    assert len(generate_recommendations(result_with_gaps, limit=3)) == 3


def test_limit_larger_than_the_gap_count_returns_everything(result_with_gaps):
    gaps = len(result_with_gaps.missing_skills)
    assert len(generate_recommendations(result_with_gaps, limit=gaps + 10)) == gaps


def test_recommendations_are_ordered_by_importance(result_with_gaps):
    items = generate_recommendations(result_with_gaps, limit=20)
    importances = [item.importance for item in items]
    assert importances == sorted(importances, reverse=True)


def test_required_skills_outrank_nice_to_have_ones(result_with_gaps):
    # The exact priority label depends on the hand-tuned thresholds, so this
    # test checks the ordering instead - that is the invariant that matters.
    items = generate_recommendations(result_with_gaps, limit=20)
    by_name = {item.skill: item for item in items}
    rank = {"High": 2, "Medium": 1, "Low": 0}

    assert by_name["kubernetes"].importance > by_name["airflow"].importance
    assert rank[by_name["kubernetes"].priority] > rank[by_name["airflow"].priority]


def test_required_gaps_are_flagged_as_asked_for_directly(result_with_gaps):
    items = generate_recommendations(result_with_gaps, limit=20)
    by_name = {item.skill: item for item in items}
    assert by_name["docker"].listed_in_ad is True
    assert by_name["docker"].priority == "High"


def test_no_recommendations_when_there_are_no_gaps(result_without_gaps):
    assert result_without_gaps.missing_skills == []
    assert generate_recommendations(result_without_gaps) == []


def test_limit_of_zero_returns_nothing(result_with_gaps):
    assert generate_recommendations(result_with_gaps, limit=0) == []


# ---------------------------------------------------------------------------
# Table output
# ---------------------------------------------------------------------------


def test_to_dataframe_uses_friendly_column_names(result_with_gaps):
    frame = to_dataframe(generate_recommendations(result_with_gaps, limit=3))
    assert list(frame.columns) == EXPECTED_COLUMNS
    assert len(frame) == 3


def test_to_dataframe_on_an_empty_list_keeps_its_columns():
    frame = to_dataframe([])
    assert frame.empty
    assert list(frame.columns) == EXPECTED_COLUMNS


# ---------------------------------------------------------------------------
# Strengths and summary
# ---------------------------------------------------------------------------


def test_highlight_strengths_respects_the_limit(result_with_gaps):
    assert len(highlight_strengths(result_with_gaps, limit=2)) == 2


def test_highlight_strengths_only_returns_matched_skills(result_with_gaps):
    strengths = highlight_strengths(result_with_gaps, limit=50)
    assert set(strengths) <= set(result_with_gaps.matched_skills)
    assert "sql" in strengths


def test_highlight_strengths_on_no_matches():
    result = analyze("cobol programming", "Requirements: python, sql")
    assert highlight_strengths(result) == []


def test_summarize_mentions_the_score_and_the_gap_count(result_with_gaps):
    items = generate_recommendations(result_with_gaps, limit=5)
    text = summarize(result_with_gaps, items)
    assert f"{result_with_gaps.match_score}/100" in text
    assert str(len(result_with_gaps.missing_skills)) in text
    assert items[0].skill in text


def test_summarize_says_so_when_there_is_nothing_to_fix(result_without_gaps):
    text = summarize(result_without_gaps, [])
    assert "No skill gaps found" in text


def test_summarize_mentions_bonus_skills(result_with_gaps):
    text = summarize(result_with_gaps, generate_recommendations(result_with_gaps))
    assert "did not ask for" in text
