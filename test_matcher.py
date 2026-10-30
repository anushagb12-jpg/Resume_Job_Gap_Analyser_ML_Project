"""Tests for src/matcher.py."""

from __future__ import annotations

import pytest

from src.matcher import (
    COVERAGE_WEIGHT,
    SIMILARITY_WEIGHT,
    MatchResult,
    _find_section_markers,
    _normalize,
    analyze,
    build_gap_table,
    compute_text_similarity,
    score_sections,
    score_skill_importance,
)
from src.skill_extractor import find_skill_mentions, prepare_text

GAP_COLUMNS = ["Skill", "Category", "Status", "Importance", "In job ad"]


# ---------------------------------------------------------------------------
# Helpers and constants
# ---------------------------------------------------------------------------


def test_the_two_score_weights_add_up_to_one():
    assert COVERAGE_WEIGHT + SIMILARITY_WEIGHT == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# _normalize
# ---------------------------------------------------------------------------


def test_normalize_scales_the_largest_value_to_one():
    assert _normalize({"a": 2.0, "b": 4.0}) == {"a": 0.5, "b": 1.0}


def test_normalize_on_empty_input():
    assert _normalize({}) == {}


def test_normalize_when_everything_is_zero():
    assert _normalize({"a": 0.0, "b": 0.0}) == {"a": 0.0, "b": 0.0}


# ---------------------------------------------------------------------------
# Text similarity
# ---------------------------------------------------------------------------


def test_similarity_of_identical_text_is_one():
    text = "We need python, sql and docker experience."
    assert compute_text_similarity(text, text) == pytest.approx(1.0, abs=1e-6)


def test_similarity_of_unrelated_text_is_low():
    similarity = compute_text_similarity(
        "python pandas scikit-learn machine learning",
        "woodworking lathe chisel joinery cabinet",
    )
    assert similarity < 0.2


def test_similarity_stays_within_bounds(resume_text, job_text):
    similarity = compute_text_similarity(resume_text, job_text)
    assert 0.0 <= similarity <= 1.0


def test_similarity_of_empty_text_is_zero_not_a_crash():
    assert compute_text_similarity("", "") == 0.0


# ---------------------------------------------------------------------------
# Job ad sections
# ---------------------------------------------------------------------------


def test_section_markers_find_required_and_optional_headings():
    markers = _find_section_markers("requirements: python. nice to have: spark.")
    assert (len(markers), [weight for _, weight in markers]) == (2, [1.0, 0.5])


def test_preferred_qualifications_stays_optional():
    # "Preferred Qualifications" must not be read as a hard requirement, even
    # though it contains the word "qualifications".
    markers = _find_section_markers("preferred qualifications: phd")
    assert len(markers) == 1
    assert markers[0][1] == 0.5


def test_no_markers_in_plain_text():
    assert _find_section_markers("we are hiring a data scientist") == []


def test_score_sections_separates_required_from_optional(job_text):
    scores = score_sections(job_text, find_skill_mentions(job_text))
    assert scores["python"] == 1.0  # under "Requirements"
    assert scores["airflow"] == 0.5  # under "Nice to have"
    assert scores["python"] > scores["airflow"]


def test_score_sections_defaults_before_the_first_heading():
    text = "Data scientist. We use Spark. Requirements: Python."
    scores = score_sections(text, find_skill_mentions(text))
    assert scores["spark"] == 0.8
    assert scores["python"] == 1.0


def test_score_sections_on_empty_mentions():
    assert score_sections("anything", {}) == {}


# ---------------------------------------------------------------------------
# Skill importance
# ---------------------------------------------------------------------------


def test_importance_ranks_required_above_nice_to_have(job_text, resume_text):
    importance = score_skill_importance(
        job_text, resume_text, find_skill_mentions(job_text)
    )
    assert importance["tensorflow"] < importance["python"]


def test_importance_is_normalised_so_the_top_skill_is_one(job_text, resume_text):
    importance = score_skill_importance(
        job_text, resume_text, find_skill_mentions(job_text)
    )
    assert max(importance.values()) == pytest.approx(1.0)
    assert all(0.0 <= value <= 1.0 for value in importance.values())


def test_importance_of_a_repeated_skill_beats_a_single_mention():
    job = "Requirements: Python. We really need Python. Python expert. Nice to have: Spark."
    importance = score_skill_importance(job, "I know spark", find_skill_mentions(job))
    assert importance["python"] > importance["spark"]


def test_importance_on_empty_mentions():
    assert score_skill_importance("job", "resume", {}) == {}


# ---------------------------------------------------------------------------
# Gap table
# ---------------------------------------------------------------------------


def test_gap_table_columns():
    table = build_gap_table({"python", "spark"}, {"python"}, {"python": 1.0}, {"python", "spark"})
    assert list(table.columns) == GAP_COLUMNS


def test_gap_table_puts_gaps_first_and_orders_them_by_importance():
    table = build_gap_table(
        {"python", "spark", "docker"},
        {"python"},
        {"python": 1.0, "spark": 0.2, "docker": 0.9},
        {"python", "spark", "docker"},
    )
    assert list(table["Status"]) == ["Gap", "Gap", "Match"]
    assert list(table["Skill"]) == ["docker", "spark", "python"]


def test_gap_table_labels_implied_skills():
    table = build_gap_table({"python"}, set(), {"python": 1.0}, set())
    assert table.loc[0, "In job ad"] == "implied"


def test_gap_table_on_no_skills_keeps_its_columns():
    table = build_gap_table(set(), set(), {}, set())
    assert table.empty
    assert list(table.columns) == GAP_COLUMNS


# ---------------------------------------------------------------------------
# analyze() - the whole pipeline
# ---------------------------------------------------------------------------


def test_analyze_returns_a_match_result(resume_text, job_text):
    assert isinstance(analyze(resume_text, job_text), MatchResult)


def test_identical_documents_are_a_perfect_match():
    text = "Requirements: Python, SQL and Docker. Great communication skills."
    result = analyze(text, text)
    assert result.coverage == 1.0
    assert result.match_score == 100
    assert result.missing_skills == []
    assert result.extra_skills == []


def test_coverage_is_matched_divided_by_job_skills():
    result = analyze("python sql", "Requirements: python, sql, docker, kubernetes")
    assert result.coverage == pytest.approx(0.5)


def test_missing_skills_are_the_ones_the_resume_lacks():
    result = analyze("python sql", "Requirements: python, sql, docker")
    assert result.missing_skills == ["docker"]


def test_extra_skills_are_the_ones_the_job_did_not_ask_for():
    result = analyze("python sql git linux", "Requirements: python, sql")
    assert set(result.extra_skills) == {"git", "linux"}


def test_match_score_is_always_between_zero_and_a_hundred(resume_text, job_text):
    assert 0 <= analyze(resume_text, job_text).match_score <= 100


def test_a_better_resume_scores_higher():
    job = "Requirements: python, pandas, sql, scikit-learn, docker, aws"
    weak = analyze("python", job)
    strong = analyze("python pandas sql scikit-learn docker aws", job)
    assert strong.match_score > weak.match_score


def test_implied_skills_are_credited_to_the_resume():
    # The resume never says "python", but Django requires it.
    result = analyze("I build apps with Django", "Requirements: python")
    assert "python" in result.matched_skills


def test_analyze_with_no_recognisable_job_skills():
    result = analyze("python sql", "We are a friendly team that values curiosity")
    assert result.job_skills == set()
    assert result.coverage == 0.0
    assert 0 <= result.match_score <= 100


def test_analyze_with_empty_inputs_does_not_crash():
    result = analyze("", "")
    assert result.match_score == 0
    assert result.gap_table.empty


def test_analyze_credits_skills_written_with_different_spellings():
    result = analyze("sklearn and k8s", "Requirements: scikit-learn, kubernetes")
    # Aliases resolve to the same canonical name on both sides, so nothing is
    # missing. Implied skills (docker, machine learning) are matched too,
    # because both documents imply them.
    assert result.missing_skills == []
    assert {"scikit-learn", "kubernetes"} <= set(result.matched_skills)


def test_implied_skills_are_compared_consistently_on_both_sides():
    # Kubernetes implies Docker. If only the job ad implies it, that is a real
    # gap; if both imply it, it is a match.
    one_sided = analyze("python", "Requirements: kubernetes")
    assert "docker" in one_sided.missing_skills
    both_sides = analyze("kubernetes", "Requirements: kubernetes")
    assert "docker" in both_sides.matched_skills


def test_analyze_is_not_fooled_by_the_excel_verb():
    result = analyze("I excel at teamwork", "Requirements: Excel")
    assert "excel" not in result.matched_skills


def test_gap_table_inside_the_result_matches_the_missing_skills(resume_text, job_text):
    result = analyze(resume_text, job_text)
    gaps = result.gap_table[result.gap_table["Status"] == "Gap"]
    assert sorted(gaps["Skill"]) == result.missing_skills


def test_every_job_skill_has_an_importance(resume_text, job_text):
    result = analyze(resume_text, job_text)
    for skill in result.job_skills:
        assert skill in result.skill_importance


def test_analysis_does_not_depend_on_the_order_of_the_input_lines():
    job_a = "Requirements: python, sql, docker\nNice to have: spark"
    job_b = "Nice to have: spark\nRequirements: python, sql, docker"
    resume = "python sql"
    assert analyze(resume, job_a).missing_skills == analyze(resume, job_b).missing_skills


def test_prepare_text_is_used_consistently_for_sections(job_text):
    # score_sections() searches the folded text, so this checks the folding we
    # rely on is stable for a real advert.
    folded = prepare_text(job_text, lowercase=True)
    assert "requirements" in folded
    assert "nice to have" in folded


# ---------------------------------------------------------------------------
# Semantic similarity (kept separate from the match score)
# ---------------------------------------------------------------------------


def test_analyze_without_semantic_leaves_the_field_empty(resume_text, job_text):
    result = analyze(resume_text, job_text)
    assert result.semantic_similarity is None


def test_analyze_can_attach_a_semantic_score(resume_text, job_text, monkeypatch):
    monkeypatch.setattr("src.matcher.compute_semantic_similarity", lambda r, j: 0.42)
    result = analyze(resume_text, job_text, use_semantic=True)
    assert result.semantic_similarity == 0.42


def test_semantic_score_never_changes_the_match_score(resume_text, job_text, monkeypatch):
    # The separation guarantee: the 0-100 score is word-based only, with or
    # without the AI opinion attached.
    before = analyze(resume_text, job_text)
    monkeypatch.setattr("src.matcher.compute_semantic_similarity", lambda r, j: 0.99)
    after = analyze(resume_text, job_text, use_semantic=True)
    assert after.match_score == before.match_score
    assert after.text_similarity == before.text_similarity
    assert after.coverage == before.coverage


def test_analyze_with_semantic_and_blank_inputs_returns_none():
    # Blank inputs short-circuit before the model is ever loaded, so this
    # works on a machine without sentence-transformers.
    result = analyze("", "", use_semantic=True)
    assert result.semantic_similarity is None
    assert result.match_score == 0
