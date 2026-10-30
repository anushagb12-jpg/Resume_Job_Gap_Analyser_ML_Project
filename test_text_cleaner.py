"""Tests for src/text_cleaner.py."""

from __future__ import annotations

from src.text_cleaner import (
    STOPWORDS,
    clean_text,
    describe_text,
    normalize_whitespace,
    remove_noise,
    remove_stopwords,
    tokenize,
)


def test_clean_text_lowercases_by_default():
    assert clean_text("Python AND Pandas") == "python and pandas"


def test_clean_text_can_keep_capital_letters():
    # The skill extractor needs capitals to tell "Go" from "go".
    assert clean_text("Know Go and React", lowercase=False) == "Know Go and React"


def test_clean_text_collapses_whitespace():
    assert clean_text("python\n\n\t  pandas   sql") == "python pandas sql"


def test_normalize_whitespace_trims_the_ends():
    assert normalize_whitespace("   hello   world  ") == "hello world"


def test_remove_noise_strips_email():
    assert "jane@example.com" not in remove_noise("Contact jane@example.com today")


def test_remove_noise_strips_url():
    result = remove_noise("See https://example.com/resume for more")
    assert "https://example.com/resume" not in result
    assert "See" in result and "for more" in result


def test_remove_noise_strips_phone_number():
    result = remove_noise("Call +1 (555) 010-2030 now")
    assert "010" not in result
    assert "Call" in result and "now" in result


def test_remove_noise_keeps_dates_and_short_numbers():
    # Removing phone numbers must not eat ordinary numbers from the resume.
    text = "Data Analyst (2022 - present), saved 6 hours a week"
    assert "2022" in remove_noise(text)
    assert "6 hours" in remove_noise(text)


def test_remove_noise_strips_bullet_characters():
    # Noise is replaced with a space, not deleted, so neighbouring words do not
    # get glued together. clean_text() collapses the extra spaces afterwards.
    result = remove_noise("\u2022 python \u25cf sql")
    assert "\u2022" not in result
    assert "\u25cf" not in result
    assert "python" in result and "sql" in result


def test_clean_text_collapses_the_gaps_left_by_bullets():
    assert clean_text("\u2022 python\n\u2022 sql") == "python sql"


def test_clean_text_on_empty_string():
    assert clean_text("") == ""


def test_clean_text_full_pipeline_on_a_realistic_header():
    raw = "Alex Morgan\nalex@example.com | +1 (555) 010-2030\nSKILLS:  Python, SQL"
    cleaned = clean_text(raw)
    assert cleaned == "alex morgan skills: python, sql"


def test_tokenize_keeps_technical_tokens():
    tokens = tokenize("c++ c# node.js scikit-learn")
    assert "c++" in tokens
    assert "c#" in tokens
    assert "node.js" in tokens
    assert "scikit-learn" in tokens


def test_tokenize_lowercases():
    assert tokenize("Python Pandas") == ["python", "pandas"]


def test_tokenize_splits_on_characters_it_does_not_allow():
    # "/" is not part of a token, so ci/cd becomes two tokens. That is fine -
    # the skill extractor has its own matching rules and does not use tokenize.
    assert tokenize("ci/cd") == ["ci", "cd"]


def test_tokenize_on_empty_string():
    assert tokenize("") == []


def test_remove_stopwords_drops_common_words():
    assert remove_stopwords(["python", "and", "the", "sql"]) == ["python", "sql"]


def test_remove_stopwords_keeps_short_skill_words():
    # "r", "c" and "go" are real skills, so they must not be in the stopword list.
    for skill in ("r", "c", "go", "sql"):
        assert skill not in STOPWORDS


def test_describe_text_counts():
    stats = describe_text("python pandas\nsql\n\n")
    assert stats["words"] == 3
    assert stats["unique_words"] == 3
    assert stats["lines"] == 2
    assert stats["characters"] == len("python pandas\nsql\n\n")
