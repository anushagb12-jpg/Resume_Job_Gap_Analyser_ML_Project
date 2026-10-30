"""Tests for src/skill_extractor.py.

Two groups of tests live here:

* Behaviour tests check that skills are found, aliases resolve, and false
  positives are avoided.
* Integrity tests check that the skill tables themselves are consistent. They
  are worth having because the tables are hand-written data - a typo in an
  alias would otherwise pass silently.
"""

from __future__ import annotations

from src.skill_extractor import (
    ALL_SKILLS,
    CASE_SENSITIVE_SKILLS,
    IMPLIED_SKILLS,
    SKILL_ALIASES,
    SKILL_CATEGORIES,
    SKILL_LEXICON,
    SURFACE_TO_CANONICAL,
    categorize_skills,
    expand_skills,
    extract_skills,
    find_skill_mentions,
    get_category,
    list_known_skills,
    matching_key,
)

# ---------------------------------------------------------------------------
# Behaviour
# ---------------------------------------------------------------------------


def test_finds_plain_skills():
    found = extract_skills("Python, SQL and Docker")
    assert {"python", "sql", "docker"} <= found


def test_resolves_alias_to_canonical_name():
    assert "scikit-learn" in extract_skills("used sklearn for the model")
    assert "kubernetes" in extract_skills("deployed on k8s")
    assert "python" in extract_skills("wrote py scripts")


def test_matches_hyphen_underscore_and_space_variants():
    for spelling in ("scikit-learn", "scikit_learn", "scikit learn", "Scikit-Learn"):
        assert "scikit-learn" in extract_skills(spelling), spelling


def test_c_does_not_match_inside_c_plus_plus_or_c_sharp():
    assert extract_skills("c++") == {"c++"}
    assert extract_skills("c#") == {"c#"}


def test_c_matches_when_it_really_is_c():
    assert "c" in extract_skills("C and assembly language")


def test_sql_does_not_match_inside_mysql():
    # mysql is its own skill; matching "sql" there too would double count it.
    assert extract_skills("mysql") == {"mysql"}


def test_sql_matches_on_its_own():
    assert "sql" in extract_skills("SQL and mysql")


def test_ambiguous_skill_ignored_when_lowercase():
    # "go", "react", "spring", "express" are ordinary English words too.
    text = "I go to the gym and react to feedback every spring."
    assert extract_skills(text) == set()


def test_ambiguous_skill_found_when_capitalised():
    assert {"go", "react"} <= extract_skills("Skilled in Go and React.")


def test_ambiguous_skill_found_when_uppercase():
    assert "go" in extract_skills("SKILLS: GO, PYTHON")


def test_unambiguous_alias_is_still_case_insensitive():
    assert "go" in extract_skills("wrote golang services")


def test_excel_the_verb_is_not_the_excel_product():
    assert "excel" not in extract_skills("I excel at problem solving")
    assert "excel" in extract_skills("Advanced Excel and problem solving")


def test_multiword_skill():
    assert "machine learning" in extract_skills("3 years of Machine Learning")


def test_skill_inside_a_bullet_list():
    text = "\u2022 Python\n\u2022 Pandas\n\u2022 SQL"
    assert {"python", "pandas", "sql"} <= extract_skills(text)


def test_ambiguous_product_names_need_a_capital_letter():
    # Documented trade-off: a job ad that writes "snowflake" in lower case is
    # missed. That is deliberate, because "snowflake schema" is an unrelated
    # data-warehousing term that would otherwise be a false positive.
    assert "snowflake" not in extract_skills("experience with snowflake schemas")
    assert "snowflake" in extract_skills("experience with Snowflake")


def test_no_skills_found_returns_empty_set():
    assert extract_skills("Enthusiastic person seeking new opportunities") == set()


def test_extract_skills_on_empty_string():
    assert extract_skills("") == set()


def test_find_skill_mentions_returns_the_surface_forms_used():
    mentions = find_skill_mentions("sklearn and scikit-learn and Scikit_Learn")
    assert set(mentions["scikit-learn"]) == {"sklearn", "scikit learn"}


def test_find_skill_mentions_counts_repeats():
    # The job ad says Python three times, which is a signal it really matters.
    mentions = find_skill_mentions("Python. python scripts. python expert")
    assert len(mentions["python"]) == 3


def test_expand_skills_follows_the_chain():
    # express -> node.js -> javascript
    assert {"express", "node.js", "javascript"} <= expand_skills({"express"})


def test_expand_skills_adds_the_general_technology():
    assert {"django", "python"} <= expand_skills({"django"})
    assert {"pyspark", "spark"} <= expand_skills({"pyspark"})


def test_expand_skills_on_empty_input():
    assert expand_skills(set()) == set()


def test_expand_skills_keeps_a_skill_that_implies_nothing():
    # "python" is a target of many implications but a source of none.
    assert expand_skills({"python"}) == {"python"}


def test_get_category_of_a_known_skill():
    assert get_category("python") == "Programming Languages"
    assert get_category("kubernetes") == "Cloud & DevOps"


def test_get_category_of_an_unknown_skill():
    assert get_category("cobol-400") == "Other"


def test_categorize_skills_groups_and_sorts():
    grouped = categorize_skills({"sql", "python", "docker"})
    assert grouped["Programming Languages"] == ["python", "sql"]
    assert grouped["Cloud & DevOps"] == ["docker"]


def test_categorize_skills_leaves_out_empty_categories():
    assert "Soft Skills" not in categorize_skills({"python"})


def test_categorize_skills_on_empty_input():
    assert categorize_skills(set()) == {}


def test_matching_key_folds_separators():
    assert matching_key("scikit-learn") == "scikit learn"
    assert matching_key("scikit_learn") == "scikit learn"


def test_list_known_skills_is_sorted_and_non_empty():
    skills = list_known_skills()
    assert skills == sorted(skills)
    assert len(skills) > 100


# ---------------------------------------------------------------------------
# Integrity of the hand-written tables
# ---------------------------------------------------------------------------


def test_every_alias_points_at_a_known_skill():
    unknown = {
        alias: canonical
        for alias, canonical in SKILL_ALIASES.items()
        if canonical not in SKILL_CATEGORIES
    }
    assert not unknown, f"aliases pointing at skills that do not exist: {unknown}"


def test_no_alias_shares_a_name_with_a_canonical_skill():
    # If an alias were spelled exactly like a different canonical skill, one of
    # them would silently never match.
    canonical_names = set(SKILL_CATEGORIES)
    clashes = {
        alias: canonical
        for alias, canonical in SKILL_ALIASES.items()
        if alias in canonical_names and alias != canonical
    }
    assert not clashes, f"alias/canonical name clashes: {clashes}"


def test_implied_skills_only_reference_known_skills():
    for source, implied in IMPLIED_SKILLS.items():
        assert source in SKILL_CATEGORIES, f"unknown source skill {source!r}"
        for target in implied:
            assert target in SKILL_CATEGORIES, f"unknown implied skill {target!r}"


def test_no_skill_implies_itself():
    self_referential = {
        source: implied
        for source, implied in IMPLIED_SKILLS.items()
        if source in implied
    }
    assert not self_referential, f"skills implying themselves: {self_referential}"


def test_case_sensitive_skills_are_all_known():
    unknown = CASE_SENSITIVE_SKILLS - set(SKILL_CATEGORIES)
    assert not unknown, f"case-sensitive entries that are not skills: {unknown}"


def test_each_skill_appears_in_exactly_one_category():
    seen: dict[str, str] = {}
    duplicates = {}
    for category, skills in SKILL_LEXICON.items():
        for skill in skills:
            if skill in seen:
                duplicates[skill] = (seen[skill], category)
            seen[skill] = category
    assert not duplicates, f"skills listed in two categories: {duplicates}"


def test_all_skills_and_surface_table_agree():
    assert set(ALL_SKILLS) == set(SKILL_CATEGORIES)
    # Every canonical skill must be reachable through the surface-form table.
    assert set(SURFACE_TO_CANONICAL.values()) == set(SKILL_CATEGORIES)


def test_every_surface_form_has_a_compiled_pattern():
    from src.skill_extractor import _PATTERNS

    assert set(_PATTERNS) == set(SURFACE_TO_CANONICAL)
