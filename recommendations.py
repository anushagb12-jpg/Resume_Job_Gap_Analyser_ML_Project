"""Turn the skill gap into ranked, actionable advice.

matcher.py answers "what is missing?". This module answers "so what should I
do first?". The ordering comes from the importance scores, so a candidate with
ten gaps is told which two are actually worth their weekend.

No URLs are hard-coded here on purpose: links rot. Each suggestion names a
platform and a search phrase instead, which stays useful over time.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from .matcher import MatchResult
from .skill_extractor import get_category

# Importance is between 0 and 1. These cut-offs are hand-tuned against real job
# ads rather than derived, so treat them as starting points to experiment with.
# Observed ranges: a skill from the "Requirements" section lands around 0.8-1.0,
# one from an unheaded paragraph around 0.6-0.8, and one from "Nice to have"
# around 0.4-0.5.
HIGH_PRIORITY_MIN = 0.7
MEDIUM_PRIORITY_MIN = 0.55

# Match score (0-100) labels used for the headline in the app.
VERDICTS = (
    (80, "Strong match"),
    (60, "Good match"),
    (40, "Partial match"),
    (0, "Early days"),
)

# What to do, and where to look, for each skill category.
LEARNING_PLAN: dict[str, tuple[str, str]] = {
    "Programming Languages": (
        "Build one small project in {skill} and push it to GitHub.",
        "Exercism or freeCodeCamp - search '{skill} for beginners'",
    ),
    "Web Frameworks & Libraries": (
        "Rebuild a page you already know in {skill} to learn its idioms.",
        "Official {skill} tutorial - search '{skill} quickstart'",
    ),
    "Machine Learning & AI": (
        "Practise {skill} on one real dataset from start to finish, then write up "
        "what you found.",
        "Kaggle Learn or the official docs - search '{skill} tutorial'",
    ),
    "Data Engineering & Databases": (
        "Put a real dataset through {skill} end to end: load it, transform it, "
        "query it.",
        "Mode SQL tutorial or the {skill} docs - search '{skill} getting started'",
    ),
    "Cloud & DevOps": (
        "Run one real task with {skill} from start to finish and write down what "
        "broke - that is what interviewers ask about.",
        "{skill} official free training - search '{skill} fundamentals course'",
    ),
    "Software Engineering Practices": (
        "Apply {skill} on your next personal project and write down what changed.",
        "Search '{skill} explained' - Refactoring Guru covers most of these",
    ),
    "Soft Skills": (
        "Collect one concrete story that shows {skill}, and add it to your resume "
        "as a measurable result rather than a bare word.",
        "Search 'STAR method interview answers' to frame the story",
    ),
}

# Used when a skill is not in one of the categories above.
DEFAULT_PLAN = (
    "Find one example of {skill} in a real project and reproduce it.",
    "Search '{skill} beginner tutorial'",
)


@dataclass
class Recommendation:
    """One ranked suggestion for closing a gap."""

    skill: str
    category: str
    priority: str
    importance: float
    listed_in_ad: bool
    action: str
    where_to_learn: str


def priority_for(importance: float) -> str:
    """Map an importance score onto High / Medium / Low."""
    if importance >= HIGH_PRIORITY_MIN:
        return "High"
    if importance >= MEDIUM_PRIORITY_MIN:
        return "Medium"
    return "Low"


def verdict_for(match_score: int) -> str:
    """Return a short human label for a match score."""
    for threshold, label in VERDICTS:
        if match_score >= threshold:
            return label
    return VERDICTS[-1][1]


def build_recommendation(
    skill: str, importance: float, listed_in_ad: bool
) -> Recommendation:
    """Create one Recommendation for a missing skill."""
    category = get_category(skill)
    action_template, resource_template = LEARNING_PLAN.get(category, DEFAULT_PLAN)
    return Recommendation(
        skill=skill,
        category=category,
        priority=priority_for(importance),
        importance=importance,
        listed_in_ad=listed_in_ad,
        action=action_template.format(skill=skill),
        where_to_learn=resource_template.format(skill=skill),
    )


def generate_recommendations(result: MatchResult, limit: int = 8) -> list[Recommendation]:
    """Return the most important gaps as ranked recommendations.

    Sorting is by importance, then by whether the job ad named the skill
    explicitly (an explicit ask beats an implied one at equal importance), then
    alphabetically so the output is stable between runs.
    """
    listed_in_ad = _listed_skills(result)

    ranked = sorted(
        result.missing_skills,
        key=lambda skill: (
            -result.skill_importance.get(skill, 0.0),
            skill not in listed_in_ad,  # False sorts before True, so listed wins
            skill,
        ),
    )

    return [
        build_recommendation(
            skill,
            result.skill_importance.get(skill, 0.0),
            skill in listed_in_ad,
        )
        for skill in ranked[:limit]
    ]


def _listed_skills(result: MatchResult) -> set[str]:
    """Skills the job ad named explicitly, read back from the gap table."""
    if result.gap_table.empty:
        return set()
    listed = result.gap_table[result.gap_table["In job ad"] == "listed"]
    return set(listed["Skill"])


def highlight_strengths(result: MatchResult, limit: int = 6) -> list[str]:
    """The matched skills the job cares about most, best first.

    Worth showing alongside the gaps: a candidate who only sees what they lack
    has no idea what is already working.
    """
    matched = sorted(
        result.matched_skills,
        key=lambda skill: (-result.skill_importance.get(skill, 0.0), skill),
    )
    return matched[:limit]


def to_dataframe(recommendations: list[Recommendation]) -> pd.DataFrame:
    """Convert recommendations into a table the app can display."""
    columns = {
        "skill": "Skill",
        "category": "Category",
        "priority": "Priority",
        "importance": "Importance",
        "listed_in_ad": "Asked for directly",
        "action": "What to do",
        "where_to_learn": "Where to learn",
    }
    if not recommendations:
        return pd.DataFrame(columns=list(columns.values()))

    frame = pd.DataFrame([asdict(item) for item in recommendations])
    return frame.rename(columns=columns)[list(columns.values())]


def summarize(result: MatchResult, recommendations: list[Recommendation]) -> str:
    """Write a short plain-English summary of the whole analysis."""
    lines = [
        f"Match score: {result.match_score}/100 ({verdict_for(result.match_score)}).",
        f"The job asks for {len(result.job_skills)} skills and your resume shows "
        f"{len(result.matched_skills)} of them ({int(result.coverage * 100)}% coverage).",
        f"Overall wording similarity between the two documents is "
        f"{int(result.text_similarity * 100)}%.",
    ]

    if not result.missing_skills:
        lines.append("No skill gaps found - every skill in the job ad appears in your resume.")
        return "\n".join(lines)

    top = ", ".join(item.skill for item in recommendations[:3])
    lines.append(
        f"{len(result.missing_skills)} skill(s) are missing. "
        f"Start with: {top}."
    )
    if result.extra_skills:
        lines.append(
            f"Your resume also shows {len(result.extra_skills)} skill(s) this job did not "
            "ask for - worth keeping, and useful for other roles."
        )
    return "\n".join(lines)
