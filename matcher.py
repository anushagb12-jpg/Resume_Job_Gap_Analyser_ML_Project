"""Compare a resume with a job description and score the match.

This module holds the two pieces of machine learning in the project:

* TF-IDF turns text into numbers. A term that appears often in the job ad but
  rarely elsewhere gets a high weight, which is exactly how we decide which
  missing skill matters most.
* Cosine similarity measures how close two TF-IDF vectors point in space.
  1.0 means "about the same words in about the same proportions", 0.0 means
  "no words in common".

The final score blends two ideas, and the weights are named constants at the
top of the file so they are easy to experiment with:

    score = 70% x skill coverage   (do you have the skills the job lists?)
          + 30% x text similarity  (do the two documents talk about the same things?)

Skill coverage is the dominant part because it is the number a recruiter
actually cares about; text similarity catches related experience that is not
in our skill dictionary.

A third signal - semantic similarity from a neural network - lives in
semantic_matcher.py. It is deliberately NOT part of the score above: it is
stored in its own field (semantic_similarity) and shown alongside the match
score, so the 0-100 number stays fully explainable while the AI opinion is
easy to compare with it.

Separately, each individual skill gets an "importance" score that decides which
gap to fix first. That score is driven mostly by WHERE in the job ad the skill
appears: a skill under "Requirements" outranks one under "Nice to have".
TF-IDF weight and mention frequency provide the tie-breakers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .semantic_matcher import compute_semantic_similarity
from .skill_extractor import (
    IMPLIED_SKILLS,
    expand_skills,
    find_skill_mentions,
    get_category,
    prepare_text,
)
from .text_cleaner import TOKEN_PATTERN

# How much of the score comes from each signal. They must add up to 1.0.
COVERAGE_WEIGHT = 0.7
SIMILARITY_WEIGHT = 0.3

# How much of a skill's importance comes from each signal. Also adds up to 1.0.
TFIDF_IMPORTANCE_WEIGHT = 0.6
FREQUENCY_IMPORTANCE_WEIGHT = 0.4

# An implied skill inherits this fraction of the importance of the tool that
# implies it. 0.8 means "Django matters, and Python matters almost as much".
IMPLIED_INHERITANCE = 0.8

# Section headings in a job ad, and how much a skill mentioned under that
# heading should count. A marker governs every skill after it until the next
# marker appears, which is how a human reads the ad too.
#
# Phrases are deliberately written as stems ("requirement" also matches
# "Requirements", "preferred" also matches "Preferred Qualifications").
REQUIRED_SECTION_MARKERS = (
    "requirement",
    "must have",
    "must-have",
    "essential",
    "we require",
    "you will need",
    "responsibilities",
    "minimum qualification",
    "basic qualification",
)
OPTIONAL_SECTION_MARKERS = (
    "nice to have",
    "nice-to-have",
    "preferred",
    "desirable",
    "bonus points",
    "a plus",
    "good to have",
    "advantage",
    "optional",
)

SECTION_WEIGHT_REQUIRED = 1.0
SECTION_WEIGHT_OPTIONAL = 0.5
# Used before the first heading is reached, or in ads with no headings at all.
SECTION_WEIGHT_DEFAULT = 0.8


@dataclass
class MatchResult:
    """Everything the app needs to show one analysis.

    Attributes
    ----------
    resume_skills, job_skills:
        All skills found on each side, including implied ones.
    matched_skills:
        Skills the job wants AND the resume shows.
    missing_skills:
        The gap - skills the job wants that the resume does not show.
    extra_skills:
        Skills the resume has that this job did not ask for. Not a bad thing:
        it is what makes a candidate interesting for a different role.
    coverage:
        Fraction of the job's skills that are matched, from 0.0 to 1.0.
    text_similarity:
        TF-IDF cosine similarity of the two documents, from 0.0 to 1.0.
    match_score:
        The headline number shown to the user, from 0 to 100.
    semantic_similarity:
        Optional AI opinion: the cosine similarity of the two texts'
        sentence-transformer embeddings (see semantic_matcher.py), or None
        when it was not requested or could not be computed. Deliberately
        kept out of match_score so the two numbers tell separate stories.
    skill_importance:
        How much the job emphasises each of its skills, from 0.0 to 1.0.
    gap_table:
        The same information as a pandas DataFrame, ready to display.
    """

    resume_skills: set[str]
    job_skills: set[str]
    matched_skills: list[str]
    missing_skills: list[str]
    extra_skills: list[str]
    coverage: float
    text_similarity: float
    match_score: int
    semantic_similarity: float | None = None
    skill_importance: dict[str, float] = field(default_factory=dict)
    gap_table: pd.DataFrame = field(default_factory=pd.DataFrame)


def _normalize(scores: dict[str, float]) -> dict[str, float]:
    """Rescale a dict of scores so the largest becomes 1.0.

    Normalising each signal separately is what lets us add TF-IDF weights (tiny
    decimals) to mention counts (whole numbers) without one swamping the other.
    """
    if not scores:
        return {}
    largest = max(scores.values())
    if largest <= 0:
        return {name: 0.0 for name in scores}
    return {name: value / largest for name, value in scores.items()}


def _skill_analyzer(text: str) -> list[str]:
    """TF-IDF analyzer that treats each skill mention as one token.

    We hand this to TfidfVectorizer instead of letting it split on spaces,
    because skills are things like "scikit learn", "ci/cd" and "node.js" that a
    generic tokenizer would chop into meaningless pieces.
    """
    mentions = find_skill_mentions(text)
    return [surface for surfaces in mentions.values() for surface in surfaces]


def _tfidf_skill_weights(job_text: str, resume_text: str) -> dict[str, float]:
    """Return the TF-IDF weight of every skill surface form in the job ad.

    The vectorizer is fitted on two documents: the job ad and the resume. That
    is what makes the "inverse document frequency" part meaningful - a skill
    mentioned only in the job ad scores higher than one both documents share,
    because it is the job ad that is insisting on it.
    """
    vectorizer = TfidfVectorizer(analyzer=_skill_analyzer, ngram_range=(1, 1))
    try:
        # Row 0 is the job ad, row 1 is the resume.
        matrix = vectorizer.fit_transform([job_text, resume_text])
    except ValueError:
        # Raised when the vocabulary is empty, i.e. no skills were recognised
        # in either document. There is nothing to weigh, so return nothing.
        return {}

    weights = dict(zip(vectorizer.get_feature_names_out(), matrix[0].toarray()[0]))
    return weights


def _find_section_markers(text: str) -> list[tuple[int, float]]:
    """Locate every section heading in a job ad, with its weight.

    Returns a list of (character offset, weight) sorted by offset. Each marker
    governs the text after it until the next marker appears.

    The marker phrases are deliberately chosen so that no optional phrase sits
    inside a required one. That is what keeps "Preferred Qualifications"
    optional: if a shorter required-sounding word matched inside it, the ad's
    nice-to-have skills would be scored as hard requirements.
    """
    markers: list[tuple[int, float]] = []
    for phrases, weight in (
        (REQUIRED_SECTION_MARKERS, SECTION_WEIGHT_REQUIRED),
        (OPTIONAL_SECTION_MARKERS, SECTION_WEIGHT_OPTIONAL),
    ):
        for phrase in phrases:
            start = text.find(phrase)
            while start != -1:
                markers.append((start, weight))
                start = text.find(phrase, start + 1)

    markers.sort()
    return markers


def score_sections(
    job_text: str, job_mentions: dict[str, list[str]]
) -> dict[str, float]:
    """Score each skill by which part of the job ad mentions it (0.0-1.0).

    A skill under "Requirements" scores 1.0, one under "Nice to have" scores
    0.5, and one before any heading scores 0.8. When a skill is mentioned in
    several places the highest score wins, because an employer that lists a
    skill as both required and desirable clearly requires it.
    """
    folded = prepare_text(job_text, lowercase=True)
    markers = _find_section_markers(folded)

    scores: dict[str, float] = {}
    for skill, surfaces in job_mentions.items():
        weights: list[float] = []
        for surface in surfaces:
            offset = folded.find(surface.lower())
            if offset == -1:
                continue
            # The governing heading is the last marker that appears BEFORE the
            # mention. If there is none, the skill sits above any heading and
            # gets the default weight.
            governing = SECTION_WEIGHT_DEFAULT
            for marker_offset, weight in markers:
                if marker_offset >= offset:
                    break
                governing = weight
            weights.append(governing)
        # A skill mentioned in two sections takes the higher weight: if an
        # employer lists it as both required and desirable, they require it.
        scores[skill] = max(weights) if weights else SECTION_WEIGHT_DEFAULT
    return scores


def score_skill_importance(
    job_text: str,
    resume_text: str,
    job_mentions: dict[str, list[str]],
) -> dict[str, float]:
    """Score how much the job ad cares about each skill it mentions (0.0-1.0).

    Three signals are combined:

    section    is the skill in the "must have" part or the "nice to have" part?
               This is by far the strongest signal.
    frequency  how many times the skill is written in the ad. A skill listed
               under two different headings is clearly important.
    tfidf      how distinctive the skill is to the ad compared with the resume.

    We take the HIGHEST TF-IDF weight among a skill's surface forms rather than
    their sum. Summing would quietly reward skills that happen to be written
    twice, which is the frequency signal's job - counting it twice lets one
    chatty skill squash the whole scale.

    The result is normalised so the most important skill is always exactly 1.0,
    which makes the priority cut-offs in recommendations.py meaningful.
    """
    if not job_mentions:
        return {}

    tfidf_weights = _tfidf_skill_weights(job_text, resume_text)

    raw_tfidf: dict[str, float] = {}
    raw_frequency: dict[str, int] = {}
    for skill, surfaces in job_mentions.items():
        raw_tfidf[skill] = max(
            (tfidf_weights.get(surface, 0.0) for surface in surfaces), default=0.0
        )
        raw_frequency[skill] = len(surfaces)

    tfidf_part = _normalize(raw_tfidf)
    frequency_part = _normalize(raw_frequency)
    section_part = score_sections(job_text, job_mentions)

    raw_importance = {
        skill: section_part[skill]
        * (
            TFIDF_IMPORTANCE_WEIGHT * tfidf_part[skill]
            + FREQUENCY_IMPORTANCE_WEIGHT * frequency_part[skill]
        )
        for skill in job_mentions
    }

    return {
        skill: round(value, 3) for skill, value in _normalize(raw_importance).items()
    }


def compute_text_similarity(resume_text: str, job_text: str) -> float:
    """Return the TF-IDF cosine similarity of the two documents (0.0-1.0).

    Unlike score_skill_importance this looks at every word, so it picks up on
    related experience even when the exact skill is not in our dictionary.
    """
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words="english",
        lowercase=True,
        token_pattern=TOKEN_PATTERN,
    )
    try:
        matrix = vectorizer.fit_transform([job_text, resume_text])
    except ValueError:
        # Empty vocabulary: neither document had a single usable word.
        return 0.0

    similarity = float(cosine_similarity(matrix[0], matrix[1])[0][0])
    # Floating point can nudge the value a hair outside 0..1.
    return min(max(similarity, 0.0), 1.0)


def _inherit_importance(importance: dict[str, float], job_skills: set[str]) -> dict[str, float]:
    """Give implied skills an importance inherited from the tool that implies them.

    A job ad that asks for Django without ever typing "Python" still needs
    Python, so Python inherits 80% of Django's importance. The loop repeats
    until nothing changes so chains (express -> node.js -> javascript) carry
    all the way through.
    """
    inherited = dict(importance)
    for _ in range(len(IMPLIED_SKILLS) + 1):
        changed = False
        for skill in job_skills:
            for implied in IMPLIED_SKILLS.get(skill, ()):
                if implied not in job_skills:
                    continue
                candidate = inherited.get(skill, 0.0) * IMPLIED_INHERITANCE
                if candidate > inherited.get(implied, 0.0):
                    inherited[implied] = round(candidate, 3)
                    changed = True
        if not changed:
            break
    return inherited


def build_gap_table(
    job_skills: set[str],
    resume_skills: set[str],
    importance: dict[str, float],
    mentioned_skills: set[str],
) -> pd.DataFrame:
    """Build the table the app displays: one row per skill the job wants.

    Rows are ordered so the most important gaps come first, which is the order
    a candidate should read them in.
    """
    rows = []
    for skill in job_skills:
        matched = skill in resume_skills
        rows.append(
            {
                "Skill": skill,
                "Category": get_category(skill),
                "Status": "Match" if matched else "Gap",
                "Importance": importance.get(skill, 0.0),
                "In job ad": "listed" if skill in mentioned_skills else "implied",
            }
        )

    table = pd.DataFrame(rows)
    if table.empty:
        # Keep the columns even when there is nothing to show, so the app can
        # always render a table header instead of crashing.
        return pd.DataFrame(
            columns=["Skill", "Category", "Status", "Importance", "In job ad"]
        )

    return table.sort_values(
        by=["Status", "Importance", "Skill"],
        ascending=[True, False, True],  # "Gap" sorts before "Match"
    ).reset_index(drop=True)


def analyze(
    resume_text: str, job_text: str, use_semantic: bool = False
) -> MatchResult:
    """Run the whole comparison and return a MatchResult.

    This is the single function app.py calls. Everything else in this module
    exists to support it.

    use_semantic also computes the neural-network similarity from
    semantic_matcher.py and attaches it as result.semantic_similarity. It is
    off by default so this module never needs the heavy
    sentence-transformers package, and the score it produces never feeds
    into match_score.
    """
    job_mentions = find_skill_mentions(job_text)
    resume_mentions = find_skill_mentions(resume_text)

    # Keep the literally-mentioned skills separate so we can label them in the
    # table and let implied skills inherit their importance.
    job_literal = set(job_mentions)
    resume_literal = set(resume_mentions)

    job_skills = expand_skills(job_literal)
    resume_skills = expand_skills(resume_literal)

    matched = job_skills & resume_skills
    missing = job_skills - resume_skills
    extra = resume_skills - job_skills

    coverage = len(matched) / len(job_skills) if job_skills else 0.0
    similarity = compute_text_similarity(resume_text, job_text)

    if job_skills:
        blended = COVERAGE_WEIGHT * coverage + SIMILARITY_WEIGHT * similarity
    else:
        # No recognisable skills in the job ad, so there is nothing to cover.
        # Fall back to the overall text similarity rather than reporting a
        # misleading zero.
        blended = similarity

    importance = _inherit_importance(
        score_skill_importance(job_text, resume_text, job_mentions),
        job_skills,
    )

    semantic_similarity = (
        compute_semantic_similarity(resume_text, job_text) if use_semantic else None
    )

    return MatchResult(
        resume_skills=resume_skills,
        job_skills=job_skills,
        matched_skills=sorted(matched),
        missing_skills=sorted(missing),
        extra_skills=sorted(extra),
        coverage=round(coverage, 3),
        text_similarity=round(similarity, 3),
        match_score=int(round(blended * 100)),
        semantic_similarity=round(semantic_similarity, 3)
        if semantic_similarity is not None
        else None,
        skill_importance=importance,
        gap_table=build_gap_table(job_skills, resume_skills, importance, job_literal),
    )
