"""Turn messy resume text into tidy text.

Resumes are full of noise that does not help us compare skills: email
addresses, phone numbers, URLs, bullet characters, tabs and random line
breaks. This module removes that noise so the other modules can focus on
the words that actually matter.

Everything here is pure text-in / text-out, which makes it easy to test.
"""

from __future__ import annotations

import re

from sklearn.feature_extraction.text import CountVectorizer

# ---------------------------------------------------------------------------
# Regular expressions (patterns used to find noise in the text)
# ---------------------------------------------------------------------------
# A web address, e.g. https://example.com or www.example.com
URL_PATTERN = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)

# An email address, e.g. jane.doe@example.com
EMAIL_PATTERN = re.compile(r"\S+@\S+\.\S+")

# A phone number with at least 10 digits, e.g. +1 (555) 123-4567
PHONE_PATTERN = re.compile(
    r"(?<!\d)"                      # not in the middle of a longer number
    r"(?:\+?\d{1,3}[\s.-]?)?"       # optional country code
    r"(?:\(?\d{3}\)?[\s.-])"        # area code
    r"\d{3}[\s.-]?\d{4}"            # remaining 7 digits
    r"(?!\d)"
)

# Characters used as bullets/decoration in resumes. They carry no meaning.
DECORATION_PATTERN = re.compile(r"[\u2022\u2023\u25cf\u25aa\u25cb\*\|\[\]{}<>~`^]")

# Any run of whitespace (spaces, tabs, newlines) collapses to one single space.
WHITESPACE_PATTERN = re.compile(r"\s+")

# A "word" for our purposes: starts with a letter or digit and may contain
# + # . _ - so that skills like "c++", "c#", "node.js" and "ci-cd" survive.
TOKEN_PATTERN = r"(?u)[a-z0-9][a-z0-9+#._-]*"

# Re-used by tokenize(). Building a vectorizer is cheap, but there is no
# reason to build a new one on every call.
_TOKENIZER = CountVectorizer(lowercase=True, token_pattern=TOKEN_PATTERN)

# A small stopword list. These words appear everywhere and tell us nothing
# about a candidate's skills. We keep it short and readable on purpose.
STOPWORDS = frozenset(
    """
    a about after all also an and any are as at be because been before being
    but by can could did do does done down each even every for from had has
    have he her here him his how i if in into is it its just like made make
    me more most my no not now of on only or other our out over own said same
    she should so some such than that the their them then there these they
    this those through to too up very was we were what when where which who
    will with would you your
    """.split()
)


def remove_noise(text: str) -> str:
    """Replace contact details and decoration characters with a space.

    We replace with a space (not with nothing) so two words on either side of
    the removed text do not get glued together.
    """
    text = URL_PATTERN.sub(" ", text)
    text = EMAIL_PATTERN.sub(" ", text)
    text = PHONE_PATTERN.sub(" ", text)
    text = DECORATION_PATTERN.sub(" ", text)
    return text


def normalize_whitespace(text: str) -> str:
    """Collapse every run of spaces/tabs/newlines into a single space."""
    return WHITESPACE_PATTERN.sub(" ", text).strip()


def clean_text(text: str, lowercase: bool = True) -> str:
    """Run the full cleaning pipeline and return one tidy line of text.

    Parameters
    ----------
    text:
        Raw text, for example everything PyMuPDF pulled out of a PDF.
    lowercase:
        When True (the default) the result is fully lowercased.

        Set this to False when you need to keep capital letters. The skill
        extractor does that, because "Go" with a capital G is the programming
        language while "go" in the middle of a sentence is just an English
        verb. Keeping the case is what lets us tell them apart.
    """
    cleaned = remove_noise(text)
    cleaned = normalize_whitespace(cleaned)
    if lowercase:
        cleaned = cleaned.lower()
    return cleaned


def tokenize(text: str) -> list[str]:
    """Split text into a list of lowercase word-like tokens.

    We let scikit-learn's CountVectorizer do the splitting so the tokenizing
    rules are exactly the same ones scikit-learn will use later when it builds
    the TF-IDF vectors. Using one definition of "a word" everywhere avoids
    subtle mismatches that are very hard to debug.
    """
    return _TOKENIZER.build_analyzer()(text)


def remove_stopwords(tokens: list[str]) -> list[str]:
    """Drop common English words from a list of tokens."""
    return [token for token in tokens if token not in STOPWORDS]


def describe_text(text: str) -> dict[str, int]:
    """Return simple statistics about a piece of text.

    The app shows these numbers so a user can confirm the PDF was read
    correctly before trusting any match score.
    """
    tokens = tokenize(text)
    return {
        "characters": len(text),
        "words": len(tokens),
        "unique_words": len(set(tokens)),
        "lines": len([line for line in text.splitlines() if line.strip()]),
    }
