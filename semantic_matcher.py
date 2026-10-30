"""Semantic similarity between a resume and a job description.

matcher.py compares the two documents by which WORDS they share (TF-IDF).
This module adds a second opinion: it compares them by MEANING, using a
"sentence transformer" neural network.

How the model works, in three steps:

1. It reads a piece of text and turns it into 384 numbers (an embedding).
   Those numbers were learned from millions of example sentences, so texts
   about the same topic end up with similar numbers - "built dashboards"
   lands close to "created visual reports" even though they share no words.
2. Each document becomes one vector, pointing somewhere in a 384-dimensional
   "meaning space".
3. The similarity of two documents is the cosine of the angle between their
   vectors: 1.0 means the same meaning, 0.0 means unrelated, and a small
   negative value means the model sees them as very different.

We use all-MiniLM-L6-v2, a deliberately small model (~80 MB) that runs on a
normal laptop CPU in well under a second. It is a general-purpose model
trained on generic sentence pairs - not on resumes or hiring decisions - so
treat its score as one rough opinion, not a measurement.

The score is kept SEPARATE from the match score on purpose: it lives in its
own field (MatchResult.semantic_similarity) and is never blended into the
0-100 number, which stays fully explainable.

The heavy `sentence-transformers` package is optional. When it is missing -
or the model cannot be downloaded - every function here returns None and
the rest of the app keeps working with word-based scores only.
"""

from __future__ import annotations

import numpy as np

# The model we pin. Small, fast, and the standard choice for exactly this
# kind of "are these two texts about the same thing?" question.
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# all-MiniLM-L6-v2 reads at most 256 word-pieces (roughly half a page). A
# resume is often 400+ words, so long texts are split into chunks of this
# many words (~230 word-pieces) whose embeddings are averaged. Without this,
# the model would silently judge a resume by its first half-page only.
CHUNK_WORDS = 180

# The model is loaded at most once per process; loading takes a few seconds.
_MODEL = None
_LOAD_ATTEMPTED = False


def _get_model():
    """Load the model once, or return None when that is not possible.

    None (rather than an exception) is returned when `sentence-transformers`
    is not installed or the model file cannot be downloaded, so the rest of
    the app keeps working without the AI score. A failed attempt is
    remembered so we do not retry it on every call.
    """
    global _MODEL, _LOAD_ATTEMPTED
    if _LOAD_ATTEMPTED:
        return _MODEL
    _LOAD_ATTEMPTED = True
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None
    try:
        _MODEL = SentenceTransformer(MODEL_NAME)
    except Exception:
        # Most common cause: no network the first time the model needs to be
        # downloaded. Once cached locally it loads without a connection.
        return None
    return _MODEL


def is_semantic_available() -> bool:
    """True when the model is installed and loaded successfully."""
    return _get_model() is not None


def _split_into_chunks(text: str, chunk_words: int = CHUNK_WORDS) -> list[str]:
    """Split a long text into consecutive chunks of at most `chunk_words` words.

    No words are dropped and the order never changes, so joining the chunks
    back together reproduces the original words exactly.
    """
    words = text.split()
    if not words:
        return []
    return [
        " ".join(words[start : start + chunk_words])
        for start in range(0, len(words), chunk_words)
    ]


def _embed_long_text(model, text: str) -> np.ndarray | None:
    """Embed a text of any length by averaging its chunk embeddings.

    The model truncates anything past ~256 word-pieces, so a whole resume
    would be judged by its first half-page. Embedding chunk by chunk and
    averaging gives every part of the document an equal vote. The average is
    re-normalised to length 1 so the cosine between two vectors still ranges
    from -1 to 1.
    """
    chunks = _split_into_chunks(text)
    if not chunks:
        return None
    embeddings = model.encode(chunks, normalize_embeddings=True)
    averaged = embeddings.mean(axis=0)
    length = float(np.linalg.norm(averaged))
    if length == 0.0:
        return None
    return averaged / length


def compute_semantic_similarity(resume_text: str, job_text: str) -> float | None:
    """Return the cosine similarity of the two texts' embeddings (-1.0 to 1.0).

    This is the model's opinion of whether the two documents discuss the same
    kind of work - it is not a probability of any real-world outcome.

    Returns None, instead of guessing a number, when either text is blank or
    when the model is unavailable (package not installed, or no network for
    the one-time download). Callers show "n/a" rather than pretending the
    score is zero.
    """
    if not (resume_text or "").strip() or not (job_text or "").strip():
        return None

    model = _get_model()
    if model is None:
        return None

    resume_vector = _embed_long_text(model, resume_text)
    job_vector = _embed_long_text(model, job_text)
    if resume_vector is None or job_vector is None:
        return None

    return float(resume_vector @ job_vector)
