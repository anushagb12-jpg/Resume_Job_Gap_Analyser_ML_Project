# 🎯 Resume-Job Gap Analyzer

Paste a job advert, upload your resume, and find out **which skills you already
have, which ones are missing, and what to learn first**.

This is a beginner-friendly machine learning project. It is built from small,
separate modules, every function has one job, and there are no hidden steps -
you can read the code top to bottom and understand exactly how a score was
produced.

Everything runs on your own machine. No resume is uploaded anywhere.

---

## What it does

| You give it | It gives you |
|---|---|
| A resume (PDF, TXT, MD or pasted text) | A **match score** from 0 to 100 |
| A job description | A **skill gap table** sorted by how much the employer cares |
| | **Two separate similarity scores**: word-based (TF-IDF) and AI semantic |
| | **Ranked recommendations** with a concrete next step for each gap |
| | Your **strongest matches** and your bonus skills |
| | An optional **local AI assistant** to chat about the gaps and get a study plan |

---

## Quick start

```bash
# 1. Create and activate a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS / Linux

# 2. Install the dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

Your browser opens at <http://localhost:8501>. To try it immediately without a
resume of your own, switch on **Use sample resume** in the sidebar, press
**Fill in a sample job description**, then press **Analyze match**.

The first analysis with **Add AI semantic similarity** switched on downloads a
small neural network (~90 MB) once, then works offline. Turn the toggle off to
skip the download entirely - everything else works without it.

Press **Start over** at the top of the page at any time to clear the inputs and
the last analysis and return to the welcome screen.

### The AI assistant tab (optional)

The **AI assistant** tab that appears with the results chats about your gaps,
recommends learning resources and drafts a 30-day study plan. It needs
[Ollama](https://ollama.com) with a small local model:

```bash
# 1. Install Ollama from https://ollama.com/download
#    (on Windows and macOS it runs automatically after installing)

# 2. Download the model (~2 GB, once)
ollama pull llama3.2:3b
```

The app talks only to `http://localhost:11434` - your resume is never sent to
an external AI service. Without Ollama the tab simply explains how to set it
up, and everything else works. If your Ollama listens on a different address,
point the app at it with the `OLLAMA_BASE_URL` environment variable. There is
no API key to configure: local Ollama has no authentication, so there is no
secret anywhere in this project.

### Run the tests

```bash
python -m pytest tests/ -v
```

---

## Project structure

```
Resume_Job_Gap_Analyzer/
├── app.py                      Streamlit UI - collects input, draws results
├── requirements.txt            Dependencies
├── README.md                   This file
├── src/
│   ├── __init__.py
│   ├── resume_parser.py        PDF / text file  ->  raw text
│   ├── text_cleaner.py         raw text  ->  tidy text
│   ├── skill_extractor.py      tidy text  ->  a set of known skills
│   ├── matcher.py              two skill sets  ->  score + gap table
│   ├── semantic_matcher.py     two texts  ->  AI meaning-similarity (optional)
│   ├── recommendations.py      gap table  ->  ranked advice
│   └── ai_assistant.py         MatchResult  ->  chat answers from local Ollama
└── tests/
    ├── conftest.py             Shared fixtures and sys.path setup
    ├── test_resume_parser.py
    ├── test_text_cleaner.py
    ├── test_skill_extractor.py
    ├── test_matcher.py
    ├── test_semantic_matcher.py
    ├── test_ai_assistant.py    Mocked Ollama API - no server needed
    ├── test_recommendations.py
    └── test_app.py             Drives the real UI with Streamlit's AppTest
```

Data flows in one direction only:

```
uploaded file ──> resume_parser ──> text_cleaner ──> skill_extractor ──┐
                                                                       ├──> matcher ──> recommendations ──> app.py
pasted job advert ──────────────────> text_cleaner ──> skill_extractor ┘
```

The AI assistant is an optional side branch off `matcher`: when you ask
something in the chat tab, the result plus both raw texts flow into
`ai_assistant` - and no further than your own Ollama server.

`app.py` never does any analysis itself. That is what makes the logic testable
without starting a web server.

---

## What each file does

### `src/resume_parser.py`
Turns a file into a string. PDFs are read page by page with
[PyMuPDF](https://pymupdf.readthedocs.io/). Text and Markdown files are decoded
as UTF-8, falling back to Windows-1252 for older documents.

It raises `ResumeParseError` with a message a human can act on - for a
password-protected PDF, an empty file, or a scanned image with no selectable
text. `app.py` catches exactly that exception, so real bugs still surface.

### `src/text_cleaner.py`
Removes the noise: URLs, email addresses, phone numbers, bullet characters and
stray whitespace. Also provides `tokenize()` and `describe_text()`.

`clean_text()` takes a `lowercase` argument. The skill extractor passes
`lowercase=False` so it can still see capital letters - which turns out to
matter a lot (see below).

### `src/skill_extractor.py`
A hand-written dictionary of **221 skills** across seven categories, plus
**148 aliases** that all point at one canonical spelling. That means
`sklearn`, `scikit learn` and `Scikit-Learn` are all recognised as the same
skill, so the resume and the job advert are always compared using the same word.

Three details do most of the work:

1. **Word boundaries.** A naive search for `c` would match inside `c++`, `c#`
   and `music`. Each skill gets a regex that stops at letters, digits, `+` and
   `#`. This is also why `sql` does not match inside `mysql` - `mysql` is a
   skill in its own right.
2. **Capitals disambiguate.** `go`, `react`, `spring`, `excel`, `rust` and
   about twenty others are ordinary English words too. A resume writes the
   *technology* capitalised and the *verb* in lower case, so those skills are
   only matched when capitalised:
   - `"I excel at problem solving"` → no Excel ✅
   - `"Advanced Excel and problem solving"` → Excel ✅
3. **Implied skills.** Listing Django credits you with Python; listing PySpark
   credits you with Spark; listing Express credits you with Node.js *and*
   JavaScript. Chains are followed to the end. Implied skills are labelled
   `implied` in the table so nothing is quietly inflated.

### `src/matcher.py`
The machine learning lives here. `analyze(resume_text, job_text)` returns one
`MatchResult` dataclass containing everything the app draws. Passing
`use_semantic=True` also attaches the AI similarity from `semantic_matcher.py`
as `result.semantic_similarity` - without ever feeding it into the match score.

### `src/semantic_matcher.py`
The deep-learning part, kept optional on purpose. It loads
[all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
once per process and turns each document into one vector of 384 numbers, so
that "built dashboards" and "created visual reports" land close together even
though they share no words. When `sentence-transformers` is not installed - or
the one-time download fails - every function returns `None` instead of
raising, and the app simply shows the word-based scores.

### `src/recommendations.py`
Turns gaps into ranked, actionable advice. Each gap gets a priority
(High / Medium / Low), a concrete next step, and a place to learn it - phrased
as a *platform plus a search phrase* rather than a hard-coded URL, because links
rot.

### `src/ai_assistant.py`
The chat assistant, powered by a local [Ollama](https://ollama.com) server
running the `llama3.2:3b` model. Three functions, no Streamlit imports, so the
whole thing is testable without a web server or even without Ollama installed:

- `check_ollama()` - one cheap GET to `/api/tags` that answers "is the server
  up, is the model pulled?".
- `build_context()` - packs the job advert, the resume (both clipped to fit the
  model's context window), the match score and the matched/missing skills with
  their importance into one text block. **This is everything the model is ever
  told**, which keeps answers grounded - and is the privacy boundary: the block
  goes to `http://localhost:11434` and nowhere else.
- `ask_assistant()` - POSTs the system prompt + context + chat history +
  question to `/api/chat` and returns the reply.

The system prompt has hard rules: ground everything in the context, **never
invent qualifications, experience or certifications**, recommend platforms and
search phrases instead of links. The request also asks for a low temperature
(`0.2`) to keep the model close to the provided facts.

Every failure has its own exception with a message written for a human:
`AssistantUnavailableError` (server down), `ModelNotInstalledError` (404 - run
`ollama pull llama3.2:3b`), `AssistantTimeoutError`, `EmptyMessageError` (blank
question) and `EmptyReplyError` (blank answer). `app.py` catches the common
base class and shows the message in the conversation, so a dead server is a
sentence in the chat, not a stack trace.

There are no secrets in this module on purpose: local Ollama has no
authentication, so there is no API key to hide. The server address comes from
the `OLLAMA_BASE_URL` environment variable (default `http://localhost:11434`).

---

## How the scoring works

### Match score (0-100)

```
match score = 100 x (0.70 x skill coverage + 0.30 x text similarity)
```

- **Skill coverage** = matched skills ÷ skills the job asks for. This is the
  number a recruiter actually cares about, so it carries most of the weight.
- **Text similarity** = TF-IDF cosine similarity of the two whole documents,
  computed with scikit-learn. It picks up related experience that is not in the
  skill dictionary.

If the job advert contains no recognisable skills at all, coverage is undefined,
so the score falls back to text similarity alone rather than reporting a
misleading zero.

### Skill importance (0.0-1.0)

This decides **which gap to fix first**. Three signals are combined:

| Signal | Weight | Why |
|---|---|---|
| **Section** | strongest | A skill under *Requirements* scores `1.0`, one under *Nice to have* scores `0.5`, one before any heading scores `0.8`. This is how a human reads an advert. |
| **TF-IDF** | 0.6 | The vectorizer is fitted on `[job advert, resume]`. A skill that appears **only** in the advert gets a higher inverse-document-frequency than one both documents share - the advert is the one insisting on it. |
| **Frequency** | 0.4 | A skill written three times matters more than one written once. |

The result is normalised so the most important skill is exactly `1.0`, which is
what makes the priority cut-offs in `recommendations.py` meaningful.

**Why section is the strongest signal:** term frequency alone cannot rank gaps.
Job adverts list each skill once, so every skill has frequency 1 and TF-IDF
alone gives them all nearly the same score. Detecting *which section* a skill
sits in is what actually separates a hard requirement from a nice-to-have.

Overlapping headings are avoided by choosing marker phrases carefully: no
optional phrase sits inside a required one, so `Preferred Qualifications` stays
optional instead of being hijacked by a shorter required-sounding word.

An implied skill inherits 80% of the importance of the tool that implies it, so
a job asking for Django ranks Python highly even though it never types "Python".

### Semantic similarity (the AI part)

A second, optional score answers a different question: not *which words do the
documents share* but *do they mean the same kind of work*.

```
document ──> all-MiniLM-L6-v2 ──> 384 numbers (a vector)
similarity = cosine of the angle between the resume's vector and the job's vector
```

The model is a small transformer neural network (~80 MB) that was trained on
millions of sentence pairs to place texts with similar meanings near each
other in a 384-dimensional space. The cosine of the angle between two vectors
then ranges from `1.0` (same meaning) through `0.0` (unrelated) to `-1.0`.

**The two similarity scores are deliberately never blended.** The match score
stays `70% coverage + 30% TF-IDF` with or without the AI, and the semantic
score is displayed beside it as `result.semantic_similarity`. Each number
tells its own half of the story, and you can always tell which is which.

Long documents are embedded in 180-word chunks whose vectors are averaged,
because the model only reads ~256 word-pieces at a time - without chunking it
would silently judge a resume by its first half-page.

**None of these numbers is a probability of getting hired.** They are
estimates of text overlap, from two very different estimation methods. See
*Limitations* below.

---

## Try it from a script

```python
from src.matcher import analyze
from src.recommendations import generate_recommendations, summarize

resume = "Python, pandas, SQL, Git, Linux, matplotlib, communication"
job = """Requirements: Python, pandas, scikit-learn, SQL, statistics,
machine learning, AWS, Docker. Nice to have: Airflow, dbt, Snowflake."""

result = analyze(resume, job, use_semantic=True)   # omit for word-based only
recommendations = generate_recommendations(result)

print(summarize(result, recommendations))
print(f"TF-IDF similarity: {result.text_similarity}")
print(f"Semantic similarity: {result.semantic_similarity}")  # None if no model
for item in recommendations:
    print(f"[{item.priority}] {item.skill}: {item.action}")
```

---

## Limitations (read this before trusting a score)

- **No score is a probability of getting hired.** They are estimates of how
  much two texts overlap. They cannot see your experience, how you interview,
  or whether the team would suit you. Use them to find gaps to work on, not
  to decide whether to apply.
- **The semantic model is small and generic.** all-MiniLM-L6-v2 was trained
  on ordinary sentence pairs - never on resumes, job adverts or hiring
  decisions. It knows what data-analysis *prose* tends to look like, not what
  a good candidate looks like. Its score is one rough opinion, which is why
  it is shown separately instead of being blended in.
- **Embeddings flatten documents.** Chunk-averaging gives every part of a
  long resume a vote, but the model still cannot reason about structure -
  it cannot tell "I built this" from "I watched a talk about this".
- **The chat assistant is a small model and can be wrong.** `llama3.2:3b` is a
  3-billion-parameter model - far smaller than the big commercial ones. It can
  misread the context or sound confident about something false, despite the
  prompt rules and the low temperature. Treat its answers as a first draft,
  and keep the deterministic gap table as the source of truth.
- **The chat is not stateless for free.** Ollama keeps no memory between
  requests, so the whole conversation is resent with every question. Long
  conversations get slower and eventually bump into the context-size limit.
- **The skill dictionary is finite.** Anything not in `SKILL_LEXICON` is
  invisible to the skill matcher. Text similarity partly compensates, but a
  niche skill will be missed. Adding it is a one-line change.
- **Capitalisation matters for ambiguous skills.** A job advert that writes
  `snowflake` in lower case will not be matched, because `snowflake schema` is
  an unrelated data-warehousing term. This trades a few misses for far fewer
  false positives.
- **No understanding of context.** "Removed the Docker dependency" still counts
  as knowing Docker. There is no sentiment or negation handling.
- **Two-column and graphical resumes parse badly.** PyMuPDF reads text in
  physical order, so a fancy layout can scramble the reading order.
- **Scanned PDFs are not supported.** There is no OCR. Export the resume as
  text, or paste it.
- **It is not a recruiter.** ATS keyword matching is a crude proxy for
  suitability. A low score means "your resume does not use these words", not
  "you cannot do this job".

---

## Ideas for extending it

- Add skills to `SKILL_LEXICON` / `SKILL_ALIASES` for your own field.
- Add OCR (Tesseract) so scanned resumes work.
- Let the user edit the detected skills before scoring, and see the score update.
- Compare one resume against many adverts and rank the jobs instead.
- Persist analyses to SQLite and track a score improving over time.
- Replace the hand-tuned priority thresholds with a model trained on real
  hiring outcomes.

---

## Built with

[Streamlit](https://streamlit.io) · [pandas](https://pandas.pydata.org) ·
[scikit-learn](https://scikit-learn.org) · [PyMuPDF](https://pymupdf.readthedocs.io) ·
[sentence-transformers](https://www.sbert.net) + [PyTorch](https://pytorch.org)
(semantic score) · [Ollama](https://ollama.com) + `llama3.2:3b` (chat assistant)

Tested on Python 3.14 with `python -m pytest tests/` (212 tests).
