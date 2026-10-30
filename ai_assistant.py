"""A chat assistant that runs entirely on the user's own machine.

The assistant is a large language model served by Ollama, a program that
downloads and runs models locally (https://ollama.com). This module talks to
Ollama's HTTP API and to nothing else, so a resume never leaves the computer -
the same privacy promise as the rest of this project.

How the pieces fit together
---------------------------
* check_ollama()  - is the Ollama server running, and is our model installed?
                    One cheap local HTTP GET, no model is loaded.
* build_context() - turn an analysis (resume text, job advert, matched and
                    missing skills) into one text block the model can read.
* ask_assistant() - send the system prompt + context + chat history + question
                    to Ollama's /api/chat endpoint and return the reply.

Every way this can fail has its own exception with a message written for a
human, because "connection refused" should become "Ollama is not running -
here is how to start it", not a stack trace.

There are no API keys or passwords in this file on purpose: Ollama is a local
server with no authentication, so there is no secret that could leak. The
server address can be overridden with the OLLAMA_BASE_URL environment
variable, for example if Ollama listens on another port.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import requests

if TYPE_CHECKING:
    # Only the type annotation is needed; importing at runtime would pull the
    # whole scikit-learn stack into every user of this module.
    from .matcher import MatchResult

MODEL_NAME = "llama3.2:3b"
DEFAULT_BASE_URL = "http://localhost:11434"

# Seconds before giving up on reaching the server. A refused local connection
# is instant, so this only bites when something is badly wrong.
CONNECT_TIMEOUT = 3
# Seconds to wait for the model to finish writing. A 3-billion-parameter model
# on a CPU can take over a minute for a long study plan.
READ_TIMEOUT = 180

# The two texts are clipped so the prompt fits the context size we request
# below (NUM_CTX tokens) with room to spare for the chat history.
MAX_RESUME_CHARS = 6000
MAX_JOB_CHARS = 3000

# A low temperature keeps the model close to the provided facts, which matters
# here: it must not dream up qualifications the candidate does not have.
TEMPERATURE = 0.2
# Ollama's default context window is small (2-4k tokens); a resume plus a job
# advert plus a conversation needs more, so we ask for 8k explicitly.
NUM_CTX = 8192

SYSTEM_PROMPT = """You are a practical, honest study-planning assistant inside a
resume analysis tool. The candidate has compared their resume with a job
advert. All the facts you may rely on are in the CONTEXT block; treat them as
ground truth.

Your job:
- Explain skill gaps in plain language, and say why each gap matters for this
  particular job.
- Recommend learning resources grouped by topic. Name real, well-known
  platforms and a search phrase (for example "pandas course on Kaggle Learn")
  instead of URLs, because you cannot check that a link works.
- When asked for a study plan, build a 30-day plan that starts with the most
  important missing skills, keeps the daily time realistic, and includes one
  small practice project per week that combines a new skill with a skill the
  candidate already has.

Hard rules:
- Ground every statement in the CONTEXT. If the context does not say it, say
  you do not know instead of guessing.
- NEVER invent qualifications, work experience, employment history or
  certifications the candidate does not have, and never imply that following
  your plan guarantees an interview or a job.
- Do not invent statistics, salaries or hiring rules.
- Be concise and skimmable: short paragraphs and lists, no long preambles.
"""


class AssistantError(Exception):
    """Base class: something stopped the assistant from replying."""


class AssistantUnavailableError(AssistantError):
    """Ollama is not running at the expected address."""


class ModelNotInstalledError(AssistantError):
    """Ollama is running but does not have our model yet."""


class AssistantTimeoutError(AssistantError):
    """Ollama accepted the request but did not answer in time."""


class EmptyMessageError(AssistantError):
    """The question sent to the assistant was blank."""


class EmptyReplyError(AssistantError):
    """The model answered with an empty string."""


@dataclass
class OllamaStatus:
    """What check_ollama() found: is the server up, is the model there?"""

    server_running: bool
    model_installed: bool
    detail: str = ""


def _base_url() -> str:
    """The Ollama server address, without a trailing slash.

    Read from the OLLAMA_BASE_URL environment variable so the address can be
    changed without editing code. The default is where Ollama listens out of
    the box.
    """
    return os.environ.get("OLLAMA_BASE_URL", DEFAULT_BASE_URL).rstrip("/")


def check_ollama() -> OllamaStatus:
    """Ask Ollama which models it has (GET /api/tags).

    This is the cheap "is the assistant usable?" check the app runs before
    showing the chat box. No model is loaded, so it answers in milliseconds
    even while Ollama is busy generating something else.
    """
    try:
        response = requests.get(f"{_base_url()}/api/tags", timeout=CONNECT_TIMEOUT)
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        return OllamaStatus(
            server_running=False,
            model_installed=False,
            detail="Ollama did not answer",
        )
    if response.status_code != 200:
        return OllamaStatus(
            server_running=False,
            model_installed=False,
            detail=f"Ollama answered with HTTP {response.status_code}",
        )
    try:
        models = response.json().get("models", [])
    except ValueError:
        return OllamaStatus(
            server_running=False,
            model_installed=False,
            detail="Ollama did not return JSON",
        )
    names = {model.get("name", "") for model in models}
    installed = MODEL_NAME in names
    return OllamaStatus(
        server_running=True,
        model_installed=installed,
        detail="" if installed else f"{MODEL_NAME} is not downloaded yet",
    )


def _clip(text: str, limit: int) -> str:
    """Cut text at a word boundary and say that we cut it."""
    if len(text) <= limit:
        return text
    # Drop the half-finished word at the cut so the model does not read
    # something like "dock" and think it means Docker.
    cut = text[:limit].rsplit(maxsplit=1)[0]
    return f"{cut}\n[... truncated for length ...]"


def build_context(resume_text: str, job_text: str, result: MatchResult) -> str:
    """Turn one analysis into the text block the model is allowed to know.

    Everything the assistant may rely on goes in here. That is what keeps its
    answers grounded, and it makes the privacy promise concrete: the resume is
    placed in a prompt that is sent to the local Ollama server and nowhere
    else.

    Missing skills are listed with their importance, most important first, so
    the model's study plan starts where the job advert hurts most.
    """
    missing = sorted(
        result.missing_skills,
        key=lambda skill: (-result.skill_importance.get(skill, 0.0), skill),
    )
    missing_lines = "\n".join(
        f"- {skill} (importance {result.skill_importance.get(skill, 0.0):.2f})"
        for skill in missing
    )
    if not missing_lines:
        missing_lines = "- none"
    matched = ", ".join(result.matched_skills) or "none"
    extra = ", ".join(result.extra_skills) or "none"

    return (
        "CONTEXT\n"
        "=======\n\n"
        "JOB ADVERT (what the candidate is applying for):\n"
        f"{_clip(job_text, MAX_JOB_CHARS)}\n\n"
        "CANDIDATE'S RESUME:\n"
        f"{_clip(resume_text, MAX_RESUME_CHARS)}\n\n"
        "ANALYSIS FACTS (computed by this tool - treat as ground truth):\n"
        f"- Match score: {result.match_score}/100\n"
        f"- Matched skills: {matched}\n"
        "- Missing skills, most important first (importance is 0.0-1.0; the "
        "higher it is, the more the advert insists on the skill):\n"
        f"{missing_lines}\n"
        f"- Extra skills the resume has but this advert did not ask for: {extra}\n"
    )


def ask_assistant(
    question: str,
    context: str,
    history: list[dict[str, str]] | None = None,
) -> str:
    """Send one question to the local model and return its reply.

    history is the conversation so far, as {"role": "user"|"assistant",
    "content": ...} dicts, so the model can refer to its own earlier answers.
    The system message (rules plus context) is resent on every call because
    Ollama keeps no memory between requests.

    Raises one of the AssistantError subclasses above, each carrying a message
    the app can show to the user as it is.
    """
    if not question or not question.strip():
        raise EmptyMessageError("The message is empty - type a question first.")

    messages: list[dict[str, str]] = [
        {"role": "system", "content": f"{SYSTEM_PROMPT}\n\n{context}"}
    ]
    for turn in history or []:
        role = turn.get("role")
        if role not in ("user", "assistant"):
            raise ValueError(f"bad role in chat history: {role!r}")
        messages.append({"role": role, "content": turn["content"]})
    messages.append({"role": "user", "content": question})

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        # One complete JSON reply is easier to handle (and to test) than a
        # stream of fragments.
        "stream": False,
        "options": {"temperature": TEMPERATURE, "num_ctx": NUM_CTX},
    }

    try:
        response = requests.post(
            f"{_base_url()}/api/chat",
            json=payload,
            timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
        )
    except requests.exceptions.ConnectionError as error:
        raise AssistantUnavailableError(
            "Could not reach Ollama. Is it running? Start the Ollama app or "
            "run `ollama serve`, then ask again."
        ) from error
    except requests.exceptions.Timeout as error:
        raise AssistantTimeoutError(
            "Ollama did not answer in time. The model may still be loading - "
            "wait a minute and try again."
        ) from error
    except requests.exceptions.RequestException as error:
        raise AssistantError(f"The request to Ollama failed: {error}") from error

    if response.status_code == 404:
        # Ollama answers 404 with "model not found, try pulling it first".
        raise ModelNotInstalledError(
            f"The model {MODEL_NAME} is not installed. Run `ollama pull "
            f"{MODEL_NAME}` in a terminal (about 2 GB, downloaded once), then "
            "ask again."
        )
    if response.status_code != 200:
        raise AssistantError(f"Ollama answered with HTTP {response.status_code}.")

    try:
        data = response.json()
    except ValueError as error:
        raise AssistantError("Ollama did not return JSON.") from error
    if not isinstance(data, dict):
        raise AssistantError("Ollama returned an unexpected reply format.")

    reply = (data.get("message") or {}).get("content", "")
    if not reply.strip():
        raise EmptyReplyError("The model returned an empty answer. Try rephrasing.")

    return reply.strip()
