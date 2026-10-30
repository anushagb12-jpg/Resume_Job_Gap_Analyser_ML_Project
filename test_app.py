"""UI smoke tests for app.py, using Streamlit's built-in AppTest runner.

AppTest executes the whole script in-process and lets us drive its widgets, so
these tests catch the mistakes unit tests cannot: a bad widget argument, a crash
when nothing has been entered yet, and a broken layout after the analysis runs.

They are slower than the other tests because each one boots the app.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src.ai_assistant import AssistantUnavailableError, OllamaStatus

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"

# Each test runs the script several times and re-does the analysis every time.
TIMEOUT_SECONDS = 180

JOB_WITH_GAPS = (
    "Requirements: Python, pandas, SQL, scikit-learn, AWS, Docker\n"
    "Nice to have: Airflow"
)
JOB_WITH_NO_SKILLS = "We are a friendly team who values curiosity and kindness."
RESUME = "Python and SQL analyst with Git and Linux experience."


def start_app() -> AppTest:
    """Load app.py and run it once, giving us the initial page."""
    return AppTest.from_file(str(APP_PATH), default_timeout=TIMEOUT_SECONDS).run()


def click_button(at: AppTest, label: str) -> AppTest:
    """Click the button with this label and return the resulting page."""
    for button in at.button:
        if button.label == label:
            return button.click().run()
    available = [button.label for button in at.button]
    raise AssertionError(f"no button labelled {label!r}; found {available}")


def metric_value(at: AppTest, label: str) -> str:
    """Read back the value of a metric tile."""
    for metric in at.metric:
        if metric.label == label:
            return str(metric.value)
    available = [metric.label for metric in at.metric]
    raise AssertionError(f"no metric labelled {label!r}; found {available}")


def submit(at: AppTest, resume: str | None, job: str | None) -> AppTest:
    """Type into the two text areas (if given) and press Analyze match."""
    if resume is not None:
        at.text_area[0].set_value(resume)
    if job is not None:
        at.text_area[1].set_value(job)
    return click_button(at, "Analyze match")


# ---------------------------------------------------------------------------
# Starting up
# ---------------------------------------------------------------------------


def test_app_starts_without_error():
    at = start_app()
    assert not at.exception
    assert "Resume-Job Gap Analyzer" in at.title[0].value


def test_welcome_message_is_shown_before_any_analysis():
    at = start_app()
    assert len(at.info) == 1
    assert "Analyze match" in at.info[0].value
    # Nothing has been analysed yet, so there are no metric tiles.
    assert len(at.metric) == 0


def test_expected_widgets_are_present():
    at = start_app()
    assert [toggle.label for toggle in at.toggle] == [
        "Use sample resume",
        "Add AI semantic similarity",
    ]
    labels = [button.label for button in at.button]
    assert "Analyze match" in labels
    assert "Fill in a sample job description" in labels


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_analyzing_with_no_input_explains_what_is_missing():
    at = submit(start_app(), resume=None, job=None)
    assert not at.exception
    assert any("resume is empty" in error.value for error in at.error)


def test_analyzing_without_a_job_description_explains_what_is_missing():
    at = submit(start_app(), resume=RESUME, job=None)
    assert not at.exception
    assert any("job description" in error.value for error in at.error)


def test_whitespace_only_input_is_treated_as_empty():
    at = submit(start_app(), resume="   \n  ", job=JOB_WITH_GAPS)
    assert any("resume is empty" in error.value for error in at.error)


def test_a_job_with_no_recognised_skills_warns_instead_of_failing():
    at = submit(start_app(), resume=RESUME, job=JOB_WITH_NO_SKILLS)
    assert not at.exception
    assert any("No recognised skills" in warning.value for warning in at.warning)


# ---------------------------------------------------------------------------
# A successful analysis
# ---------------------------------------------------------------------------


def test_analysis_renders_the_headline_metrics():
    at = submit(start_app(), resume=RESUME, job=JOB_WITH_GAPS)
    assert not at.exception
    assert not at.error

    score = metric_value(at, "Match score")
    assert score.endswith("/100")
    assert 0 <= int(score.split("/")[0]) <= 100
    assert int(metric_value(at, "Skills matched")) > 0
    assert int(metric_value(at, "Skills missing")) > 0


def test_analysis_creates_the_five_result_tabs():
    at = submit(start_app(), resume=RESUME, job=JOB_WITH_GAPS)
    assert [tab.label for tab in at.tabs] == [
        "Overview",
        "Skill gap table",
        "Recommendations",
        "What we read",
        "AI assistant",
    ]


def test_analysis_renders_the_gap_and_recommendation_tables():
    at = submit(start_app(), resume=RESUME, job=JOB_WITH_GAPS)
    # One table on the "Skill gap table" tab, one on "Recommendations".
    assert len(at.dataframe) == 2


def test_the_result_is_stored_in_session_state():
    at = submit(start_app(), resume=RESUME, job=JOB_WITH_GAPS)
    result = at.session_state["result"]
    assert result.missing_skills
    assert at.session_state["recommendations"]


def test_results_survive_a_rerun_without_re_analyzing():
    # Switching tabs reruns the script. The stored result must still be shown
    # even though Analyze match was not pressed again.
    at = submit(start_app(), resume=RESUME, job=JOB_WITH_GAPS)
    score_before = metric_value(at, "Match score")

    at = at.run()

    assert not at.exception
    assert metric_value(at, "Match score") == score_before


# ---------------------------------------------------------------------------
# Sample data helpers
# ---------------------------------------------------------------------------


def test_sample_resume_toggle_supplies_a_resume():
    at = start_app()
    at.toggle[0].set_value(True)
    at = at.run()
    assert any("Using the sample resume" in message.value for message in at.info)

    # With the toggle on, only the job description is still missing.
    at = click_button(at, "Analyze match")
    assert any("job description" in error.value for error in at.error)

    # Filling in the job advert is now enough to get a full analysis.
    at.text_area[1].set_value(JOB_WITH_GAPS)
    at = click_button(at, "Analyze match")
    assert not at.exception
    assert metric_value(at, "Match score").endswith("/100")


def test_fill_sample_job_button_fills_the_advert_box():
    at = click_button(start_app(), "Fill in a sample job description")
    assert not at.exception
    # The button's callback writes straight into the widget's state, so the
    # advert must actually be filled in on the next run.
    assert (at.text_area[1].value or "").startswith("Data Scientist")

    # Combined with the sample resume, that is enough for a full analysis.
    at.toggle[0].set_value(True)
    at = click_button(at, "Analyze match")
    assert not at.exception
    assert metric_value(at, "Match score").endswith("/100")


def test_pasted_text_is_used_as_the_resume():
    at = submit(
        start_app(),
        resume="Kubernetes and Terraform only",
        job="Requirements: Python and SQL",
    )
    assert not at.exception
    assert {"kubernetes", "terraform"} <= at.session_state["result"].resume_skills


def test_sample_toggle_takes_precedence_over_pasted_text():
    at = start_app()
    at.text_area[0].set_value("Kubernetes and Terraform only")
    at.toggle[0].set_value(True)
    at.text_area[1].set_value("Requirements: Python and SQL")
    at = click_button(at, "Analyze match")

    assert not at.exception
    result = at.session_state["result"]
    # The sample resume mentions Python; the pasted text does not. So this proves
    # the sample was used.
    assert "python" in result.resume_skills
    assert "kubernetes" not in result.resume_skills


@pytest.mark.parametrize(
    "job,expected_skill",
    [
        ("Requirements: Kubernetes", "kubernetes"),
        ("Requirements: scikit-learn", "scikit-learn"),
    ],
)
def test_gaps_are_reported_for_a_single_missing_skill(job, expected_skill):
    at = submit(start_app(), resume="Python only", job=job)
    assert expected_skill in at.session_state["result"].missing_skills


# ---------------------------------------------------------------------------
# Semantic similarity in the UI
# ---------------------------------------------------------------------------


def test_similarity_metrics_are_shown_separately(monkeypatch):
    # Patch the function matcher uses, so the app renders a deterministic
    # semantic score without needing the neural network.
    monkeypatch.setattr("src.matcher.compute_semantic_similarity", lambda r, j: 0.42)
    at = submit(start_app(), resume=RESUME, job=JOB_WITH_GAPS)

    assert not at.exception
    assert metric_value(at, "Semantic similarity (AI)") == "0.42"
    tfidf = metric_value(at, "Text similarity (TF-IDF)")
    assert 0.0 <= float(tfidf) <= 1.0


def test_semantic_unavailable_shows_a_message_and_n_a(monkeypatch):
    monkeypatch.setattr("src.semantic_matcher._get_model", lambda: None)
    at = submit(start_app(), resume=RESUME, job=JOB_WITH_GAPS)

    assert not at.exception
    assert any(
        "Semantic similarity is unavailable" in info.value for info in at.info
    )
    assert metric_value(at, "Semantic similarity (AI)") == "n/a"


def test_turning_semantic_off_leaves_the_field_empty():
    at = start_app()
    at.toggle[1].set_value(False)
    at.text_area[0].set_value(RESUME)
    at.text_area[1].set_value(JOB_WITH_GAPS)
    at = click_button(at, "Analyze match")

    assert not at.exception
    assert at.session_state["result"].semantic_similarity is None
    assert metric_value(at, "Semantic similarity (AI)") == "n/a"


def test_estimate_disclaimer_is_visible():
    at = submit(start_app(), resume=RESUME, job=JOB_WITH_GAPS)

    assert not at.exception
    # Markdown source wraps mid-phrase, so collapse all whitespace before
    # searching for the disclaimer sentences.
    texts = " ".join(markdown.value for markdown in at.markdown)
    texts = " ".join(texts.split())
    assert "not a probability of getting hired" in texts
    assert "kept separate" in texts


# ---------------------------------------------------------------------------
# Start over
# ---------------------------------------------------------------------------


def test_start_over_clears_the_analysis_and_every_input():
    at = submit(start_app(), resume=RESUME, job=JOB_WITH_GAPS)
    assert "result" in at.session_state
    assert metric_value(at, "Match score").endswith("/100")

    at = click_button(at, "Start over")

    assert not at.exception
    # The analysis is gone and we are back on the welcome screen.
    assert "result" not in at.session_state
    assert len(at.metric) == 0
    assert any("Analyze match" in info.value for info in at.info)
    # The inputs themselves were cleared, not just the results.
    assert (at.text_area[0].value or "").strip() == ""
    assert (at.text_area[1].value or "").strip() == ""
    assert at.toggle[0].value is False


def test_start_over_is_not_offered_before_any_analysis():
    at = start_app()
    assert not any(button.label == "Start over" for button in at.button)


# ---------------------------------------------------------------------------
# The AI assistant tab
#
# These tests never need Ollama: app.py binds check_ollama and ask_assistant
# when AppTest executes it, so patching the names on src.ai_assistant before
# the first run swaps in fakes for the whole test.
# ---------------------------------------------------------------------------


def test_ai_tab_explains_how_to_start_ollama_when_it_is_down(monkeypatch):
    monkeypatch.setattr(
        "src.ai_assistant.check_ollama",
        lambda: OllamaStatus(server_running=False, model_installed=False),
    )
    at = submit(start_app(), resume=RESUME, job=JOB_WITH_GAPS)

    assert not at.exception
    # The tab degrades into an explanation instead of a crash, and the rest
    # of the page is unaffected.
    assert any("Ollama" in warning.value for warning in at.warning)
    assert metric_value(at, "Match score").endswith("/100")


def test_ai_tab_names_the_model_to_pull_when_it_is_missing(monkeypatch):
    monkeypatch.setattr(
        "src.ai_assistant.check_ollama",
        lambda: OllamaStatus(server_running=True, model_installed=False),
    )
    at = submit(start_app(), resume=RESUME, job=JOB_WITH_GAPS)

    assert not at.exception
    assert any("ollama pull llama3.2:3b" in warning.value for warning in at.warning)


def test_quick_question_gets_an_answer_from_the_assistant(monkeypatch):
    monkeypatch.setattr(
        "src.ai_assistant.check_ollama",
        lambda: OllamaStatus(server_running=True, model_installed=True),
    )
    seen: dict[str, str] = {}

    def fake_ask(question, context, history=None):
        seen["question"] = question
        seen["context"] = context
        return "Learn Docker first."

    monkeypatch.setattr("src.ai_assistant.ask_assistant", fake_ask)

    at = submit(start_app(), resume=RESUME, job=JOB_WITH_GAPS)
    at = click_button(at, "Create a 30-day study plan")

    assert not at.exception
    # The canned reply is drawn in the conversation...
    texts = " ".join(markdown.value for markdown in at.markdown)
    assert "Learn Docker first." in texts
    # ...and the question went out with the analysis attached as context.
    assert "python" in seen["context"].lower()
    assert "docker" in seen["context"].lower()
    chat = at.session_state["chat_history"]
    assert chat[0]["role"] == "user"
    assert chat[-1]["content"] == "Learn Docker first."


def test_assistant_failures_stay_in_the_conversation(monkeypatch):
    monkeypatch.setattr(
        "src.ai_assistant.check_ollama",
        lambda: OllamaStatus(server_running=True, model_installed=True),
    )

    def failing_ask(question, context, history=None):
        raise AssistantUnavailableError("Could not reach Ollama.")

    monkeypatch.setattr("src.ai_assistant.ask_assistant", failing_ask)

    at = submit(start_app(), resume=RESUME, job=JOB_WITH_GAPS)
    at = click_button(at, "Explain my skill gaps")

    # A failing model is a message in the chat, not an app crash.
    assert not at.exception
    texts = " ".join(markdown.value for markdown in at.markdown)
    assert "could not answer" in texts
    assert "Could not reach Ollama." in texts


def test_start_over_clears_the_chat_history(monkeypatch):
    monkeypatch.setattr(
        "src.ai_assistant.check_ollama",
        lambda: OllamaStatus(server_running=True, model_installed=True),
    )
    monkeypatch.setattr(
        "src.ai_assistant.ask_assistant", lambda q, c, history=None: "Short answer."
    )

    at = submit(start_app(), resume=RESUME, job=JOB_WITH_GAPS)
    at = click_button(at, "Explain my skill gaps")
    assert at.session_state["chat_history"]

    at = click_button(at, "Start over")
    assert "chat_history" not in at.session_state
