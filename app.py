"""Resume-Job Gap Analyzer - the Streamlit user interface.

Run it with:

    streamlit run app.py

This file is deliberately thin. It only collects input, calls the functions in
`src/`, and draws the results. All the logic lives in the modules, which is why
they can be tested without starting a web server.
"""

from __future__ import annotations

import streamlit as st

from src.ai_assistant import (
    MODEL_NAME,
    AssistantError,
    ask_assistant,
    build_context,
    check_ollama,
)
from src.matcher import MatchResult, analyze
from src.recommendations import (
    generate_recommendations,
    highlight_strengths,
    summarize,
    to_dataframe,
    verdict_for,
)
from src.resume_parser import ResumeParseError, extract_text_from_bytes
from src.skill_extractor import categorize_skills, list_known_skills
from src.text_cleaner import describe_text

st.set_page_config(
    page_title="Resume-Job Gap Analyzer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

MAX_UPLOAD_MB = 10

# Sample input so anyone can see the tool working before uploading their own
# resume. Also used by the automated UI test.
SAMPLE_RESUME = """Alex Morgan
Data Analyst | alex.morgan@example.com | github.com/alexmorgan

SKILLS
Python, pandas, NumPy, SQL, Excel, matplotlib, Tableau, Git, Linux

EXPERIENCE
Data Analyst, Retail Co (2022 - present)
- Built dashboards in Tableau used weekly by the marketing team
- Wrote SQL queries against PostgreSQL to track campaign performance
- Automated reporting with Python and pandas, saving 6 hours a week

Junior Analyst, Startup Inc (2020 - 2022)
- Cleaned and analysed survey data, presented findings to stakeholders
- Strong teamwork and communication in a cross functional squad
"""

SAMPLE_JOB = """Data Scientist

Requirements:
- Python, pandas, NumPy and scikit-learn
- SQL and statistics
- Machine learning, A/B testing and feature engineering
- AWS and Docker
- Excellent communication and problem solving skills

Nice to have:
- Airflow, dbt and Snowflake
- Deep learning with TensorFlow
- Kubernetes

You must have 3+ years of experience turning messy data into decisions.
"""

# Ready-made questions for the AI assistant tab. Each pair is (button label,
# question sent to the local model); the labels stay short so the three
# buttons fit in one wrapping row.
QUICK_QUESTIONS = [
    (
        "Explain my skill gaps",
        "Explain my main skill gaps for this job. For each one, briefly say "
        "why it matters for this specific advert.",
    ),
    (
        "Recommend learning resources",
        "Recommend learning resources for my missing skills, grouped by "
        "topic. Name the platform and what to search for rather than links.",
    ),
    (
        "Create a 30-day study plan",
        "Create a personalized 30-day study plan for my missing skills, most "
        "important first. Keep the daily time realistic and suggest one small "
        "practice project per week.",
    ),
]


# ---------------------------------------------------------------------------
# Reading the resume
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Reading your resume...")
def read_resume(file_bytes: bytes, filename: str) -> str:
    """Extract text from uploaded bytes.

    Cached because Streamlit reruns this whole script on every interaction, and
    re-parsing a PDF each time would be wasteful. The cache key is the file's
    bytes plus its name.
    """
    return extract_text_from_bytes(file_bytes, filename)


# ---------------------------------------------------------------------------
# Sidebar: collect the two inputs
# ---------------------------------------------------------------------------
def render_sidebar() -> tuple[bool, str | None, str | None, bool]:
    """Draw the sidebar.

    Returns (analyze_pressed, resume_text, job_text, use_semantic). Either
    text may be None if the user has not supplied it yet - main() decides
    what to do about that.

    Every widget key ends in a "generation" number that Start over
    increases, because that is the one reliable way to clear inputs in the
    browser: the front end keeps a widget's on-screen state for as long as
    that widget exists, so deleting the session-state key alone leaves the
    old text visibly in the box. A new key means a brand-new widget, which
    honestly starts empty.
    """
    generation = st.session_state.get("reset_generation", 0)

    with st.sidebar:
        st.header("1. Your resume")

        # All three inputs stay on screen at all times. Hiding them would make
        # the precedence rules harder to see, and (in testing) would shuffle the
        # widget order every time the toggle changed.
        use_sample = st.toggle(
            "Use sample resume",
            value=False,
            key=f"use_sample_{generation}",
            help="Ignore the inputs below and try the tool with example data.",
        )

        uploaded = st.file_uploader(
            "Upload a PDF or text file",
            type=["pdf", "txt", "md"],
            key=f"resume_upload_{generation}",
            help=f"Maximum size {MAX_UPLOAD_MB} MB.",
        )

        resume_text: str | None = None
        if uploaded is not None and uploaded.size > MAX_UPLOAD_MB * 1024 * 1024:
            st.error(f"That file is larger than {MAX_UPLOAD_MB} MB.")
        elif uploaded is not None:
            try:
                resume_text = read_resume(uploaded.getvalue(), uploaded.name)
                st.success(f"Read {uploaded.name}.")
            except ResumeParseError as error:
                # Raised by our own parser for problems the user can fix.
                st.error(str(error))

        pasted = st.text_area(
            "...or paste your resume text here",
            height=140,
            key=f"resume_paste_{generation}",
            placeholder="Paste the contents of your resume if you do not have a file.",
        )
        if pasted.strip():
            # Pasted text wins over an uploaded file, because it is the most
            # recent thing the user did.
            resume_text = pasted

        if use_sample:
            # The toggle wins over both, because switching it on is a deliberate
            # "show me an example" action.
            resume_text = SAMPLE_RESUME
            st.info("Using the sample resume.")

        st.header("2. The job description")
        job_text = st.text_area(
            "Paste the job advert",
            height=260,
            key=f"job_advert_{generation}",
            placeholder="Copy the whole job description, including the requirements section.",
            label_visibility="collapsed",
        )

        def prefill_job_advert() -> None:
            """Write the sample advert into the box (runs before the rerun).

            Setting a widget's key from a callback is the supported way to
            change its value programmatically. Doing it in the script body
            would be refused, because the box already exists by then.
            """
            current_generation = st.session_state.get("reset_generation", 0)
            st.session_state[f"job_advert_{current_generation}"] = SAMPLE_JOB

        st.button("Fill in a sample job description", on_click=prefill_job_advert)

        st.divider()
        use_semantic = st.toggle(
            "Add AI semantic similarity",
            value=True,
            key=f"use_semantic_{generation}",
            help=(
                "Also score how close the two texts are in MEANING with a small "
                "neural network (all-MiniLM-L6-v2). The model (~90 MB) is "
                "downloaded once, then runs locally. Shown separately - it "
                "never changes the match score."
            ),
        )
        analyze_clicked = st.button(
            "Analyze match",
            type="primary",
            width="stretch",
            icon=":material/insights:",
        )

        with st.expander("How is the score calculated?"):
            st.markdown(
                """
**Match score (0-100)**

`70% x skill coverage + 30% x text similarity`

- *Skill coverage* is the fraction of the job's skills your resume shows.
- *Text similarity* is the TF-IDF cosine similarity of the two documents. It
  picks up related experience that is not in the skill dictionary.

**Skill importance (0-1)** decides which gap to fix first. It is mostly about
*where* in the advert a skill appears: "Requirements" scores highest, the
advert's title scores in the middle, and "Nice to have" scores lowest. TF-IDF
weight and how often the skill is repeated act as tie-breakers.

Skills can also be **implied**: listing Django credits you with Python, and
listing PySpark credits you with Spark. Implied rows are labelled in the table.

**Semantic similarity** (the toggle above) is a separate neural-network score
of how close the two texts are in *meaning*. It is deliberately not part of
the match score - the two numbers are always shown side by side.

Every score here estimates how much two texts overlap. None of them is a
probability of getting hired.

Recognised skills: """
                + str(len(list_known_skills()))
            )

    if not analyze_clicked:
        return False, None, None, use_semantic
    return True, resume_text, job_text, use_semantic


# ---------------------------------------------------------------------------
# Main panel: draw the results
# ---------------------------------------------------------------------------
def render_similarity_scores(result: MatchResult) -> None:
    """The two similarity measures, side by side but never blended."""
    with st.container(horizontal=True):
        st.metric(
            "Text similarity (TF-IDF)",
            f"{result.text_similarity:.2f}",
            border=True,
            help=(
                "Word overlap between the resume and the advert, from 0.00 to "
                "1.00. Catches related experience even outside the skill "
                "dictionary."
            ),
        )
        if result.semantic_similarity is None:
            st.metric(
                "Semantic similarity (AI)",
                "n/a",
                border=True,
                help=(
                    "Turn on 'Add AI semantic similarity' in the sidebar to "
                    "compute this score."
                ),
            )
        else:
            st.metric(
                "Semantic similarity (AI)",
                f"{result.semantic_similarity:.2f}",
                border=True,
                help=(
                    "The neural network's opinion of how close the two texts "
                    "are in MEANING: 1.00 = same meaning, 0.00 = unrelated, "
                    "negative = very different."
                ),
            )

    with st.expander("How should I read these scores?"):
        st.markdown(
            """
**All of these numbers are rough estimates - not a probability of getting
hired.** They measure how much two pieces of text overlap. They cannot see
your experience, how you interview, or whether you would enjoy the team.

- **Match score (0-100)** = `70% skill coverage + 30% TF-IDF similarity`.
  Transparent and simple, but only as good as the skill dictionary.
- **TF-IDF similarity** compares *which words* the documents share.
- **Semantic similarity** is a small neural network's opinion of how close
  the documents are in *meaning*. It notices "built dashboards" ~
  "created visual reports" even with no words in common, but it is a
  general-purpose model (all-MiniLM-L6-v2, 384 numbers per text) that was
  never trained on resumes or hiring, and it reads long documents as
  averaged chunks.

The two similarity scores are deliberately **kept separate** from the match
score, so each one tells its own half of the story. A high similarity does
not mean you will be hired, and a low one does not mean you will not be -
recruiters read the whole person, not a cosine.
"""
        )


def render_headline(result: MatchResult) -> None:
    """The four big numbers at the top of the page."""
    matched, missing, extra = (
        len(result.matched_skills),
        len(result.missing_skills),
        len(result.extra_skills),
    )
    # A wrapping row (not fixed columns) keeps the tiles usable on narrow
    # screens; the borders make the group read as one card row.
    with st.container(horizontal=True):
        st.metric(
            "Match score",
            f"{result.match_score}/100",
            verdict_for(result.match_score),
            border=True,
        )
        st.metric("Skills matched", matched, border=True)
        st.metric(
            "Skills missing", missing, delta=-missing if missing else None, border=True
        )
        st.metric(
            "Bonus skills",
            extra,
            border=True,
            help="Skills your resume shows that this advert did not ask for.",
        )

    st.progress(result.match_score / 100, text=f"{result.match_score}% match")
    render_similarity_scores(result)


def render_overview(result: MatchResult, recommendations) -> None:
    """Summary text, strengths, and a breakdown of the gaps by category."""
    st.subheader("Summary")
    st.text(summarize(result, recommendations))

    strengths = highlight_strengths(result)
    if strengths:
        st.subheader("Your strongest matches")
        st.markdown("  ".join(f"`{skill}`" for skill in strengths))

    gaps = result.gap_table[result.gap_table["Status"] == "Gap"]
    if not gaps.empty:
        st.subheader("Gaps by category")
        st.bar_chart(
            gaps["Category"].value_counts(),
            alt="Number of missing skills in each category",
        )

    matched_by_category = categorize_skills(result.matched_skills)
    if matched_by_category:
        with st.expander(f"Matched skills by category ({len(result.matched_skills)})"):
            for category, skills in matched_by_category.items():
                st.markdown(f"**{category}**")
                st.markdown("  ".join(f"`{skill}`" for skill in skills))

    if result.extra_skills:
        with st.expander(f"Skills this job did not ask for ({len(result.extra_skills)})"):
            st.caption(
                "Not a problem. These are worth keeping, and may matter for a "
                "different role."
            )
            for category, skills in categorize_skills(result.extra_skills).items():
                st.markdown(f"**{category}**")
                st.markdown("  ".join(f"`{skill}`" for skill in skills))


def render_recommendations(recommendations) -> None:
    """The ranked to-do list, as a table plus one card per item."""
    if not recommendations:
        st.info(
            "No gaps found - every skill in this job advert already appears in "
            "the resume."
        )
        return

    st.dataframe(
        to_dataframe(recommendations),
        width="stretch",
        hide_index=True,
        column_config={
            "Importance": st.column_config.ProgressColumn(
                "Importance", min_value=0.0, max_value=1.0, format="%.2f"
            )
        },
    )

    for item in recommendations:
        icon = {"High": "🔴", "Medium": "🟠", "Low": "🟢"}[item.priority]
        asked = "asked for directly" if item.listed_in_ad else "implied by another tool"
        with st.container(border=True):
            st.markdown(f"{icon} **{item.skill}** - {item.priority} priority")
            st.caption(f"{item.category} | importance {item.importance:.2f} | {asked}")
            st.markdown(f"**Do this:** {item.action}")
            st.markdown(f"**Where to learn:** {item.where_to_learn}")


def render_extracted_text(resume_text: str, result: MatchResult) -> None:
    """Show what the program actually read, so the score can be sanity-checked."""
    stats = describe_text(resume_text)
    with st.container(horizontal=True):
        for label, value in [
            ("Words", stats["words"]),
            ("Unique words", stats["unique_words"]),
            ("Lines", stats["lines"]),
            ("Characters", stats["characters"]),
        ]:
            st.metric(label, value, border=True)

    st.subheader("Skills found in the resume")
    resume_by_category = categorize_skills(result.resume_skills)
    if resume_by_category:
        for category, skills in resume_by_category.items():
            st.markdown(f"**{category}**")
            st.markdown("  ".join(f"`{skill}`" for skill in skills))
    else:
        st.warning("No recognised skills found. The resume may be very short.")

    with st.expander("Raw text read from the resume"):
        st.text(resume_text)


def render_ai_assistant(
    result: MatchResult, resume_text: str, job_text: str
) -> None:
    """The chat tab: a local Ollama model answers questions about this analysis.

    The conversation lives in session_state so it survives reruns, and it dies
    with Start over (or a new analysis) like everything else about this
    result. If Ollama is missing we stop after a message that explains how to
    install it, and the rest of the app is unaffected.
    """
    st.subheader("Ask the AI assistant")
    st.caption(
        f"{MODEL_NAME} running locally through Ollama. The app only ever "
        "talks to this machine - your resume is not sent to any external "
        "AI service."
    )

    status = check_ollama()
    if not status.server_running:
        st.warning(
            "The AI assistant is optional and needs Ollama, which is not "
            "running right now. Everything else on this page works without "
            "it.\n\n"
            "1. Install Ollama from https://ollama.com/download\n"
            "2. Start it - on Windows and macOS it runs automatically after "
            "installing; on Linux run `ollama serve`.\n\n"
            "Once it is running, reload this page or press Analyze match "
            "again."
        )
        return
    if not status.model_installed:
        st.warning(
            f"Ollama is running, but the `{MODEL_NAME}` model is not "
            "installed yet. Run this in a terminal, then reload the page:\n\n"
            f"```\nollama pull {MODEL_NAME}\n```\n\n"
            "(About 2 GB, downloaded once.)"
        )
        return

    st.caption(
        "Replies can take up to a minute on slow hardware. Treat them as "
        "advice from a small model, not as fact."
    )
    st.session_state.setdefault("chat_history", [])
    history = st.session_state["chat_history"]

    def respond(question: str) -> None:
        """Ask the model one question about this analysis and store the reply."""
        history.append({"role": "user", "content": question})
        context = build_context(resume_text, job_text, result)
        try:
            with st.spinner("Thinking... (the model runs on this machine)"):
                reply = ask_assistant(question, context, history=history[:-1])
        except AssistantError as error:
            # Keep the failure in the conversation so the user sees what
            # happened instead of the question quietly vanishing.
            reply = f"**Sorry - I could not answer that.** {error}"
        history.append({"role": "assistant", "content": reply})

    st.markdown("**Try one of these:**")
    with st.container(horizontal=True):
        for label, prompt in QUICK_QUESTIONS:
            if st.button(label):
                respond(prompt)
                # The history loop below has already run for this pass, so
                # rerun to have it draw the new exchange.
                st.rerun()

    for message in history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input(
        "Ask about your skill gaps, learning resources or a study plan..."
    )
    if question and question.strip():
        respond(question)
        st.rerun()


def render_results(
    result: MatchResult, resume_text: str, job_text: str, recommendations
) -> None:
    """Draw the whole results area using tabs."""
    tab_overview, tab_table, tab_advice, tab_text, tab_chat = st.tabs(
        ["Overview", "Skill gap table", "Recommendations", "What we read", "AI assistant"]
    )
    with tab_overview:
        render_overview(result, recommendations)
    with tab_table:
        if result.gap_table.empty:
            st.warning("No skills were recognised in the job description.")
        else:
            st.dataframe(
                result.gap_table,
                width="stretch",
                hide_index=True,
                column_config={
                    "Importance": st.column_config.ProgressColumn(
                        "Importance", min_value=0.0, max_value=1.0, format="%.2f"
                    )
                },
            )
            st.caption(
                "Sorted with the most important gaps first. "
                "`listed` means the advert named the skill; `implied` means it "
                "follows from another tool the advert named."
            )
    with tab_advice:
        render_recommendations(recommendations)
    with tab_text:
        render_extracted_text(resume_text, result)
    with tab_chat:
        render_ai_assistant(result, resume_text, job_text)


def render_welcome() -> None:
    """Shown before the first analysis."""
    st.info(
        "Upload a resume (or paste the text), paste a job description in the "
        "sidebar, then press **Analyze match**.\n\n"
        "In a hurry? Switch on **Use sample resume** and press "
        "**Fill in a sample job description** to try it instantly."
    )
    st.subheader("What this tool does")
    st.markdown(
        """
1. **Reads** your resume - PDF text is pulled out with PyMuPDF.
2. **Cleans** the text - contact details, bullets and stray whitespace are removed.
3. **Finds skills** in both your resume and the job advert using a dictionary of
   known skills and their aliases.
4. **Scores the match** and ranks the gaps by how much the employer seems to care.
5. **Optionally adds an AI similarity score** - a small neural network judges how
   close the two texts are in meaning. It runs locally and is shown separately
   from the match score.
6. **Suggests what to learn first**, with a concrete next step for each gap.
7. **Chats about the results** (optional) - a local AI model served by Ollama
   can explain the gaps, recommend resources and draft a 30-day study plan.

Everything happens on your own machine. No resume is uploaded anywhere.
"""
    )


def main() -> None:
    """Entry point: collect input, analyze it, render the results."""
    st.title("🎯 Resume-Job Gap Analyzer")
    st.caption(
        "Compare your resume with a job advert to see which skills you already "
        "have, which are missing, and what to learn first."
    )

    analyze_pressed, resume_text, job_text, use_semantic = render_sidebar()

    if analyze_pressed:
        if not (resume_text or "").strip():
            st.error("The resume is empty. Upload a file or paste some text.")
        elif not (job_text or "").strip():
            st.error("Please paste a job description in the sidebar.")
        else:
            if use_semantic:
                with st.spinner(
                    "Analyzing... (the AI model is downloaded once, about 90 MB)"
                ):
                    result = analyze(resume_text, job_text, use_semantic=True)
            else:
                result = analyze(resume_text, job_text)
            if not result.job_skills:
                st.warning(
                    "No recognised skills were found in that job description. "
                    "Paste the full advert, including the requirements section."
                )
            if use_semantic and result.semantic_similarity is None:
                st.info(
                    "Semantic similarity is unavailable - the sentence-transformers "
                    "package is not installed, or the model could not be "
                    "downloaded. The word-based analysis below is unaffected. "
                    "Install it with `pip install sentence-transformers` and "
                    "press Analyze again."
                )
            st.session_state["result"] = result
            st.session_state["resume_text"] = resume_text
            st.session_state["job_text"] = job_text
            st.session_state["recommendations"] = generate_recommendations(result)
            # A fresh analysis means the old conversation is about the wrong
            # job now, so it starts over too.
            st.session_state["chat_history"] = []

    # Streamlit reruns the whole script on every interaction (switching tabs,
    # opening an expander). Keeping the last result in session_state means those
    # interactions do not force the analysis to run again.
    if "result" in st.session_state:
        # Offered only once there is something to undo. Rendering it here
        # (not at the top of the script) means it also appears on the same
        # run that produces a first result.
        with st.container(horizontal=True, horizontal_alignment="right"):
            if st.button(
                "Start over",
                icon=":material/refresh:",
                help="Clear the resume, the job description and the last analysis.",
            ):
                # Wipe the analysis AND every input's widget state, then give
                # all inputs a fresh widget identity by bumping the
                # generation number (see render_sidebar for why that is
                # needed). The generation is captured before clear() so it
                # keeps growing and never reuses an identity.
                generation = st.session_state.get("reset_generation", 0) + 1
                st.session_state.clear()
                st.session_state["reset_generation"] = generation
                st.rerun()
        render_headline(st.session_state["result"])
        render_results(
            st.session_state["result"],
            st.session_state["resume_text"],
            st.session_state["job_text"],
            st.session_state["recommendations"],
        )
    else:
        render_welcome()


main()
