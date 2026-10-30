"""Find known skills inside a piece of text.

The approach is deliberately simple and transparent: we keep a hand-written
dictionary of skills (SKILL_LEXICON), turn each one into a regular expression,
and search the text for it. Nothing is guessed - if a skill is reported, the
word really was in the document.

Three small ideas make this work well on real resumes:

1. Aliases. People write the same skill many ways ("sklearn", "scikit learn",
   "Scikit-Learn"). Every alias points at one canonical name, so the resume and
   the job description always get compared using the same word.

2. Careful word boundaries. A naive search for "c" would match inside "c++",
   "c#" and "music". We build a boundary for every skill that stops at letters,
   digits, + and #, so short skill names behave properly.

3. Case-sensitive matching for ambiguous words. "go", "rust", "react", "spring"
   and "express" are also ordinary English words. The trick: a resume writes the
   technology capitalised ("Go", "React") but writes the English verb in lower
   case ("go", "react"). So for those few skills we require a capital letter.
"""

from __future__ import annotations

import re

from .text_cleaner import clean_text

# ---------------------------------------------------------------------------
# 1. The skill dictionary, grouped by category
# ---------------------------------------------------------------------------
# Names here are the CANONICAL names: the single spelling we report everywhere.
SKILL_LEXICON: dict[str, tuple[str, ...]] = {
    "Programming Languages": (
        "python", "java", "javascript", "typescript", "c", "c++", "c#",
        "go", "rust", "ruby", "php", "swift", "kotlin", "scala", "r",
        "matlab", "sql", "bash", "perl", "dart", "html", "css", "sass",
        "solidity", "haskell", "elixir", "clojure", "assembly", "vba",
        "groovy", "f#", "julia", "objective-c",
    ),
    "Web Frameworks & Libraries": (
        "react", "angular", "vue", "django", "flask", "fastapi", "express",
        "node.js", "next.js", "spring", "laravel", "ruby on rails",
        "streamlit", "dash", "bootstrap", "tailwind", "jquery", "redux",
        "svelte", "graphql", "rest api", "webpack", "vite",
    ),
    "Machine Learning & AI": (
        "machine learning", "deep learning", "data science", "data analysis",
        "nlp", "computer vision", "reinforcement learning", "mlops",
        "feature engineering", "model deployment", "a/b testing",
        "statistical modeling", "time series", "recommendation systems",
        "large language models", "prompt engineering",
        "retrieval augmented generation", "cross validation", "regression",
        "classification", "clustering", "neural networks", "cnn", "rnn",
        "transformers", "bert", "tensorflow", "pytorch", "keras",
        "scikit-learn", "xgboost", "lightgbm", "catboost", "opencv",
        "hugging face", "langchain", "mlflow", "kubeflow", "jupyter",
        "pandas", "numpy", "scipy", "statistics", "data visualization",
        "matplotlib", "seaborn", "plotly", "ggplot", "tableau", "power bi",
        "excel",
    ),
    "Data Engineering & Databases": (
        "postgresql", "mysql", "mongodb", "redis", "snowflake", "bigquery",
        "redshift", "cassandra", "elasticsearch", "dynamodb", "oracle",
        "sql server", "sqlite", "mariadb", "neo4j", "kafka", "hadoop",
        "hive", "dbt", "etl", "data warehouse", "data modeling",
        "data pipeline", "nosql", "spark", "pyspark", "databricks",
        "presto", "airflow",
    ),
    "Cloud & DevOps": (
        "aws", "azure", "gcp", "docker", "kubernetes", "terraform",
        "ansible", "jenkins", "git", "github", "gitlab", "ci/cd", "linux",
        "unix", "nginx", "serverless", "lambda", "ec2", "s3",
        "cloudformation", "helm", "prometheus", "grafana", "datadog",
        "splunk", "puppet", "chef", "microservices", "monitoring",
        "incident response", "networking", "information security", "oauth",
        "infrastructure as code",
    ),
    "Software Engineering Practices": (
        "data structures", "algorithms", "object oriented programming",
        "design patterns", "unit testing", "test driven development",
        "pytest", "jest", "selenium", "cypress", "agile", "scrum",
        "kanban", "jira", "confluence", "code review", "debugging",
        "version control", "system design", "software development",
        "solid principles", "pair programming", "continuous integration",
        "refactoring", "performance optimization", "accessibility",
        "cryptography", "technical documentation",
    ),
    "Soft Skills": (
        "communication", "teamwork", "leadership", "problem solving",
        "critical thinking", "time management", "collaboration",
        "adaptability", "stakeholder management", "mentoring",
        "presentation", "project management", "analytical skills",
        "creativity", "conflict resolution", "decision making",
        "customer service", "negotiation", "public speaking",
        "cross functional", "attention to detail", "storytelling",
        "technical writing",
    ),
}

# ---------------------------------------------------------------------------
# 2. Aliases: alternative spellings that point at a canonical name
# ---------------------------------------------------------------------------
# Only list spellings that really differ from the canonical name. The canonical
# name is always matched too, so there is no need to repeat it here.
#
# Deliberately absent: short everyday words such as "cv" (means "curriculum
# vitae" on a resume, not computer vision) and "rest" (means "the rest of").
# A false positive is worse than a miss, because it silently inflates scores.
SKILL_ALIASES: dict[str, str] = {
    # Programming languages
    "py": "python", "python3": "python", "golang": "go",
    "js": "javascript", "ecmascript": "javascript", "ts": "typescript",
    "cplusplus": "c++", "cpp": "c++", "c sharp": "c#", "csharp": "c#",
    "dotnet": "c#", ".net": "c#", "asp.net": "c#", "fsharp": "f#",
    "shell scripting": "bash", "zsh": "bash",
    # Web frameworks
    "reactjs": "react", "react.js": "react", "angularjs": "angular",
    "vuejs": "vue", "vue.js": "vue", "nodejs": "node.js",
    "node": "node.js", "expressjs": "express", "express.js": "express",
    "nextjs": "next.js", "spring boot": "spring", "springboot": "spring",
    "rails": "ruby on rails", "rest apis": "rest api",
    "restful api": "rest api", "restful apis": "rest api",
    # Machine learning & data
    "ml": "machine learning", "sklearn": "scikit-learn",
    "scikit learn": "scikit-learn", "dl": "deep learning",
    "natural language processing": "nlp", "llm": "large language models",
    "llms": "large language models",
    "large language model": "large language models",
    "gpt": "large language models", "rag": "retrieval augmented generation",
    "convolutional neural networks": "cnn",
    "recurrent neural networks": "rnn", "tf": "tensorflow",
    "torch": "pytorch", "ab testing": "a/b testing",
    "a b testing": "a/b testing", "time series forecasting": "time series",
    "timeseries": "time series", "data visualisation": "data visualization",
    "dataviz": "data visualization", "powerbi": "power bi",
    "ms excel": "excel", "microsoft excel": "excel",
    "advanced excel": "excel", "huggingface": "hugging face",
    "statistical modelling": "statistical modeling", "stats": "statistics",
    "recommender systems": "recommendation systems",
    "data analytics": "data analysis", "data scientist": "data science",
    "data engineer": "data pipeline",
    # Databases & data engineering
    "postgres": "postgresql", "psql": "postgresql", "mongo": "mongodb",
    "mssql": "sql server", "ms sql": "sql server",
    "microsoft sql server": "sql server",
    "elastic search": "elasticsearch", "dynamo db": "dynamodb",
    "google bigquery": "bigquery", "bq": "bigquery",
    "amazon redshift": "redshift", "data warehousing": "data warehouse",
    "etl pipelines": "etl", "elt": "etl", "data pipelines": "data pipeline",
    "apache spark": "spark", "apache kafka": "kafka",
    "apache airflow": "airflow", "apache hadoop": "hadoop",
    "data models": "data modeling", "google cloud": "gcp",
    "google cloud platform": "gcp", "amazon web services": "aws",
    "microsoft azure": "azure",
    # Cloud & DevOps
    "k8s": "kubernetes", "kube": "kubernetes",
    "containerization": "docker", "containers": "docker",
    "ci cd": "ci/cd", "ci/cd pipelines": "ci/cd", "cicd": "ci/cd",
    "continuous delivery": "ci/cd", "continuous deployment": "ci/cd",
    "github actions": "github", "versioning": "version control",
    "git hub": "github", "git lab": "gitlab", "aws lambda": "lambda",
    "aws ec2": "ec2", "aws s3": "s3", "micro services": "microservices",
    "iac": "infrastructure as code",
    # Software engineering practices
    "oop": "object oriented programming",
    "object oriented design": "object oriented programming",
    "ooad": "object oriented programming", "dsa": "data structures",
    "data structures and algorithms": "data structures",
    "tdd": "test driven development", "unit tests": "unit testing",
    "software testing": "unit testing",
    "scrum master": "scrum", "kanban board": "kanban",
    "atlassian jira": "jira", "code reviews": "code review",
    "refactor": "refactoring", "performance tuning": "performance optimization",
    "wcag": "accessibility", "a11y": "accessibility",
    "cyber security": "information security",
    "cybersecurity": "information security", "infosec": "information security",
    "software engineering": "software development",
    "swe": "software development", "system architecture": "system design",
    # Soft skills
    "problem solving skills": "problem solving", "team work": "teamwork",
    "team player": "teamwork", "cross functional teams": "cross functional",
    "communication skills": "communication",
    "verbal communication": "communication",
    "written communication": "communication",
    "leadership skills": "leadership",
    "analytical thinking": "analytical skills",
    "presentations": "presentation", "detail oriented": "attention to detail",
    "project planning": "project management", "pmp": "project management",
    "prince2": "project management", "customer support": "customer service",
    "adaptable": "adaptability", "flexibility": "adaptability",
    "collaborative": "collaboration", "coaching": "mentoring",
    "story telling": "storytelling", "technical writer": "technical writing",
}

# ---------------------------------------------------------------------------
# 3. Skills that are also common English words
# ---------------------------------------------------------------------------
# For these we only accept a capitalised spelling. That one rule removes almost
# all false positives: "we react to customer feedback" no longer counts as the
# React library, but "React" on a skills line still does.
CASE_SENSITIVE_SKILLS: frozenset[str] = frozenset(
    {
        "go", "r", "rust", "swift", "dart", "react", "angular", "svelte",
        "redux", "spring", "express", "flask", "dash", "bootstrap",
        "tailwind", "assembly", "bash", "oracle", "hive", "snowflake",
        "presto", "puppet", "chef", "confluence", "jest", "cypress",
        "excel", "bert",
    }
)

# ---------------------------------------------------------------------------
# 4. Implied skills
# ---------------------------------------------------------------------------
# If a resume lists a specialised tool we also credit the general skill it
# requires. Knowing Django means knowing Python; knowing PySpark means knowing
# Spark. Applied on top of what was literally found, and the app labels those
# skills separately so the score is never silently inflated.
# A tool can imply more than one general skill, so each value is a tuple:
# matplotlib means you can both program in Python and visualise data.
IMPLIED_SKILLS: dict[str, tuple[str, ...]] = {
    "mysql": ("sql",), "postgresql": ("sql",),
    "sql server": ("sql",), "sqlite": ("sql",), "oracle": ("sql",),
    "mariadb": ("sql",), "bigquery": ("sql",), "redshift": ("sql",),
    "snowflake": ("sql",), "dbt": ("sql",),
    "pyspark": ("spark",), "databricks": ("spark",),
    "next.js": ("react",), "redux": ("react",),
    "pytorch": ("deep learning",), "tensorflow": ("deep learning",),
    "keras": ("deep learning",), "cnn": ("deep learning",),
    "rnn": ("deep learning",), "transformers": ("deep learning",),
    "bert": ("nlp",), "langchain": ("nlp",), "hugging face": ("nlp",),
    "kubernetes": ("docker",), "helm": ("kubernetes",),
    "scikit-learn": ("machine learning",), "xgboost": ("machine learning",),
    "lightgbm": ("machine learning",), "catboost": ("machine learning",),
    "mlflow": ("mlops",), "kubeflow": ("mlops",),
    "airflow": ("data pipeline",), "kafka": ("data pipeline",),
    "matplotlib": ("data visualization", "python"),
    "seaborn": ("data visualization", "python"),
    "plotly": ("data visualization", "python"),
    "ggplot": ("data visualization", "r"),
    "tableau": ("data visualization",), "power bi": ("data visualization",),
    "django": ("python",), "flask": ("python",), "fastapi": ("python",),
    "streamlit": ("python",), "pandas": ("python",), "jupyter": ("python",),
    "numpy": ("python",), "scipy": ("python",), "pytest": ("python",),
    "node.js": ("javascript",), "express": ("node.js",),
    "react": ("javascript",), "vue": ("javascript",),
    "angular": ("typescript",), "jquery": ("javascript",),
    "svelte": ("javascript",), "webpack": ("javascript",),
    "vite": ("javascript",), "jest": ("javascript",),
    "cypress": ("javascript",), "spring": ("java",),
    "laravel": ("php",), "ruby on rails": ("ruby",),
}


# ---------------------------------------------------------------------------
# Lookup tables, built once when the module is imported
# ---------------------------------------------------------------------------
def matching_key(surface_form: str) -> str:
    """Return the comparison form of a skill name or a chunk of text.

    Hyphens and underscores both become spaces, so "scikit-learn",
    "scikit_learn" and "scikit learn" all end up identical. Case is preserved -
    lowercasing is the caller's decision.
    """
    return surface_form.replace("-", " ").replace("_", " ")


def _boundary_pattern(key: str, case_sensitive: bool) -> re.Pattern[str]:
    """Build a regex that matches `key` only as a whole skill name.

    The lookbehind and lookahead are the important part. They stop "c" from
    matching inside "c++", "c#" or "music", and stop "sql" from matching inside
    "mysql" (which is a skill of its own). A trailing full stop is allowed, so
    "Python." at the end of a sentence still matches.
    """
    letters = "a-zA-Z0-9+#" if case_sensitive else "a-z0-9+#"
    left = f"(?<![{letters}])"
    right = f"(?![{letters}])"

    if case_sensitive:
        # Accept "Go" and "GO" but not "go".
        variants = sorted({key.capitalize(), key.upper()}, key=len, reverse=True)
        body = "(?:" + "|".join(re.escape(variant) for variant in variants) + ")"
    else:
        body = re.escape(key)

    return re.compile(left + body + right)


def _build_lookup_tables() -> (
    tuple[dict[str, str], dict[str, str], dict[str, re.Pattern[str]]]
):
    """Pre-compute the tables used by extract_skills()."""
    categories: dict[str, str] = {}
    for category, skills in SKILL_LEXICON.items():
        for canonical in skills:
            categories[canonical] = category

    surface_to_canonical: dict[str, str] = {}
    # Canonical names first, then aliases. setdefault means an alias can never
    # overwrite a canonical name, so "sql" always refers to the skill "sql".
    for canonical in categories:
        surface_to_canonical.setdefault(matching_key(canonical), canonical)
    for alias, canonical in SKILL_ALIASES.items():
        surface_to_canonical.setdefault(matching_key(alias), canonical)

    patterns: dict[str, re.Pattern[str]] = {}
    for surface, canonical in surface_to_canonical.items():
        # Only the canonical spelling of an ambiguous skill needs the capital
        # letter rule. Aliases such as "golang" are unambiguous on their own.
        ambiguous = canonical in CASE_SENSITIVE_SKILLS
        case_sensitive = ambiguous and surface == matching_key(canonical)
        patterns[surface] = _boundary_pattern(surface, case_sensitive)

    return categories, surface_to_canonical, patterns


SKILL_CATEGORIES, SURFACE_TO_CANONICAL, _PATTERNS = _build_lookup_tables()

ALL_SKILLS: tuple[str, ...] = tuple(SKILL_CATEGORIES)


def prepare_text(text: str, lowercase: bool) -> str:
    """Clean text and fold separators, ready for skill matching."""
    return matching_key(clean_text(text, lowercase=lowercase))


def find_skill_mentions(text: str) -> dict[str, list[str]]:
    """Find every skill mentioned in `text`.

    Returns a dict mapping canonical skill name -> the surface forms that were
    actually found, e.g. {"scikit-learn": ["sklearn", "scikit learn"]}.

    The surface forms matter: matcher.py looks them up in the TF-IDF matrix to
    work out how much the job description emphasises each skill.
    """
    folded_lower = prepare_text(text, lowercase=True)
    folded_cased = prepare_text(text, lowercase=False)

    mentions: dict[str, list[str]] = {}
    for surface, canonical in SURFACE_TO_CANONICAL.items():
        # Ambiguous skills are searched in the text that still has capitals;
        # everything else is searched in the lowercased text.
        haystack = folded_cased if canonical in CASE_SENSITIVE_SKILLS else folded_lower
        matches = [match.group() for match in _PATTERNS[surface].finditer(haystack)]
        if matches:
            found = mentions.setdefault(canonical, [])
            # Lowercased so "Go" and "go" count as the same surface form. The
            # TF-IDF vocabulary in matcher.py relies on that to compute one
            # weight per spelling instead of two.
            found.extend(matching_key(match).lower() for match in matches)
    return mentions


def extract_skills(text: str) -> set[str]:
    """Return the set of canonical skills mentioned in `text`.

    Reports only skills that literally appear. Call expand_skills() to also get
    the general skills those tools imply.
    """
    return set(find_skill_mentions(text))


def expand_skills(skills: set[str] | list[str]) -> set[str]:
    """Add the general skills that the given tools require.

    Follows the chain until nothing new appears, so "express" -> "node.js" ->
    "javascript" all get credited. The loop is bounded, which makes an infinite
    cycle impossible even if IMPLIED_SKILLS ever contains one.
    """
    expanded = set(skills)
    for _ in range(len(IMPLIED_SKILLS) + 1):
        additions: set[str] = set()
        for skill in expanded:
            for implied in IMPLIED_SKILLS.get(skill, ()):
                additions.add(implied)
        additions -= expanded
        if not additions:
            break
        expanded |= additions
    return expanded


def get_category(skill: str) -> str:
    """Return the category a skill belongs to, or 'Other' if unknown."""
    return SKILL_CATEGORIES.get(skill, "Other")


def categorize_skills(skills: set[str] | list[str]) -> dict[str, list[str]]:
    """Group skills by category. Categories with no skills are left out."""
    grouped: dict[str, list[str]] = {}
    for skill in skills:
        grouped.setdefault(get_category(skill), []).append(skill)
    return {category: sorted(items) for category, items in sorted(grouped.items())}


def list_known_skills() -> list[str]:
    """Every canonical skill the extractor recognises (handy for the UI)."""
    return sorted(ALL_SKILLS)
