"""Shared pytest configuration and test data.

Two jobs happen here:

1. Put the project root on sys.path so `import src...` works no matter which
   directory pytest was started from.
2. Provide the sample resume/job texts that several test modules reuse.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# tests/conftest.py -> parents[0] is tests/, parents[1] is the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


RESUME_TEXT = """Alex Morgan
alex.morgan@example.com | +1 (555) 010-2030 | github.com/alexmorgan

SKILLS
Python, pandas, NumPy, SQL, Excel, matplotlib, Git, Linux, communication

EXPERIENCE
Data Analyst, Retail Co (2022 - present)
- Built dashboards in Tableau used weekly by the marketing team
- Wrote SQL queries against PostgreSQL to track campaign performance
- Automated reporting with Python and pandas, saving 6 hours a week
"""

JOB_TEXT = """Data Scientist

Requirements:
- Python, pandas, NumPy and scikit-learn
- SQL and statistics
- Machine learning and A/B testing
- AWS and Docker
- Excellent communication skills

Nice to have:
- Airflow, dbt and Snowflake
- Deep learning with TensorFlow
"""


@pytest.fixture
def resume_text() -> str:
    """A short but realistic data-analyst resume."""
    return RESUME_TEXT


@pytest.fixture
def job_text() -> str:
    """A job advert with both a required and a nice-to-have section."""
    return JOB_TEXT
