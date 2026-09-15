"""The eval gate. Slow (calls the local model), so it only runs when asked:

    uv run pytest -m eval
    PROMPT=v2 uv run pytest -m eval
"""

import os

import pytest

from email_triage.triage import triage
from email_triage.dataset import load
from email_triage.prompts import load_prompt


@pytest.mark.eval
def test_no_easy_case_is_missed():
    config = load_prompt(os.environ.get("PROMPT", "v1"))
    missed = [
        case.id
        for case in load().cases
        if case.difficulty == "easy" and case.verdict(triage(case.input, config).category) == "missed"
    ]
    assert missed == [], f"prompt {config.id} v{config.version} missed easy cases: {missed}"