"""A run is what one prompt config produced on one dataset version. Saved so it can be labelled,
judged and diffed later without calling the model again.

    runs/v1.json, runs/v2.json — one file per prompt name, overwritten on re-run. At temperature 0
    a re-run of the same prompt version on the same dataset version must reproduce it; if it does
    not, that is a finding, not noise.
"""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from email_triage.dataset import PROJECT_ROOT, Category, Verdict

RUNS_DIR = PROJECT_ROOT / "runs"

Verdict = Literal["exact", "acceptable", "missed", "error"]


class RunOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    verdict: Verdict
    category: Category | None = None
    """None when the call errored (token cap, schema rejection)."""
    summary: str | None = None


class Run(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str
    """The prompt file name, e.g. "v2" — what `check v2` was given."""
    prompt_version: int
    model: str
    think: bool
    dataset_version: int
    guideline_version: int
    outputs: list[RunOutput]


def save_run(run: Run) -> Path:
    RUNS_DIR.mkdir(exist_ok=True)
    path = RUNS_DIR / f"{run.prompt_version}.json"
    path.write_text(run.model_dump_json(indent=2) + "\n")
    return path


def load_run(prompt: str):
    return Run.model_validate_json((RUNS_DIR / f"{prompt}.json").read_text())