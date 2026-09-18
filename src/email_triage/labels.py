"""Your pass/fail labels on model summaries, with a critique each. You are the domain expert;
the judge is measured against this file and learns from its `train` slice.

    uv run python -m email_triage.labels v1 v2      # label every summary in runs/v1.json and runs/v2.json

Answers are saved after every label, so quitting loses nothing. Each summary is assigned to a
split before you see it — train (few-shot examples in the judge prompt), dev (tune the judge),
test (run once, at the end) — so the assignment cannot be swayed by what you thought of it.
"""

import random
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from email_triage.dataset import PROJECT_ROOT, load
from email_triage.runs import load_run

LABELS_PATH = PROJECT_ROOT / "golden" / "summary-labels.json"

Split = Literal["train", "dev", "test"]


class SummaryLabel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run: str
    case_id: str
    summary: str
    passed: bool
    critique: str
    """Why. Detailed enough that a new hire — or a judge prompt — could learn the rule from it."""
    split: Split

    @property
    def key(self) -> tuple[str, str]:
        return (self.run, self.case_id)


class LabelSet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int
    rubric_version: int
    labels: list[SummaryLabel] = []


def load_labels(path: Path = LABELS_PATH) -> LabelSet:
    if not path.exists():
        return LabelSet(version=1, rubric_version=1)
    return LabelSet.model_validate_json(path.read_text())


def save_labels(labels: LabelSet, path: Path = LABELS_PATH) -> None:
    path.write_text(labels.model_dump_json(indent=2) + "\n")


def assign_splits(keys: list(tuple[str, str]), seed: int = 0) -> dict[tuple[str, str], Split]:
    """Shuffle once, deterministically, then cut: ~20% train, ~40% dev, the rest test
    (Hamel Husain's proportions). The same keys always get the same split."""
    order = sorted(keys)
    random.Random(seed).shuffle(order)
    n_train = max(1, round(len(order) * 0.2))
    n_dev = round(len(order) * 0.4)
    splits: dict[tuple[str, str], Split] = {}
    for i, key in enumerate(order):
        splits[key] = "train" if i < n_train else "dev" if i < n_train + n_dev else "test"
    return splits


def main(run_names: list[str]) -> None:
    cases = {case.id: case for case in load().cases}
    labels = load_labels()
    done = {label.key for label in labels.labels}

    queue = [
        (run.prompt, out.case_id, out.summary)
        for run in map(load_run, run_names)
        for out in run.outputs
        if out.summary is not None # errored calls have nothing to label
    ]

    splits = assign_splits([(r, c) for r, c, _ in queue])
    todo = [(r, c, s) for r, c, s in queue if (r, c) not in done]
    print(f"{len(done)} labelled, {len(todo)} to go. p = pass, f = fail, q = quit.\n")

    for run_name, case_id, summary in todo:
        case = cases[case_id]
        print(f"[{run_name} · {case_id} · {case.difficulty}]")
        print(f"  email:   {case.input}")
        print(f"  yours:   {case.expected_summary}")
        print(f"  model:   {summary}")
        answer = ""
        while answer not in ("p", "f", "q"):
            answer = input("  pass / fail / quit [p/f/q]: ").strip().lower()
        if answer == "q":
            break
        passed = answer == "p"
        critique = ""
        while not critique:
            critique = input("  critique (why — a rule someone else could apply): ").strip()
            if passed and not critique:
                critique = "ok"
        labels.labels.append(SummaryLabel(
            run=run_name, case_id=case_id, summary=summary,
            passed=passed, critique=critique, split=splits[(run_name, case_id)],
        ))
        save_labels(labels)
        print()

    n = len(labels.labels)
    fails = sum(not label.passed for label in labels.labels)
    print(f"{n} labelled · {n - fails} pass · {fails} fail · saved {LABELS_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["v1", "v2"])


