"""The gate decides whether a merge is allowed. Both halves of that decision are plain code."""

from email_triage.dataset import GoldenCase
from email_triage.diff import compare
from email_triage.gate import blocks, render, stale
from email_triage.runs import Run, RunOutput


def run(name: str, verdicts: dict[str, str], prompt_sha: str | None = "p1", dataset_sha: str | None = "d1") -> Run:
    outputs = [RunOutput(case_id=case_id, verdict=verdict, category=None if verdict == "error" else "billing",
                         summary=None if verdict == "error" else f"{name} says {case_id}",
                         duration_ms=1000, prompt_tokens=10, output_tokens=100)
               for case_id, verdict in verdicts.items()]
    return Run(prompt=name, prompt_version=1, model="m", think=True, dataset_version=1, guideline_version=1,
               prompt_sha=prompt_sha, dataset_sha=dataset_sha, outputs=outputs)


CASES = {i: GoldenCase(id=i, input="x", expected_category="billing", expected_summary="x", difficulty="easy", notes="x")
         for i in ("c1", "c2", "c3", "c4")}
BEFORE = run("v1", {"c1": "exact", "c2": "exact", "c3": "exact", "c4": "missed"})


def test_a_fresh_run_is_evidence():
    assert stale(BEFORE, prompt_sha="p1", dataset_sha="d1") == []


def test_a_run_made_from_another_prompt_file_is_refused():
    problems = stale(BEFORE, prompt_sha="p2", dataset_sha="d1")
    assert len(problems) == 1 and "check v1" in problems[0]


def test_a_run_from_before_fingerprints_existed_is_refused():
    old = run("v1", {"c1": "exact"}, prompt_sha=None, dataset_sha=None)
    assert len(stale(old, prompt_sha="p1", dataset_sha="d1")) == 2


def test_two_regressions_in_four_block_and_are_shown_side_by_side():
    after = run("v2", {"c1": "exact", "c2": "missed", "c3": "error", "c4": "exact"})
    diff = compare(BEFORE, after, CASES)
    assert blocks(diff)
    report = render(diff, BEFORE, after)
    assert "CRITICAL" in report
    assert "| c2 | billing | v1 says c2 | billing | v2 says c2 |" in report
    assert "| c3 | billing | v1 says c3 | error |  |" in report


def test_one_improvement_and_nothing_broken_passes():
    after = run("v2", {"c1": "exact", "c2": "exact", "c3": "exact", "c4": "exact"})
    assert not blocks(compare(BEFORE, after, CASES))