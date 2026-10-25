"""The diff is ordinary code over two JSON files. Every number it prints has a right answer."""

import pytest

from email_triage.dataset import GoldenCase
from email_triage.diff import CRITICAL, WARN, compare, render
from email_triage.runs import JudgedRun, Run, RunOutput, SummaryVerdict


def case(case_id: str, expected: str) -> GoldenCase:
    return GoldenCase(id=case_id, input="x", expected_category=expected, expected_summary="x",
                      difficulty="easy", notes="fixture")


CASES = {c.id: c for c in [case("c1", "billing"), case("c2", "billing"), case("c3", "technical"), case("c4", "account")]}


def run(name: str, verdicts: dict[str, str], ms: int = 1000, tokens_out: int = 100) -> Run:
    outputs = []
    for case_id, verdict in verdicts.items():
        if verdict == "error":
            outputs.append(RunOutput(case_id=case_id, verdict="error"))
        else:
            outputs.append(RunOutput(case_id=case_id, verdict=verdict, category="billing", summary=f"{name} {case_id}",
                                     duration_ms=ms, prompt_tokens=10, output_tokens=tokens_out))
    return Run(prompt=name, prompt_version=1, model="m", think=True, dataset_version=1, guideline_version=1, outputs=outputs)


BEFORE = run("v1", {"c1": "exact", "c2": "exact", "c3": "missed", "c4": "exact"}, ms=1000, tokens_out=100)
AFTER = run("v2", {"c1": "exact", "c2": "missed", "c3": "acceptable", "c4": "error"}, ms=3000, tokens_out=400)


def test_flips_are_listed_by_direction():
    d = compare(BEFORE, AFTER, CASES)
    assert [f.case_id for f in d.category.regressions] == ["c2", "c4"]
    assert [f.case_id for f in d.category.improvements] == ["c3"]
    assert d.category.regressions[1].after_text == "error"


def test_net_delta_and_pass_counts():
    d = compare(BEFORE, AFTER, CASES)
    assert (d.category.passed_before, d.category.passed_after) == (3, 2)
    assert d.category.delta == pytest.approx(-0.25)


def test_severity_is_judged_on_regressions_not_net():
    """One regression and one improvement net to zero — and must still warn."""
    before = run("a", {"c1": "exact", "c2": "missed"} | {f"x{i}": "exact" for i in range(14)})
    after = run("b", {"c1": "missed", "c2": "exact"} | {f"x{i}": "exact" for i in range(14)})
    cases = CASES | {f"x{i}": case(f"x{i}", "general") for i in range(14)}
    d = compare(before, after, cases)
    assert d.category.delta == 0
    assert d.category.severity == "warn"  # 1/16 = 6.25% > 3%


def test_one_case_at_sixteen_is_warn_and_two_is_critical():
    assert 1 / 16 > WARN and 1 / 16 < CRITICAL
    assert 2 / 16 > CRITICAL


def test_exact_rate_by_expected_category():
    d = compare(BEFORE, AFTER, CASES)
    assert d.by_expected["billing"] == (2, 1, 2)
    assert d.by_expected["technical"] == (0, 0, 1)   # acceptable is not exact
    assert d.by_expected["account"] == (1, 0, 1)


def test_latency_and_tokens_are_summed_over_timed_cases():
    d = compare(BEFORE, AFTER, CASES)
    assert d.median_ms == (1000, 3000)
    assert d.output_tokens == (400, 1200)  # the errored c4 in v2 has no tokens to count


def test_summary_dimension_needs_both_judge_files():
    d = compare(BEFORE, AFTER, CASES)
    assert d.summary is None
    judged = lambda name, verdicts: JudgedRun(run=name, judge="judge-v1", judge_version=1, rubric_version=1,
        verdicts=[SummaryVerdict(case_id=c, passed=p, critique="") for c, p in verdicts.items()])
    d = compare(BEFORE, AFTER, CASES, judged("v1", {"c1": True, "c2": True, "c3": False}), judged("v2", {"c1": False, "c2": True, "c3": True}))
    assert d.summary.n == 3  # c4 has no summary in v2, so it is not scored
    assert [f.case_id for f in d.summary.regressions] == ["c1"]
    assert [f.case_id for f in d.summary.improvements] == ["c3"]


def test_different_datasets_refuse_to_compare():
    other = AFTER.model_copy(update={"dataset_version": 2})
    with pytest.raises(ValueError, match="different datasets"):
        compare(BEFORE, other, CASES)


def test_render_names_the_regressions():
    text = render(compare(BEFORE, AFTER, CASES))
    assert "regressions (2):" in text
    assert "c4  pass → fail" in text
    assert "verdict   CRITICAL — 2 regressions at n=4" in text