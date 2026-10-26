"""Compare two runs case by case. A total can go up while one customer's case goes down;
only the diff shows which cases moved, and in which direction.

    uv run python -m email_triage.diff v1 v2

Reads runs/<name>.json for both, and runs/<name>.judge-v1.json for each if it exists.
Ordinary code over two JSON files — so it has tests, and no model is called.
"""

import os
import sys
from statistics import median
from typing import Literal

from pydantic import BaseModel

from email_triage.dataset import GoldenCase, load
from email_triage.runs import JudgedRun, Run, RunOutput, load_judged, load_run

# The guide's thresholds, as fractions of the case count. Set GATE_WARN / GATE_CRITICAL to change them —
# they are printed next to what one case is worth, because at n = 16 a single flip is already past both.
WARN = float(os.environ.get("GATE_WARN", "0.03"))
CRITICAL = float(os.environ.get("GATE_CRITICAL", "0.08"))

Severity = Literal["ok", "warn", "critical"]


class Flip(BaseModel):
    case_id: str
    before: str
    after: str
    before_text: str | None = None
    after_text: str | None = None


class Dimension(BaseModel):
    """One scored thing — category, or summary — across the two runs."""

    name: str
    n: int
    passed_before: int
    passed_after: int
    regressions: list[Flip]
    improvements: list[Flip]

    @property
    def delta(self) -> float:
        return (self.passed_after - self.passed_before) / self.n

    @property
    def severity(self) -> Severity:
        """Judged on regressions alone, not on the net delta: one regression and one improvement
        net to zero, and zero is exactly what would hide the regression."""
        share = len(self.regressions) / self.n
        return "critical" if share > CRITICAL else "warn" if share > WARN else "ok"


class RunDiff(BaseModel):
    before: str
    after: str
    n: int
    category: Dimension
    by_expected: dict[str, tuple[int, int, int]]
    """expected category → (exact before, exact after, cases)."""
    summary: Dimension | None
    median_ms: tuple[int, int]
    max_ms: tuple[int, int]
    prompt_tokens: tuple[int, int]
    output_tokens: tuple[int, int]


def category_passed(out: RunOutput) -> bool:
    return out.verdict in ("exact", "acceptable")


def _flips(ids: list[str], before: dict[str, bool | None], after: dict[str, bool | None],
           before_text: dict[str, str | None], after_text: dict[str, str | None]) -> tuple[list[Flip], list[Flip]]:
    regressions, improvements = [], []
    for case_id in ids:
        b, a = before.get(case_id), after.get(case_id)
        if b is None or a is None or b == a:
            continue
        flip = Flip(case_id=case_id, before="pass" if b else "fail", after="pass" if a else "fail",
                    before_text=before_text.get(case_id), after_text=after_text.get(case_id))
        (improvements if a else regressions).append(flip)
    return regressions, improvements


def compare(before: Run, after: Run, cases: dict[str, GoldenCase],
            judged_before: JudgedRun | None = None, judged_after: JudgedRun | None = None) -> RunDiff:
    if before.dataset_version != after.dataset_version:
        raise ValueError(f"different datasets: v{before.dataset_version} vs v{after.dataset_version} — a diff across datasets compares nothing")
    b_out = {out.case_id: out for out in before.outputs}
    a_out = {out.case_id: out for out in after.outputs}
    ids = [case_id for case_id in b_out if case_id in a_out]
    if not ids:
        raise ValueError("the two runs share no cases")

    # Category: pass = exact or acceptable; an error is a fail like any other.
    b_pass = {i: category_passed(b_out[i]) for i in ids}
    a_pass = {i: category_passed(a_out[i]) for i in ids}
    b_cat = {i: b_out[i].category or "error" for i in ids}
    a_cat = {i: a_out[i].category or "error" for i in ids}
    reg, imp = _flips(ids, b_pass, a_pass, b_cat, a_cat)
    category = Dimension(name="category", n=len(ids), passed_before=sum(b_pass.values()),
                         passed_after=sum(a_pass.values()), regressions=reg, improvements=imp)

    by_expected: dict[str, tuple[int, int, int]] = {}
    for label in ("billing", "technical", "account", "general"):
        members = [i for i in ids if cases[i].expected_category == label]
        if members:
            by_expected[label] = (
                sum(b_out[i].verdict == "exact" for i in members),
                sum(a_out[i].verdict == "exact" for i in members),
                len(members),
            )

    summary = None
    if judged_before is not None and judged_after is not None:
        b_j = {i: judged_before.passed(i) for i in ids}
        a_j = {i: judged_after.passed(i) for i in ids}
        scored = [i for i in ids if b_j[i] is not None and a_j[i] is not None]
        reg, imp = _flips(scored, b_j, a_j, {i: b_out[i].summary for i in ids}, {i: a_out[i].summary for i in ids})
        summary = Dimension(name="summary", n=len(scored), passed_before=sum(bool(b_j[i]) for i in scored),
                            passed_after=sum(bool(a_j[i]) for i in scored), regressions=reg, improvements=imp)

    def timed(outs: dict[str, RunOutput]) -> list[RunOutput]:
        return [outs[i] for i in ids if outs[i].duration_ms is not None]

    b_t, a_t = timed(b_out), timed(a_out)
    return RunDiff(
        before=before.prompt, after=after.prompt, n=len(ids),
        category=category, by_expected=by_expected, summary=summary,
        median_ms=(round(median(o.duration_ms for o in b_t)) if b_t else 0, round(median(o.duration_ms for o in a_t)) if a_t else 0),
        max_ms=(max((o.duration_ms for o in b_t), default=0), max((o.duration_ms for o in a_t), default=0)),
        prompt_tokens=(sum(o.prompt_tokens or 0 for o in b_t), sum(o.prompt_tokens or 0 for o in a_t)),
        output_tokens=(sum(o.output_tokens or 0 for o in b_t), sum(o.output_tokens or 0 for o in a_t)),
    )


def _pct(x: float) -> str:
    return f"{x:+.0%}"


def _dimension(d: Dimension, extra: str = "") -> list[str]:
    one_case = 1 / d.n
    lines = [
        f"{d.name:<9} pass {d.passed_before}/{d.n} → {d.passed_after}/{d.n}  ({_pct(d.delta)})"
        f"   [1 case = {one_case:.0%} at n={d.n}; warn > {WARN:.0%}, critical > {CRITICAL:.0%}]{extra}"
    ]
    for kind, flips in (("regressions", d.regressions), ("improvements", d.improvements)):
        if flips:
            lines.append(f"  {kind} ({len(flips)}):")
            for f in flips:
                lines.append(f"    {f.case_id}  {f.before} → {f.after}   {f.before_text!s:.60} → {f.after_text!s:.60}")
    return lines


def render(d: RunDiff) -> str:
    lines = [f"{d.before} → {d.after} · {d.n} cases", ""]
    lines += _dimension(d.category)
    lines.append("  exact, by expected category: " + " · ".join(
        f"{label} {b}/{n} → {a}/{n}" for label, (b, a, n) in d.by_expected.items()))
    if d.summary is not None:
        lines.append("")
        lines += _dimension(d.summary)
    else:
        lines.append("")
        lines.append("summary   (no judge file for one or both runs — run `judge run <name>` first)")
    lines.append("")
    lines.append(f"latency   median {d.median_ms[0] / 1000:.1f} s → {d.median_ms[1] / 1000:.1f} s · max {d.max_ms[0] / 1000:.1f} s → {d.max_ms[1] / 1000:.1f} s")
    lines.append(f"tokens    in {d.prompt_tokens[0]} → {d.prompt_tokens[1]} · out {d.output_tokens[0]} → {d.output_tokens[1]}  (whole run)")
    lines.append("")
    worst = max([d.category.severity] + ([d.summary.severity] if d.summary else []),
                key=["ok", "warn", "critical"].index)
    regressions = len(d.category.regressions) + (len(d.summary.regressions) if d.summary else 0)
    lines.append(f"verdict   {worst.upper()} — {regressions} regression{'s' if regressions != 1 else ''} at n={d.n}. Read them before believing the net number.")
    return "\n".join(lines)


def main(before_name: str, after_name: str, judge_name: str = "judge-v1") -> None:
    cases = {case.id: case for case in load().cases}
    diff = compare(
        load_run(before_name), load_run(after_name), cases,
        load_judged(before_name, judge_name), load_judged(after_name, judge_name),
    )
    print(render(diff))


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit("usage: python -m email_triage.diff <before> <after>   e.g. v1 v2")
    main(sys.argv[1], sys.argv[2])