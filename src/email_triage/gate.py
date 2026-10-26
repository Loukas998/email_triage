"""The merge gate. CI runs it on every pull request, and it never calls a model.

    uv run python -m email_triage.gate v1 v2                    # baseline, then the changed prompts
    uv run python -m email_triage.gate v1 v2 --report report.md # CI posts report.md on the PR

The laptop measures, CI checks. A run file is evidence only if it was made from the prompt file and
the golden set in this commit, so each run carries a fingerprint of both; a run whose fingerprints
do not match is refused, not diffed. Then the diff: a critical category regression blocks the merge.
Summary verdicts are shown, not gated: the judge was validated on ten labels.

Exit code 0 = merge allowed, 1 = blocked. That exit code is the whole contract with CI.
"""

import argparse
from pathlib import Path

from email_triage.dataset import GOLDEN_PATH, load
from email_triage.diff import CRITICAL, WARN, RunDiff, compare
from email_triage.prompts import prompt_path
from email_triage.runs import RUNS_DIR, Run, fingerprint, load_judged, load_run

JUDGE = "judge-v1"
"""The judge measured against hand labels (Lesson 5). An unmeasured judge's verdicts are never shown."""


def stale(run: Run, prompt_sha: str, dataset_sha: str) -> list[str]:
    """Why this run is not evidence for the files in this commit. Empty when it is."""
    problems = []
    if run.prompt_sha != prompt_sha:
        problems.append(f"`runs/{run.prompt}.json` was not made from this `prompts/{run.prompt}.yaml` — run `check {run.prompt}` and commit the run")
    if run.dataset_sha != dataset_sha:
        problems.append(f"`runs/{run.prompt}.json` was not made from this golden set — run `check {run.prompt}` and commit the run")
    return problems


def blocks(diff: RunDiff) -> bool:
    """Only the category column can block: it has a right answer. The summary column is a judge's opinion."""
    return diff.category.severity == "critical"


def _cell(text: str | None) -> str:
    return (text or "").replace("|", "\\|").replace("\n", " ")


def render(diff: RunDiff, before: Run, after: Run) -> str:
    """One candidate's section of the PR comment, in GitHub Markdown."""
    b, a = diff.before, diff.after
    cat = diff.category
    heading = f"### {b} → {a} · {cat.severity.upper()}" + (" — blocks the merge" if blocks(diff) else "")
    lines = [
        heading, "",
        f"| | {b} | {a} | |",
        "|---|---|---|---|",
        f"| category pass | {cat.passed_before}/{cat.n} | {cat.passed_after}/{cat.n} | "
        f"{len(cat.regressions)} regressed · {len(cat.improvements)} improved |",
    ]
    if diff.summary is not None:
        s = diff.summary
        lines.append(f"| summary pass ({JUDGE}) | {s.passed_before}/{s.n} | {s.passed_after}/{s.n} | "
                     f"{len(s.regressions)} regressed · shown, not gated |")
    lines += [
        f"| median latency | {diff.median_ms[0] / 1000:.1f} s | {diff.median_ms[1] / 1000:.1f} s | "
        f"max {diff.max_ms[0] / 1000:.1f} s → {diff.max_ms[1] / 1000:.1f} s |",
        f"| tokens in / out, whole run | {diff.prompt_tokens[0]:,} / {diff.output_tokens[0]:,} | "
        f"{diff.prompt_tokens[1]:,} / {diff.output_tokens[1]:,} | |",
        "",
        f"One case is {1 / cat.n:.0%} at n={cat.n}. Warn above {WARN:.0%} of cases regressed; "
        f"critical, which blocks the merge, above {CRITICAL:.0%}.",
    ]

    old = {out.case_id: out for out in before.outputs}
    new = {out.case_id: out for out in after.outputs}
    regressed = [("category", cat.regressions)]
    if diff.summary is not None:
        regressed.append((f"summary ({JUDGE}, not gated)", diff.summary.regressions))
    for name, flips in regressed:
        if not flips:
            continue
        lines += ["", f"**Regressions — {name}**", "",
                  f"| case | {b} category | {b} summary | {a} category | {a} summary |", "|---|---|---|---|---|"]
        for flip in flips:
            o, n = old[flip.case_id], new[flip.case_id]
            lines.append(f"| {flip.case_id} | {o.category or 'error'} | {_cell(o.summary)} | "
                         f"{n.category or 'error'} | {_cell(n.summary)} |")
    if cat.improvements:
        lines += ["", "Improved: " + ", ".join(f.case_id for f in cat.improvements)]

    lines += ["", f"<sub>{after.model} · think {'on' if after.think else 'off'} · "
                  f"prompt {b} `{before.prompt_sha}` → {a} `{after.prompt_sha}` · "
                  f"golden set v{after.dataset_version} `{after.dataset_sha}`</sub>"]
    return "\n".join(lines)


def gate(baseline: str, candidates: list[str]) -> tuple[bool, str]:
    """(merge allowed, the report). Reads only the files in this commit."""
    names = list(dict.fromkeys([baseline, *candidates]))  # drop repeats, keep the order
    dataset_sha = fingerprint(GOLDEN_PATH)
    runs: dict[str, Run] = {}
    problems: list[str] = []
    for name in names:
        if not (RUNS_DIR / f"{name}.json").exists():
            problems.append(f"there is no `runs/{name}.json` — run `check {name}` and commit it")
            continue
        runs[name] = load_run(name)
        problems += stale(runs[name], fingerprint(prompt_path(name)), dataset_sha)
    if problems:
        return False, "\n".join(["## Eval gate: blocked", "",
                                 "The run files in this commit are not evidence for its prompts:", "",
                                 *[f"- {p}" for p in problems]])

    cases = {case.id: case for case in load().cases}
    allowed, sections = True, []
    for name in names[1:]:
        diff = compare(runs[baseline], runs[name], cases, load_judged(baseline, JUDGE), load_judged(name, JUDGE))
        sections.append(render(diff, runs[baseline], runs[name]))
        allowed = allowed and not blocks(diff)
    if not sections:
        sections.append(f"No prompt changed. `runs/{baseline}.json` matches `prompts/{baseline}.yaml` and the golden set.")
    return allowed, "\n\n".join([f"## Eval gate: {'passed' if allowed else 'blocked'}", *sections])



def main() -> None:
    parser = argparse.ArgumentParser(description="Allow or block a merge, from the run files in this commit.")
    parser.add_argument("baseline", help="the prompt every change is measured against, e.g. v1")
    parser.add_argument("candidates", nargs="*", help="the prompts this change adds or edits, e.g. v2")
    parser.add_argument("--report", type=Path, help="also write the report to this file")
    args = parser.parse_args()

    allowed, report = gate(args.baseline, args.candidates)
    print(report)
    if args.report:
        args.report.write_text(report + "\n")
    raise SystemExit(0 if allowed else 1)


if __name__ == "__main__":
    main()
