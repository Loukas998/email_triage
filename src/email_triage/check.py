"""Run every golden case through triage() with one prompt config and show the disagreements.

    uv run python -m email_triage.check v1
    uv run python -m email_triage.check v2
"""

import sys
from collections import Counter

from pydantic import ValidationError

from email_triage.dataset import load
from email_triage.prompts import load_prompt
from email_triage.triage import triage

MARK = {"exact": "  ", "acceptable": "~ ", "missed": "X ", "error": "! "}


def main(prompt_name: str) -> None:
    config = load_prompt(prompt_name)
    dataset = load()
    tally: Counter[str] = Counter()

    for case in dataset.cases:
        try:
            result = triage(case.input, config)
        except ValidationError as err:
            # The model hit the token cap mid-thought, or returned something the schema rejects.
            # Count it and keep going: one bad call must not take the whole run down.
            tally["error"] += 1
            print(f"{MARK['error']}{case.id:<7} {case.difficulty:<6} expected={case.expected_category:<9} got=ERROR ({err.error_count()} validation errors)")
            print()
            continue

        verdict = case.verdict(result.category)
        tally[verdict] += 1

        print(f"{MARK[verdict]}{case.id:<7} {case.difficulty:<6} expected={case.expected_category:<9} got={result.category}")
        print(f"          model: {result.summary}")
        if verdict != "exact":
            print(f"          yours: {case.expected_summary}")
            print(f"          notes: {case.notes}")
        print()

    n = len(dataset.cases)
    print(f"{n} cases · exact {tally['exact']} · acceptable {tally['acceptable']} · missed {tally['missed']} · error {tally['error']}")
    print(f"prompt {config.id} v{config.version} (think={config.think}) · dataset v{dataset.version} · guideline v{dataset.guideline_version}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "v1")