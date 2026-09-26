"""Score summaries with a stronger model, then measure how often it agrees with you.

    uv run python -m email_triage.judge dev              # tune: read every disagreement, revise the rubric/prompt, re-run
    uv run python -m email_triage.judge test             # once, at the end, with the prompt frozen
    uv run python -m email_triage.judge dev judge-v2     # a different judge config (model, rubric wording, examples)

Verdicts are saved after every call and re-runs resume from the file: on a free tier measured in
tens of calls a day, a verdict already paid for is never bought twice.

The judge is a hosted model (Gemini, free tier) because a judge weaker than the model it judges
produces noise that looks like data. Needs GEMINI_API_KEY, read from the project's .env file (gitignored).
"""

import json
import os
import sys
import time
from datetime import date
from pathlib import Path

import yaml
from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel, ConfigDict

from email_triage.agreement import describe, measure
from email_triage.dataset import PROJECT_ROOT, load
from email_triage.labels import SummaryLabel, load_labels
from email_triage.prompts import PROMPTS_DIR
from email_triage.runs import RUNS_DIR

RUBRIC_PATH = PROJECT_ROOT / "golden" / "SUMMARY-RUBRIC.md"
ENV_PATH = PROJECT_ROOT / ".env"


class JudgeConfig(BaseModel):
    """A different boundary from PromptConfig: another provider, no thinking flag, no token cap."""

    model_config = ConfigDict(extra="forbid")

    id: str
    version: int
    created: date
    description: str
    model: str
    thinking_level: str
    """How much the judge reasons before answering: minimal / low / medium / high. Cost and latency knob."""
    seed: int
    """This API has no temperature setting; the seed is the only reproducibility knob it offers."""
    system: str


class JudgeVerdict(BaseModel):
    """What the judge must return. Kept minimal on purpose — this schema is sent to the API."""

    passed: bool
    critique: str


class JudgeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run: str
    case_id: str
    split: str
    summary: str
    human_passed: bool
    human_critique: str
    judge_passed: bool
    judge_critique: str


class DailyQuotaExhausted(Exception):
    """The free tier's per-day allowance is gone. Not retryable today — the run stops and saves."""


def load_judge(name: str) -> JudgeConfig:
    return JudgeConfig.model_validate(yaml.safe_load((PROMPTS_DIR / f"{name}.yaml").read_text()))


def format_examples(examples: list[SummaryLabel], emails: dict[str, str]) -> str:
    blocks = []
    for label in examples:
        verdict = "PASS" if label.passed else "FAIL"
        blocks.append(
            f"Email: {emails[label.case_id]}\nSummary: {label.summary}\nVerdict: {verdict} — {label.critique}"
        )
    return "\n\n".join(blocks)


def build_system(config: JudgeConfig, rubric: str, examples: str) -> str:
    # str.replace, not str.format: the rubric may contain braces.
    return config.system.replace("{rubric}", rubric.strip()).replace("{examples}", examples)


def ask_gemini(config: JudgeConfig, system: str, email: str, summary: str) -> JudgeVerdict:
    load_dotenv(ENV_PATH)  # puts the file's KEY=value lines into the environment; anchored to the project, not the CWD
    if not os.environ.get("GEMINI_API_KEY"):
        raise SystemExit(f"GEMINI_API_KEY is not set — put it in {ENV_PATH} (see .env.example)")
    client = genai.Client()  # reads GEMINI_API_KEY from the environment
    user = f"Email:\n{email}\n\nSummary:\n{summary}"
    for attempt in range(3):
        try:
            interaction = client.interactions.create(
                model=config.model,
                system_instruction=system,
                input=user,
                generation_config={"thinking_level": config.thinking_level, "seed": config.seed},
                response_format={
                    "type": "text",
                    "mime_type": "application/json",
                    "schema": JudgeVerdict.model_json_schema(),
                },
            )
            return JudgeVerdict.model_validate_json(interaction.output_text)
        except Exception as err:
            text = str(err)
            if "per day" in text or "Free Tier" in text:
                # A daily quota is not something a 20-second sleep fixes. Stop, and let the caller
                # save what it already has: on 20 calls a day, every finished verdict is precious.
                raise DailyQuotaExhausted(text) from err
            if attempt == 2 or ("429" not in text and "RESOURCE_EXHAUSTED" not in text):
                raise
            print(f"  rate limited, waiting 20 s ({err.__class__.__name__})")
            time.sleep(20)
    raise AssertionError("unreachable")


def load_results(path: Path) -> list[JudgeResult]:
    if not path.exists():
        return []
    return [JudgeResult.model_validate(item) for item in json.loads(path.read_text())]


def save_results(path: Path, results: list[JudgeResult]) -> None:
    RUNS_DIR.mkdir(exist_ok=True)
    path.write_text("[\n" + ",\n".join(r.model_dump_json(indent=2) for r in results) + "\n]\n")


def main(split: str, judge_name: str = "judge-v1", ask=ask_gemini) -> None:
    config = load_judge(judge_name)
    labels = load_labels()
    emails = {case.id: case.input for case in load().cases}

    train = [label for label in labels.labels if label.split == "train"]
    targets = [label for label in labels.labels if label.split == split]
    if not targets:
        raise SystemExit(f"no labels in split {split!r} — run `python -m email_triage.labels` first")

    system = build_system(config, RUBRIC_PATH.read_text(), format_examples(train, emails))
    out = RUNS_DIR / f"{judge_name}-{split}.json"
    results = load_results(out)  # resume: a verdict already paid for is never bought twice
    done = {(r.run, r.case_id) for r in results}
    todo = [label for label in targets if (label.run, label.case_id) not in done]
    if done:
        print(f"{len(done)} verdicts already saved, {len(todo)} to go\n")

    stopped = None
    for label in todo:
        try:
            verdict = ask(config, system, emails[label.case_id], label.summary)
        except DailyQuotaExhausted as err:
            stopped = err
            break
        results.append(JudgeResult(
            run=label.run, case_id=label.case_id, split=label.split, summary=label.summary,
            human_passed=label.passed, human_critique=label.critique,
            judge_passed=verdict.passed, judge_critique=verdict.critique,
        ))
        save_results(out, results)  # after every call, not at the end: a crash must not cost the quota
        mark = "  " if verdict.passed == label.passed else "X "
        human = "pass" if label.passed else "FAIL"
        judge = "pass" if verdict.passed else "FAIL"
        print(f"{mark}{label.run:<3} {label.case_id:<5} you={human:<4} judge={judge:<4}  {verdict.critique}")
        if verdict.passed != label.passed:
            print(f"          yours: {label.critique}")

    if not results:
        raise SystemExit(f"no verdicts yet — {stopped}")

    a = measure([r.human_passed for r in results], [r.judge_passed for r in results])
    print()
    print(describe(a))
    print(f"judge {config.id} v{config.version} ({config.model}) · rubric v{labels.rubric_version} · split {split} · {len(train)} train examples in prompt")
    print(f"saved {out.relative_to(PROJECT_ROOT)}")
    if stopped:
        print(f"\nSTOPPED before {len(targets) - len(results)} of {len(targets)} labels: {stopped}")
        print("Re-run tomorrow — it resumes from the saved file. The rates above are on what is measured so far; say that n when you quote them.")


if __name__ == "__main__":
    # python -m email_triage.judge <split> [judge name]
    main(sys.argv[1] if len(sys.argv) > 1 else "dev",
         sys.argv[2] if len(sys.argv) > 2 else "judge-v1")
