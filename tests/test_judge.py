"""Everything in judge.py except the API call has a right answer. The call is measured, not tested."""

from datetime import date

from email_triage.judge import JudgeConfig, JudgeVerdict, build_system, format_examples, load_judge
from email_triage.labels import SummaryLabel


def config(system: str) -> JudgeConfig:
    return JudgeConfig(id="j", version=1, created=date(2026, 9, 16), description="t",
                       model="m", thinking_level="low", seed=0, system=system)


def test_judge_config_loads():
    assert load_judge("judge-v1").id == "summary-judge"


def test_placeholders_are_filled():
    system = build_system(config("R:\n{rubric}\nE:\n{examples}"), "1. rule {with braces}\n", "ex")
    assert system == "R:\n1. rule {with braces}\nE:\nex"


def test_examples_carry_the_verdict_and_critique():
    label = SummaryLabel(run="v1", case_id="c005", summary="…order…", passed=False,
                         critique="Invents an order.", split="train")
    text = format_examples([label], {"c005": "refund pls"})
    assert "Email: refund pls" in text
    assert "Verdict: FAIL — Invents an order." in text


def test_verdict_schema_is_what_the_api_needs():
    schema = JudgeVerdict.model_json_schema()
    assert schema["required"] == ["passed", "critique"]
    assert schema["properties"]["passed"]["type"] == "boolean"