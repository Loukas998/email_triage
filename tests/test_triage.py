"""The conversion from Ollama's response to Usage has a right answer. The model call does not."""

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from email_triage.triage import EmailTriage, TriageError, Usage


def test_from_response_converts_units():
    response = SimpleNamespace(total_duration=8_437_000_000, prompt_eval_count=61, eval_count=285)
    usage = Usage.from_response(response)
    assert (usage.duration_ms, usage.prompt_tokens, usage.output_tokens) == (8437, 61, 285)


def test_error_carries_the_cost():
    """A capped call returns empty content, which the contract rejects — but it still took 87 s."""
    response = SimpleNamespace(total_duration=87_000_000_000, prompt_eval_count=180, eval_count=2048,
                               done_reason="length", message=SimpleNamespace(content=""))
    with pytest.raises(ValidationError) as caught:
        EmailTriage.model_validate_json(response.message.content)
    err = TriageError(caught.value, response)
    assert err.usage.output_tokens == 2048
    assert err.done_reason == "length"