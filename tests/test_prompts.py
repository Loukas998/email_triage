import pytest
from pydantic import ValidationError

from email_triage.prompts import PromptConfig, load_prompt

def test_every_prompt_file_loads():
    for name in ("v1", "v2"):
        config = load_prompt(name)
        assert config.id == "triage"
        assert config.system.strip()

def test_misspelt_setting_is_rejected():
    """The Lesson 2 accident, made impossible: a key the schema does not know is an error."""
    fields = load_prompt("v1").model_dump() | {"temprature": 0.0}
    with pytest.raises(ValidationError, match="temprature"):
        PromptConfig.model_validate(fields)
