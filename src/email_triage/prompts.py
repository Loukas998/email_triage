"""Prompt configs are code. They live in files, carry a version, and load through a schema.

"What did I test?" must have an answer you can point at: prompts/<name>.yaml, at a
commit. Everything that changes the model's behaviour lives here — including the
things that are easy to forget are settings, like `think`.
"""

from datetime import date
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROMPTS_DIR = PROJECT_ROOT / "prompts"


class PromptConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    version: int
    created: date
    description: str
    model: str
    think: bool
    temperature: float
    system: str

def load_prompt(name: str) -> PromptConfig:
    path = PROMPTS_DIR / f"{name}.yaml"
    return PromptConfig.model_validate(yaml.safe_load(path.read_text()))