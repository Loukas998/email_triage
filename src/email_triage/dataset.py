from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_PATH = PROJECT_ROOT / "golden" / "v1.json"

Category = Literal["billing", "technical", "account", "general"]
Difficulty = Literal["easy", "medium", "hard"]
Verdict = Literal["exact", "acceptable", "missed"]


class GoldenCase(BaseModel):
    model_config = ConfigDict(extra="forbid")  # a typo in the JSON is an error, not a silent drop

    id: str
    input: str
    expected_category: Category
    acceptable_categories: list[Category] = []
    expected_summary: str
    difficulty: Difficulty
    tags: list[str] = []
    notes: str

    def verdict(self, category: Category) -> Verdict:
        """The one decision check.py and the tests both depend on — so it lives in one place."""
        if category == self.expected_category:
            return "exact"
        if category in self.acceptable_categories:
            return "acceptable"
        return "missed"

    def accepts(self, category: Category) -> bool:
        return self.verdict(category) != "missed"


class GoldenDataset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int
    guideline_version: int
    cases: list[GoldenCase]

    @model_validator(mode="after")
    def ids_are_unique(self) -> "GoldenDataset":
        ids = [case.id for case in self.cases]
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            raise ValueError(f"duplicate case ids: {duplicates}")
        return self


def load(path: Path = GOLDEN_PATH) -> GoldenDataset:
    return GoldenDataset.model_validate_json(path.read_text())