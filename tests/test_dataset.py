"""Tests for the part of the system that has a right answer: your own code."""

import pytest
from pydantic import ValidationError

from email_triage.dataset import GoldenCase, GoldenDataset, load


def make_case(**overrides) -> GoldenCase:
    """A valid case you can bend one field at a time."""
    fields = dict(
        id="t001",
        input="charged twice and the app crashes",
        expected_category="billing",
        acceptable_categories=["technical"],
        expected_summary="Charged twice; app crashes.",
        difficulty="hard",
        tags=["two-issues"],
        notes="test fixture",
    )
    return GoldenCase(**(fields | overrides))


def test_golden_file_loads():
    dataset = load()
    assert len(dataset.cases) >= 15


def test_tags_survive_loading():
    case = next(c for c in load().cases if c.id == "c002")
    assert "two-issues" in case.tags


def test_unknown_field_is_rejected():
    with pytest.raises(ValidationError):
        make_case(tgas=["typo"])


def test_duplicate_ids_are_rejected():
    with pytest.raises(ValidationError, match="duplicate case ids"):
        GoldenDataset(version=1, guideline_version=1, cases=[make_case(), make_case()])


@pytest.mark.parametrize(
    ("got", "expected_verdict"),
    [
        ("billing", "exact"),
        ("technical", "acceptable"),
        ("account", "missed"),
        ("general", "missed"),
    ],
)
def test_verdict(got, expected_verdict):
    assert make_case().verdict(got) == expected_verdict


def test_expected_is_never_also_listed_as_acceptable():
    for case in load().cases:
        assert case.expected_category not in case.acceptable_categories, case.id