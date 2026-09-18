"""The agreement maths has a right answer. Test it before trusting a single judge number."""

import pytest

from email_triage.agreement import measure


def test_hamel_example_always_pass_judge():
    """5% failures, judge always says pass: 95% agreement and it caught nothing."""
    human = [True] * 19 + [False]
    judge = [True] * 20
    a = measure(human, judge)
    assert a.agreement == 0.95
    assert a.tpr == 0.0
    assert a.tnr == 1.0


def test_perfect_judge():
    human = [True, False, True, False]
    a = measure(human, human)
    assert (a.agreement, a.tpr, a.tnr) == (1.0, 1.0, 1.0)


def test_judge_that_fails_everything():
    human = [True, True, False]
    a = measure(human, [False, False, False])
    assert a.tpr == 1.0  # caught the one real failure...
    assert a.tnr == 0.0  # ...by failing every pass too


def test_tpr_is_undefined_when_nothing_failed():
    a = measure([True, True], [True, False])
    assert a.tpr is None
    assert a.tnr == 0.5


def test_mismatched_lengths_are_an_error():
    with pytest.raises(ValueError, match="2 human labels but 3"):
        measure([True, False], [True, False, True])