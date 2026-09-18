"""The split assignment is ordinary code with a right answer."""

from email_triage.labels import LabelSet, SummaryLabel, assign_splits, load_labels, save_labels


def keys(n: int) -> list[tuple[str, str]]:
    return [(run, f"c{i:03d}") for run in ("v1", "v2") for i in range(1, n // 2 + 1)]


def test_splits_are_deterministic():
    assert assign_splits(keys(30)) == assign_splits(list(reversed(keys(30))))


def test_split_proportions():
    from collections import Counter

    counts = Counter(assign_splits(keys(30)).values())
    assert counts == {"train": 6, "dev": 12, "test": 12}


def test_every_key_gets_exactly_one_split():
    splits = assign_splits(keys(30))
    assert set(splits) == set(keys(30))
    assert set(splits.values()) <= {"train", "dev", "test"}


def test_labels_round_trip(tmp_path):
    path = tmp_path / "labels.json"
    labels = LabelSet(version=1, rubric_version=1, labels=[
        SummaryLabel(run="v1", case_id="c005", summary="…order…", passed=False,
                     critique="Invents an order.", split="dev"),
    ])
    save_labels(labels, path)
    assert load_labels(path) == labels


def test_missing_file_is_an_empty_set(tmp_path):
    assert load_labels(tmp_path / "nope.json").labels == []