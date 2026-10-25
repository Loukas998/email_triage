"""A run saved under one name and read under another fails silently: the reader finds an older file."""

from email_triage import runs
from email_triage.runs import Run


def test_a_saved_run_loads_back_by_its_prompt_name(tmp_path, monkeypatch):
    monkeypatch.setattr(runs, "RUNS_DIR", tmp_path)
    run = Run(prompt="v9", prompt_version=9, model="qwen3:8b", think=True,
              dataset_version=1, guideline_version=1, outputs=[])
    runs.save_run(run)
    assert runs.load_run("v9") == run