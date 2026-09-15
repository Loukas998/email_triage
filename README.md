# Email Triage

A small local-LLM pipeline that classifies customer support emails into categories and writes a one-sentence summary. Prompts are versioned YAML files, outputs are validated with Pydantic, and a golden dataset tracks how well each prompt performs.

## What it does

Given a support email, the system:

1. Sends it to a local model via [Ollama](https://ollama.com/)
2. Parses a structured JSON response with:
   - **category** — one of `billing`, `technical`, `account`, or `general`
   - **summary** — a one-sentence summary (max 200 chars)
   - **confidence** — 1–5
3. Compares the result against a hand-labelled golden set of 16 cases

Each golden case has an expected category, optional acceptable alternatives, an expected summary, and a difficulty (`easy`, `medium`, `hard`). A prediction is scored as **exact**, **acceptable**, or **missed**.

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) for dependency management
- [Ollama](https://ollama.com/) running locally
- The model used by the prompts (currently `qwen3:8b`):

  ```bash
  ollama pull qwen3:8b
  ```

## Setup

```bash
git clone https://github.com/Loukas998/email_triage.git
cd email_triage
uv sync --dev
```

## Running

### Score a prompt against the golden set

Prints every case with its verdict and a summary tally:

```bash
uv run python -m email_triage.check v1
uv run python -m email_triage.check v2
```

Legend: blank = exact, `~` = acceptable, `X` = missed, `!` = validation error.

### Triage a single email (Python)

```python
from email_triage.triage import triage
from email_triage.prompts import load_prompt

config = load_prompt("v2")
result = triage("I was charged twice this month.", config)
print(result.category, result.summary)
```

## Prompts

Prompt configs live in `prompts/` as versioned YAML files. Each file records the model, temperature, thinking mode, and system prompt — everything that changes model behaviour.

| File | Notes |
|------|-------|
| `prompts/v1.yaml` | Baseline prompt |
| `prompts/v2.yaml` | Adds explicit category definitions |

## Tests

Fast unit tests (dataset schema, prompt loading):

```bash
uv run pytest -m "not eval"
```

Slow eval gate — calls the local model for every golden case. Only run when you intend to:

```bash
uv run pytest -m eval              # uses v1 prompt (default)
PROMPT=v2 uv run pytest -m eval    # uses v2 prompt
```

The eval test asserts that no **easy** cases are **missed** for the chosen prompt.

## Project layout

```
golden/v1.json          # labelled evaluation cases
prompts/v1.yaml         # prompt configs
prompts/v2.yaml
src/email_triage/
  triage.py             # Ollama call + Pydantic schema
  prompts.py            # load versioned prompt YAML
  dataset.py            # golden set loader + verdict logic
  check.py              # run all cases, print report
tests/                  # unit + eval tests
```
