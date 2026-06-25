# fs-explorer (core)

AI-powered document search agent for regulatory documents. Explores files
dynamically (agentic mode) or queries a pre-built Postgres+pgvector index
(indexed mode), citing every claim back to its source article/clause.

See [`../CLAUDE.md`](../CLAUDE.md) for architecture, commands, and environment
variables.

## Setup

With [uv](https://docs.astral.sh/uv/getting-started/installation/) (recommended — `scripts/run.sh` and the `Makefile` both call `uv run`):

```bash
cd core
uv pip install -e ".[dev]"
```

Without `uv`, a plain venv works too, but then you'll run `pytest`/`ruff`/`ty` directly instead of via `uv run`, and `scripts/run.sh` needs a venv at the repo root (not `core/.venv`) to find it automatically. Note `[dependency-groups].dev` (PEP 735) is *not* a pip extra — `pip install -e ".[dev]"` silently skips it, so install the dev group separately (needs pip ≥ 25.1):

```bash
python3 -m venv .venv          # from the repo root, not from core/
.venv/bin/pip install -e ./core
.venv/bin/pip install --group core/pyproject.toml:dev   # pytest, ruff, ty, pre-commit
```

## Run

```bash
uv run explore --task "What is the purchase price?" --folder data/test_acquisition/
uv run uvicorn fs_explorer.server:app --host 127.0.0.1 --port 8000
```
