# fs-explorer (core)

AI-powered document search agent for regulatory documents, queried against a
pre-built Postgres+pgvector index, citing every claim back to its source
article/clause. A uv workspace split into three packages: `shared` (storage,
embeddings, basic fs helpers), `api` (chat/agent service, no Docling), and
`indexer` (Docling/langextract indexing service).

See [`../CLAUDE.md`](../CLAUDE.md) for architecture, commands, and environment
variables.

## Setup

With [uv](https://docs.astral.sh/uv/getting-started/installation/) (required — `scripts/run.sh` and the `Makefile` both call `uv run`):

```bash
cd core
uv sync --package fs-explorer-api       # chat/agent service only
uv sync --package fs-explorer-indexer   # indexing service only (pulls in Docling/langextract)
uv sync --all-packages                  # both, for full local dev
```

Always sync a specific `--package` (or `--all-packages`) — a bare `uv sync` only
installs the workspace's dev tooling group, since the workspace root itself
isn't an installable package.

## Run

```bash
uv run --package fs-explorer-api uvicorn fs_explorer_api.server:app --host 127.0.0.1 --port 8000
uv run --package fs-explorer-indexer uvicorn fs_explorer_indexer.indexer_server:app --host 127.0.0.1 --port 8001

uv run --package fs-explorer-api explore --task "What is the purchase price?" --folder data/test_acquisition/
uv run --package fs-explorer-indexer explore-index index data/test_acquisition/
```
