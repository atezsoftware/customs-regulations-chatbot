# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Agentic File Search is an AI-powered document search agent that explores files dynamically rather than using pre-computed embeddings. It uses a three-phase strategy: parallel scan, deep dive, and backtracking for cross-references. There is also a Postgres+pgvector-backed indexing pipeline for pre-indexed semantic+metadata retrieval — this is the only mode the deployed product uses (`backend` always talks to `core` indexed).

**Tech Stack:** Python 3.10+, model-agnostic LLM layer (Gemini today), LlamaIndex Workflows, Docling (document parsing), Postgres + pgvector (indexing/storage, shared with `backend`), langextract (optional metadata extraction), FastAPI + WebSocket, Typer + Rich CLI.

## Repo Layout

This repo is split into independent top-level projects:

- **`core/`** — the Python AI engine, itself a **uv workspace** split into three packages so the chat path never has to import Docling/langextract:
  - **`core/shared/`** (`fs_explorer_shared`) — code both services need: Postgres storage, embeddings, basic (non-Docling) filesystem helpers, the internal-call auth gate. No Docling, no langextract, no LLM provider layer.
  - **`core/api/`** (`fs_explorer_api`) — the chat/agent service. Workflow orchestration, the indexed-retrieval agent, `/ws/explore`, `/api/search`. Depends only on `shared`. Never imports Docling/langextract — this is what keeps its Docker image small and its process light.
  - **`core/indexer/`** (`fs_explorer_indexer`) — the indexing service. Docling document parsing, regulatory chunking, langextract metadata extraction, `/api/index*`. Depends only on `shared`. This is the one image that legitimately needs Docling/langextract (and is large/slow to start because of it).
  - All commands below are run from inside `core/` (the workspace root) unless noted.
- **`backend/`** — TypeScript API/backend service. Talks to `core-api` for chat (`CORE_INTERNAL_URL`, ws + `/api/search`) and to `core-indexer` for indexing (`CORE_INDEXER_URL`, `/api/index*`) — see `backend/src/modules/chat/services/core-bridge.service.ts` and `backend/src/modules/directories/services/core-index.service.ts`.
- **`frontend/`** — TypeScript frontend app.
- **`db/`** — shared database infra (`docker-compose.yml` for Postgres+pgvector; SQL migrations). `core` and `backend` both connect to this same Postgres instance — `core` (via `fs_explorer_shared.storage`) owns the `core_*` tables (documents/chunks/embeddings/schemas), `backend` owns everything else.

> **Deploy status:** `.github/workflows/customs-regulations-core-codebuild.yaml` is intentionally untouched by this split and still only builds/pushes one image, from `core/Dockerfile-aws` (the api/chat service — same path and same ECR repo/Helm release as before). `core/indexer/Dockerfile-aws` exists and is verified to build correctly, but nothing wires it into CI/CD yet. Before `core-indexer` can run anywhere: a new ECR repo per environment, a new EKS Deployment+Service (port 8001) in the `atezsoftware/devops` repo, a way to build/push that image (a new workflow, or extending the existing one), and `backend`'s deployed config needs `CORE_INDEXER_URL` pointed at it.

## Common Commands

Run from `core/` (`cd core` first) — this is the uv workspace root, shared by all three packages and their single `uv.lock`:

```bash
# Install dependencies for one package only (this is what keeps core-api's
# venv free of Docling/langextract — never use plain `uv sync` here, it
# only installs the dev tooling group since the workspace root itself
# isn't an installable package)
uv sync --package fs-explorer-api       # chat/agent service deps only
uv sync --package fs-explorer-indexer   # indexing service deps only (Docling, langextract)
uv sync --all-packages                  # everything, e.g. for local full-stack dev

# Run CLI (indexed query against an existing index)
uv run --package fs-explorer-api explore --task "What is the purchase price?" --folder data/test_acquisition/

# Run CLI (build/refresh an index, schema management — needs the indexer's deps)
uv run --package fs-explorer-indexer explore-index index data/test_acquisition/
uv run --package fs-explorer-indexer explore-index schema discover data/test_acquisition/
uv run --package fs-explorer-indexer explore-index schema show data/test_acquisition/

# Run the services
uv run --package fs-explorer-api uvicorn fs_explorer_api.server:app --host 127.0.0.1 --port 8000
uv run --package fs-explorer-indexer uvicorn fs_explorer_indexer.indexer_server:app --host 127.0.0.1 --port 8001

# Run tests (from core/, against whichever packages are synced)
uv run --all-packages pytest tests                  # everything (needs both packages' deps)
uv run --package fs-explorer-api pytest tests/api tests/shared    # api-only, no Docling needed
uv run --package fs-explorer-indexer pytest tests/indexer tests/shared  # indexer-only
uv run pytest tests/api/test_agent.py                # single file (after syncing the right package)
uv run pytest -k "test_name"                         # single test

# Lint, format, typecheck, build (also available via Makefile — these already
# loop/scope correctly across all 3 packages)
make test
make lint
make format
make format-check
make typecheck
make build
```

Entry points: `core/api/pyproject.toml` → `explore` = `fs_explorer_api.main:app` (indexed query CLI; `explore --task ... --folder ...` works without typing a subcommand name since `query` is the only command Typer registers), `explore-ui` = `fs_explorer_api.server:run_server`. `core/indexer/pyproject.toml` → `explore-index` = `fs_explorer_indexer.main:app` (`index`, `schema discover`, `schema show`), `explore-index-ui` = `fs_explorer_indexer.indexer_server:run_server`.

The non-indexed "raw filesystem" agentic explore mode (Docling-parsing a folder with no prior indexing step) was retired in the api/indexer split — `backend` never used it (it always operates indexed), and re-adding it would mean either duplicating the agent/workflow code into the indexer or making `agent.py` depend on Docling again, defeating the point of the split.

## Architecture

### Core Flow (Indexed Mode — the only mode `backend` uses)
```
User Query → core-api: Workflow → Agent → semantic_search/get_document → Postgres+pgvector → Ranked Results
```

### Indexing Flow
```
backend uploads a file → core-indexer: IndexingPipeline → Docling parse → RegulatoryChunker → (optional langextract metadata) → Postgres+pgvector
```

### Key Modules — `core/api/src/fs_explorer_api/` (chat/agent service)

- **workflow.py**: Event-driven orchestration using `llama-index-workflows`. Defines `FsExplorerWorkflow` with steps: `start_exploration`, `go_deeper_action`, `tool_call_action`, `receive_human_answer`. Uses singleton agent via `get_agent()`.

- **agent.py**: `FsExplorerAgent` talks to the LLM via the provider-agnostic `LLMClient` interface (`llm/base.py`; `llm/gemini.py` is the only implementation today). Chat history accumulates in `_chat_history` as a list of `ChatTurn` (provider-agnostic, not a Gemini-specific type). `take_action()` requests a structured JSON `Action`; `TokenUsage` tracks input/output/**thinking** tokens and costs. Also contains the `TOOLS` registry, `SYSTEM_PROMPT`, and indexed tool functions (`semantic_search`, `get_document`, `list_indexed_documents`). Index context is managed via `set_index_context()`/`clear_index_context()`, backed by `contextvars.ContextVar`s (not plain module globals — those raced across concurrently-running chats in the same process; see `_INDEX_CONTEXT_VAR`'s docstring in `agent.py`). Tool calls that miss the index (`preview_file`/`parse_file`/`scan_folder`) return a short "not indexed yet" message rather than falling back to raw Docling parsing — that capability lives only in `core-indexer` now.

- **llm/**: `base.py` defines `ChatTurn`/`LLMUsage`/`LLMClient` (Protocol); `gemini.py` implements it; `factory.py`'s `get_llm_client()` selects a provider via `FS_EXPLORER_LLM_PROVIDER`/`FS_EXPLORER_LLM_MODEL` env vars (only `"gemini"` registered today — this is the seam for swapping models later).

- **models.py**: Pydantic schemas for structured LLM output. `Action` contains one of: `ToolCallAction`, `GoDeeperAction`, `StopAction`, `AskHumanAction`. `Tools` TypeAlias defines all available tool names.

- **main.py**: Typer CLI — the `query` command (indexed agentic explore; also the implicit default command).

- **server.py**: FastAPI server with WebSocket endpoint `/ws/explore` for real-time streaming, plus `/api/search`, `/api/document`, `/api/folders` and the bundled demo UI (`ui.html`). Also `/api/index/document-chunks` — a pure-Postgres read (no Docling) duplicated from `core-indexer` so file-chunk lookups work wherever `core-api` is deployed, since `core-indexer` isn't wired into any deployed environment yet; `backend` calls this one on `core-api`, not `core-indexer`. All *other* `/api/index*` endpoints (build/refresh, embed, auto-profile) stay `core-indexer`-only.

- **exploration_trace.py**: Records tool call paths and extracts cited sources from final answers for the CLI summary.

- **search/**: `query.py`'s `IndexedQueryEngine` runs parallel semantic (chunk text matching) + metadata (JSON filter) retrieval paths using ThreadPoolExecutor, then merges and ranks via `RankedDocument.combined_score` (`ranker.py`). `filters.py`'s `parse_metadata_filters()` parses a human-readable filter DSL (`field=value`, `field>=num`, `field in (a, b)`, `field~substring`) into `MetadataFilter` objects.

### Key Modules — `core/indexer/src/fs_explorer_indexer/` (indexing service)

- **document_parser.py**: The Docling-dependent half of what used to be `fs.py` — `DocumentConverter`, `_DOCUMENT_CACHE` (keyed by `path:mtime`), `preview_file`/`parse_file`/`scan_folder`. The only module in either service that imports `docling`.

- **indexer_server.py**: FastAPI app with `/api/index`, `/api/index/embed`, `/api/index/auto-profile`, `/api/index/document-chunks`, `/api/index/status` — same request/response shapes `backend` always called, just on its own process now.

- **main.py**: Typer CLI (`explore-index`) — `index`, `schema discover`, `schema show`.

- **chunk_inspector.py**: standalone debug FastAPI app for previewing regulatory chunk boundaries on a single file, separate from the main pipeline.

- **indexing/pipeline.py**: `IndexingPipeline` orchestrates document parsing (via `document_parser.parse_file`) → chunking → metadata extraction → Postgres upsert. Walks a folder for supported files, delegates to `RegulatoryChunker` and `extract_metadata()`, handles schema resolution and deleted-file cleanup.

- **indexing/regulatory_chunker.py**: `RegulatoryChunker` — the production chunker. Structure-aware (MADDE/article, paragraph, clause, table) chunking with full locator metadata (`article_no`, `paragraph_no`, `parent_path`, `heading_path`, source char offsets) per chunk, persisted alongside the chunk row so every chunk is traceable to its source file.

- **indexing/chunker.py**: `SmartChunker` — naive paragraph-based chunker, kept for reference/tests; no longer wired into the production pipeline.

- **indexing/schema.py**: `SchemaDiscovery` auto-discovers metadata schemas from a corpus folder (file types, heuristic boolean fields like `mentions_currency`/`mentions_dates`). Optionally includes langextract fields.

- **indexing/metadata.py**: `extract_metadata()` produces per-document metadata dicts. Heuristic fields (filename, extension, document_type, currency/date detection) are always available. Optional langextract integration calls the `langextract` library for entity extraction (organizations, people, deal terms, etc.) via configurable profiles; LLM calls for this go straight through `google-genai` (not the `llm/` provider abstraction, which is api-only).

### Key Modules — `core/shared/src/fs_explorer_shared/` (used by both services)

- **storage/postgres.py**: `PostgresStorage` manages the `core_corpora`, `core_documents`, `core_chunks`, `core_schemas`, `core_chunk_embeddings` tables (schema owned by `db/migrations/`, not this file — `initialize()` is a no-op). Key operations: `upsert_document`, `search_chunks` (keyword-based scoring), `search_documents_by_metadata` (JSON filtering via `->>`), `search_chunks_semantic` (pgvector cosine distance, `<=>`), schema CRUD. Corpus/doc/chunk IDs are SHA1-based stable hashes (kept deterministic on purpose — re-indexing the same file must upsert the same row, not create a new one).

- **storage/base.py**: `StorageBackend` protocol, shared dataclasses (`DocumentRecord`, `ChunkRecord`, `SchemaRecord`), and the shared id helpers (`stable_id`, `make_document_id`, `make_chunk_id`).

- **fs.py**: Non-Docling filesystem helpers only — `read_file`, `grep_file_content`, `glob_paths`, `describe_dir_content`, `SUPPORTED_EXTENSIONS`. Importing this module never pulls in Docling.

- **embeddings.py**: `EmbeddingProvider` wraps `google-genai` directly (lightweight — no heavy ML deps), used by `core-api`'s `/api/search` and `core-indexer`'s `/api/index/embed`.

- **index_config.py**: `resolve_database_url()` resolves the Postgres connection string with precedence: CLI `--database-url` > `DATABASE_URL` env. `corpus_root()` normalizes a physical or virtual corpus identifier the same way storage does.

- **auth.py**: `internal_token_valid()`/`require_internal_token` — the shared `CORE_INTERNAL_TOKEN` gate both services' REST endpoints use.

### Workflow Event Types
- `InputEvent` → starts exploration
- `ToolCallEvent` → tool execution
- `GoDeeperEvent` → directory navigation
- `AskHumanEvent`/`HumanAnswerEvent` → human interaction
- `ExplorationEndEvent` → completion with `final_result` or `error`

### Adding New Tools
1. Implement function in `core/shared/src/fs_explorer_shared/fs.py` (non-Docling) or `core/api/src/fs_explorer_api/agent.py` (indexed) returning `str`
2. Add to `TOOLS` dict in `agent.py`
3. Add to `Tools` TypeAlias in `models.py`
4. Update `SYSTEM_PROMPT` in `agent.py`
5. Update `TOOL_ICONS` and `PHASE_DESCRIPTIONS` in `main.py`

## Environment

- `GOOGLE_API_KEY` (required for the Gemini provider, both services) — in a `.env` file (`core/api/.env` / `core/indexer/.env`, each service loads its own) or environment variable
- `DATABASE_URL` (required, both services) — Postgres connection string, shared with `backend`
- `CORE_INTERNAL_TOKEN` (optional, both services) — if set, gates `/ws/explore` and the indexing/search REST endpoints behind a shared secret (checked via `X-Internal-Token` header or the WS message's `internal_token` field); unset = no gate, for local/CLI use
- `FS_EXPLORER_LLM_PROVIDER` / `FS_EXPLORER_LLM_MODEL` (optional, `core-api`) — select/override the LLM provider+model (default: gemini / `gemini-3-flash-preview`)
- `FS_EXPLORER_LLM_MAX_CONCURRENCY` (optional, `core-api`) — process-wide cap on concurrent Gemini calls (default 8), enforced by a module-level `asyncio.Semaphore` in `llm/gemini.py` shared by every `GeminiLLMClient` instance (the singleton chat agent and any per-request client, e.g. the amendments pipeline); under concurrent chatbot traffic this queues excess calls instead of firing them all at once and hitting `429 RESOURCE_EXHAUSTED`
- `FS_EXPLORER_LLM_CALL_TIMEOUT_SECONDS` (optional, `core-api`) — per-call ceiling in `llm/gemini.py` (default 90s), applied to every individual Gemini call: each tool-planning step, each context-summary compaction, and (per streamed chunk) the final-answer stream. Without it a single stalled call could hang indefinitely, holding a concurrency slot forever; a stuck call now fails with a clear timeout instead of the WebSocket connection just going quiet until something upstream closes it
- `FS_EXPLORER_LLM_RETRY_ATTEMPTS` / `FS_EXPLORER_LLM_RETRY_BACKOFF_SECONDS` (optional, `core-api`, default 3 / 2s) — in `llm/gemini.py`, both `generate_structured()` (tool-planning/compaction calls) and `stream_text()` (the final-answer stream) retry on `429 RESOURCE_EXHAUSTED`, `503 UNAVAILABLE`, and per-call timeouts, waiting `FS_EXPLORER_LLM_RETRY_BACKOFF_SECONDS` between attempts instead of failing the whole run. `stream_text()` only retries if no text has been yielded to the caller yet — once real answer text has streamed out, a later failure is not retried (would risk duplicated/out-of-order output) and instead propagates so the caller's fallback path handles it
- `FS_EXPLORER_CONTEXT_SUMMARY_THRESHOLD` (optional, `core-api`) — fraction of `GEMINI_MAX_CONTEXT_TOKENS` (1,048,576) at which `agent.py`'s `_maybe_summarize_history` proactively compacts the mid-run chat history (default 0.85), instead of waiting to hit the ceiling and error out. This is the crash-prevention backstop only — see `FS_EXPLORER_CONTEXT_SUMMARY_MAX_TOKENS` for the cost-driven trigger
- `FS_EXPLORER_CONTEXT_SUMMARY_MAX_TOKENS` (optional, `core-api`) — absolute token size (default 8000) at which `_maybe_summarize_history` compacts, checked in addition to the ratio above (whichever fires first). Since every `take_action()` call resends the *entire* history, this is what actually bounds cost on a long run instead of only preventing a crash near the 1M ceiling
- `FS_EXPLORER_MAX_STEPS` (optional, `core-api`) — hard ceiling (default 10) on total agent decision-points (`take_action()` calls) in one run; once reached, the agent is forced to stop (no extra LLM call spent on the decision) and compose its final answer from whatever it has gathered
- `FS_EXPLORER_DEEP_READ_MAX_CHARS` (optional, `core-api`) — cap (default 15000) on how many characters `read`/`parse_file`/`get_document` can inject into chat history in one call; real indexed documents can run past a million characters, and these tools resend their full output on every subsequent step
- `FS_EXPLORER_GREP_MAX_MATCH_LINES` (optional, `core-api`) — cap (default 25) on how many matched-chunk lines a single `grep` call renders; an uncapped corpus-wide pattern (`file_path="all"`) can match a large fraction of every chunk in the corpus (55% of a real 30K-chunk corpus in testing, with no word-boundary anchoring)
- `FS_EXPLORER_LANGEXTRACT_MAX_CHARS` (optional, `core-indexer`) — max chars sent to langextract (default 6000)
- `FS_EXPLORER_LANGEXTRACT_MODEL` (optional, `core-indexer`) — model for langextract (default `gemini-3-flash-preview`)
- `backend` additionally needs `CORE_INTERNAL_URL` (→ `core-api`, default `ws://127.0.0.1:8000`) and `CORE_INDEXER_URL` (→ `core-indexer`, default `http://127.0.0.1:8001`) — see `.env.dev.example` at the repo root.

## Testing

`core/tests/` mirrors the package split:
- `tests/api/` — agent/workflow/models/exploration-trace/CLI tests against `fs_explorer_api`. No Postgres or Docling needed except where noted.
- `tests/indexer/` — indexing pipeline, chunker, schema, CLI, and `indexer_server` tests against `fs_explorer_indexer`. Needs Docling installed; Postgres-backed tests skip without it.
- `tests/shared/` — `fs_explorer_shared` (fs helpers, embeddings) tests.
- `tests/integration/` — tests that need **both** packages installed (e.g. seeding a corpus through the indexer's real `IndexingPipeline`, then querying it through the api's `/api/search`). Run these with `uv run --all-packages pytest tests/integration`.
- `tests/conftest.py` (root) only defines the `database_url` fixture (skips if `DATABASE_URL` isn't set) — kept free of any `fs_explorer_*` import so it loads safely regardless of which package's environment is active. `tests/api/conftest.py` holds the `MockGenAIClient`/`make_mock_llm_client()` mocks (api-only, since only the chat agent calls an LLM through the `llm/` abstraction).

Point `DATABASE_URL` at a throwaway database with the `db/migrations/` schema applied for the Postgres-backed tests.

Test documents live in `data/test_acquisition/` and `data/large_acquisition/`. Test fixtures for unit tests are in `tests/testfiles/`.
