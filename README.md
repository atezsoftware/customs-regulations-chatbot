# Customs Regulations Chatbot



An AI assistant that answers questions about regulatory documents (gümrük tebliğleri, genelgeler, kanunlar, ...) by either exploring files agentically like a human reader, or querying a pre-built semantic + metadata index — with every claim in the answer backed by a citation back to its source article/clause.

## Repo layout

This is a monorepo split into four independent projects, plus shared db infra:

- **`core/`** — Python AI engine, split into two services: `core-api` (chat/agent, indexed retrieval) and `core-indexer` (Docling parsing + chunking + embedding pipeline). See [`CLAUDE.md`](CLAUDE.md).
- **`backend/`** — TypeScript API (LoopBack 4). See [`backend/README.md`](backend/README.md).
- **`frontend/`** — TypeScript/React web app. See [`frontend/README.md`](frontend/README.md).
- **`db/`** — shared Postgres+pgvector docker-compose setup and SQL migrations, used by both `core` and `backend`. See [`db/README.md`](db/README.md).

## Prerequisites

Install these once, before the first `scripts/run.sh` run:

- **Docker + Docker Compose** — runs Postgres. If `docker ps` gives `permission denied`, either run the script with `sudo` (works, but see the warning below) or add yourself to the `docker` group once and re-login: `sudo usermod -aG docker $USER`.
- **Node.js 20+ and npm** — for `db` (migrations), `backend`, `frontend`.
- **Python 3.10+** and **[uv](https://docs.astral.sh/uv/getting-started/installation/)** — for `core` (a uv workspace; see [core/README.md](core/README.md)). `scripts/run.sh` and `core/Makefile` both assume `uv` is on `PATH`.

> **Don't run `scripts/run.sh` with `sudo` if you can avoid it.** The script also runs `npm install`/`pip install` as a side effect of starting `backend`/`frontend`/`db` migrations. If it's invoked under `sudo`, those installs run as `root` and leave `root`-owned files in `*/node_modules` — later, non-sudo `npm install` runs fail with `EACCES`, or get silently skipped because the (incompletely-installed) `node_modules` directory already exists. If you must use `sudo` (e.g. you're not in the `docker` group yet), run `npm install` by hand in `db/`, `backend/`, `frontend/` as your normal user *first*, so `scripts/run.sh` finds `node_modules` already populated and never tries to install as root.

## Quick start

```bash
# 1. Copy and fill in the env files (one set per environment)
cp .env.dev.example .env.dev
cp db/.env.dev.example db/.env.dev
# fill in OPENROUTER_API_KEY and the DB_*/POSTGRES_* secrets
# (the two files must agree: db/.env.dev's POSTGRES_USER/PASSWORD/DB must match
# the DB_USER/DB_PASSWORD/DB_NAME in the root .env.dev)

# 2. Install dependencies for each app once (as your normal user, not sudo)
(cd db && npm install)
(cd backend && npm install)
(cd frontend && npm install)
(cd core && uv sync --all-packages)
# see core/README.md for syncing just one of the two core services

# 3. Bring up the whole stack (db + core-api + core-indexer + backend + frontend) in dev mode
scripts/run.sh --env dev --apps all
```

`scripts/run.sh` starts Postgres via docker compose, runs pending migrations, and launches `core-api`, `core-indexer`, `backend`, and `frontend` with the right env vars wired together. You can start a subset instead, e.g. `scripts/run.sh --env dev --apps db,backend` (useful when iterating on one layer while the others stay up).

Once running:
- Frontend: http://localhost:5173 (or whatever Vite prints)
- Backend API: http://localhost:3000
- Core API (internal, not meant for direct browser use): http://127.0.0.1:8000
- Core Indexer (internal, not meant for direct browser use): http://127.0.0.1:8001

In `development`, the backend auto-creates and logs in a fixed local user — no registration required to start clicking around.

### Troubleshooting

- **`POSTGRES_USER: unbound variable` when starting `db`** — `db/.env.dev` is missing or doesn't define `POSTGRES_USER`/`POSTGRES_PASSWORD`/`POSTGRES_DB`. Copy `db/.env.dev.example` → `db/.env.dev` and fill it in; it's a *separate* file from the root `.env.dev` because the Postgres docker image specifically needs `POSTGRES_*` names.
- **`sh: 1: node-pg-migrate: not found` when running migrations** — `db/node_modules` exists but is incomplete (an earlier `npm install` was interrupted, e.g. by a network timeout or Ctrl-C). The script only runs `npm install` when `node_modules` doesn't exist at all, so a partial install never gets retried automatically. Fix: `rm -rf db/node_modules && (cd db && npm install)`.
- **`npm ERR! code EACCES` / `permission denied` inside `node_modules`** — a previous run installed as `root` (typically because the whole script ran under `sudo`). Fix the ownership or just delete and reinstall as your normal user: `sudo rm -rf <app>/node_modules && (cd <app> && npm install)`.
- **`npm ERR! ETIMEDOUT` reaching `registry.npmjs.org`** — usually a transient network hiccup during the first (uncached) install of a large dependency tree. Just re-run `npm install`.

## How a document goes from upload to answer

1. **Upload** — files of any type are uploaded per-directory in the frontend (`backend` stores them under `STORAGE_ROOT`, metadata in Postgres).
2. **Generate chunks** — `core-indexer` parses each file (Docling), runs it through `RegulatoryChunker` (structure-aware: madde/article, paragraph, clause, table, with full locator metadata), and writes documents + chunks to Postgres. Raw uploads are deleted once their text is safely chunked into the database.
3. **Start indexing** — `core-indexer` embeds the chunks that were just generated (Gemini embeddings) and stores the vectors in pgvector, enabling semantic search. This step is separate from chunk generation on purpose: chunks can be inspected/regenerated cheaply without re-spending on embeddings, and indexing only ever has to embed chunks that don't have a vector yet.
4. **Chat** — a chat session is linked to one or more directories (linking at least one is required). Each message goes to `core-api`'s agent, which searches that directory's indexed chunks semantically and/or by metadata filter, then answers with inline citations (`[Belge Adı, Madde X]`) and a sources list.

### Why not just one big "index" button?

Parsing+chunking and embedding are deliberately two steps so a stalled or misconfigured embedding provider never forces you to redo the (slower, Docling-based) parsing/chunking pass — chunks are durable in Postgres as soon as "Generate chunks" finishes, regardless of what happens next. Directories that only have chunks (no embeddings yet) still work in chat: search falls back to keyword matching over chunk text until embeddings are generated.

## Configuration

Environment variables are split per app (see each app's README/`.env.*.example` for the full list), but the ones you'll touch first:

| Variable | Where | Purpose |
|----------|-------|---------|
| `OPENROUTER_API_KEY` | `core-api`, `backend` | Required for chat inference (`core-api`) and model catalog sync (`backend`). |
| `FS_EXPLORER_LLM_PROVIDER` | `core-api` | Active chat provider. Set to `openrouter` for the new model selector flow. |
| `OPENROUTER_DEFAULT_MODEL` | `core-api`, `backend` | Default model for new sessions and provider fallback (`google/gemini-3.6-flash`). |
| `OPENROUTER_CATALOG_SYNC_MINUTES` | `backend` | How often the backend refreshes the available OpenRouter model catalog. |
| `FS_EXPLORER_MULTI_AGENT_ENABLED` | `core-api` | Enables hierarchical multi-agent indexed research. The API server and Core Compose default to `true`; direct library usage safely defaults to `false`. Set `false` as the production kill switch. |
| `FS_EXPLORER_PLANNER_{PROVIDER,MODEL,REASONING}` | `core-api` | Global planner policy. Defaults to OpenRouter, `openai/gpt-5.6-sol`, `medium`. |
| `FS_EXPLORER_TASK_{PROVIDER,MODEL,REASONING}` | `core-api` | Task coordinator policy. Defaults to OpenRouter, `google/gemini-3.6-flash`, `medium`. |
| `FS_EXPLORER_WORKER_{PROVIDER,MODEL,REASONING}` | `core-api` | Search worker policy. Defaults to OpenRouter, `google/gemini-3.5-flash-lite`, `low`. |
| `FS_EXPLORER_FINAL_{PROVIDER,MODEL,REASONING}` | `core-api` | Final synthesis policy. Defaults to OpenRouter, `google/gemini-3.6-flash`, `high`. |
| `FS_EXPLORER_MULTI_AGENT_MAX_{TASKS,WORKERS_PER_TASK,WORKER_ROUNDS,TOTAL_WORKERS,LLM_CALLS}` | `core-api` | Hard per-run fan-out/call budgets. Defaults: `5`, `3`, `2`, `8`, `24`. |
| `FS_EXPLORER_MULTI_AGENT_MAX_ARTIFACT_ITEMS` | `core-api` | Maximum structural items per typed plan/artifact list (`12`), preventing scenario fan-out from expanding downstream contexts. |
| `FS_EXPLORER_MULTI_AGENT_MAX_ARTIFACT_CONTEXT_CHARS` | `core-api` | Total serialized artifact-context cap per boundary (`16000`), independent of per-field limits. |
| `FS_EXPLORER_MULTI_AGENT_MAX_{QUESTION,PLANNER_CONTEXT,FINAL_CONTEXT}_CHARS` | `core-api` | Hard user-input and aggregate synthesis-context caps. Defaults: `8000`, `16000`, `48000`; the current question is retained before older conversation context. |
| `FS_EXPLORER_MULTI_AGENT_SEARCH_TIMEOUT_SECONDS` | `core-api` | Hard wait limit for each indexed search attempt and task-global rerank (`20`). |
| `FS_EXPLORER_MULTI_AGENT_LLM_TIMEOUT_SECONDS` | `core-api` | Hard wait limit for each planner/coordinator/worker/reviewer LLM stage (`120`). |
| `GOOGLE_API_KEY` | `core-api`, `core-indexer` | Gemini LLM + embeddings. Get one at [Google AI Studio](https://aistudio.google.com/apikey). |
| `DATABASE_URL` | `core-api`, `core-indexer`, `backend` | Shared Postgres connection string. |
| `CORE_INTERNAL_TOKEN` | `core-api`, `core-indexer`, `backend` | Shared secret gating both core services' internal REST/WebSocket endpoints so only `backend` can call them. |
| `CORE_INTERNAL_URL` | `backend` | Where `core-api` lives (`/ws/explore`, `/api/search`). |
| `CORE_INDEXER_URL` | `backend` | Where `core-indexer` lives (`/api/index*`). |
| `DATASET_MANAGEMENT_ENABLED` | `backend` | Opt-in admin-only dataset panel and its upload/chunk/index actions. Set to `true` only after `core-indexer` is deployed and reachable through `CORE_INDEXER_URL`. |
| `JWT_SECRET` | `backend` | Signs access tokens. |
| `STORAGE_ROOT` | `backend` | Where uploaded files live on disk before/while being chunked. |
| `VITE_API_URL` | `frontend` | Where the backend API lives. |

## Architecture

```
User question
     │
     ▼
GPT-5.6 Sol global planner ── precise lookup ───▶ one search + evidence worker
     │
     ├── coherent but multi-search question ───▶ one adaptive evidence task
     │
     └── scenario / multi-part question ────────▶ typed evidence/application DAG
                                                       │
                           shared evidence tasks run concurrently
                                                       ▼
                                     Gemini 3.6 task coordinators
                                                       │
                                      1–3 isolated search assignments
                                                       ▼
                                   Gemini 3.5 Flash Lite workers
                                                       │
                                  Postgres + pgvector indexed chunks
                                                       ▼
                                server-verified evidence claims
                                                       │
                         scenario facts + claim IDs only (no new search)
                                                       ▼
                             Gemini 3.6 application/integration tasks
                                                       │
                                                       ▼
                                    Gemini 3.6 final synthesis
                                                       │
                                                       ▼
                              scenario-specific cited answer + provenance
```

The global planner performs routing, intent recognition, and decomposition in
the same call; there is no separate intent-analysis request. Its versioned plan
maps every required answer to stable requirement IDs. Scenario plans also map
the user's facts, material unknowns, and decision branches to evidence,
application, and integration tasks. Deterministic validation rejects
topic-outline plans, uncovered required outputs, invalid references, scenario
plans without an application step, and unsafe `single_pass` routing before any
task runs.

Every search worker receives only its own assignment and retrieved hits.
Application tasks cannot search: they receive only the scenario facts assigned
to that task and compact artifacts from declared dependencies. Their
conclusions must reference existing fact, branch, claim, and dependency-output
IDs; the server rejects ungrounded references before final synthesis. Agents
exchange these compact typed artifacts instead of chat transcripts.
Task/worker fan-out, rounds, concurrency, total LLM calls (including one
reserved final-synthesis call), claims, user input, and aggregate final context
are all server-bounded. Invalid plans fall back to one direct task, and
an unrecoverable multi-agent failure falls back to the legacy stateless indexed
retrieval path.

For a precise one-query lookup, the planner selects `single_pass`. The server
then skips the redundant task coordinator and reviewer calls after verified
evidence covers every typed requirement (planner + worker + final synthesis).
If that first lookup leaves a gap, the same task automatically upgrades to the
normal adaptive second wave. Scenario and comparison plans are never eligible
for this shortcut.

Each task deduplicates candidates across its search assignments and reranks the
union against the task question before review. Evidence/context budgets are
shared fairly across tasks and sources. In-flight indexed searches and
structured LLM operations have stable identities, hard wait limits, and cached
results, so a WebSocket reconnect resumes completed work without paying for
the same operation again.

Benchmark runs keep the existing single-candidate mode and also support a
`production_roles` profile that evaluates the real heterogeneous
planner/task/worker/final model policy. Each item stores a bounded versioned
plan trace plus per-role token and cost usage. The answer judge receives only
evidence chunks that the final answer actually cited, capped at six excerpts
and 3,000 characters total.

`core-api` never imports Docling — it only ever reads chunks/embeddings that
`core-indexer` already wrote to Postgres. See [ARCHITECTURE.md](ARCHITECTURE.md)
for more detail (note: it predates the api/indexer split and still describes
the single-process agentic-exploration design; `core/CLAUDE.md` is current).
