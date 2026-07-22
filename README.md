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
| `OPENROUTER_DEFAULT_MODEL` | `core-api`, `backend` | Default model for new sessions and provider fallback (`google/gemini-3-flash-preview`). |
| `OPENROUTER_CATALOG_SYNC_MINUTES` | `backend` | How often the backend refreshes the available OpenRouter model catalog. |
| `GOOGLE_API_KEY` | `core-api`, `core-indexer` | Gemini LLM + embeddings. Get one at [Google AI Studio](https://aistudio.google.com/apikey). |
| `DATABASE_URL` | `core-api`, `core-indexer`, `backend` | Shared Postgres connection string. |
| `CORE_INTERNAL_TOKEN` | `core-api`, `core-indexer`, `backend` | Shared secret gating both core services' internal REST/WebSocket endpoints so only `backend` can call them. |
| `CORE_INTERNAL_URL` | `backend` | Where `core-api` lives (`/ws/explore`, `/api/search`). |
| `CORE_INDEXER_URL` | `backend` | Where `core-indexer` lives (`/api/index*`). |
| `JWT_SECRET` | `backend` | Signs access tokens. |
| `STORAGE_ROOT` | `backend` | Where uploaded files live on disk before/while being chunked. |
| `VITE_API_URL` | `frontend` | Where the backend API lives. |

## Architecture

```
                    core-api                              core-indexer
User Query             ↓                                       ↑
    ↓            ┌─────────────────┐                  ┌─────────────────┐
    └──────────→ │ Workflow Engine │                  │ IndexingPipeline │
                 │  (LlamaIndex)   │                  │  (Docling parse  │
                 └────────┬────────┘                  │  + chunking +    │
                          ↓                            │  embedding)      │
                 ┌─────────────────┐                  └────────┬────────┘
                 │     Agent       │ ←→ Gemini (structured JSON)│
                 └────────┬────────┘                            ↓
                          ↓                              ┌─────────────┐
            semantic_search / get_document  ──────────→  │  Postgres   │
                                                          │  + pgvector │
                                                          └─────────────┘
```

`core-api` never imports Docling — it only ever reads chunks/embeddings that
`core-indexer` already wrote to Postgres. See [ARCHITECTURE.md](ARCHITECTURE.md)
for more detail (note: it predates the api/indexer split and still describes
the single-process agentic-exploration design; `core/CLAUDE.md` is current).
