# Customs Regulations Chatbot

> **Based on**: [run-llama/fs-explorer](https://github.com/run-llama/fs-explorer) — the original CLI agent for filesystem exploration, extended here into a full product for querying Turkish customs regulations.

An AI assistant that answers questions about regulatory documents (gümrük tebliğleri, genelgeler, kanunlar, ...) by either exploring files agentically like a human reader, or querying a pre-built semantic + metadata index — with every claim in the answer backed by a citation back to its source article/clause.

## Repo layout

This is a monorepo split into four independent projects:

| App | Tech | Role |
|-----|------|------|
| [`core/`](core/) | Python (FastAPI, LlamaIndex Workflows, Docling, Gemini) | The AI engine: agentic document exploration, the regulatory chunker/indexing pipeline, and semantic+metadata search. Talked to over an internal WebSocket/REST API — never exposed directly to browsers. |
| [`backend/`](backend/) | TypeScript (LoopBack 4) | User-facing API: auth, directories/file uploads, chat sessions, and the bridge that streams `core`'s agent output to the frontend. |
| [`frontend/`](frontend/) | TypeScript (React + Vite + Tailwind) | The web UI: login/register, directory & file management, chunk inspector, and the chat interface. |
| [`db/`](db/) | Postgres + pgvector, SQL migrations | Shared database. `core` owns the `core_*` tables (documents/chunks/embeddings/schemas); `backend` owns everything else (users, directories, chat sessions). |

Each app has its own README with the details relevant to that layer — this file covers what ties them together.

## Quick start

```bash
# 1. Copy and fill in the env files (one set per environment)
cp .env.dev.example .env.dev
cp db/.env.dev.example db/.env.dev
# fill in GOOGLE_API_KEY (https://aistudio.google.com/apikey) and the DB_*/POSTGRES_* secrets

# 2. Bring up the whole stack (db + core + backend + frontend) in dev mode
scripts/run.sh --env dev --apps all
```

`scripts/run.sh` starts Postgres via docker compose, runs pending migrations, and launches `core`, `backend`, and `frontend` with the right env vars wired together. You can start a subset instead, e.g. `scripts/run.sh --env dev --apps db,backend` (useful when iterating on one layer while the others stay up).

Once running:
- Frontend: http://localhost:5173 (or whatever Vite prints)
- Backend API: http://localhost:3000
- Core (internal, not meant for direct browser use): http://127.0.0.1:8000

In `development`, the backend auto-creates and logs in a fixed local user — no registration required to start clicking around.

## How a document goes from upload to answer

1. **Upload** — files of any type are uploaded per-directory in the frontend (`backend` stores them under `STORAGE_ROOT`, metadata in Postgres).
2. **Generate chunks** — parses each file (Docling), runs it through `RegulatoryChunker` (structure-aware: madde/article, paragraph, clause, table, with full locator metadata), and writes documents + chunks to Postgres. Raw uploads are deleted once their text is safely chunked into the database.
3. **Start indexing** — embeds the chunks that were just generated (Gemini embeddings) and stores the vectors in pgvector, enabling semantic search. This step is separate from chunk generation on purpose: chunks can be inspected/regenerated cheaply without re-spending on embeddings, and indexing only ever has to embed chunks that don't have a vector yet.
4. **Chat** — a chat session is linked to one or more directories. Each message goes to `core`'s agent, which either explores the linked files directly or (if an index exists) searches indexed chunks semantically and/or by metadata filter, then answers with inline citations (`[Belge Adı, Madde X]`) and a sources list.

### Why not just one big "index" button?

Indexing without a real database happens to be optional in `core` (agentic exploration works on raw files with zero setup), but once you do index, parsing+chunking and embedding are deliberately two steps so a stalled or misconfigured embedding provider never forces you to redo the (slower, Docling-based) parsing/chunking pass — chunks are durable in Postgres as soon as "Generate chunks" finishes, regardless of what happens next. Directories that only have chunks (no embeddings yet) still work in chat: search falls back to keyword matching over chunk text until embeddings are generated.

## Configuration

Environment variables are split per app (see each app's README/`.env.*.example` for the full list), but the ones you'll touch first:

| Variable | Where | Purpose |
|----------|-------|---------|
| `GOOGLE_API_KEY` | `core` | Gemini LLM + embeddings. Get one at [Google AI Studio](https://aistudio.google.com/apikey). |
| `DATABASE_URL` | `core`, `backend` | Shared Postgres connection string. |
| `CORE_INTERNAL_TOKEN` | `core`, `backend` | Shared secret gating `core`'s internal REST/WebSocket endpoints so only `backend` can call them. |
| `JWT_SECRET` | `backend` | Signs access tokens. |
| `STORAGE_ROOT` | `backend` | Where uploaded files live on disk before/while being chunked. |
| `VITE_API_URL` | `frontend` | Where the backend API lives. |

## Architecture

```
Browser
  │  REST + chat (HTTP/SSE)
  ▼
backend (LoopBack 4)  ──auth, directories, chat sessions, file storage──▶ Postgres
  │  internal WebSocket/REST (X-Internal-Token)
  ▼
core (FastAPI)
  ├─ Agentic mode: Workflow → Agent (Gemini) → fs tools (scan/preview/parse/grep/glob) → Docling
  └─ Indexed mode: Workflow → Agent → semantic_search/get_document → Postgres + pgvector
```

See [`core/CLAUDE.md`](core/CLAUDE.md) for a deep dive into `core`'s modules (agent, workflow, chunker, search, storage), and each app's own README for its endpoints/structure.

## Development

Each app is developed and tested independently — see its README for the exact commands (`core` uses `uv`/`pytest`, `backend`/`frontend` use `npm`). A few cross-cutting things to know:

- `core` and `backend` share one Postgres database (`db/`), split by table prefix (`core_*` vs everything else) — never have one service write to the other's tables directly.
- Re-indexing is idempotent everywhere: corpus/document/chunk ids are stable content hashes, so re-running chunking or embedding on unchanged content upserts the same rows instead of duplicating them.
- Tests that need Postgres skip themselves (`pytest.skip`) if `DATABASE_URL` isn't set — point it at a throwaway database with `db/migrations/` applied to run them.

## License

MIT

## Acknowledgments

- Original concept from [run-llama/fs-explorer](https://github.com/run-llama/fs-explorer)
- Document parsing by [Docling](https://github.com/DS4SD/docling)
- Powered by [Google Gemini](https://deepmind.google/technologies/gemini/)
