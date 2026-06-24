# Customs Regulations Chatbot



An AI assistant that answers questions about regulatory documents (gümrük tebliğleri, genelgeler, kanunlar, ...) by either exploring files agentically like a human reader, or querying a pre-built semantic + metadata index — with every claim in the answer backed by a citation back to its source article/clause.

## Repo layout

This is a monorepo split into four independent projects:


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
User Query
    ↓
┌─────────────────┐
│ Workflow Engine │ ←→ LlamaIndex Workflows (event-driven)
└────────┬────────┘
         ↓
┌─────────────────┐
│     Agent       │ ←→ Gemini 3 Flash (structured JSON)
└────────┬────────┘
         ↓
┌─────────────────────────────────────────┐
│ scan_folder │ preview │ parse │ read │ grep │ glob │
└─────────────────────────────────────────┘
                    ↓
              Document Parser (Docling - local)
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed diagrams.




  /home/kubilay-payci/customs-regulations-chatbot/.venv/bin/python -m fs_explorer.chunk_inspector --host 127.0.0.1 --port 8123
