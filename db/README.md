# db

Shared database infrastructure: multi-environment `docker-compose` setup for Postgres (+ pgvector) and SQL migrations (`node-pg-migrate`).

Kept separate from `core`/`backend` because both read from and write to the same database — the schema shouldn't be owned by either service's language/framework.

`core` and `backend` share one physical database: `core` owns the `core_*` tables (`core_documents`/`core_chunks`/`core_chunk_embeddings`/`core_schemas`, plus pgvector for embeddings), `backend` owns everything else (`users`, `directories`, `chat_sessions`, ...). `core` connects via `DATABASE_URL` (see `core/src/fs_explorer/index_config.py`).

## Environments

Each environment is a base `docker-compose.yml` plus an override file, combined at run time:

- `docker-compose.dev.yml` — host port 5432, persistent named volume, `restart: unless-stopped`.
- `docker-compose.test.yml` — host port 5433, no persistent volume (fresh DB every run), `restart: "no"`.
- `docker-compose.prod.yml` — no host port exposed (reachable only from other containers on the same docker network), persistent named volume, memory limit, `restart: always`.

Each environment needs its own env file with `POSTGRES_USER` / `POSTGRES_PASSWORD` / `POSTGRES_DB` (and `POSTGRES_PORT` for dev/test). Copy the matching example and fill it in:

```bash
cp .env.dev.example .env.dev
cp .env.test.example .env.test
cp .env.prod.example .env.prod
```

Bring an environment up directly:

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml --env-file .env.dev up -d
```

Usually you don't need to do this by hand — see the root `scripts/run.sh`, which starts the right compose files *and* runs migrations for you.

## Migrations

Plain `.sql` files via [node-pg-migrate](https://github.com/salsita/node-pg-migrate) (`migration-file-language: sql` in `.node-pg-migraterc.json`), in `migrations/`. Each file is a single `.sql` with a `-- Up Migration` and a `-- Down Migration` section — no JS migration-builder API involved.

Filenames are prefixed with the date they were created, `YYYYMMDDHHMMSS_description.sql` (sorts and runs in that order):

```bash
npm install
DATABASE_URL=postgresql://user:pass@localhost:5432/db npm run migrate:up
DATABASE_URL=postgresql://user:pass@localhost:5432/db npm run migrate:down
npm run migrate:create -- some-migration-name   # creates migrations/<today>_some-migration-name.sql
```

Every table's `id` is a `SERIAL` (autoincrement integer) primary key — no UUIDs.

Current schema:
- `backend`: `users` (account, password hash, role, lockout state), `refresh_tokens` (hashed, rotation chain via `replaced_by_token_hash`), `directories`/`directory_files` (a user's file clusters and the files in them), `chat_sessions`/`chat_session_directories` (chat sessions and which directories each is scoped to).
- `core`: `core_documents`/`core_chunks`/`core_chunk_embeddings` (`vector(768)`, HNSW cosine index) — chunks produced by `RegulatoryChunker`, with full locator metadata (article/paragraph/clause, source offsets) in `core_chunks.metadata` (JSONB), traceable to their source file via `document_id → core_documents`. `core_schemas` for per-corpus metadata schemas.
