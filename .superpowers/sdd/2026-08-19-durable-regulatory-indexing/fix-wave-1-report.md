# Durable regulatory indexing — final-review fix wave 1

## Outcome

This wave resolves whole-branch review findings 1, 4, 5, and 10 without
enabling the feature by default or changing the legacy flag-off indexing path.

- PREPARING now commits canonical chunks, durable items, the next job state,
  and `UserFile.status=INDEXING` in one PostgreSQL transaction. Recovery reloads
  Markdown through the same `LocalFileConnector` boundary as initial ingestion,
  including ONYX metadata, title, semantic identifier, hierarchy filtering, and
  staging cleanup. The input identity excludes the connector's fallback
  wall-clock `doc_updated_at`, so the same stored bytes remain recoverable after
  restart while canonical metadata and content remain hash-sensitive.
- User-file deletion first commits a `DELETING` tombstone and generation-fences
  every non-cancelled durable job. The ordinary delete worker cannot remove
  search/file-store artifacts, `UserFile`, jobs, or items until the resumable
  Vertex cancel, GCS cleanup, snapshot-index deletion, and vector-clearing
  cancellation phases have reached `CANCELLED`. The existing deletion beat
  redelivers the ordinary worker after durable cancellation completes.
- Durable publication atomically marks the file complete and sets
  `secondary_reconcile_pending`. Existing canonical project-sync reconciliation
  projects PostgreSQL chunks to FUTURE before clearing that bit. Port-flow swap
  now locks PRESENT/FUTURE settings and blocks on both active durable jobs and
  pending file reconciliation, including INSTANT port promotion.
- The immutable snapshot, ORM, migration, repository idempotency key, and
  recovery validation now carry a deterministic SHA-256 chunk-generation
  identity. It binds the embedding tokenizer identity, regulatory chunk and
  adaptive-budget constants, and explicit chunker/indexing code versions.

Multi-row lifecycle operations use a consistent lock order:
`SearchSettings` (when needed), then `UserFile`, then durable job. This closes
the deletion/external-mutation deadlock and prevents a new job from appearing
after the deletion tombstone.

## TDD evidence

The unchanged focused baseline was `105 passed in 3.80s`.

Material RED observations before implementation or the corresponding fix:

- generation identity: collection failed because
  `compute_regulatory_chunk_generation_hash` did not exist;
- durable deletion: collection failed because
  `request_user_file_deletion_cleanup` did not exist;
- canonical recovery: the orchestrator had no shared
  `load_user_file_documents` boundary and reconstructed a different Document;
- port safety: the upload-during-port assertion initially allowed swap;
- tombstone/create race: the new real-PostgreSQL test failed with
  `DID NOT RAISE`, proving a job could be created for a deleting file;
- restart identity: the same Document with only a later fallback
  `doc_updated_at` produced two different SHA-256 values.

Final GREEN evidence:

- focused configuration, preparation, orchestration, publication, embedding,
  repository, and user-file task unit tests: `120 passed in 8.92s`;
- the broader regulatory unit superset plus repository and user-file task
  regressions: `517 passed in 11.82s`;
- real PostgreSQL job lifecycle, crash repair, lock/race, deletion, generation
  identity, FUTURE-port, and existing delete-queue regression tests:
  `55 passed in 10.16s`;
- disposable real PostgreSQL plus Elasticsearch 9.4.2 pipeline, including the
  deletion tombstone followed by real hidden-index cleanup and deterministic
  Vertex/GCS gateway cancellation: `8 passed in 33.78s`.

Touched-file pre-commit passed lazy-import enforcement, `ty`, Ruff, Ruff format,
large-file validation, ripsecrets, and environment-drift checks. `compileall`
and `git diff --check` also passed.

The Task 8 pipeline initially failed three cases only because its test fixture
still patched and called the removed manual-recovery boundary. The fixture now
injects its disposable file store at `LocalFileConnector`, the production
owner of the canonical load, and the three affected cases passed before the
full eight-test repeat.

## Migration evidence

Migration `d2a9c7e4b1f6` follows `c8f1a6d4e2b7`. On the isolated PostgreSQL
database, a representative pre-migration job was inserted at the old revision.

1. upgrade backfilled both snapshot hashes and the new column (`1:1:1`), made
   `chunk_generation_hash` non-null, and produced the ordered unique key
   `user_file_id,content_hash,search_settings_id,prompt_hash,chunk_generation_hash`;
2. downgrade restored the legacy snapshot exactly, removed the new column, and
   restored the four-column key;
3. a second upgrade returned the database to `d2a9c7e4b1f6`; deleting the
   disposable UserFile cascaded the disposable job, leaving zero test rows.

The migration's backfill digest equals the runtime digest for the released
OpenRouter model and code/config identity:
`c8e1ab454f0ac79eea2db7e0c1a54979d55fa97232da08130ee8fa4b8b324e04`.

## Verification boundaries and remaining risk

No production database, provider, object, document, or index was read or
mutated. Tests used an isolated local PostgreSQL database, a disposable
Elasticsearch 9.4.2 container/index namespace, a deterministic Vertex/GCS
gateway, and a local OpenRouter-compatible embedding server.

No live Vertex AI, Google Cloud Storage, OpenRouter, frontend canary, deploy,
push, or merge was performed. Those remain approval- and credential-gated live
validation; the repository-owned external pipeline covers their durable state
and failure boundaries without weakening the production contracts.
