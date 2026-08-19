# Task 5 Report: OpenRouter Embedding and Hidden Elasticsearch Publication

## Status

Complete. The implementation embeds canonical regulatory items through the maintained OpenRouter boundary, persists each validated response under the claimed lease generation, stages the complete file as hidden Elasticsearch chunks, and exposes it before completing the `UserFile`.

## Implementation

- Added bounded `embed_pending_regulatory_items(...)` processing in canonical order.
  - Validates the current `SearchSettings` against the immutable job snapshot.
  - Uses the snapshot provider/model and `effective_dimension`; no embedding dimension is hard-coded.
  - Reuses valid persisted vectors and repairs only missing or invalid vectors.
  - Places generated context before original legal text, while skipped context uses the original text unchanged.
  - Validates response count, dimensions, numeric values, and finiteness before persisting any vector from a request.
  - Persists a validated request atomically through a generation-fenced DB repository helper.
- Added complete-file hidden publication.
  - Reconstructs every canonical row, including temporally superseded rows, in stable `(position, id)` order.
  - Preserves regulatory chunk identity, normalized heading path, and validity dates.
  - Calls `DocumentIndex.index(...)` once with every chunk marked `hidden=True`.
  - Verifies exact row/item coverage, vector count/dimension/validity, insertion-record count, and document identity.
  - Uses deterministic document and chunk IDs so retries replace the same generation.
  - Calls `DocumentIndex.update(... hidden=False)` before the DB completion helper.
  - Does not advance the job stage; later orchestration owns that transition.
- Added `hidden: bool = False` to `DocMetadataAwareIndexChunk` enrichment and mapped it to Elasticsearch `DocumentChunk.hidden`, preserving legacy visibility by default.
- Added DB-layer helpers for atomic vector-batch persistence and generation-fenced `UserFile` completion. Service code does not mutate ORM state directly.

## TDD Evidence

### Embedding RED

Command:

```bash
uv run pytest -q backend/tests/unit/onyx/regulatory/indexing_jobs/test_embedding.py
```

Expected failure observed before implementation:

```text
ImportError: cannot import name 'embedding' from 'onyx.regulatory.indexing_jobs'
1 error during collection
```

Initial embedding GREEN after implementation: `7 passed in 4.56s`.

### Publication RED

Command:

```bash
uv run pytest -q backend/tests/unit/onyx/regulatory/indexing_jobs/test_publisher.py
```

Expected failure observed before implementation:

```text
ImportError: cannot import name 'publisher' from 'onyx.regulatory.indexing_jobs'
1 error during collection
```

Initial publication GREEN after implementation: `5 passed in 1.47s`.

Tests were then expanded for cancellation, stale verification, the real Elasticsearch conversion boundary, invalid OpenRouter indexes, and legacy hidden defaults. Final focused Task 5 result: `14 passed in 1.86s`.

## Final Verification

- Task 5 plus adjacent Elasticsearch/indexing unit regressions:

  ```bash
  uv run pytest -q \
    backend/tests/unit/onyx/regulatory/indexing_jobs \
    backend/tests/unit/onyx/document_index/elasticsearch \
    backend/tests/unit/onyx/indexing/test_embedder.py \
    backend/tests/unit/onyx/indexing/test_personas_in_chunks.py \
    backend/tests/unit/onyx/indexing/test_regulatory_chunk_access.py
  ```

  Final result: `237 passed in 12.05s` (an earlier identical run passed in `5.09s`).

- Adjacent DB repository regression against a disposable PostgreSQL 15 + pgvector database migrated to Alembic head `c8f1a6d4e2b7`:

  ```bash
  POSTGRES_HOST=127.0.0.1 POSTGRES_PORT=55435 \
    POSTGRES_USER=postgres POSTGRES_PASSWORD=[REDACTED_SECRET] \
    POSTGRES_DB=postgres \
    uv run pytest -q backend/tests/external_dependency_unit/db/test_regulatory_indexing_jobs.py
  ```

  Result: `14 passed in 2.60s`. The disposable container was removed after the test.

- Focused pre-commit with `SKIP=env-drift-baseline`: all applicable checks passed, including `ty`, Ruff lint/format, secret scanning, and lazy-import validation.
- `git diff --check`: passed.

## Files

- `backend/onyx/db/regulatory_indexing_jobs.py`
- `backend/onyx/document_index/elasticsearch/elasticsearch_document_index.py`
- `backend/onyx/indexing/models.py`
- `backend/onyx/regulatory/indexing_jobs/embedding.py`
- `backend/onyx/regulatory/indexing_jobs/publisher.py`
- `backend/tests/unit/onyx/regulatory/indexing_jobs/test_embedding.py`
- `backend/tests/unit/onyx/regulatory/indexing_jobs/test_publisher.py`
- `.superpowers/sdd/2026-08-19-durable-regulatory-indexing/task-5-report.md`

## Self-review

- Confirmed all embedding and publication inputs have exact job/file ownership and one-to-one canonical coverage.
- Confirmed invalid provider responses are rejected as a whole before the DB persistence helper is called.
- Confirmed each DB mutation is lease-generation fenced and transactionally commits or rolls back.
- Confirmed publication failure ordering: insertion/verification failure never exposes chunks; visibility-update failure never marks the file complete.
- Confirmed cancellation/deletion checks prevent staging and publication, and the completion helper cannot revive a canceled/deleting file.
- Confirmed stable ordering includes superseded temporal rows and preserves validity metadata.
- Confirmed the unrelated untracked design document was not modified or staged.

## Concerns and Limits

- The full unskipped pre-commit run reaches all Task 5 checks successfully but `env-drift-baseline` reports 12 environment variables introduced by other durable-indexing tasks. Task 5 does not add or own those variables; deployment-document synchronization belongs to the later deployment task.
- Private development PostgreSQL DNS was unavailable from this environment, so DB regression evidence uses a clean disposable PostgreSQL/pgvector instance instead.
- OpenRouter and Elasticsearch behaviors were verified at their maintained request/conversion interfaces with recording clients; no live external OpenRouter or Elasticsearch service call was made in this task.

## Commit

The implementation and this report are committed together with subject `feat: embed and safely publish regulatory jobs`.
