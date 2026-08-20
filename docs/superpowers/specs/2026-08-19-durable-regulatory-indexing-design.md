# Durable Regulatory Indexing Design

## Status

Approved for implementation on 2026-08-19. The user selected the durable,
database-backed orchestrator approach and explicitly authorized implementation
and verification without additional approval checkpoints.

## Issues to Address

Production-lite accepts Markdown without the heavyweight parser stack, and the
regulatory chunker already persists canonical `regulatory_chunk` rows. It does
not, however, run a production indexing worker. The current synchronous user-file
pipeline also couples contextual LLM calls, embedding, and Elasticsearch writes in
one task. That shape cannot safely support an asynchronous Vertex AI Batch job,
partial results, process restarts, or bounded recovery.

The new flow must:

- ingest Markdown directly without Docling, CUDA, a model server, or parser-only
  dependencies in `runtime-lite`;
- keep existing regulatory chunk generation as the canonical source of truth;
- make contextual retrieval mandatory for every eligible chunk before embedding;
- use Vertex AI Batch Prediction through the configured contextual model's Vertex
  provider and a configured Google Cloud Storage workspace;
- use the active Admin `SearchSettings` embedding provider/model, with production
  configured as OpenRouter `openai/text-embedding-3-large`;
- preserve the active index's effective dimension, expected to be 1024, by using
  `SearchSettings.final_embedding_dim` and sending `dimensions=1024` when that is
  the active value;
- never use the pre-existing, uncommitted batch embedding experiments;
- survive duplicate Celery delivery, provider throttling, partial Vertex output,
  worker restarts, stale jobs, and cancellation;
- keep incomplete documents out of retrieval and leave existing searchable data
  intact when a new generation fails;
- remain disabled by default so ordinary Onyx deployments keep their current
  behavior.

## Important Notes

### Configuration ownership

The Admin configuration remains the source of truth for both models. A job takes
an immutable snapshot before making an external request:

- `search_settings_id`, embedding provider, model name, native model dimension,
  reduced dimension, effective dimension, and index name;
- contextual `model_configuration_id`, Vertex model name, project, location, and
  authentication mode;
- prompt version and prompt hash;
- input content hash and chunk generation hash.

Secrets are never copied into the job tables. Provider credentials are resolved
fresh from encrypted Admin configuration when a stage executes. A job fails
closed if its referenced configuration has disappeared or no longer matches the
immutable non-secret snapshot.

Production readiness requires the active embedding model to be
`openai/text-embedding-3-large` through OpenRouter. The architecture does not
hard-code a vector length. The effective dimension is always
`reduced_dimension or model_dim`; both the provider response and Elasticsearch
mapping must match it. Changing model or dimension uses the existing FUTURE index
and reindex/swap flow rather than mixing embedding spaces in one index.

### Vertex Batch storage

Vertex AI Batch uses a Cloud Storage JSONL source and destination. The deployment
supplies a non-secret `REGULATORY_INDEXING_GCS_URI`, for example
`gs://bucket/regulatory-indexing`. Each job writes beneath a deterministic tenant
and job prefix. The service account or Workload Identity configured for the
selected Vertex provider must be able to create/read/delete those objects and
create/read/cancel Vertex batch jobs.

The input JSONL contains one request per eligible chunk. A canonical hash of the
request content maps echoed Vertex output back to a persisted item; output order
is never trusted. Temporary GCS objects are retained briefly for diagnosis and
then removed by best-effort cleanup after a terminal job.

## Architecture

### Components

1. **Regulatory indexing job repository**
   Owns all PostgreSQL reads, writes, atomic claims, fencing, heartbeat updates,
   retry scheduling, cancellation, and stale recovery.

2. **Preparation service**
   Reuses the existing Markdown loader and `RegulatoryIndexingChunker`. It stores
   canonical chunks, creates one idempotent job for the file/content revision,
   snapshots configuration, and creates one item per chunk.

3. **Vertex contextual batch gateway**
   Builds canonical JSONL, uploads it to GCS, submits/polls/cancels Vertex jobs,
   downloads output, validates every response, and returns results keyed by
   request hash. It contains no database or Celery logic.

4. **OpenRouter embedding stage**
   Reconstructs `DocAwareChunk` objects from canonical rows and persisted context,
   embeds bounded lists through the normal OpenRouter embedding endpoint, and
   validates vector count, ordering, finiteness, and exact dimension. This is a
   new stage and does not import old batch embedding modules.

5. **Elasticsearch publisher**
   Writes one complete file generation with `hidden=true`, verifies the inserted
   document/chunk count, then uses the existing metadata update path to publish
   it with `hidden=false`. Completion is recorded only after publication succeeds.

6. **Dedicated Celery worker**
   The `regulatory_indexing` app consumes `user_file_processing` for the existing
   upload trigger and `regulatory_indexing` for orchestration stages. It uses the
   same `runtime-lite` image and adds no parser, model-server, CUDA, or Docling
   dependency.

### State model

Job stage and job status are independent string enums.

Stages:

1. `PREPARING`
2. `CONTEXT_SUBMIT`
3. `CONTEXT_WAIT`
4. `CONTEXT_APPLY`
5. `EMBEDDING`
6. `INDEX_WRITE`
7. `VERIFY`
8. `PUBLISH`

Statuses:

- `QUEUED`
- `RUNNING`
- `RETRY_WAIT`
- `SUCCEEDED`
- `FAILED`
- `CANCELLING`
- `CANCELLED`

Item statuses are `PENDING`, `CONTEXT_READY`, `EMBEDDED`, `FAILED`, and `SKIPPED`.
An item with no useful contextual token reserve is `SKIPPED` for context but is
still embedded from its original legal text.

Every runnable job has a monotonic `lease_generation`. An atomic SQL update claims
the row only when its status, stage, `next_retry_at`, and lease are eligible. Every
subsequent mutation includes the claimed generation in its predicate. A late task
from an older lease therefore cannot overwrite newer state.

### End-to-end flow

1. Upload persists the file and sets `UserFile.status=PROCESSING` using the
   existing API.
2. With the feature flag disabled, the existing synchronous flow runs unchanged.
3. With the flag enabled, the existing processing entry point loads only the
   supported Markdown content, creates canonical chunks, creates or reuses the
   idempotent job, changes the user file to `INDEXING`, and enqueues the first
   orchestration task with `tenant_id` and `expires`.
4. `CONTEXT_SUBMIT` validates the snapshot, writes input JSONL to GCS, submits one
   Vertex Batch job, persists the remote name, and schedules a poll. A persisted
   remote job name prevents duplicate submission.
5. `CONTEXT_WAIT` performs one bounded status request and exits. Non-terminal jobs
   schedule another poll; a worker thread is never held while Vertex runs.
6. `CONTEXT_APPLY` downloads output, correlates by request hash, stores successful
   context, and retries only missing/transiently failed items. Empty, malformed,
   safety-blocked, or permanently invalid output is explicit item failure.
7. `EMBEDDING` loads bounded item batches, reconstructs contextualized chunks,
   requests OpenRouter embeddings with the snapshot dimension, validates all
   vectors, and persists them before advancing.
8. `INDEX_WRITE` reconstructs fully enriched chunks and writes the entire file as
   hidden Elasticsearch documents using deterministic document/chunk IDs.
9. `VERIFY` confirms insertion records, canonical row count, embedded item count,
   and effective dimension. Any mismatch retries or fails before publication.
10. `PUBLISH` sets all chunks visible through the existing metadata update path,
    updates `UserFile.chunk_count`, marks the file `COMPLETED`, and marks the job
    `SUCCEEDED`.

The Elasticsearch API has no multi-document transaction. Logical publication is
therefore achieved with hidden staging plus an idempotent bulk metadata update.
The job is never marked complete while any chunk remains hidden or missing.

## Retry and Recovery Policy

Retry decisions are typed rather than based on arbitrary exception strings.

### Retryable

- network timeout, connection reset, DNS failure after startup, and temporary GCS
  errors;
- HTTP 408, 409 where the operation is idempotent, 429, and 5xx;
- Vertex pending/running states;
- Vertex terminal jobs with a subset of retryable item failures;
- Elasticsearch timeout, rejected execution, and retryable bulk failures.

### Terminal

- authentication/authorization failures;
- missing Vertex project/location/GCS URI;
- contextual provider not Vertex AI;
- active embedding provider/model not the configured OpenRouter model contract;
- invalid request, malformed output, empty successful context, safety refusal;
- embedding vector count/index/dimension/finiteness mismatch;
- SearchSettings or index mapping drift from the job snapshot;
- deleted/cancelled user file.

Retry delay uses exponential backoff with full jitter, persisted as
`next_retry_at`. Defaults are capped and configurable: five attempts per stage,
15 seconds base delay, 15 minutes maximum delay, 30-second poll interval, and a
two-minute worker lease. Celery messages remain short-lived delivery hints; the
database is the durable scheduler.

A periodic recovery task claims stale `QUEUED`, `RUNNING`, and due `RETRY_WAIT`
jobs using `FOR UPDATE SKIP LOCKED`, advances their lease generation, and emits a
fresh task with an expiration. Redis guards limit duplicate queue traffic but are
never the source of truth.

Partial Vertex output creates a new JSONL file containing only unresolved items.
Successful context is never regenerated. Embedding retry likewise operates only
on items without a persisted valid vector.

## Cancellation and Cleanup

Deletion or explicit cancellation changes the job to `CANCELLING`. The next
orchestration step cancels an active Vertex job, cleans staged Elasticsearch
documents, clears persisted embedding payloads, and marks the job `CANCELLED`.
User-file deletion remains authoritative and must never be reversed by a late
indexing task.

Historical generations are deleted sequentially. The deletion worker locks the
UserFile and all of its jobs, overrides the sole active cancellation with
`USER_DELETE` or activates exactly one terminal historical generation, and leaves
every other history row terminal. After that job reaches `CANCELLED`, the durable
deletion scanner redelivers the tombstoned file and activates the next generation.
The UserFile and its job history are hard-deleted only after every generation is
`CANCELLED`; the partial unique active-job invariant therefore remains true
throughout cleanup and broker loss.

An Elasticsearch `INDEX_DELETE` failure for `USER_DELETE` or `SUPERSEDE` never
advances to `FINALIZE`, irrespective of error classification or attempt count. It
persists the error, incremented attempt, and a deterministic capped retry deadline
on the same phase. This makes blocked cleanup operator-visible without a hot loop
and prevents hard deletion or a successor while stale chunks may remain. Explicit
`USER_CANCEL` retains bounded best-effort cleanup semantics and remains distinct
from deletion.

GCS cleanup is best effort and separately retryable; inability to remove temporary
objects does not change an already correct search publication result. Lifecycle
rules on the configured bucket provide a final safety net.

### Compatibility hash resolution

Migration cannot infer whether an indexing job created by an earlier draft used
the legacy semantic-identifier/title/text digest or the canonical full-Document
digest. Such PREPARING rows carry an explicit compatibility-only
`legacy-or-canonical` discriminator. Recovery loads the canonical Markdown once,
computes both historical digests, and proceeds only when exactly one matches the
persisted content hash. The resolved `legacy-v1` or `canonical-v2` discriminator is
committed in the same transaction that replaces canonical chunks/items and advances
PREPARING. No match and the cryptographically exceptional ambiguous match both fail
closed without changing the immutable snapshot.

Fresh jobs always use `canonical-v2`. Compatibility migration and a
downgrade/re-upgrade may create the unresolved discriminator, but no job may leave
PREPARING while it remains unresolved.

### Chunk-generation supersession

One UserFile still has at most one active durable job. Every claimed normal or
preclaimed delivery compares the job's generation identity with the current
deterministic identity before ordinary terminal snapshot validation or stage work.
On drift, it locks the UserFile and claimed job, changes that same job to
`CANCELLING`, records typed `SUPERSEDE` intent, increments its lease fence, and
resumes the existing Vertex, GCS, and Elasticsearch cleanup phases. An unresolved
compatibility PREPARING snapshot is the sole exception: canonical loading and exact
legacy/canonical input-hash resolution occur first, and the following delivery then
performs the generation comparison.

Cancellation intent distinguishes `USER_CANCEL`, `USER_DELETE`, and `SUPERSEDE`.
Deletion is a monotonic override of either other intent. Finalizing user cancellation
keeps the UserFile cancelled; finalizing deletion keeps its deletion tombstone;
finalizing supersession marks only the old job `CANCELLED` and sets the UserFile to
`PROCESSING`. The existing durable UserFile scanner then invokes the ordinary
production worker to create the current generation after the partial unique active
job constraint has been released. A crash after finalization is safe because
`PROCESSING` is durable. Repeated supersession delivery reuses the same cancelling
job, and an identical already-successful current generation remains idempotent and
completed.

## Observability

Structured logs include tenant, job, user file, stage, attempt, lease generation,
remote Vertex job, item counts, and elapsed time, but never prompts, legal content,
credentials, or vectors. Metrics cover queue depth, jobs per status/stage, stage
latency, retries by typed reason, stale recovery count, Vertex item outcomes,
embedding request size, and publish verification failures.

Direct Vertex and embedding requests use dedicated `LLMFlow` tracing values.

## Deployment

- Add a short-lived feature flag `REGULATORY_BATCH_INDEXING_ENABLED`, default
  `false`, tested in both states.
- Add the `regulatory_indexing` worker to `supervisord-lite.conf`, production-lite
  health checks, log forwarding, monitoring queue metrics, and deployment
  readiness checks.
- Keep `DOCUMENT_IMPORT_ENABLED=false`; Markdown support remains controlled by
  the existing light-runtime capability.
- Add only Google client libraries already present in the locked runtime
  requirements. Do not add Docling, Torch, CUDA, or a model-server dependency to
  `runtime-lite`.
- Run the Alembic migration as the existing singleton migration job before the
  worker is enabled.

## Tests

### Unit tests

- state transitions, fencing, idempotent creation, retry classification/backoff,
  stale recovery, cancellation, and snapshot validation;
- JSONL construction and output correlation independent of order;
- partial Vertex result handling and retry-only-missing behavior;
- OpenRouter request model/dimension, vector count/order/dimension/finiteness;
- hidden-before-publish ordering and feature-flag-off compatibility;
- Celery app/queue/supervisor/deployment wiring.

### External dependency tests

- real PostgreSQL migration and atomic claim behavior under duplicate callers;
- real Redis/Celery enqueue expiry and recovery behavior where practical;
- real Elasticsearch hidden staging, publish, retry, and cleanup behavior.

### Live smoke test

In a controlled document set, upload a small Markdown regulation and verify:

1. canonical regulatory chunks exist;
2. a real Vertex Batch job completes and all eligible chunks have context;
3. real OpenRouter embeddings use `openai/text-embedding-3-large` and exactly the
   active effective dimension (expected 1024);
4. the Elasticsearch mapping dimension and inserted vector lengths match;
5. the file is hidden before publication and searchable afterward;
6. retrieval returns the expected provision and citation metadata;
7. injected transient failure, worker restart/resume, duplicate delivery, partial
   result, and cancellation scenarios preserve invariants.

Production documents and existing indices are not mutated for benchmark-style
validation. Live testing uses a newly uploaded canary document that can be safely
deleted afterward.

## References

- Google Gen AI SDK Batch Prediction: https://googleapis.github.io/python-genai/
- Vertex AI batch inference JSONL and output contract:
  https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/capabilities/batch-inference/new-job-from-cloud-storage
- OpenAI embedding dimensions:
  https://developers.openai.com/api/docs/guides/embeddings
- OpenRouter embeddings endpoint:
  https://openrouter.ai/docs/api/reference/embeddings
