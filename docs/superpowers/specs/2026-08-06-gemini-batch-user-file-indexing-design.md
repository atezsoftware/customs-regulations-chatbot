# Gemini Batch User-File Indexing Design

## Goal

Add an optional, progress-visible Gemini Batch API indexing path for document-set uploads while preserving the existing online indexing path. Use the batch path to rebuild a new `tır-transit` document set from `/home/kubilay-payci/tır_transit` with contextual retrieval enabled, `gemini-3.6-flash` for context generation, and `gemini-embedding-2` for embeddings.

## Requirements

- Preserve the existing online upload/indexing behavior as the default.
- Add `gemini_batch` as an explicit upload/indexing mode.
- Use the provider's asynchronous Batch API for contextual generation and embeddings; do not merely rename the existing local request batching.
- In `gemini_batch` mode, every contextual LLM request uses provider `generateContent` batch jobs. Document summaries, chunk contexts, and retries must never fall back to one-request-at-a-time online LLM calls.
- Keep contextual enrichment enabled and complete for every eligible chunk.
- Treat empty or malformed Gemini responses, including a response with no choices, as retryable item failures.
- Retry only failed batch items, with at most three provider submissions per item.
- Split provider jobs to stay below provider inline-request size and request-count limits.
- Allow up to eight batch shards to be submitted or collected concurrently. The provider still controls execution throughput.
- Persist job, stage, item, retry, and error state so a worker restart can resume safely.
- Show document-set batch progress in the frontend without exposing credentials or prompt contents.
- Delete the current `tır-transit` set, create a new set, upload the source directory, and index that upload through the new batch path.

## Why Provider Batch Instead of Only More Threads

The current Vertex embedding implementation already sends multiple texts per online request and the indexing pipeline parallelizes API batches. The observed failure occurs earlier: contextual `generateContent` occasionally returns a response with no choices, and one such response aborts the whole file after many successful chunk calls.

The Gemini Batch API supports asynchronous generation jobs and embedding jobs, including `gemini-embedding-2`. It is a throughput and cost mechanism rather than a latency guarantee; Google documents a target turnaround of up to 24 hours. The UI must therefore report submitted/running/completed stages honestly rather than promise a 15–20 minute completion time.

References:

- https://ai.google.dev/gemini-api/docs/batch-api
- https://ai.google.dev/gemini-api/docs/embeddings
- https://ai.google.dev/api/embeddings

## Architecture

### Upload Contract

The document-set upload endpoint accepts a new form field:

- `indexing_mode=online` (default): unchanged existing behavior.
- `indexing_mode=gemini_batch`: stores uploaded files, creates one batch indexing job for the upload, and leaves the files in `PROCESSING` until the job reaches a terminal state.

The upload response adds an optional `batch_job_id`. Existing clients remain compatible because the new field is optional and the default mode is unchanged.

### Persistent State

Add two PostgreSQL models.

`UserFileBatchJob` stores:

- job UUID, tenant, document-set ID, creator, mode, and captured search-settings ID;
- contextual and embedding model identifiers;
- current stage and terminal status;
- total/completed/failed file and request counts;
- retry count summary, safe error summary, timestamps, and cancellation timestamp.

`UserFileBatchItem` stores:

- job ID and user-file ID;
- stage (`prepare`, `document_summary`, `chunk_context`, `embedding`, `index_write`);
- stable request key and regulatory chunk ID when applicable;
- provider batch job name and provider request key;
- status, attempt count, safe error text, and timestamps;
- generated contextual text where it is required by the next stage.

Embedding vectors are not stored long-term in PostgreSQL. They are consumed from a completed provider job and written to Elasticsearch during collection. Provider job names and stable request keys are durable, allowing collection to resume after restart.

### Batch State Machine

1. **Prepare files**
   - Parse each uploaded blob with the existing user-file loader.
   - Run the structure-aware regulatory chunker and persist canonical `regulatory_chunk` rows.
   - Record total files and eligible chunks.
2. **Document-summary batches**
   - Submit one request for each distinct document context that requires a summary, grouped into provider `generateContent` batch jobs rather than awaited one by one.
   - Poll provider jobs and persist successful summaries.
   - Resubmit only failed/malformed items, up to three attempts.
3. **Chunk-context batches**
   - Build each existing contextual prompt from its persisted document summary and legal chunk.
   - Submit all eligible chunk requests in provider `generateContent` batch jobs, poll, validate non-empty responses, and retry failed items only in new batch jobs.
4. **Embedding batches**
   - Build the same final embedding text used by the online pipeline.
   - Submit `gemini-embedding-2` embedding batch jobs with the configured output dimensionality.
   - Validate result count and vector dimensionality before indexing.
5. **Elasticsearch write**
   - Reuse the existing enrichment and document-index write contracts with precomputed embeddings.
   - Mark each file `COMPLETED` only after every expected chunk is written successfully.
6. **Terminal state**
   - Mark the job completed only when all files complete.
   - Mark a file and job failed with a safe error if an item exhausts three attempts.

Every transition is idempotent. A task may be delivered more than once without creating duplicate provider requests or duplicate Elasticsearch documents.

### Workers and Scheduling

Add a dedicated `user_file_batch` Celery queue consumed by the local background worker. Separate tasks prepare, submit, poll, collect, and write stages. Poll tasks have expirations and re-enqueue themselves with bounded intervals while provider jobs are non-terminal.

Configuration:

- `CONTEXTUAL_RAG_MAX_WORKERS=8` for the online fallback path.
- `GEMINI_BATCH_MAX_IN_FLIGHT_JOBS=8` for batch submission/collection.
- `GEMINI_BATCH_MAX_ITEM_ATTEMPTS=3`.
- Provider polling uses bounded backoff and does not occupy a worker thread while waiting.

### Progress API and Frontend

Add document-set endpoints to fetch the latest batch job and cancel a non-terminal job. The progress response includes:

- overall status and current stage;
- total/completed/failed files;
- total/completed/failed requests for the current stage;
- percentage derived from completed stage weights and current-stage items;
- creation/update timestamps and a safe terminal error;
- per-file stage/status/chunk counts.

The document-set files page adds a Batch Indexing progress card and polls every four seconds while the job is non-terminal. Upload controls offer Online and Gemini Batch modes; Online remains the default. The card reports provider-waiting states explicitly because the provider does not expose continuous per-request completion inside a running batch.

### Error Handling

- Empty `choices`, empty generated text, malformed response payloads, rate limits, timeouts, and retryable provider 5xx responses are item-level retryable failures.
- Authentication, permission, invalid model, invalid prompt, and dimension mismatches fail immediately with a safe user-visible error.
- One failed item does not discard successful items from the same provider job.
- Cancellation stops new submissions, requests provider cancellation where supported, and marks unfinished files `CANCELED` without deleting uploaded blobs.
- Secrets, raw prompts, provider payloads, and legal document content are never returned by progress endpoints or logged in errors.

## Data Cleanup and Rebuild Procedure

After the implementation and tests pass:

1. Keep the existing background and reindex workers stopped.
2. Delete the current `tır-transit` document set through the application API.
3. Delete its detached user files and the prior orphaned upload through the existing user-file deletion workflow so PostgreSQL, MinIO, and Elasticsearch are reconciled.
4. Run the deletion worker until the old set, user files, regulatory chunks, and Elasticsearch projections are gone.
5. Create a new document set named `tır-transit`.
6. Upload every supported file under `/home/kubilay-payci/tır_transit` with `indexing_mode=gemini_batch`.
7. Start the batch worker and monitor the progress endpoint until completion.

## Testing

- Unit-test provider response classification: empty choices and empty text are retryable; permanent configuration errors are not.
- Unit-test item retry and exhaustion without rerunning successful items.
- Unit-test state-machine idempotency and restart-safe collection.
- Unit-test progress aggregation and safe error serialization.
- Integration-test batch upload, mocked provider job polling, precomputed embedding indexing, and final file/job states.
- Frontend-test mode selection, progress polling, stage labels, terminal errors, and default online compatibility.
- Live smoke-test one small file through the configured Vertex batch provider before deleting and rebuilding the full document set.

## Explicit Non-Goals

- Replacing the default online path.
- Guaranteeing a provider batch completion time.
- Persisting full embedding vectors in PostgreSQL.
- Building a generic batch abstraction for non-Gemini providers in this change.
