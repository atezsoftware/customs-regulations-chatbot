# Gemini Files Batch with Admin Service Account Design

## Goal

Replace the durable regulatory contextualization job's Vertex AI + Google Cloud
Storage transport with Gemini Developer API Files + Batch while authenticating
from the service-account JSON already stored in the selected Admin Vertex model
provider.

## Requirements

- Do not require or use `REGULATORY_INDEXING_GCS_URI`.
- Keep the Admin model provider as the credential source of truth.
- Parse the stored service-account JSON only at execution time; never persist it
  in job snapshots, provider resource names, errors, or logs.
- Exchange the service-account credential for short-lived OAuth access tokens and
  call the Generative Language API with `Authorization: Bearer` and
  `x-goog-user-project` headers.
- Upload one bounded JSONL input with Gemini Files API, submit one Gemini Batch
  job, poll it durably, download its generated JSONL result, and delete both
  input and output Files resources during cleanup.
- Keep existing PostgreSQL `vertex_*` columns and cleanup enum values as legacy
  persistence names. Add only a submission-state constraint value for
  crash-durable partial-attempt cleanup.
- Preserve the current per-item correlation, retries, cancellation, OpenRouter
  embedding, Elasticsearch publication, and feature flag behavior.
- Preserve service-account and workload-identity Admin validation, but fail
  closed when the credential identity or Google project cannot be resolved.

## Architecture

The pure contextual request/result contracts stay in `vertex_batch.py` for
snapshot and database compatibility. New jobs use a
`GoogleGeminiFilesBatchGateway` in `gemini_files_batch.py`. It owns OAuth token
refresh, resumable Files upload, Batch CRUD/list operations, result download,
and Files cleanup. A narrow `legacy_vertex_batch.py` compatibility gateway
continues draining GCS-backed snapshots created before rollout. The
orchestrator routes by the persisted snapshot transport; new resource values
are `files/...` and `batches/...` names rather than `gs://...` URIs.

`VertexBatchConfig` remains the serialized field name for backward-compatible
job snapshots. It accepts but never serializes a legacy `gcs_uri`. `location`
is retained only because the selected Admin provider and tokenizer/model
configuration use it; the Gemini Developer REST base URL is global. New
snapshots therefore contain no storage location.

## Error and Recovery Contract

- Timeouts, network failures, HTTP 408/429/5xx, and indeterminate Batch create
  responses retain the existing retry/reconciliation behavior.
- Reconciliation lists Batch jobs and matches the stable `displayName`.
- A successful job must expose `response.responsesFile` (with the SDK metadata
  shape retained as a compatibility fallback); otherwise it is malformed.
- Cleanup accepts only `files/...` resource names captured on the job. A 404 is
  idempotent success at the orchestrator cleanup boundary.
- Partial-result retries persist `RETRY_CLEANUP_REQUIRED` before any external
  deletion, then idempotently delete the batch and both Files resources before
  clearing the attempt references.
- Raw upstream response bodies are never included in exceptions.

## Tests

- Unit-test JSONL contract and result parsing unchanged.
- Unit-test service-account JSON parsing, OAuth headers, resumable upload,
  Batch create/get/list/cancel/delete, result download, and file deletion with a
  fake HTTP session.
- Unit-test configuration resolution without a GCS environment variable.
- Unit-test orchestrator submission, polling, apply, cancellation, and terminal
  cleanup using Files resource names.
- Run the complete regulatory indexing unit package, readiness script tests,
  targeted external-dependency contract tests, typing/lint hooks, and a live
  canary only when valid production Admin and database access is available.

## Non-goals

- Changing OpenRouter embedding transport.
- Renaming legacy database columns or cleanup enum values in this increment.
- Creating a new Gemini API key or storing a second credential.
- Starting a production canary that creates user data without authenticated
  Admin access.
