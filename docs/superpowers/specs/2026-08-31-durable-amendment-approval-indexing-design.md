# Gemini Amendment Approval Projection Design

## Problem

The approval endpoint returns before indexing, but the background task was
changed to prepare the OpenRouter-only durable regulatory indexing pipeline.
The active DEV search settings instead use Google `gemini-embedding-2` with a
1024-dimensional Elasticsearch index. Durable preparation therefore failed
before creating a job, while stale recovery kept re-enqueueing the same
impossible operation and left the UI in `approving` indefinitely.

The fix must preserve the legal-versioning contract: the historical chunk
remains stored and receives an end date, while the replacement is a new active
chunk. A proposal is successful only after the active search projection has
been written.

## Chosen Design

Approval remains asynchronous and uses the active regulatory projection path.

1. The approval worker locks the proposal, creates the replacement chunk,
   closes the old chunk, records the replacement on the proposal, and commits.
2. In a new transaction, the worker validates that the current search settings
   are Google `gemini-embedding-2` with final embedding dimension 1024.
3. The worker reprojects the canonical chunk rows for the affected user file
   through `project_user_file_to_index` and the exact validated PRESENT index.
   It never invokes a FUTURE embedding provider; instead, an existing FUTURE
   copy is marked for normal reconciliation. A separate approval heartbeat
   keeps stale recovery from dispatching a duplicate while this full-file
   operation is running.
4. Successful projection marks the proposal `approved`. A projection failure
   marks it `approval_failed` with a bounded safe message; provider details are
   retained only in server logs.
5. Retry reuses `applied_new_chunk_id`, repeats only the search projection, and
   never creates another replacement chunk.

The general-purpose durable regulatory indexing pipeline remains available for
its existing workflows. Amendment approval does not use it because that path
has a different OpenRouter embedding contract.

## State and Recovery

`applied_new_chunk_id` distinguishes a proposal whose legal DB transition has
already been applied from a queued proposal that never started. The existing
`approval_failed` and `approval_error` fields provide a terminal, visible
failure state.

Recovery behaves as follows:

- a live projection renews the proposal timestamp;
- an applied, stale `approving` proposal is re-enqueued and resumes from the
  existing replacement chunk;
- a stale proposal with no applied chunk returns to `pending`;
- a terminal projection failure remains `approval_failed` until an admin
  explicitly retries it.

A retry may reclaim a file left `FAILED` by a terminal legacy job. It does not
claim an `INDEXING` file, because an active durable job may still own that file.
The worker verifies and locks the same PRESENT settings row again after the
long projection before it marks the proposal approved, so a concurrent index
promotion becomes a safe terminal retry instead of a false success.

## API and UI

The approve endpoint remains asynchronous and returns HTTP 202. Pending
proposals accept reviewed field edits. `approval_failed` proposals expose a
retry endpoint and a `Retry indexing` action without allowing further legal
content edits, because the replacement chunk already exists in PostgreSQL.

The admin UI polls while a proposal is `approving`, shows success only after
projection, and replaces the endless running message with a clear terminal
error and retry action when projection fails.

## Tests

Backend tests verify that:

- legal mutation commits before active-index projection;
- only Google `gemini-embedding-2` with 1024-dimensional vectors is accepted;
- long projection renews its approval heartbeat;
- success and failure become explicit terminal proposal states;
- retry reuses the existing applied chunk;
- worker registration, queue expiry, and stale recovery remain valid.

Frontend tests verify the visible terminal failure and retry action. Live DEV
verification checks the proposal transition, Elasticsearch mapping/vector
dimension, and retrieval of the amended content after the `develop` deploy.
