# Durable Amendment Approval Indexing Design

## Problem

The amendment approval endpoint correctly returns before indexing, but the
background approval task still performs a full `project_user_file_to_index`
call inside the same database transaction that applies the legal version
change. For large regulatory files this can run for more than an hour while
holding proposal and chunk locks. The proposal has no approval heartbeat or
durable indexing-job identity, so the UI can remain in `approving` forever if
the worker is lost or the projection stalls.

The fix must preserve the existing legal-versioning contract: the historical
chunk remains stored and receives an end date, while the replacement is a new
active chunk. Search must expose the new version only after its index
projection has been published successfully.

## Chosen Design

Approval becomes a short, durable state transition followed by the existing
regulatory indexing job workflow.

1. The approval worker locks the proposal, creates the replacement chunk,
   closes the old chunk, records the replacement on the proposal, and commits.
   It does not call an embedding provider or Elasticsearch while this
   transaction is open.
2. The worker creates a durable regulatory indexing job from the canonical
   chunk rows for that user file and stores the job ID on the proposal.
3. The regulatory indexing worker performs contextualization, embedding,
   hidden index staging, verification, and atomic publication using its
   existing heartbeat, retry, fencing, and stale-recovery behavior.
4. Successful publication marks every linked proposal `approved` in the same
   database transaction that marks the indexing job successful. This prevents
   the UI from claiming success before the new search projection is visible.
5. Terminal indexing failure marks linked proposals with a visible approval
   error state. A retry reuses the already-created legal version and restarts
   only its failed indexing job; it never creates another replacement chunk.

## State and Schema

`amendment_proposal` gains:

- `approval_indexing_job_id`: nullable foreign key to
  `regulatory_indexing_job.id`.
- `approval_error`: nullable safe user-facing failure message.

The proposal status constraint gains `approval_failed`. The existing
`applied_new_chunk_id` distinguishes a proposal whose legal DB transition has
already been applied from a legacy queued proposal that never started.

The durable chunk-row preparation path is extended to accept a completed user
file during an amendment reprojection. Preparation moves that file to
`INDEXING`; successful publication returns it to `COMPLETED`. Existing active
job fencing remains the single authority for concurrent file projections.

## Recovery and Compatibility

The periodic amendment recovery task also reconciles approval state:

- A linked succeeded job finalizes the proposal as `approved`.
- A linked terminal failed or cancelled job marks it `approval_failed` with a
  safe message.
- A stale legacy `approving` proposal with neither an applied chunk nor an
  indexing job is returned to `pending`; this repairs approvals created by the
  previous deployment.
- A proposal with an applied chunk but no job is re-enqueued so indexing can be
  prepared idempotently.

The approval task is safe to redeliver. If `applied_new_chunk_id` exists, it
skips the legal mutation and resumes from durable indexing preparation.

## API and UI

The existing approve endpoint remains asynchronous and returns HTTP 202.
Pending proposals accept reviewed field edits. `approval_failed` proposals
offer retry without accepting further legal-content edits, because their new
chunk version already exists in PostgreSQL.

The admin UI continues polling while a proposal is `approving`. It shows:

- informational progress while the durable job is active;
- success only after index publication;
- a clear error and retry action for `approval_failed`;
- no infinite generic running state for terminal failures.

## Error Handling and Observability

Logs include proposal ID, user-file ID, indexing job ID, and lifecycle event at
dispatch, DB application, job linkage, publication completion, failure, and
recovery. Provider exception details stay in server logs; the proposal stores
only a bounded safe message.

Dispatch failure after the DB transition does not roll back or duplicate the
legal version. The recovery task detects the applied proposal without a job
and resumes it.

## Tests

Backend tests verify that:

- the approval task never calls `project_user_file_to_index`;
- legal mutation commits before durable job preparation and no duplicate chunk
  is created on redelivery;
- proposal and indexing-job linkage is persisted;
- publication success atomically finalizes linked proposals;
- terminal failure becomes `approval_failed`;
- stale legacy proposals and applied-but-unlinked proposals recover correctly;
- retry does not reapply the legal amendment;
- worker registration, queue expiry, and deployment queue wiring remain valid.

Frontend tests verify polling success, visible terminal failure, and retry.
After deployment, a browser test approves a real DEV proposal, observes the
transition out of `approving`, and verifies the amended text is retrievable.
