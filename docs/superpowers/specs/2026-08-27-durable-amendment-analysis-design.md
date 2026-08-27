# Durable Amendment Analysis Design

## Status

Approved for implementation on 2026-08-27. The synchronous analysis request
will be replaced by a durable, checkpointed Celery workflow. The design assumes
document sets with roughly 15,000 regulatory chunks and potentially long,
multi-instruction amendment documents.

## Problem

`POST /regulatory/amendments/analyze` currently performs the entire amendment
pipeline before returning:

1. one LLM call segments the submitted text into atomic instructions;
2. indexed PostgreSQL queries find a small candidate set for each instruction;
3. one LLM call selects the matching existing chunk for each instruction; and
4. one LLM call drafts the replacement chunk and its effective dates.

Steps 3 and 4 run for every instruction. A multi-article PDF can therefore keep
the HTTP request open beyond the outer proxy's response limit. The proxy returns
524 even though the API may still be working. Raising the repository's Nginx
timeout cannot remove a timeout imposed by an upstream proxy.

The amendment tables already expose an `analyzing` state, but the work is still
owned by the request process and no durable progress is recorded. A worker or
pod restart can repeat or abandon the whole analysis.

## Goals

- Return from the analyze request quickly without waiting for any LLM output.
- Continue analysis after browser disconnects and API deployments.
- Resume after the last committed instruction instead of repeating completed
  LLM calls.
- Prevent duplicate Celery delivery from creating duplicate proposals.
- Show queued, running, completed, failed, and numeric progress in Updates.
- Preserve the exact document-set file scope captured when the batch is queued.
- Keep a 15,000-chunk document set out of the LLM context; only the top bounded
  candidates may be sent to matching and drafting.
- Keep the production-lite memory footprint bounded.

## Non-goals

- Sending all document-set chunks to an LLM.
- Replacing PostgreSQL candidate search with vector search.
- Applying proposals automatically after analysis.
- Running multiple instructions from the same batch concurrently in the first
  version. Sequential instruction execution makes checkpointing, provider rate
  limits, and proposal ordering predictable.
- Adding another heavyweight Python worker process to the 4 GiB production-lite
  background pod.

## LLM Boundaries

The LLM remains responsible only for semantic work that deterministic code
cannot safely perform:

1. **Segmentation** (`AMENDMENT_SEGMENTATION`): converts extracted or pasted
   amendment text into typed atomic instructions, article references, date
   phrases, and a publication/reference date.
2. **Match confirmation** (`AMENDMENT_MATCH_CONFIRMATION`): receives one
   instruction and at most five candidates found by PostgreSQL, then selects a
   candidate ID or explicitly identifies a new provision.
3. **Drafting** (`AMENDMENT_DRAFTING`): receives one instruction, the selected
   old chunk (or one sibling reference for a new provision), and the reference
   date, then produces the complete proposed chunk and validity dates.

PostgreSQL performs scope enforcement and candidate narrowing. The existing GIN
trigram indexes on chunk text and heading text, the expression index on
`article_no`, status filtering, document-set file filtering, and `LIMIT` remain
the scale boundary. Approximately 15,000 chunks are searched in the database;
they are never serialized into a prompt. Candidate IDs returned by the LLM are
still checked against the supplied candidate set.

## Durable State

`amendment_batch` gains the state required to make PostgreSQL the source of
truth for orchestration:

- status: `queued`, `analyzing`, `analyzed`, or `failed`;
- stage: `queued`, `segmenting`, `processing`, or `finalizing`;
- immutable `user_file_ids` scope snapshot;
- serialized, schema-validated segmented instructions;
- serialized unmatched instructions;
- `instruction_count` and `processed_instruction_count`;
- `lease_generation` for fencing stale workers;
- `heartbeat_at`, `started_at`, and `completed_at`; and
- the existing `error_message` and `reference_date` fields.

`amendment_proposal` gains a unique constraint on
`(batch_id, instruction_index)`. A transaction that persists a proposal or an
unmatched instruction also advances `processed_instruction_count`. A crash can
therefore leave either the complete instruction result and its checkpoint, or
neither; it cannot leave a half-checkpointed instruction.

The stored file-ID scope is captured after the API authorizes the selected
document set. Background processing uses that immutable list even if the set is
edited while analysis is running. Approval keeps its current authorization and
scope checks.

## Queue and Worker Design

A new logical queue and task names are added:

- queue: `regulatory_amendment`;
- task: `regulatory_amendment_run`;
- recovery task: `regulatory_amendment_recover_stale`.

The existing production-lite regulatory LLM Celery app discovers the amendment
task module and consumes both `regulatory_benchmark` and
`regulatory_amendment`. Its thread concurrency is configured with enough
capacity for one benchmark run and one amendment delivery without starting a
new Python process. Amendment tasks use high priority; benchmark tasks retain
medium priority. Only one instruction within a batch executes at a time.

Every send includes `tenant_id`, a bounded expiration, `acks_late`, and
`reject_on_worker_lost`. The task payload contains only `batch_id` and
`tenant_id`; legal text and file IDs remain in PostgreSQL.

An atomic database claim increments `lease_generation`. All progress and
terminal writes include the claimed generation in their update predicate, so a
late worker cannot overwrite a newer owner. A lightweight heartbeat renews the
database lease while an LLM call is in flight.

The existing durable regulatory Beat process dispatches recovery scans. A scan
claims queued batches whose initial delivery was lost and analyzing batches
whose heartbeat is stale, advances their lease, and emits a fresh short-lived
Celery message. Duplicate messages are safe because the database claim and
proposal uniqueness are authoritative.

## Execution Flow

1. The API authorizes the administrator and document set, snapshots its file
   IDs, creates a `queued` batch, commits it, and sends two idempotent delivery
   hints (immediate and short delayed), following the established regulatory
   benchmark dispatch pattern.
2. The API returns the batch snapshot with HTTP 202. It does not construct an
   LLM client or wait for analysis.
3. The worker claims the batch. If no segmented instruction checkpoint exists,
   it marks the stage `segmenting`, runs segmentation, validates the structured
   result, and commits the instructions, reference date, and total count.
4. Starting at `processed_instruction_count`, the worker finds candidates for
   one instruction using the immutable file scope.
5. No candidates produces an unmatched checkpoint without an LLM match or draft
   call. Otherwise the worker confirms the match, validates candidate scope,
   drafts the new chunk, and commits one proposal plus progress.
6. The worker repeats step 4 sequentially until every instruction is committed.
7. Finalization verifies that proposal plus unmatched counts equal the total,
   then marks the batch `analyzed` with `completed_at`.
8. A terminal validation/provider error marks the batch `failed` without
   deleting checkpoints. An admin retry changes it back to `queued`; processing
   resumes from the first incomplete instruction.

If the initial broker send fails after the batch commit, periodic recovery picks
up the queued row. If the worker dies, late acknowledgement and stale recovery
redeliver it. If a user retries while an old delivery exists, lease fencing
prevents the old delivery from mutating the new run.

## API Contract

`POST /regulatory/amendments/analyze` keeps its request body and authorization,
but returns an `AmendmentBatchSnapshot` with HTTP 202 instead of waiting for an
`AnalyzeAmendmentResponse`.

`GET /regulatory/amendments/batches/{batch_id}/analysis` returns:

- the current batch snapshot and progress fields;
- proposals committed so far; and
- unmatched instructions committed so far.

It applies the same editable-document-set authorization as the existing batch
and proposal endpoints.

`POST /regulatory/amendments/batches/{batch_id}/retry` is permitted only for a
failed batch. It clears the terminal error/timestamps, preserves segmentation
and completed instruction checkpoints, advances the lease generation, queues a
new delivery, and returns the queued batch snapshot.

Existing list, approve, and reject endpoints remain compatible. Response errors
use `OnyxError` and never expose prompts, provider payloads, or stack traces.

## Frontend Behavior

Selecting Analyze queues the batch and immediately switches the page to its
history entry. The editable source text is cleared only after the queue request
succeeds.

While a selected batch is active, the page polls its analysis endpoint. It
shows the stage and `processed / total` progress when the total is known. Polling
stops on `analyzed` or `failed`, when the component unmounts, or when another
batch is selected. The interval is bounded and does not start overlapping
requests.

Committed proposals may be displayed as they arrive, but approval and rejection
remain disabled until the batch is fully analyzed. This prevents reviewing a
partial result without knowing later instructions may target the same chunk.

A failed batch shows its safe error message and a Retry action. Reloading the
page reconstructs the same state from the batch history; browser memory is not
the source of truth.

## Capacity and Failure Policy

- Candidate queries keep their current top-five prompt boundary and use the
  existing PostgreSQL indexes. No full chunk collection is loaded into Python.
- One batch processes instructions sequentially. Worker concurrency provides
  bounded cross-job capacity, not unbounded per-instruction fan-out.
- Structured LLM calls retain their existing provider retry and timeout policy.
  Checkpointing prevents completed calls from being repeated after a later
  failure.
- Queue messages expire; database recovery, not an indefinitely retained broker
  message, owns eventual execution.
- Error text stored for administrators is length-bounded and excludes submitted
  legal text and provider response bodies.
- Logs include tenant, batch, stage, instruction index, lease generation,
  candidate count, elapsed time, and status, but not raw legal text.

## Testing

### Backend

- The analyze endpoint commits and returns a queued batch without invoking an
  LLM synchronously.
- Dispatch includes tenant, queue, priority, expiration, and duplicate-safe
  delivery hints; total dispatch failure still leaves a recoverable queued row.
- Segmentation is persisted once and reused on redelivery.
- Each proposal/unmatched result and progress increment commit atomically.
- Duplicate delivery cannot create duplicate proposals.
- A stale lease generation cannot write progress or terminal state.
- A killed worker/redelivery resumes at the first incomplete instruction.
- Stale queued/analyzing batches are recovered; fresh heartbeats are not.
- Failed batches can be retried; analyzed and active batches cannot.
- Batch analysis reads require authorization for the batch's document set.
- Candidate matching remains scoped and capped at five for a 15,000-row test
  corpus or an equivalent query-plan/index integration fixture.
- Celery app, queue, Beat schedule, supervisor, log forwarding, and deployment
  readiness wiring include amendment analysis.

### Frontend

- Analyze handles a queued response and never waits for proposal data.
- The selected active batch polls without overlapping requests.
- Progress, completed proposals, unmatched instructions, and terminal errors are
  rendered from server state.
- Polling stops on terminal state and batch selection changes.
- Retry queues only a failed batch.
- Existing approval/rejection behavior remains disabled for partial analysis and
  enabled after completion.

### Verification

- Run focused backend unit tests and migration tests.
- Run focused frontend service/component tests, formatting, and type checks.
- Run runtime dependency, Celery wiring, and production-lite supervisor tests.
- Deploy to `test/v1`, upload a multi-instruction PDF, confirm the POST returns
  before the proxy timeout, observe progress, and verify completion.
- During a canary run, restart the background pod and verify the same batch
  resumes without duplicate proposals or repeated completed instructions.

## Deployment

The Alembic migration runs before the updated API and worker become ready. The
background release continues using the runtime-lite image and its 4 GiB memory
limit. No PDF parser or new model dependency is added to the worker.

Deployment readiness verifies that the regulatory LLM worker consumes the new
queue and has registered both run and recovery tasks. Monitoring adds amendment
queue depth and batches by status/stage. The old synchronous behavior is removed
rather than retained behind a second endpoint, so all input modes receive the
same durable behavior.
