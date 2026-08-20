# Durable Regulatory Indexing Fix Wave 4

Date: 2026-08-20

## Outcome

The three confirmed lifecycle findings are fixed:

- a claimed normal/recovery delivery fences chunk-generation drift into the same
  job's durable `SUPERSEDE` cancellation before snapshot validation or stage work;
- deletion drains historical job generations sequentially with at most one active
  `USER_DELETE` cancellation; and
- Elasticsearch `INDEX_DELETE` failure for supersession or deletion stays on the
  durable cleanup phase with a visible error, attempt, and capped retry deadline.

Compatibility-only unresolved PREPARING jobs still load and resolve their exact
input hash before generation drift can supersede them. `USER_CANCEL` deliberately
retains its existing bounded best-effort cleanup behavior.

## RED evidence

- The unresolved PREPARING and claimed-drift unit slice failed `2` tests: PREPARING
  raised generation mismatch before hash resolution, while ordinary drift reached
  snapshot validation and returned `SKIPPED` instead of a fenced cancellation.
- The real PostgreSQL drift test failed because the atomic supersession repository
  operation did not exist.
- The real PostgreSQL three-generation deletion test failed with
  `uq_regulatory_indexing_job_active_user_file`, proving the prior loop activated
  multiple historical rows in one transaction.
- The cleanup safety matrix failed `2` cases because unclassified and
  attempt-exhausted `INDEX_DELETE` errors advanced to `FINALIZE`.

## GREEN evidence

- Snapshot/orchestrator unit suites: `59 passed in 2.21s`.
- Real PostgreSQL durable job repository suite: `56 passed in 8.02s`.
- Real PostgreSQL/Redis user-file processing and deletion queue suites:
  `11 passed in 1.81s`.
- Disposable PostgreSQL + Elasticsearch `9.4.2` + HTTP embedding pipeline:
  `1 passed in 13.20s`. This invokes the real regulatory Celery delivery, real
  processing scanner, and `process_single_user_file` worker. A one-shot
  Elasticsearch deletion failure leaves old chunks visible, records same-phase
  retry state, prevents scanner successor creation, then recovers and publishes
  the current generation.
- Final post-format combined unit, real PostgreSQL, and Elasticsearch pipeline
  run: `116 passed in 15.86s`.
- Target-file pre-commit (including `ty`, Ruff lint/format, secret scan, and
  environment-drift checks): passed on the second clean run after formatter edits.

## Data and migration impact

No schema change was needed. The existing partial unique active-job index, typed
cancellation intent, phase, attempt, retry deadline, error fields, and lease
generation provide the required durable state.

## Deferred scope

Concrete provider-specific Elasticsearch transient exception normalization remains
in the later provider wave. Unknown cleanup errors already fail safe and remain
observable; they are never treated as successful cleanup.
