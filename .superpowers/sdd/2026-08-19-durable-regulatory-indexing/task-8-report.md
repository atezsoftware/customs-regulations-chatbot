# Task 8 report — integration verification and readiness

## Outcome

Task 8 implements the clean-baseline repair, external dependency pipeline,
read-only readiness command, runbook cutover/canary procedure, and every deferred
minor from the ledger. The automated pipeline uses only disposable PostgreSQL,
Elasticsearch, tenant, user, file, job, and index identities. No production
document, index, database record, object, or provider state was mutated.

## TDD evidence

- Projection baseline RED: `2 failed, 6 passed`. The setting-specific test leaked
  the contextual-RAG environment default; the contextual test patched a removed
  private boundary. GREEN after test-only repairs: `8 passed in 1.44s` (final
  repeat: `8 passed in 1.41s`). Production projection behavior is unchanged.
- Deferred real-PostgreSQL bound RED: a 200-character error code raised
  `StringDataRightTruncation`. GREEN after bounding every repository write to the
  persisted `VARCHAR(128)` contract.
- Deferred client cleanup RED: submit, reconcile, get, and cancel fake clients
  remained open. GREEN after a managed google-genai client boundary.
- Deferred ownership RED: contextual apply accepted a canonical row from another
  user file. GREEN after fail-closed ownership validation.
- Readiness RED: the new test initially failed collection with
  `ModuleNotFoundError: scripts.regulatory_indexing_readiness`; the runtime image
  copy test then failed because the script was not shipped; the memory-headroom
  CLI test failed because the attestation was absent. GREEN after implementation:
  `6 passed in 1.07s`, with Ruff and ty passing.
- The real pipeline exposed Elasticsearch 9.4.2 excluding dense-vector fields
  from ordinary `_source`. Hidden-stage verification first failed with
  `DocumentChunkVerificationError`. Explicit vector source includes made the
  direct mget parser and the complete pipeline green. The adjacent client suite
  exposed the same behavior at the single-document GET boundary; all seven
  affected tests passed after the same explicit include contract.

## External pipeline coverage

`test_durable_indexing_pipeline.py` runs through the durable database claim and
orchestration boundaries with:

- PostgreSQL 15.2 on `127.0.0.1:55433`, migrated to real head;
- Elasticsearch 9.4.2 on `127.0.0.1:59200`, a unique disposable index, real
  hidden documents, refresh, readback, verification, publish, and deletion;
- a local OpenRouter-compatible HTTP server reached through the production
  embedding client, deliberately returning out-of-order vectors;
- a deterministic fake Vertex Batch/GCS gateway, including a partial first
  result and a retry containing only the missing request;
- Markdown canonical chunk persistence, contextual text before original text,
  active effective dimension `1024` because that disposable SearchSettings row
  says `1024`, duplicate delivery fencing, stale-claim/recovery token restart,
  hidden staging, publication, completion, cancellation, remote cancel/cleanup,
  vector clearing, and staged Elasticsearch cleanup.

The test cleans the unique index and disposable database identities in `finally`.
The `1024` value is only test fixture data; production readiness always derives
the effective dimension from the active SearchSettings.

## Deferred-minor disposition

1. `error_code VARCHAR(128)`: fixed at every repository write boundary and proven
   by real PostgreSQL.
2. google-genai client close: fixed for submit/get/reconcile/cancel and asserted.
3. contextual apply ownership: fixed and asserted.
4. preparation duplicate/takeover/partial recovery: existing real-PostgreSQL
   takeover/repair tests remain green; the new pipeline also proves restart and
   idempotent duplicate delivery.
5. atomic preparation callback rollback: new real-PostgreSQL test proves callback
   chunk writes roll back on validation failure.
6. out-of-order OpenRouter success: new regression test plus the local HTTP
   pipeline prove correlation by response index.
7. direct Elasticsearch mget parser: new direct unit test covers out-of-order
   documents and exact vector-source request; real Elasticsearch confirms it.
8. live memory headroom: repository-owned proof remains impossible because Helm
   limits and node capacity are external. Readiness now fails closed unless an
   operator supplies `--memory-headroom-reviewed`, reports cgroup
   `memory.current`, `memory.peak`, `memory.max`, and OOM counters without
   secrets, and still fails on OOM. The runbook retains the measured cold-import
   evidence (~216568 KiB) and requires archived pod/node review.

## Verification

- Focused regulatory unit, dedicated worker/runtime wiring, readiness, prod-lite
  deployment, dependency split, and benchmark environment tests:
  `576 passed in 16.84s`.
- Real PostgreSQL job repository plus real PostgreSQL/Elasticsearch pipeline:
  `29 passed in 15.07s`.
- Pipeline-only post-type-fix repeat: `1 passed in 12.03s`.
- Projection clean baseline final repeat: `8 passed in 1.41s`.
- Real migration cycle on the disposable PostgreSQL:
  `c8f1a6d4e2b7 (head)` -> downgrade `f7b2e4c6a8d1` -> upgrade
  `c8f1a6d4e2b7 (head)`, all exit 0.
- Ruff on touched Python: pass.
- ty on touched Python: pass. Pyright/basedpyright executables are not installed
  in the locked environment; `uv run pyright` and `uv run basedpyright` each
  failed to spawn with `No such file or directory`.
- Pre-commit touched files: first run formatted files; second run passed every
  applicable hook (lazy imports, ty, Ruff, Ruff format, large-file, ripsecrets,
  and env-drift checks).
- Existing real Elasticsearch client suite initial run: `62 passed, 8 failed in
  407.70s`; seven failures shared the ES 9 dense-vector `_source` root cause and
  one accepted only the older two-span highlight representation. Targeted GREEN:
  `7 passed in 44.90s` and `1 passed in 6.71s`. Final full-suite GREEN after both
  fixes: `70 passed in 404.80s`.

## Live readiness and canary gate

The read-only readiness command was run from `backend` with
`PYTHONPATH=.`. It returned exit `1`, `NOT_READY`: migration and Admin snapshot
failed with redacted `OperationalError`; the local supervisor query failed; GCS,
Vertex, OpenRouter, and Elasticsearch checks were blocked by the absent Admin
snapshot. The first invocation without `PYTHONPATH=.` also recorded the local
checkout limitation (`ModuleNotFoundError: onyx`); the shipped runtime image sets
`PYTHONPATH=/app`, and the runbook command executes there.

Current environment checks:

- `getent hosts psql.dev.singlewindow.io` returned exit `2` (private DNS still
  unresolved).
- frontend/nginx `http://localhost:7000/api/health` and `:7001/api/health`
  returned success, but `onyx-api_server-1` remains Docker `unhealthy`.
- `onyx-relational_db-1`, `onyx-cache-1`, and `onyx-opensearch-1` have been
  stopped for 13 days; their service ports are not exposed.
- host Elasticsearch logged `User limit of inotify watches reached` during the
  extended disposable test suite. Tests continued, but this host-level resource
  warning must be corrected before treating the host as production-ready.

Because the migration/Admin/provider/data-service readiness gate failed, no live
Vertex Batch, real OpenRouter, frontend upload, retry/restart, retrieval/citation,
or deletion canary was started. This follows the required stop rule and avoids
touching existing production documents or indices.
