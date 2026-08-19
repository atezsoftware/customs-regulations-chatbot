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

---

## Review fix round 1/5

All five Important findings and the one Minor finding were addressed without a
production write, deployment, push, or merge.

### RED/GREEN evidence

- Readiness contract RED: `7 failed, 2 passed in 1.46s`. The failures proved the
  missing capability-attestation report/CLI, exact supervisor queue parser,
  direct dense-vector dimension check, and attestation scope/permission/age
  validator. GREEN after implementation: `10 passed`. Expanded field/attribute
  coverage plus the Vertex gateway file finished at `49 passed in 2.36s`.
- Live Celery queue validation had a separate RED:
  `1 failed, 9 deselected in 1.20s` because the exact active-queue validator did
  not exist. It now accepts only the two production queues on every matching
  `regulatory_indexing@...` worker.
- Public gateway probe RED: `1 failed, 34 deselected in 1.38s` because the public
  GCS probe did not exist. GREEN: all `35 passed in 1.27s`. Both public probes
  report only credential identity, perform only list/get observations, and own
  and close their GCS/google-genai clients. All existing GCS gateway operations
  now also close their storage client.
- The first refactored pipeline invocation was blocked before test behavior by
  the absent disposable PostgreSQL (`password authentication failed`); after
  starting fresh disposable PostgreSQL 15.2 and Elasticsearch 9.4.2 and applying
  migrations, the production-path test passed: `1 passed in 16.65s`.

### Hardened contracts

- Elasticsearch readiness retains the broad schema compatibility check and now
  directly reads the mapping. Both `content_vector` and `title_vector` must
  match the active snapshot for `type`, `dims`, implicit/explicit
  `element_type`, `index`, and `similarity`. A `768` versus `1536` regression
  fails on `content_vector.dims`, and the client closes on failure.
- Supervisor configuration is parsed with `ConfigParser` and `shlex`; queue
  names in app paths, hostnames, or log paths cannot satisfy readiness. The one
  exact `-Q`/`--queues` value and each live Celery `active_queues` response must
  equal `regulatory_indexing,user_file_processing`.
- Readiness fails closed without an owner-only mode-`0600`, at-most-24-hour IAM
  attestation for the exact active GCS URI, Vertex project/location/model, and
  runtime credential identity. It enumerates GCS object create/get/delete/list
  and Vertex batch create/get/cancel/list plus model get. The command does not
  issue write/delete/cancel probes; the runbook now states that observational
  list/get calls cannot prove mutation permissions.
- The durable pipeline creates real disposable OpenRouter embedding, Vertex LLM
  provider/model, and active SearchSettings rows in PostgreSQL. It resolves the
  immutable snapshot through `prepare_regulatory_indexing_job`, leaves
  `validate_snapshot_for_stage` active for every stage, writes Markdown bytes to
  `PostgresBackedFileStore`, and loads them through the production Markdown
  loader. Its chosen reduced dimension is `768`, but every client, mapping,
  request, and vector assertion derives from `SearchSettings.final_embedding_dim`
  and explicitly proves it is not `1024`. Duplicate delivery, partial
  retry-only-missing context, stale recovery, hidden staging, publish, and
  cancellation cleanup assertions remain intact.
- The broad Elasticsearch highlight assertion now accepts only the two exact
  known representations: `['Artificial', 'intelligence']` or
  `['Artificial intelligence']`.

### Fix-round verification

- Focused readiness and gateway final repeat: `49 passed in 1.90s`.
- Unit/runtime superset of the earlier 576-test command: `668 passed in 21.11s`.
- Real PostgreSQL repository plus real PostgreSQL/Elasticsearch pipeline:
  `29 passed in 15.11s`.
- Full real Elasticsearch client suite: `70 passed in 408.01s`.
- Disposable migration round trip:
  `c8f1a6d4e2b7 -> f7b2e4c6a8d1 -> c8f1a6d4e2b7`; final database head query
  returned `c8f1a6d4e2b7`.
- Ruff on touched Python: pass. Ruff formatting: pass. `ty` on touched Python:
  pass. File-scoped pre-commit passed every applicable hook, including lazy
  imports, `ty`, Ruff, Ruff formatting, large-file, ripsecrets, and environment
  drift validation.
- Disposable cleanup query returned zero Task 8 SearchSettings, LLM provider,
  and file-store records, and the Task 8 index glob returned no index. The
  isolated `task8-review-postgres` and `task8-review-elasticsearch` containers
  were then removed.

### Live readiness recheck

`getent hosts psql.dev.singlewindow.io` still returned exit `2`. Both local nginx
ports returned HTTP `200` for `/api/health`, but the read-only readiness command
returned exit `1`: migration/Admin snapshot `OperationalError`, supervisor query
failure, unavailable host cgroup evidence, and all snapshot-dependent checks
blocked. No capability evidence was fabricated. Therefore no live Vertex,
OpenRouter, upload, retrieval/citation, retry/restart, or cleanup canary was
started. Existing production documents, indices, database/provider state, and
object storage were not mutated.
