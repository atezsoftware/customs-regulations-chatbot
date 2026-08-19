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

---

## Review fix round 2/5

All four Important findings were closed without a production write, deployment,
push, or merge.

### RED/GREEN evidence

- Local-replica worker RED: the new multi-replica fixture failed with
  `TypeError` because live queue validation had no exact local-node boundary.
  Queue inspection now targets only
  `regulatory_indexing@<container-hostname>`, fails when only a healthy remote
  replica responds, checks the exact two-queue set, and correlates Celery
  `stats.pid` with the local Supervisor PID. The A/B focused repeat passed
  (`4 passed`), and the expanded readiness/gateway/wiring/deploy set finished
  at `109 passed in 10.19s`.
- Workload Identity RED: both observational probes returned the initial
  Compute Engine sentinel identity `default`. They now perform the list/get
  observation first, resolve the refreshed effective service-account identity,
  reject the sentinel, and still close every owned client. A valid post-request
  identity passes; GCS and Vertex identity mismatches against the attestation
  both fail closed.
- Deployment-attestation RED: the Compose wiring test raised `KeyError:
  'volumes'`, and executable preflight accepted a writable attestation mount
  with exit `0`. The canonical overlay now requires one read-only bind at
  `/run/readiness/regulatory-capabilities.json`; preflight validates that mount,
  a regular host file, mode `0600`, and numeric owner/group `1001:1001`.
  Readiness validates intended runtime UID `1001` rather than the invoking
  operator, and the runbook invokes it with `--user 1001:1001`.
- Attestation authenticity now requires a lowercase 64-character
  `evidence_sha256` and an archived evidence reference ending in the exact
  `#sha256=<digest>` binding. Permission, identity, exact scope, and 24-hour
  freshness checks remain mandatory; no write/delete/cancel probe was added.
- Cleanup-boundary RED first failed with the intentionally absent disposable
  scope (`NameError`). A forced exception after the real Markdown file and
  durable job were prepared then exposed and drove fixes for wildcard-safe
  Elasticsearch enumeration, continuation after partial cleanup errors, eager
  User loading, and explicit model/provider deletion order. Final forced-failure
  GREEN: `1 passed in 11.18s`; both real pipeline tests together passed in
  `17.94s`. The ordinary pipeline and every setup mutation now share the same
  `finally` cleanup boundary.

### Cleanup and deployment contracts

- Every disposable database, file-store, user-file, job, and index identity is
  recorded immediately. Prefix fallback removes file records through the real
  PostgreSQL file-store deletion path (including large-object unlink), user
  files/jobs/users, SearchSettings, model/provider rows, and every exact
  Elasticsearch index returned for the disposable prefix. Cleanup tolerates
  failure before the first DB commit or before index creation.
- The forced-failure regression directly verifies zero SearchSettings, LLM
  provider/model, file-record/file-content, UserFile, durable-job,
  `pg_largeobject_metadata`, and Elasticsearch state. The final independent
  disposable-environment query returned `0|0|0|0|0|0` for Task 8 settings,
  LLM providers, file records, user files, jobs, and OpenRouter provider; the
  Task 8 Elasticsearch index listing was empty.
- `env.regulatory-prod.template` now requires the host attestation path and
  documents its `1001:1001`/`0600` contract. The runbook explains that the
  digest binds the attestation to archived operator evidence but the archive's
  own access controls establish authenticity; observational probes still do
  not prove mutation permissions.

### Round-2 verification

- Focused readiness, gateway, Compose wiring, and executable preflight:
  `109 passed in 10.19s`.
- Regulatory unit/runtime superset (broader than the prior 668-test command):
  `777 passed in 17.69s`.
- Real PostgreSQL repository plus the two real PostgreSQL/Elasticsearch
  pipeline tests: `30 passed in 21.78s`.
- Ruff and Ruff formatting on touched Python: pass. `ty` on touched Python:
  pass. Touched-file pre-commit passed every applicable hook, including lazy
  imports, `ty`, Ruff, Ruff formatting, YAML, shellcheck, large-file,
  ripsecrets, and environment drift. The final post-hook repeat was
  `109 passed in 10.32s` plus `2 passed in 17.92s`. Migration files were not
  touched; the fresh disposable PostgreSQL was upgraded from empty to
  `c8f1a6d4e2b7` successfully before the real tests.
- The Elasticsearch production client was not changed in this round, so the
  seven-minute 70-test client suite was not repeated. The affected cleanup and
  durable publication paths were exercised against Elasticsearch 9.4.2 by the
  two pipeline tests and the 30-test real-dependency suite.

### Live readiness recheck

`getent hosts psql.dev.singlewindow.io` again returned exit `2`. Both nginx
ports returned HTTP `200`, while `onyx-api_server-1` remained `unhealthy`.
The read-only readiness command returned exit `1`, `NOT_READY`: migration and
Admin snapshot failed with redacted `OperationalError`, Supervisor inspection
failed, host cgroup evidence was unavailable, and attestation/provider/index
checks were blocked by the absent snapshot. No live canary was started and no
existing production document, index, database/provider record, GCS object, or
Vertex job was mutated.

---

## Review fix round 3/5

All five Important findings were implemented without a production write,
deployment, push, or merge.

### RED/GREEN evidence

- PID fail-closed RED: the four missing, zero, non-numeric, and non-running PID
  cases produced `1 failed` because the parser returned `None`. It now requires
  a positive PID from an exact `RUNNING` Supervisor status line and always
  compares that integer with the exact-destination Celery `stats.pid`.
- Archived-evidence RED: the separate-file regression initially failed with
  `TypeError: unexpected keyword argument 'expected_owner_gid'`; the Compose
  wiring regression separately failed because the evidence volume was absent.
  Readiness now validates two distinct regular, non-symlink files, numeric
  owner/group `1001:1001`, modes `0600`/`0400`, non-empty bounded sizes, and a
  streaming SHA-256 of the archived evidence's actual bytes. The attestation
  digest and reference must bind that computed digest; evidence content is
  never parsed or emitted.
- Supervisor socket RED: the wiring assertion reported mode `0700`, not the
  required app-user contract. The canonical socket is now numeric
  `1001:1001`, mode `0770`, with no world bits. The test also starts a real
  Supervisor 4.3.0 instance using the canonical numeric/mode shape, performs a
  successful `supervisorctl status` through its Unix socket, and verifies the
  resulting owner/group/mode.
- Cleanup/provider regressions were authored against the prior unconditional
  `finally` cleanup and global OpenRouter delete behavior. The disposable
  services were started after the staged cleanup/advisory-lock implementation,
  so there is no standalone pre-change pytest transcript for these two cases.
  Their final real-infrastructure run passed: an injected Elasticsearch cleanup
  failure is attached to, but does not replace, the original sentinel; later
  file-store, UserFile/job, configuration, provider, and unlock stages still
  execute. A PostgreSQL session advisory lock excludes a contender, the exact
  pre-existing raw provider row (including encrypted API-key bytes) is restored,
  and an absent singleton is removed only through the recorded ORM instance
  after its entire test state is revalidated.

### Deployment and safety contracts

- The production-lite overlay requires distinct read-only bind mounts at
  `/run/readiness/regulatory-capabilities.json` and
  `/run/readiness/regulatory-capability-evidence.json`. Executable preflight
  validates uniqueness, distinct paths, file kind, symlink rejection,
  owner/group, modes, and size bounds. The environment template and runbook
  define both host inputs and the exact app-UID readiness command.
- The runbook now instructs operators to hash the archived IAM artifact's
  actual bytes and place that digest in the separate metadata attestation. It
  continues to state that observational GCS/Vertex probes do not prove
  create/delete/cancel authority.
- Cleanup attempts every stage independently and aggregates failures only when
  no primary pipeline error exists. When a primary error exists, cleanup
  failures are attached as exception notes and cannot replace it. The final
  disposable-environment query returned zero Task 8 SearchSettings, LLM
  providers, file records, UserFiles, durable jobs, and OpenRouter provider
  rows; the Task 8 Elasticsearch index listing was empty.

### Round-3 verification

- Focused readiness, Vertex/GCS gateway, Supervisor/Compose wiring, and
  executable preflight: final repeat `120 passed in 13.89s` (the executable Supervisor
  socket test was then repeated after numeric ownership was finalized:
  `1 passed, 24 deselected in 1.42s`).
- Regulatory unit/runtime superset: final repeat `788 passed in 21.66s`.
- Real PostgreSQL 15.2 repository plus PostgreSQL/Elasticsearch 9.4.2 pipeline:
  final repeat `32 passed in 40.21s`; the pipeline file independently passed `4 passed in
  37.12s`, including forced setup failure, forced cleanup-stage failure,
  provider restoration/lock contention, and the full durable pipeline.
- Fresh disposable PostgreSQL migration from empty to head succeeded. No
  migration file or Elasticsearch production client changed in this round, so
  a downgrade cycle and the seven-minute 70-test ES client suite were not
  repeated.
- Ruff and Ruff formatting on touched Python passed; `ty` on touched Python
  passed. Host `shellcheck` was unavailable, while the repository pre-commit
  shellcheck hook and every other applicable touched-file hook passed. The
  final touched-file pre-commit run passed every applicable hook, including
  lazy imports, `ty`, Ruff, Ruff formatting, YAML, shellcheck, large-file,
  ripsecrets, and environment drift.

### Live readiness recheck

`getent hosts psql.dev.singlewindow.io` returned exit `2` again.
`onyx-api_server-1` remained `unhealthy`; nginx ports `7000` and `7001` both
returned HTTP `200`. The corrected local read-only command, with both canonical
evidence paths, returned exit `1`, `NOT_READY`: migration/Admin snapshot failed
with redacted `OperationalError`, Supervisor inspection failed, host cgroup
evidence was unavailable, and all snapshot-dependent checks were blocked. No
attestation was fabricated and no live canary was started. Existing production
documents, indices, database/provider state, GCS objects, and Vertex jobs were
not mutated.
