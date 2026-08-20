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

---

## Review fix round 4/5

All three remaining Important findings were implemented without a production
write, deployment, push, or merge.

### RED/GREEN evidence

- Secure-reader and Supervisor contract RED was `7 failed in 2.92s`: the
  file-only readiness mode and descriptor-owned reader did not exist, preflight
  did not delegate to the canonical reader, and a host-identity-rewritten
  Supervisor test could not prove numeric `1001:1001`. Readiness now opens each
  path exactly once with `O_RDONLY|O_NOFOLLOW|O_CLOEXEC`, validates that same
  descriptor with `fstat`, requires the exact owner/group and `0600`/`0400`
  mode contract, and reads at most the configured maximum plus one byte. The
  exact bounded bytes are parsed/hashed and every descriptor is closed. Tests
  deterministically cover path replacement after open, final-component
  symlinks, initial oversize, growth after `fstat`, absent safe-open flags, and
  validation-failure closure without emitting evidence contents.
- Production-lite preflight removed its separate host `stat` security claim.
  After the immutable background image and read-only mounts are resolved, it
  runs that image as `1001:1001` in a disposable read-only, no-network,
  capability-dropped container and invokes
  `--validate-capability-files-only`. This mode returns before database engine
  initialization or any network-backed check. The final readiness, wiring, and
  executable-preflight repeat was `91 passed in 13.44s`.
- The prior Supervisor executable test substituted the invoking host UID/GID.
  The replacement external/runtime test builds the current `runtime-lite`
  target, starts Supervisor as root, applies the canonical numeric
  `1001:1001`/`0770` socket configuration, proves `supervisorctl status` works
  after dropping to UID/GID `1001:1001`, and proves UID/GID `2002:2002` cannot
  connect. The same read-only/no-network runtime also executes the shipped
  file-only secure reader as `1001:1001` without emitting the evidence marker.
  Final result: `1 passed in 38.77s`; its container, probe directory, and
  test-only image tag were removed.
- Provider post-commit RED was `1 failed in 1.40s` because the disposable scope
  could not inject a failure at the commit boundary. The exact raw OpenRouter
  test state is now captured immediately after `flush` and before `commit`, and
  the failure callback is the first statement after that commit. The resulting
  regression proves the new global provider is removed and its Task 8
  SearchSettings/LLM rows are cleaned.
- Pre-existing-provider cleanup-boundary RED was `1 failed in 1.02s`. Its
  initial mutation, flush, raw ciphertext/field snapshot, commit, and yield now
  all live inside one exception-safe context. Dedicated failures immediately
  after commit and while the raw test snapshot is being obtained both restore
  or remove safely. Exact raw ciphertext and every provider field are restored
  while the existing PostgreSQL session advisory lock remains held for the
  disposable scope. A separate OpenAI row is unchanged by forced OpenRouter
  post-commit cleanup.

### Round-4 verification

- Clean pre-change focused baseline: `84 passed in 11.64s`.
- Final focused readiness, Supervisor/Compose wiring, and executable preflight:
  `91 passed in 13.44s`.
- Regulatory unit/runtime superset: `922 passed in 20.08s`.
- Fresh disposable PostgreSQL 15.2 migration from empty to head
  `c8f1a6d4e2b7` succeeded. Final real PostgreSQL repository plus
  PostgreSQL/Elasticsearch 9.4.2 pipeline repeat: `36 passed in 44.22s`.
- The independent post-test query returned `0|0|0|0|0|0` for Task 8
  SearchSettings, LLM providers, file records, UserFiles, durable jobs, and
  OpenRouter/OpenAI providers. The Task 8 Elasticsearch index listing was
  empty.
- Ruff, Ruff formatting, and `ty` passed on all touched Python. `bash -n`
  passed. Host `shellcheck` was unavailable; the repository shellcheck hook
  passed. The touched-code pre-commit run passed every applicable hook,
  including lazy imports, `ty`, Ruff, Ruff formatting, shellcheck, large-file,
  and ripsecrets.

### Live readiness recheck

`getent hosts psql.dev.singlewindow.io` returned exit `2` again.
`onyx-api_server-1` was `running unhealthy`; nginx ports `7000` and `7001`
both returned HTTP `200`. The corrected local read-only readiness command with
both canonical evidence paths returned exit `1`, `NOT_READY`: migration and
Admin snapshot failed with redacted `OperationalError`, Supervisor inspection
failed, host cgroup evidence was unavailable, and all snapshot-dependent checks
were blocked. No attestation was fabricated, no live canary was started, and no
existing production document, index, database/provider record, GCS object, or
Vertex job was mutated.

---

## Review fix round 5/5

Both remaining Important findings were implemented without a production write,
deployment, push, or merge.

### RED/GREEN evidence

- Nonblocking secure-open RED was `2 failed, 2 passed, 28 deselected in
  2.41s`: readiness did not fail closed when `O_NONBLOCK` was unavailable, and
  opening a writerless FIFO blocked beyond the deterministic one-second child
  process deadline. `_read_secure_file` now includes
  `O_RDONLY|O_NONBLOCK|O_NOFOLLOW|O_CLOEXEC` in its one `os.open`, immediately
  `fstat`s and rejects non-regular files, reads at most the bound plus one byte,
  and closes on every path. Focused GREEN was `4 passed, 28 deselected in
  1.07s`.
- Snapshot-helper RED was `3 failed, 68 deselected in 1.55s`: the stdlib-only
  helper and snapshot-only runtime validation mode did not exist. Timeout RED
  was separately `2 failed, 37 deselected in 0.50s`: preflight issued neither
  a hard timeout nor deterministic named-container cleanup. The helper now
  opens each original source exactly once with the same nonblocking,
  no-follow, close-on-exec contract; validates regular-file type, exact
  `1001:1001` ownership, exact `0600`/`0400` modes, and bounded size from that
  descriptor; reads only those bounded bytes; and writes two fixed-name,
  exclusive mode-`0600` snapshots in a unique mode-`0700` directory with file
  and directory `fsync`. It does not enumerate the source directory or print
  paths or evidence. Focused helper/runtime GREEN was `3 passed, 68 deselected
  in 1.13s`; timeout/delegation GREEN was `2 passed, 37 deselected in 0.51s`.
- Production-lite preflight bind-mounts only those trusted snapshots into a
  named `--rm`, no-network, read-only-root, capability-dropped runtime
  container. The invocation has a foreground 30-second timeout with a
  five-second kill grace; the EXIT/HUP/INT/TERM cleanup trap uses a separately
  bounded forced container removal and deletes only the two fixed snapshots
  and their private directory. The runtime image reopens both snapshots with
  the canonical secure reader and independently revalidates the exact archived
  evidence digest binding before returning without database or network access.
- The real runtime-lite regression maps the invoking test identity to numeric
  `1001:1001` in an isolated user namespace. Exact regular source files pass
  descriptor snapshotting and runtime digest validation; a final-component
  host attestation symlink fails before any readiness container run; unrelated
  sibling data is never read; all snapshots, containers, probe directories,
  and test image tags are removed. Focused result was `2 passed, 1 deselected
  in 8.37s`; the complete privileged runtime module, including canonical
  Supervisor `1001:1001`/`0770` allow/deny proof, passed `3 passed in 46.31s`.

### Round-5 verification

- Final focused readiness, Supervisor/Compose wiring, and executable preflight:
  `97 passed in 11.25s`.
- Regulatory/indexing unit and runtime superset, broader than round 4:
  `975 passed, 6 skipped in 26.88s`.
- Ruff and Ruff formatting passed on every touched Python file. `ty` passed.
  `bash -n` passed. Host `shellcheck` was unavailable; the repository
  shellcheck pre-commit hook passed. The complete touched-code pre-commit run
  passed every applicable hook, including lazy imports, `ty`, Ruff, Ruff
  formatting, shellcheck, large-file, and ripsecrets.
- No durable pipeline or repository code changed in this breaker round, so the
  disposable PostgreSQL 15.2/Elasticsearch 9.4.2 pipeline suite was not
  repeated. The immediately preceding unchanged-code evidence remains
  `36 passed in 44.22s`, with the independent zero-residue database and index
  checks recorded above.

### Environment and live-readiness limitation

This host's snap-packaged Docker 29.6.1 daemon rejects every initial executable,
including `/bin/true`, when Docker applies
`--security-opt no-new-privileges`, with `operation not permitted`. The
canonical preflight retains that fail-closed security option. The external
regression probes this daemon behavior and, only for the test runtime on an
affected host, uses `setpriv --no-new-privs`; it separately proves
`NoNewPrivs: 1` before exercising
the same runtime validation. Docker-visible snapshots also use a unique
repository-local test directory because this snap daemon cannot bind the
caller's `/tmp`; production keeps the ordinary private `TMPDIR` contract.

The live gate otherwise remains unchanged: `getent hosts
psql.dev.singlewindow.io` returned exit `2`; `onyx-api_server-1` was running but
unhealthy; nginx ports `7000` and `7001` returned HTTP `200`. No attestation was
fabricated, no live canary was started, and no production document, index,
database/provider record, GCS object, or Vertex job was mutated.

---

## Task 8: breaker adjudication

Both load-bearing findings were implemented without a production write,
deployment, push, or merge.

### RED/GREEN evidence

- Operator/ownership RED was `5 failed, 36 deselected in 0.78s`: the standalone
  preflight accepted a non-root operator, the helper accepted caller-selected
  owner IDs and emitted the mapped caller identity, and the validation
  container inherited that identity instead of the canonical app identity.
  GREEN was `5 passed, 36 deselected in 0.58s`; the complete preflight/deploy
  unit file then passed `41 passed in 6.43s` before cleanup cases were added.
- The canonical preflight now fails before Docker unless `EUID=0`. Its
  stdlib-only helper also requires effective UID/GID `0:0`, opens and validates the
  original sources against exact `1001:1001` plus `0600`/`0400`, creates each
  snapshot exclusively in the root-owned mode-`0700` directory, then performs
  descriptor-owned `fchown(1001, 1001)`, `fchmod(0600)`, `fsync`, and final
  descriptor metadata/size verification. It emits neither evidence nor an
  operator-derived identity. The runtime validator is always launched with
  `--user 1001:1001`. The deploy wrapper grants `sudo` only to this bounded
  preflight subprocess; rollout commands continue under the operator identity.
  The runbook's exact preflight commands use `sudo --` and explicitly forbid
  `sudo -E` or environment/secret printing.
- The bounded-sudo integration exposed one topology guard that otherwise would
  have been stripped by ordinary `sudo`: RED was `1 failed, 44 deselected in
  0.56s` because an operator `COMPOSE_PROFILES` value reached deployment after
  the root preflight no longer saw it. GREEN was `1 passed, 44 deselected in
  0.05s`; the unprivileged wrapper now rejects this non-secret control before
  sudo, while the root preflight still rejects it in its own environment and
  in the selected `.env`.
- Exact-cleanup RED was `5 failed, 39 deselected in 1.19s`: preflight still used
  a guessed name, had no private cidfile or ownership label, swallowed cleanup
  ownership/removal errors, and did not verify exact-ID absence. GREEN was
  `5 passed, 39 deselected in 1.29s`; the final complete unit file is
  `45 passed in 9.85s`.
- Preflight now creates a cryptographically random 64-hex ownership token and
  passes it as the `io.regulatory.readiness-preflight-owner` label. Docker alone
  writes the cidfile inside the private snapshot directory. Cleanup accepts
  only a 64-hex ID read from that private file, queries that exact full ID,
  verifies the exact token before `docker rm -f`, and then proves the same ID is
  absent. There is no `--name` and no name-based deletion. A label mismatch
  leaves the unrelated container untouched and reports failure; a forced
  removal failure changes an otherwise successful preflight to nonzero with a
  controlled diagnostic. When validation already failed, its original message
  remains visible alongside any cleanup failure. Normal cleanup removes the
  cidfile, two snapshots, and private directory.
- The real-timeout regression initially failed `1 failed, 3 deselected in
  44.57s` because its injection did not yet keep the validation container
  alive. GREEN was `1 passed, 3 deselected in 75.87s`: the isolated runtime
  process ignores `TERM`, forcing the five-second kill grace and the cidfile
  cleanup path; the exact ID appears in the removal command and a subsequent
  Docker inspect proves it absent.

### Privileged/runtime proof

The disposable operator test derives a test-only image from the exact
`runtime-lite` target, runs the canonical preflight as root with the local
Docker socket, and uses original host-visible files whose real numeric metadata
is `1001:1001`. A distinct UID/GID `2002:2002` direct invocation first fails
before Docker. The root invocation succeeds, and the actual inner Docker run
uses `--user 1001:1001`; a final-component host symlink fails before any inner
run. The same module retains the Supervisor root startup and canonical socket
`1001:1001`/`0770` app-user allow/unrelated-user deny proof. With the forced
timeout cleanup case, the complete module passed `4 passed in 92.62s`. Every
test-only container, probe directory, and image tag was removed.

### Verification and live limit

- Focused readiness, executable preflight/deploy, and Supervisor/Compose
  wiring: final repeat `103 passed in 14.70s`.
- Broad regulatory, user-file, Vertex, and runtime unit superset:
  `871 passed in 23.62s`.
- Ruff, Ruff formatting, `ty`, and `bash -n` passed. Host `shellcheck` remains
  unavailable; the repository pre-commit shellcheck hook passed. The complete
  touched-file pre-commit run passed all applicable hooks, including lazy
  imports, `ty`, Ruff, Ruff formatting, shellcheck, large-file, and ripsecrets.
- No durable pipeline/repository code changed in this adjudication, so the
  immediately preceding unchanged PostgreSQL 15.2/Elasticsearch 9.4.2 evidence
  remains `36 passed in 44.22s` with zero provider, pipeline, and index residue.

The live read-only check remains blocked: private PostgreSQL DNS returned exit
`2`, `onyx-api_server-1` was up but unhealthy, and nginx ports `7000` and `7001`
returned HTTP `200`. No trusted live attestation was fabricated, no root
preflight was run against unapproved evidence, and no live canary or production
mutation was attempted.

---

## Task 8: post-adjudication deployment-target binding

The three reviewed trust-boundary findings were reproduced and fixed without a
production write, deployment, push, merge, or security-contract downgrade.

### RED/GREEN and security contract

- The focused RED selector was `10 failed, 45 deselected in 0.84s`. It proved
  that Docker/Compose/image ambient overrides reached different sides of the
  sudo boundary, sudo was interactive-capable, and a container created before
  Docker wrote the private cidfile was not recovered. The same selector passed
  GREEN as `10 passed, 45 deselected in 1.10s`; the completed deploy/preflight
  unit module passed `56 passed in 10.14s` before the final ambiguity and
  runbook assertions were added.
- The deploy wrapper now rejects ambient `DOCKER_*`, `COMPOSE_*`, application,
  edge, and infrastructure image interpolation overrides before sudo or
  Docker. It resolves `PATH`, `HOME`, `$HOME/.docker`, and the fixed local
  `unix:///var/run/docker.sock` target once, then prefixes the root preflight
  and every Compose/direct-Docker rollout operation with the same explicit
  `env -i` allowlist. Compose interpolation therefore comes from the selected
  mode-`0600` environment file, not the operator shell. The regression's fake
  sudo performs a realistic environment reset; every observed preflight and
  rollout Docker invocation had the same fixed host, no context, no ambient
  image override, and no unrelated marker.
- Non-root handoff now uses `sudo -n` and emits a controlled non-secret refusal
  when exact noninteractive authorization is unavailable. The runbook's exact
  commands go through `regulatory-prod-lite-deploy.sh preflight`, require an
  exact-argv `NOPASSWD` rule only for the bounded preflight handoff, and forbid
  generic `env`, Docker, shell, or whole-deployer privilege. The compatibility
  alias also routes through that canonical wrapper.
- Cleanup no longer depends on Docker successfully writing the cidfile. On a
  missing, non-regular, or invalid private cidfile it queries the exact
  cryptographically random ownership label, accepts only zero or one full
  64-hex container ID, independently inspects the exact label value, removes
  only that full ID, and proves no container with the token remains. A cidfile
  and label disagreement, multi-ID/invalid output, label mismatch, query
  failure, removal failure, or residual container fails closed without
  removing an unrelated container or hiding the primary validation failure.

### Real Docker, broad, and static verification

- The new real-Docker pre-cid regression passed `1 passed, 4 deselected in
  38.15s`. The fake CLI used the real daemon to create the labeled validation
  container and deliberately omitted the cidfile. Cleanup recovered and
  removed the exact owned ID, an exact-token daemon query returned zero
  residue, and a deliberately pre-existing container with the same label key
  but a different token remained inspectable until fixture teardown. The full
  privileged runtime module passed `5 passed in 95.62s`, retaining the
  Supervisor numeric `1001:1001`/`0770` allow/deny proof and normal, symlink,
  timeout, and pre-cid validation cases.
- Final focused readiness, executable preflight/deploy, compatibility alias,
  and Supervisor/Compose wiring passed `120 passed in 15.21s`. The broad
  regulatory, user-file, Vertex, and runtime unit superset passed `888 passed
  in 30.71s`.
- Ruff check, Ruff formatting, `ty`, and `bash -n` passed. Host `shellcheck`
  remained unavailable; the repository shellcheck pre-commit hook and the
  complete touched-file pre-commit run passed all applicable hooks.
- Durable pipeline/provider/repository code did not change, so disposable
  PostgreSQL 15.2 and Elasticsearch 9.4.2 were not rerun. The immediately
  preceding unchanged-code evidence remains `36 passed in 44.22s` with exact
  provider restoration and zero provider, pipeline, large-object, or index
  residue.

### Live-readiness limit

The final read-only check remained blocked before any canary: private
`psql.dev.singlewindow.io` DNS returned exit `2`; `onyx-api_server-1` was up but
unhealthy; `onyx-nginx-1` was healthy; ports `7000` and `7001` returned HTTP
`200`. The test ownership label had zero remaining Docker containers. No
trusted attestation was fabricated, no live root preflight was attempted, and
no production document, index, database/provider record, object, or job was
mutated.

---

## Task 8: privileged executable boundary closure

The post-commit Critical finding was valid. The deployment wrapper placed the
operator's `PATH`, `HOME`, and `$HOME/.docker` inside the root handoff and
resolved `sudo`, Docker, and the Compose plugin through caller-controlled
lookup. The internal preflight also used `/usr/bin/env bash`, so its privileged
interpreter and later unqualified commands depended on that environment.

### TDD and corrected contract

- Focused RED was `4 failed, 56 deselected in 0.93s`. A deployment completed
  while hostile Bash, sudo, Docker, Python, timeout/stat, `BASH_ENV`, and a
  `$HOME/.docker/cli-plugins/docker-compose` marker executed; a mode-`0770`
  Docker configuration directory was accepted and Docker was queried; the
  scripts lacked the absolute boundary; and the runbook still selected the
  caller's home configuration.
- The production scripts now start with `/bin/bash -p`, immediately install the
  fixed `/usr/sbin:/usr/bin:/sbin:/bin` path, and clear `BASH_ENV`, `ENV`,
  `PYTHONPATH`, `PYTHONHOME`, `SUDO_ASKPASS`, loader variables, and related
  shell controls. The only privilege transition is absolute `/usr/bin/sudo -n`
  to `/usr/bin/env -i ... /bin/bash -p` with the same fixed environment used by
  every later rollout command.
- Docker is always `/usr/bin/docker`; Compose is invoked directly as
  `/usr/libexec/docker/cli-plugins/docker-compose`, so neither caller `PATH`
  nor home-directory CLI plugins can select an executable. Both phases use the
  fixed local socket and fixed `/etc/onyx/regulatory-docker` configuration;
  ambient Docker/Compose/image variables remain rejected at the wrapper.
- Root preflight fails before Docker unless `/etc/onyx`, the dedicated config
  directory, `config.json`, Docker, and Compose satisfy their fixed trust
  contracts. Directories must be root-owned, non-symlink, canonical, not group
  writable, and inaccessible to world users. `config.json` must be a
  root-owned non-symlink regular file, at most 1 MiB, non-executable, with no
  unsafe group/world access; executables must be root-owned regular files,
  executable, and not group/world writable. Locked parents make the subsequent
  use non-operator-rebindable.
- Unit tests use a copied test-only Compose bundle whose compile-time constants
  point to fixture-owned tools. This preserves fixed production constants and
  adds no environment-controlled runtime escape hatch. GREEN for the new
  boundary cases was `3 passed, 57 deselected in 0.55s`; after compatibility
  assertions the complete deploy/preflight unit module passed `60 passed in
  12.53s`; the final post-hook repeat was `60 passed in 12.65s`. The
  hostile-path regression also proves the root argv contains the trusted path,
  `/var/empty` home, fixed config, and none of the poison variables.

### Runtime, broad, and static evidence

- The isolated real-Docker normal case passed `1 passed, 4 deselected in
  45.63s`. The final complete runtime module passed `5 passed in 93.56s`, retaining
  Supervisor's root-started numeric `1001:1001`/`0770` allow/deny proof plus
  normal, host-symlink, forced-timeout, and pre-cid cleanup scenarios. Its
  outer image contained the canonical root-owned Docker config; root-owned
  adapters were mounted at the exact Docker/Compose paths while the real daemon
  handled the inner runtime container. The dedicated outer temp mount avoided
  exposing sibling host `/tmp` data.
- Final readiness, deployment, runtime-dependency, environment-inventory,
  Supervisor, and Celery wiring focus passed `197 passed in 23.13s`. The broad
  55-file regulatory, user-file, Vertex, Supervisor, and runtime unit superset
  passed `929 passed in 31.30s`.
- Ruff check, Ruff formatting, `ty`, `bash -n`, and `git diff --check` passed.
  Host `shellcheck` is unavailable; the repository shellcheck pre-commit hook
  passed. Absolute Compose `2.40.3+ds1-0ubuntu1` successfully rendered the
  production-lite overlay directly, without Docker plugin discovery.
- Real-Docker teardown left zero readiness-label containers, test images, or
  probe directories. Durable pipeline/provider/repository code did not change,
  so PostgreSQL/Elasticsearch was not rerun; the immediately preceding
  unchanged-code evidence remains `36 passed in 44.22s` with exact provider
  restoration and zero provider, pipeline, large-object, or index residue.

### Operator and live-readiness limits

The deployment host must provision root-owned
`/etc/onyx/regulatory-docker/config.json` and update the exact `NOPASSWD` argv
for the fixed path, config, and `/bin/bash -p` boundary before a change window;
the current local host does not have `/etc/onyx`, so canonical preflight fails
closed rather than falling back to an operator config. The runbook now gives
mode-`0750`/`0640` provisioning and explicitly requires removal of any older
caller-derived sudoers route.

The read-only live check remains blocked: private PostgreSQL DNS returned exit
`2`; `onyx-api_server-1` was up but unhealthy; `onyx-nginx-1` was healthy; and
ports `7000` and `7001` returned HTTP `200`. No trusted config or attestation
was fabricated, no live root preflight or rollout was attempted, and no
production document, index, database/provider record, object, or job was
mutated.

---

## Task 8: final privileged-bundle adjudication

The remaining Critical finding was valid: despite the fixed executables and
sanitized environment, the bounded sudo handoff still executed
`regulatory-prod-lite-preflight.sh` from the operator-owned release directory,
and that script selected its Python helper and trusted Compose overlays from
the same mutable directory.

### TDD and installed trust boundary

- The focused RED was `2 failed in 0.23s`: the deploy wrapper had no fixed
  privileged-bundle root and the required installed entrypoint, installer, and
  manifest did not exist. The first contract GREEN was `2 passed in 0.04s`.
- The only sudo target is now the fixed
  `/usr/local/libexec/onyx/regulatory-prod-lite/regulatory-prod-lite-preflight`.
  The wrapper invokes it directly through absolute `/usr/bin/sudo -n`, without
  an intervening shell or `env`, and passes the exact active manifest digest.
  The same digest-addressed installed release supplies every trusted overlay
  to the later non-root Compose rollout; checkout preflight/helper/overlay
  copies are never selected by root.
- Before dispatch, the installed entrypoint validates `/`, both fixed PATH
  directories, every fixed bundle ancestor, `current`, the selected release,
  the exact member set, and every required file as exact `root:root`,
  non-symlink, and non-group/world-writable. It binds the release-directory
  name to the SHA-256 of `REGULATORY_PRIVILEGED_MANIFEST.sha256`, verifies each
  listed digest, and proves the stable entrypoint equals the selected release
  copy. Root then runs only the validated preflight through `/bin/bash -p`.
  The preflight invokes the descriptor-owned snapshot helper and ownership
  token generator through fixed `/usr/bin/python3 -I -S`, excluding cwd,
  script-directory, `PYTHONPATH`, site, and startup-hook import paths.
- The root-only installer is itself accepted only at
  `/usr/local/sbin/install-regulatory-prod-lite-privileged-bundle`. It has no
  sudo call or caller-selected destination. It accepts only a digest-named,
  root-owned private source below
  `/var/lib/onyx/regulatory-prod-lite-staging`, verifies the exact manifest and
  file set, installs a versioned release through a private temporary directory,
  fsyncs files/directories, validates the installed copy, then atomically
  updates the fixed entrypoint and `current`. Existing corrupt releases,
  symlinks, unsafe ownership/modes, extra members, and digest mismatches fail
  before activation. The runbook requires organization signature/archive
  verification, inert allowlisted staging, a verified fixed-path installer,
  a literal no-wildcard exact-argv sudoers rule, `visudo -c`, and removal of all
  older checkout/shell/env/wildcard grants.
- Behavior regressions prove a mode-`0777` checkout preflight, Python helper,
  and overlay cannot run or affect deployment; a hostile cwd `secrets.py`
  marker is not imported; installed file/directory writable modes, final
  symlinks, and a non-root-owned helper fail before Docker; the installer
  atomically installs an exact reviewed bundle while rejecting symlinked or
  writable staging and direct checkout execution. The final focused readiness,
  deploy, runtime dependency, environment, Celery wiring/task, and Supervisor
  unit set passed `176 passed in 27.44s`.

### Privileged runtime, broad, and static evidence

- A real setuid-sudo/NOPASSWD test in a disposable runtime-derived image uses
  an exact literal rule for the fixed entrypoint and digest. UID/GID
  `2002:2002` can run that one noninteractive `--help` handoff; sudo rejects a
  hostile checkout shell path and rejects extra/non-exact preflight arguments,
  and no marker executes. The complete privileged module passed `6 passed in
  128.93s`, retaining Supervisor numeric `1001:1001`/`0770` allow/deny plus
  normal readiness, final-component symlink, forced-timeout cleanup, and
  pre-cidfile exact-label cleanup proofs.
- The exact 55-file regulatory, user-file, Vertex, Supervisor, and runtime unit
  superset passed `940 passed in 41.12s`. Durable pipeline/provider/repository
  code did not change, so the immediately preceding unchanged PostgreSQL
  15.2/Elasticsearch 9.4.2 proof remains `36 passed in 44.22s` with exact
  provider restoration and zero pipeline/index residue.
- The privileged manifest's eight SHA-256 entries passed `sha256sum --check
  --strict`. Ruff check, Ruff formatting, `ty`, `bash -n`, and `git diff
  --check` passed. Host shellcheck is unavailable, while the repository
  touched-file pre-commit run passed all applicable hooks, including lazy
  imports, `ty`, Ruff, Ruff format, shellcheck, large-file, and ripsecrets.
  Disposable containers, ownership-label containers, test images, and probe
  directories had zero residue.

### Live-readiness limit

The final read-only check remains blocked before any canary: private
`psql.dev.singlewindow.io` DNS returned exit `2`; `onyx-api_server-1` was up
but unhealthy; `onyx-nginx-1` was healthy; ports `7000` and `7001` returned
HTTP `200`; and the readiness ownership label matched zero containers. The
local host also intentionally has no approved installed privileged bundle or
trusted capability attestation. No installer, sudoers rule, root preflight,
rollout, live canary, production document, index, database/provider record,
object, or job was created or mutated.

---

## Task 8: breaker adjudication — ancestor-mode test evidence correction

The remaining Important finding was valid and test-only. The preceding report
overstated two ancestor-mode regressions: the installer's
`directory-writable` parameter changed the staged helper file instead of the
staging-root ancestor, while the installed-release-ancestor mutation branch
existed but its parameter was absent and therefore could not run.

- TDD RED was the corrected staging-root assertion failing against mode `0755`
  (`1 failed, 70 deselected in 0.21s`), proving the old parameter did not apply
  the claimed directory mutation.
- The installer regression now changes the actual staging-root ancestor to
  mode `0777`, observes that mode before invocation, requires failure before
  the install root/current activation exists, and restores the original mode
  in `finally`. The dispatcher regression now includes the formerly
  unreachable installed `releases`-ancestor case, observes mode `0777`,
  requires failure before any Docker log/call exists, and likewise restores
  the original mode in `finally`. The two corrected cases passed (`2 passed,
  70 deselected in 0.23s`).
- A scan of every `mutation` conditional in this module found no other
  parameter/branch mismatch. The complete deploy/preflight test module passed
  `72 passed in 20.62s`; the relevant deploy, secure-readiness, and runtime
  dependency boundary set passed `110 passed in 25.26s`.
- Ruff check, Ruff formatting, `ty`, `git diff --check`, and the touched-file
  pre-commit suite passed. No production source or runtime artifact changed, so
  the prior privileged runtime, broad unit, and PostgreSQL/Elasticsearch
  evidence remains applicable without rerunning external infrastructure. No
  production operation occurred.
