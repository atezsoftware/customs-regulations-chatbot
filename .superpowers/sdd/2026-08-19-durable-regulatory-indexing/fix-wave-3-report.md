# Durable regulatory indexing: compatibility and supersession fix wave

Date: 2026-08-20

Scope: follow-up for the two Important findings from review of fix wave 2. No
production mutation, provider submission, deployment, push, merge, or live data read
was performed.

## Outcomes

### Migration-safe input hash resolution

- Fresh jobs continue to snapshot metadata-sensitive `canonical-v2` input identity.
- Rows whose migration provenance cannot distinguish the historical
  semantic-identifier/title/text digest from the canonical full-Document digest now
  carry `legacy-or-canonical`; migrations no longer guess `legacy-v1`.
- PREPARING recovery loads the stored Markdown through the canonical production
  loader, computes the exact legacy-v1 and canonical-v2 digests, and proceeds only
  when exactly one matches. No match and the ambiguous case fail closed.
- The resolved discriminator is persisted in the same PostgreSQL transaction as
  canonical chunk/item replacement, the PREPARING stage advance, and
  `UserFile.status=INDEXING`. Callback validation failure rolls back the resolution
  as well as chunk and item writes.
- A new f4 migration also repairs PREPARING rows already labelled by the earlier e3
  draft. Its cancellation-intent backfill distinguishes deletion, explicit user
  cancellation, and a downgraded/re-upgraded supersession from durable UserFile
  state.

The real compatibility pipeline covers three origins with embedded
`ONYX_METADATA`: a fresh c8 legacy row, an already-d2 canonical row without a
discriminator, and a canonical head row taken through head to c8 to head. Each row
starts unresolved after upgrade, resolves to the exact matching algorithm while
leaving PREPARING, and publishes visible chunks successfully.

### Durable chunk-generation supersession

- Job creation still locks UserFile and reuses the one active job. When that job has
  an older chunk-generation hash, it is atomically fenced into `CANCELLING` with
  typed `SUPERSEDE` intent rather than being enqueued for normal validation or
  terminally failing the file.
- The existing durable Vertex, GCS, and Elasticsearch cancellation phases own cleanup.
  Repeated generation-mismatch delivery does not increment the fence again or create
  a second active job. Every active stage and each remote/non-remote cleanup entry
  path is covered.
- Elasticsearch cancellation waits for a refreshed delete boundary. Without that
  boundary, an immediate successor could observe the stale pre-delete search
  snapshot and fail its own delete-by-query with a 409 conflict. The new refresh
  option defaults off for every legacy caller and is enabled only for durable
  cancellation cleanup.
- Supersession finalization marks only the old job `CANCELLED`, leaves user
  cancellation as `CANCELED`, preserves deletion as `DELETING`, and atomically sets
  the superseded file to `PROCESSING`. Deletion monotonically overrides either other
  intent and increments the generation fence exactly once.
- The real processing scanner selects that durable `PROCESSING` row. The ordinary
  production user-file worker then creates and prepares the current generation after
  the partial unique active-job constraint is released.
- A different terminal generation is explicitly reprocessable. An identical current
  successful job remains idempotent, creates no new row, and normalizes a redelivered
  PROCESSING/INDEXING file back to `COMPLETED`. Stale old-generation failures cannot
  overwrite the successor or its UserFile status.

The disposable production-path test completes one generation, invokes the real
user-file worker for a new generation, supersedes that active generation with a third
identity, completes real cancellation cleanup, and invokes the same worker again to
prepare and fully publish the successor. A final identical terminal delivery creates
no additional job.

## RED evidence

- Compatibility resolver tests initially failed because the unresolved enum and
  resolver did not exist. After adding the compatibility state, the ambiguous case
  reported `DID NOT RAISE` and the no-match case followed the unsupported-version
  path until exact dual-hash resolution was implemented.
- Claimed PREPARING recovery initially rejected the unresolved discriminator before
  reaching atomic persistence. The repository atomicity test initially rejected the
  new `resolved_input_hash_version` argument.
- The migration expectation showed missing discriminators being labelled
  `legacy-v1`; the f4 schema test initially found no cancellation-intent column.
- A generation-mismatch repository test observed the old job remain `QUEUED`; the
  finalization test observed the UserFile become `CANCELED` instead of `PROCESSING`.
- Deletion-intent tests observed `NONE` instead of `USER_DELETE`, and deletion did not
  override in-flight supersession until the monotonic override was added.
- Historical f4 backfill first produced `USER_CANCEL` for a `DELETING` row. After
  deletion inference was added, the downgrade/re-upgrade supersession case still
  produced `USER_CANCEL` for `PROCESSING`; it became `SUPERSEDE` only after typed
  lifecycle inference was completed.
- The production-worker regression initially followed the flag-off legacy path
  because its fixture had not enabled durable regulatory indexing. Enabling the test
  flag exercised the intended production branch; no flag-off source behavior changed.
- A repeated real Elasticsearch 9.4.2 run exposed the successor job terminally
  failing at `INDEX_WRITE` with `ConflictError`. The durable cancellation delete had
  completed without refreshing the index, so the immediate successor occasionally
  queried the stale pre-delete snapshot. A focused unit assertion first failed on the
  missing `refresh=True` contract before the synchronous cleanup boundary was added.

## GREEN evidence

- Focused regulatory indexing and user-file unit suite after the final fix:
  `223 passed in 5.08s`.
- Complete document-index unit suite: `120 passed in 1.24s`.
- Broad regulatory, user-file, database, document-index, and server unit suite:
  `704 passed in 11.32s`.
- Real PostgreSQL repository, migration, and actual Redis-backed processing scanner
  after the final fix: `63 passed in 10.66s`.
- All eight active supersession stages plus rollback and stale-failure focus:
  `10 passed in 1.82s` before the complete repository repeat.
- Isolated real PostgreSQL port and swap invariant suite: `100 passed in 7.90s`.
- Three-origin ONYX_METADATA recovery and publication on Elasticsearch 9.4.2:
  `3 passed in 58.28s`.
- Production-worker supersession and terminal-idempotency pipeline on Elasticsearch
  9.4.2: `1 passed in 14.38s`.
- The same real production-worker successor pipeline repeated after the refreshed
  cancellation fix: `10 passed in 115.43s`.
- Final full disposable PostgreSQL, HTTP embedding, deterministic Vertex/GCS gateway,
  and Elasticsearch 9.4.2 pipeline file: `11 passed in 84.33s`.
- Alembic reports the single head `f4e8a2c6d1b3`.
- File-scoped pre-commit, including `ty`, Ruff, secret scanning, and environment
  drift checks: passed.

One deliberately over-combined external-dependency run shared one database across
unrelated startup/user-file and port modules. Thirteen unchanged port assertions saw
the portable scopes left by earlier modules, and one real model-server case could not
connect. The port modules passed `100/100` on a new empty migrated database; the model
server-only case was excluded through its existing `MODEL_SERVER_HOST=disabled` gate.
No product code was changed for those environment/isolation failures.

## Residual risk and boundaries

- Vertex AI and Google Cloud Storage behavior remains covered by the deterministic
  gateway, including cancel and cleanup calls; no live cloud credential or provider
  mutation was used.
- The feature remains disabled by default. The legacy flag-off indexing branch is
  unchanged; one pre-existing unit assertion was updated to account for the already
  introduced durable deletion-fence transaction before the ordinary delete read.
- Downgrading this unshipped feature still deliberately compacts duplicate generation
  history as documented in fix wave 2. Typed intent is reconstructed from durable
  UserFile lifecycle state when f4 is re-applied.
