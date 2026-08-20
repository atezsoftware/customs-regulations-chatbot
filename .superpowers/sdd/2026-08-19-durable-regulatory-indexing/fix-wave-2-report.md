# Durable regulatory indexing: fix wave 2

Date: 2026-08-20

Scope: follow-up for the three Important findings from the review of fix wave 1.
No production mutation, deployment, push, merge, or external provider submission was
performed.

## Outcomes

### Versioned input identity and legacy PREPARING recovery

- `RegulatoryIndexingConfigSnapshot` now requires `input_hash_version`.
- `legacy-v1` exactly reproduces the original semantic-identifier/title/chunker-text
  digest. `canonical-v2` hashes the stable JSON representation of the whole `Document`
  and excludes only `doc_updated_at`, the loader clock.
- New jobs always snapshot `canonical-v2`; d2 backfill labels pre-d2 rows
  `legacy-v1`; the e3 compatibility backfill labels databases that had already run an
  earlier d2 draft.
- Claimed PREPARING recovery selects the hash algorithm from the immutable snapshot.
  Metadata remains identity-sensitive for v2.
- A duplicate delivery that resolves a newer chunk-generation identity reuses, but
  does not claim or prepare, the older active job. The durable orchestrator owns
  validating and terminally fencing that older generation.

The real recovery test creates a fresh PostgreSQL database from `template0`, migrates
it only to c8, stores Markdown containing embedded `ONYX_METADATA` title and a custom
regulation tag, inserts an actual QUEUED/PREPARING c8 job, upgrades to e3, and then
recovers it through the canonical loader. The job reaches VERIFY, publishes visible
chunks to disposable Elasticsearch 9.4.2, and atomically completes the UserFile.

### One active generation per UserFile

- Creation locks UserFile first and preserves lifecycle lock order before looking at
  jobs and SearchSettings.
- Creation returns the existing QUEUED/RUNNING/RETRY_WAIT/CANCELLING job regardless of
  a caller's newer content or chunk-generation identity.
- A different immutable generation can create a reindex only after every previous job
  is terminal. An identical terminal identity remains idempotent and is returned.
- The new e3 head installs a PostgreSQL partial unique index on active states as a
  database-level defense. Development rows that predate the unshipped invariant are
  reconciled deterministically before the index is created.
- Real concurrent PostgreSQL sessions requesting different generation hashes return
  one job ID. A direct second active insert fails on the partial unique index. A stale
  failure delivery for an older terminal generation cannot change the new RUNNING job
  or its INDEXING UserFile.

### Safe d2 downgrade

- Before restoring the legacy four-column idempotency constraint, d2 now ranks rows in
  each legacy-key group by live/current status and then update/create/generation/ID
  recency. It keeps one deterministic best/current row and deletes the rest.
- `regulatory_indexing_item` dependents of discarded jobs are removed by the existing
  `ON DELETE CASCADE`; the retained job's items survive.
- Downgrade deliberately removes input/chunk-generation identity because c8 has no
  representation for it. A later upgrade therefore labels the retained opaque c8 hash
  as `legacy-v1` and assigns the then-current chunk-generation identity. This is the
  bounded data-loss semantic for an unshipped feature; only one retained job history
  per legacy idempotency key survives the downgrade.

## RED evidence

1. `uv run pytest -q backend/tests/unit/onyx/regulatory/indexing_jobs/test_preparation.py -x`
   failed during collection because `RegulatoryInputHashVersion` did not exist.
2. The new real-PostgreSQL active-generation test failed because two calls returned
   distinct IDs (`a357f373-... != c17d5ac7-...`).
3. The repository boundary test for an unknown input-hash version failed with
   `DID NOT RAISE ValueError`.
4. Pre-fix d2 inspection showed that downgrade dropped the five-column unique and
   immediately recreated the four-column unique without reconciling rows. Two rows
   differing only by generation therefore collide. The data-bearing regression now
   exercises that exact downgrade shape.

## GREEN evidence

- Focused unit lifecycle/configuration/task suite:
  `201 passed in 14.59s`.
- Broad regulatory/unit lifecycle suite: `520 passed in 7.34s` on the final
  implementation.
- Versioned preparation tests: `10 passed in 2.91s`.
- Repository boundary tests: `6 passed in 0.33s`.
- Real PostgreSQL repository, migration, port, and deletion-queue tests:
  `60 passed in 11.72s`.
- Data-bearing c8/head/two-generation/c8/head migration cycle:
  `1 passed in 0.64s`.
- Actual pre-migration ONYX_METADATA recovery and publish:
  `1 passed, 8 deselected in 23.41s`.
- Full disposable PostgreSQL + HTTP embedding + Elasticsearch 9.4.2 pipeline file:
  `9 passed in 48.48s`.
- Focused Ruff: passed.
- Focused `ty check`: passed.
- File-scoped pre-commit (including `ty`, Ruff, Ruff format, secret scan, lazy
  imports, and environment drift): passed.
- Final post-format real PostgreSQL repository and migration rerun:
  `42 passed in 7.08s`.

## Deferred and residual risk

- The migration recovery test requires a PostgreSQL role allowed to create and drop a
  disposable database. The repository's external-dependency PostgreSQL test contract
  currently uses such an administrative test role.
- Downgrading across this unshipped feature intentionally compacts duplicate
  generation history; it is not a lossless archival downgrade. The deterministic
  survivor and cascade behavior are tested and documented above.
