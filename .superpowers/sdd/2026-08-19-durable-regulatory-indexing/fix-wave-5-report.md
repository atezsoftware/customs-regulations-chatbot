# Final provider/publication hardening wave

Date: 2026-08-20

This bounded wave resolves final-review findings 2, 3, 6, 7, 8, and 9 without
changing canonical Markdown chunking, feature flags, tenant fencing, dynamic
SearchSettings dimensions, deletion/supersession behavior, or the lite runtime
boundary.

## Implemented contracts

- Vertex create identity now includes tenant, durable job, GCS object prefix,
  request set, and persisted submission attempt. Create intent is committed
  before the provider call. Ambiguous and unknown create outcomes are charged
  once and reconciled under a persisted visibility horizon; a list miss never
  authorizes another create. Definite rejections reset intent and use the normal
  retry budget. Late-visible terminal jobs are durably reconciled, named, deleted,
  and then have their GCS prefix swept.
- Each submitted contextual item has a persisted attempt count. Bounded shards
  select only pending items under both request-count and exact UTF-8 JSONL byte
  caps; omitted, never-submitted items are not charged. Partial/remote-error
  subsets retain their charge and eventually fail closed at the configured
  budget. JSONL upload, result reads, and GCS deletion are streamed/batched.
- Context preparation reuses sorted rows and tokenized legal snapshots, with a
  bounded LRU across distinct validity dates. Existing Markdown chunks remain
  canonical. Gemini receives each chunk's contextual prompt, and OpenRouter
  embeds only generated context plus that original chunk at the persisted
  SearchSettings dimension; no whole-document embedding was added.
- Publication now converges hidden, visible, and process-death mixed generations
  through hide-all/verify then show-all/verify. Terminal publish and snapshot
  drift obey classification/budgets and enter durable cancellation/cleanup;
  explicitly indeterminate visibility recovery remains idempotent.
- Maintained OpenRouter HTTP failures and concrete `elastic_transport`/
  Elasticsearch failures map to secret-safe timeout, network, retryable HTTP, or
  terminal HTTP decisions. Legal text, prompts, vectors, provider response bodies,
  keys, and exception details are absent from added diagnostics.
- Success, failure, cancellation, and settings-drift termination schedule durable
  Vertex cancel/reconcile/delete and GCS cleanup. One-use generation-fenced
  cleanup deliveries survive restarts; exhausted cleanup remains sweepable.
- Added and documented `REGULATORY_INDEXING_CONTEXT_REQUEST_SIZE`,
  `REGULATORY_INDEXING_CONTEXT_JSONL_MAX_BYTES`, and
  `REGULATORY_INDEXING_SUBMISSION_RECONCILE_SECONDS` in the lite environment and
  operator inventory.

## Search compatibility

The repository client, Compose services, and ECK development manifest pin
Elasticsearch 8.6.2. The durable production-path pipeline passed against both
Elasticsearch 8.6.2 and 9.4.2. The runbook now fails rollout closed unless the
authenticated live root response identifies one of those exact tested versions;
other distributions/versions require the same bulk visibility, refresh/readback,
vector `_source`, mapping, and transport-error suite first. `kubectl` was not
installed in this environment, so no live cluster version was claimed.

## Verification

- Focused changed unit/runtime suite: `282 passed in 9.93s` before the final
  late-visible cleanup regression; the final orchestrator repeat passed
  `35 passed in 1.87s`.
- Broad regulatory unit suite: `540 passed in 8.43s` before the final
  late-visible cleanup regression; repeated in the final verification below.
- Real PostgreSQL repository suite: `56 passed in 9.45s` before the additional
  late-visible cleanup persistence test; that new test passed independently in
  `0.93s` and the full suite was repeated in the final verification below.
- Full PostgreSQL + local HTTP embedder + Elasticsearch 8.6.2 pipeline:
  `11 passed in 81.16s`.
- The production-path PostgreSQL + local HTTP embedder pipeline against
  Elasticsearch 9.4.2: `1 passed in 19.58s`.
- Fresh PostgreSQL 15.2 upgraded from empty to `a6d4c8e2f1b7`; the final
  `a6d4c8e2f1b7 -> f4e8a2c6d1b3 -> a6d4c8e2f1b7` roundtrip succeeded and
  Alembic reports the sole head `a6d4c8e2f1b7`.
- Focused Ruff and `ty` passed throughout. Final touched-file pre-commit and
  final suite repeats are recorded below.

No live Vertex AI, GCS, OpenRouter, production database/index, deployment, push,
or production mutation was performed. Provider tests used deterministic fakes and
a local OpenRouter-compatible HTTP server; PostgreSQL, Redis, and Elasticsearch
resources were disposable local containers.

## Final verification repeat

- Broad regulatory unit suite after the late-visible cleanup phase:
  `543 passed in 7.07s`.
- Full real PostgreSQL repository suite after the migration roundtrip:
  `57 passed in 8.54s`.
- Final touched-file pre-commit passed every applicable hook, including lazy
  imports, Compose-template synchronization, `ty`, Ruff, Ruff formatting, YAML,
  secret scanning, large-file, and environment-drift checks.
- `git diff --check` and the final single-head/current checks passed before
  commit.
