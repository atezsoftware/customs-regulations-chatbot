# SDD ledger — plan: docs/superpowers/plans/2026-08-19-durable-regulatory-indexing.md

## Pre-flight interface scan

| Producer task | Consumer task | Shared file or interface | Finding |
| --- | --- | --- | --- |
| 1 | 2 | Job enums/models and immutable snapshot JSON | Compatible: Task 1 stores a non-secret JSON snapshot; Task 2 defines and validates its typed shape. |
| 1 | 4 | Job/item repository and canonical `regulatory_chunk` references | Compatible: preparation creates idempotent jobs/items through the Task 1 repository. |
| 1 | 5 | Item context/vector persistence and lease fencing | Compatible: embedding and publication consume persisted items and must use the claimed generation. |
| 1 | 6 | Claim, transition, retry, and stale-recovery repository APIs | Compatible: Task 6 orchestrates exclusively through Task 1 atomic operations. |
| 1 | 8 | Alembic migration and real PostgreSQL behavior | Compatible: Task 8 validates migration and claim behavior without changing the schema contract. |
| 2 | 3 | Vertex batch configuration snapshot | Compatible: Task 3 resolves credentials at execution time and does not persist secrets. |
| 2 | 5 | Embedding model/provider/effective-dimension snapshot | Compatible: Task 5 uses `final_embedding_dim`, with no hard-coded production dimension. |
| 2 | 6 | Retry decisions, delays, feature flag, lease and poll values | Compatible: Task 6 consumes typed retry policy and persists scheduling state. |
| 2 | 7 | Environment configuration contract | Compatible: deployment wiring exposes the Task 2 settings while defaulting the feature off. |
| 2 | 8 | Readiness validation | Compatible: readiness resolves and reports the same non-secret snapshot invariants. |
| 3 | 4 | `VertexBatchRequest` request-hash contract | Compatible: Task 4 builds requests; Task 3 serializes/correlates them independent of output order. |
| 3 | 6 | Submit/get/read/cancel/cleanup gateway operations | Compatible: one orchestration step performs at most one bounded gateway operation. |
| 3 | 7 | Google SDK runtime dependencies | Compatible: Task 7 must prove existing runtime-lite dependency coverage without adding heavy parser/model dependencies. |
| 3 | 8 | Fake gateway and real readiness checks | Compatible: integration uses a deterministic fake; live readiness checks real access separately. |
| 4 | 5 | Canonical chunks plus persisted contextual text | Compatible: Task 5 embeds contextual text followed by original chunk text and skips only contextual generation when ineligible. |
| 4 | 6 | Preparation and contextual-apply stage boundaries | Compatible: Task 6 advances stages only after Task 4 persistence summaries succeed. |
| 5 | 6 | Embedding and hidden-stage/publish operations | Compatible: Task 6 invokes each as a separate durable stage. |
| 5 | 8 | Hidden publication and vector dimension invariants | Compatible: Task 8 asserts staging, verification, visibility, and dimension end to end. |
| 6 | 7 | Celery app name, queues, task discovery, recovery schedule | Compatible: Task 7 wires the exact Task 6 task names and consumes both required queues. |
| 6 | 8 | Restart, duplicate delivery, partial retry, cancellation | Compatible: Task 8 exercises Task 6 recovery semantics against real infrastructure where available. |
| 7 | 8 | Supervisor/compose/workflow/readiness contract | Compatible: Task 8 reports exact deployment prerequisites and verifies the artifacts agree. |

## Per-task consistency scan

| Task | Internal requirements, tests, and files | Finding |
| --- | --- | --- |
| 1 | Repository tests, enums/models, Alembic migration, atomic claims | Consistent; migration downgrade target is the current clean head. |
| 2 | Frozen snapshot, validation, retry classification/jitter, config defaults | Consistent; secrets remain outside the snapshot. |
| 3 | Pure JSONL contracts and bounded Vertex/GCS gateway methods | Consistent; gateway explicitly contains no polling loop. |
| 4 | Existing regulatory chunker boundary and contextual mapping | Consistent; no legal chunking duplication. |
| 5 | OpenRouter embedding, hidden Elasticsearch staging, verify then publish | Consistent; legacy indexing retains `hidden=False`. |
| 6 | One-step orchestrator, expiring sends, flag-compatible upload entry | Consistent; database remains durable scheduler. |
| 7 | Dedicated light Celery app and production-lite artifacts | Consistent; no model-server or heavyweight parsing dependency. |
| 8 | Baseline repair, external dependency pipeline, readiness, live canary | Consistent; production documents remain untouched and external blockers are reported explicitly. |

Pre-flight result: no contradictory task requirements or review-rubric conflicts found.

Task 1: minor (deferred): bound or validate `error_code` before writing into VARCHAR(128); final review must triage.
Task 1: fix round 1/5 (5 addressed, 0 open; commits e960276..325c20f)
Task 1: complete (commits 4bc0166..325c20f, review clean)
Task 2: fix round 1/5 (3 addressed, 0 open; commits d057b57..69a8829)
Task 2: minor (deferred): environment contract documentation awaits Task 7 wiring.
Task 2: complete (commits 325c20f..69a8829, review clean)
Task 3: minor (deferred): close each google-genai client after bounded submit/get/cancel calls; final review must triage.
Task 3: fix round 1/5 (4 addressed, 0 open; commits 753a403..8862bb2)
Task 3: complete (commits 69a8829..8862bb2, review clean)
Task 4: minor (deferred): apply_contextual_results must validate row.user_file_id equals job.user_file_id; final review must triage.
Task 4: minor (deferred): duplicate preparation tests need real takeover/partial recovery coverage; final review must triage.
Task 4: fix round 1/5 (4 addressed, 1 open — contextual Vertex tokenizer still falls back to default embedding tokenizer; commits 813c7f5..07673f6)
Task 4: Ruling: use a local UTF-8-byte upper-bound contextual tokenizer because the installed stack has no local Gemini tokenizer and preparation may not call a provider — this guarantees the Vertex input window is not exceeded at the cost of conservative context trimming if the bound is loose.
Task 4: minor (deferred): add an explicit callback/validation-failure rollback test for atomic preparation; final review must triage.
Task 4: fix round 2/5 (1 addressed, 0 open; commits 07673f6..5222c72)
Task 4: complete (commits 8862bb2..5222c72, review clean)
Task 5: minor (deferred): add an out-of-order successful OpenRouter response regression test; final review must triage.
Task 5: fix round 1/5 (4 addressed, 2 new open — post-exposure DB failure compensation and VERIFY/PUBLISH snapshot validation; commits 3d8d680..176aba5)
Task 5: minor (deferred): add direct ElasticsearchIndexClient mget parser coverage; final review must triage.
Task 5: fix round 2/5 (2 addressed, 0 open; commits 176aba5..a4e9e31)
Task 5: complete (commits 5222c72..a4e9e31, review clean)
Task 6: review round 1/5 (5 open — publish crash idempotency, partial apply resume, duplicate preclaimed delivery, stranded cancellation, and in-flight submit lease race; commit b63955d)
Task 6: fix round 1/5 implementation complete (5 addressed pending scoped re-review; commits b63955d..ba755de)
Task 6: fix round 1/5 re-review (5 addressed, 0 open; commits b63955d..ba755de)
Task 6: complete (commits a4e9e31..ba755de, review clean)
Task 7: review round 1/5 (2 open — production-lite has no recovery/monitoring scheduler, and canonical runbook contradicts the new worker/queue; commit ad1a45b)
Task 7: minor (deferred): live background-pod memory headroom for the additional worker is not verifiable from repository-owned Helm values; readiness/final report must surface it.
Task 7: fix round 1/5 implementation complete (2 addressed pending scoped re-review; commits ad1a45b..f5d52bd)
Task 7: fix round 1/5 re-review (0 fully addressed, 2 open — replica-safe Beat ownership/dedup and stale probe freshness; commits ad1a45b..f5d52bd)
Task 7: fix round 2/5 implementation complete (2 addressed pending scoped re-review; commits f5d52bd..71dcc20)
Task 7: fix round 2/5 re-review (probe truth addressed; replica tick test, next-slot contract wording, and all-replica CodeBuild verification remain open; commits f5d52bd..71dcc20)
Task 7: fix round 3/5 implementation complete (remaining test/contract/deployment-gate items addressed pending scoped re-review; commits 71dcc20..e3b5919)
Task 7: fix round 3/5 re-review (runtime/deployment/docs addressed; one Important test gap remains — real tick test must cover both entry types; commits 71dcc20..e3b5919)
Task 7: fix round 4/5 implementation complete (entry-level isolation test added pending scoped re-review; commits e3b5919..b5f4ace)
Task 7: fix round 4/5 re-review (entry-level isolation gap closed, 0 open; commits e3b5919..b5f4ace)
Task 7: complete (commits ba755de..b5f4ace, review clean)
Task 8: deferred-minor triage complete (error-code bound, google-genai close, contextual ownership, preparation rollback/recovery, out-of-order embedding, direct ES mget, and external memory-headroom gate all addressed or explicitly bounded)
Task 8: live validation blocked before canary (private PostgreSQL DNS unresolved; local production DB/cache/search stopped; API container unhealthy; read-only readiness exit 1; no production mutation)
Task 8: implementation complete pending controller review (focused unit/runtime 576 passed; real PG+ES 29 passed; migration downgrade/upgrade passed; Ruff/ty/pre-commit passed)
