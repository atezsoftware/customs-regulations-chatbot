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
Task 8: fix round 1/5 implementation complete pending re-review (6 findings addressed; readiness/gateway 49 passed; unit/runtime 668 passed; real PG+ES 29 passed; ES client 70 passed; migration cycle, Ruff, and ty passed)
Task 8: fix round 2/5 implementation complete pending re-review (4 findings addressed; focused readiness/gateway/wiring/deploy 109 passed; unit/runtime superset 777 passed; real PG+ES 30 passed; forced cleanup left zero DB/large-object/index residue; Ruff and ty passed; live readiness remained blocked without production mutation)
Task 8: fix round 3/5 implementation complete pending re-review (5 findings addressed; focused readiness/gateway/wiring/preflight 120 passed; unit/runtime superset 788 passed; real PG+ES 32 passed; forced cleanup preserved the primary error and left zero residue; exact OpenRouter state restored under advisory lock; Ruff/ty/pre-commit passed; live readiness remained blocked without production mutation)
Task 8: fix round 4/5 implementation complete pending re-review (3 findings addressed; descriptor-owned readiness/preflight validation and runtime numeric Supervisor proof passed; focused readiness/wiring/preflight 91 passed; unit/runtime superset 922 passed; real PG+ES 36 passed with zero provider/pipeline/index residue; Ruff/ty/shellcheck/pre-commit passed; live readiness remained blocked without production mutation)
Task 8: fix round 5/5 breaker implementation complete pending re-review (2 findings addressed; O_NONBLOCK/FIFO and descriptor-snapshot trust boundaries passed; focused readiness/wiring/preflight 97 passed; privileged real-runtime module 3 passed; unit/runtime superset 975 passed with 6 skipped; Ruff/ty/bash-n/shellcheck/pre-commit passed; unchanged prior real PG+ES 36-pass evidence cited; live readiness remained blocked without production mutation)
Task 8: breaker adjudication complete pending re-review (root-only canonical preflight and fixed 1001:1001 snapshot handoff passed; bounded-sudo wrapper preserves the COMPOSE_PROFILES fail-closed topology guard; cidfile plus random-label exact-ID cleanup passed collision, timeout, removal-failure, and no-residue proofs; focused readiness/wiring/preflight 103 passed; privileged real-runtime module 4 passed; broad unit/runtime superset 871 passed; Ruff/ty/bash-n/shellcheck/pre-commit passed; unchanged prior real PG+ES 36-pass evidence cited; live readiness remained blocked without production mutation)
Task 8: post-adjudication deployment-target binding complete pending re-review (one sanitized env-file-only environment and fixed local Docker target now span root preflight plus every rollout command; ambient Docker/Compose/image overrides fail before sudo; sudo is noninteractive and exactly bounded; missing/invalid cidfiles recover only through a random-label full-CID proof with zero residue; focused readiness/wiring/preflight 120 passed; privileged real-runtime module 5 passed including pre-cid and unrelated-container proof; broad unit/runtime superset 888 passed; Ruff/ty/bash-n/shellcheck/pre-commit passed; unchanged real PG+ES 36-pass evidence cited; live readiness remained blocked without production mutation)
Task 8: privileged executable boundary closure complete pending re-review (absolute sudo/Bash/Docker/Compose, fixed system-only PATH and root-owned Docker config now span root preflight and rollout; hostile PATH/HOME/BASH_ENV/CLI-plugin markers cannot cross sudo; focused readiness/wiring 197 passed; privileged real-runtime module 5 passed; broad regulatory/user-file/Vertex/runtime superset 929 passed; Ruff/ty/bash-n/shellcheck passed; unchanged real PG+ES 36-pass evidence cited; live readiness remains blocked and local trusted Docker config is intentionally unprovisioned)
Task 8: final privileged-bundle adjudication complete pending closure (root executes only a manifest-bound release below the fixed root-owned libexec path; deploy uses the same installed digest and overlays across exact noninteractive sudo preflight and non-root rollout; fixed root-only installer, exact sudoers contract, hostile checkout/cwd and ownership/mode/symlink regressions passed; focused unit 176 passed; privileged real-runtime/sudo module 6 passed; broad 55-file superset 940 passed; Ruff/ty/bash-n/shellcheck/pre-commit passed; unchanged real PG+ES 36-pass evidence cited; live readiness remains blocked without any production mutation)
Task 8: breaker adjudication test-evidence correction complete pending closure (the installer staging-root and installed-release-ancestor writable-mode cases now execute their intended branches, assert failure before activation/Docker, and restore modes reliably; RED 1 failed; corrected cases 2 passed; full deploy/preflight module 72 passed; relevant boundary set 110 passed; Ruff/ty/pre-commit passed; no production code or runtime artifact changed)
