# Agentic Benchmark System Design

## 1. Purpose

This document specifies an in-product benchmark for the customs-regulations
agentic search chatbot. An admin authors benchmark questions, selects which
catalog LLMs (from the existing OpenRouter integration) participate, starts a
run, and reviews per-model metrics plus an LLM-judge score — all from the
frontend, with the run executing in the background so it never blocks or
degrades the live `/ws/explore` chat path.

This is a design specification. It does not authorize implementation or
contain an implementation commit.

## 2. Confirmed Product Decisions

- Benchmark questions, model selection, run configuration, and results review
  all happen from the frontend admin area; nothing requires a code deploy to
  add/edit/remove a question or to change which models are tested.
- Models under test are exactly the rows already exposed by the existing
  OpenRouter catalog (`GET /llm/models`) — no separate model registry.
- Each benchmark question is linked to one or more existing indexed
  directories (corpora), the same way a chat session is, so answers are
  produced against real indexed regulatory content, never synthetic evidence.
- A run tests every selected (model × question) pair once by default.
- A single, fixed judge model scores every candidate answer in a run, so
  scores are comparable across candidates within that run.
- The benchmark runs as a backend-owned background job; the admin polls for
  progress. It must not add any new load-bearing code path to
  `core-api`'s `/ws/explore` handler or `runs.py`'s resume registry.
- Every reported metric is either already computed by the existing agent/chat
  instrumentation (`stats` in the `"complete"` WS event, `llm_calls`) or is a
  standard, well-known agentic/RAG-evaluation metric (success rate, cost,
  latency percentiles, groundedness/citation rate, LLM-as-judge rubric
  scoring) — nothing invented without a clear analog.

## 3. Why Reuse Rather Than Extend the Live Chat Path

The chat path (`core/api/src/fs_explorer_api/server.py`'s `_run_fresh_session`,
`workflow.py`, `runs.py`) already implements exactly the agent-loop mechanics a
benchmark needs (`new_workflow()`, `ExplorationTrace`, indexed tool calls,
per-call `LLMCallStats` via the `on_llm_call` hook, final `stats` computation)
but is also the most latency- and correctness-sensitive code in the system: it
streams live to real users and supports resuming an interrupted run via
`runs.py`'s TTL-swept registry. Reusing its *building blocks* while keeping a
*separate, additive orchestration path* for benchmarking means a bug in the
benchmark runner cannot regress live chat, and the existing WS test suite
(`tests/api/test_server.py`, `tests/api/test_workflow.py`) staying green,
unmodified, is itself proof of that isolation.

## 4. Architecture

### 4.1 Components

1. **`core-api` headless single-question runner** (`benchmark_runner.py`,
   new) — drives one `FsExplorerAgent` run to completion against a specified
   corpus/model/provider, with no WebSocket dependency, returning the same
   `stats` shape as the existing `"complete"` event.
2. **`core-api` judge endpoint** — a single non-agentic structured LLM call
   (`generate_structured`) that scores a candidate answer against a
   question's reference answer/expected facts using a fixed rubric.
3. **`backend` benchmark module** — questions CRUD, run configuration/
   progress/results endpoints, and a background orchestration service that
   pulls pending (model × question) work items with bounded concurrency and
   an advisory lock, mirroring `openrouter-catalog.service.ts`.
4. **`frontend` admin page** — questions CRUD, model/question/judge
   selection, run launch, polling progress view, per-model results table and
   comparison chart, per-question drill-down.

### 4.2 Request Flow

1. Admin opens `/admin/benchmark`, authors/edits questions (each linked to
   one or more existing indexed directories), and picks models from the
   existing `GET /llm/models` catalog plus a judge model.
2. `POST /admin/benchmark/runs` creates a `benchmark_runs` row and one
   `benchmark_run_items` row per (model × question) pair, all `pending`, and
   returns `202` immediately.
3. Backend's `BenchmarkRunnerService`, ticking on a `setInterval` guarded by a
   Postgres advisory lock (same pattern as catalog sync), pulls up to
   `BENCHMARK_MAX_CONCURRENCY` pending items, marks them `running`, and for
   each: calls `core-api`'s `POST /api/benchmark/run-question` with the
   question's task text, resolved `index_folders` (via the same
   `virtualCorpusKey` mapping `core-bridge.service.ts` already uses for chat
   sessions), and the candidate provider/model.
4. On response, the backend calls `core-api`'s `POST /api/benchmark/judge`
   with the candidate answer, the question's reference answer/expected
   facts/rubric notes, and the run's fixed judge provider/model.
5. The backend persists the `benchmark_run_items` row (stats + answer +
   citations) and the `benchmark_run_judgments` row, updates the parent run's
   progress counters, and flags the run `completed` once every item is
   terminal.
6. The frontend polls `GET /admin/benchmark/runs/{id}` every ~3s while
   `status=running`, rendering progress and, once items complete, per-model
   aggregate metrics computed at read time from `benchmark_run_items`/
   `benchmark_run_judgments`.

A stale-item sweep (same tick) reclaims items stuck `running` past a timeout
(process restart, crashed call) back to `pending`.

## 5. Environment Configuration

```text
BENCHMARK_MAX_CONCURRENCY=3   # backend; bounded parallel (model,question) calls
BENCHMARK_STALE_ITEM_MINUTES=10  # backend; stale "running" item reclaim window
```

No new secrets: benchmark calls reuse the existing `OPENROUTER_API_KEY` (via
`core-api`) and the existing `CORE_INTERNAL_TOKEN` gate for the two new
`core-api` endpoints, exactly like every other internal `core-api` route.

## 6. Core Additions

### 6.1 `benchmark_runner.py` (new module, `core-api`)

`async def run_agentic_session(*, task: str, index_folders: list[str],
database_url: str | None, provider: str | None, model: str | None,
temperature: float | None) -> dict` — builds `ExplorationTrace`, calls
`set_index_context(index_folders, database_url)` /
`set_search_flags(enable_semantic=True, enable_metadata=True)`, constructs
`new_workflow(provider=, model=, temperature=, on_llm_call=<collect into a
list>)`, drives `handler.stream_events()` counting `step_number` and feeding
`trace` exactly as `_run_fresh_session` does today but with no
`websocket.send_json` calls anywhere, then calls
`agent.stream_final_answer()` and `extract_cited_sources()`, and returns:

```python
{
    "final_result": str,
    "error": str | None,
    "incomplete": bool,           # forced_stop without an error
    "cited_sources": list[str],
    "step_path": list[...],        # trace.step_path, for drill-down
    "llm_calls": list[LLMCallStats-shaped dict],
    "stats": {                     # identical field names to the existing
        "steps": int,               # "complete" WS event's stats dict
        "api_calls": int,
        "prompt_tokens": int,
        "completion_tokens": int,
        "thinking_tokens": int,
        "total_tokens": int,
        "tool_result_chars": int,
        "context_summaries": int,
        "duration_ms": int,
        "estimated_cost": float,
    },
}
```

Always clears index context on exit (`finally`) so a benchmark call can never
leak corpus state into a concurrently-running chat request in the same
process — same discipline `_run_fresh_session` already follows.

### 6.2 New endpoint: `POST /api/benchmark/run-question`

Internal-token gated (`require_internal_token`, `fs_explorer_shared.auth`).
Body: `{task, index_folders: list[str], database_url?, provider, model,
temperature?}`. Returns `benchmark_runner.run_agentic_session(...)` as JSON.
Blocking/synchronous — the backend's own bounded-concurrency loop provides
parallelism across items; this endpoint does not stream.

### 6.3 New endpoint: `POST /api/benchmark/judge`

Internal-token gated. Body: `{question, reference_answer?, expected_facts?,
rubric_notes?, candidate_answer, cited_sources, judge_provider, judge_model}`.

Implementation: `get_llm_client(provider=judge_provider, model=judge_model)`
then one `generate_structured(history, JUDGE_SYSTEM_PROMPT, JudgmentResult)`
call — no tools, no agent loop.

New `models.py` schema:

```python
class JudgmentResult(BaseModel):
    correctness: int   # 1-5
    groundedness: int  # 1-5
    completeness: int  # 1-5
    clarity: int       # 1-5
    rationale: str
```

### 6.4 Judge Rubric (Standardization)

A fixed, versioned `JUDGE_SYSTEM_PROMPT` constant gives explicit per-level
anchors so scores stay comparable across very different candidate models and
question types:

| Dimension | Weight | 1 | 3 | 5 |
| --- | --- | --- | --- | --- |
| Correctness | 0.4 | Contradicts reference answer/expected facts, or fabricates a rule | Partially correct with a material gap or minor factual error | Fully matches reference answer/expected facts, no fabrication |
| Groundedness | 0.3 | No citations, or citations that don't support the claim | Citations present but incomplete coverage of claims made | Every material claim backed by a cited source consistent with retrieved evidence |
| Completeness | 0.2 | Ignores the actual question | Answers the main question but misses a clearly-relevant exception/cross-reference | Fully addresses the question including relevant exceptions |
| Clarity | 0.1 | Confusing or contradictory | Serviceable but verbose/unfocused | Direct, well-structured, actionable |

`overall_score = round(100 * (0.4*correctness + 0.3*groundedness +
0.2*completeness + 0.1*clarity) / 5)`.

Using one fixed judge model per run (never swapped per candidate) plus this
fixed rubric plus a per-question authored reference answer is what makes
scores comparable across the models being compared — this is the intentional
answer to "how do we standardize this": fix the judge, fix the rubric, fix
the reference answer; let only the candidate vary.

This is a pointwise LLM-as-judge design in the spirit of the RAGAS
faithfulness/answer-correctness metrics standard for RAG evaluation, adapted
to a single structured-output call per item since each question carries an
authored reference answer rather than requiring retrieval-recall ground
truth.

## 7. Database Design

### 7.1 `benchmark_questions`

| Column | Type | Rules |
| --- | --- | --- |
| `id` | `SERIAL` | Primary key |
| `prompt` | `TEXT` | Required |
| `reference_answer` | `TEXT` | Nullable |
| `expected_facts` | `JSONB` | Nullable array of short strings |
| `rubric_notes` | `TEXT` | Nullable, extra judge guidance |
| `tags` | `JSONB` | Nullable array |
| `is_active` | `BOOLEAN` | Required, default `true` |
| `created_by` / `updated_by` | `INTEGER` | User id |
| `created_at` / `updated_at` | `TIMESTAMPTZ` | Required |

### 7.2 `benchmark_question_directories`

| Column | Type |
| --- | --- |
| `id` | `SERIAL` |
| `question_id` | `INTEGER` FK → `benchmark_questions` |
| `directory_id` | `INTEGER` FK → existing `directories` table |

Mirrors `chat_session_directory`; unique on `(question_id, directory_id)`.

### 7.3 `benchmark_runs`

| Column | Type |
| --- | --- |
| `id` | `SERIAL` |
| `label` | `TEXT` |
| `status` | `TEXT` (`pending`/`running`/`completed`/`error`/`cancelled`) |
| `judge_provider` | `TEXT` |
| `judge_model` | `TEXT` |
| `created_by` | `INTEGER` |
| `total_items` | `INTEGER` |
| `completed_items` | `INTEGER` |
| `failed_items` | `INTEGER` |
| `started_at` / `completed_at` / `created_at` | `TIMESTAMPTZ` |

### 7.4 `benchmark_run_items`

| Column | Type | Rules |
| --- | --- | --- |
| `id` | `SERIAL` | Primary key |
| `run_id` | `INTEGER` | FK → `benchmark_runs` |
| `provider` | `TEXT` | |
| `model_id` | `TEXT` | |
| `question_id` | `INTEGER` | FK → `benchmark_questions` |
| `repeat_index` | `INTEGER` | Default `1` |
| `status` | `TEXT` | `pending`/`running`/`completed`/`error` |
| `final_result` | `TEXT` | Nullable |
| `error_message` | `TEXT` | Nullable |
| `steps` | `INTEGER` | |
| `api_calls` | `INTEGER` | |
| `prompt_tokens` / `completion_tokens` / `thinking_tokens` / `total_tokens` | `INTEGER` | |
| `tool_result_chars` | `INTEGER` | |
| `context_summaries` | `INTEGER` | |
| `duration_ms` | `INTEGER` | |
| `cost_usd` | `NUMERIC(20,10)` | Nullable |
| `cost_source` | `TEXT` | `provider`/`estimated` |
| `cited_sources` | `JSONB` | |
| `step_path` | `JSONB` | For drill-down |
| `started_at` / `completed_at` | `TIMESTAMPTZ` | |

Unique on `(run_id, provider, model_id, question_id, repeat_index)`. Column
names deliberately match the existing `"complete"` WS event's `stats` dict
(`server.py`) and `llm_calls` — one shared vocabulary across the codebase.

### 7.5 `benchmark_run_judgments`

| Column | Type |
| --- | --- |
| `id` | `SERIAL` |
| `run_item_id` | `INTEGER` FK → `benchmark_run_items`, unique |
| `judge_provider` / `judge_model` | `TEXT` |
| `correctness_score` / `groundedness_score` / `completeness_score` / `clarity_score` | `SMALLINT` (1-5) |
| `overall_score` | `SMALLINT` (0-100) |
| `rationale` | `TEXT` |
| `created_at` | `TIMESTAMPTZ` |

No stored aggregate columns anywhere: per-model averages are computed at read
time via a grouped SQL query over `benchmark_run_items`/
`benchmark_run_judgments`, avoiding a second place aggregate figures could
drift from the source rows.

## 8. Backend API Contracts

### 8.1 Questions CRUD (admin-only)

```
GET    /admin/benchmark/questions
POST   /admin/benchmark/questions
PATCH  /admin/benchmark/questions/{id}
DELETE /admin/benchmark/questions/{id}
```

Admin gating uses the canonical `getCurrentUser` + `requireAdmin` pattern
(`backend/src/common/auth/current-user.ts`), as used in
`support.controller.ts`.

### 8.2 Runs

```
POST /admin/benchmark/runs
  { label, providerModelPairs: [{provider, modelId}],
    questionIds: number[] | "all-active",
    judgeProvider, judgeModel }
  -> 202 { runId }

GET  /admin/benchmark/runs               -> list with rollup counts
GET  /admin/benchmark/runs/{id}          -> status + per-model aggregate metrics
GET  /admin/benchmark/runs/{id}/items    -> per-item rows (drill-down)
POST /admin/benchmark/runs/{id}/cancel
```

`POST /admin/benchmark/runs` mirrors the existing `POST
/admin/llm-models/sync` precedent: `202` + id, no second job started while
work is in flight.

## 9. Frontend Design

New page `AdminBenchmarkPage.tsx` at `/admin/benchmark`, gated by the
existing `<AdminRoute />` wrapper in `App.tsx` (same as `/admin/support`,
`/admin/amendments`). List/detail split-pane layout matching
`AdminSupportPage.tsx`'s existing pattern.

- **Questions tab**: table + add/edit/delete modal (prompt, reference answer,
  expected facts as tags, directory multi-select, active toggle).
- **Run tab**: model multi-select fed by `GET /llm/models`, question
  multi-select ("all active" shortcut), judge-model single-select, "Start"
  button.
- **Run detail/progress view**: polls `GET /admin/benchmark/runs/{id}` every
  ~3s while running; progress bar; per-model results table (metrics below,
  §10); row click opens per-question drill-down (answer, citations, judge
  rationale); small comparison chart across selected models.

## 10. Metrics

Computed per (run, model) from `benchmark_run_items`/`benchmark_run_judgments`
as pooled sums-over-sums where relevant (not averages-of-ratios):

1. `avg_steps`
2. `avg_tokens_per_step = sum(total_tokens) / sum(steps)`
3. `avg_total_tokens`
4. `avg_duration_ms`
5. `avg_duration_per_step_ms = sum(duration_ms) / sum(steps)`
6. `avg_cost_usd`, `total_cost_usd`
7. `success_rate` (% completed, no error/forced-stop)
8. `citation_rate` (% completed answers with ≥1 cited source)
9. `avg_api_calls`
10. `avg_context_summaries`
11. `p50_duration_ms`, `p95_duration_ms`
12. `judge_overall_score` (avg 0-100) plus the 4 sub-dimension averages
13. `error_rate`

## 11. Security and Isolation

- New `core-api` endpoints reuse the existing `CORE_INTERNAL_TOKEN` gate — no
  new auth surface.
- Admin-only backend endpoints reuse the canonical `requireAdmin` pattern.
- Benchmark calls compete for `core-api`'s existing process-wide
  `FS_EXPLORER_LLM_MAX_CONCURRENCY` semaphore alongside live chat traffic;
  `BENCHMARK_MAX_CONCURRENCY` defaults low specifically so a large benchmark
  run cannot starve real users.
- `benchmark_runner.py` always clears index context in a `finally`, matching
  `_run_fresh_session`'s existing discipline, so no corpus state can leak
  across concurrent requests in the same process.
- Zero modifications to `server.py`'s existing WS routes, `workflow.py`, or
  `runs.py`.

## 12. Testing Strategy

### 12.1 Core Unit Tests

- `run_agentic_session` produces the same `stats` shape/values as the WS
  path's `"complete"` event for an equivalent scripted run.
- Index context is cleared on both success and exception paths.
- `/api/benchmark/judge` sends the fixed rubric prompt and parses/validates
  `JudgmentResult`; rejects out-of-range scores.
- Existing `tests/api/test_server.py`/`test_workflow.py` remain green,
  unmodified — proof the WS path is untouched.

### 12.2 Backend Unit Tests

- Questions CRUD (ownership/validation).
- Aggregation query: known fixture rows produce known averages/percentiles
  (including the pooled-ratio metrics, not naive per-item averages).
- Advisory lock prevents concurrent runner ticks across replicas.
- Stale-item sweep reclaims a `running` item past the timeout.
- Run status flips to `completed` only once every item is terminal.

### 12.3 Frontend Tests

- Questions CRUD interactions.
- Run-start flow (model/question/judge selection → `202` → redirect to
  progress view).
- Polling renders progress and stops polling once `status != running`.
- Results table sorting/drill-down.

### 12.4 Integration

One small end-to-end run (1-2 models × 2-3 questions) against a fake
OpenRouter transport, reusing the existing `httpx.MockTransport` pattern from
`tests/api/test_llm_openrouter.py`.

## 13. Acceptance Criteria

1. Adding, editing, deactivating, and deleting a benchmark question requires
   no code deploy.
2. Selecting which catalog models participate in a run is a frontend action
   backed by the existing `GET /llm/models` catalog.
3. A running benchmark never blocks, slows measurably, or errors the live
   `/ws/explore` chat path; existing chat test suites pass unmodified.
4. Progress is visible and updates while a run is in flight, without a
   persistent connection.
5. Every one of the 5 originally requested metrics (avg steps, avg tokens/step,
   avg total tokens, avg duration, avg duration/step) is present and computed
   from real per-item data, per model.
6. A judge score is produced per item using one fixed judge model and one
   fixed rubric per run, with rationale text stored for audit.
7. A backend restart mid-run does not permanently strand items in `running`.
8. No new secret or auth surface is introduced; existing internal-token and
   admin-role gates are reused as-is.

## 14. Rollout Sequence

1. DB migration (all-new tables, zero changes to existing tables).
2. `core-api`: `benchmark_runner.py` + `/api/benchmark/run-question` +
   `/api/benchmark/judge`. Zero diff to existing routes/functions.
3. `backend`: benchmark module (models/repositories/controllers/service) +
   advisory-lock runner loop + stale-item sweep.
4. `frontend`: admin page (questions → run config → progress/results).
5. Seed an initial real customs question set with reference answers (content
   work, not code).
6. Dry run with 1-2 cheap models to sanity-check metrics and judge output
   before opening the model list to the full catalog.

## 15. Out of Scope for the First Release

- Pass@k / multi-repeat sampling exposed in the UI (schema allows it later via
  `repeat_index`).
- Automatic model recommendation/ranking beyond showing the metrics.
- Re-judging historical runs with a different judge model (schema keeps this
  possible later — judgments are a separate table keyed off run items).
- Server-enforced budget/spend caps (only a bounded-concurrency guard).
- Streaming live per-token output for a benchmark item (blocking call is
  sufficient; progress granularity is per-item, not per-token).
