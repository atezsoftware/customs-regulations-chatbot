# Agentic Benchmark System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans`
> to implement this plan task-by-task, if that skill is available in your
> environment. Steps use checkbox (`- [ ]`) syntax for tracking. If the skill
> is not installed, track the same steps with the harness's own todo-list tool
> instead — the task/step structure below is still the source of truth.

**Goal:** Let an admin author benchmark questions, pick which OpenRouter
catalog models to test, run them in the background, and see per-model
metrics plus an LLM-judge score — all from the frontend, without touching the
live `/ws/explore` chat path.

**Architecture:** A standalone `core-api` headless runner
(`benchmark_runner.py`) and judge endpoint reuse existing agent/LLM building
blocks with zero changes to the WS chat path. A backend `benchmark` module
(mirroring `llm-catalog`) owns questions CRUD, run orchestration (advisory-
lock + bounded concurrency, mirroring `openrouter-catalog.service.ts`), and
read-time metric aggregation. A frontend admin page drives everything via
polling.

**Tech Stack:** Python 3.10/FastAPI/Pydantic (core-api), TypeScript/LoopBack/
PostgreSQL (backend), React 19/Vite (frontend), reuses the existing
OpenRouter integration end to end.

## Global Constraints

- Zero modifications to `core/api/src/fs_explorer_api/server.py`'s existing
  WS routes, `workflow.py`, or `runs.py`. `tests/api/test_server.py` and
  `tests/api/test_workflow.py` must pass unmodified throughout.
- Every new `core-api` endpoint is internal-token gated exactly like existing
  internal routes (`require_internal_token`).
- Every new backend admin endpoint uses the canonical `getCurrentUser` +
  `requireAdmin` pattern (`backend/src/common/auth/current-user.ts`), not the
  inline duplicate style.
- No stored aggregate columns — all per-model metrics are computed at read
  time from item-level rows.
- `benchmark_runner.py` clears index context in a `finally` on every path.
- `BENCHMARK_MAX_CONCURRENCY` defaults low (3) so a benchmark run cannot
  starve live chat traffic sharing `core-api`'s LLM concurrency semaphore.
- No commit is created for this work unless the user requests one.

---

## File Structure

| File | Responsibility |
| --- | --- |
| `db/migrations/<ts>_add-agentic-benchmark.sql` | All new benchmark tables. |
| `core/api/src/fs_explorer_api/benchmark_runner.py` | Headless single-question agent run (new). |
| `core/api/src/fs_explorer_api/models.py` | Add `JudgmentResult` schema. |
| `core/api/src/fs_explorer_api/server.py` | Add `/api/benchmark/run-question`, `/api/benchmark/judge` routes only. |
| `core/tests/api/test_benchmark_runner.py` | New unit tests. |
| `backend/src/modules/benchmark/*` | New module: models, repositories, controllers, service. |
| `backend/src/index.ts` | Register the benchmark runner tick alongside the existing catalog-sync interval. |
| `frontend/src/pages/AdminBenchmarkPage.tsx` | New admin page. |
| `frontend/src/App.tsx` | Add `/admin/benchmark` route inside `<AdminRoute />`. |
| `frontend/src/lib/endpoints.ts`, `frontend/src/types.ts` | Benchmark API client + DTOs. |

## Task 1: Headless Agent Runner in core-api

**Files:**
- Create: `core/api/src/fs_explorer_api/benchmark_runner.py`
- Test: `core/tests/api/test_benchmark_runner.py`

**Consumes:** `new_workflow`, `InputEvent`, `ExplorationTrace`,
`set_index_context`/`clear_index_context`, `set_search_flags`,
`agent.stream_final_answer`, `extract_cited_sources` — all already exported
by `workflow.py`/`agent.py`/`exploration_trace.py`.

**Produces:** `run_agentic_session(...)` returning the same `stats` shape as
the existing `"complete"` WS event.

- [ ] **Step 1: Write failing tests**

```python
async def test_run_agentic_session_returns_stats_shape(mock_llm_client) -> None:
    result = await run_agentic_session(
        task="What is the transit penalty?",
        index_folders=["virtual://corpus-1"],
        database_url="postgresql://...",
        provider="openrouter",
        model="google/gemini-3-flash-preview",
        temperature=None,
    )
    assert set(result["stats"]) >= {
        "steps", "api_calls", "prompt_tokens", "completion_tokens",
        "thinking_tokens", "total_tokens", "duration_ms",
    }
    assert result["final_result"]

async def test_run_agentic_session_clears_index_context_on_error(mock_llm_client_raises) -> None:
    with pytest.raises(SomeExpectedError):
        await run_agentic_session(task="x", index_folders=["y"], ...)
    assert get_index_context() is None
```

- [ ] **Step 2: Verify RED**

Run: `cd core && uv run --package fs-explorer-api pytest tests/api/test_benchmark_runner.py -v`

Expected: FAIL, module does not exist.

- [ ] **Step 3: Implement `benchmark_runner.py`**

Mirror `_run_fresh_session`'s orchestration (`server.py`) minus every
`websocket.send_json` call: set up `ExplorationTrace`, call
`set_index_context`/`set_search_flags`, build `new_workflow(provider=,
model=, temperature=, on_llm_call=<append to local list>)`, drive
`handler.stream_events()` incrementing a local `step_number` and feeding
`trace`, then `agent.stream_final_answer()` + `extract_cited_sources()`, and
assemble the `stats`/result dict per the design doc §6.1. Wrap the whole body
in `try/finally: clear_index_context()`.

- [ ] **Step 4: Verify GREEN and that existing suites are untouched**

Run: `cd core && uv run --package fs-explorer-api pytest tests/api -v`

Expected: PASS, including `test_server.py`/`test_workflow.py` unchanged.

## Task 2: Judge Endpoint and Rubric

**Files:**
- Modify: `core/api/src/fs_explorer_api/models.py` (add `JudgmentResult`)
- Modify: `core/api/src/fs_explorer_api/server.py` (add two routes only)
- Test: `core/tests/api/test_benchmark_runner.py` (or a sibling
  `test_benchmark_judge.py`)

**Consumes:** `get_llm_client`, `generate_structured`, `require_internal_token`.

**Produces:** `POST /api/benchmark/run-question`, `POST /api/benchmark/judge`.

- [ ] **Step 1: Write failing tests**

```python
async def test_judge_endpoint_scores_within_range(client, mock_judge_llm) -> None:
    response = await client.post("/api/benchmark/judge", json={
        "question": "...", "reference_answer": "...", "candidate_answer": "...",
        "cited_sources": ["doc_x#241"], "judge_provider": "openrouter",
        "judge_model": "anthropic/claude-...",
    }, headers={"X-Internal-Token": TOKEN})
    body = response.json()
    assert 1 <= body["correctness"] <= 5
    assert 0 <= body["overall_score"] <= 100

async def test_run_question_endpoint_requires_internal_token(client) -> None:
    response = await client.post("/api/benchmark/run-question", json={})
    assert response.status_code == 401
```

- [ ] **Step 2: Verify RED**

Run: `cd core && uv run --package fs-explorer-api pytest tests/api/test_benchmark_runner.py -v`

Expected: FAIL, routes do not exist.

- [ ] **Step 3: Implement**

Add `JudgmentResult` to `models.py`. Add the `JUDGE_SYSTEM_PROMPT` constant
with the 4-dimension weighted rubric from the design doc §6.4 (embed the
question, reference answer/expected facts/rubric notes, candidate answer, and
cited sources into the single user turn sent to `generate_structured`).
Compute `overall_score` in Python from the validated `JudgmentResult`, not by
asking the model for it directly, so weighting stays server-controlled.
Both new routes go at the bottom of `server.py`, gated with the existing
`require_internal_token` dependency used elsewhere in the file — do not touch
any existing route.

- [ ] **Step 4: Verify GREEN**

Run: `cd core && uv run --package fs-explorer-api pytest tests/api -v && make format-check && make typecheck`

Expected: PASS.

## Task 3: Database Schema

**Files:**
- Create: `db/migrations/<timestamp>_add-agentic-benchmark.sql`

**Consumes:** Existing `directories` table (for the join table FK).

**Produces:** `benchmark_questions`, `benchmark_question_directories`,
`benchmark_runs`, `benchmark_run_items`, `benchmark_run_judgments`.

- [ ] **Step 1: Write a migration verification check**

Assert (via `information_schema.columns`/`information_schema.table_constraints`)
that all 5 tables exist with their documented columns, and that the unique
constraint on `benchmark_run_items(run_id, provider, model_id, question_id,
repeat_index)` and the FK from `benchmark_question_directories.directory_id`
to `directories` both exist.

- [ ] **Step 2: Verify RED**

Run: `npm --prefix db run migrate:up` against a test DB, then query
`information_schema` — expect the tables to be absent before the migration
file exists.

- [ ] **Step 3: Write the forward + down migration**

Follow the exact `-- Up Migration` / `-- Down Migration` structure of
`db/migrations/20260721090000_add-openrouter-model-catalog-and-costs.sql`.
Use `SERIAL` primary keys, `TIMESTAMPTZ` for all timestamps,
`NUMERIC(20,10)` for `cost_usd`. Down migration drops the 5 tables in FK-safe
order.

- [ ] **Step 4: Verify migration round-trips**

Run: `npm --prefix db run migrate:up && npm --prefix db run migrate:down && npm --prefix db run migrate:up`

Expected: succeeds both directions with no orphaned constraints.

## Task 4: Backend Benchmark Module — Questions CRUD

**Files:**
- Create: `backend/src/modules/benchmark/models/benchmark-question.model.ts`
- Create: `backend/src/modules/benchmark/models/benchmark-question-directory.model.ts`
- Create: `backend/src/modules/benchmark/repositories/*.ts`
- Create: `backend/src/modules/benchmark/controllers/benchmark-questions.controller.ts`
- Modify: `backend/src/application.ts` (register module)
- Test: `backend/src/modules/benchmark/controllers/benchmark-questions.controller.test.ts`

**Consumes:** `getCurrentUser`/`requireAdmin` (`current-user.ts`), existing
`DirectoryRepository`.

**Produces:** `GET/POST/PATCH/DELETE /admin/benchmark/questions`.

- [ ] **Step 1: Write failing tests**

```ts
it('rejects a non-admin caller', async () => {
  await expect(controller.create(nonAdminUser, questionInput)).rejects.toMatchObject({statusCode: 403});
});

it('creates a question with linked directories', async () => {
  const question = await controller.create(adminUser, {prompt: '...', directoryIds: [1, 2]});
  expect(await directoryLinkRepository.find({where: {questionId: question.id}})).toHaveLength(2);
});
```

- [ ] **Step 2: Verify RED**

Run: `cd backend && node --import tsx --test src/modules/benchmark/controllers/benchmark-questions.controller.test.ts`

Expected: FAIL, module does not exist.

- [ ] **Step 3: Implement models/repositories/controller**

Follow the `chat-research-step.model.ts`/`llm-model.model.ts` style
(`@model`/`@property`, explicit `postgresql.columnName`). Controller methods
call `getCurrentUser` + `requireAdmin` first, exactly like
`support.controller.ts`.

- [ ] **Step 4: Verify GREEN**

Run: `cd backend && node --import tsx --test src/modules/benchmark/controllers/benchmark-questions.controller.test.ts && npm run build`

Expected: PASS.

## Task 5: Backend Run Orchestration and Aggregation

**Files:**
- Create: `backend/src/modules/benchmark/models/benchmark-run.model.ts`
- Create: `backend/src/modules/benchmark/models/benchmark-run-item.model.ts`
- Create: `backend/src/modules/benchmark/models/benchmark-run-judgment.model.ts`
- Create: `backend/src/modules/benchmark/repositories/benchmark-run-item.repository.ts` (+ aggregation query method)
- Create: `backend/src/modules/benchmark/services/benchmark-runner.service.ts`
- Create: `backend/src/modules/benchmark/controllers/benchmark-runs.controller.ts`
- Modify: `backend/src/index.ts` (register the tick, mirroring the existing
  catalog-sync `setInterval` registration)
- Test: `backend/src/modules/benchmark/services/benchmark-runner.service.test.ts`
- Test: `backend/src/modules/benchmark/repositories/benchmark-run-item.repository.test.ts`

**Consumes:** `core-api`'s new `/api/benchmark/run-question` and
`/api/benchmark/judge` endpoints, `core-bridge.service.ts`'s
`virtualCorpusKey` helper (for resolving a question's linked directories into
`index_folders`), the advisory-lock pattern from
`openrouter-catalog.service.ts`.

**Produces:** `POST /admin/benchmark/runs`, `GET /admin/benchmark/runs`,
`GET /admin/benchmark/runs/{id}`, `GET /admin/benchmark/runs/{id}/items`,
`POST /admin/benchmark/runs/{id}/cancel`; the background tick that actually
executes items; the stale-item sweep.

- [ ] **Step 1: Write failing tests**

```ts
it('computes pooled avg_tokens_per_step as sum(tokens)/sum(steps), not avg-of-ratios', async () => {
  await seedItems([{steps: 2, totalTokens: 100}, {steps: 8, totalTokens: 100}]);
  const metrics = await repository.aggregateByModel(runId);
  expect(metrics[0].avgTokensPerStep).toBeCloseTo(200 / 10);
});

it('does not start a second tick while the advisory lock is held', async () => {
  const first = service.tick();
  const second = service.tick();
  await Promise.all([first, second]);
  expect(coreApiMock.runQuestion).toHaveBeenCalledTimes(expectedSingleBatchSize);
});

it('reclaims a stale running item past the timeout', async () => {
  await seedItems([{status: 'running', startedAt: elevenMinutesAgo}]);
  await service.sweepStaleItems();
  expect(await repository.findById(itemId)).toMatchObject({status: 'pending'});
});
```

- [ ] **Step 2: Verify RED**

Run: `cd backend && node --import tsx --test src/modules/benchmark/services/benchmark-runner.service.test.ts src/modules/benchmark/repositories/benchmark-run-item.repository.test.ts`

Expected: FAIL, service/repository do not exist.

- [ ] **Step 3: Implement orchestration**

`POST /admin/benchmark/runs` creates the run + one `pending` item per
(model × question) pair, returns `202 {runId}` immediately. The service's
tick: try `pg_try_advisory_lock(<new distinct id>)`; while held, pull up to
`BENCHMARK_MAX_CONCURRENCY` pending items across all `running` runs, mark
`running`, for each resolve the question's linked directories into
`index_folders` via `virtualCorpusKey`, call `run-question` then `judge`,
persist results, update run counters, flip to `completed` when all items are
terminal; release the lock in `finally`. A companion sweep marks items
`running` past `BENCHMARK_STALE_ITEM_MINUTES` back to `pending`.
Aggregation query groups `benchmark_run_items` (+ left join
`benchmark_run_judgments`) by `(provider, model_id)`, computing every metric
in design doc §10 with `SUM`/`AVG`/`PERCENTILE_CONT` in SQL — no
average-of-ratios.

- [ ] **Step 4: Verify GREEN**

Run: `cd backend && node --import tsx --test src/modules/benchmark/services/benchmark-runner.service.test.ts src/modules/benchmark/repositories/benchmark-run-item.repository.test.ts && npm run build`

Expected: PASS.

## Task 6: Frontend Admin Benchmark Page

**Files:**
- Create: `frontend/src/pages/AdminBenchmarkPage.tsx`
- Modify: `frontend/src/App.tsx` (route)
- Modify: `frontend/src/lib/endpoints.ts`, `frontend/src/types.ts`
- Test: component/interaction tests matching whatever pattern
  `AdminSupportPage`'s tests (if any) already use

**Consumes:** `GET /llm/models` (existing), the new
`/admin/benchmark/questions` and `/admin/benchmark/runs*` endpoints.

**Produces:** questions CRUD UI, run configuration UI, polling progress/
results view with per-model comparison.

- [ ] **Step 1: Write failing tests for the polling/results logic**

```ts
test('stops polling once run status leaves running', async () => {
  mockRunStatus(['running', 'running', 'completed']);
  render(<AdminBenchmarkPage />);
  await waitFor(() => expect(screen.getByText(/completed/i)).toBeInTheDocument());
  expect(fetchRunSpy).toHaveBeenCalledTimes(3);
});
```

- [ ] **Step 2: Verify RED**

Run: `cd frontend && npm run test -- AdminBenchmarkPage` (or the project's
equivalent test invocation)

Expected: FAIL, page does not exist.

- [ ] **Step 3: Implement the page**

Split-pane layout matching `AdminSupportPage.tsx`. Questions tab (CRUD
table + modal). Run tab (model multi-select off `GET /llm/models`, question
multi-select/"all active", judge single-select, start button). Run detail
view polls every ~3s while `status === 'running'`, renders a progress bar,
per-model results table with the metrics from design doc §10, row
drill-down panel, and a small comparison chart per the `dataviz` skill's
guidance (no existing chart library in `package.json` — justify any new
dependency or use a dependency-free SVG bar chart for v1).

- [ ] **Step 4: Verify GREEN**

Run: `cd frontend && npm run lint && npm run build`

Expected: PASS. Then manually exercise the page in a running dev server:
create a question, start a run with 1-2 cheap models, watch progress reach
`completed`, confirm the results table and drill-down render real data.

## Task 7: End-to-End Verification

**Files:**
- Modify: `core/tests/api/test_server.py` (add coverage confirming the new
  routes don't affect existing route registration/behavior, if not already
  implied by Task 2's tests)

**Consumes:** complete implementation.

**Produces:** executable evidence the live chat path is unaffected and the
benchmark works end to end.

- [ ] **Step 1: Full regression pass**

```bash
cd core && uv run --package fs-explorer-api pytest tests/api tests/shared
cd backend && npm run build
cd frontend && npm run lint && npm run build
```

Expected: all green, with `tests/api/test_server.py`/`test_workflow.py`
byte-for-byte unmodified in behavior.

- [ ] **Step 2: Live dry run**

With the dev stack running, create 2-3 real benchmark questions against a
real indexed directory, select 1-2 cheap OpenRouter models plus a judge
model, start a run from the admin page, watch it complete, and sanity-check
that the 5 originally-requested metrics plus the judge score look
reasonable (non-zero steps/tokens, judge rationale is coherent, cost is
non-negative and small).

- [ ] **Step 3: Confirm isolation from live chat**

While the dry-run benchmark is executing, use a normal chat session
concurrently and confirm no latency regression or error — this is the
concrete verification of the "ana akış bozulmasın" (main flow must not
break) requirement.

## Plan Self-Review

- Spec coverage: headless runner + judge (Tasks 1-2), schema (Task 3),
  backend CRUD/orchestration/aggregation (Tasks 4-5), frontend (Task 6),
  end-to-end/isolation verification (Task 7).
- No implementation commits are included unless the user requests one.
- The live `/ws/explore` path, `workflow.py`, and `runs.py` are never
  modified — only additive new files/routes/tables.
- Metrics are computed as pooled sums-over-sums where the design doc
  specifies it, avoiding average-of-ratios distortion.
