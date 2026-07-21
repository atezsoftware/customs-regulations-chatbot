# OpenRouter Multi-Model and Cost Tracking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let every user select any OpenRouter model that is compatible with the structured research agent, persist that selection per chat, and record/display accurate USD cost per provider call.

**Architecture:** Add an OpenRouter implementation behind the existing core `LLMClient` protocol. Backend owns an authenticated, synchronized compatible-model catalog, price history, session selection validation, and cost persistence. Frontend consumes only that safe catalog and persists model changes via the session endpoint.

**Tech Stack:** Python 3.10/FastAPI/Pydantic/httpx, TypeScript/LoopBack/PostgreSQL, React 19/Vite/Tailwind, OpenRouter Chat Completions and Models APIs.

## Global Constraints

- `OPENROUTER_API_KEY` exists only in server/deployment secrets; never browser, database, log, source, or error response.
- Default provider/model are `openrouter` / `google/gemini-3-flash-preview` for new chats.
- Expose all and only current text-input/text-output models that support `structured_outputs`; no user allowlist.
- Persist a session model until the user changes it; active runs retain their start-time selection.
- Provider-reported `usage.cost` is authoritative. Decimal values are stored as PostgreSQL `NUMERIC`, never JavaScript/Python floats.
- Normalize OpenRouter reasoning tokens as `output = completion - reasoning`; never double-count reasoning in total token or cost calculations.
- Show USD as `$` with two to six fraction digits; mark fallback-calculated costs as estimated.
- Preserve direct Gemini as an unlisted rollback provider.
- No commit is created for this work unless the user requests one.

---

## File Structure

| File | Responsibility |
| --- | --- |
| `core/api/pyproject.toml` | Add `httpx` HTTP client dependency. |
| `core/api/src/fs_explorer_api/llm/base.py` | Provider-neutral extended usage contract. |
| `core/api/src/fs_explorer_api/llm/openrouter.py` | Chat Completions adapter, stream parser, retry/error/usage normalization. |
| `core/api/src/fs_explorer_api/llm/factory.py` | Register `openrouter` while retaining direct Gemini. |
| `core/tests/api/test_llm_openrouter.py` | Unit tests with an injected `httpx.MockTransport`. |
| `db/migrations/20260721090000_add-openrouter-model-catalog-and-costs.sql` | Catalog, immutable price snapshot, sync-run, session, and call-cost schema. |
| `backend/src/modules/llm-catalog/*` | Catalog models, repositories, synchronization service, public/admin controllers. |
| `backend/src/modules/chat/*` | Session model persistence/validation and enriched provider-call persistence. |
| `backend/src/modules/analytics/*` | Decimal-safe cost aggregates. |
| `backend/src/application.ts` | Register catalog lifecycle/sync binding. |
| `frontend/src/components/chat/ModelSelector.tsx` | Accessible searchable model combobox. |
| `frontend/src/components/chat/ChatInput.tsx` | Embed selector beside send/stop controls. |
| `frontend/src/lib/endpoints.ts`, `frontend/src/types.ts` | Safe catalog/session/cost API types. |
| `frontend/src/components/chat/UsageFooter.tsx`, `frontend/src/pages/DashboardPage.tsx` | Actual/estimated USD presentation. |

## Task 1: Extend the Core Usage Contract

**Files:**
- Modify: `core/api/src/fs_explorer_api/llm/base.py`
- Modify: `core/api/src/fs_explorer_api/agent.py`
- Modify: `core/api/src/fs_explorer_api/server.py`
- Test: `core/tests/api/test_llm_openrouter.py`

**Consumes:** Existing `LLMUsage`, `LLMCallStats`, and core WebSocket `llm_call` event.

**Produces:** A stable event contract for all providers:

```python
class LLMUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    cached_input_tokens: int = 0
    cache_write_tokens: int = 0
    duration_ms: float = 0
    generation_id: str | None = None
    billed_cost_usd: Decimal | None = None
    upstream_cost_usd: Decimal | None = None
    cost_source: Literal['provider', 'estimated'] | None = None
```

- [ ] **Step 1: Write failing usage-normalization tests**

```python
def test_openrouter_completion_total_is_not_double_counted() -> None:
    usage = usage_from_openrouter({
        'prompt_tokens': 100,
        'completion_tokens': 30,
        'completion_tokens_details': {'reasoning_tokens': 12},
    })
    assert (usage.input_tokens, usage.output_tokens, usage.thinking_tokens) == (100, 18, 12)
    assert usage.input_tokens + usage.output_tokens + usage.thinking_tokens == 130
```

- [ ] **Step 2: Run the test to verify RED**

Run: `cd core && uv run --package fs-explorer-api pytest tests/api/test_llm_openrouter.py::test_openrouter_completion_total_is_not_double_counted -v`

Expected: FAIL because `usage_from_openrouter` does not exist.

- [ ] **Step 3: Add only the provider-neutral fields and forward them through `LLMCallStats`/WebSocket events**

Keep direct Gemini values at defaults. Serialize decimal costs as strings in WebSocket JSON so downstream TypeScript never receives an imprecise number.

- [ ] **Step 4: Run focused core tests**

Run: `cd core && uv run --package fs-explorer-api pytest tests/api/test_agent.py tests/api/test_server.py tests/api/test_llm_openrouter.py -v`

Expected: PASS.

## Task 2: Implement the OpenRouter Core Adapter

**Files:**
- Modify: `core/api/pyproject.toml`
- Create: `core/api/src/fs_explorer_api/llm/openrouter.py`
- Modify: `core/api/src/fs_explorer_api/llm/factory.py`
- Modify: `core/api/src/fs_explorer_api/llm/__init__.py`
- Test: `core/tests/api/test_llm_openrouter.py`

**Consumes:** `ChatTurn`, `ThinkingLevel`, `LLMUsage`, OpenRouter key/config environment variables.

**Produces:** `OpenRouterLLMClient` implementing `generate_structured`, `stream_text`, and `last_stream_usage`.

- [ ] **Step 1: Write failing adapter tests using `httpx.MockTransport`**

Cover (as separate tests): role conversion; JSON-schema request body; structured Pydantic validation; provider cost; reasoning split; `Retry-After` on a pre-stream 429; keepalive SSE frames; final stream usage; and in-band `error` after a partial `delta`.

```python
async def test_structured_request_uses_strict_json_schema() -> None:
    client, seen = make_openrouter_client(response_json=structured_response())
    result, _ = await client.generate_structured([ChatTurn(role='user', text='x')], 'system', Action)
    assert seen['json']['response_format']['json_schema']['strict'] is True
    assert result.type == 'stop'
```

- [ ] **Step 2: Verify RED**

Run: `cd core && uv run --package fs-explorer-api pytest tests/api/test_llm_openrouter.py -v`

Expected: FAIL with `ModuleNotFoundError` or missing factory branch.

- [ ] **Step 3: Implement the adapter**

Use `httpx.AsyncClient`, injected as an optional constructor dependency for tests. Always send `Authorization: Bearer <key>`, `HTTP-Referer`, and `X-Title` server side. Map `model` turns to OpenAI-compatible `assistant` turns and use a first `system` message. For structured calls send:

```python
payload['response_format'] = {
    'type': 'json_schema',
    'json_schema': {'name': schema.__name__, 'strict': True, 'schema': schema.model_json_schema()},
}
```

Only retry timeout/408/429/502/503 before a stream yields content. Parse SSE `data:` frames, ignore `[DONE]` and comments, emit `choices[0].delta.content`, and raise a typed sanitized error for a stream error frame. Capture final `usage`, resolved `model`, and generation id.

- [ ] **Step 4: Register provider and configuration**

`get_llm_client(provider='openrouter')` reads `OPENROUTER_API_KEY`, errors with a sanitized missing-key exception, and defaults to `OPENROUTER_DEFAULT_MODEL` then `google/gemini-3-flash-preview`. Gemini remains unchanged.

- [ ] **Step 5: Verify GREEN and formatting**

Run: `cd core && uv run --package fs-explorer-api pytest tests/api/test_llm_openrouter.py tests/api/test_llm_gemini.py -v && make format-check && make typecheck`

Expected: PASS.

## Task 3: Add Durable Catalog, Price, and Call-Cost Schema

**Files:**
- Create: `db/migrations/20260721090000_add-openrouter-model-catalog-and-costs.sql`
- Modify: `backend/src/modules/chat/models/chat-session.model.ts`
- Modify: `backend/src/modules/chat/models/llm-call.model.ts`
- Test: migration against `db/docker-compose.test.yml`

**Consumes:** Existing `chat_sessions` and `llm_calls` tables.

**Produces:** Schema that can preserve price history and audit every provider request.

- [ ] **Step 1: Write a migration verification script/test**

Assert `llm_models`, `llm_model_price_snapshots`, and `llm_model_sync_runs` exist; `chat_sessions.llm_provider` is non-null after migration; and `llm_calls.billed_cost_usd` is `NUMERIC`.

- [ ] **Step 2: Verify RED**

Run: `npm --prefix db run migrate:up` against the test database, then query `information_schema.columns`.

Expected: FAIL before the migration exists.

- [ ] **Step 3: Write the forward-only migration**

Create catalog primary key `(provider, model_id)`, store raw pricing as `JSONB`, current active/expiry/compatibility data, immutable snapshot rows keyed by a pricing hash, and sync-run rows with sanitized error text. Add `llm_provider TEXT NOT NULL DEFAULT 'openrouter'` to sessions and add generation/cached/cache-write/decimal-cost/snapshot fields to `llm_calls`. Backfill existing sessions as `gemini` so historical direct calls retain their origin.

- [ ] **Step 4: Update LoopBack entity fields**

Map every added column with explicit `postgresql.columnName`; use strings for decimal values at the API boundary.

- [ ] **Step 5: Verify migration round-trip in disposable test DB**

Run: `npm --prefix db run migrate:up && npm --prefix backend run build`

Expected: migration succeeds and TypeScript compiles.

## Task 4: Build Backend Catalog Synchronization and APIs

**Files:**
- Create: `backend/src/modules/llm-catalog/models/llm-model.model.ts`
- Create: `backend/src/modules/llm-catalog/models/llm-model-price-snapshot.model.ts`
- Create: `backend/src/modules/llm-catalog/models/llm-model-sync-run.model.ts`
- Create: `backend/src/modules/llm-catalog/repositories/*.ts`
- Create: `backend/src/modules/llm-catalog/services/openrouter-catalog.service.ts`
- Create: `backend/src/modules/llm-catalog/controllers/llm-models.controller.ts`
- Create: `backend/src/modules/llm-catalog/controllers/admin-llm-models.controller.ts`
- Modify: `backend/src/application.ts`
- Test: `backend/src/modules/llm-catalog/services/openrouter-catalog.service.test.ts`

**Consumes:** Server-side `OPENROUTER_API_KEY`, `/api/v1/models/user`, database schema.

**Produces:** `GET /llm/models`, `POST /admin/llm-models/sync`, hourly/startup synchronization.

- [ ] **Step 1: Write failing catalog service tests**

```ts
it('keeps only active text structured-output models', async () => {
  const result = filterCompatibleModels([textStructured(), imageOnly(), noStructuredOutput(), expired()]);
  expect(result.map(model => model.modelId)).toEqual(['vendor/compatible']);
});

it('retains the last successful catalog when refresh fails', async () => {
  await service.sync();
  fetchMock.mockRejectedValueOnce(new Error('network down'));
  await expect(service.sync()).rejects.toThrow('network down');
  await expect(repository.find({where: {isActive: true}})).resolves.toHaveLength(1);
});
```

- [ ] **Step 2: Verify RED**

Run: `cd backend && node --import tsx --test src/modules/llm-catalog/services/openrouter-catalog.service.test.ts`

Expected: FAIL because service/filter do not exist.

- [ ] **Step 3: Implement sync and readiness behavior**

Use one `undici` request with the server key. A model is publishable only when text input/output, `supported_parameters` includes `structured_outputs`, expiry is absent/future, and required model/pricing data parses. Acquire PostgreSQL `pg_try_advisory_lock`, record a sync-run row, skip malformed rows, upsert catalog, add a snapshot only when a deterministic SHA-256 pricing hash changes, and deactivate only after a non-empty successful response.

On missing/rejected key or failed first-ever sync, expose a sanitized admin diagnostic and return `503` from `GET /llm/models`; never return a key or authorization header. After one successful sync, return last-known-good data with `lastSyncedAt` even if a later sync fails.

- [ ] **Step 4: Implement safe API shapes**

`GET /llm/models` returns only compatible active models, default model id, and `lastSyncedAt`. Pricing is string-valued. `POST /admin/llm-models/sync` returns `202` and starts no duplicate sync while the lock is held.

- [ ] **Step 5: Verify GREEN**

Run: `cd backend && node --import tsx --test src/modules/llm-catalog/services/openrouter-catalog.service.test.ts && npm run build`

Expected: PASS.

## Task 5: Persist and Validate Chat Model Selection; Record Actual Costs

**Files:**
- Modify: `backend/src/modules/chat/controllers/chat-sessions.controller.ts`
- Modify: `backend/src/modules/chat/controllers/chat-messages.controller.ts`
- Modify: `backend/src/modules/chat/services/core-bridge.service.ts`
- Modify: `backend/src/modules/chat/repositories/llm-call.repository.ts`
- Modify: `backend/src/modules/analytics/controllers/usage.controller.ts`
- Test: `backend/src/modules/chat/controllers/chat-sessions.controller.test.ts`
- Test: `backend/src/modules/chat/services/core-bridge.service.test.ts`

**Consumes:** active catalog and enriched core `llm_call` events.

**Produces:** `PATCH /chat-sessions/{id}/model`, session-snapshotted run configuration, persisted provider cost and cost analytics.

- [ ] **Step 1: Write failing session validation tests**

```ts
it('rejects another user and inactive model selections', async () => {
  await expect(controller.setModel(7, {provider: 'openrouter', modelId: 'vendor/offline'})).rejects.toMatchObject({statusCode: 400});
});

it('uses the session model instead of a message-body model', async () => {
  await controller.create(7, {content: 'question', model: 'attacker/model'} as never);
  expect(sessionRepository.updateById).not.toHaveBeenCalledWith(7, expect.objectContaining({model: 'attacker/model'}));
});
```

- [ ] **Step 2: Verify RED**

Run: `cd backend && node --import tsx --test src/modules/chat/controllers/chat-sessions.controller.test.ts src/modules/chat/services/core-bridge.service.test.ts`

Expected: FAIL because no dedicated model endpoint/validation exists.

- [ ] **Step 3: Implement selection and start-time snapshot**

Create `PATCH /chat-sessions/{id}/model` with `{provider, modelId}`. Validate ownership and active structured-output catalog membership. New sessions receive configured OpenRouter Gemini Flash only if active; otherwise creation fails safely. Remove `model` from message-send API and never trust it from the browser. The bridge reads provider/model/temperature once before opening its core socket and passes both provider and model in the start frame.

- [ ] **Step 4: Persist real and estimated cost**

On each `llm_call`, persist the resolved provider/model, generation id, separated output/reasoning/cache quantities, duration, and decimal `billed_cost_usd`. If core marks cost unavailable, calculate from that call's immutable snapshot using:

```text
request + uncached_input*prompt + cached_input*cache_read + cache_write*cache_write + (output+thinking)*completion
```

Persist `cost_source='estimated'`; do not replace a provider figure with a catalog estimate. Update repository SQL and analytics response to include actual cost, estimated subtotal, and model cost breakdown without changing legacy total-token semantics.

- [ ] **Step 5: Verify GREEN**

Run: `cd backend && node --import tsx --test src/modules/chat/controllers/chat-sessions.controller.test.ts src/modules/chat/services/core-bridge.service.test.ts && npm run build`

Expected: PASS.

## Task 6: Add Frontend Model Picker and Cost UI

**Files:**
- Create: `frontend/src/components/chat/ModelSelector.tsx`
- Modify: `frontend/src/components/chat/ChatInput.tsx`
- Modify: `frontend/src/pages/ChatPage.tsx`
- Modify: `frontend/src/components/chat/UsageFooter.tsx`
- Modify: `frontend/src/pages/DashboardPage.tsx`
- Modify: `frontend/src/lib/endpoints.ts`
- Modify: `frontend/src/types.ts`
- Create: `frontend/src/lib/currency.ts`
- Test: `frontend/src/lib/currency.test.ts`

**Consumes:** `GET /llm/models`, selection PATCH, enriched message/analytics usage.

**Produces:** searchable, keyboard-accessible selector that persists the user choice and presents actual versus estimated USD accurately.

- [ ] **Step 1: Write failing pure formatter tests**

```ts
test('formats very small USD amounts without losing the dollar sign', () => {
  expect(formatUsd('0.000123')).toBe('$0.000123');
});

test('limits fraction digits to six', () => {
  expect(formatUsd('12.3456789')).toBe('$12.345679');
});
```

- [ ] **Step 2: Verify RED**

Run: `cd frontend && node --import tsx --test src/lib/currency.test.ts`

Expected: FAIL because `formatUsd` does not exist.

- [ ] **Step 3: Implement data types/API/formatter**

Represent all price and cost values as strings at the API boundary. `formatUsd` uses `Intl.NumberFormat('en-US', {style: 'currency', currency: 'USD', minimumFractionDigits: 2, maximumFractionDigits: 6})` only after controlled decimal-to-display conversion; avoid aggregate arithmetic in the browser.

- [ ] **Step 4: Implement `ModelSelector` and connect it to `ChatInput`**

Fetch catalog on chat page load, render selected model name plus input/output `$ / 1M` pricing, and use a filterable list with ArrowUp/ArrowDown/Enter/Escape. On selection, PATCH the session then update local `sessions`; do not send a model with the next message. While a request runs, make selector read-only and state that the selected model applies to the next message. Keep an inactive historical session visible, show `Model no longer available`, and switch future messages to server-validated default only.

- [ ] **Step 5: Add footer/dashboard cost fields**

Show model, calls, tokens, duration, and actual USD in `UsageFooter`. Add an `Estimated cost` label only if any constituent call is estimated. Dashboard receives/display actual total, estimated subtotal, and provider/model cost breakdown.

- [ ] **Step 6: Verify frontend build, lint, and formatter tests**

Run: `cd frontend && node --import tsx --test src/lib/currency.test.ts && npm run lint && npm run build`

Expected: PASS.

## Task 7: End-to-End Verification and Rollout Checks

**Files:**
- Modify: `core/tests/api/test_server.py`
- Modify: `backend/src/modules/analytics/controllers/usage.controller.test.ts`
- Modify: `README.md` only if existing deployment environment-variable documentation has a matching section.

**Consumes:** complete OpenRouter core/backend/frontend implementation.

**Produces:** executable evidence that accuracy contract, accounting, failure behavior, and rollback are intact.

- [ ] **Step 1: Add fake-provider end-to-end test**

Use a fake OpenRouter model catalog and SSE response. Assert it produces: valid structured action, final answer, provider/model/generation id, `input=100`, `output=18`, `thinking=12`, actual cost string, and a `llm_calls` row with no double-counted total.

- [ ] **Step 2: Verify RED then GREEN**

Run the test before wiring the final event path (expected FAIL), then after implementation:

`cd core && uv run --package fs-explorer-api pytest tests/api/test_server.py tests/api/test_llm_openrouter.py -v`

Expected: PASS.

- [ ] **Step 3: Run regression and static checks**

```bash
cd core && uv run --package fs-explorer-api pytest tests/api tests/shared
cd backend && npm run build
cd frontend && npm run lint && npm run build
git diff --check
```

Expected: all commands succeed; no whitespace errors.

- [ ] **Step 4: Run the approved accuracy/cost benchmark before enabling production default**

Execute the existing fixed customs benchmark once with direct Gemini and once with OpenRouter Gemini Flash using identical corpus, prompts, temperature, thinking level, and questions. Record accuracy/citation assertions, provider call count, separated token categories, duration, and actual USD. Do not promote the default unless the established factual/citation assertions remain green.

- [ ] **Step 5: Deployment readiness review**

Set only `OPENROUTER_API_KEY`, `FS_EXPLORER_LLM_PROVIDER=openrouter`, and `OPENROUTER_DEFAULT_MODEL=google/gemini-3-flash-preview` in managed server secrets. Verify first sync/readiness, model endpoint, one new session selection, actual cost footer, and `FS_EXPLORER_LLM_PROVIDER=gemini` rollback before production enablement.

## Plan Self-Review

- Spec coverage: core adapter (Tasks 1–2), durable catalog and price history (Tasks 3–4), session safety/cost analytics (Task 5), model selector/USD UI (Task 6), accuracy/error/rollback verification (Task 7).
- No implementation commits are included because the user explicitly requested no commit.
- Cost fields are strings/decimals across boundaries; total-token accounting uses normalized output plus thinking exactly once.
- No API key is introduced into any frontend type, endpoint, storage model, or error contract.
