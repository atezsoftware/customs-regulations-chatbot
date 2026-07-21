# OpenRouter Multi-Model Selection and Cost Tracking Design

## 1. Purpose

This document specifies how the customs-regulations chatbot will use one
server-side OpenRouter account to offer a dynamically updated list of compatible
LLM models. A user selects a model per chat session; that selection remains in
effect until the user changes it. Every LLM call records native token usage and
the actual USD amount charged by OpenRouter.

This is a design specification. It does not authorize implementation or contain
an implementation commit.

## 2. Confirmed Product Decisions

- OpenRouter is the default LLM provider for chatbot research and final answers.
- One shared `OPENROUTER_API_KEY` is stored only as a server/deployment secret.
- Every OpenRouter model compatible with the agent's structured-output
  requirement is automatically available to users. There is no first-release
  allowlist.
- A selected model is persisted on the chat session and remains active until the
  user changes it.
- New chats default to Gemini Flash through OpenRouter.
- Costs are displayed in USD with a dollar sign and fractional digits.
- Catalog prices are synchronized automatically, while completed-call costs use
  the actual amount returned by OpenRouter.
- Historical calls are never repriced after a catalog price change.

## 3. Why Compatibility Filtering Is Required

The existing agent does not use provider-native tool calling. Instead, each
research-planning turn asks the model for an `Action` object validated against a
Pydantic JSON Schema. A model that cannot enforce structured JSON can break the
research loop even if it can generate ordinary chat text.

The public user catalog therefore contains every model that satisfies all of
these conditions:

1. The model is present in the latest successful OpenRouter catalog response.
2. It accepts text input and produces text output.
3. `supported_parameters` contains `structured_outputs`.
4. The model is not expired at synchronization time.
5. Its model ID and required pricing fields are valid.

Streaming is supported by the OpenRouter Chat Completions API across models, so
it is handled by the adapter rather than used as an additional catalog filter.
Models may also support images, files, audio, or tools; those extra capabilities
do not affect first-release eligibility.

References:

- [OpenRouter model catalog API](https://openrouter.ai/docs/api/api-reference/models/get-models)
- [OpenRouter structured outputs](https://openrouter.ai/docs/guides/features/structured-outputs)
- [OpenRouter streaming](https://openrouter.ai/docs/api/reference/streaming)

## 4. Architecture

### 4.1 Components

The implementation adds four focused units:

1. **OpenRouter LLM adapter in core-api**
   - Implements the existing `LLMClient` protocol.
   - Translates provider-independent chat turns, thinking levels, JSON Schemas,
     streaming text, errors, and usage into/from OpenRouter Chat Completions.

2. **Model catalog and price synchronization in backend**
   - Fetches the OpenRouter model catalog using the server-side key.
   - Filters compatible models.
   - Stores current metadata and immutable price history.
   - Exposes the safe catalog to the frontend.

3. **Session model selection in backend/frontend**
   - Persists `provider=openrouter` and the OpenRouter model slug on each chat.
   - Validates every selection against the active catalog.
   - Sends only the server-validated session selection to core-api.

4. **Actual cost accounting**
   - Extends the core usage event and backend `llm_calls` record.
   - Stores OpenRouter's billed cost as the authoritative amount.
   - Uses a price snapshot only when the provider does not return a cost.

### 4.2 Request Flow

1. The frontend loads `GET /llm/models` when the authenticated application
   starts and refreshes it when the catalog timestamp changes.
2. A new chat receives `provider=openrouter` and the configured Gemini Flash
   default.
3. The user may select another active model. The frontend sends the selected
   model to the dedicated session-model endpoint.
4. When the user submits a question, the backend reads the model from the chat
   session. It does not trust a model slug supplied in the message body.
5. The backend opens the existing internal core WebSocket and sends provider,
   model, and temperature.
6. Core constructs a fresh `OpenRouterLLMClient` for the run.
7. Structured planning calls and the streamed final answer use the same selected
   model.
8. Each core LLM usage event contains provider, resolved model, generation ID,
   token details, duration, and provider-reported cost.
9. The backend persists one `llm_calls` row per actual provider request.
10. The final SSE event updates the chat UI with the completed answer, tokens,
    duration, model, and cost.

The selected model is snapshotted when a run starts. Changing the session model
while a run is active affects the next user message, not the request already in
progress.

## 5. Environment and Secret Configuration

### 5.1 Required Variables

```text
OPENROUTER_API_KEY=<deployment secret; user supplies value>
FS_EXPLORER_LLM_PROVIDER=openrouter
OPENROUTER_DEFAULT_MODEL=google/gemini-3-flash-preview
OPENROUTER_CATALOG_SYNC_MINUTES=60
OPENROUTER_HTTP_REFERER=https://dev-customs-regulations.singlewindow.io
OPENROUTER_APP_TITLE=Customs Regulations Chatbot
```

`OPENROUTER_API_KEY` is required by both backend catalog synchronization and
core-api inference. Both workloads reference the same secret value. The key is
never written to source control, database rows, client bundles, API responses,
structured logs, or error payloads.

Example and local environment files contain only this placeholder:

```text
OPENROUTER_API_KEY=
```

### 5.2 Optional Variables

```text
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_REQUIRE_ZDR=false
OPENROUTER_REQUEST_TIMEOUT_SECONDS=90
OPENROUTER_MAX_RETRIES=3
```

`OPENROUTER_REQUIRE_ZDR=false` preserves the confirmed requirement that all
compatible models are available. If the organization later requires zero data
retention, setting it to `true` changes catalog synchronization to the
OpenRouter ZDR-filtered catalog. This is a deployment policy, not a user-facing
toggle.

## 6. Core LLM Adapter

### 6.1 Provider Factory

`get_llm_client()` retains the existing provider abstraction and gains the
`openrouter` branch. Provider precedence remains:

1. Explicit per-run provider.
2. `FS_EXPLORER_LLM_PROVIDER`.
3. Existing provider default during migration.

The OpenRouter branch requires `OPENROUTER_API_KEY` and returns a dedicated
`OpenRouterLLMClient`. Gemini's direct client remains available for rollback and
controlled A/B validation but is not exposed in the first-release model menu.

### 6.2 Chat Turn Mapping

The internal roles map as follows:

| Internal role | OpenRouter role |
| --- | --- |
| `user` | `user` |
| `model` | `assistant` |

The existing `system_prompt` is sent as the first `system` message. Text is sent
as plain Unicode; the existing HTML-entity normalization remains in place.

### 6.3 Structured Planning

`generate_structured()` sends a non-streaming Chat Completions request with:

```json
{
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "name": "agent_action",
      "strict": true,
      "schema": "<schema.model_json_schema()>"
    }
  }
}
```

The adapter validates the returned content with the original Pydantic model.
If provider output fails validation, the adapter performs at most one corrective
retry using the same schema and a compact validation-error instruction. A
second invalid response fails the call with a typed structured-output error.

### 6.4 Reasoning Mapping

The agent's existing thinking levels map to OpenRouter's normalized reasoning
configuration:

| Agent level | OpenRouter request |
| --- | --- |
| `minimal` | `reasoning.effort=minimal` |
| `low` | `reasoning.effort=low` |
| `medium` | `reasoning.effort=medium` |
| `high` | `reasoning.effort=high` |

The model catalog's reasoning metadata determines whether a model exposes
explicit effort control. If it does not, the adapter omits the reasoning
parameter and accepts the model/provider default. Mandatory reasoning models
must never be sent `effort=none`.

Reasoning tokens are stored separately for observability but are treated as
output tokens for billing, matching OpenRouter accounting.

Reference: [OpenRouter reasoning tokens](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens)

### 6.5 Streaming Final Answers

`stream_text()` uses `stream=true` and:

- Ignores SSE comment/keepalive frames.
- Emits only `choices[].delta.content` to the existing answer stream.
- Captures the final `usage` object.
- Captures generation ID and resolved model.
- Detects in-band mid-stream errors even when HTTP status is `200`.
- Preserves partial output on a mid-stream failure.
- Does not silently retry with another model after content has been emitted.

### 6.6 Usage Contract

`LLMUsage` and `LLMCallStats` are extended with optional provider-independent
accounting fields:

```text
provider
model
generation_id
input_tokens
output_tokens
thinking_tokens
cached_input_tokens
cache_write_tokens
duration_ms
billed_cost_usd
upstream_cost_usd
cost_source
```

`cost_source` is `provider` when OpenRouter supplied `usage.cost`, and
`estimated` only when the backend calculated a fallback from a stored price
snapshot.

OpenRouter reports reasoning tokens inside `completion_tokens`. The adapter
must normalize this without double-counting:

```text
thinking_tokens = completion_tokens_details.reasoning_tokens ?? 0
output_tokens = max(completion_tokens - thinking_tokens, 0)
```

The application's existing `input + output + thinking` total therefore remains
equal to the provider's prompt-plus-completion total. Cached input and cache
write tokens are reported as detail dimensions and are not added to that total
a second time.

## 7. Model Catalog and Price Synchronization

### 7.1 Source

The backend calls the authenticated OpenRouter user-model endpoint so account
preferences, privacy settings, guardrails, and provider availability are
reflected in the catalog. Compatibility filtering is then performed locally.

Reference: [OpenRouter user-filtered models](https://openrouter.ai/docs/api/api-reference/models/list-models-user)

### 7.2 Schedule

- Run once during backend startup without blocking server boot beyond the
  configured HTTP timeout.
- Run every 60 minutes.
- Allow an admin-triggered refresh.
- Use a PostgreSQL advisory lock so only one backend replica synchronizes.
- Record every attempt, result, duration, item count, and sanitized error.

### 7.3 Failure Behavior

- A failed sync never deletes or deactivates the last successful catalog.
- The frontend continues to receive the last successful models and
  `lastSyncedAt` timestamp.
- A successful sync marks missing/expired models inactive.
- Reappearing compatible models become active again.
- Invalid individual model rows are skipped and counted; they do not fail the
  entire catalog.
- An empty successful-looking response is treated as a sync failure, preventing
  accidental mass deactivation.
- A missing or rejected API key makes OpenRouter readiness degraded and records
  a sanitized admin diagnostic.
- If no last-known-good catalog exists on first boot, `/llm/models` returns
  `503` and OpenRouter chat runs cannot start. The normal user receives the
  existing safe error message; the admin view identifies the configuration or
  synchronization failure without exposing the key.

### 7.4 Price Semantics

OpenRouter catalog price strings are stored exactly and normalized to
high-precision decimal columns. Display prices per one million tokens are:

```text
prompt_usd_per_million = prompt_usd_per_token * 1,000,000
completion_usd_per_million = completion_usd_per_token * 1,000,000
```

The raw pricing object is retained because some models also charge per request,
image, web search, cache read/write, or other units.

When provider-reported cost is temporarily unavailable, the estimate uses the
price snapshot captured for that call:

```text
estimated_cost_usd =
  request_usd
  + uncached_input_tokens * prompt_usd_per_token
  + cached_input_tokens * cache_read_usd_per_token
  + cache_write_tokens * cache_write_usd_per_token
  + (output_tokens + thinking_tokens) * completion_usd_per_token
```

Missing optional cache prices fall back to the prompt rate. Every fallback is
stored and displayed as `estimated`; it never replaces a later provider-reported
actual cost silently.

Catalog pricing supports estimates and model comparison. It is not used to
overwrite provider-reported historical costs.

## 8. Database Design

### 8.1 `llm_models`

Stores current catalog metadata.

| Column | Type | Rules |
| --- | --- | --- |
| `provider` | `TEXT` | Part of primary key; `openrouter` |
| `model_id` | `TEXT` | Part of primary key; full OpenRouter slug |
| `canonical_slug` | `TEXT` | Nullable canonical identifier |
| `display_name` | `TEXT` | Required |
| `description` | `TEXT` | Nullable |
| `context_length` | `INTEGER` | Required positive value |
| `max_completion_tokens` | `INTEGER` | Nullable |
| `input_modalities` | `JSONB` | Required array |
| `output_modalities` | `JSONB` | Required array |
| `supported_parameters` | `JSONB` | Required array |
| `reasoning_config` | `JSONB` | Nullable provider metadata |
| `architecture` | `JSONB` | Required provider metadata |
| `raw_pricing` | `JSONB` | Required |
| `is_active` | `BOOLEAN` | Required, indexed |
| `created_at` | `TIMESTAMPTZ` | Required |
| `updated_at` | `TIMESTAMPTZ` | Required |
| `last_synced_at` | `TIMESTAMPTZ` | Required |

Indexes support active-list lookup, display-name search, provider search, and
model-slug search.

### 8.2 `llm_model_price_snapshots`

Stores immutable price history. A row is inserted only when the normalized or
raw price representation changes.

| Column | Type | Rules |
| --- | --- | --- |
| `id` | `SERIAL` | Primary key |
| `provider` | `TEXT` | Model foreign-key component |
| `model_id` | `TEXT` | Model foreign-key component |
| `prompt_usd_per_token` | `NUMERIC(24,12)` | Required, non-negative |
| `completion_usd_per_token` | `NUMERIC(24,12)` | Required, non-negative |
| `request_usd` | `NUMERIC(24,12)` | Nullable |
| `image_usd` | `NUMERIC(24,12)` | Nullable |
| `cache_read_usd_per_token` | `NUMERIC(24,12)` | Nullable |
| `cache_write_usd_per_token` | `NUMERIC(24,12)` | Nullable |
| `raw_pricing` | `JSONB` | Required |
| `pricing_hash` | `TEXT` | Required; change detector |
| `effective_from` | `TIMESTAMPTZ` | Required |
| `effective_to` | `TIMESTAMPTZ` | Nullable; one open row per model |

### 8.3 `llm_model_sync_runs`

Records catalog synchronization observability.

| Column | Type |
| --- | --- |
| `id` | `SERIAL` |
| `status` | `TEXT` (`running`, `completed`, `error`) |
| `models_received` | `INTEGER` |
| `models_activated` | `INTEGER` |
| `models_deactivated` | `INTEGER` |
| `models_skipped` | `INTEGER` |
| `error_message` | `TEXT` nullable |
| `started_at` | `TIMESTAMPTZ` |
| `completed_at` | `TIMESTAMPTZ` nullable |

### 8.4 `chat_sessions` Changes

Add:

| Column | Type | Behavior |
| --- | --- | --- |
| `llm_provider` | `TEXT` | Required; defaults to `openrouter` for new chats |
| `model` | Existing `TEXT` | Stores full OpenRouter model slug |

Migration behavior:

- Sessions with null model or `gemini-3-flash-preview` become
  `provider=openrouter`, `model=google/gemini-3-flash-preview`.
- Historical `llm_calls.provider=gemini` rows remain unchanged.
- No historical cost is fabricated for existing Gemini calls.

### 8.5 `llm_calls` Changes

Add:

| Column | Type | Behavior |
| --- | --- | --- |
| `generation_id` | `TEXT` | Nullable provider generation ID |
| `cached_input_tokens` | `INTEGER` | Required default `0` |
| `cache_write_tokens` | `INTEGER` | Required default `0` |
| `billed_cost_usd` | `NUMERIC(20,10)` | Nullable actual/fallback cost |
| `upstream_cost_usd` | `NUMERIC(20,10)` | Nullable provider upstream cost |
| `cost_source` | `TEXT` | Nullable `provider` or `estimated` |
| `price_snapshot_id` | `INTEGER` | Nullable audit reference |

The numeric precision permits costs far below one cent without floating-point
drift. Application DTOs serialize these decimals as strings and format them at
the UI boundary.

## 9. Backend API Contracts

### 9.1 List Models

`GET /llm/models`

Authenticated response:

```json
{
  "defaultModelId": "google/gemini-3-flash-preview",
  "lastSyncedAt": "2026-07-21T10:00:00.000Z",
  "stale": false,
  "models": [
    {
      "provider": "openrouter",
      "id": "google/gemini-3-flash-preview",
      "name": "Google: Gemini 3 Flash Preview",
      "description": "...",
      "contextLength": 1000000,
      "maxCompletionTokens": 65536,
      "inputModalities": ["text"],
      "outputModalities": ["text"],
      "supportsReasoning": true,
      "reasoning": {},
      "promptUsdPerMillion": "0.000000",
      "completionUsdPerMillion": "0.000000",
      "requestUsd": "0.000000",
      "pricingUpdatedAt": "2026-07-21T10:00:00.000Z"
    }
  ]
}
```

The numbers in this example are illustrative values, not asserted model
prices. Production values always come from the synchronized catalog.

### 9.2 Change Session Model

`PATCH /chat-sessions/{id}/model`

Request:

```json
{
  "provider": "openrouter",
  "modelId": "anthropic/claude-sonnet-4.6"
}
```

Validation:

- The user owns the session.
- Provider is supported.
- Model exists and is active.
- Model remains structured-output compatible.

The endpoint returns the updated safe session. Invalid/inactive models return a
`400` with a stable error code. The existing message-send body no longer
selects a model; the server uses session state.

### 9.3 Manual Catalog Refresh

`POST /admin/llm-models/sync`

- Admin-only.
- Returns `202` with sync-run ID.
- Does not launch a second job if the advisory lock is held.
- Admin status endpoints expose the sanitized result, never the API key or
  request authorization headers.

## 10. Frontend Design

### 10.1 Placement

The searchable model combobox is placed in the ChatInput toolbar, left of the
Send/Stop controls. It remains visible when the input is disabled so the user
can inspect the selected model, but changing it requires an owned active chat.

### 10.2 Closed State

The closed control displays:

```text
Gemini 3 Flash Preview · $X.XXXXXX in / $Y.YYYYYY out
```

The compact layout may collapse pricing to the dropdown details on narrow
screens, but the model name always remains visible.

### 10.3 Open State

The menu follows the supplied visual reference: a large searchable list above
the field with keyboard navigation and internal scrolling.

Each result contains:

- Human-readable model name.
- Full `provider/model` slug.
- `$… / 1M input` and `$… / 1M output`.
- Context length.
- `Reasoning` and `Structured` badges when applicable.

Ordering:

1. Configured default Gemini Flash, labeled `Default`.
2. Exact search matches.
3. Remaining models alphabetically by provider and display name.

Search matches display name, model slug, and model author/provider. The menu
supports arrow keys, Enter, Escape, focus restoration, and screen-reader labels.

### 10.4 Persistence and Inactive Models

- A successful selection calls the session-model endpoint immediately.
- The session list and selected chat state are updated without a page refresh.
- Active requests keep their start-time model snapshot.
- If the selected model becomes inactive, historical messages retain it.
- On returning to an inactive-model session, the UI shows `Model no longer
  available` and switches future messages to the configured default Gemini
  Flash after the backend validates that default as active.
- No different model is chosen silently because it is cheaper or faster.

### 10.5 Cost Display

Use `Intl.NumberFormat('en-US', {style: 'currency', currency: 'USD'})` with a
minimum of two and maximum of six fractional digits.

Examples:

```text
$0.000123
$0.084321
$12.3456
```

Message footer:

```text
CLAUDE SONNET · 12,480 TOKENS · 18.2S · 4 CALLS · $0.084321
```

If any call was estimated, the footer includes `Estimated cost`. Provider costs
and estimated costs are never visually conflated.

Dashboard additions:

- Total actual USD cost.
- Estimated-cost subtotal.
- Model/provider cost breakdown.
- Calls, input tokens, output tokens, reasoning tokens, cache tokens, and cost.
- Catalog last-updated indicator.

## 11. Error and Retry Policy

OpenRouter errors are normalized into typed internal errors with HTTP/provider
code, stable `error_type`, sanitized message, and retryability.

| Error | Behavior |
| --- | --- |
| `400` invalid request/schema/context | No generic retry; surface actionable error |
| `401` authentication | No retry; fail readiness/admin diagnostics |
| `402` insufficient credits | No retry; admin diagnostic and safe user message |
| `403` permission/guardrail | No retry |
| `408` timeout | Retry within current bounded policy |
| `429` rate limit | Honor `Retry-After`, then bounded retry |
| `502` provider unavailable | Allow OpenRouter pre-stream provider fallback |
| `503` overloaded/no provider | Honor `Retry-After`, then bounded retry |
| Mid-stream in-band error | Preserve partial output; do not switch model silently |

The adapter reads mid-stream error events even though the HTTP response is
already `200`. Provider error details are persisted to the existing admin-only
chat error field. The normal user sees a safe message and may Continue or
Regenerate according to the existing chat behavior.

Reference: [OpenRouter errors and debugging](https://openrouter.ai/docs/api/reference/errors-and-debugging)

## 12. Security and Privacy

- API keys exist only in managed deployment secrets and process environments.
- Authorization headers are never logged.
- Provider response bodies are sanitized before entering admin error storage.
- Frontend model endpoints expose metadata and prices only.
- Model IDs from the client are validated against the server catalog.
- Chat ownership checks apply to session-model changes.
- Admin sync endpoints require the existing admin role.
- OpenRouter debug echo is disabled in production because it may include prompt
  content.
- All selected providers receive the prompt and retrieved customs evidence
  needed for inference. Enabling every compatible model therefore accepts the
  retention policies allowed by the shared OpenRouter account. Organization
  privacy requirements must be enforced through OpenRouter account preferences
  or `OPENROUTER_REQUIRE_ZDR=true`.

## 13. Observability

Required structured logs and metrics:

- Catalog sync duration, success/failure, received/active/skipped counts.
- LLM provider/model, purpose, duration, token categories, billed cost, and
  generation ID.
- Retry count and typed provider error.
- Percentage of calls with provider cost versus estimated fallback.
- Cost totals by user, session, model, provider, and day through existing
  analytics authorization boundaries.

Never include task text, retrieved evidence, API keys, or authorization headers
in routine cost/model logs.

## 14. Testing Strategy

### 14.1 Core Unit Tests

- Provider factory selects OpenRouter and preserves direct Gemini rollback.
- Role and system-prompt conversion.
- Pydantic JSON Schema conversion.
- Structured response validation and one corrective retry.
- Reasoning-level mapping and omission for unsupported models.
- Streaming chunk parsing, keepalive handling, final usage capture.
- Reasoning tokens are split from completion tokens without increasing the
  provider-reported total.
- Mid-stream error detection with HTTP `200`.
- `Retry-After` handling and retry ceilings.
- Unicode and HTML-entity behavior remains unchanged.

### 14.2 Backend Unit Tests

- Compatibility filter includes all and only eligible models.
- Invalid catalog rows are skipped without emptying the catalog.
- Price hash detects real price changes.
- Unchanged price does not create a duplicate snapshot.
- Failed/empty sync preserves last-known-good models.
- First boot without a usable key or catalog reports degraded readiness and
  returns `503` from the model catalog endpoint.
- Missing models deactivate only after a successful non-empty sync.
- Advisory lock prevents concurrent replica sync.
- Session model endpoint validates ownership and active status.
- Provider cost is stored exactly as a decimal.
- Estimated fallback references the correct price snapshot.
- Historical calls are unchanged by later price updates.

### 14.3 Frontend Tests

- Model search by name, slug, and provider.
- Default model ordering and label.
- Keyboard navigation and accessible combobox roles.
- Selection persists across session reload.
- Selection during an active run affects only the next message.
- Inactive model warning/default transition.
- USD formatting from very small per-call cost through aggregate totals.
- Provider and estimated cost labels remain distinct.

### 14.4 Integration Tests

A fake OpenRouter HTTP/SSE server covers:

- Catalog synchronization.
- Structured planning response.
- Streamed final answer.
- Native usage and billed-cost persistence.
- Pre-stream HTTP errors.
- Mid-stream in-band errors.
- Rate-limit retry with `Retry-After`.

CI tests do not require a real OpenRouter key. A real-provider smoke test is
opt-in and skipped unless explicitly enabled with secret credentials.

### 14.5 Accuracy and Cost Regression

Run the existing fixed customs benchmark using:

1. Direct Gemini baseline.
2. OpenRouter Gemini Flash with the same prompts, retrieval tools, thinking
   levels, temperature, and questions.

The OpenRouter path must preserve every existing factual/citation accuracy
assertion. Report actual calls, token categories, elapsed time, and USD cost.
Differences caused by provider tokenizer accounting are recorded rather than
normalized away.

## 15. Acceptance Criteria

The feature is complete only when all conditions hold:

1. No OpenRouter key is present in source, database, browser traffic, or logs.
2. The backend exposes every current compatible model from the last successful
   catalog.
3. A new compatible model appears without a code deployment after sync.
4. A removed model cannot be selected for a new request, while its history
   remains readable.
5. New chats start with configured Gemini Flash.
6. A user's model selection persists until changed.
7. The running request uses its start-time model even if the session selection
   changes.
8. Structured research actions and streamed final answers work across the
   catalog contract.
9. Every OpenRouter call records provider, resolved model, generation ID,
   native token categories, duration, and actual billed USD cost when supplied.
10. Catalog price changes create history instead of rewriting old cost.
11. USD values display with `$` and up to six fractional digits.
12. Dashboard totals equal the sum of persisted call costs without binary
    floating-point drift.
13. Catalog outages use last-known-good data.
14. Typed OpenRouter errors remain visible to admins and safe for users.
15. Existing customs accuracy and citation benchmark assertions pass.
16. Reasoning tokens appear as their own category without being counted twice
    in total tokens or estimated completion cost.
17. A deployment with no usable OpenRouter key or last-known-good catalog fails
    readiness clearly and cannot start an OpenRouter chat run.

## 16. Rollout Sequence

1. Add database tables/columns and deploy backward-compatible backend models.
2. Add catalog sync in disabled/read-only mode and verify model/price data.
3. Add OpenRouter core adapter behind `FS_EXPLORER_LLM_PROVIDER`.
4. Run direct-Gemini versus OpenRouter-Gemini accuracy/cost benchmark.
5. Add backend catalog/session endpoints.
6. Add frontend selector and cost UI.
7. Configure `OPENROUTER_API_KEY` and default model in dev secrets.
8. Enable OpenRouter in dev and run end-to-end acceptance tests.
9. Promote migrations and secrets through test and production environments.

Rollback is performed by setting `FS_EXPLORER_LLM_PROVIDER=gemini`; historical
OpenRouter catalog and cost data remain intact. Database down migrations are not
part of normal rollback because dropping cost history would destroy audit data.

## 17. Out of Scope for the First Release

- Per-user OpenRouter keys.
- User-created model allowlists or deny lists.
- User budgets, quotas, approval limits, or role-based model access.
- Automatic model choice based on price, latency, or quality.
- Silent fallback to a different model slug.
- TRY currency conversion or exchange-rate history.
- Vision/file/audio inputs in chat.
- OpenRouter BYOK provider-key management.
- Reconstructing actual historical cost for pre-OpenRouter Gemini calls.
