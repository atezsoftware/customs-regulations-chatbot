# OpenRouter Request Compatibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every selectable OpenRouter model receive only portable request parameters and make rejections actionable.

**Architecture:** Keep compatibility logic inside `OpenRouterLLMClient`. Structured calls require the `response_format` capability at routing time; all OpenRouter calls omit optional reasoning controls. Error handling extracts only a bounded response error message.

**Tech Stack:** Python 3.10, httpx, pytest.

## Global Constraints

- Do not log or return API keys, authorization headers, or request bodies.
- Keep Gemini behavior unchanged.
- Preserve strict JSON schema for agent planning.

---

### Task 1: OpenRouter adapter compatibility

**Files:**
- Modify: `core/api/src/fs_explorer_api/llm/openrouter.py`
- Test: `core/tests/api/test_llm_openrouter.py`

**Interfaces:**
- Produces: structured OpenRouter requests with `provider.require_parameters=True` and no `reasoning` key.
- Produces: `OpenRouterError` text containing bounded upstream error detail and HTTP status for a rejected request.

- [ ] **Step 1: Write failing tests**

```python
assert "reasoning" not in seen
assert seen["provider"] == {"require_parameters": True}
assert "reasoning is not supported" in str(error.value)
```

- [ ] **Step 2: Run the focused test file and verify the new tests fail**

Run: `cd core && uv run --package fs-explorer-api pytest tests/api/test_llm_openrouter.py -q`

Expected: FAIL because the client currently includes `reasoning`, lacks provider requirements, and replaces upstream error text.

- [ ] **Step 3: Implement the minimal adapter changes**

```python
payload["provider"] = {"require_parameters": True}
# Do not send optional reasoning controls to heterogeneous selected models.
```

Extract `error.message` from the JSON response, truncate it to 300 characters, and use it with the status code in `OpenRouterError`.

- [ ] **Step 4: Re-run the focused tests**

Run: `cd core && uv run --package fs-explorer-api pytest tests/api/test_llm_openrouter.py -q`

Expected: PASS.

- [ ] **Step 5: Run API test coverage**

Run: `cd core && uv run --package fs-explorer-api pytest tests/api -q`

Expected: PASS.
