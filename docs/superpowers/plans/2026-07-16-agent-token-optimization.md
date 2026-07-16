# Agent Token Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce repeated planning/context tokens while preserving accuracy and retrieval depth.

**Architecture:** Measure one stable benchmark before and after each isolated optimization. Separate planning from synthesis, compact by token budget, prefer relevant chunk neighborhoods, and batch independent tool calls while retaining full-document fallback.

**Tech Stack:** Python 3.10+, Pydantic, LlamaIndex Workflows, Google GenAI, pytest.

## Global Constraints

- Do not lower the 60-step safety ceiling.
- Keep `get_document` available as a fallback.
- Batch at most three tool calls.
- Every phase must pass the accuracy benchmark and existing API tests.

---

### Task 1: Baseline benchmark

**Files:**
- Create: `core/tests/api/test_token_optimization.py`
- Create: `core/scripts/benchmark_agent_tokens.py`

**Interfaces:**
- Produces: deterministic accuracy assertions and JSON token metrics.
- Consumes: `FsExplorerAgent`, `LLMClient`, and fixed synthetic tool results.

- [ ] Add a capturing LLM client that records system prompt and history payload sizes.
- [ ] Add direct-rule, exception, and cross-reference scenarios with required facts/citations.
- [ ] Run the benchmark on commit `17b64cc` and save the output outside tracked source.

### Task 2: Purpose-specific prompts and thinking

**Files:**
- Modify: `core/api/src/fs_explorer_api/agent.py`
- Modify: `core/api/src/fs_explorer_api/llm/base.py`
- Modify: `core/api/src/fs_explorer_api/llm/gemini.py`
- Test: `core/tests/api/test_agent.py`

**Interfaces:**
- Produces: purpose-specific `thinking_level` values and separate action/final prompts.
- Consumes: existing structured and streaming LLM calls.

- [ ] Write failing tests for compact action prompt and low/high thinking routing.
- [ ] Implement optional thinking level in the LLM protocol and Gemini config.
- [ ] Re-run accuracy and token benchmark.

### Task 3: Token-budgeted compaction

**Files:**
- Modify: `core/api/src/fs_explorer_api/agent.py`
- Modify: `core/api/src/fs_explorer_api/models.py`
- Test: `core/tests/api/test_agent.py`

**Interfaces:**
- Produces: bounded history retaining original task, evidence summary, and latest exchange.
- Consumes: per-call prompt token observations and `ContextSummary`.

- [ ] Write failing tests for large recent tool results and repeated-summary avoidance.
- [ ] Implement token-budgeted recent-turn selection and summary fingerprinting.
- [ ] Re-run accuracy and token benchmark.

### Task 4: Relevant chunk neighborhoods

**Files:**
- Modify: `core/api/src/fs_explorer_api/agent.py`
- Modify: `core/api/src/fs_explorer_api/models.py`
- Test: `core/tests/api/test_agent.py`

**Interfaces:**
- Produces: `get_chunk_context(chunk_id, before=1, after=1)`.
- Consumes: `PostgresStorage.get_chunk` and `list_document_chunks`.

- [ ] Write failing tests for bounded neighboring chunk selection and missing chunks.
- [ ] Implement the tool and update retrieval guidance.
- [ ] Re-run accuracy and token benchmark.

### Task 5: Bounded batch tool actions

**Files:**
- Modify: `core/api/src/fs_explorer_api/models.py`
- Modify: `core/api/src/fs_explorer_api/workflow.py`
- Modify: `core/api/src/fs_explorer_api/agent.py`
- Modify: `core/api/src/fs_explorer_api/server.py`
- Test: `core/tests/api/test_models.py`
- Test: `core/tests/api/test_workflow.py`

**Interfaces:**
- Produces: `ToolBatchAction` and `ToolBatchEvent`, limited to three calls.
- Consumes: existing tool registry, deduplication, trace, and stream event translation.

- [ ] Write failing schema/workflow tests for two calls and the size limit.
- [ ] Execute batch calls and append one combined result turn.
- [ ] Preserve individual research events and trace entries.
- [ ] Re-run accuracy and token benchmark.

### Task 6: Final comparison

**Files:**
- Update: `docs/superpowers/plans/2026-07-16-agent-token-optimization.md`

**Interfaces:**
- Produces: old/new accuracy and token comparison.
- Consumes: the benchmark and full API test suite.

- [ ] Run old and new benchmark with identical scenarios.
- [ ] Run API tests, frontend build, lint for changed files, and diff checks.
- [ ] Report exact call/token deltas and any limitations of mocked versus real-provider measurements.

## Verified Results

Deterministic benchmark (three scenarios, all accuracy assertions passed):

- Baseline `17b64cc`: 60,749 total tokens, 18 calls, 4 summaries.
- Optimized: 9,213 total tokens, 12 calls, 0 summaries.
- Reduction: 84.8%.

Real Gemini benchmark on Vertex `global`, using the same synthetic customs evidence and accuracy gates:

- Baseline `17b64cc` plus only the SDK awaitable-stream compatibility shim: 139,155 total tokens, 24 calls, 7 summaries, 3/3 accuracy.
- Optimized: 25,856 total tokens, 12 calls, 0 summaries, 3/3 accuracy.
- Reduction: 81.4% total tokens and 50% model calls, with no benchmark accuracy loss.
