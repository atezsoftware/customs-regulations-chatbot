"""Tests for the FsExplorerAgent class."""

import pytest
import os

from unittest.mock import Mock, patch
from google.genai import Client as GenAIClient

from fs_explorer_api.agent import (
    GEMINI_MAX_CONTEXT_TOKENS,
    DEFAULT_EFFORT,
    FsExplorerAgent,
    IndexedCorpus,
    PlannedSearchResult,
    RetrievalStats,
    SYSTEM_PROMPT,
    FINAL_SYSTEM_PROMPT,
    TokenUsage,
    _build_system_prompt,
    _chunk_context_from_storage,
    set_search_flags,
    get_search_flags,
    set_effort,
    get_effort,
    clear_index_context,
)
from fs_explorer_api.llm import LLMUsage
from fs_explorer_api.models import (
    Action,
    ContextSummary,
    GoDeeperAction,
    RetrievalPlan,
    RetrievalQuery,
    StopAction,
)
from fs_explorer_api.search import SearchHit
from fs_explorer_api.search.ranker import RankedDocument
from .conftest import make_mock_llm_client


class TestAgentInitialization:
    """Tests for agent initialization."""

    @patch.dict(os.environ, {"GOOGLE_API_KEY": "test-api-key"})
    def test_agent_init_with_env_key(self) -> None:
        """Test agent initialization with API key from environment."""
        agent = FsExplorerAgent()
        assert isinstance(agent._llm.raw_client, GenAIClient)
        assert len(agent._chat_history) == 0  # No system prompt in history
        assert isinstance(agent.token_usage, TokenUsage)

    def test_agent_init_with_explicit_key(self) -> None:
        """Test agent initialization with explicit API key."""
        agent = FsExplorerAgent(api_key="explicit-test-key")
        assert isinstance(agent._llm.raw_client, GenAIClient)

    def test_agent_init_without_key_raises(self) -> None:
        """Test that initialization without Google credentials raises ValueError."""
        # Ensure no credentials in environment
        env = os.environ.copy()
        for name in (
            "GOOGLE_API_KEY",
            "GOOGLE_APPLICATION_CREDENTIALS",
            "GOOGLE_APPLICATION_CREDENTIALS_JSON",
            "GOOGLE_GENAI_USE_VERTEXAI",
        ):
            env.pop(name, None)

        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ValueError, match="Google GenAI credentials"):
                FsExplorerAgent()


class TestAgentConfiguration:
    """Tests for agent task configuration."""

    @patch.dict(os.environ, {"GOOGLE_API_KEY": "test-api-key"})
    def test_configure_task_adds_to_history(self) -> None:
        """Test that configure_task adds message to chat history."""
        agent = FsExplorerAgent()
        agent.configure_task("this is a task")

        assert len(agent._chat_history) == 1
        assert agent._chat_history[0].role == "user"
        assert agent._chat_history[0].text == "this is a task"

    @patch.dict(os.environ, {"GOOGLE_API_KEY": "test-api-key"})
    def test_multiple_configure_task_calls(self) -> None:
        """Test that multiple configure_task calls accumulate."""
        agent = FsExplorerAgent()
        agent.configure_task("task 1")
        agent.configure_task("task 2")

        assert len(agent._chat_history) == 2
        assert agent._chat_history[0].text == "task 1"
        assert agent._chat_history[1].text == "task 2"


class TestChunkContextRetrieval:
    def test_returns_target_and_bounded_neighbor_chunks(self) -> None:
        chunks = [
            {
                "id": f"chunk_{position}",
                "doc_id": "doc_1",
                "text": f"text {position}",
                "position": position,
                "start_char": position * 10,
                "end_char": position * 10 + 9,
                "chunk_type": "article",
                "metadata": {"article_no": str(position)},
            }
            for position in range(5)
        ]
        storage = _ChunkStorage(chunks)

        result = _chunk_context_from_storage(storage, "chunk_2", before=1, after=1)

        assert "chunk 1" in result
        assert "chunk 2" in result
        assert "chunk 3" in result
        assert "chunk 0" not in result
        assert "chunk 4" not in result

    def test_missing_chunk_returns_clear_message(self) -> None:
        storage = _ChunkStorage([])

        result = _chunk_context_from_storage(storage, "missing", before=1, after=1)

        assert result == "No indexed chunk found for chunk_id='missing'"


class _ChunkStorage:
    def __init__(self, chunks) -> None:
        self.chunks = chunks

    def get_chunk(self, *, chunk_id):
        return next((chunk for chunk in self.chunks if chunk["id"] == chunk_id), None)

    def list_document_chunks(self, *, doc_id):
        return [chunk for chunk in self.chunks if chunk["doc_id"] == doc_id]


class TestAgentActions:
    """Tests for agent action handling."""

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"GOOGLE_API_KEY": "test-api-key"})
    async def test_take_action_returns_action(self) -> None:
        """Test that take_action returns an action from the model."""
        agent = FsExplorerAgent(llm_client=make_mock_llm_client())
        agent.configure_task("this is a task")

        result = await agent.take_action()

        assert result is not None
        action, action_type = result
        assert isinstance(action, Action)
        assert isinstance(action.action, StopAction)
        assert action.action.final_result == "this is a final result"
        assert action.reason == "I am done"
        assert action_type == "stop"

    @patch.dict(os.environ, {"GOOGLE_API_KEY": "test-api-key"})
    def test_reset_clears_history(self) -> None:
        """Test that reset clears chat history and token usage."""
        agent = FsExplorerAgent()
        agent.configure_task("task 1")
        agent.token_usage.api_calls = 5

        agent.reset()

        assert len(agent._chat_history) == 0
        assert agent.token_usage.api_calls == 0


class TestTokenUsage:
    """Tests for TokenUsage tracking."""

    def test_add_api_call(self) -> None:
        """Test adding API call metrics."""
        usage = TokenUsage()
        usage.add_api_call(100, 50)

        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert usage.total_tokens == 150
        assert usage.api_calls == 1

    def test_context_usage_ratio_uses_last_call_not_cumulative_sum(self) -> None:
        """A multi-step run's prompt_tokens sums every call (correct for
        billing), but context_usage_ratio must reflect only the size of
        the most recent request — not that inflated running total, which
        for a several-step run can overshoot the real context usage by
        multiple times over."""
        usage = TokenUsage()
        usage.add_api_call(prompt_tokens=20_000, completion_tokens=100)
        usage.add_api_call(prompt_tokens=50_000, completion_tokens=100)
        usage.add_api_call(prompt_tokens=110_000, completion_tokens=100)

        assert usage.prompt_tokens == 180_000  # cumulative, for cost/billing
        assert usage.last_prompt_tokens == 110_000  # current history size

        ratio = usage.context_usage_ratio(GEMINI_MAX_CONTEXT_TOKENS)
        assert ratio == pytest.approx(110_000 / GEMINI_MAX_CONTEXT_TOKENS)
        assert ratio < 0.15

    def test_add_tool_result_parse_file(self) -> None:
        """Test tracking parse_file tool usage."""
        usage = TokenUsage()
        usage.add_tool_result("document content here", "parse_file")

        assert usage.documents_parsed == 1
        assert usage.tool_result_chars == len("document content here")

    def test_add_tool_result_scan_folder(self) -> None:
        """Test tracking scan_folder tool usage."""
        usage = TokenUsage()
        # Simulating scan output with document markers
        result = "│ [1/3] doc1.pdf\n│ [2/3] doc2.pdf\n│ [3/3] doc3.pdf"
        usage.add_tool_result(result, "scan_folder")

        assert usage.documents_scanned == 3

    def test_add_tool_result_counts_semantic_search_hits_as_retrieval(self) -> None:
        """`semantic_search`'s rendered hits are counted via the same
        `[idx] doc_id:` marker the tool itself already emits per hit."""
        usage = TokenUsage()
        result = (
            "=== INDEXED SEARCH RESULTS ===\n"
            "Query: test\n\n"
            "[1] doc_id: doc_a\n    title: a\n\n"
            "[2] doc_id: doc_b\n    title: b\n\n"
            "[3] doc_id: doc_c\n    title: c\n\n"
        )
        usage.add_tool_result(result, "semantic_search")

        assert usage.retrieval_chunks == 3
        assert usage.retrieval_chars == len(result)
        assert usage.retrieval_estimated_tokens() == (len(result) + 3) // 4

    def test_add_tool_result_counts_chunk_headers_as_retrieval(self) -> None:
        """Deep reads and full-chunk grep share the same chunk marker."""
        usage = TokenUsage()
        result = (
            "=== INDEXED DOCUMENT FROM CHUNKS ===\n"
            "doc_id: doc_a\n\n"
            "--- chunk 0 (text, chars 0-10) ---\nfirst\n"
            "--- chunk 1 (text, chars 10-20) ---\nsecond\n"
        )
        for tool_name in ("get_document", "parse_file", "read", "grep"):
            usage = TokenUsage()
            usage.add_tool_result(result, tool_name)
            assert usage.retrieval_chunks == 2, tool_name
            assert usage.retrieval_chars == len(result), tool_name

    def test_add_tool_result_excludes_preview_file_from_retrieval(self) -> None:
        """`preview_file` shares `_document_from_chunks`'s rendering with
        `get_document`/`parse_file`/`read` but is deliberately excluded —
        it's a fixed, cheap, truncated peek, not real retrieval breadth."""
        usage = TokenUsage()
        result = "--- chunk 0 (text, chars 0-10) ---\nfirst\n"
        usage.add_tool_result(result, "preview_file")

        assert usage.retrieval_chunks == 0
        assert usage.retrieval_chars == 0

    def test_add_tool_result_excludes_non_chunk_tools_from_retrieval(self) -> None:
        usage = TokenUsage()
        usage.add_tool_result("some glob output", "glob")

        assert usage.retrieval_chunks == 0
        assert usage.retrieval_chars == 0

    def test_summary_format(self) -> None:
        """Test that summary produces formatted output."""
        usage = TokenUsage()
        usage.add_api_call(1000, 500)

        summary = usage.summary()

        assert "TOKEN USAGE SUMMARY" in summary
        assert "1,000" in summary  # Formatted prompt tokens
        assert "API Calls:" in summary
        assert "Est. Cost" in summary


class TestSystemPrompt:
    """Tests for system prompt configuration."""

    def test_system_prompt_contains_tools(self) -> None:
        """Test that system prompt documents all tools."""
        assert "scan_folder" in SYSTEM_PROMPT
        assert "preview_file" in SYSTEM_PROMPT
        assert "parse_file" in SYSTEM_PROMPT
        assert "read" in SYSTEM_PROMPT
        assert "grep" in SYSTEM_PROMPT
        assert "glob" in SYSTEM_PROMPT

    def test_system_prompt_contains_strategy(self) -> None:
        """Test that system prompt includes exploration strategy."""
        assert "Strategy" in SYSTEM_PROMPT
        assert "semantic_search" in SYSTEM_PROMPT
        assert "cross-reference" in SYSTEM_PROMPT

    def test_system_prompt_contains_index_tools(self) -> None:
        """Test that system prompt documents index-aware tools."""
        assert "semantic_search" in SYSTEM_PROMPT
        assert "get_document" in SYSTEM_PROMPT
        assert "list_indexed_documents" in SYSTEM_PROMPT
        assert "get_chunk_context" not in SYSTEM_PROMPT
        assert "complete selected chunks" in SYSTEM_PROMPT

    def test_action_and_final_prompts_are_purpose_specific(self) -> None:
        assert "Tools:" in SYSTEM_PROMPT
        assert "Sources section" not in SYSTEM_PROMPT
        assert "Available Tools" not in FINAL_SYSTEM_PROMPT
        assert "## Sources" in FINAL_SYSTEM_PROMPT
        assert len(SYSTEM_PROMPT) < 4000


class TestPurposeSpecificThinking:
    @pytest.mark.asyncio
    async def test_action_uses_low_thinking(self) -> None:
        client = _PurposeCapturingClient()
        agent = FsExplorerAgent(llm_client=client)
        agent.configure_task("test")

        await agent.take_action()

        assert client.structured_thinking_levels == ["low"]

    @pytest.mark.asyncio
    async def test_final_answer_uses_high_thinking(self) -> None:
        client = _PurposeCapturingClient()
        agent = FsExplorerAgent(llm_client=client)
        agent.configure_task("evidence")

        chunks = [chunk async for chunk in agent.stream_final_answer("fallback")]

        assert chunks == ["answer"]
        assert client.stream_thinking_levels == ["high"]


class _PurposeCapturingClient:
    model = "test"

    def __init__(self) -> None:
        self.structured_thinking_levels: list[str | None] = []
        self.stream_thinking_levels: list[str | None] = []

    async def generate_structured(
        self, history, system_prompt, schema, *, thinking_level=None
    ):
        self.structured_thinking_levels.append(thinking_level)
        return Action(reason="done", action=StopAction(final_result="done")), LLMUsage()

    async def stream_text(self, history, system_prompt, *, thinking_level=None):
        self.stream_thinking_levels.append(thinking_level)
        yield "answer"

    def last_stream_usage(self):
        return None


class TestSearchFlags:
    """Tests for search flag state and dynamic system prompt."""

    def setup_method(self) -> None:
        clear_index_context()

    def teardown_method(self) -> None:
        clear_index_context()

    def test_set_and_get_search_flags(self) -> None:
        assert get_search_flags() == (False, False)
        set_search_flags(enable_semantic=True, enable_metadata=False)
        assert get_search_flags() == (True, False)
        set_search_flags(enable_semantic=False, enable_metadata=False)
        assert get_search_flags() == (False, False)

    def test_clear_index_context_resets_flags(self) -> None:
        set_search_flags(enable_semantic=True, enable_metadata=True)
        clear_index_context()
        assert get_search_flags() == (False, False)

    def test_build_system_prompt_no_index(self) -> None:
        prompt = _build_system_prompt(False, False)
        assert prompt == SYSTEM_PROMPT

    def test_build_system_prompt_semantic_only(self) -> None:
        prompt = _build_system_prompt(True, False)
        assert "Semantic Only" in prompt
        assert "WITHOUT the `filters`" in prompt

    def test_build_system_prompt_metadata_only(self) -> None:
        prompt = _build_system_prompt(False, True)
        assert "Metadata Only" in prompt
        assert "metadata filtering" in prompt

    def test_build_system_prompt_both(self) -> None:
        prompt = _build_system_prompt(True, True)
        assert "Semantic + Metadata" in prompt

    @patch.dict(os.environ, {"GOOGLE_API_KEY": "test-api-key"})
    def test_all_tools_always_available(self) -> None:
        """Filesystem and indexed tools are never blocked."""
        set_search_flags(enable_semantic=False, enable_metadata=False)
        agent = FsExplorerAgent()
        agent.configure_task("test")
        agent.call_tool("glob", {"directory": "/tmp", "pattern": "*.md"})

        last = agent._chat_history[-1]
        assert "not available" not in last.text


class _ScriptedLLMClient:
    """LLMClient whose `generate_structured` responses are scripted per call.

    `action_token_counts` gives the reported `input_tokens` for each
    non-summary ("action") call, in order; the last one always resolves to
    a StopAction (so a test-driven exploration loop terminates on its own).
    Any call using the `ContextSummary` schema is intercepted separately
    and does not consume from that list, mirroring how
    `_maybe_summarize_history` issues an extra, distinct call.
    """

    def __init__(self, action_token_counts: list[int]) -> None:
        self._action_token_counts = list(action_token_counts)
        self.summary_calls = 0

    async def generate_structured(
        self, history, system_prompt, schema, *, thinking_level=None
    ):
        if schema is ContextSummary:
            self.summary_calls += 1
            return ContextSummary(summary="compact summary"), LLMUsage(
                input_tokens=500, output_tokens=50
            )

        tokens = self._action_token_counts.pop(0)
        is_last = not self._action_token_counts
        action = Action(
            reason="done" if is_last else "continuing",
            action=StopAction(final_result="done")
            if is_last
            else GoDeeperAction(directory="."),
        )
        return action, LLMUsage(input_tokens=tokens, output_tokens=10)

    async def stream_text(self, history, system_prompt, *, thinking_level=None):
        return
        yield ""  # pragma: no cover - makes this an async generator

    def last_stream_usage(self):
        return None


class TestContextSummarization:
    """Tests for mid-run chat history compaction (`_maybe_summarize_history`)."""

    @pytest.mark.asyncio
    @patch("fs_explorer_api.agent._MAX_STEPS", 100)
    async def test_triggers_above_threshold_and_shrinks_history(self) -> None:
        # 8 small calls to build up history, then one that crosses 85% of
        # the context ceiling and should trigger a compaction. Step budget
        # patched well above 9 so this run of the step-count guard doesn't
        # interfere with what this test is actually exercising.
        over_threshold = int(GEMINI_MAX_CONTEXT_TOKENS * 0.9)
        client = _ScriptedLLMClient([1000] * 8 + [over_threshold])
        agent = FsExplorerAgent(llm_client=client)

        for i in range(9):
            agent.configure_task(f"step {i}")
            await agent.take_action()

        assert client.summary_calls == 1
        assert agent.token_usage.context_summaries == 1
        # leading task turn + summary + token-budgeted recent turns
        assert len(agent._chat_history) <= 5
        assert agent._chat_history[0].text == "step 0"
        assert "compact summary" in agent._chat_history[1].text

    @pytest.mark.asyncio
    async def test_large_tool_result_is_summarized_instead_of_kept_recent(self) -> None:
        client = _ScriptedLLMClient([5000])
        agent = FsExplorerAgent(llm_client=client)
        agent.configure_task("original task")
        agent.configure_task("Tool result:\n" + "evidence " * 3000)
        agent.configure_task("choose next action")

        await agent.take_action()

        assert client.summary_calls == 1
        assert all("evidence " * 100 not in turn.text for turn in agent._chat_history)
        assert "compact summary" in agent._chat_history[1].text

    @pytest.mark.asyncio
    async def test_compacted_history_is_not_summarized_again_without_new_evidence(
        self,
    ) -> None:
        client = _ScriptedLLMClient([5000])
        agent = FsExplorerAgent(llm_client=client)
        agent.configure_task("original task")
        for i in range(6):
            agent.configure_task(f"evidence {i} " * 300)

        await agent.take_action()
        await agent._maybe_summarize_history(5000)

        assert client.summary_calls == 1

    @pytest.mark.asyncio
    @patch("fs_explorer_api.agent._MAX_STEPS", 100)
    async def test_does_not_trigger_below_threshold(self) -> None:
        client = _ScriptedLLMClient([1000] * 9)
        agent = FsExplorerAgent(llm_client=client)

        for i in range(9):
            agent.configure_task(f"step {i}")
            await agent.take_action()

        assert client.summary_calls == 0
        assert agent.token_usage.context_summaries == 0
        assert len(agent._chat_history) == 18

    @pytest.mark.asyncio
    async def test_no_trigger_when_history_too_short(self) -> None:
        # Crosses the ratio on the very first call, but there's nothing
        # worth compacting yet (history shorter than leading+recent).
        over_threshold = int(GEMINI_MAX_CONTEXT_TOKENS * 0.9)
        client = _ScriptedLLMClient([over_threshold])
        agent = FsExplorerAgent(llm_client=client)

        agent.configure_task("only step")
        await agent.take_action()

        assert client.summary_calls == 0


class TestMaxSteps:
    """Tests for the `_MAX_STEPS` hard step budget in `take_action()`."""

    @pytest.mark.asyncio
    @patch("fs_explorer_api.agent._MAX_STEPS", 3)
    async def test_forces_stop_after_budget_without_extra_llm_call(self) -> None:
        # Every scripted action is a GoDeeperAction (never resolves itself
        # to a stop) so the *only* way this loop terminates is the budget.
        client = _ScriptedLLMClient([100] * 10)
        agent = FsExplorerAgent(llm_client=client)

        results = []
        for i in range(5):
            agent.configure_task(f"step {i}")
            results.append(await agent.take_action())

        # First 3 calls really hit the LLM; the 4th and 5th are forced
        # stops that must not consume any more scripted responses.
        assert len(client._action_token_counts) == 10 - 3
        assert results[3][1] == "stop"
        assert results[4][1] == "stop"
        assert "step budget" in results[3][0].reason

    @pytest.mark.asyncio
    @patch("fs_explorer_api.agent._MAX_STEPS", 2)
    async def test_forced_stop_still_produces_valid_history_turn(self) -> None:
        client = _ScriptedLLMClient([100] * 5)
        agent = FsExplorerAgent(llm_client=client)

        agent.configure_task("step 0")
        await agent.take_action()
        agent.configure_task("step 1")
        await agent.take_action()
        agent.configure_task("step 2")
        action, action_type = await agent.take_action()

        assert action_type == "stop"
        assert agent._chat_history[-1].role == "model"

    @pytest.mark.asyncio
    @patch("fs_explorer_api.agent._MAX_STEPS", 2)
    async def test_forced_stop_sets_forced_stop_flag(self) -> None:
        client = _ScriptedLLMClient([100] * 5)
        agent = FsExplorerAgent(llm_client=client)

        agent.configure_task("step 0")
        await agent.take_action()
        agent.configure_task("step 1")
        await agent.take_action()
        agent.configure_task("step 2")
        await agent.take_action()

        assert agent.forced_stop is True

    @pytest.mark.asyncio
    @patch("fs_explorer_api.agent._MAX_STEPS", 2)
    async def test_grant_more_steps_lets_a_resumed_run_actually_continue(self) -> None:
        # Regression test: resuming a forced-stopped agent must not
        # immediately re-trigger the exact same forced stop on the very
        # next take_action() call — grant_more_steps() (called by
        # server.py's resume path) must raise the ceiling for real.
        client = _ScriptedLLMClient([100] * 10)
        agent = FsExplorerAgent(llm_client=client)

        agent.configure_task("step 0")
        await agent.take_action()
        agent.configure_task("step 1")
        await agent.take_action()
        agent.configure_task("step 2")
        _forced_action, forced_type = await agent.take_action()
        assert forced_type == "stop"
        assert agent.forced_stop is True

        agent.grant_more_steps()
        assert agent.forced_stop is False

        agent.configure_task("step 3 (resumed)")
        _action, action_type = await agent.take_action()

        # A real scripted action was returned (there were 10 queued, only 3
        # consumed before the forced stop) — not another forced stop.
        assert action_type == "godeeper"
        assert agent.forced_stop is False


class TestDuplicateCallGuard:
    """Tests for the near-duplicate tool-call short-circuit in `call_tool()`."""

    def test_exact_duplicate_query_is_skipped(self) -> None:
        from fs_explorer_api.agent import TOOLS

        calls = []
        original = TOOLS["semantic_search"]
        TOOLS["semantic_search"] = lambda **kwargs: (
            calls.append(kwargs) or "real result"
        )
        try:
            agent = FsExplorerAgent(llm_client=make_mock_llm_client())
            agent.call_tool("semantic_search", {"query": "TIR karnesi ekstre teminat"})
            agent.call_tool("semantic_search", {"query": "TIR karnesi ekstre teminat"})
        finally:
            TOOLS["semantic_search"] = original

        assert len(calls) == 1  # second call never reached the real tool
        assert "SKIPPED" in agent._chat_history[-1].text

    def test_near_duplicate_reworded_query_is_skipped(self) -> None:
        from fs_explorer_api.agent import TOOLS

        calls = []
        original = TOOLS["semantic_search"]
        TOOLS["semantic_search"] = lambda **kwargs: (
            calls.append(kwargs) or "real result"
        )
        try:
            agent = FsExplorerAgent(llm_client=make_mock_llm_client())
            agent.call_tool(
                "semantic_search",
                {
                    "query": "TIR karnesi ekstre teminat hassas eşya yüksek riskli eşya listesi"
                },
            )
            agent.call_tool(
                "semantic_search",
                {
                    "query": "TIR karnesi kapsamında ek teminat veya hassas eşya listesi yüksek riskli eşyalar"
                },
            )
        finally:
            TOOLS["semantic_search"] = original

        assert len(calls) == 1

    def test_genuinely_different_query_is_not_skipped(self) -> None:
        from fs_explorer_api.agent import TOOLS

        calls = []
        original = TOOLS["semantic_search"]
        TOOLS["semantic_search"] = lambda **kwargs: (
            calls.append(kwargs) or "real result"
        )
        try:
            agent = FsExplorerAgent(llm_client=make_mock_llm_client())
            agent.call_tool("semantic_search", {"query": "TIR karnesi ekstra teminat"})
            agent.call_tool("semantic_search", {"query": "gümrük vergisi iade süresi"})
        finally:
            TOOLS["semantic_search"] = original

        assert len(calls) == 2

    def test_same_document_via_different_tool_name_is_skipped(self) -> None:
        """parse_file/get_document/read all fetch the same underlying
        document content — a repeat via a different tool name must still
        count as a duplicate, not get a fresh "first one's free"."""
        from fs_explorer_api.agent import TOOLS

        calls = []
        originals = {name: TOOLS[name] for name in ("parse_file", "get_document")}
        for name in originals:
            TOOLS[name] = lambda **kwargs: calls.append(kwargs) or "doc text"
        try:
            agent = FsExplorerAgent(llm_client=make_mock_llm_client())
            agent.call_tool("parse_file", {"file_path": "doc_abc123"})
            agent.call_tool("get_document", {"doc_id": "doc_abc123"})
        finally:
            TOOLS.update(originals)

        assert len(calls) == 1
        assert "SKIPPED" in agent._chat_history[-1].text


class _RecordingEngine:
    """Stand-in for `IndexedQueryEngine` that just records `.search()` kwargs
    instead of touching Postgres, so effort-driven limit/overfetch clamping
    can be asserted without a real index."""

    calls: list[dict] = []

    def __init__(self, storage, embedding_provider=None) -> None:
        pass

    def search(self, **kwargs):
        _RecordingEngine.calls.append(kwargs)
        return []


class TestEffortLevels:
    """Tests for the user-selectable effort level clamping `semantic_search`'s
    retrieval breadth — this is what makes `limit` a hard ceiling instead of
    a suggestion the model could previously set to any value."""

    def setup_method(self) -> None:
        clear_index_context()
        set_effort(DEFAULT_EFFORT)
        _RecordingEngine.calls = []

    def teardown_method(self) -> None:
        clear_index_context()
        set_effort(DEFAULT_EFFORT)

    def _run_semantic_search(self, *, limit=None) -> dict:
        import fs_explorer_api.agent as agent_module

        with (
            patch.object(
                agent_module,
                "_get_index_storage_and_corpora",
                return_value=(
                    Mock(),
                    [IndexedCorpus(root_folder="/x", corpus_id="c1")],
                    None,
                ),
            ),
            patch.object(agent_module, "IndexedQueryEngine", _RecordingEngine),
        ):
            agent_module.semantic_search("query", limit=limit)
        return _RecordingEngine.calls[-1]

    def test_default_effort_matches_low_baseline(self) -> None:
        # DEFAULT_EFFORT is "low", tuned to match the pre-effort behavior
        # exactly (limit=5, overfetch_multiplier=4) — never a regression
        # for callers that omit `effort` entirely.
        call = self._run_semantic_search()
        assert call["limit"] == 5
        assert call["overfetch_multiplier"] == 4

    def test_low_effort_caps_limit_even_when_model_asks_for_more(self) -> None:
        set_effort("low")
        call = self._run_semantic_search(limit=100)
        assert call["limit"] == 6  # low's max_limit — a hard ceiling
        assert call["overfetch_multiplier"] == 4

    def test_high_effort_default_limit_applies_when_model_omits_limit(self) -> None:
        set_effort("high")
        call = self._run_semantic_search(limit=None)
        assert call["limit"] == 12  # high's default_limit
        assert call["overfetch_multiplier"] == 6

    def test_model_supplied_limit_within_ceiling_is_honored(self) -> None:
        set_effort("high")
        call = self._run_semantic_search(limit=2)
        assert call["limit"] == 2

    def test_unknown_effort_falls_back_to_default(self) -> None:
        set_effort("bogus-value")
        assert get_effort() == DEFAULT_EFFORT


class TestFullChunkSearchResults:
    def test_semantic_search_renders_complete_selected_chunk(self) -> None:
        import fs_explorer_api.agent as agent_module

        full_text = "başlangıç " + ("uzun kanıt " * 80) + "TAM_CHUNK_SONU"
        hit = Mock(
            doc_id="doc_a",
            relative_path="a.md",
            absolute_path="/a.md",
            position=4,
            text=full_text,
            semantic_score=0.8,
            metadata_score=0,
            score=0.95,
            matched_by="semantic",
            chunk_id="chunk_a",
            chunk_type="text",
            metadata={"article_no": "54"},
        )

        class FullHitEngine:
            def __init__(self, storage, embedding_provider=None) -> None:
                pass

            def search(self, **kwargs):
                return [hit]

        storage = Mock()
        storage.get_active_schema.return_value = None
        with (
            patch.object(
                agent_module,
                "_get_index_storage_and_corpora",
                return_value=(
                    storage,
                    [IndexedCorpus(root_folder="/x", corpus_id="c1")],
                    None,
                ),
            ),
            patch.object(agent_module, "IndexedQueryEngine", FullHitEngine),
        ):
            result = agent_module.semantic_search("TIR karnesi")

        assert full_text in result
        assert "TAM_CHUNK_SONU" in result
        assert "excerpt:" not in result
        assert "--- chunk 4 (text)" in result
        assert "already includes its complete chunk text" in result

    def test_indexed_grep_reranks_and_renders_complete_chunks(self) -> None:
        import fs_explorer_api.agent as agent_module

        storage = Mock()
        storage.list_document_chunks.return_value = [
            {
                "id": f"chunk_{index}",
                "text": (
                    f"aranan ifade aday {index} "
                    + ("tam bağlam " * 30)
                    + f"GREP_CHUNK_SONU_{index}"
                ),
                "position": index,
                "chunk_type": "text",
                "metadata": {"article_no": str(index + 5)},
            }
            for index in range(2, 5)
        ]
        document = {
            "id": "doc_a",
            "relative_path": "a.md",
            "absolute_path": "/a.md",
        }
        captured: list[RankedDocument] = []

        def fake_rank(*, query, documents, limit):
            captured.extend(documents)
            return [(documents[-1], 0.91)]

        with (
            patch.object(agent_module, "_index_tools_available", return_value=True),
            patch.object(
                agent_module,
                "_get_index_storage_and_corpora",
                return_value=(
                    storage,
                    [IndexedCorpus(root_folder="/x", corpus_id="c1")],
                    None,
                ),
            ),
            patch.object(
                agent_module,
                "_all_index_documents",
                return_value=[document],
            ),
            patch.object(
                agent_module.IndexedQueryEngine,
                "rank_candidates",
                side_effect=fake_rank,
            ),
        ):
            result = agent_module._indexed_grep_file_content("all", "aranan ifade")

        assert len(captured) == 3
        assert [candidate.position for candidate in captured] == [2, 3, 4]
        assert all("GREP_CHUNK_SONU_" in candidate.text for candidate in captured)
        assert "GREP_CHUNK_SONU_4" in result
        assert "--- chunk 4 (text)" in result
        assert "complete chunk text" in result


class TestRetrievalHook:
    """Tests for `on_retrieval`/`RetrievalStats` — the retrieval-specific
    analog of `on_llm_call`, firing once per chunk-bearing tool call from
    both the sequential (`call_tool`) and concurrent-batch (`call_tools`)
    dispatch paths."""

    def _fake_semantic_search_result(self, hit_count: int) -> str:
        return "".join(f"[{i}] doc_id: doc_{i}\n" for i in range(1, hit_count + 1))

    def test_call_tool_fires_hook_for_chunk_bearing_tool(self) -> None:
        from fs_explorer_api.agent import TOOLS, RetrievalStats

        original = TOOLS["semantic_search"]
        TOOLS["semantic_search"] = lambda **kwargs: self._fake_semantic_search_result(3)
        received: list[RetrievalStats] = []
        try:
            agent = FsExplorerAgent(
                llm_client=make_mock_llm_client(), on_retrieval=received.append
            )
            agent._step_count = 5
            agent.call_tool("semantic_search", {"query": "test"})
        finally:
            TOOLS["semantic_search"] = original

        assert len(received) == 1
        assert received[0].step == 5
        assert received[0].tool_name == "semantic_search"
        assert received[0].chunk_count == 3

    def test_call_tool_fires_hook_even_for_zero_chunk_result(self) -> None:
        """A chunk-bearing tool that genuinely found nothing still reports
        a (chunk_count=0) event — server.py's WS layer relies on exactly
        one retrieval_stats event per chunk-bearing tool_call to correlate
        the two in order, including within a batch; silently skipping
        zero-chunk calls would desync every call after the first one."""
        from fs_explorer_api.agent import TOOLS, RetrievalStats

        original = TOOLS["get_document"]
        TOOLS["get_document"] = lambda **kwargs: "=== INDEXED DOCUMENT FROM CHUNKS ===\nno chunks here"
        received: list[RetrievalStats] = []
        try:
            agent = FsExplorerAgent(
                llm_client=make_mock_llm_client(), on_retrieval=received.append
            )
            agent.call_tool("get_document", {"doc_id": "doc_empty"})
        finally:
            TOOLS["get_document"] = original

        assert len(received) == 1
        assert received[0].chunk_count == 0

    def test_call_tool_does_not_fire_hook_for_non_chunk_tool(self) -> None:
        from fs_explorer_api.agent import TOOLS, RetrievalStats

        original = TOOLS["glob"]
        TOOLS["glob"] = lambda **kwargs: "some/path.txt"
        received: list[RetrievalStats] = []
        try:
            agent = FsExplorerAgent(
                llm_client=make_mock_llm_client(), on_retrieval=received.append
            )
            agent.call_tool("glob", {"directory": ".", "pattern": "*.txt"})
        finally:
            TOOLS["glob"] = original

        assert received == []

    @pytest.mark.asyncio
    async def test_call_tools_batch_fires_hook_for_each_chunk_bearing_call(
        self,
    ) -> None:
        from fs_explorer_api.agent import TOOLS, RetrievalStats

        original_search = TOOLS["semantic_search"]
        original_glob = TOOLS["glob"]
        TOOLS["semantic_search"] = lambda **kwargs: self._fake_semantic_search_result(2)
        TOOLS["glob"] = lambda **kwargs: "some/path.txt"
        received: list[RetrievalStats] = []
        try:
            agent = FsExplorerAgent(
                llm_client=make_mock_llm_client(), on_retrieval=received.append
            )
            agent._step_count = 2
            await agent.call_tools(
                [
                    ("semantic_search", {"query": "a"}),
                    ("glob", {"directory": ".", "pattern": "*.txt"}),
                ]
            )
        finally:
            TOOLS["semantic_search"] = original_search
            TOOLS["glob"] = original_glob

        # Only the chunk-bearing call in the batch reports retrieval stats —
        # `glob` produces no chunk markers, so it's silently skipped.
        assert len(received) == 1
        assert received[0].tool_name == "semantic_search"
        assert received[0].chunk_count == 2
        assert received[0].step == 2

    def test_set_retrieval_hook_rebinds_for_resume(self) -> None:
        from fs_explorer_api.agent import TOOLS, RetrievalStats

        original = TOOLS["semantic_search"]
        TOOLS["semantic_search"] = lambda **kwargs: self._fake_semantic_search_result(1)
        first_run: list[RetrievalStats] = []
        second_run: list[RetrievalStats] = []
        try:
            agent = FsExplorerAgent(
                llm_client=make_mock_llm_client(), on_retrieval=first_run.append
            )
            agent.set_retrieval_hook(second_run.append)
            agent.call_tool("semantic_search", {"query": "test"})
        finally:
            TOOLS["semantic_search"] = original

        assert first_run == []
        assert len(second_run) == 1


class _StatelessRetrievalClient:
    model = "test"

    def __init__(self) -> None:
        self.structured_histories = []
        self.stream_histories = []
        self._stream_usage = None

    async def generate_structured(
        self, history, system_prompt, schema, *, thinking_level=None
    ):
        self.structured_histories.append(list(history))
        assert schema is RetrievalPlan
        return (
            RetrievalPlan(
                searches=[
                    RetrievalQuery(query="ana kural"),
                    RetrievalQuery(query="istisnalar"),
                    RetrievalQuery(query="ana kural"),
                    RetrievalQuery(query="süre ve usul"),
                    RetrievalQuery(query="fazladan sorgu"),
                ]
            ),
            LLMUsage(input_tokens=500, output_tokens=80),
        )

    async def stream_text(self, history, system_prompt, *, thinking_level=None):
        self.stream_histories.append(list(history))
        self._stream_usage = LLMUsage(input_tokens=2400, output_tokens=300)
        yield "nihai cevap"

    def last_stream_usage(self):
        return self._stream_usage


def _search_hit(
    *,
    chunk_id: str,
    text: str,
    score: float,
    position: int,
) -> SearchHit:
    return SearchHit(
        doc_id=f"doc_{chunk_id}",
        relative_path=f"{chunk_id}.md",
        absolute_path=f"/{chunk_id}.md",
        position=position,
        text=text,
        semantic_score=score,
        metadata_score=0,
        score=score,
        matched_by="semantic",
        chunk_id=chunk_id,
        chunk_type="text",
        metadata={"article_no": str(position + 1)},
    )


class TestStatelessParallelRetrieval:
    @pytest.mark.asyncio
    async def test_uses_one_plan_and_one_final_call_without_search_history(
        self,
    ) -> None:
        import fs_explorer_api.agent as agent_module

        client = _StatelessRetrievalClient()
        retrieval_events: list[RetrievalStats] = []
        agent = FsExplorerAgent(
            llm_client=client, on_retrieval=retrieval_events.append
        )
        set_effort("low")

        calls = await agent.plan_indexed_retrieval("Transit süresi nedir?")
        assert [call.to_fn_args()["query"] for call in calls] == [
            "ana kural",
            "istisnalar",
            "süre ve usul",
        ]

        def fake_search(*, query, filters=None, as_of_date=None):
            common = _search_hit(
                chunk_id="common",
                text="ortak tam chunk",
                score=0.8,
                position=0,
            )
            unique = _search_hit(
                chunk_id=query.replace(" ", "_"),
                text=f"{query} için tam chunk",
                score=0.7,
                position=1,
            )
            return PlannedSearchResult(query=query, hits=[common, unique])

        captured_candidates: list[RankedDocument] = []

        def fake_global_rank(*, query, documents, limit):
            captured_candidates.extend(documents)
            return [
                (document, 0.9 - index * 0.1)
                for index, document in enumerate(documents[:limit])
            ]

        with (
            patch.object(
                agent_module, "_run_planned_index_search", side_effect=fake_search
            ),
            patch.object(
                agent_module.IndexedQueryEngine,
                "rank_candidates",
                side_effect=fake_global_rank,
            ),
        ):
            evidence = await agent.collect_parallel_indexed_evidence(
                "Transit süresi nedir?", calls
            )

        assert len(captured_candidates) == 4  # shared hit deduplicated
        assert evidence.count("ortak tam chunk") == 1
        assert len(agent._chat_history) == 2
        assert agent._chat_history[0].text == (
            "Original question:\nTransit süresi nedir?"
        )
        assert agent._chat_history[1].text == evidence
        assert len(retrieval_events) == 1
        assert retrieval_events[0].chunk_count == 4

        answer = "".join(
            [part async for part in agent.stream_final_answer("fallback")]
        )

        assert answer == "nihai cevap"
        assert agent.token_usage.api_calls == 2
        assert agent.token_usage.total_tokens == 3280
        assert len(client.structured_histories) == 1
        assert len(client.stream_histories) == 1
        assert len(client.stream_histories[0]) == 3  # question, evidence, final prompt
