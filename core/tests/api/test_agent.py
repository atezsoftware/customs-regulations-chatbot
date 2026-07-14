"""Tests for the FsExplorerAgent class."""

import pytest
import os

from unittest.mock import patch
from google.genai import Client as GenAIClient

from fs_explorer_api.agent import (
    GEMINI_MAX_CONTEXT_TOKENS,
    FsExplorerAgent,
    SYSTEM_PROMPT,
    TokenUsage,
    _build_system_prompt,
    set_search_flags,
    get_search_flags,
    clear_index_context,
)
from fs_explorer_api.llm import LLMUsage
from fs_explorer_api.models import Action, ContextSummary, GoDeeperAction, StopAction
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
        assert "Three-Phase" in SYSTEM_PROMPT or "PHASE" in SYSTEM_PROMPT
        assert "Parallel Scan" in SYSTEM_PROMPT or "PARALLEL" in SYSTEM_PROMPT
        assert "Backtracking" in SYSTEM_PROMPT or "BACKTRACK" in SYSTEM_PROMPT

    def test_system_prompt_contains_index_tools(self) -> None:
        """Test that system prompt documents index-aware tools."""
        assert "semantic_search" in SYSTEM_PROMPT
        assert "get_document" in SYSTEM_PROMPT
        assert "list_indexed_documents" in SYSTEM_PROMPT


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

    async def generate_structured(self, history, system_prompt, schema):
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

    async def stream_text(self, history, system_prompt):
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
        # leading task turn (1) + summary turn (1) + recent turns (4)
        assert len(agent._chat_history) == 6
        assert agent._chat_history[0].text == "step 0"
        assert "compact summary" in agent._chat_history[1].text

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


class TestDuplicateCallGuard:
    """Tests for the near-duplicate tool-call short-circuit in `call_tool()`."""

    def test_exact_duplicate_query_is_skipped(self) -> None:
        from fs_explorer_api.agent import TOOLS

        calls = []
        original = TOOLS["semantic_search"]
        TOOLS["semantic_search"] = lambda **kwargs: calls.append(kwargs) or "real result"
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
        TOOLS["semantic_search"] = lambda **kwargs: calls.append(kwargs) or "real result"
        try:
            agent = FsExplorerAgent(llm_client=make_mock_llm_client())
            agent.call_tool(
                "semantic_search",
                {"query": "TIR karnesi ekstre teminat hassas eşya yüksek riskli eşya listesi"},
            )
            agent.call_tool(
                "semantic_search",
                {"query": "TIR karnesi kapsamında ek teminat veya hassas eşya listesi yüksek riskli eşyalar"},
            )
        finally:
            TOOLS["semantic_search"] = original

        assert len(calls) == 1

    def test_genuinely_different_query_is_not_skipped(self) -> None:
        from fs_explorer_api.agent import TOOLS

        calls = []
        original = TOOLS["semantic_search"]
        TOOLS["semantic_search"] = lambda **kwargs: calls.append(kwargs) or "real result"
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
