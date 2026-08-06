"""Tests for llm_loop.py, including history construction and empty-response paths."""

import json
import queue
import re
from typing import Any, cast
from unittest.mock import Mock, patch

import pytest

from onyx.chat.chat_state import ChatStateContainer
from onyx.chat.citation_processor import DynamicCitationProcessor
from onyx.chat.emitter import BufferedEmitter, Emitter
from onyx.chat.llm_loop import (
    _REFUSAL_FINISH_REASONS,
    EmptyLLMResponseError,
    SearchEvidenceLedgerEntry,
    _build_candidate_answer_evidence_chunks,
    _build_empty_llm_response_error,
    _commit_canonical_tool_decision_step,
    _compact_regulatory_search_history_for_reconsideration,
    _compact_repeated_search_results_for_history,
    _constrain_regulatory_tool_calls,
    _effective_regulatory_search_call_budget,
    _extract_llm_visible_search_results,
    _format_regulatory_tool_call_batch_feedback,
    _format_search_evidence_ledger,
    _hide_projected_tool_decision_output,
    _join_search_work_reminders,
    _merge_gathered_search_docs,
    _project_regulatory_history_for_tool_decision,
    _regulatory_llm_step_max_tokens,
    _regulatory_search_call_budget,
    _regulatory_search_chunk_cap,
    _try_fallback_tool_extraction,
    construct_message_history,
    run_llm_loop,
    select_reminder_text,
)
from onyx.chat.models import (
    ChatLoadedFile,
    ChatMessageSimple,
    ContextFileMetadata,
    ExtractedContextFiles,
    FileToolMetadata,
    LlmStepResult,
    ToolCallSimple,
)
from onyx.chat.staged_generation import commit_staged_llm_step
from onyx.configs.constants import DocumentSource, MessageType
from onyx.context.search.models import BaseFilters, SearchDoc, SearchDocsResponse
from onyx.file_store.models import ChatFileType
from onyx.llm.interfaces import LLMConfig, ToolChoiceOptions
from onyx.llm.models import ReasoningEffort
from onyx.prompts.chat_prompts import IMAGE_GEN_REMINDER, OPEN_URL_REMINDER
from onyx.regulatory.candidate_answer_review import (
    CandidateAnswerClaimIssue,
    CandidateAnswerClaimSpan,
    CandidateAnswerReviewResult,
    ClaimKind,
    format_candidate_answer_review,
    format_candidate_resolution_review,
)
from onyx.server.query_and_chat.placement import Placement
from onyx.server.query_and_chat.streaming_models import (
    AgentResponseDelta,
    AgentResponseStart,
    CitationInfo,
    Packet,
)
from onyx.tools.models import ParallelToolCallResponse, ToolCallKickoff, ToolResponse
from onyx.tools.tool_implementations.search.search_tool import SearchTool


def create_message(
    content: str, message_type: MessageType, token_count: int | None = None
) -> ChatMessageSimple:
    """Helper to create a ChatMessageSimple for testing."""
    if token_count is None:
        # Simple token estimation: ~1 token per 4 characters
        token_count = max(1, len(content) // 4)
    return ChatMessageSimple(
        message=content,
        token_count=token_count,
        message_type=message_type,
    )


def create_assistant_with_tool_call(
    tool_call_id: str, tool_name: str, token_count: int
) -> ChatMessageSimple:
    """Helper to create an ASSISTANT message with tool_calls for testing."""
    tool_call = ToolCallSimple(
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        tool_arguments={},
        token_count=token_count,
    )
    return ChatMessageSimple(
        message="",
        token_count=token_count,
        message_type=MessageType.ASSISTANT,
        tool_calls=[tool_call],
    )


def create_tool_response(
    tool_call_id: str, content: str, token_count: int
) -> ChatMessageSimple:
    """Helper to create a TOOL_CALL_RESPONSE message for testing."""
    return ChatMessageSimple(
        message=content,
        token_count=token_count,
        message_type=MessageType.TOOL_CALL_RESPONSE,
        tool_call_id=tool_call_id,
    )


def create_context_files(
    num_files: int = 0, num_images: int = 0, tokens_per_file: int = 100
) -> ExtractedContextFiles:
    """Helper to create ExtractedContextFiles for testing."""
    file_texts = [f"Project file {i} content" for i in range(num_files)]
    file_metadata = [
        ContextFileMetadata(
            file_id=f"file_{i}",
            filename=f"file_{i}.txt",
            file_content=f"Project file {i} content",
        )
        for i in range(num_files)
    ]
    image_files = [
        ChatLoadedFile(
            file_id=f"image_{i}",
            content=b"",
            file_type=ChatFileType.IMAGE,
            filename=f"image_{i}.png",
            content_text=None,
            token_count=50,
        )
        for i in range(num_images)
    ]
    return ExtractedContextFiles(
        file_texts=file_texts,
        image_files=image_files,
        use_as_search_filter=False,
        total_token_count=num_files * tokens_per_file,
        file_metadata=file_metadata,
        uncapped_token_count=num_files * tokens_per_file,
    )


def test_commit_staged_llm_step_publishes_state_and_packets_atomically() -> None:
    search_doc = SearchDoc(
        document_id="doc",
        chunk_ind=7,
        semantic_identifier="Rule > Article 7",
        blurb="Operative text",
        source_type=DocumentSource.FILE,
        boost=1,
        hidden=False,
        metadata={},
        match_highlights=[],
    )
    staged_processor = DynamicCitationProcessor()
    staged_processor.update_citation_mapping({1: search_doc})
    staged_state = ChatStateContainer()
    staged_state.set_reasoning_tokens("reasoning")
    staged_state.set_answer_tokens("answer")
    staged_state.set_pre_answer_processing_time(1.0)
    staged_state.add_emitted_citation(1)

    buffered = BufferedEmitter()
    buffered.emit(
        Packet(
            placement=Placement(turn_index=2),
            obj=AgentResponseStart(
                final_documents=[search_doc], pre_answer_processing_seconds=1.0
            ),
        )
    )
    buffered.emit(
        Packet(
            placement=Placement(turn_index=2),
            obj=CitationInfo(
                citation_number=1,
                document_id="doc",
                chunk_ind=7,
                semantic_identifier="Rule > Article 7",
            ),
        )
    )
    buffered.emit(
        Packet(
            placement=Placement(turn_index=2),
            obj=AgentResponseDelta(content="answer"),
        )
    )
    live_state = ChatStateContainer()
    merged_queue: queue.Queue = queue.Queue()
    destination = Emitter(merged_queue=merged_queue, model_idx=1)

    commit_staged_llm_step(
        buffered_emitter=buffered,
        staged_state=staged_state,
        staged_citation_processor=staged_processor,
        emitter=destination,
        state_container=live_state,
        pre_answer_processing_time=4.5,
    )

    assert live_state.get_reasoning_tokens() == "reasoning"
    assert live_state.get_answer_tokens() == "answer"
    assert live_state.get_pre_answer_processing_time() == 4.5
    assert live_state.get_citation_to_doc() == {1: search_doc}
    assert live_state.get_emitted_citations() == {1}
    packets = [merged_queue.get_nowait()[1] for _ in range(3)]
    assert all(packet.placement.model_index == 1 for packet in packets)
    assert isinstance(packets[0].obj, AgentResponseStart)
    assert packets[0].obj.pre_answer_processing_seconds == 4.5
    assert isinstance(packets[1].obj, CitationInfo)
    assert isinstance(packets[2].obj, AgentResponseDelta)


def test_projected_tool_decision_does_not_publish_or_mutate_partial_save_state() -> (
    None
):
    staged_state = ChatStateContainer()
    staged_state.set_reasoning_tokens("projected reasoning")
    staged_state.set_answer_tokens("projected narration")
    buffered = BufferedEmitter()
    buffered.emit(
        Packet(
            placement=Placement(turn_index=3),
            obj=AgentResponseDelta(content="projected narration"),
        )
    )

    live_state = ChatStateContainer()
    live_state.set_reasoning_tokens("canonical reasoning")
    live_state.set_answer_tokens("canonical answer")
    canonical_processor = DynamicCitationProcessor()
    staged_processor = canonical_processor.fork()
    merged_queue: queue.Queue = queue.Queue()

    selected_processor = _commit_canonical_tool_decision_step(
        projected_tool_decision_history=True,
        buffered_emitter=buffered,
        staged_state=staged_state,
        staged_citation_processor=staged_processor,
        canonical_citation_processor=canonical_processor,
        emitter=Emitter(merged_queue=merged_queue),
        state_container=live_state,
        pre_answer_processing_time=2.0,
    )

    assert selected_processor is canonical_processor
    assert merged_queue.empty()
    assert live_state.get_reasoning_tokens() == "canonical reasoning"
    assert live_state.get_answer_tokens() == "canonical answer"


class TestConstructMessageHistory:
    """Tests for the construct_message_history function."""

    def test_basic_no_truncation(self) -> None:
        """Test basic functionality when all messages fit within token budget."""
        system_prompt = create_message(
            "You are a helpful assistant", MessageType.SYSTEM, 10
        )
        user_msg1 = create_message("Hello", MessageType.USER, 5)
        assistant_msg1 = create_message("Hi there!", MessageType.ASSISTANT, 5)
        user_msg2 = create_message("How are you?", MessageType.USER, 5)

        simple_chat_history = [user_msg1, assistant_msg1, user_msg2]
        context_files = create_context_files()

        result = construct_message_history(
            system_prompt=system_prompt,
            custom_agent_prompt=None,
            simple_chat_history=simple_chat_history,
            reminder_message=None,
            context_files=context_files,
            available_tokens=1000,
        )

        # Should have: system, user1, assistant1, user2
        assert len(result) == 4
        assert result[0] == system_prompt
        assert result[1] == user_msg1
        assert result[2] == assistant_msg1
        assert result[3] == user_msg2

    def test_with_custom_agent_prompt(self) -> None:
        """Test that custom agent prompt is inserted before the last user message."""
        system_prompt = create_message("System", MessageType.SYSTEM, 10)
        user_msg1 = create_message("First message", MessageType.USER, 5)
        assistant_msg1 = create_message("Response", MessageType.ASSISTANT, 5)
        user_msg2 = create_message("Second message", MessageType.USER, 5)
        custom_agent = create_message("Custom instructions", MessageType.USER, 10)

        simple_chat_history = [user_msg1, assistant_msg1, user_msg2]
        context_files = create_context_files()

        result = construct_message_history(
            system_prompt=system_prompt,
            custom_agent_prompt=custom_agent,
            simple_chat_history=simple_chat_history,
            reminder_message=None,
            context_files=context_files,
            available_tokens=1000,
        )

        # Should have: system, user1, assistant1, custom_agent, user2
        assert len(result) == 5
        assert result[0] == system_prompt
        assert result[1] == user_msg1
        assert result[2] == assistant_msg1
        assert result[3] == custom_agent  # Before last user message
        assert result[4] == user_msg2

    def test_with_context_files(self) -> None:
        """Test that project files are inserted before the last user message."""
        system_prompt = create_message("System", MessageType.SYSTEM, 10)
        user_msg1 = create_message("First message", MessageType.USER, 5)
        user_msg2 = create_message("Second message", MessageType.USER, 5)

        simple_chat_history = [user_msg1, user_msg2]
        context_files = create_context_files(num_files=2, tokens_per_file=50)

        result = construct_message_history(
            system_prompt=system_prompt,
            custom_agent_prompt=None,
            simple_chat_history=simple_chat_history,
            reminder_message=None,
            context_files=context_files,
            available_tokens=1000,
        )

        # Should have: system, user1, context_files_message, user2
        assert len(result) == 4
        assert result[0] == system_prompt
        assert result[1] == user_msg1
        assert (
            result[2].message_type == MessageType.USER
        )  # Project files as user message
        assert "documents" in result[2].message  # Should contain JSON structure
        assert result[3] == user_msg2

    def test_with_reminder_message(self) -> None:
        """Test that reminder message is added at the very end."""
        system_prompt = create_message("System", MessageType.SYSTEM, 10)
        user_msg = create_message("Hello", MessageType.USER, 5)
        reminder = create_message("Remember to cite sources", MessageType.USER, 10)

        simple_chat_history = [user_msg]
        context_files = create_context_files()

        result = construct_message_history(
            system_prompt=system_prompt,
            custom_agent_prompt=None,
            simple_chat_history=simple_chat_history,
            reminder_message=reminder,
            context_files=context_files,
            available_tokens=1000,
        )

        # Should have: system, user, reminder
        assert len(result) == 3
        assert result[0] == system_prompt
        assert result[1] == user_msg
        assert result[2] == reminder  # At the end

    def test_tool_calls_after_last_user_message(self) -> None:
        """Test that tool calls and responses after last user message are preserved."""
        system_prompt = create_message("System", MessageType.SYSTEM, 10)
        user_msg1 = create_message("First message", MessageType.USER, 5)
        assistant_msg1 = create_message("Response", MessageType.ASSISTANT, 5)
        user_msg2 = create_message("Search for X", MessageType.USER, 5)
        assistant_with_tool = create_assistant_with_tool_call("tc_1", "search", 5)
        tool_response = create_tool_response("tc_1", "Search results...", 10)

        simple_chat_history = [
            user_msg1,
            assistant_msg1,
            user_msg2,
            assistant_with_tool,
            tool_response,
        ]
        context_files = create_context_files()

        result = construct_message_history(
            system_prompt=system_prompt,
            custom_agent_prompt=None,
            simple_chat_history=simple_chat_history,
            reminder_message=None,
            context_files=context_files,
            available_tokens=1000,
        )

        # Should have: system, user1, assistant1, user2, assistant_with_tool, tool_response
        assert len(result) == 6
        assert result[0] == system_prompt
        assert result[1] == user_msg1
        assert result[2] == assistant_msg1
        assert result[3] == user_msg2
        assert result[4] == assistant_with_tool
        assert result[5] == tool_response

    def test_custom_agent_and_project_before_last_user_with_tools_after(self) -> None:
        """Test correct ordering with custom agent, project files, and tool calls."""
        system_prompt = create_message("System", MessageType.SYSTEM, 10)
        user_msg1 = create_message("First", MessageType.USER, 5)
        user_msg2 = create_message("Second", MessageType.USER, 5)
        assistant_with_tool = create_assistant_with_tool_call("tc_1", "tool", 5)
        custom_agent = create_message("Custom", MessageType.USER, 10)

        simple_chat_history = [user_msg1, user_msg2, assistant_with_tool]
        context_files = create_context_files(num_files=1, tokens_per_file=50)

        result = construct_message_history(
            system_prompt=system_prompt,
            custom_agent_prompt=custom_agent,
            simple_chat_history=simple_chat_history,
            reminder_message=None,
            context_files=context_files,
            available_tokens=1000,
        )

        # Should have: system, user1, custom_agent, context_files, user2, assistant_with_tool
        assert len(result) == 6
        assert result[0] == system_prompt
        assert result[1] == user_msg1
        assert result[2] == custom_agent  # Before last user message
        assert result[3].message_type == MessageType.USER  # Project files
        assert "documents" in result[3].message
        assert result[4] == user_msg2  # Last user message
        assert result[5] == assistant_with_tool  # After last user message

    def test_construct_message_history_does_not_duplicate_project_images(
        self,
    ) -> None:
        """Project images are attached upstream in convert_chat_history; this
        function must not re-attach them. Simulates the realistic state where
        the last user message in simple_chat_history already carries the
        project images, and asserts they appear exactly once."""
        system_prompt = create_message("System", MessageType.SYSTEM, 10)

        project_image = ChatLoadedFile(
            file_id="project_image",
            content=b"",
            file_type=ChatFileType.IMAGE,
            filename="project.png",
            content_text=None,
            token_count=50,
        )
        # Simulate convert_chat_history's output: the last user message already
        # has the project image attached.
        user_msg = ChatMessageSimple(
            message="What is in this image?",
            token_count=5,
            message_type=MessageType.USER,
            image_files=[project_image],
        )

        simple_chat_history = [user_msg]
        context_files = ExtractedContextFiles(
            file_texts=[],
            image_files=[project_image],
            use_as_search_filter=False,
            total_token_count=0,
            file_metadata=[],
            uncapped_token_count=0,
        )

        result = construct_message_history(
            system_prompt=system_prompt,
            custom_agent_prompt=None,
            simple_chat_history=simple_chat_history,
            reminder_message=None,
            context_files=context_files,
            available_tokens=1000,
        )

        last_message = result[-1]
        assert last_message.message == "What is in this image?"
        assert last_message.image_files is not None
        assert len(last_message.image_files) == 1
        assert last_message.image_files[0].file_id == "project_image"

    def test_truncation_from_top(self) -> None:
        """Test that history is truncated from the top when token budget is exceeded."""
        system_prompt = create_message("System", MessageType.SYSTEM, 10)
        user_msg1 = create_message("First", MessageType.USER, 20)
        assistant_msg1 = create_message("Response 1", MessageType.ASSISTANT, 20)
        user_msg2 = create_message("Second", MessageType.USER, 20)
        assistant_msg2 = create_message("Response 2", MessageType.ASSISTANT, 20)
        user_msg3 = create_message("Third", MessageType.USER, 20)

        simple_chat_history = [
            user_msg1,
            assistant_msg1,
            user_msg2,
            assistant_msg2,
            user_msg3,
        ]
        context_files = create_context_files()

        # Budget only allows last 3 messages + system (10 + 20 + 20 + 20 = 70 tokens)
        result = construct_message_history(
            system_prompt=system_prompt,
            custom_agent_prompt=None,
            simple_chat_history=simple_chat_history,
            reminder_message=None,
            context_files=context_files,
            available_tokens=80,
        )

        # Should have: system, user2, assistant2, user3
        # user1 and assistant1 should be truncated
        assert len(result) == 4
        assert result[0] == system_prompt
        assert result[1] == user_msg2  # user1 truncated
        assert result[2] == assistant_msg2
        assert result[3] == user_msg3

    def test_truncation_preserves_last_user_and_messages_after(self) -> None:
        """Test that truncation preserves the last user message and everything after it."""
        system_prompt = create_message("System", MessageType.SYSTEM, 10)
        user_msg1 = create_message("First", MessageType.USER, 30)
        user_msg2 = create_message("Second", MessageType.USER, 20)
        assistant_with_tool = create_assistant_with_tool_call("tc_1", "tool", 20)
        tool_response = create_tool_response("tc_1", "tool_response", 20)

        simple_chat_history = [user_msg1, user_msg2, assistant_with_tool, tool_response]
        context_files = create_context_files()

        # Budget only allows last user message and messages after + system
        # (10 + 20 + 20 + 20 = 70 tokens)
        result = construct_message_history(
            system_prompt=system_prompt,
            custom_agent_prompt=None,
            simple_chat_history=simple_chat_history,
            reminder_message=None,
            context_files=context_files,
            available_tokens=80,
        )

        # Should have: system, user2, assistant_with_tool, tool_response
        # user1 should be truncated, but user2 and everything after preserved
        assert len(result) == 4
        assert result[0] == system_prompt
        assert result[1] == user_msg2  # user1 truncated
        assert result[2] == assistant_with_tool
        assert result[3] == tool_response

    def test_truncation_drops_orphaned_tool_response(self) -> None:
        """If truncation drops an assistant tool call, its orphaned tool response is removed."""
        system_prompt = create_message("System", MessageType.SYSTEM, 10)
        user_msg1 = create_message("First", MessageType.USER, 10)
        assistant_with_tool = create_assistant_with_tool_call("tc_1", "tool", 25)
        tool_response = create_tool_response("tc_1", "tool_response", 5)
        assistant_msg1 = create_message("Used the tool above", MessageType.ASSISTANT, 5)
        user_msg2 = create_message("Latest question", MessageType.USER, 10)

        simple_chat_history = [
            user_msg1,
            assistant_with_tool,
            tool_response,
            assistant_msg1,
            user_msg2,
        ]
        context_files = create_context_files()

        # Remaining history budget is 10 tokens (30 total - 10 system - 10 last user):
        # keeps [tool_response, assistant_msg1] from history_before_last_user,
        # but drops assistant_with_tool, making tool_response orphaned.
        result = construct_message_history(
            system_prompt=system_prompt,
            custom_agent_prompt=None,
            simple_chat_history=simple_chat_history,
            reminder_message=None,
            context_files=context_files,
            available_tokens=30,
        )

        # Orphaned tool response should be removed from final history.
        assert len(result) == 3
        assert result[0] == system_prompt
        assert result[1] == assistant_msg1
        assert result[2] == user_msg2

    def test_preserves_non_orphaned_tool_response(self) -> None:
        """Tool responses remain when their assistant tool call is present."""
        system_prompt = create_message("System", MessageType.SYSTEM, 10)
        user_msg1 = create_message("First", MessageType.USER, 10)
        assistant_with_tool = create_assistant_with_tool_call("tc_1", "tool", 20)
        tool_response = create_tool_response("tc_1", "tool_response", 5)
        user_msg2 = create_message("Latest question", MessageType.USER, 10)

        simple_chat_history = [user_msg1, assistant_with_tool, tool_response, user_msg2]
        context_files = create_context_files()

        # Remaining history budget is 25 tokens (45 total - 10 system - 10 last user):
        # keeps both assistant_with_tool and tool_response in history_before_last_user.
        result = construct_message_history(
            system_prompt=system_prompt,
            custom_agent_prompt=None,
            simple_chat_history=simple_chat_history,
            reminder_message=None,
            context_files=context_files,
            available_tokens=45,
        )

        assert len(result) == 4
        assert result[0] == system_prompt
        assert result[1] == assistant_with_tool
        assert result[2] == tool_response
        assert result[3] == user_msg2

    def test_empty_history(self) -> None:
        """Test handling of empty chat history."""
        system_prompt = create_message("System", MessageType.SYSTEM, 10)
        custom_agent = create_message("Custom", MessageType.USER, 10)
        reminder = create_message("Reminder", MessageType.USER, 10)

        simple_chat_history: list[ChatMessageSimple] = []
        context_files = create_context_files(num_files=1, tokens_per_file=50)

        result = construct_message_history(
            system_prompt=system_prompt,
            custom_agent_prompt=custom_agent,
            simple_chat_history=simple_chat_history,
            reminder_message=reminder,
            context_files=context_files,
            available_tokens=1000,
        )

        # Should have: system, custom_agent, context_files, reminder
        assert len(result) == 4
        assert result[0] == system_prompt
        assert result[1] == custom_agent
        assert result[2].message_type == MessageType.USER  # Project files
        assert result[3] == reminder

    def test_no_user_message_raises_error(self) -> None:
        """Test that an error is raised when there's no user message in history."""
        system_prompt = create_message("System", MessageType.SYSTEM, 10)
        assistant_msg = create_message("Response", MessageType.ASSISTANT, 5)
        assistant_with_tool = create_assistant_with_tool_call("tc_1", "tool", 5)

        simple_chat_history = [assistant_msg, assistant_with_tool]
        context_files = create_context_files()

        with pytest.raises(ValueError, match="No user message found"):
            construct_message_history(
                system_prompt=system_prompt,
                custom_agent_prompt=None,
                simple_chat_history=simple_chat_history,
                reminder_message=None,
                context_files=context_files,
                available_tokens=1000,
            )

    def test_not_enough_tokens_for_required_elements(self) -> None:
        """Test error when there aren't enough tokens for required elements."""
        system_prompt = create_message("System", MessageType.SYSTEM, 50)
        user_msg = create_message("Message", MessageType.USER, 50)
        custom_agent = create_message("Custom", MessageType.USER, 50)

        simple_chat_history = [user_msg]
        context_files = create_context_files(num_files=1, tokens_per_file=100)

        # Total required: 50 (system) + 50 (custom) + 100 (project) + 50 (user) = 250
        # But only 200 available
        with pytest.raises(ValueError, match="Not enough tokens"):
            construct_message_history(
                system_prompt=system_prompt,
                custom_agent_prompt=custom_agent,
                simple_chat_history=simple_chat_history,
                reminder_message=None,
                context_files=context_files,
                available_tokens=200,
            )

    def test_not_enough_tokens_for_last_user_and_messages_after(self) -> None:
        """Test error when last user message and messages after don't fit."""
        system_prompt = create_message("System", MessageType.SYSTEM, 10)
        user_msg1 = create_message("First", MessageType.USER, 10)
        user_msg2 = create_message("Second", MessageType.USER, 30)
        assistant_with_tool = create_assistant_with_tool_call("tc_1", "tool", 30)

        simple_chat_history = [user_msg1, user_msg2, assistant_with_tool]
        context_files = create_context_files()

        # Budget: 50 tokens
        # Required: 10 (system) + 30 (user2) + 30 (assistant_with_tool) = 70 tokens
        # After subtracting system: 40 tokens available, but need 60 for user2 + assistant_with_tool
        with pytest.raises(
            ValueError, match="Not enough tokens to include the last user message"
        ):
            construct_message_history(
                system_prompt=system_prompt,
                custom_agent_prompt=None,
                simple_chat_history=simple_chat_history,
                reminder_message=None,
                context_files=context_files,
                available_tokens=50,
            )

    def test_complex_scenario_all_elements(self) -> None:
        """Test a complex scenario with all elements combined."""
        system_prompt = create_message("System", MessageType.SYSTEM, 10)
        user_msg1 = create_message("First", MessageType.USER, 10)
        assistant_msg1 = create_message("Response 1", MessageType.ASSISTANT, 10)
        user_msg2 = create_message("Second", MessageType.USER, 10)
        assistant_msg2 = create_message("Response 2", MessageType.ASSISTANT, 10)
        user_msg3 = create_message("Third", MessageType.USER, 10)
        assistant_with_tool = create_assistant_with_tool_call("tc_1", "search", 10)
        tool_response = create_tool_response("tc_1", "Results", 10)
        custom_agent = create_message("Custom instructions", MessageType.USER, 15)
        reminder = create_message("Cite sources", MessageType.USER, 10)

        simple_chat_history = [
            user_msg1,
            assistant_msg1,
            user_msg2,
            assistant_msg2,
            user_msg3,
            assistant_with_tool,
            tool_response,
        ]
        context_files = create_context_files(num_files=2, tokens_per_file=20)

        result = construct_message_history(
            system_prompt=system_prompt,
            custom_agent_prompt=custom_agent,
            simple_chat_history=simple_chat_history,
            reminder_message=reminder,
            context_files=context_files,
            available_tokens=1000,
        )

        # Expected order:
        # system, user1, assistant1, user2, assistant2,
        # custom_agent, context_files, user3, assistant_with_tool, tool_response, reminder
        assert len(result) == 11
        assert result[0] == system_prompt
        assert result[1] == user_msg1
        assert result[2] == assistant_msg1
        assert result[3] == user_msg2
        assert result[4] == assistant_msg2
        assert result[5] == custom_agent  # Before last user
        assert (
            result[6].message_type == MessageType.USER
        )  # Project files before last user
        assert "documents" in result[6].message
        assert result[7] == user_msg3  # Last user message
        assert result[8] == assistant_with_tool  # After last user
        assert result[9] == tool_response  # After last user
        assert result[10] == reminder  # At the very end

    def test_context_files_json_format(self) -> None:
        """Test that project files are formatted correctly as JSON."""
        system_prompt = create_message("System", MessageType.SYSTEM, 10)
        user_msg = create_message("Hello", MessageType.USER, 5)

        simple_chat_history = [user_msg]
        context_files = create_context_files(num_files=2, tokens_per_file=50)

        result = construct_message_history(
            system_prompt=system_prompt,
            custom_agent_prompt=None,
            simple_chat_history=simple_chat_history,
            reminder_message=None,
            context_files=context_files,
            available_tokens=1000,
        )

        # Find the project files message
        project_message = result[1]  # Should be between system and user

        # Verify it's formatted as JSON
        assert "Here are some documents provided for context" in project_message.message
        assert '"documents"' in project_message.message
        assert '"document": 1' in project_message.message
        assert '"document": 2' in project_message.message
        assert '"contents"' in project_message.message
        assert "Project file 0 content" in project_message.message
        assert "Project file 1 content" in project_message.message

    def test_file_metadata_for_tool_produces_message(self) -> None:
        """When context_files has file_metadata_for_tool, a metadata listing
        message should be injected into the history."""
        system_prompt = create_message("System", MessageType.SYSTEM, 10)
        user_msg = create_message("Analyze the spreadsheet", MessageType.USER, 5)

        context_files = ExtractedContextFiles(
            file_texts=[],
            image_files=[],
            use_as_search_filter=False,
            total_token_count=0,
            file_metadata=[],
            uncapped_token_count=0,
            file_metadata_for_tool=[
                FileToolMetadata(
                    file_id="xlsx-1",
                    filename="report.xlsx",
                    approx_char_count=100000,
                ),
            ],
        )

        result = construct_message_history(
            system_prompt=system_prompt,
            custom_agent_prompt=None,
            simple_chat_history=[user_msg],
            reminder_message=None,
            context_files=context_files,
            available_tokens=1000,
            token_counter=_simple_token_counter,
        )

        # Should have: system, tool_metadata_message, user
        assert len(result) == 3
        metadata_msg = result[1]
        assert metadata_msg.message_type == MessageType.USER
        assert "report.xlsx" in metadata_msg.message
        assert "xlsx-1" in metadata_msg.message

    def test_metadata_only_and_text_files_both_present(self) -> None:
        """When both text content and tool metadata are present, both messages
        should appear in the history."""
        system_prompt = create_message("System", MessageType.SYSTEM, 10)
        user_msg = create_message("Summarize everything", MessageType.USER, 5)

        context_files = ExtractedContextFiles(
            file_texts=["Text file content here"],
            image_files=[],
            use_as_search_filter=False,
            total_token_count=100,
            file_metadata=[
                ContextFileMetadata(
                    file_id="txt-1",
                    filename="notes.txt",
                    file_content="Text file content here",
                ),
            ],
            uncapped_token_count=100,
            file_metadata_for_tool=[
                FileToolMetadata(
                    file_id="xlsx-1",
                    filename="data.xlsx",
                    approx_char_count=50000,
                ),
            ],
        )

        result = construct_message_history(
            system_prompt=system_prompt,
            custom_agent_prompt=None,
            simple_chat_history=[user_msg],
            reminder_message=None,
            context_files=context_files,
            available_tokens=2000,
            token_counter=_simple_token_counter,
        )

        # Should have: system, context_files_message, tool_metadata_message, user
        assert len(result) == 4
        # Context files message (text content)
        assert "documents" in result[1].message
        assert "Text file content here" in result[1].message
        # Tool metadata message
        assert "data.xlsx" in result[2].message
        assert result[3] == user_msg


def _simple_token_counter(text: str) -> int:
    """Approximate token counter for tests (~4 chars per token)."""
    return max(1, len(text) // 4)


def _make_file_metadata(
    file_id: str, filename: str, approx_chars: int = 50_000
) -> FileToolMetadata:
    return FileToolMetadata(
        file_id=file_id, filename=filename, approx_char_count=approx_chars
    )


class TestForgottenFileMetadata:
    """Tests for the forgotten-files mechanism in construct_message_history.

    These cover the scenario where a user attaches a large file to a chat
    message. On the first turn the file content message is in the context
    window. On subsequent turns, it may be truncated by either:
      a) context-window budget limits, or
      b) summary-based truncation removing the message before
         convert_chat_history ever runs — leaving an "orphaned" metadata
         entry with no corresponding file_id-tagged ChatMessageSimple.

    The forgotten-files mechanism must detect both cases and inject a
    lightweight metadata message so the LLM knows to use read_file.
    """

    def _build(
        self,
        simple_chat_history: list[ChatMessageSimple],
        available_tokens: int = 10_000,
        all_injected_file_metadata: dict[str, FileToolMetadata] | None = None,
    ) -> list[ChatMessageSimple]:
        """Shorthand wrapper around construct_message_history."""
        return construct_message_history(
            system_prompt=create_message("system", MessageType.SYSTEM, 5),
            custom_agent_prompt=None,
            simple_chat_history=simple_chat_history,
            reminder_message=None,
            context_files=create_context_files(),
            available_tokens=available_tokens,
            token_counter=_simple_token_counter,
            all_injected_file_metadata=all_injected_file_metadata,
        )

    @staticmethod
    def _find_forgotten_message(
        result: list[ChatMessageSimple],
    ) -> ChatMessageSimple | None:
        """Find the forgotten-files metadata message in the result, if any."""
        for msg in result:
            if "Use the read_file tool" in msg.message:
                return msg
        return None

    # ------------------------------------------------------------------
    # Case 1: file message is still in context — no forgotten-files needed
    # ------------------------------------------------------------------

    def test_file_message_present_no_forgotten_metadata(self) -> None:
        """When the file message fits in context, no forgotten-file message
        should be injected.
        """
        file_meta = _make_file_metadata("file-abc", "moby_dick.txt")
        file_msg = create_message("Contents of moby dick...", MessageType.USER, 50)
        file_msg.file_id = "file-abc"

        history = [
            file_msg,
            create_message("Summarize this", MessageType.ASSISTANT, 20),
            create_message("What's chapter 1?", MessageType.USER, 10),
        ]
        result = self._build(
            history,
            available_tokens=10_000,
            all_injected_file_metadata={"file-abc": file_meta},
        )

        forgotten = self._find_forgotten_message(result)
        assert forgotten is None, (
            "Should not inject forgotten-files when file is in context"
        )
        # The file message itself should still be present
        assert any(m.file_id == "file-abc" for m in result)

    # ------------------------------------------------------------------
    # Case 2: file message dropped by context-window truncation
    # ------------------------------------------------------------------

    def test_file_message_dropped_by_truncation_gets_forgotten_metadata(self) -> None:
        """When the context budget is too tight and the file message gets
        truncated, a forgotten-files metadata message must appear.
        """
        file_meta = _make_file_metadata("file-abc", "moby_dick.txt")
        file_msg = create_message("x" * 2000, MessageType.USER, 500)
        file_msg.file_id = "file-abc"

        history = [
            file_msg,
            create_message("Got it", MessageType.ASSISTANT, 10),
            create_message("Tell me about ch1", MessageType.USER, 10),
        ]

        # Budget is just enough for the system prompt + last messages but
        # NOT the 500-token file message.
        result = self._build(
            history,
            available_tokens=100,
            all_injected_file_metadata={"file-abc": file_meta},
        )

        forgotten = self._find_forgotten_message(result)
        assert forgotten is not None, "Forgotten-files message should be injected"
        assert "moby_dick.txt" in forgotten.message
        assert "file-abc" in forgotten.message

        # The original file message should NOT be in context
        assert not any(
            getattr(m, "file_id", None) == "file-abc"
            and m.message_type == MessageType.USER
            for m in result
            if m is not forgotten
        )

    # ------------------------------------------------------------------
    # Case 3: file message removed by summary truncation ("orphaned" metadata)
    # ------------------------------------------------------------------

    def test_orphaned_metadata_triggers_forgotten_files(self) -> None:
        """Simulates the scenario where summary truncation in process_message
        removed the file's original message BEFORE convert_chat_history ran,
        so no ChatMessageSimple has the file_id tag. The metadata is still
        passed via all_injected_file_metadata and must be treated as dropped.
        """
        file_meta = _make_file_metadata("file-abc", "moby_dick.txt")

        # History has no file_id-tagged message — it was already removed by
        # summary truncation. Only later conversation remains.
        history = [
            create_message("Summary of earlier convo", MessageType.ASSISTANT, 20),
            create_message("Now tell me about chapter 2", MessageType.USER, 10),
        ]

        result = self._build(
            history,
            available_tokens=10_000,
            all_injected_file_metadata={"file-abc": file_meta},
        )

        forgotten = self._find_forgotten_message(result)
        assert forgotten is not None, (
            "Orphaned file metadata should trigger forgotten-files message"
        )
        assert "moby_dick.txt" in forgotten.message
        assert "file-abc" in forgotten.message

    # ------------------------------------------------------------------
    # Case 4: multiple files — one survives, one is dropped
    # ------------------------------------------------------------------

    def test_mixed_files_only_dropped_ones_appear_in_forgotten(self) -> None:
        """When two files exist but only one's message is truncated, only the
        truncated file should appear in the forgotten-files metadata.
        """
        meta_a = _make_file_metadata("file-a", "big_file.txt")
        meta_b = _make_file_metadata("file-b", "small_file.txt")

        # file-a has a huge message that will be dropped, file-b fits
        file_msg_a = create_message("x" * 2000, MessageType.USER, 500)
        file_msg_a.file_id = "file-a"
        file_msg_b = create_message("small content", MessageType.USER, 5)
        file_msg_b.file_id = "file-b"

        history = [
            file_msg_a,
            create_message("ok", MessageType.ASSISTANT, 3),
            file_msg_b,
            create_message("ok", MessageType.ASSISTANT, 3),
            create_message("Compare the two files", MessageType.USER, 10),
        ]

        # Tight budget: system(5) + last-user(10) = 15 min. Give ~50 so
        # file_msg_b(5)+assistant(3)+assistant(3) fit but file_msg_a(500) won't.
        result = self._build(
            history,
            available_tokens=80,
            all_injected_file_metadata={"file-a": meta_a, "file-b": meta_b},
        )

        forgotten = self._find_forgotten_message(result)
        assert forgotten is not None
        assert "big_file.txt" in forgotten.message
        assert "file-a" in forgotten.message
        # file-b should NOT be in the forgotten message — it's still in context
        assert "small_file.txt" not in forgotten.message

    # ------------------------------------------------------------------
    # Case 5: no metadata dict → no forgotten-files message even if dropped
    # ------------------------------------------------------------------

    def test_no_metadata_dict_means_no_forgotten_message(self) -> None:
        """If all_injected_file_metadata is None (FileReaderTool not enabled),
        no forgotten-files message should be emitted even if file messages
        are dropped by truncation.
        """
        file_msg = create_message("x" * 2000, MessageType.USER, 500)
        file_msg.file_id = "file-abc"

        history = [
            file_msg,
            create_message("Got it", MessageType.ASSISTANT, 10),
            create_message("Tell me more", MessageType.USER, 10),
        ]

        result = self._build(
            history,
            available_tokens=100,
            all_injected_file_metadata=None,
        )

        forgotten = self._find_forgotten_message(result)
        assert forgotten is None, (
            "No forgotten-files message when metadata dict is None"
        )

    # ------------------------------------------------------------------
    # Case 6: orphaned metadata with multiple files, all summarized away
    # ------------------------------------------------------------------

    def test_multiple_orphaned_files_all_appear_in_forgotten(self) -> None:
        """All files from summarized-away messages should be listed in the
        forgotten-files message.
        """
        meta_a = _make_file_metadata("file-a", "report.pdf")
        meta_b = _make_file_metadata("file-b", "data.csv")

        # Both original messages were removed by summary truncation;
        # only post-summary messages remain.
        history = [
            create_message("Earlier discussion summarized", MessageType.ASSISTANT, 15),
            create_message("What patterns do you see?", MessageType.USER, 10),
        ]

        result = self._build(
            history,
            available_tokens=10_000,
            all_injected_file_metadata={"file-a": meta_a, "file-b": meta_b},
        )

        forgotten = self._find_forgotten_message(result)
        assert forgotten is not None
        assert "report.pdf" in forgotten.message
        assert "data.csv" in forgotten.message

    # ------------------------------------------------------------------
    # Case 7: file metadata persists across many turns after truncation
    # ------------------------------------------------------------------

    def test_forgotten_metadata_persists_across_many_turns(self) -> None:
        """Simulates the real bug: after the file's original message is
        summarized away, every subsequent turn should still include the
        forgotten-files metadata — not just the first turn after truncation.
        """
        file_meta = _make_file_metadata("file-abc", "moby_dick.txt")

        # Build several turns AFTER the file was already summarized away.
        # Each turn, construct_message_history is called fresh with the
        # same all_injected_file_metadata.
        for turn in range(5):
            messages = [
                create_message("Summary", MessageType.ASSISTANT, 15),
            ]
            # Add some back-and-forth after the summary
            for i in range(turn):
                messages.append(create_message(f"Question {i}", MessageType.USER, 5))
                messages.append(create_message(f"Answer {i}", MessageType.ASSISTANT, 5))
            messages.append(
                create_message(f"Latest question (turn {turn})", MessageType.USER, 5)
            )

            result = self._build(
                messages,
                available_tokens=10_000,
                all_injected_file_metadata={"file-abc": file_meta},
            )

            forgotten = self._find_forgotten_message(result)
            assert forgotten is not None, (
                f"Turn {turn}: forgotten-files message must persist every turn"
            )
            assert "moby_dick.txt" in forgotten.message


class TestFallbackToolExtraction:
    def _tool_defs(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "internal_search",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "queries": {
                                "type": "array",
                                "items": {"type": "string"},
                            }
                        },
                        "required": ["queries"],
                    },
                },
            }
        ]

    def test_noop_if_fallback_was_already_attempted(self) -> None:
        llm_step_result = LlmStepResult(
            reasoning=None,
            answer='{"name":"internal_search","arguments":{"queries":["alpha"]}}',
            tool_calls=None,
        )

        result, attempted = _try_fallback_tool_extraction(
            llm_step_result=llm_step_result,
            tool_choice=ToolChoiceOptions.REQUIRED,
            fallback_extraction_attempted=True,
            tool_defs=self._tool_defs(),
            turn_index=0,
        )

        assert result is llm_step_result
        assert attempted is False

    def test_extracts_from_answer_when_required_and_no_tool_calls(self) -> None:
        llm_step_result = LlmStepResult(
            reasoning=None,
            answer='{"name":"internal_search","arguments":{"queries":["alpha"]}}',
            tool_calls=None,
        )

        result, attempted = _try_fallback_tool_extraction(
            llm_step_result=llm_step_result,
            tool_choice=ToolChoiceOptions.REQUIRED,
            fallback_extraction_attempted=False,
            tool_defs=self._tool_defs(),
            turn_index=3,
        )

        assert attempted is True
        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].tool_name == "internal_search"
        assert result.tool_calls[0].tool_args == {"queries": ["alpha"]}
        assert result.tool_calls[0].placement == Placement(turn_index=3)

    def test_falls_back_to_reasoning_when_answer_has_no_tool_calls(self) -> None:
        llm_step_result = LlmStepResult(
            reasoning='{"name":"internal_search","arguments":{"queries":["beta"]}}',
            answer="I should search first.",
            tool_calls=None,
        )

        result, attempted = _try_fallback_tool_extraction(
            llm_step_result=llm_step_result,
            tool_choice=ToolChoiceOptions.REQUIRED,
            fallback_extraction_attempted=False,
            tool_defs=self._tool_defs(),
            turn_index=5,
        )

        assert attempted is True
        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].tool_name == "internal_search"
        assert result.tool_calls[0].tool_args == {"queries": ["beta"]}
        assert result.tool_calls[0].placement == Placement(turn_index=5)

    def test_extracts_xml_style_invoke_from_answer_when_required(self) -> None:
        llm_step_result = LlmStepResult(
            reasoning=None,
            answer=(
                '<function_calls><invoke name="internal_search">'
                '<parameter name="queries" string="false">'
                '["Onyx documentation", "Onyx docs", "Onyx platform"]'
                "</parameter></invoke></function_calls>"
            ),
            tool_calls=None,
        )

        result, attempted = _try_fallback_tool_extraction(
            llm_step_result=llm_step_result,
            tool_choice=ToolChoiceOptions.REQUIRED,
            fallback_extraction_attempted=False,
            tool_defs=self._tool_defs(),
            turn_index=7,
        )

        assert attempted is True
        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].tool_name == "internal_search"
        assert result.tool_calls[0].tool_args == {
            "queries": ["Onyx documentation", "Onyx docs", "Onyx platform"]
        }
        assert result.tool_calls[0].placement == Placement(turn_index=7)

    def test_extracts_xml_style_invoke_from_answer_when_auto(self) -> None:
        llm_step_result = LlmStepResult(
            reasoning=None,
            # Runtime-faithful shape: filtered answer is empty, raw answer has XML payload.
            answer=None,
            raw_answer=(
                '<function_calls><invoke name="internal_search">'
                '<parameter name="queries" string="false">'
                '["Onyx documentation", "Onyx docs", "Onyx internal docs"]'
                "</parameter></invoke></function_calls>"
            ),
            tool_calls=None,
        )

        result, attempted = _try_fallback_tool_extraction(
            llm_step_result=llm_step_result,
            tool_choice=ToolChoiceOptions.AUTO,
            fallback_extraction_attempted=False,
            tool_defs=self._tool_defs(),
            turn_index=9,
        )

        assert attempted is True
        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].tool_name == "internal_search"
        assert result.tool_calls[0].tool_args == {
            "queries": ["Onyx documentation", "Onyx docs", "Onyx internal docs"]
        }
        assert result.tool_calls[0].placement == Placement(turn_index=9)

    def test_extracts_from_raw_answer_when_filtered_answer_has_no_xml(self) -> None:
        llm_step_result = LlmStepResult(
            reasoning=None,
            answer="",
            raw_answer=(
                '<function_calls><invoke name="internal_search">'
                '<parameter name="queries" string="false">'
                '["Onyx documentation", "Onyx docs"]'
                "</parameter></invoke></function_calls>"
            ),
            tool_calls=None,
        )

        result, attempted = _try_fallback_tool_extraction(
            llm_step_result=llm_step_result,
            tool_choice=ToolChoiceOptions.AUTO,
            fallback_extraction_attempted=False,
            tool_defs=self._tool_defs(),
            turn_index=10,
        )

        assert attempted is True
        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].tool_name == "internal_search"
        assert result.tool_calls[0].tool_args == {
            "queries": ["Onyx documentation", "Onyx docs"]
        }
        assert result.tool_calls[0].placement == Placement(turn_index=10)

    def test_does_not_attempt_fallback_for_auto_without_tool_call_hints(self) -> None:
        llm_step_result = LlmStepResult(
            reasoning=None,
            answer="Here is a normal answer with no tool call payload.",
            tool_calls=None,
        )

        result, attempted = _try_fallback_tool_extraction(
            llm_step_result=llm_step_result,
            tool_choice=ToolChoiceOptions.AUTO,
            fallback_extraction_attempted=False,
            tool_defs=self._tool_defs(),
            turn_index=2,
        )

        assert result is llm_step_result
        assert attempted is False

    def test_returns_unchanged_when_required_but_nothing_extractable(self) -> None:
        llm_step_result = LlmStepResult(
            reasoning="Need more info.",
            answer="Let me think.",
            tool_calls=None,
        )

        result, attempted = _try_fallback_tool_extraction(
            llm_step_result=llm_step_result,
            tool_choice=ToolChoiceOptions.REQUIRED,
            fallback_extraction_attempted=False,
            tool_defs=self._tool_defs(),
            turn_index=1,
        )

        assert result is llm_step_result
        assert attempted is True
        assert result.tool_calls is None

    def test_noop_when_tool_calls_already_present(self) -> None:
        existing_call = ToolCallKickoff(
            tool_call_id="call_existing",
            tool_name="internal_search",
            tool_args={"queries": ["already-set"]},
            placement=Placement(turn_index=0),
        )
        llm_step_result = LlmStepResult(
            reasoning=None,
            answer='{"name":"internal_search","arguments":{"queries":["alpha"]}}',
            tool_calls=[existing_call],
        )

        result, attempted = _try_fallback_tool_extraction(
            llm_step_result=llm_step_result,
            tool_choice=ToolChoiceOptions.REQUIRED,
            fallback_extraction_attempted=False,
            tool_defs=self._tool_defs(),
            turn_index=0,
        )

        assert result is llm_step_result
        assert attempted is False


class TestEmptyLlmResponseClassification:
    def _make_llm(self, provider: str = "openai", model: str = "gpt-5.2") -> Mock:
        llm = Mock()
        llm.config = LLMConfig(
            model_provider=provider,
            model_name=model,
            temperature=0.0,
            max_input_tokens=4096,
        )
        return llm

    def test_openai_empty_stream_is_classified_as_budget_exceeded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("onyx.chat.llm_loop.is_true_openai_model", lambda *_: True)

        err = _build_empty_llm_response_error(
            llm=self._make_llm(),
            llm_step_result=LlmStepResult(
                reasoning=None,
                answer=None,
                tool_calls=None,
                raw_answer=None,
            ),
            tool_choice=ToolChoiceOptions.AUTO,
        )

        assert isinstance(err, EmptyLLMResponseError)
        assert err.error_code == "BUDGET_EXCEEDED"
        assert err.is_retryable is False
        assert "quota" in err.client_error_msg.lower()

    def test_reasoning_only_response_uses_generic_empty_response_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("onyx.chat.llm_loop.is_true_openai_model", lambda *_: True)

        err = _build_empty_llm_response_error(
            llm=self._make_llm(),
            llm_step_result=LlmStepResult(
                reasoning="scratchpad only",
                answer=None,
                tool_calls=None,
                raw_answer=None,
            ),
            tool_choice=ToolChoiceOptions.AUTO,
        )

        assert isinstance(err, EmptyLLMResponseError)
        assert err.error_code == "EMPTY_LLM_RESPONSE"
        assert err.is_retryable is True
        assert "quota" not in err.client_error_msg.lower()

    def test_refusal_finish_reason_is_classified_as_model_refusal(self) -> None:
        """Anthropic refusal: HTTP 200, stop_reason="refusal" (normalized by
        LiteLLM to "content_filter"), no text or tool calls. Must surface as a
        refusal, not a generic empty-stream error."""
        err = _build_empty_llm_response_error(
            llm=self._make_llm(provider="anthropic", model="claude-fable-5"),
            llm_step_result=LlmStepResult(
                reasoning=None,
                answer=None,
                tool_calls=None,
                raw_answer=None,
                finish_reason="content_filter",
            ),
            tool_choice=ToolChoiceOptions.AUTO,
        )

        assert isinstance(err, EmptyLLMResponseError)
        assert err.error_code == "MODEL_REFUSAL"
        assert err.is_retryable is False
        assert err.finish_reason == "content_filter"
        assert "declined" in err.client_error_msg.lower()
        # Anthropic-specific fallback suggestion from the issue.
        assert "Claude Opus 4.8" in err.client_error_msg

    @pytest.mark.parametrize("finish_reason", sorted(_REFUSAL_FINISH_REASONS))
    def test_refusal_finish_reasons_take_precedence_over_budget_heuristic(
        self, monkeypatch: pytest.MonkeyPatch, finish_reason: str
    ) -> None:
        """Native provider refusal reasons may pass through gateways unchanged."""
        monkeypatch.setattr("onyx.chat.llm_loop.is_true_openai_model", lambda *_: True)

        err = _build_empty_llm_response_error(
            llm=self._make_llm(),
            llm_step_result=LlmStepResult(
                reasoning=None,
                answer=None,
                tool_calls=None,
                raw_answer=None,
                finish_reason=finish_reason,
            ),
            tool_choice=ToolChoiceOptions.AUTO,
        )

        assert err.error_code == "MODEL_REFUSAL"
        assert err.is_retryable is False
        assert err.finish_reason == finish_reason
        assert "Claude Opus 4.8" not in err.client_error_msg


class TestSelectReminderText:
    """The open_url nudge must be suppressed when the open_url tool is disabled,
    otherwise the model is told to call a tool it doesn't have (confusing
    "open_url is not available" replies)."""

    def _select(self, **overrides: Any) -> str | None:
        kwargs: dict[str, Any] = dict(
            ran_image_gen=False,
            just_ran_web_search=False,
            has_open_url_tool=True,
            out_of_cycles=False,
            persona_task_prompt=None,
            include_citation_reminder=False,
            include_file_reminder=False,
        )
        kwargs.update(overrides)
        return select_reminder_text(**kwargs)

    def test_open_url_reminder_when_tool_available(self) -> None:
        result = self._select(just_ran_web_search=True, has_open_url_tool=True)
        assert result == OPEN_URL_REMINDER

    def test_no_open_url_reminder_when_tool_disabled(self) -> None:
        """Web search ran but open_url is disabled -> fall back, never nudge open_url."""
        result = self._select(just_ran_web_search=True, has_open_url_tool=False)
        assert result != OPEN_URL_REMINDER
        assert result is None  # nothing else to remind about in this scenario

    def test_open_url_reminder_suppressed_on_last_cycle(self) -> None:
        result = self._select(
            just_ran_web_search=True, has_open_url_tool=True, out_of_cycles=True
        )
        assert result != OPEN_URL_REMINDER

    def test_image_gen_reminder_takes_precedence(self) -> None:
        result = self._select(
            ran_image_gen=True, just_ran_web_search=True, has_open_url_tool=True
        )
        assert result == IMAGE_GEN_REMINDER

    def test_search_ledger_is_appended_to_standard_reminder(self) -> None:
        result = self._select(
            persona_task_prompt="Task reminder",
            search_ledger_reminder="Search receipt",
        )

        assert result == "Task reminder\n\nSearch receipt"

    def test_search_ledger_survives_special_tool_reminder(self) -> None:
        result = self._select(
            just_ran_web_search=True,
            has_open_url_tool=True,
            search_ledger_reminder="Search receipt",
        )

        assert result == OPEN_URL_REMINDER + "\n\nSearch receipt"


def test_format_search_evidence_ledger_is_compact_and_non_evidentiary() -> None:
    ledger = [
        SearchEvidenceLedgerEntry(
            query="authorization issuing authority territory",
            search_mode="hybrid",
            result_count=9,
        ),
        SearchEvidenceLedgerEntry(
            query="authorization suspension conditions",
            search_mode="keyword",
            result_count=4,
            repeated_result_count=2,
        ),
    ]

    reminder = _format_search_evidence_ledger(ledger)

    assert reminder is not None
    assert "execution receipt, not legal evidence" in reminder
    assert "query: authorization suspension conditions" in reminder
    assert "mode: keyword; returned new chunks: 4" in reminder
    assert "exact repeats omitted: 2" in reminder
    assert "remaining distinct internal-search safety allowance" not in reminder
    assert "Do not repeat an equivalent retrieval attempt" in reminder
    assert "Retry an unresolved point" in reminder


def test_format_search_evidence_ledger_preserves_all_bounded_attempts() -> None:
    entries = [
        SearchEvidenceLedgerEntry(
            query=f"query {index}",
            search_mode="hybrid",
            result_count=index,
        )
        for index in range(10)
    ]

    reminder = _format_search_evidence_ledger(entries)

    assert reminder is not None
    assert "query 0" in reminder
    assert "query 2" in reminder
    assert "query 9" in reminder
    assert "earlier attempt(s) omitted" not in reminder


def test_candidate_answer_review_feedback_describes_bounded_correction() -> None:
    reminder = format_candidate_answer_review(
        CandidateAnswerReviewResult(
            needs_reconsideration=True,
            advisory_claim_issues=[
                CandidateAnswerClaimIssue(
                    claim_reference="A material draft claim",
                    advisory_feedback=(
                        "The attributed chunk establishes a condition but not its effect."
                    ),
                )
            ],
        )
    )

    assert reminder is not None
    assert "not legal evidence" in reminder
    assert "At most one focused support search" in reminder
    assert "You retain the decision whether further retrieval is useful" not in reminder
    assert "Do not silently drop a concern" in reminder
    assert "A material draft claim" in reminder
    assert "query:" not in reminder
    assert "search_mode" not in reminder


def test_candidate_resolution_feedback_requires_bounded_final_disposition() -> None:
    reminder = format_candidate_resolution_review(
        CandidateAnswerReviewResult(
            needs_reconsideration=True,
            advisory_claim_issues=[
                CandidateAnswerClaimIssue(
                    claim_reference="An earlier unsupported sequence",
                    advisory_feedback="The revised answer repeats the same sequence.",
                )
            ],
        )
    )

    assert reminder is not None
    assert "next synthesis is final" in reminder
    assert "precise controlling-source gap" in reminder
    assert "Do not silently omit or repeat it" in reminder


def test_candidate_answer_review_without_findings_adds_no_instruction() -> None:
    reminder = format_candidate_answer_review(
        CandidateAnswerReviewResult(
            needs_reconsideration=False,
            advisory_claim_issues=[],
        )
    )

    assert reminder is None


def test_candidate_answer_evidence_uses_exact_visible_content_and_citation_order() -> (
    None
):
    first_chunk = SearchDoc(
        document_id="shared-doc",
        chunk_ind=3,
        semantic_identifier="Fallback title",
        blurb="First short blurb",
        source_type=DocumentSource.FILE,
        boost=1,
        hidden=False,
        metadata={
            "regulatory_chunk_id": "rc-first",
            "regulatory_heading_path": ["Rule", "Article 3"],
        },
        match_highlights=[],
    )
    cited_chunk = first_chunk.model_copy(
        update={
            "chunk_ind": 9,
            "blurb": "Cited short blurb",
            "metadata": {
                "regulatory_chunk_id": "rc-cited",
                "regulatory_heading_path": ["Rule", "Article 9"],
            },
        }
    )
    uncited_chunk = first_chunk.model_copy(
        update={
            "chunk_ind": 11,
            "blurb": "Uncited short blurb",
            "metadata": {
                "regulatory_chunk_id": "rc-uncited",
                "regulatory_heading_path": ["Rule", "Article 11"],
            },
        }
    )
    unseen_rich_doc = first_chunk.model_copy(
        update={
            "chunk_ind": 13,
            "blurb": "Never shown to the answer model",
            "metadata": {"regulatory_chunk_id": "rc-unseen"},
        }
    )

    evidence = _build_candidate_answer_evidence_chunks(
        candidate_answer="Later-numbered claim [4], then the earlier citation [1].",
        citation_mapping={
            1: first_chunk,
            2: uncited_chunk,
            4: cited_chunk,
            7: unseen_rich_doc,
        },
        # Deliberately insert citation 1 first: candidate appearance, not dict or
        # numeric order, must prioritize cited evidence for the bounded review.
        llm_visible_results_by_citation={
            1: ("Visible title 1", "Full exact text shown for citation 1"),
            2: ("Visible title 2", "Full exact text shown for citation 2"),
            4: ("Visible title 4", "Full exact text shown for citation 4"),
        },
    )

    assert len(evidence) == 3
    assert evidence[0].citation_number == 4
    assert evidence[0].retrieval_number == 4
    assert (evidence[0].document_id, evidence[0].chunk_id) == ("shared-doc", 9)
    assert evidence[0].chunk_identifier == "rc-cited"
    assert evidence[0].heading == "Rule > Article 9"
    assert evidence[0].content == "Full exact text shown for citation 4"
    assert evidence[1].citation_number == 1
    assert (evidence[1].document_id, evidence[1].chunk_id) == ("shared-doc", 3)
    assert evidence[1].chunk_identifier == "rc-first"
    assert evidence[1].content == "Full exact text shown for citation 1"
    assert evidence[2].citation_number is None
    assert evidence[2].retrieval_number == 2
    assert (evidence[2].document_id, evidence[2].chunk_id) == ("shared-doc", 11)
    assert evidence[2].chunk_identifier == "rc-uncited"
    assert evidence[2].content == "Full exact text shown for citation 2"
    assert all(item.chunk_identifier != "rc-unseen" for item in evidence)


def test_join_search_work_reminders_omits_missing_sections() -> None:
    assert _join_search_work_reminders(None, "ledger") == "ledger"
    assert _join_search_work_reminders("feedback", None, "ledger") == (
        "feedback\n\nledger"
    )
    assert _join_search_work_reminders(None, None) is None


def _search_tool_call(
    call_id: str,
    *,
    query: str = "model-written query",
    search_mode: str = "hybrid",
    coverage_item: str = "Requested issue",
    evidence_target: str = "Unresolved proposition",
) -> ToolCallKickoff:
    return ToolCallKickoff(
        tool_call_id=call_id,
        tool_name="internal_search",
        tool_args={
            "coverage_item": coverage_item,
            "evidence_target": evidence_target,
            "queries": [query],
            "search_mode": search_mode,
        },
        placement=Placement(turn_index=0),
    )


def test_constrain_regulatory_calls_enforces_parallel_search_cap() -> None:
    calls = [
        _search_tool_call("first", query="first query"),
        _search_tool_call("second", query="second query"),
        _search_tool_call("third", query="third query"),
    ]

    constrained = _constrain_regulatory_tool_calls(
        calls,
        search_slots=2,
    )

    assert [call.tool_call_id for call in constrained] == ["first", "second"]


def test_constrain_regulatory_calls_deduplicates_query_and_mode_within_batch() -> None:
    calls = [
        _search_tool_call("first"),
        _search_tool_call(
            "duplicate",
            query="  MODEL-WRITTEN   QUERY ",
            coverage_item="Another issue",
            evidence_target="Another proposition",
        ),
    ]

    constrained = _constrain_regulatory_tool_calls(
        calls,
        search_slots=2,
    )

    assert [call.tool_call_id for call in constrained] == ["first"]


def test_constrain_regulatory_calls_drops_previously_attempted_query_and_mode() -> None:
    constrained = _constrain_regulatory_tool_calls(
        [_search_tool_call("duplicate", query="  MODEL-WRITTEN   QUERY ")],
        search_slots=1,
        attempted_query_modes={("model-written query", "hybrid")},
    )

    assert constrained == []


def test_constrain_regulatory_calls_allows_same_query_with_different_mode() -> None:
    calls = [
        _search_tool_call("hybrid", search_mode="hybrid"),
        _search_tool_call("keyword", search_mode="keyword"),
    ]

    constrained = _constrain_regulatory_tool_calls(
        calls,
        search_slots=2,
    )

    assert [call.tool_call_id for call in constrained] == ["hybrid", "keyword"]


def test_constrain_regulatory_calls_allows_rephrased_query() -> None:
    constrained = _constrain_regulatory_tool_calls(
        [_search_tool_call("rephrased", query="different focused query")],
        search_slots=1,
        attempted_query_modes={("model-written query", "hybrid")},
    )

    assert [call.tool_call_id for call in constrained] == ["rephrased"]


def test_regulatory_tool_call_batch_feedback_reports_only_counts() -> None:
    feedback = _format_regulatory_tool_call_batch_feedback(
        requested_search_calls=17,
        executed_search_calls=8,
    )

    assert feedback is not None
    assert "Requested search calls: 17" in feedback
    assert "executed now: 8" in feedback
    assert "not executed: 9" in feedback
    assert "produced no evidence" in feedback
    assert "material unresolved proposition" in feedback


def test_regulatory_tool_call_batch_feedback_omits_completed_batch() -> None:
    assert (
        _format_regulatory_tool_call_batch_feedback(
            requested_search_calls=8,
            executed_search_calls=8,
        )
        is None
    )


def test_partial_regulatory_search_batch_reaches_next_auto_decision() -> None:
    raw_tool_calls = [
        _search_tool_call(
            f"call-{letter}",
            query=f"model-query-{letter}",
        )
        for letter in "abcdefghijklmnopq"
    ]
    deferred_queries = [call.tool_args["queries"][0] for call in raw_tool_calls[8:]]
    dispatched_batches: list[list[ToolCallKickoff]] = []
    llm_step_kwargs: list[dict[str, Any]] = []

    def fake_run_llm_step(**kwargs: Any) -> tuple[LlmStepResult, bool]:
        llm_step_kwargs.append(kwargs)
        if len(llm_step_kwargs) == 1:
            return (
                LlmStepResult(
                    reasoning=None,
                    answer=None,
                    tool_calls=raw_tool_calls,
                    raw_answer=None,
                    finish_reason="tool_calls",
                ),
                False,
            )
        return (
            LlmStepResult(
                reasoning=None,
                answer="Supported answer.",
                tool_calls=None,
                raw_answer="Supported answer.",
                finish_reason="stop",
            ),
            False,
        )

    def fake_run_tool_calls(**kwargs: Any) -> ParallelToolCallResponse:
        tool_calls = list(kwargs["tool_calls"])
        dispatched_batches.append(tool_calls)
        return ParallelToolCallResponse(
            tool_responses=[
                ToolResponse(
                    rich_response=SearchDocsResponse(
                        search_docs=[],
                        citation_mapping={},
                    ),
                    llm_facing_response=json.dumps(
                        {
                            "receipt": {
                                "coverage_item": "Requested issue",
                                "evidence_target": "Unresolved proposition",
                            },
                            "results": [],
                        }
                    ),
                    tool_call=tool_call,
                )
                for tool_call in tool_calls
            ],
            updated_citation_mapping={},
        )

    search_tool = Mock(spec=SearchTool)
    search_tool.id = 1
    search_tool.name = SearchTool.NAME
    search_tool.tool_definition.return_value = {
        "type": "function",
        "function": {
            "name": SearchTool.NAME,
            "parameters": {"type": "object", "properties": {}},
        },
    }
    persona = Mock()
    persona.id = 0
    persona.datetime_aware = False
    persona.replace_base_system_prompt = False
    persona.system_prompt = None
    persona.task_prompt = None
    llm = Mock()
    llm.config = LLMConfig(
        model_provider="openai",
        model_name="test-model",
        temperature=0.0,
        max_input_tokens=100_000,
    )
    state_container = ChatStateContainer()

    with (
        patch("onyx.chat.llm_loop.run_llm_step", side_effect=fake_run_llm_step),
        patch("onyx.chat.llm_loop.run_tool_calls", side_effect=fake_run_tool_calls),
        patch(
            "onyx.chat.llm_loop.review_regulatory_candidate_answer",
            return_value=CandidateAnswerReviewResult(needs_reconsideration=False),
        ),
        patch(
            "onyx.chat.llm_loop.get_default_base_system_prompt",
            return_value="",
        ),
        patch("onyx.chat.llm_loop.get_session_with_current_tenant"),
        patch("onyx.llm.litellm_singleton.config.initialize_litellm"),
    ):
        run_llm_loop(
            emitter=Emitter(merged_queue=queue.Queue()),
            state_container=state_container,
            simple_chat_history=[
                create_message(
                    "Analyze every independently material legal issue.",
                    MessageType.USER,
                )
            ],
            tools=[search_tool],
            custom_agent_prompt=None,
            context_files=create_context_files(),
            persona=persona,
            user_memory_context=None,
            llm=llm,
            token_counter=len,
        )

    assert len(dispatched_batches) == 1
    assert dispatched_batches[0] == raw_tool_calls[:8]
    persisted_tool_calls = state_container.get_tool_calls()
    assert [call.tool_call_id for call in persisted_tool_calls] == [
        call.tool_call_id for call in raw_tool_calls[:8]
    ]
    assert all(call.tool_call_response for call in persisted_tool_calls)

    second_decision = llm_step_kwargs[1]
    assert second_decision["tool_choice"] is ToolChoiceOptions.AUTO
    assert second_decision["tool_definitions"] == [
        search_tool.tool_definition.return_value
    ]
    reminders = [
        message
        for message in second_decision["history"]
        if message.message_type == MessageType.USER_REMINDER
    ]
    assert len(reminders) == 1
    reminder_text = reminders[0].message
    batch_receipt = reminder_text.split("# Internal-search batch receipt\n", 1)[
        1
    ].split("\n\n", 1)[0]
    assert re.findall(r"\b\d+\b", batch_receipt) == ["17", "8", "9"]
    assert all(query not in reminder_text for query in deferred_queries)


def test_sequential_focused_searches_keep_bounded_query_expansion() -> None:
    first_call = _search_tool_call("first", query="ilk hukuki mesele")
    second_call = _search_tool_call("second", query="ikinci hukuki mesele")
    steps = iter(
        [
            LlmStepResult(
                reasoning=None,
                answer=None,
                tool_calls=[first_call],
                raw_answer=None,
                finish_reason="tool_calls",
            ),
            LlmStepResult(
                reasoning=None,
                answer=None,
                tool_calls=[second_call],
                raw_answer=None,
                finish_reason="tool_calls",
            ),
            LlmStepResult(
                reasoning=None,
                answer="Sonuç.",
                tool_calls=None,
                raw_answer="Sonuç.",
                finish_reason="stop",
            ),
        ]
    )
    tool_run_kwargs: list[dict[str, Any]] = []

    def fake_run_llm_step(**_kwargs: Any) -> tuple[LlmStepResult, bool]:
        return next(steps), False

    def fake_run_tool_calls(**kwargs: Any) -> ParallelToolCallResponse:
        tool_calls = cast(list[ToolCallKickoff], kwargs["tool_calls"])
        if not tool_calls:
            return ParallelToolCallResponse(
                tool_responses=[],
                updated_citation_mapping={},
            )
        tool_run_kwargs.append(kwargs)
        tool_call = tool_calls[0]
        return ParallelToolCallResponse(
            tool_responses=[
                ToolResponse(
                    rich_response=SearchDocsResponse(
                        search_docs=[],
                        citation_mapping={},
                    ),
                    llm_facing_response=json.dumps({"results": []}),
                    tool_call=tool_call,
                )
            ],
            updated_citation_mapping={},
        )

    search_tool = Mock(spec=SearchTool)
    search_tool.id = 1
    search_tool.name = SearchTool.NAME
    search_tool.tool_definition.return_value = {
        "type": "function",
        "function": {
            "name": SearchTool.NAME,
            "parameters": {"type": "object", "properties": {}},
        },
    }
    persona = Mock(
        id=1,
        datetime_aware=False,
        replace_base_system_prompt=False,
        system_prompt=None,
        task_prompt=None,
    )
    llm = Mock()
    llm.config = LLMConfig(
        model_provider="openai",
        model_name="test-model",
        temperature=0.0,
        max_input_tokens=100_000,
    )

    with (
        patch("onyx.chat.llm_loop.run_llm_step", side_effect=fake_run_llm_step),
        patch("onyx.chat.llm_loop.run_tool_calls", side_effect=fake_run_tool_calls),
        patch("onyx.chat.llm_loop.get_default_base_system_prompt", return_value=""),
        patch("onyx.chat.llm_loop.get_session_with_current_tenant"),
        patch("onyx.llm.litellm_singleton.config.initialize_litellm"),
    ):
        run_llm_loop(
            emitter=Emitter(merged_queue=queue.Queue()),
            state_container=ChatStateContainer(),
            simple_chat_history=[
                create_message("Hukuki meseleleri incele.", MessageType.USER)
            ],
            tools=[search_tool],
            custom_agent_prompt=None,
            context_files=create_context_files(),
            persona=persona,
            user_memory_context=None,
            llm=llm,
            token_counter=len,
        )

    assert [kwargs["tool_calls"] for kwargs in tool_run_kwargs] == [
        [first_call],
        [second_call],
    ]
    assert [kwargs["skip_search_query_expansion"] for kwargs in tool_run_kwargs] == [
        False,
        False,
    ]


def test_candidate_review_runs_one_direct_search_after_ordinary_research() -> None:
    original_doc = SearchDoc(
        document_id="original-document",
        chunk_ind=1,
        semantic_identifier="Kanun > Madde 1",
        blurb="İlk araştırma metni",
        source_type=DocumentSource.FILE,
        boost=1,
        hidden=False,
        metadata={
            "regulatory_chunk_id": "original-chunk",
            "regulatory_heading_path": ["Kanun", "Madde 1"],
        },
        match_highlights=[],
    )
    recovered_doc = original_doc.model_copy(
        update={
            "document_id": "recovered-document",
            "chunk_ind": 2,
            "semantic_identifier": "Kanun > Madde 2",
            "metadata": {
                "regulatory_chunk_id": "recovered-chunk",
                "regulatory_heading_path": ["Kanun", "Madde 2"],
            },
        }
    )
    ordinary_call = _search_tool_call("ordinary", query="ordinary research")
    steps = iter(
        [
            LlmStepResult(
                reasoning=None,
                answer=None,
                tool_calls=[ordinary_call],
                raw_answer=None,
                finish_reason="tool_calls",
            ),
            LlmStepResult(
                reasoning=None,
                answer="Desteksiz yükümlülük [1].",
                tool_calls=None,
                raw_answer="Desteksiz yükümlülük [1].",
                finish_reason="stop",
            ),
            LlmStepResult(
                reasoning=None,
                answer="Düzeltilmiş yükümlülük [2].",
                tool_calls=None,
                raw_answer="Düzeltilmiş yükümlülük [2].",
                finish_reason="stop",
            ),
        ]
    )

    def fake_run_llm_step(**kwargs: Any) -> tuple[LlmStepResult, bool]:
        result = next(steps)
        cast(ChatStateContainer, kwargs["state_container"]).set_answer_tokens(
            result.answer
        )
        return result, False

    ordinary_response = ToolResponse(
        rich_response=SearchDocsResponse(
            search_docs=[original_doc],
            citation_mapping={1: original_doc.document_id},
            citation_chunk_mapping={1: original_doc.chunk_ind},
        ),
        llm_facing_response=json.dumps(
            {
                "results": [
                    {
                        "document": 1,
                        "title": original_doc.semantic_identifier,
                        "content": "İlk araştırma metni",
                        "metadata": json.dumps(original_doc.metadata),
                    }
                ]
            },
            ensure_ascii=False,
        ),
        tool_call=ordinary_call,
    )
    recovery_response = ToolResponse(
        rich_response=SearchDocsResponse(
            search_docs=[recovered_doc],
            citation_mapping={2: recovered_doc.document_id},
            citation_chunk_mapping={2: recovered_doc.chunk_ind},
        ),
        llm_facing_response=json.dumps(
            {
                "results": [
                    {
                        "document": 2,
                        "title": recovered_doc.semantic_identifier,
                        "content": "Kurtarılan kesin hüküm metni",
                        "metadata": json.dumps(recovered_doc.metadata),
                    }
                ]
            },
            ensure_ascii=False,
        ),
    )
    search_tool = Mock(spec=SearchTool)
    search_tool.id = 1
    search_tool.name = SearchTool.NAME
    search_tool.user_selected_filters = BaseFilters(regulatory_chunks_only=True)
    search_tool.tool_definition.return_value = {
        "type": "function",
        "function": {
            "name": SearchTool.NAME,
            "parameters": {"type": "object", "properties": {}},
        },
    }
    search_tool.run.return_value = recovery_response
    persona = Mock(
        id=0,
        datetime_aware=False,
        replace_base_system_prompt=False,
        system_prompt=None,
        task_prompt=None,
    )
    llm = Mock()
    llm.config = LLMConfig(
        model_provider="openai",
        model_name="test-model",
        temperature=0.0,
        max_input_tokens=100_000,
    )
    review = CandidateAnswerReviewResult(
        needs_reconsideration=True,
        advisory_claim_issues=[
            CandidateAnswerClaimIssue(
                claim_kind=ClaimKind.LEGAL_RULE,
                claim_span=CandidateAnswerClaimSpan(start=0, end=25),
                claim_reference="Desteksiz yükümlülük [1].",
                advisory_feedback="Atıf yükümlülüğü desteklemiyor.",
                related_citation_numbers=[1],
                recovery_query="yükümlülüğün kesin kanuni dayanağı",
            )
        ],
    )
    state = ChatStateContainer()

    with (
        patch("onyx.chat.llm_loop.run_llm_step", side_effect=fake_run_llm_step),
        patch(
            "onyx.chat.llm_loop.run_tool_calls",
            return_value=ParallelToolCallResponse(
                tool_responses=[ordinary_response],
                updated_citation_mapping={1: original_doc.document_id},
            ),
        ) as ordinary_research,
        patch(
            "onyx.chat.llm_loop.review_regulatory_candidate_answer",
            return_value=review,
        ),
        patch(
            "onyx.chat.llm_loop.review_regulatory_candidate_resolution",
            return_value=CandidateAnswerReviewResult(needs_reconsideration=False),
        ),
        patch("onyx.chat.llm_loop.get_default_base_system_prompt", return_value=""),
        patch("onyx.chat.llm_loop.get_session_with_current_tenant"),
        patch("onyx.llm.litellm_singleton.config.initialize_litellm"),
    ):
        run_llm_loop(
            emitter=Emitter(merged_queue=queue.Queue()),
            state_container=state,
            simple_chat_history=[
                create_message("Yükümlülüğü hukuken incele.", MessageType.USER)
            ],
            tools=[search_tool],
            custom_agent_prompt=None,
            context_files=create_context_files(),
            persona=persona,
            user_memory_context=None,
            llm=llm,
            token_counter=len,
        )

    ordinary_research.assert_called_once()
    search_tool.run.assert_called_once()
    recovery_kwargs = search_tool.run.call_args.kwargs
    assert recovery_kwargs["override_kwargs"].starting_citation_num == 2
    assert recovery_kwargs["override_kwargs"].skip_query_expansion is False
    assert recovery_kwargs["queries"] == ["yükümlülüğün kesin kanuni dayanağı"]
    assert state.get_answer_tokens() == "Düzeltilmiş yükümlülük [2]."
    assert state.get_citation_to_doc() == {1: original_doc, 2: recovered_doc}


def test_extract_llm_visible_search_results_uses_only_history_payload() -> None:
    response = json.dumps(
        {
            "receipt": {"coverage_item": "item", "evidence_target": "target"},
            "results": [
                {"document": 1, "title": "Provision A", "content": "Rule A"},
                {"document": 2, "title": "Provision B", "content": "Rule B"},
                {"document": 3, "title": "Missing content"},
                {"document": 0, "title": "Invalid citation", "content": "Rule C"},
            ],
        }
    )

    assert _extract_llm_visible_search_results(response) == [
        (1, "Provision A", "Rule A"),
        (2, "Provision B", "Rule B"),
    ]
    assert _extract_llm_visible_search_results("not json") == []


def test_compact_repeated_search_results_only_changes_model_history_copy() -> None:
    original_response = json.dumps(
        {
            "receipt": {
                "coverage_item": "independent issue",
                "evidence_target": "operative rule",
            },
            "results": [
                {"document": 7, "title": "Rule A", "content": "Exact text A"},
                {"document": 8, "title": "Rule B", "content": "Exact text B"},
            ],
            "note": "Search scope note",
        },
        ensure_ascii=False,
    )

    compacted_response, repeated_count = _compact_repeated_search_results_for_history(
        original_response,
        previously_visible_results_by_citation={7: ("Rule A", "Exact text A")},
    )

    assert repeated_count == 1
    assert len(json.loads(original_response)["results"]) == 2
    compacted_payload = json.loads(compacted_response)
    assert compacted_payload["receipt"]["coverage_item"] == "independent issue"
    assert compacted_payload["note"] == "Search scope note"
    assert compacted_payload["results"] == [
        {"document": 8, "title": "Rule B", "content": "Exact text B"}
    ]
    assert compacted_payload["history_compaction"]["omitted_exact_repeats"] == 1


def test_compact_repeated_search_results_retains_changed_chunk_payload() -> None:
    response = json.dumps(
        {
            "results": [
                {
                    "document": 7,
                    "title": "Rule A",
                    "content": "A newly expanded exact excerpt",
                }
            ]
        }
    )

    compacted_response, repeated_count = _compact_repeated_search_results_for_history(
        response,
        previously_visible_results_by_citation={
            7: ("Rule A", "Earlier shorter excerpt")
        },
    )

    assert repeated_count == 0
    assert compacted_response == response


def _regulatory_history_result(
    citation_number: int,
    *,
    content: str,
) -> dict[str, object]:
    return {
        "document": citation_number,
        "title": f"Rule > Article {citation_number}",
        "content": content,
        "metadata": json.dumps(
            {
                "regulatory_chunk_id": f"rc-{citation_number}",
                "regulatory_heading_path": ["Rule", f"Article {citation_number}"],
            }
        ),
    }


def test_tool_decision_projection_replaces_every_batch_with_bounded_inventory() -> None:
    first_response = json.dumps(
        {
            "results": [
                _regulatory_history_result(1, content="old-one"),
                _regulatory_history_result(2, content="old-two"),
            ]
        }
    )
    second_response = json.dumps(
        {
            "results": [
                _regulatory_history_result(2, content="old-two-variant"),
                _regulatory_history_result(3, content="old-three"),
            ]
        }
    )
    latest_response = json.dumps(
        {
            "results": [
                _regulatory_history_result(4, content="latest-four"),
                _regulatory_history_result(5, content="latest-five"),
            ]
        }
    )
    history = [
        create_assistant_with_tool_call("search-1", "internal_search", 1),
        create_tool_response("search-1", first_response, 10),
        create_assistant_with_tool_call("search-2", "internal_search", 1),
        create_tool_response("search-2", second_response, 10),
        create_assistant_with_tool_call("search-3", "internal_search", 1),
        create_tool_response("search-3", latest_response, 10),
        create_assistant_with_tool_call("other", "other_tool", 1),
        create_tool_response("other", "unchanged", 1),
    ]
    canonical_messages = [message.message for message in history]

    projected, omitted = _project_regulatory_history_for_tool_decision(
        history,
        token_counter=len,
    )

    assert omitted == 6
    assert [message.message for message in history] == canonical_messages
    first_projected = json.loads(projected[1].message)
    second_projected = json.loads(projected[3].message)
    latest_projected = json.loads(projected[5].message)
    assert first_projected["results"] == []
    assert second_projected["results"] == []
    assert latest_projected["results"] == []
    assert [
        item["document"] for item in first_projected["search_result_inventory"]
    ] == [1]
    assert [
        item["document"] for item in second_projected["search_result_inventory"]
    ] == [2]
    assert [
        item["document"] for item in latest_projected["search_result_inventory"]
    ] == [4, 5]
    assert all(
        "content" not in item
        for payload in (first_projected, second_projected, latest_projected)
        for item in payload["search_result_inventory"]
    )
    assert first_projected["search_result_inventory"][0] == {
        "document": 1,
        "regulatory_chunk_id": "rc-1",
        "heading": "Rule > Article 1",
    }
    assert latest_projected["search_result_inventory"][0]["decision_excerpt"] == (
        "latest-four"
    )
    assert '"content": "latest-four"' not in projected[5].message
    assert projected[7].message == "unchanged"
    assert projected[1].token_count == len(projected[1].message)


def test_tool_decision_projection_skips_one_bounded_search_call() -> None:
    response = json.dumps(
        {"results": [_regulatory_history_result(1, content="x" * 1_000)]}
    )
    history = [
        create_assistant_with_tool_call("search-1", "internal_search", 1),
        create_tool_response("search-1", response, 10),
    ]

    projected, omitted = _project_regulatory_history_for_tool_decision(
        history,
        token_counter=len,
    )

    assert projected is history
    assert omitted == 0


def test_tool_decision_projection_bounds_excerpts_after_multiple_searches() -> None:
    first_response = json.dumps(
        {"results": [_regulatory_history_result(1, content="first result")]}
    )
    second_response = json.dumps(
        {"results": [_regulatory_history_result(2, content="x" * 1_000)]}
    )
    history = [
        create_assistant_with_tool_call("search-1", "internal_search", 1),
        create_tool_response("search-1", first_response, 10),
        create_assistant_with_tool_call("search-2", "internal_search", 1),
        create_tool_response("search-2", second_response, 10),
    ]

    projected, omitted = _project_regulatory_history_for_tool_decision(
        history,
        token_counter=len,
    )

    assert omitted == 2
    inventory = json.loads(projected[3].message)["search_result_inventory"]
    assert len(inventory[0]["decision_excerpt"]) == 240
    assert inventory[0]["decision_excerpt"].endswith("…")


def test_tool_decision_projection_only_excerpts_top_results_per_search() -> None:
    first_response = json.dumps(
        {
            "results": [
                _regulatory_history_result(index, content=f"result-{index}")
                for index in range(1, 5)
            ]
        }
    )
    second_response = json.dumps(
        {
            "results": [
                _regulatory_history_result(index, content=f"latest-{index}")
                for index in range(5, 9)
            ]
        }
    )
    history = [
        create_assistant_with_tool_call("search-1", "internal_search", 1),
        create_tool_response("search-1", first_response, 10),
        create_assistant_with_tool_call("search-2", "internal_search", 1),
        create_tool_response("search-2", second_response, 10),
    ]

    projected, omitted = _project_regulatory_history_for_tool_decision(
        history,
        token_counter=len,
    )

    assert omitted == 8
    inventory = json.loads(projected[3].message)["search_result_inventory"]
    assert ["decision_excerpt" in item for item in inventory] == [
        True,
        True,
        False,
        False,
    ]


def test_tool_decision_projection_duplicates_do_not_consume_excerpt_quota() -> None:
    first_response = json.dumps(
        {
            "results": [
                _regulatory_history_result(1, content="first-one"),
                _regulatory_history_result(2, content="first-two"),
            ]
        }
    )
    second_response = json.dumps(
        {
            "results": [
                _regulatory_history_result(1, content="duplicate-one"),
                _regulatory_history_result(2, content="duplicate-two"),
                _regulatory_history_result(3, content="new-three"),
                _regulatory_history_result(4, content="new-four"),
                _regulatory_history_result(5, content="new-five"),
            ]
        }
    )
    history = [
        create_assistant_with_tool_call("search-1", "internal_search", 1),
        create_tool_response("search-1", first_response, 10),
        create_assistant_with_tool_call("search-2", "internal_search", 1),
        create_tool_response("search-2", second_response, 10),
    ]

    projected, omitted = _project_regulatory_history_for_tool_decision(
        history,
        token_counter=len,
    )

    assert omitted == 7
    inventory = json.loads(projected[3].message)["search_result_inventory"]
    assert [item["document"] for item in inventory] == [2, 3, 4, 5]
    assert ["decision_excerpt" in item for item in inventory] == [
        True,
        True,
        False,
        False,
    ]


def test_tool_decision_projection_includes_review_priority_excerpt() -> None:
    first_response = json.dumps(
        {
            "results": [
                _regulatory_history_result(index, content=f"result-{index}")
                for index in range(1, 5)
            ]
        }
    )
    second_response = json.dumps(
        {"results": [_regulatory_history_result(5, content="second search")]}
    )
    history = [
        create_assistant_with_tool_call("search-1", "internal_search", 1),
        create_tool_response("search-1", first_response, 10),
        create_assistant_with_tool_call("search-2", "internal_search", 1),
        create_tool_response("search-2", second_response, 10),
    ]

    projected, omitted = _project_regulatory_history_for_tool_decision(
        history,
        token_counter=len,
        priority_citation_numbers={4},
    )

    assert omitted == 5
    inventory = json.loads(projected[1].message)["search_result_inventory"]
    assert [item["document"] for item in inventory] == [1, 4]
    assert ["decision_excerpt" in item for item in inventory] == [False, True]


def test_tool_decision_projection_preserves_long_structural_heading() -> None:
    source = "SOURCE " + ("A" * 120)
    scope = "SECTION " + ("B" * 100)
    article = "MADDE 42A — " + ("C" * 60)
    long_result = {
        "document": 1,
        "title": "display title",
        "content": "operative text",
        "metadata": json.dumps(
            {
                "regulatory_chunk_id": "rc-long",
                "regulatory_heading_path": [source, scope, article],
            }
        ),
    }
    history = [
        create_assistant_with_tool_call("search-1", "internal_search", 1),
        create_tool_response("search-1", json.dumps({"results": [long_result]}), 10),
        create_assistant_with_tool_call("search-2", "internal_search", 1),
        create_tool_response(
            "search-2",
            json.dumps({"results": [_regulatory_history_result(2, content="second")]}),
            10,
        ),
    ]

    projected, omitted = _project_regulatory_history_for_tool_decision(
        history,
        token_counter=len,
    )

    assert omitted == 2
    heading = json.loads(projected[1].message)["search_result_inventory"][0]["heading"]
    assert source in heading
    assert "MADDE 42A" in heading


def test_tool_decision_projection_bounds_provision_navigation() -> None:
    first_response = json.dumps(
        {
            "results": [_regulatory_history_result(1, content="first")],
            "regulatory_provision_navigation": {
                "type": "regulatory_provision_heading_navigation",
                "document_title": "Rule",
                "usage_note": "long canonical note",
                "headings": [
                    {
                        "article_key": f"article:{index}",
                        "heading_label": f"Article {index}",
                    }
                    for index in range(1, 25)
                ],
            },
        }
    )
    second_response = json.dumps(
        {"results": [_regulatory_history_result(2, content="second")]}
    )
    history = [
        create_assistant_with_tool_call("search-1", "internal_search", 1),
        create_tool_response("search-1", first_response, 10),
        create_assistant_with_tool_call("search-2", "internal_search", 1),
        create_tool_response("search-2", second_response, 10),
    ]

    projected, omitted = _project_regulatory_history_for_tool_decision(
        history,
        token_counter=len,
    )

    assert omitted == 2
    navigation = json.loads(projected[1].message)["regulatory_provision_navigation"]
    assert len(navigation["headings"]) == 16
    assert navigation["headings"][0]["article_key"] == "article:1"
    assert navigation["headings"][-1]["article_key"] == "article:16"
    assert navigation["headings_omitted_for_tool_decision"] == 8
    assert "not legal evidence" in navigation["usage_note"]


@pytest.mark.parametrize(
    "broken_old_response",
    [
        "not-json",
        json.dumps(
            {
                "results": [
                    {
                        "document": 1,
                        "title": "Rule",
                        "content": "exact text",
                    }
                ]
            }
        ),
    ],
)
def test_tool_decision_projection_fails_open_to_canonical_history(
    broken_old_response: str,
) -> None:
    history = [
        create_assistant_with_tool_call("search-1", "internal_search", 1),
        create_tool_response("search-1", broken_old_response, 10),
        create_assistant_with_tool_call("search-2", "internal_search", 1),
        create_tool_response(
            "search-2",
            json.dumps({"results": [_regulatory_history_result(2, content="latest")]}),
            10,
        ),
    ]

    projected, omitted = _project_regulatory_history_for_tool_decision(
        history,
        token_counter=len,
    )

    assert projected is history
    assert omitted == 0


def test_projected_tool_action_discards_narration_and_normalizes_ui_placement() -> None:
    original_tool_call = ToolCallKickoff(
        tool_call_id="search-1",
        tool_name="internal_search",
        tool_args={"queries": ["Article 7"], "search_mode": "keyword"},
        placement=Placement(turn_index=9, tab_index=4, sub_turn_index=2),
    )
    result = LlmStepResult(
        reasoning="projected reasoning",
        answer="I will search again",
        raw_answer="I will search again",
        tool_calls=[original_tool_call],
        finish_reason="tool_calls",
    )

    hidden = _hide_projected_tool_decision_output(result, turn_index=3)

    assert hidden.reasoning is None
    assert hidden.answer is None
    assert hidden.raw_answer is None
    assert hidden.finish_reason == "tool_calls"
    assert hidden.tool_calls is not None
    assert hidden.tool_calls[0].tool_args == original_tool_call.tool_args
    assert hidden.tool_calls[0].placement == Placement(turn_index=3, tab_index=0)
    assert result.reasoning == "projected reasoning"
    assert result.answer == "I will search again"
    assert result.tool_calls == [original_tool_call]


def _history_search_doc(
    citation_number: int,
    *,
    document_id: str,
    heading_path: list[str],
) -> SearchDoc:
    return SearchDoc(
        document_id=document_id,
        chunk_ind=citation_number,
        semantic_identifier=" > ".join(heading_path),
        blurb=f"Rule {citation_number}",
        source_type=DocumentSource.FILE,
        boost=1,
        hidden=False,
        metadata={
            "regulatory_chunk_id": f"rc-{citation_number}",
            "regulatory_heading_path": heading_path,
        },
        match_highlights=[],
    )


def test_gathered_search_docs_preserve_first_seen_unique_chunks() -> None:
    first = _history_search_doc(
        1,
        document_id="doc-a",
        heading_path=["Rule", "Article 1"],
    )
    duplicate = first.model_copy(update={"blurb": "later duplicate"})
    second = _history_search_doc(
        2,
        document_id="doc-a",
        heading_path=["Rule", "Article 2"],
    )

    merged = _merge_gathered_search_docs([first], [duplicate, second])

    assert merged == [first, second]


def test_tool_decision_projection_supports_legacy_payload_via_citation_mapping() -> (
    None
):
    legacy_response = json.dumps(
        {
            "results": [
                {
                    "document": 1,
                    "title": "Rule > Article 1",
                    "content": "old exact text",
                }
            ]
        }
    )
    latest_response = json.dumps(
        {"results": [_regulatory_history_result(2, content="latest exact text")]}
    )
    history = [
        create_assistant_with_tool_call("search-1", "internal_search", 1),
        create_tool_response("search-1", legacy_response, 10),
        create_assistant_with_tool_call("search-2", "internal_search", 1),
        create_tool_response("search-2", latest_response, 10),
    ]
    citation_mapping = {
        1: _history_search_doc(
            1,
            document_id="legacy-doc",
            heading_path=["Rule", "Article 1"],
        )
    }

    projected, omitted = _project_regulatory_history_for_tool_decision(
        history,
        token_counter=len,
        citation_mapping=citation_mapping,
    )

    assert omitted == 2
    assert json.loads(projected[1].message)["search_result_inventory"] == [
        {
            "document": 1,
            "regulatory_chunk_id": "rc-1",
            "heading": "Rule > Article 1",
        }
    ]
    latest_inventory = json.loads(projected[3].message)["search_result_inventory"]
    assert latest_inventory[0]["decision_excerpt"] == "latest exact text"


def test_reconsideration_history_compaction_preserves_selected_provision_groups_and_lanes() -> (
    None
):
    first_response = json.dumps(
        {
            "receipt": {"evidence_target": "first issue"},
            "results": [
                {"document": 1, "title": "Other rule", "content": "Other"},
                {"document": 2, "title": "Article 7(a)", "content": "Cited"},
                {"document": 3, "title": "Article 7(b)", "content": "Sibling"},
            ],
        }
    )
    second_response = json.dumps(
        {
            "receipt": {"evidence_target": "second issue"},
            "results": [
                {"document": 4, "title": "Lane representative", "content": "Keep"},
                {"document": 5, "title": "Inventory rule", "content": "Omit"},
            ],
        }
    )
    history = [
        create_assistant_with_tool_call("search-1", "internal_search", 1),
        create_tool_response("search-1", first_response, 100),
        create_assistant_with_tool_call("search-2", "internal_search", 1),
        create_tool_response("search-2", second_response, 100),
        create_assistant_with_tool_call("other-1", "other_tool", 1),
        create_tool_response("other-1", "unchanged non-search response", 7),
    ]
    citation_mapping = {
        1: _history_search_doc(1, document_id="other", heading_path=["Article 1"]),
        2: _history_search_doc(2, document_id="shared", heading_path=["Article 7"]),
        3: _history_search_doc(3, document_id="shared", heading_path=["Article 7"]),
        4: _history_search_doc(4, document_id="lane", heading_path=["Article 4"]),
        5: _history_search_doc(5, document_id="inventory", heading_path=["Article 5"]),
    }

    with patch(
        "onyx.chat.llm_loop._REGULATORY_RECONSIDERATION_HISTORY_RESULT_THRESHOLD",
        2,
    ):
        compacted, retained, omitted = (
            _compact_regulatory_search_history_for_reconsideration(
                history,
                candidate_answer="The supported proposition [[2]]().",
                citation_mapping=citation_mapping,
                token_counter=len,
            )
        )

    assert retained == {1, 2, 3, 4}
    assert omitted == 1
    assert (
        json.loads(compacted[1].message)["results"]
        == json.loads(first_response)["results"]
    )
    compacted_second = json.loads(compacted[3].message)
    assert [result["document"] for result in compacted_second["results"]] == [4]
    assert compacted_second["omitted_result_inventory"] == [
        {"document": 5, "title": "Inventory rule"}
    ]
    assert compacted_second["history_compaction"]["omitted_after_candidate_review"] == 1
    assert compacted[5].message == "unchanged non-search response"
    assert compacted[3].token_count == len(compacted[3].message)


def test_reconsideration_history_compaction_does_not_touch_small_retrieval() -> None:
    response = json.dumps(
        {"results": [{"document": 1, "title": "Rule", "content": "Exact text"}]}
    )
    history = [
        create_assistant_with_tool_call("search", "internal_search", 1),
        create_tool_response("search", response, 10),
    ]

    compacted, retained, omitted = (
        _compact_regulatory_search_history_for_reconsideration(
            history,
            candidate_answer="Answer [[1]]().",
            citation_mapping={
                1: _history_search_doc(
                    1, document_id="document", heading_path=["Article 1"]
                )
            },
            token_counter=len,
        )
    )

    assert compacted is history
    assert retained is None
    assert omitted == 0


def test_reconsideration_history_compaction_ignores_unmapped_only_citations() -> None:
    response = json.dumps(
        {
            "results": [
                {"document": 1, "title": "First rule", "content": "First"},
                {"document": 2, "title": "Second rule", "content": "Second"},
            ]
        }
    )
    history = [
        create_assistant_with_tool_call("search", "internal_search", 1),
        create_tool_response("search", response, 10),
    ]
    citation_mapping = {
        1: _history_search_doc(1, document_id="document", heading_path=["Article 1"]),
        2: _history_search_doc(2, document_id="document", heading_path=["Article 2"]),
    }

    with patch(
        "onyx.chat.llm_loop._REGULATORY_RECONSIDERATION_HISTORY_RESULT_THRESHOLD",
        0,
    ):
        compacted, retained, omitted = (
            _compact_regulatory_search_history_for_reconsideration(
                history,
                candidate_answer="Unsupported reference [[9999]]().",
                citation_mapping=citation_mapping,
                token_counter=len,
            )
        )

    assert compacted is history
    assert retained is None
    assert omitted == 0


def test_reconsideration_history_compaction_preserves_only_near_structural_relatives() -> (
    None
):
    response = json.dumps(
        {
            "results": [
                {"document": 10, "title": "Cited leaf", "content": "Cited"},
                {"document": 20, "title": "Cited root", "content": "Root"},
                {"document": 1, "title": "Lane representative", "content": "Lane"},
                {"document": 9, "title": "Lead-in", "content": "Parent"},
                {"document": 11, "title": "Sibling", "content": "Sibling"},
                {"document": 100, "title": "Distant sibling", "content": "Far"},
                {"document": 21, "title": "Other root rule", "content": "Other"},
            ]
        }
    )
    history = [
        create_assistant_with_tool_call("search", "internal_search", 1),
        create_tool_response("search", response, 10),
    ]
    citation_mapping = {
        1: _history_search_doc(1, document_id="other-document", heading_path=["Other"]),
        9: _history_search_doc(
            9,
            document_id="shared-document",
            heading_path=["Part", "Article 7"],
        ),
        10: _history_search_doc(
            10,
            document_id="shared-document",
            heading_path=["Part", "Article 7", "a"],
        ),
        11: _history_search_doc(
            11,
            document_id="shared-document",
            heading_path=["Part", "Article 7", "b"],
        ),
        20: _history_search_doc(
            20, document_id="shared-document", heading_path=["Article 20"]
        ),
        21: _history_search_doc(
            21, document_id="shared-document", heading_path=["Article 21"]
        ),
        100: _history_search_doc(
            100,
            document_id="shared-document",
            heading_path=["Part", "Article 7", "c"],
        ),
    }

    with patch(
        "onyx.chat.llm_loop._REGULATORY_RECONSIDERATION_HISTORY_RESULT_THRESHOLD",
        0,
    ):
        compacted, retained, omitted = (
            _compact_regulatory_search_history_for_reconsideration(
                history,
                candidate_answer="Claims [[10]]() and [[20]]().",
                citation_mapping=citation_mapping,
                token_counter=len,
            )
        )

    assert retained == {1, 9, 10, 11, 20}
    assert omitted == 2
    compacted_response = json.loads(compacted[1].message)
    assert [result["document"] for result in compacted_response["results"]] == [
        10,
        20,
        1,
        9,
        11,
    ]
    assert [
        item["document"] for item in compacted_response["omitted_result_inventory"]
    ] == [100, 21]


def test_regulatory_search_chunk_cap_is_bounded_only_when_enabled() -> None:
    assert _regulatory_search_chunk_cap(True) == 8
    assert _regulatory_search_chunk_cap(False) is None


def test_regulatory_search_call_budget_is_bounded_only_for_regulatory_request() -> None:
    assert _regulatory_search_call_budget(True) == 16
    assert _regulatory_search_call_budget(False) is None


@pytest.mark.parametrize(
    (
        "complex_regulatory_request",
        "tool_choice",
        "projected_tool_decision_history",
        "reasoning_effort",
        "expected",
    ),
    [
        (False, ToolChoiceOptions.AUTO, False, ReasoningEffort.AUTO, None),
        (False, ToolChoiceOptions.NONE, False, ReasoningEffort.HIGH, None),
        (True, ToolChoiceOptions.AUTO, True, ReasoningEffort.AUTO, 3584),
        (True, ToolChoiceOptions.REQUIRED, False, ReasoningEffort.HIGH, 5632),
        (True, ToolChoiceOptions.AUTO, False, ReasoningEffort.AUTO, 5632),
        (True, ToolChoiceOptions.NONE, False, ReasoningEffort.OFF, 5632),
        (True, ToolChoiceOptions.NONE, False, ReasoningEffort.HIGH, 5632),
    ],
)
def test_regulatory_llm_steps_bound_provider_output_reservations(
    complex_regulatory_request: bool,
    tool_choice: ToolChoiceOptions,
    projected_tool_decision_history: bool,
    reasoning_effort: ReasoningEffort,
    expected: int | None,
) -> None:
    assert (
        _regulatory_llm_step_max_tokens(
            complex_regulatory_request=complex_regulatory_request,
            tool_choice=tool_choice,
            projected_tool_decision_history=projected_tool_decision_history,
            reasoning_effort=reasoning_effort,
        )
        == expected
    )


def test_regulatory_search_breadth_does_not_raise_evidence_ceiling() -> None:
    chunk_cap = _regulatory_search_chunk_cap(True)
    call_budget = _regulatory_search_call_budget(True)

    assert chunk_cap is not None
    assert call_budget is not None
    assert call_budget * chunk_cap <= 16 * 8
    assert (call_budget + 4) * chunk_cap <= (16 + 4) * 8


@pytest.mark.parametrize(
    ("base_budget", "candidate_was_rejected", "expected"),
    [
        (None, False, None),
        (None, True, None),
        (16, False, 16),
        (16, True, 16),
    ],
)
def test_effective_regulatory_search_budget_excludes_direct_review_recovery(
    base_budget: int | None,
    candidate_was_rejected: bool,
    expected: int | None,
) -> None:
    assert (
        _effective_regulatory_search_call_budget(
            base_budget,
            candidate_was_rejected=candidate_was_rejected,
        )
        == expected
    )
