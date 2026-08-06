import traceback
from collections import defaultdict
from typing import Any

import onyx.tracing.framework._error_tracing as _error_tracing
from onyx.chat.models import ChatMessageSimple
from onyx.configs.chat_configs import MAX_CHUNKS_FED_TO_CHAT
from onyx.configs.constants import MessageType
from onyx.context.search.models import SearchDocsResponse
from onyx.db.memory import UserMemoryContext
from onyx.server.query_and_chat.streaming_models import (
    Packet,
    PacketException,
    SectionEnd,
)
from onyx.tools.interface import Tool
from onyx.tools.models import (
    ChatFile,
    ChatMinimalTextMessage,
    OpenURLToolOverrideKwargs,
    ParallelToolCallResponse,
    PythonToolOverrideKwargs,
    SearchToolOverrideKwargs,
    ToolCallException,
    ToolCallKickoff,
    ToolExecutionException,
    ToolResponse,
    WebSearchToolOverrideKwargs,
)
from onyx.tools.tool_implementations.coding_agent.coding_agent_tool import (
    CodingAgentTool,
    CodingAgentToolOverrideKwargs,
)
from onyx.tools.tool_implementations.memory.memory_tool import (
    MemoryTool,
    MemoryToolOverrideKwargs,
)
from onyx.tools.tool_implementations.open_url.open_url_tool import OpenURLTool
from onyx.tools.tool_implementations.python.python_tool import PythonTool
from onyx.tools.tool_implementations.search.search_tool import SearchTool
from onyx.tools.tool_implementations.web_search.web_search_tool import WebSearchTool
from onyx.tracing.framework.create import function_span
from onyx.tracing.framework.spans import SpanError
from onyx.utils.logger import setup_logger
from onyx.utils.threadpool_concurrency import run_functions_tuples_in_parallel

logger = setup_logger()

QUERIES_FIELD = "queries"
COVERAGE_ITEM_FIELD = "coverage_item"
URLS_FIELD = "urls"
GENERIC_TOOL_ERROR_MESSAGE = "Tool failed with error: {error}"

DEFAULT_SEARCH_MAX_LLM_CHUNKS = MAX_CHUNKS_FED_TO_CHAT
PARALLEL_SEARCH_TARGET_TOTAL_LLM_CHUNKS = 84
PARALLEL_SEARCH_MIN_LLM_CHUNKS_PER_CALL = 8

# 10 minute timeout for tool execution to prevent indefinite hangs
TOOL_EXECUTION_TIMEOUT_SECONDS = 10 * 60

# Mapping of tool name to the field that should be merged when multiple calls exist
MERGEABLE_TOOL_FIELDS: dict[str, str] = {
    WebSearchTool.NAME: QUERIES_FIELD,
    OpenURLTool.NAME: URLS_FIELD,
}


def _merge_tool_calls(tool_calls: list[ToolCallKickoff]) -> list[ToolCallKickoff]:
    """Merge multiple web-search or open-URL calls into a single call.

    Internal-search calls deliberately remain separate so independent legal issues
    retain their own retrieval result sets. For WebSearchTool, if there are multiple
    calls, their queries are merged into a single tool call.
    For OpenURLTool (open_url), multiple calls have their urls merged.
    Other tool calls are left unchanged.

    Args:
        tool_calls: List of tool calls to potentially merge

    Returns:
        List of merged tool calls
    """
    # Group tool calls by tool name
    tool_calls_by_name: dict[str, list[ToolCallKickoff]] = defaultdict(list)
    merged_calls: list[ToolCallKickoff] = []

    for tool_call in tool_calls:
        tool_calls_by_name[tool_call.tool_name].append(tool_call)

    # Process each tool name group
    for tool_name, calls in tool_calls_by_name.items():
        if tool_name in MERGEABLE_TOOL_FIELDS and len(calls) > 1:
            merge_field = MERGEABLE_TOOL_FIELDS[tool_name]

            # Merge field values from all calls
            all_values: list[str] = []
            for call in calls:
                values = call.tool_args.get(merge_field, [])
                if isinstance(values, list):
                    all_values.extend(values)
                elif values:
                    # Handle case where it might be a single string
                    all_values.append(str(values))

            # Create a merged tool call using the first call's ID and merging the field
            merged_args = calls[0].tool_args.copy()
            merged_args[merge_field] = all_values

            merged_call = ToolCallKickoff(
                tool_call_id=calls[0].tool_call_id,  # Use first call's ID
                tool_name=tool_name,
                tool_args=merged_args,
                # Use first call's placement since merged calls become a single call
                placement=calls[0].placement,
            )
            merged_calls.append(merged_call)
        else:
            # No merging needed, add all calls as-is
            merged_calls.extend(calls)

    return merged_calls


def _safe_run_single_tool(
    tool: Tool,
    tool_call: ToolCallKickoff,
    override_kwargs: Any,
) -> ToolResponse:
    """Execute a single tool and return its response.

    This function is designed to be run in parallel via run_functions_tuples_in_parallel.

    Exception handling:
    - ToolCallException: Expected errors from tool execution (e.g., invalid input,
      API failures). Uses the exception's llm_facing_message for LLM consumption.
    - Other exceptions: Unexpected errors. Uses a generic error message.

    In all cases (success or failure):
    - SectionEnd packet is emitted to signal tool completion
    - tool_call is set on the response for downstream processing
    """
    tool_response: ToolResponse | None = None

    with function_span(tool.name) as span_fn:
        span_fn.span_data.input = str(tool_call.tool_args)
        try:
            tool_response = tool.run(
                placement=tool_call.placement,
                override_kwargs=override_kwargs,
                **tool_call.tool_args,
            )
            span_fn.span_data.output = tool_response.llm_facing_response
        except ToolCallException as e:
            # ToolCallException is an expected error from tool execution
            # Use llm_facing_message which is specifically designed for LLM consumption
            logger.error("Tool call error for %s: %s", tool.name, e)
            tool_response = ToolResponse(
                rich_response=None,
                llm_facing_response=GENERIC_TOOL_ERROR_MESSAGE.format(
                    error=e.llm_facing_message
                ),
            )
            _error_tracing.attach_error_to_current_span(
                SpanError(
                    message="Tool call error (expected)",
                    data={
                        "tool_name": tool.name,
                        "tool_call_id": tool_call.tool_call_id,
                        "tool_args": tool_call.tool_args,
                        "error": str(e),
                        "llm_facing_message": e.llm_facing_message,
                        "stack_trace": traceback.format_exc(),
                        "error_type": "ToolCallException",
                    },
                )
            )
        except ToolExecutionException as e:
            # Unexpected error during tool execution
            logger.error("Unexpected error running tool %s: %s", tool.name, e)
            tool_response = ToolResponse(
                rich_response=None,
                llm_facing_response=GENERIC_TOOL_ERROR_MESSAGE.format(error=str(e)),
            )
            _error_tracing.attach_error_to_current_span(
                SpanError(
                    message="Tool execution error (unexpected)",
                    data={
                        "tool_name": tool.name,
                        "tool_call_id": tool_call.tool_call_id,
                        "tool_args": tool_call.tool_args,
                        "error": str(e),
                        "stack_trace": traceback.format_exc(),
                        "error_type": type(e).__name__,
                    },
                )
            )
            if e.emit_error_packet:
                tool.emitter.emit(
                    Packet(
                        placement=tool_call.placement,
                        obj=PacketException(exception=e),
                    )
                )
        except Exception as e:
            # Unexpected error during tool execution
            logger.error("Unexpected error running tool %s: %s", tool.name, e)
            tool_response = ToolResponse(
                rich_response=None,
                llm_facing_response=GENERIC_TOOL_ERROR_MESSAGE.format(error=str(e)),
            )
            _error_tracing.attach_error_to_current_span(
                SpanError(
                    message="Tool execution error (unexpected)",
                    data={
                        "tool_name": tool.name,
                        "tool_call_id": tool_call.tool_call_id,
                        "tool_args": tool_call.tool_args,
                        "error": str(e),
                        "stack_trace": traceback.format_exc(),
                        "error_type": type(e).__name__,
                    },
                )
            )

    # Emit SectionEnd after tool completes (success or failure)
    tool.emitter.emit(
        Packet(
            placement=tool_call.placement,
            obj=SectionEnd(),
        )
    )

    # Set tool_call on the response for downstream processing
    tool_response.tool_call = tool_call
    return tool_response


def _search_input_context(
    tool_call: ToolCallKickoff,
    last_user_message: str,
) -> tuple[str, list[ChatMinimalTextMessage]]:
    raw_queries = tool_call.tool_args.get(QUERIES_FIELD)
    if (
        isinstance(raw_queries, list)
        and len(raw_queries) == 1
        and isinstance(raw_queries[0], str)
        and raw_queries[0].strip()
    ):
        focused_query = raw_queries[0].strip()
        raw_coverage_item = tool_call.tool_args.get(COVERAGE_ITEM_FIELD)
        focused_context = (
            raw_coverage_item.strip()
            if isinstance(raw_coverage_item, str) and raw_coverage_item.strip()
            else focused_query
        )
        return focused_query, [
            ChatMinimalTextMessage(
                message=focused_context,
                message_type=MessageType.USER,
            )
        ]

    return last_user_message, [
        ChatMinimalTextMessage(
            message=last_user_message,
            message_type=MessageType.USER,
        )
    ]


def _has_completed_internal_search_in_current_turn(
    message_history: list[ChatMessageSimple],
) -> bool:
    """Return whether this user turn already contains an internal-search result."""

    last_user_index: int | None = None
    for index in range(len(message_history) - 1, -1, -1):
        if message_history[index].message_type == MessageType.USER:
            last_user_index = index
            break

    if last_user_index is None:
        return False

    current_turn_messages = message_history[last_user_index + 1 :]
    internal_search_call_ids = {
        tool_call.tool_call_id
        for message in current_turn_messages
        if message.message_type == MessageType.ASSISTANT and message.tool_calls
        for tool_call in message.tool_calls
        if tool_call.tool_name == SearchTool.NAME
    }
    return any(
        message.message_type == MessageType.TOOL_CALL_RESPONSE
        and message.tool_call_id in internal_search_call_ids
        for message in current_turn_messages
    )


def _search_filter_message_history(
    tool_calls: list[ToolCallKickoff],
    last_user_message: str,
    has_completed_internal_search: bool,
) -> list[ChatMinimalTextMessage]:
    """Build bounded context for SearchTool's source and time decisions.

    The first decision retains the latest user request so temporal and source intent
    are not lost. Later decisions see only the current search batch; prior tool
    results remain available to the top-level planner but are not repeated here.
    """

    if not has_completed_internal_search:
        return [
            ChatMinimalTextMessage(
                message=last_user_message,
                message_type=MessageType.USER,
            )
        ]

    focused_messages: list[ChatMinimalTextMessage] = []
    seen_messages: set[str] = set()
    for tool_call in tool_calls:
        if tool_call.tool_name != SearchTool.NAME:
            continue
        _, search_history = _search_input_context(
            tool_call=tool_call,
            last_user_message=last_user_message,
        )
        for message in search_history:
            if message.message in seen_messages:
                continue
            seen_messages.add(message.message)
            focused_messages.append(message)

    return focused_messages or [
        ChatMinimalTextMessage(
            message=last_user_message,
            message_type=MessageType.USER,
        )
    ]


def _should_skip_search_query_expansion(
    *, skip_requested: bool, internal_search_call_count: int
) -> bool:
    """Honor only an explicit retry decision; lane construction enforces bounds."""
    del internal_search_call_count
    return skip_requested


def _parallel_search_filter_queries(
    tool_calls: list[ToolCallKickoff],
) -> list[str]:
    """Collect the model-written queries used by one parallel routing decision."""

    queries: list[str] = []
    for tool_call in tool_calls:
        if tool_call.tool_name != SearchTool.NAME:
            continue
        raw_queries = tool_call.tool_args.get(QUERIES_FIELD)
        if not isinstance(raw_queries, list):
            continue
        queries.extend(
            query.strip()
            for query in raw_queries
            if isinstance(query, str) and query.strip()
        )
    return list(dict.fromkeys(queries))


def _max_llm_chunks_per_search_call(
    internal_search_call_count: int,
    *,
    per_call_cap: int | None = None,
) -> int:
    if internal_search_call_count <= 1:
        computed_limit = DEFAULT_SEARCH_MAX_LLM_CHUNKS
    else:
        shared_budget = (
            PARALLEL_SEARCH_TARGET_TOTAL_LLM_CHUNKS // internal_search_call_count
        )
        computed_limit = min(
            DEFAULT_SEARCH_MAX_LLM_CHUNKS,
            max(PARALLEL_SEARCH_MIN_LLM_CHUNKS_PER_CALL, shared_budget),
        )

    if per_call_cap is None:
        return computed_limit
    if per_call_cap <= 0:
        raise ValueError("per_call_cap must be positive")
    return min(computed_limit, per_call_cap)


def run_tool_calls(
    tool_calls: list[ToolCallKickoff],
    tools: list[Tool],
    # The stuff below is needed for the different individual built-in tools
    message_history: list[ChatMessageSimple],
    user_memory_context: UserMemoryContext | None,
    user_info: str | None,
    citation_mapping: dict[int, str],
    next_citation_num: int,
    # Max number of tools to run concurrently (and overall) in this batch.
    # If set, tool calls beyond this limit are dropped.
    max_concurrent_tools: int | None = None,
    # Explicitly skip SearchTool query expansion. Model-written internal-search
    # calls are preserved regardless; this flag remains for caller compatibility.
    skip_search_query_expansion: bool = False,
    # Files from the chat session to pass to tools like PythonTool
    chat_files: list[ChatFile] | None = None,
    # A map of url -> summary for passing web results to open url tool
    url_snippet_map: dict[str, str] = {},
    # When False, don't pass memory context to search tools for query expansion
    # (but still pass it to the memory tool for persistence)
    inject_memories_in_prompt: bool = True,
    search_llm_chunks_per_call_cap: int | None = None,
) -> ParallelToolCallResponse:
    """Run (optionally merged) tool calls in parallel and update citation mappings.

    Before execution, calls for `WebSearchTool` and `OpenURLTool` are merged:
    - `WebSearchTool`: merge the `queries` list
    - `OpenURLTool`: merge the `urls` list

    Separate `SearchTool` calls are preserved and run on isolated tool instances so
    their retrieval state cannot race or blend independent result sets.

    Tools are executed in parallel (threadpool). For tools that generate citations,
    each tool call is assigned a **distinct** `starting_citation_num` range to avoid
    citation number collisions when running concurrently (the range is advanced by
    100 per tool call).

    The provided `citation_mapping` may be mutated in-place: any new
    `SearchDocsResponse.citation_mapping` entries are merged into it.

    Args:
        tool_calls: List of tool calls to execute.
        tools: List of available tool instances.
        message_history: Chat message history (used to find the most recent user query
            for `SearchTool` override kwargs).
        user_memory_context: User memory context, if available (passed through to `SearchTool`).
        user_info: User information string, if available (passed through to `SearchTool`).
        citation_mapping: Current citation number to URL mapping. May be updated with
            new citations produced by search tools.
        next_citation_num: The next citation number to allocate from.
        max_concurrent_tools: Max number of tools to run in this batch. If set, any
            tool calls after this limit are dropped (not queued).
        skip_search_query_expansion: Explicitly request skipping query expansion for
            `SearchTool`. Tool-runner calls already preserve model-written internal
            queries; the flag remains for compatibility with existing callers.

    Returns:
        A `ParallelToolCallResponse` containing:
        - `tool_responses`: `ToolResponse` objects for successfully dispatched tool calls
          (each has `tool_call` set). If a tool execution fails at the threadpool layer,
          its entry will be omitted.
        - `updated_citation_mapping`: The updated citation mapping dictionary.
    """
    # Merge tool calls for SearchTool, WebSearchTool, and OpenURLTool
    merged_tool_calls = _merge_tool_calls(tool_calls)

    if not merged_tool_calls:
        return ParallelToolCallResponse(
            tool_responses=[],
            updated_citation_mapping=citation_mapping,
        )

    tools_by_name = {tool.name: tool for tool in tools}

    # Drop unknown tools (and don't let them count against the cap)
    filtered_tool_calls: list[ToolCallKickoff] = []
    for tool_call in merged_tool_calls:
        if tool_call.tool_name not in tools_by_name:
            logger.warning("Tool %s not found in tools list", tool_call.tool_name)
            continue
        filtered_tool_calls.append(tool_call)

    # Apply safety cap (drop tool calls beyond the cap)
    if max_concurrent_tools is not None:
        if max_concurrent_tools <= 0:
            return ParallelToolCallResponse(
                tool_responses=[],
                updated_citation_mapping=citation_mapping,
            )
        filtered_tool_calls = filtered_tool_calls[:max_concurrent_tools]

    internal_search_call_count = sum(
        tool_call.tool_name == SearchTool.NAME for tool_call in filtered_tool_calls
    )
    search_filter_queries = _parallel_search_filter_queries(filtered_tool_calls)
    parallel_search_forks: list[SearchTool] = []
    if internal_search_call_count > 1:
        search_tool = tools_by_name.get(SearchTool.NAME)
        if isinstance(search_tool, SearchTool):
            parallel_search_forks = search_tool.fork_for_parallel_calls(
                internal_search_call_count
            )
    parallel_search_fork_index = 0

    # Get starting citation number from citation processor to avoid conflicts with project files
    starting_citation_num = next_citation_num

    # Prepare minimal history for SearchTool (computed once, shared by all)
    minimal_history = [
        ChatMinimalTextMessage(message=msg.message, message_type=msg.message_type)
        for msg in message_history
    ]
    last_user_message = None
    for i in range(len(minimal_history) - 1, -1, -1):
        if minimal_history[i].message_type == MessageType.USER:
            last_user_message = minimal_history[i].message
            break

    search_filter_history = (
        _search_filter_message_history(
            tool_calls=filtered_tool_calls,
            last_user_message=last_user_message,
            has_completed_internal_search=(
                _has_completed_internal_search_in_current_turn(message_history)
            ),
        )
        if internal_search_call_count > 0 and last_user_message is not None
        else []
    )

    # Convert citation_mapping for OpenURLTool (computed once, shared by all)
    url_to_citation: dict[str, int] = {
        url: citation_num for citation_num, url in citation_mapping.items()
    }

    # Prepare all tool calls with their override_kwargs
    # Each tool gets a unique starting citation number to avoid conflicts when running in parallel
    tool_run_params: list[tuple[Tool, ToolCallKickoff, Any]] = []

    for tool_call in filtered_tool_calls:
        tool = tools_by_name[tool_call.tool_name]
        if isinstance(tool, SearchTool) and internal_search_call_count > 1:
            tool = parallel_search_forks[parallel_search_fork_index]
            parallel_search_fork_index += 1

        # Emit the tool start packet before running the tool
        tool.emit_start(placement=tool_call.placement)

        override_kwargs: (
            SearchToolOverrideKwargs
            | WebSearchToolOverrideKwargs
            | OpenURLToolOverrideKwargs
            | PythonToolOverrideKwargs
            | MemoryToolOverrideKwargs
            | CodingAgentToolOverrideKwargs
            | None
        ) = None

        if isinstance(tool, SearchTool):
            if last_user_message is None:
                raise ValueError("No user message found in message history")

            search_original_query, search_message_history = _search_input_context(
                tool_call=tool_call,
                last_user_message=last_user_message,
            )

            search_memory_context = (
                user_memory_context
                if inject_memories_in_prompt
                else (
                    user_memory_context.without_memories()
                    if user_memory_context
                    else None
                )
            )
            override_kwargs = SearchToolOverrideKwargs(
                starting_citation_num=starting_citation_num,
                original_query=search_original_query,
                message_history=search_message_history,
                filter_message_history=search_filter_history,
                filter_queries=search_filter_queries or None,
                user_memory_context=search_memory_context,
                user_info=user_info,
                skip_query_expansion=_should_skip_search_query_expansion(
                    skip_requested=skip_search_query_expansion,
                    internal_search_call_count=internal_search_call_count,
                ),
                max_llm_chunks=_max_llm_chunks_per_search_call(
                    internal_search_call_count,
                    per_call_cap=search_llm_chunks_per_call_cap,
                ),
            )
            # Increment citation number for next search tool to avoid conflicts
            # Estimate: reserve 100 citation slots per search tool
            starting_citation_num += 100

        elif isinstance(tool, WebSearchTool):
            override_kwargs = WebSearchToolOverrideKwargs(
                starting_citation_num=starting_citation_num,
            )
            # Increment citation number for next search tool to avoid conflicts
            starting_citation_num += 100

        elif isinstance(tool, OpenURLTool):
            override_kwargs = OpenURLToolOverrideKwargs(
                starting_citation_num=starting_citation_num,
                citation_mapping=url_to_citation,
                url_snippet_map=url_snippet_map,
            )
            starting_citation_num += 100

        elif isinstance(tool, PythonTool):
            override_kwargs = PythonToolOverrideKwargs(
                chat_files=chat_files or [],
            )
        elif isinstance(tool, CodingAgentTool):
            override_kwargs = CodingAgentToolOverrideKwargs()
        elif isinstance(tool, MemoryTool):
            override_kwargs = MemoryToolOverrideKwargs(
                user_name=(
                    user_memory_context.user_info.name if user_memory_context else None
                ),
                user_email=(
                    user_memory_context.user_info.email if user_memory_context else None
                ),
                user_role=(
                    user_memory_context.user_info.role if user_memory_context else None
                ),
                existing_memories=(
                    list(user_memory_context.memories) if user_memory_context else []
                ),
                chat_history=minimal_history,
            )

        tool_run_params.append((tool, tool_call, override_kwargs))

    # Run all tools in parallel
    functions_with_args = [
        (_safe_run_single_tool, (tool, tool_call, override_kwargs))
        for tool, tool_call, override_kwargs in tool_run_params
    ]

    tool_run_results: list[ToolResponse | None] = run_functions_tuples_in_parallel(
        functions_with_args,
        allow_failures=True,  # Continue even if some tools fail
        max_workers=max_concurrent_tools,
        timeout=TOOL_EXECUTION_TIMEOUT_SECONDS,
    )

    # Process results and update citation_mapping
    for result in tool_run_results:
        if result is None:
            continue

        if result and isinstance(result.rich_response, SearchDocsResponse):
            new_citations = result.rich_response.citation_mapping
            if new_citations:
                # Merge new citations into the existing mapping
                citation_mapping.update(new_citations)

    tool_responses = [result for result in tool_run_results if result is not None]
    return ParallelToolCallResponse(
        tool_responses=tool_responses,
        updated_citation_mapping=citation_mapping,
    )
