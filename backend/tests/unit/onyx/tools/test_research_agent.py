import json
from collections.abc import Callable
from datetime import datetime
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest

from onyx.chat.chat_state import ChatStateContainer
from onyx.chat.citation_processor import CitationMode, DynamicCitationProcessor
from onyx.chat.emitter import Emitter
from onyx.chat.llm_loop import construct_message_history
from onyx.chat.llm_step import translate_history_to_llm_format
from onyx.chat.models import ChatMessageSimple, LlmStepResult, ToolCallSimple
from onyx.configs.constants import DocumentSource, MessageType
from onyx.context.search.models import BaseFilters, SearchDoc, SearchDocsResponse
from onyx.deep_research.dr_mock_tools import (
    GENERATE_REPORT_TOOL_NAME,
    RESEARCH_AGENT_TASK_KEY,
    THINK_TOOL_NAME,
)
from onyx.deep_research.models import ResearchAgentCallResult
from onyx.llm.constants import LlmProviderNames
from onyx.llm.interfaces import LLMConfig
from onyx.llm.models import AssistantMessage, ToolMessage
from onyx.regulatory.candidate_answer_review import (
    build_candidate_answer_evidence_chunk,
)
from onyx.server.query_and_chat.placement import Placement
from onyx.tools.fake_tools import research_agent
from onyx.tools.models import (
    ParallelToolCallResponse,
    ToolCallInfo,
    ToolCallKickoff,
    ToolResponse,
)
from onyx.tools.tool_implementations.search.search_tool import SearchTool


def _make_search_tool(*, regulatory_chunks_only: bool = False) -> SearchTool:
    return SearchTool(
        tool_id=1,
        emitter=MagicMock(),
        user=MagicMock(is_anonymous=False),
        persona_search_info=MagicMock(document_set_names=[]),
        llm=MagicMock(),
        document_index=MagicMock(),
        user_selected_filters=(
            BaseFilters(regulatory_chunks_only=True) if regulatory_chunks_only else None
        ),
        project_id_filter=None,
        enable_slack_search=False,
        auto_detect_filters=True,
    )


def _research_agent_call(task: str, *, tab_index: int = 0) -> ToolCallKickoff:
    return ToolCallKickoff(
        tool_call_id=f"research-{tab_index}",
        tool_name="research_agent",
        tool_args={RESEARCH_AGENT_TASK_KEY: task},
        placement=Placement(turn_index=0, tab_index=tab_index),
    )


def _llm() -> MagicMock:
    llm = MagicMock()
    llm.config = MagicMock(max_input_tokens=100_000)
    return llm


def _llm_step(tool_call: ToolCallKickoff) -> tuple[LlmStepResult, bool]:
    return (
        LlmStepResult(
            reasoning=None,
            answer=None,
            tool_calls=[tool_call],
        ),
        False,
    )


def _search_call(
    call_id: str, query: str, *, search_mode: str = "hybrid"
) -> ToolCallKickoff:
    return ToolCallKickoff(
        tool_call_id=call_id,
        tool_name=SearchTool.NAME,
        tool_args={
            "queries": [query],
            "search_mode": search_mode,
            "coverage_item": "focused legal issue",
            "evidence_target": "controlling rule and material qualifications",
        },
        placement=Placement(turn_index=0),
    )


def _search_response(title: str, content: str) -> str:
    return json.dumps(
        {
            "receipt": {
                "coverage_item": "focused legal issue",
                "evidence_target": "controlling rule and material qualifications",
            },
            "results": [
                {
                    "document": 1,
                    "title": title,
                    "content": content,
                    "metadata": json.dumps(
                        {
                            "regulatory_chunk_id": f"chunk-{title}",
                            "regulatory_heading_path": ["Part A", title],
                        }
                    ),
                }
            ],
            "regulatory_provision_navigation": {"nearby": [title]},
        }
    )


def _regulatory_search_doc(
    *,
    document_id: str,
    chunk_ind: int,
    chunk_identifier: str,
    heading: str,
) -> SearchDoc:
    return SearchDoc(
        document_id=document_id,
        chunk_ind=chunk_ind,
        semantic_identifier=heading,
        blurb="WRONG SHORT BLURB",
        source_type=DocumentSource.FILE,
        boost=1,
        hidden=False,
        metadata={
            "regulatory_chunk_id": chunk_identifier,
            "regulatory_heading_path": ["Rule", heading],
        },
        match_highlights=[],
        updated_at=datetime.now(),
    )


def test_exact_regulatory_evidence_uses_llm_visible_content_and_global_citation() -> (
    None
):
    search_doc = _regulatory_search_doc(
        document_id="document-1",
        chunk_ind=9,
        chunk_identifier="regulatory-chunk-9",
        heading="Article 9",
    )
    local_response = ToolResponse(
        rich_response=SearchDocsResponse(
            search_docs=[search_doc],
            citation_mapping={7: search_doc.document_id},
            citation_chunk_mapping={7: search_doc.chunk_ind},
        ),
        llm_facing_response=json.dumps(
            {
                "results": [
                    {
                        "document": 7,
                        "title": "Uploaded file",
                        "content": "EXACT LLM-VISIBLE OPERATIVE TEXT",
                        "metadata": json.dumps(search_doc.metadata),
                    }
                ]
            }
        ),
        tool_call=_search_call("search-exact", "operative relationship"),
    )

    evidence = research_agent._exact_regulatory_evidence_from_search_response(
        local_response
    )

    assert len(evidence) == 1
    assert (evidence[0].document_id, evidence[0].chunk_id) == (
        search_doc.document_id,
        search_doc.chunk_ind,
    )
    assert evidence[0].content == "EXACT LLM-VISIBLE OPERATIVE TEXT"
    assert evidence[0].content != search_doc.blurb
    assert evidence[0].chunk_identifier == "regulatory-chunk-9"
    remapped = research_agent._remap_exact_evidence_chunks(
        ResearchAgentCallResult(
            intermediate_report="supported [7]",
            citation_mapping={7: search_doc},
            evidence_citation_mapping={7: search_doc},
            exact_evidence_chunks=evidence,
        ),
        {4: search_doc},
    )
    assert len(remapped) == 1
    assert remapped[0].citation_number == 4
    assert remapped[0].retrieval_number == 4


def test_research_agent_result_keeps_uncited_exact_evidence_addressable() -> None:
    cited_doc = _regulatory_search_doc(
        document_id="document-cited",
        chunk_ind=1,
        chunk_identifier="chunk-cited",
        heading="Article 1",
    )
    uncited_doc = _regulatory_search_doc(
        document_id="document-uncited",
        chunk_ind=2,
        chunk_identifier="chunk-uncited",
        heading="Article 2",
    )
    citation_processor = DynamicCitationProcessor(
        citation_mode=CitationMode.KEEP_MARKERS
    )
    citation_processor.update_citation_mapping({1: cited_doc, 2: uncited_doc})
    list(citation_processor.process_token("Supported proposition [1]."))

    result = research_agent._build_research_agent_call_result(
        intermediate_report="Supported proposition [1].",
        citation_processor=citation_processor,
        exact_evidence_chunks=[
            build_candidate_answer_evidence_chunk(
                document_id="document-cited",
                chunk_id=1,
                citation_number=1,
                retrieval_number=1,
                chunk_identifier="chunk-cited",
                heading="Rule > Article 1",
                content="cited exact text",
            ),
            build_candidate_answer_evidence_chunk(
                document_id="document-uncited",
                chunk_id=2,
                citation_number=2,
                retrieval_number=2,
                chunk_identifier="chunk-uncited",
                heading="Rule > Article 2",
                content="uncited exact text",
            ),
        ],
    )

    assert result.citation_mapping == {1: cited_doc}
    assert result.evidence_citation_mapping == {1: cited_doc, 2: uncited_doc}
    assert result.exact_evidence_chunks[0].citation_number == 1
    assert result.exact_evidence_chunks[0].retrieval_number == 1
    assert result.exact_evidence_chunks[1].citation_number is None
    assert result.exact_evidence_chunks[1].retrieval_number == 2


def test_uncited_exact_evidence_remaps_only_its_retrieval_number() -> None:
    search_doc = _regulatory_search_doc(
        document_id="document-uncited",
        chunk_ind=8,
        chunk_identifier="chunk-uncited",
        heading="Article 8",
    )
    evidence = build_candidate_answer_evidence_chunk(
        document_id="document-uncited",
        chunk_id=8,
        citation_number=None,
        retrieval_number=7,
        chunk_identifier="chunk-uncited",
        heading="Rule > Article 8",
        content="uncited exact text",
    )

    remapped = research_agent._remap_exact_evidence_chunks(
        ResearchAgentCallResult(
            intermediate_report="No citation used.",
            citation_mapping={},
            evidence_citation_mapping={7: search_doc},
            exact_evidence_chunks=[evidence],
        ),
        {4: search_doc},
    )

    assert remapped[0].citation_number is None
    assert remapped[0].retrieval_number == 4


def _search_response_with_navigation(
    *,
    document_title: str,
    content: str,
    headings: list[tuple[str, str]],
) -> str:
    return json.dumps(
        {
            "results": [
                {
                    "document": 1,
                    "title": document_title,
                    "content": content,
                    "metadata": json.dumps(
                        {
                            "regulatory_chunk_id": f"chunk-{document_title}",
                            "regulatory_heading_path": [document_title],
                        }
                    ),
                }
            ],
            "regulatory_provision_navigation": {
                "type": "regulatory_provision_heading_navigation",
                "document_title": document_title,
                "usage_note": "navigation leads only",
                "headings": [
                    {"article_key": article_key, "heading_label": heading_label}
                    for article_key, heading_label in headings
                ],
            },
        },
        ensure_ascii=False,
    )


def test_regulatory_research_caps_only_llm_facing_search_evidence() -> None:
    search_call = _search_call("search-1", "focused legal relationship")
    search_doc = _regulatory_search_doc(
        document_id="document-controlling",
        chunk_ind=2,
        chunk_identifier="chunk-Controlling provision",
        heading="Controlling provision",
    )
    llm_facing_response = _search_response("Controlling provision", "material evidence")
    report_call = ToolCallKickoff(
        tool_call_id="report",
        tool_name=GENERATE_REPORT_TOOL_NAME,
        tool_args={},
        placement=Placement(turn_index=0),
    )

    with (
        patch.object(
            research_agent,
            "run_llm_step",
            side_effect=[_llm_step(search_call), _llm_step(report_call)],
        ),
        patch.object(
            research_agent,
            "run_tool_calls",
            return_value=ParallelToolCallResponse(
                tool_responses=[
                    ToolResponse(
                        rich_response=SearchDocsResponse(
                            search_docs=[search_doc],
                            citation_mapping={1: search_doc.document_id},
                            citation_chunk_mapping={1: search_doc.chunk_ind},
                        ),
                        llm_facing_response=llm_facing_response,
                        tool_call=search_call,
                    )
                ],
                updated_citation_mapping={},
            ),
        ) as run_tool_calls,
        patch.object(
            research_agent,
            "generate_intermediate_report",
            return_value="report",
        ) as generate_report,
    ):
        result = research_agent.run_research_agent_call(
            research_agent_call=_research_agent_call("focused topic"),
            parent_tool_call_id="parent",
            tools=[_make_search_tool(regulatory_chunks_only=True)],
            emitter=MagicMock(),
            state_container=ChatStateContainer(),
            llm=_llm(),
            is_reasoning_model=True,
            token_counter=len,
            user_identity=None,
        )

    assert result is not None
    assert run_tool_calls.call_args.kwargs["search_llm_chunks_per_call_cap"] == (
        research_agent._REGULATORY_RESEARCH_MAX_LLM_CHUNKS_PER_CALL
    )
    assert result.exact_evidence_chunks[0].content == "material evidence"
    canonical_history = cast(
        list[ChatMessageSimple], generate_report.call_args.kwargs["history"]
    )
    assert generate_report.call_args.kwargs["max_tokens"] == (
        research_agent.REGULATORY_MAX_INTERMEDIATE_REPORT_LENGTH_TOKENS
    )
    assert [
        message.message
        for message in canonical_history
        if message.message_type == MessageType.TOOL_CALL_RESPONSE
    ] == [llm_facing_response]


def test_regulatory_research_uses_compact_exact_history_for_report() -> None:
    search_call = _search_call("search-1", "unknown operative relationship")
    report_call = ToolCallKickoff(
        tool_call_id="report",
        tool_name=GENERATE_REPORT_TOOL_NAME,
        tool_args={},
        placement=Placement(turn_index=0),
    )
    search_doc = _regulatory_search_doc(
        document_id="document-1",
        chunk_ind=4,
        chunk_identifier="chunk-1",
        heading="Article 4",
    )
    exact_content = "controlling operative text"
    search_response = json.dumps(
        {
            "receipt": {
                "coverage_item": "unknown operative relationship",
                "evidence_target": "controlling rule and material qualifications",
            },
            "results": [
                {
                    "document": 1,
                    "title": "Article 4",
                    "content": exact_content,
                    "metadata": json.dumps(search_doc.metadata),
                }
            ],
            "regulatory_provision_navigation": {
                "usage_note": "navigation only " + ("x" * 8_000)
            },
        }
    )

    with (
        patch.object(
            research_agent,
            "run_llm_step",
            side_effect=[_llm_step(search_call), _llm_step(report_call)],
        ),
        patch.object(
            research_agent,
            "run_tool_calls",
            return_value=ParallelToolCallResponse(
                tool_responses=[
                    ToolResponse(
                        rich_response=SearchDocsResponse(
                            search_docs=[search_doc],
                            citation_mapping={1: search_doc.document_id},
                            citation_chunk_mapping={1: search_doc.chunk_ind},
                        ),
                        llm_facing_response=search_response,
                        tool_call=search_call,
                    )
                ],
                updated_citation_mapping={},
            ),
        ),
        patch.object(
            research_agent,
            "generate_intermediate_report",
            return_value="report",
        ) as generate_report,
    ):
        result = research_agent.run_research_agent_call(
            research_agent_call=_research_agent_call("focused topic"),
            parent_tool_call_id="parent",
            tools=[_make_search_tool(regulatory_chunks_only=True)],
            emitter=MagicMock(),
            state_container=ChatStateContainer(),
            llm=_llm(),
            is_reasoning_model=True,
            token_counter=len,
            user_identity=None,
        )

    assert result is not None
    report_history = cast(
        list[ChatMessageSimple], generate_report.call_args.kwargs["history"]
    )
    assert [message.message_type for message in report_history] == [
        MessageType.USER,
        MessageType.ASSISTANT,
        MessageType.TOOL_CALL_RESPONSE,
    ]
    compact_payload = json.loads(report_history[-1].message)
    assert compact_payload["type"] == "validated_regulatory_search_evidence"
    assert compact_payload["results"][0]["content"] == exact_content
    assert "navigation only" not in report_history[-1].message


def test_regulatory_report_history_compacts_exact_evidence_atomically() -> None:
    repeated_content = "controlling operative text " + ("x" * 4_000)
    additional_content = "independent exception text " + ("y" * 1_000)
    first_doc = _regulatory_search_doc(
        document_id="document-1",
        chunk_ind=4,
        chunk_identifier="chunk-1",
        heading="Article 4",
    )
    second_doc = _regulatory_search_doc(
        document_id="document-2",
        chunk_ind=8,
        chunk_identifier="chunk-2",
        heading="Article 8",
    )
    citation_processor = DynamicCitationProcessor(
        citation_mode=CitationMode.KEEP_MARKERS
    )
    citation_processor.update_citation_mapping(
        {1: first_doc, 2: first_doc, 3: second_doc}
    )
    evidence_chunks = [
        build_candidate_answer_evidence_chunk(
            document_id="document-1",
            chunk_id=1,
            citation_number=1,
            chunk_identifier="chunk-1",
            heading="Rule > Article 4",
            content=repeated_content,
        ),
        build_candidate_answer_evidence_chunk(
            document_id="document-1",
            chunk_id=1,
            citation_number=2,
            chunk_identifier="chunk-1",
            heading="Rule > Article 4",
            content=repeated_content,
        ),
        build_candidate_answer_evidence_chunk(
            document_id="document-2",
            chunk_id=2,
            citation_number=3,
            chunk_identifier="chunk-2",
            heading="Rule > Article 8",
            content=additional_content,
        ),
    ]
    calls = [
        _search_call("search-1", "first legal relationship"),
        _search_call("search-2", "distinct exception relationship"),
        _search_call("search-3", "remaining source gap"),
    ]

    def response(results: list[tuple[int, str, str]], *, evidence_target: str) -> str:
        return json.dumps(
            {
                "receipt": {
                    "coverage_item": "focused issue",
                    "evidence_target": evidence_target,
                },
                "results": [
                    {
                        "document": citation_number,
                        "title": title,
                        "content": content,
                        "metadata": json.dumps(
                            {
                                "regulatory_chunk_id": (
                                    "chunk-1"
                                    if citation_number in {1, 2}
                                    else "chunk-2"
                                )
                            }
                        ),
                    }
                    for citation_number, title, content in results
                ],
                "regulatory_provision_navigation": {
                    "usage_note": "navigation only",
                    "headings": ["z" * 2_000],
                },
            },
            ensure_ascii=False,
        )

    responses = [
        response([(1, "Article 4", repeated_content)], evidence_target="rule"),
        response(
            [
                (2, "Article 4", repeated_content),
                (3, "Article 8", additional_content),
            ],
            evidence_target="exception",
        ),
        response([], evidence_target="remaining gap"),
    ]
    topic_message = ChatMessageSimple(
        message="focused regulatory topic",
        token_count=len("focused regulatory topic"),
        message_type=MessageType.USER,
    )
    history: list[ChatMessageSimple] = [topic_message]
    for call, tool_response in zip(calls, responses):
        tool_call_text = call.to_msg_str()
        history.extend(
            [
                ChatMessageSimple(
                    message="",
                    token_count=len(tool_call_text),
                    message_type=MessageType.ASSISTANT,
                    tool_calls=[
                        ToolCallSimple(
                            tool_call_id=call.tool_call_id,
                            tool_name=call.tool_name,
                            tool_arguments=call.tool_args,
                            token_count=len(tool_call_text),
                        )
                    ],
                ),
                ChatMessageSimple(
                    message=tool_response,
                    token_count=len(tool_response),
                    message_type=MessageType.TOOL_CALL_RESPONSE,
                    tool_call_id=call.tool_call_id,
                ),
            ]
        )
    original_messages = [message.message for message in history]

    compacted = research_agent._compact_regulatory_report_history(
        research_topic=topic_message.message,
        history=history,
        exact_evidence_chunks=evidence_chunks,
        citation_processor=citation_processor,
        token_counter=len,
    )

    assert compacted is not history
    assert [message.message for message in history] == original_messages
    assert [message.message_type for message in compacted] == [
        MessageType.USER,
        MessageType.ASSISTANT,
        MessageType.TOOL_CALL_RESPONSE,
    ]
    compact_call = compacted[1].tool_calls
    assert compact_call is not None and len(compact_call) == 1
    assert compact_call[0].tool_call_id == compacted[2].tool_call_id
    constructed = construct_message_history(
        system_prompt=None,
        custom_agent_prompt=None,
        simple_chat_history=compacted,
        reminder_message=None,
        context_files=None,
        available_tokens=100_000,
    )
    translated = translate_history_to_llm_format(
        history=constructed,
        llm_config=LLMConfig(
            model_provider=LlmProviderNames.OPENAI,
            model_name="provider-compatibility-test",
            temperature=0,
            max_input_tokens=100_000,
        ),
    )
    assert isinstance(translated, list)
    translated_assistant = translated[1]
    translated_response = translated[2]
    assert isinstance(translated_assistant, AssistantMessage)
    assert translated_assistant.tool_calls is not None
    assert translated_assistant.tool_calls[0].id == compacted[2].tool_call_id
    assert isinstance(translated_response, ToolMessage)
    assert translated_response.tool_call_id == compacted[2].tool_call_id
    compact_payload = json.loads(compacted[2].message)
    assert [result["document"] for result in compact_payload["results"]] == [1, 3]
    assert compacted[2].message.count(repeated_content) == 1
    assert compact_payload["search_attempts"][1]["returned_citation_numbers"] == [1, 3]
    assert compact_payload["search_attempts"][2]["status"] == "zero_results"
    assert "zero results do not prove" in compact_payload["usage_note"]
    assert sum(message.token_count for message in compacted) < sum(
        message.token_count for message in history
    )


def test_regulatory_report_history_falls_back_on_untrusted_compaction_input() -> None:
    search_call = _search_call("search-1", "focused query")
    history = [
        ChatMessageSimple(
            message="focused topic",
            token_count=13,
            message_type=MessageType.USER,
        ),
        ChatMessageSimple(
            message="",
            token_count=1,
            message_type=MessageType.ASSISTANT,
            tool_calls=[
                ToolCallSimple(
                    tool_call_id=search_call.tool_call_id,
                    tool_name=search_call.tool_name,
                    tool_arguments=search_call.tool_args,
                    token_count=1,
                )
            ],
        ),
        ChatMessageSimple(
            message="not valid search JSON",
            token_count=21,
            message_type=MessageType.TOOL_CALL_RESPONSE,
            tool_call_id=search_call.tool_call_id,
        ),
    ]
    search_doc = _regulatory_search_doc(
        document_id="document-1",
        chunk_ind=1,
        chunk_identifier="chunk-1",
        heading="Article 1",
    )
    citation_processor = DynamicCitationProcessor()
    citation_processor.update_citation_mapping({1: search_doc})
    evidence = build_candidate_answer_evidence_chunk(
        document_id="document-1",
        chunk_id=1,
        citation_number=1,
        chunk_identifier="chunk-1",
        heading="Rule > Article 1",
        content="operative text",
    )

    assert (
        research_agent._compact_regulatory_report_history(
            research_topic="focused topic",
            history=history,
            exact_evidence_chunks=[evidence],
            citation_processor=citation_processor,
            token_counter=len,
        )
        is history
    )
    assert (
        research_agent._compact_regulatory_report_history(
            research_topic="focused topic",
            history=history,
            exact_evidence_chunks=[],
            citation_processor=citation_processor,
            token_counter=len,
        )
        is history
    )

    exact_content = "operative text " + ("x" * 4_000)
    trusted_payload = {
        "receipt": {
            "coverage_item": "focused legal issue",
            "evidence_target": "controlling rule and material qualifications",
        },
        "results": [
            {
                "document": 1,
                "title": "Article 1",
                "content": exact_content,
                "metadata": json.dumps({"regulatory_chunk_id": "chunk-1"}),
            }
        ],
        "regulatory_provision_navigation": {"headings": ["y" * 8_000]},
    }
    trusted_response = json.dumps(trusted_payload)
    trusted_history = [
        history[0],
        history[1].model_copy(
            update={
                "token_count": len(search_call.to_msg_str()),
                "tool_calls": [
                    ToolCallSimple(
                        tool_call_id=search_call.tool_call_id,
                        tool_name=search_call.tool_name,
                        tool_arguments=search_call.tool_args,
                        token_count=len(search_call.to_msg_str()),
                    )
                ],
            }
        ),
        history[2].model_copy(
            update={"message": trusted_response, "token_count": len(trusted_response)}
        ),
    ]
    trusted_evidence = build_candidate_answer_evidence_chunk(
        document_id="document-1",
        chunk_id=1,
        citation_number=1,
        chunk_identifier="chunk-1",
        heading="Rule > Article 1",
        content=exact_content,
    )
    assert (
        research_agent._compact_regulatory_report_history(
            research_topic="focused topic",
            history=trusted_history,
            exact_evidence_chunks=[trusted_evidence],
            citation_processor=citation_processor,
            token_counter=len,
        )
        is not trusted_history
    )

    forged_evidence = trusted_evidence.model_copy(
        update={"content": "different sidecar text"}
    )
    assert (
        research_agent._compact_regulatory_report_history(
            research_topic="focused topic",
            history=trusted_history,
            exact_evidence_chunks=[forged_evidence],
            citation_processor=citation_processor,
            token_counter=len,
        )
        is trusted_history
    )

    missing_receipt_payload = dict(trusted_payload)
    missing_receipt_payload.pop("receipt")
    missing_receipt_response = json.dumps(missing_receipt_payload)
    missing_receipt_history = [
        trusted_history[0],
        trusted_history[1],
        trusted_history[2].model_copy(
            update={
                "message": missing_receipt_response,
                "token_count": len(missing_receipt_response),
            }
        ),
    ]
    assert (
        research_agent._compact_regulatory_report_history(
            research_topic="focused topic",
            history=missing_receipt_history,
            exact_evidence_chunks=[trusted_evidence],
            citation_processor=citation_processor,
            token_counter=len,
        )
        is missing_receipt_history
    )

    truncated_evidence = trusted_evidence.model_copy(update={"content_truncated": True})
    assert (
        research_agent._compact_regulatory_report_history(
            research_topic="focused topic",
            history=trusted_history,
            exact_evidence_chunks=[truncated_evidence],
            citation_processor=citation_processor,
            token_counter=len,
        )
        is trusted_history
    )


def test_non_regulatory_research_does_not_add_search_evidence_cap() -> None:
    assert (
        research_agent._regulatory_search_llm_chunk_cap([_make_search_tool()]) is None
    )


def test_regulatory_search_novelty_tracks_chunk_versions_by_stable_id() -> None:
    seen: set[research_agent._SearchResultIdentity] = set()

    assert research_agent._update_regulatory_search_result_novelty(
        _search_response("Provision A", "first text"),
        seen_result_identities=seen,
    ) == (1, 0)
    assert research_agent._update_regulatory_search_result_novelty(
        _search_response("Provision A", "first text"),
        seen_result_identities=seen,
    ) == (0, 1)
    assert research_agent._update_regulatory_search_result_novelty(
        _search_response("Provision A", "updated text"),
        seen_result_identities=seen,
    ) == (1, 0)
    assert research_agent._update_regulatory_search_result_novelty(
        _search_response("Provision B", "updated text"),
        seen_result_identities=seen,
    ) == (1, 0)


def test_regulatory_research_surfaces_local_repeat_novelty_once() -> None:
    search_calls = [
        _search_call("search-1", "first focused query"),
        _search_call("search-2", "materially different focused query"),
    ]
    report_call = ToolCallKickoff(
        tool_call_id="report",
        tool_name=GENERATE_REPORT_TOOL_NAME,
        tool_args={},
        placement=Placement(turn_index=0),
    )
    repeated_response = _search_response("Same provision", "same evidence")

    def fake_run_tool_calls(**kwargs: Any) -> ParallelToolCallResponse:
        tool_call = cast(list[ToolCallKickoff], kwargs["tool_calls"])[0]
        return ParallelToolCallResponse(
            tool_responses=[
                ToolResponse(
                    rich_response=None,
                    llm_facing_response=repeated_response,
                    tool_call=tool_call,
                )
            ],
            updated_citation_mapping={},
        )

    with (
        patch.object(
            research_agent,
            "run_llm_step",
            side_effect=[
                _llm_step(search_calls[0]),
                _llm_step(search_calls[1]),
                _llm_step(report_call),
            ],
        ) as run_llm_step,
        patch.object(
            research_agent,
            "run_tool_calls",
            side_effect=fake_run_tool_calls,
        ),
        patch.object(
            research_agent,
            "generate_intermediate_report",
            return_value="report",
        ) as generate_report,
    ):
        result = research_agent.run_research_agent_call(
            research_agent_call=_research_agent_call("focused topic"),
            parent_tool_call_id="parent",
            tools=[_make_search_tool(regulatory_chunks_only=True)],
            emitter=MagicMock(),
            state_container=ChatStateContainer(),
            llm=_llm(),
            is_reasoning_model=True,
            token_counter=len,
            user_identity=None,
        )

    assert result is not None
    second_decision_history = cast(
        list[ChatMessageSimple], run_llm_step.call_args_list[1].kwargs["history"]
    )
    assert "Retrieval novelty" not in "\n".join(
        message.message for message in second_decision_history
    )
    third_decision_history = cast(
        list[ChatMessageSimple], run_llm_step.call_args_list[2].kwargs["history"]
    )
    third_decision_text = "\n".join(
        message.message for message in third_decision_history
    )
    assert "Retrieval novelty" in third_decision_text
    assert "0 previously unseen regulatory evidence chunk version(s)" in (
        third_decision_text
    )
    assert "1 exact result(s) already seen" in third_decision_text

    canonical_history = cast(
        list[ChatMessageSimple], generate_report.call_args.kwargs["history"]
    )
    assert [
        message.message
        for message in canonical_history
        if message.message_type == MessageType.TOOL_CALL_RESPONSE
    ] == [repeated_response, repeated_response]


def test_regulatory_research_skips_exact_duplicate_search_with_paired_feedback() -> (
    None
):
    first_search = _search_call(
        "search-first",
        "  MODEL-WRITTEN   QUERY  ",
    )
    duplicate_search = _search_call(
        "search-duplicate",
        "model-written query",
    )
    report_call = ToolCallKickoff(
        tool_call_id="report",
        tool_name=GENERATE_REPORT_TOOL_NAME,
        tool_args={},
        placement=Placement(turn_index=0),
    )
    executed_call_ids: list[str] = []

    def fake_run_tool_calls(**kwargs: Any) -> ParallelToolCallResponse:
        tool_call = cast(list[ToolCallKickoff], kwargs["tool_calls"])[0]
        executed_call_ids.append(tool_call.tool_call_id)
        return ParallelToolCallResponse(
            tool_responses=[
                ToolResponse(
                    rich_response=None,
                    llm_facing_response=_search_response(
                        "Provision A", "material evidence"
                    ),
                    tool_call=tool_call,
                )
            ],
            updated_citation_mapping={},
        )

    with (
        patch.object(
            research_agent,
            "run_llm_step",
            side_effect=[
                _llm_step(first_search),
                _llm_step(duplicate_search),
                _llm_step(report_call),
            ],
        ) as run_llm_step,
        patch.object(
            research_agent,
            "run_tool_calls",
            side_effect=fake_run_tool_calls,
        ) as run_tool_calls,
        patch.object(
            research_agent,
            "generate_intermediate_report",
            return_value="report",
        ),
    ):
        result = research_agent.run_research_agent_call(
            research_agent_call=_research_agent_call("focused topic"),
            parent_tool_call_id="parent",
            tools=[_make_search_tool(regulatory_chunks_only=True)],
            emitter=MagicMock(),
            state_container=ChatStateContainer(),
            llm=_llm(),
            is_reasoning_model=True,
            token_counter=len,
            user_identity=None,
        )

    assert result is not None
    assert executed_call_ids == ["search-first"]
    assert run_tool_calls.call_count == 1
    third_decision_history = cast(
        list[ChatMessageSimple], run_llm_step.call_args_list[2].kwargs["history"]
    )
    duplicate_assistant_index = next(
        index
        for index, message in enumerate(third_decision_history)
        if message.message_type == MessageType.ASSISTANT
        and message.tool_calls is not None
        and any(
            tool_call.tool_call_id == "search-duplicate"
            for tool_call in message.tool_calls
        )
    )
    duplicate_assistant = third_decision_history[duplicate_assistant_index]
    duplicate_feedback = third_decision_history[duplicate_assistant_index + 1]
    assert duplicate_assistant.tool_calls is not None
    assert duplicate_feedback.message_type == MessageType.TOOL_CALL_RESPONSE
    assert duplicate_feedback.tool_call_id == "search-duplicate"
    assert "was not executed" in duplicate_feedback.message
    assert "materially different query or search_mode" in duplicate_feedback.message


def test_regulatory_research_defers_extra_same_decision_searches_with_feedback() -> (
    None
):
    first_search = _search_call("search-first", "first focused query")
    extra_search = _search_call("search-extra", "second focused query")
    report_call = ToolCallKickoff(
        tool_call_id="report",
        tool_name=GENERATE_REPORT_TOOL_NAME,
        tool_args={},
        placement=Placement(turn_index=0),
    )
    multi_search_decision = (
        LlmStepResult(
            reasoning=None,
            answer=None,
            tool_calls=[first_search, extra_search],
        ),
        False,
    )
    executed_call_ids: list[str] = []

    def fake_run_tool_calls(**kwargs: Any) -> ParallelToolCallResponse:
        tool_call = cast(list[ToolCallKickoff], kwargs["tool_calls"])[0]
        executed_call_ids.append(tool_call.tool_call_id)
        return ParallelToolCallResponse(
            tool_responses=[
                ToolResponse(
                    rich_response=None,
                    llm_facing_response=_search_response(
                        "Provision A", "material evidence"
                    ),
                    tool_call=tool_call,
                )
            ],
            updated_citation_mapping={},
        )

    with (
        patch.object(
            research_agent,
            "run_llm_step",
            side_effect=[multi_search_decision, _llm_step(report_call)],
        ) as run_llm_step,
        patch.object(
            research_agent,
            "run_tool_calls",
            side_effect=fake_run_tool_calls,
        ),
        patch.object(
            research_agent,
            "generate_intermediate_report",
            return_value="report",
        ),
    ):
        result = research_agent.run_research_agent_call(
            research_agent_call=_research_agent_call("focused topic"),
            parent_tool_call_id="parent",
            tools=[_make_search_tool(regulatory_chunks_only=True)],
            emitter=MagicMock(),
            state_container=ChatStateContainer(),
            llm=_llm(),
            is_reasoning_model=True,
            token_counter=len,
            user_identity=None,
        )

    assert result is not None
    assert executed_call_ids == ["search-first"]
    second_decision_history = cast(
        list[ChatMessageSimple], run_llm_step.call_args_list[1].kwargs["history"]
    )
    second_decision_text = "\n".join(
        message.message for message in second_decision_history
    )
    assert "Only the first internal_search call" in second_decision_text
    assert "1 additional retrieval call(s) were deferred" in second_decision_text


def test_regulatory_research_allows_different_mode_and_rephrased_searches() -> None:
    search_calls = [
        _search_call("search-hybrid", "same focused query"),
        _search_call(
            "search-full-text",
            " SAME   FOCUSED QUERY ",
            search_mode="full_text",
        ),
        _search_call(
            "search-rephrased",
            "same focused query with exception",
            search_mode="full_text",
        ),
    ]
    report_call = ToolCallKickoff(
        tool_call_id="report",
        tool_name=GENERATE_REPORT_TOOL_NAME,
        tool_args={},
        placement=Placement(turn_index=0),
    )
    executed_call_ids: list[str] = []

    def fake_run_tool_calls(**kwargs: Any) -> ParallelToolCallResponse:
        tool_call = cast(list[ToolCallKickoff], kwargs["tool_calls"])[0]
        executed_call_ids.append(tool_call.tool_call_id)
        return ParallelToolCallResponse(
            tool_responses=[
                ToolResponse(
                    rich_response=None,
                    llm_facing_response=_search_response(
                        tool_call.tool_call_id,
                        f"evidence for {tool_call.tool_call_id}",
                    ),
                    tool_call=tool_call,
                )
            ],
            updated_citation_mapping={},
        )

    with (
        patch.object(
            research_agent,
            "run_llm_step",
            side_effect=[
                *[_llm_step(search_call) for search_call in search_calls],
                _llm_step(report_call),
            ],
        ),
        patch.object(
            research_agent,
            "run_tool_calls",
            side_effect=fake_run_tool_calls,
        ),
        patch.object(
            research_agent,
            "generate_intermediate_report",
            return_value="report",
        ),
    ):
        result = research_agent.run_research_agent_call(
            research_agent_call=_research_agent_call("focused topic"),
            parent_tool_call_id="parent",
            tools=[_make_search_tool(regulatory_chunks_only=True)],
            emitter=MagicMock(),
            state_container=ChatStateContainer(),
            llm=_llm(),
            is_reasoning_model=True,
            token_counter=len,
            user_identity=None,
        )

    assert result is not None
    assert executed_call_ids == [
        "search-hybrid",
        "search-full-text",
        "search-rephrased",
    ]


def test_non_regulatory_research_still_executes_exact_duplicate_searches() -> None:
    first_search = _search_call("search-first", "same query")
    duplicate_search = _search_call("search-duplicate", " SAME   QUERY ")
    report_call = ToolCallKickoff(
        tool_call_id="report",
        tool_name=GENERATE_REPORT_TOOL_NAME,
        tool_args={},
        placement=Placement(turn_index=0),
    )
    executed_call_ids: list[str] = []

    def fake_run_tool_calls(**kwargs: Any) -> ParallelToolCallResponse:
        tool_call = cast(list[ToolCallKickoff], kwargs["tool_calls"])[0]
        executed_call_ids.append(tool_call.tool_call_id)
        return ParallelToolCallResponse(
            tool_responses=[
                ToolResponse(
                    rich_response=None,
                    llm_facing_response=_search_response(
                        tool_call.tool_call_id,
                        f"evidence for {tool_call.tool_call_id}",
                    ),
                    tool_call=tool_call,
                )
            ],
            updated_citation_mapping={},
        )

    with (
        patch.object(
            research_agent,
            "run_llm_step",
            side_effect=[
                _llm_step(first_search),
                _llm_step(duplicate_search),
                _llm_step(report_call),
            ],
        ),
        patch.object(
            research_agent,
            "run_tool_calls",
            side_effect=fake_run_tool_calls,
        ),
        patch.object(
            research_agent,
            "generate_intermediate_report",
            return_value="report",
        ),
    ):
        result = research_agent.run_research_agent_call(
            research_agent_call=_research_agent_call("focused topic"),
            parent_tool_call_id="parent",
            tools=[_make_search_tool()],
            emitter=MagicMock(),
            state_container=ChatStateContainer(),
            llm=_llm(),
            is_reasoning_model=True,
            token_counter=len,
            user_identity=None,
        )

    assert result is not None
    assert executed_call_ids == ["search-first", "search-duplicate"]


def test_research_decision_projects_only_older_local_search_results() -> None:
    research_topic = "LOCAL_RESEARCH_TOPIC_SENTINEL"
    sibling_topic = "UNRELATED_SIBLING_TOPIC_SENTINEL"
    older_tail = "OLDER_FULL_TAIL_SENTINEL"
    latest_tail = "LATEST_FULL_TAIL_SENTINEL"
    older_response = _search_response(
        "Older provision",
        "older evidence " + ("x" * 4_000) + older_tail,
    )
    latest_response = _search_response(
        "Latest controlling provision",
        "latest evidence " + ("y" * 400) + latest_tail,
    )
    search_calls = [
        _search_call("search-1", "first focused query"),
        _search_call("search-2", "second focused query"),
    ]
    report_call = ToolCallKickoff(
        tool_call_id="report",
        tool_name=GENERATE_REPORT_TOOL_NAME,
        tool_args={},
        placement=Placement(turn_index=0),
    )
    response_iter = iter([older_response, latest_response])

    def fake_run_tool_calls(**kwargs: Any) -> ParallelToolCallResponse:
        tool_call = cast(list[ToolCallKickoff], kwargs["tool_calls"])[0]
        return ParallelToolCallResponse(
            tool_responses=[
                ToolResponse(
                    rich_response=None,
                    llm_facing_response=next(response_iter),
                    tool_call=tool_call,
                )
            ],
            updated_citation_mapping={},
        )

    with (
        patch.object(
            research_agent,
            "run_llm_step",
            side_effect=[
                _llm_step(search_calls[0]),
                _llm_step(search_calls[1]),
                _llm_step(report_call),
            ],
        ) as run_llm_step,
        patch.object(
            research_agent,
            "run_tool_calls",
            side_effect=fake_run_tool_calls,
        ),
        patch.object(
            research_agent,
            "generate_intermediate_report",
            return_value="report",
        ) as generate_report,
    ):
        result = research_agent.run_research_agent_call(
            research_agent_call=_research_agent_call(research_topic),
            parent_tool_call_id="parent",
            tools=[_make_search_tool()],
            emitter=MagicMock(),
            state_container=ChatStateContainer(),
            llm=_llm(),
            is_reasoning_model=True,
            token_counter=len,
            user_identity=None,
        )

    assert result is not None
    assert result.intermediate_report == "report"
    assert generate_report.call_args.kwargs["max_tokens"] == (
        research_agent.MAX_INTERMEDIATE_REPORT_LENGTH_TOKENS
    )

    decision_history = cast(
        list[ChatMessageSimple], run_llm_step.call_args_list[2].kwargs["history"]
    )
    decision_text = "\n".join(message.message for message in decision_history)
    assert research_topic in decision_text
    assert sibling_topic not in decision_text
    assert older_tail not in decision_text
    assert latest_tail in decision_text

    projected_older_message = next(
        message
        for message in decision_history
        if message.tool_call_id == search_calls[0].tool_call_id
    )
    projected_older_payload = json.loads(projected_older_message.message)
    assert projected_older_payload["results"] == []
    assert projected_older_payload["search_result_inventory"][0]["title"] == (
        "Older provision"
    )
    assert (
        "older evidence"
        in projected_older_payload["search_result_inventory"][0]["decision_excerpt"]
    )
    assert projected_older_message.token_count < len(older_response)

    latest_message = next(
        message
        for message in decision_history
        if message.tool_call_id == search_calls[1].tool_call_id
    )
    assert latest_message.message == latest_response

    canonical_history = cast(
        list[ChatMessageSimple], generate_report.call_args.kwargs["history"]
    )
    canonical_responses = {
        message.tool_call_id: message.message
        for message in canonical_history
        if message.message_type == MessageType.TOOL_CALL_RESPONSE
    }
    assert canonical_responses[search_calls[0].tool_call_id] == older_response
    assert canonical_responses[search_calls[1].tool_call_id] == latest_response
    assert older_tail in canonical_responses[search_calls[0].tool_call_id]
    assert latest_tail in canonical_responses[search_calls[1].tool_call_id]


def test_research_decision_does_not_expand_a_small_search_response() -> None:
    older_response = _search_response("Older provision", "short evidence")
    latest_response = _search_response("Latest provision", "new evidence")
    older_call = _search_call("search-1", "first query")
    latest_call = _search_call("search-2", "second query")
    history = [
        ChatMessageSimple(
            message="",
            token_count=0,
            message_type=MessageType.ASSISTANT,
            tool_calls=[
                ToolCallSimple(
                    tool_call_id=older_call.tool_call_id,
                    tool_name=older_call.tool_name,
                    tool_arguments=older_call.tool_args,
                    token_count=0,
                )
            ],
        ),
        ChatMessageSimple(
            message=older_response,
            token_count=len(older_response),
            message_type=MessageType.TOOL_CALL_RESPONSE,
            tool_call_id=older_call.tool_call_id,
        ),
        ChatMessageSimple(
            message="",
            token_count=0,
            message_type=MessageType.ASSISTANT,
            tool_calls=[
                ToolCallSimple(
                    tool_call_id=latest_call.tool_call_id,
                    tool_name=latest_call.tool_name,
                    tool_arguments=latest_call.tool_args,
                    token_count=0,
                )
            ],
        ),
        ChatMessageSimple(
            message=latest_response,
            token_count=len(latest_response),
            message_type=MessageType.TOOL_CALL_RESPONSE,
            tool_call_id=latest_call.tool_call_id,
        ),
    ]

    projected, projected_result_count = (
        research_agent._project_research_history_for_tool_decision(
            history,
            token_counter=len,
        )
    )

    assert projected_result_count == 0
    assert projected == history


def test_research_decision_compacts_old_inventory_without_mutating_history() -> None:
    source_identity = (
        "1975_tir_sozlesmesi_source_identity_" + ("source-segment-" * 10) + ".docx"
    )
    terminal_heading = "MADDE 4A - Taraflar arasindaki belirli tehlikeli atik hareketi"
    long_title = (
        f"{source_identity} — DOCUMENT_TITLE > "
        + ("MIDDLE_PATH_SENTINEL > " * 20)
        + terminal_heading
    )
    older_response = json.dumps(
        {
            "receipt": {
                "coverage_item": "focused legal issue",
                "evidence_target": "unknown controlling provision",
            },
            "note": "Repeated scope and query narration that is not legal evidence.",
            "results": [
                {
                    "document": 1,
                    "title": long_title,
                    "updated_at": "2026-08-01T00:00:00+00:00",
                    "source_type": "user_file",
                    "content": "operative evidence " + ("x" * 4_000),
                    "metadata": json.dumps(
                        {
                            "regulatory_chunk_id": "chunk-article-4a",
                            "regulatory_heading_path": [
                                "DOCUMENT_TITLE",
                                "MIDDLE_PATH_SENTINEL",
                                terminal_heading,
                            ],
                        }
                    ),
                }
            ],
        },
        ensure_ascii=False,
    )
    latest_response = _search_response("Latest provision", "latest exact evidence")
    search_calls = [
        _search_call("search-old", "unknown prohibition relationship"),
        _search_call("search-latest", "discovered source identity"),
    ]
    history: list[ChatMessageSimple] = []
    for search_call, response in zip(
        search_calls, [older_response, latest_response], strict=True
    ):
        history.extend(
            [
                ChatMessageSimple(
                    message="",
                    token_count=0,
                    message_type=MessageType.ASSISTANT,
                    tool_calls=[
                        ToolCallSimple(
                            tool_call_id=search_call.tool_call_id,
                            tool_name=search_call.tool_name,
                            tool_arguments=search_call.tool_args,
                            token_count=0,
                        )
                    ],
                ),
                ChatMessageSimple(
                    message=response,
                    token_count=len(response),
                    message_type=MessageType.TOOL_CALL_RESPONSE,
                    tool_call_id=search_call.tool_call_id,
                ),
            ]
        )
    canonical_messages = [message.message for message in history]

    projected, projected_result_count = (
        research_agent._project_research_history_for_tool_decision(
            history,
            token_counter=len,
        )
    )

    assert projected_result_count == 1
    older_projected_message = next(
        message
        for message in projected
        if message.tool_call_id == search_calls[0].tool_call_id
    )
    older_payload = json.loads(older_projected_message.message)
    inventory_item = older_payload["search_result_inventory"][0]
    compact_title = inventory_item["title"]
    assert len(compact_title) <= research_agent._OLDER_SEARCH_DECISION_TITLE_CHARS
    assert compact_title.startswith("1975_tir_sozlesmesi_source_identity_")
    assert compact_title.endswith(terminal_heading)
    assert "MIDDLE_PATH_SENTINEL > MIDDLE_PATH_SENTINEL" not in compact_title
    assert "updated_at" not in inventory_item
    assert "source_type" not in inventory_item
    assert "receipt" not in older_payload
    assert "note" not in older_payload
    assert "note" not in older_payload["history_compaction"]

    latest_projected_message = next(
        message
        for message in projected
        if message.tool_call_id == search_calls[1].tool_call_id
    )
    assert latest_projected_message.message == latest_response
    assert [message.message for message in history] == canonical_messages


def test_research_decision_deduplicates_old_results_globally_newest_first() -> None:
    def result(
        document: int,
        title: str,
        content: str,
        chunk_id: str | None,
    ) -> dict[str, object]:
        item: dict[str, object] = {
            "document": document,
            "title": title,
            "content": content,
        }
        if chunk_id is not None:
            item["metadata"] = json.dumps(
                {
                    "regulatory_chunk_id": chunk_id,
                    "regulatory_heading_path": [title],
                }
            )
        return item

    fallback_content = "fallback exact evidence " + ("f" * 4_000)
    responses = [
        json.dumps(
            {
                "results": [
                    result(1, "Older A", "old A " + ("a" * 4_000), "chunk-a"),
                    result(2, "Older B", "old B " + ("b" * 4_000), "chunk-b"),
                    result(3, "Fallback X", fallback_content, None),
                ]
            }
        ),
        json.dumps(
            {
                "results": [
                    result(4, "Newer A", "new A " + ("n" * 4_000), "chunk-a"),
                    result(5, "Only C", "only C " + ("c" * 4_000), "chunk-c"),
                    result(6, "Fallback X", fallback_content, None),
                ]
            }
        ),
        json.dumps(
            {
                "results": [
                    result(7, "Latest B", "latest B", "chunk-b"),
                    result(8, "Latest D", "latest D", "chunk-d"),
                ]
            }
        ),
    ]
    search_calls = [
        _search_call(f"search-{index}", f"query {index}") for index in range(3)
    ]
    history: list[ChatMessageSimple] = []
    for search_call, response in zip(search_calls, responses, strict=True):
        history.extend(
            [
                ChatMessageSimple(
                    message="",
                    token_count=0,
                    message_type=MessageType.ASSISTANT,
                    tool_calls=[
                        ToolCallSimple(
                            tool_call_id=search_call.tool_call_id,
                            tool_name=search_call.tool_name,
                            tool_arguments=search_call.tool_args,
                            token_count=0,
                        )
                    ],
                ),
                ChatMessageSimple(
                    message=response,
                    token_count=len(response),
                    message_type=MessageType.TOOL_CALL_RESPONSE,
                    tool_call_id=search_call.tool_call_id,
                ),
            ]
        )
    canonical_messages = [message.message for message in history]

    projected, projected_result_count = (
        research_agent._project_research_history_for_tool_decision(
            history,
            token_counter=len,
        )
    )

    assert projected_result_count == 6
    projected_payloads = {
        message.tool_call_id: json.loads(message.message)
        for message in projected
        if message.message_type == MessageType.TOOL_CALL_RESPONSE
    }
    oldest_payload = projected_payloads[search_calls[0].tool_call_id]
    newer_payload = projected_payloads[search_calls[1].tool_call_id]
    assert oldest_payload["search_result_inventory"] == []
    assert (
        oldest_payload["history_compaction"][
            "duplicate_results_omitted_for_research_decision"
        ]
        == 3
    )
    assert {
        item.get("regulatory_chunk_id")
        for item in newer_payload["search_result_inventory"]
        if item.get("regulatory_chunk_id") is not None
    } == {"chunk-a", "chunk-c"}
    assert [item["title"] for item in newer_payload["search_result_inventory"]] == [
        "Newer A",
        "Only C",
        "Fallback X",
    ]
    assert projected_payloads[search_calls[2].tool_call_id] == json.loads(responses[2])
    assert [message.message for message in history] == canonical_messages


def test_research_decision_deduplicates_older_navigation_newest_first() -> None:
    search_calls = [
        _search_call(f"search-{index}", f"query {index}") for index in range(1, 5)
    ]
    responses = [
        _search_response_with_navigation(
            document_title="Other Rule",
            content="other evidence " + ("o" * 4_000),
            headings=[("article:2", "Other Rule Article 2")],
        ),
        _search_response_with_navigation(
            document_title="Shared Rule",
            content="old evidence " + ("a" * 4_000),
            headings=[
                ("article:1", "Article 1"),
                ("article:2", "Older Article 2 label"),
            ],
        ),
        _search_response_with_navigation(
            document_title="Shared Rule",
            content="newer evidence " + ("b" * 4_000),
            headings=[
                ("article:2", "Newer Article 2 label"),
                ("article:3", "Article 3"),
            ],
        ),
        _search_response_with_navigation(
            document_title="Shared Rule",
            content="latest evidence",
            headings=[
                ("article:3", "Latest Article 3 label"),
                ("article:4", "Article 4"),
            ],
        ),
    ]
    history: list[ChatMessageSimple] = []
    for search_call, response in zip(search_calls, responses):
        history.extend(
            [
                ChatMessageSimple(
                    message="",
                    token_count=0,
                    message_type=MessageType.ASSISTANT,
                    tool_calls=[
                        ToolCallSimple(
                            tool_call_id=search_call.tool_call_id,
                            tool_name=search_call.tool_name,
                            tool_arguments=search_call.tool_args,
                            token_count=0,
                        )
                    ],
                ),
                ChatMessageSimple(
                    message=response,
                    token_count=len(response),
                    message_type=MessageType.TOOL_CALL_RESPONSE,
                    tool_call_id=search_call.tool_call_id,
                ),
            ]
        )
    canonical_messages = [message.message for message in history]

    projected, projected_result_count = (
        research_agent._project_research_history_for_tool_decision(
            history,
            token_counter=len,
        )
    )

    assert projected_result_count == 3
    projected_payloads = {
        message.tool_call_id: json.loads(message.message)
        for message in projected
        if message.message_type == MessageType.TOOL_CALL_RESPONSE
    }
    assert projected_payloads[search_calls[0].tool_call_id][
        "regulatory_provision_navigation"
    ]["headings"] == [
        {"article_key": "article:2", "heading_label": "Other Rule Article 2"}
    ]
    assert projected_payloads[search_calls[1].tool_call_id][
        "regulatory_provision_navigation"
    ]["headings"] == [{"article_key": "article:1", "heading_label": "Article 1"}]
    assert projected_payloads[search_calls[2].tool_call_id][
        "regulatory_provision_navigation"
    ]["headings"] == [
        {"article_key": "article:2", "heading_label": "Newer Article 2 label"}
    ]
    duplicate_count_key = (
        "duplicate_regulatory_navigation_headings_omitted_for_research_decision"
    )
    assert (
        projected_payloads[search_calls[1].tool_call_id]["history_compaction"][
            duplicate_count_key
        ]
        == 1
    )
    assert (
        projected_payloads[search_calls[2].tool_call_id]["history_compaction"][
            duplicate_count_key
        ]
        == 1
    )

    latest_message = next(
        message
        for message in projected
        if message.tool_call_id == search_calls[-1].tool_call_id
    )
    assert latest_message.message == responses[-1]
    assert [message.message for message in history] == canonical_messages


def test_parallel_research_agents_get_distinct_search_tools_and_task_contexts() -> None:
    first_topic = "FIRST_AGENT_ONLY_SENTINEL"
    second_topic = "SECOND_AGENT_ONLY_SENTINEL"
    original_tool = _make_search_tool()
    original_tool._search_cycles.append(MagicMock())
    captured_jobs: list[
        tuple[Callable[..., ResearchAgentCallResult | None], tuple[Any, ...]]
    ] = []
    decision_histories: list[list[ChatMessageSimple]] = []

    def fake_run_in_parallel(
        functions_with_args: list[
            tuple[Callable[..., ResearchAgentCallResult | None], tuple[Any, ...]]
        ],
        **_: Any,
    ) -> list[ResearchAgentCallResult | None]:
        captured_jobs.extend(functions_with_args)
        return [function(*args) for function, args in functions_with_args]

    def fake_run_llm_step(**kwargs: Any) -> tuple[LlmStepResult, bool]:
        decision_histories.append(cast(list[ChatMessageSimple], kwargs["history"]))
        return _llm_step(
            ToolCallKickoff(
                tool_call_id="report",
                tool_name=GENERATE_REPORT_TOOL_NAME,
                tool_args={},
                placement=cast(Placement, kwargs["placement"]),
            )
        )

    with (
        patch.object(
            research_agent,
            "run_functions_tuples_in_parallel",
            side_effect=fake_run_in_parallel,
        ),
        patch.object(
            research_agent,
            "run_llm_step",
            side_effect=fake_run_llm_step,
        ),
        patch.object(
            research_agent,
            "generate_intermediate_report",
            return_value="report",
        ),
    ):
        result = research_agent.run_research_agent_calls(
            research_agent_calls=[
                _research_agent_call(first_topic, tab_index=0),
                _research_agent_call(second_topic, tab_index=1),
            ],
            parent_tool_call_ids=["parent-1", "parent-2"],
            tools=[original_tool],
            emitter=MagicMock(),
            state_container=ChatStateContainer(),
            llm=_llm(),
            is_reasoning_model=True,
            token_counter=len,
            citation_mapping={},
        )

    assert result.intermediate_reports == ["report", "report"]
    assert len(captured_jobs) == 2
    first_tool = cast(list[SearchTool], captured_jobs[0][1][2])[0]
    second_tool = cast(list[SearchTool], captured_jobs[1][1][2])[0]
    assert first_tool is not original_tool
    assert second_tool is not original_tool
    assert first_tool is not second_tool
    assert first_tool._search_cycles == []
    assert second_tool._search_cycles == []
    assert first_tool._shared_time_filter_decision is not (
        second_tool._shared_time_filter_decision
    )

    first_prompt = "\n".join(message.message for message in decision_histories[0])
    second_prompt = "\n".join(message.message for message in decision_histories[1])
    assert first_topic in first_prompt
    assert second_topic not in first_prompt
    assert second_topic in second_prompt
    assert first_topic not in second_prompt
    assert [
        message.message
        for message in decision_histories[0]
        if message.message_type == MessageType.USER
    ] == [first_topic]
    assert [
        message.message
        for message in decision_histories[1]
        if message.message_type == MessageType.USER
    ] == [second_topic]


def test_timed_out_research_agent_cannot_write_to_shared_output() -> None:
    live_emitter = MagicMock(spec=Emitter)
    live_state = ChatStateContainer()
    before_packet = MagicMock()
    after_packet = MagicMock()
    before_search_packet = MagicMock()
    after_search_packet = MagicMock()
    before_tool_call = MagicMock(spec=ToolCallInfo)
    after_tool_call = MagicMock(spec=ToolCallInfo)
    before_doc = _regulatory_search_doc(
        document_id="before-timeout",
        chunk_ind=1,
        chunk_identifier="before-timeout-chunk",
        heading="Before timeout",
    )
    after_doc = _regulatory_search_doc(
        document_id="after-timeout",
        chunk_ind=2,
        chunk_identifier="after-timeout-chunk",
        heading="After timeout",
    )

    def fake_run_in_parallel(
        functions_with_args: list[
            tuple[Callable[..., ResearchAgentCallResult | None], tuple[Any, ...]]
        ],
        **kwargs: Any,
    ) -> list[ResearchAgentCallResult]:
        function, args = functions_with_args[0]
        agent_emitter = cast(Emitter, args[3])
        agent_state = args[4]
        agent_search_tool = cast(list[SearchTool], args[2])[0]
        agent_emitter.emit(before_packet)
        agent_search_tool.emitter.emit(before_search_packet)
        agent_state.add_tool_call(before_tool_call)
        agent_state.add_search_docs([before_doc])

        timeout_callback = cast(
            Callable[..., ResearchAgentCallResult], kwargs["timeout_callback"]
        )
        timeout_result = timeout_callback(0, function, args)

        # This is what the still-running Python worker can do after the timeout.
        agent_emitter.emit(after_packet)
        agent_search_tool.emitter.emit(after_search_packet)
        agent_state.add_tool_call(after_tool_call)
        agent_state.add_search_docs([after_doc])
        return [timeout_result]

    with patch.object(
        research_agent,
        "run_functions_tuples_in_parallel",
        side_effect=fake_run_in_parallel,
    ):
        result = research_agent.run_research_agent_calls(
            research_agent_calls=[_research_agent_call("slow focused topic")],
            parent_tool_call_ids=["parent"],
            tools=[_make_search_tool()],
            emitter=live_emitter,
            state_container=live_state,
            llm=_llm(),
            is_reasoning_model=True,
            token_counter=len,
            citation_mapping={},
        )

    assert [emission.args[0] for emission in live_emitter.emit.call_args_list] == [
        before_packet,
        before_search_packet,
    ]
    assert live_state.get_tool_calls() == [before_tool_call]
    assert live_state.get_all_search_docs() == {
        (before_doc.document_id, before_doc.chunk_ind): before_doc
    }
    assert result.intermediate_reports == [
        research_agent.RESEARCH_AGENT_TIMEOUT_MESSAGE
    ]


def test_research_agent_call_and_parent_id_lengths_must_match() -> None:
    with pytest.raises(ValueError, match="must have equal lengths"):
        research_agent.run_research_agent_calls(
            research_agent_calls=[_research_agent_call("focused topic")],
            parent_tool_call_ids=[],
            tools=[],
            emitter=MagicMock(),
            state_container=ChatStateContainer(),
            llm=_llm(),
            is_reasoning_model=True,
            token_counter=len,
            citation_mapping={},
        )


def test_public_and_evidence_namespaces_reject_number_reassignment() -> None:
    public_doc = _regulatory_search_doc(
        document_id="public-document",
        chunk_ind=1,
        chunk_identifier="public-chunk",
        heading="Public",
    )
    different_evidence_doc = _regulatory_search_doc(
        document_id="evidence-document",
        chunk_ind=2,
        chunk_identifier="evidence-chunk",
        heading="Evidence",
    )

    with pytest.raises(ValueError, match="global citation 1"):
        research_agent._merge_citation_namespaces(
            {1: public_doc},
            {1: different_evidence_doc},
        )


def test_combined_research_keeps_hidden_evidence_in_global_namespace() -> None:
    prior_public_doc = _regulatory_search_doc(
        document_id="prior-public",
        chunk_ind=4,
        chunk_identifier="prior-public-chunk",
        heading="Prior public",
    )
    prior_evidence_doc = _regulatory_search_doc(
        document_id="prior-evidence",
        chunk_ind=5,
        chunk_identifier="prior-evidence-chunk",
        heading="Prior evidence",
    )
    cited_doc = _regulatory_search_doc(
        document_id="new-cited",
        chunk_ind=1,
        chunk_identifier="new-cited-chunk",
        heading="New cited",
    )
    hidden_doc = _regulatory_search_doc(
        document_id="new-hidden",
        chunk_ind=2,
        chunk_identifier="new-hidden-chunk",
        heading="New hidden",
    )
    local_result = ResearchAgentCallResult(
        intermediate_report="Supported [1].",
        citation_mapping={1: cited_doc},
        evidence_citation_mapping={1: cited_doc, 2: hidden_doc},
        exact_evidence_chunks=[
            build_candidate_answer_evidence_chunk(
                document_id="new-cited",
                chunk_id=1,
                citation_number=1,
                retrieval_number=1,
                chunk_identifier="new-cited-chunk",
                heading="Rule > New cited",
                content="new cited exact text",
            ),
            build_candidate_answer_evidence_chunk(
                document_id="new-hidden",
                chunk_id=2,
                citation_number=None,
                retrieval_number=2,
                chunk_identifier="new-hidden-chunk",
                heading="Rule > New hidden",
                content="new hidden exact text",
            ),
        ],
    )

    with patch.object(
        research_agent,
        "run_functions_tuples_in_parallel",
        return_value=[local_result],
    ):
        combined = research_agent.run_research_agent_calls(
            research_agent_calls=[_research_agent_call("focused topic")],
            parent_tool_call_ids=["parent"],
            tools=[],
            emitter=MagicMock(),
            state_container=ChatStateContainer(),
            llm=_llm(),
            is_reasoning_model=True,
            token_counter=len,
            citation_mapping={4: prior_public_doc},
            evidence_citation_mapping={5: prior_evidence_doc},
        )

    assert combined.intermediate_reports == ["Supported [6]."]
    assert combined.citation_mapping == {4: prior_public_doc, 6: cited_doc}
    assert combined.evidence_citation_mapping == {
        5: prior_evidence_doc,
        6: cited_doc,
        7: hidden_doc,
    }
    assert combined.exact_evidence_chunks[0].citation_number == 6
    assert combined.exact_evidence_chunks[0].retrieval_number == 6
    assert combined.exact_evidence_chunks[1].citation_number is None
    assert combined.exact_evidence_chunks[1].retrieval_number == 7


def test_repeated_think_calls_stop_at_decision_limit_and_generate_report() -> None:
    think_call = ToolCallKickoff(
        tool_call_id="think",
        tool_name=THINK_TOOL_NAME,
        tool_args={"thought": "reassess"},
        placement=Placement(turn_index=0),
    )

    with (
        patch.object(
            research_agent,
            "run_llm_step",
            return_value=_llm_step(think_call),
        ) as run_llm_step,
        patch.object(
            research_agent,
            "generate_intermediate_report",
            return_value="bounded report",
        ) as generate_report,
    ):
        result = research_agent.run_research_agent_call(
            research_agent_call=_research_agent_call("focused topic"),
            parent_tool_call_id="parent",
            tools=[_make_search_tool()],
            emitter=MagicMock(),
            state_container=ChatStateContainer(),
            llm=_llm(),
            is_reasoning_model=False,
            token_counter=len,
            user_identity=None,
        )

    assert result is not None
    assert result.intermediate_report == "bounded report"
    assert run_llm_step.call_count == research_agent.MAX_RESEARCH_AGENT_LLM_DECISIONS
    generate_report.assert_called_once()


def test_decision_limit_preserves_full_search_then_think_pattern() -> None:
    decision_number = 0
    completed_searches = 0

    def fake_run_llm_step(**_: Any) -> tuple[LlmStepResult, bool]:
        nonlocal decision_number
        decision_number += 1
        if decision_number % 2 == 1:
            search_number = (decision_number + 1) // 2
            return _llm_step(
                _search_call(f"search-{search_number}", f"query {search_number}")
            )
        return _llm_step(
            ToolCallKickoff(
                tool_call_id=f"think-{decision_number}",
                tool_name=THINK_TOOL_NAME,
                tool_args={"thought": "assess the latest evidence"},
                placement=Placement(turn_index=0),
            )
        )

    def fake_run_tool_calls(**kwargs: Any) -> ParallelToolCallResponse:
        nonlocal completed_searches
        completed_searches += 1
        tool_call = cast(list[ToolCallKickoff], kwargs["tool_calls"])[0]
        return ParallelToolCallResponse(
            tool_responses=[
                ToolResponse(
                    rich_response=None,
                    llm_facing_response=_search_response(
                        f"Provision {completed_searches}",
                        f"material evidence {completed_searches}",
                    ),
                    tool_call=tool_call,
                )
            ],
            updated_citation_mapping={},
        )

    with (
        patch.object(
            research_agent,
            "run_llm_step",
            side_effect=fake_run_llm_step,
        ) as run_llm_step,
        patch.object(
            research_agent,
            "run_tool_calls",
            side_effect=fake_run_tool_calls,
        ),
        patch.object(
            research_agent,
            "generate_intermediate_report",
            return_value="complete report",
        ) as generate_report,
    ):
        result = research_agent.run_research_agent_call(
            research_agent_call=_research_agent_call("focused topic"),
            parent_tool_call_id="parent",
            tools=[_make_search_tool()],
            emitter=MagicMock(),
            state_container=ChatStateContainer(),
            llm=_llm(),
            is_reasoning_model=False,
            token_counter=len,
            user_identity=None,
        )

    assert result is not None
    assert result.intermediate_report == "complete report"
    assert completed_searches == research_agent.MAX_RESEARCH_CYCLES
    assert run_llm_step.call_count == research_agent.MAX_RESEARCH_CYCLES * 2 - 1
    generate_report.assert_called_once()


def test_per_call_budget_bounds_recovery_without_changing_defaults() -> None:
    decision_number = 0
    completed_searches = 0

    def fake_run_llm_step(**_: Any) -> tuple[LlmStepResult, bool]:
        nonlocal decision_number
        decision_number += 1
        return _llm_step(
            _search_call(
                f"bounded-search-{decision_number}",
                f"materially distinct query {decision_number}",
            )
        )

    def fake_run_tool_calls(**kwargs: Any) -> ParallelToolCallResponse:
        nonlocal completed_searches
        completed_searches += 1
        tool_call = cast(list[ToolCallKickoff], kwargs["tool_calls"])[0]
        return ParallelToolCallResponse(
            tool_responses=[
                ToolResponse(
                    rich_response=None,
                    llm_facing_response=_search_response(
                        f"Provision {completed_searches}",
                        f"bounded evidence {completed_searches}",
                    ),
                    tool_call=tool_call,
                )
            ],
            updated_citation_mapping={},
        )

    budget = research_agent.ResearchAgentRunBudget(
        max_research_cycles=2,
        max_llm_decisions=4,
        max_report_tokens=1536,
    )
    with (
        patch.object(
            research_agent,
            "run_llm_step",
            side_effect=fake_run_llm_step,
        ) as run_llm_step,
        patch.object(
            research_agent,
            "run_tool_calls",
            side_effect=fake_run_tool_calls,
        ),
        patch.object(
            research_agent,
            "generate_intermediate_report",
            return_value="bounded recovery report",
        ) as generate_report,
    ):
        result = research_agent.run_research_agent_call(
            research_agent_call=_research_agent_call("one evidence gap"),
            parent_tool_call_id="parent",
            tools=[_make_search_tool(regulatory_chunks_only=True)],
            emitter=MagicMock(),
            state_container=ChatStateContainer(),
            llm=_llm(),
            is_reasoning_model=True,
            token_counter=len,
            user_identity=None,
            run_budget=budget,
        )

    assert result is not None
    assert result.intermediate_report == "bounded recovery report"
    assert completed_searches == 2
    assert run_llm_step.call_count == 2
    assert run_llm_step.call_count <= budget.max_llm_decisions
    generate_report.assert_called_once()
    assert generate_report.call_args.kwargs["max_tokens"] == 1536
