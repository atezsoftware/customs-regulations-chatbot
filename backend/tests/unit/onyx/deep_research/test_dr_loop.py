import json
import queue
from collections.abc import Callable
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest

from onyx.chat.chat_state import ChatStateContainer
from onyx.chat.emitter import Emitter
from onyx.chat.empty_response import EmptyLLMResponseError
from onyx.chat.models import ChatMessageSimple, LlmStepResult
from onyx.configs.constants import DocumentSource, MessageType
from onyx.context.search.models import BaseFilters, SearchDoc, SearchDocsResponse
from onyx.deep_research import dr_loop
from onyx.regulatory.candidate_answer_review import (
    CandidateAnswerClaimIssue,
    CandidateAnswerClaimSpan,
    CandidateAnswerEvidenceChunk,
    CandidateAnswerReviewError,
    CandidateAnswerReviewResult,
    ClaimKind,
)
from onyx.regulatory.coverage_plan import (
    RegulatoryCoverageItem,
    RegulatoryCoveragePlan,
)
from onyx.server.query_and_chat.placement import Placement
from onyx.server.query_and_chat.streaming_models import (
    AgentResponseDelta,
    AgentResponseStart,
    CitationInfo,
    Packet,
)
from onyx.tools.models import (
    ParallelToolCallResponse,
    ToolCallInfo,
    ToolCallKickoff,
    ToolResponse,
)
from onyx.tools.tool_implementations.search.search_tool import SearchTool


def test_regulatory_final_report_uses_full_report_output_budget() -> None:
    assert dr_loop.MAX_REGULATORY_FINAL_REPORT_TOKENS == dr_loop.MAX_FINAL_REPORT_TOKENS


def _search_doc() -> SearchDoc:
    return SearchDoc(
        document_id="document-1",
        chunk_ind=4,
        semantic_identifier="Rule > Article 4",
        blurb="WRONG SHORT BLURB",
        source_type=DocumentSource.FILE,
        boost=1,
        hidden=False,
        metadata={
            "regulatory_chunk_id": "regulatory-chunk-4",
            "regulatory_heading_path": ["Rule", "Article 4"],
        },
        match_highlights=[],
    )


def _hidden_search_doc(
    citation_number: int,
    *,
    document_id: str,
) -> SearchDoc:
    return SearchDoc(
        document_id=document_id,
        chunk_ind=citation_number,
        semantic_identifier=f"Rule > Article {citation_number}",
        blurb="WRONG SHORT BLURB",
        source_type=DocumentSource.FILE,
        boost=1,
        hidden=False,
        metadata={
            "regulatory_chunk_id": f"regulatory-chunk-{citation_number}",
            "regulatory_heading_path": ["Rule", f"Article {citation_number}"],
        },
        match_highlights=[],
    )


def _evidence() -> CandidateAnswerEvidenceChunk:
    return CandidateAnswerEvidenceChunk(
        document_id="document-1",
        chunk_id=4,
        citation_number=1,
        retrieval_number=1,
        chunk_identifier="regulatory-chunk-4",
        heading="Rule > Article 4",
        content="EXACT LLM-VISIBLE OPERATIVE TEXT",
    )


def _hidden_evidence(citation_number: int) -> CandidateAnswerEvidenceChunk:
    return CandidateAnswerEvidenceChunk(
        document_id=f"document-{citation_number}",
        chunk_id=citation_number,
        citation_number=None,
        retrieval_number=citation_number,
        chunk_identifier=f"regulatory-chunk-{citation_number}",
        heading=f"Rule > Article {citation_number}",
        content=f"EXACT HIDDEN OPERATIVE TEXT {citation_number}",
    )


def _history() -> list[ChatMessageSimple]:
    return [
        ChatMessageSimple(
            message="Analyze every material legal issue.",
            token_count=35,
            message_type=MessageType.USER,
        )
    ]


def _llm() -> MagicMock:
    llm = MagicMock()
    llm.config = MagicMock(
        max_input_tokens=100_000,
        model_provider="vertex_ai",
        model_name="gemini-3.6-flash",
    )
    return llm


def _recovery_search_tool() -> SearchTool:
    return SearchTool(
        tool_id=1,
        emitter=MagicMock(),
        user=MagicMock(is_anonymous=False),
        persona_search_info=MagicMock(document_set_names=[]),
        llm=MagicMock(),
        document_index=MagicMock(),
        user_selected_filters=BaseFilters(regulatory_chunks_only=True),
        project_id_filter=None,
        enable_slack_search=False,
        auto_detect_filters=True,
    )


def _fake_llm_step(answers: list[str]) -> Callable[..., tuple[LlmStepResult, bool]]:
    remaining_answers = iter(answers)

    def run(**kwargs: Any) -> tuple[LlmStepResult, bool]:
        answer = next(remaining_answers)
        state = cast(ChatStateContainer, kwargs["state_container"])
        emitter = cast(Emitter, kwargs["emitter"])
        placement = cast(Placement, kwargs["placement"])
        final_documents = cast(list[SearchDoc], kwargs["final_documents"])
        pre_answer_time = cast(float | None, kwargs["pre_answer_processing_time"])
        citation_processor = kwargs["citation_processor"]
        state.set_reasoning_tokens(f"reasoning for {answer}")
        state.set_answer_tokens(answer or None)
        state.set_pre_answer_processing_time(pre_answer_time)
        emits_citation = bool(
            answer and "[1]" in answer and 1 in citation_processor.citation_to_doc
        )
        if emits_citation:
            search_doc = citation_processor.citation_to_doc[1]
            citation_processor.seen_citations[1] = search_doc
            state.add_emitted_citation(1)
        emitter.emit(
            Packet(
                placement=placement,
                obj=AgentResponseStart(
                    final_documents=final_documents,
                    pre_answer_processing_seconds=pre_answer_time,
                ),
            )
        )
        if emits_citation:
            search_doc = citation_processor.citation_to_doc[1]
            emitter.emit(
                Packet(
                    placement=placement,
                    obj=CitationInfo(
                        citation_number=1,
                        document_id=search_doc.document_id,
                        chunk_ind=search_doc.chunk_ind,
                        semantic_identifier=search_doc.semantic_identifier,
                    ),
                )
            )
        if answer:
            emitter.emit(
                Packet(
                    placement=placement,
                    obj=AgentResponseDelta(content=answer),
                )
            )
        return (
            LlmStepResult(
                reasoning=f"reasoning for {answer}",
                answer=answer or None,
                raw_answer=answer or None,
                tool_calls=None,
                finish_reason="stop",
            ),
            True,
        )

    return run


def _fake_llm_step_with_finish_reasons(
    steps: list[tuple[str, str]],
) -> Callable[..., tuple[LlmStepResult, bool]]:
    fake_step = _fake_llm_step([answer for answer, _ in steps])
    finish_reasons = iter(finish_reason for _, finish_reason in steps)

    def run(**kwargs: Any) -> tuple[LlmStepResult, bool]:
        result, has_reasoned = fake_step(**kwargs)
        return (
            result.model_copy(update={"finish_reason": next(finish_reasons)}),
            has_reasoned,
        )

    return run


def _fake_llm_step_with_citations(
    steps: list[tuple[str, list[int]]],
) -> Callable[..., tuple[LlmStepResult, bool]]:
    remaining_steps = iter(steps)

    def run(**kwargs: Any) -> tuple[LlmStepResult, bool]:
        answer, citation_numbers = next(remaining_steps)
        state = cast(ChatStateContainer, kwargs["state_container"])
        emitter = cast(Emitter, kwargs["emitter"])
        placement = cast(Placement, kwargs["placement"])
        citation_processor = kwargs["citation_processor"]
        final_documents = cast(list[SearchDoc], kwargs["final_documents"])
        pre_answer_time = cast(float | None, kwargs["pre_answer_processing_time"])
        state.set_answer_tokens(answer)
        state.set_pre_answer_processing_time(pre_answer_time)
        emitter.emit(
            Packet(
                placement=placement,
                obj=AgentResponseStart(
                    final_documents=final_documents,
                    pre_answer_processing_seconds=pre_answer_time,
                ),
            )
        )
        for citation_number in citation_numbers:
            search_doc = citation_processor.citation_to_doc[citation_number]
            citation_processor.seen_citations[citation_number] = search_doc
            state.add_emitted_citation(citation_number)
            emitter.emit(
                Packet(
                    placement=placement,
                    obj=CitationInfo(
                        citation_number=citation_number,
                        document_id=search_doc.document_id,
                        chunk_ind=search_doc.chunk_ind,
                        semantic_identifier=search_doc.semantic_identifier,
                    ),
                )
            )
        emitter.emit(
            Packet(
                placement=placement,
                obj=AgentResponseDelta(content=answer),
            )
        )
        return (
            LlmStepResult(
                reasoning=None,
                answer=answer,
                raw_answer=answer,
                tool_calls=None,
                finish_reason="stop",
            ),
            False,
        )

    return run


def _run_final_report(
    *,
    is_regulatory_research: bool,
    live_state: ChatStateContainer,
    destination: Emitter,
    exact_evidence_chunks: list[CandidateAnswerEvidenceChunk] | None = None,
    citation_mapping: dict[int, SearchDoc] | None = None,
    evidence_citation_mapping: dict[int, SearchDoc] | None = None,
    history: list[ChatMessageSimple] | None = None,
    recovery_tools: list[SearchTool] | None = None,
    reasoning_effort: dr_loop.ReasoningEffort = dr_loop.ReasoningEffort.LOW,
    custom_agent_prompt: str | None = None,
) -> bool:
    return dr_loop.generate_final_report(
        history=_history() if history is None else history,
        research_plan="Investigate the material legal relationships.",
        llm=_llm(),
        token_counter=len,
        state_container=live_state,
        emitter=destination,
        turn_index=3,
        citation_mapping=(
            {1: _search_doc()} if citation_mapping is None else citation_mapping
        ),
        is_regulatory_research=is_regulatory_research,
        user_identity=None,
        reasoning_effort=reasoning_effort,
        pre_answer_processing_time=2.0,
        exact_evidence_chunks=(
            [_evidence()] if exact_evidence_chunks is None else exact_evidence_chunks
        ),
        evidence_citation_mapping=evidence_citation_mapping,
        recovery_tools=recovery_tools,
        custom_agent_prompt=custom_agent_prompt,
    )


def _packets(merged_queue: queue.Queue[tuple[int, object]]) -> list[Packet]:
    packets: list[Packet] = []
    while not merged_queue.empty():
        packets.append(cast(Packet, merged_queue.get_nowait()[1]))
    return packets


def test_orchestrator_schedule_runs_every_decision_before_forced_report() -> None:
    schedule = list(dr_loop._orchestrator_cycle_schedule(4))

    assert schedule[:-1] == [0, 1, 2, 3]
    assert schedule[-1] == 4


def test_coverage_evidence_selection_preserves_every_call_before_deeper_ranks() -> None:
    first_call = [_hidden_evidence(1), _hidden_evidence(2)]
    second_call = [_hidden_evidence(3), _hidden_evidence(4)]

    selected = dr_loop._round_robin_coverage_evidence(
        [first_call, second_call],
        limit=3,
    )

    assert [chunk.retrieval_number for chunk in selected] == [1, 3, 2]


def test_regulatory_evidence_citations_are_dense_after_parallel_searches() -> None:
    first_doc = _hidden_search_doc(1, document_id="document-1")
    second_doc = _hidden_search_doc(101, document_id="document-101")

    citation_mapping, evidence = dr_loop._densify_regulatory_evidence_citations(
        {1: first_doc, 101: second_doc},
        [_hidden_evidence(1), _hidden_evidence(101)],
    )

    assert citation_mapping == {1: first_doc, 2: second_doc}
    assert [chunk.citation_number for chunk in evidence] == [1, 2]
    assert [chunk.retrieval_number for chunk in evidence] == [1, 2]


def test_direct_coverage_search_builds_exact_evidence_and_citation_mapping() -> None:
    search_doc = _search_doc()
    call = ToolCallKickoff(
        tool_call_id="coverage-search",
        tool_name=SearchTool.NAME,
        tool_args={
            "queries": ["Focused controlling text"],
            "search_mode": "hybrid",
            "coverage_item": "Requested issue",
            "evidence_target": "Controlling trigger",
            "source_anchors": ["Named Instrument"],
        },
        placement=Placement(turn_index=1, tab_index=0),
    )
    response = ToolResponse(
        rich_response=SearchDocsResponse(
            search_docs=[search_doc],
            citation_mapping={1: search_doc.document_id},
            citation_chunk_mapping={1: search_doc.chunk_ind},
        ),
        llm_facing_response=json.dumps(
            {
                "results": [
                    {
                        "document": 1,
                        "title": search_doc.semantic_identifier,
                        "content": "EXACT LLM-VISIBLE OPERATIVE TEXT",
                        "metadata": json.dumps(search_doc.metadata),
                    }
                ]
            }
        ),
        tool_call=call,
    )
    plan = RegulatoryCoveragePlan(
        coverage_items=[
            RegulatoryCoverageItem(
                research_question="Resolve the requested issue.",
                evidence_dimensions=["Controlling trigger"],
                completion_test="Close the requested issue.",
            )
        ]
    )
    state = ChatStateContainer()

    with patch.object(
        dr_loop,
        "run_tool_calls",
        return_value=ParallelToolCallResponse(
            tool_responses=[response],
            updated_citation_mapping={1: search_doc.document_id},
        ),
    ):
        citation_mapping, evidence = dr_loop._run_regulatory_coverage_searches(
            coverage_plan=plan,
            tools=[_recovery_search_tool()],
            history=_history(),
            state_container=state,
            turn_index=1,
        )

    assert citation_mapping == {1: search_doc}
    assert len(evidence) == 1
    assert evidence[0].content == "EXACT LLM-VISIBLE OPERATIVE TEXT"
    assert evidence[0].citation_number == 1
    assert evidence[0].research_target == (
        "Specific evidence target: Controlling trigger. Coverage item: Requested issue"
    )
    assert len(state.get_tool_calls()) == 1


def test_regulatory_final_report_pass_publishes_hidden_draft_once() -> None:
    live_state = ChatStateContainer()
    merged_queue: queue.Queue[tuple[int, object]] = queue.Queue()
    destination = Emitter(merged_queue)
    review = CandidateAnswerReviewResult(needs_reconsideration=False)
    recovery_tool = _recovery_search_tool()

    with (
        patch.object(
            dr_loop,
            "run_llm_step",
            side_effect=_fake_llm_step(["Initial grounded answer [1]"]),
        ) as run_step,
        patch.object(
            dr_loop,
            "review_regulatory_candidate_answer",
            return_value=review,
        ) as reviewer,
        patch.object(dr_loop, "run_research_agent_calls") as run_recovery,
    ):
        has_reasoned = _run_final_report(
            is_regulatory_research=True,
            live_state=live_state,
            destination=destination,
            recovery_tools=[recovery_tool],
        )

    assert has_reasoned is True
    assert run_step.call_count == 1
    reviewer.assert_called_once()
    run_recovery.assert_not_called()
    reviewed_evidence = reviewer.call_args.kwargs["evidence_chunks"]
    assert reviewed_evidence[0].content == "EXACT LLM-VISIBLE OPERATIVE TEXT"
    assert reviewed_evidence[0].content != _search_doc().blurb
    assert reviewed_evidence[0].citation_number == 1
    packets = _packets(merged_queue)
    starts = [
        packet.obj for packet in packets if isinstance(packet.obj, AgentResponseStart)
    ]
    assert len(starts) == 1
    assert starts[0].final_documents == [_search_doc()]
    assert [
        packet.obj.content
        for packet in packets
        if isinstance(packet.obj, AgentResponseDelta)
    ] == ["Initial grounded answer [1]"]
    assert live_state.get_answer_tokens() == "Initial grounded answer [1]"


def test_regulatory_final_report_reject_publishes_only_one_correction() -> None:
    search_doc = _search_doc()
    live_state = ChatStateContainer()
    live_state.add_search_docs([search_doc])
    merged_queue: queue.Queue[tuple[int, object]] = queue.Queue()
    destination = Emitter(merged_queue)
    review = CandidateAnswerReviewResult(
        needs_reconsideration=True,
        advisory_claim_issues=[
            CandidateAnswerClaimIssue(
                claim_reference="A material conclusion",
                advisory_feedback="The cited text does not support its scope.",
                related_citation_numbers=[1],
            )
        ],
    )

    with (
        patch.object(
            dr_loop,
            "run_llm_step",
            side_effect=_fake_llm_step(
                ["Rejected hidden draft [1]", "Corrected final answer [1]"]
            ),
        ) as run_step,
        patch.object(
            dr_loop,
            "review_regulatory_candidate_answer",
            return_value=review,
        ) as reviewer,
        patch.object(
            dr_loop,
            "review_regulatory_candidate_resolution",
            return_value=CandidateAnswerReviewResult(needs_reconsideration=False),
        ) as resolution_reviewer,
    ):
        _run_final_report(
            is_regulatory_research=True,
            live_state=live_state,
            destination=destination,
            reasoning_effort=dr_loop.ReasoningEffort.HIGH,
        )

    reviewer.assert_called_once()
    resolution_reviewer.assert_called_once()
    assert run_step.call_count == 2
    selected_llm = run_step.call_args_list[0].kwargs["llm"]
    assert run_step.call_args_list[1].kwargs["llm"] is selected_llm
    assert reviewer.call_args.args[0] is selected_llm
    assert all(
        call.kwargs["max_tokens"] == dr_loop.MAX_REGULATORY_FINAL_REPORT_TOKENS
        for call in run_step.call_args_list
    )
    assert all(
        call.kwargs["tool_choice"] is dr_loop.ToolChoiceOptions.NONE
        for call in run_step.call_args_list
    )
    assert all(
        call.kwargs["reasoning_effort"] is dr_loop.ReasoningEffort.LOW
        for call in run_step.call_args_list
    )
    correction_history = cast(
        list[ChatMessageSimple], run_step.call_args_list[1].kwargs["history"]
    )
    assert any(
        message.message == "Rejected hidden draft [1]"
        and message.message_type == MessageType.ASSISTANT
        for message in correction_history
    )
    assert any(
        "A material conclusion" in message.message
        and "EXACT LLM-VISIBLE OPERATIVE TEXT" in message.message
        for message in correction_history
    )
    assert all(
        "Investigate the material legal relationships." not in message.message
        for message in correction_history
    )
    packets = _packets(merged_queue)
    assert sum(isinstance(packet.obj, AgentResponseStart) for packet in packets) == 1
    deltas = [
        packet.obj.content
        for packet in packets
        if isinstance(packet.obj, AgentResponseDelta)
    ]
    assert deltas == ["Corrected final answer [1]"]
    assert "Rejected hidden draft" not in "".join(deltas)
    assert live_state.get_answer_tokens() == "Corrected final answer [1]"
    assert live_state.get_all_search_docs() == {
        (search_doc.document_id, search_doc.chunk_ind): search_doc
    }


def test_regulatory_final_review_runs_one_issue_only_recovery_and_merges_mapping() -> (
    None
):
    public_doc = _search_doc()
    recovered_doc = _hidden_search_doc(2, document_id="recovered-document")
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
                        "title": "Convention > Article 4A",
                        "content": "RECOVERED EXACT OPERATIVE TEXT",
                        "metadata": json.dumps(recovered_doc.metadata),
                    }
                ]
            }
        ),
    )
    first_issue = CandidateAnswerClaimIssue(
        claim_kind=ClaimKind.LEGAL_RULE,
        claim_span=CandidateAnswerClaimSpan(start=0, end=18),
        claim_reference="DRAFT_SENTINEL [1]",
        advisory_feedback=("The cited chunk does not support this legal consequence."),
        related_citation_numbers=[1],
        recovery_query="Article 4A legal consequence exact provision",
    )
    review = CandidateAnswerReviewResult(
        needs_reconsideration=True,
        advisory_claim_issues=[
            first_issue,
            CandidateAnswerClaimIssue(
                claim_kind=ClaimKind.LEGAL_RULE,
                claim_span=CandidateAnswerClaimSpan(start=19, end=30),
                claim_reference="second issue",
                advisory_feedback="SECOND_ISSUE_SENTINEL",
                recovery_query="second unsupported rule",
            ),
        ],
    )
    history = [
        ChatMessageSimple(
            message="EARLIER_CONTEXT_SENTINEL",
            token_count=24,
            message_type=MessageType.USER,
        ),
        ChatMessageSimple(
            message="PRIOR_REPORT_SENTINEL",
            token_count=21,
            message_type=MessageType.ASSISTANT,
        ),
        ChatMessageSimple(
            message=(
                "FULL_SCENARIO_SENTINEL. Analyze Article 4A consequence. "
                "REQUEST_TAIL_SENTINEL."
            ),
            token_count=84,
            message_type=MessageType.USER,
        ),
    ]
    recovery_tool = _recovery_search_tool()
    live_state = ChatStateContainer()
    destination = Emitter(queue.Queue())

    with (
        patch.object(
            dr_loop,
            "run_llm_step",
            side_effect=_fake_llm_step_with_citations(
                [
                    ("DRAFT_SENTINEL [1]", [1]),
                    ("Corrected from recovered evidence [2]", [2]),
                ]
            ),
        ) as run_step,
        patch.object(
            dr_loop,
            "review_regulatory_candidate_answer",
            return_value=review,
        ),
        patch.object(
            dr_loop,
            "run_single_gap_recovery",
            return_value=recovery_response,
        ) as run_recovery,
        patch.object(dr_loop, "run_research_agent_calls") as research_agent,
        patch.object(
            dr_loop,
            "review_regulatory_candidate_resolution",
            return_value=CandidateAnswerReviewResult(needs_reconsideration=False),
        ) as resolution_reviewer,
    ):
        _run_final_report(
            is_regulatory_research=True,
            live_state=live_state,
            destination=destination,
            citation_mapping={1: public_doc},
            evidence_citation_mapping={1: public_doc},
            exact_evidence_chunks=[_evidence()],
            history=history,
            recovery_tools=[recovery_tool],
        )

    run_recovery.assert_called_once()
    recovery_kwargs = run_recovery.call_args.kwargs
    assert recovery_kwargs["search_tool"] is recovery_tool
    assert recovery_kwargs["issue"] is first_issue
    assert recovery_kwargs["starting_citation_num"] == 2
    research_agent.assert_not_called()

    assert run_step.call_count == 2
    correction_history = cast(
        list[ChatMessageSimple], run_step.call_args_list[1].kwargs["history"]
    )
    assert any(
        "RECOVERED EXACT OPERATIVE TEXT" in message.message
        for message in correction_history
    )
    resolution_evidence = resolution_reviewer.call_args.kwargs["evidence_chunks"]
    assert any(
        (chunk.document_id, chunk.chunk_id)
        == (
            recovered_doc.document_id,
            recovered_doc.chunk_ind,
        )
        and chunk.citation_number == 2
        for chunk in resolution_evidence
    )
    assert live_state.get_citation_to_doc() == {1: public_doc, 2: recovered_doc}


@pytest.mark.parametrize("raises", [False, True])
def test_regulatory_gap_recovery_failure_or_empty_result_is_not_retried(
    raises: bool,
) -> None:
    public_doc = _search_doc()
    review = CandidateAnswerReviewResult(
        needs_reconsideration=True,
        advisory_claim_issues=[
            CandidateAnswerClaimIssue(
                claim_kind=ClaimKind.LEGAL_RULE,
                claim_span=CandidateAnswerClaimSpan(start=0, end=31),
                claim_reference="Analyze every material legal issue.",
                advisory_feedback=(
                    "The express deliverable is wholly unanswered without exact evidence."
                ),
                recovery_query="material legal issue controlling rule",
            )
        ],
    )
    empty_recovery = ToolResponse(
        rich_response=SearchDocsResponse(
            search_docs=[],
            citation_mapping={},
            citation_chunk_mapping={},
        ),
        llm_facing_response=json.dumps({"results": []}),
    )
    live_state = ChatStateContainer()
    run_recovery = MagicMock(
        side_effect=RuntimeError("bounded recovery failed") if raises else None,
        return_value=None if raises else empty_recovery,
    )

    with (
        patch.object(
            dr_loop,
            "run_llm_step",
            side_effect=_fake_llm_step(
                ["Incomplete hidden draft [1]", "Qualified correction [1]"]
            ),
        ) as run_step,
        patch.object(
            dr_loop,
            "review_regulatory_candidate_answer",
            return_value=review,
        ),
        patch.object(dr_loop, "run_single_gap_recovery", new=run_recovery),
        patch.object(dr_loop, "run_research_agent_calls") as research_agent,
        patch.object(
            dr_loop,
            "review_regulatory_candidate_resolution",
            return_value=CandidateAnswerReviewResult(needs_reconsideration=False),
        ),
    ):
        _run_final_report(
            is_regulatory_research=True,
            live_state=live_state,
            destination=Emitter(queue.Queue()),
            citation_mapping={1: public_doc},
            evidence_citation_mapping={1: public_doc},
            exact_evidence_chunks=[_evidence()],
            recovery_tools=[_recovery_search_tool()],
        )

    run_recovery.assert_called_once()
    research_agent.assert_not_called()
    assert run_step.call_count == 2
    correction_history = cast(
        list[ChatMessageSimple], run_step.call_args_list[1].kwargs["history"]
    )
    assert all(
        "RECOVERED EXACT OPERATIVE TEXT" not in message.message
        for message in correction_history
    )


def test_regulatory_correction_can_cite_hidden_exact_evidence_without_ui_leak() -> None:
    public_doc = _search_doc()
    corrected_doc = _hidden_search_doc(2, document_id="document-2")
    unused_doc = _hidden_search_doc(3, document_id="document-3")
    unformatted_doc = _hidden_search_doc(4, document_id="document-4")
    live_state = ChatStateContainer()
    merged_queue: queue.Queue[tuple[int, object]] = queue.Queue()
    destination = Emitter(merged_queue)
    review = CandidateAnswerReviewResult(
        needs_reconsideration=True,
        advisory_claim_issues=[
            CandidateAnswerClaimIssue(
                claim_reference="Omitted controlling qualification",
                advisory_feedback="Use the retrieved exact qualification.",
                related_citation_numbers=[1],
            )
        ],
    )

    with (
        patch.object(
            dr_loop,
            "run_llm_step",
            side_effect=_fake_llm_step_with_citations(
                [
                    ("Incomplete draft [1]", [1]),
                    ("Corrected answer [2]", [2]),
                ]
            ),
        ) as run_step,
        patch.object(
            dr_loop,
            "review_regulatory_candidate_answer",
            return_value=review,
        ),
        patch.object(
            dr_loop,
            "review_regulatory_candidate_resolution",
            return_value=CandidateAnswerReviewResult(needs_reconsideration=False),
        ) as resolution_reviewer,
    ):
        _run_final_report(
            is_regulatory_research=True,
            live_state=live_state,
            destination=destination,
            citation_mapping={1: public_doc},
            evidence_citation_mapping={
                1: public_doc,
                2: corrected_doc,
                3: unused_doc,
                4: unformatted_doc,
            },
            exact_evidence_chunks=[
                _evidence(),
                _hidden_evidence(2),
                _hidden_evidence(3),
            ],
        )

    assert run_step.call_args_list[0].kwargs["citation_processor"].citation_to_doc == {
        1: public_doc
    }
    assert run_step.call_args_list[1].kwargs["citation_processor"].citation_to_doc == {
        1: public_doc,
        2: corrected_doc,
        3: unused_doc,
    }
    correction_history = cast(
        list[ChatMessageSimple], run_step.call_args_list[1].kwargs["history"]
    )
    correction_prompt = "\n".join(message.message for message in correction_history)
    assert "EXACT HIDDEN OPERATIVE TEXT 2" in correction_prompt
    assert '"citation_number": 2' in correction_prompt
    resolution_evidence = resolution_reviewer.call_args.kwargs["evidence_chunks"]
    corrected_evidence = next(
        evidence for evidence in resolution_evidence if evidence.retrieval_number == 2
    )
    assert corrected_evidence.citation_number == 2

    packets = _packets(merged_queue)
    starts = [
        packet.obj for packet in packets if isinstance(packet.obj, AgentResponseStart)
    ]
    assert len(starts) == 1
    assert starts[0].final_documents == [corrected_doc]
    assert [
        packet.obj.citation_number
        for packet in packets
        if isinstance(packet.obj, CitationInfo)
    ] == [2]
    assert live_state.get_emitted_citations() == {2}
    assert live_state.get_citation_to_doc() == {
        1: public_doc,
        2: corrected_doc,
        3: unused_doc,
    }


def test_regulatory_final_report_review_unavailable_fails_open() -> None:
    live_state = ChatStateContainer()
    merged_queue: queue.Queue[tuple[int, object]] = queue.Queue()
    destination = Emitter(merged_queue)
    unavailable = CandidateAnswerReviewResult(
        needs_reconsideration=False,
        review_error=CandidateAnswerReviewError.REVIEW_UNAVAILABLE,
    )

    with (
        patch.object(
            dr_loop,
            "run_llm_step",
            side_effect=_fake_llm_step(["Fail-open grounded draft [1]"]),
        ) as run_step,
        patch.object(
            dr_loop,
            "review_regulatory_candidate_answer",
            return_value=unavailable,
        ),
    ):
        _run_final_report(
            is_regulatory_research=True,
            live_state=live_state,
            destination=destination,
        )

    assert run_step.call_count == 1
    assert live_state.get_answer_tokens() == "Fail-open grounded draft [1]"


def test_regulatory_final_report_retries_truncated_correction_without_leaking() -> None:
    live_state = ChatStateContainer()
    merged_queue: queue.Queue[tuple[int, object]] = queue.Queue()
    destination = Emitter(merged_queue)
    review = CandidateAnswerReviewResult(
        needs_reconsideration=True,
        advisory_claim_issues=[
            CandidateAnswerClaimIssue(
                claim_reference="Unsupported conclusion",
                advisory_feedback="Qualify the unsupported scope.",
                related_citation_numbers=[1],
            )
        ],
    )

    with (
        patch.object(
            dr_loop,
            "run_llm_step",
            side_effect=_fake_llm_step_with_finish_reasons(
                [
                    ("Safe initial answer [1]", "stop"),
                    ("Incomplete correction [1]", "length"),
                    ("Complete correction [1]", "stop"),
                ]
            ),
        ) as run_step,
        patch.object(
            dr_loop,
            "review_regulatory_candidate_answer",
            return_value=review,
        ),
        patch.object(
            dr_loop,
            "review_regulatory_candidate_resolution",
            return_value=CandidateAnswerReviewResult(needs_reconsideration=False),
        ) as resolution_reviewer,
    ):
        has_reasoned = _run_final_report(
            is_regulatory_research=True,
            live_state=live_state,
            destination=destination,
            custom_agent_prompt="Act as a customs compliance specialist.",
        )

    assert [call.kwargs["reasoning_effort"] for call in run_step.call_args_list] == [
        dr_loop.ReasoningEffort.LOW,
        dr_loop.ReasoningEffort.LOW,
        dr_loop.ReasoningEffort.OFF,
    ]
    correction_histories = [
        cast(list[ChatMessageSimple], call.kwargs["history"])
        for call in run_step.call_args_list[1:]
    ]
    assert all(
        any(
            message.message == "Act as a customs compliance specialist."
            for message in history
        )
        for history in correction_histories
    )
    packets = _packets(merged_queue)
    assert [
        packet.obj.content
        for packet in packets
        if isinstance(packet.obj, AgentResponseDelta)
    ] == ["Complete correction [1]"]
    assert live_state.get_answer_tokens() == "Complete correction [1]"
    assert has_reasoned is True
    resolution_reviewer.assert_called_once()


def test_regulatory_final_report_truncated_correction_exhaustion_is_typed() -> None:
    live_state = ChatStateContainer()
    merged_queue: queue.Queue[tuple[int, object]] = queue.Queue()
    destination = Emitter(merged_queue)
    review = CandidateAnswerReviewResult(
        needs_reconsideration=True,
        advisory_claim_issues=[
            CandidateAnswerClaimIssue(
                claim_reference="Unsupported conclusion",
                advisory_feedback="Qualify the unsupported scope.",
                related_citation_numbers=[1],
            )
        ],
    )

    with (
        patch.object(
            dr_loop,
            "run_llm_step",
            side_effect=_fake_llm_step_with_finish_reasons(
                [
                    ("Safe initial answer [1]", "stop"),
                    ("Incomplete correction one [1]", "length"),
                    ("Incomplete correction two [1]", "length"),
                ]
            ),
        ),
        patch.object(
            dr_loop,
            "review_regulatory_candidate_answer",
            return_value=review,
        ),
        patch.object(
            dr_loop,
            "review_regulatory_candidate_resolution",
        ) as resolution_reviewer,
        pytest.raises(EmptyLLMResponseError) as exc_info,
    ):
        _run_final_report(
            is_regulatory_research=True,
            live_state=live_state,
            destination=destination,
        )

    assert exc_info.value.finish_reason == "length"
    assert _packets(merged_queue) == []
    assert live_state.get_answer_tokens() is None
    resolution_reviewer.assert_not_called()


def test_regulatory_final_report_correction_provider_exception_propagates() -> None:
    live_state = ChatStateContainer()
    merged_queue: queue.Queue[tuple[int, object]] = queue.Queue()
    destination = Emitter(merged_queue)
    review = CandidateAnswerReviewResult(
        needs_reconsideration=True,
        advisory_claim_issues=[
            CandidateAnswerClaimIssue(
                claim_reference="Unsupported conclusion",
                advisory_feedback="Qualify the unsupported scope.",
                related_citation_numbers=[1],
            )
        ],
    )
    initial_step = _fake_llm_step(["Rejected initial answer [1]"])
    step_number = 0

    def fail_correction(**kwargs: Any) -> tuple[LlmStepResult, bool]:
        nonlocal step_number
        step_number += 1
        if step_number == 1:
            return initial_step(**kwargs)
        raise RuntimeError("correction provider failed")

    with (
        patch.object(
            dr_loop,
            "run_llm_step",
            side_effect=fail_correction,
        ) as run_step,
        patch.object(
            dr_loop,
            "review_regulatory_candidate_answer",
            return_value=review,
        ),
        patch.object(
            dr_loop,
            "review_regulatory_candidate_resolution",
        ) as resolution_reviewer,
        pytest.raises(RuntimeError, match="correction provider failed"),
    ):
        _run_final_report(
            is_regulatory_research=True,
            live_state=live_state,
            destination=destination,
        )

    assert run_step.call_count == 2
    resolution_reviewer.assert_not_called()
    assert _packets(merged_queue) == []
    assert live_state.get_answer_tokens() is None


def test_regulatory_final_report_correction_review_exception_uses_source_gap() -> None:
    live_state = ChatStateContainer()
    merged_queue: queue.Queue[tuple[int, object]] = queue.Queue()
    destination = Emitter(merged_queue)
    review = CandidateAnswerReviewResult(
        needs_reconsideration=True,
        advisory_claim_issues=[
            CandidateAnswerClaimIssue(
                claim_reference="Unsupported conclusion",
                advisory_feedback="Qualify the unsupported scope.",
                related_citation_numbers=[1],
            )
        ],
    )

    with (
        patch.object(
            dr_loop,
            "run_llm_step",
            side_effect=_fake_llm_step(
                ["Rejected initial answer [1]", "Corrected answer [1]"]
            ),
        ),
        patch.object(
            dr_loop,
            "review_regulatory_candidate_answer",
            return_value=review,
        ),
        patch.object(
            dr_loop,
            "review_regulatory_candidate_resolution",
            side_effect=RuntimeError("review provider failed"),
        ),
    ):
        has_reasoned = _run_final_report(
            is_regulatory_research=True,
            live_state=live_state,
            destination=destination,
        )

    packets = _packets(merged_queue)
    assert [
        packet.obj.content
        for packet in packets
        if isinstance(packet.obj, AgentResponseDelta)
    ] == [dr_loop._REGULATORY_SOURCE_GAP_FALLBACK]
    assert not any(isinstance(packet.obj, CitationInfo) for packet in packets)
    assert live_state.get_citation_to_doc() == {}
    assert has_reasoned is False


def test_regulatory_final_report_reviews_even_without_exact_evidence() -> None:
    live_state = ChatStateContainer()
    merged_queue: queue.Queue[tuple[int, object]] = queue.Queue()
    destination = Emitter(merged_queue)
    review = CandidateAnswerReviewResult(
        needs_reconsideration=True,
        advisory_claim_issues=[
            CandidateAnswerClaimIssue(
                claim_reference="Unsupported definite conclusion",
                advisory_feedback="No exact source text supports this conclusion.",
            )
        ],
    )

    with (
        patch.object(
            dr_loop,
            "run_llm_step",
            side_effect=_fake_llm_step(
                ["Unsupported hidden draft", "Qualified final answer"]
            ),
        ) as run_step,
        patch.object(
            dr_loop,
            "review_regulatory_candidate_answer",
            return_value=review,
        ) as reviewer,
        patch.object(
            dr_loop,
            "review_regulatory_candidate_resolution",
            return_value=CandidateAnswerReviewResult(needs_reconsideration=False),
        ),
    ):
        _run_final_report(
            is_regulatory_research=True,
            live_state=live_state,
            destination=destination,
            exact_evidence_chunks=[],
        )

    reviewer.assert_called_once()
    assert reviewer.call_args.kwargs["evidence_chunks"] == []
    assert run_step.call_args_list[1].kwargs["citation_processor"].citation_to_doc == {}
    assert live_state.get_answer_tokens() == "Qualified final answer"


def test_regulatory_final_report_separates_current_request_from_earlier_context() -> (
    None
):
    live_state = ChatStateContainer()
    merged_queue: queue.Queue[tuple[int, object]] = queue.Queue()
    destination = Emitter(merged_queue)
    review = CandidateAnswerReviewResult(
        needs_reconsideration=True,
        advisory_claim_issues=[
            CandidateAnswerClaimIssue(
                claim_reference="Current authorization effect",
                advisory_feedback="Qualify the current conclusion.",
                related_citation_numbers=[1],
            )
        ],
    )
    history = [
        ChatMessageSimple(
            message="Earlier jurisdiction and event facts.",
            token_count=5,
            message_type=MessageType.USER,
        ),
        ChatMessageSimple(
            message="Earlier assistant answer.",
            token_count=4,
            message_type=MessageType.ASSISTANT,
        ),
        ChatMessageSimple(
            message="What is its current effect on the authorization?",
            token_count=8,
            message_type=MessageType.USER,
        ),
    ]

    with (
        patch.object(
            dr_loop,
            "run_llm_step",
            side_effect=_fake_llm_step(
                ["Hidden draft [1]", "Corrected current answer [1]"]
            ),
        ) as run_step,
        patch.object(
            dr_loop,
            "review_regulatory_candidate_answer",
            return_value=review,
        ) as reviewer,
        patch.object(
            dr_loop,
            "review_regulatory_candidate_resolution",
            return_value=CandidateAnswerReviewResult(needs_reconsideration=False),
        ),
    ):
        _run_final_report(
            is_regulatory_research=True,
            live_state=live_state,
            destination=destination,
            history=history,
        )

    assert reviewer.call_args.kwargs["user_request"] == (
        "What is its current effect on the authorization?"
    )
    assert reviewer.call_args.kwargs["earlier_user_context"] == (
        "Earlier jurisdiction and event facts.",
    )
    correction_history = cast(
        list[ChatMessageSimple], run_step.call_args_list[1].kwargs["history"]
    )
    correction_user_messages = [
        message.message
        for message in correction_history
        if message.message_type == MessageType.USER
    ]
    assert correction_user_messages == [
        "What is its current effect on the authorization?"
    ]
    correction_reminder = next(
        message.message
        for message in correction_history
        if message.message_type == MessageType.USER_REMINDER
    )
    assert "Earlier jurisdiction and event facts." in correction_reminder
    assert "not additional deliverables" in correction_reminder
    assert "Earlier assistant answer." not in correction_reminder


def test_regulatory_final_report_rechecks_and_repairs_unresolved_correction() -> None:
    public_doc = _search_doc()
    unformatted_doc = _hidden_search_doc(2, document_id="document-2")
    live_state = ChatStateContainer()
    merged_queue: queue.Queue[tuple[int, object]] = queue.Queue()
    destination = Emitter(merged_queue)
    initial_issue = CandidateAnswerClaimIssue(
        claim_reference="Unsupported conclusion",
        advisory_feedback="The exact text does not support the stated scope.",
        related_citation_numbers=[1],
    )
    initial_review = CandidateAnswerReviewResult(
        needs_reconsideration=True,
        advisory_claim_issues=[initial_issue],
    )
    unresolved_review = CandidateAnswerReviewResult(
        needs_reconsideration=True,
        advisory_claim_issues=[initial_issue],
    )

    with (
        patch.object(
            dr_loop,
            "run_llm_step",
            side_effect=_fake_llm_step(
                [
                    "Rejected hidden draft [1]",
                    "Still unsupported correction [1]",
                    "Final qualified answer [1]",
                ]
            ),
        ) as run_step,
        patch.object(
            dr_loop,
            "review_regulatory_candidate_answer",
            return_value=initial_review,
        ),
        patch.object(
            dr_loop,
            "review_regulatory_candidate_resolution",
            side_effect=[
                unresolved_review,
                CandidateAnswerReviewResult(needs_reconsideration=False),
            ],
        ) as resolution_reviewer,
    ):
        _run_final_report(
            is_regulatory_research=True,
            live_state=live_state,
            destination=destination,
            citation_mapping={1: public_doc},
            evidence_citation_mapping={2: unformatted_doc},
            custom_agent_prompt="Act as a customs compliance specialist.",
        )

    assert resolution_reviewer.call_count == 2
    assert resolution_reviewer.call_args_list[0].kwargs["candidate_answer"] == (
        "Still unsupported correction [1]"
    )
    assert resolution_reviewer.call_args_list[1].kwargs["candidate_answer"] == (
        "Final qualified answer [1]"
    )
    assert run_step.call_count == 3
    assert run_step.call_args_list[1].kwargs["citation_processor"].citation_to_doc == {
        1: public_doc
    }
    assert run_step.call_args_list[2].kwargs["citation_processor"].citation_to_doc == {
        1: public_doc
    }
    final_history = cast(
        list[ChatMessageSimple], run_step.call_args_list[2].kwargs["history"]
    )
    assert all(
        any(
            message.message == "Act as a customs compliance specialist."
            for message in cast(list[ChatMessageSimple], call.kwargs["history"])
        )
        for call in run_step.call_args_list[1:]
    )
    assert any(
        "Revised candidate resolution review" in message.message
        and "EXACT LLM-VISIBLE OPERATIVE TEXT" in message.message
        for message in final_history
    )
    packets = _packets(merged_queue)
    assert [
        packet.obj.content
        for packet in packets
        if isinstance(packet.obj, AgentResponseDelta)
    ] == ["Final qualified answer [1]"]
    assert live_state.get_answer_tokens() == "Final qualified answer [1]"


def test_regulatory_final_report_retries_truncated_final_correction() -> None:
    live_state = ChatStateContainer()
    merged_queue: queue.Queue[tuple[int, object]] = queue.Queue()
    destination = Emitter(merged_queue)
    issue = CandidateAnswerClaimIssue(
        claim_reference="Unsupported conclusion",
        advisory_feedback="The exact text does not support the stated scope.",
        related_citation_numbers=[1],
    )
    initial_review = CandidateAnswerReviewResult(
        needs_reconsideration=True,
        advisory_claim_issues=[issue],
    )
    unresolved_review = CandidateAnswerReviewResult(
        needs_reconsideration=True,
        advisory_claim_issues=[issue],
    )

    with (
        patch.object(
            dr_loop,
            "run_llm_step",
            side_effect=_fake_llm_step_with_finish_reasons(
                [
                    ("Rejected hidden draft [1]", "stop"),
                    ("Still unsupported correction [1]", "stop"),
                    ("Incomplete final correction [1]", "length"),
                    ("Complete final correction [1]", "stop"),
                ]
            ),
        ) as run_step,
        patch.object(
            dr_loop,
            "review_regulatory_candidate_answer",
            return_value=initial_review,
        ),
        patch.object(
            dr_loop,
            "review_regulatory_candidate_resolution",
            side_effect=[
                unresolved_review,
                CandidateAnswerReviewResult(needs_reconsideration=False),
            ],
        ) as resolution_reviewer,
    ):
        has_reasoned = _run_final_report(
            is_regulatory_research=True,
            live_state=live_state,
            destination=destination,
        )

    assert [call.kwargs["reasoning_effort"] for call in run_step.call_args_list] == [
        dr_loop.ReasoningEffort.LOW,
        dr_loop.ReasoningEffort.LOW,
        dr_loop.ReasoningEffort.LOW,
        dr_loop.ReasoningEffort.OFF,
    ]
    assert resolution_reviewer.call_count == 2
    packets = _packets(merged_queue)
    assert [
        packet.obj.content
        for packet in packets
        if isinstance(packet.obj, AgentResponseDelta)
    ] == ["Complete final correction [1]"]
    assert live_state.get_answer_tokens() == "Complete final correction [1]"
    assert has_reasoned is True


def test_regulatory_final_report_truncated_final_correction_exhaustion_is_typed() -> (
    None
):
    live_state = ChatStateContainer()
    merged_queue: queue.Queue[tuple[int, object]] = queue.Queue()
    destination = Emitter(merged_queue)
    issue = CandidateAnswerClaimIssue(
        claim_reference="Unsupported conclusion",
        advisory_feedback="The exact text does not support the stated scope.",
        related_citation_numbers=[1],
    )
    initial_review = CandidateAnswerReviewResult(
        needs_reconsideration=True,
        advisory_claim_issues=[issue],
    )
    unresolved_review = CandidateAnswerReviewResult(
        needs_reconsideration=True,
        advisory_claim_issues=[issue],
    )

    with (
        patch.object(
            dr_loop,
            "run_llm_step",
            side_effect=_fake_llm_step_with_finish_reasons(
                [
                    ("Rejected hidden draft [1]", "stop"),
                    ("Still unsupported correction [1]", "stop"),
                    ("Incomplete final correction one [1]", "length"),
                    ("Incomplete final correction two [1]", "length"),
                ]
            ),
        ),
        patch.object(
            dr_loop,
            "review_regulatory_candidate_answer",
            return_value=initial_review,
        ),
        patch.object(
            dr_loop,
            "review_regulatory_candidate_resolution",
            return_value=unresolved_review,
        ) as resolution_reviewer,
        pytest.raises(EmptyLLMResponseError) as exc_info,
    ):
        _run_final_report(
            is_regulatory_research=True,
            live_state=live_state,
            destination=destination,
        )

    assert exc_info.value.finish_reason == "length"
    resolution_reviewer.assert_called_once()
    assert _packets(merged_queue) == []
    assert live_state.get_answer_tokens() is None


def test_regulatory_final_report_final_review_exception_uses_source_gap() -> None:
    live_state = ChatStateContainer()
    merged_queue: queue.Queue[tuple[int, object]] = queue.Queue()
    destination = Emitter(merged_queue)
    issue = CandidateAnswerClaimIssue(
        claim_reference="Unsupported conclusion",
        advisory_feedback="The exact text does not support the stated scope.",
        related_citation_numbers=[1],
    )
    initial_review = CandidateAnswerReviewResult(
        needs_reconsideration=True,
        advisory_claim_issues=[issue],
    )
    unresolved_review = CandidateAnswerReviewResult(
        needs_reconsideration=True,
        advisory_claim_issues=[issue],
    )

    with (
        patch.object(
            dr_loop,
            "run_llm_step",
            side_effect=_fake_llm_step(
                [
                    "Rejected hidden draft [1]",
                    "Still unsupported correction [1]",
                    "Final correction [1]",
                ]
            ),
        ),
        patch.object(
            dr_loop,
            "review_regulatory_candidate_answer",
            return_value=initial_review,
        ),
        patch.object(
            dr_loop,
            "review_regulatory_candidate_resolution",
            side_effect=[
                unresolved_review,
                RuntimeError("final review provider failed"),
            ],
        ),
    ):
        has_reasoned = _run_final_report(
            is_regulatory_research=True,
            live_state=live_state,
            destination=destination,
        )

    packets = _packets(merged_queue)
    assert [
        packet.obj.content
        for packet in packets
        if isinstance(packet.obj, AgentResponseDelta)
    ] == [dr_loop._REGULATORY_SOURCE_GAP_FALLBACK]
    assert not any(isinstance(packet.obj, CitationInfo) for packet in packets)
    assert live_state.get_citation_to_doc() == {}
    assert has_reasoned is False


def test_regulatory_final_report_uses_source_gap_after_repeated_final_defect() -> None:
    live_state = ChatStateContainer()
    merged_queue: queue.Queue[tuple[int, object]] = queue.Queue()
    destination = Emitter(merged_queue)
    initial_issue = CandidateAnswerClaimIssue(
        claim_reference="Unsupported conclusion",
        advisory_feedback="The exact text does not support the stated scope.",
        related_citation_numbers=[1],
    )
    initial_review = CandidateAnswerReviewResult(
        needs_reconsideration=True,
        advisory_claim_issues=[initial_issue],
    )
    unresolved_review = CandidateAnswerReviewResult(
        needs_reconsideration=True,
        advisory_claim_issues=[initial_issue],
    )

    with (
        patch.object(
            dr_loop,
            "run_llm_step",
            side_effect=_fake_llm_step(
                [
                    "Rejected hidden draft [1]",
                    "Still unsupported correction [1]",
                    "Still unsupported final correction [1]",
                ]
            ),
        ) as run_step,
        patch.object(
            dr_loop,
            "review_regulatory_candidate_answer",
            return_value=initial_review,
        ),
        patch.object(
            dr_loop,
            "review_regulatory_candidate_resolution",
            side_effect=[unresolved_review, unresolved_review],
        ) as resolution_reviewer,
    ):
        has_reasoned = _run_final_report(
            is_regulatory_research=True,
            live_state=live_state,
            destination=destination,
        )

    assert run_step.call_count == 3
    assert resolution_reviewer.call_count == 2
    assert resolution_reviewer.call_args_list[1].kwargs["candidate_answer"] == (
        "Still unsupported final correction [1]"
    )
    packets = _packets(merged_queue)
    starts = [
        packet.obj for packet in packets if isinstance(packet.obj, AgentResponseStart)
    ]
    assert len(starts) == 1
    assert starts[0].final_documents == []
    assert not any(isinstance(packet.obj, CitationInfo) for packet in packets)
    assert [
        packet.obj.content
        for packet in packets
        if isinstance(packet.obj, AgentResponseDelta)
    ] == [dr_loop._REGULATORY_SOURCE_GAP_FALLBACK]
    assert live_state.get_answer_tokens() == dr_loop._REGULATORY_SOURCE_GAP_FALLBACK
    assert live_state.get_citation_to_doc() == {}
    assert live_state.get_emitted_citations() == set()
    assert has_reasoned is False


def test_research_agent_batch_caps_parallel_calls_and_aligns_parent_ids() -> None:
    calls = [
        ToolCallKickoff(
            tool_call_id="unexpected",
            tool_name="unexpected_tool",
            tool_args={},
            placement=Placement(turn_index=1),
        ),
        *[
            ToolCallKickoff(
                tool_call_id=f"research-{index}",
                tool_name=dr_loop.RESEARCH_AGENT_TOOL_NAME,
                tool_args={"task": f"Focused issue {index}"},
                placement=Placement(turn_index=1, tab_index=index),
            )
            for index in range(5)
        ],
    ]

    selected, parent_ids = dr_loop._bounded_research_agent_batch(calls)

    assert [call.tool_call_id for call in selected] == [
        "research-0",
        "research-1",
        "research-2",
    ]
    assert parent_ids == ["research-0", "research-1", "research-2"]

    final_slot, final_parent_ids = dr_loop._bounded_research_agent_batch(
        calls,
        remaining_call_budget=1,
    )
    assert [call.tool_call_id for call in final_slot] == ["research-0"]
    assert final_parent_ids == ["research-0"]

    exhausted, exhausted_parent_ids = dr_loop._bounded_research_agent_batch(
        calls,
        remaining_call_budget=0,
    )
    assert exhausted == []
    assert exhausted_parent_ids == []


def test_regulatory_research_task_runtime_guard_rejects_without_truncation() -> None:
    oversized_task = "x" * (dr_loop.MAX_RESEARCH_AGENT_TASK_CHARS + 1)
    call = ToolCallKickoff(
        tool_call_id="research-oversized",
        tool_name=dr_loop.RESEARCH_AGENT_TOOL_NAME,
        tool_args={"task": oversized_task},
        placement=Placement(turn_index=1),
    )

    rejection = dr_loop._regulatory_research_task_rejection(call)

    assert rejection is not None
    assert "too broad" in rejection
    assert call.tool_args["task"] == oversized_task


def test_rejected_research_task_feedback_pairs_tool_call_for_retry() -> None:
    call = ToolCallKickoff(
        tool_call_id="research-invalid",
        tool_name=dr_loop.RESEARCH_AGENT_TOOL_NAME,
        tool_args={"task": " "},
        placement=Placement(turn_index=1),
    )
    rejection = dr_loop._regulatory_research_task_rejection(call)
    assert rejection is not None
    history: list[ChatMessageSimple] = []

    dr_loop._append_rejected_research_agent_feedback(
        history,
        [(call, rejection)],
        len,
    )

    assert len(history) == 2
    assert history[0].message_type == MessageType.ASSISTANT
    assert history[0].tool_calls is not None
    assert history[0].tool_calls[0].tool_call_id == call.tool_call_id
    assert history[1].message_type == MessageType.TOOL_CALL_RESPONSE
    assert history[1].tool_call_id == call.tool_call_id
    assert "focused research fragment" in history[1].message


def test_unrun_research_task_feedback_is_short_and_retryable() -> None:
    history: list[ChatMessageSimple] = []

    dr_loop._append_unrun_research_agent_feedback(
        history,
        unrun_call_count=2,
        total_budget_exhausted=False,
        token_counter=len,
    )

    assert len(history) == 1
    reminder = history[0]
    assert reminder.message_type == MessageType.USER_REMINDER
    assert reminder.token_count == len(reminder.message)
    assert len(reminder.message) < 400
    assert "2 proposed research_agent call(s) were not run" in reminder.message
    assert "produced no evidence" in reminder.message
    assert "reassess any material unresolved gap" in reminder.message
    assert "emit a needed focused fragment again if useful" in reminder.message
    assert "Focused issue" not in reminder.message


def test_unrun_research_task_feedback_marks_exhausted_total_budget() -> None:
    history: list[ChatMessageSimple] = []

    dr_loop._append_unrun_research_agent_feedback(
        history,
        unrun_call_count=3,
        total_budget_exhausted=True,
        token_counter=len,
    )

    assert len(history) == 1
    reminder = history[0]
    assert reminder.message_type == MessageType.USER_REMINDER
    assert "total research-agent call budget is exhausted" in reminder.message
    assert "produced no evidence" in reminder.message
    assert "do not treat their topics as researched" in reminder.message
    assert "Preserve any material unresolved gap" in reminder.message
    assert "emit a needed focused fragment again" not in reminder.message


def test_unrun_research_task_feedback_ignores_zero_count() -> None:
    history: list[ChatMessageSimple] = []

    dr_loop._append_unrun_research_agent_feedback(
        history,
        unrun_call_count=0,
        total_budget_exhausted=False,
        token_counter=len,
    )

    assert history == []


def test_non_regulatory_final_report_stages_success_before_publishing() -> None:
    live_state = ChatStateContainer()
    merged_queue: queue.Queue[tuple[int, object]] = queue.Queue()
    destination = Emitter(merged_queue)

    with (
        patch.object(
            dr_loop,
            "run_llm_step",
            side_effect=_fake_llm_step(["Direct non-regulatory answer"]),
        ) as run_step,
        patch.object(
            dr_loop,
            "review_regulatory_candidate_answer",
        ) as reviewer,
    ):
        _run_final_report(
            is_regulatory_research=False,
            live_state=live_state,
            destination=destination,
        )

    reviewer.assert_not_called()
    assert run_step.call_count == 1
    assert run_step.call_args.kwargs["emitter"] is not destination
    assert run_step.call_args.kwargs["state_container"] is not live_state
    assert run_step.call_args.kwargs["max_tokens"] == dr_loop.MAX_FINAL_REPORT_TOKENS
    assert live_state.get_answer_tokens() == "Direct non-regulatory answer"


@pytest.mark.parametrize("is_regulatory_research", [False, True])
def test_final_report_retries_length_at_off_without_leaking_partial_packets(
    is_regulatory_research: bool,
) -> None:
    live_state = ChatStateContainer()
    merged_queue: queue.Queue[tuple[int, object]] = queue.Queue()
    destination = Emitter(merged_queue)

    with (
        patch.object(
            dr_loop,
            "run_llm_step",
            side_effect=_fake_llm_step_with_finish_reasons(
                [
                    ("Incomplete first draft", "length"),
                    ("Complete retry report", "stop"),
                ]
            ),
        ) as run_step,
        patch.object(
            dr_loop,
            "review_regulatory_candidate_answer",
            return_value=CandidateAnswerReviewResult(needs_reconsideration=False),
        ),
    ):
        _run_final_report(
            is_regulatory_research=is_regulatory_research,
            live_state=live_state,
            destination=destination,
            reasoning_effort=dr_loop.ReasoningEffort.HIGH,
        )

    assert [call.kwargs["reasoning_effort"] for call in run_step.call_args_list] == [
        dr_loop.ReasoningEffort.LOW,
        dr_loop.ReasoningEffort.OFF,
    ]
    deltas = [
        packet.obj.content
        for packet in _packets(merged_queue)
        if isinstance(packet.obj, AgentResponseDelta)
    ]
    assert deltas == ["Complete retry report"]
    assert live_state.get_answer_tokens() == "Complete retry report"


def test_non_regulatory_final_report_two_empty_attempts_raise_terminal_error() -> None:
    live_state = ChatStateContainer()
    merged_queue: queue.Queue[tuple[int, object]] = queue.Queue()
    destination = Emitter(merged_queue)

    with (
        patch.object(
            dr_loop,
            "run_llm_step",
            side_effect=_fake_llm_step_with_finish_reasons(
                [("", "length"), ("", "length")]
            ),
        ),
        pytest.raises(EmptyLLMResponseError) as exc_info,
    ):
        _run_final_report(
            is_regulatory_research=False,
            live_state=live_state,
            destination=destination,
            reasoning_effort=dr_loop.ReasoningEffort.AUTO,
        )

    assert exc_info.value.finish_reason == "length"
    assert exc_info.value.provider == "vertex_ai"
    assert exc_info.value.model == "gemini-3.6-flash"
    assert _packets(merged_queue) == []
    assert live_state.get_answer_tokens() is None


def test_regulatory_empty_final_report_uses_bounded_source_gap_fallback() -> None:
    live_state = ChatStateContainer()
    merged_queue: queue.Queue[tuple[int, object]] = queue.Queue()

    with patch.object(
        dr_loop,
        "run_llm_step",
        side_effect=_fake_llm_step_with_finish_reasons(
            [("", "length"), ("", "length")]
        ),
    ):
        has_reasoned = _run_final_report(
            is_regulatory_research=True,
            live_state=live_state,
            destination=Emitter(merged_queue),
            reasoning_effort=dr_loop.ReasoningEffort.AUTO,
        )

    assert has_reasoned is False
    assert live_state.get_answer_tokens() == dr_loop._REGULATORY_SOURCE_GAP_FALLBACK
    assert [
        packet.obj.content
        for packet in _packets(merged_queue)
        if isinstance(packet.obj, AgentResponseDelta)
    ] == [dr_loop._REGULATORY_SOURCE_GAP_FALLBACK]


def test_regulatory_final_retry_flattens_completed_research_reports() -> None:
    live_state = ChatStateContainer()
    live_state.add_tool_call(
        ToolCallInfo(
            parent_tool_call_id=None,
            turn_index=2,
            tab_index=0,
            tool_name=dr_loop.RESEARCH_AGENT_TOOL_NAME,
            tool_call_id="research-1",
            tool_id=9,
            reasoning_tokens=None,
            tool_call_arguments={"task": "Find the controlling rule"},
            tool_call_response="RESEARCH_REPORT_SENTINEL [1]",
            search_docs=None,
            generated_images=None,
        )
    )
    merged_queue: queue.Queue[tuple[int, object]] = queue.Queue()

    with (
        patch.object(
            dr_loop,
            "run_llm_step",
            side_effect=_fake_llm_step_with_finish_reasons(
                [("", "length"), ("Compact grounded report [1]", "stop")]
            ),
        ) as run_step,
        patch.object(
            dr_loop,
            "review_regulatory_candidate_answer",
            return_value=CandidateAnswerReviewResult(needs_reconsideration=False),
        ),
    ):
        _run_final_report(
            is_regulatory_research=True,
            live_state=live_state,
            destination=Emitter(merged_queue),
        )

    retry_history = cast(
        list[ChatMessageSimple], run_step.call_args_list[1].kwargs["history"]
    )
    assert any(
        "RESEARCH_REPORT_SENTINEL" in message.message for message in retry_history
    )
    assert all(not message.tool_calls for message in retry_history)
    assert live_state.get_answer_tokens() == "Compact grounded report [1]"


def test_final_report_retries_twice_when_request_effort_is_off() -> None:
    merged_queue: queue.Queue[tuple[int, object]] = queue.Queue()

    with patch.object(
        dr_loop,
        "run_llm_step",
        side_effect=_fake_llm_step_with_finish_reasons(
            [("", "length"), ("Complete retry report", "stop")]
        ),
    ) as run_step:
        _run_final_report(
            is_regulatory_research=False,
            live_state=ChatStateContainer(),
            destination=Emitter(merged_queue),
            reasoning_effort=dr_loop.ReasoningEffort.OFF,
        )

    assert [call.kwargs["reasoning_effort"] for call in run_step.call_args_list] == [
        dr_loop.ReasoningEffort.LOW,
        dr_loop.ReasoningEffort.OFF,
    ]


def test_deep_research_search_allowlist_rejects_same_name_non_search_tool() -> None:
    real_search = _recovery_search_tool()
    same_name_custom_tool = MagicMock()
    same_name_custom_tool.name = SearchTool.NAME

    allowed = dr_loop._deep_research_search_tools([same_name_custom_tool, real_search])

    assert allowed == [real_search]


def test_final_report_history_includes_custom_agent_prompt() -> None:
    live_state = ChatStateContainer()
    merged_queue: queue.Queue[tuple[int, object]] = queue.Queue()
    destination = Emitter(merged_queue)

    with (
        patch.object(
            dr_loop,
            "construct_message_history",
            wraps=dr_loop.construct_message_history,
        ) as construct_history,
        patch.object(
            dr_loop,
            "run_llm_step",
            side_effect=_fake_llm_step(["Complete report"]),
        ),
    ):
        _run_final_report(
            is_regulatory_research=False,
            live_state=live_state,
            destination=destination,
            custom_agent_prompt="Act as a customs compliance specialist.",
        )

    prompt_message = construct_history.call_args.kwargs["custom_agent_prompt"]
    assert prompt_message.message == "Act as a customs compliance specialist."
    assert prompt_message.message_type == MessageType.USER


def test_custom_agent_prompt_is_in_clarification_history() -> None:
    merged_queue: queue.Queue[tuple[int, object]] = queue.Queue()

    with (
        patch.object(dr_loop, "SKIP_DEEP_RESEARCH_CLARIFICATION", False),
        patch.object(
            dr_loop,
            "construct_message_history",
            wraps=dr_loop.construct_message_history,
        ) as construct_history,
        patch.object(
            dr_loop,
            "run_llm_step",
            return_value=(
                LlmStepResult(
                    reasoning=None,
                    answer="Please clarify the jurisdiction.",
                    tool_calls=None,
                ),
                False,
            ),
        ),
        patch(
            "onyx.llm.litellm_singleton.config.initialize_litellm",
        ),
    ):
        dr_loop.run_deep_research_llm_loop(
            emitter=Emitter(merged_queue),
            state_container=ChatStateContainer(),
            simple_chat_history=_history(),
            tools=[],
            custom_agent_prompt="Act as a customs compliance specialist.",
            llm=_llm(),
            token_counter=len,
            skip_clarification=False,
        )

    prompt_message = construct_history.call_args.kwargs["custom_agent_prompt"]
    assert prompt_message.message == "Act as a customs compliance specialist."


def test_custom_agent_prompt_is_in_plan_orchestration_and_report_calls() -> None:
    merged_queue: queue.Queue[tuple[int, object]] = queue.Queue()

    def plan_generator(**_kwargs: Any):
        if False:
            yield None
        return (
            LlmStepResult(
                reasoning=None,
                answer="Research the applicable customs rules.",
                tool_calls=None,
            ),
            False,
        )

    report_call = ToolCallKickoff(
        tool_call_id="generate-report",
        tool_name="generate_report",
        tool_args={},
        placement=Placement(turn_index=2),
    )
    with (
        patch.object(
            dr_loop,
            "construct_message_history",
            wraps=dr_loop.construct_message_history,
        ) as construct_history,
        patch.object(
            dr_loop,
            "run_llm_step_pkt_generator",
            side_effect=plan_generator,
        ),
        patch.object(
            dr_loop,
            "run_llm_step",
            return_value=(
                LlmStepResult(
                    reasoning=None,
                    answer=None,
                    tool_calls=[report_call],
                ),
                False,
            ),
        ),
        patch.object(
            dr_loop,
            "generate_final_report",
            return_value=False,
        ) as generate_report,
        patch.object(dr_loop, "model_is_reasoning_model", return_value=True),
        patch(
            "onyx.llm.litellm_singleton.config.initialize_litellm",
        ),
    ):
        dr_loop.run_deep_research_llm_loop(
            emitter=Emitter(merged_queue),
            state_container=ChatStateContainer(),
            simple_chat_history=_history(),
            tools=[],
            custom_agent_prompt="Act as a customs compliance specialist.",
            llm=_llm(),
            token_counter=len,
            skip_clarification=True,
        )

    assert len(construct_history.call_args_list) == 2
    assert all(
        call.kwargs["custom_agent_prompt"].message
        == "Act as a customs compliance specialist."
        for call in construct_history.call_args_list
    )
    assert generate_report.call_args.kwargs["custom_agent_prompt"] == (
        "Act as a customs compliance specialist."
    )


def test_regulatory_deep_research_carries_structured_coverage_into_plan() -> None:
    merged_queue: queue.Queue[tuple[int, object]] = queue.Queue()

    def plan_generator(**_kwargs: Any):
        if False:
            yield None
        return (
            LlmStepResult(
                reasoning=None,
                answer="Research the unresolved coverage rows.",
                tool_calls=None,
            ),
            False,
        )

    coverage_plan = RegulatoryCoveragePlan(
        coverage_items=[
            RegulatoryCoverageItem(
                research_question="Resolve the expressly requested legal result.",
                evidence_dimensions=["Controlling trigger", "Direct consequence"],
                source_anchors=["Named Instrument"],
                request_segment_ids=["R1"],
                completion_test="Close the requested result with exact evidence.",
            )
        ]
    )
    report_call = ToolCallKickoff(
        tool_call_id="generate-report",
        tool_name="generate_report",
        tool_args={},
        placement=Placement(turn_index=2),
    )

    review_llm = MagicMock()
    with (
        patch.object(
            dr_loop,
            "build_regulatory_review_llm",
            return_value=review_llm,
        ),
        patch.object(
            dr_loop,
            "build_regulatory_coverage_plan",
            return_value=coverage_plan,
        ) as build_coverage,
        patch.object(
            dr_loop,
            "_run_regulatory_coverage_searches",
            return_value=({1: _search_doc()}, [_evidence()]),
        ) as run_coverage_searches,
        patch.object(
            dr_loop,
            "generate_final_report",
            return_value=False,
        ) as generate_report,
        patch.object(
            dr_loop,
            "run_llm_step_pkt_generator",
            side_effect=plan_generator,
        ) as run_plan_generator,
        patch.object(
            dr_loop,
            "run_llm_step",
            return_value=(
                LlmStepResult(
                    reasoning=None,
                    answer=None,
                    tool_calls=[report_call],
                ),
                False,
            ),
        ),
        patch.object(dr_loop, "model_is_reasoning_model", return_value=True),
        patch("onyx.llm.litellm_singleton.config.initialize_litellm"),
    ):
        dr_loop.run_deep_research_llm_loop(
            emitter=Emitter(merged_queue),
            state_container=ChatStateContainer(),
            simple_chat_history=_history(),
            tools=[_recovery_search_tool()],
            custom_agent_prompt=None,
            llm=_llm(),
            token_counter=len,
            skip_clarification=True,
        )

    build_coverage.assert_called_once_with(
        review_llm,
        user_request="Analyze every material legal issue.",
    )
    run_coverage_searches.assert_called_once()
    run_plan_generator.assert_called_once()
    final_history = generate_report.call_args.kwargs["history"]
    assert any(
        message.message_type == MessageType.USER_REMINDER
        and "# Request coverage contract" in message.message
        and "Named Instrument" in message.message
        for message in final_history
    )
    assert any(
        message.message_type == MessageType.USER_REMINDER
        and "# Retrieved exact evidence matrix" in message.message
        and "EXACT LLM-VISIBLE OPERATIVE TEXT" in message.message
        and "bootstrap evidence" in message.message
        for message in final_history
    )
    research_plan = generate_report.call_args.kwargs["research_plan"]
    assert research_plan.startswith("# Request coverage contract")
    assert research_plan.endswith("Research the unresolved coverage rows.")
    assert generate_report.call_args.kwargs["citation_mapping"] == {1: _search_doc()}
    assert generate_report.call_args.kwargs["evidence_citation_mapping"] == {
        1: _search_doc()
    }
    assert generate_report.call_args.kwargs["exact_evidence_chunks"] == [_evidence()]
