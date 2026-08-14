import json
from unittest.mock import MagicMock, patch

import pytest

from onyx.chat.models import ChatMessageSimple
from onyx.configs.chat_configs import REGULATORY_REVIEW_TIMEOUT_S
from onyx.configs.constants import MessageType
from onyx.context.search.models import SearchDocsResponse
from onyx.llm.interfaces import LLMConfig
from onyx.llm.models import ReasoningEffort
from onyx.prompts.regulatory_candidate_answer_review import (
    REGULATORY_CANDIDATE_ANSWER_REVIEW_SYSTEM_PROMPT,
    REGULATORY_CANDIDATE_RESOLUTION_REVIEW_SYSTEM_PROMPT,
    REGULATORY_EVIDENCE_MATRIX_CLOSURE_REVIEW_SYSTEM_PROMPT,
)
from onyx.regulatory.candidate_answer_review import (
    CandidateAnswerClaimIssue,
    CandidateAnswerClaimSpan,
    CandidateAnswerEvidenceChunk,
    CandidateAnswerIssueResolutionStatus,
    CandidateAnswerReviewError,
    CandidateAnswerReviewResult,
    ClaimKind,
    ClaimSpanSource,
    _CandidateAnswerIssueResolutionDraft,
    _CandidateAnswerResolutionReviewDraft,
    _CandidateAnswerReviewDraft,
    _CandidateAnswerReviewDraftClaimIssue,
    _compact_evidence_chunks,
    build_candidate_answer_evidence_chunk,
    build_regulatory_review_llm,
    build_regulatory_review_user_context,
    build_regulatory_review_user_request,
    format_candidate_correction_evidence,
    review_regulatory_candidate_answer,
    review_regulatory_candidate_answer_with_fallback,
    review_regulatory_candidate_matrix_closure,
    review_regulatory_candidate_resolution,
    review_regulatory_candidate_resolution_with_fallback,
)
from onyx.regulatory.gap_recovery import (
    run_batched_gap_recovery,
    select_priority_recovery_issue,
    select_priority_recovery_issues,
)
from onyx.server.query_and_chat.placement import Placement
from onyx.tools.models import ToolResponse
from onyx.tracing.flows import LLMFlow


@pytest.fixture(autouse=True)
def _stable_candidate_review_context_budget():
    with (
        patch(
            "onyx.regulatory.candidate_answer_review.get_llm_token_counter",
            return_value=lambda text: max(1, len(text) // 4),
        ),
        patch(
            "onyx.regulatory.candidate_answer_review._selected_review_max_input_tokens",
            return_value=1_000_000,
        ),
    ):
        yield


def _evidence(
    *, citation_number: int | None = 1, content: str = "Operative text."
) -> CandidateAnswerEvidenceChunk:
    return build_candidate_answer_evidence_chunk(
        document_id=f"document-{citation_number}",
        chunk_id=citation_number or 0,
        citation_number=citation_number,
        chunk_identifier=f"chunk-{citation_number}",
        heading=f"Instrument > Provision {citation_number}",
        content=content,
    )


def test_regulatory_review_llm_defaults_to_answer_model() -> None:
    answer_llm = MagicMock()
    answer_llm.config.model_name = "candidate-model"

    with patch("onyx.regulatory.candidate_answer_review.REGULATORY_REVIEW_MODEL", None):
        assert build_regulatory_review_llm(answer_llm) is answer_llm


def test_regulatory_review_llm_reuses_provider_route_for_stronger_model() -> None:
    answer_llm = MagicMock()
    answer_llm.config = LLMConfig(
        model_provider="vertex_ai",
        model_name="candidate-model",
        temperature=0.2,
        api_key=None,
        api_base=None,
        api_version=None,
        deployment_name=None,
        custom_config={"vertex_project": "project"},
        max_input_tokens=250_000,
    )
    review_llm = MagicMock()

    with (
        patch(
            "onyx.regulatory.candidate_answer_review.REGULATORY_REVIEW_MODEL",
            "review-model",
        ),
        patch(
            "onyx.regulatory.candidate_answer_review.get_llm",
            return_value=review_llm,
        ) as get_llm,
    ):
        result = build_regulatory_review_llm(answer_llm)

    assert result is review_llm
    get_llm.assert_called_once_with(
        provider="vertex_ai",
        model="review-model",
        max_input_tokens=250_000,
        deployment_name=None,
        api_key=None,
        api_base=None,
        api_version=None,
        custom_config={"vertex_project": "project"},
        temperature=0,
        timeout=REGULATORY_REVIEW_TIMEOUT_S,
    )


def test_unavailable_independent_candidate_review_retries_answer_model() -> None:
    review_llm = MagicMock()
    answer_llm = MagicMock()
    unavailable = CandidateAnswerReviewResult(
        needs_reconsideration=False,
        review_error=CandidateAnswerReviewError.REVIEW_UNAVAILABLE,
    )
    completed = CandidateAnswerReviewResult(needs_reconsideration=False)

    with patch(
        "onyx.regulatory.candidate_answer_review.review_regulatory_candidate_answer",
        side_effect=[unavailable, completed],
    ) as review:
        result = review_regulatory_candidate_answer_with_fallback(
            review_llm,
            answer_llm,
            user_request="Request",
            candidate_answer="Answer",
            evidence_chunks=[],
        )

    assert result is completed
    assert [call.args[0] for call in review.call_args_list] == [review_llm, answer_llm]


def test_unavailable_resolution_review_retries_answer_model() -> None:
    review_llm = MagicMock()
    answer_llm = MagicMock()
    issue = CandidateAnswerClaimIssue(
        claim_reference="Claim",
        advisory_feedback="Missing support",
    )
    unavailable = CandidateAnswerReviewResult(
        needs_reconsideration=False,
        review_error=CandidateAnswerReviewError.REVIEW_UNAVAILABLE,
    )
    completed = CandidateAnswerReviewResult(needs_reconsideration=False)

    with patch(
        "onyx.regulatory.candidate_answer_review."
        "review_regulatory_candidate_resolution",
        side_effect=[unavailable, completed],
    ) as review:
        result = review_regulatory_candidate_resolution_with_fallback(
            review_llm,
            answer_llm,
            candidate_answer="Revised answer",
            prior_issues=[issue],
            evidence_chunks=[],
        )

    assert result is completed
    assert [call.args[0] for call in review.call_args_list] == [review_llm, answer_llm]


def test_evidence_keeps_canonical_chunk_identity() -> None:
    evidence = build_candidate_answer_evidence_chunk(
        document_id="doc",
        chunk_id=7,
        citation_number=1,
        chunk_identifier="Madde 5",
        heading="Madde 5",
        content="Metin",
    )

    assert (evidence.document_id, evidence.chunk_id) == ("doc", 7)


def test_cited_but_unsupported_legal_rule_is_recovery_eligible() -> None:
    legal_issue = CandidateAnswerClaimIssue(
        claim_kind=ClaimKind.LEGAL_RULE,
        claim_span=CandidateAnswerClaimSpan(start=42, end=63),
        claim_reference="İzin zorunludur [3].",
        advisory_feedback="Atıf bu zorunluluğu desteklemiyor.",
        related_citation_numbers=[3],
        recovery_query="izin zorunluluğu uygulanacak hüküm",
    )
    earlier_fact = CandidateAnswerClaimIssue(
        claim_kind=ClaimKind.MATERIAL_FACT,
        claim_span=CandidateAnswerClaimSpan(start=3, end=18),
        claim_reference="Başvuru yapıldı.",
        advisory_feedback="Bu olgu kaynakta yer almıyor.",
        recovery_query="başvuru yapılma tarihi",
    )
    review = CandidateAnswerReviewResult(
        needs_reconsideration=True,
        advisory_claim_issues=[earlier_fact, legal_issue],
    )

    selected = select_priority_recovery_issue(review)

    assert selected is legal_issue
    assert selected.claim_kind is ClaimKind.LEGAL_RULE
    assert selected.related_citation_numbers == [3]


def test_recovery_priority_uses_earliest_span_within_claim_kind() -> None:
    later = CandidateAnswerClaimIssue(
        claim_kind=ClaimKind.LEGAL_RULE,
        claim_span=CandidateAnswerClaimSpan(start=90, end=104),
        claim_reference="Sonraki yasak.",
        advisory_feedback="Doğrudan destek yok.",
        recovery_query="sonraki yasak hükmü",
    )
    earlier = CandidateAnswerClaimIssue(
        claim_kind=ClaimKind.LEGAL_RULE,
        claim_span=CandidateAnswerClaimSpan(start=10, end=25),
        claim_reference="Önceki istisna.",
        advisory_feedback="Doğrudan destek yok.",
        recovery_query="önceki istisna hükmü",
    )
    review = CandidateAnswerReviewResult(
        needs_reconsideration=True,
        advisory_claim_issues=[later, earlier],
    )

    assert select_priority_recovery_issue(review) is earlier


def test_batched_recovery_selects_five_query_distinct_issues() -> None:
    issues = [
        CandidateAnswerClaimIssue(
            claim_kind=ClaimKind.LEGAL_RULE,
            claim_reference=f"claim {index}",
            advisory_feedback="support gap",
            recovery_query=query,
        )
        for index, query in enumerate(
            (
                "rule one",
                "RULE   ONE",
                "rule two",
                "rule three",
                "rule four",
                "rule five",
            )
        )
    ]
    review = CandidateAnswerReviewResult(
        needs_reconsideration=True,
        advisory_claim_issues=issues,
    )

    assert select_priority_recovery_issues(review) == [
        issues[0],
        issues[2],
        issues[3],
        issues[4],
        issues[5],
    ]


def test_batched_recovery_uses_one_focused_search_call_per_issue() -> None:
    issues = [
        CandidateAnswerClaimIssue(
            claim_kind=ClaimKind.LEGAL_RULE,
            claim_reference=f"claim {index}",
            advisory_feedback="support gap",
            recovery_query=query,
        )
        for index, query in enumerate(("rule one", "rule two"))
    ]
    search_tool = MagicMock()
    search_tool.run.side_effect = [
        ToolResponse(
            rich_response=SearchDocsResponse(
                search_docs=[], citation_mapping={}, citation_chunk_mapping={}
            ),
            llm_facing_response=json.dumps(
                {"results": [], "receipt": {"query": query}}
            ),
        )
        for query in ("rule one", "rule two")
    ]
    placement = Placement(turn_index=4, tab_index=0)

    result = run_batched_gap_recovery(
        search_tool=search_tool,
        issues=issues,
        starting_citation_num=9,
        placement=placement,
    )

    assert search_tool.run.call_count == 2
    first_kwargs = search_tool.run.call_args_list[0].kwargs
    second_kwargs = search_tool.run.call_args_list[1].kwargs
    assert first_kwargs["queries"] == ["rule one"]
    assert second_kwargs["queries"] == ["rule two"]
    assert first_kwargs["search_mode"] == second_kwargs["search_mode"] == "hybrid"
    assert first_kwargs["placement"] == placement
    assert second_kwargs["placement"] == placement.model_copy(update={"tab_index": 1})
    assert first_kwargs["override_kwargs"].starting_citation_num == 9
    assert second_kwargs["override_kwargs"].starting_citation_num == 9
    assert first_kwargs["override_kwargs"].original_query == "rule one"
    assert second_kwargs["override_kwargs"].original_query == "rule two"
    assert isinstance(result.rich_response, SearchDocsResponse)
    assert json.loads(result.llm_facing_response)["recovery_receipts"] == [
        {"query": "rule one"},
        {"query": "rule two"},
    ]


def test_review_user_context_separates_current_request_from_earlier_facts() -> None:
    history = [
        ChatMessageSimple(
            message="Earlier jurisdiction and event facts.",
            token_count=5,
            message_type=MessageType.USER,
        ),
        ChatMessageSimple(
            message="Assistant speculation must not become request facts.",
            token_count=7,
            message_type=MessageType.ASSISTANT,
        ),
        ChatMessageSimple(
            message="What is its effect on the authorization?",
            token_count=7,
            message_type=MessageType.USER,
        ),
    ]

    context = build_regulatory_review_user_context(history)

    assert context.current_request == "What is its effect on the authorization?"
    assert context.earlier_user_context == ("Earlier jurisdiction and event facts.",)
    assert build_regulatory_review_user_request(history) == context.current_request


def test_review_user_context_does_not_carry_topic_switch_as_a_deliverable() -> None:
    history = [
        ChatMessageSimple(
            message="Analyze the permit revocation procedure.",
            token_count=6,
            message_type=MessageType.USER,
        ),
        ChatMessageSimple(
            message="New topic: when does the filing deadline expire?",
            token_count=8,
            message_type=MessageType.USER,
        ),
    ]

    context = build_regulatory_review_user_context(history)

    assert context.current_request == (
        "New topic: when does the filing deadline expire?"
    )
    assert context.earlier_user_context == ("Analyze the permit revocation procedure.",)


def test_review_user_context_keeps_correction_in_current_request() -> None:
    history = [
        ChatMessageSimple(
            message="The operator is established in France.",
            token_count=6,
            message_type=MessageType.USER,
        ),
        ChatMessageSimple(
            message=(
                "Correction: the operator is established in Italy. Reassess liability."
            ),
            token_count=10,
            message_type=MessageType.USER,
        ),
    ]

    context = build_regulatory_review_user_context(history)

    assert "Italy" in context.current_request
    assert "France" not in context.current_request
    assert context.earlier_user_context == ("The operator is established in France.",)


def test_evidence_builder_bounds_without_paraphrasing_source_text() -> None:
    content = "start:" + ("x" * 20_000) + ":end"

    evidence = _evidence(content=content)

    assert len(evidence.content) == 12_000
    assert evidence.content.startswith("start:")
    assert evidence.content.endswith(":end")
    assert evidence.content_truncated is True


def test_correction_evidence_can_cite_a_retrieved_chunk_omitted_by_draft() -> None:
    omitted_by_draft = build_candidate_answer_evidence_chunk(
        document_id="document-7",
        chunk_id=7,
        citation_number=None,
        retrieval_number=7,
        chunk_identifier="chunk-7",
        heading="Instrument > Provision 7",
        content="Exact condition omitted by the hidden draft.",
    )

    payload = json.loads(format_candidate_correction_evidence([omitted_by_draft]))

    assert payload["evidence_chunks"] == [
        {
            "document_id": "document-7",
            "chunk_id": 7,
            "citation_number": 7,
            "retrieval_number": 7,
            "chunk_identifier": "chunk-7",
            "heading": "Instrument > Provision 7",
            "research_target": "",
            "content": "Exact condition omitted by the hidden draft.",
            "content_truncated": False,
        }
    ]


def test_correction_evidence_has_a_separate_total_content_bound() -> None:
    evidence_chunks = [
        _evidence(citation_number=index + 1, content="x" * 12_000)
        for index in range(48)
    ]

    payload = json.loads(format_candidate_correction_evidence(evidence_chunks))

    assert len(payload["evidence_chunks"]) == 48
    assert sum(len(chunk["content"]) for chunk in payload["evidence_chunks"]) <= 120_000
    assert all(chunk["content_truncated"] for chunk in payload["evidence_chunks"])


def test_review_makes_one_bounded_structured_call() -> None:
    draft = _CandidateAnswerReviewDraft(
        needs_reconsideration=True,
        advisory_claim_issues=[
            _CandidateAnswerReviewDraftClaimIssue(
                claim_kind=ClaimKind.LEGAL_RULE,
                claim_reference="A material claim",
                advisory_feedback="The attributed chunk does not entail this effect.",
                related_citation_numbers=[1],
                recovery_query="material claim controlling legal rule",
            )
        ],
    )
    with patch(
        "onyx.regulatory.candidate_answer_review.generate_structured",
        return_value=draft,
    ) as generate:
        result = review_regulatory_candidate_answer(
            MagicMock(),
            user_request="Analyze the supplied facts.",
            earlier_user_context=["The authorization was issued last year."],
            coverage_contract="Close the distinct authorization consequence.",
            evidence_matrix="Supported matrix row with exact document 7.",
            candidate_answer="A material claim [1].",
            evidence_chunks=[_evidence()],
        )

    assert result.completed is True
    assert result.needs_reconsideration is True
    assert result.advisory_claim_issues == [
        CandidateAnswerClaimIssue(
            claim_kind=ClaimKind.LEGAL_RULE,
            claim_span=CandidateAnswerClaimSpan(start=0, end=16),
            claim_reference="A material claim",
            advisory_feedback="The attributed chunk does not entail this effect.",
            related_citation_numbers=[1],
            recovery_query="material claim controlling legal rule",
        )
    ]
    generate.assert_called_once()
    kwargs = generate.call_args.kwargs
    assert kwargs["flow"] is LLMFlow.REGULATORY_ANSWER_AUDIT
    assert kwargs["timeout_override"] == REGULATORY_REVIEW_TIMEOUT_S
    assert kwargs["max_tokens"] == 24_000
    assert kwargs["reasoning_effort"] is ReasoningEffort.HIGH
    assert kwargs["max_attempts"] == 2
    payload = json.loads(kwargs["user_prompt"])
    assert payload["user_request"] == {
        "text": "Analyze the supplied facts.",
        "truncated": False,
    }
    assert payload["earlier_user_context"] == [
        {
            "text": "The authorization was issued last year.",
            "truncated": False,
        }
    ]
    assert payload["coverage_contract"] == {
        "text": "Close the distinct authorization consequence.",
        "truncated": False,
    }
    assert payload["evidence_matrix"] == {
        "text": "Supported matrix row with exact document 7.",
        "truncated": False,
    }
    assert payload["candidate_answer"]["text"] == "A material claim [1]."
    assert payload["retrieval_inventory"] == {
        "total_result_count": 1,
        "represented_by_exact_evidence_count": 1,
        "inventory_only_result_count": 0,
        "included_result_count": 0,
        "truncated": False,
        "items": [],
    }
    assert payload["evidence_chunks"][0]["citation_number"] == 1
    assert payload["evidence_chunks"][0]["retrieval_number"] == 1
    assert payload["evidence_chunks"][0]["content"] == "Operative text."


def test_matrix_closure_review_uses_the_focused_prompt_and_payload() -> None:
    with patch(
        "onyx.regulatory.candidate_answer_review.generate_structured",
        return_value=_CandidateAnswerReviewDraft(
            needs_reconsideration=False,
            advisory_claim_issues=[],
        ),
    ) as generate:
        result = review_regulatory_candidate_matrix_closure(
            MagicMock(),
            user_request="Analyze the requested procedure.",
            candidate_answer="The supported result is stated [[1]]().",
            evidence_chunks=[_evidence()],
            evidence_matrix="One supported procedure row.",
        )

    assert result.completed is True
    assert result.needs_reconsideration is False
    kwargs = generate.call_args.kwargs
    assert (
        kwargs["system_prompt"]
        == REGULATORY_EVIDENCE_MATRIX_CLOSURE_REVIEW_SYSTEM_PROMPT
    )
    payload = json.loads(kwargs["user_prompt"])
    assert payload["coverage_contract"] is None
    assert payload["earlier_user_context"] == []
    assert payload["evidence_matrix"]["text"] == "One supported procedure row."


def test_review_preserves_structured_recovery_identity_and_query() -> None:
    draft = _CandidateAnswerReviewDraft(
        needs_reconsideration=True,
        advisory_claim_issues=[
            _CandidateAnswerReviewDraftClaimIssue(
                claim_kind=ClaimKind.LEGAL_RULE,
                claim_reference="Analyze the authorization consequence.",
                advisory_feedback=(
                    "The express deliverable is wholly unanswered and no exact "
                    "evidence supports it."
                ),
                related_citation_numbers=[],
                recovery_query="authorization consequence controlling rule",
            )
        ],
    )
    with patch(
        "onyx.regulatory.candidate_answer_review.generate_structured",
        return_value=draft,
    ):
        result = review_regulatory_candidate_answer(
            MagicMock(),
            user_request="Analyze the authorization consequence.",
            candidate_answer="The draft discusses only a different issue.",
            evidence_chunks=[],
        )

    assert result.advisory_claim_issues == [
        CandidateAnswerClaimIssue(
            claim_kind=ClaimKind.LEGAL_RULE,
            claim_span=CandidateAnswerClaimSpan(
                start=0,
                end=38,
                source=ClaimSpanSource.USER_REQUEST,
            ),
            claim_reference="Analyze the authorization consequence.",
            advisory_feedback=(
                "The express deliverable is wholly unanswered and no exact evidence "
                "supports it."
            ),
            recovery_query="authorization consequence controlling rule",
        )
    ]


def test_review_bounds_all_payload_sections_and_retains_each_selected_chunk() -> None:
    evidence_chunks = [
        _evidence(citation_number=index + 1, content="x" * 12_000)
        for index in range(40)
    ]
    with patch(
        "onyx.regulatory.candidate_answer_review.generate_structured",
        return_value=_CandidateAnswerReviewDraft(
            needs_reconsideration=False,
            advisory_claim_issues=[],
        ),
    ) as generate:
        review_regulatory_candidate_answer(
            MagicMock(),
            user_request="u" * 40_000,
            candidate_answer="a" * 50_000,
            evidence_chunks=evidence_chunks,
        )

    payload = json.loads(generate.call_args.kwargs["user_prompt"])
    assert len(payload["user_request"]["text"]) == 24_000
    assert payload["user_request"]["truncated"] is True
    assert len(payload["candidate_answer"]["text"]) == 36_000
    assert payload["candidate_answer"]["truncated"] is True
    assert len(payload["evidence_chunks"]) == 40
    assert all(chunk["content"] for chunk in payload["evidence_chunks"])
    assert all(chunk["content_truncated"] for chunk in payload["evidence_chunks"])
    assert sum(len(chunk["content"]) for chunk in payload["evidence_chunks"]) <= 200_000
    inventory = payload["retrieval_inventory"]
    assert inventory["total_result_count"] == 40
    assert inventory["represented_by_exact_evidence_count"] == 40
    assert inventory["inventory_only_result_count"] == 0
    assert inventory["included_result_count"] == 0
    assert inventory["truncated"] is False
    assert inventory["items"] == []


def test_review_fits_small_context_by_dropping_lower_priority_sections() -> None:
    current_request = "CURRENT REQUEST: " + ("u" * 6_000)
    candidate_answer = "CANDIDATE ANSWER: " + ("a" * 6_000)
    cited_evidence = build_candidate_answer_evidence_chunk(
        document_id="document-1",
        chunk_id=1,
        citation_number=1,
        retrieval_number=1,
        chunk_identifier="cited-chunk",
        heading="Instrument > Provision 1",
        content="CITED EVIDENCE: " + ("c" * 1_900),
    )
    uncited_evidence = [
        build_candidate_answer_evidence_chunk(
            document_id=f"document-{index + 2}",
            chunk_id=index + 2,
            citation_number=None,
            retrieval_number=index + 2,
            chunk_identifier=f"uncited-{index}",
            heading=f"Instrument > Provision {index + 2}",
            content="UNCITED EVIDENCE: " + ("x" * 2_300),
        )
        for index in range(20)
    ]
    max_input_tokens = 60_000
    with (
        patch(
            "onyx.regulatory.candidate_answer_review.get_llm_token_counter",
            return_value=len,
        ),
        patch(
            "onyx.regulatory.candidate_answer_review._selected_review_max_input_tokens",
            return_value=max_input_tokens,
        ),
        patch(
            "onyx.regulatory.candidate_answer_review.generate_structured",
            return_value=_CandidateAnswerReviewDraft(
                needs_reconsideration=False,
                advisory_claim_issues=[],
            ),
        ) as generate,
    ):
        result = review_regulatory_candidate_answer(
            MagicMock(),
            user_request=current_request,
            earlier_user_context=[
                f"EARLIER {index}: " + ("e" * 6_000) for index in range(4)
            ],
            candidate_answer=candidate_answer,
            evidence_chunks=[cited_evidence, *uncited_evidence],
        )

    assert result.completed is True
    user_prompt = generate.call_args.kwargs["user_prompt"]
    schema_chars = len(
        json.dumps(_CandidateAnswerReviewDraft.model_json_schema(), ensure_ascii=False)
    )
    payload_budget = (
        max_input_tokens
        - len(REGULATORY_CANDIDATE_ANSWER_REVIEW_SYSTEM_PROMPT)
        - schema_chars
        - 16_000
        - 2_048
    )
    assert len(user_prompt) <= payload_budget
    payload = json.loads(user_prompt)
    assert payload["user_request"]["text"] == current_request
    assert payload["candidate_answer"]["text"] == candidate_answer
    inventory = payload["retrieval_inventory"]
    assert inventory["represented_by_exact_evidence_count"] == len(
        payload["evidence_chunks"]
    )
    assert inventory["represented_by_exact_evidence_count"] >= 1
    assert (
        inventory["represented_by_exact_evidence_count"]
        + inventory["inventory_only_result_count"]
        == 21
    )
    assert 0 < inventory["included_result_count"] <= 20
    assert inventory["truncated"] is (
        inventory["included_result_count"] < inventory["inventory_only_result_count"]
    )
    assert len(payload["earlier_user_context"]) < 4 or any(
        item["truncated"] for item in payload["earlier_user_context"]
    )
    assert payload["evidence_chunks"][0]["citation_number"] == 1
    assert payload["evidence_chunks"][0]["content"] == cited_evidence.content
    exact_chunk_ids = {
        chunk["chunk_identifier"] for chunk in payload["evidence_chunks"]
    }
    inventory_chunk_ids = {item["chunk_identifier"] for item in inventory["items"]}
    assert exact_chunk_ids.isdisjoint(inventory_chunk_ids)


def test_review_is_explicitly_unavailable_when_minimum_payload_cannot_fit() -> None:
    with (
        patch(
            "onyx.regulatory.candidate_answer_review.get_llm_token_counter",
            return_value=len,
        ),
        patch(
            "onyx.regulatory.candidate_answer_review._selected_review_max_input_tokens",
            return_value=1_000,
        ),
        patch(
            "onyx.regulatory.candidate_answer_review.generate_structured"
        ) as generate,
    ):
        result = review_regulatory_candidate_answer(
            MagicMock(),
            user_request="Analyze the issue.",
            candidate_answer="Candidate answer.",
            evidence_chunks=[_evidence()],
        )

    assert result.completed is False
    assert result.review_error is CandidateAnswerReviewError.REVIEW_UNAVAILABLE
    generate.assert_not_called()


def test_review_bounds_inventory_while_covering_all_normally_reachable_results() -> (
    None
):
    evidence_chunks = [
        build_candidate_answer_evidence_chunk(
            document_id=f"document-{index + 1}",
            chunk_id=index + 1,
            citation_number=None,
            retrieval_number=index + 1,
            chunk_identifier=f"chunk-{index}-" + ("i" * 200),
            heading=f"Instrument {index} > " + ("h" * 480),
            content="Operative text.",
        )
        for index in range(400)
    ]
    with patch(
        "onyx.regulatory.candidate_answer_review.generate_structured",
        return_value=_CandidateAnswerReviewDraft(
            needs_reconsideration=False,
            advisory_claim_issues=[],
        ),
    ) as generate:
        review_regulatory_candidate_answer(
            MagicMock(),
            user_request="Analyze.",
            candidate_answer="Candidate answer.",
            evidence_chunks=evidence_chunks,
        )

    inventory = json.loads(generate.call_args.kwargs["user_prompt"])[
        "retrieval_inventory"
    ]
    assert inventory["total_result_count"] == 400
    assert inventory["represented_by_exact_evidence_count"] == 256
    assert inventory["inventory_only_result_count"] == 144
    assert inventory["included_result_count"] == 144
    assert inventory["truncated"] is False
    evidence_retrieval_numbers = {
        chunk["retrieval_number"]
        for chunk in json.loads(generate.call_args.kwargs["user_prompt"])[
            "evidence_chunks"
        ]
    }
    inventory_retrieval_numbers = {
        item["retrieval_number"] for item in inventory["items"]
    }
    assert evidence_retrieval_numbers.isdisjoint(inventory_retrieval_numbers)
    assert (
        sum(
            len(item["chunk_identifier"]) + len(item["heading"])
            for item in inventory["items"]
        )
        <= 80_000
    )


def test_review_prioritizes_cited_chunks_when_evidence_limit_is_reached() -> None:
    evidence_chunks = [
        build_candidate_answer_evidence_chunk(
            document_id=f"document-{index + 1}",
            chunk_id=index + 1,
            citation_number=None,
            retrieval_number=index + 1,
            chunk_identifier=f"uncited-{index}",
            heading=f"Instrument > Provision {index}",
            content=f"Uncited {index}",
        )
        for index in range(48)
    ] + [
        build_candidate_answer_evidence_chunk(
            document_id="document-7",
            chunk_id=49,
            citation_number=7,
            retrieval_number=49,
            chunk_identifier="cited-7",
            heading="Instrument > Provision 7",
            content="Cited evidence",
        )
    ]
    with patch(
        "onyx.regulatory.candidate_answer_review.generate_structured",
        return_value=_CandidateAnswerReviewDraft(
            needs_reconsideration=False,
            advisory_claim_issues=[],
        ),
    ) as generate:
        review_regulatory_candidate_answer(
            MagicMock(),
            user_request="Analyze.",
            candidate_answer="Candidate answer [7].",
            evidence_chunks=evidence_chunks,
        )

    payload = json.loads(generate.call_args.kwargs["user_prompt"])
    assert payload["evidence_chunks"][0]["citation_number"] == 7
    assert len(payload["evidence_chunks"]) == 49
    selected_uncited = payload["evidence_chunks"][1:]
    assert selected_uncited[0]["content"] == "Uncited 0"
    assert selected_uncited[-1]["content"] == "Uncited 47"
    assert payload["retrieval_inventory"]["represented_by_exact_evidence_count"] == 49
    assert payload["retrieval_inventory"]["inventory_only_result_count"] == 0
    assert payload["retrieval_inventory"]["included_result_count"] == 0
    assert payload["retrieval_inventory"]["truncated"] is False


def test_review_reserves_uncited_paragraphs_from_cited_provision() -> None:
    cited = build_candidate_answer_evidence_chunk(
        document_id="instrument",
        chunk_id=1,
        citation_number=1,
        retrieval_number=1,
        chunk_identifier="cited",
        heading="Instrument > MADDE 6",
        content="Cited paragraph.",
    )
    unrelated = [
        build_candidate_answer_evidence_chunk(
            document_id="other-instrument",
            chunk_id=index + 2,
            citation_number=None,
            retrieval_number=index + 2,
            chunk_identifier=f"unrelated-{index}",
            heading=f"Other Instrument > MADDE {index + 100}",
            content=f"Unrelated {index}",
        )
        for index in range(100)
    ]
    siblings = [
        build_candidate_answer_evidence_chunk(
            document_id="instrument",
            chunk_id=200 + index,
            citation_number=None,
            retrieval_number=200 + index,
            chunk_identifier=f"sibling-{index}",
            heading="Instrument > MADDE 6",
            content=f"Sibling paragraph {index}",
        )
        for index in range(3)
    ]

    compacted = _compact_evidence_chunks([cited, *unrelated, *siblings])

    assert compacted[0].chunk_identifier == "cited"
    assert [chunk.chunk_identifier for chunk in compacted[1:4]] == [
        "sibling-0",
        "sibling-1",
        "sibling-2",
    ]
    assert len(compacted) == 104


def test_review_reserves_and_samples_uncited_evidence_across_retrieval_history() -> (
    None
):
    cited_chunks = [
        _evidence(citation_number=index + 1, content=f"Cited {index}")
        for index in range(20)
    ]
    uncited_chunks = [
        build_candidate_answer_evidence_chunk(
            document_id=f"document-{index}",
            chunk_id=index,
            citation_number=None,
            retrieval_number=index,
            chunk_identifier=f"uncited-{index}",
            heading=f"Instrument > Provision {index}",
            content=f"Uncited {index}",
        )
        for index in range(101, 181)
    ]
    with patch(
        "onyx.regulatory.candidate_answer_review.generate_structured",
        return_value=_CandidateAnswerReviewDraft(
            needs_reconsideration=False,
            advisory_claim_issues=[],
        ),
    ) as generate:
        review_regulatory_candidate_answer(
            MagicMock(),
            user_request="Analyze comprehensively.",
            candidate_answer="Candidate answer "
            + " ".join(f"[{index}]" for index in range(1, 21)),
            evidence_chunks=cited_chunks + uncited_chunks,
        )

    payload = json.loads(generate.call_args.kwargs["user_prompt"])
    selected = payload["evidence_chunks"]
    assert len(selected) == 100
    assert sum(chunk["citation_number"] is not None for chunk in selected) == 20
    selected_uncited = [chunk for chunk in selected if chunk["citation_number"] is None]
    assert len(selected_uncited) == 80
    assert selected_uncited[0]["retrieval_number"] == 101
    assert selected_uncited[-1]["retrieval_number"] == 180
    assert sum(len(chunk["content"]) for chunk in selected) <= 200_000


def test_review_preserves_cited_evidence_before_using_remaining_uncited_slots() -> None:
    cited_chunks = [
        _evidence(citation_number=index + 1, content=f"Cited {index}")
        for index in range(45)
    ]
    uncited_chunks = [
        build_candidate_answer_evidence_chunk(
            document_id=f"document-{index}",
            chunk_id=index,
            citation_number=None,
            retrieval_number=index,
            chunk_identifier=f"uncited-{index}",
            heading=f"Instrument > Provision {index}",
            content=f"Uncited {index}",
        )
        for index in range(100, 120)
    ]
    with patch(
        "onyx.regulatory.candidate_answer_review.generate_structured",
        return_value=_CandidateAnswerReviewDraft(
            needs_reconsideration=False,
            advisory_claim_issues=[],
        ),
    ) as generate:
        review_regulatory_candidate_answer(
            MagicMock(),
            user_request="Analyze comprehensively.",
            candidate_answer="Candidate answer "
            + " ".join(f"[{index}]" for index in range(1, 46)),
            evidence_chunks=cited_chunks + uncited_chunks,
        )

    selected = json.loads(generate.call_args.kwargs["user_prompt"])["evidence_chunks"]
    assert len(selected) == 65
    assert [chunk["citation_number"] for chunk in selected[:45]] == list(range(1, 46))
    selected_uncited = selected[45:]
    assert len(selected_uncited) == 20
    assert selected_uncited[0]["retrieval_number"] == 100
    assert selected_uncited[-1]["retrieval_number"] == 119


def test_review_samples_uncited_chunks_across_retrieval_order() -> None:
    cited_chunks = [
        build_candidate_answer_evidence_chunk(
            document_id=f"cited-document-{index}",
            chunk_id=index,
            citation_number=index + 1,
            retrieval_number=index + 1,
            chunk_identifier=f"cited-{index}",
            heading=f"Instrument > Article {index}",
            content=f"Cited rule {index}.",
        )
        for index in range(254)
    ]
    ordinary_first = build_candidate_answer_evidence_chunk(
        document_id="ordinary-first",
        chunk_id=100,
        citation_number=None,
        retrieval_number=100,
        chunk_identifier="ordinary-first",
        heading="Other Instrument > Background",
        content="General background text.",
    )
    middle = build_candidate_answer_evidence_chunk(
        document_id="middle",
        chunk_id=101,
        citation_number=None,
        retrieval_number=101,
        chunk_identifier="middle",
        heading="Other Instrument > Eligibility",
        content="Middle excerpt.",
    )
    ordinary_last = build_candidate_answer_evidence_chunk(
        document_id="ordinary-last",
        chunk_id=102,
        citation_number=None,
        retrieval_number=102,
        chunk_identifier="ordinary-last",
        heading="Other Instrument > Notes",
        content="Additional general background.",
    )

    compacted = _compact_evidence_chunks(
        [*cited_chunks, ordinary_first, middle, ordinary_last]
    )

    assert len(compacted) == 256
    assert [chunk.chunk_identifier for chunk in compacted[-2:]] == [
        "ordinary-first",
        "ordinary-last",
    ]


def test_review_reserves_most_relevant_chunk_per_focused_target() -> None:
    cited_chunks = [
        build_candidate_answer_evidence_chunk(
            document_id=f"cited-document-{index}",
            chunk_id=index,
            citation_number=index + 1,
            retrieval_number=index + 1,
            chunk_identifier=f"cited-{index}",
            heading=f"Instrument > Article {index}",
            content=f"Cited rule {index}.",
        )
        for index in range(62)
    ]
    first_target_unrelated = build_candidate_answer_evidence_chunk(
        document_id="target-one-unrelated",
        chunk_id=100,
        citation_number=None,
        retrieval_number=100,
        chunk_identifier="target-one-unrelated",
        heading="Instrument > Background",
        research_target=(
            "Specific evidence target: alpha protocol. Full query context."
        ),
        content="General background.",
    )
    first_target_relevant = build_candidate_answer_evidence_chunk(
        document_id="target-one-relevant",
        chunk_id=101,
        citation_number=None,
        retrieval_number=101,
        chunk_identifier="target-one-relevant",
        heading="Instrument > Alpha protocol",
        research_target=(
            "Specific evidence target: alpha protocol. Full query context."
        ),
        content="The alpha protocol is described here.",
    )
    first_target_partly_relevant = build_candidate_answer_evidence_chunk(
        document_id="target-one-partly-relevant",
        chunk_id=103,
        citation_number=None,
        retrieval_number=103,
        chunk_identifier="target-one-partly-relevant",
        heading="Instrument > Alpha notes",
        research_target=(
            "Specific evidence target: alpha protocol. Full query context."
        ),
        content="Alpha background.",
    )
    second_target_relevant = build_candidate_answer_evidence_chunk(
        document_id="target-two-relevant",
        chunk_id=102,
        citation_number=None,
        retrieval_number=102,
        chunk_identifier="target-two-relevant",
        heading="Instrument > Beta sequence",
        research_target=(
            "Specific evidence target: beta sequence. Full query context."
        ),
        content="The beta sequence is described here.",
    )

    compacted = _compact_evidence_chunks(
        [
            *cited_chunks,
            first_target_unrelated,
            first_target_partly_relevant,
            first_target_relevant,
            second_target_relevant,
        ]
    )

    assert len(compacted) == 66
    assert {
        "target-one-relevant",
        "target-two-relevant",
    }.issubset({chunk.chunk_identifier for chunk in compacted})


def test_review_returns_explicit_fail_open_result_on_transport_or_parse_error() -> None:
    with patch(
        "onyx.regulatory.candidate_answer_review.generate_structured",
        side_effect=RuntimeError("provider unavailable"),
    ):
        result = review_regulatory_candidate_answer(
            MagicMock(),
            user_request="Analyze.",
            candidate_answer="Candidate answer.",
            evidence_chunks=[_evidence()],
        )

    assert result.completed is False
    assert result.review_error is CandidateAnswerReviewError.REVIEW_UNAVAILABLE
    assert result.needs_reconsideration is False
    assert result.advisory_claim_issues == []


def test_review_without_actionable_issue_is_not_rejected() -> None:
    inconsistent_draft = _CandidateAnswerReviewDraft(
        needs_reconsideration=True, advisory_claim_issues=[]
    )
    with patch(
        "onyx.regulatory.candidate_answer_review.generate_structured",
        return_value=inconsistent_draft,
    ):
        result = review_regulatory_candidate_answer(
            MagicMock(),
            user_request="Analyze.",
            candidate_answer="Candidate answer.",
            evidence_chunks=[],
        )

    assert result.review_error is None
    assert result.needs_reconsideration is False


def test_review_normalizes_provider_length_and_count_drift() -> None:
    draft = _CandidateAnswerReviewDraft(
        needs_reconsideration=True,
        advisory_claim_issues=[
            _CandidateAnswerReviewDraftClaimIssue(
                claim_kind=ClaimKind.MATERIAL_FACT,
                claim_reference=f"Claim {index} " + ("x" * 400),
                advisory_feedback=f"Issue {index} " + ("y" * 700),
                related_citation_numbers=[99, 1, 1, 2, 98, 3, 4, 5, 6],
            )
            for index in range(8)
        ],
    )
    with patch(
        "onyx.regulatory.candidate_answer_review.generate_structured",
        return_value=draft,
    ):
        result = review_regulatory_candidate_answer(
            MagicMock(),
            user_request="Analyze.",
            candidate_answer="Candidate answer.",
            evidence_chunks=[
                _evidence(citation_number=index) for index in range(1, 11)
            ],
        )

    assert result.review_error is None
    assert result.needs_reconsideration is True
    assert len(result.advisory_claim_issues) == 8
    assert all(
        len(issue.claim_reference) <= 280 for issue in result.advisory_claim_issues
    )
    assert all(
        len(issue.advisory_feedback) <= 520 for issue in result.advisory_claim_issues
    )
    assert all(
        issue.related_citation_numbers == list(range(1, 6))
        for issue in result.advisory_claim_issues
    )


def test_review_retains_sixteen_material_issues_for_complex_requests() -> None:
    draft = _CandidateAnswerReviewDraft(
        needs_reconsideration=True,
        advisory_claim_issues=[
            _CandidateAnswerReviewDraftClaimIssue(
                claim_kind=ClaimKind.MATERIAL_FACT,
                claim_reference=f"Claim {index}",
                advisory_feedback=f"Issue {index}",
            )
            for index in range(16)
        ],
    )

    assert len(draft.advisory_claim_issues) == 16


def test_review_validates_nonblank_request_and_candidate_before_call() -> None:
    with patch(
        "onyx.regulatory.candidate_answer_review.generate_structured"
    ) as generate:
        with pytest.raises(ValueError, match="text must not be blank"):
            review_regulatory_candidate_answer(
                MagicMock(),
                user_request="Analyze.",
                candidate_answer="  ",
                evidence_chunks=[],
            )

    generate.assert_not_called()


def test_resolution_review_checks_each_prior_issue_with_cited_evidence_only() -> None:
    prior_issues = [
        CandidateAnswerClaimIssue(
            claim_reference="Unsupported category mapping",
            advisory_feedback="The cited rule names a category but does not classify the facts.",
            related_citation_numbers=[1],
        ),
        CandidateAnswerClaimIssue(
            claim_reference="Unsupported procedural sequence",
            advisory_feedback="The evidence does not establish a mandatory sequence.",
            related_citation_numbers=[1],
        ),
    ]
    draft = _CandidateAnswerResolutionReviewDraft(
        issue_resolutions=[
            _CandidateAnswerIssueResolutionDraft(
                issue_index=0,
                status=(
                    CandidateAnswerIssueResolutionStatus.CLAIM_REMOVED_OR_QUALIFIED
                ),
                advisory_feedback="The revised answer now states the classification gap.",
            ),
            _CandidateAnswerIssueResolutionDraft(
                issue_index=1,
                status=CandidateAnswerIssueResolutionStatus.STILL_UNRESOLVED,
                advisory_feedback="The same mandatory sequence is repeated.",
            ),
        ]
    )
    with patch(
        "onyx.regulatory.candidate_answer_review.generate_structured",
        return_value=draft,
    ) as generate:
        result = review_regulatory_candidate_resolution(
            MagicMock(),
            candidate_answer="Revised answer [1].",
            prior_issues=prior_issues,
            evidence_chunks=[
                _evidence(citation_number=1, content="Exact cited text"),
                _evidence(citation_number=None, content="Uncited retrieved text"),
            ],
        )

    assert result.completed is True
    assert result.needs_reconsideration is True
    assert result.advisory_claim_issues == [
        CandidateAnswerClaimIssue(
            claim_reference="Unsupported procedural sequence",
            advisory_feedback="The same mandatory sequence is repeated.",
            related_citation_numbers=[1],
        )
    ]
    kwargs = generate.call_args.kwargs
    assert kwargs["flow"] is LLMFlow.REGULATORY_ANSWER_AUDIT
    assert kwargs["max_tokens"] == 16_000
    assert kwargs["reasoning_effort"] is ReasoningEffort.HIGH
    assert kwargs["max_attempts"] == 2
    payload = json.loads(kwargs["user_prompt"])
    assert [issue["issue_index"] for issue in payload["prior_issues"]] == [0, 1]
    assert [issue["related_citation_numbers"] for issue in payload["prior_issues"]] == [
        [1],
        [1],
    ]
    assert payload["revised_candidate_answer"] == {
        "text": "Revised answer [1].",
        "truncated": False,
    }
    assert len(payload["evidence_chunks"]) == 1
    assert payload["evidence_chunks"][0]["content"] == "Exact cited text"


def test_resolution_review_fits_small_context_without_dropping_required_core() -> None:
    prior_issues = [
        CandidateAnswerClaimIssue(
            claim_reference="First disputed claim",
            advisory_feedback="Verify the first claim against its exact source.",
            related_citation_numbers=[1],
        ),
        CandidateAnswerClaimIssue(
            claim_reference="Second disputed claim",
            advisory_feedback="Verify the second claim against its exact source.",
            related_citation_numbers=[2],
        ),
    ]
    candidate_answer = "REVISED CANDIDATE: " + ("a" * 3_500)
    issue_evidence = [
        build_candidate_answer_evidence_chunk(
            document_id=f"document-{index}",
            chunk_id=index,
            citation_number=index,
            retrieval_number=index,
            chunk_identifier=f"issue-{index}",
            heading=f"Instrument > Provision {index}",
            content=f"ISSUE EVIDENCE {index}: " + ("e" * 1_450),
        )
        for index in (1, 2)
    ]
    supplemental_evidence = [
        build_candidate_answer_evidence_chunk(
            document_id=f"document-{index}",
            chunk_id=index,
            citation_number=index,
            retrieval_number=index,
            chunk_identifier=f"supplemental-{index}",
            heading=f"Instrument > Provision {index}",
            content=f"SUPPLEMENTAL {index}: " + ("s" * 1_450),
        )
        for index in range(3, 13)
    ]
    max_input_tokens = 26_000
    draft = _CandidateAnswerResolutionReviewDraft(
        issue_resolutions=[
            _CandidateAnswerIssueResolutionDraft(
                issue_index=0,
                status=(
                    CandidateAnswerIssueResolutionStatus.CLAIM_REMOVED_OR_QUALIFIED
                ),
            ),
            _CandidateAnswerIssueResolutionDraft(
                issue_index=1,
                status=(
                    CandidateAnswerIssueResolutionStatus.RESOLVED_BY_EXACT_EVIDENCE
                ),
            ),
        ]
    )
    with (
        patch(
            "onyx.regulatory.candidate_answer_review.get_llm_token_counter",
            return_value=len,
        ),
        patch(
            "onyx.regulatory.candidate_answer_review._selected_review_max_input_tokens",
            return_value=max_input_tokens,
        ),
        patch(
            "onyx.regulatory.candidate_answer_review.generate_structured",
            return_value=draft,
        ) as generate,
    ):
        result = review_regulatory_candidate_resolution(
            MagicMock(),
            candidate_answer=candidate_answer,
            prior_issues=prior_issues,
            evidence_chunks=[*issue_evidence, *supplemental_evidence],
        )

    assert result.completed is True
    user_prompt = generate.call_args.kwargs["user_prompt"]
    schema_chars = len(
        json.dumps(
            _CandidateAnswerResolutionReviewDraft.model_json_schema(),
            ensure_ascii=False,
        )
    )
    payload_budget = (
        max_input_tokens
        - len(REGULATORY_CANDIDATE_RESOLUTION_REVIEW_SYSTEM_PROMPT)
        - schema_chars
        - 13_000
        - 2_048
    )
    assert len(user_prompt) <= payload_budget
    assert generate.call_args.kwargs["max_tokens"] == 13_000
    payload = json.loads(user_prompt)
    assert payload["revised_candidate_answer"] == {
        "text": candidate_answer,
        "truncated": False,
    }
    assert [issue["issue_index"] for issue in payload["prior_issues"]] == [0, 1]
    selected_citations = {
        chunk["citation_number"] for chunk in payload["evidence_chunks"]
    }
    assert {1, 2} <= selected_citations
    assert len(selected_citations) < len(issue_evidence + supplemental_evidence)


def test_resolution_review_is_unavailable_when_required_core_cannot_fit() -> None:
    prior_issue = CandidateAnswerClaimIssue(
        claim_reference="Disputed claim",
        advisory_feedback="Verify whether the revision resolved this issue.",
        related_citation_numbers=[1],
    )
    with (
        patch(
            "onyx.regulatory.candidate_answer_review.get_llm_token_counter",
            return_value=len,
        ),
        patch(
            "onyx.regulatory.candidate_answer_review._selected_review_max_input_tokens",
            return_value=9_000,
        ),
        patch(
            "onyx.regulatory.candidate_answer_review.generate_structured"
        ) as generate,
    ):
        result = review_regulatory_candidate_resolution(
            MagicMock(),
            candidate_answer="REQUIRED CANDIDATE: " + ("a" * 2_000),
            prior_issues=[prior_issue],
            evidence_chunks=[_evidence()],
        )

    assert result.completed is False
    assert result.review_error is CandidateAnswerReviewError.REVIEW_UNAVAILABLE
    generate.assert_not_called()


def test_resolution_review_prioritizes_issue_related_citations_within_bound() -> None:
    prior_issues = [
        CandidateAnswerClaimIssue(
            claim_reference="Unsupported application",
            advisory_feedback="The answer applies two provisions without support.",
            related_citation_numbers=[40, 39],
        )
    ]
    evidence_chunks = [
        _evidence(citation_number=index, content=f"Evidence {index}")
        for index in range(1, 41)
    ]
    draft = _CandidateAnswerResolutionReviewDraft(
        issue_resolutions=[
            _CandidateAnswerIssueResolutionDraft(
                issue_index=0,
                status=(
                    CandidateAnswerIssueResolutionStatus.RESOLVED_BY_EXACT_EVIDENCE
                ),
            )
        ]
    )
    with patch(
        "onyx.regulatory.candidate_answer_review.generate_structured",
        return_value=draft,
    ) as generate:
        result = review_regulatory_candidate_resolution(
            MagicMock(),
            candidate_answer="Revised answer [39] [40].",
            prior_issues=prior_issues,
            evidence_chunks=evidence_chunks,
        )

    assert result.needs_reconsideration is False
    payload = json.loads(generate.call_args.kwargs["user_prompt"])
    assert len(payload["evidence_chunks"]) == 40
    assert [chunk["citation_number"] for chunk in payload["evidence_chunks"][:2]] == [
        40,
        39,
    ]
    assert payload["evidence_chunks"][-1]["citation_number"] == 38
    assert sum(len(chunk["content"]) for chunk in payload["evidence_chunks"]) <= 120_000


def test_resolution_review_reports_one_serious_existing_evidence_regression() -> None:
    prior_issue = CandidateAnswerClaimIssue(
        claim_reference="Earlier unsupported proposition",
        advisory_feedback="The cited text did not entail it.",
        related_citation_numbers=[1],
    )
    draft = _CandidateAnswerResolutionReviewDraft(
        issue_resolutions=[
            _CandidateAnswerIssueResolutionDraft(
                issue_index=0,
                status=(
                    CandidateAnswerIssueResolutionStatus.CLAIM_REMOVED_OR_QUALIFIED
                ),
            )
        ],
        new_grounding_regression=_CandidateAnswerReviewDraftClaimIssue(
            claim_kind=ClaimKind.MATERIAL_FACT,
            claim_reference="New unsupported proposition",
            advisory_feedback=(
                "Existing evidence contradicts it; remove or accurately qualify it."
            ),
            related_citation_numbers=[2, 2, 1],
        ),
    )
    with patch(
        "onyx.regulatory.candidate_answer_review.generate_structured",
        return_value=draft,
    ):
        result = review_regulatory_candidate_resolution(
            MagicMock(),
            candidate_answer="Revised answer [1] [2].",
            prior_issues=[prior_issue],
            evidence_chunks=[
                _evidence(citation_number=1),
                _evidence(citation_number=2),
            ],
        )

    assert result.completed is True
    assert result.needs_reconsideration is True
    assert result.advisory_claim_issues == [
        CandidateAnswerClaimIssue(
            claim_reference="New unsupported proposition",
            advisory_feedback=(
                "Existing evidence contradicts it; remove or accurately qualify it."
            ),
            related_citation_numbers=[2, 1],
        )
    ]


def test_resolution_review_preserves_single_regression_with_six_unresolved_issues() -> (
    None
):
    prior_issues = [
        CandidateAnswerClaimIssue(
            claim_reference=f"Prior issue {index}",
            advisory_feedback=f"Prior feedback {index}.",
            related_citation_numbers=[1],
        )
        for index in range(6)
    ]
    draft = _CandidateAnswerResolutionReviewDraft(
        issue_resolutions=[
            _CandidateAnswerIssueResolutionDraft(
                issue_index=index,
                status=CandidateAnswerIssueResolutionStatus.STILL_UNRESOLVED,
            )
            for index in range(6)
        ],
        new_grounding_regression=_CandidateAnswerReviewDraftClaimIssue(
            claim_kind=ClaimKind.MATERIAL_FACT,
            claim_reference="Single serious regression",
            advisory_feedback="Remove or qualify it using the supplied evidence.",
            related_citation_numbers=[1],
        ),
    )
    with patch(
        "onyx.regulatory.candidate_answer_review.generate_structured",
        return_value=draft,
    ):
        result = review_regulatory_candidate_resolution(
            MagicMock(),
            candidate_answer="Revised answer [1].",
            prior_issues=prior_issues,
            evidence_chunks=[_evidence()],
        )

    assert len(result.advisory_claim_issues) == 7
    assert [issue.claim_reference for issue in result.advisory_claim_issues[:5]] == [
        f"Prior issue {index}" for index in range(5)
    ]
    assert result.advisory_claim_issues[-1].claim_reference == (
        "Single serious regression"
    )


def test_resolution_review_fails_open_when_not_every_issue_is_assessed() -> None:
    prior_issues = [
        CandidateAnswerClaimIssue(
            claim_reference="First issue",
            advisory_feedback="First problem.",
        ),
        CandidateAnswerClaimIssue(
            claim_reference="Second issue",
            advisory_feedback="Second problem.",
        ),
    ]
    incomplete_draft = _CandidateAnswerResolutionReviewDraft(
        issue_resolutions=[
            _CandidateAnswerIssueResolutionDraft(
                issue_index=0,
                status=(
                    CandidateAnswerIssueResolutionStatus.RESOLVED_BY_EXACT_EVIDENCE
                ),
            )
        ]
    )
    with patch(
        "onyx.regulatory.candidate_answer_review.generate_structured",
        return_value=incomplete_draft,
    ):
        result = review_regulatory_candidate_resolution(
            MagicMock(),
            candidate_answer="Revised answer.",
            prior_issues=prior_issues,
            evidence_chunks=[],
        )

    assert result.completed is False
    assert result.review_error is CandidateAnswerReviewError.REVIEW_UNAVAILABLE
    assert result.needs_reconsideration is False


def test_resolution_review_preserves_fail_open_on_provider_error() -> None:
    with patch(
        "onyx.regulatory.candidate_answer_review.generate_structured",
        side_effect=RuntimeError("provider unavailable"),
    ):
        result = review_regulatory_candidate_resolution(
            MagicMock(),
            candidate_answer="Revised answer.",
            prior_issues=[
                CandidateAnswerClaimIssue(
                    claim_reference="Prior issue",
                    advisory_feedback="Prior feedback.",
                )
            ],
            evidence_chunks=[],
        )

    assert result.completed is False
    assert result.review_error is CandidateAnswerReviewError.REVIEW_UNAVAILABLE
    assert result.needs_reconsideration is False


def test_resolution_review_requires_prior_issues_without_calling_provider() -> None:
    with patch(
        "onyx.regulatory.candidate_answer_review.generate_structured"
    ) as generate:
        with pytest.raises(ValueError, match="prior issues must not be empty"):
            review_regulatory_candidate_resolution(
                MagicMock(),
                candidate_answer="Revised answer.",
                prior_issues=[],
                evidence_chunks=[],
            )

    generate.assert_not_called()


def test_review_schema_and_prompt_bound_one_support_recovery_query() -> None:
    schema = _CandidateAnswerReviewDraft.model_json_schema()
    serialized_schema = json.dumps(schema)
    prompt = REGULATORY_CANDIDATE_ANSWER_REVIEW_SYSTEM_PROMPT
    resolution_prompt = REGULATORY_CANDIDATE_RESOLUTION_REVIEW_SYSTEM_PROMPT
    normalized_prompt = " ".join(prompt.split())

    assert "recovery_query" in serialized_schema
    assert "search_mode" not in serialized_schema.casefold()
    for scenario_term in ("2207", "UND", "DAC", "Basel", "İtalya", "Azerbaycan"):
        assert scenario_term not in prompt
        assert scenario_term not in resolution_prompt

    assert "current user request" in normalized_prompt
    assert "not an independent instruction" in normalized_prompt
    assert "exact chunk" in normalized_prompt
    assert "Do not introduce a predicted source" in normalized_prompt
    assert "at most sixteen" in normalized_prompt

    normalized_resolution_prompt = " ".join(resolution_prompt.split())
    assert "Assess every prior issue exactly once" in normalized_resolution_prompt
    assert "resolved_by_exact_evidence" in normalized_resolution_prompt
    assert "claim_removed_or_qualified" in normalized_resolution_prompt
    assert "still_unresolved" in normalized_resolution_prompt
    assert "at most one new_grounding_regression" in normalized_resolution_prompt
    return

    assert "not a predetermined checklist" in normalized_prompt
    assert "Request coverage and evidentiary support are independent" in (
        normalized_prompt
    )
    assert "related but different object does not answer" in normalized_prompt
    assert "slash, conjunction, punctuation, acronym, or parenthetical" in (
        normalized_prompt
    )
    assert "verified acronym/full-name pair" in normalized_prompt
    assert "no evidence for it was retrieved" in normalized_prompt
    assert "Both cited and uncited unsupported propositions can be recoverable" in (
        normalized_prompt
    )
    assert "provide one concise recovery_query" in normalized_prompt
    assert "choose a search mode, propose multiple queries" in normalized_prompt
    assert "logical direction, included or excluded category" in normalized_prompt
    assert "converse, an adjacent range or category" in normalized_prompt
    assert "inherit a negation, permission, prohibition" in normalized_prompt
    assert "omits or leaves that operator grammatically ambiguous" in normalized_prompt
    assert "neither semantic coverage nor claim-to-evidence support" in (
        normalized_prompt
    )
    assert "territorial, personal, material, procedural, and regime scope" in (
        normalized_prompt
    )
    assert "Do not supply a corrected legal conclusion" in normalized_prompt
    assert "Apart from the single bounded recovery_query field" in normalized_prompt
    assert "at most ten issues in descending order" in normalized_prompt
    assert "ten most material rather than the first ten" in normalized_prompt
    assert "fact-to-category mapping separately" in normalized_prompt
    assert "legal regime active at the relevant event" in normalized_prompt
    assert "changes legal classification during the fact sequence" in normalized_prompt
    assert "earlier differently classified movement" in normalized_prompt
    assert "separately regulated objects" in normalized_prompt
    assert "cumulative and alternative members" in normalized_prompt
    assert "failure or impossibility branch" in normalized_prompt
    assert "separately verify the governing membership logic" in normalized_prompt
    assert "inclusion, exclusion, precedence, or characteristic rule" in (
        normalized_prompt
    )
    assert "actor, status or class, recurrence or stage" in normalized_prompt
    assert "unproved special-status discriminator" in normalized_prompt
    assert "mandatory order or exhaustion condition" in normalized_prompt
    assert "discussion of one procedure does not by itself cover" in normalized_prompt
    assert "Flag such an omission only when" in normalized_prompt
    assert "do not invent a rule from background knowledge" in normalized_prompt
    assert "does not establish an automatic sanction" in normalized_prompt
    assert "support for both its trigger and its stated effect" in normalized_prompt
    assert "provision-content pairing" in normalized_prompt
    assert "heading, identifier, neighboring text" in normalized_prompt
    assert "supplied evidence materially conflicts" in normalized_prompt
    assert "reconcile the relevant version, scope, authority" in normalized_prompt
    assert "retrieval_inventory is a bounded map" in normalized_prompt
    assert "Inventory identifiers and headings are not positive legal evidence" in (
        normalized_prompt
    )
    assert "Only exact text in evidence_chunks" in normalized_prompt
    assert "binding negative scope boundary" in normalized_prompt
    assert (
        "never detach descendant text from its actor, status or class"
        in normalized_prompt
    )
    assert "Only user_request defines the deliverables" in normalized_prompt
    assert "solely to resolve references and retain user-supplied facts" in (
        normalized_prompt
    )
    assert "current statement controls" in normalized_prompt
    assert "earlier topic into a current issue" in normalized_prompt
    assert "at most five deduplicated positive related_citation_numbers" in (
        normalized_prompt
    )
    assert "never a retrieval_number" in normalized_prompt
    normalized_resolution_prompt = " ".join(resolution_prompt.split())
    assert "Assess every prior issue exactly once" in normalized_resolution_prompt
    assert "resolved_by_exact_evidence" in normalized_resolution_prompt
    assert "claim_removed_or_qualified" in normalized_resolution_prompt
    assert "still_unresolved" in normalized_resolution_prompt
    assert "at most one new_grounding_regression" in normalized_resolution_prompt
    assert "must not request more retrieval" in normalized_resolution_prompt
    assert "open another review or research loop" in normalized_resolution_prompt
    assert "Except for that single narrow regression" in normalized_resolution_prompt
    assert "classification change during an event sequence" in (
        normalized_resolution_prompt
    )
    assert "separately regulated" in normalized_resolution_prompt
    assert "post-performance duties" in normalized_resolution_prompt
    assert "provision-content pairing intact" in normalized_resolution_prompt
    assert "supports both its trigger and effect" in normalized_resolution_prompt
    assert "supplied evidence materially conflicts" in normalized_resolution_prompt
    assert "reconcile the governing version, scope, authority" in (
        normalized_resolution_prompt
    )
    assert "binding negative scope boundary" in normalized_resolution_prompt
