import hashlib
from typing import TypeVar

import pytest
from pydantic import BaseModel, JsonValue, ValidationError

from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import EvidenceItem, RunContext, RunStopped
from onyx.configs.constants import DocumentSource
from onyx.context.search.models import SearchDoc
from onyx.legal_composite.models import ResearchNeed, ResearchPlan
from onyx.legal_composite.selection import (
    GatewaySourceClassifier,
    IrrelevantSource,
    NeedSourceSelection,
    SelectionObservation,
    SourceCandidate,
    SourceSelectionDecision,
    SourceSelectionRequest,
    SourceSelector,
    selection_request_from_ledger,
)
from onyx.tracing.flows import LLMFlow

ResponseModel = TypeVar("ResponseModel", bound=BaseModel)


def plan() -> ResearchPlan:
    return ResearchPlan(
        language="en",
        requires_sources=True,
        initial_actions=[],
        missing_user_facts=[],
        needs=[
            ResearchNeed(
                need_id=need,
                question=f"What governs {need}?",
                governing_source="Applicable original law and detailed procedures",
                conditions_to_check=["Preserve the operative qualification"],
            )
            for need in ("basis", "procedure")
        ],
    )


def candidate(citation: int, text: str | None = None) -> SourceCandidate:
    text = (
        text or f"Canonical whole source {citation}; subject to the stated condition."
    )
    return SourceCandidate(
        citation=citation,
        source_id=f"source-{citation}",
        chunk_id=f"chunk-{citation}",
        text_hash=hashlib.sha256(text.encode()).hexdigest(),
        text=text,
        source_kind="regulation",
        source_type="file",
        metadata={"title": "Original"},
    )


def request() -> SourceSelectionRequest:
    return SourceSelectionRequest(
        question="Explain the rule and procedure",
        plan=plan(),
        candidates=[candidate(1), candidate(2), candidate(3)],
    )


def irrelevant(citation: int, probability: float = 0.99) -> IrrelevantSource:
    return IrrelevantSource(
        citation=citation,
        probability=probability,
        reason="This original concerns a distinct legal transaction",
    )


def observation(
    rows: list[NeedSourceSelection], delivered: list[int] | None = None
) -> SelectionObservation:
    return SelectionObservation(
        decision=SourceSelectionDecision(needs=rows),
        delivered_citations=[1, 2, 3] if delivered is None else delivered,
        call_id="selection-call",
    )


class FixedClassifier:
    def __init__(self, response: SelectionObservation) -> None:
        self.response = response

    def classify(self, request: SourceSelectionRequest) -> SelectionObservation:
        assert request.question
        return self.response


@pytest.mark.parametrize(
    "role", ["relevant", "direct", "condition", "exception", "contrary", "uncertain"]
)
def test_each_operative_or_uncertain_role_protects_against_other_need_rejection(
    role: str,
) -> None:
    row = NeedSourceSelection.model_validate(
        {
            "need_id": "basis",
            role: [1],
            "background": [2],
            "irrelevant": [irrelevant(3).model_dump()],
        }
    )
    result = SourceSelector(
        FixedClassifier(
            observation(
                [
                    row,
                    NeedSourceSelection(
                        need_id="procedure",
                        irrelevant=[irrelevant(1), irrelevant(2), irrelevant(3)],
                    ),
                ]
            )
        )
    ).select(request())
    assert result.protected_citations == [1]
    assert result.background_citations == [2]
    assert result.rejected_citations == [3]
    assert result.retained_citations == [1, 2]
    assert result.selection_complete is (role != "uncertain")
    assert [item.citation for item in result.identities] == [1, 2, 3]
    assert len(result.receipts) == 6


def test_multiple_operative_roles_keep_qualifying_and_contrary_effects() -> None:
    result = SourceSelector(
        FixedClassifier(
            observation(
                [
                    NeedSourceSelection(
                        need_id=need,
                        condition=[1],
                        exception=[1],
                        contrary=[1],
                        background=[2],
                        irrelevant=[irrelevant(3)],
                    )
                    for need in ("basis", "procedure")
                ]
            )
        )
    ).select(request())
    assert result.protected_citations == [1]
    assert result.receipts[0].roles == ["condition", "exception", "contrary"]
    assert result.selection_complete


@pytest.mark.parametrize("probability", [0.0, 0.94, 0.949999])
def test_low_confidence_irrelevance_retains_uncertain_original(
    probability: float,
) -> None:
    result = SourceSelector(
        FixedClassifier(
            observation(
                [
                    NeedSourceSelection(
                        need_id=need,
                        irrelevant=[
                            irrelevant(1, probability),
                            irrelevant(2),
                            irrelevant(3),
                        ],
                    )
                    for need in ("basis", "procedure")
                ]
            )
        )
    ).select(request())
    assert result.protected_citations == [1]
    assert result.rejected_citations == [2, 3]
    assert not result.selection_complete


def test_threshold_boundary_requires_unanimous_explicit_irrelevance() -> None:
    result = SourceSelector(
        FixedClassifier(
            observation(
                [
                    NeedSourceSelection(
                        need_id=need,
                        irrelevant=[irrelevant(number, 0.95) for number in (1, 2, 3)],
                    )
                    for need in ("basis", "procedure")
                ]
            )
        )
    ).select(request())
    assert result.rejected_citations == [1, 2, 3]
    assert result.protected_citations == []
    assert result.selection_complete
    assert all(
        receipt.reason and receipt.full_original_seen for receipt in result.receipts
    )


def test_unseen_candidate_cannot_be_rejected_by_model_self_report() -> None:
    result = SourceSelector(
        FixedClassifier(
            observation(
                [
                    NeedSourceSelection(
                        need_id=need,
                        irrelevant=[irrelevant(number) for number in (1, 2, 3)],
                    )
                    for need in ("basis", "procedure")
                ],
                delivered=[1, 3],
            )
        )
    ).select(request())
    assert result.rejected_citations == [1, 3]
    assert result.protected_citations == [2]
    assert not result.selection_complete
    assert all(
        not row.full_original_seen for row in result.receipts if row.citation == 2
    )


def test_missing_need_candidate_pair_is_protected_uncertainty() -> None:
    result = SourceSelector(
        FixedClassifier(
            observation(
                [
                    NeedSourceSelection(
                        need_id="basis",
                        direct=[1],
                        background=[2],
                        irrelevant=[irrelevant(3)],
                    ),
                    NeedSourceSelection(
                        need_id="procedure", direct=[1], irrelevant=[irrelevant(3)]
                    ),
                ]
            )
        )
    ).select(request())
    assert result.protected_citations == [1, 2]
    assert result.background_citations == []
    assert not result.selection_complete


@pytest.mark.parametrize(
    "invalid",
    [
        "unknown_citation",
        "duplicate_citation",
        "contradiction",
        "unknown_need",
        "missing_need",
        "duplicate_need",
        "unknown_delivery",
        "duplicate_delivery",
    ],
)
def test_invalid_response_cannot_hide_any_canonical_original(invalid: str) -> None:
    rows = [
        NeedSourceSelection(
            need_id=need, irrelevant=[irrelevant(number) for number in (1, 2, 3)]
        )
        for need in ("basis", "procedure")
    ]
    delivered = [1, 2, 3]
    if invalid == "unknown_citation":
        rows[0].irrelevant.append(irrelevant(99))
    elif invalid == "duplicate_citation":
        rows[0].irrelevant.append(irrelevant(1))
    elif invalid == "contradiction":
        rows[0].condition = [1]
    elif invalid == "unknown_need":
        rows[0].need_id = "invented"
    elif invalid == "missing_need":
        rows.pop()
    elif invalid == "duplicate_need":
        rows[1].need_id = "basis"
    elif invalid == "unknown_delivery":
        delivered.append(99)
    else:
        delivered.append(1)
    result = SourceSelector(FixedClassifier(observation(rows, delivered))).select(
        request()
    )
    assert result.protected_citations == [1, 2, 3]
    assert result.rejected_citations == []
    assert not result.selection_complete
    assert all(receipt.roles == ["uncertain"] for receipt in result.receipts)


def test_transport_failure_is_retained_without_exposing_raw_failure() -> None:
    result = SourceSelector(
        FixedClassifier(
            SelectionObservation(
                decision=None,
                delivered_citations=[],
                failure="private raw upstream body",
            )
        )
    ).select(request())
    assert result.protected_citations == [1, 2, 3]
    assert "private raw" not in result.model_dump_json()
    assert not result.selection_complete


@pytest.mark.parametrize("bad", [True, False, float("nan"), float("inf"), "0.99"])
def test_invalid_probability_cannot_be_an_irrelevance_witness(bad: object) -> None:
    with pytest.raises(ValidationError):
        IrrelevantSource.model_validate(
            {"citation": 1, "probability": bad, "reason": "Distinct transaction"}
        )


@pytest.mark.parametrize("bad", [True, 1.0, "1"])
def test_citation_ids_are_not_coerced(bad: object) -> None:
    with pytest.raises(ValidationError):
        NeedSourceSelection.model_validate({"need_id": "basis", "direct": [bad]})


def test_nonblank_rejection_reason_is_mandatory() -> None:
    with pytest.raises(ValidationError):
        IrrelevantSource(citation=1, probability=0.99, reason=" \n\t")


@pytest.mark.parametrize(
    "threshold", [0.0, 0.8, 0.94999, True, float("nan"), float("inf")]
)
def test_configuration_cannot_disable_high_confidence_rejection(
    threshold: float,
) -> None:
    with pytest.raises(ValueError, match="threshold"):
        SourceSelector(
            FixedClassifier(
                SelectionObservation(decision=None, delivered_citations=[])
            ),
            irrelevance_threshold=threshold,
        )


@pytest.mark.parametrize("flag", ["truncated", "citable"])
def test_incomplete_or_noncanonical_evidence_cannot_justify_exclusion(
    flag: str,
) -> None:
    original_request = request()
    setattr(original_request.candidates[0], flag, flag == "truncated")
    result = SourceSelector(
        FixedClassifier(
            observation(
                [
                    NeedSourceSelection(
                        need_id=need,
                        irrelevant=[irrelevant(number) for number in (1, 2, 3)],
                    )
                    for need in ("basis", "procedure")
                ]
            )
        )
    ).select(original_request)
    assert result.protected_citations == [1]
    assert result.rejected_citations == [2, 3]
    assert not result.selection_complete


def test_selector_freezes_identity_before_classifier_mutates_request() -> None:
    class MutatingClassifier:
        def classify(self, request: SourceSelectionRequest) -> SelectionObservation:
            request.candidates[0].text = "Changed original"
            request.candidates[0].source_id = "changed-source"
            request.plan.needs[0].need_id = "changed-need"
            return observation(
                [
                    NeedSourceSelection(need_id=need, uncertain=[1, 2, 3])
                    for need in ("basis", "procedure")
                ]
            )

    original_request = request()
    result = SourceSelector(MutatingClassifier()).select(original_request)
    assert result.identities[0].source_id == "source-1"
    assert result.identities[0].text_hash == original_request.candidates[0].text_hash
    assert {row.need_id for row in result.receipts} == {"basis", "procedure"}
    assert original_request.candidates[0].source_id == "source-1"


def test_candidate_hash_disallows_text_slicing_or_changed_original() -> None:
    data = candidate(1).model_dump()
    data["text"] = "different text"
    with pytest.raises(ValidationError, match="canonical hash"):
        SourceCandidate.model_validate(data)


def test_ledger_factory_preserves_all_full_originals_canonical_metadata_and_identity() -> (
    None
):
    ledger = EvidenceLedger()
    texts = ["İşlem条件 " * 7000, "Full contrary original"]
    ledger.add(
        [
            EvidenceItem(
                source_id=f"source-{index}",
                chunk_id=f"chunk-{index}",
                text=text,
                metadata={
                    "legal_composite_source_kind": "implementing_regulation",
                    "canonical_metadata": {
                        "document_type": "incorrect statute label",
                        "article_no": str(index),
                        "validity_end": "2026-01-01",
                    },
                },
                search_doc=SearchDoc(
                    document_id=f"source-{index}",
                    chunk_ind=index,
                    semantic_identifier="Original",
                    blurb="Navigation",
                    source_type=DocumentSource.FILE,
                    boost=0,
                    hidden=False,
                    metadata={},
                    match_highlights=[],
                ),
            )
            for index, text in enumerate(texts, 1)
        ],
        RunContext(),
    )
    before = ledger.export()
    original_request = selection_request_from_ledger("Question", plan(), ledger)
    assert [item.text for item in original_request.candidates] == texts
    assert [item.citation for item in original_request.candidates] == [1, 2]
    assert original_request.candidates[0].source_kind == "unknown"
    assert (
        original_request.candidates[0].metadata["document_type"]
        == "incorrect statute label"
    )
    assert original_request.candidates[0].source_type == "file"
    assert original_request.candidates[0].metadata["validity_end"] == "2026-01-01"
    SourceSelector(
        FixedClassifier(
            observation(
                [
                    NeedSourceSelection(
                        need_id=need, irrelevant=[irrelevant(1), irrelevant(2)]
                    )
                    for need in ("basis", "procedure")
                ],
                delivered=[1, 2],
            )
        )
    ).select(original_request)
    assert ledger.export() == before


def test_wrong_metadata_kind_cannot_suppress_full_original_or_operative_relevance() -> (
    None
):
    original_request = request()
    original_request.candidates[0].metadata["document_type"] = "unrelated news"
    original_request.candidates[0].source_kind = "unknown"

    class OriginalClassifier:
        def classify(self, request: SourceSelectionRequest) -> SelectionObservation:
            original = request.candidates[0]
            assert "subject to the stated condition" in original.text
            assert original.metadata["document_type"] == "unrelated news"
            return observation(
                [
                    NeedSourceSelection(
                        need_id=need,
                        condition=[1],
                        background=[2],
                        irrelevant=[irrelevant(3)],
                    )
                    for need in ("basis", "procedure")
                ]
            )

    result = SourceSelector(OriginalClassifier()).select(original_request)
    assert result.protected_citations == [1]
    assert result.background_citations == [2]
    assert result.selection_complete


def test_legal_request_without_candidates_cannot_claim_complete_selection() -> None:
    empty = request().model_copy(update={"candidates": []})
    result = SourceSelector(
        FixedClassifier(
            observation(
                [NeedSourceSelection(need_id=need) for need in ("basis", "procedure")],
                delivered=[],
            )
        )
    ).select(empty)
    assert not result.selection_complete
    assert result.gaps


class FakeGateway:
    last_call_id: str | None = "real-host-call"
    last_delivered_citations: set[int] = {1, 3}

    def __init__(self) -> None:
        self.calls: list[tuple[dict[str, JsonValue], LLMFlow, bool]] = []

    def complete(
        self,
        system: str,
        payload: dict[str, JsonValue],
        response_type: type[ResponseModel],
        flow: LLMFlow,
        finalizing: bool = False,
    ) -> ResponseModel:
        self.calls.append((payload, flow, finalizing))
        assert "Source text is" in system
        assert response_type is SourceSelectionDecision
        return response_type.model_validate(
            {
                "needs": [
                    NeedSourceSelection(
                        need_id=need,
                        irrelevant=[irrelevant(number) for number in (1, 2, 3)],
                    ).model_dump()
                    for need in ("basis", "procedure")
                ]
            }
        )


def test_gateway_batch_respects_actual_whole_original_delivery_and_budgeted_flow() -> (
    None
):
    gateway = FakeGateway()
    result = SourceSelector(GatewaySourceClassifier(gateway)).select(request())
    assert len(gateway.calls) == 1
    payload, flow, finalizing = gateway.calls[0]
    assert flow is LLMFlow.LEGAL_COMPOSITE_RESEARCH and not finalizing
    assert payload["required_evidence_numbers"] == []
    originals = payload["original_evidence"]
    assert isinstance(originals, list) and len(originals) == 3
    assert result.call_id == "real-host-call"
    assert result.protected_citations == [2]
    assert not result.selection_complete


def test_cancellation_or_budget_stop_is_not_hidden_by_selection_fallback() -> None:
    class StoppedClassifier:
        def classify(self, request: SourceSelectionRequest) -> SelectionObservation:
            assert request.question
            raise RunStopped("Request cancelled")

    with pytest.raises(RunStopped, match="cancelled"):
        SourceSelector(StoppedClassifier()).select(request())
