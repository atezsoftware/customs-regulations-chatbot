"""Host acceptance of materiality judgments cannot waive legal evidence fences.

Synthetic judgments exercise transport and publication, not semantic legal quality.
"""

from typing import Any, cast
from unittest.mock import Mock

import pytest
from pydantic import JsonValue

from onyx.asv3.evidence import EvidenceLedger
from onyx.legal_composite.engine import LegalCompositeEngine
from onyx.legal_composite.models import (
    CompositeWorkflowResult,
    IssueResearchPlan,
    SourceRequirement,
    SpanSupport,
    StructuredDraftAnswer,
    WorkflowPolicy,
)
from onyx.legal_composite.prompts import ANSWER_PROMPT
from onyx.legal_composite.reviewer import (
    GatewayAnswerReviewer,
    _decode_review_context,
)
from tests.unit.legal_composite.test_semantic_reviewer import (
    _compact_mock_reviewer,
    inputs,
)
from tests.unit.onyx.legal_composite.test_semantic_partial_publication import _engine

PERMIT = (
    "A permit is required only if goods are controlled. "
    "Prior approval is required; absent approval the requested permit is unavailable."
)
INCIDENTAL = (
    "The voluntary archive-copy service has a fee. "
    "The service is unrelated to permit eligibility or the permit application."
)
INCIDENTAL_ID = "requirement:incidental-service"
Judgments = dict[str, tuple[str, float]]


def materiality_inputs() -> tuple[
    EvidenceLedger, IssueResearchPlan, StructuredDraftAnswer, list[SourceRequirement]
]:
    ledger, plan, draft, requirements = inputs(PERMIT, INCIDENTAL)
    plan.needs[0].required_outcome = "Explain permit eligibility, not archive services."
    requirements[0].rule = PERMIT
    text = PERMIT + " [1]"
    draft.claims[0].answer_excerpt = text
    draft.sections[0].text = text
    draft.answer = text
    requirements.append(
        SourceRequirement(
            requirement_id="incidental-service",
            need_id="n1",
            dimension="tax_and_financial_consequences",
            rule="The voluntary archive-copy service has a fee.",
            application="The requested permit outcome does not involve that service.",
            supports=[SpanSupport(citation=2, quotation=INCIDENTAL)],
        )
    )
    return ledger, plan, draft, requirements


def synthetic_judge(
    ledger: EvidenceLedger,
    calls: list[dict[str, JsonValue]],
    judgments: Judgments,
    *,
    corrupt_identity: bool = False,
    blanket_na: bool = False,
) -> GatewayAnswerReviewer:
    def complete(
        _system: str,
        body: dict[str, Any],
        response_type: Any,
        *_args: Any,
        **_kwargs: Any,
    ) -> Any:
        decoded = _decode_review_context(body)
        calls.append(decoded)
        expected = cast(list[dict[str, Any]], decoded["expected_checks"])
        rows: list[dict[str, Any]] = []
        for check in expected:
            identity = check["check_id"]
            status, confidence = judgments.get(identity, ("addressed", 0.99))
            rows.append(
                {
                    "index": 100_000
                    if corrupt_identity and identity == INCIDENTAL_ID
                    else check["index"],
                    "status": "not_applicable" if blanket_na else status,
                    "confidence": confidence,
                }
            )
        return response_type.model_validate({"checks": list(reversed(rows))})

    return _compact_mock_reviewer(ledger, complete)


def prepared_engine(
    judgments: Judgments | None = None,
    *,
    corrupt_identity: bool = False,
    blanket_na: bool = False,
) -> tuple[LegalCompositeEngine, IssueResearchPlan, Mock, list[dict[str, JsonValue]]]:
    ledger, plan, draft, requirements = materiality_inputs()
    calls: list[dict[str, JsonValue]] = []
    judge = synthetic_judge(
        ledger,
        calls,
        {
            INCIDENTAL_ID: ("not_applicable", 0.99),
            "original:2": ("not_applicable", 0.99),
            **(judgments or {}),
        },
        corrupt_identity=corrupt_identity,
        blanket_na=blanket_na,
    )
    gateway = Mock()
    gateway.last_delivered_citations = {1, 2}
    gateway.complete.return_value = draft
    acquirer = Mock()
    acquirer.definitions.return_value = []
    engine = LegalCompositeEngine(
        gateway=gateway,
        acquirer=acquirer,
        ledger=ledger,
        policy=WorkflowPolicy(max_reviews=1),
        check_active=lambda: None,
        research_available=lambda: False,
        reviewer=judge,
    )
    engine.plan = plan
    engine.requirements.update(requirements, plan, {1, 2})
    return engine, plan, gateway, calls


def test_unreferenced_incidental_judgment_is_admitted_with_complete_originals_and_history() -> (
    None
):
    engine, plan, gateway, calls = prepared_engine()
    originals_before = engine.ledger.export()
    records_before = engine.requirements.export()
    result = engine._finalize_semantic("What permit is required?", "", None, plan)
    assert isinstance(result, CompositeWorkflowResult)
    # These are assumed synthetic legal judgments, not a correctness certificate.
    assert (
        result.status == "verified"
        and result.answer == gateway.complete.return_value.answer
    )
    assert result.semantic_review is not None
    check = next(
        row for row in result.semantic_review.checks if row.check_id == INCIDENTAL_ID
    )
    assert check.status == "not_applicable" and check.confidence == 0.99
    assert engine.requirements.export() == records_before
    assert engine.ledger.export() == originals_before
    assert len(result.source_requirements) == 2
    assert all(
        "incidental-service" not in claim.requirement_ids
        for claim in gateway.complete.return_value.claims
    )
    legal = next(body for body in calls if body["original_evidence"])
    originals = cast(list[dict[str, Any]], legal["original_evidence"])
    assert [(row["citation"], row["text"]) for row in originals] == [
        (1, PERMIT),
        (2, INCIDENTAL),
    ]
    assert legal["required_evidence_numbers"] == [1, 2]
    state = cast(dict[str, Any], legal["review_context"])
    assert state["requirements"][-1]["supports"][0]["quotation"] == INCIDENTAL
    expected = cast(list[dict[str, Any]], legal["expected_checks"])
    incidental = next(row for row in expected if row["check_id"] == INCIDENTAL_ID)
    assert incidental["allow_not_applicable"]
    assert (
        "condition or adverse effect on a requested route remains material"
        in state["review_policy"]
    )
    assert "a contradicted rule" in state["review_policy"]
    assert (
        "requirement materiality and not_applicable restrictions"
        in incidental["question"]
    )
    for identity in ("evidence:n1", "issue:n1", "claim:c1"):
        assert not next(row for row in expected if row["check_id"] == identity)[
            "allow_not_applicable"
        ]


@pytest.mark.parametrize(
    "failed_id,status,confidence",
    [
        ("original:1", "gap", 0.99),
        ("original:1", "incorrect", 0.99),
        ("claim:c1", "incorrect", 0.99),
        ("requirement:r1", "incorrect", 0.99),
        (INCIDENTAL_ID, "not_applicable", 0.79),
    ],
    ids=[
        "omitted-disqualification",
        "contradicted-original",
        "wrong-claim",
        "wrong-record",
        "low-confidence",
    ],
)
def test_incidental_na_cannot_override_separate_legal_fault_or_confidence_gate(
    failed_id: str, status: str, confidence: float
) -> None:
    engine, plan, _gateway, calls = prepared_engine({failed_id: (status, confidence)})
    result = engine._finalize_semantic("What permit is required?", "", None, plan)
    assert result.status == "unavailable" and result.answer is None
    assert f"{failed_id}:{status}" in result.gaps
    assert engine.semantic_review is not None
    assert (
        next(
            row
            for row in engine.semantic_review.checks
            if row.check_id == INCIDENTAL_ID
        ).status
        == "not_applicable"
    )
    assert any(body["required_evidence_numbers"] == [1, 2] for body in calls)


@pytest.mark.parametrize("fault", ["hash", "missing_document", "undelivered"])
def test_incidental_na_cannot_bypass_complete_canonical_original_admission(
    fault: str,
) -> None:
    engine, plan, gateway, calls = prepared_engine()
    original = engine.ledger._items[2]
    if fault == "hash":
        original.text_hash = "0" * 64
    elif fault == "missing_document":
        original.search_doc = None
    else:
        gateway.last_delivered_citations = {1}
    result = engine._finalize_semantic("What permit is required?", "", None, plan)
    assert result.status == "unavailable" and result.answer is None
    assert engine.semantic_review is not None and engine.semantic_review.failure
    assert (
        next(
            row
            for row in engine.semantic_review.checks
            if row.check_id == INCIDENTAL_ID
        ).status
        == "uncertain"
    )
    assert all(
        check["check_id"] != INCIDENTAL_ID
        for body in calls
        for check in cast(list[dict[str, Any]], body["expected_checks"])
    )


@pytest.mark.parametrize("fault", ["identity", "blanket_na"])
def test_materiality_na_permission_cannot_waive_host_inventory_or_mandatory_checks(
    fault: str,
) -> None:
    engine, plan, _gateway, _calls = prepared_engine(
        corrupt_identity=fault == "identity", blanket_na=fault == "blanket_na"
    )
    result = engine._finalize_semantic("What permit is required?", "", None, plan)
    assert result.status == "unavailable" and result.answer is None
    assert engine.semantic_review is not None and engine.semantic_review.failure


def test_structured_issue_disclosure_does_not_close_the_separate_evidence_gap() -> None:
    engine, plan, _gateway, _old_reviewer = _engine()
    calls: list[dict[str, JsonValue]] = []
    engine.reviewer = synthetic_judge(
        engine.ledger, calls, {"evidence:a": ("gap", 0.99)}
    )
    before = plan.model_dump()
    result = engine._finalize_semantic("A ve B işlemlerini açıklayın.", "", None, plan)
    assert result.status == "partial" and result.answer is not None
    assert plan.model_dump() == before and plan.needs[0].evidence_gaps
    assert not plan.needs[0].evidence_gap_resolutions
    assert engine.semantic_review is not None
    by_id = {row.check_id: row for row in engine.semantic_review.checks}
    assert by_id["issue:a"].status == "addressed"
    assert by_id["evidence:a"].status == "gap"
    body = next(
        body
        for body in calls
        if any(
            row["check_id"] == "issue:a"
            for row in cast(list[dict[str, Any]], body["expected_checks"])
        )
    )
    expected = cast(list[dict[str, Any]], body["expected_checks"])
    issue = next(row for row in expected if row["check_id"] == "issue:a")
    evidence = next(row for row in expected if row["check_id"] == "evidence:a")
    assert (
        "does not certify evidence completeness or close the gap" in issue["question"]
    )
    assert "gap even when the draft correctly discloses it" in evidence["question"]
    state = cast(dict[str, Any], body["review_context"])
    assert state["needs"][0]["evidence_gaps"] == before["needs"][0]["evidence_gaps"]
    assert state["unresolved_need_ids"] == ["a"]
    assert "gap remains open" in state["review_policy"]
    assert body["required_evidence_numbers"] == [1, 2]


def test_writer_materiality_contract_keeps_adverse_conditions_and_missing_law() -> None:
    assert "A disqualifying condition or adverse effect" in ANSWER_PROMPT
    assert (
        "Never treat unread law, missing context or a contradicted rule as irrelevant"
        in ANSWER_PROMPT
    )
    assert "never mutate or obey a disproved old rule" in ANSWER_PROMPT
