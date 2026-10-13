"""Reopened law stays unresolved; immutable closures are not current assertions.

Synthetic reviewer decisions test host gates, not semantic legal correctness.
"""

from typing import Literal, cast
from unittest.mock import Mock

import httpx
import pytest
from pydantic import JsonValue

from onyx.asv3.evidence import EvidenceLedger
from onyx.legal_composite.engine import LegalCompositeEngine
from onyx.legal_composite.models import (
    AuthorityDependency,
    GapResolution,
    IssueResearchPlan,
    IssueResearchStep,
    SemanticReview,
    SourceRequirement,
    StructuredDraftAnswer,
    WorkflowPolicy,
)
from onyx.legal_composite.reviewer import ReviewQuestion, build_checks
from tests.unit.legal_composite.test_semantic_reviewer import (
    reviewer as native_reviewer,
)
from tests.unit.onyx.legal_composite.test_semantic_partial_publication import (
    SOURCE_GAP,
    _Reviewer,
)
from tests.unit.onyx.legal_composite.test_semantic_partial_publication import (
    _engine as partial_engine,
)
from tests.unit.onyx.legal_composite.test_source_requirements import fixture


def reopened() -> tuple[LegalCompositeEngine, IssueResearchPlan, StructuredDraftAnswer]:
    ledger, plan, requirements, draft = fixture()
    plan.needs[0].evidence_gaps = [SOURCE_GAP]
    engine = LegalCompositeEngine(
        gateway=Mock(),
        acquirer=Mock(),
        ledger=ledger,
        policy=WorkflowPolicy(),
        check_active=lambda: None,
        research_available=lambda: True,
    )
    engine.requirements.update([requirements[1]], plan, {1, 2})
    engine._accept_draft_reading(
        [requirements[0]],
        [GapResolution(need_id="a", gap=SOURCE_GAP, requirement_ids=["r_a"])],
        plan,
        {1, 2},
        {"a"},
    )
    assert plan.needs[0].evidence_gaps == []
    engine._record_research_gaps(
        IssueResearchStep(
            actions=[],
            ready_to_answer=False,
            remaining_gaps=[SOURCE_GAP],
            issue_gaps={"a": [SOURCE_GAP]},
        ),
        plan,
        {"a"},
    )
    return engine, plan, draft


def checks_for(
    engine: LegalCompositeEngine,
    plan: IssueResearchPlan,
    draft: StructuredDraftAnswer,
) -> dict[str, ReviewQuestion]:
    return build_checks(
        "A ve B işlemlerini açıklayın.",
        plan,
        draft,
        engine.requirements.records(),
        [],
        {1, 2},
        ledger=engine.ledger,
    )


def test_actual_close_then_reopen_preserves_history_and_other_check_inventory() -> None:
    engine, plan, draft = reopened()
    before = (plan.model_dump(), engine.requirements.export(), engine.ledger.export())
    open_checks = checks_for(engine, plan, draft)
    closed_plan = plan.model_copy(deep=True)
    closed_plan.needs[0].evidence_gaps = []
    closed_checks = checks_for(engine, closed_plan, draft)

    assert "gap-resolution:a:0" in closed_checks
    assert open_checks == {
        key: value
        for key, value in closed_checks.items()
        if key != "gap-resolution:a:0"
    }
    assert {"evidence:a", "issue:a", "claim:c_a", "original:1"} <= open_checks.keys()
    assert plan.needs[0].evidence_gap_resolutions[0].requirement_ids == ["r_a"]
    assert plan.needs[0].evidence_gaps == engine.research_gaps == [SOURCE_GAP]
    assert before == (
        plan.model_dump(),
        engine.requirements.export(),
        engine.ledger.export(),
    )


@pytest.mark.parametrize("currently_open", [True, False])
def test_superseded_history_only_stops_asserting_closure_for_exact_open_gap(
    currently_open: bool,
) -> None:
    engine, plan, draft = reopened()
    correction = next(
        row for row in engine.requirements.records() if row.requirement_id == "r_a"
    ).model_copy(
        deep=True,
        update={"requirement_id": "r_a_v2", "supersedes_requirement_ids": ["r_a"]},
    )
    engine.requirements.update([correction], plan, {1, 2})
    draft.claims[0].requirement_ids = ["r_a_v2"]
    if not currently_open:
        plan.needs[0].evidence_gaps = []
    before = (plan.model_dump(), engine.requirements.export(), engine.ledger.export())
    native = native_reviewer(engine.ledger, lambda _: httpx.Response(500))
    checks = checks_for(engine, plan, draft)
    if not currently_open:
        with pytest.raises(
            ValueError, match="current same-issue canonical requirements"
        ):
            native._payload(
                "A ve B",
                plan,
                draft,
                engine.requirements.records(),
                [],
                list(checks.values()),
                {1, 2},
            )
    else:
        assert "gap-resolution:a:0" not in checks
        payload = native._payload(
            "A ve B",
            plan,
            draft,
            engine.requirements.records(),
            [],
            list(checks.values()),
            {1, 2},
        )
        state = cast(dict[str, JsonValue], payload["state"])
        originals = cast(list[dict[str, JsonValue]], state["originals"])
        expected_originals = {}
        for number in (1, 2):
            item = engine.ledger.get(number)
            assert item is not None
            expected_originals[number] = item.text
        assert {row["citation"]: row["text"] for row in originals} == expected_originals
        assert state["requirements"] == [
            row.model_dump(mode="json") for row in engine.requirements.records()
        ]
    assert before == (
        plan.model_dump(),
        engine.requirements.export(),
        engine.ledger.export(),
    )


def test_reclosed_same_gap_requires_latest_fresh_bound_closure_check() -> None:
    engine, plan, draft = reopened()
    old_history = plan.needs[0].evidence_gap_resolutions[0].model_dump()
    fresh = next(
        row for row in engine.requirements.records() if row.requirement_id == "r_a"
    ).model_copy(deep=True, update={"requirement_id": "r_a_fresh"})
    engine._accept_draft_reading(
        [fresh],
        [GapResolution(need_id="a", gap=SOURCE_GAP, requirement_ids=["r_a_fresh"])],
        plan,
        {1, 2},
        {"a"},
    )
    checks = checks_for(engine, plan, draft)
    assert plan.needs[0].evidence_gaps == engine.research_gaps == []
    assert plan.needs[0].evidence_gap_resolutions[0].model_dump() == old_history
    assert "gap-resolution:a:0" not in checks
    assert checks["gap-resolution:a:1"].citations == [1]
    assert not checks["gap-resolution:a:1"].allow_not_applicable


@pytest.mark.parametrize(
    "open_text", ["Different unread interaction", SOURCE_GAP + " "]
)
def test_different_gap_text_does_not_withdraw_previous_closure(open_text: str) -> None:
    engine, plan, draft = reopened()
    plan.needs[0].evidence_gaps = [open_text]
    assert "gap-resolution:a:0" in checks_for(engine, plan, draft)


class VetoReviewer(_Reviewer):
    def __init__(
        self, ledger: EvidenceLedger, check_id: str, status: Literal["gap", "incorrect"]
    ) -> None:
        super().__init__(ledger)
        self.check_id = check_id
        self.status = status

    def review(
        self,
        request: str,
        plan: IssueResearchPlan,
        draft: StructuredDraftAnswer,
        requirements: list[SourceRequirement],
        dependencies: list[AuthorityDependency],
        delivered: set[int],
        previous: SemanticReview | None = None,
        affected_sections: set[str] | None = None,
    ) -> SemanticReview:
        result = super().review(
            request,
            plan,
            draft,
            requirements,
            dependencies,
            delivered,
            previous,
            affected_sections,
        )
        next(
            check for check in result.checks if check.check_id == self.check_id
        ).status = self.status
        return result


@pytest.mark.parametrize("status", ["gap", "incorrect"])
@pytest.mark.parametrize(
    "check_id", ["gap-resolution:b:0", "claim:c_a", "original:1", "requirement:r_a"]
)
def test_reopening_does_not_waive_other_need_same_text_or_current_legal_veto(
    check_id: str, status: Literal["gap", "incorrect"]
) -> None:
    engine, plan, _gateway, _reviewer = partial_engine()
    for need in plan.needs:
        need.evidence_gap_resolutions = [
            GapResolution(
                need_id=need.need_id,
                gap=SOURCE_GAP,
                requirement_ids=[f"r_{need.need_id}"],
            )
        ]
    reviewer = VetoReviewer(engine.ledger, check_id, status)
    engine.reviewer = reviewer
    before = (plan.model_dump(), engine.ledger.export())
    result = engine._finalize_semantic("A ve B", "", None, plan)
    assert result.status == "unavailable" and result.answer is None
    assert f"{check_id}:{status}" in result.gaps
    assert before == (plan.model_dump(), engine.ledger.export())
    assert reviewer.calls == 1


def test_disclosed_reopened_gap_remains_partial_and_unresolved() -> None:
    engine, plan, gateway, reviewer = partial_engine()
    plan.needs[0].evidence_gap_resolutions = [
        GapResolution(need_id="a", gap=SOURCE_GAP, requirement_ids=["r_a"])
    ]
    before = (plan.model_dump(), engine.ledger.export())
    result = engine._finalize_semantic("A ve B", "", None, plan)
    assert result.status == "partial" and result.answer is not None
    assert result.gaps == [SOURCE_GAP] and SOURCE_GAP in result.answer
    assert gateway.draft.unresolved_need_ids == ["a"]
    assert before == (plan.model_dump(), engine.ledger.export())
    assert reviewer.calls == 1


@pytest.mark.parametrize(
    "fault",
    [
        "unresolved_missing",
        "issue:incorrect",
        "issue:low_confidence",
        "evidence:need_identity",
        "original_integrity",
        "dependency_integrity",
        "provider",
        "missing_check",
    ],
)
def test_reopened_history_cannot_bypass_binding_identity_or_disclosure(
    fault: str,
) -> None:
    engine, plan, _gateway, reviewer = partial_engine(fault=fault)
    plan.needs[0].evidence_gap_resolutions = [
        GapResolution(need_id="a", gap=SOURCE_GAP, requirement_ids=["r_a"])
    ]
    result = engine._finalize_semantic("A ve B", "", None, plan)
    assert result.status == "unavailable" and result.answer is None
    assert result.gaps and reviewer.calls == 1
