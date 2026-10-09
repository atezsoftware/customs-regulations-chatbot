"""Task coverage cannot require a corpus review or certify legal correctness."""

import json
from copy import deepcopy
from typing import Any, cast
from unittest.mock import Mock

import httpx
import pytest
from pydantic import JsonValue

from onyx.asv3.evidence import EvidenceLedger
from onyx.legal_composite.budget import WorkflowBudget
from onyx.legal_composite.engine import LegalCompositeEngine
from onyx.legal_composite.gateway import BudgetedGateway, ModelContextLimit
from onyx.legal_composite.models import AnswerSection, WorkflowPolicy
from onyx.legal_composite.reviewer import (
    GatewayAnswerReviewer,
    _decode_review_context,
    _review_generation_policy,
)
from onyx.llm.interfaces import LLMConfig
from tests.unit.legal_composite.test_semantic_reviewer import inputs, reviewer
from tests.unit.onyx.legal_composite.draft_composition_fixture import composition_for


def coverage_state(body: dict[str, JsonValue]) -> dict[str, JsonValue]:
    return cast(dict[str, JsonValue], body["review_context"])


def test_coverage_context_preserves_complete_request_facts_needs_and_every_section() -> (
    None
):
    ledger, plan, draft, requirements = inputs(
        "An incidental complete source. " * 10_000
    )
    request = "Explain permission; separately explain the requested alternative."
    plan.needs[
        0
    ].required_outcome = "Permission conditions and the user's alternatives."
    plan.needs[0].relevant_facts = ["The user supplied one event date."]
    plan.missing_user_facts = ["Goods' actual control status."]
    draft.unresolved_need_ids = ["n1"]
    draft.sections.append(
        AnswerSection(
            section_id="other",
            need_ids=["n1"],
            text='ALL literal additional prose: {"quotation_ref":0} / İ / I\u0307 / 🚢',
        )
    )
    draft.answer = "\n\n".join(section.text for section in draft.sections)
    native = reviewer(ledger, lambda _: httpx.Response(500))
    check = native.expected_checks(request, plan, draft, requirements, [], {1})[
        "request:coverage"
    ]
    before = deepcopy(
        (plan.model_dump(), draft.model_dump(), requirements[0].model_dump())
    )
    body = native._payload(request, plan, draft, requirements, [], [check], {1})
    state = cast(dict[str, Any], body["state"])
    assert check.citations == [] and not check.allow_not_applicable
    assert state["request"] == request
    assert state["needs"] == [need.model_dump(mode="json") for need in plan.needs]
    assert state["sections"] == [
        section.model_dump(mode="json") for section in draft.sections
    ]
    assert state["missing_user_facts"] == plan.missing_user_facts
    assert state["unresolved_need_ids"] == draft.unresolved_need_ids
    assert state["review_purpose"] == "request_coverage"
    for field in (
        "originals",
        "requirements",
        "claims",
        "canonical_witnesses",
        "dependencies",
    ):
        assert state[field] == []
    criteria = cast(dict[str, Any], body["questions"])["request:coverage"]["criteria"]
    assert "never legal correctness" in criteria["addressed"]
    assert "not_applicable" not in criteria
    assert "intentionally absent" in state["review_policy"]
    assert before == (
        plan.model_dump(),
        draft.model_dump(),
        requirements[0].model_dump(),
    )


def test_mixed_batch_and_all_legal_checks_keep_complete_originals_and_witnesses() -> (
    None
):
    ledger, plan, draft, requirements = inputs(
        "Complete governing source with exception."
    )
    native = reviewer(ledger, lambda _: httpx.Response(500))
    checks = native.expected_checks("Question", plan, draft, requirements, [], {1})
    coverage = checks["request:coverage"]
    issue = checks["issue:n1"]
    alone = native._payload("Question", plan, draft, requirements, [], [issue], {1})
    mixed = native._payload(
        "Question", plan, draft, requirements, [], [issue, coverage], {1}
    )
    alone_state = cast(dict[str, Any], alone["state"])
    mixed_state = cast(dict[str, Any], mixed["state"])
    assert {k: v for k, v in alone_state.items() if k != "checks"} == {
        k: v for k, v in mixed_state.items() if k != "checks"
    }
    assert "review_purpose" not in mixed_state
    assert mixed_state["requirements"] and mixed_state["claims"]
    assert mixed_state["canonical_witnesses"]
    for identity in (
        "original:1",
        "claim:c1",
        "requirement:r1",
        "issue:n1",
        "evidence:n1",
        "dimension:n1:case_law_and_rulings",
    ):
        body = native._payload(
            "Question", plan, draft, requirements, [], [checks[identity]], {1}
        )
        state = cast(dict[str, Any], body["state"])
        assert state["originals"][0]["text"] == ledger._items[1].text
        assert state["originals"][0]["text_hash"] == ledger._items[1].text_hash
        assert state["original_context_scope"]["full_original_citations"] == [1]
        assert "review_purpose" not in state


def generation_reviewer(
    ledger: EvidenceLedger,
    calls: list[dict[str, JsonValue]],
    *,
    coverage_status: str = "addressed",
    failed_legal_id: str | None = None,
    context_cap: int | None = None,
) -> GatewayAnswerReviewer:
    config = LLMConfig(
        model_provider="openrouter",
        model_name="synthetic-test",
        temperature=1,
        max_input_tokens=192_000,
    )

    def factory() -> BudgetedGateway:
        gateway = Mock(spec=BudgetedGateway)
        gateway.selected_llm = Mock(config=config)
        gateway.last_call_id = None
        gateway.token_counter = None

        def fit(
            _system: str, body: dict[str, JsonValue], *_args: Any, **_kwargs: Any
        ) -> Any:
            if context_cap is not None and len(json.dumps(body)) > context_cap:
                raise ModelContextLimit("Synthetic complete context does not fit")
            return [], 100, body["original_evidence"]

        def complete(
            system: str,
            body: dict[str, JsonValue],
            response_type: Any,
            *_args: Any,
            **_kwargs: Any,
        ) -> Any:
            decoded = _decode_review_context(body)
            calls.append(decoded)
            state = coverage_state(decoded)
            expected = cast(list[dict[str, Any]], decoded["expected_checks"])
            coverage = state.get("review_purpose") == "request_coverage"
            if coverage:
                assert [check["check_id"] for check in expected] == ["request:coverage"]
                assert decoded["original_evidence"] == []
                assert decoded["required_evidence_numbers"] == []
                assert "task coverage only" in system
            else:
                assert all(
                    check["check_id"] != "request:coverage" for check in expected
                )
            return response_type.model_validate(
                {
                    "checks": [
                        {
                            "index": check["index"],
                            "status": coverage_status
                            if coverage
                            else "gap"
                            if check["check_id"] == failed_legal_id
                            else "addressed",
                            "confidence": 0.99,
                        }
                        for check in expected
                    ]
                }
            )

        gateway._fit_messages.side_effect = fit
        gateway.complete.side_effect = complete
        return gateway

    return GatewayAnswerReviewer(
        config=config,
        gateway_factory=factory,
        budget=WorkflowBudget(
            WorkflowPolicy(max_model_calls=100, max_context_tokens=192_000)
        ),
        ledger=ledger,
    )


def test_missing_requested_alternative_still_reaches_coverage_when_plan_omits_it() -> (
    None
):
    ledger, plan, draft, requirements = inputs()
    request = "What permit is required? Separately explain the refund alternative."
    assert all("refund" not in need.question for need in plan.needs)
    calls: list[dict[str, JsonValue]] = []
    judge = generation_reviewer(ledger, calls, coverage_status="gap")
    result = judge.review(request, plan, draft, requirements, [], {1})
    expected = judge.expected_checks(request, plan, draft, requirements, [], {1})
    assert len(calls) == 2
    coverage = next(
        body for body in calls if coverage_state(body).get("review_purpose")
    )
    assert coverage_state(coverage)["request"] == request
    assert coverage_state(coverage)["sections"] == [
        s.model_dump(mode="json") for s in draft.sections
    ]
    assert result.failure is None
    assert (
        next(
            check for check in result.checks if check.check_id == "request:coverage"
        ).status
        == "gap"
    )
    observed = [
        check["check_id"]
        for body in calls
        for check in cast(list[dict[str, Any]], body["expected_checks"])
    ]
    assert len(observed) == len(set(observed)) and set(observed) == set(expected)
    gateway = Mock()
    gateway.last_delivered_citations = {1}
    gateway.complete.return_value = composition_for(draft)
    engine = LegalCompositeEngine(
        gateway=gateway,
        acquirer=Mock(),
        ledger=ledger,
        policy=WorkflowPolicy(max_reviews=1),
        check_active=lambda: None,
        research_available=lambda: False,
        reviewer=judge,
    )
    engine.plan = plan
    engine.requirements.update(requirements, plan, {1})
    published = engine._finalize_semantic(request, "", None, plan)
    assert published.answer is None and published.status == "unavailable"
    assert "request:coverage:gap" in published.gaps


@pytest.mark.parametrize(
    "failed_legal_id", ["original:1", "claim:c1", "requirement:r1"]
)
def test_positive_coverage_cannot_waive_a_separate_legal_gap(
    failed_legal_id: str,
) -> None:
    ledger, plan, draft, requirements = inputs()
    calls: list[dict[str, JsonValue]] = []
    judge = generation_reviewer(ledger, calls, failed_legal_id=failed_legal_id)
    result = judge.review("Question", plan, draft, requirements, [], {1})
    by_id = {check.check_id: check for check in result.checks}
    assert by_id["request:coverage"].status == "addressed"
    assert by_id[failed_legal_id].status == "gap"
    legal = next(body for body in calls if body["original_evidence"])
    assert legal["required_evidence_numbers"] == [1]
    assert (
        cast(list[dict[str, JsonValue]], legal["original_evidence"])[0]["text"]
        == ledger._items[1].text
    )
    gateway = Mock()
    gateway.last_delivered_citations = {1}
    gateway.complete.return_value = composition_for(draft)
    engine = LegalCompositeEngine(
        gateway=gateway,
        acquirer=Mock(),
        ledger=ledger,
        policy=WorkflowPolicy(max_reviews=1),
        check_active=lambda: None,
        research_available=lambda: False,
        reviewer=judge,
    )
    engine.plan = plan
    engine.requirements.update(requirements, plan, {1})
    published = engine._finalize_semantic("Question", "", None, plan)
    assert published.answer is None and published.status == "unavailable"
    assert failed_legal_id + ":gap" in published.gaps


def test_coverage_can_fit_while_oversized_legal_singletons_remain_uncertain() -> None:
    ledger, plan, draft, requirements = inputs("Complete unshortened source. " * 10_000)
    calls: list[dict[str, JsonValue]] = []
    judge = generation_reviewer(ledger, calls, context_cap=20_000)
    result = judge.review("Question", plan, draft, requirements, [], {1})
    assert len(calls) == 1
    assert coverage_state(calls[0])["review_purpose"] == "request_coverage"
    assert (
        next(
            check for check in result.checks if check.check_id == "request:coverage"
        ).status
        == "addressed"
    )
    assert result.failure
    assert all(
        check.status == "uncertain"
        for check in result.checks
        if check.check_id != "request:coverage"
    )
    assert ledger._items[1].text == "Complete unshortened source. " * 10_000


def test_coverage_policy_is_only_selected_for_the_declared_singleton() -> None:
    only: dict[str, JsonValue] = {
        "review_context": {"review_purpose": "request_coverage"},
        "expected_checks": [{"check_id": "request:coverage"}],
    }
    assert "explicit task coverage only" in _review_generation_policy(only)
    mixed = deepcopy(only)
    cast(list[JsonValue], mixed["expected_checks"]).append({"check_id": "original:1"})
    assert "Never mark missing decisive law" in _review_generation_policy(mixed)
    assert "explicit task coverage only" not in _review_generation_policy(mixed)
