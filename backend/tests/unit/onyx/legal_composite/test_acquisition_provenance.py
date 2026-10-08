"""Reuse preserves every need binding without duplicating or rereading originals."""

from unittest.mock import Mock

import pytest

from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import OutcomeStatus, RunContext, ToolOutcome, ToolSpec
from onyx.asv3.registry import CapabilityRegistry, build_core_specs
from onyx.legal_composite.acquisition import CanonicalAcquirer
from onyx.legal_composite.engine import LegalCompositeEngine
from onyx.legal_composite.models import (
    ResearchNeed,
    ResearchPlan,
    ResearchStep,
    SourceAction,
    WorkflowPolicy,
)
from tests.unit.onyx.legal_composite.test_review_assessment import original


def plan() -> ResearchPlan:
    return ResearchPlan(
        language="tr",
        requires_sources=True,
        needs=[
            ResearchNeed(
                need_id=need_id,
                question=f"{need_id} sonucu nedir?",
                governing_source="Law",
                conditions_to_check=[],
            )
            for need_id in ("first", "second")
        ],
        initial_actions=[],
        missing_user_facts=[],
    )


def acquirer() -> tuple[CanonicalAcquirer, Mock]:
    ledger = EvidenceLedger()
    context = RunContext(corpus_only=True)
    handler = Mock(
        return_value=ToolOutcome(
            status=OutcomeStatus.FOUND,
            summary="Controlling original delivered.",
            evidence=[original("Immutable original text.", "shared")],
        )
    )
    registry = CapabilityRegistry(
        [
            ToolSpec(
                name="read_original",
                description="Read an authorized original.",
                parameters={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
                handler=handler,
            ),
            ToolSpec(
                name="external_source",
                description="Outside sources.",
                parameters={"type": "object", "properties": {}},
                handler=handler,
                external=True,
            ),
        ]
    )
    for spec in build_core_specs(registry, ledger, state_provider=lambda: {}):
        if spec.name == "read_evidence":
            registry.register(spec)
    return CanonicalAcquirer(registry, context, ledger, WorkflowPolicy()), handler


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("different_public_update", [False, True])
def test_duplicate_or_cached_action_merges_every_need_without_extra_work(
    cached: bool,
    different_public_update: bool,
) -> None:
    instance, handler = acquirer()
    request_plan = plan()
    first = SourceAction(need_ids=["first"], tool="read_original", arguments={})
    second = SourceAction(
        need_ids=["second", "first"], tool="read_original", arguments={}
    )
    if different_public_update:
        first.arguments["_public_update"] = ["Original read", "Read the original"]
        second.arguments["_public_update"] = ["Original reuse", "Reuse the original"]
    if cached:
        instance.acquire([first], request_plan)
        bytes_before = instance.context.budget.snapshot()["evidence_bytes"]
        receipts = instance.acquire([second], request_plan)
        assert receipts[0]["reused"] is True
        assert instance.context.budget.snapshot()["evidence_bytes"] == bytes_before
    else:
        receipts = instance.acquire([first, second], request_plan)
        assert receipts[0]["need_ids"] == ["first", "second"]
    handler.assert_called_once()
    assert "_public_update" not in handler.call_args.args[0]
    assert len(receipts) == 1 and receipts[0]["citations"] == [1]
    assert instance.context.budget.snapshot()["tools"] == 1
    assert instance.ledger.citation_numbers() == (1,)
    item = instance.ledger.get(1)
    assert item is not None
    assert item.question_ids == ["first", "second"]
    assert item.text == "Immutable original text." and item.chunk_id == "shared"


def test_different_source_queries_are_not_collapsed_with_same_public_update() -> None:
    instance, handler = acquirer()
    instance.registry.register(
        ToolSpec(
            name="resolve_source",
            description="Resolve an authorized source identity.",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
                "additionalProperties": False,
            },
            handler=handler,
        )
    )
    actions = [
        SourceAction(
            need_ids=[need_id],
            tool="resolve_source",
            arguments={"query": query, "_public_update": ["Resolve", "Resolve source"]},
        )
        for need_id, query in [
            ("first", "Governing instrument"),
            ("second", "Implementing instrument"),
        ]
    ]
    receipts = instance.acquire(actions, plan())
    assert handler.call_count == 2 and len(receipts) == 2
    assert {call.args[0]["query"] for call in handler.call_args_list} == {
        "Governing instrument",
        "Implementing instrument",
    }
    assert instance.context.budget.snapshot()["tools"] == 2


def test_reopened_original_binds_new_need_without_new_text_or_identity() -> None:
    instance, handler = acquirer()
    request_plan = plan()
    instance.acquire(
        [SourceAction(need_ids=["first"], tool="read_original", arguments={})],
        request_plan,
    )
    before = instance.ledger.get(1)
    bytes_before = instance.context.budget.snapshot()["evidence_bytes"]
    receipts = instance.acquire(
        [
            SourceAction(
                need_ids=["second"], tool="read_evidence", arguments={"citation": 1}
            )
        ],
        request_plan,
    )
    after = instance.ledger.get(1)
    assert before is not None and after is not None
    assert after.question_ids == ["first", "second"]
    assert after.identity == before.identity and after.text == before.text
    assert receipts[0]["citations"] == [1]
    assert instance.ledger.citation_numbers() == (1,)
    assert instance.context.budget.snapshot()["evidence_bytes"] == bytes_before
    handler.assert_called_once()


@pytest.mark.parametrize("invalid", ["unknown_need", "unknown_tool", "external_tool"])
def test_invalid_planned_action_becomes_unavailable_without_tool_execution(
    invalid: str,
) -> None:
    instance, handler = acquirer()
    request_plan = plan()
    action = SourceAction(
        need_ids=["unknown"] if invalid == "unknown_need" else ["first"],
        tool={
            "unknown_need": "read_original",
            "unknown_tool": "missing_tool",
            "external_tool": "external_source",
        }[invalid],
        arguments={},
    )
    gateway = Mock()
    if invalid == "unknown_need":
        gateway.complete.side_effect = [
            request_plan,
            ResearchStep(actions=[action], ready_to_answer=False, remaining_gaps=[]),
        ]
    else:
        request_plan.initial_actions = [action]
        gateway.complete.return_value = request_plan
    engine = LegalCompositeEngine(
        gateway=gateway,
        acquirer=instance,
        ledger=instance.ledger,
        policy=WorkflowPolicy(),
        check_active=instance.context.check_active,
        research_available=lambda: invalid == "unknown_need",
    )
    result = engine.run("Kaynaklı hukuki sonucu açıkla.")
    assert result.status == "unavailable" and result.answer is None
    assert result.gaps and result.plan is not None
    assert gateway.complete.call_count == (2 if invalid == "unknown_need" else 1)
    assert all(
        call.args[2] in (ResearchPlan, ResearchStep)
        for call in gateway.complete.call_args_list
    )
    handler.assert_not_called()
    assert instance.context.budget.snapshot()["tools"] == 0


def test_programming_error_is_not_swallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    instance, _handler = acquirer()
    request_plan = plan()

    monkeypatch.setattr(
        instance.registry, "dispatch", Mock(side_effect=ValueError("Programming error"))
    )
    with pytest.raises(ValueError, match="Programming error"):
        instance.acquire(
            [SourceAction(need_ids=["first"], tool="read_original", arguments={})],
            request_plan,
        )
