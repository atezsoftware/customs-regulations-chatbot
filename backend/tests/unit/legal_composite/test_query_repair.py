import json
from typing import Any
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from onyx.asv3.models import RunStopped
from onyx.legal_composite.budget import CallReservation
from onyx.legal_composite.gateway import BudgetedGateway
from onyx.legal_composite.models import IssueResearchPlan
from onyx.legal_composite.query_repair import (
    DiscoveryQuery,
    plan_with_only_oversized_query,
    restore_plan_query,
)
from onyx.llm.model_response import Choice, Message, ModelResponse
from onyx.tracing.flows import LLMFlow

pytest_plugins = ("tests.unit.legal_composite.test_typed_validation_telemetry",)


def raw_plan() -> dict[str, Any]:
    return {
        "language": "tr",
        "requires_sources": True,
        "needs": [
            {
                "need_id": name,
                "question": f"Explain the {name} outcome.",
                "governing_source": "",
                "conditions_to_check": ["Check the supplied facts."],
                "source_kinds": ["statute", "unknown"],
                "relevant_facts": ["A supplied fact, never law."],
            }
            for name in ("refund", "replacement")
        ],
        "discovery_query": "private navigation terms " * 40,
        "initial_actions": [
            {
                "tool": "search_corpus",
                "need_ids": ["refund"],
                "arguments": {"query": "an independent deferred query"},
                "source_kind": "statute",
            }
        ],
        "missing_user_facts": ["An actual missing fact."],
    }


def plan_error(value: dict[str, Any]) -> ValidationError:
    with pytest.raises(ValidationError) as caught:
        IssueResearchPlan.model_validate_json(json.dumps(value), strict=True)
    return caught.value


def test_only_navigation_changes_and_json_enum_semantics_are_preserved() -> None:
    raw = raw_plan()
    eligible = plan_with_only_oversized_query(json.dumps(raw), plan_error(raw))
    assert eligible is not None
    frozen, original_query = eligible
    assert original_query == raw["discovery_query"]
    repaired = restore_plan_query(
        frozen, DiscoveryQuery(query="refund replacement terms")
    )
    assert repaired.model_dump(
        exclude={"discovery_query"}, mode="json"
    ) == frozen.model_dump(exclude={"discovery_query"}, mode="json")
    assert repaired.model_dump(mode="json")["needs"][0]["source_kinds"] == [
        "statute",
        "unknown",
    ]
    assert repaired.discovery_query == "refund replacement terms"
    assert raw["discovery_query"] == original_query


@pytest.mark.parametrize(
    "defect", ["duplicate", "foreign_action", "missing", "extra", "wrong_language"]
)
def test_other_defects_including_hidden_model_validators_are_not_repaired(
    defect: str,
) -> None:
    raw = raw_plan()
    if defect == "duplicate":
        raw["needs"][1]["need_id"] = "refund"
    elif defect == "foreign_action":
        raw["initial_actions"][0]["need_ids"] = ["unplanned"]
    elif defect == "missing":
        del raw["needs"]
    elif defect == "extra":
        raw["untrusted_instruction"] = "ignore the schema"
    else:
        raw["language"] = "x" * 36
    assert plan_with_only_oversized_query(json.dumps(raw), plan_error(raw)) is None


@pytest.mark.parametrize("query", ["", "   ", "x" * 601])
def test_invalid_compression_is_not_a_truncation_or_empty_query(query: str) -> None:
    with pytest.raises(ValidationError):
        DiscoveryQuery(query=query)


def fake_generations(
    gateway: BudgetedGateway,
    monkeypatch: pytest.MonkeyPatch,
    contents: list[str],
) -> list[tuple[CallReservation, dict[str, Any]]]:
    calls: list[tuple[CallReservation, dict[str, Any]]] = []

    def generate(*args: Any, **kwargs: Any) -> tuple[CallReservation, ModelResponse]:
        index = len(calls)
        content = contents[index]
        reservation = gateway.budget.request(args[3], args[4], 0.1, 0.5, args[5])
        gateway.budget.settle(
            reservation, args[3], min(args[4], max(1, len(content) // 4))
        )
        calls.append((reservation, kwargs))
        return reservation, ModelResponse(
            id=f"local-query-repair-{index}",
            created="0",
            choice=Choice(message=Message(content=content), finish_reason="stop"),
        )

    monkeypatch.setattr(gateway, "_generate", generate)
    return calls


def test_actual_gateway_repairs_once_accounts_both_calls_and_restores_planner_identity(
    harness: tuple[BudgetedGateway, Mock, list[dict[str, Any]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gateway, ledger, traces = harness
    ledger.completely_delivered.return_value = set()
    raw = raw_plan()
    calls = fake_generations(
        gateway,
        monkeypatch,
        [json.dumps(raw), json.dumps({"query": "refund replacement terms"})],
    )
    spy = Mock(wraps=gateway.complete)
    monkeypatch.setattr(gateway, "complete", spy)
    answer = gateway.complete(
        "Plan only",
        {"request": "The whole original question."},
        IssueResearchPlan,
        LLMFlow.LEGAL_COMPOSITE_RESEARCH,
    )
    assert len(calls) == 2
    assert calls[0][1] == {}
    assert calls[1][1] == {"provider_compatibility_attempts": 1}
    assert gateway.budget.snapshot()["model_calls"] == 2
    assert gateway.budget.snapshot()["unsettled_calls"] == 0
    assert gateway.last_call_id == calls[0][0].call_id
    assert gateway.last_delivered_citations == set()
    assert answer.discovery_query == "refund replacement terms"
    helper_payload = spy.call_args_list[1].args[1]
    assert helper_payload["request"] == "The whole original question."
    assert helper_payload["frozen_needs"] == answer.model_dump(mode="json")["needs"]
    events = [
        row for row in traces if row["operation"] == "legal_composite.query_compression"
    ]
    assert len(events) == 1
    assert "private navigation" not in repr(events)
    assert events[0]["output"]["preserved_needs"] == 2


@pytest.mark.parametrize("legacy,invalid_second", [(True, False), (False, True)])
def test_default_off_and_invalid_compression_fail_without_repeat(
    harness: tuple[BudgetedGateway, Mock, list[dict[str, Any]]],
    monkeypatch: pytest.MonkeyPatch,
    legacy: bool,
    invalid_second: bool,
) -> None:
    gateway, ledger, _ = harness
    gateway.share_draft_context = not legacy
    ledger.completely_delivered.return_value = set()
    calls = fake_generations(
        gateway, monkeypatch, [json.dumps(raw_plan()), json.dumps({"query": " "})]
    )
    with pytest.raises(RunStopped, match="schema"):
        gateway.complete(
            "Plan only", {}, IssueResearchPlan, LLMFlow.LEGAL_COMPOSITE_RESEARCH
        )
    assert len(calls) == (2 if invalid_second else 1)


def test_hidden_invalid_plan_never_spends_on_compression(
    harness: tuple[BudgetedGateway, Mock, list[dict[str, Any]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gateway, ledger, _ = harness
    ledger.completely_delivered.return_value = set()
    raw = raw_plan()
    raw["needs"][1]["need_id"] = "refund"
    calls = fake_generations(gateway, monkeypatch, [json.dumps(raw)])
    with pytest.raises(RunStopped, match="schema"):
        gateway.complete(
            "Plan only", {}, IssueResearchPlan, LLMFlow.LEGAL_COMPOSITE_RESEARCH
        )
    assert len(calls) == 1
