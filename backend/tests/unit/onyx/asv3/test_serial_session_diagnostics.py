"""Hosted serial diagnostics expose current IDs and preserve every source guard."""

import copy
from typing import Any, cast

import pytest
from pydantic import JsonValue

from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import CapabilityCall, OutcomeStatus
from onyx.asv3.registry import native_research_bindings
from onyx.asv3.scenario import question_determinations
from onyx.prompts.asv3.experimental import parallel_metadata_instructions
from tests.unit.onyx.asv3.test_legal_source_reviews import (
    deliver,
    navigation,
    review,
    setup_reviews,
)
from tests.unit.onyx.asv3.test_native_model_adapter import last_payload
from tests.unit.onyx.asv3.test_parallel_research_bindings import (
    need_properties,
    parameters,
)
from tests.unit.onyx.asv3.test_runtime import response
from tests.unit.onyx.asv3.test_serial_experimental_session_parity import (
    INSTRUCTION,
    outer,
    serial_run,
    session,
)

pytestmark = pytest.mark.usefixtures("empty_source_inventory")


def test_first_hosted_decision_exposes_exact_local_ids_without_an_extra_model_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = serial_run(monkeypatch, "conversation", "Merhaba!")
    base_prompt = run.calls[0]["prompt"]
    base_schema = need_properties(
        parameters(run.harness.registry.definitions(run.harness.context))
    )
    assert "enum" not in cast(dict[str, Any], base_schema["question_ids"])["items"]
    assert "research_bindings" not in last_payload(run.selected)
    run.selected.reset_mock()
    child = session(run, outer(run), EvidenceLedger())
    assert child.context.services["experimental_parallel"] is False
    assert child.context.depth == 0
    assert "serial_session_diagnostics" not in child.outer_context.services
    spec = child.registry.get("update_research")
    assert spec is not None
    original_schema = copy.deepcopy(spec.parameters)
    determinations = question_determinations(list(child.research.questions))
    valid: dict[str, JsonValue] = {
        "needs": [
            {
                "need_id": "operative_condition",
                "question_ids": ["q0"],
                "determination_ids": [determinations[0]["determination_id"]],
                "purpose": "Read the condition relevant to the supplied transaction.",
                "completion_test": "The operative condition is delivered in full.",
            }
        ]
    }
    run.selected.invoke.side_effect = lambda **_kwargs: response(
        calls=[("update_research", valid)]
    )
    decision = child.model.decide(child.harness.view())
    assert run.selected.invoke.call_count == 1
    actual = run.selected.invoke.call_args.kwargs
    assert base_prompt[0].content == INSTRUCTION
    assert actual["prompt"][0].content == parallel_metadata_instructions(INSTRUCTION)
    bindings = last_payload(run.selected)["research_bindings"]
    assert bindings["question_ids"] == ["q0"]
    assert bindings["determinations"] == [
        {key: item[key] for key in ("determination_id", "question_id", "question")}
        for item in determinations
    ]
    schema = need_properties(parameters(child.registry.definitions(child.context)))
    assert cast(dict[str, Any], schema["question_ids"])["items"]["enum"] == ["q0"]
    assert cast(dict[str, Any], schema["determination_ids"])["items"]["enum"] == [
        item["determination_id"] for item in determinations
    ]
    before = child.research.export()
    invented = copy.deepcopy(valid)
    invented_need = cast(list[dict[str, JsonValue]], invented["needs"])[0]
    invented_need["determination_ids"] = ["procedure"]
    rejected = child.registry.dispatch(
        CapabilityCall(name="update_research", arguments=invented), child.context
    )
    assert rejected.status == OutcomeStatus.INVALID
    assert child.research.export() == before
    accepted = child.registry.dispatch(decision.calls[0], child.context)
    assert accepted.status == OutcomeStatus.FOUND
    assert child.research.question_ids("operative_condition") == ["q0"]
    assert spec.parameters == original_schema


@pytest.mark.parametrize(
    "override",
    [
        {"serial_session_diagnostics": False},
        {"serial_session_diagnostics": "true"},
        {"lean_native_mode": False},
        {"research_profile": "normal"},
        {"research_profile": "deep"},
        {"experimental_parallel": True},
    ],
)
def test_hosted_diagnostics_flag_does_not_enable_other_native_policies(
    monkeypatch: pytest.MonkeyPatch, override: dict[str, object]
) -> None:
    run = serial_run(monkeypatch, "conversation", "Merhaba!")
    child = session(run, outer(run), EvidenceLedger())
    child.context.services.update(override)
    assert native_research_bindings(child.context) is None
    child.context.services.update(
        lean_native_mode=True,
        research_profile="experimental",
        experimental_parallel=False,
        serial_session_diagnostics=True,
    )
    child.context.depth = 1
    assert native_research_bindings(child.context) is None


@pytest.mark.parametrize(
    "failure,code",
    [
        ("wrong_source", "wrong_source"),
        ("undelivered", "not_fully_delivered"),
        ("foreign_lead", "duplicate_or_foreign_lead"),
    ],
)
def test_hosted_terminal_rejects_bad_witnesses_with_precise_existing_diagnostics(
    monkeypatch: pytest.MonkeyPatch, failure: str, code: str
) -> None:
    run = serial_run(monkeypatch, "conversation", "Merhaba!")
    ledger = EvidenceLedger()
    child = session(run, outer(run), ledger)
    _, fixtures, _ = setup_reviews()
    for number in (1, 2, 3):
        item = fixtures.get(number)
        assert item is not None
        ledger.add([item], child.context)
    deliver(ledger, "anchor-call", [1])
    child.reviews.record_delivery("anchor-call", child.context, navigation(), ledger)
    delivered = [1, 3] if failure == "undelivered" else [1, 2, 3]
    deliver(ledger, "terminal-call", delivered)
    child.context.services["last_model_call_id"] = "terminal-call"
    candidate = review()
    if failure == "wrong_source":
        candidate["witnesses"] = [{"citation": 3, "start_char": 0, "end_char": 10}]
    elif failure == "foreign_lead":
        candidate["lead_id"] = "lead_" + "f" * 64
    body = "The source-supported result retains its condition [1]."
    before = child.reviews.export()
    result = child.registry.dispatch(
        CapabilityCall(
            name="submit_partial_answer",
            arguments={"answer": body, "_related_source_reviews": [candidate]},
        ),
        child.context,
    )
    assert result.status == OutcomeStatus.INVALID
    assert result.data["invalid_related_source_review"] is True
    diagnostic = cast(dict[str, JsonValue], result.data["related_source_review_error"])
    assert diagnostic["code"] == code
    assert child.reviews.export() == before
    assert "submitted_answer" not in child.context.services
    if failure != "foreign_lead":
        assert diagnostic["source_id"] == "decision"
        assert diagnostic["available_original_witnesses"] == (
            []
            if failure == "undelivered"
            else [{"citation": 2, "start_char": 0, "end_char": 39}]
        )
    child.context.services.pop("serial_session_diagnostics")
    ordinary = child.registry.dispatch(
        CapabilityCall(
            name="submit_partial_answer",
            arguments={"answer": body, "_related_source_reviews": [candidate]},
        ),
        child.context,
    )
    assert ordinary.status == OutcomeStatus.INVALID
    assert ordinary.summary == result.summary
    assert ordinary.data["detail"] == result.data["detail"]
    assert "related_source_review_error" not in ordinary.data
    assert child.reviews.export() == before
