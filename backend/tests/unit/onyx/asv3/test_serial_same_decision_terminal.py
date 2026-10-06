"""Hosted terminal batches preserve current-call originals and ordered mutations."""

from typing import Any

import pytest
from pydantic import JsonValue

from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import CapabilityCall, OutcomeStatus, ToolOutcome
from tests.unit.onyx.asv3.test_native_cache_projection import actual_originals
from tests.unit.onyx.asv3.test_runtime import response
from tests.unit.onyx.asv3.test_serial_experimental_session_parity import (
    declaration,
    outer,
    serial_run,
    session,
)

pytestmark = pytest.mark.usefixtures("empty_source_inventory")
BODY = "İşleme uygulanacak özgün koşul okunmuştur [1]."


def update(*, valid: bool = True) -> dict[str, JsonValue]:
    return {
        "needs": [
            {
                "need_id": "operative_condition",
                "question_ids": ["q0"],
                "determination_ids": ["q0:d0" if valid else "invented"],
                "purpose": "Read the condition relevant to the supplied transaction.",
                "completion_test": "The operative condition is delivered in full.",
            }
        ]
    }


def answer() -> dict[str, JsonValue]:
    return {
        "answer": BODY,
        "basis": "originals",
        "_language": "tr",
        "_outcomes": [declaration()],
    }


@pytest.mark.parametrize("terminal_first", [False, True])
def test_delivered_original_and_same_decision_update_publish_without_third_call(
    monkeypatch: pytest.MonkeyPatch, terminal_first: bool
) -> None:
    run = serial_run(monkeypatch, "conversation", "Merhaba!")
    run.calls.clear()
    run.selected.reset_mock()
    child = session(run, outer(run), EvidenceLedger())
    publications: list[list[str]] = []
    real_gap = child.publication_gap

    def publication_gap(*args: Any, **kwargs: Any) -> ToolOutcome | None:
        publications.append(child.research.question_ids("operative_condition"))
        return real_gap(*args, **kwargs)

    monkeypatch.setattr(child, "publication_gap", publication_gap)

    def invoke(**arguments: Any) -> Any:
        run.calls.append(arguments)
        if len(run.calls) == 1:
            return response(
                calls=[
                    ("read_source_range", {"source_id": str(run.broker.sources[0].id)})
                ]
            )
        assert len(run.calls) == 2
        originals = actual_originals(arguments["prompt"])
        assert len(originals) == 1
        assert run.broker.chunks[str(run.broker.sources[0].id)].text in str(originals)
        calls = [("update_research", update()), ("submit_answer", answer())]
        if terminal_first:
            calls.reverse()
        return response(calls=calls)

    run.selected.invoke.side_effect = invoke
    result = child.run()
    assert result.status == OutcomeStatus.FOUND
    assert result.summary == BODY
    assert publications and all(ids == ["q0"] for ids in publications)
    assert run.selected.invoke.call_count == 2
    assert run.selected.config.model_name == "gpt-6-luna"
    assert child.ledger.completely_delivered(child.model.last_call_id or "") == {1}
    assert [receipt.call.name for receipt in child.harness.receipts][-2:] == [
        "update_research",
        "submit_answer",
    ]


def test_read_beside_terminal_cannot_create_current_call_delivery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = serial_run(monkeypatch, "conversation", "Merhaba!")
    child = session(run, outer(run), EvidenceLedger())
    run.selected.invoke.side_effect = lambda **_kwargs: response(
        calls=[
            ("read_source_range", {"source_id": str(run.broker.sources[0].id)}),
            ("submit_answer", answer()),
        ]
    )
    decision = child.model.decide(child.harness.view())
    child._on_decision(decision)
    assert "hosted_terminal_batch" not in child.context.services
    receipts = child.harness._dispatch(decision.calls)
    assert receipts[1].outcome.status == OutcomeStatus.DENIED
    assert child.context.services.get("submitted_answer") is None
    # Reading during this decision does not prove the model received the text.
    call = CapabilityCall(name="submit_answer", arguments=answer())
    child._standalone_answer_call = True
    outcome = child.registry.dispatch(call, child.context)
    assert outcome.status == OutcomeStatus.PARTIAL
    assert outcome.data["undelivered_citations"] == [1]


def test_rejected_same_decision_update_blocks_terminal_and_retains_draft(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = serial_run(monkeypatch, "conversation", "Merhaba!")
    child = session(run, outer(run), EvidenceLedger())
    run.selected.invoke.side_effect = lambda **_kwargs: response(
        calls=[("submit_answer", answer()), ("update_research", update(valid=False))]
    )
    decision = child.model.decide(child.harness.view())
    child._on_decision(decision)
    receipts = child.harness._dispatch(decision.calls)
    assert [row.call.call_id for row in receipts] == [
        call.call_id for call in decision.calls
    ]
    assert [row.outcome.status for row in receipts] == [
        OutcomeStatus.PARTIAL,
        OutcomeStatus.INVALID,
    ]
    assert "hosted_terminal_completed" not in child.context.services
    assert child.context.services.get("submitted_answer") is None
    assert child.harness.last_draft == BODY


@pytest.mark.parametrize("failure", ["not_completed", "wrong_id", "stale_model"])
def test_terminal_batch_cannot_bypass_completion_and_exact_call_fences(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    run = serial_run(monkeypatch, "conversation", "Merhaba!")
    child = session(run, outer(run), EvidenceLedger())
    run.selected.invoke.side_effect = lambda **_kwargs: response(
        calls=[("update_research", update()), ("submit_answer", answer())]
    )
    decision = child.model.decide(child.harness.view())
    child._on_decision(decision)
    batch = child.context.services["hosted_terminal_batch"]
    dispatch_context = child.context.child()
    dispatch_context.depth = 0
    if failure != "not_completed":
        dispatch_context.services["hosted_terminal_completed"] = dict(batch)
    dispatch_context.services["hosted_terminal_execution_id"] = (
        "different-terminal" if failure == "wrong_id" else batch["terminal_call_id"]
    )
    if failure == "stale_model":
        dispatch_context.services["hosted_terminal_completed"]["model_call_id"] = "old"
    outcome = child.registry.dispatch(decision.calls[1], dispatch_context)
    assert outcome.status == OutcomeStatus.DENIED
    assert child.context.services.get("submitted_answer") is None


@pytest.mark.parametrize(
    "override",
    [
        {"serial_session_diagnostics": False},
        {"serial_session_diagnostics": "true"},
        {"lean_native_mode": False},
        {"research_profile": "deep"},
        {"experimental_parallel": True},
        {"task_id": ""},
    ],
)
def test_protected_serial_and_non_hosted_modes_keep_standalone_rule(
    monkeypatch: pytest.MonkeyPatch, override: dict[str, object]
) -> None:
    run = serial_run(monkeypatch, "conversation", "Merhaba!")
    child = session(run, outer(run), EvidenceLedger())
    run.selected.invoke.side_effect = lambda **_kwargs: response(
        calls=[("update_research", update()), ("submit_answer", answer())]
    )
    decision = child.model.decide(child.harness.view())
    child.context.services.update(override)
    child._on_decision(decision)
    assert "hosted_terminal_batch" not in child.context.services
    spec = child.registry.get("submit_answer")
    assert spec is not None
    outcome = spec.handler(answer(), child.context)
    assert outcome.status == OutcomeStatus.DENIED
