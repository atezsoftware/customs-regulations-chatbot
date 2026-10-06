"""Owned references select strict source validation without a paid metadata patch."""

import copy
import json
from typing import Any

import pytest
from pydantic import JsonValue

from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import OutcomeStatus
from onyx.asv3.retained_answer import normalize_retained_answer_basis
from tests.unit.onyx.asv3.test_model_adapter import tool_response
from tests.unit.onyx.asv3.test_native_cache_projection import actual_originals, payloads
from tests.unit.onyx.asv3.test_retained_answer import REQUEST, context, reference
from tests.unit.onyx.asv3.test_serial_experimental_session_parity import (
    declaration,
    outer,
    serial_run,
    session,
)
from tests.unit.onyx.asv3.test_shared_originals import original

pytestmark = pytest.mark.usefixtures("empty_source_inventory")


@pytest.mark.parametrize("hosted", [False, True])
def test_owned_source_reference_defaults_only_validation_mode(hosted: bool) -> None:
    ctx = context(hosted=hosted)
    ledger = EvidenceLedger()
    ctx.services["evidence"] = ledger
    ledger.add([original("known", "The operative original.")], ctx)
    body = "The condition applies [1]."
    arguments: dict[str, JsonValue] = {"retained_answer_id": reference(ctx, body)}
    untouched = copy.deepcopy(arguments)
    before = ledger.export()
    assert normalize_retained_answer_basis(
        "submit_answer", arguments, ctx, body, request=REQUEST
    ) == {**arguments, "basis": "originals"}
    assert arguments == untouched and ledger.export() == before
    assert ledger.completely_delivered("any-call") == set()


@pytest.mark.parametrize(
    "invalid",
    [
        "foreign",
        "stale",
        "scope",
        "plain",
        "partial",
        "literal",
        "uncited",
        "unknown",
        "empty_edits",
        "foreign_unit",
        "explicit_basis",
        "no_ledger",
        "uncitable",
    ],
)
def test_default_does_not_salvage_invalid_or_unscoped_references(invalid: str) -> None:
    ctx = context()
    ledger = EvidenceLedger()
    ctx.services["evidence"] = ledger
    ledger.add(
        [original("known", "Exact original.", citable=invalid != "uncitable")], ctx
    )
    body = (
        "Condition [1]."
        if invalid not in {"uncited", "unknown"}
        else "Facts only."
        if invalid == "uncited"
        else "Condition [2]."
    )
    arguments: dict[str, JsonValue] = {"retained_answer_id": reference(ctx, body)}
    name = "submit_answer"
    if invalid == "foreign":
        arguments["retained_answer_id"] = "retained_foreign"
    elif invalid == "stale":
        body += " Changed."
    elif invalid == "scope":
        ctx.scope["tenant"] = "B"
    elif invalid == "plain":
        ctx.services["experimental_parallel"] = False
    elif invalid == "partial":
        name = "submit_partial_answer"
    elif invalid == "literal":
        arguments = {"answer": body}
    elif invalid == "empty_edits":
        arguments["retained_answer_edits"] = []
    elif invalid == "foreign_unit":
        arguments["retained_answer_edits"] = [
            {"unit_id": "another-owner-unit", "replacement": "Condition [1]."}
        ]
    elif invalid == "explicit_basis":
        arguments["basis"] = None
    elif invalid == "no_ledger":
        del ctx.services["evidence"]
    assert (
        normalize_retained_answer_basis(name, arguments, ctx, body, request=REQUEST)
        is arguments
    )


@pytest.mark.parametrize("missing", [False, True])
def test_real_hosted_terminal_without_basis_keeps_actual_delivery_guard(
    monkeypatch: pytest.MonkeyPatch, missing: bool
) -> None:
    run = serial_run(monkeypatch, "conversation", "Merhaba!")
    run.selected.reset_mock()
    ledger = EvidenceLedger()
    child = session(run, outer(run), ledger)
    own = original("own", "The genuine original.")
    ledger.add(
        [
            own,
            original(
                "unseen",
                "Not supplied to this invocation.",
                source="other",
                headings=["Other instrument"],
            ),
        ],
        child.context,
    )
    child.harness.evidence_working_set.remember(1, 0, len(own.text))
    body = f"The condition applies [{2 if missing else 1}]."
    wire = ""

    def invoke(**kwargs: Any) -> Any:
        nonlocal wire
        assert {row["citation"] for row in actual_originals(kwargs["prompt"])} == {1}
        current = payloads(kwargs["prompt"])[-1]
        wire = json.dumps(
            {
                "retained_answer_id": current["retained_answer"]["retained_answer_id"],
                "_language": "tr",
                "_outcomes": [declaration()],
            }
        )
        return tool_response(wire, "submit_answer")

    run.selected.invoke.side_effect = invoke
    current_view = child.harness.view()
    current_view.draft_to_repair = body
    current_view.publication_gap = {"repair": True}
    decision = child.model.decide(current_view)
    assert run.selected.invoke.call_count == 1
    assert decision.calls[0].argument_error is None
    assert decision.calls[0].arguments["basis"] == "originals"
    assert decision.assistant_message and decision.assistant_message.tool_calls
    assert decision.assistant_message.tool_calls[0].function.arguments == wire
    assert "basis" not in json.loads(wire)
    call = child.model.last_call_id or ""
    assert ledger.completely_delivered(call) == {1}
    child._on_decision(decision)
    receipts = child.harness._dispatch(decision.calls)
    if missing:
        assert receipts[0].outcome.status == OutcomeStatus.PARTIAL
        assert receipts[0].outcome.data["undelivered_citations"] == [2]
        assert child.context.services.get("submitted_answer") is None
    else:
        assert receipts[0].outcome.status == OutcomeStatus.FOUND
        assert child.context.services["submitted_answer"] == body
