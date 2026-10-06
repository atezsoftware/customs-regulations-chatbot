"""Real hosted terminal handlers retain selected sources without extra review calls."""

import copy
import hashlib
from dataclasses import replace
from typing import Any
from uuid import uuid4

import pytest
from pydantic import JsonValue

from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import OutcomeStatus, RunContext, SharedBudget
from onyx.asv3.parallel_answers import ParallelAnswerReceipts
from onyx.prompts.asv3.experimental import parallel_metadata_instructions
from tests.unit.onyx.asv3.test_native_cache_projection import actual_originals
from tests.unit.onyx.asv3.test_native_model_adapter import last_payload
from tests.unit.onyx.asv3.test_runtime import response
from tests.unit.onyx.asv3.test_serial_experimental_session_parity import (
    FOCUS,
    INSTRUCTION,
    SCENARIO,
    declaration,
    outer,
    serial_run,
    session,
)

pytestmark = pytest.mark.usefixtures("empty_source_inventory")
OMITTED = "İlk koşul uygulanır [1]."
CORRECTED = "İlk koşul uygulanır [1]. Sonucu sınırlayan diğer koşul da korunur [2]."


@pytest.mark.parametrize("trusted_hosted", [True, False])
def test_real_terminal_retains_selected_second_source_and_seals_exact_corrected_body(
    monkeypatch: pytest.MonkeyPatch, trusted_hosted: bool
) -> None:
    run = serial_run(monkeypatch, "conversation", "Merhaba!")
    # A third delivered source has no selected positive outcome and stays optional.
    third = replace(
        run.broker.sources[1], id=uuid4(), name="Unrelated Source", file_id="unrelated"
    )
    run.broker.sources.append(third)
    run.broker.chunks[str(third.id)] = replace(
        run.broker.chunks[str(run.broker.sources[1].id)],
        id="unrelated",
        source_id=third.id,
        heading_path=("Unrelated Source", "MADDE 299"),
    )
    run.calls.clear()
    run.selected.reset_mock()
    ledger = EvidenceLedger()
    envelope = outer(run)
    child = session(run, envelope, ledger)
    if not trusted_hosted:
        child.context.services["serial_session_diagnostics"] = False

    def invoke(**arguments: Any) -> Any:
        run.calls.append(arguments)
        assert arguments["prompt"][0].content == (
            parallel_metadata_instructions(INSTRUCTION)
            if trusted_hosted
            else INSTRUCTION
        )
        if len(run.calls) == 1:
            return response(
                calls=[
                    ("read_source_range", {"source_id": str(selected.id)})
                    for selected in run.broker.sources
                ]
            )
        originals = actual_originals(arguments["prompt"])
        assert {row["citation"] for row in originals} == {1, 2, 3}
        for row in originals:
            item = ledger.get(row["citation"])
            assert item is not None
            assert row["text"] == item.text
            assert row["source_id"] == item.source_id
            assert row["chunk_id"] == item.chunk_id
            assert row["text_hash"] == item.text_hash
        second = ledger.get(2)
        assert second is not None
        body = OMITTED
        if len(run.calls) == 3:
            assert trusted_hosted
            rejected = child.harness.receipts[-1].outcome
            assert rejected.status == OutcomeStatus.PARTIAL
            assert rejected.data["missing_selected_outcome_sources"] is True
            assert rejected.data["outcome_ids"] == ["local-result"]
            assert rejected.data["condition_ids"] == ["material_condition"]
            assert rejected.data["missing_inline_citations"] == [2]
            assert rejected.data["undelivered_citations"] == []
            assert child.context.services.get("submitted_answer") is None
            assert child.harness.last_draft == OMITTED
            assert (
                last_payload(run.selected)["publication_gap"][
                    "missing_selected_outcome_sources"
                ]
                is True
            )
            body = CORRECTED
        else:
            assert len(run.calls) == 2
        return response(
            calls=[
                (
                    "submit_answer",
                    {
                        "answer": body,
                        "basis": "originals",
                        "_language": "tr",
                        "_outcomes": [declaration()],
                        "_coverage": {
                            "conditions": [
                                {
                                    "condition_id": "material_condition",
                                    "outcome_ids": ["local-result"],
                                    "detail": "A second material source qualifies this result.",
                                    "witnesses": [
                                        {
                                            "citation": 2,
                                            "start_char": 0,
                                            "end_char": len(second.text),
                                        }
                                    ],
                                }
                            ],
                            "resolutions": [
                                {
                                    "outcome_id": "local-result",
                                    "status": "supported",
                                    "condition_ids": ["material_condition"],
                                    "evidence_numbers": [1, 2],
                                }
                            ],
                        },
                    },
                )
            ]
        )

    run.selected.invoke.side_effect = invoke
    result = child.run()
    expected = CORRECTED if trusted_hosted else OMITTED
    assert result.status == OutcomeStatus.FOUND and result.summary == expected
    assert run.selected.invoke.call_count == (3 if trusted_hosted else 2)
    assert run.selected.config.model_name == "gpt-6-luna"
    assert [row.call.name for row in child.harness.receipts[:3]] == [
        "read_source_range"
    ] * 3
    terminal_receipts = child.harness.receipts[3:]
    assert [row.outcome.status for row in terminal_receipts] == (
        [OutcomeStatus.PARTIAL, OutcomeStatus.FOUND]
        if trusted_hosted
        else [OutcomeStatus.FOUND]
    )
    call_id = child.model.last_call_id
    assert call_id is not None and ledger.completely_delivered(call_id) == {1, 2, 3}
    assert child.validate_accepted(expected, call_id, result.status) is None
    state = child.source_state()
    accepted = state["accepted"]
    assert isinstance(accepted, dict)
    assert accepted["answer_hash"] == hashlib.sha256(expected.encode()).hexdigest()
    assert accepted["model_call_id"] == call_id
    assert accepted["status"] == OutcomeStatus.FOUND.value
    if not trusted_hosted:
        return
    root = RunContext(
        run_id=child.context.run_id,
        scope=copy.deepcopy(child.context.scope),
        budget=SharedBudget(unlimited_execution=True),
    )
    assignment: dict[str, JsonValue] = {
        "question_id": "independent-a",
        "question": FOCUS,
        "task_id": "owned-task",
        "parent_question_ids": [1],
    }
    envelope.services["last_model_call_id"] = call_id
    receipts = ParallelAnswerReceipts(root, SCENARIO, user_id="owner")
    receipt_id = receipts.seal(
        envelope,
        assignment=assignment,
        answer=expected,
        status=result.status,
        model_call_id=call_id,
        ledger=ledger,
        validate_body=lambda: child.validate_accepted(expected, call_id, result.status),
        source_state=state,
    )
    assert ledger.export()["pinned_delivery_calls"] == [call_id]
    receipts.verify(
        root,
        receipt_id=receipt_id,
        task_id="owned-task",
        assignment=assignment,
        answer=expected,
        status=result.status,
        ledger=ledger,
        validate_body=lambda: child.validate_accepted(expected, call_id, result.status),
        source_state=state,
    )
    with pytest.raises(ValueError, match="changed"):
        child.validate_accepted(OMITTED, call_id, result.status)
    assert run.selected.invoke.call_count == 3
