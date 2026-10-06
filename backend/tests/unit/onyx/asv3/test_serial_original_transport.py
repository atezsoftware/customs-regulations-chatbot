"""Exercise lossless original transport inside owned ordinary serial sessions."""

import copy
import json
from typing import cast

import pytest
from pydantic import JsonValue

from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.harness import Harness
from onyx.asv3.llm_adapter import ResearchModel
from onyx.asv3.models import (
    CapabilityCall,
    Decision,
    HarnessView,
    OutcomeStatus,
    RunContext,
    ToolOutcome,
    ToolReceipt,
)
from onyx.asv3.native_cache_projection import (
    decode_compact_originals,
    lossless_original_transport_enabled,
)
from onyx.asv3.parallel_answers import ParallelAnswerReceipts
from onyx.asv3.registry import CapabilityRegistry
from onyx.llm.models import ChatCompletionMessage, ToolMessage
from tests.unit.onyx.asv3 import (
    test_native_cache_projection as cache_cases,
)
from tests.unit.onyx.asv3 import (
    test_parallel_compact_native_invocation as invocation_cases,
)
from tests.unit.onyx.asv3 import (
    test_parallel_native_original_transport as acquisition_cases,
)
from tests.unit.onyx.asv3.test_native_cache_projection import payloads
from tests.unit.onyx.asv3.test_native_metadata_projection import setup_original
from tests.unit.onyx.asv3.test_native_model_adapter import model, original, turn, view
from tests.unit.onyx.asv3.test_serial_experimental_session_parity import (
    FOCUS,
    SCENARIO,
    outer,
    serial_run,
    session,
)

pytestmark = pytest.mark.usefixtures("empty_source_inventory")


def hosted(context: RunContext) -> None:
    context.services.update(
        lean_native_mode=True,
        research_profile="experimental",
        experimental_parallel=False,
        serial_session_diagnostics=True,
        serial_original_transport=True,
        task_id="owned-task",
    )


def invocation_originals(
    prompt: list[ChatCompletionMessage], ledger: EvidenceLedger
) -> list[dict[str, JsonValue]]:
    parsed = cast(list[dict[str, JsonValue]], payloads(prompt))
    catalogue = cast(
        list[dict[str, JsonValue]], parsed[-1].get("original_metadata_catalogue", [])
    )
    return decode_compact_originals(parsed, catalogue, ledger)


@pytest.mark.parametrize(
    "change,value",
    [
        ("research_profile", "normal"),
        ("research_profile", "deep"),
        ("experimental_parallel", None),
        ("serial_original_transport", False),
        ("serial_original_transport", "true"),
        ("serial_session_diagnostics", False),
        ("serial_session_diagnostics", 1),
        ("lean_native_mode", False),
        ("task_id", ""),
        ("task_id", "  "),
        ("task_id", 1),
    ],
)
def test_hosted_transport_requires_its_typed_local_scope(
    change: str, value: object
) -> None:
    context = RunContext()
    hosted(context)
    assert lossless_original_transport_enabled(context)
    context.services[change] = value
    assert not lossless_original_transport_enabled(context)


def test_hosted_flag_does_not_enable_optional_worker_or_plain_serial_transport() -> (
    None
):
    local = RunContext(depth=1)
    hosted(local)
    assert not lossless_original_transport_enabled(local)
    normal = RunContext(services={"research_profile": "experimental"})
    assert not lossless_original_transport_enabled(normal)
    normal.services["experimental_parallel"] = True
    assert lossless_original_transport_enabled(normal)


def test_plain_serial_original_prompt_bytes_are_unchanged_by_absent_opt_in() -> None:
    ledger, context, first = setup_original()
    context.services.update(
        research_profile="experimental", experimental_parallel=False
    )
    selected = model(limit=1000000)
    adapter = ResearchModel(selected, context, lean_native_mode=True)
    state = view(turns=[turn("read", [first])], original_evidence=[first])
    baseline, schemas, output = adapter._fit_native_decision(state)
    context.services.update(serial_original_transport=True, task_id="owned-task")
    untrusted, actual_schemas, actual_output = adapter._fit_native_decision(state)
    assert [row.model_dump_json() for row in untrusted] == [
        row.model_dump_json() for row in baseline
    ]
    assert actual_schemas == schemas and actual_output == output
    assert "original_metadata_catalogue" not in payloads(untrusted)[-1]
    assert ledger.export()["deliveries"] == []


def test_long_originals_are_preserved_once_with_exact_provenance_and_raw_history() -> (
    None
):
    ledger, context, first = setup_original()
    second = original(ledger, context, "İstisna VE koşul; 東京.\n  " * 1300)
    records = [first, second]
    receipt = ToolReceipt(
        call=CapabilityCall(call_id="acquire", name="read_provision", arguments={}),
        outcome=ToolOutcome(status=OutcomeStatus.FOUND, summary="Originals acquired"),
        evidence_ids=[1, 2],
        elapsed_seconds=0,
    )

    def no_decision(_: HarnessView) -> Decision:
        raise AssertionError("No model step")

    harness = Harness(
        request=FOCUS,
        context=context,
        registry=CapabilityRegistry(),
        decide=no_decision,
        evidence=ledger,
    )
    context.services.update(
        research_profile="experimental", experimental_parallel=False
    )
    legacy = json.loads(harness._model_tool_result(receipt).content)
    assert [row["citation"] for row in legacy["original_evidence"]] == [1]
    hosted(context)
    acquired = harness._model_tool_result(receipt)
    assert [
        row["text"] for row in json.loads(acquired.content)["original_evidence"]
    ] == [row["text"] for row in records]
    native = [turn("acquire", [])]
    native[0].results = [acquired]
    raw_before = native[0].model_dump_json()
    ledger_before = copy.deepcopy(ledger.export()["records"])
    selected = model(limit=1000000)
    adapter = ResearchModel(selected, context, lean_native_mode=True)
    adapter.decide(view(turns=native, original_evidence=records))
    prompt = selected.invoke.call_args.kwargs["prompt"]
    decoded = invocation_originals(prompt, ledger)
    expected = [
        {**row, "start_char": 0, "end_char": len(cast(str, row["text"]))}
        for row in records
    ]
    assert decoded == expected
    assert len(payloads(prompt)[-1]["original_metadata_catalogue"]) == 2
    assert "original_evidence" not in payloads(prompt)[-1]
    assert ledger.completely_delivered(adapter.last_call_id or "") == {1, 2}
    assert native[0].model_dump_json() == raw_before
    assert ledger.export()["records"] == ledger_before
    assert context.services["experimental_parallel"] is False
    selected.invoke.assert_called_once()


def test_hosted_actual_acquisition_keeps_all_large_originals_in_later_invocation() -> (
    None
):
    ledger = EvidenceLedger()
    context = acquisition_cases.context(parallel=False)
    hosted(context)
    research, adapter, selected = acquisition_cases.harness(context, ledger)
    assert research.run().answer == acquisition_cases.BODY
    first_receipt = research.receipts[0]
    assert len(first_receipt.evidence_ids) == 24
    acquired = acquisition_cases.tool_originals(research.turns[0].results[0])
    assert len(json.dumps(acquired, ensure_ascii=False)) > 20000
    assert (
        len(
            json.loads(
                ledger.serialize_records(first_receipt.evidence_ids, max_chars=20000)
            )
        )
        < 24
    )
    canonical_first = json.loads(
        ledger.serialize_records(first_receipt.evidence_ids, max_chars=None)
    )
    assert acquired == canonical_first
    assert all(row["truncated"] is False for row in acquired)
    early = selected.invoke.call_args_list[1].kwargs["prompt"]
    assert len(invocation_originals(early, ledger)) == 24
    final = selected.invoke.call_args_list[2].kwargs["prompt"]
    canonical_all = json.loads(
        ledger.serialize_records(list(range(1, 26)), max_chars=None)
    )
    expected = [
        {**row, "start_char": 0, "end_char": len(row["text"])} for row in canonical_all
    ]
    assert invocation_originals(final, ledger) == expected
    assert len(payloads(final)[-1]["original_metadata_catalogue"]) == 25
    assert ledger.completely_delivered(adapter.last_call_id or "") == set(range(1, 26))
    assert context.services["experimental_parallel"] is False
    assert selected.invoke.call_count == 3


def test_transport_removes_repeated_identities_without_dropping_original_characters() -> (
    None
):
    ledger, context, first = setup_original()
    context.services.update(
        research_profile="experimental", experimental_parallel=False
    )
    selected = model(limit=1000000)
    adapter = ResearchModel(selected, context, lean_native_mode=True, token_counter=len)
    native = [turn(f"read-{index}", [first]) for index in range(4)]
    state = view(turns=native, original_evidence=[first])
    baseline, schemas, _ = adapter._fit_native_decision(state)
    baseline_cost = adapter._input_cost(baseline, schemas)
    hosted(context)
    compact, actual_schemas, _ = adapter._fit_native_decision(state)
    assert actual_schemas == schemas
    assert compact[0].content == baseline[0].content
    assert invocation_originals(compact, ledger) == [
        {**first, "start_char": 0, "end_char": len(cast(str, first["text"]))}
    ]
    assert adapter._input_cost(compact, schemas) < baseline_cost
    selected.invoke.assert_not_called()


def test_hosted_required_originals_survive_atomic_tool_pair_eviction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cache_cases, "parallel", hosted)
    cache_cases.test_group_eviction_redelivers_required_original_in_tail(monkeypatch)


@pytest.mark.parametrize("recovery", ["empty", "envelope", "truncated"])
def test_hosted_actual_invocation_retains_catalogue_through_recovery(
    monkeypatch: pytest.MonkeyPatch, recovery: str
) -> None:
    monkeypatch.setattr(invocation_cases, "parallel", hosted)
    invocation_cases.test_compact_originals_remain_delivered_in_actual_native_recovery(
        recovery
    )


@pytest.mark.parametrize(
    "mutation", ["removed", "altered", "foreign_catalogue", "pointer_only"]
)
def test_hosted_catalogue_and_physical_delivery_fail_closed(
    monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    monkeypatch.setattr(invocation_cases, "parallel", hosted)
    invocation_cases.test_actual_compact_scan_fails_closed_before_provider_or_ignores_pointer_only(
        mutation
    )


def test_accepted_serial_body_and_sealed_proof_are_exact_with_transport_on_or_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = (
        "Koşul, istisna ve sonraki aşama korunur [1].\n\n"
        + "İĞŞçöü 東京.\n" * 2000
        + "TAM_SON"
    )
    run = serial_run(monkeypatch, "originals", body)
    for enabled in (False, True):
        run.calls.clear()
        ledger = EvidenceLedger()
        envelope = outer(run)
        child = session(run, envelope, ledger)
        assert child.context.services["serial_original_transport"] is True
        child.context.services["serial_original_transport"] = enabled
        result = child.run()
        assert result.summary.encode() == body.encode()
        assert result.status == OutcomeStatus.FOUND
        assert child.context.services["experimental_parallel"] is False
        assert len(run.calls) == 2
        accepted_call = child.model.last_call_id
        assert accepted_call is not None
        assert ledger.completely_delivered(accepted_call) == {1}
        envelope.services["last_model_call_id"] = accepted_call
        assignment: dict[str, JsonValue] = {
            "question_id": "independent-a",
            "question": FOCUS,
            "task_id": "owned-task",
            "parent_question_ids": [1],
        }
        root = RunContext(run_id=envelope.run_id, scope=envelope.scope)
        seals = ParallelAnswerReceipts(root, SCENARIO, user_id="owner")
        receipt_id = seals.seal(
            envelope,
            assignment=assignment,
            answer=body,
            status=result.status,
            model_call_id=accepted_call,
            ledger=ledger,
            validate_body=lambda: child.validate_accepted(
                body, accepted_call, result.status
            ),
            source_state=child.source_state(),
        )
        assert ledger.export()["pinned_delivery_calls"] == [accepted_call]
        seals.verify(
            root,
            receipt_id=receipt_id,
            task_id="owned-task",
            assignment=assignment,
            answer=body,
            status=result.status,
            ledger=ledger,
            validate_body=lambda: child.validate_accepted(
                body, accepted_call, result.status
            ),
            source_state=child.source_state(),
        )
        if enabled:
            actual = invocation_originals(run.calls[-1]["prompt"], ledger)
            item = ledger.get(1)
            assert item is not None
            assert [(row["citation"], row["text"]) for row in actual] == [
                (1, item.text)
            ]
            assert any(isinstance(row, ToolMessage) for row in run.calls[-1]["prompt"])
