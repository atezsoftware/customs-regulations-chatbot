import json
from collections.abc import Callable
from contextlib import closing
from typing import cast
from unittest.mock import MagicMock

import pytest
from pydantic import JsonValue

from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.harness import Harness
from onyx.asv3.llm_adapter import ResearchModel
from onyx.asv3.models import (
    EvidenceItem,
    OriginalEvidenceRead,
    OutcomeStatus,
    RunContext,
    SharedBudget,
    ToolOutcome,
    ToolSpec,
)
from onyx.asv3.parallel_checkpoint import compact_parallel_checkpoint
from onyx.asv3.registry import CapabilityRegistry
from onyx.asv3.workers import WorkerPool
from onyx.db.asv3_runs import decode_asv3_checkpoint, encode_asv3_checkpoint
from onyx.llm.model_response import Choice, Message, ModelResponse
from onyx.llm.models import ChatCompletionMessage, ToolMessage
from tests.unit.onyx.asv3.test_native_cache_projection import actual_originals, payloads
from tests.unit.onyx.asv3.test_native_model_adapter import model, native_response

REQUEST = "Apply the conditions and subsequent steps to this fixed scenario."
BODY = "All supported conditions and exceptions remain in this answer [1].\n" * 45


def originals(*, large_provenance: bool = False) -> list[EvidenceItem]:
    items = [
        EvidenceItem(
            source_id="canonical-source",
            chunk_id=f"operative-{index}",
            text=f"Clause {index}: authorization AND the stated proof are required.\n",
            metadata={
                "heading_path": ["Official instrument", "Applicable scope " * 80],
                "source_sha256": "a" * 64,
                "publication_revision": 4,
                "read_as_of_date": "2026-10-06",
                "version_unknown": False,
            },
        )
        for index in range(24)
    ]
    if large_provenance:
        items[0].metadata["additional_context"] = "Immutable provenance. " * 110000
    return items


def context(*, profile: str = "experimental", parallel: bool = True) -> RunContext:
    return RunContext(
        scope={"captured_authorized_documents": [15]},
        budget=SharedBudget(unlimited_execution=True),
        timeout_seconds=float("inf"),
        services={
            "lean_native_mode": True,
            "research_profile": profile,
            "experimental_parallel": parallel,
            "scenario_request": REQUEST,
        },
    )


def answer() -> ModelResponse:
    return ModelResponse(
        id="complete", created="0", choice=Choice(message=Message(content=BODY))
    )


def harness(
    captured: RunContext,
    ledger: EvidenceLedger,
    *,
    request: str = REQUEST,
    supplied: list[EvidenceItem] | None = None,
    later: bool = True,
    navigation: str = "",
) -> tuple[Harness, ResearchModel, MagicMock]:
    first = supplied if supplied is not None else originals()
    second = EvidenceItem(
        source_id="separate-source",
        chunk_id="later-stage",
        text="The later stage requires a separate source-supported application.\n",
    )
    registry = CapabilityRegistry(
        [
            ToolSpec(
                name="acquire_originals",
                description="Read the specified original acquisition.",
                parameters={
                    "type": "object",
                    "properties": {"batch": {"type": "integer", "enum": [0, 1]}},
                    "required": ["batch"],
                    "additionalProperties": False,
                },
                handler=lambda args, _context: ToolOutcome(
                    status=OutcomeStatus.FOUND,
                    summary="Acquired originals",
                    evidence=first if args["batch"] == 0 else [second],
                    data={"navigation": navigation if args["batch"] == 0 else ""},
                ),
            )
        ]
    )
    selected = model(limit=1000000)
    selected.invoke.side_effect = [
        native_response("acquire_originals", '{"batch":0}', call_id="first-read"),
        *(
            [native_response("acquire_originals", '{"batch":1}', call_id="later-read")]
            if later
            else []
        ),
        answer(),
    ]
    adapter = ResearchModel(selected, captured, lean_native_mode=True)
    return (
        Harness(
            request=request,
            context=captured,
            registry=registry,
            decide=adapter.decide,
            evidence=ledger,
        ),
        adapter,
        selected,
    )


def tool_originals(message: ToolMessage) -> list[dict[str, JsonValue]]:
    return cast(
        list[dict[str, JsonValue]], json.loads(message.content)["original_evidence"]
    )


@pytest.mark.parametrize(
    "profile,parallel,complete",
    [
        ("experimental", True, True),
        ("experimental", False, False),
        ("normal", True, False),
        ("normal", False, False),
        ("deep", True, False),
        ("deep", False, False),
    ],
)
def test_real_acquisition_only_parallel_preserves_all_initial_originals(
    profile: str, parallel: bool, complete: bool
) -> None:
    ledger, captured = EvidenceLedger(), context(profile=profile, parallel=parallel)
    research, _, selected = harness(captured, ledger, later=False)
    assert research.run().answer == BODY
    receipt = research.receipts[0]
    message = research.turns[0].results[0]
    legacy = ledger.serialize_records(receipt.evidence_ids, max_chars=20000)
    assert len(json.loads(legacy)) < len(receipt.evidence_ids) == 24
    recorded = tool_originals(message)
    if complete:
        assert len(json.dumps(recorded, ensure_ascii=False)) > 20000
        assert [row["citation"] for row in recorded] == receipt.evidence_ids
        assert all(row["truncated"] is False for row in recorded)
        assert [row["text"] for row in recorded] == [item.text for item in originals()]
    else:
        assert json.dumps(recorded, ensure_ascii=False) == legacy
    assert selected.invoke.call_count == 2
    assert len(ledger.citation_numbers()) == 24


def test_real_native_acquisition_has_stable_prefix_and_exact_actual_delivery() -> None:
    ledger, captured = EvidenceLedger(), context()
    research, adapter, selected = harness(captured, ledger)
    assert research.run().answer == BODY
    early: list[ChatCompletionMessage] = selected.invoke.call_args_list[1].kwargs[
        "prompt"
    ]
    final: list[ChatCompletionMessage] = selected.invoke.call_args_list[2].kwargs[
        "prompt"
    ]
    assert [message.model_dump_json() for message in early[:-1]] == [
        message.model_dump_json() for message in final[: len(early) - 1]
    ]
    assert (
        selected.invoke.call_args_list[1].kwargs["tools"]
        == selected.invoke.call_args_list[2].kwargs["tools"]
    )
    assert "original_evidence" not in payloads(early)[-1]
    assert "original_evidence" not in payloads(final)[-1]
    actual = actual_originals(final)
    assert len(actual) == 25
    assert [row["citation"] for row in actual] == list(range(1, 26))
    assert all(
        (item := ledger.get(row["citation"])) is not None
        and (row["source_id"], row["chunk_id"], row["text_hash"], row["text"])
        == (item.source_id, item.chunk_id, item.text_hash, item.text)
        for row in actual
    )
    assert ledger.completely_delivered(adapter.last_call_id or "") == set(range(1, 26))
    assert selected.invoke.call_count == 3
    assert BODY == research.last_draft


def test_explicit_partial_original_read_keeps_only_its_acquired_range() -> None:
    ledger, captured = EvidenceLedger(), context()
    item = originals()[0]
    ledger.add([item], captured)
    registry = CapabilityRegistry(
        [
            ToolSpec(
                name="read_selected_range",
                description="Read the requested range only.",
                parameters={"type": "object", "additionalProperties": False},
                handler=lambda _args, _context: ToolOutcome(
                    status=OutcomeStatus.PARTIAL,
                    summary="Only the specified partial range was acquired",
                    original_reads=[
                        OriginalEvidenceRead(
                            citation=1,
                            text_hash=item.text_hash,
                            start_char=10,
                            end_char=30,
                        )
                    ],
                ),
            )
        ]
    )
    selected = model()
    selected.invoke.side_effect = [native_response("read_selected_range"), answer()]
    adapter = ResearchModel(selected, captured, lean_native_mode=True)
    research = Harness(
        request=REQUEST,
        context=captured,
        registry=registry,
        decide=adapter.decide,
        evidence=ledger,
    )
    research.run()
    assert research.receipts[0].outcome.status == OutcomeStatus.PARTIAL
    acquired = tool_originals(research.turns[0].results[0])[0]
    assert (acquired["start_char"], acquired["end_char"], acquired["text"]) == (
        10,
        30,
        item.text[10:30],
    )
    actual = actual_originals(selected.invoke.call_args.kwargs["prompt"])
    assert [(row["start_char"], row["end_char"], row["text"]) for row in actual] == [
        (10, 30, item.text[10:30])
    ]
    assert ledger.completely_delivered(adapter.last_call_id or "") == set()
    assert ledger.get(1) == item


def test_physical_history_eviction_redelivers_every_required_original(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger, captured = EvidenceLedger(), context()
    research, adapter, selected = harness(
        captured, ledger, navigation="OLD_NAVIGATION" * 4000
    )
    research.run()
    accepted_call = adapter.last_call_id or ""
    ledger.pin_delivery(accepted_call)
    current = research.view().model_copy(
        update={"required_evidence_numbers": list(range(1, 26))}
    )
    adapter.token_counter = len
    monkeypatch.setattr(
        adapter, "_research_instruction", lambda: "Use the exact originals."
    )
    retained = current.model_copy(update={"turns": research.turns[-1:]})
    fitting_prompt, tools, _ = adapter._fit_native_decision(retained)
    ceiling = adapter._input_cost(fitting_prompt, tools) + 5
    monkeypatch.setattr(adapter, "_limits", lambda _: (ceiling, 100))
    selected.invoke.side_effect = None
    selected.invoke.return_value = answer()
    adapter.decide(current)
    prompt = selected.invoke.call_args.kwargs["prompt"]
    assert [
        message.tool_call_id for message in prompt if isinstance(message, ToolMessage)
    ] == ["later-read"]
    assert len(payloads(prompt)[-1]["original_evidence"]) == 24
    assert {row["citation"] for row in actual_originals(prompt)} == set(range(1, 26))
    assert ledger.completely_delivered(adapter.last_call_id or "") == set(range(1, 26))
    assert ledger.completely_delivered(accepted_call) == set(range(1, 26))
    assert ledger.export()["pinned_delivery_calls"] == [accepted_call]
    assert adapter._input_cost(prompt, tools) <= ceiling


def test_owned_root_native_transport_checkpoints_survive_compact_storage_and_resume() -> (
    None
):
    root, ledger = context(), EvidenceLedger()
    supplied = originals(large_provenance=True)

    def runner(
        task: str, child: RunContext, _updates: Callable[[], list[str]]
    ) -> ToolOutcome:
        transport_context = RunContext(
            run_id=child.run_id,
            scope=child.scope,
            services=dict(child.services),
            budget=child.budget,
            deadline=child.deadline,
            research_deadline=child.research_deadline,
            cancelled=child.is_cancelled,
        )
        research, adapter, _ = harness(
            transport_context, ledger, request=task, supplied=supplied, later=False
        )
        result = research.run()
        ledger.pin_delivery(adapter.last_call_id or "")
        callback = cast(
            Callable[[dict[str, JsonValue]], None],
            child.services["record_child_checkpoint"],
        )
        callback(research.snapshot())
        return ToolOutcome(status=result.status, summary=result.answer or "")

    with closing(WorkerPool(root, runner)) as pool:
        for index in range(3):
            task_id = pool.spawn(
                f"Assigned independent question {index}",
                independent_question=True,
                assignment_id=f"q{index}",
                outcome_ids=[f"outcome{index}"],
            )
            pool.wait_until_all([task_id])
        workers = pool.export()
    snapshot: dict[str, JsonValue] = {
        "version": 1,
        "run_id": root.run_id,
        "sequence": 2,
        "research_profile": "experimental",
        "parallel_research": True,
        "evidence": ledger.export(),
        "workers": workers,
        "last_draft": BODY,
    }
    raw = json.dumps(snapshot, ensure_ascii=False)
    assert len(raw.encode()) > 8_000_000
    assert (
        len(
            json.dumps(
                compact_parallel_checkpoint(snapshot), ensure_ascii=False
            ).encode()
        )
        < 8_000_000
    )
    restored = decode_asv3_checkpoint(encode_asv3_checkpoint(snapshot))
    assert restored == snapshot
    assert json.dumps(restored, ensure_ascii=False) == raw
    with closing(WorkerPool(root, runner)) as pool:
        pool.restore(cast(dict[str, JsonValue], restored["workers"]))
        for task in pool.results():
            assert task.outcome is not None and task.outcome.summary == BODY
            saved = pool.checkpoint(task.task_id)
            assert saved is not None
            resumed_context = context()
            resumed_context.run_id = root.run_id
            resumed_ledger = EvidenceLedger()
            resumed_llm = model(limit=1000000)
            resumed_llm.invoke.return_value = answer()
            adapter = ResearchModel(resumed_llm, resumed_context, lean_native_mode=True)
            resumed = Harness(
                request=task.task,
                context=resumed_context,
                registry=CapabilityRegistry(),
                decide=adapter.decide,
                evidence=resumed_ledger,
            )
            resumed.restore(saved)
            assert len(tool_originals(resumed.turns[0].results[0])) == 24
            assert resumed.run().answer == BODY
            assert resumed_ledger.completely_delivered(
                adapter.last_call_id or ""
            ) == set(range(1, 25))
            assert (
                resumed_ledger.export()["pinned_delivery_calls"]
                == cast(dict[str, JsonValue], saved["evidence"])[
                    "pinned_delivery_calls"
                ]
            )
            assert resumed_llm.invoke.call_count == 1


@pytest.mark.parametrize("status", [OutcomeStatus.DENIED, OutcomeStatus.UNAVAILABLE])
def test_unbounded_transport_preserves_actual_source_failure(
    status: OutcomeStatus,
) -> None:
    ledger, captured = EvidenceLedger(), context()
    failure = ToolOutcome(
        status=status,
        summary="The original cannot be acquired in this scope",
        data={"source_gap": True},
    )
    registry = CapabilityRegistry(
        [
            ToolSpec(
                name="read_unavailable_original",
                description="Acquire the applicable original.",
                parameters={"type": "object", "additionalProperties": False},
                handler=lambda _args, _context: failure,
            )
        ]
    )
    selected = model()
    selected.invoke.side_effect = [
        native_response("read_unavailable_original"),
        ModelResponse(
            id="partial",
            created="0",
            choice=Choice(
                message=Message(content="The applicable original remains unavailable.")
            ),
        ),
    ]
    adapter = ResearchModel(selected, captured, lean_native_mode=True)
    research = Harness(
        request=REQUEST,
        context=captured,
        registry=registry,
        decide=adapter.decide,
        evidence=ledger,
    )
    research.run()
    result = json.loads(research.turns[0].results[0].content)
    assert result["outcome"]["status"] == status.value
    assert result["outcome"]["summary"] == failure.summary
    assert result["outcome"]["data"] == {"source_gap": True}
    assert result["original_evidence"] == []
    assert ledger.citation_numbers() == ()
    assert ledger.completely_delivered(adapter.last_call_id or "") == set()
