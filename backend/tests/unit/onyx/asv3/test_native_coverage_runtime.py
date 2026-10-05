"""Native coverage metadata survives parallel dispatch and answer-model handoff."""

import json
import threading

import pytest
from pydantic import JsonValue

from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.harness import Harness
from onyx.asv3.llm_adapter import ResearchModel
from onyx.asv3.models import (
    CapabilityCall,
    Decision,
    OutcomeStatus,
    RunContext,
    ToolOutcome,
    ToolSpec,
)
from onyx.asv3.outcome_map import OutcomeMap, OutcomeUpdate
from onyx.asv3.registry import CapabilityRegistry
from tests.unit.onyx.asv3.test_native_model_adapter import (
    adaptive_tool_view,
    last_payload,
    model,
    native_action,
    original,
)


def test_parallel_dispatch_cannot_reapply_earlier_outcome_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, ledger = RunContext(), EvidenceLedger()
    outcomes = OutcomeMap(["Requested result?"], context)
    context.services.update(evidence=ledger, outcome_map=outcomes)
    registry = CapabilityRegistry(
        [
            ToolSpec(
                name=name,
                description="Independent local operation",
                parameters={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
                handler=lambda _args, _context: ToolOutcome(
                    status=OutcomeStatus.FOUND, summary="Done"
                ),
            )
            for name in ("earlier", "later")
        ]
    )
    completed = threading.Event()
    dispatch = registry.dispatch

    def reverse_execution(call: CapabilityCall, child: RunContext) -> ToolOutcome:
        if call.name == "earlier":
            assert completed.wait(timeout=2)
        result = dispatch(call, child)
        if call.name == "later":
            completed.set()
        return result

    monkeypatch.setattr(registry, "dispatch", reverse_execution)
    calls = [
        CapabilityCall(
            name=name,
            arguments={
                "_outcomes": [
                    {
                        "outcome_id": "requested",
                        "question_ids": ["q0"],
                        "detail": detail,
                    }
                ]
            },
        )
        for name, detail in (
            ("earlier", "Initial interpretation"),
            ("later", "Clarified interpretation"),
        )
    ]
    harness = Harness(
        request="Requested result?",
        context=context,
        registry=registry,
        evidence=ledger,
        decide=lambda _view: Decision(answer="Unused"),
    )
    receipts = harness._dispatch(calls)
    assert all(receipt.outcome.status == OutcomeStatus.FOUND for receipt in receipts)
    assert "Clarified interpretation" in json.dumps(outcomes.view())
    assert "Initial interpretation" not in json.dumps(outcomes.view())


def test_candidate_source_conditions_reach_selected_answer_model() -> None:
    selected, cheap = model(), model()
    context = RunContext(
        depth=1,
        services={
            "independent_question": True,
            "task_outcome_ids": ["release"],
            "lean_native_mode": True,
        },
    )
    ledger = EvidenceLedger()
    record = original(ledger, context, "Permission requires a separate approval.")
    outcomes = OutcomeMap(["Release?"], context)
    outcomes.update(
        OutcomeUpdate.model_validate(
            {
                "outcomes": [
                    {
                        "outcome_id": "release",
                        "question_ids": ["q0"],
                        "detail": "Release effect",
                    }
                ]
            }
        ),
        ledger,
    )
    context.services.update(evidence=ledger, outcome_map=outcomes)
    registry = CapabilityRegistry(
        [
            ToolSpec(
                name="submit_answer",
                description="Finish with originals",
                parameters={
                    "type": "object",
                    "properties": {
                        "answer": {"type": "string"},
                        "basis": {"type": "string", "enum": ["originals"]},
                    },
                    "required": ["answer", "basis"],
                    "additionalProperties": False,
                },
                handler=lambda _args, _context: ToolOutcome(
                    status=OutcomeStatus.FOUND, summary="Retained"
                ),
            )
        ]
    )
    context.services["registry"] = registry
    condition_detail = "The authority decides the release separately."
    candidate: dict[str, JsonValue] = {
        "answer": "Release requires a separate approval [1].",
        "basis": "originals",
        "_coverage": {
            "conditions": [
                {
                    "condition_id": "approval",
                    "outcome_ids": ["release"],
                    "detail": condition_detail,
                    "witnesses": [
                        {
                            "citation": 1,
                            "start_char": 0,
                            "end_char": len(str(record["text"])),
                        }
                    ],
                }
            ],
        },
    }
    cheap.invoke.return_value = native_action("submit_answer", candidate)
    selected.invoke.return_value = native_action(
        "submit_answer",
        {"answer": "Release requires a separate approval [1].", "basis": "originals"},
    )
    adapter = ResearchModel(
        selected, context, research_llm=cheap, lean_native_mode=True
    )
    current = adaptive_tool_view(original_evidence=[record])
    current.tools.extend(registry.definitions(context))
    decision = adapter.decide(current)
    assert [call.name for call in decision.calls] == ["submit_answer"]
    assert cheap.invoke.call_count == selected.invoke.call_count == 1
    assert condition_detail in json.dumps(last_payload(selected))
    assert outcomes.view()["resolutions"] == []


def test_selected_assembly_returns_source_work_to_lite_without_extra_review() -> None:
    selected, cheap = model(), model()
    ledger = EvidenceLedger()
    context = RunContext(
        services={
            "evidence": ledger,
            "independent_question_mode": True,
            "question_research_started": True,
            "independent_answers": [
                {
                    "question_id": "q0",
                    "answer": "Complete original-supported body [1].",
                    "evidence_numbers": [1],
                },
            ],
        }
    )
    record = original(ledger, context, "The governing rule has a further condition.")
    selected.invoke.side_effect = [
        native_action("read_provision", {"source_id": "law", "article": "7"}),
        native_action("assemble_answers", {"order": ["q0"]}),
    ]
    cheap.invoke.side_effect = [
        native_action("read_provision", {"source_id": "law", "article": "8"}),
        native_action("assemble_answers", {"order": ["q0"]}),
    ]
    adapter = ResearchModel(
        selected, context, research_llm=cheap, lean_native_mode=True
    )
    current = adaptive_tool_view(original_evidence=[record])
    assert adapter.decide(current).calls[0].name == "read_provision"
    assert selected.invoke.call_count == 1 and cheap.invoke.call_count == 0
    assert adapter.decide(current).calls[0].name == "read_provision"
    assert selected.invoke.call_count == cheap.invoke.call_count == 1
    assert adapter.decide(current).calls[0].name == "assemble_answers"
    assert selected.invoke.call_count == cheap.invoke.call_count == 2
