"""A real hosted serial decision repairs metadata without retransmitting its body."""

import copy
import json
from dataclasses import replace
from typing import Any, cast

import pytest
from pydantic import JsonValue

from onyx.asv3.corpus_tools import CorpusBroker, build_corpus_specs
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.llm_adapter import ResearchModel
from onyx.asv3.models import (
    OutcomeStatus,
    RunContext,
    SharedBudget,
    ToolOutcome,
    ToolSpec,
)
from onyx.asv3.parallel_answers import ParallelAnswerReceipts
from onyx.asv3.sandbox import build_sandbox_specs
from onyx.asv3.serial_experimental_session import SerialExperimentalSession
from onyx.asv3.source_tools import build_source_specs
from onyx.asv3.supplemental_tools import build_supplemental_specs
from onyx.llm.model_response import ModelResponse
from onyx.llm.models import AssistantMessage, ReasoningEffort, ToolMessage
from tests.unit.onyx.asv3.test_native_cache_projection import payloads
from tests.unit.onyx.asv3.test_runtime import response, setup_run
from tests.unit.onyx.asv3.test_serial_experimental_session_parity import (
    FOCUS,
    SCENARIO,
    declaration,
    tool_definitions,
)
from tests.unit.onyx.asv3.test_serial_original_transport import invocation_originals

pytestmark = pytest.mark.usefixtures("empty_source_inventory")


def test_rejected_native_body_rehydrates_shared_original_not_in_owner_ranges() -> None:
    from onyx.asv3.harness import Harness
    from onyx.asv3.models import CapabilityCall, ToolReceipt
    from onyx.asv3.registry import CapabilityRegistry
    from tests.unit.onyx.asv3.test_native_model_adapter import model
    from tests.unit.onyx.asv3.test_shared_originals import original

    ledger = EvidenceLedger()
    context = RunContext(
        run_id="shared-ledger-repair",
        scope={"tenant_id": "authorized-owner"},
        services={
            "research_profile": "experimental",
            "experimental_parallel": True,
            "lean_native_mode": True,
            "evidence": ledger,
            "scenario_request": "Explain the applicable rule and condition.",
        },
    )
    first = original("own-read", "The first applicable provision.")
    shared = original(
        "other-owner-read",
        "A separately acquired material condition must be retained.",
        source="another-authorized-instrument",
    )
    assert ledger.add([first, shared], context) == [1, 2]
    selected = model()
    adapter = ResearchModel(selected, context, token_counter=len, lean_native_mode=True)
    harness = Harness(
        request="Explain the applicable rule and condition.",
        context=context,
        registry=CapabilityRegistry([]),
        decide=adapter.decide,
        evidence=ledger,
    )
    harness.evidence_working_set.remember(1, 0, len(first.text))
    owner_ranges = harness.evidence_working_set.export()
    adapter.decide(harness.view())
    earlier_call = adapter.last_call_id or ""
    assert ledger.completely_delivered(earlier_call) == {1}
    assert {
        row["citation"]
        for row in invocation_originals(
            selected.invoke.call_args.kwargs["prompt"], ledger
        )
    } == {1}

    body = "The rule applies [1], subject to the material condition [2]."
    harness._commit_receipt(
        ToolReceipt(
            elapsed_seconds=0,
            call=CapabilityCall(
                name="submit_answer",
                call_id="rejected-native-terminal",
                arguments={"answer": body, "basis": "originals"},
            ),
            outcome=ToolOutcome(
                status=OutcomeStatus.INVALID,
                summary="The second citation was not delivered in this model call.",
                data={"undelivered_citations": [2]},
            ),
        )
    )
    repair = harness.view()
    assert harness.last_draft == repair.draft_to_repair == body
    assert repair.required_evidence_numbers == [1, 2]
    assert harness.evidence_working_set.export() == owner_ranges
    assert ledger.completely_delivered(earlier_call) == {1}
    adapter.decide(repair)
    actual = invocation_originals(selected.invoke.call_args.kwargs["prompt"], ledger)
    assert {(row["citation"], row["text"], row["text_hash"]) for row in actual} == {
        (1, first.text, first.text_hash),
        (2, shared.text, shared.text_hash),
    }
    repaired_call = adapter.last_call_id or ""
    assert repaired_call != earlier_call
    assert ledger.completely_delivered(repaired_call) == {1, 2}
    assert ledger.completely_delivered(earlier_call) == {1}
    assert selected.invoke.call_count == 2


@pytest.mark.parametrize("edit_unit", [False, True])
def test_real_hosted_serial_reuses_rejected_body_with_fresh_original_delivery(
    monkeypatch: pytest.MonkeyPatch,
    edit_unit: bool,
) -> None:
    _kwargs, broker, selected, _checkpoints, _queue = setup_run(monkeypatch)
    selected.config = selected.config.model_copy(update={"model_name": "gpt-6-luna"})
    for index, source in enumerate(broker.sources):
        broker.sources[index] = replace(source, name=f"Example {index} Law")
        broker.chunks[str(source.id)] = replace(
            broker.chunks[str(source.id)],
            heading_path=(f"Example {index} Law", f"MADDE {142 + index}"),
        )
    source_ids = [str(source.id) for source in broker.sources]
    navigation: list[dict[str, JsonValue]] = [
        {
            "anchor_source_id": source_ids[0],
            "article_no": "142",
            "qualifier": None,
            "candidates": [
                {
                    "source_id": source_ids[1],
                    "name": "Related instrument",
                    "candidate_role": "judicial_candidate",
                }
            ],
        }
    ]
    monkeypatch.setattr(broker, "related_source_navigation", lambda: navigation)
    ledger = EvidenceLedger()
    envelope = RunContext(
        run_id="owned-retained-run",
        depth=1,
        scope={"tenant_id": "owned-tenant", "document_sets": [15]},
        budget=SharedBudget(unlimited_execution=True),
        timeout_seconds=float("inf"),
        services={
            "research_profile": "experimental",
            "experimental_parallel": True,
            "task_id": "owned-task",
            "assignment_id": "owned-assignment",
        },
    )

    def capabilities(context: RunContext) -> list[ToolSpec]:
        context.services.update(
            legal_source_navigation=broker.related_source_navigation,
            legal_source_navigation_acquire=broker.related_sources_for_evidence,
        )
        boundary = cast(CorpusBroker, broker)
        return [
            *build_corpus_specs(boundary, require_search_targets=True),
            *build_source_specs(boundary),
            *build_sandbox_specs(boundary),
            *build_supplemental_specs(),
        ]

    child = SerialExperimentalSession(
        outer_context=envelope,
        request=FOCUS,
        scenario_request=SCENARIO,
        history="",
        ledger=ledger,
        llm=selected,
        reasoning_effort=ReasoningEffort.AUTO,
        token_counter=len,
        capability_factory=capabilities,
        verify=lambda _context, _args: ToolOutcome(
            status=OutcomeStatus.FOUND, summary="Verification complete"
        ),
    )
    body = (
        "## Applicable condition\n\n"
        + (
            "The original condition and its scope remain unchanged [1][2].\n\n" * 200
        ).rstrip()
    )
    assert len(body) > 10000
    accepted_body = body
    invocations: list[dict[str, Any]] = []
    emitted: list[ModelResponse] = []
    acquisition_records: list[dict[str, JsonValue]] = []
    model_calls: list[str] = []
    rejected_native = ""

    def invoke(**arguments: Any) -> ModelResponse:
        nonlocal rejected_native, accepted_body
        invocations.append(arguments)
        model_calls.append(child.model.last_call_id or "")
        index = len(invocations)
        if index == 1:
            result = response(
                calls=[
                    ("read_source_range", {"source_id": source, "_language": "tr"})
                    for source in source_ids
                ]
            )
        else:
            originals = invocation_originals(arguments["prompt"], ledger)
            assert {row["text"] for row in originals} == {
                chunk.text for chunk in broker.chunks.values()
            }
            if index == 2:
                acquisition_records.extend(copy.deepcopy(originals))
            else:
                assert index == 3, [
                    receipt.outcome.summary for receipt in child.harness.receipts
                ]
                assert originals == acquisition_records
            definitions = tool_definitions(arguments)
            properties = definitions["submit_answer"]["function"]["parameters"][
                "properties"
            ]
            lead_ids = properties["_related_source_reviews"]["items"]["properties"][
                "lead_id"
            ]["enum"]
            assert len(lead_ids) == 1
            citations = {row["source_id"]: row["citation"] for row in originals}
            review: dict[str, JsonValue] = {
                "lead_id": lead_ids[0],
                "status": "examined",
                "source_role": "operative_text",
                "effect": "The original condition and its scope remain unchanged",
                "limitations": "its scope remain unchanged",
                "witnesses": [
                    {
                        "citation": citations[source_ids[0 if index == 2 else 1]],
                        "start_char": 0,
                        "end_char": 20,
                    }
                ],
                "gap": "",
            }
            terminal: dict[str, JsonValue] = {
                "basis": "originals",
                "_language": "tr",
                "_outcomes": [declaration()],
                "_related_source_reviews": [review],
            }
            if index == 2:
                terminal["answer"] = body
                assert "retained_answer_id" not in properties
            else:
                rejected = child.harness.receipts[-1]
                assert rejected.outcome.status == OutcomeStatus.INVALID
                assert rejected.outcome.data["invalid_related_source_review"] is True
                assert child.harness.last_draft == body
                rejected_native = child.harness.turns[-1].model_dump_json()
                current = payloads(arguments["prompt"])[-1]
                descriptor = current["retained_answer"]
                identifier = descriptor["retained_answer_id"]
                assert properties["retained_answer_id"]["enum"] == [identifier]
                assert (
                    "answer"
                    not in definitions["submit_answer"]["function"]["parameters"][
                        "required"
                    ]
                )
                assert body not in json.dumps(descriptor)
                units = current["draft_to_repair"]["units"]
                assert "units" not in descriptor
                assert "".join(unit["text"] for unit in units) == body
                assert all(
                    json.dumps(body) not in json.dumps(payload)
                    for payload in payloads(arguments["prompt"])
                )
                assert not any(
                    isinstance(message, AssistantMessage)
                    and any(
                        call.id == rejected.call.call_id
                        for call in message.tool_calls or []
                    )
                    for message in arguments["prompt"]
                )
                assert any(
                    isinstance(message, ToolMessage) for message in arguments["prompt"]
                )
                terminal["retained_answer_id"] = identifier
                if edit_unit:
                    unit = units[1]
                    replacement = "The original condition and its scope remain unchanged; the supplied fact is distinct [1][2]."
                    terminal["retained_answer_edits"] = [
                        {"unit_id": unit["unit_id"], "replacement": replacement}
                    ]
                    original = unit["text"]
                    prefix = original[: len(original) - len(original.lstrip())]
                    suffix = original[len(original.rstrip()) :]
                    accepted_body = (
                        body[: unit["start_char"]]
                        + prefix
                        + replacement
                        + suffix
                        + body[unit["end_char"] :]
                    )
            result = response(calls=[("submit_answer", terminal)])
        emitted.append(result)
        return result

    selected.invoke.side_effect = invoke
    outcome = child.run()
    assert selected.invoke.call_count == 3
    assert outcome.status == OutcomeStatus.FOUND and outcome.summary == accepted_body
    assert child.harness.last_draft == accepted_body
    assert child.harness.receipts[-1].call.arguments["answer"] == accepted_body
    assert "retained_answer_id" not in child.harness.receipts[-1].call.arguments
    assert "retained_answer_edits" not in child.harness.receipts[-1].call.arguments
    assert child.harness.turns[-2].model_dump_json() == rejected_native
    final_wire = child.harness.turns[-1].assistant
    assert final_wire is not None and final_wire.tool_calls
    final_arguments = json.loads(final_wire.tool_calls[0].function.arguments)
    assert "answer" not in final_arguments and "retained_answer_id" in final_arguments
    assert ("retained_answer_edits" in final_arguments) is edit_unit
    assert len(final_wire.model_dump_json()) < len(body) // 5
    accepted_call = child.model.last_call_id or ""
    assert ledger.completely_delivered(accepted_call) == {1, 2}
    envelope.services["last_model_call_id"] = accepted_call
    receipts = ParallelAnswerReceipts(envelope, SCENARIO, user_id="owner")
    sealed = receipts.seal(
        envelope,
        assignment={"question_id": "owned-assignment", "task_id": "owned-task"},
        answer=accepted_body,
        status=outcome.status,
        model_call_id=accepted_call,
        ledger=ledger,
        validate_body=lambda: child.validate_accepted(
            accepted_body, accepted_call, outcome.status
        ),
        source_state=child.source_state(),
    )
    assert sealed.startswith("parallel_")
    pins = ledger.export()["pinned_delivery_calls"]
    assert isinstance(pins, list) and accepted_call in pins
    assert ledger.completely_delivered(model_calls[2]) == {1, 2}
    assert (
        child.validate_accepted(accepted_body, accepted_call, OutcomeStatus.FOUND)
        is None
    )
    assert (
        child.publication_gap(accepted_body, "different-call", requires_sources=True)
        is not None
    )

    # The same rejected journal remains visible without the trusted hosted opt-in.
    plain = RunContext(
        run_id=child.context.run_id,
        scope=copy.deepcopy(child.context.scope),
        services={"research_profile": "experimental", "experimental_parallel": False},
    )
    adapter = ResearchModel(selected, plain, token_counter=len, lean_native_mode=True)
    view = child.harness.view()
    view.draft_to_repair = body
    view.publication_gap = {"invalid_related_source_review": True}
    prompt, schemas, _output = adapter._fit_native_decision(view)
    assert "retained_answer" not in payloads(prompt)[-1]
    submit = tool_definitions({"tools": schemas})["submit_answer"]
    assert "retained_answer_id" not in submit["function"]["parameters"]["properties"]
    assert any(
        isinstance(message, AssistantMessage)
        and any(
            call.id == child.harness.receipts[-2].call.call_id
            for call in message.tool_calls or []
        )
        for message in prompt
    )
    assert selected.invoke.call_count == 3
