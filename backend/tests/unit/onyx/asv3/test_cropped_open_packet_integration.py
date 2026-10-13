"""Open topic scheduling uses the actual working view and native source decoder."""

import hashlib
import json
from typing import cast
from unittest.mock import MagicMock

import pytest
import tiktoken
from pydantic import JsonValue

from onyx.asv3.cropped_sections import CroppedSectionMaterialContext
from onyx.asv3.cropped_task_packets import (
    CroppedTaskPacketPolicy,
    CroppedTaskPacketStore,
    expand_task_packet_receipts,
)
from onyx.asv3.cropped_transport import prepare_cropped_writer_view
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.harness import Harness
from onyx.asv3.llm_adapter import ResearchModel, _NativeOriginalCatalogue
from onyx.asv3.models import (
    CapabilityCall,
    Decision,
    HarnessView,
    OutcomeStatus,
    ResearchTurn,
    ToolOutcome,
    ToolReceipt,
    ToolSpec,
)
from onyx.asv3.native_cache_projection import decode_compact_originals
from onyx.asv3.registry import CapabilityRegistry
from onyx.asv3.research_state import ResearchNeed, ResearchState, ResearchUpdate
from onyx.asv3.workflow_variant import ASV3_TUNED_VARIANT
from onyx.llm.interfaces import LLM
from onyx.llm.models import (
    AssistantMessage,
    ChatCompletionMessage,
    FunctionCall,
    ToolCall,
)
from onyx.tracing.flows import LLMFlow
from tests.unit.onyx.asv3.test_cropped_task_packets import REQUEST, fixture, receipt
from tests.unit.onyx.asv3.test_native_authority import original
from tests.unit.onyx.asv3.test_native_model_adapter import model


def open_fixture(
    *, dependency: bool = False, small: bool = False
) -> tuple[Harness, ResearchModel, HarnessView, EvidenceLedger, set[int], set[int]]:
    context, _, _, _ = fixture()
    context.services.update(
        lean_native_mode=True,
        research_profile="normal",
        asv3_workflow_variant=ASV3_TUNED_VARIANT,
    )
    a_count, b_count = (3, 8) if small else (64, 192)
    a_numbers = set(range(1, a_count + 1))
    b_numbers = set(range(a_count + 1, a_count + b_count + 1))
    ledger = EvidenceLedger()
    items = []
    for number in range(1, a_count + b_count + 2):
        topic = (
            "A" if number in a_numbers else "B" if number in b_numbers else "unknown"
        )
        title = f"Synthetic {topic} Kanunu"
        item = original(
            title,
            str(number),
            text=(
                f"**MADDE {number}** .- (1) SYNTHETIC-{topic}-{number}: "
                "Başvurunun aktörü, maddi koşulu ve tarih kapsamı birlikte değerlendirilir.\n"
                "(2) İstisna, karşı delil ve sonraki başvuru adımı bağımsız olarak korunur.\n"
                "(3) Bu sentetik metin test verisidir; gerçek bir hukuki hüküm değildir."
            ),
        )
        canonical = item.metadata["canonical_metadata"]
        assert isinstance(canonical, dict)
        canonical.update(
            article_no=str(number),
            paragraph_no=None,
            provision_identifiers=[f"madde {number}"],
            read_as_of_date="2026-10-13",
            validity_start="2026-01-01",
            validity_end=None,
        )
        item.metadata["position"] = number - 1
        items.append(item)
    assert len(ledger.add(items, context)) == a_count + b_count + 1
    state = ResearchState([REQUEST], context, require_need_bindings=True)
    state.update(
        ResearchUpdate(
            needs=[
                ResearchNeed(
                    need_id=key,
                    question_ids=["q0"],
                    purpose=f"Assess topic {key}, with its conditions and exceptions",
                    completion_test="Read the operative originals and address the remaining material issue",
                    status="researching",
                    depends_on=["B"] if dependency and key == "A" else [],
                )
                for key in ("A", "B")
            ],
            active_need_ids=["A"],
        ),
        ledger,
    )
    context.services.update(evidence=ledger, research_state=state)
    # A broad view uses the same tools, state and sources before the activation gate.
    context.services["cropped_task_packet_store"] = CroppedTaskPacketStore(
        context,
        REQUEST,
        policy=CroppedTaskPacketPolicy(
            cumulative_input_tokens=100_000_000, original_records=1000
        ),
    )
    registry = CapabilityRegistry()
    registry.register(
        ToolSpec(
            name="read_provision",
            description="Read an exact complete source provision",
            parameters={
                "type": "object",
                "properties": {
                    "source_id": {"type": "string"},
                    "article": {"type": "string"},
                },
                "required": ["source_id", "article"],
                "additionalProperties": False,
            },
            handler=lambda _arguments, _context: ToolOutcome(
                status=OutcomeStatus.NOT_FOUND, summary="Unused offline handler"
            ),
        )
    )
    for name in ("submit_answer", "submit_partial_answer"):
        registry.register(
            ToolSpec(
                name=name,
                description="Return supported findings and precise unresolved limitations",
                parameters={
                    "type": "object",
                    "properties": {"answer": {"type": "string", "minLength": 1}},
                    "required": ["answer"],
                    "additionalProperties": False,
                },
                handler=lambda _arguments, _context: ToolOutcome(
                    status=OutcomeStatus.PARTIAL, summary="Unused offline terminal"
                ),
                consumes_tool_budget=False,
            )
        )
    active = Harness(
        request=REQUEST,
        context=context,
        registry=registry,
        decide=lambda _view: Decision(),
        evidence=ledger,
    )
    active.facts = ["Common user-supplied transaction date remains 2026-10-13"]
    turns = []
    for number in ledger.citation_numbers():
        key = "A" if number in a_numbers else "B" if number in b_numbers else None
        actual = receipt(number, key, ledger)
        item = ledger.get(number)
        assert item is not None
        actual.outcome.evidence = [item]
        if number == min(b_numbers):
            actual.outcome.status = OutcomeStatus.PARTIAL
            actual.outcome.data = {
                "has_more": True,
                "next_position": 900,
                "absence_proven": False,
            }
        actual.outcome.data["original_evidence"] = json.loads(
            ledger.serialize_records([number], max_chars=None)
        )
        active._commit_receipt(actual)
        turns.append(
            ResearchTurn(
                assistant=AssistantMessage(
                    content="Keep the source conditions and shared scenario unchanged.",
                    tool_calls=[
                        ToolCall(
                            id=actual.call.call_id,
                            function=FunctionCall(
                                name=actual.call.name,
                                arguments=json.dumps(actual.call.arguments),
                            ),
                        )
                    ],
                ),
                results=[active._model_tool_result(actual)],
            )
        )
    failed = ToolReceipt(
        call=CapabilityCall(
            name="read_provision",
            call_id="topic-B-unavailable",
            arguments={"source_id": "unresolved-B", "article": "900", "_need_id": "B"},
        ),
        outcome=ToolOutcome(
            status=OutcomeStatus.NOT_FOUND,
            summary="Exact operative source was unavailable; no absence conclusion",
            data={
                "not_read": True,
                "absence_proven": False,
                "unlisted_marker": {
                    "literal_error": "Exact source endpoint unavailable; continuation remains unverified.",
                    "probe": False,
                    "nullable": None,
                    "attempt_code": 17,
                },
            },
        ),
        elapsed_seconds=0,
    )
    active._commit_receipt(failed)
    active.turns = turns
    broad = active.view()
    assert "cropped_task_packet" not in broad.research_state
    context.services["cropped_task_packet_store"] = CroppedTaskPacketStore(
        context, REQUEST
    )
    adapter = ResearchModel(
        model(limit=1_000_000),
        context,
        lean_native_mode=True,
        token_counter=lambda text: len(
            tiktoken.get_encoding("o200k_base").encode(text)
        ),
    )
    return active, adapter, broad, ledger, a_numbers, b_numbers


def payloads(prompt: list[ChatCompletionMessage]) -> list[dict[str, JsonValue]]:
    from onyx.asv3.cropped_handoff_transport import expand_cropped_handoff_transport

    result = []
    for message in prompt:
        if not isinstance(message.content, str):
            continue
        try:
            value = json.loads(message.content)
        except ValueError:
            continue
        if isinstance(value, dict):
            result.append(expand_cropped_handoff_transport(value))
    return result


def contains_mapping(value: JsonValue, expected: dict[str, JsonValue]) -> bool:
    if isinstance(value, dict):
        return value == expected or any(
            contains_mapping(child, expected) for child in value.values()
        )
    if isinstance(value, list):
        return any(contains_mapping(child, expected) for child in value)
    return False


def test_active_open_topic_does_not_reimport_other_open_topic_source_bodies() -> None:
    active, adapter, broad, ledger, a_numbers, b_numbers = open_fixture()
    before_ledger = ledger.export()
    selected = active.view()
    unknown = max(ledger.citation_numbers())
    assert selected.required_evidence_numbers == sorted(a_numbers | {unknown})
    assert selected.turns == []
    assert selected.facts == broad.facts and selected.questions == broad.questions
    manifest = cast(
        dict[str, JsonValue], selected.research_state["cropped_task_packet"]
    )
    assert manifest["legal_approval"] is False
    index = cast(list[dict[str, JsonValue]], manifest["detached_need_index"])
    assert len(index) == 1 and index[0]["status"] == "retained_open_need_not_completed"
    need = cast(dict[str, JsonValue], index[0]["need"])
    assert need["need_id"] == "B" and need["status"] == "researching"
    assert index[0]["originals_unread_this_packet"] is True
    pending = {
        row["call_id"]: row
        for row in cast(list[dict[str, JsonValue]], index[0]["pending_operations"])
    }
    assert pending[f"read-{min(b_numbers)}"]["continuation"] == {
        "has_more": True,
        "next_position": 900,
        "absence_proven": False,
    }
    assert pending["topic-B-unavailable"]["status"] == "not_found"
    assert pending["topic-B-unavailable"]["continuation"] == {
        "not_read": True,
        "absence_proven": False,
    }
    prompt, tools, _ = adapter._fit_native_decision(selected)
    wire = json.dumps([m.model_dump(mode="json") for m in prompt], ensure_ascii=False)
    catalogue = _NativeOriginalCatalogue.bind(prompt, cheap_metadata=True)
    assert {row["citation"] for row in catalogue.records} == a_numbers | {unknown}
    for number in b_numbers:
        item = ledger.get(number)
        assert item is not None and item.text not in wire
    broad_prompt, broad_tools, _ = adapter._fit_native_decision(broad)
    assert adapter._input_cost(prompt, tools) < adapter._input_cost(
        broad_prompt, broad_tools
    )
    assert ledger.export() == before_ledger


def test_reactivated_open_topic_restores_exact_receipts_then_final_restores_all() -> (
    None
):
    active, adapter, broad, ledger, a_numbers, b_numbers = open_fixture()
    active.view()
    state = active.context.services["research_state"]
    assert isinstance(state, ResearchState)
    state.update(ResearchUpdate(active_need_ids=["B"]), ledger)
    selected = active.view()
    unknown = max(ledger.citation_numbers())
    assert set(selected.required_evidence_numbers) == b_numbers | {unknown}
    manifest = cast(
        dict[str, JsonValue], selected.research_state["cropped_task_packet"]
    )
    restored = expand_task_packet_receipts(manifest, selected.original_evidence)
    actual = {
        row.call.call_id: row.model_dump(mode="json")
        for row in active.receipts
        if row.call.arguments.get("_need_id") == "B"
        or not row.call.arguments.get("_need_id")
    }
    assert {
        cast(dict[str, JsonValue], row["call"])["call_id"]: row for row in restored
    } == actual
    prompt, _, _ = adapter._fit_native_decision(selected)
    assert {
        row["citation"]
        for row in _NativeOriginalCatalogue.bind(prompt, cheap_metadata=True).records
    } == b_numbers | {unknown}
    state.update(ResearchUpdate(active_need_ids=["A"]), ledger)
    selected = active.view()
    assert set(selected.required_evidence_numbers) == a_numbers | {unknown}
    final = prepare_cropped_writer_view(
        selected, active.context, adapter._native_original_records(broad.turns)
    )
    assert set(final.required_evidence_numbers) == a_numbers | b_numbers | {unknown}
    assert "cropped_task_packet" not in final.research_state
    assert {row.call.call_id for row in final.receipts} == {
        row.call.call_id for row in active.receipts
    }
    adapter._writing_answer = True
    final_prompt, _, _ = adapter._fit_native_decision(final)
    catalogue = _NativeOriginalCatalogue.bind(final_prompt, cheap_metadata=True)
    expanded = payloads(final_prompt)
    final_records = decode_compact_originals(expanded, catalogue.records, ledger)
    assert {row["citation"] for row in final_records} == set(ledger.citation_numbers())
    for row in final_records:
        item = ledger.get(cast(int, row["citation"]))
        assert item is not None and row["text"] == item.text
        assert (
            hashlib.sha256(cast(str, row["text"]).encode()).hexdigest()
            == item.text_hash
        )
    restored_contexts = [
        cast(dict[str, JsonValue], observation["cropped_restored_topic_context"])
        for value in expanded
        if isinstance(observations := value.get("repair_research_observations"), list)
        for observation in observations
        if isinstance(observation, dict)
        and "cropped_restored_topic_context" in observation
    ]
    assert len(restored_contexts) == 1
    actual_all = {
        row.call.call_id: row.model_dump(mode="json") for row in active.receipts
    }
    restored_all = expand_task_packet_receipts(restored_contexts[0], final_records)
    assert {
        cast(dict[str, JsonValue], row["call"])["call_id"]: row for row in restored_all
    } == actual_all
    assert all(
        "cropped_restored_topic_context"
        not in cast(dict[str, JsonValue], value.get("research_state", {}))
        for value in expanded
    )
    failed = next(
        row for row in active.receipts if row.call.call_id == "topic-B-unavailable"
    )
    marker = cast(dict[str, JsonValue], failed.outcome.data["unlisted_marker"])
    assert any(contains_mapping(value, marker) for value in expanded)
    assert any(
        "topic-B-unavailable" in json.dumps(value, ensure_ascii=False)
        for value in expanded
    )


def test_explicit_dependency_keeps_open_topic_full_source_union() -> None:
    active, adapter, _, ledger, a_numbers, b_numbers = open_fixture(dependency=True)
    selected = active.view()
    prompt, _, _ = adapter._fit_native_decision(selected)
    assert {
        row["citation"]
        for row in _NativeOriginalCatalogue.bind(prompt, cheap_metadata=True).records
    } == a_numbers | b_numbers | {max(ledger.citation_numbers())}
    assert selected.facts == active.facts


@pytest.mark.parametrize("cropped", [False, True])
def test_untriggered_small_open_topic_native_input_remains_identical(
    cropped: bool,
) -> None:
    active, adapter, _, _, _, _ = open_fixture(small=True)
    active.context.services["asv3_exp_cropped"] = cropped
    before = active.view()
    before_prompt, before_tools, _ = adapter._fit_native_decision(before)
    after = active.view()
    after_prompt, after_tools, _ = adapter._fit_native_decision(after)
    assert "cropped_task_packet" not in after.research_state
    assert [m.model_dump(mode="json") for m in before_prompt] == [
        m.model_dump(mode="json") for m in after_prompt
    ]
    assert before_tools == after_tools


@pytest.mark.parametrize("overflow", [False, True])
def test_actual_source_writer_and_section_boundary_receive_parked_receipt_context(
    monkeypatch: pytest.MonkeyPatch, overflow: bool
) -> None:
    from onyx.asv3.cropped_transport import CroppedInputOverflow

    active, adapter, _, ledger, _, _ = open_fixture()
    selected = active.view()
    failed = next(
        row for row in active.receipts if row.call.call_id == "topic-B-unavailable"
    )
    marker = cast(dict[str, JsonValue], failed.outcome.data["unlisted_marker"])
    seen: list[str] = []

    def invoke(
        prepared: HarnessView,
        prompt: list[ChatCompletionMessage],
        _tools: list[dict[str, JsonValue]],
        _output: int,
        *,
        call_llm_override: LLM | None = None,
        flow_override: LLMFlow | None = None,
    ) -> Decision:
        assert adapter._writing_answer is True
        assert call_llm_override is adapter.llm and flow_override == LLMFlow.ASV3_FINAL
        assert {row.call.call_id for row in prepared.receipts} == {
            row.call.call_id for row in active.receipts
        }
        assert any(contains_mapping(value, marker) for value in payloads(prompt))
        assert {
            row["citation"]
            for row in _NativeOriginalCatalogue.bind(
                prompt, cheap_metadata=True
            ).records
        } == set(ledger.citation_numbers())
        seen.append("writer")
        if overflow:
            raise CroppedInputOverflow(240001)
        return Decision(answer="Synthetic supported candidate [1].")

    def sections(
        prepared: HarnessView,
        *,
        material_context: CroppedSectionMaterialContext | None = None,
    ) -> Decision:
        assert material_context is not None
        assert {row.call.call_id for row in prepared.receipts} == {
            row.call.call_id for row in active.receipts
        }
        assert contains_mapping(
            cast(JsonValue, material_context.repair_observations()), marker
        )
        seen.append("sections")
        return Decision(answer="Synthetic section candidate [1].")

    monkeypatch.setattr(adapter, "_invoke_decision", invoke)
    monkeypatch.setattr(adapter, "_assemble_cropped_sections", sections)
    adapter._decide_cropped_source_answer(selected)
    assert seen == (["writer", "sections"] if overflow else ["writer"])
    cast(MagicMock, adapter.llm).invoke.assert_not_called()
