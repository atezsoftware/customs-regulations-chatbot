import copy
import json
from typing import cast

import pytest
from pydantic import JsonValue

from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.llm_adapter import ResearchModel
from onyx.asv3.models import EvidenceItem, ResearchTurn, RunContext
from onyx.asv3.native_projection import reference_duplicate_metadata
from onyx.llm.models import ToolMessage
from tests.unit.onyx.asv3.test_native_model_adapter import (
    last_payload,
    model,
    turn,
    view,
)


def setup_original() -> tuple[EvidenceLedger, RunContext, dict[str, JsonValue]]:
    context, ledger = RunContext(), EvidenceLedger()
    context.services["evidence"] = ledger
    number = ledger.add(
        [
            EvidenceItem(
                source_id="canonical-instrument",
                chunk_id="operative-clause",
                text="The authorization AND the proof are required.\n  ",
                metadata={
                    "heading_path": ["Official instrument", "Detailed scope " * 45],
                    "validity_start": "2026-01-01",
                    "validity_end": None,
                    "version_unknown": False,
                },
            )
        ],
        context,
    )[0]
    record = cast(
        dict[str, JsonValue], json.loads(ledger.serialize_records([number]))[0]
    )
    return ledger, context, record


def history(record: dict[str, JsonValue], count: int = 1) -> list[ResearchTurn]:
    return ResearchModel._native_turns_with_original_references(
        [turn(f"read-{index}", [record]) for index in range(count)]
    )


def test_repeated_metadata_is_referenced_without_mutating_text_or_audit() -> None:
    ledger, _, record = setup_original()
    turns = history(record, 24)
    before = [item.model_dump(mode="json") for item in turns]
    saved_ledger = ledger.export()
    projected = reference_duplicate_metadata(turns, [record], ledger)
    for previous, current in zip(turns, projected, strict=True):
        old_payload = json.loads(previous.results[0].content)
        new_payload = json.loads(current.results[0].content)
        expected = copy.deepcopy(old_payload)
        reference = expected["original_evidence_refs"][0]
        reference.pop("metadata")
        reference["metadata_ref"] = {
            "citation": record["citation"],
            "text_hash": record["text_hash"],
        }
        assert new_payload == expected
        assert current.assistant == previous.assistant
        assert current.results[0].tool_call_id == previous.results[0].tool_call_id
    saved_bytes = sum(
        len(old.results[0].content) - len(new.results[0].content)
        for old, new in zip(turns, projected, strict=True)
    )
    assert saved_bytes > 12000
    assert [item.model_dump(mode="json") for item in turns] == before
    assert ledger.export() == saved_ledger
    item = ledger.get(1)
    assert item is not None
    assert item.text == record["text"]


@pytest.mark.parametrize(
    "change",
    ["missing", "partial", "start", "text", "hash", "source", "chunk", "metadata"],
)
def test_only_full_current_canonical_metadata_can_replace_a_history_copy(
    change: str,
) -> None:
    ledger, _, record = setup_original()
    current = copy.deepcopy(record)
    if change == "partial":
        current["text"] = str(current["text"])[:12]
    elif change == "start":
        current["start_char"] = 1
    elif change == "text":
        current["text"] = str(current["text"]) + "Altered qualification"
    elif change == "hash":
        current["text_hash"] = "0" * 64
    elif change in {"source", "chunk"}:
        current[change + "_id"] = "another-identity"
    elif change == "metadata":
        current["metadata"] = {"version_unknown": True}
    turns = history(record)
    projected = reference_duplicate_metadata(
        turns, [] if change == "missing" else [current], ledger
    )
    assert projected == turns


@pytest.mark.parametrize("change", ["source_id", "chunk_id", "text_hash", "metadata"])
def test_distinct_historical_identity_or_acquisition_metadata_is_retained(
    change: str,
) -> None:
    ledger, _, record = setup_original()
    old = copy.deepcopy(record)
    old[change] = {"version_unknown": True} if change == "metadata" else "different"
    turns = history(old)
    assert reference_duplicate_metadata(turns, [record], ledger) == turns


def test_non_reference_tool_data_and_malformed_history_are_preserved() -> None:
    ledger, _, record = setup_original()
    turns = history(record)
    turns[0].results = [
        ToolMessage(tool_call_id="one", content='{"outcome":{"data":{"text":"keep"}}}'),
        ToolMessage(tool_call_id="two", content="not json"),
        ToolMessage(tool_call_id="three", content="[]"),
    ]
    assert reference_duplicate_metadata(turns, [record], ledger) == turns


def test_existing_metadata_reference_is_preserved_with_full_metadata() -> None:
    ledger, _, record = setup_original()
    turns = history(record)
    payload = json.loads(turns[0].results[0].content)
    payload["original_evidence_refs"][0]["metadata_ref"] = {
        "citation": 91,
        "text_hash": "prior-reference",
    }
    turns[0].results[0].content = json.dumps(payload)
    assert reference_duplicate_metadata(turns, [record], ledger) == turns


@pytest.mark.parametrize(
    "profile,parallel,projected",
    [
        ("experimental", True, True),
        ("experimental", False, False),
        ("normal", True, False),
        ("deep", True, False),
    ],
)
def test_real_native_decision_keeps_complete_originals_and_only_parallel_refs(
    profile: str, parallel: bool, projected: bool
) -> None:
    ledger, context, record = setup_original()
    context.services.update(
        research_profile=profile,
        experimental_parallel=parallel,
        scenario_request="Explain both outcomes without dropping conditions.",
    )
    selected = model(limit=1000000)
    adapter = ResearchModel(selected, context, lean_native_mode=True)
    native_turns = [turn("read-one", [record]), turn("read-two", [record])]
    before = [item.model_dump(mode="json") for item in native_turns]
    adapter.decide(view(turns=native_turns, original_evidence=[record]))
    prompt = selected.invoke.call_args.kwargs["prompt"]
    final = last_payload(selected)
    assert final["original_evidence"][0]["text"] == record["text"]
    assert final["original_evidence"][0]["metadata"] == record["metadata"]
    for message in prompt:
        if isinstance(message, ToolMessage):
            reference = json.loads(message.content)["original_evidence_refs"][0]
            assert ("metadata_ref" in reference) is projected
            assert ("metadata" in reference) is not projected
    assert ledger.completely_delivered(adapter.last_call_id or "") == {1}
    assert selected.invoke.call_count == 1
    assert [item.model_dump(mode="json") for item in native_turns] == before
