import json
from typing import cast

import pytest
from pydantic import JsonValue

from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.llm_adapter import ResearchModel
from onyx.asv3.models import EvidenceItem, RunContext
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
    tool_payloads = [
        json.loads(message.content)
        for message in prompt
        if isinstance(message, ToolMessage)
    ]
    if projected:
        assert "original_evidence" not in final
        assert final["original_metadata_catalogue"][0]["metadata"] == record["metadata"]
        actual = tool_payloads[0]["original_evidence"][0]
        assert actual["text"] == record["text"]
        assert actual["metadata_ref"] == {
            "citation": record["citation"],
            "text_hash": record["text_hash"],
        }
        assert "metadata" not in actual
        assert "original_evidence" not in tool_payloads[1]
        assert "metadata_ref" in tool_payloads[1]["original_evidence_refs"][0]
    else:
        assert final["original_evidence"][0]["text"] == record["text"]
        assert final["original_evidence"][0]["metadata"] == record["metadata"]
        for payload in tool_payloads:
            reference = payload["original_evidence_refs"][0]
            assert "metadata_ref" not in reference
            assert "metadata" in reference
    assert ledger.completely_delivered(adapter.last_call_id or "") == {1}
    assert selected.invoke.call_count == 1
    assert [item.model_dump(mode="json") for item in native_turns] == before
