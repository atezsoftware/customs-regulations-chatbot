"""Exact metadata transport and delivery receipts through the provider projection."""

import hashlib
import json
from contextlib import nullcontext
from copy import deepcopy
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock

import pytest
from pydantic import JsonValue

from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import EvidenceItem, RunContext, SharedBudget
from onyx.supersearch import gateway
from onyx.supersearch.models import WriterDecision
from onyx.supersearch.payload import (
    METADATA_DECODER,
    METADATA_POOL_KEY,
    compact_original_evidence_payload,
    expand_original_evidence_payload,
)
from onyx.supersearch.receipts import (
    RECEIPT_DATA_DECODER,
    RECEIPT_DATA_POOL_KEY,
    expand_receipt_data_payload,
)
from onyx.tracing.flows import LLMFlow
from tests.unit.onyx.supersearch.test_canonical_runtime import (
    selected_llm,
    structured_stream,
)
from tests.unit.onyx.supersearch.test_engine import original


def records_payload() -> dict[str, JsonValue]:
    records: list[JsonValue] = []
    for index in range(40):
        text = (
            f"Özgün metin {index}: hükümler, istisnalar ve koşullar.\nİkinci paragraf."
        )
        metadata: dict[str, JsonValue] = {
            "title": "PC Külliyatı kaynak başlığı " * 8,
            "heading_path": ["Başlangıç ve koşullar " * 8, f"Madde {index % 3}"],
            "legal_dates": ["2009-10-08", "2026-10-08"],
            "validity_end": None,
            "access_control_list": ["group:regulatory", "user:authorized"],
            "labels": {"numeric": 1, "boolean": True, "empty": [], "nullable": None},
            "reference_like_value": {"metadata_ref": 1, "value_refs": [0, 1]},
            "true": True,
            "one": 1,
            "float_one": 1.0,
            "false": False,
            "zero": 0,
            "float_zero": 0.0,
            "empty_string": "",
            "empty_list": [],
            "empty_object": {},
        }
        if index % 2:
            metadata["optional_null"] = None
        records.append(
            {
                "citation": index + 1,
                "source_id": "pc-source",
                "chunk_id": f"atomic-{index}",
                "text_hash": hashlib.sha256(text.encode()).hexdigest(),
                "text": text,
                "truncated": False,
                "citable": True,
                "metadata": metadata,
                "witness_spans": [{"start_char": 0, "end_char": len(text)}],
            }
        )
    return {"request": "Başvuru koşulları nelerdir?", "original_evidence": records}


def serialized(value: JsonValue) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def test_round_trip_retains_every_value_field_order_and_original_byte() -> None:
    payload = records_payload()
    before = serialized(payload)
    projection, compacted = compact_original_evidence_payload(payload)
    assert compacted
    assert serialized(expand_original_evidence_payload(projection)) == before
    assert serialized(payload) == before
    assert len(serialized(projection).encode()) + len(METADATA_DECODER.encode()) < len(
        before.encode()
    )
    raw_records = cast(list[dict[str, JsonValue]], payload["original_evidence"])
    projected_records = cast(
        list[dict[str, JsonValue]], projection["original_evidence"]
    )
    for raw, projected in zip(raw_records, projected_records, strict=True):
        assert serialized(
            {key: value for key, value in raw.items() if key != "metadata"}
        ) == serialized(
            {key: value for key, value in projected.items() if key != "metadata_ref"}
        )


@pytest.mark.parametrize(
    "payload",
    [
        {"request": "Hello", "original_evidence": []},
        {"original_evidence": [{"citation": 1, "metadata": {"title": "Short"}}]},
        {"original_evidence": [{"citation": 1, "metadata": None}]},
        {"original_metadata": {"existing": True}, **records_payload()},
    ],
)
def test_small_or_reserved_payloads_keep_the_original_transport(
    payload: dict[str, JsonValue],
) -> None:
    projection, compacted = compact_original_evidence_payload(payload)
    assert not compacted
    assert projection is payload


def test_malformed_metadata_references_cannot_silently_decode_other_fields() -> None:
    projection, compacted = compact_original_evidence_payload(records_payload())
    assert compacted
    rows = cast(list[dict[str, JsonValue]], projection["original_evidence"])
    rows[0]["metadata_ref"] = -1
    with pytest.raises(ValueError, match="reference is out of range"):
        expand_original_evidence_payload(projection)


@pytest.mark.parametrize(
    "flow", [LLMFlow.SUPERSEARCH_ANSWER, LLMFlow.SUPERSEARCH_REVIEW]
)
def test_gateway_preserves_actual_complete_original_delivery_when_metadata_is_pooled(
    monkeypatch: pytest.MonkeyPatch,
    flow: LLMFlow,
) -> None:
    llm = selected_llm()
    context = RunContext(
        timeout_seconds=float("inf"),
        budget=SharedBudget(unlimited_execution=True),
        scope={"asv3_document_set_id": 442},
    )
    ledger = EvidenceLedger()
    base = original()
    assert base.search_doc is not None
    items = [
        EvidenceItem(
            source_id=base.source_id,
            chunk_id=f"atomic-{index}",
            text=f"{base.text}\nÖzgün ek paragraf {index}.",
            search_doc=base.search_doc.model_copy(deep=True),
            metadata={
                "title": "PC Külliyatı kaynak başlığı " * 20,
                "document_date": "2026-10-08",
            },
        )
        for index in range(40)
    ]
    numbers = ledger.add(items, context)
    payload = {
        "original_evidence": json.loads(
            ledger.serialize_records(numbers, max_chars=None)
        ),
        "receipts": [
            {
                "status": "ambiguous",
                "data": {"sources": ["PC kaynak başlığı " * 20] * 20},
            }
            for _ in range(4)
        ],
    }
    before = deepcopy(payload)
    span = SimpleNamespace(span_data=SimpleNamespace(model_config={}))
    monkeypatch.setattr(
        gateway, "llm_generation_span", lambda **_kwargs: nullcontext(span)
    )
    monkeypatch.setattr(gateway, "record_llm_response", lambda *_args: None)
    cast(MagicMock, llm).stream.return_value = iter(
        structured_stream(
            json.dumps(
                {
                    "answer": "Kaynaklı sonuç [1]",
                    "unresolved_need_ids": [],
                    "actions": [],
                }
            )
        )
    )
    selected = gateway.SelectedModelGateway(llm=llm, ledger=ledger, context=context)
    selected.complete("PC originals only", payload, WriterDecision, flow, True)
    messages = cast(MagicMock, llm).stream.call_args.args[0]
    cast(MagicMock, llm).invoke.assert_not_called()
    assert cast(MagicMock, llm).stream.call_args.kwargs["timeout_override"] == 120
    provider_payload = json.loads(messages[1].content)
    assert METADATA_POOL_KEY in provider_payload
    assert METADATA_DECODER in messages[0].content
    assert (RECEIPT_DATA_POOL_KEY in provider_payload) == (
        flow == LLMFlow.SUPERSEARCH_REVIEW
    )
    assert (RECEIPT_DATA_DECODER in messages[0].content) == (
        flow == LLMFlow.SUPERSEARCH_REVIEW
    )
    assert (
        expand_original_evidence_payload(expand_receipt_data_payload(provider_payload))
        == before
    )
    assert payload == before
    assert selected.last_call_id is not None
    assert ledger.completely_delivered(selected.last_call_id) == set(numbers)
    assert selected.last_delivered_citations == set(numbers)
