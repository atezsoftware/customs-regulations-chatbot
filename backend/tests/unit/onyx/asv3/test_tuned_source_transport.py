import copy
import json
from typing import cast

import pytest
from pydantic import JsonValue

from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.llm_adapter import ResearchModel
from onyx.asv3.models import EvidenceItem, RunContext
from onyx.asv3.native_cache_projection import (
    decode_compact_originals,
    lossless_original_transport_enabled,
)
from onyx.asv3.source_metadata_transport import (
    expand_source_metadata,
    share_source_metadata,
)
from onyx.asv3.workflow_variant import ASV3_TUNED_VARIANT
from onyx.llm.models import ToolMessage
from tests.unit.onyx.asv3.test_native_cache_projection import payloads
from tests.unit.onyx.asv3.test_native_model_adapter import model, turn, view


def records() -> tuple[EvidenceLedger, RunContext, list[dict[str, JsonValue]]]:
    ledger = EvidenceLedger()
    context = RunContext(
        scope={"document_set_id": 15, "tenant_id": "owned"},
        services={
            "asv3_workflow_variant": ASV3_TUNED_VARIANT,
            "research_profile": "normal",
        },
    )
    context.services["evidence"] = ledger
    numbers = ledger.add(
        [
            EvidenceItem(
                source_id="owned-source",
                chunk_id=f"clause-{index}",
                text=f"Koşul {index} VE istisna; izin verilmeden uygulanmaz. 東京.\n  ",
                metadata={
                    "title": "Verified official source " * 16,
                    "document_type": "kanun",
                    "legal_dates": {
                        "validity_start": "2026-01-01",
                        "validity_end": None,
                    },
                    "version_unknown": False,
                    "heading_path": ["Shared title", f"Provision {index}"],
                    "article_no": str(index),
                },
            )
            for index in range(1, 6)
        ],
        context,
    )
    originals = cast(
        list[dict[str, JsonValue]], json.loads(ledger.serialize_records(numbers))
    )
    return ledger, context, originals


def catalogue(originals: list[dict[str, JsonValue]]) -> list[dict[str, JsonValue]]:
    return [
        {
            key: row[key]
            for key in ("citation", "source_id", "chunk_id", "text_hash", "metadata")
        }
        for row in originals
    ]


def test_source_factoring_preserves_exact_metadata_and_does_not_mutate_records() -> (
    None
):
    _, _, originals = records()
    full = catalogue(originals)
    before = copy.deepcopy(full)
    encoded, shared = share_source_metadata(full)
    first_metadata = originals[0]["metadata"]
    assert isinstance(first_metadata, dict)
    assert shared["owned-source"] == {
        "title": first_metadata["title"],
        "document_type": "kanun",
        "legal_dates": {"validity_start": "2026-01-01", "validity_end": None},
        "version_unknown": False,
    }
    payload: dict[str, JsonValue] = {
        "original_metadata_catalogue": cast(JsonValue, encoded),
        "original_source_metadata": shared,
    }
    assert expand_source_metadata(payload) == full
    assert full == before
    assert len(json.dumps(payload)) < len(json.dumps(full))
    assert all(
        row["metadata"] != full[index]["metadata"] for index, row in enumerate(encoded)
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("version_unknown", True),
        ("legal_dates", {}),
        ("title", "Another title"),
        ("article_no", None),
    ],
)
def test_fields_that_differ_or_are_missing_remain_citation_specific(
    field: str, value: JsonValue
) -> None:
    _, _, originals = records()
    full = catalogue(originals)
    metadata = full[-1]["metadata"]
    assert isinstance(metadata, dict)
    metadata[field] = value
    second = full[-2]["metadata"]
    assert isinstance(second, dict)
    second.pop(field, None)
    encoded, shared = share_source_metadata(full)
    assert field not in cast(dict[str, JsonValue], shared["owned-source"])
    assert (
        expand_source_metadata(
            {
                "original_metadata_catalogue": cast(JsonValue, encoded),
                "original_source_metadata": shared,
            }
        )
        == full
    )


@pytest.mark.parametrize(
    "change", ["missing_source", "wrong_source", "override", "unused_source"]
)
def test_foreign_or_conflicting_metadata_binding_fails_closed(change: str) -> None:
    _, _, originals = records()
    encoded, shared = share_source_metadata(catalogue(originals))
    if change == "missing_source":
        shared.clear()
    elif change == "wrong_source":
        encoded[0]["source_metadata_ref"] = "foreign-source"
    elif change == "override":
        metadata = encoded[0]["metadata"]
        assert isinstance(metadata, dict)
        metadata["version_unknown"] = True
    else:
        shared["foreign-source"] = {"title": "Foreign title"}
    with pytest.raises(ValueError, match="Native source metadata"):
        expand_source_metadata(
            {
                "original_metadata_catalogue": cast(JsonValue, encoded),
                "original_source_metadata": shared,
            }
        )


def test_tuned_native_invocation_keeps_literal_text_once_and_stable_acquisition_prefix() -> (
    None
):
    ledger, context, originals = records()
    native = [turn("first", originals[:3]), turn("repeat", originals[:3])]
    before = [item.model_dump_json() for item in native]
    before_ledger = copy.deepcopy(ledger.export()["records"])
    llm = model(limit=1000000)
    adapter = ResearchModel(llm, context, lean_native_mode=True)
    first_prompt, _, _ = adapter._fit_native_decision(
        view(turns=native, original_evidence=originals[:3])
    )
    adapter.decide(view(turns=native, original_evidence=originals))
    prompt = llm.invoke.call_args.kwargs["prompt"]
    parsed = cast(list[dict[str, JsonValue]], payloads(prompt))
    host = parsed[-1]
    identity = expand_source_metadata(host)
    decoded = decode_compact_originals(parsed, identity, ledger)
    assert [row["text"] for row in decoded] == [row["text"] for row in originals]
    assert [row["metadata"] for row in decoded] == [
        row["metadata"] for row in originals
    ]
    assert ledger.completely_delivered(adapter.last_call_id or "") == set(range(1, 6))
    tools = [message for message in prompt if isinstance(message, ToolMessage)]
    old_tools = [
        message for message in first_prompt if isinstance(message, ToolMessage)
    ]
    assert [row.model_dump_json() for row in tools] == [
        row.model_dump_json() for row in old_tools
    ]
    assert len(json.loads(tools[0].content)["original_evidence"]) == 3
    assert "original_evidence" not in json.loads(tools[1].content)
    assert [
        row["citation"]
        for row in cast(list[dict[str, JsonValue]], host["original_evidence"])
    ] == [4, 5]
    assert llm.invoke.call_count == 1
    assert [item.model_dump_json() for item in native] == before
    assert ledger.export()["records"] == before_ledger


def test_evicted_history_is_delivered_in_full_and_metadata_alone_is_not_evidence() -> (
    None
):
    ledger, context, originals = records()
    llm = model(limit=1000000)
    adapter = ResearchModel(llm, context, lean_native_mode=True)
    adapter.decide(view(original_evidence=originals))
    parsed = cast(
        list[dict[str, JsonValue]], payloads(llm.invoke.call_args.kwargs["prompt"])
    )
    decoded = decode_compact_originals(
        parsed, expand_source_metadata(parsed[-1]), ledger
    )
    assert [row["text"] for row in decoded] == [row["text"] for row in originals]
    assert (
        decode_compact_originals(
            [
                {
                    "original_metadata_catalogue": cast(
                        JsonValue, expand_source_metadata(parsed[-1])
                    )
                }
            ],
            expand_source_metadata(parsed[-1]),
            ledger,
        )
        == []
    )


@pytest.mark.parametrize("profile", ["normal", "deep", "experimental"])
def test_protected_modes_cannot_enable_tuned_transport(profile: str) -> None:
    _, context, originals = records()
    context.services.update(asv3_workflow_variant="standard", research_profile=profile)
    assert not lossless_original_transport_enabled(context)
    adapter = ResearchModel(model(limit=1000000), context, lean_native_mode=True)
    state = view(turns=[turn("read", originals)], original_evidence=originals)
    before, schemas, output = adapter._fit_native_decision(state)
    context.services["unused_tuned_transport_flag"] = True
    after, actual_schemas, actual_output = adapter._fit_native_decision(state)
    assert [row.model_dump_json() for row in after] == [
        row.model_dump_json() for row in before
    ]
    assert actual_schemas == schemas and actual_output == output
    host = payloads(after)[-1]
    assert (
        "original_source_metadata" not in host
        and "original_metadata_catalogue" not in host
    )
