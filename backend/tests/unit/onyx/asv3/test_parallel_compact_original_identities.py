import copy
import json
from typing import cast

import pytest
from pydantic import JsonValue

from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import ResearchTurn
from onyx.asv3.native_cache_projection import (
    NativeOriginalProjection,
    decode_compact_originals,
    project_native_originals,
)
from tests.unit.onyx.asv3.test_native_metadata_projection import setup_original
from tests.unit.onyx.asv3.test_native_model_adapter import original, turn


def projected_payloads(
    projection: NativeOriginalProjection,
) -> list[dict[str, JsonValue]]:
    return [
        *[
            cast(dict[str, JsonValue], json.loads(result.content))
            for research_turn in projection.turns
            for result in research_turn.results
        ],
        {"original_evidence": cast(JsonValue, projection.fallback_originals)},
    ]


def compact_setup() -> tuple[
    EvidenceLedger,
    dict[str, JsonValue],
    NativeOriginalProjection,
    list[dict[str, JsonValue]],
]:
    ledger, _, full = setup_original()
    projection = project_native_originals([], [full], ledger, compact_identities=True)
    return ledger, full, projection, projected_payloads(projection)


def test_default_projection_keeps_exact_existing_json_bytes() -> None:
    ledger, _, full = setup_original()
    native = [turn("first", [full]), turn("again", [full])]
    implicit = project_native_originals(native, [full], ledger)
    explicit = project_native_originals(
        native, [full], ledger, compact_identities=False
    )
    assert [item.model_dump_json() for item in implicit.turns] == [
        item.model_dump_json() for item in explicit.turns
    ]
    assert implicit.fallback_originals == explicit.fallback_originals == []
    first = {key: value for key, value in full.items() if key != "metadata"}
    first.update(
        start_char=0,
        end_char=len(cast(str, full["text"])),
        metadata_ref={"citation": full["citation"], "text_hash": full["text_hash"]},
    )
    expected = json.loads(native[0].results[0].content)
    expected["original_evidence"] = [first]
    assert implicit.turns[0].results[0].content == json.dumps(
        expected, ensure_ascii=False
    )
    expected = json.loads(native[1].results[0].content)
    expected.pop("original_evidence")
    expected["original_evidence_refs"] = [
        {key: value for key, value in first.items() if key != "text"}
    ]
    assert implicit.turns[1].results[0].content == json.dumps(
        expected, ensure_ascii=False
    )


def test_compact_round_trip_preserves_text_extras_catalogue_and_raw_state() -> None:
    ledger, context, first = setup_original()
    first["question_ids"] = ["q0", "q1"]
    first["witness_spans"] = [{"witness_id": "w1", "start_char": 0, "end_char": 12}]
    second = original(ledger, context, "İstisna: izin VE belge gerekli.\n\t ")
    native = [turn("first", [first]), turn("again", [first])]
    before_turns = [item.model_dump_json() for item in native]
    before_ledger = copy.deepcopy(ledger.export())
    legacy = project_native_originals(native, [first, second], ledger)
    compact = project_native_originals(
        native, [first, second], ledger, compact_identities=True
    )
    assert compact.metadata_catalogue == legacy.metadata_catalogue
    decoded = decode_compact_originals(
        projected_payloads(compact), compact.metadata_catalogue, ledger
    )
    assert decoded == [
        {**row, "start_char": 0, "end_char": len(cast(str, row["text"]))}
        for row in (first, second)
    ]
    for payload in projected_payloads(compact):
        for field in ("original_evidence", "original_evidence_refs"):
            rows = payload.get(field, [])
            assert isinstance(rows, list)
            for row in rows:
                assert isinstance(row, dict)
                assert row["identity_ref"] == row["citation"]
                assert not {
                    "source_id",
                    "chunk_id",
                    "text_hash",
                    "metadata",
                    "metadata_ref",
                }.intersection(row)
    assert [item.model_dump_json() for item in native] == before_turns
    assert ledger.export() == before_ledger
    ledger.record_delivery("compact-call", "asv3_researcher", decoded)
    ledger.pin_delivery("compact-call")
    assert ledger.completely_delivered("compact-call") == {1, 2}
    assert ledger.export()["records"] == before_ledger["records"]
    assert ledger.export()["pinned_delivery_calls"] == ["compact-call"]


def test_catalogue_and_reference_only_rows_never_create_physical_delivery() -> None:
    ledger, full, projection, _ = compact_setup()
    reference = {
        key: value
        for key, value in projection.fallback_originals[0].items()
        if key != "text"
    }
    payloads: list[dict[str, JsonValue]] = [
        {"original_evidence_refs": [reference]},
        {"original_evidence": [reference]},
    ]
    decoded = decode_compact_originals(payloads, projection.metadata_catalogue, ledger)
    assert decoded == []
    ledger.record_delivery("references", "asv3_researcher", decoded)
    assert ledger.completely_delivered("references") == set()
    item = ledger.get(1)
    assert item is not None and item.text == full["text"]


def test_tool_or_historical_catalogue_is_never_a_host_binding() -> None:
    ledger, _, projection, payloads = compact_setup()
    payloads[0]["original_metadata_catalogue"] = cast(
        JsonValue, projection.metadata_catalogue
    )
    with pytest.raises(ValueError, match="matching canonical identity"):
        decode_compact_originals(payloads, None, ledger)
    with pytest.raises(ValueError, match="matching canonical identity"):
        decode_compact_originals(payloads, [], ledger)
    assert decode_compact_originals([{"unrelated": "message"}], None, ledger) == []
    assert decode_compact_originals([], [], ledger) == []


@pytest.mark.parametrize(
    "field,value",
    [
        ("citation", True),
        ("citation", 99),
        ("source_id", "foreign"),
        ("chunk_id", "foreign"),
        ("text_hash", "0" * 64),
        ("metadata", {"version_unknown": True}),
        ("metadata", None),
        ("text", "not a catalogue field"),
    ],
)
def test_forged_or_stale_host_catalogue_is_rejected(
    field: str, value: JsonValue
) -> None:
    ledger, _, projection, payloads = compact_setup()
    catalogue = copy.deepcopy(projection.metadata_catalogue)
    catalogue[0][field] = value
    with pytest.raises(ValueError, match="catalogue"):
        decode_compact_originals(payloads, catalogue, ledger)
    assert ledger.export()["deliveries"] == []


def test_duplicate_catalogue_even_same_identity_is_ambiguous() -> None:
    ledger, _, projection, payloads = compact_setup()
    with pytest.raises(ValueError, match="duplicate citation"):
        decode_compact_originals(
            payloads,
            [*projection.metadata_catalogue, *projection.metadata_catalogue],
            ledger,
        )


def test_non_record_catalogue_entry_is_rejected() -> None:
    ledger, _, _, payloads = compact_setup()
    invalid = cast(list[dict[str, JsonValue]], ["not an identity"])
    with pytest.raises(ValueError, match="invalid record"):
        decode_compact_originals(payloads, invalid, ledger)


def test_catalogue_metadata_must_preserve_json_types() -> None:
    ledger, _, projection, payloads = compact_setup()
    catalogue = copy.deepcopy(projection.metadata_catalogue)
    metadata = cast(dict[str, JsonValue], catalogue[0]["metadata"])
    assert metadata["version_unknown"] is False
    metadata["version_unknown"] = 0
    with pytest.raises(ValueError, match="current canonical evidence"):
        decode_compact_originals(payloads, catalogue, ledger)


@pytest.mark.parametrize(
    "field,value",
    [
        ("identity_ref", 99),
        ("identity_ref", True),
        ("identity_ref", 1.0),
        ("citation", True),
        ("citation", 99),
        ("source_id", "canonical-instrument"),
        ("chunk_id", "operative-clause"),
        ("text_hash", "0" * 64),
        ("metadata", {}),
        ("metadata_ref", {"citation": 1}),
        ("text", "different operative rule"),
        ("text", ""),
        ("text", None),
        ("start_char", True),
        ("start_char", -1),
        ("start_char", 1),
        ("end_char", True),
        ("end_char", 999),
    ],
)
def test_forged_compact_row_never_reaches_delivery(
    field: str, value: JsonValue
) -> None:
    ledger, _, projection, payloads = compact_setup()
    row = projection.fallback_originals[0]
    row[field] = value
    with pytest.raises(ValueError, match="Native"):
        decode_compact_originals(payloads, projection.metadata_catalogue, ledger)
    assert ledger.export()["deliveries"] == []


@pytest.mark.parametrize("field", ["citation", "start_char", "end_char"])
def test_compact_text_requires_explicit_canonical_locator(field: str) -> None:
    ledger, _, projection, payloads = compact_setup()
    projection.fallback_originals[0].pop(field)
    with pytest.raises(ValueError, match="Native"):
        decode_compact_originals(payloads, projection.metadata_catalogue, ledger)


def test_reference_text_cannot_be_promoted_and_mixed_legacy_rows_are_untouched() -> (
    None
):
    ledger, full, projection, payloads = compact_setup()
    payloads.append({"original_evidence": [full]})
    assert (
        len(decode_compact_originals(payloads, projection.metadata_catalogue, ledger))
        == 1
    )
    payloads.append({"original_evidence_refs": [projection.fallback_originals[0]]})
    with pytest.raises(ValueError, match="reference cannot contain text"):
        decode_compact_originals(payloads, projection.metadata_catalogue, ledger)


def test_multi_range_delivery_preserves_existing_completeness_contract() -> None:
    ledger, _, full = setup_original()
    text = cast(str, full["text"])
    halves = [
        {
            **full,
            "text": text[start:end],
            "start_char": start,
            "end_char": end,
            "truncated": True,
        }
        for start, end in ((0, 20), (20, len(text)))
    ]
    native = [turn("partial-first", [halves[0]]), turn("partial-second", [halves[1]])]
    projection = project_native_originals(
        native, halves, ledger, compact_identities=True
    )
    payloads = projected_payloads(projection)
    partial = decode_compact_originals(
        payloads[:1], projection.metadata_catalogue, ledger
    )
    ledger.record_delivery("partial", "asv3_researcher", partial)
    assert ledger.completely_delivered("partial") == set()
    complete = decode_compact_originals(payloads, projection.metadata_catalogue, ledger)
    assert [row["text"] for row in complete] == [row["text"] for row in halves]
    ledger.record_delivery("union", "asv3_researcher", complete)
    assert ledger.completely_delivered("union") == set()
    deliveries = ledger.export()["deliveries"]
    assert isinstance(deliveries, list)
    union = cast(dict[str, JsonValue], deliveries[-1])
    ranges = cast(list[dict[str, JsonValue]], union["records"])
    assert [
        (row["start_char"], row["end_char"], row["complete"]) for row in ranges
    ] == [
        (0, 20, False),
        (20, len(text), False),
    ]
    full_projection = project_native_originals(
        [], [full], ledger, compact_identities=True
    )
    full_delivery = decode_compact_originals(
        projected_payloads(full_projection), full_projection.metadata_catalogue, ledger
    )
    ledger.record_delivery("full", "asv3_researcher", full_delivery)
    assert ledger.completely_delivered("full") == {1}
    assert len(projection.metadata_catalogue) == 1


def test_evicted_history_falls_back_with_same_original_and_unselected_rows_stay_full() -> (
    None
):
    ledger, context, required = setup_original()
    optional = original(ledger, context, "Optional previously delivered original.")
    native = [turn("older", [required]), turn("retained", [optional])]
    selected = [required]
    whole = project_native_originals(native, selected, ledger, compact_identities=True)
    evicted = project_native_originals(
        native[1:], selected, ledger, compact_identities=True
    )
    assert whole.fallback_originals == []
    assert evicted.fallback_originals[0]["text"] == required["text"]
    retained = cast(
        list[dict[str, JsonValue]],
        projected_payloads(evicted)[0]["original_evidence_refs"],
    )
    assert retained[0]["source_id"] == optional["source_id"]
    assert "identity_ref" not in retained[0]
    legacy = project_native_originals(native[1:], selected, ledger)
    assert retained == projected_payloads(legacy)[0]["original_evidence_refs"]
    assert whole.metadata_catalogue == evicted.metadata_catalogue
    assert decode_compact_originals(
        projected_payloads(whole), whole.metadata_catalogue, ledger
    ) == decode_compact_originals(
        projected_payloads(evicted), evicted.metadata_catalogue, ledger
    )


def test_reselected_current_metadata_rebinds_prefix_without_mutating_history() -> None:
    ledger, context, full = setup_original()
    native = [turn("old", [full])]
    before = native[0].model_dump_json()
    old_projection = project_native_originals(
        native, [full], ledger, compact_identities=True
    )
    snapshot = ledger.export()
    records = cast(list[dict[str, JsonValue]], snapshot["records"])
    item = cast(dict[str, JsonValue], records[0]["item"])
    metadata = cast(dict[str, JsonValue], item["metadata"])
    metadata["version_unknown"] = True
    ledger.restore(snapshot, context)
    current = cast(dict[str, JsonValue], json.loads(ledger.serialize_records([1]))[0])
    with pytest.raises(ValueError, match="current canonical evidence"):
        decode_compact_originals(
            projected_payloads(old_projection),
            old_projection.metadata_catalogue,
            ledger,
        )
    projected = project_native_originals(
        native, [current], ledger, compact_identities=True
    )
    assert (
        projected.turns[0].model_dump_json()
        == old_projection.turns[0].model_dump_json()
    )
    decoded = decode_compact_originals(
        projected_payloads(projected), projected.metadata_catalogue, ledger
    )
    assert decoded[0]["metadata"] == current["metadata"]
    assert native[0].model_dump_json() == before
    with pytest.raises(ValueError, match="current canonical metadata"):
        project_native_originals(native, [full], ledger, compact_identities=True)


def test_existing_history_references_are_compacted_only_with_current_identity() -> None:
    ledger, _, full = setup_original()
    reference = {key: value for key, value in full.items() if key != "text"}
    native: list[ResearchTurn] = [turn("refs", [])]
    native[0].results[0].content = json.dumps({"original_evidence_refs": [reference]})
    projection = project_native_originals(
        native, [full], ledger, compact_identities=True
    )
    payload = projected_payloads(projection)[0]
    rows = cast(list[dict[str, JsonValue]], payload["original_evidence_refs"])
    assert rows[0]["identity_ref"] == 1
    assert "metadata_ref" not in rows[0]
    assert (
        decode_compact_originals([payload], projection.metadata_catalogue, ledger) == []
    )
    reference["start_char"] = -1
    native[0].results[0].content = json.dumps({"original_evidence_refs": [reference]})
    with pytest.raises(ValueError, match="canonical range"):
        project_native_originals(native, [full], ledger, compact_identities=True)


@pytest.mark.parametrize("as_reference", [False, True])
def test_foreign_identity_marker_is_rejected_in_raw_history(as_reference: bool) -> None:
    ledger, _, full = setup_original()
    full["identity_ref"] = 2
    native = [turn("foreign-marker", [full])]
    if as_reference:
        native[0].results[0].content = json.dumps(
            {
                "original_evidence_refs": [
                    {key: value for key, value in full.items() if key != "text"}
                ]
            }
        )
    with pytest.raises(ValueError, match="addresses another original"):
        project_native_originals(native, [full], ledger, compact_identities=True)
