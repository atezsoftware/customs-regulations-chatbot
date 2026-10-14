"""File annotations reach models without altering canonical chunk evidence."""

import copy
import hashlib
import json
from dataclasses import replace
from typing import cast
from uuid import UUID

import pytest
from pydantic import JsonValue

from onyx.asv3.assertions import AssertionWitness, assertion_witness_valid
from onyx.asv3.corpus_tools import evidence_for_chunk
from onyx.asv3.cropped_handoff_transport import expand_cropped_handoff_transport
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.legistation_metadata_transport import LEGISTATION_METADATA_USAGE_RULE
from onyx.asv3.llm_adapter import ResearchModel, _NativeOriginalCatalogue
from onyx.asv3.models import (
    RunContext,
    compact_evidence_metadata,
    model_evidence_metadata,
)
from onyx.asv3.native_cache_projection import decode_compact_originals
from onyx.asv3.source_metadata_transport import (
    expand_source_metadata,
    share_source_metadata,
)
from onyx.asv3.workflow_variant import ASV3_TUNED_VARIANT
from onyx.db.asv3_corpus import CorpusChunk, CorpusSource
from onyx.regulatory.legistation_metadata import LegistationMetadata
from tests.unit.onyx.asv3.test_native_cache_projection import payloads
from tests.unit.onyx.asv3.test_native_model_adapter import model, turn, view

SOURCE_ID = UUID("04d319cb-94eb-4083-a6b0-000000009366")
REPEAL_NOTE = "Repeal by 9515 (from 01.02.2025)"
ANNOTATION = LegistationMetadata(
    document_type="Karar",
    subject="Kaynak açıklaması",
    official_title="9366 sayılı Karar",
    official_gazette_date="2025-01-27",
    validity_source="Kullanıcı tarafından doğrulanmış dosya metadata'sı",
    validity_status="Mülga",
    repeal_note=REPEAL_NOTE,
)
LLM_FIELDS = {"validity_status": "Mülga", "repeal_note": REPEAL_NOTE}


def chunk(number: int = 1) -> CorpusChunk:
    return CorpusChunk(
        id=f"canonical-{number}",
        source_id=SOURCE_ID,
        text=f"**MADDE {number}** — Özgün şart ve istisna.\n  東京",
        position=number - 1,
        projection_ordinal=number - 1,
        heading_path=("9366 sayılı Karar", f"MADDE {number}"),
        metadata={"article_no": str(number), "read_as_of_date": "2026-10-14"},
        validity_start=None,
        validity_end=None,
        status="active",
    )


def source(annotation: LegistationMetadata | None = ANNOTATION) -> CorpusSource:
    return CorpusSource(SOURCE_ID, "KARARLAR/9366.md", "file-9366", annotation)


def records(
    *, annotation: LegistationMetadata | None = ANNOTATION, count: int = 4
) -> tuple[EvidenceLedger, RunContext, list[dict[str, JsonValue]]]:
    ledger, context = EvidenceLedger(), RunContext()
    context.services.update(
        evidence=ledger,
        asv3_workflow_variant=ASV3_TUNED_VARIANT,
        research_profile="normal",
    )
    numbers = ledger.add(
        [
            evidence_for_chunk(source(annotation), chunk(number))
            for number in range(1, count + 1)
        ],
        context,
    )
    originals = cast(
        list[dict[str, JsonValue]], json.loads(ledger.serialize_records(numbers))
    )
    return ledger, context, originals


def test_file_annotation_does_not_change_text_hash_locator_citation_or_source() -> None:
    original = chunk()
    before = copy.deepcopy(original)
    plain = evidence_for_chunk(source(None), original)
    annotated = evidence_for_chunk(source(), original)
    assert original == before
    assert annotated.identity == plain.identity
    assert annotated.text == plain.text == original.text
    assert annotated.text_hash == hashlib.sha256(original.text.encode()).hexdigest()
    assert annotated.search_doc == plain.search_doc
    assert (
        annotated.metadata["canonical_metadata"] == plain.metadata["canonical_metadata"]
    )
    assert {
        key: value
        for key, value in annotated.metadata.items()
        if key != "legistation_metadata"
    } == plain.metadata
    assert annotated.metadata["legistation_metadata"] == ANNOTATION.model_dump(
        mode="json"
    )
    assert (
        model_evidence_metadata(annotated.metadata)["legistation_metadata"]
        == LLM_FIELDS
    )


@pytest.mark.parametrize("value", [None, {}, [], "Mülga", {"validity_status": 1}])
def test_absent_or_malformed_annotation_is_the_exact_previous_model_projection(
    value: object,
) -> None:
    baseline = evidence_for_chunk(source(None), chunk()).metadata
    supplied = {**baseline, "legistation_metadata": value}
    assert compact_evidence_metadata(supplied) == baseline
    assert model_evidence_metadata(supplied) == model_evidence_metadata(baseline)


def test_non_llm_file_fields_are_kept_in_ledger_but_not_sent_to_model() -> None:
    ledger, _, originals = records()
    item = ledger.get(1)
    assert item is not None
    annotation = item.metadata["legistation_metadata"]
    assert isinstance(annotation, dict) and len(annotation) == 7
    assert (
        cast(dict[str, JsonValue], originals[0]["metadata"])["legistation_metadata"]
        == LLM_FIELDS
    )
    only_title = replace(
        source(), legistation_metadata=LegistationMetadata(official_title="Title")
    )
    assert "legistation_metadata" not in model_evidence_metadata(
        evidence_for_chunk(only_title, chunk()).metadata
    )


def test_validity_annotations_remain_present_for_every_selected_chunk() -> None:
    _, _, originals = records(count=12)
    catalogue = [
        {
            key: row[key]
            for key in ("citation", "source_id", "chunk_id", "text_hash", "metadata")
        }
        for row in originals
    ]
    # An unrelated shared field makes factoring profitable; annotations stay per row.
    for row in catalogue:
        metadata = cast(dict[str, JsonValue], row["metadata"])
        metadata["title"] = "Long official title " * 20
    before = copy.deepcopy(catalogue)
    encoded, shared = share_source_metadata(catalogue)
    assert shared and "legistation_metadata" not in cast(
        dict[str, JsonValue], shared[str(SOURCE_ID)]
    )
    assert all(
        cast(dict[str, JsonValue], row["metadata"])["legistation_metadata"]
        == LLM_FIELDS
        for row in encoded
    )
    assert (
        expand_source_metadata(
            {
                "original_metadata_catalogue": cast(JsonValue, encoded),
                "original_source_metadata": shared,
            }
        )
        == before
    )
    assert catalogue == before


@pytest.mark.parametrize("variant", ["tuned", "cheap", "cropped"])
def test_actual_native_render_roundtrip_contains_two_fields_for_each_original(
    variant: str,
) -> None:
    ledger, context, originals = records(count=12)
    if variant != "tuned":
        context.services[f"asv3_exp_{variant}"] = True
    selected = model(limit=1_000_000)
    adapter = ResearchModel(selected, context, lean_native_mode=True)
    before = copy.deepcopy(ledger.export())
    prompt, _, _ = adapter._fit_native_decision(
        view(turns=[turn("read", originals)], original_evidence=originals)
    )
    parsed = cast(list[dict[str, JsonValue]], payloads(prompt))
    bound = _NativeOriginalCatalogue.bind(prompt, cheap_metadata=variant != "tuned")
    bound.validate(prompt)
    decoded = decode_compact_originals(parsed, bound.records, ledger)
    rendered = "\n".join(row.model_dump_json() for row in prompt)
    assert rendered.count(LEGISTATION_METADATA_USAGE_RULE) == 1
    assert [row["text"] for row in decoded] == [row["text"] for row in originals]
    assert all(
        cast(dict[str, JsonValue], row["metadata"])["legistation_metadata"]
        == LLM_FIELDS
        for row in decoded
    )
    assert ledger.export() == before
    selected.invoke.assert_not_called()
    if variant == "cropped":
        expanded = expand_cropped_handoff_transport(parsed[-1])
        assert expanded["original_metadata_catalogue"] == list(bound.records)
    # Independent assessments and repairs use the same conditional guidance.
    fitted, _, _ = adapter._fit(
        "Independent assessment",
        json.dumps(parsed[-1], ensure_ascii=False),
        [],
        max_tokens=100,
    )
    assert cast(str, fitted[0].content).count(LEGISTATION_METADATA_USAGE_RULE) == 1


@pytest.mark.parametrize("variant", ["tuned", "cheap", "cropped"])
def test_no_validity_annotation_preserves_exact_native_prompt(
    variant: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    for annotation in (
        None,
        LegistationMetadata(official_title="Only stored title"),
        LegistationMetadata(
            official_title="Only stored title", validity_status=" ", repeal_note=""
        ),
    ):
        _, context, originals = records(annotation=annotation)
        if variant != "tuned":
            context.services[f"asv3_exp_{variant}"] = True
        adapter = ResearchModel(model(limit=1_000_000), context, lean_native_mode=True)
        current = view(original_evidence=originals)
        prompt, schemas, output = adapter._fit_native_decision(current)
        before = ([row.model_dump_json() for row in prompt], schemas, output)
        assert LEGISTATION_METADATA_USAGE_RULE not in str(before)
        # Compare the same ledger against the previous unannotated prompt branch;
        # a different ledger legitimately has a different completion fingerprint.
        with monkeypatch.context() as patched:
            patched.setattr(
                "onyx.asv3.llm_adapter.has_legistation_annotations", lambda _: False
            )
            previous, previous_schemas, previous_output = adapter._fit_native_decision(
                current
            )
        assert before == (
            [row.model_dump_json() for row in previous],
            previous_schemas,
            previous_output,
        )
        independent, _, _ = adapter._fit(
            "Unchanged instruction",
            json.dumps({"evidence": originals}, ensure_ascii=False),
            [],
            max_tokens=100,
        )
        assert independent[0].content == "Unchanged instruction"


def test_annotation_footnote_is_not_a_literal_source_witness() -> None:
    annotated = evidence_for_chunk(source(), chunk())
    assert not assertion_witness_valid(
        AssertionWitness(citation=1, source_quote=REPEAL_NOTE), {1: annotated.text}
    )
    assert assertion_witness_valid(
        AssertionWitness(citation=1, source_quote="Özgün şart ve istisna"),
        {1: annotated.text},
    )


@pytest.mark.parametrize("encoded", [False, True])
def test_independent_assessment_has_conditional_rule_for_list_or_json_evidence(
    encoded: bool,
) -> None:
    _, context, originals = records()
    adapter = ResearchModel(model(limit=1_000_000), context, lean_native_mode=True)
    evidence = json.dumps(originals, ensure_ascii=False) if encoded else originals
    prompt, _, _ = adapter._fit(
        "Independent assessment",
        json.dumps({"evidence": evidence}, ensure_ascii=False),
        [],
        max_tokens=100,
    )
    assert cast(str, prompt[0].content).count(LEGISTATION_METADATA_USAGE_RULE) == 1


def test_complete_native_input_budget_counts_annotations() -> None:
    _, plain_context, plain = records(annotation=None)
    _, annotated_context, annotated = records()
    plain_adapter = ResearchModel(
        model(limit=1_000_000), plain_context, lean_native_mode=True, token_counter=len
    )
    annotated_adapter = ResearchModel(
        model(limit=1_000_000),
        annotated_context,
        lean_native_mode=True,
        token_counter=len,
    )
    plain_prompt, tools, _ = plain_adapter._fit_native_decision(
        view(original_evidence=plain)
    )
    annotated_prompt, annotated_tools, _ = annotated_adapter._fit_native_decision(
        view(original_evidence=annotated)
    )
    assert annotated_tools == tools
    assert annotated_adapter._input_cost(
        annotated_prompt, annotated_tools
    ) > plain_adapter._input_cost(plain_prompt, tools)
