"""The actual logo introduction permits a strict decision, never inherited labels."""

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from onyx.regulatory.amendments.annexes.context_dependencies import context_hash
from onyx.regulatory.indexing_jobs.vertex_batch import VertexBatchRequest
from onyx.regulatory.labeling.article_lead_in import article_lead_in_candidate
from onyx.regulatory.labeling.chapter_caption import chapter_caption_candidate
from onyx.regulatory.labeling.intake_abstention_repair import (
    build_abstention_repair,
    build_abstention_repair_audit,
    parse_abstention_repair_output,
    validate_abstention_repair,
    validate_stored_abstention_repair_audit,
)
from onyx.regulatory.labeling.intake_dependent_clause_repair import (
    DependentClauseRepairSource,
)
from onyx.regulatory.labeling.intake_logo_intro_repair import (
    legacy_logo_intro_repair_candidate,
    legacy_logo_intro_shape,
)
from onyx.regulatory.labeling.provider import TaxonomyDefinition
from onyx.regulatory.labeling.self_repeal import self_repeal_candidate
from tests.unit.onyx.regulatory.labeling.abstention_repair_test_helpers import line


@pytest.fixture
def frozen() -> dict[str, Any]:
    path = Path(__file__).parent / "fixtures/legacy_logo_intro_context.json"
    return json.loads(path.read_text())


def _sources(frozen: dict[str, Any]) -> list[DependentClauseRepairSource]:
    return [DependentClauseRepairSource(**row) for row in frozen["sources"]]


def _index(sources: list[DependentClauseRepairSource], position: int) -> int:
    return next(
        i for i, row in enumerate(sources) if row.snapshot["position"] == position
    )


def _changed(
    source: DependentClauseRepairSource,
    *,
    text: str | None = None,
    snapshot: dict[str, Any] | None = None,
) -> DependentClauseRepairSource:
    text = source.text if text is None else text
    snapshot = deepcopy(source.snapshot) if snapshot is None else snapshot
    return replace(
        source,
        text=text,
        canonical_text_sha256=context_hash(text),
        snapshot=snapshot,
        current_text=text,
        current_snapshot=deepcopy(snapshot),
    )


def test_actual_27_members_preserve_child_a_heading_and_other_body_facts(
    frozen: dict[str, Any],
) -> None:
    sources = _sources(frozen)
    before = deepcopy(sources)
    target = sources[_index(sources, 142)]
    assert legacy_logo_intro_shape(target.text)
    assert legacy_logo_intro_repair_candidate(chunk_id=target.chunk_id, sources=sources)
    assert not chapter_caption_candidate(target.text)
    assert not self_repeal_candidate(target.text)
    assert not article_lead_in_candidate(target.text)
    a = sources[_index(sources, 143)].snapshot["chunk_metadata"]
    assert a.get("section_title") is None
    assert "section_title" not in a["intake_legal_metadata"]["fields"]
    assert (
        a["intake_legal_metadata"]["fields"]["paragraph_no"]["evidence_scope"]
        == "heading"
    )
    assert sources == before
    base = VertexBatchRequest.model_validate(frozen["base_request"])
    assert base.request_hash == frozen["actual_base_request_hash"]
    repair = build_abstention_repair(
        base_request=base,
        rejected_response=frozen["base_response"],
        text=target.text,
        chunk_id=target.chunk_id,
        source_hash=target.canonical_text_sha256,
        taxonomy=TaxonomyDefinition.model_validate(frozen["taxonomy"]),
        provider_binding=frozen["provider_binding"],
    )
    original, reconsideration = (
        json.loads(base.prompt),
        json.loads(repair.request.prompt),
    )
    assert reconsideration["target"] == original["target"]
    assert (
        reconsideration["interpretation_context"] == original["interpretation_context"]
    )
    assert repair.request.generation_config == base.generation_config
    for position in range(141, 150):
        assert (
            sources[_index(sources, position)].text
            in original["interpretation_context"]
        )


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "duplicate",
        "members",
        "duplicate-position",
        "foreign-file",
        "inactive",
        "current-text",
        "current-metadata",
        "current-dates",
        "hash",
        "context-derived",
        "context-version",
        "orphan-article-fact",
        "operative-target",
        "wrong-type",
        "heading-depth",
        "metadata-heading",
        "wrong-paragraph",
        "article",
        "table",
        "derived",
        "version",
        "stale-audit",
        "missing-audit",
        "forged-body",
        "foreign-witness",
        "heading-substitution",
        "bad-request-hash",
        "reference-fact",
        "a-invented-section",
        "a-wrong-paragraph-scope",
        "b-missing-section",
        "b-heading-substitution",
        "d-body-substitution",
        "e-wrong-prefix",
        "child-parent",
        "child-label",
        "child-gap",
        "child-extra-body",
        "prior-title",
        "prior-appendix",
        "next-title",
        "next-appendix",
        "missing-boundary",
        "wrong-child-version",
    ],
)
def test_equal_current_snapshots_cannot_replace_scope_or_honest_evidence(
    frozen: dict[str, Any], change: str
) -> None:
    sources = _sources(frozen)
    index = _index(sources, 142)
    target = sources[index]
    if change in {"missing", "duplicate", "members", "duplicate-position"}:
        if change == "missing":
            sources.pop(0)
        elif change == "duplicate":
            sources.append(target)
        elif change == "members":
            target.snapshot["context_member_ids"].append("not-a-member")
        else:
            snapshot = deepcopy(sources[0].snapshot)
            snapshot["position"] = 142
            sources[0] = _changed(sources[0], snapshot=snapshot)
    elif change in {
        "foreign-file",
        "inactive",
        "current-text",
        "current-metadata",
        "current-dates",
        "hash",
        "context-derived",
        "context-version",
        "orphan-article-fact",
    }:
        source = sources[0]
        if change == "foreign-file":
            sources[0] = replace(source, file_id="foreign-file")
        elif change == "inactive":
            sources[0] = replace(source, active=False)
        elif change == "current-text":
            sources[0] = replace(source, current_text=source.text + " changed")
        elif change in {"current-metadata", "current-dates"}:
            current = deepcopy(source.current_snapshot)
            if change == "current-dates":
                current["validity_start_date"] = "2026-11-03"
            else:
                current["chunk_metadata"]["paragraph_no"] = "9"
            sources[0] = replace(source, current_snapshot=current)
        else:
            sources[0] = replace(source, canonical_text_sha256="f" * 64)
    else:
        position = (
            143
            if change.startswith("a-")
            else 144
            if change.startswith("b-")
            else 147
            if change.startswith("d-")
            else 148
            if change.startswith("e-")
            else 141
            if change.startswith("prior-")
            else 149
            if change.startswith("next-") or change == "missing-boundary"
            else 145
            if change.startswith("child-") or change == "wrong-child-version"
            else 142
        )
        index = _index(sources, position)
        source = sources[index]
        snapshot = deepcopy(source.snapshot)
        meta = snapshot["chunk_metadata"]
        fields = meta["intake_legal_metadata"]["fields"]
        text = source.text
        if change in {"context-derived", "context-version"}:
            index = 0
            source = sources[index]
            snapshot = deepcopy(source.snapshot)
            meta = snapshot["chunk_metadata"]
            meta[
                "chunk_variant"
                if change == "context-derived"
                else "intake_chunking_version"
            ] = "derived" if change == "context-derived" else "4"
        elif change == "orphan-article-fact":
            fields["article_no"] = deepcopy(fields["paragraph_no"])
        elif change == "operative-target":
            text += "\nBaşvuru yapan kişiler ödeme yapmakla yükümlüdür."
        elif change == "wrong-type":
            snapshot["chunk_type"] = "free_text"
        elif change == "heading-depth":
            snapshot["heading_path"].append("Başka bölüm")
            meta["heading_path"] = deepcopy(snapshot["heading_path"])
        elif change == "metadata-heading":
            meta["heading_path"] = ["Başka belge"]
        elif change == "wrong-paragraph":
            meta["paragraph_no"] = "2"
        elif change in {"article", "table"}:
            meta["article_no" if change == "article" else "table_index"] = "1"
        elif change == "derived":
            meta["chunk_variant"] = "derived"
        elif change in {"version", "wrong-child-version"}:
            meta["intake_chunking_version"] = "4"
        elif change == "stale-audit":
            meta["intake_legal_metadata"]["input_stale"] = True
        elif change == "missing-audit":
            fields.pop("paragraph_no")
        elif change in {
            "forged-body",
            "foreign-witness",
            "heading-substitution",
            "bad-request-hash",
            "reference-fact",
        }:
            fact = fields["paragraph_no"]
            key, value = {
                "forged-body": ("evidence_quote", "1. İlgisiz bölüm"),
                "foreign-witness": ("source_chunk_id", sources[0].chunk_id),
                "heading-substitution": ("evidence_scope", "heading"),
                "bad-request-hash": ("request_hash", "bad"),
                "reference-fact": ("reference_quote", "Başka belge"),
            }[change]
            fact[key] = value
        elif change == "a-invented-section":
            meta["section_title"] = "1. Logo Tasarımı"
            fields["section_title"] = deepcopy(fields["paragraph_no"])
        elif change == "a-wrong-paragraph-scope":
            fields["paragraph_no"]["evidence_scope"] = "body"
        elif change == "b-missing-section":
            meta.pop("section_title")
            fields.pop("section_title")
        elif change == "b-heading-substitution":
            fields["paragraph_no"].update(
                evidence_scope="heading", source_chunk_id=source.chunk_id
            )
        elif change == "d-body-substitution":
            fields["clause_label"].update(
                evidence_scope="body", evidence_quote=source.text
            )
        elif change == "e-wrong-prefix":
            text = text.replace("Ürünün etiketi", "Vergi tutarı")
        elif change == "child-parent":
            snapshot["heading_path"][2] = "2. Başka kapsam"
            meta["heading_path"] = deepcopy(snapshot["heading_path"])
        elif change == "child-label":
            meta["clause_label"] = "d"
        elif change in {"child-gap", "missing-boundary"}:
            snapshot["position"] += 1000
        elif change == "child-extra-body":
            text += "\nMADDE 8 – Başka kapsam."
        elif change in {"prior-title", "next-title"}:
            text += " Başka kapsam"
        elif change in {"prior-appendix", "next-appendix"}:
            meta["appendix_label"] = "EK v"
        sources[index] = _changed(source, text=text, snapshot=snapshot)
    assert not legacy_logo_intro_repair_candidate(
        chunk_id=target.chunk_id, sources=sources
    )


def test_continued_abstention_and_context_only_quotes_never_complete(
    frozen: dict[str, Any],
) -> None:
    target = next(
        source
        for source in _sources(frozen)
        if source.chunk_id == frozen["target_chunk_id"]
    )
    taxonomy = TaxonomyDefinition.model_validate(frozen["taxonomy"])
    repair = build_abstention_repair(
        base_request=VertexBatchRequest.model_validate(frozen["base_request"]),
        rejected_response=frozen["base_response"],
        text=target.text,
        chunk_id=target.chunk_id,
        source_hash=target.canonical_text_sha256,
        taxonomy=taxonomy,
        provider_binding=frozen["provider_binding"],
    )
    output = parse_abstention_repair_output(
        [line(repair, frozen["base_response"])], repair=repair, on_progress=lambda: None
    )
    audit = build_abstention_repair_audit(
        output=output,
        repair=repair,
        output_uri="fixture://continued",
        text=target.text,
        taxonomy=taxonomy,
    )
    with pytest.raises(ValueError, match="accepted assignments"):
        validate_stored_abstention_repair_audit(
            audit,
            base_request=repair.base_request,
            text=target.text,
            chunk_id=target.chunk_id,
            source_hash=target.canonical_text_sha256,
            taxonomy=taxonomy,
            provider_binding=repair.provider_binding,
            labels=[],
            assignments=[],
        )
    for label, quote in (
        ("unknown-label", target.text),
        (
            taxonomy.labels[0].id,
            "c) Logoda kullanılacak yazı tipi Quicksand Bold olur.",
        ),
    ):
        response = json.dumps(
            {
                "labels": [{"label_id": label, "evidence_quote": quote}],
                "abstained": False,
            }
        )
        with pytest.raises(ValueError):
            validate_abstention_repair(
                response, repair=repair, text=target.text, taxonomy=taxonomy
            )
