import json
from typing import cast
from uuid import uuid4

import pytest
from pydantic import JsonValue

from onyx.document_index.publication_models import IndexedProjectionEvidence
from onyx.regulatory.publication_baseline import observed_baseline_binding
from tests.unit.onyx.document_index.elasticsearch.test_observed_publication import (
    observed,
)
from tests.unit.onyx.regulatory.indexing_jobs.owned_publication_test_helpers import (
    canonical_row,
    writer_inputs,
)


def baseline_case():
    file_id = uuid4()
    inputs = writer_inputs(file_id, [canonical_row(file_id, 5, "Legal text")])
    index = observed().observed_index
    source = json.loads(observed().source_json)
    source.update(
        document_id=str(file_id),
        regulatory_chunk_id="row-5",
        validity_start_date=None,
        validity_end_date=None,
        doc_summary="",
        chunk_context="",
        metadata_suffix="",
        heading_path=inputs.canonical[0].heading_path,
        source_links=json.dumps({0: ""}),
        image_file_id=None,
    )
    evidence = IndexedProjectionEvidence(
        index=index,
        source_json=json.dumps(source),
        frozen_projection=None,
        payload_sha256=None,
    )
    return inputs, evidence


def test_baseline_retains_exact_source_and_vectors_without_encoder_receipt() -> None:
    inputs, evidence = baseline_case()
    binding = observed_baseline_binding(evidence, inputs.canonical)
    assert json.loads(binding.projection.source_json) == json.loads(
        evidence.source_json
    )
    assert binding.representation_text == "Legal text"
    assert binding.context is None
    assert binding.dependency_ids == []
    assert binding.projection.model_dump()["evidence_kind"] == "observed-v1"


@pytest.mark.parametrize(
    "field,value",
    [
        ("content", "wrong source"),
        ("regulatory_chunk_id", "other chunk"),
        ("document_id", str(uuid4())),
        ("chunk_index", 15),
        ("heading_path", ["wrong article"]),
    ],
)
def test_baseline_rejects_mismatched_existing_source(field: str, value: object) -> None:
    inputs, evidence = baseline_case()
    source = json.loads(evidence.source_json)
    source[field] = value
    with pytest.raises(ValueError):
        observed_baseline_binding(
            evidence.model_copy(update={"source_json": json.dumps(source)}),
            inputs.canonical,
        )


@pytest.mark.parametrize(
    "field,value", [("source_links", '{"0":"wrong"}'), ("image_file_id", "wrong-image")]
)
def test_baseline_checks_citation_evidence_before_publication(
    field: str, value: str
) -> None:
    inputs, evidence = baseline_case()
    source = json.loads(evidence.source_json)
    source[field] = value
    with pytest.raises(ValueError, match="source/image evidence"):
        observed_baseline_binding(
            evidence.model_copy(update={"source_json": json.dumps(source)}),
            inputs.canonical,
        )


def test_inventory_audit_distinguishes_legacy_ready_and_missing_source() -> None:
    from onyx.regulatory.publication_baseline import audit_baseline_inventory

    inputs, evidence = baseline_case()
    report = audit_baseline_inventory(inputs.canonical, [evidence], [])
    assert report.state == "legacy"
    assert report.retained_count == 1
    binding = observed_baseline_binding(evidence, inputs.canonical)
    evidence = evidence.model_copy(update={"observed_projection": binding.projection})
    ready = audit_baseline_inventory(inputs.canonical, [evidence], [binding])
    assert ready.state == "ready"
    assert ready.source_sha256 == report.source_sha256
    assert ready.vectors_sha256 == report.vectors_sha256
    missing = audit_baseline_inventory(inputs.canonical, [], [])
    assert missing.state == "unresolved"
    assert missing.issues[0].code == "missing_indexed_source"


def test_inventory_audit_keeps_legacy_digests_without_retaining_decoded_vectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import weakref

    from onyx.document_index.publication_models import publication_digest
    from onyx.regulatory import publication_baseline as baseline

    class TrackedVector(list[float]):
        pass

    inputs, item = baseline_case()
    sources = []
    evidence = []
    bindings = []
    canonical = []
    for ordinal in (5, 1, 12, 2, 9, 3):
        source = json.loads(item.source_json)
        source.update(chunk_index=ordinal, regulatory_chunk_id=f"row-{ordinal}")
        sources.append(source)
        row = inputs.canonical[0].model_copy(
            update={"id": f"row-{ordinal}", "projection_ordinal": ordinal}
        )
        current = item.model_copy(update={"source_json": json.dumps(source)})
        binding = observed_baseline_binding(current, [row])
        canonical.append(row)
        evidence.append(
            current.model_copy(update={"observed_projection": binding.projection})
        )
        bindings.append(binding)
    ordered = sorted(sources, key=lambda source: str(source["chunk_index"]))
    source_digest = publication_digest(ordered)
    vector_digest = publication_digest(
        [
            {
                "ordinal": s["chunk_index"],
                "id": s["regulatory_chunk_id"],
                "content": s["content_vector"],
                "title": s["title_vector"],
            }
            for s in ordered
        ]
    )
    original = baseline.publication_source
    tracked: list[weakref.ReferenceType[TrackedVector]] = []

    def parse(value: str) -> dict[str, JsonValue]:
        source = original(value)
        source["content_vector"] = TrackedVector(
            cast(list[float], source["content_vector"])
        )
        tracked.append(weakref.ref(source["content_vector"]))
        assert sum(ref() is not None for ref in tracked) <= 3, (
            "audit retains decoded inventory"
        )
        return source

    monkeypatch.setattr(baseline, "publication_source", parse)
    report = baseline.audit_baseline_inventory(canonical, evidence, bindings)
    assert report.state == "ready" and report.issues == []
    assert report.source_sha256 == source_digest
    assert report.vectors_sha256 == vector_digest


@pytest.mark.parametrize("recorded_split", [False, True])
def test_observed_image_retains_own_asset_with_exact_split_parent(
    recorded_split: bool,
) -> None:
    inputs, evidence = baseline_case()
    image = inputs.canonical[0].model_copy(
        update={
            "metadata": {
                "chunk_variant": "image_companion",
                "bound_to_regulatory_chunk_id": "old-parent",
                "image_file_id": "asset-1",
            }
        }
    )
    parent = image.model_copy(
        update={
            "id": "parent-part",
            "projection_ordinal": 6,
            "metadata": {
                "oversized_split": {
                    "source_chunk_id": "old-parent",
                    "part": 1,
                    "parts": 2,
                },
            }
            if recorded_split
            else {},
        }
    )
    source = json.loads(evidence.source_json)
    source["image_file_id"] = "asset-1"
    binding = observed_baseline_binding(
        evidence.model_copy(update={"source_json": json.dumps(source)}), [parent, image]
    )
    assert binding.dependency_ids == [parent.id]
    assert binding.representation_metadata["bound_to_regulatory_chunk_id"] == parent.id
    assert binding.representation_metadata["image_file_id"] == "asset-1"


@pytest.mark.parametrize(
    "split,unrecorded", [(False, False), (True, False), (False, True)]
)
@pytest.mark.parametrize("missing_heading", [False, True])
def test_observed_image_activation_validates_own_citation_asset(
    split: bool,
    unrecorded: bool,
    missing_heading: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from unittest.mock import MagicMock

    from sqlalchemy.orm import Session

    from onyx.db.models import RegulatoryChunk
    from onyx.db.regulatory_context_projections import activate_temporal_projection
    from onyx.regulatory.amendments.annexes.staging import canonical_snapshot_rows

    inputs, evidence = baseline_case()
    parent = inputs.canonical[0].model_copy(
        update={
            "id": "parent",
            "projection_ordinal": 6,
            "metadata": {
                "oversized_split": {
                    "source_chunk_id": "old-parent",
                    "part": 1,
                    "parts": 2,
                }
            }
            if split
            else {},
        }
    )
    image = inputs.canonical[0].model_copy(
        update={
            "metadata": {
                "chunk_variant": "image_companion",
                "bound_to_regulatory_chunk_id": "old-parent"
                if split or unrecorded
                else "parent",
                "image_file_id": "asset-1",
            }
        }
    )
    source = json.loads(evidence.source_json)
    source["image_file_id"] = "asset-1"
    if missing_heading:
        parent = parent.model_copy(update={"heading_path": []})
        image = image.model_copy(update={"heading_path": []})
        source["heading_path"] = None
    binding = observed_baseline_binding(
        evidence.model_copy(update={"source_json": json.dumps(source)}), [parent, image]
    )
    rows = {r.id: r for r in canonical_snapshot_rows([parent, image])}
    session = MagicMock(spec=Session)
    session.get.side_effect = lambda model, identifier, **_kw: (
        rows.get(identifier) if model is RegulatoryChunk else None
    )
    session.scalar.return_value = None
    monkeypatch.setattr(
        "onyx.db.regulatory_context_projections.get_indexed_temporal_projection",
        lambda *_a, **_kw: None,
    )
    monkeypatch.setattr(
        "onyx.db.regulatory_canonical_revisions.retain_canonical_revision",
        lambda *_a, **_kw: uuid4(),
    )
    monkeypatch.setattr(
        "onyx.db.regulatory_annex_changes.capture_canonical_scope",
        lambda *_a, **_kw: [parent, image],
    )
    activate_temporal_projection(session, user_file_id=inputs.file.id, binding=binding)
    persisted = session.add.call_args.args[0]
    assert persisted.payload["dependency_ids"] == ["parent"]
    assert (
        json.loads(persisted.payload["projection"]["source_json"])["image_file_id"]
        == "asset-1"
    )


def test_cyclic_legacy_dependencies_are_rejected_before_closing_read_gate() -> None:
    from onyx.regulatory.publication_baseline import audit_baseline_inventory

    inputs, evidence = baseline_case()
    first = inputs.canonical[0].model_copy(
        update={
            "metadata": {
                "chunk_variant": "hierarchical_aggregate",
                "hierarchy_root_path": ["Legal"],
                "source_regulatory_chunk_ids": ["row-6"],
            }
        }
    )
    second = first.model_copy(
        update={
            "id": "row-6",
            "projection_ordinal": 6,
            "metadata": {**first.metadata, "source_regulatory_chunk_ids": [first.id]},
        }
    )
    other = json.loads(evidence.source_json)
    other.update(regulatory_chunk_id=second.id, chunk_index=6)
    report = audit_baseline_inventory(
        [first, second],
        [evidence, evidence.model_copy(update={"source_json": json.dumps(other)})],
        [],
    )
    assert report.state == "unresolved"
    assert any(issue.code == "dependency_cycle" for issue in report.issues)


def test_existing_but_unrelated_image_parent_is_not_accepted_as_provenance() -> None:
    inputs, evidence = baseline_case()
    image = inputs.canonical[0].model_copy(
        update={
            "metadata": {
                "chunk_variant": "image_companion",
                "bound_to_regulatory_chunk_id": "parent",
                "image_file_id": "asset",
            }
        }
    )
    parent = image.model_copy(
        update={
            "id": "parent",
            "text": "unrelated provision",
            "metadata": {},
            "projection_ordinal": 6,
        }
    )
    source = json.loads(evidence.source_json)
    source["image_file_id"] = "asset"
    with pytest.raises(ValueError, match="image membership"):
        observed_baseline_binding(
            evidence.model_copy(update={"source_json": json.dumps(source)}),
            [parent, image],
        )


def test_observed_reader_keeps_exact_historical_citation_ordinal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from datetime import date, datetime, timezone
    from unittest.mock import MagicMock

    from onyx.db import regulatory_public_reads as reads
    from onyx.regulatory.amendments.annexes.staging import canonical_snapshot_rows

    inputs, evidence = baseline_case()
    binding = observed_baseline_binding(evidence, inputs.canonical)
    boundary = date(2027, 1, 1)
    source = json.loads(binding.projection.source_json)
    source["validity_end_date"] = int(
        datetime(2027, 1, 1, tzinfo=timezone.utc).timestamp()
    )
    projection = type(binding.projection).model_validate(
        {**binding.projection.model_dump(), "source_json": json.dumps(source)}
    )
    binding = binding.model_copy(
        update={"projection": projection, "effective_end": boundary}
    )
    canonical = canonical_snapshot_rows(
        [inputs.canonical[0].model_copy(update={"validity_end_date": boundary})]
    )[0]
    session = MagicMock()
    session.scalars.return_value = [canonical]
    session.get.return_value = canonical
    monkeypatch.setattr(
        reads, "load_file_temporal_bindings", lambda *_a, **_kw: [binding]
    )
    assert reads.load_public_temporal_bindings(
        session, inputs.file.id, index=binding.index, as_of_date=date(2026, 12, 31)
    ) == [binding]
    assert (
        reads.load_public_temporal_bindings(
            session, inputs.file.id, index=binding.index, as_of_date=boundary
        )
        == []
    )
    assert reads.citation_chunk_as_of_date(
        session,
        document_id=str(inputs.file.id),
        chunk_id=5,
        search_settings_id=11,
        index_name=binding.index.index_name,
    ) == date(2026, 12, 31)
    assert (
        reads.citation_chunk_as_of_date(
            session,
            document_id=str(inputs.file.id),
            chunk_id=99,
            search_settings_id=11,
            index_name=binding.index.index_name,
        )
        is None
    )


@pytest.mark.parametrize("fault", [None, "ambiguous", "contradictory_split"])
def test_exact_image_source_recovery_requires_unambiguous_evidence(
    fault: str | None,
) -> None:
    inputs, evidence = baseline_case()
    image = inputs.canonical[0].model_copy(
        update={
            "metadata": {
                "chunk_variant": "image_companion",
                "bound_to_regulatory_chunk_id": "missing-parent",
                "image_file_id": "asset",
                "image_alt": "caption",
            }
        }
    )
    parent = image.model_copy(
        update={"id": "parent", "projection_ordinal": 6, "metadata": {}}
    )
    rows = [parent, image]
    if fault == "ambiguous":
        rows.append(
            parent.model_copy(update={"id": "duplicate", "projection_ordinal": 7})
        )
    if fault == "contradictory_split":
        rows[0] = parent.model_copy(
            update={
                "metadata": {"oversized_split": {"source_chunk_id": "other-parent"}}
            }
        )
    source = json.loads(evidence.source_json)
    source["image_file_id"] = "asset"
    actual = evidence.model_copy(update={"source_json": json.dumps(source)})
    if fault:
        with pytest.raises(ValueError, match="image membership"):
            observed_baseline_binding(actual, rows)
    else:
        original = [r.model_dump(mode="json") for r in rows]
        binding = observed_baseline_binding(actual, rows)
        assert binding.dependency_ids == [parent.id]
        assert binding.projection.source_json == actual.source_json
        assert [r.model_dump(mode="json") for r in rows] == original


def test_observed_aggregate_accepts_exact_recorded_article_title_without_metadata_changes() -> (
    None
):
    inputs, evidence = baseline_case()
    root = ["Karar 7238", "MADDE 4"]
    title = "Ortak Komisyonun Amaçları"
    first = inputs.canonical[0].model_copy(
        update={
            "id": "first",
            "position": 1,
            "projection_ordinal": 1,
            "text": "**Madde 4**\n**Ortak Komisyonun Amaçları**\nKomisyonun amaçları:",
            "heading_path": root,
            "metadata": {"chunk_variant": "atomic"},
        }
    )
    second = first.model_copy(
        update={
            "id": "second",
            "position": 2,
            "projection_ordinal": 2,
            "text": "a) İşlemleri kolaylaştırmak;",
            "heading_path": [root[0], root[1] + " - " + title, "a)"],
        }
    )
    aggregate = inputs.canonical[0].model_copy(
        update={
            "chunk_type": "hierarchical_aggregate",
            "heading_path": root,
            "text": root[-1]
            + " - "
            + title
            + "\n\n"
            + first.text
            + "\n\n"
            + second.text,
            "metadata": {
                "chunk_variant": "hierarchical_aggregate",
                "hierarchy_root_path": root,
                "article_title": title,
                "source_regulatory_chunk_ids": [first.id, second.id],
            },
        }
    )
    source = json.loads(evidence.source_json)
    source.update(content=aggregate.text, heading_path=root)
    actual = evidence.model_copy(update={"source_json": json.dumps(source)})
    rows = [first, second, aggregate]
    original = [r.model_dump(mode="json") for r in rows]
    binding = observed_baseline_binding(actual, rows)
    assert binding.dependency_ids == [first.id, second.id]
    assert binding.projection.source_json == actual.source_json
    assert [r.model_dump(mode="json") for r in rows] == original


@pytest.mark.parametrize("indexed_heading", [None, [], False, ""])
def test_empty_canonical_heading_accepts_only_null_or_empty_index_heading(
    indexed_heading: object,
) -> None:
    inputs, evidence = baseline_case()
    canonical = inputs.canonical[0].model_copy(update={"heading_path": []})
    source = json.loads(evidence.source_json)
    source["heading_path"] = indexed_heading
    actual = evidence.model_copy(update={"source_json": json.dumps(source)})
    if indexed_heading is None or indexed_heading == []:
        binding = observed_baseline_binding(actual, [canonical])
        assert binding.projection.source_json == actual.source_json
        assert canonical.heading_path == []
    else:
        with pytest.raises(ValueError, match="identity, heading"):
            observed_baseline_binding(actual, [canonical])
