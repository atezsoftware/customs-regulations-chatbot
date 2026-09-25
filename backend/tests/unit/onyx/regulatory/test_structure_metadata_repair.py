from uuid import uuid4

import pytest

from onyx.regulatory.amendments.annexes.publication_representations import _snapshot
from tests.unit.onyx.regulatory.indexing_jobs.owned_publication_test_helpers import (
    canonical_row,
)


def test_structure_repair_derives_identity_from_unique_original_body() -> None:
    from onyx.regulatory.structure_metadata_repair import plan_structure_repair

    file_id = uuid4()
    source = "ÖRNEK KANUN\n\nMADDE 2 - (1) Tanımlar:\n\ng) İlk tanım.\n\nğ) Ayrı tanım.\n\nEK MADDE 8 - (1) İlave hüküm."
    rows = [
        _snapshot(canonical_row(file_id, 1, "ğ) Ayrı tanım.")),
        _snapshot(canonical_row(file_id, 2, "EK MADDE 8 - (1) İlave hüküm.")),
    ]
    plan = plan_structure_repair(rows, markdown=source, source_file="kanun.md")
    assert not plan.unresolved
    assert len(plan.changes) == 2
    first, additional = [change.after for change in plan.changes]
    assert first.metadata["article_no"] == "2"
    assert first.metadata["paragraph_no"] == "1"
    assert first.metadata["clause_label"] == "ğ"
    assert additional.metadata["article_no"] == "EK 8"
    assert additional.metadata["paragraph_no"] == "1"
    assert first.chunk_type == "clause" and additional.chunk_type == "paragraph"
    for before, change in zip(rows, plan.changes):
        assert change.after.id == before.id
        assert change.after.text == before.text
        assert change.after.projection_ordinal == before.projection_ordinal
        assert change.after.position == before.position
        assert change.after.validity_start_date == before.validity_start_date
        assert source[change.source_start : change.source_end].strip() == before.text


def test_structure_repair_never_guesses_repeated_or_missing_text() -> None:
    from onyx.regulatory.structure_metadata_repair import plan_structure_repair

    file_id = uuid4()
    source = "KANUN\n\nMADDE 1 - (1) Hüküm.\n\na) Aynı metin.\n\nMADDE 2 - (1) Başka hüküm.\n\na) Aynı metin."
    rows = [
        _snapshot(canonical_row(file_id, 1, "a) Aynı metin.")),
        _snapshot(canonical_row(file_id, 2, "a) Kaynakta yok.")),
    ]
    plan = plan_structure_repair(rows, markdown=source, source_file="kanun.md")
    assert plan.changes == []
    assert set(plan.unresolved) == {row.id for row in rows}


def test_structure_repair_rejects_mixed_source_files() -> None:
    from onyx.regulatory.structure_metadata_repair import plan_structure_repair

    rows = [
        _snapshot(canonical_row(uuid4(), index, "MADDE 1 - (1) Hüküm."))
        for index in range(2)
    ]
    with pytest.raises(ValueError, match="one source"):
        plan_structure_repair(rows, markdown=rows[0].text, source_file="kanun.md")


def test_source_proof_accepts_only_parser_whitespace_joining() -> None:
    from onyx.regulatory.structure_metadata_repair import plan_structure_repair

    row = _snapshot(canonical_row(uuid4(), 1, "ğ) Tanım.\nh)"))
    source = "KANUN\n\nMADDE 2 - (1) Tanımlar:\n\nğ) Tanım.\n\nh)"
    plan = plan_structure_repair([row], markdown=source, source_file="kanun.md")
    assert not plan.unresolved
    assert plan.changes[0].after.metadata["clause_label"] == "ğ"
    assert plan.changes[0].after.text == row.text


def test_metadata_manifest_preserves_vectors_content_ordinals_and_prior_bindings() -> (
    None
):
    import json
    from dataclasses import replace

    from onyx.document_index.publication_models import publication_digest
    from onyx.regulatory.publication_baseline import observed_baseline_binding
    from onyx.regulatory.structure_metadata_repair import (
        plan_structure_repair,
        prepare_structure_metadata_manifest,
    )
    from tests.unit.onyx.regulatory.indexing_jobs.owned_publication_test_helpers import (
        OwnedAuthority,
    )
    from tests.unit.onyx.regulatory.test_publication_baseline import baseline_case

    inputs, evidence = baseline_case()
    before = inputs.canonical[0].model_copy(update={"text": "ğ) Ayrı tanım."})
    source = json.loads(evidence.source_json)
    source.update(content=before.text, blurb=before.text)
    evidence = evidence.model_copy(update={"source_json": json.dumps(source)})
    binding = observed_baseline_binding(evidence, [before])
    inputs = replace(
        inputs, canonical=[before], bindings=[binding], revisions={binding.id: uuid4()}
    )
    markdown = "KANUN\n\nMADDE 8 - (1) Tanımlar:\n\nğ) Ayrı tanım."
    plan = plan_structure_repair(
        [before], markdown=markdown, source_file=inputs.file.name
    )
    manifest = prepare_structure_metadata_manifest(
        OwnedAuthority(inputs.file.id).owner,
        inputs,
        plan=plan,
        markdown=markdown,
    )
    after = manifest.bindings[0]
    changed = json.loads(after.projection.source_json)
    assert manifest.canonical_after == [plan.changes[0].after]
    assert manifest.previous_binding_ids == [binding.id]
    assert after.id != binding.id
    assert after.projection.ordinal == binding.projection.ordinal
    from onyx.document_index.publication_models import ObservedPublicationProjection

    assert isinstance(after.projection, ObservedPublicationProjection)
    assert isinstance(binding.projection, ObservedPublicationProjection)
    assert after.projection.observed_index == binding.projection.observed_index
    assert after.representation_text == before.text
    assert after.representation_metadata["clause_label"] == "ğ"
    assert changed["heading_path"] == plan.changes[0].after.heading_path
    assert {k: v for k, v in changed.items() if k != "heading_path"} == {
        k: v for k, v in source.items() if k != "heading_path"
    }
    assert publication_digest(
        json.loads(binding.projection.source_json)
    ) == publication_digest(source)
    with pytest.raises(ValueError, match="source proof"):
        prepare_structure_metadata_manifest(
            OwnedAuthority(inputs.file.id).owner,
            inputs,
            plan=plan,
            markdown=markdown + " changed",
        )


def test_already_correct_structure_does_not_remove_nulls_or_add_duplicate_fields() -> (
    None
):
    from onyx.regulatory.chunker import RegulatoryChunker
    from onyx.regulatory.structure_metadata_repair import plan_structure_repair

    source = "KANUN\n\nMADDE 3 - (1) Hüküm."
    parsed = next(
        c
        for c in RegulatoryChunker().chunk_text(source, source_file="kanun.md").chunks
        if c.metadata.article_no == "3"
    )
    row = _snapshot(canonical_row(uuid4(), 2, parsed.text)).model_copy(
        update={
            "heading_path": parsed.metadata.heading_path,
            "chunk_type": parsed.metadata.chunk_type,
            "metadata": {"article_no": "3", "paragraph_no": "1", "clause_label": None},
        }
    )
    plan = plan_structure_repair([row], markdown=source, source_file="kanun.md")
    assert plan.changes == []
    assert plan.unchanged == [row.id]


@pytest.mark.parametrize("broken_chain", [False, True])
def test_amended_text_inherits_only_proven_structure_from_its_source_version(
    broken_chain: bool,
) -> None:
    from datetime import date

    from onyx.regulatory.structure_metadata_repair import plan_structure_repair

    file_id = uuid4()
    old = _snapshot(canonical_row(file_id, 1, "ğ) Eski tanım."))
    new = old.model_copy(
        update={
            "id": "new",
            "text": "ğ) Yeni tanım.",
            "source": "amendment",
            "supersedes_chunk_id": old.id,
            "validity_start_date": date(2026, 1, 1),
        }
    )
    old = old.model_copy(
        update={
            "status": "superseded",
            "superseded_by_chunk_id": "wrong" if broken_chain else new.id,
            "validity_end_date": date(2026, 1, 1),
        }
    )
    plan = plan_structure_repair(
        [new],
        history=[old],
        source_file="kanun.md",
        markdown="KANUN\n\nMADDE 8 - (1) Tanımlar:\n\nğ) Eski tanım.",
    )
    if broken_chain:
        assert plan.changes == [] and new.id in plan.unresolved
    else:
        assert not plan.unresolved
        change = plan.changes[0]
        assert change.after.text == new.text
        assert change.after.metadata["article_no"] == "8"
        assert change.after.metadata["paragraph_no"] == "1"
        assert change.after.metadata["clause_label"] == "ğ"
        assert change.lineage_ids == [new.id, old.id]
        assert change.after.heading_path[-1] == "ğ) Yeni tanım"
