from uuid import uuid4

import pytest

from onyx.regulatory.amendments.annexes.models import (
    AnnexExtraction,
    AnnexOriginalEvidence,
    AnnexReviewEvidence,
    ExtractedAnnexElement,
)
from onyx.regulatory.amendments.models import AmendmentInstruction


def test_full_annex_labels_group_once_without_truncating_suffix() -> None:
    from onyx.regulatory.amendments.annexes.analysis import group_annex_instructions

    instructions = [
        AmendmentInstruction(instruction_text=value)
        for value in [
            "EK-1/A değiştirilmiştir",
            "EK-1 değiştirilmiştir",
            "EK-1/A ikinci değişiklik",
            "Madde 5 değişti",
        ]
    ]
    groups = group_annex_instructions(instructions)
    assert [(group.annex_label, group.instruction_indices) for group in groups] == [
        ("ek:1/a", [0, 2]),
        ("ek:1", [1]),
    ]


def test_new_evidence_mapping_binds_actual_asset_and_selected_positions() -> None:
    from onyx.regulatory.amendments.annexes.evidence import (
        build_new_evidence_remapping,
        validate_new_evidence_remapping,
    )

    original = AnnexOriginalEvidence(
        file_id="package-file", sha256="a" * 64, mime_type="text/html", available=True
    )
    assert original.sha256 is not None
    extraction = AnnexExtraction(
        source_sha256=original.sha256,
        mime_type="text/html",
        elements=[ExtractedAnnexElement(kind="text", text="new")],
    )
    evidence = [
        AnnexReviewEvidence(
            id=uuid4(),
            side="new",
            kind="original",
            file_id="frozen-new",
            sha256=original.sha256,
            mime_type="text/html",
            byte_count=3,
            parent_file_id="package-file",
            parent_sha256=original.sha256,
        )
    ]
    asset_id = uuid4()
    mapping = build_new_evidence_remapping(
        new=extraction, evidence=evidence, asset_ids={"package-file": asset_id}
    )
    assert mapping.elements[0].source_asset_id == asset_id
    assert (
        mapping.elements[0].position == 0 and mapping.elements[0].image_file_ids == []
    )
    validate_new_evidence_remapping(
        mapping=mapping,
        new=extraction,
        evidence=evidence,
        asset_ids={"package-file": asset_id},
    )
    with pytest.raises(ValueError, match="mapping"):
        validate_new_evidence_remapping(
            mapping=mapping,
            new=extraction,
            evidence=evidence,
            asset_ids={"package-file": uuid4()},
        )


def test_semantic_visual_successor_can_reuse_only_exact_verified_input() -> None:
    from onyx.regulatory.amendments.annexes.context_dependencies import (
        compare_context_views,
    )
    from onyx.regulatory.amendments.annexes.models import (
        FrozenContextProjection,
        PreparedContextView,
    )

    old = FrozenContextProjection(
        canonical_chunk_id="old",
        source_snapshot_sha256="source",
        generation_path="normal",
        request_hashes=[],
        embedding_input_sha256="input",
        embedding_config_sha256="config",
        embedding_texts=["same"],
        canonical_text_sha256="same",
        metadata_sha256="old",
        vector_reuse_verified=True,
    )
    new = old.model_copy(
        update={
            "canonical_chunk_id": "new",
            "vector_reuse_verified": False,
            "metadata_sha256": "new",
        }
    )
    result = compare_context_views(
        old=PreparedContextView(projections=[old]),
        new=PreparedContextView(projections=[new]),
        direct_canonical_changes=["new"],
        canonical_predecessors={"new": "old"},
    )
    assert result.embedding_changes == []
    assert result.retire_history == ["old"]
    assert "embedding_input_unchanged" in result.reasons["new"]
    unverified = old.model_copy(update={"vector_reuse_verified": False})
    result = compare_context_views(
        old=PreparedContextView(projections=[unverified]),
        new=PreparedContextView(projections=[new]),
        direct_canonical_changes=["new"],
        canonical_predecessors={"new": "old"},
    )
    assert result.embedding_changes == ["new"]
    assert result.reasons["new"] == ["legacy_provenance_unavailable"]


def test_article_reference_to_annex_stays_legacy_and_instruments_do_not_merge() -> None:
    from onyx.regulatory.amendments.annexes.analysis import group_annex_instructions

    instructions = [
        AmendmentInstruction(
            article_reference="Madde 5",
            instruction_text='5 inci maddesinde yer alan "Liste" ibaresi "EK-1" şeklinde değiştirilmiştir.',
        ),
        AmendmentInstruction(
            target_source="Instrument A", instruction_text="EK-1 değiştirilmiştir"
        ),
        AmendmentInstruction(
            target_source="Instrument B", instruction_text="EK-1 değiştirilmiştir"
        ),
    ]
    assert [
        group.instruction_indices for group in group_annex_instructions(instructions)
    ] == [[1], [2]]


def test_bound_human_corrections_preserve_raw_extraction_and_reject_stale_before() -> (
    None
):
    from onyx.regulatory.amendments.annexes.corrections import apply_bound_corrections
    from onyx.regulatory.amendments.annexes.models import AnnexElementCorrection

    raw = AnnexExtraction(
        source_sha256="raw-hash",
        mime_type="text/html",
        elements=[ExtractedAnnexElement(kind="text", text="OCR value")],
    )
    correction = AnnexElementCorrection(
        position=0,
        before_text="OCR value",
        corrected_text="Printed value",
        reason="OCR typo",
    )
    corrected = apply_bound_corrections(raw, [correction])
    assert raw.elements[0].text == "OCR value"
    assert corrected.elements[0].text == "Printed value"
    assert corrected.source_sha256 == raw.source_sha256
    assert corrected.elements[0].locator == raw.elements[0].locator
    with pytest.raises(ValueError, match="before"):
        apply_bound_corrections(
            raw, [correction.model_copy(update={"before_text": "forged"})]
        )


def test_analysis_delivery_carries_environment_database_tenant_and_expiration() -> None:
    from unittest.mock import MagicMock

    from onyx.background.celery.tasks.regulatory_amendments.tasks import (
        enqueue_amendment_batch,
    )
    from onyx.regulatory.amendments.annexes import config

    broker = MagicMock()
    enqueue_amendment_batch(celery_app=broker, batch_id=3, tenant_id="tenant-one")
    for call in broker.send_task.call_args_list:
        assert (
            call.kwargs["kwargs"]["environment"] == config.REGULATORY_ANNEX_ENVIRONMENT
        )
        assert (
            call.kwargs["kwargs"]["database_identity"] == config.ANNEX_DATABASE_IDENTITY
        )
        assert call.kwargs["kwargs"]["tenant_id"] == "tenant-one"
        assert call.kwargs["expires"] > 0


def test_explicit_article_target_takes_precedence_over_annex_mentions() -> None:
    from onyx.regulatory.amendments.annexes.analysis import group_annex_instructions

    instruction = AmendmentInstruction(
        article_reference="Madde 5", instruction_text='"EK-1" ibaresi eklenmiştir.'
    )
    assert group_annex_instructions([instruction]) == []


def test_new_views_combine_by_source_occurrences_without_canonical_coverage() -> None:
    from onyx.regulatory.amendments.annexes.evidence import (
        combine_annex_evidence_views,
        select_annex_evidence_view,
        validate_evidence_view,
    )
    from onyx.regulatory.amendments.annexes.models import SourceLink

    views = []
    links = []
    for index in range(2):
        digest = str(index) * 64
        original = AnnexOriginalEvidence(
            file_id=str(index), sha256=digest, mime_type="text/html", available=True
        )
        extraction = AnnexExtraction(
            source_sha256=digest,
            mime_type="text/html",
            elements=[
                ExtractedAnnexElement(
                    kind="text", text="EK-1", extraction_method="native"
                ),
                ExtractedAnnexElement(
                    kind="text", text=f"part {index}", extraction_method="native"
                ),
            ],
        )
        views.append(
            select_annex_evidence_view(
                extraction=extraction,
                original=original,
                annex_label="EK-1",
                canonical_labels=["EK-1"],
                canonical_chunk_ids=["legal-scope"],
            )
        )
        links.append(
            SourceLink(
                parent_asset_hash="f" * 64,
                target_asset_hash=digest,
                source_field=f"html:{index}",
                label="EK-1",
                kind="url",
            )
        )
    combined = combine_annex_evidence_views(
        views, canonical_chunk_ids=[], source_occurrences=links
    )
    assert combined.evidence_view is not None
    assert len(combined.evidence_view.parents) == 2
    assert not validate_evidence_view(combined)
    assert [element.text for element in combined.elements] == [
        "EK-1",
        "part 0",
        "EK-1",
        "part 1",
    ]
    with pytest.raises(ValueError, match="source"):
        combine_annex_evidence_views(
            views, canonical_chunk_ids=[], source_occurrences=links[::-1]
        )


def test_new_source_selection_uses_ordered_disjoint_parts_and_native_containment() -> (
    None
):
    from onyx.regulatory.amendments.annexes.evidence import (
        select_annex_evidence_view,
        select_new_annex_sources,
    )
    from onyx.regulatory.amendments.annexes.models import SourceLink

    def part(number: int, text: str) -> tuple[AnnexOriginalEvidence, AnnexExtraction]:
        digest = str(number) * 64
        original = AnnexOriginalEvidence(
            file_id=str(number), sha256=digest, mime_type="text/html", available=True
        )
        extraction = AnnexExtraction(
            source_sha256=digest,
            mime_type="text/html",
            elements=[
                ExtractedAnnexElement(
                    kind="text", text="EK-1", extraction_method="native"
                ),
                *[
                    ExtractedAnnexElement(
                        kind="text", text=value, extraction_method="native"
                    )
                    for value in text.split("|")
                ],
            ],
        )
        return original, select_annex_evidence_view(
            extraction=extraction,
            original=original,
            annex_label="EK-1",
            canonical_labels=["EK-1"],
            canonical_chunk_ids=["scope"],
        )

    first, second = part(1, "first"), part(2, "second")
    links = [
        SourceLink(
            parent_asset_hash="0" * 64,
            target_asset_hash=str(index) * 64,
            source_field=f"html:{index}",
            label="EK-1",
            kind="url",
        )
        for index in (1, 2)
    ]
    originals, view = select_new_annex_sources([second, first], links)
    assert [item.file_id for item in originals] == ["1", "2"]
    assert (
        view.evidence_view is not None
        and view.evidence_view.selection_method == "ordered_source_occurrences"
    )
    with pytest.raises(ValueError, match="incomplete"):
        select_new_annex_sources([first], links)
    with pytest.raises(ValueError, match="overlapping"):
        select_new_annex_sources([first, part(2, "first")], links)
    parent = part(0, "first|second")
    from onyx.regulatory.amendments.annexes.comparison import compare_annexes

    comparison = compare_annexes(
        old=parent[1], new=view, old_pages=[], new_pages=[], llm=None
    )
    assert comparison.ready, comparison.issues
    assert comparison.changes == []
    originals, _ = select_new_annex_sources([parent, first, second], links)
    assert [item.file_id for item in originals] == ["0"]
    with pytest.raises(ValueError, match="overlapping"):
        select_new_annex_sources([parent, part(2, "second|first")], [links[1]])
    attachment = part(2, "independent attachment")
    with pytest.raises(ValueError, match="overlapping"):
        select_new_annex_sources(
            [parent, attachment], [links[1].model_copy(update={"kind": "embedded"})]
        )


@pytest.mark.parametrize(
    "wording", ["2 nci maddesinde", "ikinci fıkrasında", "(a) bendinde", "article 2"]
)
def test_explicit_annex_target_owns_internal_article_wording(wording: str) -> None:
    from onyx.regulatory.amendments.annexes.analysis import group_annex_instructions

    groups = group_annex_instructions(
        [
            AmendmentInstruction(
                article_reference="EK-1",
                instruction_text=f"EK-1’in {wording} yer alan eski ibaresi yeni olarak değiştirilmiştir.",
            )
        ]
    )
    assert len(groups) == 1 and groups[0].annex_label == "ek:1"
