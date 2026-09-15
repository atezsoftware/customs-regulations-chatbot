import hashlib
from uuid import uuid4

import pytest

from onyx.regulatory.amendments.annexes.canonical_evidence import (
    canonical_baseline_extraction,
    render_selected_text,
    validate_canonical_baseline,
)
from onyx.regulatory.amendments.annexes.comparison import compare_annexes
from onyx.regulatory.amendments.annexes.evidence import (
    has_visual_evidence,
    identical_evidence_scope,
    select_annex_evidence_view,
    validate_compared_evidence,
)
from onyx.regulatory.amendments.annexes.models import (
    AnnexBaseline,
    AnnexExtraction,
    AnnexLocator,
    AnnexModelSnapshot,
    AnnexOriginalEvidence,
    AnnexReviewEvidence,
    ExtractedAnnexElement,
)


def baseline(*, image: bool = True) -> AnnexBaseline:
    elements = [
        ExtractedAnnexElement(
            kind="text", text="Approved text", canonical_chunk_id="text"
        )
    ]
    originals = []
    if image:
        elements.append(
            ExtractedAnnexElement(
                kind="image_region",
                text="Photo caption",
                canonical_chunk_id="image",
                canonical_role="supporting",
                bound_to_regulatory_chunk_id="text",
                image_file_id="image-file",
            )
        )
        originals.append(
            AnnexOriginalEvidence(
                file_id="image-file",
                sha256="a" * 64,
                mime_type="image/png",
                available=True,
                canonical_chunk_ids=["image"],
            )
        )
    return AnnexBaseline(
        baseline_sha256="baseline",
        canonical_text="Approved text",
        elements=elements,
        originals=originals,
        visual_evidence_available=image,
    )


def test_partial_image_coverage_supplements_all_canonical_chunks() -> None:
    source = baseline()
    old = canonical_baseline_extraction(source)
    assert old.elements == source.elements
    assert old.canonical_evidence is not None
    assert old.canonical_evidence.images[0].canonical_chunk_ids == ["image"]
    assert has_visual_evidence(old)
    assert validate_canonical_baseline(source, old) == []
    changed = old.model_copy(deep=True)
    changed.elements[0].text = "Invented"
    assert validate_canonical_baseline(source, changed)
    changed = old.model_copy(deep=True)
    assert changed.canonical_evidence is not None
    changed.canonical_evidence.images[0] = changed.canonical_evidence.images[
        0
    ].model_copy(update={"canonical_chunk_ids": ["text"]})
    assert validate_canonical_baseline(source, changed)
    assert not identical_evidence_scope(old, changed)
    assert "visual_evidence_unavailable" in compare_annexes(old=old, new=changed).issues


def test_missing_linked_image_is_explicitly_incomplete() -> None:
    source = baseline()
    source.originals[0].available = False
    old = canonical_baseline_extraction(source)
    assert old.elements == source.elements
    assert old.issues == ["canonical_image_unavailable"]
    source.elements[1].image_file_id = None
    source.originals[0].mime_type = None
    source.originals[0].linked_from_chunks = True
    assert canonical_baseline_extraction(source).issues == [
        "canonical_image_unavailable"
    ]


def test_reviewed_unchanged_canonical_rows_do_not_require_identical_ocr_wrapping() -> (
    None
):
    from datetime import date

    from onyx.regulatory.amendments.annexes.comparison import annex_snapshot_hash
    from onyx.regulatory.amendments.annexes.models import (
        AnnexComparison,
        AnnexCoverage,
        AnnexDifference,
        AnnexElementReference,
    )
    from onyx.regulatory.amendments.annexes.patch_plan import prepare_annex_patch

    source = baseline(image=False)
    source.elements.append(
        ExtractedAnnexElement(
            kind="text", text="Old rule", canonical_chunk_id="changed"
        )
    )
    source.canonical_text += "\n\nOld rule"
    old = canonical_baseline_extraction(source)
    new = AnnexExtraction(
        source_sha256="new",
        mime_type="text/plain",
        elements=[
            ExtractedAnnexElement(kind="text", text="Approved\ntext"),
            ExtractedAnnexElement(kind="text", text="New rule"),
        ],
    )
    comparison = AnnexComparison(
        old_source_sha256=old.source_sha256,
        new_source_sha256=new.source_sha256,
        old_snapshot_sha256=annex_snapshot_hash(old),
        new_snapshot_sha256=annex_snapshot_hash(new),
        changes=[
            AnnexDifference(
                operation="replace",
                old=[
                    AnnexElementReference(
                        position=1, text="Old rule", locator=old.elements[1].locator
                    )
                ],
                new=[
                    AnnexElementReference(
                        position=1, text="New rule", locator=new.elements[1].locator
                    )
                ],
                explanation="Changed rule",
            )
        ],
        coverage=AnnexCoverage(
            old_positions=[0, 1],
            new_positions=[0, 1],
            old_pages=[],
            new_pages=[],
            method="native_structure",
        ),
        issues=[],
        ready=True,
    )
    plan = prepare_annex_patch(
        baseline=source,
        old=old,
        new=new,
        comparison=comparison,
        effective_date=date(2026, 7, 4),
        package_complete=True,
    )
    assert plan.ready and plan.unchanged == ["text"]
    assert [(patch.old_chunk_id, patch.new_text) for patch in plan.patches] == [
        ("changed", "New rule")
    ]
    comparison = comparison.model_copy(
        update={
            "coverage": comparison.coverage.model_copy(update={"old_positions": [1]})
        }
    )
    assert not prepare_annex_patch(
        baseline=source,
        old=old,
        new=new,
        comparison=comparison,
        effective_date=date(2026, 7, 4),
        package_complete=True,
    ).ready


def test_canonical_text_approval_requires_baseline_identity_not_an_original_pdf() -> (
    None
):
    source = baseline(image=False)
    old = canonical_baseline_extraction(source)
    new = AnnexExtraction(
        source_sha256=hashlib.sha256(b"new").hexdigest(),
        mime_type="text/plain",
        elements=[ExtractedAnnexElement(kind="text", text="New text")],
    )
    original = AnnexOriginalEvidence(
        file_id="new-file",
        sha256=new.source_sha256,
        mime_type=new.mime_type,
        available=True,
    )
    evidence = [
        AnnexReviewEvidence(
            id=uuid4(),
            side="new",
            kind="original",
            file_id="frozen",
            sha256=new.source_sha256,
            mime_type=new.mime_type,
            byte_count=3,
            parent_file_id=original.file_id,
            parent_sha256=new.source_sha256,
        )
    ]
    comparison = compare_annexes(old=old, new=new)
    validate_compared_evidence(
        old=old,
        new=new,
        old_originals=[],
        new_originals=[original],
        comparison=comparison,
        evidence=evidence,
        baseline=source,
    )
    with pytest.raises(ValueError, match="original identity"):
        validate_compared_evidence(
            old=old,
            new=new,
            old_originals=[],
            new_originals=[original],
            comparison=comparison,
            evidence=evidence,
        )
    source.elements[0].text = "Changed after review"
    with pytest.raises(ValueError, match="canonical baseline"):
        validate_compared_evidence(
            old=old,
            new=new,
            old_originals=[],
            new_originals=[original],
            comparison=comparison,
            evidence=evidence,
            baseline=source,
        )


def test_verified_scanned_headings_bound_complete_annex_pages() -> None:
    def text(value: str, page: int, top: float) -> ExtractedAnnexElement:
        return ExtractedAnnexElement(
            kind="text",
            text=value,
            extraction_method="vision",
            locator=AnnexLocator(page=page, normalized_box=(0.1, top, 0.9, top + 0.04)),
        )

    extraction = AnnexExtraction(
        source_sha256="a" * 64,
        mime_type="application/pdf",
        page_count=4,
        model_snapshot=AnnexModelSnapshot(
            model_provider="fixture", model_name="vision"
        ),
        elements=[
            text("“EK-3: TAŞIT ONAY BELGESİ", 1, 0.03),
            text("Form", 1, 0.3),
            text("Continuation", 2, 0.3),
            text("EK-4 TIR KARNESİ", 3, 0.03),
            text("Manifest", 3, 0.3),
            text("“EK-10", 4, 0.03),
            text("Goods", 4, 0.3),
        ],
    )
    original = AnnexOriginalEvidence(
        file_id="pdf",
        sha256=extraction.source_sha256,
        mime_type=extraction.mime_type,
        available=True,
    )

    def select(verified: bool = False) -> AnnexExtraction:
        return select_annex_evidence_view(
            extraction=extraction,
            original=original,
            annex_label="EK-3",
            canonical_labels=["EK-3"],
            canonical_chunk_ids=["text"],
            verified_visual_source_sha256=original.sha256 if verified else None,
        )

    with pytest.raises(ValueError, match="boundary"):
        select()
    view = select(True)
    assert view.evidence_view is not None
    assert [page.original_page for page in view.evidence_view.pages] == [1, 2]
    assert [element.text for element in view.elements] == [
        "“EK-3: TAŞIT ONAY BELGESİ",
        "Form",
        "Continuation",
    ]
    extraction.elements.insert(1, text("EK-3", 1, 0.1))
    with pytest.raises(ValueError, match="ambiguous"):
        select(True)


def test_selected_table_cells_become_markdown_and_photos_stay_captions() -> None:
    elements = [
        ExtractedAnnexElement(
            kind="table_cell",
            text=value,
            extraction_method="vision",
            locator=AnnexLocator(page=5, normalized_box=box),
        )
        for value, box in [
            ("Name", (0.1, 0.1, 0.4, 0.3)),
            ("Rate", (0.4, 0.1, 0.9, 0.3)),
            ("Goods", (0.1, 0.3, 0.4, 0.5)),
            ("17%", (0.4, 0.3, 0.9, 0.5)),
        ]
    ]
    extraction = AnnexExtraction(
        source_sha256="a" * 64,
        mime_type="application/pdf",
        page_count=12,
        elements=elements,
        model_snapshot=AnnexModelSnapshot(
            model_provider="fixture", model_name="vision"
        ),
    )
    assert (
        render_selected_text(extraction, [0, 1, 2, 3])
        == "| Name | Rate |\n| --- | --- |\n| Goods | 17% |"
    )
    assert all(element.locator.page == 5 for element in extraction.elements)
    extraction.elements = [
        ExtractedAnnexElement(kind="image_region", text="Photograph of a vehicle")
    ]
    assert render_selected_text(extraction, [0]) == "Photograph of a vehicle"
