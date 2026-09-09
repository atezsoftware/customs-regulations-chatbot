from onyx.regulatory.amendments.annexes.models import (
    AnnexExtraction,
    AnnexLocator,
    AnnexOriginalEvidence,
    ExtractedAnnexElement,
)


def mixed_pdf() -> AnnexExtraction:
    def text(
        value: str, page: int, top: float, *, native: bool = False
    ) -> ExtractedAnnexElement:
        return ExtractedAnnexElement(
            kind="text",
            text=value,
            aggregate=native,
            evidence_kind="original" if native else "rendered_preview",
            extraction_method="native" if native else "vision",
            locator=AnnexLocator(
                page=page,
                normalized_box=(0.1, top, 0.9, top + 0.05),
                coordinate_system="top_left_points",
            ),
        )

    return AnnexExtraction(
        source_sha256="a" * 64,
        mime_type="application/pdf",
        page_count=3,
        elements=[
            text("Notice: replace EK-1", 1, 0.1, native=True),
            text("EK-1", 1, 0.4, native=True),
            text("EK-1", 1, 0.4),
            text("First row", 1, 0.5),
            text("Continuing row", 2, 0.1),
            text("Footnote to EK-1", 2, 0.6),
            text("EK-2", 2, 0.8, native=True),
            text("EK-2", 2, 0.8),
            text("Other annex", 3, 0.2),
        ],
    )


def test_mixed_notice_scopes_complete_annex_and_excludes_adjacent_annex() -> None:
    from onyx.regulatory.amendments.annexes.evidence import select_annex_evidence_view

    original = AnnexOriginalEvidence(
        file_id="original", sha256="a" * 64, mime_type="application/pdf", available=True
    )
    view = select_annex_evidence_view(
        extraction=mixed_pdf(),
        original=original,
        annex_label="EK-1",
        canonical_labels=["EK-1", "EK-2"],
        canonical_chunk_ids=["chunk"],
    )
    assert [element.text for element in view.elements if not element.aggregate] == [
        "EK-1",
        "First row",
        "Continuing row",
        "Footnote to EK-1",
    ]
    assert view.page_count == 3  # Actual parent page count remains unchanged.
    assert view.evidence_view is not None
    assert [page.original_page for page in view.evidence_view.pages] == [1, 2]
    assert view.evidence_view.pages[0].normalized_box == (0, 0.4, 1, 1)
    assert view.evidence_view.pages[1].normalized_box == (0, 0, 1, 0.8)
    assert view.evidence_view.parents[0].sha256 == original.sha256
    assert view.source_sha256 == original.sha256
    assert view.evidence_view.sha256 != original.sha256


def test_same_parent_different_selected_scope_never_uses_identical_asset_shortcut() -> (
    None
):
    from onyx.regulatory.amendments.annexes.comparison import compare_annexes
    from onyx.regulatory.amendments.annexes.evidence import select_annex_evidence_view

    original = AnnexOriginalEvidence(
        file_id="original", sha256="a" * 64, mime_type="application/pdf", available=True
    )
    views = [
        select_annex_evidence_view(
            extraction=mixed_pdf(),
            original=original,
            annex_label=label,
            canonical_labels=["EK-1", "EK-2"],
            canonical_chunk_ids=["chunk"],
        )
        for label in ["EK-1", "EK-2"]
    ]
    comparison = compare_annexes(old=views[0], new=views[1])
    assert comparison.coverage.method == "simultaneous_vision"
    assert not comparison.ready


def test_vision_only_boundary_cannot_authorize_scope() -> None:
    import pytest

    from onyx.regulatory.amendments.annexes.evidence import select_annex_evidence_view

    extraction = mixed_pdf()
    extraction.elements = [
        element for element in extraction.elements if not element.aggregate
    ]
    with pytest.raises(ValueError, match="boundary"):
        select_annex_evidence_view(
            extraction=extraction,
            original=AnnexOriginalEvidence(
                file_id="original",
                sha256="a" * 64,
                mime_type="application/pdf",
                available=True,
            ),
            annex_label="EK-1",
            canonical_labels=["EK-1"],
            canonical_chunk_ids=["chunk"],
        )


def test_modified_selected_view_cannot_be_review_ready() -> None:
    from onyx.regulatory.amendments.annexes.comparison import compare_annexes
    from onyx.regulatory.amendments.annexes.evidence import select_annex_evidence_view

    original = AnnexOriginalEvidence(
        file_id="original", sha256="a" * 64, mime_type="application/pdf", available=True
    )
    view = select_annex_evidence_view(
        extraction=mixed_pdf(),
        original=original,
        annex_label="EK-1",
        canonical_labels=["EK-1"],
        canonical_chunk_ids=["chunk"],
    )
    view.elements[0].text = "forged"
    result = compare_annexes(old=view, new=view)
    assert not result.ready and "evidence_view_integrity_mismatch" in result.issues


def test_patch_rejects_selected_parent_outside_baseline() -> None:
    from datetime import date

    from onyx.regulatory.amendments.annexes.comparison import compare_annexes
    from onyx.regulatory.amendments.annexes.evidence import select_annex_evidence_view
    from onyx.regulatory.amendments.annexes.models import AnnexBaseline
    from onyx.regulatory.amendments.annexes.patch_plan import prepare_annex_patch

    original = AnnexOriginalEvidence(
        file_id="foreign-original",
        sha256="a" * 64,
        mime_type="application/pdf",
        available=True,
    )
    view = select_annex_evidence_view(
        extraction=mixed_pdf(),
        original=original,
        annex_label="EK-1",
        canonical_labels=["EK-1"],
        canonical_chunk_ids=["chunk"],
    )
    baseline = AnnexBaseline(
        baseline_sha256="baseline",
        canonical_text="First row",
        elements=[
            ExtractedAnnexElement(
                kind="text", text="First row", canonical_chunk_id="chunk"
            )
        ],
        originals=[original.model_copy(update={"file_id": "owned-original"})],
        visual_evidence_available=True,
    )
    plan = prepare_annex_patch(
        baseline=baseline,
        old=view,
        new=view,
        comparison=compare_annexes(old=view, new=view),
        effective_date=date(2026, 9, 10),
        package_complete=True,
    )
    assert not plan.ready and "evidence_parent_outside_baseline" in plan.issues


def test_original_selection_requires_complete_ordered_bindings() -> None:
    import pytest

    from onyx.regulatory.amendments.annexes.evidence import choose_original_evidence

    complete = AnnexOriginalEvidence(
        file_id="full",
        sha256="a" * 64,
        available=True,
        mime_type="application/pdf",
        canonical_chunk_ids=["a", "b"],
    )
    piece = AnnexOriginalEvidence(
        file_id="part",
        sha256="b" * 64,
        available=True,
        mime_type="image/png",
        canonical_chunk_ids=["b"],
    )
    assert choose_original_evidence(
        [piece, complete], canonical_chunk_ids=["a", "b"]
    ) == [complete]
    with pytest.raises(ValueError, match="missing"):
        choose_original_evidence([piece], canonical_chunk_ids=["a", "b"])
    first = piece.model_copy(update={"file_id": "first", "canonical_chunk_ids": ["a"]})
    with pytest.raises(ValueError, match="unordered"):
        choose_original_evidence([piece, first], canonical_chunk_ids=["a", "b"])
    assert choose_original_evidence([first, piece], canonical_chunk_ids=["a", "b"]) == [
        first,
        piece,
    ]


def test_selected_comparison_uses_original_page_mapping_and_exact_region_images() -> (
    None
):
    import base64
    from unittest.mock import MagicMock, patch

    from onyx.regulatory.amendments.annexes.comparison import (
        compare_annexes,
        comparison_page_evidence,
    )
    from onyx.regulatory.amendments.annexes.evidence import select_annex_evidence_view
    from onyx.regulatory.amendments.annexes.models import AnnexComparisonResponse
    from tests.unit.onyx.regulatory.annexes.test_comparison import page

    views = []
    for digest in ["a" * 64, "b" * 64]:
        extraction = mixed_pdf().model_copy(update={"source_sha256": digest})
        views.append(
            select_annex_evidence_view(
                extraction=extraction,
                original=AnnexOriginalEvidence(
                    file_id=digest,
                    sha256=digest,
                    mime_type="application/pdf",
                    available=True,
                ),
                annex_label="EK-1",
                canonical_labels=["EK-1"],
                canonical_chunk_ids=["chunk"],
            )
        )
    positions = [
        index
        for index, element in enumerate(views[0].elements)
        if not element.aggregate
    ]
    response = AnnexComparisonResponse(
        old_positions=positions,
        new_positions=positions,
        old_pages=[1, 2],
        new_pages=[1, 2],
    )
    llm = MagicMock()
    llm.config.model_name, llm.config.model_provider = "vision", "configured"
    pages = [page(number=1), page(number=2)]
    with patch(
        "onyx.regulatory.amendments.annexes.comparison.generate_structured",
        return_value=response,
    ) as call:
        result = compare_annexes(
            old=views[0], new=views[1], old_pages=pages, new_pages=pages, llm=llm
        )
    assert result.ready
    assert views[0].evidence_view is not None
    expected = [
        image.png
        for rendered, mapping in zip(pages, views[0].evidence_view.pages)
        for image in comparison_page_evidence(
            rendered, normalized_box=mapping.normalized_box
        )
    ]
    submitted = [
        base64.b64decode(part.image_url.url.split(",", 1)[1])
        for part in call.call_args.kwargs["image_parts"]
    ]
    assert submitted == expected + expected
    prompt = call.call_args.kwargs["user_prompt"]
    assert "selected_positions" not in prompt
    assert "original_position" not in prompt
    assert "EXACT required old_positions: " + str(positions) in prompt
    assert "EXACT required new_positions: " + str(positions) in prompt


def test_vision_heading_box_can_include_space_above_native_glyphs() -> None:
    from onyx.regulatory.amendments.annexes.evidence import select_annex_evidence_view

    extraction = mixed_pdf()
    extraction.elements[2].locator.normalized_box = (0.1, 0.38, 0.9, 0.47)
    view = select_annex_evidence_view(
        extraction=extraction,
        original=AnnexOriginalEvidence(
            file_id="original",
            sha256="a" * 64,
            mime_type="application/pdf",
            available=True,
        ),
        annex_label="EK-1",
        canonical_labels=["EK-1"],
        canonical_chunk_ids=["chunk"],
    )
    assert "First row" in [element.text for element in view.elements]
    assert "Notice: replace EK-1" not in [element.text for element in view.elements]


def test_native_sheet_selection_keeps_only_complete_matching_sheet() -> None:
    from onyx.regulatory.amendments.annexes.evidence import select_annex_evidence_view

    extraction = AnnexExtraction(
        source_sha256="a" * 64,
        mime_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        elements=[
            ExtractedAnnexElement(
                kind="table_cell",
                text=label,
                extraction_method="native",
                locator=AnnexLocator(sheet=label, row=1, column=1),
            )
            for label in ["EK-1", "EK-2"]
        ],
    )
    view = select_annex_evidence_view(
        extraction=extraction,
        original=AnnexOriginalEvidence(
            file_id="book",
            sha256=extraction.source_sha256,
            mime_type=extraction.mime_type,
            available=True,
        ),
        annex_label="EK-1",
        canonical_labels=["EK-1"],
        canonical_chunk_ids=["row"],
    )
    assert [element.locator.sheet for element in view.elements] == ["EK-1"]


def test_bound_image_views_combine_only_in_complete_original_order() -> None:
    import pytest

    from onyx.regulatory.amendments.annexes.evidence import (
        combine_annex_evidence_views,
        select_annex_evidence_view,
    )

    views = []
    for index, canonical_id in enumerate(["a", "b"]):
        digest = str(index) * 64
        extraction = AnnexExtraction(
            source_sha256=digest,
            mime_type="image/png",
            page_count=1,
            elements=[
                ExtractedAnnexElement(
                    kind="image_region",
                    text=f"Region {index}",
                    extraction_method="vision",
                    locator=AnnexLocator(page=1, normalized_box=(0, 0, 1, 1)),
                )
            ],
        )
        views.append(
            select_annex_evidence_view(
                extraction=extraction,
                original=AnnexOriginalEvidence(
                    file_id=canonical_id,
                    sha256=digest,
                    mime_type="image/png",
                    available=True,
                    canonical_chunk_ids=[canonical_id],
                ),
                annex_label="EK-1",
                canonical_labels=["EK-1"],
                canonical_chunk_ids=[canonical_id],
            )
        )
    combined = combine_annex_evidence_views(views, canonical_chunk_ids=["a", "b"])
    assert combined.evidence_view is not None
    assert [parent.file_id for parent in combined.evidence_view.parents] == ["a", "b"]
    assert [
        (page.parent_index, page.original_page, page.view_page)
        for page in combined.evidence_view.pages
    ] == [(0, 1, 1), (1, 1, 2)]
    assert [element.locator.page for element in combined.elements] == [1, 2]
    with pytest.raises(ValueError, match="unordered"):
        combine_annex_evidence_views(views[::-1], canonical_chunk_ids=["a", "b"])


def test_comparison_rejects_benign_prose_as_blocking_issues() -> None:
    import pytest
    from pydantic import ValidationError

    from onyx.regulatory.amendments.annexes.models import AnnexComparisonResponse

    with pytest.raises(ValidationError):
        AnnexComparisonResponse.model_validate(
            {"issues": ["The complete selected scope was reviewed"]}
        )
