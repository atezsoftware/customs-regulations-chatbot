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
    from hashlib import sha256
    from uuid import uuid4

    import pytest

    from onyx.regulatory.amendments.annexes.evidence import validate_compared_evidence
    from onyx.regulatory.amendments.annexes.models import AnnexReviewEvidence

    assert [item.sha256 for item in result.image_manifest] == [
        sha256(content).hexdigest() for content in submitted
    ]
    originals = [
        AnnexOriginalEvidence(
            file_id=view.evidence_view.parents[0].file_id,
            sha256=view.source_sha256,
            mime_type=view.mime_type,
            available=True,
        )
        for view in views
    ]
    frozen = [
        AnnexReviewEvidence(
            id=uuid4(),
            side="old" if index == 0 else "new",
            kind="original",
            file_id=f"frozen-{index}",
            sha256=original.sha256 or "",
            mime_type=original.mime_type or "",
            byte_count=1,
            parent_file_id=original.file_id,
            parent_sha256=original.sha256 or "",
        )
        for index, original in enumerate(originals)
    ]
    for item in result.image_manifest:
        original = originals[0 if item.side == "old" else 1]
        frozen.append(
            AnnexReviewEvidence(
                id=uuid4(),
                side=item.side,
                kind=item.kind,
                file_id=str(uuid4()),
                sha256=item.sha256,
                mime_type="image/png",
                byte_count=item.byte_count,
                parent_file_id=original.file_id,
                parent_sha256=original.sha256 or "",
                locator=AnnexLocator(
                    page=item.page, normalized_box=item.normalized_box
                ),
            )
        )
    validate_compared_evidence(
        old=views[0],
        new=views[1],
        old_originals=[originals[0]],
        new_originals=[originals[1]],
        comparison=result,
        evidence=frozen,
    )
    for changed in [
        frozen[:-1],
        [*frozen[:-1], frozen[-1].model_copy(update={"sha256": "wrong"})],
    ]:
        with pytest.raises(ValueError, match="manifest"):
            validate_compared_evidence(
                old=views[0],
                new=views[1],
                old_originals=[originals[0]],
                new_originals=[originals[1]],
                comparison=result,
                evidence=changed,
            )
    with pytest.raises(ValueError, match="region"):
        validate_compared_evidence(
            old=views[0],
            new=views[1],
            old_originals=[originals[0]],
            new_originals=[originals[1]],
            comparison=result.model_copy(
                update={
                    "image_manifest": [
                        item
                        for item in result.image_manifest
                        if item.kind != "comparison_region"
                    ]
                }
            ),
            evidence=frozen,
        )
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


def _linked_image(number: int) -> tuple[AnnexOriginalEvidence, AnnexExtraction]:
    digest = str(number) * 64
    original = AnnexOriginalEvidence(
        file_id=f"image-{number}", sha256=digest, mime_type="image/png", available=True
    )
    return original, AnnexExtraction(
        source_sha256=digest,
        mime_type="image/png",
        page_count=1,
        elements=[
            ExtractedAnnexElement(
                kind="image_region",
                text=f"Table part {number}",
                extraction_method="vision",
                locator=AnnexLocator(page=1, normalized_box=(0, 0, 1, 1)),
            )
        ],
    )


def test_descriptive_image_occurrences_preserve_complete_hash_bound_order() -> None:
    import hashlib
    import json
    from contextlib import contextmanager
    from io import BytesIO
    from typing import cast

    import pytest

    from onyx.db.models import RegulatorySourceAsset
    from onyx.db.regulatory_annexes import normalize_annex_label
    from onyx.file_store.file_store import FileStore
    from onyx.regulatory.amendments.annexes.evidence import (
        evidence_view_hash,
        read_source_graph,
        select_annex_evidence_view,
        select_new_annex_sources,
        validate_source_occurrence_order,
    )
    from onyx.regulatory.amendments.annexes.models import SourceLink

    titles = ["EK-1 Oran Tablosu - sayfa 1", "EK-1 Oran Tablosu - devam ve dipnot"]
    parts = [_linked_image(number) for number in (1, 2)]
    links = [
        SourceLink(
            parent_asset_hash="0" * 64,
            target_asset_hash=original.sha256,
            source_field=f"html:img:{index}",
            label=title,
            kind="url",
        )
        for index, ((original, _), title) in enumerate(zip(parts, titles))
    ]
    links = [
        SourceLink(
            parent_asset_hash="9" * 64,
            target_asset_hash="0" * 64,
            source_page=1,
            source_field="pdf:annotation:0",
            kind="url",
            label="EKLER - EK-1 Oran Tablosu (iki sayfa)\n https://annex-fixture.invalid/updates/annexes.html\n",
        ),
        SourceLink(
            parent_asset_hash="9" * 64,
            target_asset_hash="0" * 64,
            source_page=1,
            source_field="pdf:text:0",
            kind="url",
            label="",
        ),
        *links,
    ]
    assets = [
        RegulatorySourceAsset(sha256=digest, mime_type=mime)
        for digest, mime in [
            ("9" * 64, "application/pdf"),
            ("0" * 64, "text/html"),
            ("1" * 64, "image/png"),
            ("2" * 64, "image/png"),
        ]
    ]
    raw = json.dumps(
        {
            "status": "ready",
            "issues": [],
            "assets": [
                {"sha256": asset.sha256, "mime_type": asset.mime_type}
                for asset in assets
            ],
            "links": [link.model_dump(mode="json") for link in links],
        }
    ).encode()

    class Store:
        @contextmanager
        def read_file(self, _identifier: str):
            yield BytesIO(raw)

    store = cast(FileStore, Store())
    verified_links = read_source_graph(
        store,
        manifest_file_id="manifest",
        manifest_sha256=hashlib.sha256(raw).hexdigest(),
        assets=assets,
    )
    selected = [
        (
            original,
            select_annex_evidence_view(
                extraction=extraction,
                original=original,
                annex_label="EK-1",
                canonical_labels=["EK-1"],
                canonical_chunk_ids=["canonical"],
                source_labels=[
                    link.label
                    for link in verified_links
                    if link.target_asset_hash == original.sha256
                ],
            ),
        )
        for original, extraction in parts
    ]
    originals, combined = select_new_annex_sources(selected[::-1], verified_links)
    assert [original.sha256 for original in originals] == ["1" * 64, "2" * 64]
    view = combined.evidence_view
    assert view is not None and view.label == "ek:1"
    assert view.source_occurrences == verified_links[-2:]
    assert evidence_view_hash(combined) == view.sha256
    with pytest.raises(ValueError, match="incomplete"):
        select_new_annex_sources(selected[:1], verified_links)
    with pytest.raises(ValueError, match="incomplete"):
        select_new_annex_sources(
            selected[:1],
            [*verified_links[:-1], verified_links[-1].model_copy(update={"label": ""})],
        )
    with pytest.raises(ValueError, match="identity mismatch"):
        validate_source_occurrence_order([view], verified_links[::-1])
    with pytest.raises(ValueError, match="integrity"):
        read_source_graph(
            store, manifest_file_id="manifest", manifest_sha256="f" * 64, assets=assets
        )
    for title in titles:
        with pytest.raises(ValueError, match="ambiguous"):
            normalize_annex_label(title)


def test_image_occurrence_labels_refuse_ambiguous_or_unproven_scope() -> None:
    import pytest

    from onyx.regulatory.amendments.annexes.evidence import select_annex_evidence_view

    original, extraction = _linked_image(1)
    refused = [
        [],
        ["Oran tablosu"],
        ["EK-1 ve EK-2"],
        ["EK-1 ve 2"],
        ["EK-1, 2"],
        ["EK-1/"],
        ["EK-1 /"],
        ["EK-1 -"],
        ["EK-1 (ve 2)"],
        ["EK-1-"],
        ["EK-1..."],
        ["EK-1/A/"],
        ["EK-1A Oran Tablosu"],
        ["EK-1/A Oran Tablosu"],
        ["EK-1 Oran Tablosu", "EK-2 Dipnot"],
        ["EK-1 ve EK-"],
        ["https://example.test/EK-1"],
        ["EK-1.png"],
        ["EK-1 Oran Tablosu.png"],
        ["EK-1 EK-1"],
        ["EK-12 Oran Tablosu"],
    ]
    for labels in refused:
        with pytest.raises(ValueError):
            select_annex_evidence_view(
                extraction=extraction,
                original=original,
                annex_label="EK-1",
                canonical_labels=["EK-1"],
                canonical_chunk_ids=["canonical"],
                source_labels=labels,
            )
    extraction.elements[0].text = "EK-2"
    with pytest.raises(ValueError, match="ambiguous"):
        select_annex_evidence_view(
            extraction=extraction,
            original=original,
            annex_label="EK-1",
            canonical_labels=["EK-1"],
            canonical_chunk_ids=["canonical"],
            source_labels=["EK-1 Oran Tablosu - sayfa 1"],
        )


def test_image_occurrence_caption_can_corroborate_scope_beside_a_url() -> None:
    from onyx.regulatory.amendments.annexes.evidence import select_annex_evidence_view

    original, extraction = _linked_image(1)
    selected = select_annex_evidence_view(
        extraction=extraction,
        original=original,
        annex_label="EK-1",
        canonical_labels=["EK-1"],
        canonical_chunk_ids=["canonical"],
        source_labels=["EK-1 Oran Tablosu - sayfa 1\nhttps://example.test/EK-2.png"],
    )
    assert selected.evidence_view is not None and selected.evidence_view.label == "ek:1"


def test_visual_continuation_headers_and_values_are_not_body_overlap() -> None:
    import pytest

    from onyx.regulatory.amendments.annexes.evidence import (
        select_annex_evidence_view,
        select_new_annex_sources,
    )
    from onyx.regulatory.amendments.annexes.models import SourceLink

    def part(
        number: int,
        codes: list[str],
        *,
        located: bool = True,
        header: bool = True,
        rates: list[str] | None = None,
        roles: bool = True,
        vision: bool = True,
    ):
        original, extraction = _linked_image(number)
        extraction.elements = [
            ExtractedAnnexElement(
                kind="table_cell",
                table_role=("column_header" if header and row == 0 else "data")
                if roles
                else "unknown",
                text=text,
                extraction_method="vision" if vision else "native",
                locator=AnnexLocator(
                    page=1,
                    normalized_box=(
                        column * 0.4,
                        0.2 + row * 0.1,
                        (column + 1) * 0.4,
                        0.3 + row * 0.1,
                    )
                    if located
                    else None,
                ),
            )
            for row, cells in enumerate(
                [
                    *([["Code", "Rate"]] if header else []),
                    *[
                        [code, rate]
                        for code, rate in zip(codes, rates or ["5%"] * len(codes))
                    ],
                ]
            )
            for column, text in enumerate(cells)
        ]
        extraction.elements.extend(
            [
                ExtractedAnnexElement(
                    kind="text",
                    text="Shared footer",
                    extraction_method="vision",
                    locator=AnnexLocator(page=1, normalized_box=(0, 0.95, 1, 0.99)),
                ),
                ExtractedAnnexElement(
                    kind="image_region",
                    text="Horizontal rule",
                    extraction_method="vision",
                    locator=AnnexLocator(page=1, normalized_box=(0, 0.35, 1, 0.351)),
                ),
            ]
        )
        return original, select_annex_evidence_view(
            extraction=extraction,
            original=original,
            annex_label="EK-1",
            canonical_labels=["EK-1"],
            canonical_chunk_ids=["canonical"],
            source_labels=[f"EK-1 Table page {number}"],
        )

    links = [
        SourceLink(
            parent_asset_hash="0" * 64,
            target_asset_hash=str(number) * 64,
            source_field=f"html:{number}",
            label=f"EK-1 Table page {number}",
            kind="url",
        )
        for number in (1, 2)
    ]
    first, second = part(1, ["A", "B"]), part(2, ["C", "D"])
    _, combined = select_new_annex_sources([first, second], links)
    assert len(combined.elements) == len(first[1].elements) + len(second[1].elements)
    assert sum(element.text == "5%" for element in combined.elements) == 4
    assert sum(element.text == "Shared footer" for element in combined.elements) == 2
    for conflicting in [
        part(2, ["A", "B"]),
        part(2, ["B", "D"]),
        part(2, ["C", "D"], located=False),
        part(2, ["C", "D"], roles=False),
        part(2, ["C", "D"], vision=False),
    ]:
        with pytest.raises(ValueError, match="overlapping"):
            select_new_annex_sources([first, conflicting], links)

    for unclassified in [
        [
            part(1, ["Buğday", "Pirinç"], header=False, rates=["Muaf", "5%"]),
            part(2, ["Buğday", "Mısır"], header=False, rates=["Muaf", "10%"]),
        ],
        [
            part(1, ["1001.10", "2001.20"], header=False, rates=["5%", "15%"]),
            part(2, ["1001.10", "3001.30"], header=False, rates=["5%", "25%"]),
        ],
        [
            part(1, ["Apples", "Berries"], header=False, rates=["Fresh", "Dried"]),
            part(2, ["Apples", "Citrus"], header=False, rates=["Fresh", "Whole"]),
        ],
    ]:
        with pytest.raises(ValueError, match="overlapping"):
            select_new_annex_sources(unclassified, links)


def test_alternative_annex_shorthand_does_not_authorize_an_image() -> None:
    import pytest

    from onyx.regulatory.amendments.annexes.evidence import select_annex_evidence_view

    original, extraction = _linked_image(1)
    for connector in ("veya", "yahut", "veyahut", "ya da", "ve/veya", "and/or"):
        with pytest.raises(ValueError, match="ambiguous"):
            select_annex_evidence_view(
                extraction=extraction,
                original=original,
                annex_label="EK-1",
                canonical_labels=["EK-1"],
                canonical_chunk_ids=["canonical"],
                source_labels=[f"EK-1 {connector} 2"],
            )


def test_optional_table_role_preserves_legacy_nested_json_and_hash() -> None:
    import json
    from pathlib import Path

    from onyx.regulatory.amendments.annexes.evidence import evidence_view_hash
    from onyx.regulatory.amendments.annexes.models import AnnexVisionResult

    frozen = (
        (Path(__file__).parent / "fixtures" / "roleless_evidence_view.json")
        .read_text()
        .strip()
    )
    payload = json.loads(frozen)
    for explicit_unknown in (False, True):
        if explicit_unknown:
            payload["elements"][0]["table_role"] = "unknown"
        restored = AnnexExtraction.model_validate(payload)
        assert restored.model_dump_json() == frozen
        assert "table_role" not in restored.model_dump()["elements"][0]
        assert restored.evidence_view is not None
        assert evidence_view_hash(restored) == restored.evidence_view.sha256
        marked = restored.model_copy(deep=True)
        marked.elements[0].table_role = "data"
        assert marked.model_dump_json() != frozen
        assert evidence_view_hash(marked) != restored.evidence_view.sha256
    old_vision = {
        "elements": [
            {
                "kind": "table_cell",
                "text": "Legacy cell",
                "box": [0, 0, 1, 1],
                "status": "readable",
                "issues": [],
            }
        ]
    }
    result = AnnexVisionResult.model_validate(old_vision)
    assert result.model_dump(mode="json") == old_vision
    old_vision["elements"][0]["table_role"] = "unknown"
    assert (
        AnnexVisionResult.model_validate(old_vision).model_dump_json()
        == result.model_dump_json()
    )
