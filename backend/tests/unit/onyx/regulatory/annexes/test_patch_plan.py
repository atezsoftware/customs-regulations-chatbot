from datetime import date

import pytest

from onyx.regulatory.amendments.annexes.models import (
    AnnexBaseline,
    AnnexExtraction,
    AnnexPatchPlan,
    ExtractedAnnexElement,
)
from tests.unit.onyx.regulatory.annexes.test_comparison import extraction


def baseline(text: str) -> AnnexBaseline:
    return AnnexBaseline(
        baseline_sha256="a" * 64,
        canonical_text=text,
        elements=[
            ExtractedAnnexElement(
                kind="table_cell",
                text=text,
                canonical_chunk_id="canonical-rate",
                semantic_key="rate:A",
            )
        ],
        originals=[],
        visual_evidence_available=True,
        canonical_amendment_chunk_ids=["canonical-rate"],
    )


@pytest.mark.parametrize("new_value,expected", [("5%", ["canonical-rate"]), ("7%", [])])
def test_approved_canonical_overlay_controls_patch_despite_old_raw_pixels(
    new_value: str, expected: list[str]
) -> None:
    from onyx.regulatory.amendments.annexes.comparison import compare_annexes
    from onyx.regulatory.amendments.annexes.patch_plan import prepare_annex_patch

    old, new = extraction("5%"), extraction(new_value)
    comparison = compare_annexes(old=old, new=new)
    result = prepare_annex_patch(
        baseline=baseline("7%"),
        old=old,
        new=new,
        comparison=comparison,
        effective_date=date(2026, 9, 10),
        package_complete=True,
    )
    assert result.ready
    assert result.direct_canonical_changes == expected
    if expected:
        assert result.patches[0].old_text == "7%" and result.patches[0].new_text == "5%"


def test_canonical_text_old_against_visual_new_is_not_an_evidence_mismatch() -> None:
    """A markdown-only baseline (no retained original) compared against a
    freshly acquired visual NEW is the intended fallback, not the
    inconsistency `original_visual_evidence_unavailable` exists to catch.
    """
    from onyx.regulatory.amendments.annexes.comparison import compare_annexes
    from onyx.regulatory.amendments.annexes.patch_plan import prepare_annex_patch

    old, new = extraction("7%", visual=False), extraction("7%", visual=True)
    comparison = compare_annexes(old=old, new=new)
    canonical_only_baseline = baseline("7%").model_copy(
        update={"visual_evidence_available": False}
    )
    result = prepare_annex_patch(
        baseline=canonical_only_baseline,
        old=old,
        new=new,
        comparison=comparison,
        effective_date=date(2026, 9, 10),
        package_complete=True,
    )
    assert "original_visual_evidence_unavailable" not in result.issues


def test_visual_old_still_requires_baseline_to_confirm_it() -> None:
    """Guard the original intent: if OLD itself claims visual evidence, the
    baseline backing it must confirm that evidence actually exists.
    """
    from onyx.regulatory.amendments.annexes.comparison import compare_annexes
    from onyx.regulatory.amendments.annexes.patch_plan import prepare_annex_patch

    old, new = extraction("7%", visual=True), extraction("7%", visual=True)
    comparison = compare_annexes(old=old, new=new)
    result = prepare_annex_patch(
        baseline=baseline("7%").model_copy(
            update={"visual_evidence_available": False}
        ),
        old=old,
        new=new,
        comparison=comparison,
        effective_date=date(2026, 9, 10),
        package_complete=True,
    )
    assert not result.ready
    assert "original_visual_evidence_unavailable" in result.issues


def test_changed_unmapped_visual_region_cannot_authorize_arbitrary_chunk() -> None:
    from onyx.regulatory.amendments.annexes.comparison import compare_annexes
    from onyx.regulatory.amendments.annexes.patch_plan import prepare_annex_patch

    old, new = extraction("5%"), extraction("7%")
    old.elements[0].semantic_key = None
    comparison = compare_annexes(old=old, new=new)
    result = prepare_annex_patch(
        baseline=baseline("8%"),
        old=old,
        new=new,
        comparison=comparison,
        effective_date=date.today(),
        package_complete=True,
    )
    assert not result.ready and "canonical_correspondence_unresolved" in result.issues


def test_partial_package_and_unknown_effective_date_block_patch() -> None:
    from onyx.regulatory.amendments.annexes.comparison import compare_annexes
    from onyx.regulatory.amendments.annexes.patch_plan import prepare_annex_patch

    old, new = extraction("5%"), extraction("7%")
    result = prepare_annex_patch(
        baseline=baseline("5%"),
        old=old,
        new=new,
        comparison=compare_annexes(old=old, new=new),
        effective_date=None,
        package_complete=False,
    )
    assert not result.ready
    assert {"incomplete_source_package", "effective_date_unresolved"} <= set(
        result.issues
    )


@pytest.mark.parametrize("label", ["EK-1/A", "EK-1/B", "EK iiia", "EK a", "EK ivb"])
def test_new_annex_routing_preserves_full_corroborated_labels(label: str) -> None:
    from onyx.regulatory.amendments.annexes.patch_plan import resolve_annex_instruction

    assert (
        resolve_annex_instruction(
            f"{label} aşağıdaki şekilde değiştirilmiştir.",
            source_labels=[label],
            file_labels=[label],
        )
        == label
    )
    with pytest.raises(ValueError, match="scope"):
        resolve_annex_instruction(
            f"{label} değiştirilmiştir.", source_labels=[label], file_labels=["EK-1"]
        )


def test_non_annex_keeps_legacy_route_and_noise_blocks() -> None:
    from onyx.regulatory.amendments.annexes.patch_plan import resolve_annex_instruction

    assert (
        resolve_annex_instruction(
            "Madde 5 değiştirilmiştir", source_labels=[], file_labels=[]
        )
        is None
    )
    with pytest.raises(ValueError):
        resolve_annex_instruction(
            "EK EK değiştirilmiştir", source_labels=["EK EK"], file_labels=["EK EK"]
        )


def test_cell_patch_preserves_rest_of_existing_canonical_row() -> None:
    current, old, new = linewise_case()
    current.elements[0].text = current.canonical_text = "1001 | Wheat | 5%"
    result = prepare_linewise_case(current, old, new)
    assert result.ready
    assert result.patches[0].old_text == "1001 | Wheat | 5%"
    assert result.patches[0].new_text == "1001 | Wheat | 7%"


def test_merge_does_not_duplicate_new_content_in_two_canonical_chunks() -> None:
    from onyx.regulatory.amendments.annexes.comparison import annex_snapshot_hash
    from onyx.regulatory.amendments.annexes.models import (
        AnnexComparison,
        AnnexCoverage,
        AnnexDifference,
        AnnexElementReference,
    )
    from onyx.regulatory.amendments.annexes.patch_plan import prepare_annex_patch

    current = baseline("A")
    current.elements.append(
        ExtractedAnnexElement(kind="text", text="B", canonical_chunk_id="second")
    )
    old = AnnexExtraction(
        source_sha256="a",
        mime_type="text/html",
        elements=[
            ExtractedAnnexElement(kind="text", text="A"),
            ExtractedAnnexElement(kind="text", text="B"),
        ],
    )
    new = AnnexExtraction(
        source_sha256="b",
        mime_type="text/html",
        elements=[ExtractedAnnexElement(kind="text", text="Merged AB")],
    )
    comparison = AnnexComparison(
        old_source_sha256="a",
        new_source_sha256="b",
        old_snapshot_sha256=annex_snapshot_hash(old),
        new_snapshot_sha256=annex_snapshot_hash(new),
        coverage=AnnexCoverage(
            old_positions=[0, 1],
            new_positions=[0],
            old_pages=[],
            new_pages=[],
            method="native_structure",
        ),
        changes=[
            AnnexDifference(
                operation="merge",
                old=[
                    AnnexElementReference(
                        position=index, text=element.text, locator=element.locator
                    )
                    for index, element in enumerate(old.elements)
                ],
                new=[
                    AnnexElementReference(
                        position=0, text="Merged AB", locator=new.elements[0].locator
                    )
                ],
                explanation="Merged rows",
            )
        ],
        issues=[],
        ready=True,
    )
    result = prepare_annex_patch(
        baseline=current,
        old=old,
        new=new,
        comparison=comparison,
        effective_date=date.today(),
        package_complete=True,
    )
    assert result.ready
    assert [patch.new_text for patch in result.patches] == ["Merged AB", None]


@pytest.mark.parametrize(
    "changed_cell,overlay,ready",
    [(False, False, True), (True, False, False), (False, True, False)],
)
def test_source_backed_placeholder_companion_only_allows_independent_visual_change(
    changed_cell: bool, overlay: bool, ready: bool
) -> None:
    from onyx.regulatory.amendments.annexes.comparison import annex_snapshot_hash
    from onyx.regulatory.amendments.annexes.models import (
        AnnexComparison,
        AnnexCoverage,
        AnnexDifference,
        AnnexElementReference,
        AnnexOriginalEvidence,
    )
    from onyx.regulatory.amendments.annexes.patch_plan import prepare_annex_patch

    current = baseline("Diagram header")
    current.canonical_amendment_chunk_ids = ["canonical-rate"] if overlay else []
    current.elements[0].semantic_key = "header-lineage"
    current.elements.append(
        ExtractedAnnexElement(
            kind="image_region",
            text="Diagram header\n\n[Görsel: needs review]",
            canonical_chunk_id="caption",
            canonical_role="supporting",
            image_file_id="old-image",
            bound_to_regulatory_chunk_id="canonical-rate",
        )
    )
    old = AnnexExtraction(
        source_sha256="a",
        mime_type="image/png",
        page_count=1,
        elements=[
            ExtractedAnnexElement(
                kind="table_cell" if changed_cell else "image_region",
                text="5%" if changed_cell else "Green square",
            )
        ],
    )
    new = AnnexExtraction(
        source_sha256="b",
        mime_type="image/png",
        page_count=1,
        elements=[
            ExtractedAnnexElement(
                kind="table_cell" if changed_cell else "image_region",
                text="7%" if changed_cell else "Blue square",
            )
        ],
    )
    current.originals = [
        AnnexOriginalEvidence(
            file_id="old-image",
            sha256="a",
            available=True,
            canonical_chunk_ids=["caption"],
            mime_type="image/png",
        )
    ]
    comparison = AnnexComparison(
        old_source_sha256="a",
        new_source_sha256="b",
        old_snapshot_sha256=annex_snapshot_hash(old),
        new_snapshot_sha256=annex_snapshot_hash(new),
        coverage=AnnexCoverage(
            old_positions=[0],
            new_positions=[0],
            old_pages=[1],
            new_pages=[1],
            method="simultaneous_vision",
        ),
        changes=[
            AnnexDifference(
                operation="replace" if changed_cell else "visual",
                old=[
                    AnnexElementReference(
                        position=0,
                        text=old.elements[0].text,
                        locator=old.elements[0].locator,
                    )
                ],
                new=[
                    AnnexElementReference(
                        position=0,
                        text=new.elements[0].text,
                        locator=new.elements[0].locator,
                    )
                ],
                explanation="Observed source change",
            )
        ],
        ready=True,
        issues=[],
    )
    result = prepare_annex_patch(
        baseline=current,
        old=old,
        new=new,
        comparison=comparison,
        effective_date=date.today(),
        package_complete=True,
    )
    assert result.ready == ready
    if ready:
        caption = next(
            patch for patch in result.patches if patch.old_chunk_id == "caption"
        )
        assert (
            caption.canonical_role == "supporting" and caption.new_text == "Blue square"
        )
        assert "canonical-rate" in result.metadata_only
        assert not any(
            patch.old_chunk_id == "canonical-rate" and patch.old_text != patch.new_text
            for patch in result.patches
        )


def test_serialized_ready_flag_cannot_bypass_reference_validation() -> None:
    from onyx.regulatory.amendments.annexes.comparison import compare_annexes
    from onyx.regulatory.amendments.annexes.models import (
        AnnexDifference,
        AnnexElementReference,
    )
    from onyx.regulatory.amendments.annexes.patch_plan import prepare_annex_patch

    old, new = extraction("5%"), extraction("7%")
    comparison = compare_annexes(old=old, new=new)
    corrupt = comparison.model_copy(
        update={
            "changes": [
                AnnexDifference(
                    operation="replace",
                    old=[
                        AnnexElementReference(
                            position=0, text="invented", locator=old.elements[0].locator
                        )
                    ],
                    new=[
                        AnnexElementReference(
                            position=0, text="7%", locator=new.elements[0].locator
                        )
                    ],
                    explanation="untrusted",
                )
            ],
            "ready": True,
        }
    )
    result = prepare_annex_patch(
        baseline=baseline("5%"),
        old=old,
        new=new,
        comparison=corrupt,
        effective_date=date.today(),
        package_complete=True,
    )
    assert not result.ready and result.patches == []


def test_padded_markdown_row_preserves_separate_nonlegal_delimiter() -> None:

    current, old, new = linewise_case()
    current.elements[0].text = current.canonical_text = "| 1001     | Wheat    | 5% |"
    current.elements.append(
        ExtractedAnnexElement(
            kind="text", text="| :--- | ---: | --- |", canonical_chunk_id="scaffold"
        )
    )
    result = prepare_linewise_case(current, old, new)
    assert result.ready
    assert result.patches[0].new_text == "| 1001     | Wheat    | 7% |"
    assert "scaffold" in result.unchanged


def test_ready_comparison_cannot_hide_extraction_uncertainty() -> None:
    from onyx.regulatory.amendments.annexes.comparison import (
        annex_snapshot_hash,
        compare_annexes,
    )
    from onyx.regulatory.amendments.annexes.patch_plan import prepare_annex_patch

    old, new = extraction("5%"), extraction("7%")
    comparison = compare_annexes(old=old, new=new)
    new.elements[0].status = "uncertain"
    comparison = comparison.model_copy(
        update={"new_snapshot_sha256": annex_snapshot_hash(new)}
    )
    result = prepare_annex_patch(
        baseline=baseline("5%"),
        old=old,
        new=new,
        comparison=comparison,
        effective_date=date.today(),
        package_complete=True,
    )
    assert not result.ready and "incomplete_extraction" in result.issues


@pytest.mark.parametrize("approved", ["15%", "5%5", "0.5%", "5% surcharge"])
def test_partial_cell_value_cannot_patch_an_approved_numeric_overlay(
    approved: str,
) -> None:
    from onyx.regulatory.amendments.annexes.comparison import compare_annexes
    from onyx.regulatory.amendments.annexes.patch_plan import prepare_annex_patch

    current = baseline(f"| A | {approved} |")
    current.elements[0].semantic_key = "approved-row-lineage"
    old, new = extraction("5%"), extraction("7%")
    result = prepare_annex_patch(
        baseline=current,
        old=old,
        new=new,
        comparison=compare_annexes(old=old, new=new),
        effective_date=date.today(),
        package_complete=True,
    )
    assert not result.ready
    assert "canonical_correspondence_unresolved" in result.issues
    assert not result.patches


def linewise_case() -> tuple[AnnexBaseline, AnnexExtraction, AnnexExtraction]:
    from onyx.regulatory.amendments.annexes.models import AnnexLocator

    rows = [
        ("Code", "Product", "Rate"),
        ("1001", "Wheat", "5%"),
        ("2001", "Rice", "15%"),
    ]
    elements = [
        ExtractedAnnexElement(
            kind="table_cell",
            text=text,
            extraction_method="vision",
            locator=AnnexLocator(
                page=2,
                normalized_box=(
                    0.1 + col * 0.25,
                    0.2 + row * 0.1,
                    0.35 + col * 0.25,
                    0.3 + row * 0.1,
                ),
            ),
        )
        for row, values in enumerate(rows)
        for col, text in enumerate(values)
    ]
    old = AnnexExtraction(source_sha256="old", mime_type="text/html", elements=elements)
    new = old.model_copy(deep=True)
    new.source_sha256 = "new"
    new.elements[5].text = "7%"
    current = baseline("\n".join(e.text for e in old.elements))
    current.canonical_amendment_chunk_ids = []
    current.elements[0].semantic_key = "canonical-row"
    current.elements[0].kind = "text"
    return current, old, new


def prepare_linewise_case(
    current: AnnexBaseline, old: AnnexExtraction, new: AnnexExtraction
) -> AnnexPatchPlan:
    from onyx.regulatory.amendments.annexes.comparison import annex_snapshot_hash
    from onyx.regulatory.amendments.annexes.models import (
        AnnexComparison,
        AnnexCoverage,
        AnnexDifference,
        AnnexElementReference,
    )
    from onyx.regulatory.amendments.annexes.patch_plan import prepare_annex_patch

    comparison = AnnexComparison(
        old_source_sha256=old.source_sha256,
        new_source_sha256=new.source_sha256,
        old_snapshot_sha256=annex_snapshot_hash(old),
        new_snapshot_sha256=annex_snapshot_hash(new),
        coverage=AnnexCoverage(
            old_positions=list(range(len(old.elements))),
            new_positions=list(range(len(new.elements))),
            old_pages=[2],
            new_pages=[2],
            method="native_structure",
        ),
        changes=[
            AnnexDifference(
                operation="replace",
                old=[
                    AnnexElementReference(
                        position=5,
                        text=old.elements[5].text,
                        locator=old.elements[5].locator,
                    )
                ],
                new=[
                    AnnexElementReference(
                        position=5,
                        text=new.elements[5].text,
                        locator=new.elements[5].locator,
                    )
                ],
                explanation="Changed rate",
            )
        ],
        issues=[],
        ready=True,
    )
    return prepare_annex_patch(
        baseline=current,
        old=old,
        new=new,
        comparison=comparison,
        effective_date=date(2026, 9, 10),
        package_complete=True,
    )


def test_native_linewise_table_maps_complete_cell_with_unique_row_context() -> None:
    current, old, new = linewise_case()
    result = prepare_linewise_case(current, old, new)
    assert result.ready, result.issues
    assert result.patches[0].new_text == current.elements[0].text.replace(
        "\n5%\n", "\n7%\n"
    )
    assert result.patches[0].new_text is not None
    assert "15%" in result.patches[0].new_text


@pytest.mark.parametrize(
    "defect",
    [
        "missing_geometry",
        "overlapping_cells",
        "missing_cell",
        "duplicate_canonical_row",
        "duplicate_source_row",
        "stale_overlay",
        "partial_cell",
        "wrong_neighbor",
    ],
)
def test_native_linewise_table_refuses_unproven_row_context(defect: str) -> None:
    current, old, new = linewise_case()
    if defect == "missing_geometry":
        old.elements[4].locator.normalized_box = None
    elif defect == "overlapping_cells":
        old.elements[4].locator.normalized_box = old.elements[5].locator.normalized_box
    elif defect == "missing_cell":
        old.elements[4].kind = "text"
    elif defect == "duplicate_source_row":
        for i in range(3):
            old.elements[6 + i].text = old.elements[3 + i].text
            new.elements[6 + i].text = old.elements[3 + i].text
    else:
        text = current.elements[0].text
        if defect == "duplicate_canonical_row":
            text += "\n1001\nWheat\n5%"
        elif defect == "stale_overlay":
            text = text.replace("\n5%\n", "\n8%\n") + "\nOther\nThing\n5%"
            current.canonical_amendment_chunk_ids = ["canonical-rate"]
        elif defect == "partial_cell":
            text = text.replace("\n5%\n", "\n5% surcharge\n")
        else:
            text = text.replace("Wheat", "Changed product")
        current.elements[0].text = current.canonical_text = text
    result = prepare_linewise_case(current, old, new)
    assert not result.ready
    assert "canonical_correspondence_unresolved" in result.issues


def test_pipe_cell_boundaries_precede_substring_uniqueness() -> None:
    current, old, new = linewise_case()
    current.elements[0].text = current.canonical_text = (
        "| Code | Product | Rate |\n| 1001 | Wheat | 5% |\n| 2001 | Rice | 15% |"
    )
    result = prepare_linewise_case(current, old, new)
    assert result.ready, result.issues
    assert (
        result.patches[0].new_text
        == "| Code | Product | Rate |\n| 1001 | Wheat | 7% |\n| 2001 | Rice | 15% |"
    )


@pytest.mark.parametrize(
    "defect",
    [
        None,
        "missing_geometry",
        "different_page",
        "no_anchor",
        "repeated_text",
        "moved_region",
    ],
)
def test_unchanged_text_footnote_requires_unique_aligned_page_neighborhood(
    defect: str | None,
) -> None:
    from onyx.regulatory.amendments.annexes.models import AnnexLocator

    current, old, new = linewise_case()
    for view in (old, new):
        view.elements.append(
            ExtractedAnnexElement(
                kind="text",
                text="Continuation remains",
                locator=AnnexLocator(page=2, normalized_box=(0.1, 0.6, 0.8, 0.65)),
            )
        )
        view.elements.append(
            ExtractedAnnexElement(
                kind="text",
                text="Original page two",
                locator=AnnexLocator(page=2, normalized_box=(0.75, 0.9, 0.9, 0.95)),
            )
        )
    new.elements[-1].kind = "footnote"
    current.elements[0].text += "\nContinuation remains\nOriginal page two"
    current.canonical_text = current.elements[0].text
    if defect == "missing_geometry":
        new.elements[-1].locator.normalized_box = None
    elif defect == "different_page":
        new.elements[-1].locator.page = 3
    elif defect == "no_anchor":
        for e in new.elements[:-1]:
            e.locator.normalized_box = None
    elif defect == "repeated_text":
        new.elements.append(new.elements[-1].model_copy(deep=True))
    elif defect == "moved_region":
        new.elements[-1].locator.normalized_box = (0.1, 0.05, 0.4, 0.1)
    result = prepare_linewise_case(current, old, new)
    assert result.ready is (defect is None), result.issues


def test_pipe_row_cannot_patch_unrelated_scalar_after_approved_correction() -> None:
    current, old, new = linewise_case()
    current.elements[0].text = current.canonical_text = (
        "| Code | Product | Rate |\n| 1001 | Wheat | 8% |\n| 2001 | Rice | 15% |\n| Other | Thing | 5% |"
    )
    current.canonical_amendment_chunk_ids = ["canonical-rate"]
    result = prepare_linewise_case(current, old, new)
    assert not result.ready
    assert "canonical_correspondence_unresolved" in result.issues
    assert not result.patches


def test_vertical_row_overlap_cannot_authorize_native_cell() -> None:
    current, old, new = linewise_case()
    for element in old.elements[3:6]:
        box = element.locator.normalized_box
        assert box is not None
        element.locator.normalized_box = (box[0], box[1] - 0.03, box[2], box[3] - 0.03)
    result = prepare_linewise_case(current, old, new)
    assert not result.ready
    assert "canonical_correspondence_unresolved" in result.issues


def test_pipe_scalar_without_row_or_semantic_authority_refuses() -> None:
    from onyx.regulatory.amendments.annexes.comparison import compare_annexes
    from onyx.regulatory.amendments.annexes.patch_plan import prepare_annex_patch

    current = baseline("| A | 5% | B | 15% |")
    current.elements[0].semantic_key = "unrelated-row"
    old, new = extraction("5%"), extraction("7%")
    result = prepare_annex_patch(
        baseline=current,
        old=old,
        new=new,
        comparison=compare_annexes(old=old, new=new),
        effective_date=date.today(),
        package_complete=True,
    )
    assert not result.ready
