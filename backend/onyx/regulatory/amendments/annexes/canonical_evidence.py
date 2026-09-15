"""Indexed annex state with supplementary, explicitly bound image evidence."""

import hashlib

from onyx.regulatory.amendments.annexes.models import (
    AnnexBaseline,
    AnnexCanonicalEvidence,
    AnnexCanonicalImage,
    AnnexExtraction,
)


def canonical_baseline_extraction(baseline: AnnexBaseline) -> AnnexExtraction:
    canonical_ids = {
        element.canonical_chunk_id
        for element in baseline.elements
        if element.canonical_chunk_id
    }
    images: list[AnnexCanonicalImage] = []
    issues: list[str] = []
    for original in baseline.originals:
        bindings = set(original.canonical_chunk_ids) & canonical_ids
        if not bindings:
            continue
        if not (original.mime_type or "").startswith("image/"):
            if original.linked_from_chunks:
                issues.append("canonical_image_unavailable")
            continue
        if not original.available or not original.sha256:
            issues.append("canonical_image_unavailable")
            continue
        images.append(
            AnnexCanonicalImage(
                file_id=original.file_id,
                sha256=original.sha256,
                mime_type=original.mime_type or "",
                canonical_chunk_ids=sorted(bindings),
                page=len(images) + 1,
            )
        )
    bound_ids = {image.file_id for image in images}
    if any(
        element.image_file_id and element.image_file_id not in bound_ids
        for element in baseline.elements
    ):
        issues.append("canonical_image_unavailable")
    return AnnexExtraction(
        source_sha256=hashlib.sha256(baseline.canonical_text.encode()).hexdigest(),
        mime_type="text/markdown",
        elements=[element.model_copy(deep=True) for element in baseline.elements],
        canonical_evidence=AnnexCanonicalEvidence(
            baseline_sha256=baseline.baseline_sha256,
            images=images,
        ),
        issues=sorted(set(issues)),
    )


def validate_canonical_baseline(
    baseline: AnnexBaseline, extraction: AnnexExtraction
) -> list[str]:
    expected = canonical_baseline_extraction(baseline)
    if extraction.canonical_evidence is None:
        # Previously frozen canonical-only reviews predate explicit bindings.
        expected = expected.model_copy(update={"canonical_evidence": None})
        if any(element.image_file_id for element in baseline.elements):
            return ["canonical_image_evidence_missing"]
    return [] if extraction == expected else ["canonical_baseline_evidence_mismatch"]


def render_selected_text(extraction: AnnexExtraction, positions: list[int]) -> str:
    """Retain grounded table geometry when visual cells become chunk text."""
    elements = [extraction.elements[position] for position in positions]
    if not any(element.kind == "table_cell" for element in elements) or not all(
        element.extraction_method == "vision"
        and element.locator.page is not None
        and element.locator.normalized_box is not None
        for element in elements
    ):
        return "\n".join(element.text for element in elements)
    from onyx.regulatory.amendments.pdf_vision import pdf_transcript

    pages = sorted(
        {
            element.locator.page
            for element in elements
            if element.locator.page is not None
        }
    )
    local_pages = {page: index + 1 for index, page in enumerate(pages)}
    selected = extraction.model_copy(
        update={
            "page_count": len(pages),
            "elements": [
                element.model_copy(
                    update={
                        "locator": element.locator.model_copy(
                            update={"page": local_pages[element.locator.page or 0]}
                        )
                    }
                )
                for element in elements
            ],
        }
    )
    return pdf_transcript(selected)
