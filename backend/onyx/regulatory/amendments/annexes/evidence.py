"""Server-derived annex scope with original coordinates and immutable view identity."""

import hashlib
import re
from collections.abc import Mapping
from io import BytesIO
from typing import Literal, cast
from uuid import uuid4

from onyx.configs.constants import FileOrigin
from onyx.db.regulatory_annexes import normalize_annex_label
from onyx.file_store.file_store import FileStore
from onyx.regulatory.amendments.annexes.context_dependencies import context_hash
from onyx.regulatory.amendments.annexes.models import (
    AnnexBaseline,
    AnnexEvidenceElementMap,
    AnnexEvidencePage,
    AnnexEvidenceParent,
    AnnexEvidenceView,
    AnnexExtraction,
    AnnexLocator,
    AnnexOriginalEvidence,
    AnnexRenderedPage,
    AnnexReviewEvidence,
    AnnexReviewEvidenceScope,
    ExtractedAnnexElement,
)

_STANDALONE_LABEL = re.compile(
    r"(?:EK|annex|appendix)[\s:–—-]*(?:\d+[a-z]?|[ivxlcdm]+[a-z]?|[a-z])(?:[/.-][0-9a-z]+)*",
    re.IGNORECASE,
)


def _boundary_label(element: ExtractedAnnexElement) -> str | None:
    if element.extraction_method != "native" or element.kind != "text":
        return None
    text = element.text.strip()
    if not _STANDALONE_LABEL.fullmatch(text):
        return None
    return normalize_annex_label(text)


def _boundary_top(extraction: AnnexExtraction, position: int) -> float:
    heading = extraction.elements[position]
    locator = heading.locator
    if locator.normalized_box is None:
        raise ValueError("annex boundary coordinates unavailable")
    top = locator.normalized_box[1]
    for element in extraction.elements:
        box = element.locator.normalized_box
        if (
            element.text.strip() == heading.text.strip()
            and element.locator.page == locator.page
            and box is not None
            and box[3] >= top
            and box[1] <= locator.normalized_box[3]
        ):
            top = min(top, box[1])
    if any(
        element.extraction_method == "native"
        and element.locator.page == locator.page
        and element.locator.normalized_box is not None
        and element.locator.normalized_box[1] < locator.normalized_box[1]
        and element.locator.normalized_box[3] > top
        and element.text.strip() != heading.text.strip()
        for element in extraction.elements
    ):
        raise ValueError("annex heading expansion overlaps native content")
    return top


def evidence_view_hash(extraction: AnnexExtraction) -> str:
    view = extraction.evidence_view
    if view is None:
        raise ValueError("scoped evidence view missing")
    return context_hash(
        {
            "view": view.model_dump(mode="json", exclude={"sha256"}),
            "source_sha256": extraction.source_sha256,
            "mime_type": extraction.mime_type,
            "elements": [
                element.model_dump(mode="json") for element in extraction.elements
            ],
        }
    )


def selected_evidence_pages(extraction: AnnexExtraction) -> list[int]:
    if extraction.evidence_view is not None:
        return [page.view_page for page in extraction.evidence_view.pages]
    return list(range(1, (extraction.page_count or 0) + 1))


def identical_evidence_scope(old: AnnexExtraction, new: AnnexExtraction) -> bool:
    if old.source_sha256 != new.source_sha256:
        return False
    if old.evidence_view is None and new.evidence_view is None:
        return True
    return (
        old.evidence_view is not None
        and new.evidence_view is not None
        and old.evidence_view.sha256 == new.evidence_view.sha256
        and evidence_view_hash(old) == old.evidence_view.sha256
        and evidence_view_hash(new) == new.evidence_view.sha256
    )


def select_annex_evidence_view(
    *,
    extraction: AnnexExtraction,
    original: AnnexOriginalEvidence,
    annex_label: str,
    canonical_labels: list[str],
    canonical_chunk_ids: list[str],
    source_labels: list[str] | None = None,
) -> AnnexExtraction:
    """Select native, standalone annex boundaries corroborated by relational scope.

    The caller supplies labels/IDs from scoped canonical rows, never model output.
    Source page count remains the actual original's count; selected page coverage
    lives in the view and does not masquerade as an original-document property.
    """
    if extraction.evidence_view is not None:
        raise ValueError("nested evidence scope")
    if (
        not original.available
        or original.sha256 != extraction.source_sha256
        or original.mime_type != extraction.mime_type
    ):
        raise ValueError("original evidence unavailable or hash mismatch")
    label = normalize_annex_label(annex_label)
    if (
        label not in {normalize_annex_label(value) for value in canonical_labels}
        or not canonical_chunk_ids
    ):
        raise ValueError("canonical annex scope is not corroborated")
    boundaries = [
        (position, found)
        for position, element in enumerate(extraction.elements)
        if (found := _boundary_label(element)) is not None
    ]
    starts = [position for position, found in boundaries if found == label]
    if not starts and extraction.mime_type.endswith("spreadsheetml.sheet"):
        sheets = {
            element.locator.sheet
            for element in extraction.elements
            if element.locator.sheet
            and normalize_annex_label(element.locator.sheet) == label
        }
        if len(sheets) != 1:
            raise ValueError("annex sheet boundary missing or ambiguous")
        selected = [
            index
            for index, element in enumerate(extraction.elements)
            if element.locator.sheet in sheets
        ]
        if any(
            extraction.elements[index].extraction_method != "native"
            for index in selected
        ):
            raise ValueError("annex sheet boundary is not native")
        return _build_selected_view(
            extraction,
            original,
            label,
            canonical_chunk_ids,
            selected,
            [],
            [],
            "native_sheet",
        )
    if not starts and extraction.mime_type.startswith("image/"):
        source_scope = {normalize_annex_label(value) for value in source_labels or []}
        if set(original.canonical_chunk_ids) != set(
            canonical_chunk_ids
        ) and source_scope != {label}:
            raise ValueError(
                "whole image lacks canonical binding or source occurrence label"
            )
        other_labels = {
            normalize_annex_label(element.text.strip())
            for element in extraction.elements
            if _STANDALONE_LABEL.fullmatch(element.text.strip())
        } - {label}
        if other_labels or extraction.page_count is None:
            raise ValueError("whole image annex scope ambiguous")
        pages = [
            AnnexEvidencePage(
                parent_index=0,
                original_page=page,
                view_page=page,
                normalized_box=(0, 0, 1, 1),
            )
            for page in range(1, extraction.page_count + 1)
        ]
        return _build_selected_view(
            extraction,
            original,
            label,
            canonical_chunk_ids,
            list(range(len(extraction.elements))),
            pages,
            [],
            "bound_whole_original",
        )
    if len(starts) != 1:
        raise ValueError("annex native boundary missing or ambiguous")
    start = starts[0]
    end = next(
        (position for position, _ in boundaries if position > start),
        len(extraction.elements),
    )
    visual = extraction.mime_type.startswith(("application/pdf", "image/"))
    pages: list[AnnexEvidencePage] = []
    if visual:
        first = extraction.elements[start].locator
        last = (
            extraction.elements[end].locator if end < len(extraction.elements) else None
        )
        if (
            extraction.page_count is None
            or first.page is None
            or first.normalized_box is None
            or (last is not None and (last.page is None or last.normalized_box is None))
        ):
            raise ValueError("annex boundary coordinates unavailable")
        first_key = (first.page, _boundary_top(extraction, start))
        end_key = (
            (last.page, _boundary_top(extraction, end))
            if last is not None
            and last.page is not None
            and last.normalized_box is not None
            else (extraction.page_count + 1, 0.0)
        )
        if end_key <= first_key:
            raise ValueError("annex boundaries unordered")
        selected: list[int] = []
        for position, element in enumerate(extraction.elements):
            locator = element.locator
            if locator.page is None or locator.normalized_box is None:
                raise ValueError("annex element coordinates unavailable")
            top, bottom = (
                (locator.page, locator.normalized_box[1]),
                (locator.page, locator.normalized_box[3]),
            )
            if bottom <= first_key or top >= end_key:
                continue
            if top < first_key or bottom > end_key:
                raise ValueError("annex boundary intersects source element")
            selected.append(position)
        last_page = end_key[0] if end_key[1] > 0 else end_key[0] - 1
        if end_key[1] > 0 and not any(
            extraction.elements[position].locator.page == last_page
            for position in selected
        ):
            last_page -= 1
        for page in range(first.page, last_page + 1):
            pages.append(
                AnnexEvidencePage(
                    parent_index=0,
                    original_page=page,
                    view_page=page,
                    normalized_box=(
                        0,
                        first_key[1] if page == first.page else 0,
                        1,
                        end_key[1] if page == end_key[0] else 1,
                    ),
                )
            )
    else:
        selected = list(range(start, end))
        if any(element.kind == "footnote" for element in extraction.elements[end:]):
            raise ValueError("annex trailing footnote scope unresolved")
    return _build_selected_view(
        extraction,
        original,
        label,
        canonical_chunk_ids,
        selected,
        pages,
        [start, *([end] if end < len(extraction.elements) else [])],
        "native_boundaries",
    )


def _build_selected_view(
    extraction: AnnexExtraction,
    original: AnnexOriginalEvidence,
    label: str,
    canonical_chunk_ids: list[str],
    selected: list[int],
    pages: list[AnnexEvidencePage],
    boundaries: list[int],
    method: Literal["native_boundaries", "bound_whole_original", "native_sheet"],
) -> AnnexExtraction:
    parent = AnnexEvidenceParent(
        file_id=original.file_id,
        sha256=extraction.source_sha256,
        mime_type=extraction.mime_type,
        extraction_sha256=context_hash(extraction.model_dump(mode="json")),
        element_count=len(extraction.elements),
        page_count=extraction.page_count,
        canonical_chunk_ids=canonical_chunk_ids,
    )
    view = AnnexEvidenceView(
        sha256="",
        label=label,
        parents=[parent],
        pages=pages,
        selected_positions=selected,
        boundary_positions=boundaries,
        selection_method=method,
        element_mappings=[
            AnnexEvidenceElementMap(
                parent_index=0,
                original_position=position,
                original_locator=extraction.elements[position].locator.model_copy(
                    deep=True
                ),
                view_position=index,
            )
            for index, position in enumerate(selected)
        ],
    )
    result = extraction.model_copy(
        deep=True,
        update={
            "elements": [
                extraction.elements[position].model_copy(deep=True)
                for position in selected
            ],
            "evidence_view": view,
        },
    )
    result.evidence_view = view.model_copy(
        update={"sha256": evidence_view_hash(result)}
    )
    return result


def validate_evidence_view(extraction: AnnexExtraction) -> list[str]:
    view = extraction.evidence_view
    if view is None:
        return []
    issues: list[str] = []
    if evidence_view_hash(extraction) != view.sha256:
        issues.append("evidence_view_integrity_mismatch")
    if not view.parents or extraction.source_sha256 != view.parents[0].sha256:
        issues.append("evidence_parent_hash_mismatch")
    if len(view.selected_positions) != len(
        extraction.elements
    ) or view.selected_positions != sorted(set(view.selected_positions)):
        issues.append("evidence_element_mapping_mismatch")
    if len(view.element_mappings) != len(extraction.elements) or [
        mapping.view_position for mapping in view.element_mappings
    ] != list(range(len(extraction.elements))):
        issues.append("evidence_element_mapping_mismatch")
    seen_elements: set[tuple[int, int]] = set()
    for element, mapping in zip(extraction.elements, view.element_mappings):
        key = (mapping.parent_index, mapping.original_position)
        if key in seen_elements or not 0 <= mapping.parent_index < len(view.parents):
            issues.append("evidence_element_mapping_mismatch")
            continue
        seen_elements.add(key)
        if (
            not 0
            <= mapping.original_position
            < view.parents[mapping.parent_index].element_count
        ):
            issues.append("evidence_element_mapping_mismatch")
        if element.locator.model_dump(
            exclude={"page"}
        ) != mapping.original_locator.model_dump(exclude={"page"}):
            issues.append("evidence_coordinate_mapping_mismatch")
        if mapping.original_locator.page is not None and not any(
            page.parent_index == mapping.parent_index
            and page.original_page == mapping.original_locator.page
            and page.view_page == element.locator.page
            for page in view.pages
        ):
            issues.append("evidence_coordinate_mapping_mismatch")
    if len({page.view_page for page in view.pages}) != len(view.pages):
        issues.append("evidence_page_mapping_mismatch")
    for page in view.pages:
        if page.parent_index < 0 or page.parent_index >= len(view.parents):
            issues.append("evidence_parent_mapping_mismatch")
            continue
        parent = view.parents[page.parent_index]
        left, top, right, bottom = page.normalized_box
        if (
            parent.page_count is None
            or not 1 <= page.original_page <= parent.page_count
            or not (0 <= left < right <= 1 and 0 <= top < bottom <= 1)
        ):
            issues.append("evidence_page_mapping_mismatch")
    return sorted(set(issues))


def validate_baseline_evidence_view(
    baseline: AnnexBaseline, extraction: AnnexExtraction
) -> list[str]:
    issues = validate_evidence_view(extraction)
    view = extraction.evidence_view
    if view is None:
        return issues
    canonical_ids = {
        element.canonical_chunk_id
        for element in baseline.elements
        if element.canonical_chunk_id
    }
    covered: set[str] = set()
    for parent in view.parents:
        originals = [
            original
            for original in baseline.originals
            if original.file_id == parent.file_id
            and original.available
            and original.sha256 == parent.sha256
            and original.mime_type == parent.mime_type
        ]
        if len(originals) != 1:
            issues.append("evidence_parent_outside_baseline")
            continue
        bindings = set(parent.canonical_chunk_ids)
        if not bindings or not bindings.issubset(
            canonical_ids & set(originals[0].canonical_chunk_ids)
        ):
            issues.append("evidence_parent_binding_mismatch")
        covered.update(bindings)
    if covered != canonical_ids:
        issues.append("evidence_baseline_coverage_incomplete")
    return sorted(set(issues))


def choose_original_evidence(
    originals: list[AnnexOriginalEvidence], *, canonical_chunk_ids: list[str]
) -> list[AnnexOriginalEvidence]:
    """Prefer one complete original; disjoint pieces must prove canonical order."""
    expected = set(canonical_chunk_ids)
    if not expected or len(expected) != len(canonical_chunk_ids):
        raise ValueError("canonical scope missing or duplicated")
    available = [
        original for original in originals if original.available and original.sha256
    ]
    complete = [
        original
        for original in available
        if set(original.canonical_chunk_ids) == expected
    ]
    if len(complete) == 1:
        return complete
    if len(complete) > 1:
        raise ValueError("multiple complete originals are ambiguous")
    covered = [
        chunk_id for original in available for chunk_id in original.canonical_chunk_ids
    ]
    if set(covered) != expected:
        raise ValueError("original scope missing")
    if covered != canonical_chunk_ids:
        raise ValueError("original scope unordered or overlapping")
    return available


def freeze_review_original(
    store: FileStore,
    *,
    scope: AnnexReviewEvidenceScope,
    side: str,
    original: AnnexOriginalEvidence,
) -> AnnexReviewEvidence:
    if side not in ("old", "new"):
        raise ValueError("invalid evidence side")
    allowed = (
        scope.old_original_file_ids if side == "old" else scope.new_original_file_ids
    )
    if original.file_id not in allowed:
        raise ValueError("original outside authorized review scope")
    if not original.available or not original.sha256 or not original.mime_type:
        raise ValueError("original unavailable")
    with store.read_file(original.file_id) as stream:
        content = stream.read(25 * 1024 * 1024 + 1)
    if (
        len(content) > 25 * 1024 * 1024
        or hashlib.sha256(content).hexdigest() != original.sha256
    ):
        raise ValueError("original integrity mismatch")
    evidence_id = uuid4()
    file_id = store.save_file(
        BytesIO(content),
        "Frozen annex original",
        FileOrigin.OTHER,
        original.mime_type,
        file_metadata={
            "annex_review_scope": scope.storage_identity(),
            "sha256": original.sha256,
            "parent_sha256": original.sha256,
            "parent_file_id": original.file_id,
        },
    )
    return AnnexReviewEvidence(
        id=evidence_id,
        side="old" if side == "old" else "new",
        kind="original",
        file_id=file_id,
        sha256=original.sha256,
        mime_type=original.mime_type,
        byte_count=len(content),
        parent_file_id=original.file_id,
        parent_sha256=original.sha256,
    )


def freeze_review_pages(
    store: FileStore,
    *,
    scope: AnnexReviewEvidenceScope,
    original: AnnexReviewEvidence,
    extraction: AnnexExtraction | None = None,
) -> tuple[list[AnnexRenderedPage], list[AnnexReviewEvidence]]:
    """Render the frozen original once; return those same pages to the comparator."""
    from onyx.regulatory.amendments.annexes.comparison import comparison_page_evidence
    from onyx.regulatory.amendments.annexes.rendering import render_annex_pages
    from onyx.utils.process_isolation import run_in_isolated_process

    record = store.read_file_record(original.file_id)
    if (
        not isinstance(record.file_metadata, Mapping)
        or cast(Mapping[str, object], record.file_metadata).get("annex_review_scope")
        != scope.storage_identity()
        or original.kind != "original"
    ):
        raise ValueError("frozen original outside review scope")
    with store.read_file(original.file_id) as stream:
        content = stream.read(25 * 1024 * 1024 + 1)
    if (
        len(content) != original.byte_count
        or hashlib.sha256(content).hexdigest() != original.sha256
    ):
        raise ValueError("frozen original integrity mismatch")
    if not original.mime_type.startswith(("image/", "application/pdf")):
        return [], []
    pages = run_in_isolated_process(
        render_annex_pages, content, original.mime_type, timeout=30
    )
    mappings: dict[int, AnnexEvidencePage] = {}
    if extraction is not None and extraction.evidence_view is not None:
        if validate_evidence_view(extraction):
            raise ValueError("invalid evidence view")
        view = extraction.evidence_view
        parent_indices = [
            index
            for index, parent in enumerate(view.parents)
            if parent.file_id == original.parent_file_id
            and parent.sha256 == original.parent_sha256
        ]
        if len(parent_indices) != 1:
            raise ValueError("frozen original outside selected parents")
        mappings = {
            mapping.original_page: mapping
            for mapping in view.pages
            if mapping.parent_index == parent_indices[0]
        }
        pages = [page for page in pages if page.page in mappings]
        if len(pages) != len(mappings):
            raise ValueError("selected page unavailable")
    evidence: list[AnnexReviewEvidence] = []
    compared_pages: list[AnnexRenderedPage] = []
    for page in pages:
        mapping = mappings.get(page.page)
        region = mapping.normalized_box if mapping else (0, 0, 1, 1)
        compared_pages.append(
            page.model_copy(update={"page": mapping.view_page}) if mapping else page
        )
        for image in comparison_page_evidence(page, normalized_box=region):
            digest = hashlib.sha256(image.png).hexdigest()
            evidence_id = uuid4()
            file_id = store.save_file(
                BytesIO(image.png),
                "Frozen annex comparison",
                FileOrigin.OTHER,
                "image/png",
                file_metadata={
                    "annex_review_scope": scope.storage_identity(),
                    "sha256": digest,
                    "parent_sha256": original.parent_sha256,
                    "parent_file_id": original.parent_file_id,
                },
            )
            evidence.append(
                AnnexReviewEvidence(
                    id=evidence_id,
                    side=original.side,
                    kind=image.kind,
                    file_id=file_id,
                    sha256=digest,
                    mime_type="image/png",
                    byte_count=len(image.png),
                    parent_file_id=original.parent_file_id,
                    parent_sha256=original.parent_sha256,
                    locator=AnnexLocator(
                        page=page.page,
                        normalized_box=image.normalized_box,
                        original_box=(
                            image.normalized_box[0] * page.width,
                            image.normalized_box[1] * page.height,
                            image.normalized_box[2] * page.width,
                            image.normalized_box[3] * page.height,
                        ),
                        original_width=page.width,
                        original_height=page.height,
                        coordinate_system="top_left_points"
                        if original.mime_type == "application/pdf"
                        else "top_left_pixels",
                    ),
                )
            )
    return compared_pages, evidence


def combine_annex_evidence_views(
    views: list[AnnexExtraction], *, canonical_chunk_ids: list[str]
) -> AnnexExtraction:
    if not views or any(
        view.evidence_view is None or validate_evidence_view(view) for view in views
    ):
        raise ValueError("original scope missing or invalid")
    scopes = [view.evidence_view for view in views if view.evidence_view is not None]
    if any(not scope.pages for scope in scopes):
        raise ValueError("combined native originals need explicit structural ordering")
    if len({parent.file_id for scope in scopes for parent in scope.parents}) != len(
        scopes
    ):
        raise ValueError("original parents duplicated")
    if len({scope.label for scope in scopes}) != 1 or any(
        len(scope.parents) != 1 for scope in scopes
    ):
        raise ValueError("original scope ambiguous")
    covered = [
        chunk_id
        for scope in scopes
        for parent in scope.parents
        for chunk_id in parent.canonical_chunk_ids
    ]
    if covered != canonical_chunk_ids or len(set(covered)) != len(covered):
        raise ValueError("original scope unordered or incomplete")
    parents: list[AnnexEvidenceParent] = []
    pages: list[AnnexEvidencePage] = []
    elements: list[ExtractedAnnexElement] = []
    mappings: list[AnnexEvidenceElementMap] = []
    positions: list[int] = []
    boundaries: list[int] = []
    original_offset = 0
    for parent_index, (extraction, scope) in enumerate(zip(views, scopes)):
        parents.extend(scope.parents)
        page_map: dict[int, int] = {}
        for page in scope.pages:
            view_page = len(pages) + 1
            page_map[page.view_page] = view_page
            pages.append(
                page.model_copy(
                    update={"parent_index": parent_index, "view_page": view_page}
                )
            )
        for element, mapping in zip(extraction.elements, scope.element_mappings):
            copied = element.model_copy(deep=True)
            if copied.locator.page is not None:
                if copied.locator.page not in page_map:
                    raise ValueError("original element page missing")
                copied.locator.page = page_map[copied.locator.page]
            mappings.append(
                mapping.model_copy(
                    update={
                        "parent_index": parent_index,
                        "view_position": len(elements),
                    }
                )
            )
            elements.append(copied)
        positions.extend(
            original_offset + position for position in scope.selected_positions
        )
        boundaries.extend(
            original_offset + position for position in scope.boundary_positions
        )
        original_offset += scope.parents[0].element_count
    scope = AnnexEvidenceView(
        sha256="",
        label=scopes[0].label,
        parents=parents,
        pages=pages,
        selected_positions=positions,
        boundary_positions=boundaries,
        element_mappings=mappings,
        selection_method="ordered_bound_originals",
    )
    result = views[0].model_copy(
        deep=True,
        update={
            "elements": elements,
            "evidence_view": scope,
            "issues": sorted({issue for view in views for issue in view.issues}),
        },
    )
    result.evidence_view = scope.model_copy(
        update={"sha256": evidence_view_hash(result)}
    )
    return result
