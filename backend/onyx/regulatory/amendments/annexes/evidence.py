"""Server-derived annex scope with original coordinates and immutable view identity."""

import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping
from io import BytesIO
from typing import Literal, cast
from uuid import UUID, uuid4

from onyx.configs.constants import FileOrigin
from onyx.db.models import RegulatorySourceAsset
from onyx.db.regulatory_annexes import normalize_annex_label
from onyx.file_store.file_store import FileStore
from onyx.regulatory.amendments.annexes.context_dependencies import context_hash
from onyx.regulatory.amendments.annexes.models import (
    AnnexBaseline,
    AnnexComparison,
    AnnexEvidenceElementMap,
    AnnexEvidencePage,
    AnnexEvidenceParent,
    AnnexEvidenceView,
    AnnexExtraction,
    AnnexLocator,
    AnnexNewEvidenceRemapping,
    AnnexOriginalEvidence,
    AnnexRenderedPage,
    AnnexReviewEvidence,
    AnnexReviewEvidenceScope,
    ExtractedAnnexElement,
    SourceLink,
)

_STANDALONE_LABEL = re.compile(
    r"(?:EK|annex|appendix)[\s:–—-]*(?:\d+[a-z]?|[ivxlcdm]+[a-z]?|[a-z])(?:[/.-][0-9a-z]+)*",
    re.IGNORECASE,
)


_SOURCE_ANNEX_WORD = re.compile(r"\b(?:EK|annex|appendix)\b", re.IGNORECASE)
_SOURCE_ANNEX_REFERENCE = re.compile(
    r"(?<![\w/\\.-])" + _STANDALONE_LABEL.pattern + r"(?![\w/\\.–—…-])",
    re.IGNORECASE,
)
_SOURCE_CONJUNCTION = r"(?:ve(?:ya(?:hut)?)?|ya(?:hut|\s+da)|and|or|ile)"
_SOURCE_SHORTHAND = re.compile(
    rf"^\s*(?:{_SOURCE_CONJUNCTION}(?:\s*/\s*{_SOURCE_CONJUNCTION})*\b|[,;&+/–—-])"
    r"\s*(?:[0-9]|[ivxlcdm]+\b|[a-z]\b)",
    re.IGNORECASE,
)


def _source_occurrence_label(value: str) -> str | None:
    """Read one complete reference from acquired link text, never a file location."""
    text = re.sub(
        r"(?:[a-z][a-z0-9+.-]*://|www\.)[^\s]+", "", value, flags=re.IGNORECASE
    ).strip()
    if re.search(
        r"\.(?:pdf|png|jpe?g|webp|tiff?|html?|docx?|xlsx?)(?:[?#].*)?$",
        text,
        re.IGNORECASE,
    ):
        return None
    words = list(_SOURCE_ANNEX_WORD.finditer(text))
    if not words:
        return None
    references = list(_SOURCE_ANNEX_REFERENCE.finditer(text))
    if len(words) != 1 or len(references) != 1:
        raise ValueError("source occurrence annex label missing or ambiguous")
    reference = references[0]
    remainder = text[reference.end() :].strip()
    if (
        remainder.startswith(("/", ".", "…"))
        or re.fullmatch(r"[-–—]", remainder)
        or _SOURCE_SHORTHAND.match(remainder.lstrip("(["))
    ):
        raise ValueError("source occurrence annex label shorthand is ambiguous")
    return normalize_annex_label(reference.group())


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
            "view": view.model_dump(
                mode="json",
                exclude={"sha256"}
                if view.source_occurrences
                else {"sha256", "source_occurrences"},
            ),
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
        source_scope = {
            _source_occurrence_label(value) for value in source_labels or []
        }
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
    if view.selection_method == "ordered_source_occurrences":
        try:
            validate_source_occurrence_order([view], view.source_occurrences)
        except ValueError:
            issues.append("source_occurrence_scope_mismatch")
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
    if store.read_file_record(original.file_id).file_type != original.mime_type:
        raise ValueError("original MIME differs from stored evidence")
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
            "byte_count": len(content),
            "side": side,
            "kind": "original",
            "locator": AnnexLocator().model_dump(mode="json"),
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
            locator = AnnexLocator(
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
            )
            file_id = store.save_file(
                BytesIO(image.png),
                "Frozen annex comparison",
                FileOrigin.OTHER,
                "image/png",
                file_metadata={
                    "annex_review_scope": scope.storage_identity(),
                    "sha256": digest,
                    "byte_count": len(image.png),
                    "side": original.side,
                    "kind": image.kind,
                    "locator": locator.model_dump(mode="json"),
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
                    locator=locator,
                )
            )
    return compared_pages, evidence


def combine_annex_evidence_views(
    views: list[AnnexExtraction],
    *,
    canonical_chunk_ids: list[str],
    source_occurrences: list[SourceLink] | None = None,
) -> AnnexExtraction:
    if not views or any(
        view.evidence_view is None or validate_evidence_view(view) for view in views
    ):
        raise ValueError("original scope missing or invalid")
    scopes = [view.evidence_view for view in views if view.evidence_view is not None]
    if source_occurrences is None and any(not scope.pages for scope in scopes):
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
    if source_occurrences is not None:
        validate_source_occurrence_order(scopes, source_occurrences)
    elif covered != canonical_chunk_ids or len(set(covered)) != len(covered):
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
        selection_method="ordered_source_occurrences"
        if source_occurrences is not None
        else "ordered_bound_originals",
        source_occurrences=source_occurrences or [],
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


def validate_compared_evidence(
    *,
    old: AnnexExtraction,
    new: AnnexExtraction,
    old_originals: list[AnnexOriginalEvidence],
    new_originals: list[AnnexOriginalEvidence],
    comparison: AnnexComparison,
    evidence: list[AnnexReviewEvidence],
) -> None:
    """Bind every compared parent and image to authorized frozen review bytes."""
    expected_images: list[tuple[object, ...]] = []
    actual_images: list[tuple[object, ...]] = []
    for side, extraction, originals in (
        ("old", old, old_originals),
        ("new", new, new_originals),
    ):
        if extraction.evidence_view:
            parents = [
                (parent.file_id, parent.sha256, parent.mime_type)
                for parent in extraction.evidence_view.parents
            ]
        else:
            parents = [
                (original.file_id, original.sha256, original.mime_type)
                for original in originals
                if original.available
                and original.sha256 == extraction.source_sha256
                and original.mime_type == extraction.mime_type
            ]
            if len(parents) != 1:
                raise ValueError(
                    "compared whole original identity is missing or ambiguous"
                )
        authorized = {
            (original.file_id, original.sha256, original.mime_type)
            for original in originals
            if original.available
        }
        if not parents or not set(parents).issubset(authorized):
            raise ValueError("compared parent identity outside authorized originals")
        for parent_id, digest, mime in parents:
            frozen = [
                item
                for item in evidence
                if item.side == side
                and item.kind == "original"
                and item.parent_file_id == parent_id
            ]
            if len(frozen) != 1 or (
                frozen[0].parent_sha256,
                frozen[0].sha256,
                frozen[0].mime_type,
            ) != (digest, digest, mime):
                raise ValueError("compared original differs from frozen review bytes")
        manifest = [item for item in comparison.image_manifest if item.side == side]
        if extraction.mime_type.startswith(("image/", "application/pdf")):
            pages = selected_evidence_pages(extraction)
            if not pages or sorted(
                item.page for item in manifest if item.kind == "comparison_page"
            ) != sorted(pages):
                raise ValueError("compared page manifest is incomplete")
        elif manifest:
            raise ValueError("nonvisual original has comparison image manifest")
        for item in manifest:
            if extraction.evidence_view:
                mappings = [
                    page
                    for page in extraction.evidence_view.pages
                    if page.view_page == item.page
                ]
                if len(mappings) != 1:
                    raise ValueError("comparison image page outside selected view")
                mapping = mappings[0]
                parent_id, digest, _ = parents[mapping.parent_index]
                original_page = mapping.original_page
            else:
                parent_id, digest, _ = parents[0]
                original_page = item.page
            expected_images.append(
                (
                    side,
                    parent_id,
                    digest,
                    original_page,
                    item.kind,
                    item.normalized_box,
                    item.sha256,
                    item.byte_count,
                    "image/png",
                )
            )
        if extraction.evidence_view:
            for mapping in extraction.evidence_view.pages:
                if mapping.normalized_box != (0, 0, 1, 1) and not any(
                    item.page == mapping.view_page
                    and item.kind == "comparison_region"
                    and item.normalized_box == mapping.normalized_box
                    for item in manifest
                ):
                    raise ValueError(
                        "selected comparison region manifest is incomplete"
                    )
    for item in evidence:
        if item.kind != "original":
            actual_images.append(
                (
                    item.side,
                    item.parent_file_id,
                    item.parent_sha256,
                    item.locator.page,
                    item.kind,
                    item.locator.normalized_box,
                    item.sha256,
                    item.byte_count,
                    item.mime_type,
                )
            )
    if Counter(actual_images) != Counter(expected_images):
        raise ValueError(
            "frozen comparison images differ from exact comparison manifest"
        )


def build_new_evidence_remapping(
    *,
    new: AnnexExtraction,
    evidence: list[AnnexReviewEvidence],
    asset_ids: dict[str, UUID],
) -> "AnnexNewEvidenceRemapping":
    from uuid import NAMESPACE_URL, uuid5

    from onyx.regulatory.amendments.annexes.comparison import annex_snapshot_hash
    from onyx.regulatory.amendments.annexes.models import (
        AnnexNewElementEvidence,
        AnnexNewEvidenceRemapping,
    )

    digest = annex_snapshot_hash(new)
    view = new.evidence_view
    elements: list[AnnexNewElementEvidence] = []
    for position, element in enumerate(new.elements):
        locator = element.locator
        if view is not None:
            mappings = [
                item for item in view.element_mappings if item.view_position == position
            ]
            if len(mappings) != 1:
                raise ValueError("NEW evidence mapping position ambiguous")
            parent = view.parents[mappings[0].parent_index]
            parent_file_id, parent_sha256 = parent.file_id, parent.sha256
            locator = mappings[0].original_locator
        else:
            originals = [
                item
                for item in evidence
                if item.side == "new"
                and item.kind == "original"
                and item.parent_sha256 == new.source_sha256
            ]
            if len(originals) != 1:
                raise ValueError("NEW evidence mapping original ambiguous")
            parent_file_id, parent_sha256 = (
                originals[0].parent_file_id,
                originals[0].parent_sha256,
            )
        if parent_file_id not in asset_ids:
            raise ValueError("NEW evidence mapping asset outside package")
        selected = [
            item
            for item in evidence
            if item.side == "new"
            and item.parent_file_id == parent_file_id
            and item.parent_sha256 == parent_sha256
            and (item.kind == "original" or item.locator.page == locator.page)
        ]
        if not any(item.kind == "original" for item in selected):
            raise ValueError("NEW evidence mapping original missing")
        images = [
            item.file_id for item in selected if item.kind == "comparison_region"
        ] or [item.file_id for item in selected if item.kind == "comparison_page"]
        elements.append(
            AnnexNewElementEvidence(
                position=position,
                element_id=uuid5(NAMESPACE_URL, f"annex:{digest}:{position}"),
                source_asset_id=asset_ids[parent_file_id],
                parent_file_id=parent_file_id,
                evidence_ids=[item.id for item in selected],
                image_file_ids=images,
            )
        )
    return AnnexNewEvidenceRemapping(extraction_sha256=digest, elements=elements)


def validate_new_evidence_remapping(
    *,
    mapping: "AnnexNewEvidenceRemapping",
    new: AnnexExtraction,
    evidence: list[AnnexReviewEvidence],
    asset_ids: dict[str, UUID],
) -> None:
    if mapping != build_new_evidence_remapping(
        new=new, evidence=evidence, asset_ids=asset_ids
    ):
        raise ValueError("NEW evidence mapping differs from reviewed package/view")


# A draft's `original_source_text_version` pins which join algorithm produced
# its frozen `original_source_text_sha256`, so upgrading this constant can
# never retroactively change a hash a live review is still checked against —
# see `AnnexChangeDraft.original_source_text_version`.
CURRENT_SOURCE_TEXT_VERSION: Literal[1, 2] = 2


def _annex_section_heading(
    asset: "RegulatorySourceAsset", links: list[SourceLink]
) -> str | None:
    """Mark `asset` as the resolved target of a discovered link, if it is one.

    Reuses `_source_occurrence_label` so a link whose own anchor text names
    its annex (e.g. "Ek-3'ü görüntülemek için tıklayınız") gets that exact
    label; a generic anchor (e.g. "Ekleri için tıklayınız") still gets a
    clear attachment boundary, just without a specific annex number — no
    number is invented. Only a cross-asset link counts as an attachment; an
    internal anchor that resolves back to the page it came from isn't a
    separate section.
    """
    link = next(
        (
            candidate
            for candidate in links
            if candidate.target_asset_hash == asset.sha256
            and candidate.target_asset_hash != candidate.parent_asset_hash
        ),
        None,
    )
    if link is None:
        return None
    label = link.label.strip() or "linked attachment"
    try:
        annex_label = _source_occurrence_label(link.label)
    except ValueError:
        annex_label = None
    heading = f"Annex {annex_label}" if annex_label else "Attachment"
    display_name = asset.display_name or "attachment"
    return f'## {heading} — resolved from "{label}": {display_name}'


def read_original_source_text(
    store: FileStore,
    assets: list["RegulatorySourceAsset"],
    *,
    links: list[SourceLink] | None = None,
    version: Literal[1, 2] = CURRENT_SOURCE_TEXT_VERSION,
) -> tuple[str, str]:
    """Assemble one immutable source's per-asset text into one reviewable string.

    Version 1 is a bare join in asset order — the original behavior, kept so
    a draft frozen before version 2 shipped keeps verifying against its
    original hash forever. Version 2 threads the already-discovered link
    graph through so a resolved attachment (e.g. a page whose only reference
    to it was an "Ekleri için tıklayınız" anchor) is marked as its own
    section instead of silently concatenated after the anchor's own literal
    text, for both the segmenter/drafter's input and the admin's pre-analysis
    review text.
    """
    headings_by_hash = (
        {}
        if version == 1 or links is None
        else {
            asset.sha256: heading
            for asset in assets
            if (heading := _annex_section_heading(asset, links)) is not None
        }
    )
    parts: list[str] = []
    for asset in assets:
        if asset.text_file_id is None:
            continue
        with store.read_file(asset.text_file_id) as stream:
            content = stream.read(2_000_000 + 1)
        if (
            len(content) > 2_000_000
            or hashlib.sha256(content).hexdigest() != asset.text_sha256
        ):
            raise ValueError("original extracted source text integrity changed")
        asset_text = content.decode("utf-8")
        heading = headings_by_hash.get(asset.sha256)
        parts.append(f"{heading}\n\n{asset_text}" if heading else asset_text)
    text = "\n\n".join(parts)
    if len(text) > 2_000_000:
        raise ValueError("original extracted source text limit")
    return text, hashlib.sha256(text.encode()).hexdigest()


def read_source_graph(
    store: FileStore,
    *,
    manifest_file_id: str,
    manifest_sha256: str,
    assets: list[RegulatorySourceAsset],
) -> list[SourceLink]:
    with store.read_file(manifest_file_id) as stream:
        content = stream.read(150 * 1024 * 1024 + 1)
    if hashlib.sha256(content).hexdigest() != manifest_sha256:
        raise ValueError("source_manifest_integrity_mismatch")
    manifest = json.loads(content)
    expected_assets = {(asset.sha256, asset.mime_type) for asset in assets}
    actual_assets = {
        (asset["sha256"], asset["mime_type"]) for asset in manifest.get("assets", [])
    }
    if (
        manifest.get("status") != "ready"
        or manifest.get("issues")
        or actual_assets != expected_assets
    ):
        raise ValueError("source_manifest_graph_incomplete")
    links = [SourceLink.model_validate(value) for value in manifest["links"]]
    hashes = {asset.sha256 for asset in assets}
    if any(
        link.parent_asset_hash not in hashes or link.target_asset_hash not in hashes
        for link in links
    ):
        raise ValueError("source_manifest_graph_incomplete")
    return links


def source_occurrence_key(link: SourceLink) -> tuple[int, str, int]:
    prefix, separator, ordinal = link.source_field.rpartition(":")
    if not separator or not ordinal.isdigit() or prefix.startswith("pdf:attachment"):
        raise ValueError("source occurrence has no native document order")
    return link.source_page or 0, prefix, int(ordinal)


def validate_source_occurrence_order(
    scopes: list[AnnexEvidenceView], links: list[SourceLink]
) -> None:
    parents = [parent for scope in scopes for parent in scope.parents]
    keys = [source_occurrence_key(link) for link in links]
    if (
        len(links) != len(parents)
        or not links
        or len({link.parent_asset_hash for link in links}) != 1
        or len({key[1] for key in keys}) != 1
        or keys != sorted(set(keys))
        or [link.target_asset_hash for link in links]
        != [parent.sha256 for parent in parents]
        or len({parent.sha256 for parent in parents}) != len(parents)
    ):
        raise ValueError("source occurrence order or complete parent identity mismatch")
    labels = {scope.label for scope in scopes}
    if any(_source_occurrence_label(link.label) not in labels for link in links):
        raise ValueError("source occurrence annex label mismatch")


def _visual_table_rows(
    view: AnnexExtraction,
) -> list[list[ExtractedAnnexElement]] | None:
    if not view.mime_type.startswith("image/"):
        return None
    atomic = [element for element in view.elements if not element.aggregate]
    if any(
        element.locator.page is None or element.locator.normalized_box is None
        for element in atomic
    ):
        return None
    cells = [element for element in atomic if element.kind == "table_cell"]
    rows: list[list[ExtractedAnnexElement]] = []
    # Group horizontal cell bands by their vertical centers, not rounded text values.
    for cell in sorted(
        cells,
        key=lambda element: (
            element.locator.page,
            cast(tuple[float, float, float, float], element.locator.normalized_box)[1],
            cast(tuple[float, float, float, float], element.locator.normalized_box)[0],
        ),
    ):
        box = cell.locator.normalized_box
        assert box is not None
        previous = rows[-1][0] if rows else None
        previous_box = previous.locator.normalized_box if previous else None
        if (
            previous is not None
            and previous_box is not None
            and previous.locator.page == cell.locator.page
            and previous_box[1] < (box[1] + box[3]) / 2 < previous_box[3]
        ):
            rows[-1].append(cell)
        else:
            rows.append([cell])
    if len(rows) < 2 or any(len(row) < 2 for row in rows):
        return None
    for row in rows:
        row.sort(
            key=lambda element: cast(
                tuple[float, float, float, float], element.locator.normalized_box
            )[0]
        )
        for left, right in zip(row, row[1:]):
            left_box, right_box = (
                left.locator.normalized_box,
                right.locator.normalized_box,
            )
            assert left_box is not None and right_box is not None
            if left_box[2] > right_box[0]:
                return None
    return rows


def _has_explicit_column_headers(rows: list[list[ExtractedAnnexElement]]) -> bool:
    """Only explicit visual heading evidence can exempt a repeated leading row."""
    header = rows[0]
    return all(len(row) == len(header) for row in rows[1:]) and all(
        cell.table_role == "column_header"
        and cell.extraction_method == "vision"
        and cell.status == "readable"
        and not cell.issues
        and bool(cell.text.strip())
        for cell in header
    )


def _source_parts_overlap(left: AnnexExtraction, right: AnnexExtraction) -> bool:
    def content(element: ExtractedAnnexElement) -> tuple[str, str, str | None]:
        return element.kind, element.text, element.formula

    atoms = [
        [
            element
            for element in view.elements
            if not element.aggregate
            and element.text.strip()
            and _boundary_label(element) is None
        ]
        for view in (left, right)
    ]
    shared = {content(element) for element in atoms[0]} & {
        content(element) for element in atoms[1]
    }
    if not shared:
        return False
    left_rows, right_rows = _visual_table_rows(left), _visual_table_rows(right)
    if left_rows is None or right_rows is None:
        return True
    signatures = [
        [tuple(content(element) for element in row) for row in rows]
        for rows in (left_rows, right_rows)
    ]
    if signatures[0] == signatures[1]:
        return True
    leading_header = _has_explicit_column_headers(
        left_rows
    ) and _has_explicit_column_headers(right_rows)
    if any(
        row == other and (index != 0 or other_index != 0 or not leading_header)
        for index, row in enumerate(signatures[0])
        for other_index, other in enumerate(signatures[1])
    ):
        return True
    # Unknown header roles retain strict duplicate-row refusal.
    # Other repeated text inside the table body still lacks disjointness proof.
    for elements, rows in zip(atoms, (left_rows, right_rows)):
        for element in elements:
            if content(element) not in shared or element.kind in {
                "table_cell",
                "image_region",
            }:
                continue
            box = element.locator.normalized_box
            assert box is not None
            table_boxes = [
                cell.locator.normalized_box
                for row in rows
                for cell in row
                if cell.locator.page == element.locator.page
            ]
            bounds = [value for value in table_boxes if value is not None]
            if (
                bounds
                and box[3] > min(value[1] for value in bounds)
                and box[1] < max(value[3] for value in bounds)
            ):
                return True
    return False


def select_new_annex_sources(
    selected: list[tuple[AnnexOriginalEvidence, AnnexExtraction]],
    links: list[SourceLink],
) -> tuple[list[AnnexOriginalEvidence], AnnexExtraction]:
    """Resolve disjoint NEW parts from native graph order, never OLD row bindings."""
    if not selected:
        raise ValueError("new_annex_evidence_missing")
    by_hash = {original.sha256: (original, view) for original, view in selected}
    if len(by_hash) != len(selected):
        raise ValueError("new_annex_evidence_duplicate")

    def native_content(
        view: AnnexExtraction,
    ) -> list[tuple[str, str, str | None]] | None:
        atomic = [
            element
            for element in view.elements
            if not element.aggregate and _boundary_label(element) is None
        ]
        if not atomic or any(
            element.extraction_method != "native" or element.kind == "image_region"
            for element in atomic
        ):
            return None
        return [(element.kind, element.text, element.formula) for element in atomic]

    label = selected[0][1].evidence_view.label if selected[0][1].evidence_view else ""
    labeled_links = [(link, _source_occurrence_label(link.label)) for link in links]
    covered_containers: set[str | None] = set()
    for parent, parent_label in labeled_links:
        if parent_label != label:
            continue
        children = [
            (link, child_label)
            for link, child_label in labeled_links
            if link.parent_asset_hash == parent.target_asset_hash
        ]
        if (
            children
            and all(link.source_field.startswith("html:") for link, _ in children)
            and any(link.target_asset_hash in by_hash for link, _ in children)
            and all(
                child_label not in {None, label} or link.target_asset_hash in by_hash
                for link, child_label in children
            )
        ):
            covered_containers.add(parent.target_asset_hash)
    if any(
        source_label == label
        and link.target_asset_hash not in by_hash
        and link.target_asset_hash not in covered_containers
        for link, source_label in labeled_links
    ):
        raise ValueError("new_annex_evidence_incomplete_linked_parts")
    contained: set[str | None] = set()
    for link in links:
        if (
            link.parent_asset_hash not in by_hash
            or link.target_asset_hash not in by_hash
            or link.target_asset_hash == link.parent_asset_hash
        ):
            continue
        parent_content = native_content(by_hash[link.parent_asset_hash][1])
        child_content = native_content(by_hash[link.target_asset_hash][1])
        if (
            parent_content is not None
            and child_content is not None
            and any(
                parent_content[start : start + len(child_content)] == child_content
                for start in range(len(parent_content) - len(child_content) + 1)
            )
        ):
            contained.add(link.target_asset_hash)
    selected = [pair for pair in selected if pair[0].sha256 not in contained]
    if len(selected) == 1:
        return [selected[0][0]], selected[0][1]
    hashes = {original.sha256 for original, _ in selected}
    occurrences = [
        link
        for link in links
        if link.target_asset_hash in hashes and link.parent_asset_hash not in hashes
    ]
    if len(occurrences) != len(selected):
        raise ValueError("new_annex_evidence_missing_or_overlapping_source_order")
    occurrences.sort(key=source_occurrence_key)
    ordered = [by_hash[link.target_asset_hash] for link in occurrences]
    for index, (_, view) in enumerate(ordered):
        if any(
            _source_parts_overlap(previous, view) for _, previous in ordered[:index]
        ):
            raise ValueError("new_annex_evidence_overlapping_parts")
    combined = combine_annex_evidence_views(
        [view for _, view in ordered],
        canonical_chunk_ids=[],
        source_occurrences=occurrences,
    )
    return [original for original, _ in ordered], combined


def visual_element_positions(extraction: AnnexExtraction) -> list[int]:
    if extraction.evidence_view is None:
        return (
            list(range(len(extraction.elements)))
            if extraction.mime_type.startswith(("image/", "application/pdf"))
            else []
        )
    view = extraction.evidence_view
    return [
        mapping.view_position
        for mapping in view.element_mappings
        if view.parents[mapping.parent_index].mime_type.startswith(
            ("image/", "application/pdf")
        )
    ]


def has_visual_original(extraction: AnnexExtraction) -> bool:
    return (
        any(
            parent.mime_type.startswith(("image/", "application/pdf"))
            for parent in extraction.evidence_view.parents
        )
        if extraction.evidence_view
        else extraction.mime_type.startswith(("image/", "application/pdf"))
    )
