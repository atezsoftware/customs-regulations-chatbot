"""Guarded annex routing and canonical patches; neither publishes nor checkpoints."""

import hashlib
import re
from collections import Counter
from datetime import date

from onyx.db.regulatory_annexes import normalize_annex_label
from onyx.regulatory.amendments.annexes.comparison import (
    annex_snapshot_hash,
    validate_annex_comparison,
)
from onyx.regulatory.amendments.annexes.models import (
    AnnexBaseline,
    AnnexCanonicalPatch,
    AnnexCanonicalSpan,
    AnnexComparison,
    AnnexExtraction,
    AnnexPatchPlan,
)

_ANNEX_REFERENCE = re.compile(
    r"\b(?:EK|annex|appendix)[\s:–—-]*(?:\d+[a-z]?|[ivxlcdm]+[a-z]?|[a-z])(?:[/.-][0-9a-z]+)*(?![\w/.-])",
    re.IGNORECASE,
)
_MARKDOWN_DELIMITER = re.compile(r"\s*\|\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)*\|\s*")
_ANNEX_WORD = re.compile(r"\b(?:EK|annex|appendix)\b", re.IGNORECASE)


def resolve_annex_instruction(
    instruction: str, *, source_labels: list[str], file_labels: list[str]
) -> str | None:
    """Require the full source-backed label before loading the complete file scope.

    `None` retains the legacy textual route. An unresolved annex raises; it must
    not fall through to the legacy regexp, which truncates suffixes/Roman labels.
    Labels must come from acquired evidence and relational file/heading inventory.
    """
    if not _ANNEX_WORD.search(instruction):
        return None
    intended = {
        normalize_annex_label(match.group())
        for match in _ANNEX_REFERENCE.finditer(instruction)
    }
    sources = {normalize_annex_label(label): label for label in source_labels}
    files = {normalize_annex_label(label) for label in file_labels}
    if len(intended) != 1 or not intended.issubset(sources.keys() & files):
        raise ValueError("annex instruction scope is unresolved or ambiguous")
    return sources[next(iter(intended))]


def prepare_annex_patch(
    *,
    baseline: AnnexBaseline,
    old: AnnexExtraction,
    new: AnnexExtraction,
    comparison: AnnexComparison,
    effective_date: date | None,
    package_complete: bool,
) -> AnnexPatchPlan:
    """Resolve extraction references against approved text; freeze a reviewable patch.

    Exact unique structure/content is required for automatic canonical mapping.
    Ambiguous mappings retain comparison evidence but never become approval-ready.
    Raw OCR cannot override an approved correction as the OLD legal value.
    """
    issues = validate_annex_comparison(comparison, old=old, new=new)
    if not comparison.ready:
        issues.append("comparison_not_ready")
    if not package_complete:
        issues.append("incomplete_source_package")
    if effective_date is None:
        issues.append("effective_date_unresolved")
    if comparison.old_snapshot_sha256 != annex_snapshot_hash(
        old
    ) or comparison.new_snapshot_sha256 != annex_snapshot_hash(new):
        issues.append("comparison_snapshot_changed")
    if (
        any(
            item.mime_type.startswith(("image/", "application/pdf"))
            for item in (old, new)
        )
        and not baseline.visual_evidence_available
    ):
        issues.append("original_visual_evidence_unavailable")
    if issues:
        return AnnexPatchPlan(
            baseline_sha256=baseline.baseline_sha256,
            comparison_sha256=hashlib.sha256(
                comparison.model_dump_json().encode()
            ).hexdigest(),
            effective_date=effective_date,
            patches=[],
            direct_canonical_changes=[],
            metadata_only=[],
            retire_history=[],
            unchanged=[],
            issues=sorted(set(issues)),
            ready=False,
        )
    canonical = [element for element in baseline.elements if element.canonical_chunk_id]
    canonical_ids = {
        element.canonical_chunk_id
        for element in canonical
        if element.canonical_chunk_id
    }
    independent_visual = (
        bool(comparison.changes)
        and not baseline.canonical_amendment_chunk_ids
        and Counter(
            element.text
            for element in old.elements
            if not element.aggregate and element.kind != "image_region"
        )
        == Counter(
            element.text
            for element in new.elements
            if not element.aggregate and element.kind != "image_region"
        )
        and all(
            change.operation in ("visual", "replace")
            and all(
                old.elements[reference.position].kind == "image_region"
                for reference in change.old
            )
            and all(
                new.elements[reference.position].kind == "image_region"
                for reference in change.new
            )
            for change in comparison.changes
        )
    )
    visual_bound_chunks: set[str] = set()
    if independent_visual:
        verified_companions = [
            element
            for element in canonical
            if element.canonical_role == "supporting"
            and element.bound_to_regulatory_chunk_id in canonical_ids
            and any(
                original.available
                and original.file_id == element.image_file_id
                and original.sha256 == old.source_sha256
                and element.canonical_chunk_id in original.canonical_chunk_ids
                for original in baseline.originals
            )
        ]
        if len(verified_companions) == 1:
            binding = verified_companions[0].bound_to_regulatory_chunk_id
            assert binding is not None
            visual_bound_chunks.add(binding)
    bindings: dict[int, AnnexCanonicalSpan] = {}
    canonical_by_id = {element.canonical_chunk_id: element for element in canonical}
    for old_position, element in enumerate(old.elements):
        if element.aggregate:
            continue
        matches: list[AnnexCanonicalSpan] = []
        for candidate in canonical:
            if candidate.canonical_role == "supporting":
                continue
            chunk_id = candidate.canonical_chunk_id
            assert chunk_id is not None
            if (
                element.canonical_chunk_id is not None
                and element.canonical_chunk_id != chunk_id
            ):
                continue
            if (
                element.semantic_key is not None
                and element.semantic_key == candidate.semantic_key
            ):
                start, end = 0, len(candidate.text)
            elif element.text and candidate.text.count(element.text) == 1:
                start = candidate.text.index(element.text)
                end = start + len(element.text)
                if element.kind == "table_cell":
                    cell_start = candidate.text.rfind("|", 0, start) + 1
                    cell_end = candidate.text.find("|", end)
                    if cell_end < 0:
                        cell_end = len(candidate.text)
                    if candidate.text[cell_start:cell_end].strip() != element.text:
                        continue
                elif (start and re.match(r"[\w.,%]", candidate.text[start - 1])) or (
                    end < len(candidate.text)
                    and re.match(r"[\w.,%]", candidate.text[end])
                ):
                    continue
            else:
                continue
            matches.append(
                AnnexCanonicalSpan(
                    canonical_chunk_id=chunk_id,
                    old_position=old_position,
                    start=start,
                    end=end,
                    old_text=candidate.text[start:end],
                )
            )
        if len(matches) == 1:
            bindings[old_position] = matches[0]
    # Reconcile correspondence across the whole annex, including hash-skipped raw
    # comparisons: a reused old asset can still reverse an approved text overlay.
    correspondence: dict[int, list[int]] = {}
    changed_old: set[int] = set()
    changed_new: set[int] = set()
    operations: dict[int, str] = {}
    for change in comparison.changes:
        old_positions = [reference.position for reference in change.old]
        new_positions = [reference.position for reference in change.new]
        changed_old.update(old_positions)
        changed_new.update(new_positions)
        for ordinal, position in enumerate(old_positions):
            correspondence[position] = (
                [] if change.operation == "merge" and ordinal else new_positions
            )
            operations[position] = change.operation
    for old_position, element in enumerate(old.elements):
        if element.aggregate or old_position in changed_old:
            continue
        matching_positions = [
            index
            for index, candidate in enumerate(new.elements)
            if not candidate.aggregate
            and index not in changed_new
            and (
                (
                    element.semantic_key is not None
                    and element.semantic_key == candidate.semantic_key
                )
                or (element.text == candidate.text and element.kind == candidate.kind)
            )
        ]
        if len(matching_positions) == 1:
            correspondence[old_position] = matching_positions
        elif old_position in bindings:
            issues.append("canonical_correspondence_unresolved")
    patches: list[AnnexCanonicalPatch] = []
    used_chunks: set[str] = set(visual_bound_chunks)
    grouped: dict[str, list[AnnexCanonicalSpan]] = {}
    for old_position in correspondence:
        binding = bindings.get(old_position)
        if binding is None:
            if old_position in changed_old and not visual_bound_chunks:
                issues.append("canonical_correspondence_unresolved")
            continue
        grouped.setdefault(binding.canonical_chunk_id, []).append(binding)
    for chunk_id, spans in grouped.items():
        current = canonical_by_id[chunk_id]
        used_chunks.add(chunk_id)
        ordered_spans = sorted(spans, key=lambda span: span.start)
        if any(
            left.end > right.start
            for left, right in zip(ordered_spans, ordered_spans[1:])
        ):
            issues.append("overlapping_canonical_spans")
            continue
        replacement = current.text
        new_positions: list[int] = []
        for span in reversed(ordered_spans):
            positions = correspondence[span.old_position]
            new_positions.extend(positions)
            text = "\n".join(new.elements[position].text for position in positions)
            replacement = replacement[: span.start] + text + replacement[span.end :]
        new_text = replacement or None
        operation = "remove" if new_text is None else "replace"
        if new_text == current.text:
            if any(operations.get(span.old_position) == "visual" for span in spans):
                operation = "visual"
            elif any(operations.get(span.old_position) == "move" for span in spans):
                operation = "move"
            else:
                continue
        patches.append(
            AnnexCanonicalPatch(
                old_chunk_id=chunk_id,
                old_text=current.text,
                new_text=new_text,
                old_positions=[span.old_position for span in ordered_spans],
                new_positions=sorted(set(new_positions)),
                validated_spans=ordered_spans,
                operation=operation,
            )
        )
    for chunk_id in visual_bound_chunks:
        if chunk_id not in grouped:
            current = canonical_by_id[chunk_id]
            patches.append(
                AnnexCanonicalPatch(
                    old_chunk_id=chunk_id,
                    old_text=current.text,
                    new_text=current.text,
                    old_positions=[],
                    new_positions=[],
                    operation="visual",
                )
            )
    insertions: dict[tuple[object, ...], list[int]] = {}
    for change in comparison.changes:
        if change.operation != "insert":
            continue
        for reference in change.new:
            element = new.elements[reference.position]
            locator = element.locator
            key = (
                (locator.page, locator.sheet, locator.path, locator.row)
                if element.kind == "table_cell" and locator.row is not None
                else (reference.position,)
            )
            insertions.setdefault(key, []).append(reference.position)
    for positions in insertions.values():
        separator = (
            " | "
            if all(
                new.elements[position].kind == "table_cell" for position in positions
            )
            else "\n"
        )
        patches.append(
            AnnexCanonicalPatch(
                old_chunk_id=None,
                old_text=None,
                new_text=separator.join(
                    new.elements[position].text for position in sorted(positions)
                ),
                old_positions=[],
                new_positions=sorted(positions),
                operation="insert",
            )
        )
    for element in canonical:
        chunk_id = element.canonical_chunk_id
        assert chunk_id is not None
        if element.canonical_role == "supporting":
            originals = [
                original
                for original in baseline.originals
                if original.file_id == element.image_file_id
                and original.sha256 == old.source_sha256
                and original.available
            ]
            if old.source_sha256 == new.source_sha256:
                used_chunks.add(chunk_id)
            elif len(originals) == 1:
                text = "\n".join(
                    item.text
                    for item in new.elements
                    if not item.aggregate and item.text
                )
                if not text:
                    issues.append("supporting_caption_disposition_required")
                else:
                    used_chunks.add(chunk_id)
                    if text != element.text:
                        patches.append(
                            AnnexCanonicalPatch(
                                canonical_role="supporting",
                                old_chunk_id=chunk_id,
                                old_text=element.text,
                                new_text=text,
                                old_positions=[],
                                new_positions=[
                                    index
                                    for index, item in enumerate(new.elements)
                                    if not item.aggregate
                                ],
                                operation="replace",
                            )
                        )
            else:
                issues.append("supporting_caption_disposition_required")
        elif chunk_id not in used_chunks:
            # A Markdown delimiter has no legal values or visual transcription.
            # Keep its canonical identity while changed content still needs mapping.
            if _MARKDOWN_DELIMITER.fullmatch(element.text):
                used_chunks.add(chunk_id)
            else:
                issues.append("canonical_correspondence_unresolved")
    direct = sorted(
        {
            patch.old_chunk_id
            for patch in patches
            if patch.old_chunk_id and patch.old_text != patch.new_text
        }
    )
    metadata = sorted(
        {
            patch.old_chunk_id
            for patch in patches
            if patch.old_chunk_id and patch.old_text == patch.new_text
        }
    )
    return AnnexPatchPlan(
        baseline_sha256=baseline.baseline_sha256,
        comparison_sha256=hashlib.sha256(
            comparison.model_dump_json().encode()
        ).hexdigest(),
        effective_date=effective_date,
        patches=patches,
        direct_canonical_changes=direct,
        metadata_only=metadata,
        retire_history=direct,
        unchanged=sorted(canonical_ids - set(direct) - set(metadata)),
        issues=sorted(set(issues)),
        ready=not issues,
    )
