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
from onyx.regulatory.amendments.annexes.evidence import (
    has_visual_original,
    validate_baseline_evidence_view,
)
from onyx.regulatory.amendments.annexes.models import (
    AnnexBaseline,
    AnnexCanonicalPatch,
    AnnexCanonicalSpan,
    AnnexComparison,
    AnnexExtraction,
    AnnexPatchPlan,
    ExtractedAnnexElement,
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


def _source_table_rows(view: AnnexExtraction) -> dict[int, list[int]]:
    """Require complete, aligned visual rows before binding native linewise cells."""
    pages: dict[tuple[int, str | None], list[int]] = {}
    for position, element in enumerate(view.elements):
        if element.aggregate or element.kind != "table_cell":
            continue
        box = element.locator.normalized_box
        if (
            element.locator.page is None
            or box is None
            or not (0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1)
        ):
            return {}
        pages.setdefault((element.locator.page, element.source_asset_id), []).append(
            position
        )
    result: dict[int, list[int]] = {}
    signatures: Counter[tuple[str, ...]] = Counter()
    all_rows: list[list[int]] = []
    for positions in pages.values():

        def box_at(position: int) -> tuple[float, float, float, float]:
            box = view.elements[position].locator.normalized_box
            assert box is not None
            return box

        rows: list[list[int]] = []
        for position in sorted(
            positions, key=lambda item: (box_at(item)[1], box_at(item)[0])
        ):
            box = box_at(position)
            previous = box_at(rows[-1][0]) if rows else None
            if (
                previous is not None
                and previous[1] < (box[1] + box[3]) / 2 < previous[3]
            ):
                rows[-1].append(position)
            else:
                rows.append([position])
        if len(rows) < 2:
            return {}
        for row in rows:
            row.sort(key=lambda item: box_at(item)[0])
        if any(
            max(box_at(item)[3] for item in above)
            > min(box_at(item)[1] for item in below) + 1e-9
            for above, below in zip(rows, rows[1:])
        ):
            return {}
        columns = [(box_at(item)[0], box_at(item)[2]) for item in rows[0]]
        if len(columns) < 2 or any(
            len(row) != len(columns)
            or any(
                abs(box_at(item)[0] - columns[column][0]) > 0.01
                or abs(box_at(item)[2] - columns[column][1]) > 0.01
                or abs(box_at(item)[1] - box_at(row[0])[1]) > 0.01
                or abs(box_at(item)[3] - box_at(row[0])[3]) > 0.01
                for column, item in enumerate(row)
            )
            or any(
                box_at(left)[2] > box_at(right)[0] + 1e-9
                for left, right in zip(row, row[1:])
            )
            for row in rows
        ):
            return {}
        for row in rows:
            signature = tuple(view.elements[item].text for item in row)
            if any(
                not text.strip() or "\n" in text or "|" in text for text in signature
            ):
                continue
            signatures[signature] += 1
            all_rows.append(row)
    for row in all_rows:
        if signatures[tuple(view.elements[item].text for item in row)] == 1:
            for position in row:
                result[position] = row
    return result


def _anchored_cell_spans(
    text: str, position: int, row: list[int], old: AnnexExtraction
) -> list[tuple[int, int]]:
    """A complete source row must uniquely identify the cell in either text format."""
    expected = [old.elements[item].text for item in row]
    column = row.index(position)
    matches: list[tuple[int, int]] = []
    lines: list[tuple[str, int]] = []
    offset = 0
    for line in text.splitlines(keepends=True):
        lines.append((line.strip(), offset + len(line) - len(line.lstrip())))
        if "|" in line:
            cells: list[tuple[str, int]] = []
            cell_offset = offset
            for part in line.split("|"):
                cells.append(
                    (part.strip(), cell_offset + len(part) - len(part.lstrip()))
                )
                cell_offset += len(part) + 1
            if cells and not cells[0][0]:
                cells.pop(0)
            if cells and not cells[-1][0]:
                cells.pop()
            if [value for value, _ in cells] == expected:
                start = cells[column][1]
                matches.append((start, start + len(old.elements[position].text)))
        offset += len(line)
    for start in range(len(lines) - len(row) + 1):
        if [value for value, _ in lines[start : start + len(row)]] == expected:
            cell_start = lines[start + column][1]
            matches.append((cell_start, cell_start + len(old.elements[position].text)))
    return matches


def _aligned_text_kind_drift(
    old: AnnexExtraction,
    new: AnnexExtraction,
    position: int,
    changed_old: set[int],
    changed_new: set[int],
) -> list[int]:
    element = old.elements[position]
    if element.kind not in {"text", "footnote"} or not element.text:
        return []
    old_atomic = [item for item in old.elements if not item.aggregate]
    matches = [
        index
        for index, item in enumerate(new.elements)
        if not item.aggregate and item.text == element.text
    ]
    if sum(item.text == element.text for item in old_atomic) != 1 or len(matches) != 1:
        return []
    target = new.elements[matches[0]]
    if matches[0] in changed_new or {element.kind, target.kind} != {"text", "footnote"}:
        return []

    def aligned(left: ExtractedAnnexElement, right: ExtractedAnnexElement) -> bool:
        a, b = left.locator.normalized_box, right.locator.normalized_box
        return (
            left.locator.page is not None
            and right.locator.page is not None
            and a is not None
            and b is not None
            and all(
                0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1
                for box in (a, b)
            )
            and all(abs(x - y) <= 0.01 for x, y in zip(a, b))
        )

    if not aligned(element, target):
        return []
    anchors = 0
    for index, anchor in enumerate(old.elements):
        if (
            index == position
            or index in changed_old
            or anchor.aggregate
            or not anchor.text
        ):
            continue
        if (anchor.locator.page, anchor.source_asset_id) != (
            element.locator.page,
            element.source_asset_id,
        ):
            continue
        if sum(item.text == anchor.text for item in old_atomic) != 1:
            continue
        candidates = [
            (index, item)
            for index, item in enumerate(new.elements)
            if not item.aggregate and item.text == anchor.text
        ]
        if len(candidates) != 1:
            continue
        candidate_position, candidate = candidates[0]
        if (
            candidate_position not in changed_new
            and anchor.kind == candidate.kind
            and (candidate.locator.page, candidate.source_asset_id)
            == (target.locator.page, target.source_asset_id)
            and aligned(anchor, candidate)
        ):
            anchors += 1
    return matches if anchors >= 2 else []


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
    issues = [
        *validate_annex_comparison(comparison, old=old, new=new),
        *validate_baseline_evidence_view(baseline, old),
    ]
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
    if has_visual_original(old) and not baseline.visual_evidence_available:
        # OLD's own evidence chain must match what the baseline claims to
        # back it. NEW's visual evidence comes from the freshly acquired
        # amendment source, an independent chain baseline never speaks to —
        # a canonical-text-only OLD (baseline.visual_evidence_available is
        # False by design here, not by error) compared against a visual NEW
        # is the intended fallback, not an inconsistency to block.
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
    source_rows = _source_table_rows(old)
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
            elif element.kind == "table_cell" and element.text:
                spans = (
                    _anchored_cell_spans(
                        candidate.text, old_position, source_rows[old_position], old
                    )
                    if old_position in source_rows
                    else []
                )
                if len(spans) != 1:
                    continue
                start, end = spans[0]
            elif element.text and candidate.text.count(element.text) == 1:
                start = candidate.text.index(element.text)
                end = start + len(element.text)
                if (start and re.match(r"[\w.,%]", candidate.text[start - 1])) or (
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
        if not matching_positions:
            matching_positions = _aligned_text_kind_drift(
                old, new, old_position, changed_old, changed_new
            )
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
