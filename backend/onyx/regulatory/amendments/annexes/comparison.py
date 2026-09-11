"""Whole-annex comparison over immutable evidence, with scoped model references."""

import base64
import hashlib
from collections import Counter
from io import BytesIO

from pydantic import ValidationError

from onyx.llm.interfaces import LLM
from onyx.llm.models import ImageContentPart, ImageUrlDetail
from onyx.prompts.regulatory_annex_comparison import ANNEX_COMPARISON_PROMPT
from onyx.regulatory.amendments.annexes.evidence import (
    has_visual_original,
    identical_evidence_scope,
    selected_evidence_pages,
    validate_evidence_view,
    visual_element_positions,
)
from onyx.regulatory.amendments.annexes.models import (
    AnnexComparedImage,
    AnnexComparison,
    AnnexComparisonImage,
    AnnexComparisonProposal,
    AnnexComparisonResponse,
    AnnexCoverage,
    AnnexDifference,
    AnnexElementReference,
    AnnexExtraction,
    AnnexModelSnapshot,
    AnnexRenderedPage,
)
from onyx.regulatory.structured_llm import generate_structured
from onyx.tracing.flows import LLMFlow

MAX_COMPARISON_PROMPT_CHARS = 200_000
MAX_PAGE_IMAGES = 50
DETAIL_TILE_PIXELS = 1024


def annex_snapshot_hash(extraction: AnnexExtraction) -> str:
    return hashlib.sha256(extraction.model_dump_json().encode()).hexdigest()


def _reference(extraction: AnnexExtraction, position: int) -> AnnexElementReference:
    element = extraction.elements[position]
    return AnnexElementReference(
        position=position, text=element.text, locator=element.locator
    )


def _atomic_positions(
    extraction: AnnexExtraction, *, omit_scope_boundaries: bool = False
) -> list[int]:
    return [
        index
        for index, element in enumerate(extraction.elements)
        if not element.aggregate
        and not (
            omit_scope_boundaries
            and extraction.evidence_view is not None
            and extraction.evidence_view.selected_positions[index]
            in extraction.evidence_view.boundary_positions
            and element.extraction_method == "native"
        )
    ]


def _native_changes(
    old: AnnexExtraction, new: AnnexExtraction
) -> list[AnnexDifference]:
    multipart = any(
        extraction.evidence_view is not None
        and len(extraction.evidence_view.parents) > 1
        for extraction in (old, new)
    )
    old_positions, new_positions = (
        _atomic_positions(old, omit_scope_boundaries=multipart),
        _atomic_positions(new, omit_scope_boundaries=multipart),
    )

    # Unique structural keys anchor changed cells; exact content can anchor moves.
    def keys(
        extraction: AnnexExtraction, positions: list[int]
    ) -> dict[int, tuple[str, str, str]]:
        return {
            index: (
                element.kind,
                "semantic" if element.semantic_key else "text",
                element.semantic_key or element.text,
            )
            for index in positions
            for element in [extraction.elements[index]]
        }

    old_keys, new_keys = keys(old, old_positions), keys(new, new_positions)
    old_counts, new_counts = Counter(old_keys.values()), Counter(new_keys.values())
    if any(count > 1 for count in [*old_counts.values(), *new_counts.values()]):
        raise ValueError("ambiguous_native_correspondence")
    same_order = list(old_keys.values()) == list(new_keys.values())
    by_key = {key: index for index, key in new_keys.items()}
    changes: list[AnnexDifference] = []
    matched: set[int] = set()
    for index, key in old_keys.items():
        target = by_key.get(key)
        before = _reference(old, index)
        if target is None:
            changes.append(
                AnnexDifference(
                    operation="remove",
                    old=[before],
                    explanation="Element absent from complete native structure",
                )
            )
            continue
        matched.add(target)
        if old.elements[index].formula != new.elements[target].formula:
            raise ValueError("formula_change_requires_review")
        after = _reference(new, target)
        if before.text != after.text:
            changes.append(
                AnnexDifference(
                    operation="replace",
                    old=[before],
                    new=[after],
                    explanation="Exact anchored value changed",
                )
            )
        elif not (
            same_order
            and (
                before.locator.sheet,
                before.locator.cell,
                before.locator.row,
                before.locator.column,
            )
            == (
                after.locator.sheet,
                after.locator.cell,
                after.locator.row,
                after.locator.column,
            )
        ) and (
            before.locator != after.locator
            or old_positions.index(index) != new_positions.index(target)
        ):
            changes.append(
                AnnexDifference(
                    operation="move",
                    old=[before],
                    new=[after],
                    explanation="Anchored element moved",
                )
            )
    changes.extend(
        AnnexDifference(
            operation="insert",
            new=[_reference(new, index)],
            explanation="Element added to complete native structure",
        )
        for index in new_positions
        if index not in matched
    )
    return changes


def comparison_page_evidence(
    page: AnnexRenderedPage,
    *,
    normalized_box: tuple[float, float, float, float] = (0, 0, 1, 1),
) -> list[AnnexComparisonImage]:
    """The exact compared bytes and original-page coordinates, also frozen for review."""
    from PIL import Image

    images = [
        AnnexComparisonImage(
            png=page.png, kind="comparison_page", normalized_box=(0, 0, 1, 1)
        )
    ]
    with Image.open(BytesIO(page.png)) as image:
        width, height = image.size
        if width > DETAIL_TILE_PIXELS or height > DETAIL_TILE_PIXELS:
            for top in range(0, height, DETAIL_TILE_PIXELS):
                for left in range(0, width, DETAIL_TILE_PIXELS):
                    right, bottom = (
                        min(left + DETAIL_TILE_PIXELS, width),
                        min(top + DETAIL_TILE_PIXELS, height),
                    )
                    buffer = BytesIO()
                    image.crop((left, top, right, bottom)).save(buffer, format="PNG")
                    images.append(
                        AnnexComparisonImage(
                            png=buffer.getvalue(),
                            kind="comparison_tile",
                            normalized_box=(
                                left / width,
                                top / height,
                                right / width,
                                bottom / height,
                            ),
                        )
                    )
        if normalized_box != (0, 0, 1, 1):
            left, top, right, bottom = normalized_box
            if not (0 <= left < right <= 1 and 0 <= top < bottom <= 1):
                raise ValueError("invalid_selection_region")
            buffer = BytesIO()
            image.crop(
                (
                    int(left * width),
                    int(top * height),
                    int(right * width),
                    int(bottom * height),
                )
            ).save(buffer, format="PNG")
            images.append(
                AnnexComparisonImage(
                    png=buffer.getvalue(),
                    kind="comparison_region",
                    normalized_box=normalized_box,
                )
            )
    if len(images) > MAX_PAGE_IMAGES // 2:
        raise ValueError("comparison_image_budget")
    return images


def _page_images(
    page: AnnexRenderedPage,
    *,
    normalized_box: tuple[float, float, float, float] = (0, 0, 1, 1),
) -> list[ImageContentPart]:
    return [
        ImageContentPart(
            image_url=ImageUrlDetail(
                url="data:image/png;base64," + base64.b64encode(image.png).decode(),
                detail="high",
            )
        )
        for image in comparison_page_evidence(page, normalized_box=normalized_box)
    ]


def _validate_response(
    response: AnnexComparisonResponse,
    *,
    old: AnnexExtraction,
    new: AnnexExtraction,
    old_positions: list[int],
    new_positions: list[int],
    old_pages: list[int],
    new_pages: list[int],
) -> list[str]:
    issues: list[str] = list(response.issues)
    for actual, expected in (
        (response.old_positions, old_positions),
        (response.new_positions, new_positions),
        (response.old_pages, old_pages),
        (response.new_pages, new_pages),
    ):
        if sorted(actual) != sorted(expected):
            issues.append("incomplete_comparison_coverage")
    used_old: set[int] = set()
    used_new: set[int] = set()
    for change in response.changes:
        if change.uncertain:
            issues.append("uncertain_difference")
        counts = (len(change.old), len(change.new))
        valid_shape = {
            "replace": counts == (1, 1),
            "move": counts == (1, 1),
            "visual": counts[0] > 0 and counts[1] > 0,
            "insert": counts[0] == 0 and counts[1] > 0,
            "remove": counts[0] > 0 and counts[1] == 0,
            "split": counts[0] == 1 and counts[1] > 1,
            "merge": counts[0] > 1 and counts[1] == 1,
        }[change.operation]
        if not valid_shape:
            issues.append("invalid_operation_shape")
        for references, extraction, allowed, used in (
            (change.old, old, old_positions, used_old),
            (change.new, new, new_positions, used_new),
        ):
            for reference in references:
                if reference.position not in allowed or reference != _reference(
                    extraction, reference.position
                ):
                    issues.append("invalid_snapshot_reference")
                elif reference.position in used:
                    issues.append("overlapping_change_reference")
                used.add(reference.position)
    return issues


def _scope_prompt(extraction: AnnexExtraction) -> str:
    view = extraction.evidence_view
    if view is None:
        return "whole original"
    return (
        f"label={view.label}; view_sha256={view.sha256}; selected page regions="
        + "; ".join(
            f"view_page={page.view_page}, parent_sha256={view.parents[page.parent_index].sha256}, original_page={page.original_page}, normalized_box={page.normalized_box}"
            for page in view.pages
        )
    )


def compare_annexes(
    *,
    old: AnnexExtraction,
    new: AnnexExtraction,
    old_pages: list[AnnexRenderedPage] | None = None,
    new_pages: list[AnnexRenderedPage] | None = None,
    llm: LLM | None = None,
    instruction: str = "",
) -> AnnexComparison:
    """Compare complete extractions; callers must supply originals for visual files.

    The returned references are extraction-local positions, not write authority.
    Patch preparation separately verifies canonical baseline, dates and mappings.
    """
    before_pages, after_pages = old_pages or [], new_pages or []
    multipart = any(
        extraction.evidence_view is not None
        and len(extraction.evidence_view.parents) > 1
        for extraction in (old, new)
    )
    old_positions, new_positions = (
        _atomic_positions(old, omit_scope_boundaries=multipart),
        _atomic_positions(new, omit_scope_boundaries=multipart),
    )
    issues = [
        *old.issues,
        *new.issues,
        *validate_evidence_view(old),
        *validate_evidence_view(new),
    ]
    if any(
        element.status != "readable" or element.issues
        for extraction in (old, new)
        for element in extraction.elements
    ):
        issues.append("incomplete_extraction")
    if not old_positions or not new_positions:
        issues.append("empty_extraction")
    visual = any(has_visual_original(extraction) for extraction in (old, new))
    method = (
        "identical_asset"
        if identical_evidence_scope(old, new)
        else "simultaneous_vision"
        if visual
        else "native_structure"
    )
    coverage = AnnexCoverage(
        old_positions=old_positions,
        new_positions=new_positions,
        old_pages=[page.page for page in before_pages],
        new_pages=[page.page for page in after_pages],
        method=method,
    )
    image_manifest: list[AnnexComparedImage] = []
    for side, extraction, pages in (
        ("old", old, before_pages),
        ("new", new, after_pages),
    ):
        for page in pages:
            region = (
                next(
                    (
                        mapping.normalized_box
                        for mapping in extraction.evidence_view.pages
                        if mapping.view_page == page.page
                    ),
                    (0, 0, 1, 1),
                )
                if extraction.evidence_view
                else (0, 0, 1, 1)
            )
            try:
                for image in comparison_page_evidence(page, normalized_box=region):
                    image_manifest.append(
                        AnnexComparedImage(
                            side="old" if side == "old" else "new",
                            page=page.page,
                            kind=image.kind,
                            normalized_box=image.normalized_box,
                            sha256=hashlib.sha256(image.png).hexdigest(),
                            byte_count=len(image.png),
                        )
                    )
            except ValueError as error:
                issues.append(str(error))
    changes: list[AnnexDifference] = []
    model_snapshot = None
    if method == "native_structure" and not issues:
        try:
            changes = _native_changes(old, new)
        except ValueError as error:
            issues.append(str(error))
    elif method == "simultaneous_vision":
        for extraction, pages in ((old, before_pages), (new, after_pages)):
            if (
                has_visual_original(extraction)
                and extraction.evidence_view is None
                and extraction.page_count is None
            ):
                issues.append("page_count_unverified")
            elif [page.page for page in pages] != selected_evidence_pages(extraction):
                issues.append("page_coverage_mismatch")
            if has_visual_original(extraction) and not pages:
                issues.append("visual_evidence_unavailable")
            if not {
                extraction.elements[position].locator.page
                for position in visual_element_positions(extraction)
            }.issubset({page.page for page in pages}):
                issues.append("extraction_page_unmapped")
        if llm is None:
            issues.append("vision_model_unavailable")
        if not issues and llm is not None:
            model_snapshot = AnnexModelSnapshot(
                model_provider=llm.config.model_provider,
                model_name=llm.config.model_name,
            )
            images: list[ImageContentPart] = []
            manifest: list[str] = []
            for side, pages in (("OLD", before_pages), ("NEW", after_pages)):
                for page in pages:
                    try:
                        extraction = old if side == "OLD" else new
                        region = (
                            next(
                                (
                                    mapping.normalized_box
                                    for mapping in extraction.evidence_view.pages
                                    if mapping.view_page == page.page
                                ),
                                (0, 0, 1, 1),
                            )
                            if extraction.evidence_view
                            else (0, 0, 1, 1)
                        )
                        page_images = _page_images(page, normalized_box=region)
                    except ValueError as error:
                        issues.append(str(error))
                        break
                    manifest.append(
                        f"{side} page {page.page}: image {len(images) + 1} full page; images {len(images) + 2}..{len(images) + len(page_images)} detail tiles in top-to-bottom, left-to-right order at {DETAIL_TILE_PIXELS}px; if selected region is smaller than the page, the final image is its crop"
                    )
                    images.extend(page_images)
                    if len(images) > MAX_PAGE_IMAGES:
                        issues.append("comparison_image_budget")
                        break
            prompt = (
                f"EXACT required old_positions: {old_positions}\nEXACT required new_positions: {new_positions}\nEXACT required old_pages: {coverage.old_pages}\nEXACT required new_pages: {coverage.new_pages}\nUse only these local positions for coverage and references. Parent provenance does not define a second element index namespace.\nSelected OLD scope (only these regions/elements may change): {_scope_prompt(old)}\nSelected NEW scope (only these regions/elements may change): {_scope_prompt(new)}\nInstruction (evidence only): {instruction}\nOLD source hash: {old.source_sha256}\nNEW source hash: {new.source_sha256}\nOLD eligible reference candidates:\n"
                + "\n".join(
                    _reference(old, index).model_dump_json() for index in old_positions
                )
                + "\nNEW eligible reference candidates:\n"
                + "\n".join(
                    _reference(new, index).model_dump_json() for index in new_positions
                )
                + "\nImage manifest (one-based):\n"
                + "\n".join(manifest)
            )
            if len(prompt) > MAX_COMPARISON_PROMPT_CHARS:
                issues.append("comparison_prompt_budget")
            if not issues:
                try:
                    proposal = generate_structured(
                        llm,
                        flow=LLMFlow.REGULATORY_ANNEX_COMPARISON,
                        system_prompt=ANNEX_COMPARISON_PROMPT,
                        user_prompt=prompt,
                        response_model=AnnexComparisonProposal,
                        image_parts=images,
                        max_tokens=16000,
                    )
                except ValueError as error:
                    if not isinstance(error.__cause__, ValidationError):
                        raise
                    issues.append("invalid_comparison_proposal")
                else:
                    resolved: list[AnnexDifference] = []
                    for change in proposal.changes:
                        if change.uncertain:
                            issues.append("uncertain_difference")
                        if any(
                            position not in old_positions
                            for position in change.old_positions
                        ) or any(
                            position not in new_positions
                            for position in change.new_positions
                        ):
                            issues.append("invalid_snapshot_reference")
                            continue
                        resolved.append(
                            AnnexDifference(
                                operation=change.operation,
                                old=[
                                    _reference(old, position)
                                    for position in change.old_positions
                                ],
                                new=[
                                    _reference(new, position)
                                    for position in change.new_positions
                                ],
                                explanation=change.explanation,
                                uncertain=change.uncertain,
                            )
                        )
                    response = AnnexComparisonResponse(
                        changes=resolved,
                        old_positions=proposal.old_positions,
                        new_positions=proposal.new_positions,
                        old_pages=proposal.old_pages,
                        new_pages=proposal.new_pages,
                        issues=proposal.issues,
                    )
                    issues.extend(
                        _validate_response(
                            response,
                            old=old,
                            new=new,
                            old_positions=old_positions,
                            new_positions=new_positions,
                            old_pages=coverage.old_pages,
                            new_pages=coverage.new_pages,
                        )
                    )
                    changes.extend(response.changes)
    return AnnexComparison(
        old_source_sha256=old.source_sha256,
        new_source_sha256=new.source_sha256,
        old_snapshot_sha256=annex_snapshot_hash(old),
        new_snapshot_sha256=annex_snapshot_hash(new),
        changes=changes,
        image_manifest=image_manifest,
        coverage=coverage,
        model_snapshot=model_snapshot,
        prompt_version="annex-comparison-v3"
        if model_snapshot is not None
        else "annex-comparison-v2",
        issues=sorted(set(issues)),
        ready=not issues,
    )


def validate_annex_comparison(
    comparison: AnnexComparison, *, old: AnnexExtraction, new: AnnexExtraction
) -> list[str]:
    """Revalidate persisted/editable comparison data before deriving write targets."""
    issues = [
        *comparison.issues,
        *old.issues,
        *new.issues,
        *validate_evidence_view(old),
        *validate_evidence_view(new),
    ]
    if any(
        element.status != "readable" or element.issues
        for extraction in (old, new)
        for element in extraction.elements
    ):
        issues.append("incomplete_extraction")
    multipart = any(
        extraction.evidence_view is not None
        and len(extraction.evidence_view.parents) > 1
        for extraction in (old, new)
    )
    old_positions = _atomic_positions(old, omit_scope_boundaries=multipart)
    new_positions = _atomic_positions(new, omit_scope_boundaries=multipart)
    if not old_positions or not new_positions:
        issues.append("empty_extraction")
    visual = any(has_visual_original(extraction) for extraction in (old, new))
    expected_method = (
        "identical_asset"
        if identical_evidence_scope(old, new)
        else "simultaneous_vision"
        if visual
        else "native_structure"
    )
    if comparison.coverage.method != expected_method:
        issues.append("comparison_method_mismatch")
    if expected_method == "simultaneous_vision":
        for extraction, pages in (
            (old, comparison.coverage.old_pages),
            (new, comparison.coverage.new_pages),
        ):
            if (
                has_visual_original(extraction)
                and extraction.evidence_view is None
                and extraction.page_count is None
            ):
                issues.append("page_count_unverified")
            elif pages != selected_evidence_pages(extraction):
                issues.append("page_coverage_mismatch")
    if (
        comparison.old_snapshot_sha256 != annex_snapshot_hash(old)
        or comparison.new_snapshot_sha256 != annex_snapshot_hash(new)
        or comparison.old_source_sha256 != old.source_sha256
        or comparison.new_source_sha256 != new.source_sha256
    ):
        issues.append("comparison_snapshot_changed")
    try:
        response = AnnexComparisonResponse(
            changes=comparison.changes,
            old_positions=comparison.coverage.old_positions,
            new_positions=comparison.coverage.new_positions,
            old_pages=comparison.coverage.old_pages,
            new_pages=comparison.coverage.new_pages,
        )
    except ValueError:
        return [*issues, "overlapping_change_reference"]
    issues.extend(
        _validate_response(
            response,
            old=old,
            new=new,
            old_positions=old_positions,
            new_positions=new_positions,
            old_pages=comparison.coverage.old_pages,
            new_pages=comparison.coverage.new_pages,
        )
    )
    return sorted(set(issues))
