"""Whole-annex comparison over immutable evidence, with scoped model references."""

import base64
import hashlib
from collections import Counter
from io import BytesIO

from onyx.llm.interfaces import LLM
from onyx.llm.models import ImageContentPart, ImageUrlDetail
from onyx.prompts.regulatory_annex_comparison import ANNEX_COMPARISON_PROMPT
from onyx.regulatory.amendments.annexes.models import (
    AnnexComparison,
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


def _atomic_positions(extraction: AnnexExtraction) -> list[int]:
    return [
        index
        for index, element in enumerate(extraction.elements)
        if not element.aggregate
    ]


def _native_changes(
    old: AnnexExtraction, new: AnnexExtraction
) -> list[AnnexDifference]:
    old_positions, new_positions = _atomic_positions(old), _atomic_positions(new)

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
        elif before.locator != after.locator:
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


def _page_images(page: AnnexRenderedPage) -> list[ImageContentPart]:
    from PIL import Image

    images: list[bytes] = [page.png]
    with Image.open(BytesIO(page.png)) as image:
        width, height = image.size
        if width > DETAIL_TILE_PIXELS or height > DETAIL_TILE_PIXELS:
            for top in range(0, height, DETAIL_TILE_PIXELS):
                for left in range(0, width, DETAIL_TILE_PIXELS):
                    buffer = BytesIO()
                    image.crop(
                        (
                            left,
                            top,
                            min(left + DETAIL_TILE_PIXELS, width),
                            min(top + DETAIL_TILE_PIXELS, height),
                        )
                    ).save(buffer, format="PNG")
                    images.append(buffer.getvalue())
                    if len(images) > MAX_PAGE_IMAGES // 2:
                        raise ValueError("comparison_image_budget")
    return [
        ImageContentPart(
            image_url=ImageUrlDetail(
                url="data:image/png;base64," + base64.b64encode(content).decode(),
                detail="high",
            )
        )
        for content in images
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
    issues = list(response.issues)
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
    old_positions, new_positions = _atomic_positions(old), _atomic_positions(new)
    issues = [*old.issues, *new.issues]
    if any(
        element.status != "readable" or element.issues
        for extraction in (old, new)
        for element in extraction.elements
    ):
        issues.append("incomplete_extraction")
    if not old_positions or not new_positions:
        issues.append("empty_extraction")
    visual = any(
        extraction.mime_type.startswith(("image/", "application/pdf"))
        for extraction in (old, new)
    )
    method = (
        "identical_asset"
        if old.source_sha256 == new.source_sha256
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
    changes: list[AnnexDifference] = []
    model_snapshot = None
    if method == "native_structure" and not issues:
        try:
            changes = _native_changes(old, new)
        except ValueError as error:
            issues.append(str(error))
    elif method == "simultaneous_vision":
        for extraction, pages in ((old, before_pages), (new, after_pages)):
            if extraction.page_count is None:
                issues.append("page_count_unverified")
            elif [page.page for page in pages] != list(
                range(1, extraction.page_count + 1)
            ):
                issues.append("page_coverage_mismatch")
            if not pages:
                issues.append("visual_evidence_unavailable")
            if not {element.locator.page for element in extraction.elements}.issubset(
                {page.page for page in pages}
            ):
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
                        page_images = _page_images(page)
                    except ValueError as error:
                        issues.append(str(error))
                        break
                    manifest.append(
                        f"{side} page {page.page}: image {len(images) + 1} full page; images {len(images) + 2}..{len(images) + len(page_images)} detail tiles in top-to-bottom, left-to-right order at {DETAIL_TILE_PIXELS}px"
                    )
                    images.extend(page_images)
                    if len(images) > MAX_PAGE_IMAGES:
                        issues.append("comparison_image_budget")
                        break
            prompt = (
                f"Instruction (evidence only): {instruction}\nOLD source hash: {old.source_sha256}\nNEW source hash: {new.source_sha256}\nOLD complete elements:\n"
                + "\n".join(
                    _reference(old, index).model_dump_json() for index in old_positions
                )
                + "\nNEW complete elements:\n"
                + "\n".join(
                    _reference(new, index).model_dump_json() for index in new_positions
                )
                + "\nImage manifest (one-based):\n"
                + "\n".join(manifest)
            )
            if len(prompt) > MAX_COMPARISON_PROMPT_CHARS:
                issues.append("comparison_prompt_budget")
            if not issues:
                response = generate_structured(
                    llm,
                    flow=LLMFlow.REGULATORY_ANNEX_COMPARISON,
                    system_prompt=ANNEX_COMPARISON_PROMPT,
                    user_prompt=prompt,
                    response_model=AnnexComparisonResponse,
                    image_parts=images,
                    max_tokens=16000,
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
        coverage=coverage,
        model_snapshot=model_snapshot,
        issues=sorted(set(issues)),
        ready=not issues,
    )


def validate_annex_comparison(
    comparison: AnnexComparison, *, old: AnnexExtraction, new: AnnexExtraction
) -> list[str]:
    """Revalidate persisted/editable comparison data before deriving write targets."""
    issues = [*comparison.issues, *old.issues, *new.issues]
    if any(
        element.status != "readable" or element.issues
        for extraction in (old, new)
        for element in extraction.elements
    ):
        issues.append("incomplete_extraction")
    if not _atomic_positions(old) or not _atomic_positions(new):
        issues.append("empty_extraction")
    visual = any(
        extraction.mime_type.startswith(("image/", "application/pdf"))
        for extraction in (old, new)
    )
    expected_method = (
        "identical_asset"
        if old.source_sha256 == new.source_sha256
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
            if extraction.page_count is None:
                issues.append("page_count_unverified")
            elif pages != list(range(1, extraction.page_count + 1)):
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
            old_positions=_atomic_positions(old),
            new_positions=_atomic_positions(new),
            old_pages=comparison.coverage.old_pages,
            new_pages=comparison.coverage.new_pages,
        )
    )
    return sorted(set(issues))
