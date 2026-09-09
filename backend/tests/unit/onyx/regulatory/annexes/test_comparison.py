import hashlib
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from onyx.regulatory.amendments.annexes.models import (
    AnnexExtraction,
    AnnexLocator,
    AnnexRenderedPage,
    ExtractedAnnexElement,
)


def extraction(value: str, *, visual: bool = False) -> AnnexExtraction:
    return AnnexExtraction(
        page_count=1 if visual else None,
        source_sha256=hashlib.sha256(value.encode()).hexdigest(),
        mime_type="image/png" if visual else "text/html",
        elements=[
            ExtractedAnnexElement(
                kind="table_cell",
                text=value,
                semantic_key="rate:A",
                locator=AnnexLocator(
                    page=1 if visual else None,
                    row=2,
                    column=2,
                    normalized_box=(0.1, 0.1, 0.9, 0.9) if visual else None,
                ),
            )
        ],
    )


def page(color: str = "white", number: int = 1) -> AnnexRenderedPage:
    buffer = BytesIO()
    Image.new("RGB", (200, 200), color).save(buffer, format="PNG")
    return AnnexRenderedPage(page=number, width=200, height=200, png=buffer.getvalue())


def test_identical_asset_skips_vision_and_covers_all_elements() -> None:
    from onyx.regulatory.amendments.annexes.comparison import compare_annexes

    old = extraction("5%", visual=True)
    llm = MagicMock()
    result = compare_annexes(old=old, new=old, llm=llm)
    assert result.ready and result.changes == []
    assert result.coverage.old_positions == [0] == result.coverage.new_positions
    llm.invoke.assert_not_called()


def test_native_one_cell_change_and_move_preserve_correspondence() -> None:
    from onyx.regulatory.amendments.annexes.comparison import compare_annexes

    old, new = extraction("5%"), extraction("7%")
    result = compare_annexes(old=old, new=new)
    assert result.ready
    assert [
        (item.operation, item.old[0].text, item.new[0].text) for item in result.changes
    ] == [("replace", "5%", "7%")]
    new.elements[0].text = "5%"
    new.elements[0].locator.row = 3
    result = compare_annexes(old=old, new=new)
    assert result.changes[0].operation == "move"


def test_different_pixels_with_equal_ocr_requires_simultaneous_vision() -> None:
    from onyx.regulatory.amendments.annexes.comparison import compare_annexes
    from onyx.regulatory.amendments.annexes.models import (
        AnnexComparisonResponse,
        AnnexDifference,
        AnnexElementReference,
    )

    old, new = extraction("", visual=True), extraction("", visual=True)
    new.source_sha256 = "b" * 64
    reference = AnnexElementReference(
        position=0, text="", locator=old.elements[0].locator
    )
    response = AnnexComparisonResponse(
        changes=[
            AnnexDifference(
                operation="visual",
                old=[reference],
                new=[reference],
                explanation="Red warning symbol replaced black symbol",
            )
        ],
        old_positions=[0],
        new_positions=[0],
        old_pages=[1],
        new_pages=[1],
    )
    llm = MagicMock()
    llm.config.model_provider = "configured"
    llm.config.model_name = "vision"
    with patch(
        "onyx.regulatory.amendments.annexes.comparison.generate_structured",
        return_value=response,
    ) as invoke:
        result = compare_annexes(
            old=old, new=new, old_pages=[page()], new_pages=[page("red")], llm=llm
        )
    assert result.ready and result.changes[0].operation == "visual"
    assert len(invoke.call_args.kwargs["image_parts"]) >= 2
    assert (
        "OLD" in invoke.call_args.kwargs["user_prompt"]
        and "NEW" in invoke.call_args.kwargs["user_prompt"]
    )


def test_missing_page_or_uncertain_extraction_never_ready() -> None:
    from onyx.regulatory.amendments.annexes.comparison import compare_annexes

    old, new = extraction("5%", visual=True), extraction("7%", visual=True)
    result = compare_annexes(
        old=old, new=new, old_pages=[page(), page(number=2)], new_pages=[page()]
    )
    assert not result.ready and "page_coverage_mismatch" in result.issues
    old = extraction("5%")
    old.elements[0].status = "uncertain"
    assert not compare_annexes(old=old, new=extraction("7%")).ready


@pytest.mark.parametrize("invalid", ["position", "text", "locator"])
def test_model_references_are_checked_against_actual_snapshot(invalid: str) -> None:
    from onyx.regulatory.amendments.annexes.comparison import compare_annexes
    from onyx.regulatory.amendments.annexes.models import (
        AnnexComparisonResponse,
        AnnexDifference,
        AnnexElementReference,
    )

    old, new = extraction("5%", visual=True), extraction("7%", visual=True)
    ref = AnnexElementReference(
        position=99 if invalid == "position" else 0,
        text="invented" if invalid == "text" else "5%",
        locator=AnnexLocator(page=7)
        if invalid == "locator"
        else old.elements[0].locator,
    )
    response = AnnexComparisonResponse(
        changes=[
            AnnexDifference(
                operation="replace",
                old=[ref],
                new=[
                    AnnexElementReference(
                        position=0, text="7%", locator=new.elements[0].locator
                    )
                ],
                explanation="Rate",
            )
        ],
        old_positions=[0],
        new_positions=[0],
        old_pages=[1],
        new_pages=[1],
    )
    llm = MagicMock()
    llm.config.model_provider = "configured"
    llm.config.model_name = "vision"
    with patch(
        "onyx.regulatory.amendments.annexes.comparison.generate_structured",
        return_value=response,
    ):
        result = compare_annexes(
            old=old, new=new, old_pages=[page()], new_pages=[page("red")], llm=llm
        )
    assert not result.ready and any("reference" in issue for issue in result.issues)


def test_visual_region_count_difference_is_not_legal_deletion() -> None:
    from onyx.regulatory.amendments.annexes.comparison import compare_annexes
    from onyx.regulatory.amendments.annexes.models import AnnexComparisonResponse

    old, new = extraction("5%", visual=True), extraction("5%", visual=True)
    new.source_sha256 = "b" * 64
    old.elements.append(
        ExtractedAnnexElement(
            kind="image_region", text="", locator=AnnexLocator(page=1)
        )
    )
    response = AnnexComparisonResponse(
        old_positions=[0, 1], new_positions=[0], old_pages=[1], new_pages=[1]
    )
    llm = MagicMock()
    llm.config.model_provider = "configured"
    llm.config.model_name = "vision"
    with patch(
        "onyx.regulatory.amendments.annexes.comparison.generate_structured",
        return_value=response,
    ):
        result = compare_annexes(
            old=old, new=new, old_pages=[page()], new_pages=[page()], llm=llm
        )
    assert result.ready and result.changes == []


def test_one_physical_visual_change_cannot_be_returned_twice() -> None:
    from pydantic import ValidationError

    from onyx.regulatory.amendments.annexes.models import AnnexComparisonResponse

    reference = {"position": 0, "text": "Green square", "locator": {"page": 1}}
    change = {
        "operation": "replace",
        "old": [reference],
        "new": [{**reference, "text": "Blue square"}],
        "explanation": "Caption describes color",
    }
    with pytest.raises(ValidationError, match="overlapping"):
        AnnexComparisonResponse.model_validate(
            {
                "changes": [change, {**change, "operation": "visual"}],
                "old_positions": [0],
                "new_positions": [0],
                "old_pages": [1],
                "new_pages": [1],
            }
        )


def test_complete_annex_can_grow_and_move_an_element_across_pages() -> None:
    from onyx.regulatory.amendments.annexes.comparison import compare_annexes
    from onyx.regulatory.amendments.annexes.models import (
        AnnexComparisonResponse,
        AnnexDifference,
        AnnexElementReference,
    )

    old, new = extraction("5%", visual=True), extraction("5%", visual=True)
    old.page_count = 1
    new.page_count = 2
    new.source_sha256 = "b" * 64
    new.elements[0].locator.page = 2
    response = AnnexComparisonResponse(
        changes=[
            AnnexDifference(
                operation="move",
                old=[
                    AnnexElementReference(
                        position=0, text="5%", locator=old.elements[0].locator
                    )
                ],
                new=[
                    AnnexElementReference(
                        position=0, text="5%", locator=new.elements[0].locator
                    )
                ],
                explanation="Row moved to continuation page",
            )
        ],
        old_positions=[0],
        new_positions=[0],
        old_pages=[1],
        new_pages=[1, 2],
    )
    llm = MagicMock()
    llm.config.model_provider = "configured"
    llm.config.model_name = "vision"
    with patch(
        "onyx.regulatory.amendments.annexes.comparison.generate_structured",
        return_value=response,
    ) as invoke:
        result = compare_annexes(
            old=old,
            new=new,
            old_pages=[page()],
            new_pages=[page(), page(number=2)],
            llm=llm,
        )
    assert result.ready and result.changes[0].operation == "move"
    assert invoke.call_count == 1


def test_native_formula_change_is_explicit_when_printed_value_is_identical() -> None:
    from onyx.regulatory.amendments.annexes.comparison import compare_annexes

    old, new = extraction("5%"), extraction("5%")
    new.source_sha256 = "b" * 64
    old.elements[0].formula = "=A1/100"
    new.elements[0].formula = "=B1/100"
    result = compare_annexes(old=old, new=new)
    assert not result.ready and "formula_change_requires_review" in result.issues
