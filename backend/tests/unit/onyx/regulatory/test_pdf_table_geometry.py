import hashlib
import json
from io import BytesIO
from unittest.mock import MagicMock

import pytest

from onyx.llm.model_response import Choice, Message, ModelResponse
from onyx.regulatory.amendments.annexes.models import (
    AcquiredAsset,
    AnnexExtraction,
    AnnexLocator,
    AnnexModelSnapshot,
    AnnexVisionWireResult,
    ExtractedAnnexElement,
)
from onyx.regulatory.amendments.pdf_vision import pdf_transcript
from onyx.regulatory.structured_llm import generate_structured
from onyx.tracing.flows import LLMFlow


def cell(text: str, box: tuple[float, float, float, float]) -> ExtractedAnnexElement:
    return ExtractedAnnexElement(
        kind="table_cell",
        text=text,
        extraction_method="vision",
        locator=AnnexLocator(page=1, normalized_box=box),
    )


def page(elements: list[ExtractedAnnexElement]) -> AnnexExtraction:
    return AnnexExtraction(
        source_sha256="a" * 64,
        mime_type="application/pdf",
        page_count=1,
        model_snapshot=AnnexModelSnapshot(
            model_provider="fixture", model_name="vision"
        ),
        elements=elements,
    )


@pytest.mark.parametrize("tall_column", ["left", "right"])
def test_merged_form_preserves_each_cell_without_inventing_rows(
    tall_column: str,
) -> None:
    if tall_column == "left":
        elements = [
            cell("Merged field", (0.1, 0.1, 0.4, 0.7)),
            cell("Upper field", (0.4, 0.1, 0.9, 0.4)),
            cell("Lower field", (0.4, 0.4, 0.9, 0.7)),
        ]
        expected = "Merged field\nUpper field\nLower field"
    else:
        elements = [
            cell("Upper field", (0.1, 0.1, 0.4, 0.4)),
            cell("Merged field", (0.4, 0.1, 0.9, 0.7)),
            cell("Lower field", (0.1, 0.4, 0.4, 0.7)),
        ]
        expected = "Upper field\nMerged field\nLower field"
    assert pdf_transcript(page(elements)) == expected


def test_small_vertical_overlap_retains_cells_individually() -> None:
    assert (
        pdf_transcript(
            page(
                [
                    cell("First field", (0.1, 0.1, 0.4, 0.3)),
                    cell("Second field", (0.5, 0.29, 0.9, 0.5)),
                ]
            )
        )
        == "First field\nSecond field"
    )


def test_regular_grid_keeps_pipe_rows_and_touching_borders() -> None:
    assert (
        pdf_transcript(
            page(
                [
                    cell("Name", (0.1, 0.1, 0.4, 0.3)),
                    cell("Rate", (0.4, 0.1, 0.9, 0.3)),
                    cell("Fixture", (0.1, 0.3, 0.4, 0.5)),
                    cell("17%", (0.4, 0.3, 0.9, 0.5)),
                ]
            )
        )
        == "| Name | Rate |\n| --- | --- |\n| Fixture | 17% |"
    )


def test_regular_grid_keeps_legacy_pipe_rows_at_version_two() -> None:
    # Frozen version-2 transcripts must never gain the version-3 markdown
    # header separator retroactively.
    assert (
        pdf_transcript(
            page(
                [
                    cell("Name", (0.1, 0.1, 0.4, 0.3)),
                    cell("Rate", (0.4, 0.1, 0.9, 0.3)),
                    cell("Fixture", (0.1, 0.3, 0.4, 0.5)),
                    cell("17%", (0.4, 0.3, 0.9, 0.5)),
                ]
            ),
            version=2,
        )
        == "Name | Rate\nFixture | 17%"
    )


def test_actual_cell_collision_is_rejected_across_intervening_text() -> None:
    note = cell("Grounded note", (0.7, 0.2, 0.9, 0.25)).model_copy(
        update={"kind": "text"}
    )
    with pytest.raises(ValueError, match="pdf_table_cells_overlap"):
        pdf_transcript(
            page(
                [
                    cell("First field", (0.1, 0.1, 0.5, 0.4)),
                    note,
                    cell("Colliding field", (0.2, 0.3, 0.6, 0.5)),
                ]
            )
        )


def test_actual_cell_collision_uses_existing_structured_correction_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from onyx.regulatory import structured_llm

    now = [100.0]
    monkeypatch.setattr(structured_llm.time, "monotonic", lambda: now[0])
    model = MagicMock()
    model.config.model_provider = "fixture"
    model.config.model_name = "vision"

    def response(left: float) -> ModelResponse:
        return ModelResponse(
            id="fixture",
            created="2026-09-14",
            choice=Choice(
                message=Message(
                    content=json.dumps(
                        {
                            "elements": [
                                {
                                    "kind": "table_cell",
                                    "text": "A",
                                    "status": "readable",
                                    "box": {
                                        "left": 0.1,
                                        "top": 0.1,
                                        "right": 0.5,
                                        "bottom": 0.3,
                                    },
                                },
                                {
                                    "kind": "table_cell",
                                    "text": "B",
                                    "status": "readable",
                                    "box": {
                                        "left": left,
                                        "top": 0.1,
                                        "right": 0.9,
                                        "bottom": 0.3,
                                    },
                                },
                            ]
                        }
                    )
                )
            ),
        )

    def invoke(*_args: object, **_kwargs: object) -> ModelResponse:
        if model.invoke.call_count == 1:
            now[0] += 3
            return response(0.4)
        return response(0.5)

    model.invoke.side_effect = invoke
    result = generate_structured(
        model,
        flow=LLMFlow.REGULATORY_ANNEX_EXTRACTION,
        system_prompt="Preserve grounded table cells.",
        user_prompt="Fixture page",
        response_model=AnnexVisionWireResult,
        deadline=110,
        use_streaming=False,
    )
    assert result.elements[1].box.left == 0.5
    assert [
        call.kwargs["timeout_override"] for call in model.invoke.call_args_list
    ] == [10, 7]
    assert "pdf_table_cells_overlap" in model.invoke.call_args.args[0][-1].content


@pytest.mark.parametrize("legacy_case", ["separated_rows", "separated_collision"])
def test_frozen_legacy_transcripts_are_verified_and_reused_without_regeneration(
    legacy_case: str,
) -> None:
    from onyx.regulatory.amendments import pdf_vision

    if legacy_case == "separated_rows":
        elements = [
            cell("A", (0.1, 0.1, 0.3, 0.3)),
            cell("Heading", (0.35, 0.15, 0.45, 0.2)).model_copy(
                update={"kind": "text"}
            ),
            cell("B", (0.4, 0.29, 0.6, 0.5)),
            cell("C", (0.6, 0.29, 0.8, 0.5)),
        ]
        frozen_text = "A\nHeading\nB | C"
    else:
        elements = [
            cell("A", (0.1, 0.1, 0.5, 0.4)),
            cell("Heading", (0.7, 0.2, 0.9, 0.25)).model_copy(update={"kind": "text"}),
            cell("B", (0.2, 0.3, 0.6, 0.5)),
        ]
        frozen_text = "A\nHeading\nB"
    original = b"frozen source PDF"
    source_sha256 = hashlib.sha256(original).hexdigest()
    extraction = page(elements).model_copy(update={"source_sha256": source_sha256})
    data = extraction.model_dump_json().encode()
    reference = {
        "file_id": "old-derivative",
        "sha256": hashlib.sha256(data).hexdigest(),
        "transcript_sha256": pdf_vision.digest(frozen_text),
    }
    saved = {
        "sha256": source_sha256,
        "mime_type": "application/pdf",
        "text": frozen_text,
        "native_text": "Old native text",
        "pdf_vision": reference,
    }
    store = MagicMock()
    store.read_file.side_effect = lambda _key: BytesIO(data)
    assert pdf_vision.load_pdf_extraction(saved, source_sha256, store) == extraction
    reused = pdf_vision.reuse_pdf_source(
        AcquiredAsset(
            sha256=source_sha256,
            content=original,
            mime_type="application/pdf",
            display_name="old.pdf",
        ),
        saved,
        store,
    )
    assert reused.text == frozen_text and reused.native_text == "Old native text"
    assert reused.pdf_vision is not None
    assert reused.pdf_vision.model_dump(mode="json") == reference
    store.save_file.assert_not_called()
    with pytest.raises(ValueError):
        pdf_vision.load_pdf_extraction(
            {**saved, "pdf_vision": {**reference, "transcript_version": 2}},
            source_sha256,
            store,
        )


def test_new_pdf_derivatives_explicitly_select_current_transcript_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from onyx.regulatory.amendments import pdf_vision

    extraction = page([cell("A", (0.1, 0.1, 0.5, 0.3))])
    monkeypatch.setattr(
        pdf_vision, "extract_annex_structure", lambda *_args, **_kwargs: extraction
    )
    store = MagicMock()
    store.save_file.return_value = "new-derivative"
    prepared = pdf_vision.prepare_pdf_source(
        AcquiredAsset(
            sha256="a" * 64,
            content=b"PDF",
            mime_type="application/pdf",
            display_name="new.pdf",
        ),
        store=store,
        llm=MagicMock(),
        deadline=pdf_vision.time.monotonic() + 60,
    )
    assert prepared.pdf_vision is not None
    assert prepared.pdf_vision.model_dump(mode="json")["transcript_version"] == 3
