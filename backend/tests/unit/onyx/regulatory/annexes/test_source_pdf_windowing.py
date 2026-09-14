import time
from io import BytesIO
from unittest.mock import MagicMock

import pytest
from pypdf import PdfWriter

from onyx.regulatory.amendments.annexes.models import AnnexVisionWireResult
from onyx.regulatory.amendments.annexes.rendering import render_annex_pages
from onyx.utils.process_isolation import run_in_isolated_process


def fifty_page_pdf() -> bytes:
    writer = PdfWriter()
    for _ in range(50):
        writer.add_blank_page(width=595, height=842)
    stream = BytesIO()
    writer.write(stream)
    return stream.getvalue()


def test_selected_pdf_pages_keep_original_numbers_with_bounded_rendering() -> None:
    pages = run_in_isolated_process(
        render_annex_pages,
        fifty_page_pdf(),
        "application/pdf",
        page_numbers=(46, 47, 48, 49),
        timeout=30,
    )
    assert [page.page for page in pages] == [46, 47, 48, 49]
    assert all(page.png.startswith(b"\x89PNG") for page in pages)
    assert all(page.width == 595 and page.height == 842 for page in pages)


def test_fifty_page_source_reads_every_page_without_rendering_entire_pdf(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from onyx.regulatory.amendments.annexes import extraction
    from onyx.regulatory.amendments.pdf_vision import pdf_transcript

    observed: list[int] = []

    def generate(*_args: object, **kwargs: object) -> AnnexVisionWireResult:
        prompt = str(kwargs["user_prompt"])
        page = int(prompt.split()[1].rstrip("."))
        observed.append(page)
        assert kwargs["use_streaming"] is False
        return AnnexVisionWireResult.model_validate(
            {
                "elements": [
                    {
                        "kind": "text",
                        "text": f"Page {page}",
                        "box": {"left": 0.1, "top": 0.1, "right": 0.9, "bottom": 0.2},
                        "status": "readable",
                        "issues": [],
                    }
                ]
            }
        )

    monkeypatch.setattr(extraction, "generate_structured", generate)
    model = MagicMock()
    model.config.model_provider = "fixture"
    model.config.model_name = "vision"
    result = extraction.extract_annex_structure(
        fifty_page_pdf(),
        "application/pdf",
        vision_llm=model,
        vision_deadline=time.monotonic() + 7200,
    )
    assert observed == list(range(1, 51))
    assert result.page_count == 50 and not result.issues
    assert pdf_transcript(result).split("\n\n") == [f"Page {n}" for n in range(1, 51)]


def test_source_stops_reading_more_pages_when_structure_budget_is_exhausted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from onyx.regulatory.amendments.annexes import extraction

    response = AnnexVisionWireResult.model_validate(
        {
            "elements": [
                {
                    "kind": "text",
                    "text": "x" * 20000,
                    "box": {"left": 0.1, "top": 0.1, "right": 0.9, "bottom": 0.2},
                    "status": "readable",
                }
                for _ in range(60)
            ]
        }
    )
    generate = MagicMock(return_value=response)
    monkeypatch.setattr(extraction, "generate_structured", generate)
    model = MagicMock()
    model.config.model_provider = "fixture"
    model.config.model_name = "vision"
    with pytest.raises(ValueError, match="annex_structure_limit"):
        extraction.extract_annex_structure(
            fifty_page_pdf(),
            "application/pdf",
            vision_llm=model,
            vision_deadline=time.monotonic() + 7200,
        )
    assert generate.call_count == 2
