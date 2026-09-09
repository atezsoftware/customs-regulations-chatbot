"""Bounded native rendering; only called in a disposable process."""

from contextlib import closing
from io import BytesIO

from onyx.regulatory.amendments.annexes.models import (
    AnnexLocator,
    AnnexRenderedPage,
    ExtractedAnnexElement,
)
from onyx.regulatory.amendments.annexes.source_parser import apply_source_process_limits

MAX_RENDER_PIXELS = 64_000_000
MAX_RENDER_BYTES = 100 * 1024 * 1024


def render_annex_pages(content: bytes, mime_type: str) -> list[AnnexRenderedPage]:
    apply_source_process_limits()
    from PIL import Image

    pages: list[AnnexRenderedPage] = []
    pixels = 0
    output_bytes = 0

    def append_page(page: AnnexRenderedPage, pixel_count: int) -> None:
        nonlocal pixels, output_bytes
        pixels += pixel_count
        output_bytes += len(page.png)
        if pixels > MAX_RENDER_PIXELS or output_bytes > MAX_RENDER_BYTES:
            raise ValueError("annex_render_limit")
        pages.append(page)

    if mime_type == "application/pdf":
        import pypdfium2 as pdfium

        with pdfium.PdfDocument(content) as document:
            if len(document) > 500:
                raise ValueError("annex_page_limit")
            for index in range(len(document)):
                with closing(document[index]) as page:
                    width, height = page.get_size()
                    if width * height * 4 + pixels > MAX_RENDER_PIXELS:
                        raise ValueError("annex_render_limit")
                    elements: list[ExtractedAnnexElement] = []
                    with closing(page.get_textpage()) as textpage:
                        for rectangle in range(textpage.count_rects()):
                            left, bottom, right, top = textpage.get_rect(rectangle)
                            text = textpage.get_text_bounded(left, bottom, right, top)
                            elements.append(
                                ExtractedAnnexElement(
                                    kind="text",
                                    text=text,
                                    locator=AnnexLocator(
                                        page=index + 1,
                                        original_box=(
                                            left,
                                            height - top,
                                            right,
                                            height - bottom,
                                        ),
                                        normalized_box=(
                                            left / width,
                                            (height - top) / height,
                                            right / width,
                                            (height - bottom) / height,
                                        ),
                                        original_width=width,
                                        original_height=height,
                                        coordinate_system="top_left_points",
                                    ),
                                )
                            )
                    if page.get_rotation() != 0:
                        # PDFium text rectangles use unrotated PDF user space;
                        # the rendered preview's display coordinates differ.
                        with closing(page.get_textpage()) as textpage:
                            textpage.count_rects()
                            for index_in_page, element in enumerate(elements):
                                element.locator.original_box = textpage.get_rect(
                                    index_in_page
                                )
                                element.locator.normalized_box = None
                                element.locator.coordinate_system = "pdf_user_space"
                                element.status = "uncertain"
                                element.issues.append("rotated_pdf_native_coordinates")
                    bitmap = page.render(scale=2)
                    try:
                        buffer = BytesIO()
                        bitmap.to_pil().save(buffer, format="PNG")
                        append_page(
                            AnnexRenderedPage(
                                page=index + 1,
                                width=width,
                                height=height,
                                png=buffer.getvalue(),
                                text_elements=elements,
                            ),
                            int(width * height * 4),
                        )
                    finally:
                        bitmap.close()
    else:
        with Image.open(BytesIO(content)) as image:
            if getattr(image, "n_frames", 1) > 500:
                raise ValueError("annex_page_limit")
            for frame in range(getattr(image, "n_frames", 1)):
                image.seek(frame)
                width, height = image.size
                if width * height + pixels > MAX_RENDER_PIXELS:
                    raise ValueError("annex_render_limit")
                buffer = BytesIO()
                normalized = image.convert("RGB")
                normalized.info.clear()
                normalized.save(buffer, format="PNG")
                append_page(
                    AnnexRenderedPage(
                        original_orientation=int(image.getexif().get(274, 1)),
                        page=frame + 1,
                        width=width,
                        height=height,
                        png=buffer.getvalue(),
                    ),
                    width * height,
                )
    return pages
