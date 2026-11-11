import base64
import io
from pathlib import Path

import pytest
from PIL import Image

from onyx.regulatory.document_intake import markdown_package as package


def _image_uri(format: str = "PNG") -> str:
    stream = io.BytesIO()
    Image.new("RGB", (3, 2), "white").save(stream, format=format)
    mime = "png" if format == "PNG" else "jpeg"
    return f"data:image/{mime};base64," + base64.b64encode(stream.getvalue()).decode()


def test_embedded_png_is_valid_markdown_without_rewriting_source() -> None:
    text = f"# Kural\n\nÖnce. ![Çizelge]({_image_uri()}) Sonra.\n"
    references, title = package._markdown_references(text, "ek.md")
    assert references == []
    assert title == "Kural"


def test_actual_docling_annex_embedded_png_is_accepted() -> None:
    source = Path(__file__).parent / "fixtures/docling_737039bf/20230928-1-1.md"
    text = source.read_bytes().decode("utf-8")
    package._markdown_references(text, source.name)


@pytest.mark.parametrize("format", ["PNG", "JPEG"])
def test_exact_offsets_and_original_bytes_for_repeated_images(format: str) -> None:
    from onyx.regulatory.document_intake.markdown_images import embedded_markdown_images

    uri = _image_uri(format)
    syntax = f'![Çizelge *A*]({uri} "Başlık")'
    text = f"Önce Türkçe.\r\n{syntax}\r\nArada. {syntax} Sonra."
    images = embedded_markdown_images(text, path="ek.md")
    assert len(images) == 2
    assert all(text[item.start : item.end] == syntax for item in images)
    assert images[0].start == text.index(syntax)
    assert images[1].start == text.rindex(syntax)
    assert images[0].end < images[1].start
    assert images[0].content == base64.b64decode(uri.split(",", 1)[1])
    assert images[0].descriptor.width == 3
    assert images[0].descriptor.height == 2
    assert images[0].descriptor.mime_type == (
        "image/png" if format == "PNG" else "image/jpeg"
    )
    assert images[0].alt == "Çizelge A"


@pytest.mark.parametrize("syntax", ["![alt][resim]", "![resim][]", "![resim]"])
def test_reference_images_have_occurrence_spans_not_definition_spans(
    syntax: str,
) -> None:
    from onyx.regulatory.document_intake.markdown_images import embedded_markdown_images

    text = f"Önce {syntax} sonra.\n\n[resim]: <{_image_uri()}> 'Çizelge'\n"
    (image,) = embedded_markdown_images(text, path="ek.md")
    assert text[image.start : image.end] == syntax
    assert image.end < text.index("[resim]:")


def test_images_in_fences_code_and_escaped_syntax_are_not_extracted() -> None:
    from onyx.regulatory.document_intake.markdown_images import embedded_markdown_images

    syntax = f"![alt]({_image_uri()})"
    prefix = f"```md\n{syntax}\n```\n\n`{syntax}`\n\n\\{syntax}\n\n    {syntax}\n\n"
    text = prefix + syntax + "\n\n" + prefix
    (image,) = embedded_markdown_images(text, path="ek.md")
    assert image.start == len(prefix)
    assert image.end == len(prefix) + len(syntax)
    assert embedded_markdown_images(prefix, path="ek.md") == ()


@pytest.mark.parametrize(
    "prefix,suffix",
    [("> ", ""), ("- ", ""), ("[", "](https://example.com)"), ("| A | ", " |")],
)
def test_image_spans_in_quotes_lists_links_and_tables(prefix: str, suffix: str) -> None:
    from onyx.regulatory.document_intake.markdown_images import embedded_markdown_images

    syntax = f"![alt]({_image_uri()})"
    text = prefix + syntax + suffix
    (image,) = embedded_markdown_images(text, path="ek.md")
    assert image.start == len(prefix)
    assert text[image.start : image.end] == syntax


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/image.png",
        "file:///tmp/image.png",
        "../image.png",
        "data:image/svg+xml;base64,PHN2Zy8+",
        "data:image/png,not-base64",
        "data:image/png;base64,@@@@",
        "data:image/png;base64,aA==",
        "data:image/png;base64,",
    ],
)
def test_invalid_or_external_images_fail_without_returning_source(url: str) -> None:
    from onyx.regulatory.document_intake.markdown_images import (
        EmbeddedMarkdownImageError,
        embedded_markdown_images,
    )

    with pytest.raises(EmbeddedMarkdownImageError) as caught:
        embedded_markdown_images(f"Secret text ![alt]({url})", path="ek.md")
    assert "Secret text" not in str(caught.value)
    assert url not in str(caught.value)
    assert caught.value.path == "ek.md"


def test_mime_mismatch_and_corrupt_bytes_fail_validation() -> None:
    from onyx.regulatory.document_intake.markdown_images import (
        EmbeddedMarkdownImageError,
        embedded_markdown_images,
    )

    for uri in [
        _image_uri().replace("image/png", "image/jpeg"),
        _image_uri()[:-8] + "AAAAAAAA",
    ]:
        with pytest.raises(
            EmbeddedMarkdownImageError, match="invalid_embedded_markdown_image"
        ):
            embedded_markdown_images(f"![alt]({uri})", path="ek.md")


def test_image_size_and_occurrence_limits_are_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from onyx.regulatory.document_intake import markdown_images as images

    uri = _image_uri()
    monkeypatch.setattr(images, "MAX_SOURCE_IMAGE_BYTES", 1)
    with pytest.raises(images.EmbeddedMarkdownImageError, match="image_byte_limit"):
        images.embedded_markdown_images(f"![alt]({uri})", path="ek.md")
    monkeypatch.setattr(images, "MAX_SOURCE_IMAGE_BYTES", 8 * 1024 * 1024)
    monkeypatch.setattr(images, "MAX_EMBEDDED_MARKDOWN_IMAGES", 1)
    with pytest.raises(images.EmbeddedMarkdownImageError, match="image_count_limit"):
        images.embedded_markdown_images(f"![alt]({uri}) ![alt]({uri})", path="ek.md")


def test_unmappable_multiline_quote_image_fails_explicitly() -> None:
    from onyx.regulatory.document_intake.markdown_images import (
        EmbeddedMarkdownImageError,
        embedded_markdown_images,
    )

    text = f"> ![alt\n> continuation]({_image_uri()})"
    with pytest.raises(
        EmbeddedMarkdownImageError, match="unsupported_markdown_image_source_span"
    ):
        embedded_markdown_images(text, path="ek.md")


def test_package_keeps_raw_html_image_and_binary_as_markdown_guards() -> None:
    with pytest.raises(
        package.MarkdownPackageError, match="images_or_embedded_resources"
    ):
        package._markdown_references(f'<img src="{_image_uri()}">', "ek.md")
    with pytest.raises(
        package.MarkdownPackageError,
        match="binary_source_requires_markdown_transcription",
    ):
        package._decode(base64.b64decode(_image_uri().split(",", 1)[1]), "ek.md")


def test_source_candidate_and_aggregate_limits_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from onyx.regulatory.document_intake import markdown_images as images

    text = f"![alt]({_image_uri()})"
    monkeypatch.setattr(images, "MAX_MARKDOWN_IMAGE_SOURCE_CHARS", 2)
    with pytest.raises(
        images.EmbeddedMarkdownImageError, match="source_character_limit"
    ):
        images.embedded_markdown_images(text, path="ek.md")
    monkeypatch.setattr(images, "MAX_MARKDOWN_IMAGE_SOURCE_CHARS", 2_000_000)
    monkeypatch.setattr(images, "MAX_MARKDOWN_IMAGE_CANDIDATES", 1)
    with pytest.raises(images.EmbeddedMarkdownImageError, match="candidate_limit"):
        images.embedded_markdown_images(f"`![fake]`\n\n{text}", path="ek.md")
    monkeypatch.setattr(images, "MAX_MARKDOWN_IMAGE_CANDIDATES", 256)
    monkeypatch.setattr(images, "MAX_EMBEDDED_MARKDOWN_IMAGE_BYTES", 1)
    with pytest.raises(images.EmbeddedMarkdownImageError, match="total_byte_limit"):
        images.embedded_markdown_images(text, path="ek.md")


def test_oversized_dimensions_fail_source_image_validator() -> None:
    from onyx.regulatory.document_intake.markdown_images import (
        EmbeddedMarkdownImageError,
        embedded_markdown_images,
    )

    stream = io.BytesIO()
    Image.new("RGB", (8193, 1)).save(stream, format="PNG")
    uri = "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode()
    with pytest.raises(
        EmbeddedMarkdownImageError, match="invalid_embedded_markdown_image"
    ):
        embedded_markdown_images(f"![alt]({uri})", path="ek.md")


def test_reference_definition_span_and_title_are_preserved() -> None:
    from onyx.regulatory.document_intake.markdown_images import embedded_markdown_images

    definition = f'[resim]: <{_image_uri()}> "Takvim &amp; kapsam"\r\n'
    text = f"Önce ![Şema][resim] sonra.\r\n\r\n{definition}\r\nSon kural.\r\n"
    (image,) = embedded_markdown_images(text, path="ek.md")
    assert image.alt == "Şema"
    assert image.title == "Takvim & kapsam"
    assert image.resource_spans == (
        (text.index(definition), text.index(definition) + len(definition)),
    )
    assert text[image.resource_spans[0][0] : image.resource_spans[0][1]] == definition


def test_inline_image_title_has_no_definition_resource_span() -> None:
    from onyx.regulatory.document_intake.markdown_images import embedded_markdown_images

    (image,) = embedded_markdown_images(
        f'![Şema]({_image_uri()} "Yıllık takvim")', path="ek.md"
    )
    assert image.title == "Yıllık takvim"
    assert image.resource_spans == ()


def test_repeated_reference_images_share_only_active_definition_span() -> None:
    from onyx.regulatory.document_intake.markdown_images import embedded_markdown_images

    definition = f'[resim]: {_image_uri()} "Takvim"\n'
    prefix = f"```md\n{definition}```\n\n`{definition.rstrip()}`\n\n\\{definition}\n"
    text = prefix + "![A][resim] ve ![B][resim]\n\n" + definition
    images = embedded_markdown_images(text, path="ek.md")
    assert len(images) == 2
    span = (text.rindex(definition), len(text))
    assert images[0].resource_spans == images[1].resource_spans == (span,)
    assert all(item.title == "Takvim" for item in images)


def test_multiline_definition_resource_span_includes_title_only() -> None:
    from onyx.regulatory.document_intake.markdown_images import embedded_markdown_images

    definition = f'[resim]:\n  <{_image_uri()}>\n  "Satır takvimi"\n'
    text = f"![Şema][resim]\n\n{definition}\nSon kural.\n"
    (image,) = embedded_markdown_images(text, path="ek.md")
    assert image.title == "Satır takvimi"
    assert image.resource_spans == (
        (text.index(definition), text.index(definition) + len(definition)),
    )


def test_definition_shared_with_normal_link_fails_explicitly() -> None:
    from onyx.regulatory.document_intake.markdown_images import (
        EmbeddedMarkdownImageError,
        embedded_markdown_images,
    )

    text = f"![Şema][resim] ve [indirme][resim]\n\n[resim]: {_image_uri()}\n"
    with pytest.raises(
        EmbeddedMarkdownImageError, match="shared_markdown_image_reference"
    ):
        embedded_markdown_images(text, path="ek.md")


def test_unmappable_quote_reference_definition_fails_explicitly() -> None:
    from onyx.regulatory.document_intake.markdown_images import (
        EmbeddedMarkdownImageError,
        embedded_markdown_images,
    )

    text = f"> ![Şema][resim]\n>\n> [resim]: {_image_uri()}\n"
    with pytest.raises(
        EmbeddedMarkdownImageError, match="unsupported_markdown_image_resource_span"
    ):
        embedded_markdown_images(text, path="ek.md")


def test_multiple_image_reference_definitions_can_be_removed_together() -> None:
    from onyx.regulatory.document_intake.markdown_images import embedded_markdown_images

    first = f'[first]: <{_image_uri()}> "Birinci"\n'
    second = f'[second]: <{_image_uri()}> "İkinci"'
    text = f"Önce ![A][first] ve ![B][second].\n\n{first}{second}"
    first_image, second_image = embedded_markdown_images(text, path="ek.md")
    assert first_image.title == "Birinci"
    assert second_image.title == "İkinci"
    assert first_image.resource_spans == (
        (text.index(first), text.index(first) + len(first)),
    )
    assert second_image.resource_spans == ((text.index(second), len(text)),)
    edits = [
        (image.start, image.end, image.alt) for image in (first_image, second_image)
    ]
    edits += [
        (start, end, "")
        for image in (first_image, second_image)
        for start, end in image.resource_spans
    ]
    for start, end, replacement in sorted(edits, reverse=True):
        text = text[:start] + replacement + text[end:]
    assert text == "Önce A ve B.\n\n"
