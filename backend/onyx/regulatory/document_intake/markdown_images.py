"""Bounded embedded-image validation and verified original Markdown source spans."""

import base64
import binascii
import re
from bisect import bisect_left
from collections.abc import Callable
from dataclasses import dataclass, replace
from html import unescape
from typing import Match, TypedDict, cast

from mistune import BlockState, InlineState, Markdown
from mistune.helpers import parse_link_label
from mistune.inline_parser import _parse_link_text

from onyx.regulatory.source_image_evidence import (
    MAX_SOURCE_IMAGE_BYTES,
    SourceImageDescriptor,
    validate_source_image,
)

MAX_MARKDOWN_IMAGE_SOURCE_CHARS = 2_000_000
MAX_EMBEDDED_MARKDOWN_IMAGES = 32
MAX_MARKDOWN_IMAGE_CANDIDATES = 256
MAX_EMBEDDED_MARKDOWN_IMAGE_BYTES = 16 * 1024 * 1024
_DATA_URI = re.compile(r"data:(image/png|image/jpeg);base64,([A-Za-z0-9+/]*={0,2})\Z")
_IMAGE_START = re.compile(r"!\[")


class EmbeddedMarkdownImageError(ValueError):
    """A source-free, bounded diagnostic for the Markdown intake boundary."""

    def __init__(self, code: str, *, path: str) -> None:
        self.code = code
        self.path = path
        super().__init__(f"{code}: {ascii(path[:512])}")


@dataclass(frozen=True)
class EmbeddedMarkdownImage:
    start: int
    end: int
    alt: str
    content: bytes
    descriptor: SourceImageDescriptor
    title: str | None = None
    # Image-only reference definitions may be omitted from derived text, never raw source.
    resource_spans: tuple[tuple[int, int], ...] = ()


class _Node(TypedDict, total=False):
    type: str
    attrs: dict[str, object]
    children: list["_Node"]
    raw: str
    ref: str


def _image_nodes(nodes: list[_Node]) -> list[_Node]:
    result: list[_Node] = []
    for node in nodes:
        if node.get("type") == "image":
            result.append(node)
        result.extend(_image_nodes(node.get("children", [])))
    return result


def _canonical(
    nodes: list[_Node], *, ignore_blank_lines: bool = False
) -> list[dict[str, object]]:
    """Reference and inline forms render identically; retain every other AST field."""
    result: list[dict[str, object]] = []
    for node in nodes:
        if ignore_blank_lines and node.get("type") == "blank_line":
            continue
        item: dict[str, object] = {key: value for key, value in node.items()}
        if node.get("type") in {"image", "link"}:
            for key in ("literal_url", "ref", "label"):
                item.pop(key, None)
        if "children" in node:
            item["children"] = _canonical(
                node["children"], ignore_blank_lines=ignore_blank_lines
            )
        result.append(item)
    return result


def _alt(node: _Node) -> str:
    return unescape(node.get("raw", "")) + "".join(
        _alt(child) for child in node.get("children", [])
    )


def _decode_image(url: object, *, path: str) -> tuple[bytes, SourceImageDescriptor]:
    if not isinstance(url, str) or not url.startswith("data:"):
        raise EmbeddedMarkdownImageError("images_require_transcription", path=path)
    if len(url) > 64 + 4 * ((MAX_SOURCE_IMAGE_BYTES + 2) // 3):
        raise EmbeddedMarkdownImageError(
            "embedded_markdown_image_byte_limit", path=path
        )
    match = _DATA_URI.fullmatch(url)
    if match is None:
        raise EmbeddedMarkdownImageError("invalid_embedded_markdown_image", path=path)
    try:
        content = base64.b64decode(match.group(2), validate=True)
        # Reject noncanonical padding and nonzero unused bits as malformed data.
        if base64.b64encode(content).decode("ascii") != match.group(2):
            raise ValueError("Noncanonical base64")
        descriptor = validate_source_image(content, match.group(1))
    except (ValueError, binascii.Error):
        raise EmbeddedMarkdownImageError(
            "invalid_embedded_markdown_image", path=path
        ) from None
    return content, descriptor


def _link_references(nodes: list[_Node]) -> set[str]:
    result: set[str] = set()
    for node in nodes:
        if node.get("type") == "link" and isinstance(node.get("ref"), str):
            result.add(node["ref"])
        result.update(_link_references(node.get("children", [])))
    return result


def _edit_source(text: str, edits: list[tuple[int, int, str]]) -> str:
    for start, end, replacement in sorted(edits, reverse=True):
        text = text[:start] + replacement + text[end:]
    return text


def _resource_spans(
    text: str,
    nodes: list[_Node],
    images: list[EmbeddedMarkdownImage],
    image_references: list[str | None],
    original_references: dict[str, dict[str, object]],
    parser: Callable[[], Markdown],
    *,
    path: str,
) -> dict[str, tuple[tuple[int, int], ...]]:
    references = {key for key in image_references if key is not None}
    if not references:
        return {}
    if references & _link_references(nodes):
        raise EmbeddedMarkdownImageError("shared_markdown_image_reference", path=path)

    # Work in Mistune's normalized coordinates, retaining exact original offsets.
    crlf_positions = [
        match.start() - index for index, match in enumerate(re.finditer("\r\n", text))
    ]
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    source = normalized if normalized.endswith("\n") else normalized + "\n"

    def original_offset(position: int) -> int:
        return min(len(text), position + bisect_left(crlf_positions, position))

    image_edits = [
        (image.start, image.end, f"ONYX_IMAGE_{index}")
        for index, image in enumerate(images)
    ]
    marker_tokens = cast(list[_Node], parser()(_edit_source(text, image_edits)))
    marker_ast = _canonical(marker_tokens, ignore_blank_lines=True)
    result: dict[str, tuple[tuple[int, int], ...]] = {}
    block = parser().block
    candidates = 0
    for match in block.compile_sc(["ref_link"]).finditer(source):
        candidates += 1
        if candidates > MAX_MARKDOWN_IMAGE_CANDIDATES:
            raise EmbeddedMarkdownImageError(
                "markdown_image_candidate_limit", path=path
            )
        candidate = BlockState()
        candidate.process(source)
        candidate.cursor = match.start()
        end = block.parse_ref_link(match, candidate)
        candidate_references = cast(
            dict[str, dict[str, object]], candidate.env["ref_links"]
        )
        if end is None or len(candidate_references) != 1:
            continue
        key = next(iter(candidate_references))
        if (
            key not in references
            or candidate_references[key] != original_references[key]
        ):
            continue
        start, finish = original_offset(match.start()), original_offset(end)
        # Removing a definition-looking string in code must not count as removing
        # the active binding. The remaining reference environment must be identical.
        _removed_tokens, removed_state = parser().parse(
            _edit_source(text, [(start, finish, "")])
        )
        remaining = cast(dict[str, dict[str, object]], removed_state.env["ref_links"])
        if remaining != {
            ref: value for ref, value in original_references.items() if ref != key
        }:
            continue
        if any(start < image.end and finish > image.start for image in images):
            continue
        removed = _edit_source(text, image_edits + [(start, finish, "")])
        removed_tokens = cast(list[_Node], parser()(removed))
        if _canonical(removed_tokens, ignore_blank_lines=True) != marker_ast:
            continue
        if key in result:
            raise EmbeddedMarkdownImageError(
                "unsupported_markdown_image_resource_span", path=path
            )
        result[key] = ((start, finish),)
    if set(result) != references:
        raise EmbeddedMarkdownImageError(
            "unsupported_markdown_image_resource_span", path=path
        )
    definition_edits = [
        (start, end, "") for spans in result.values() for start, end in spans
    ]
    combined_tokens = cast(
        list[_Node], parser()(_edit_source(text, image_edits + definition_edits))
    )
    if _canonical(combined_tokens, ignore_blank_lines=True) != marker_ast:
        raise EmbeddedMarkdownImageError(
            "unsupported_markdown_image_resource_span", path=path
        )
    return result


def embedded_markdown_images(
    text: str, *, path: str
) -> tuple[EmbeddedMarkdownImage, ...]:
    """Extract genuine image nodes without fetching or changing original text.

    Mistune omits source offsets and rewrites block/list prefixes. For a possible
    original span, substitute only its destination with a probe, parse again, and
    require the complete AST to be identical after restoring that image's attrs.
    This proves both the image context and its exact span, even when identical
    syntax occurs in code or reference definitions. Unmappable spans fail closed.
    """
    # Runtime import keeps the package's shared literal/reference grammar authoritative.
    from onyx.regulatory.document_intake.markdown_package import (
        MarkdownPackageError,
        _LiteralBlockParser,
        _LiteralInlineParser,
    )

    def parser() -> Markdown:
        return Markdown(block=_LiteralBlockParser(), inline=_LiteralInlineParser())

    if len(text) > MAX_MARKDOWN_IMAGE_SOURCE_CHARS:
        raise EmbeddedMarkdownImageError(
            "markdown_image_source_character_limit", path=path
        )
    try:
        tokens_raw, state = parser().parse(text)
        tokens = cast(list[_Node], tokens_raw)
        nodes = _image_nodes(tokens)
        if not nodes:
            return ()
        if len(nodes) > MAX_EMBEDDED_MARKDOWN_IMAGES:
            raise EmbeddedMarkdownImageError(
                "embedded_markdown_image_count_limit", path=path
            )
        # Validate actual nodes first, including images whose spans cannot be mapped.
        validated: dict[str, tuple[bytes, SourceImageDescriptor]] = {}
        total_bytes = 0
        for node in nodes:
            url = node.get("attrs", {}).get("url")
            content, descriptor = _decode_image(url, path=path)
            assert isinstance(url, str)
            validated[url] = content, descriptor
            total_bytes += len(content)
            if total_bytes > MAX_EMBEDDED_MARKDOWN_IMAGE_BYTES:
                raise EmbeddedMarkdownImageError(
                    "embedded_markdown_image_total_byte_limit", path=path
                )
        original = _canonical(tokens)
        expected_images = [_canonical([node]) for node in nodes]
        inline = _LiteralInlineParser()
        images: list[EmbeddedMarkdownImage] = []
        image_references: list[str | None] = []
        candidates: list[Match[str]] = []
        for match in _IMAGE_START.finditer(text):
            candidates.append(match)
            if len(candidates) > MAX_MARKDOWN_IMAGE_CANDIDATES:
                raise EmbeddedMarkdownImageError(
                    "markdown_image_candidate_limit", path=path
                )
        for match in candidates:
            candidate = InlineState(state.env)
            candidate.src = text
            end = inline.parse_link(match, candidate)
            if end is None or len(candidate.tokens) != 1:
                continue
            node = cast(_Node, candidate.tokens[0])
            if node.get("type") != "image" or _canonical([node]) not in expected_images:
                continue
            label, label_end = parse_link_label(text, match.end())
            if label is None:
                _label, label_end = _parse_link_text(candidate, match.end())
            if label_end is None:
                continue
            marker = f"urn:onyx:markdown-image-span:{match.start()}"
            probe = text[:label_end] + f"({marker})" + text[end:]
            probe_tokens = cast(list[_Node], parser()(probe))
            marked = [
                image
                for image in _image_nodes(probe_tokens)
                if image.get("attrs", {}).get("url") == marker
            ]
            if len(marked) != 1:
                continue
            marked[0]["attrs"] = node.get("attrs", {})
            if (
                _canonical([marked[0]]) != _canonical([node])
                or _canonical(probe_tokens) != original
            ):
                continue
            url = node.get("attrs", {}).get("url")
            if not isinstance(url, str) or url not in validated:
                continue
            content, descriptor = validated[url]
            title = node.get("attrs", {}).get("title")
            image_references.append(node.get("ref"))
            images.append(
                EmbeddedMarkdownImage(
                    start=match.start(),
                    end=end,
                    alt=_alt(node),
                    content=content,
                    descriptor=descriptor,
                    title=unescape(title) if isinstance(title, str) else None,
                )
            )
        if len(images) != len(nodes) or any(
            left.end > right.start for left, right in zip(images, images[1:])
        ):
            raise EmbeddedMarkdownImageError(
                "unsupported_markdown_image_source_span", path=path
            )
        resources = _resource_spans(
            text,
            tokens,
            images,
            image_references,
            cast(dict[str, dict[str, object]], state.env["ref_links"]),
            parser,
            path=path,
        )
        return tuple(
            replace(
                image, resource_spans=resources.get(reference, ()) if reference else ()
            )
            for image, reference in zip(images, image_references)
        )
    except MarkdownPackageError as error:
        raise EmbeddedMarkdownImageError(error.code, path=path) from None
    except (RecursionError, ValueError) as error:
        if isinstance(error, EmbeddedMarkdownImageError):
            raise
        raise EmbeddedMarkdownImageError("invalid_markdown", path=path) from None
