"""Source-backed legal units and displaced publication headers in Markdown."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from onyx.regulatory.provision_identity import canonical_clause_label

if TYPE_CHECKING:
    from onyx.regulatory.chunker import SourceBlock

_ALPHABET = "abcçdefgğhıijklmnoöprsştuüvyz"
_INLINE_CLAUSE = re.compile(
    r"(?<!\S)(?P<label>(?P<letter>[a-zçğıöşü])(?P=letter))\)\s+", re.I
)
_QUOTED = re.compile(
    r'“[^”]*”|"[^"\n]*"|«[^»]*»|‘[^’]*’|(?<!`)(?P<ticks>`+)(?!`).*?(?<!`)(?P=ticks)(?!`)'
)
_CLAUSE = re.compile(
    r"^(?:[-+*]\s+)?(?:\d{1,3}[.)]\s+)?(?P<label>[a-zçğıöşü]{1,2})\)\s+\S", re.I
)
_FIRST_ARTICLE = re.compile(r"^(?:#{1,6}\s+)?MADDE\s+1\s*[-–—:]", re.I)
_SECTION = re.compile(
    r"^(?:#{1,6}\s+)?(?:BİRİNCİ|İKİNCİ|ÜÇÜNCÜ|DÖRDÜNCÜ)\s+BÖLÜM", re.I
)
_PUBLISHER = re.compile(
    r"^[^.;:\n]{3,120}(?:Bakanlığından|Başkanlığından|Kurulundan):$", re.I
)
_TITLE = re.compile(r"\b(?:YÖNETMELİK|YÖNETMELİĞİ|TEBLİĞ|TEBLİĞİ|KANUNU|KARARI)$", re.I)


@dataclass(frozen=True)
class SourceDocumentHeader:
    title: str
    chapter: str
    source_block: int


def _plain(text: str) -> str:
    return re.sub(r"^#{1,6}\s+", "", text).strip()


def source_document_header(blocks: list[SourceBlock]) -> SourceDocumentHeader | None:
    for block in blocks:
        if block.kind == "document_header":
            lines = [_plain(line) for line in block.text.splitlines() if line.strip()]
            return SourceDocumentHeader(
                title=lines[2],
                chapter=f"{lines[3]} — {lines[4]}",
                source_block=block.block_id,
            )
    return None


def _inline_units(block: SourceBlock) -> list[SourceBlock]:
    if block.kind != "paragraph":
        return [block]
    if block.text.count('"') % 2 or block.text.count("“") != block.text.count("”"):
        return [block]
    quoted = [match.span() for match in _QUOTED.finditer(block.text)]
    markers = [
        match
        for match in _INLINE_CLAUSE.finditer(block.text)
        if not any(start <= match.start() < end for start, end in quoted)
    ]
    if len(markers) < 2 or markers[0].start() != 0:
        return [block]
    labels = [canonical_clause_label(match["label"]) for match in markers]
    if any(len(set(label)) != 1 or label[0] not in _ALPHABET for label in labels):
        return [block]
    positions = [_ALPHABET.index(label[0]) for label in labels]
    if any(right != left + 1 for left, right in zip(positions, positions[1:])):
        return [block]
    result = []
    source_start = block.start_char
    # A line block may include indentation before its stripped text.
    for index, marker in enumerate(markers):
        end = (
            markers[index + 1].start() if index + 1 < len(markers) else len(block.text)
        )
        fragment = block.text[marker.start() : end]
        result.append(
            replace(
                block,
                text=fragment.rstrip(),
                start_char=source_start + marker.start(),
                end_char=source_start + end,
            )
        )
    return result


def _outside_fences(blocks: list[SourceBlock]) -> set[int]:
    allowed: set[int] = set()
    fence: str | None = None
    for index, block in enumerate(blocks):
        marker = re.match(r"^\s*(`{3,}|~{3,})", block.text)
        if marker:
            token = marker[1]
            if fence is None:
                fence = token
            elif token[0] == fence[0] and len(token) >= len(fence):
                fence = None
            continue
        if fence is None:
            allowed.add(index)
    return allowed


def legal_markdown_blocks(blocks: list[SourceBlock], source: str) -> list[SourceBlock]:
    """Interpret formatting without moving or rewriting any operative source text."""
    outside = _outside_fences(blocks)
    expanded = []
    for index, block in enumerate(blocks):
        # Resolve leading whitespace against the original source before slicing.
        offset = source[block.start_char : block.end_char].find(block.text)
        aligned = replace(block, start_char=block.start_char + max(0, offset))
        slices = _inline_units(aligned) if index in outside else [block]
        expanded.extend(slices if len(slices) > 1 else [block])
    outside = _outside_fences(expanded)
    headers: list[tuple[int, int]] = []
    mastheads = [
        index
        for index, block in enumerate(expanded)
        if index in outside
        and block.kind == "heading"
        and _plain(block.text).casefold() in {"resmî gazete", "resmi gazete"}
    ]
    for index in range(1, len(expanded) - 6):
        if index not in outside or len(mastheads) != 1:
            continue
        group = expanded[index : index + 6]
        lines = [_plain(block.text) for block in group]
        if (
            group[0].kind != "heading"
            or lines[0].casefold() not in {"resmî gazete", "resmi gazete"}
            or _PUBLISHER.fullmatch(lines[1]) is None
            or len(lines[2]) > 240
            or lines[2].upper() != lines[2]
            or _TITLE.search(lines[2]) is None
            or lines[3].upper() != "BİRİNCİ BÖLÜM"
            or not re.fullmatch(r"[A-Za-zÇĞİÖŞÜçğıöşü ]{3,100}", lines[4])
            or lines[5].upper() not in {"YÖNETMELİK", "TEBLİĞ", "KANUN"}
            or any(block.kind == "table" for block in group)
            or not all(position in outside for position in range(index, index + 6))
        ):
            continue
        before = _CLAUSE.match(expanded[index - 1].text)
        after = _CLAUSE.match(expanded[index + 6].text)
        if not before or not after:
            continue
        left = canonical_clause_label(before["label"])
        right = canonical_clause_label(after["label"])
        if (
            len(left) != 1
            or len(right) != 1
            or left not in _ALPHABET
            or right not in _ALPHABET
        ):
            continue
        if _ALPHABET.index(right) != _ALPHABET.index(left) + 1:
            continue
        prior = expanded[:index]
        if not any(_FIRST_ARTICLE.match(block.text) for block in prior):
            continue
        if any(_SECTION.match(block.text) for block in prior):
            continue
        if any(
            _TITLE.search(_plain(block.text))
            and _plain(block.text).upper() == _plain(block.text)
            for block in prior
        ):
            continue
        headers.append((index, index + 6))
    if len(headers) == 1:
        start, end = headers[0]
        first, last = expanded[start], expanded[end - 1]
        header = replace(
            first,
            kind="document_header",
            # Keep the already-validated header fields; the original envelope may
            # also contain page markers removed by the ordinary Markdown reader.
            text="\n".join(block.text for block in expanded[start:end]),
            end_char=last.end_char,
        )
        expanded[start:end] = [header]
    return [replace(block, block_id=index) for index, block in enumerate(expanded)]
