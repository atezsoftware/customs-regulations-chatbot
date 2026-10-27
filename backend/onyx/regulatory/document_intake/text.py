"""Source-preserving preparation and coverage gates used only for new documents."""

import re
from collections.abc import Sequence
from html import escape, unescape
from html.parser import HTMLParser
from typing import Any, Match
from urllib.parse import unquote, urlsplit

from mistune import BlockParser, BlockState, InlineParser, InlineState
from mistune.helpers import unescape_char

from onyx.regulatory.amendments.annexes.models import SourceLink
from onyx.regulatory.chunker import RegulatoryChunker
from onyx.regulatory.document_intake.models import IntakeSourceAsset

_INLINE_ARTICLE = re.compile(
    r"^(?P<heading>[ \t]*(?:#{1,6}[ \t]+)?(?:\*\*)?"
    r"(?:(?:GEÇİCİ|GECICI|EK|MÜKERRER|MUKERRER)[ \t]+)?MADDE[ \t]*"
    r"\d+(?:[ \t]*/[ \t]*[A-ZÇĞİÖŞÜa-zçğıöşü0-9]+)?[ \t]*(?:\*\*)?[ \t]*[-–—−])"
    r"[ \t]*(?P<body>\S[^\n]*)$",
    re.IGNORECASE | re.MULTILINE,
)
_ANNEX_NUMBER_VALUE = r"\d+(?:[/.]\d+)*|[IVXLCDM]+|[A-ZÇĞİÖŞÜ]"
_ATTACHMENT_PLACEHOLDER = re.compile(
    rf"\bek(?:(?:leri|ler)|\s*[-–—:_]?\s*(?P<number>{_ANNEX_NUMBER_VALUE}))?"
    r"\s+(?:için|icin)\s+t[ıi]klay[ıi]n[ıi]z\b",
    re.IGNORECASE,
)
_ANNEX_NUMBER = re.compile(
    rf"\bEK\s*[-–—:_]?\s*({_ANNEX_NUMBER_VALUE})\b",
    re.IGNORECASE,
)
_WRAPPED_ANNEX_REFERENCE = re.compile(
    r"(?m)^(?P<prose>[^\n|]*\S[ \t]+EK[ \t]*[-–—])[ \t]*"
    r"(?P<line_break>\n[ \t]*)(?P<number>\d+(?:[/.]\d+)*[A-Z]?)\b",
    re.IGNORECASE,
)
_WRAPPED_ARTICLE = re.compile(
    r"(?m)^(?P<label>[ \t]*(?:(?:GEÇİCİ|GECICI|MÜKERRER|MUKERRER|EK)"
    r"[ \t]*(?:\n[ \t]*)?)?MADDE[ \t]+\d+(?:/?[A-Z])?)"
    r"[ \t]*(?:\n[ \t]*)?(?P<dash>[-–—])[ \t]*(?=\n[ \t]*\(\d{1,3}\))",
    re.I,
)


def _navigation_suffix(label: str) -> str | None:
    match = _ATTACHMENT_PLACEHOLDER.match(label.strip())
    if match is None:
        return None
    suffix = label.strip()[match.end() :].strip(" \t\r\n—–-:;.,()[]")
    suffix = re.sub(r"\s*\(?PDF\)?$", "", suffix, flags=re.IGNORECASE).strip()
    if suffix:
        number = _ANNEX_NUMBER.fullmatch(suffix)
        if number is None or (
            match["number"] and match["number"].upper() != number[1].upper()
        ):
            return None
    if match["number"]:
        return f"EK-{match['number'].upper()}"
    return suffix


def _reference_key(value: str) -> str:
    return unquote(unescape(unescape_char(value))).split("#", 1)[0]


def _annex_label(
    label: str, link: SourceLink, assets: dict[str, IntakeSourceAsset]
) -> str:
    suffix = _navigation_suffix(label)
    number = _ANNEX_NUMBER.fullmatch(suffix or label.strip())
    if number is not None:
        return f"EK-{number[1].upper()}"
    asset = assets.get(link.target_asset_hash or "")
    names = [
        asset.display_name if asset else "",
        urlsplit(
            link.final_url or link.requested_url or link.original_url or ""
        ).path.rsplit("/", 1)[-1],
    ]
    for name in names:
        number = _ANNEX_NUMBER.search(unquote(name))
        if number:
            return f"EK-{number[1].upper()}"
    title = re.sub(
        r"\.(?:md|pdf|html?|docx?|txt)$", "", names[0], flags=re.IGNORECASE
    ).strip()
    if title and _ATTACHMENT_PLACEHOLDER.search(title) is None:
        return title
    return "Eki"


def attachment_reference_label(
    link: SourceLink,
    *,
    package_assets: Sequence[IntakeSourceAsset] = (),
) -> str:
    """A concise display label; immutable source relationships retain their wording."""
    label = link.label or ""
    if _navigation_suffix(label) is None:
        return label
    return _annex_label(label, link, {asset.sha256: asset for asset in package_assets})


def _mask_code(text: str) -> tuple[str, dict[str, Any]]:
    state = BlockState()
    state.process(text if text.endswith("\n") else text + "\n")
    BlockParser().parse(state)
    ranges: list[tuple[int, int]] = []

    def visit(tokens: list[dict[str, Any]]) -> None:
        for token in tokens:
            raw = token.get("raw")
            if token.get("type") == "block_code" and isinstance(raw, str) and raw:
                start = text.find(raw)
                while start >= 0:
                    ranges.append((start, start + len(raw)))
                    start = text.find(raw, start + len(raw))
            visit(token.get("children", []))

    visit(state.tokens)
    delimiters = list(re.finditer(r"`+", text))
    following: dict[int, int] = {}
    next_same: dict[int, int] = {}
    for index in range(len(delimiters) - 1, -1, -1):
        width = delimiters[index].end() - delimiters[index].start()
        if width in following:
            next_same[index] = following[width]
        following[width] = index
    index = 0
    while index < len(delimiters):
        closing = next_same.get(index)
        if closing is None:
            index += 1
            continue
        ranges.append((delimiters[index].start(), delimiters[closing].end()))
        index = closing + 1
    masked = list(text)
    for start, end in ranges:
        for position in range(start, end):
            if masked[position] != "\n":
                masked[position] = " "
    return "".join(masked), dict(state.env)


def _token_text(token: dict[str, Any]) -> str:
    return token.get("raw", "") + "".join(
        _token_text(child) for child in token.get("children", [])
    )


def _apply_replacements(text: str, replacements: list[tuple[int, int, str]]) -> str:
    for start, end, value in sorted(replacements, reverse=True):
        text = text[:start] + value + text[end:]
    return text


class _NavigationLinks(InlineParser):
    def __init__(
        self, links: dict[str, SourceLink], assets: dict[str, IntakeSourceAsset]
    ) -> None:
        super().__init__()
        self.links, self.assets = links, assets
        self.replacements: list[tuple[int, int, str]] = []

    def parse_link(self, m: Match[str], state: InlineState) -> int | None:
        end = super().parse_link(m, state)
        if end is None or not state.tokens or state.in_link:
            return end
        token = state.tokens[-1]
        if token["type"] != "link":
            return end
        label = _token_text(token)
        if _navigation_suffix(label) is None:
            return end
        target = self.links.get(_reference_key(token["attrs"]["url"]))
        if target is None:
            raise ValueError(
                "Attachment navigation does not match a resolved source asset"
            )
        # Replace only the visible label, retaining inline/reference targets verbatim.
        depth, closing = 0, m.end()
        while closing < end:
            if state.src[closing] == "\\":
                closing += 2
                continue
            if state.src[closing] == "[":
                depth += 1
            elif state.src[closing] == "]":
                if depth == 0:
                    break
                depth -= 1
            closing += 1
        self.replacements.append(
            (m.end(), closing, _annex_label(label, target, self.assets))
        )
        return end


class _NavigationHtml(HTMLParser):
    def __init__(
        self,
        text: str,
        links: dict[str, SourceLink],
        assets: dict[str, IntakeSourceAsset],
    ) -> None:
        super().__init__(convert_charrefs=True)
        self.lines = [0]
        self.lines.extend(match.end() for match in re.finditer("\n", text))
        self.links, self.assets = links, assets
        self.anchor: tuple[int, str, list[str]] | None = None
        self.replacements: list[tuple[int, int, str]] = []

    def _position(self) -> int:
        line, column = self.getpos()
        return self.lines[line - 1] + column

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        href = dict(attrs).get("href")
        if tag == "a" and href is not None:
            self.anchor = (
                self._position() + len(self.get_starttag_text() or ""),
                href,
                [],
            )

    def handle_data(self, data: str) -> None:
        if self.anchor:
            self.anchor[2].append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag != "a" or self.anchor is None:
            return
        start, href, parts = self.anchor
        label = "".join(parts)
        self.anchor = None
        if _navigation_suffix(label) is None:
            return
        target = self.links.get(_reference_key(href))
        if target is None:
            raise ValueError(
                "Attachment navigation does not match a resolved source asset"
            )
        self.replacements.append(
            (start, self._position(), escape(_annex_label(label, target, self.assets)))
        )


def normalize_attachment_navigation(
    text: str,
    *,
    asset_sha256: str,
    links: Sequence[SourceLink],
    package_assets: Sequence[IntakeSourceAsset] = (),
) -> str:
    """Remove source navigation prompts while retaining verified annex references."""
    if _ATTACHMENT_PLACEHOLDER.search(text) is None:
        return text
    outgoing = [
        link
        for link in links
        if link.parent_asset_hash == asset_sha256
        and link.target_asset_hash
        and link.target_asset_hash != asset_sha256
    ]
    assets = {asset.sha256: asset for asset in package_assets}
    by_url = {
        _reference_key(url): link
        for link in outgoing
        for url in (link.original_url, link.requested_url, link.final_url)
        if url
    }
    masked, environment = _mask_code(text)
    parser = _NavigationLinks(by_url, assets)
    state = InlineState(environment)
    state.src = masked
    parser.parse(state)
    text = _apply_replacements(text, parser.replacements)
    masked, _environment = _mask_code(text)
    html = _NavigationHtml(masked, by_url, assets)
    html.feed(masked)
    html.close()
    text = _apply_replacements(text, html.replacements)
    masked, _environment = _mask_code(text)
    replacements = []
    # A plain navigation sentence may follow operative PDF text on the same line.
    for match in re.finditer(
        rf"(?m)(?:^[ \t]*|(?<=[.!?])[ \t]+)(?P<label>{_ATTACHMENT_PLACEHOLDER.pattern}[^\n]*)$",
        masked,
        flags=re.IGNORECASE,
    ):
        label = match["label"]
        suffix = _navigation_suffix(label)
        if suffix is None or not outgoing:
            continue
        selected = [
            link
            for link in outgoing
            if not suffix
            or _annex_label(link.label or "", link, assets)
            == _annex_label(label, link, assets)
        ]
        if not selected:
            raise ValueError(
                "Attachment navigation does not match a resolved source asset"
            )
        references = list(
            dict.fromkeys(
                f"[{_annex_label(link.label or '', link, assets)}]({link.original_url or link.requested_url or link.final_url})"
                if (link.original_url or link.requested_url or link.final_url)
                else _annex_label(link.label or "", link, assets)
                for link in selected
            )
        )
        replacements.append(
            (match.start("label"), match.end("label"), ", ".join(references))
        )
    return _apply_replacements(text, replacements)


def require_resolved_attachment_references(
    text: str, *, asset_sha256: str, links: Sequence[SourceLink]
) -> None:
    if _ATTACHMENT_PLACEHOLDER.search(text) and not any(
        link.parent_asset_hash == asset_sha256
        and link.target_asset_hash
        and link.target_asset_hash != asset_sha256
        for link in links
    ):
        raise ValueError(
            "The source says to click for attachments, but no attachment target was "
            "resolved. Supply the original source URL or a file that retains its links."
        )


def normalize_inline_articles(
    text: str, *, preserve_markdown_headings: bool = False
) -> str:
    """Insert only whitespace so an inline article body is not consumed as its title."""
    return _INLINE_ARTICLE.sub(
        lambda match: (
            match[0]
            if preserve_markdown_headings and match["heading"].lstrip().startswith("#")
            else f"{match['heading']}\n{match['body']}"
        ),
        text,
    )


def normalize_wrapped_annex_references(text: str) -> str:
    """Keep a wrapped prose reference from becoming a numbered section heading."""
    masked, _environment = _mask_code(text)
    replacements = []
    for match in _WRAPPED_ANNEX_REFERENCE.finditer(masked):
        next_line_end = masked.find("\n", match.end())
        if next_line_end < 0:
            next_line_end = len(masked)
        if (
            match["prose"].lstrip().startswith("#")
            or re.match(r"^(?: {4}| {0,3}\t)", match["prose"])
            or "|" in masked[match.end() : next_line_end]
        ):
            continue
        replacements.append((*match.span("line_break"), ""))
    return _apply_replacements(text, replacements)


def normalize_wrapped_article_headings(text: str) -> str:
    """Join only a line-start article label/dash followed by a real numbered body."""
    masked, _environment = _mask_code(text)
    return _apply_replacements(
        text,
        [
            (
                match.start(),
                match.end(),
                f"{' '.join(match['label'].split())}{match['dash']}",
            )
            for match in _WRAPPED_ARTICLE.finditer(masked)
            if "\n" in match[0]
        ],
    )


def require_chunk_source_coverage(text: str) -> None:
    """Every source letter/digit must reach a chunk or one of its delivered headings."""
    chunked = RegulatoryChunker().chunk_text(text)
    if not chunked.chunks:
        raise ValueError("The source has no reviewable canonical chunks")
    intervals = [
        (chunk.metadata.source_start_char, chunk.metadata.source_end_char)
        for chunk in chunked.chunks
    ]
    referenced_nodes = {
        node_id for chunk in chunked.chunks for node_id in chunk.metadata.parent_ids
    }
    blocks = {block.block_id: block for block in chunked.blocks}
    for node in chunked.structure:
        if node.node_id in referenced_nodes and node.source_block is not None:
            block = blocks[node.source_block]
            intervals.append((block.start_char, block.end_char))
    title = chunked.metadata.title
    first_line = chunked.content.split("\n", 1)[0]
    visible_first_line = first_line.lstrip("# \t").strip("*_ \t")
    if title == visible_first_line and any(
        title in chunk.metadata.heading_path for chunk in chunked.chunks
    ):
        intervals.append((0, len(first_line)))
    cursor = 0
    for start, end in sorted(intervals):
        if any(character.isalnum() for character in chunked.content[cursor:start]):
            raise ValueError(
                "Canonical chunking would omit source content; review the source layout"
            )
        cursor = max(cursor, end)
    if any(character.isalnum() for character in chunked.content[cursor:]):
        raise ValueError(
            "Canonical chunking would omit source content; review the source layout"
        )


def prepare_intake_text(
    text: str,
    *,
    asset_sha256: str | None = None,
    links: Sequence[SourceLink] = (),
    package_assets: Sequence[IntakeSourceAsset] = (),
    preserve_markdown_headings: bool = False,
    formatter_version: int = 4,
) -> str:
    if formatter_version not in {2, 3, 4}:
        raise ValueError("The intake formatter version is not supported")
    if asset_sha256 is not None:
        text = normalize_attachment_navigation(
            text, asset_sha256=asset_sha256, links=links, package_assets=package_assets
        )
    normalized = normalize_inline_articles(
        normalize_wrapped_article_headings(text) if formatter_version >= 4 else text,
        preserve_markdown_headings=preserve_markdown_headings,
    )
    if formatter_version >= 3:
        normalized = normalize_wrapped_annex_references(normalized)
    if "".join(text.split()) != "".join(normalized.split()):
        raise ValueError("New document normalization changed the source wording")
    require_chunk_source_coverage(normalized)
    return normalized
