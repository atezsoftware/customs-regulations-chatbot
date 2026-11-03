"""Exact retained-source evidence for an unclassified whole repealed article."""

import datetime
import hashlib
import re
import unicodedata
from collections.abc import Sequence
from typing import Literal
from uuid import UUID

from bs4 import BeautifulSoup, Comment, NavigableString, Tag
from pydantic import BaseModel, ConfigDict, Field, model_validator

from onyx.document_index.publication_models import publication_digest
from onyx.regulatory.amendments.annexes.models import AnnexCanonicalSnapshot
from onyx.regulatory.amendments.annexes.sources import MAX_ASSET_BYTES
from onyx.regulatory.document_intake.structural_readers import (
    _NativeStyles,
    is_ordinal_chapter_heading,
)
from onyx.regulatory.labeling.intake_abstention_repair import AbstentionRepairAudit
from onyx.regulatory.law_identity import LAW_DOCUMENT_TYPES
from onyx.utils.web_content import decode_html_bytes

SOURCE_ONLY_KEY = "intake_source_only_self_repeal"
_SHA = r"^[0-9a-f]{64}$"
_ARTICLE = re.compile(r"^MADDE\s+(\d+)\s*[-–—:]", re.I)
_WHOLE = re.compile(
    r"MADDE\s+(?P<article>[1-9]\d*)\s*[-–—:]\s*"
    r"\(Mülga:\s*(?P<date>\d{1,2}/\d{1,2}/\d{4})\s*-\s*"
    r"(?P<instrument>(?:KHK-)?[1-9]\d*)/(?P<provision>[1-9]\d*)\s+md\.\)",
    re.I,
)


def normalized_text(text: str) -> str:
    return " ".join(text.split())


def _title(text: str) -> str:
    text = unicodedata.normalize("NFC", text).translate(
        str.maketrans({"I": "ı", "İ": "i"})
    )
    return unicodedata.normalize(
        "NFC", normalized_text(re.sub(r"\[\d+\]", "", text)).casefold()
    )


def _own_article_path(row: AnnexCanonicalSnapshot) -> tuple[str, ...]:
    matches = [
        index
        for index, heading in enumerate(row.heading_path)
        if (match := _ARTICLE.match(normalized_text(heading)))
        and match[1] == str(row.metadata.get("article_no"))
    ]
    if len(matches) != 1:
        raise ValueError("The canonical own article heading is ambiguous")
    return tuple(row.heading_path[: matches[0] + 1])


def _caption(row: AnnexCanonicalSnapshot) -> str:
    return _ARTICLE.sub("", _own_article_path(row)[-1]).strip()


class SelfRepealSourceProof(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    version: Literal[1] = 1
    user_file_id: UUID
    source_asset_id: UUID
    source_sha256: str = Field(pattern=_SHA)
    chunk_id: str
    document_number: str
    document_title: str
    article_number: str
    source_paragraph: int = Field(ge=0)
    canonical_scope_sha256: str = Field(pattern=_SHA)


class PriorFailedAbstention(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: UUID
    item_id: UUID
    request_hash: str = Field(pattern=_SHA)
    error: Literal[
        "Model abstained: insufficient evidence after bounded semantic reconsideration"
    ]
    source_snapshot_sha256: str = Field(pattern=_SHA)


class SourceOnlySelfRepealReceipt(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    version: Literal[1] = 1
    disposition: Literal["source_only_self_repeal"] = "source_only_self_repeal"
    classification: Literal["abstained"] = "abstained"
    proof: SelfRepealSourceProof
    frozen_target_sha256: str = Field(pattern=_SHA)
    taxonomy_sha256: str = Field(pattern=_SHA)
    provider_binding_sha256: str = Field(pattern=_SHA)
    provider_abstention: AbstentionRepairAudit | None = None
    prior_failed_abstention: PriorFailedAbstention | None = None

    @model_validator(mode="after")
    def one_abstention(self) -> "SourceOnlySelfRepealReceipt":
        if (self.provider_abstention is None) == (self.prior_failed_abstention is None):
            raise ValueError("One unchanged provider abstention lineage is required")
        if self.provider_abstention is not None and (
            not self.provider_abstention.outcome.abstained
            or self.provider_abstention.outcome.labels
        ):
            raise ValueError("Source-only disposition preserves an empty abstention")
        return self


def _visible_native(paragraph: Tag, styles: _NativeStyles) -> bool:
    if not any(
        isinstance(parent, Tag) and parent.name == "body"
        for parent in paragraph.parents
    ):
        return False
    for tag in (paragraph, *paragraph.parents, *paragraph.find_all(True)):
        if isinstance(tag, Tag) and (
            tag.name
            in {
                "table",
                "blockquote",
                "q",
                "template",
                "script",
                "noscript",
                "head",
                "style",
                "img",
                "svg",
                "canvas",
                "iframe",
                "object",
                "embed",
                "video",
                "audio",
            }
            or tag.has_attr("hidden")
            or styles.computed(tag)["display"] == "none"
            or styles.computed(tag)["visibility"] != "visible"
        ):
            return False
    return True


def _require_unambiguous_visibility(soup: BeautifulSoup) -> None:
    css = ";\n".join(
        [tag.get_text() for tag in soup.find_all("style")]
        + [str(tag.get("style", "")) for tag in soup.find_all(style=True)]
    )
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    if (
        re.search(
            r"(?:^|[;{])\s*(?:opacity|filter|clip|clip-path|content-visibility|transform)\s*:",
            css,
            re.I,
        )
        or re.search(
            r"font-size\s*:\s*(?:0+(?:\.0+)?|\.0+)(?:[a-z%]|\s|;|}|$)", css, re.I
        )
        or re.search(
            r"(?:^|[;{])\s*color\s*:[^;}]*?(?:transparent|rgba|hsla|#[0-9a-f]{8}\b|#[0-9a-f]{4}\b)",
            css,
            re.I,
        )
    ):
        raise ValueError("The native source has ambiguous visibility-affecting CSS")


def _native_interval_text(start: Tag, end: Tag, styles: _NativeStyles) -> str:
    texts: list[str] = []
    for node in start.next_elements:
        if node is end:
            return normalized_text(" ".join(texts))
        if start in node.parents:
            continue
        if isinstance(node, Tag) and node.name in {
            "img",
            "svg",
            "canvas",
            "iframe",
            "object",
            "embed",
            "video",
            "audio",
            "table",
        }:
            raise ValueError(
                "The native article region contains unmodeled visual content"
            )
        if (
            isinstance(node, NavigableString)
            and not isinstance(node, Comment)
            and node.strip()
        ):
            parent = node.parent
            if not isinstance(parent, Tag) or not _visible_native(parent, styles):
                raise ValueError(
                    "The native article region contains hidden or unsupported content"
                )
            texts.append(str(node))
    raise ValueError("The native own article DOM boundary is incomplete")


def _next_section_transition(
    *,
    start: Tag,
    end: Tag,
    styles: _NativeStyles,
    target: AnnexCanonicalSnapshot,
    next_rows: Sequence[AnnexCanonicalSnapshot],
    rows: Sequence[AnnexCanonicalSnapshot],
) -> None:
    """Account for a new canonical branch before the next own article."""
    paths = {_own_article_path(row) for row in next_rows}
    if len(paths) != 1:
        raise ValueError("The next own article branch is ambiguous")
    next_path = next(iter(paths))
    current_path = _own_article_path(target)
    common = 0
    for left, right in zip(current_path[:-1], next_path[:-1]):
        if left != right:
            break
        common += 1
    markers = list(next_path[common:-1])
    if (
        not common
        or not 1 <= len(markers) <= 4
        or any(
            not is_ordinal_chapter_heading(
                marker.removesuffix(" KISIM") + " BÖLÜM"
                if marker.endswith(" KISIM")
                else marker
            )
            for marker in markers
        )
    ):
        raise ValueError("The next branch has no exact canonical ordinal boundary")
    next_position = min(row.position for row in next_rows)
    transitions = sorted(
        (row for row in rows if target.position < row.position < next_position),
        key=lambda row: (row.position, row.id),
    )
    if len(transitions) > 8 or any(
        row.status != "active"
        or row.superseded_by_chunk_id is not None
        or row.metadata.get("chunk_variant") != "atomic"
        or row.chunk_type != "free_text"
        or row.metadata.get("article_no") is not None
        or not common < len(row.heading_path) < len(next_path)
        or tuple(row.heading_path) != next_path[: len(row.heading_path)]
        for row in transitions
    ):
        raise ValueError("The next section has no complete canonical caption scope")
    captions = [
        normalized_text(line)
        for row in transitions
        for line in row.text.splitlines()
        if line.strip()
    ]
    if len(captions) > 16:
        raise ValueError("The next canonical caption scope is too broad")
    blocks: list[Tag] = []
    for node in start.next_elements:
        if node is end:
            break
        if start in node.parents:
            continue
        if (
            isinstance(node, Tag)
            and node.name in {"p", "h1", "h2", "h3", "h4", "h5", "h6"}
            and normalized_text(node.get_text(" ", strip=True))
        ):
            blocks.append(node)
    values = [normalized_text(block.get_text(" ", strip=True)) for block in blocks]
    next_caption = _caption(next_rows[0])
    if not values or values[0] != markers[0] or values[-1] != next_caption:
        raise ValueError(
            "The next canonical section boundary or own caption is missing"
        )
    marker_index = caption_index = 0
    for block, value in zip(blocks[:-1], values[:-1]):
        if not _visible_native(block, styles) or not styles.centered_bold(block):
            raise ValueError("The next native section heading is hidden or ambiguous")
        if marker_index < len(markers) and value == markers[marker_index]:
            marker_index += 1
        elif caption_index < len(captions) and value == captions[caption_index]:
            caption_index += 1
        else:
            raise ValueError(
                "The native section contains extra or noncanonical content"
            )
    if marker_index != len(markers) or caption_index != len(captions):
        raise ValueError("The native next section is incomplete")
    if not _visible_native(blocks[-1], styles) or not styles.visible_bold(blocks[-1]):
        raise ValueError("The next native own article caption is hidden or ambiguous")
    if _native_interval_text(start, end, styles) != normalized_text(" ".join(values)):
        raise ValueError("The native next section contains unmodeled content")


def _own_header(paragraph: Tag, styles: _NativeStyles) -> bool:
    first = next(
        (
            node
            for node in paragraph.descendants
            if isinstance(node, NavigableString) and node.strip()
        ),
        None,
    )
    return (
        _visible_native(paragraph, styles)
        and first is not None
        and isinstance(first.parent, Tag)
        and re.match(r"madde\b", normalized_text(str(first)), re.I) is not None
        and styles.computed(first.parent)["font-weight"]
        in {"bold", "700", "800", "900"}
    )


def _closed_quotes(texts: Sequence[str]) -> bool:
    curly = False
    straight = False
    for text in texts:
        for char in text:
            if char == "“":
                if curly:
                    return False
                curly = True
            elif char == "”":
                if not curly:
                    return False
                curly = False
            elif char == '"':
                straight = not straight
    return not curly and not straight


# Public source checks shared by independently serialized source-only receipts.
NativeSourceStyles = _NativeStyles
native_title = _title
native_own_article_path = _own_article_path
native_article_caption = _caption
native_visible_content = _visible_native
require_native_visibility = _require_unambiguous_visibility
native_interval_text = _native_interval_text
native_own_header = _own_header
native_closed_quotes = _closed_quotes


def self_repeal_candidate(text: str) -> bool:
    return _WHOLE.fullmatch(normalized_text(text)) is not None


def build_self_repeal_source_proof(
    *,
    raw_html: bytes,
    source_sha256: str,
    source_asset_id: UUID,
    user_file_id: UUID,
    chunk_id: str,
    rows: Sequence[AnnexCanonicalSnapshot],
) -> SelfRepealSourceProof:
    """Accept only a complete native own article, never an excerpt or quotation."""
    if (
        len(raw_html) > MAX_ASSET_BYTES
        or hashlib.sha256(raw_html).hexdigest() != source_sha256
    ):
        raise ValueError("The retained self-repeal source bytes changed")
    by_id = {row.id: row for row in rows}
    row = by_id.get(chunk_id)
    if len(by_id) != len(rows) or any(
        r.user_file_id != str(user_file_id) for r in rows
    ):
        raise ValueError("The self-repeal canonical scope changed")
    if row is None or row.status != "active" or row.superseded_by_chunk_id is not None:
        raise ValueError("A current own self-repeal article is required")
    full = _WHOLE.fullmatch(normalized_text(row.text))
    if (
        full is None
        or row.chunk_type != "article"
        or row.metadata.get("chunk_variant") != "atomic"
        or str(row.metadata.get("article_no")) != full["article"]
        or str(row.metadata.get("document_type", "")).casefold()
        not in LAW_DOCUMENT_TYPES
    ):
        raise ValueError(
            "The target is not one complete unconditional self-repeal article"
        )
    try:
        datetime.datetime.strptime(full["date"], "%d/%m/%Y")
    except ValueError as error:
        raise ValueError("The self-repeal annotation date is invalid") from error
    article = full["article"]
    number = str(row.metadata.get("document_number", ""))
    title = row.metadata.get("document_title")
    if title is None and row.heading_path:
        root = row.heading_path[0]
        if re.fullmatch(r".+\s+kanunu?", _title(root)):
            title = root
    if not number.isdecimal() or not isinstance(title, str) or not title.strip():
        raise ValueError("The self-repeal article has no verified own law identity")
    atomics = [r for r in rows if r.metadata.get("chunk_variant") == "atomic"]
    if [r.id for r in atomics if str(r.metadata.get("article_no")) == article] != [
        chunk_id
    ]:
        raise ValueError("The self-repeal article is split or ambiguous")
    soup = BeautifulSoup(decode_html_bytes(raw_html, "text/html"), "html.parser")
    _require_unambiguous_visibility(soup)
    styles = _NativeStyles(soup)
    paragraphs = list(soup.find_all("p"))
    texts = [normalized_text(p.get_text(" ", strip=True)) for p in paragraphs]
    markers = [(i, m[1]) for i, text in enumerate(texts) if (m := _ARTICLE.match(text))]
    candidates = [(j, i) for j, (i, n) in enumerate(markers) if n == article]
    if len(candidates) != 1:
        raise ValueError("The native own article is missing or duplicated")
    j, index = candidates[0]
    if j == 0 or j + 1 == len(markers):
        raise ValueError("The native article has no complete own neighboring boundary")
    previous, following = markers[j - 1], markers[j + 1]
    if [int(n) for _, n in markers[: j + 2]] != list(range(1, int(following[1]) + 1)):
        raise ValueError(
            "The native article hierarchy contains quoted or missing provisions"
        )
    if int(previous[1]) + 1 != int(article) or int(article) + 1 != int(following[1]):
        raise ValueError("The native article sequence does not prove own identity")
    mastheads = [
        (i, m[1])
        for i, text in enumerate(texts[: markers[0][0]])
        if (m := re.fullmatch(r"Kanun Numarası\s*:\s*(\d+)", text, re.I))
    ]
    if len(mastheads) != 1 or mastheads[0][1] != number:
        raise ValueError("The retained source own law number differs")
    if _title(" ".join(texts[: mastheads[0][0]])) != _title(title):
        raise ValueError("The retained source own formal title differs")
    if any(not _own_header(paragraphs[i], styles) for i, _ in markers[: j + 2]):
        raise ValueError("A native own article boundary is hidden, quoted or ambiguous")
    if any(not _visible_native(p, styles) for p in paragraphs[: mastheads[0][0] + 1]):
        raise ValueError("The native source masthead is hidden or ambiguous")
    paragraph = paragraphs[index]
    prefix: list[str] = []
    if soup.body is None:
        raise ValueError("The native source has no visible body")
    for node in soup.body.descendants:
        if node is paragraph:
            break
        if isinstance(node, NavigableString) and not isinstance(node, Comment):
            prefix.append(str(node))
    if not _closed_quotes(prefix):
        raise ValueError("The native marker is inside an unresolved quotation")
    if texts[index] != normalized_text(row.text) or not styles.visible_bold(paragraph):
        raise ValueError("The native marker does not prove the complete canonical body")
    prior_caption_index = next(
        (i for i in range(index - 1, previous[0], -1) if texts[i]), None
    )
    if (
        prior_caption_index is None
        or not _visible_native(paragraphs[prior_caption_index], styles)
        or not styles.visible_bold(paragraphs[prior_caption_index])
    ):
        raise ValueError("The native own article caption is hidden or ambiguous")
    preceding = [text for text in texts[previous[0] + 1 : index] if text]
    if not preceding or preceding[-1] != _caption(row):
        raise ValueError("The native self-repeal article caption differs")
    next_rows = [
        r for r in atomics if str(r.metadata.get("article_no")) == following[1]
    ]
    next_captions = {_caption(r) for r in next_rows}
    if any(
        not _visible_native(paragraphs[i], styles)
        for i in range(index + 1, following[0])
    ):
        raise ValueError("The native own article region is hidden or ambiguous")
    between = [text for text in texts[index + 1 : following[0]] if text]
    simple_caption = (
        len(next_captions) == 1
        and between == list(next_captions)
        and bool(next(iter(next_captions)))
    )
    if simple_caption:
        if _native_interval_text(
            paragraph, paragraphs[following[0]], styles
        ) != normalized_text(" ".join(between)):
            raise ValueError(
                "The native whole article has unmodeled content outside paragraphs"
            )
    else:
        _next_section_transition(
            start=paragraph,
            end=paragraphs[following[0]],
            styles=styles,
            target=row,
            next_rows=next_rows,
            rows=rows,
        )
    return SelfRepealSourceProof(
        user_file_id=user_file_id,
        source_asset_id=source_asset_id,
        source_sha256=source_sha256,
        chunk_id=chunk_id,
        document_number=number,
        document_title=title,
        article_number=article,
        source_paragraph=index,
        canonical_scope_sha256=publication_digest(
            [
                r.model_dump(mode="json")
                for r in sorted(rows, key=lambda r: (r.position, r.id))
            ]
        ),
    )
