"""Source-proven chapter captions, separate from immutable canonical metadata."""

import hashlib
import re
from collections.abc import Sequence
from typing import Literal
from uuid import UUID

from bs4 import BeautifulSoup, NavigableString, Tag
from pydantic import BaseModel, ConfigDict, Field
from soupsieve import match

from onyx.document_index.publication_models import publication_digest
from onyx.regulatory.amendments.annexes.models import AnnexCanonicalSnapshot
from onyx.regulatory.amendments.annexes.sources import MAX_ASSET_BYTES
from onyx.utils.web_content import decode_html_bytes

_CHAPTER = re.compile(
    r"^(?:(?:ON|YİRMİ|OTUZ|KIRK|ELLİ|ALTMIŞ|YETMİŞ|SEKSEN|DOKSAN) )?"
    r"(?:BİRİNCİ|İKİNCİ|ÜÇÜNCÜ|DÖRDÜNCÜ|BEŞİNCİ|ALTINCI|YEDİNCİ|SEKİZİNCİ|"
    r"DOKUZUNCU|ONUNCU|YİRMİNCİ|OTUZUNCU|KIRKINCI|ELLİNCİ|ALTMIŞINCI|"
    r"YETMİŞİNCİ|SEKSENİNCİ|DOKSANINCI|YÜZÜNCÜ) BÖLÜM$"
)
_ARTICLE = re.compile(r"^MADDE\s+(\d+(?:/[A-Z])?)\s*[-–—:]", re.I)
_SELECTOR = re.compile(r"^(?:[a-zA-Z][\w-]*|\*)?(?:[.#][\w-]+)*$")
_PROPERTIES = {"font-weight", "text-align", "display", "visibility"}


def is_ordinal_chapter_heading(text: str) -> bool:
    return _CHAPTER.fullmatch(text) is not None


class NativeChapterReaderProof(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    version: Literal[1] = 1
    heading_chunk_id: str
    user_file_id: UUID
    source_asset_id: UUID
    source_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    chapter_label: str
    heading_text: str
    governed_chunk_ids: tuple[str, ...]
    governed_article_numbers: tuple[str, ...]
    scope_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


def _text(value: str) -> str:
    return " ".join(value.split())


def _scope_digest(rows: Sequence[AnnexCanonicalSnapshot]) -> str:
    return publication_digest(
        [
            row.model_dump(mode="json")
            for row in sorted(rows, key=lambda r: (r.position, r.id))
        ]
    )


def _atomic(row: AnnexCanonicalSnapshot) -> bool:
    return (
        row.status == "active"
        and row.metadata.get("chunk_variant") == "atomic"
        and row.superseded_by_chunk_id is None
    )


def native_chapter_reader_candidate(row: AnnexCanonicalSnapshot) -> bool:
    return (
        _atomic(row)
        and row.chunk_type == "free_text"
        and row.metadata.get("article_no") is None
        and len(row.heading_path) >= 2
        and _CHAPTER.fullmatch(row.heading_path[-1]) is not None
    )


def _chapter_children(
    rows: Sequence[AnnexCanonicalSnapshot],
    heading: AnnexCanonicalSnapshot,
    articles: tuple[str, ...],
) -> list[AnnexCanonicalSnapshot]:
    prefix = heading.heading_path
    family = [
        row
        for row in rows
        if _atomic(row) and row.heading_path[: len(prefix)] == prefix
    ]
    children = [
        row for row in family if str(row.metadata.get("article_no")) in articles
    ]
    end = max((row.position for row in children), default=heading.position)
    interlopers = [
        row
        for row in rows
        if _atomic(row)
        and heading.position < row.position <= end
        and row not in children
    ]
    if (
        not children
        or interlopers
        or any(
            row.position <= heading.position or not row.metadata.get("article_no")
            for row in children
        )
        or any(
            row.chunk_type == "free_text" and row.metadata.get("article_no") is None
            for row in children
        )
    ):
        return []
    return sorted(children, key=lambda r: (r.position, r.id))


def validate_native_chapter_reader_proof(
    proof: NativeChapterReaderProof,
    *,
    rows: Sequence[AnnexCanonicalSnapshot],
    expected_source_sha256: str,
    expected_source_asset_id: UUID | str,
    user_file_id: UUID | str,
) -> NativeChapterReaderProof:
    """Recheck the complete frozen canonical scope without reading raw HTML again."""
    by_id = {row.id: row for row in rows}
    if (
        len(by_id) != len(rows)
        or any(row.user_file_id != str(user_file_id) for row in rows)
        or proof.user_file_id != UUID(str(user_file_id))
        or proof.source_asset_id != UUID(str(expected_source_asset_id))
        or proof.source_sha256 != expected_source_sha256
        or proof.scope_sha256 != _scope_digest(rows)
    ):
        raise ValueError("Native chapter reader source scope changed")
    heading = by_id.get(proof.heading_chunk_id)
    if (
        heading is None
        or not _atomic(heading)
        or heading.chunk_type != "free_text"
        or heading.metadata.get("article_no") is not None
        or _text(heading.text) != proof.heading_text
        or len(heading.heading_path) < 2
        or heading.heading_path[-1] != proof.chapter_label
        or not _CHAPTER.fullmatch(proof.chapter_label)
    ):
        raise ValueError("Native chapter reader heading changed")
    children = _chapter_children(rows, heading, proof.governed_article_numbers)
    articles = tuple(dict.fromkeys(str(row.metadata["article_no"]) for row in children))
    if (
        not children
        or tuple(row.id for row in children) != proof.governed_chunk_ids
        or articles != proof.governed_article_numbers
    ):
        raise ValueError("Native chapter reader governed scope changed")
    return proof


def _declarations(text: str) -> dict[str, tuple[str, bool]]:
    result: dict[str, tuple[str, bool]] = {}
    for declaration in text.split(";"):
        key, separator, value = declaration.partition(":")
        if not separator:
            continue
        key, value = key.strip().lower(), value.strip().lower()
        if key in {"font", "all"}:
            raise ValueError("Native heading CSS reset or shorthand is ambiguous")
        if key in _PROPERTIES:
            important = value.endswith("!important")
            if key not in result or important or not result[key][1]:
                result[key] = (value.removesuffix("!important").strip(), important)
    return result


class _NativeStyles:
    """A conservative cascade for the inherited properties needed by this proof."""

    def __init__(self, soup: BeautifulSoup) -> None:
        self.rules: list[
            tuple[str, tuple[int, int, int, int], dict[str, tuple[str, bool]]]
        ] = []
        self.cache: dict[int, dict[str, str]] = {}
        if soup.find("link", rel="stylesheet"):
            raise ValueError("Native heading has external CSS")
        for style in soup.find_all("style"):
            css = (
                re.sub(r"/\*.*?\*/", "", style.get_text(), flags=re.S)
                .replace("<!--", "")
                .replace("-->", "")
            )
            if re.search(r"@(import|media|supports|layer)\b", css, re.I):
                raise ValueError("Native heading has conditional CSS")
            for selectors, body in re.findall(r"([^{}]+)\{([^{}]*)\}", css):
                declarations = _declarations(body)
                if not declarations:
                    continue
                for selector in selectors.split(","):
                    selector = selector.strip()
                    if not _SELECTOR.fullmatch(selector) or not selector:
                        raise ValueError("Native heading CSS selector is unsupported")
                    specificity = (
                        0,
                        selector.count("#"),
                        selector.count("."),
                        int(selector[0].isalpha()),
                    )
                    self.rules.append((selector, specificity, declarations))

    def computed(self, tag: Tag) -> dict[str, str]:
        if id(tag) in self.cache:
            return self.cache[id(tag)]
        parent = self.computed(tag.parent) if isinstance(tag.parent, Tag) else {}
        values = {
            "font-weight": parent.get("font-weight", "normal"),
            "text-align": parent.get("text-align", "start"),
            "visibility": parent.get("visibility", "visible"),
            "display": "inline",
        }
        if tag.name in {"b", "strong"}:
            values["font-weight"] = "bold"
        winners: dict[str, tuple[bool, tuple[int, int, int, int], int]] = {}
        declarations = [
            (spec, rule) for selector, spec, rule in self.rules if match(selector, tag)
        ]
        declarations.append(((1, 0, 0, 0), _declarations(str(tag.get("style", "")))))
        for order, (specificity, rule) in enumerate(declarations):
            for key, (value, important) in rule.items():
                priority = (important, specificity, order)
                if key not in winners or priority >= winners[key]:
                    winners[key] = priority
                    values[key] = (
                        parent.get(key, "normal" if key == "font-weight" else "start")
                        if value in {"inherit", "unset"}
                        else value
                    )
        self.cache[id(tag)] = values
        return values

    def centered_bold(self, paragraph: Tag) -> bool:
        if self.computed(paragraph)["text-align"] != "center":
            return False
        for node in paragraph.descendants:
            if not isinstance(node, NavigableString) or not node.strip():
                continue
            parent = node.parent
            if not isinstance(parent, Tag) or self.computed(parent)[
                "font-weight"
            ] not in {"bold", "700", "800", "900"}:
                return False
            for ancestor in (parent, *parent.parents):
                if isinstance(ancestor, Tag):
                    if ancestor.has_attr("hidden") or ancestor.name in {
                        "template",
                        "script",
                        "noscript",
                    }:
                        return False
                    style = self.computed(ancestor)
                    if style["display"] == "none" or style["visibility"] != "visible":
                        return False
        return True


def build_native_chapter_reader_proofs(
    *,
    raw_html: bytes,
    expected_sha256: str,
    user_file_id: UUID | str,
    source_asset_id: UUID | str,
    rows: Sequence[AnnexCanonicalSnapshot],
) -> dict[str, NativeChapterReaderProof]:
    """Promote only a standalone caption between its native chapter and articles."""
    if (
        len(raw_html) > MAX_ASSET_BYTES
        or hashlib.sha256(raw_html).hexdigest() != expected_sha256
    ):
        raise ValueError("Native chapter reader raw source changed")
    if any(row.user_file_id != str(user_file_id) for row in rows) or len(
        {row.id for row in rows}
    ) != len(rows):
        raise ValueError("Native chapter reader canonical file mismatch")
    soup = BeautifulSoup(decode_html_bytes(raw_html, "text/html"), "html.parser")
    try:
        styles = _NativeStyles(soup)
    except ValueError:
        return {}
    paragraphs = list(soup.find_all("p"))
    markers = [
        (i, _text(p.get_text(" ", strip=True)))
        for i, p in enumerate(paragraphs)
        if _CHAPTER.fullmatch(_text(p.get_text(" ", strip=True)))
    ]
    result: dict[str, NativeChapterReaderProof] = {}
    for marker_index, (start, label) in enumerate(markers):
        if sum(other == label for _, other in markers) != 1 or start + 1 >= len(
            paragraphs
        ):
            continue
        marker, caption = paragraphs[start], paragraphs[start + 1]
        sibling = marker.next_sibling
        while isinstance(sibling, NavigableString) and not sibling.strip():
            sibling = sibling.next_sibling
        try:
            visible_caption = styles.centered_bold(marker) and styles.centered_bold(
                caption
            )
        except ValueError:
            continue
        if sibling is not caption or not visible_caption:
            continue
        text = _text(caption.get_text(" ", strip=True))
        candidates = [
            row
            for row in rows
            if native_chapter_reader_candidate(row)
            and row.heading_path[-1] == label
            and _text(row.text) == text
        ]
        if not text or len(candidates) != 1:
            continue
        end = (
            markers[marker_index + 1][0]
            if marker_index + 1 < len(markers)
            else len(paragraphs)
        )
        articles = tuple(
            match.group(1)
            for p in paragraphs[start + 2 : end]
            if (match := _ARTICLE.match(_text(p.get_text(" ", strip=True))))
        )
        children = _chapter_children(rows, candidates[0], articles)
        if (
            not articles
            or len(set(articles)) != len(articles)
            or tuple(dict.fromkeys(str(row.metadata["article_no"]) for row in children))
            != articles
        ):
            continue
        proof = NativeChapterReaderProof(
            heading_chunk_id=candidates[0].id,
            user_file_id=UUID(str(user_file_id)),
            source_asset_id=UUID(str(source_asset_id)),
            source_sha256=expected_sha256,
            chapter_label=label,
            heading_text=text,
            governed_chunk_ids=tuple(row.id for row in children),
            governed_article_numbers=articles,
            scope_sha256=_scope_digest(rows),
        )
        result[proof.heading_chunk_id] = validate_native_chapter_reader_proof(
            proof,
            rows=rows,
            expected_source_sha256=expected_sha256,
            expected_source_asset_id=source_asset_id,
            user_file_id=user_file_id,
        )
    return result
