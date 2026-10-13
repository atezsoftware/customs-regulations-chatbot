"""Canonical text witnesses for Cropped navigation, never legal acceptance."""

import re
from dataclasses import dataclass, field
from typing import Literal, cast

from pydantic import JsonValue

from onyx.asv3.cheap_authority import (
    _AMENDING_PASSAGE,
    _OWN_ARTICLE_OPENING,
    _own_article_witness,
    folded,
)
from onyx.asv3.models import EvidenceItem
from onyx.db.asv3_corpus import CorpusChunk
from onyx.regulatory.heading_path import parse_regulatory_article_heading


def verified_own_article_identity(
    chunk: CorpusChunk,
) -> tuple[str, str | None] | None:
    if any(
        chunk.metadata.get(flag)
        for flag in ("derived", "external", "untrusted", "truncated")
    ):
        return None
    previous = chunk.metadata.get("article_no")
    qualifier = chunk.metadata.get("article_qualifier")
    for heading in reversed(chunk.heading_path):
        parsed = parse_regulatory_article_heading(heading)
        if parsed is not None:
            previous, qualifier = parsed.article_no, parsed.qualifier
            break
    witness = _own_article_witness(
        0,
        EvidenceItem(
            source_id=str(chunk.source_id),
            chunk_id=chunk.id,
            text=chunk.text,
            metadata={"canonical_metadata": cast(dict[str, JsonValue], chunk.metadata)},
        ),
        {
            "article_no": previous,
            "article_qualifier": qualifier,
            "clause_label": chunk.metadata.get("clause_label"),
        },
    )
    return (witness.article_no, witness.qualifier) if witness is not None else None


_INSERTED_NUMBER = re.compile(r"(\d{1,4})/[A-ZÇĞİÖŞÜ]")
_PARAGRAPH_OPENING = re.compile(r"^(?:\((\d{1,3})\)|(\d{1,3})\.)\s+\S")


def _canonical_navigation_chunk(chunk: CorpusChunk) -> bool:
    return (
        bool(chunk.id and chunk.heading_path)
        and folded(str(chunk.metadata.get("document_type") or ""))
        in {"kanun", "law", "act", "statute", "yonetmelik", "teblig"}
        and not any(
            chunk.metadata.get(flag)
            for flag in ("derived", "external", "untrusted", "truncated")
        )
        and all(
            chunk.metadata.get(field) is None
            for field in (
                "target_article_no",
                "target_id",
                "asv3_navigation_position_authority",
            )
        )
        and not _AMENDING_PASSAGE.search(folded(chunk.text))
    )


def _structural_identity(chunk: CorpusChunk) -> tuple[str, str | None] | None:
    for heading in reversed(chunk.heading_path):
        parsed = parse_regulatory_article_heading(heading)
        if parsed is not None:
            return parsed.article_no, parsed.qualifier
    return None


def _literal_opening(chunk: CorpusChunk) -> tuple[str, str | None, int] | None:
    if not _canonical_navigation_chunk(chunk):
        return None
    lines = chunk.text.lstrip().splitlines()
    if not lines:
        return None
    opening = _OWN_ARTICLE_OPENING.match(lines[0])
    if opening is None:
        return None
    heading = parse_regulatory_article_heading(opening["heading"].replace("**", ""))
    if heading is None or any(
        parse_regulatory_article_heading(line.replace("**", "").replace("__", ""))
        for line in lines[1:]
    ):
        return None
    remainder = lines[0][opening.end("heading") :].lstrip(" .-–—:\t\xa0")
    paragraph = _PARAGRAPH_OPENING.match(remainder)
    if paragraph is None:
        return None
    original_path = chunk.metadata.get("original_heading_path")
    if isinstance(original_path, list) and any(
        isinstance(part, str)
        and (parsed := parse_regulatory_article_heading(part)) is not None
        and (parsed.article_no, parsed.qualifier)
        != (heading.article_no, heading.qualifier)
        for part in original_path
    ):
        return None
    return heading.article_no, heading.qualifier, int(paragraph[1] or paragraph[2])


def verified_inserted_article_opening(
    chunk: CorpusChunk, target: tuple[str, str | None]
) -> bool:
    """Identify an inserted opening; continuation must be proved separately."""
    inserted = _INSERTED_NUMBER.fullmatch(target[0])
    if inserted is None or target[1] is not None:
        return False
    previous = _structural_identity(chunk)
    return (
        _literal_opening(chunk) == (*target, 1)
        and previous in {target, (inserted[1], None)}
        and chunk.metadata.get("article_no") in {target[0], inserted[1]}
        and chunk.metadata.get("article_qualifier") is None
        and chunk.metadata.get("clause_label") is None
    )


def _continuation_root(chunk: CorpusChunk) -> tuple[str, ...]:
    path = chunk.heading_path
    return path[:-1] if _PARAGRAPH_OPENING.match(path[-1]) else path


@dataclass
class InsertedArticleRead:
    """A narrowly proved literal paragraph sequence under a stale base heading."""

    first: CorpusChunk
    target: tuple[str, str | None]
    last_position: int = field(init=False)
    paragraph: int = 1
    complete: bool = False

    def __post_init__(self) -> None:
        if not verified_inserted_article_opening(self.first, self.target):
            raise ValueError("An exact canonical inserted-article opening is required.")
        self.last_position = self.first.position

    def consume(
        self, chunk: CorpusChunk
    ) -> Literal["continuation", "boundary", "unverified"]:
        if (
            chunk.source_id != self.first.source_id
            or chunk.position != self.last_position + 1
            or not _canonical_navigation_chunk(chunk)
        ):
            return "unverified"
        opening = _literal_opening(chunk)
        if opening is not None:
            first_article_index = next(
                index
                for index, heading in enumerate(self.first.heading_path)
                if parse_regulatory_article_heading(heading) is not None
            )
            own = opening[:2]
            if (
                own != self.target
                and own == _structural_identity(chunk)
                and chunk.metadata.get("article_no") == own[0]
                and chunk.metadata.get("article_qualifier") == own[1]
                and chunk.heading_path[:first_article_index]
                == self.first.heading_path[:first_article_index]
            ):
                self.complete = True
                return "boundary"
            return "unverified"
        lines = chunk.text.lstrip().replace("**", "").splitlines()
        marker = _PARAGRAPH_OPENING.match(lines[0]) if lines else None
        number = int(marker[1] or marker[2]) if marker else None
        if (
            number is None
            or number != self.paragraph + 1
            or _continuation_root(chunk) != _continuation_root(self.first)
            or _structural_identity(chunk) != _structural_identity(self.first)
            or chunk.metadata.get("article_no") != self.first.metadata.get("article_no")
            or chunk.metadata.get("article_qualifier") is not None
            or chunk.metadata.get("clause_label") is not None
            or chunk.metadata.get("paragraph_no") not in {None, str(number), number}
            or any(
                _PARAGRAPH_OPENING.match(line) or parse_regulatory_article_heading(line)
                for line in lines[1:]
            )
        ):
            return "unverified"
        self.paragraph = number
        self.last_position = chunk.position
        return "continuation"
