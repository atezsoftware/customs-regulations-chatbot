"""Read-only legal identity for document-level predecessor suggestions."""

import re
import unicodedata
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal
from uuid import UUID

from sqlalchemy import ColumnElement, func, or_, select
from sqlalchemy.orm import Session

from onyx.db.models import RegulatoryChunk, UserFile

_TRANSLATION = str.maketrans("ıİşŞğĞüÜöÖçÇ", "iIsSgGuUoOcC")
_SUFFIX = re.compile(r"\.(?:md|pdf|html?|docx?|txt)$", re.IGNORECASE)
_TECHNICAL_NAME = re.compile(
    r"https?://|www\.|(?:[a-z0-9-]+\.)+(?:gov|com|org|net|edu|tr)(?:[./_\s]|$)",
    re.IGNORECASE,
)
_LEGAL_NUMBER = re.compile(r"\b([a-z]{2,12})\s*(?:[-:]\s*|\s+)(\d{4})\s*/\s*(\d+)\b")
_NUMBERED_LAW = re.compile(r"\b(\d{1,6})\s+sayili\b")
_REPEALED = re.compile(r"\byururlukten\s+kaldirilmistir\b")
_STOP_WORDS = {"sayili", "sayisi", "say", "no", "numarali", "tarihli"}


def normalized_legal_text(value: str, *, retain_numbers: bool = False) -> str:
    value = unicodedata.normalize("NFKD", value.translate(_TRANSLATION))
    value = "".join(char for char in value if not unicodedata.combining(char)).lower()
    return " ".join(
        token
        for token in re.findall(r"[a-z0-9]+" if retain_numbers else r"[a-z]+", value)
        if token not in _STOP_WORDS
    )


def _plain_text(value: str) -> str:
    return "".join(
        char
        for char in unicodedata.normalize("NFKD", value.translate(_TRANSLATION))
        if not unicodedata.combining(char)
    ).lower()


def _meaningful_title(value: str) -> str | None:
    value = value.strip()
    if not value or _TECHNICAL_NAME.search(value):
        return None
    while _SUFFIX.search(value):
        value = _SUFFIX.sub("", value)
    tokens = normalized_legal_text(value).split()
    if len(tokens) < 2 or set(tokens) <= {"ek", "madde", "belge", "dosya", "download"}:
        return None
    return value


def legal_number_keys(value: str) -> frozenset[str]:
    plain = _plain_text(value)
    keys = {
        f"{family}:{int(year)}/{int(number)}"
        for family, year, number in _LEGAL_NUMBER.findall(plain)
    }
    keys.update(f"number:{int(number)}" for number in _NUMBERED_LAW.findall(plain))
    if not keys and re.fullmatch(r"\s*\d{1,6}\s*", plain):
        keys.add(f"number:{int(plain)}")
    return frozenset(keys)


@dataclass(frozen=True)
class DocumentLegalIdentity:
    title: str | None
    number_keys: frozenset[str]
    title_source: Literal["metadata", "heading", "filename"]


def _current_field(value: Any, audit: Any, input_stale: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    if isinstance(audit, dict):
        if audit.get("stale") or (input_stale and audit.get("source") != "manual"):
            return None
    elif input_stale:
        return None
    return value.strip()


def _unique(values: Sequence[str]) -> str | None:
    distinct: dict[str, str] = {}
    for value in values:
        key = normalized_legal_text(value, retain_numbers=True)
        if key:
            distinct.setdefault(key, value)
    return next(iter(distinct.values())) if len(distinct) == 1 else None


def document_legal_identities(
    session: Session, files: Sequence[UserFile]
) -> dict[UUID, DocumentLegalIdentity]:
    """Read identity fields together; aggregates and stale inference are not evidence."""
    if not files:
        return {}
    rows: dict[UUID, list[Any]] = defaultdict(list)
    metadata = RegulatoryChunk.chunk_metadata
    audit = metadata["intake_legal_metadata"]
    for row in session.execute(
        select(
            RegulatoryChunk.user_file_id,
            metadata["document_title"].astext.label("title"),
            metadata["document_number"].astext.label("number"),
            audit["fields"]["document_title"].label("title_audit"),
            audit["fields"]["document_number"].label("number_audit"),
            audit["input_stale"].label("input_stale"),
            RegulatoryChunk.heading_path[0].astext.label("heading"),
        ).where(
            RegulatoryChunk.user_file_id.in_([file.id for file in files]),
            RegulatoryChunk.status == "active",
            RegulatoryChunk.superseded_by_chunk_id.is_(None),
            RegulatoryChunk.chunk_type.is_distinct_from("hierarchical_aggregate"),
        )
    ):
        rows[row.user_file_id].append(row)
    result: dict[UUID, DocumentLegalIdentity] = {}
    for file in files:
        titles, numbers, headings = [], [], []
        for row in rows[file.id]:
            title = _current_field(row.title, row.title_audit, row.input_stale)
            number = _current_field(row.number, row.number_audit, row.input_stale)
            meaningful = _meaningful_title(title) if title else None
            if meaningful:
                titles.append(meaningful)
            if number:
                numbers.append(number)
            heading = (
                _meaningful_title(row.heading) if isinstance(row.heading, str) else None
            )
            if heading:
                headings.append(heading)
        title = _unique(titles)
        source: Literal["metadata", "heading", "filename"] = "metadata"
        if title is None:
            title, source = _unique(headings), "heading"
        if title is None:
            title, source = _meaningful_title(file.name), "filename"
        number_values = {legal_number_keys(number) for number in numbers}
        number_values.discard(frozenset())
        keys = next(iter(number_values)) if len(number_values) == 1 else frozenset()
        if len(keys) != 1:
            keys = frozenset()
        if not numbers:
            title_keys = legal_number_keys(
                title or (file.name if not _TECHNICAL_NAME.search(file.name) else "")
            )
            keys = title_keys if len(title_keys) == 1 else frozenset()
        result[file.id] = DocumentLegalIdentity(title, keys, source)
    return result


def explicit_repeal_targets(session: Session, file_id: UUID) -> frozenset[str]:
    """Only completed repeal wording identifies a target; no date or relation is inferred."""
    groups: dict[str, list[str]] = defaultdict(list)
    for row in session.execute(
        select(
            RegulatoryChunk.text,
            RegulatoryChunk.heading_path,
            RegulatoryChunk.chunk_metadata["article_no"].astext,
        )
        .where(
            RegulatoryChunk.user_file_id == file_id,
            RegulatoryChunk.status == "active",
            RegulatoryChunk.superseded_by_chunk_id.is_(None),
            RegulatoryChunk.chunk_type.is_distinct_from("hierarchical_aggregate"),
        )
        .order_by(RegulatoryChunk.position, RegulatoryChunk.id)
    ):
        text, headings, article = row
        heading = next(
            (
                part
                for part in reversed(headings)
                if re.search(r"\bMADDE\s+\d+", part, re.IGNORECASE)
            ),
            None,
        )
        # A split provision may span chunks, but never join unrelated articles.
        key = str(article or heading or len(groups))
        groups[key].append(text)
    targets: set[str] = set()
    for parts in groups.values():
        plain = _plain_text("\n".join(parts))
        for repeal in _REPEALED.finditer(plain):
            preceding = plain[max(0, repeal.start() - 1200) : repeal.start()]
            prefixed = list(_LEGAL_NUMBER.finditer(preceding))
            if prefixed:
                match = prefixed[-1]
                targets.update(legal_number_keys(match.group()))
            else:
                numbered = list(_NUMBERED_LAW.finditer(preceding))
                if numbered:
                    targets.update(legal_number_keys(numbered[-1].group()))
    return frozenset(targets)


def identity_search_condition(tokens: Sequence[str]) -> ColumnElement[bool]:
    """SQL prefilter uses only own identity fields; final matching stays in Python."""
    metadata = RegulatoryChunk.chunk_metadata
    texts = (
        metadata["document_title"].astext,
        metadata["document_number"].astext,
        RegulatoryChunk.heading_path[0].astext,
    )
    conditions = [
        func.lower(func.translate(text, "ıİşŞğĞüÜöÖçÇ", "iIsSgGuUoOcC")).contains(
            token, autoescape=True
        )
        for text in texts
        for token in tokens
    ]
    return or_(
        *(
            func.lower(
                func.translate(UserFile.name, "ıİşŞğĞüÜöÖçÇ", "iIsSgGuUoOcC")
            ).contains(token, autoescape=True)
            for token in tokens
        ),
        _matching_chunks(or_(*conditions)),
    )


def _matching_chunks(condition: ColumnElement[bool]) -> ColumnElement[bool]:
    return (
        select(RegulatoryChunk.id)
        .where(
            RegulatoryChunk.user_file_id == UserFile.id,
            RegulatoryChunk.status == "active",
            RegulatoryChunk.superseded_by_chunk_id.is_(None),
            RegulatoryChunk.chunk_type.is_distinct_from("hierarchical_aggregate"),
            condition,
        )
        .correlate(UserFile)
        .exists()
    )


def exact_identity_search_condition(keys: frozenset[str]) -> ColumnElement[bool]:
    def matches(value: Any) -> ColumnElement[bool]:
        return or_(
            *(
                func.regexp_replace(
                    func.lower(func.translate(value, "ıİşŞğĞüÜöÖçÇ", "iIsSgGuUoOcC")),
                    "[^a-z0-9]",
                    "",
                    "g",
                ).contains(
                    (key.removeprefix("number:") if key.startswith("number:") else key)
                    .replace(":", "")
                    .replace("/", ""),
                    autoescape=True,
                )
                for key in keys
            )
        )

    return or_(
        matches(UserFile.name),
        _matching_chunks(
            or_(
                matches(RegulatoryChunk.chunk_metadata["document_number"].astext),
                matches(RegulatoryChunk.chunk_metadata["document_title"].astext),
            )
        ),
    )
