"""Read-only, authorized source classification for Legal Composite lanes."""

import json
import re
import unicodedata
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from contextvars import copy_context
from datetime import date
from enum import StrEnum
from hashlib import sha256
from typing import cast
from uuid import UUID

from pydantic import BaseModel, ConfigDict, JsonValue
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from onyx.context.search.models import IndexFilters
from onyx.db.asv3_corpus import (
    CorpusScopeUnavailable,
    CorpusSource,
    find_sources,
    read_source_chunks,
    require_source,
    resolve_source_query_index,
)
from onyx.db.engine.sql_engine import get_session_with_current_tenant
from onyx.db.models import RegulatoryChunk, User

SOURCE_PAGE_SIZE = 100
MAX_SOURCE_INVENTORY = 10_000
MAX_METADATA_TYPES_PER_PAGE = 1_600
MAX_OPENING_CHUNKS = 3
MAX_OPENING_CHARS = 4_096
MAX_OPENING_LINES = 12


class SourceKind(StrEnum):
    CONSTITUTION = "constitution"
    STATUTE = "statute"
    TREATY = "treaty"
    PRESIDENTIAL_DECREE = "presidential_decree"
    REGULATION = "regulation"
    COMMUNIQUE = "communique"
    CIRCULAR = "circular"
    JUDICIAL_DECISION = "judicial_decision"
    EXECUTIVE_DECISION = "executive_decision"
    PRIVATE_RULING = "private_ruling"
    OTHER = "other"
    UNKNOWN = "unknown"


class SourceClassification(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    source_id: UUID
    name: str
    kind: SourceKind
    method: str
    uncertain: bool
    observed_document_types: tuple[str, ...]
    opening_identity_sha256: str | None = None
    original_kind: SourceKind | None = None

    def admits(self, kind: SourceKind) -> bool:
        return self.kind == kind or (kind == SourceKind.UNKNOWN and self.uncertain)


class SourceLaneCatalogue(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    user_id: UUID
    scope_sha256: str
    records: tuple[SourceClassification, ...]
    complete: bool
    limitations: tuple[str, ...] = ()

    def source_ids(self, kind: SourceKind) -> tuple[UUID, ...]:
        return tuple(row.source_id for row in self.records if row.admits(kind))

    def provenance(self) -> dict[str, JsonValue]:
        return {
            "scope_sha256": self.scope_sha256,
            "inventory_complete": self.complete,
            "source_count": len(self.records),
            "uncertain_source_count": sum(row.uncertain for row in self.records),
            "limitations": list(self.limitations),
        }

    def kinds_by_source_id(self) -> dict[str, str]:
        return {str(row.source_id): row.kind.value for row in self.records}


def source_scope_sha256(user: User, filters: IndexFilters) -> str:
    body = {"user_id": str(user.id), "filters": filters.model_dump(mode="json")}
    return sha256(
        json.dumps(
            body, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode()
    ).hexdigest()


def _fold(value: str) -> str:
    replaced = value.casefold().replace("ı", "i")
    return "".join(
        character
        for character in unicodedata.normalize("NFKD", replaced)
        if not unicodedata.combining(character)
    )


def _name_kind(name: str) -> SourceKind | None:
    basename = re.split(r"[/\\]", _fold(name))[-1]
    tokens = " " + re.sub(r"[^a-z0-9]+", " ", basename) + " "
    for kind, markers in (
        (SourceKind.PRIVATE_RULING, ("ozelge", "mukteza", "private ruling")),
        (
            SourceKind.PRESIDENTIAL_DECREE,
            (
                "cumhurbaskanligi kararnamesi",
                "cumhurbaskanligi kararname",
                "presidential decree",
            ),
        ),
        (
            SourceKind.JUDICIAL_DECISION,
            (
                "anayasa mahkemesi",
                "danistay",
                "yargitay",
                "mahkeme",
                "judgment",
                "court",
            ),
        ),
        (SourceKind.CONSTITUTION, ("anayasa", "constitution")),
        (
            SourceKind.EXECUTIVE_DECISION,
            ("cumhurbaskani karari", "bakanlar kurulu", "bkk", "executive decision"),
        ),
        (
            SourceKind.TREATY,
            (
                "sozlesme",
                "sozlesmesi",
                "anlasma",
                "anlasmasi",
                "andlasma",
                "andlasmasi",
                "treaty",
            ),
        ),
        (SourceKind.REGULATION, ("yonetmelik", "yonetmeligi", "regulation")),
        (SourceKind.COMMUNIQUE, ("teblig", "tebligi", "communique")),
        (SourceKind.CIRCULAR, ("genelge", "genelgesi", "circular")),
        (SourceKind.STATUTE, ("kanun", "kanunu", "statute")),
        (SourceKind.OTHER, ("yonerge", "protokol")),
    ):
        if any(" " + marker + " " in tokens for marker in markers):
            return kind
    return None


_METADATA_KINDS = {
    "anayasa": SourceKind.CONSTITUTION,
    "constitution": SourceKind.CONSTITUTION,
    "kanun": SourceKind.STATUTE,
    "statute": SourceKind.STATUTE,
    "sozlesme": SourceKind.TREATY,
    "andlasma": SourceKind.TREATY,
    "anlasma": SourceKind.TREATY,
    "treaty": SourceKind.TREATY,
    "cumhurbaskanligi_kararnamesi": SourceKind.PRESIDENTIAL_DECREE,
    "presidential_decree": SourceKind.PRESIDENTIAL_DECREE,
    "yonetmelik": SourceKind.REGULATION,
    "regulation": SourceKind.REGULATION,
    "teblig": SourceKind.COMMUNIQUE,
    "communique": SourceKind.COMMUNIQUE,
    "genelge": SourceKind.CIRCULAR,
    "circular": SourceKind.CIRCULAR,
    "yargi_karari": SourceKind.JUDICIAL_DECISION,
    "judicial_decision": SourceKind.JUDICIAL_DECISION,
    "idari_karar": SourceKind.EXECUTIVE_DECISION,
    "executive_decision": SourceKind.EXECUTIVE_DECISION,
    "ozelge": SourceKind.PRIVATE_RULING,
    "mukteza": SourceKind.PRIVATE_RULING,
    "private_ruling": SourceKind.PRIVATE_RULING,
    "yonerge": SourceKind.OTHER,
    "protokol": SourceKind.OTHER,
}


def classify_source(
    source: CorpusSource,
    document_types: tuple[str, ...],
    *,
    metadata_complete: bool = True,
    opening_texts: tuple[str, ...] = (),
) -> SourceClassification:
    observed = tuple(
        sorted({_fold(value).strip() for value in document_types if value.strip()})
    )
    name_kind = _name_kind(source.name)
    effective = {value for value in observed if value not in {"", "unknown"}}
    kinds = {_METADATA_KINDS[value] for value in effective if value in _METADATA_KINDS}
    generic_decision = effective == {"karar"}
    original_kind, identity = _opening_identity(opening_texts)
    if original_kind is None:
        kind, method, uncertain = (
            SourceKind.UNKNOWN,
            "original_identity_unverified",
            True,
        )
    else:
        kind, method = original_kind, "original_opening_identity"
        compatible_decision = generic_decision and kind in {
            SourceKind.JUDICIAL_DECISION,
            SourceKind.EXECUTIVE_DECISION,
            SourceKind.PRESIDENTIAL_DECREE,
        }
        uncertain = (
            not metadata_complete
            or (name_kind is not None and name_kind != kind)
            or (bool(effective) and kinds != {kind} and not compatible_decision)
        )
    return SourceClassification(
        source_id=source.id,
        name=source.name,
        kind=kind,
        method=method,
        uncertain=uncertain,
        observed_document_types=observed,
        opening_identity_sha256=identity,
        original_kind=original_kind,
    )


def _heading_kind(line: str) -> SourceKind | None:
    folded = _fold(line).strip()
    if folded.endswith((".", ";", ":", "?")):
        return None
    if re.match(r"(?:bu|ilgili|anilan|dayanak|ilgi|konu|referans|bakiniz)\b", folded):
        return None
    folded = re.sub(r"\s*\([^()]*\)\s*$", "", folded)
    folded = re.sub(r"\s*[-:]?\s*\d+(?:[/.-]\d+)*\s*$", "", folded)
    tokens = re.sub(r"[^a-z0-9]+", " ", folded).strip()
    if re.search(
        r"\b(?:uyarinca|geregince|hukmu|hukumleri|kapsaminda|dayanilarak|dayanak|referans|bakiniz)\b",
        tokens,
    ):
        return None
    if re.search(
        r"(?:mahkemesi|mahkeme|danistay|yargitay|court)\b", tokens
    ) and re.search(r"\b(?:karar|karari|judgment)\b", tokens):
        return SourceKind.JUDICIAL_DECISION
    if re.search(r"\bcumhurbaskanligi\b", tokens) and re.search(
        r"\bkararname(?:si)?$", tokens
    ):
        return SourceKind.PRESIDENTIAL_DECREE
    if re.search(r"\b(?:cumhurbaskani|bakanlar kurulu)\b", tokens) and re.search(
        r"\bkarar(?:i)?$", tokens
    ):
        return SourceKind.EXECUTIVE_DECISION
    for kind, suffix in (
        (SourceKind.CONSTITUTION, r"anayasa(?:si)?|constitution"),
        (SourceKind.REGULATION, r"yonetmelik|yonetmeligi|regulation"),
        (SourceKind.COMMUNIQUE, r"teblig|tebligi|communique"),
        (SourceKind.CIRCULAR, r"genelge|genelgesi|circular"),
        (SourceKind.TREATY, r"sozlesme(?:si)?|andlasma(?:si)?|anlasma(?:si)?|treaty"),
        (SourceKind.PRIVATE_RULING, r"ozelge|mukteza|private ruling"),
        (SourceKind.STATUTE, r"kanun|kanunu|statute"),
        (SourceKind.OTHER, r"yonerge|protokol"),
    ):
        if re.search(r"\b(?:" + suffix + r")$", tokens):
            return kind
    return None


def _opening_identity(texts: tuple[str, ...]) -> tuple[SourceKind | None, str | None]:
    lines = "\n".join(texts)[:MAX_OPENING_CHARS].splitlines()
    identified: list[tuple[SourceKind, str]] = []
    meaningful = 0
    for raw in lines:
        line = raw.strip().strip("#*_| []")
        if not line:
            continue
        folded = _fold(line)
        if re.match(r"(?:madde|gecici madde|mukerrer madde)\b|\(?\d+[.)]\s", folded):
            break
        meaningful += 1
        kind = _heading_kind(line)
        if kind is not None:
            identified.append((kind, line))
        if meaningful >= MAX_OPENING_LINES:
            break
    kinds = {kind for kind, _line in identified}
    if len(kinds) != 1:
        return None, None
    body = json.dumps(
        [(kind.value, line) for kind, line in identified],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return next(iter(kinds)), sha256(body.encode()).hexdigest()


def _opening_texts(
    session: Session,
    *,
    user: User,
    filters: IndexFilters,
    source: CorpusSource,
    check_active: Callable[[], None],
) -> tuple[str, ...]:
    check_active()
    index = resolve_source_query_index(session, source.id)
    _source, chunks, _more = read_source_chunks(
        session,
        user=user,
        filters=filters,
        source_id=source.id,
        start=0,
        limit=MAX_OPENING_CHUNKS,
        query_indexes={source.id: index} if index is not None else {},
    )
    remaining = MAX_OPENING_CHARS
    result: list[str] = []
    for chunk in chunks:
        check_active()
        if remaining <= 0:
            break
        text = chunk.text[:remaining]
        result.append(text)
        remaining -= len(text)
    return tuple(result)


def _document_types(
    session: Session,
    source_ids: tuple[UUID, ...],
    filters: IndexFilters,
    check_active: Callable[[], None],
) -> tuple[dict[UUID, tuple[str, ...]], bool]:
    if not source_ids:
        return {}, True
    statement = select(
        RegulatoryChunk.user_file_id,
        RegulatoryChunk.chunk_metadata["document_type"].astext,
    ).where(
        RegulatoryChunk.user_file_id.in_(source_ids),
        RegulatoryChunk.chunk_type.is_distinct_from("hierarchical_aggregate"),
    )
    effective_date = filters.as_of_date or date.today()
    if filters.as_of_date is None:
        statement = statement.where(RegulatoryChunk.status == "active")
    else:
        statement = statement.where(
            or_(
                RegulatoryChunk.validity_start_date.is_(None),
                RegulatoryChunk.validity_start_date <= effective_date,
            ),
            or_(
                RegulatoryChunk.validity_end_date.is_(None),
                RegulatoryChunk.validity_end_date > effective_date,
            ),
        )
    rows = session.execute(
        statement.distinct().limit(MAX_METADATA_TYPES_PER_PAGE + 1)
    ).all()
    found: dict[UUID, list[str]] = {}
    for source_id, value in rows[:MAX_METADATA_TYPES_PER_PAGE]:
        check_active()
        if isinstance(value, str):
            found.setdefault(source_id, []).append(value)
    return {key: tuple(value) for key, value in found.items()}, len(
        rows
    ) <= MAX_METADATA_TYPES_PER_PAGE


def _source_openings(
    session: Session,
    source_page: list[CorpusSource],
    *,
    user: User,
    filters: IndexFilters,
    check_active: Callable[[], None],
    opening_workers: int,
) -> list[tuple[str, ...] | None]:
    def read(source: CorpusSource) -> tuple[str, ...] | None:
        try:
            if opening_workers == 1:
                return _opening_texts(
                    session,
                    user=user,
                    filters=filters,
                    source=source,
                    check_active=check_active,
                )
            # SQLAlchemy sessions never cross concurrent worker boundaries.
            with get_session_with_current_tenant() as opening_session:
                return _opening_texts(
                    opening_session,
                    user=user,
                    filters=filters,
                    source=source,
                    check_active=check_active,
                )
        except CorpusScopeUnavailable:
            return None

    if opening_workers == 1:
        return [read(source) for source in source_page]
    executor = ThreadPoolExecutor(max_workers=opening_workers)
    try:
        futures = [
            cast(
                Future[tuple[str, ...] | None],
                executor.submit(copy_context().run, read, source),
            )
            for source in source_page
        ]
        result: list[tuple[str, ...] | None] = []
        for future in futures:
            while True:
                check_active()
                try:
                    result.append(future.result(timeout=0.05))
                    break
                except FutureTimeout:
                    continue
        return result
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def load_source_lane_catalogue(
    session: Session,
    *,
    user: User,
    filters: IndexFilters,
    check_active: Callable[[], None],
    max_sources: int = MAX_SOURCE_INVENTORY,
    opening_workers: int = 1,
) -> SourceLaneCatalogue:
    if max_sources < 1:
        raise ValueError("Source inventory must have a positive bound")
    if (
        not isinstance(opening_workers, int)
        or isinstance(opening_workers, bool)
        or not 1 <= opening_workers <= 4
    ):
        raise ValueError("Opening workers must be an integer between one and four")
    records: list[SourceClassification] = []
    limitations: list[str] = []
    offset = 0
    more = True
    while more and offset < max_sources:
        check_active()
        page_limit = min(SOURCE_PAGE_SIZE, max_sources - offset)
        sources, more = find_sources(
            session, user=user, filters=filters, offset=offset, limit=page_limit
        )
        types, metadata_complete = _document_types(
            session, tuple(source.id for source in sources), filters, check_active
        )
        if (
            not metadata_complete
            and "Metadata type inventory was partial." not in limitations
        ):
            limitations.append("Metadata type inventory was partial.")
        openings_page = _source_openings(
            session,
            sources,
            user=user,
            filters=filters,
            check_active=check_active,
            opening_workers=opening_workers,
        )
        for source, openings in zip(sources, openings_page, strict=True):
            if openings is None:
                if (
                    "Some original opening identities were unavailable."
                    not in limitations
                ):
                    limitations.append(
                        "Some original opening identities were unavailable."
                    )
            records.append(
                classify_source(
                    source,
                    types.get(source.id, ()),
                    metadata_complete=metadata_complete,
                    opening_texts=openings or (),
                )
            )
        offset += page_limit
    if more:
        limitations.append(
            "Authorized source inventory exceeded the bounded page budget; omitted sources remain unsearched."
        )
    check_active()
    return SourceLaneCatalogue(
        user_id=user.id,
        scope_sha256=source_scope_sha256(user, filters),
        records=tuple(records),
        complete=not more and not limitations,
        limitations=tuple(limitations),
    )


def revalidate_source_classification(
    session: Session,
    *,
    user: User,
    filters: IndexFilters,
    recorded: SourceClassification,
    check_active: Callable[[], None],
) -> CorpusSource:
    check_active()
    source = require_source(
        session, user=user, filters=filters, source_id=recorded.source_id
    )
    types, complete = _document_types(session, (source.id,), filters, check_active)
    current = classify_source(
        source,
        types.get(source.id, ()),
        metadata_complete=complete,
        opening_texts=_opening_texts(
            session,
            user=user,
            filters=filters,
            source=source,
            check_active=check_active,
        ),
    )
    if (
        current.kind != recorded.kind
        or current.opening_identity_sha256 != recorded.opening_identity_sha256
    ):
        raise CorpusScopeUnavailable(
            "Source classification changed after the lane inventory."
        )
    check_active()
    return source
