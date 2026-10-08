"""Read-only, authorized source classification for Legal Composite lanes."""

import json
import re
import time
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
from sqlalchemy import Integer, and_, func, or_, select
from sqlalchemy import cast as sql_cast
from sqlalchemy.orm import Session
from sqlalchemy.sql.selectable import Subquery

from onyx.access.access import get_access_for_user_files, get_acl_for_user
from onyx.context.search.models import IndexFilters
from onyx.db.asv3_corpus import (
    CorpusScopeUnavailable,
    CorpusSource,
    _source_statement,
    _validate_filters,
    read_source_chunks,
    require_source,
    resolve_source_query_index,
)
from onyx.db.engine.sql_engine import get_session_with_current_tenant
from onyx.db.models import (
    RegulatoryChunk,
    RegulatoryTemporalProjection,
    SearchSettings,
    User,
    UserFile,
)
from onyx.db.regulatory_canonical_revisions import (
    get_canonical_revisions,
    validate_temporal_canonical_revisions,
)
from onyx.db.regulatory_public_reads import qualified_file_ids
from onyx.db.search_settings import get_current_search_settings
from onyx.document_index.encoder_authority import effective_runtime_authority
from onyx.document_index.publication_models import (
    PublicationEncoderAuthority,
    PublicationIndexSnapshot,
    accepts_publication_projection,
    publication_digest,
)
from onyx.regulatory.amendments.annexes.context_dependencies import context_hash
from onyx.regulatory.amendments.annexes.models import AnnexTemporalProjection
from onyx.regulatory.publication_baseline import observed_index_snapshot
from onyx.regulatory.publication_reads import (
    filter_publication_read,
    observe_publication_read,
    require_publication_files,
)
from shared_configs.configs import MULTI_TENANT

SOURCE_PAGE_SIZE = 1_000
MAX_OPENING_BATCH_SOURCES = 100
MAX_SOURCE_INVENTORY = 10_000
MAX_METADATA_TYPES_PER_PAGE = 16 * SOURCE_PAGE_SIZE
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


def find_source_inventory_page(
    session: Session,
    *,
    user: User,
    filters: IndexFilters,
    source_ids: tuple[UUID, ...] | None = None,
    offset: int = 0,
    limit: int = SOURCE_PAGE_SIZE,
) -> tuple[list[CorpusSource], bool]:
    """Page authorized inventory using canonical scope, ACL and publication guards."""
    if (
        isinstance(limit, bool)
        or not isinstance(limit, int)
        or not 1 <= limit <= SOURCE_PAGE_SIZE
        or offset < 0
    ):
        raise ValueError("Inventory pages require offset >= 0 and limit 1..1000.")
    _validate_filters(session, user, filters)
    statement = _source_statement(filters)
    if source_ids is not None:
        statement = statement.where(UserFile.id.in_(source_ids))
    records = session.execute(
        statement.order_by(UserFile.id).offset(offset).limit(limit + 1)
    ).all()
    access = get_access_for_user_files([str(row.id) for row in records], session)
    user_acl = get_acl_for_user(user, session)
    candidates = [
        CorpusSource(row.id, row.name, row.file_id)
        for row in records[:limit]
        if row.id and str(row.id) in access and access[str(row.id)].to_acl() & user_acl
    ]
    retained = filter_publication_read(
        observe_publication_read(), candidates, lambda row: str(row.id)
    )
    return retained, len(records) > limit


class SourceOpeningWitness(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk_id: str
    text_sha256: str
    binding_id: UUID | None = None
    index_uuid: str | None = None
    payload_sha256: str | None = None
    canonical_revision_id: UUID | None = None
    revision_sha256: str | None = None
    canonical_base_sha256: str | None = None
    effective_start: date | None = None
    effective_end: date | None = None


class RoutingOpening(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    texts: tuple[str, ...] = ()
    witnesses: tuple[SourceOpeningWitness, ...] = ()
    identity_available: bool = False


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
    routing_only: bool = False
    opening_witnesses: tuple[SourceOpeningWitness, ...] = ()
    prepared: bool = False
    prepared_revision: int | None = None
    prepared_window: str | None = None
    preparation_id: UUID | None = None

    def admits(self, kind: SourceKind) -> bool:
        return self.kind == kind or self.kind == SourceKind.UNKNOWN or self.uncertain


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
            "provisional_routing_source_count": sum(
                row.routing_only for row in self.records
            ),
            "classification_is_full_source_proof": False,
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


def classify_candidate_sources(
    session: Session,
    *,
    user: User,
    filters: IndexFilters,
    source_ids: tuple[UUID, ...],
    check_active: Callable[[], None],
) -> dict[UUID, SourceClassification]:
    """Classify only authorized retrieval candidates; full read proof remains separate."""
    identifiers = tuple(dict.fromkeys(source_ids))
    if not identifiers:
        return {}
    if len(identifiers) > MAX_OPENING_BATCH_SOURCES:
        raise ValueError("Candidate opening batches cannot exceed 100 sources.")
    check_active()
    sources, more = find_source_inventory_page(
        session,
        user=user,
        filters=filters,
        source_ids=identifiers,
        limit=MAX_OPENING_BATCH_SOURCES,
    )
    if more:
        raise CorpusScopeUnavailable("Candidate authorization page was incomplete.")
    authorized = tuple(source.id for source in sources)
    if not authorized:
        return {}
    try:
        openings = _routing_opening_batch(session, authorized, filters, check_active)
    except CorpusScopeUnavailable:
        # Authorization is rechecked below before an unavailable identity becomes UNKNOWN.
        openings = {}
    current, more = find_source_inventory_page(
        session,
        user=user,
        filters=filters,
        source_ids=identifiers,
        limit=MAX_OPENING_BATCH_SOURCES,
    )
    if more:
        raise CorpusScopeUnavailable("Candidate authorization recheck was incomplete.")
    current_by_id = {source.id: source for source in current}
    check_active()
    records: dict[UUID, SourceClassification] = {}
    for source in sources:
        if current_by_id.get(source.id) != source:
            continue
        opening = openings.get(source.id)
        record = classify_source(
            source,
            (),
            metadata_complete=False,
            opening_texts=opening.texts
            if opening and opening.identity_available
            else (),
        )
        records[source.id] = record.model_copy(
            update={
                "routing_only": True,
                "opening_witnesses": opening.witnesses if opening else (),
                "method": "candidate_" + record.method,
            }
        )
    return records


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


def _opening_query_indexes(
    session: Session, source_ids: tuple[UUID, ...]
) -> dict[UUID, PublicationIndexSnapshot]:
    """Resolve each file's own activated receipts without merging files' authority."""
    if not source_ids:
        return {}
    name = get_current_search_settings(session).index_name
    settings = session.scalars(
        select(SearchSettings).where(SearchSettings.index_name == name)
    ).one_or_none()
    if settings is None:
        raise CorpusScopeUnavailable("Configured query index authority is unavailable.")
    actual = PublicationEncoderAuthority(
        provider=str(settings.provider_type) if settings.provider_type else None,
        model=settings.model_name,
        effective_dimension=settings.final_embedding_dim,
        endpoint_sha256=context_hash(settings.api_url),
        deployment_name=settings.deployment_name,
        api_version=settings.api_version,
        normalize=settings.normalize,
        passage_prefix=settings.passage_prefix,
    )
    payloads: dict[UUID, list[tuple[str, object]]] = {}
    for source_id, index_uuid, payload in session.execute(
        select(
            RegulatoryTemporalProjection.user_file_id,
            RegulatoryTemporalProjection.index_uuid,
            RegulatoryTemporalProjection.payload["index"],
        )
        .where(
            RegulatoryTemporalProjection.user_file_id.in_(source_ids),
            RegulatoryTemporalProjection.retired_at.is_(None),
            RegulatoryTemporalProjection.payload["index"]["index_name"].astext == name,
        )
        .distinct()
    ):
        payloads.setdefault(source_id, []).append((index_uuid, payload))
    result: dict[UUID, PublicationIndexSnapshot] = {}
    for source_id, values in payloads.items():
        identifiers = {identifier for identifier, _ in values}
        if len(identifiers) != 1:
            continue
        identifier = next(iter(identifiers))
        observed = observed_index_snapshot(settings, identifier)
        accepted: PublicationIndexSnapshot | None = None
        receipts = {}
        for _, payload in values:
            index = PublicationIndexSnapshot.model_validate(payload)
            if (
                index.index_name != name
                or index.search_settings_id != settings.id
                or index.multitenant != MULTI_TENANT
            ):
                continue
            if index.encoder_authority is None:
                if index != observed:
                    raise CorpusScopeUnavailable(
                        "Activated query configuration changed."
                    )
            elif index.effective_authority() != effective_runtime_authority(
                actual, query_prefix=settings.query_prefix
            ):
                raise CorpusScopeUnavailable("Activated encoder authority changed.")
            if accepted is not None and not accepted.matches_temporal_index(index):
                raise CorpusScopeUnavailable(
                    "Activated query index authority is ambiguous."
                )
            if accepted is None or index.encoder_authority is not None:
                accepted = index
            receipts.update(
                {
                    receipt.configuration_json: receipt
                    for receipt in index.encoder_receipts
                }
            )
        if accepted is not None:
            result[source_id] = PublicationIndexSnapshot.model_validate(
                accepted.model_copy(
                    update={"encoder_receipts": tuple(receipts.values())}
                ).model_dump(mode="json")
            )
    return result


def _ranked_temporal_openings(
    indexes: dict[UUID, str], filters: IndexFilters
) -> Subquery:
    as_of = filters.as_of_date or date.today()
    sources_by_index: dict[str, list[UUID]] = {}
    for source_id, index_uuid in indexes.items():
        sources_by_index.setdefault(index_uuid, []).append(source_id)
    position = sql_cast(
        RegulatoryTemporalProjection.payload["semantic_position"].astext, Integer
    )
    return (
        select(
            RegulatoryTemporalProjection.id.label("id"),
            func.row_number()
            .over(
                partition_by=RegulatoryTemporalProjection.user_file_id,
                order_by=(
                    position,
                    RegulatoryTemporalProjection.projection_ordinal,
                ),
            )
            .label("opening_rank"),
        )
        .join(
            RegulatoryChunk,
            RegulatoryChunk.id == RegulatoryTemporalProjection.canonical_chunk_id,
        )
        .where(
            or_(
                *(
                    and_(
                        RegulatoryTemporalProjection.index_uuid == index_uuid,
                        RegulatoryTemporalProjection.user_file_id.in_(source_ids),
                    )
                    for index_uuid, source_ids in sources_by_index.items()
                )
            ),
            RegulatoryTemporalProjection.retired_at.is_(None),
            RegulatoryTemporalProjection.payload["derived_role"].astext == "canonical",
            position >= 0,
            or_(
                RegulatoryTemporalProjection.effective_start.is_(None),
                RegulatoryTemporalProjection.effective_start <= as_of,
            ),
            or_(
                RegulatoryTemporalProjection.effective_end.is_(None),
                RegulatoryTemporalProjection.effective_end > as_of,
            ),
            or_(
                RegulatoryChunk.validity_start_date.is_(None),
                RegulatoryChunk.validity_start_date <= as_of,
            ),
            or_(
                RegulatoryChunk.validity_end_date.is_(None),
                RegulatoryChunk.validity_end_date > as_of,
            ),
        )
        .subquery()
    )


def _opening_rows(
    session: Session,
    source_ids: tuple[UUID, ...],
    filters: IndexFilters,
    indexes: dict[UUID, PublicationIndexSnapshot],
    witnesses: dict[UUID, tuple[SourceOpeningWitness, ...]] | None = None,
) -> dict[UUID, list[str]]:
    result: dict[UUID, list[str]] = {source_id: [] for source_id in source_ids}
    if indexes:
        ranked = _ranked_temporal_openings(
            {source_id: index.index_uuid for source_id, index in indexes.items()},
            filters,
        )
        rows = list(
            session.scalars(
                select(RegulatoryTemporalProjection).where(
                    RegulatoryTemporalProjection.id.in_(
                        select(ranked.c.id).where(
                            ranked.c.opening_rank <= MAX_OPENING_CHUNKS + 1
                        )
                    )
                )
            )
        )
        validate_temporal_canonical_revisions(session, rows)
        revisions = (
            get_canonical_revisions(
                session,
                [
                    row.canonical_revision_id
                    for row in rows
                    if row.canonical_revision_id
                ],
            )
            if witnesses is not None
            else {}
        )
        bindings: dict[UUID, list[AnnexTemporalProjection]] = {}
        for row in rows:
            if publication_digest(row.payload) != row.payload_sha256:
                raise CorpusScopeUnavailable("Temporal binding payload changed.")
            binding = AnnexTemporalProjection.model_validate(row.payload)
            index = indexes[row.user_file_id]
            if not binding.index.matches_temporal_index(
                index
            ) or not accepts_publication_projection(index, binding.projection):
                raise CorpusScopeUnavailable(
                    "Temporal binding has no accepted publication/encoder receipt."
                )
            bindings.setdefault(row.user_file_id, []).append(binding)
        for source_id, values in bindings.items():
            result[source_id] = [
                binding.representation_text
                for binding in sorted(
                    values,
                    key=lambda binding: (
                        binding.semantic_position,
                        binding.projection.ordinal,
                    ),
                )[:MAX_OPENING_CHUNKS]
            ]
            if witnesses is not None:
                own_rows = sorted(
                    (row for row in rows if row.user_file_id == source_id),
                    key=lambda row: (
                        int(row.payload["semantic_position"]),
                        row.projection_ordinal,
                    ),
                )
                # Full payload/revision validation above is the authority gate.
                witnesses[source_id] = tuple(
                    SourceOpeningWitness(
                        chunk_id=row.canonical_chunk_id,
                        text_sha256=context_hash(
                            str(row.payload["representation_text"])
                        ),
                        binding_id=row.id,
                        index_uuid=row.index_uuid,
                        payload_sha256=row.payload_sha256,
                        canonical_revision_id=row.canonical_revision_id,
                        revision_sha256=revisions[row.canonical_revision_id].sha256
                        if row.canonical_revision_id is not None
                        else None,
                        canonical_base_sha256=str(row.payload["canonical_base_sha256"]),
                        effective_start=row.effective_start,
                        effective_end=row.effective_end,
                    )
                    for row in own_rows[:MAX_OPENING_CHUNKS]
                )
    timeless = tuple(source_id for source_id in source_ids if source_id not in indexes)
    if timeless:
        statement = select(
            RegulatoryChunk.id.label("id"),
            func.row_number()
            .over(
                partition_by=RegulatoryChunk.user_file_id,
                order_by=(RegulatoryChunk.position, RegulatoryChunk.id),
            )
            .label("opening_rank"),
        ).where(
            RegulatoryChunk.user_file_id.in_(timeless),
            RegulatoryChunk.position >= 0,
            RegulatoryChunk.chunk_type.is_distinct_from("hierarchical_aggregate"),
        )
        if filters.as_of_date is None:
            statement = statement.where(RegulatoryChunk.status == "active")
        else:
            statement = statement.where(
                or_(
                    RegulatoryChunk.validity_start_date.is_(None),
                    RegulatoryChunk.validity_start_date <= filters.as_of_date,
                ),
                or_(
                    RegulatoryChunk.validity_end_date.is_(None),
                    RegulatoryChunk.validity_end_date > filters.as_of_date,
                ),
            )
        ranked = statement.subquery()
        rows = list(
            session.scalars(
                select(RegulatoryChunk).where(
                    RegulatoryChunk.id.in_(
                        select(ranked.c.id).where(
                            ranked.c.opening_rank <= MAX_OPENING_CHUNKS + 1
                        )
                    )
                )
            )
        )
        for row in sorted(
            rows, key=lambda row: (str(row.user_file_id), row.position, row.id)
        ):
            if len(result[row.user_file_id]) < MAX_OPENING_CHUNKS:
                result[row.user_file_id].append(row.text)
                if witnesses is not None:
                    witnesses[row.user_file_id] = (
                        *witnesses.get(row.user_file_id, ()),
                        SourceOpeningWitness(
                            chunk_id=row.id, text_sha256=context_hash(row.text)
                        ),
                    )
    return result


def _opening_batch(
    session: Session,
    source_ids: tuple[UUID, ...],
    filters: IndexFilters,
    check_active: Callable[[], None],
) -> dict[UUID, tuple[str, ...] | None]:
    check_active()
    observation = observe_publication_read()
    require_publication_files(observation, source_ids)
    qualified = qualified_file_ids(session, source_ids)
    indexes = _opening_query_indexes(session, tuple(qualified))
    # A qualified file with unavailable authority never receives timeless fallback.
    readable = tuple(
        source_id
        for source_id in source_ids
        if source_id not in qualified or source_id in indexes
    )
    texts = _opening_rows(session, readable, filters, indexes)
    current_qualified = qualified_file_ids(session, source_ids)
    require_publication_files(observation, source_ids)
    check_active()
    result: dict[UUID, tuple[str, ...] | None] = {}
    for source_id in source_ids:
        if source_id not in texts or (source_id in qualified) != (
            source_id in current_qualified
        ):
            result[source_id] = None
            continue
        remaining = MAX_OPENING_CHARS
        retained: list[str] = []
        for text in texts[source_id]:
            if remaining <= 0:
                break
            retained.append(text[:remaining])
            remaining -= len(retained[-1])
        result[source_id] = tuple(retained)
    return result


def _bounded_opening_texts(texts: list[str]) -> tuple[str, ...]:
    retained: list[str] = []
    remaining = MAX_OPENING_CHARS
    for text in texts[:MAX_OPENING_CHUNKS]:
        if remaining <= 0:
            break
        retained.append(text[:remaining])
        remaining -= len(retained[-1])
    return tuple(retained)


def _routing_opening_batch(
    session: Session,
    source_ids: tuple[UUID, ...],
    filters: IndexFilters,
    check_active: Callable[[], None],
) -> dict[UUID, RoutingOpening]:
    """Read provisional headings without hydrating search payloads or encoder receipts."""
    check_active()
    observation = observe_publication_read()
    require_publication_files(observation, source_ids)
    qualified = qualified_file_ids(session, source_ids)
    result = {source_id: RoutingOpening() for source_id in source_ids}
    if qualified:
        name = get_current_search_settings(session).index_name
        physical: dict[UUID, set[str]] = {}
        for source_id, index_uuid in session.execute(
            select(
                RegulatoryTemporalProjection.user_file_id,
                RegulatoryTemporalProjection.index_uuid,
            )
            .where(
                RegulatoryTemporalProjection.user_file_id.in_(qualified),
                RegulatoryTemporalProjection.retired_at.is_(None),
            )
            .distinct()
        ):
            physical.setdefault(source_id, set()).add(index_uuid)
        indexes = {
            source_id: next(iter(values))
            for source_id, values in physical.items()
            if len(values) == 1
        }
        if indexes:
            ranked = _ranked_temporal_openings(indexes, filters)
            rows = list(
                session.execute(
                    select(
                        RegulatoryTemporalProjection.id,
                        RegulatoryTemporalProjection.user_file_id,
                        RegulatoryTemporalProjection.canonical_chunk_id,
                        RegulatoryTemporalProjection.canonical_revision_id,
                        RegulatoryTemporalProjection.index_uuid,
                        RegulatoryTemporalProjection.projection_ordinal,
                        RegulatoryTemporalProjection.payload_sha256,
                        RegulatoryTemporalProjection.effective_start,
                        RegulatoryTemporalProjection.effective_end,
                        RegulatoryTemporalProjection.payload["index"][
                            "index_name"
                        ].astext.label("index_name"),
                        RegulatoryTemporalProjection.payload[
                            "semantic_position"
                        ].astext.label("semantic_position"),
                        RegulatoryTemporalProjection.payload[
                            "canonical_base_sha256"
                        ].astext.label("canonical_base_sha256"),
                        RegulatoryTemporalProjection.payload[
                            "representation_text"
                        ].astext.label("representation_text"),
                    ).where(
                        RegulatoryTemporalProjection.id.in_(
                            select(ranked.c.id).where(
                                ranked.c.opening_rank <= MAX_OPENING_CHUNKS + 1
                            )
                        )
                    )
                ).all()
            )
            try:
                revisions = get_canonical_revisions(
                    session,
                    [
                        row.canonical_revision_id
                        for row in rows
                        if row.canonical_revision_id
                    ],
                )
            except ValueError:
                revisions = {}
            for source_id in indexes:
                check_active()
                own = sorted(
                    (row for row in rows if row.user_file_id == source_id),
                    key=lambda row: (
                        int(row.semantic_position),
                        row.projection_ordinal,
                    ),
                )
                if not own or any(
                    row.index_name != name
                    or not isinstance(row.representation_text, str)
                    or not isinstance(row.canonical_base_sha256, str)
                    for row in own
                ):
                    continue
                witnesses: list[SourceOpeningWitness] = []
                valid = True
                for row in own:
                    revision = revisions.get(row.canonical_revision_id)
                    if revision is None or (
                        revision.snapshot.id != row.canonical_chunk_id
                        or UUID(revision.snapshot.user_file_id) != source_id
                        or context_hash(revision.snapshot.text)
                        != row.canonical_base_sha256
                    ):
                        valid = False
                    witnesses.append(
                        SourceOpeningWitness(
                            chunk_id=row.canonical_chunk_id,
                            text_sha256=context_hash(row.representation_text),
                            binding_id=row.id,
                            index_uuid=row.index_uuid,
                            payload_sha256=row.payload_sha256,
                            canonical_revision_id=row.canonical_revision_id,
                            revision_sha256=revision.sha256 if revision else None,
                            canonical_base_sha256=row.canonical_base_sha256,
                            effective_start=row.effective_start,
                            effective_end=row.effective_end,
                        )
                    )
                result[source_id] = RoutingOpening(
                    texts=_bounded_opening_texts(
                        [row.representation_text for row in own]
                    ),
                    witnesses=tuple(witnesses[:MAX_OPENING_CHUNKS]),
                    identity_available=valid,
                )
    timeless = tuple(
        source_id for source_id in source_ids if source_id not in qualified
    )
    if timeless:
        statement = select(
            RegulatoryChunk.id.label("id"),
            func.row_number()
            .over(
                partition_by=RegulatoryChunk.user_file_id,
                order_by=(RegulatoryChunk.position, RegulatoryChunk.id),
            )
            .label("opening_rank"),
        ).where(
            RegulatoryChunk.user_file_id.in_(timeless),
            RegulatoryChunk.position >= 0,
            RegulatoryChunk.chunk_type.is_distinct_from("hierarchical_aggregate"),
        )
        if filters.as_of_date is None:
            statement = statement.where(RegulatoryChunk.status == "active")
        else:
            statement = statement.where(
                or_(
                    RegulatoryChunk.validity_start_date.is_(None),
                    RegulatoryChunk.validity_start_date <= filters.as_of_date,
                ),
                or_(
                    RegulatoryChunk.validity_end_date.is_(None),
                    RegulatoryChunk.validity_end_date > filters.as_of_date,
                ),
            )
        ranked = statement.subquery()
        texts: dict[UUID, list[str]] = {}
        witnesses_by_source: dict[UUID, list[SourceOpeningWitness]] = {}
        for row in session.execute(
            select(
                RegulatoryChunk.user_file_id, RegulatoryChunk.id, RegulatoryChunk.text
            )
            .where(
                RegulatoryChunk.id.in_(
                    select(ranked.c.id).where(
                        ranked.c.opening_rank <= MAX_OPENING_CHUNKS
                    )
                )
            )
            .order_by(
                RegulatoryChunk.user_file_id,
                RegulatoryChunk.position,
                RegulatoryChunk.id,
            )
        ):
            texts.setdefault(row.user_file_id, []).append(row.text)
            witnesses_by_source.setdefault(row.user_file_id, []).append(
                SourceOpeningWitness(
                    chunk_id=row.id, text_sha256=context_hash(row.text)
                )
            )
        for source_id, values in texts.items():
            result[source_id] = RoutingOpening(
                texts=_bounded_opening_texts(values),
                witnesses=tuple(witnesses_by_source[source_id]),
                identity_available=True,
            )
    current_qualified = qualified_file_ids(session, source_ids)
    require_publication_files(observation, source_ids)
    check_active()
    return {
        source_id: value
        if (source_id in qualified) == (source_id in current_qualified)
        else RoutingOpening()
        for source_id, value in result.items()
    }


def _source_openings(
    session: Session,
    source_page: list[CorpusSource],
    *,
    user: User,
    filters: IndexFilters,
    check_active: Callable[[], None],
    opening_workers: int,
    heartbeat: Callable[[], None] | None = None,
    routing_only: bool = False,
) -> list[tuple[str, ...] | RoutingOpening | None]:
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

    if opening_workers == 1 and not routing_only:
        return [read(source) for source in source_page]
    if not source_page:
        return []
    identifiers = tuple(source.id for source in source_page)
    authorized, _ = find_source_inventory_page(
        session,
        user=user,
        filters=filters,
        source_ids=identifiers,
        limit=SOURCE_PAGE_SIZE,
    )
    captured = {source.id: source for source in source_page}
    admitted = tuple(
        source.id for source in authorized if captured.get(source.id) == source
    )

    def read_batch(
        group: tuple[UUID, ...],
    ) -> dict[UUID, tuple[str, ...] | RoutingOpening | None]:
        try:
            with get_session_with_current_tenant() as opening_session:
                if routing_only:
                    return dict(
                        _routing_opening_batch(
                            opening_session, group, filters, check_active
                        )
                    )
                return dict(
                    _opening_batch(opening_session, group, filters, check_active)
                )
        except CorpusScopeUnavailable:
            return {source_id: None for source_id in group}

    width = min(
        MAX_OPENING_BATCH_SOURCES,
        max(1, (len(admitted) + opening_workers - 1) // opening_workers),
    )
    executor = ThreadPoolExecutor(max_workers=opening_workers)
    try:
        futures = [
            cast(
                Future[dict[UUID, tuple[str, ...] | RoutingOpening | None]],
                executor.submit(
                    copy_context().run, read_batch, admitted[offset : offset + width]
                ),
            )
            for offset in range(0, len(admitted), width)
        ]
        result: dict[UUID, tuple[str, ...] | RoutingOpening | None] = {}
        last_heartbeat = time.monotonic()
        for future in futures:
            while True:
                check_active()
                if heartbeat is not None and time.monotonic() - last_heartbeat >= 5:
                    heartbeat()
                    last_heartbeat = time.monotonic()
                try:
                    result.update(future.result(timeout=0.05))
                    break
                except FutureTimeout:
                    continue
        current, _ = find_source_inventory_page(
            session,
            user=user,
            filters=filters,
            source_ids=identifiers,
            limit=SOURCE_PAGE_SIZE,
        )
        retained = {
            source.id for source in current if captured.get(source.id) == source
        }
        return [
            result.get(source.id) if source.id in retained else None
            for source in source_page
        ]
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def load_source_lane_catalogue(
    session: Session,
    *,
    user: User,
    filters: IndexFilters,
    check_active: Callable[[], None],
    max_sources: int | None = None,
    opening_workers: int = 1,
    on_progress: Callable[[int, bool], None] | None = None,
    routing_only: bool = False,
) -> SourceLaneCatalogue:
    if not isinstance(routing_only, bool):
        raise ValueError("Routing-only inventory mode must be boolean")
    if max_sources is not None and (
        not isinstance(max_sources, int)
        or isinstance(max_sources, bool)
        or max_sources < 1
    ):
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
    while more and (max_sources is None or offset < max_sources):
        check_active()
        page_limit = (
            SOURCE_PAGE_SIZE
            if max_sources is None
            else min(SOURCE_PAGE_SIZE, max_sources - offset)
        )
        sources, more = find_source_inventory_page(
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
            heartbeat=(lambda: on_progress(len(records), True))
            if on_progress
            else None,
            routing_only=routing_only,
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
            routing = openings if isinstance(openings, RoutingOpening) else None
            texts = (
                (routing.texts if routing and routing.identity_available else ())
                if routing is not None
                else (openings or ())
            )
            record = classify_source(
                source,
                types.get(source.id, ()),
                metadata_complete=metadata_complete,
                opening_texts=cast(tuple[str, ...], texts),
            )
            if routing_only:
                record = record.model_copy(
                    update={
                        "routing_only": True,
                        "opening_witnesses": routing.witnesses if routing else (),
                        "method": "provisional_" + record.method,
                    }
                )
            records.append(record)
        offset += page_limit
        if on_progress is not None:
            on_progress(len(records), more)
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
    if recorded.routing_only:
        observation = observe_publication_read()
        qualified = qualified_file_ids(session, (source.id,))
        indexes = _opening_query_indexes(session, tuple(qualified))
        if qualified and source.id not in indexes:
            raise CorpusScopeUnavailable(
                "Source has no verified query index authority."
            )
        witnesses: dict[UUID, tuple[SourceOpeningWitness, ...]] = {}
        texts = _opening_rows(
            session, (source.id,), filters, indexes, witnesses=witnesses
        ).get(source.id, [])
        if (
            recorded.opening_witnesses
            and witnesses.get(source.id, ()) != recorded.opening_witnesses
        ) or (recorded.kind != SourceKind.UNKNOWN and not recorded.opening_witnesses):
            raise CorpusScopeUnavailable(
                "Source opening witness changed after inventory."
            )
        if qualified != qualified_file_ids(session, (source.id,)):
            raise CorpusScopeUnavailable(
                "Source qualification changed during validation."
            )
        require_publication_files(observation, (source.id,))
        opening_texts = _bounded_opening_texts(texts)
    else:
        opening_texts = _opening_texts(
            session,
            user=user,
            filters=filters,
            source=source,
            check_active=check_active,
        )
    types, complete = _document_types(session, (source.id,), filters, check_active)
    current = classify_source(
        source,
        types.get(source.id, ()),
        metadata_complete=complete,
        opening_texts=opening_texts,
    )
    if not (recorded.routing_only and recorded.kind == SourceKind.UNKNOWN) and (
        current.kind != recorded.kind
        or current.opening_identity_sha256 != recorded.opening_identity_sha256
    ):
        raise CorpusScopeUnavailable(
            "Source classification changed after the lane inventory."
        )
    check_active()
    return source
