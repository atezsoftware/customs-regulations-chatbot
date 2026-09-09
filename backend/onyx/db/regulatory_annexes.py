"""Scoped annex identities, immutable snapshots and canonical chunk relations."""

import datetime
import hashlib
import json
import re
from collections import defaultdict
from uuid import UUID, uuid4

from sqlalchemy import or_, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from onyx.db.models import (
    AmendmentSourcePackage,
    DocumentSet__UserFile,
    RegulatoryAnnex,
    RegulatoryAnnexElement,
    RegulatoryAnnexElementChunk,
    RegulatoryAnnexRevision,
    RegulatoryAnnexRevisionElement,
    RegulatoryChunk,
    RegulatorySourceAsset,
    UserFile,
)
from onyx.db.regulatory_chunks import is_hierarchical_aggregate_chunk
from onyx.regulatory.amendments.annexes.models import (
    AnnexExtraction,
    ExtractedAnnexElement,
)


def normalize_annex_label(value: str) -> str:
    normalized = re.sub(r"\s+", "", value.casefold())
    match = re.fullmatch(r"(ek|annex|appendix)[-–—:]?(.*)", normalized)
    if match:
        body = match.group(2)
        if not re.fullmatch(
            r"(?:[0-9]+[a-z]?|[ivxlcdm]+[a-z]?|[a-z])(?:[/.-][0-9a-z]+)*", body
        ):
            raise ValueError("ambiguous annex label requires explicit scope review")
        return f"{match.group(1)}:{body}"
    if not normalized:
        raise ValueError("annex label is required")
    return normalized


def require_annex_file_scope(
    session: Session, document_set_id: int, user_file_id: UUID
) -> UserFile:
    file = session.scalar(
        select(UserFile)
        .join(DocumentSet__UserFile, DocumentSet__UserFile.user_file_id == UserFile.id)
        .where(
            UserFile.id == user_file_id,
            DocumentSet__UserFile.document_set_id == document_set_id,
        )
    )
    if file is None:
        raise ValueError("annex file scope mismatch")
    return file


def get_or_create_annex(
    session: Session, *, document_set_id: int, user_file_id: UUID, annex_label: str
) -> RegulatoryAnnex:
    require_annex_file_scope(session, document_set_id, user_file_id)
    label = normalize_annex_label(annex_label)
    session.execute(
        insert(RegulatoryAnnex)
        .values(
            id=uuid4(),
            document_set_id=document_set_id,
            user_file_id=user_file_id,
            label=label,
        )
        .on_conflict_do_nothing(constraint="uq_regulatory_annex_scope")
    )
    return session.scalars(
        select(RegulatoryAnnex)
        .where(
            RegulatoryAnnex.document_set_id == document_set_id,
            RegulatoryAnnex.user_file_id == user_file_id,
            RegulatoryAnnex.label == label,
        )
        .with_for_update()
    ).one()


def load_legacy_annex_chunks(
    session: Session,
    *,
    document_set_id: int,
    user_file_id: UUID,
    annex_label: str,
    as_of_date: datetime.date,
) -> list[RegulatoryChunk]:
    require_annex_file_scope(session, document_set_id, user_file_id)
    label = normalize_annex_label(annex_label)
    # Full relational scope, deliberately without search ranking/top-k.
    rows = list(
        session.scalars(
            select(RegulatoryChunk)
            .where(
                RegulatoryChunk.user_file_id == user_file_id,
                or_(
                    RegulatoryChunk.validity_start_date.is_(None),
                    RegulatoryChunk.validity_start_date <= as_of_date,
                ),
                or_(
                    RegulatoryChunk.validity_end_date.is_(None),
                    RegulatoryChunk.validity_end_date > as_of_date,
                ),
            )
            .order_by(RegulatoryChunk.position, RegulatoryChunk.id)
        )
    )

    def matches(row: RegulatoryChunk) -> bool:
        metadata_label = row.chunk_metadata.get("appendix_label")
        if isinstance(metadata_label, str) and metadata_label.strip():
            try:
                return normalize_annex_label(metadata_label) == label
            except ValueError:
                return False
        for heading in row.heading_path:
            if not heading.strip():
                continue
            try:
                if normalize_annex_label(heading) == label:
                    return True
            except ValueError:
                continue
        return False

    atomic = [
        row
        for row in rows
        if not is_hierarchical_aggregate_chunk(row)
        and not row.chunk_metadata.get("bound_to_regulatory_chunk_id")
        and matches(row)
    ]
    ids = {row.id for row in atomic}
    roots = set(canonical_chunk_lineage_keys(session, atomic).values())
    for row in rows:
        if is_hierarchical_aggregate_chunk(row):
            continue
        bound_id = row.chunk_metadata.get("bound_to_regulatory_chunk_id")
        if not isinstance(bound_id, str):
            continue
        bound = session.get(RegulatoryChunk, bound_id)
        if bound is not None and bound.user_file_id == user_file_id:
            if canonical_chunk_lineage_keys(session, [bound])[bound.id] in roots:
                ids.add(row.id)
        elif matches(row):
            raise ValueError(
                "legacy annex scope has unresolved image-companion binding"
            )
    return [row for row in rows if row.id in ids]


def get_revision_elements(
    session: Session, revision_id: UUID
) -> list[RegulatoryAnnexRevisionElement]:
    return list(
        session.scalars(
            select(RegulatoryAnnexRevisionElement)
            .where(RegulatoryAnnexRevisionElement.revision_id == revision_id)
            .order_by(RegulatoryAnnexRevisionElement.position)
        )
    )


def get_effective_annex_revision(
    session: Session, annex_id: UUID, as_of_date: datetime.date
) -> RegulatoryAnnexRevision | None:
    return session.scalar(
        select(RegulatoryAnnexRevision)
        .where(
            RegulatoryAnnexRevision.annex_id == annex_id,
            RegulatoryAnnexRevision.approved_at.is_not(None),
            or_(
                RegulatoryAnnexRevision.effective_start.is_(None),
                RegulatoryAnnexRevision.effective_start <= as_of_date,
            ),
            or_(
                RegulatoryAnnexRevision.effective_end.is_(None),
                RegulatoryAnnexRevision.effective_end > as_of_date,
            ),
        )
        .order_by(
            RegulatoryAnnexRevision.approved_at.desc(),
            RegulatoryAnnexRevision.created_at.desc(),
        )
        .limit(1)
    )


def _content_key(element: ExtractedAnnexElement) -> str:
    return hashlib.sha256(
        json.dumps(
            [element.kind, element.text, element.formula, element.value],
            ensure_ascii=False,
        ).encode()
    ).hexdigest()


def persist_annex_revision(
    session: Session,
    *,
    annex_id: UUID,
    extraction: AnnexExtraction,
    baseline_sha256: str,
    effective_start: datetime.date | None,
    effective_end: datetime.date | None,
    approved_at: datetime.datetime | None,
    predecessor_revision_id: UUID | None = None,
    source_asset_id: UUID | None = None,
    chunk_ids_by_position: dict[int, list[str]] | None = None,
    element_id_by_position: dict[int, UUID] | None = None,
    baseline_snapshot: dict[str, object] | None = None,
) -> RegulatoryAnnexRevision:
    annex = session.scalars(
        select(RegulatoryAnnex).where(RegulatoryAnnex.id == annex_id).with_for_update()
    ).one()
    existing = session.scalar(
        select(RegulatoryAnnexRevision).where(
            RegulatoryAnnexRevision.annex_id == annex_id,
            RegulatoryAnnexRevision.baseline_sha256 == baseline_sha256,
        )
    )
    if existing is not None:
        if (
            existing.snapshot.get("extraction") != extraction.model_dump(mode="json")
            or existing.effective_start != effective_start
            or existing.effective_end != effective_end
        ):
            raise ValueError("annex baseline hash reused with conflicting revision")
        return existing
    if (
        effective_start is not None
        and effective_end is not None
        and effective_end <= effective_start
    ):
        raise ValueError("invalid annex revision dates")
    if source_asset_id is not None:
        asset = session.scalar(
            select(RegulatorySourceAsset)
            .join(AmendmentSourcePackage)
            .where(
                RegulatorySourceAsset.id == source_asset_id,
                AmendmentSourcePackage.document_set_id == annex.document_set_id,
            )
        )
        if asset is None or asset.sha256 != extraction.source_sha256:
            raise ValueError("annex source asset scope or hash mismatch")
    previous: list[RegulatoryAnnexRevisionElement] = []
    if predecessor_revision_id is not None:
        predecessor = session.get(RegulatoryAnnexRevision, predecessor_revision_id)
        if predecessor is None or predecessor.annex_id != annex_id:
            raise ValueError("predecessor annex scope mismatch")
        previous = get_revision_elements(session, predecessor_revision_id)
    correspondence = element_id_by_position or {}
    prior_by_id = {row.element_id: row for row in previous}
    if len(set(correspondence.values())) != len(correspondence) or any(
        position < 0
        or position >= len(extraction.elements)
        or identity not in prior_by_id
        or prior_by_id[identity].payload["kind"] != extraction.elements[position].kind
        for position, identity in correspondence.items()
    ):
        raise ValueError(
            "annex element correspondence is outside the predecessor scope"
        )
    revision = RegulatoryAnnexRevision(
        id=uuid4(),
        annex_id=annex_id,
        predecessor_revision_id=predecessor_revision_id,
        source_asset_id=source_asset_id,
        baseline_sha256=baseline_sha256,
        snapshot={
            "extraction": extraction.model_dump(mode="json"),
            "baseline": baseline_snapshot,
        },
        created_at=datetime.datetime.now(datetime.timezone.utc),
        effective_start=effective_start,
        effective_end=effective_end,
        approved_at=approved_at,
    )
    session.add(revision)
    session.flush()
    semantic: dict[str, list[UUID]] = defaultdict(list)
    content: dict[str, list[UUID]] = defaultdict(list)
    for prior in previous:
        element = ExtractedAnnexElement.model_validate(prior.payload)
        if element.semantic_key:
            semantic[element.semantic_key].append(prior.element_id)
        if element.semantic_key is None:
            content[_content_key(element)].append(prior.element_id)
    prior_ids = {item.element_id for item in previous}
    used: set[UUID] = set(correspondence.values())
    new_semantic_counts: dict[str, int] = defaultdict(int)
    new_content_counts: dict[str, int] = defaultdict(int)
    for element in extraction.elements:
        if element.semantic_key:
            new_semantic_counts[element.semantic_key] += 1
        new_content_counts[_content_key(element)] += 1
    for position, element in enumerate(extraction.elements):
        candidates = (
            semantic.get(element.semantic_key, [])
            if element.semantic_key and new_semantic_counts[element.semantic_key] == 1
            else []
        )
        if len(candidates) != 1:
            candidates = (
                content.get(_content_key(element), [])
                if element.semantic_key is None
                and new_content_counts[_content_key(element)] == 1
                else []
            )
        element_id = (
            candidates[0]
            if len(candidates) == 1 and candidates[0] not in used
            else uuid4()
        )
        if position in correspondence:
            element_id = correspondence[position]
        if element_id not in prior_ids:
            session.add(RegulatoryAnnexElement(id=element_id, annex_id=annex_id))
            session.flush()
        used.add(element_id)
        session.add(
            RegulatoryAnnexRevisionElement(
                revision_id=revision.id,
                element_id=element_id,
                position=position,
                payload=element.model_dump(mode="json"),
            )
        )
        session.flush()
        for chunk_id in (chunk_ids_by_position or {}).get(position, []):
            chunk = session.get(RegulatoryChunk, chunk_id)
            if chunk is None or chunk.user_file_id != annex.user_file_id:
                raise ValueError("element chunk scope mismatch")
            session.add(
                RegulatoryAnnexElementChunk(
                    revision_id=revision.id, element_id=element_id, chunk_id=chunk_id
                )
            )
    if approved_at is not None:
        latest = (
            session.get(RegulatoryAnnexRevision, annex.latest_approved_revision_id)
            if annex.latest_approved_revision_id
            else None
        )
        if (
            latest is None
            or latest.approved_at is None
            or latest.approved_at <= approved_at
        ):
            annex.latest_approved_revision_id = revision.id
    session.flush()
    return revision


def copy_annex_chunk_links(
    session: Session, *, old_chunk_id: str, new_chunk_id: str
) -> None:
    """Carry source-element lineage through an approved textual correction."""
    old = session.get(RegulatoryChunk, old_chunk_id)
    new = session.get(RegulatoryChunk, new_chunk_id)
    if old is None or new is None or old.user_file_id != new.user_file_id:
        raise ValueError("annex chunk lineage scope mismatch")
    links = session.scalars(
        select(RegulatoryAnnexElementChunk).where(
            RegulatoryAnnexElementChunk.chunk_id == old_chunk_id
        )
    ).all()
    for link in links:
        session.execute(
            insert(RegulatoryAnnexElementChunk)
            .values(
                revision_id=link.revision_id,
                element_id=link.element_id,
                chunk_id=new_chunk_id,
            )
            .on_conflict_do_nothing()
        )
    session.flush()


def canonical_chunk_lineage_keys(
    session: Session, rows: list[RegulatoryChunk]
) -> dict[str, str]:
    """Use the original canonical identity across any number of approvals."""
    cache = {row.id: row for row in rows}
    keys: dict[str, str] = {}
    for row in rows:
        ancestor = row
        visited = {row.id}
        while ancestor.supersedes_chunk_id:
            parent_id = ancestor.supersedes_chunk_id
            if parent_id in visited:
                raise ValueError("canonical chunk lineage cycle")
            visited.add(parent_id)
            parent = cache.get(parent_id) or session.get(RegulatoryChunk, parent_id)
            if parent is None or parent.user_file_id != row.user_file_id:
                raise ValueError("canonical chunk lineage scope mismatch")
            cache[parent_id] = parent
            ancestor = parent
        keys[row.id] = f"canonical:{ancestor.id}"
    return keys
