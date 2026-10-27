"""Document-level succession is explicitly reviewed, never inferred from numbering."""

import datetime
import hashlib
import json
from difflib import SequenceMatcher
from typing import Any, Literal
from uuid import UUID

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from onyx.db.document_lineage_identity import (
    document_legal_identities,
    exact_identity_search_condition,
    explicit_repeal_targets,
    identity_search_condition,
    legal_number_keys,
    normalized_legal_text,
)
from onyx.db.enums import UserFileStatus
from onyx.db.models import (
    DocumentSet,
    DocumentSet__UserFile,
    RegulatoryDocumentIntake,
    RegulatoryDocumentIntakeFile,
    RegulatoryDocumentLineageReview,
    UserFile,
)

LineageRelation = Literal["successor", "repeals"]


def _member(
    session: Session, batch: RegulatoryDocumentIntake, document_id: UUID
) -> RegulatoryDocumentIntakeFile:
    member = session.get(RegulatoryDocumentIntakeFile, document_id)
    if member is None or member.intake_id != batch.id:
        raise ValueError("Intake document not found")
    return member


def _rows(
    session: Session, batch: RegulatoryDocumentIntake, *, lock: bool = False
) -> list[RegulatoryDocumentLineageReview]:
    statement = (
        select(RegulatoryDocumentLineageReview)
        .join(
            RegulatoryDocumentIntakeFile,
            RegulatoryDocumentIntakeFile.id
            == RegulatoryDocumentLineageReview.intake_document_id,
        )
        .where(RegulatoryDocumentIntakeFile.intake_id == batch.id)
        .order_by(RegulatoryDocumentLineageReview.intake_document_id)
    )
    if lock:
        statement = statement.with_for_update(of=RegulatoryDocumentLineageReview)
    return list(session.scalars(statement))


def _existing_file(
    session: Session, document_set_id: int, file_id: UUID, *, lock: bool = False
) -> UserFile:
    statement = (
        select(UserFile)
        .join(DocumentSet__UserFile, UserFile.id == DocumentSet__UserFile.user_file_id)
        .where(
            UserFile.id == file_id,
            DocumentSet__UserFile.document_set_id == document_set_id,
            UserFile.status == UserFileStatus.COMPLETED,
        )
    )
    if lock:
        statement = statement.with_for_update(of=UserFile).execution_options(
            populate_existing=True
        )
    file = session.scalar(statement)
    if file is None:
        raise ValueError("Önceki belge PC Külliyatı'nda tamamlanmış bir dosya olmalı")
    return file


def _file_fingerprint(file: UserFile) -> str:
    payload = {
        "id": str(file.id),
        "name": file.name,
        "file_id": file.file_id,
        "status": file.status.value,
        "generation_hash": file.regulatory_chunk_generation_hash,
        "link_url": file.link_url,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def predecessor_candidates(
    session: Session,
    batch: RegulatoryDocumentIntake,
    document_id: UUID,
    query: str | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Suggest own legal identities; selecting a relation remains an explicit review."""
    member = _member(session, batch, document_id)
    new_file = session.get_one(UserFile, member.user_file_id)
    new_identity = document_legal_identities(session, [new_file])[new_file.id]
    targets = (
        explicit_repeal_targets(session, new_file.id) if query is None else frozenset()
    )
    exact_target = targets if len(targets) == 1 else frozenset()
    normalized = normalized_legal_text(
        query if query is not None else new_identity.title or "",
        retain_numbers=query is not None,
    )
    tokens = sorted(set(normalized.split()), key=lambda token: (-len(token), token))[:8]
    if not tokens and not exact_target:
        return []
    unapproved = (
        select(RegulatoryDocumentIntakeFile.user_file_id)
        .join(
            RegulatoryDocumentIntake,
            RegulatoryDocumentIntake.id == RegulatoryDocumentIntakeFile.intake_id,
        )
        .where(RegulatoryDocumentIntake.status != "completed")
    )
    current_batch_files = select(RegulatoryDocumentIntakeFile.user_file_id).where(
        RegulatoryDocumentIntakeFile.intake_id == batch.id
    )
    files = list(
        session.scalars(
            select(UserFile)
            .join(
                DocumentSet__UserFile, UserFile.id == DocumentSet__UserFile.user_file_id
            )
            .where(
                DocumentSet__UserFile.document_set_id == batch.document_set_id,
                UserFile.status == UserFileStatus.COMPLETED,
                UserFile.id.not_in(unapproved),
                UserFile.id.not_in(current_batch_files),
                exact_identity_search_condition(exact_target)
                if exact_target
                else identity_search_condition(tokens),
            )
            .order_by(UserFile.created_at.desc(), UserFile.id)
            .limit(300)
        )
    )
    identities = document_legal_identities(session, files)
    today = datetime.datetime.now(datetime.timezone.utc).date()
    repeal_dates: dict[UUID, datetime.date | None] = {
        file_id: effective_date
        for file_id, effective_date in session.execute(
            select(
                RegulatoryDocumentLineageReview.predecessor_user_file_id,
                func.min(RegulatoryDocumentLineageReview.effective_date),
            )
            .where(
                RegulatoryDocumentLineageReview.document_set_id
                == batch.document_set_id,
                RegulatoryDocumentLineageReview.status == "active",
                RegulatoryDocumentLineageReview.relation == "repeals",
                RegulatoryDocumentLineageReview.predecessor_user_file_id.in_(
                    [file.id for file in files]
                ),
            )
            .group_by(RegulatoryDocumentLineageReview.predecessor_user_file_id)
        )
    }
    result = []
    wanted = set(normalized.split())
    query_keys = legal_number_keys(query) if query else frozenset()
    for file in files:
        identity = identities[file.id]
        if exact_target:
            if not exact_target.intersection(identity.number_keys):
                continue
            score = 100
            reason = "Kaynakta açıkça kaldırılan mevzuatın kendi numarası eşleşti. Hukuki ilişkiyi onaylayın."
        elif query_keys.intersection(identity.number_keys):
            score = 100
            reason = "Aranan numara belgenin kendi mevzuat kimliğiyle eşleşti. Hukuki ilişkiyi onaylayın."
        else:
            title = normalized_legal_text(
                identity.title or "", retain_numbers=query is not None
            )
            if not title:
                continue
            if query is not None:
                title = " ".join(
                    (
                        title,
                        normalized_legal_text(
                            " ".join(identity.number_keys), retain_numbers=True
                        ),
                    )
                )
            available = set(title.split())
            overlap = len(wanted & available) / max(len(wanted | available), 1)
            score = round(
                100
                * (
                    0.65 * overlap
                    + 0.35 * SequenceMatcher(None, normalized, title).ratio()
                )
            )
            if score < 25:
                continue
            basis = (
                "Anlamlı dosya adı"
                if identity.title_source == "filename"
                else "Belgenin kendi hukuki başlığı"
            )
            reason = f"{basis} benzerliği; numara farklılığı tek başına hukuki ilişki kurmaz. Hukuki ilişkiyi onaylayın."
        repeal_date = repeal_dates.get(file.id)
        result.append(
            {
                "user_file_id": str(file.id),
                "title": identity.title or file.name,
                "reason": reason,
                "score": score,
                "status": "repealed"
                if repeal_date and repeal_date <= today
                else "repeal_scheduled"
                if repeal_date
                else "current",
            }
        )
    return sorted(
        result, key=lambda row: (-row["score"], row["title"], row["user_file_id"])
    )[: max(1, min(limit, 20))]


def _validate_graph(
    session: Session,
    document_set_id: int,
    predecessor_id: UUID,
    successor_id: UUID,
    *,
    exclude_document_id: UUID,
) -> None:
    if predecessor_id == successor_id:
        raise ValueError("Bir belge kendisinin önceki sürümü olamaz")
    edges = session.execute(
        select(
            RegulatoryDocumentLineageReview.predecessor_user_file_id,
            RegulatoryDocumentLineageReview.successor_user_file_id,
        ).where(
            RegulatoryDocumentLineageReview.document_set_id == document_set_id,
            RegulatoryDocumentLineageReview.intake_document_id != exclude_document_id,
            RegulatoryDocumentLineageReview.status.in_(("frozen", "active")),
        )
    )
    graph: dict[UUID, set[UUID]] = {}
    for previous, following in edges:
        graph.setdefault(previous, set()).add(following)
    pending = [successor_id]
    visited: set[UUID] = set()
    while pending:
        current = pending.pop()
        if current == predecessor_id:
            raise ValueError("Önceki/sonraki belge ilişkisi döngü oluşturuyor")
        if current not in visited:
            visited.add(current)
            pending.extend(graph.get(current, ()))


def _validate_selection(
    session: Session,
    batch: RegulatoryDocumentIntake,
    row: RegulatoryDocumentLineageReview,
    *,
    check_fingerprint: bool = True,
) -> None:
    member = _member(session, batch, row.intake_document_id)
    if (
        row.successor_user_file_id != member.user_file_id
        or row.document_set_id != batch.document_set_id
    ):
        raise ValueError("Belge ilişkisi kapsamı değişti")
    predecessor = _existing_file(
        session, batch.document_set_id, row.predecessor_user_file_id, lock=True
    )
    if check_fingerprint and _file_fingerprint(predecessor) != row.predecessor_sha256:
        raise ValueError("Önceki belge seçimden sonra değişti; ilişkiyi yeniden seçin")
    _validate_graph(
        session,
        batch.document_set_id,
        predecessor.id,
        member.user_file_id,
        exclude_document_id=member.id,
    )


def set_predecessor_selection(
    session: Session,
    batch: RegulatoryDocumentIntake,
    document_id: UUID,
    *,
    predecessor_user_file_id: UUID | None,
    relation: LineageRelation,
    effective_date: datetime.date | None,
    evidence: str | None,
) -> None:
    session.get_one(DocumentSet, batch.document_set_id, with_for_update=True)
    if batch.status != "ready" or batch.stage != "review":
        raise ValueError("Önceki belge yalnızca onay öncesi incelemede seçilebilir")
    member = _member(session, batch, document_id)
    row = session.get(
        RegulatoryDocumentLineageReview, document_id, with_for_update=True
    )
    if row is not None and row.status != "draft":
        raise ValueError("Onaylanmış belge ilişkisi değiştirilemez")
    if predecessor_user_file_id is None:
        if row is not None:
            session.delete(row)
            session.flush()
        return
    if relation not in {"successor", "repeals"}:
        raise ValueError("Geçersiz belge ilişkisi")
    evidence = evidence.strip() if evidence else None
    if evidence and len(evidence) > 4000:
        raise ValueError("Dayanak en fazla 4000 karakter olabilir")
    if relation == "repeals" and (effective_date is None or not evidence):
        raise ValueError("Mülga işareti için yürürlük tarihi ve açık dayanak gerekir")
    predecessor = _existing_file(
        session, batch.document_set_id, predecessor_user_file_id
    )
    same_intake = session.scalar(
        select(RegulatoryDocumentIntakeFile.id).where(
            RegulatoryDocumentIntakeFile.intake_id == batch.id,
            RegulatoryDocumentIntakeFile.user_file_id == predecessor.id,
        )
    )
    pending_intake = session.scalar(
        select(RegulatoryDocumentIntakeFile.id)
        .join(
            RegulatoryDocumentIntake,
            RegulatoryDocumentIntake.id == RegulatoryDocumentIntakeFile.intake_id,
        )
        .where(
            RegulatoryDocumentIntakeFile.user_file_id == predecessor.id,
            RegulatoryDocumentIntake.status != "completed",
        )
    )
    if same_intake or pending_intake:
        raise ValueError("Önceki belge tamamlanmış başka bir dosya olmalı")
    _validate_graph(
        session,
        batch.document_set_id,
        predecessor.id,
        member.user_file_id,
        exclude_document_id=member.id,
    )
    successor = session.get_one(UserFile, member.user_file_id)
    if row is None:
        row = RegulatoryDocumentLineageReview(intake_document_id=member.id)
        session.add(row)
    row.document_set_id = batch.document_set_id
    row.predecessor_user_file_id = predecessor.id
    row.successor_user_file_id = member.user_file_id
    row.predecessor_title, row.successor_title = predecessor.name, successor.name
    row.predecessor_sha256 = _file_fingerprint(predecessor)
    row.relation, row.effective_date, row.evidence = relation, effective_date, evidence
    row.status, row.confirmed_by = "draft", batch.created_by
    row.confirmed_at = datetime.datetime.now(datetime.timezone.utc)
    session.flush()


def _selection_snapshot(row: RegulatoryDocumentLineageReview) -> dict[str, Any]:
    return {
        "predecessor_user_file_id": str(row.predecessor_user_file_id),
        "title": row.predecessor_title,
        "relation": row.relation,
        "effective_date": row.effective_date.isoformat()
        if row.effective_date
        else None,
        "evidence": row.evidence,
        "status": row.status,
    }


def intake_lineage_selections(
    session: Session, batch: RegulatoryDocumentIntake
) -> dict[UUID, dict[str, Any]]:
    return {
        row.intake_document_id: _selection_snapshot(row)
        for row in _rows(session, batch)
    }


def lineage_review_payload(
    session: Session, batch: RegulatoryDocumentIntake
) -> list[dict[str, Any]]:
    payload = []
    for row in _rows(session, batch):
        file = session.get(UserFile, row.predecessor_user_file_id)
        member = session.get(
            DocumentSet__UserFile, (batch.document_set_id, row.predecessor_user_file_id)
        )
        selection = _selection_snapshot(row)
        selection.pop("status")
        payload.append(
            {
                **selection,
                "document_id": str(row.intake_document_id),
                "successor_user_file_id": str(row.successor_user_file_id),
                "selected_sha256": row.predecessor_sha256,
                "current_sha256": (_file_fingerprint(file) if file and member else None)
                if row.status == "draft"
                else row.predecessor_sha256,
                "confirmed_by": str(row.confirmed_by),
            }
        )
    return payload


def freeze_intake_lineage(session: Session, batch: RegulatoryDocumentIntake) -> None:
    session.get_one(DocumentSet, batch.document_set_id, with_for_update=True)
    rows = _rows(session, batch, lock=True)
    for row in rows:
        _validate_selection(session, batch, row)
    for row in rows:
        if row.status == "draft":
            row.status = "frozen"
            row.frozen_at = datetime.datetime.now(datetime.timezone.utc)
    session.flush()


def activate_intake_lineage(session: Session, batch: RegulatoryDocumentIntake) -> None:
    if batch.status != "completed":
        return
    session.get_one(DocumentSet, batch.document_set_id, with_for_update=True)
    rows = _rows(session, batch, lock=True)
    if not rows or all(row.status == "active" for row in rows):
        return
    members = list(
        session.scalars(
            select(RegulatoryDocumentIntakeFile).where(
                RegulatoryDocumentIntakeFile.intake_id == batch.id
            )
        )
    )
    if not members or any(
        session.get_one(UserFile, member.user_file_id).status
        != UserFileStatus.COMPLETED
        for member in members
    ):
        raise ValueError(
            "Belge ilişkisi tüm yeni dosyalar indekslendikten sonra etkinleşir"
        )
    for row in rows:
        if row.status != "frozen":
            raise ValueError("Belge ilişkisi onaylanmadı")
        _validate_selection(session, batch, row, check_fingerprint=False)
    for row in rows:
        row.status = "active"
        row.activated_at = datetime.datetime.now(datetime.timezone.utc)
    session.flush()


def reconcile_file_lineage(
    session: Session, document_set_id: int, file_id: UUID
) -> None:
    """Reading either side publishes its fully indexed, previously approved relation."""
    from onyx.db.document_intake import refresh_intake

    batch_ids = list(
        session.scalars(
            select(RegulatoryDocumentIntakeFile.intake_id)
            .join(
                RegulatoryDocumentLineageReview,
                RegulatoryDocumentLineageReview.intake_document_id
                == RegulatoryDocumentIntakeFile.id,
            )
            .where(
                RegulatoryDocumentLineageReview.document_set_id == document_set_id,
                RegulatoryDocumentLineageReview.status == "frozen",
                or_(
                    RegulatoryDocumentLineageReview.predecessor_user_file_id == file_id,
                    RegulatoryDocumentLineageReview.successor_user_file_id == file_id,
                ),
            )
            .distinct()
            .order_by(RegulatoryDocumentIntakeFile.intake_id)
        )
    )
    for batch_id in batch_ids:
        batch = session.get_one(
            RegulatoryDocumentIntake, batch_id, with_for_update=True
        )
        if batch.document_set_id != document_set_id:
            raise ValueError("Belge ilişkisi kapsamı değişti")
        refresh_intake(session, batch)


def reconcile_completed_intake_file(session: Session, file_id: UUID) -> None:
    """Complete only the admission owning this freshly published file."""
    from onyx.db.document_intake import refresh_intake

    batch_id = session.scalar(
        select(RegulatoryDocumentIntakeFile.intake_id).where(
            RegulatoryDocumentIntakeFile.user_file_id == file_id
        )
    )
    if batch_id is None:
        return
    batch = session.get_one(RegulatoryDocumentIntake, batch_id, with_for_update=True)
    if batch.stage == "indexing" and batch.status in {"approving", "failed"}:
        refresh_intake(session, batch)


def repealed_file_ids(
    session: Session,
    file_ids: tuple[UUID, ...] | None,
    *,
    as_of_date: datetime.date | None = None,
) -> frozenset[UUID]:
    """A confirmed whole-document repeal bounds legal query visibility, not storage."""
    if file_ids == ():
        return frozenset()
    statement = (
        select(RegulatoryDocumentLineageReview.predecessor_user_file_id)
        .where(
            RegulatoryDocumentLineageReview.status == "active",
            RegulatoryDocumentLineageReview.relation == "repeals",
            RegulatoryDocumentLineageReview.effective_date
            <= (as_of_date or datetime.date.today()),
        )
        .distinct()
    )
    if file_ids is not None:
        statement = statement.where(
            RegulatoryDocumentLineageReview.predecessor_user_file_id.in_(file_ids)
        )
    # Never silently truncate a confirmed repeal when building one ES terms filter.
    ids = frozenset(session.scalars(statement.limit(65_536)))
    if len(ids) > 65_535:
        raise ValueError("Document repeal query exceeds the safe terms-filter limit")
    return ids


def current_repealed_file_ids(
    file_ids: tuple[UUID, ...] | None = None, *, as_of_date: datetime.date | None = None
) -> frozenset[UUID]:
    from onyx.db.engine.sql_engine import get_session_with_current_tenant

    if file_ids == ():
        return frozenset()
    with get_session_with_current_tenant() as session:
        return repealed_file_ids(session, file_ids, as_of_date=as_of_date)


def file_lineage_snapshot(
    session: Session,
    document_set_id: int,
    file_id: UUID,
    as_of: datetime.date | None = None,
) -> dict[str, Any]:
    file = session.scalar(
        select(UserFile)
        .join(DocumentSet__UserFile, UserFile.id == DocumentSet__UserFile.user_file_id)
        .where(
            UserFile.id == file_id,
            DocumentSet__UserFile.document_set_id == document_set_id,
        )
    )
    if file is None:
        raise ValueError("Belge PC Külliyatı'nda bulunamadı")
    rows = list(
        session.scalars(
            select(RegulatoryDocumentLineageReview)
            .where(
                RegulatoryDocumentLineageReview.document_set_id == document_set_id,
                RegulatoryDocumentLineageReview.status == "active",
                or_(
                    RegulatoryDocumentLineageReview.predecessor_user_file_id == file_id,
                    RegulatoryDocumentLineageReview.successor_user_file_id == file_id,
                ),
            )
            .order_by(RegulatoryDocumentLineageReview.confirmed_at)
        )
    )
    as_of = as_of or datetime.datetime.now(datetime.timezone.utc).date()
    predecessors, successors = [], []
    repeal_dates = []
    for row in rows:
        is_previous = row.successor_user_file_id == file_id
        entry = {
            "user_file_id": str(
                row.predecessor_user_file_id
                if is_previous
                else row.successor_user_file_id
            ),
            "title": row.predecessor_title if is_previous else row.successor_title,
            "relation": row.relation,
            "effective_date": row.effective_date.isoformat()
            if row.effective_date
            else None,
            "evidence": row.evidence,
            "confirmed_by": str(row.confirmed_by),
            "confirmed_at": row.confirmed_at.isoformat(),
            "activated_at": row.activated_at.isoformat() if row.activated_at else None,
        }
        (predecessors if is_previous else successors).append(entry)
        if not is_previous and row.relation == "repeals" and row.effective_date:
            repeal_dates.append(row.effective_date)
    earliest = min(repeal_dates) if repeal_dates else None
    status = (
        "repealed"
        if earliest and earliest <= as_of
        else "repeal_scheduled"
        if earliest
        else "current"
    )
    return {
        "user_file_id": str(file.id),
        "title": file.name,
        "status": status,
        "effective_date": earliest.isoformat() if earliest else None,
        "predecessors": predecessors,
        "successors": successors,
    }
