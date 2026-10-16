"""Validate reviewed insertion boundaries under the existing file ownership."""

from datetime import date
from typing import Any
from uuid import UUID

from sqlalchemy import Date, Text, and_, case, func, literal, select, update
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Session
from sqlalchemy.sql.elements import ColumnElement, SQLColumnExpression

from onyx.db.models import AmendmentProposal, RegulatoryChunk
from onyx.db.regulatory_chunks import make_regulatory_chunk_id
from onyx.document_index.publication_models import publication_digest
from onyx.regulatory.amendments.insertion_order import (
    InsertionOrder,
    OrderMember,
    is_direct_article_opening,
    plan_insertion,
    reconcile_insertion_order,
)
from onyx.regulatory.amendments.models import ReviewedAmendmentChunkDraft
from onyx.regulatory.chunk_evidence import RegulatoryChunkEvidence

_MAX_SIBLING_REBASE_OPERATIONS = 64


def _source_hash_sql(*values: SQLColumnExpression[Any]) -> ColumnElement[str]:
    return func.encode(
        func.sha256(
            func.convert_to(func.jsonb_build_array(*values).cast(Text), "UTF8")
        ),
        "hex",
    )


def _snapshot_source_hash(session: Session, snapshot: dict[str, Any]) -> str:
    return session.execute(
        select(
            _source_hash_sql(
                literal(snapshot["text"], Text),
                literal(snapshot["heading_path"], JSONB),
                literal(snapshot["metadata"], JSONB),
                literal(snapshot["chunk_type"], Text),
                literal(
                    date.fromisoformat(snapshot["validity_start_date"])
                    if snapshot["validity_start_date"] is not None
                    else None,
                    Date,
                ),
                literal(
                    date.fromisoformat(snapshot["validity_end_date"])
                    if snapshot["validity_end_date"] is not None
                    else None,
                    Date,
                ),
            )
        )
    ).scalar_one()


def load_amendment_order(session: Session, user_file_id: UUID) -> list[OrderMember]:
    records = session.execute(
        select(
            RegulatoryChunk.id,
            RegulatoryChunk.position,
            RegulatoryChunk.status,
            RegulatoryChunk.chunk_metadata["article_no"].astext,
            RegulatoryChunk.chunk_metadata["paragraph_no"].astext,
            RegulatoryChunk.chunk_metadata["clause_label"].astext,
            RegulatoryChunk.chunk_metadata["chunk_variant"].astext,
            RegulatoryChunk.chunk_metadata["bound_to_regulatory_chunk_id"].astext,
            _source_hash_sql(
                RegulatoryChunk.text,
                RegulatoryChunk.heading_path,
                RegulatoryChunk.chunk_metadata,
                RegulatoryChunk.chunk_type,
                RegulatoryChunk.validity_start_date,
                RegulatoryChunk.validity_end_date,
            ),
            case(
                (
                    and_(
                        RegulatoryChunk.chunk_metadata["article_no"].astext.is_not(
                            None
                        ),
                        RegulatoryChunk.chunk_metadata["paragraph_no"].astext.is_(None),
                        RegulatoryChunk.chunk_metadata["clause_label"].astext.is_(None),
                    ),
                    func.left(RegulatoryChunk.text, 512),
                ),
                else_=None,
            ),
        )
        .where(RegulatoryChunk.user_file_id == user_file_id)
        .order_by(RegulatoryChunk.position, RegulatoryChunk.id)
        .limit(100_001)
    ).all()
    if len(records) > 100_000:
        raise ValueError("Insertion source exceeds the bounded ordering budget")
    return [
        OrderMember(
            id=row[0],
            position=row[1],
            active=row[2] == "active",
            article_no=row[3],
            paragraph_no=row[4],
            clause_label=row[5],
            derived=row[6] == "hierarchical_aggregate" or row[7] is not None,
            source_sha256=row[8],
            direct_clause_parent=is_direct_article_opening(row[9], row[3]),
        )
        for row in records
    ]


def validate_insertion_order(
    session: Session,
    user_file_id: UUID,
    order: InsertionOrder,
    *,
    proposal: AmendmentProposal | None = None,
) -> InsertionOrder:
    members = load_amendment_order(session, user_file_id)
    current = plan_insertion(
        members,
        article_no=order.article_no,
        paragraph_no=order.paragraph_no,
        clause_label=order.clause_label,
        after_article_no=order.after_article_no,
    )
    try:
        return reconcile_insertion_order(order, current)
    except ValueError:
        if proposal is None or not _approved_siblings_reconstruct_reviewed_order(
            session, user_file_id, proposal, order, members
        ):
            raise
    return current


def _approved_siblings_reconstruct_reviewed_order(
    session: Session,
    user_file_id: UUID,
    proposal: AmendmentProposal,
    reviewed: InsertionOrder,
    members: list[OrderMember],
) -> bool:
    """Prove the frozen full baseline by undoing only attested sibling approvals."""
    if (
        proposal.status != "approving"
        or proposal.old_chunk_id is not None
        or len(proposal.chunk_changes or []) > 1
        or proposal.new_chunk_draft.get("user_file_id") != str(user_file_id)
        or proposal.new_chunk_draft.get("insertion_order")
        != reviewed.model_dump(mode="json")
    ):
        return False
    siblings = session.execute(
        select(AmendmentProposal, RegulatoryChunk)
        .join(
            RegulatoryChunk,
            RegulatoryChunk.id == AmendmentProposal.applied_new_chunk_id,
        )
        .where(
            AmendmentProposal.batch_id == proposal.batch_id,
            AmendmentProposal.id != proposal.id,
            AmendmentProposal.status == "approved",
            RegulatoryChunk.user_file_id == user_file_id,
        )
        .order_by(
            RegulatoryChunk.projection_ordinal.desc(), RegulatoryChunk.created_at.desc()
        )
        .limit(_MAX_SIBLING_REBASE_OPERATIONS + 1)
    ).all()
    if len(siblings) > _MAX_SIBLING_REBASE_OPERATIONS:
        return False
    reconstructed = {member.id: member for member in members}
    for sibling, applied in siblings:
        if any(
            sibling.new_chunk_draft.get(field) != proposal.new_chunk_draft.get(field)
            for field in ("effective_start_date", "effective_end_date")
        ):
            return False
        if not _reverse_approved_sibling(
            session, user_file_id, sibling, applied, reconstructed
        ):
            return False
        ordered = sorted(
            reconstructed.values(), key=lambda member: (member.position, member.id)
        )
        if (
            publication_digest([member.model_dump(mode="json") for member in ordered])
            == reviewed.baseline_sha256
        ):
            historical = plan_insertion(
                ordered,
                article_no=reviewed.article_no,
                paragraph_no=reviewed.paragraph_no,
                clause_label=reviewed.clause_label,
                after_article_no=reviewed.after_article_no,
            )
            reconcile_insertion_order(reviewed, historical)
            return True
    return False


def _reverse_approved_sibling(
    session: Session,
    user_file_id: UUID,
    sibling: AmendmentProposal,
    applied: RegulatoryChunk,
    members: dict[str, OrderMember],
) -> bool:
    applied_ids = sibling.applied_new_chunk_ids or []
    if (
        len(sibling.chunk_changes or []) > 1
        or (applied_ids and applied_ids != [applied.id])
        or applied.source != "amendment"
        or applied.status != "active"
        or applied.superseded_by_chunk_id is not None
        or applied.supersedes_chunk_id != sibling.old_chunk_id
    ):
        return False
    try:
        draft = ReviewedAmendmentChunkDraft.model_validate(
            sibling.new_chunk_draft
        ).model_dump(mode="json")
    except ValueError:
        return False
    if draft.get("user_file_id") != str(user_file_id) or not draft.get(
        "effective_start_date"
    ):
        return False
    if applied.id != make_regulatory_chunk_id(
        user_file_id,
        draft["position"],
        draft["text"],
        version_key=f"amendment:{sibling.id}",
    ):
        return False
    old = (
        session.get(RegulatoryChunk, sibling.old_chunk_id)
        if sibling.old_chunk_id
        else None
    )
    metadata = dict(draft.get("metadata") or {})
    for key in RegulatoryChunkEvidence.model_fields:
        metadata.pop(key, None)
        if old is not None and key in old.chunk_metadata:
            metadata[key] = old.chunk_metadata[key]
    metadata.setdefault("chunk_variant", "atomic")
    metadata.setdefault("source_chunk_orders", [])
    metadata.setdefault("source_regulatory_chunk_ids", [])
    if (
        applied.text != draft.get("text")
        or applied.heading_path != draft.get("heading_path")
        or applied.chunk_type != draft.get("chunk_type")
        or applied.chunk_metadata != metadata
    ):
        return False
    expected_hash = _snapshot_source_hash(
        session,
        {
            "text": draft["text"],
            "heading_path": draft["heading_path"],
            "metadata": metadata,
            "chunk_type": draft["chunk_type"],
            "validity_start_date": draft["effective_start_date"],
            "validity_end_date": draft.get("effective_end_date"),
        },
    )
    member = members.get(applied.id)
    if (
        member is None
        or not member.active
        or member.derived
        or member.source_sha256 != expected_hash
        or member.position != draft.get("position")
    ):
        return False
    if old is None:
        insertion = draft.get("insertion_order")
        if (
            sibling.old_chunk_id is not None
            or sibling.old_chunk_snapshot
            or not isinstance(insertion, dict)
        ):
            return False
        reviewed_insertion = InsertionOrder.model_validate(insertion)
        if (
            reviewed_insertion.position != member.position
            or reviewed_insertion.article_no != member.article_no
            or reviewed_insertion.paragraph_no != member.paragraph_no
            or reviewed_insertion.clause_label != member.clause_label
        ):
            return False
        del members[applied.id]
        for identifier, item in list(members.items()):
            if item.position >= member.position:
                members[identifier] = item.model_copy(
                    update={"position": item.position - 1}
                )
        return True
    snapshot = sibling.old_chunk_snapshot
    required = {
        "id",
        "user_file_id",
        "text",
        "chunk_type",
        "heading_path",
        "metadata",
        "validity_start_date",
        "validity_end_date",
        "status",
        "source",
        "supersedes_chunk_id",
        "superseded_by_chunk_id",
    }
    if (
        not required <= snapshot.keys()
        or snapshot["id"] != old.id
        or snapshot["user_file_id"] != str(user_file_id)
        or old.user_file_id != user_file_id
        or snapshot["status"] != "active"
        or snapshot["superseded_by_chunk_id"] is not None
        or old.status != "superseded"
        or old.superseded_by_chunk_id != applied.id
        or old.supersedes_chunk_id != snapshot["supersedes_chunk_id"]
        or old.text != snapshot["text"]
        or old.heading_path != snapshot["heading_path"]
        or old.chunk_type != snapshot["chunk_type"]
        or old.chunk_metadata != snapshot["metadata"]
        or old.source != snapshot["source"]
        or (old.validity_start_date.isoformat() if old.validity_start_date else None)
        != snapshot["validity_start_date"]
        or old.validity_end_date != applied.validity_start_date
        or sibling.old_chunk_snapshot.get("descendant_snapshots")
        or snapshot.get("replacement_scope")
        or snapshot.get("heading_change_scope")
    ):
        return False
    predecessor = members.get(old.id)
    if (
        predecessor is None
        or predecessor.active
        or predecessor.position != member.position
    ):
        return False
    try:
        predecessor_hash = _snapshot_source_hash(session, snapshot)
    except (TypeError, ValueError):
        return False
    del members[applied.id]
    members[old.id] = predecessor.model_copy(
        update={
            "active": True,
            "source_sha256": predecessor_hash,
        }
    )
    return True


def apply_insertion_order(
    session: Session, user_file_id: UUID, order: InsertionOrder, inserted_id: str
) -> None:
    # An order-preserving shift retains predecessor/successor alignment and leaves
    # physical projection ordinals and canonical identities untouched.
    session.execute(
        update(RegulatoryChunk)
        .where(
            RegulatoryChunk.user_file_id == user_file_id,
            RegulatoryChunk.position >= order.position,
            RegulatoryChunk.id != inserted_id,
        )
        .values(position=RegulatoryChunk.position + 1)
    )
    session.flush()
