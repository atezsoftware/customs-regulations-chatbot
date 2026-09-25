"""Validate reviewed insertion boundaries under the existing file ownership."""

from uuid import UUID

from sqlalchemy import Text, and_, case, func, select, update
from sqlalchemy.orm import Session

from onyx.db.models import RegulatoryChunk
from onyx.regulatory.amendments.insertion_order import (
    InsertionOrder,
    OrderMember,
    is_direct_article_opening,
    plan_insertion,
)


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
            func.encode(
                func.sha256(
                    func.convert_to(
                        func.jsonb_build_array(
                            RegulatoryChunk.text,
                            RegulatoryChunk.heading_path,
                            RegulatoryChunk.chunk_metadata,
                            RegulatoryChunk.chunk_type,
                            RegulatoryChunk.validity_start_date,
                            RegulatoryChunk.validity_end_date,
                        ).cast(Text),
                        "UTF8",
                    )
                ),
                "hex",
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
    session: Session, user_file_id: UUID, order: InsertionOrder
) -> None:
    current = plan_insertion(
        load_amendment_order(session, user_file_id),
        article_no=order.article_no,
        paragraph_no=order.paragraph_no,
        clause_label=order.clause_label,
    )
    if current != order:
        raise ValueError(
            "Insertion source order changed after review; reanalyze before approval"
        )


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
