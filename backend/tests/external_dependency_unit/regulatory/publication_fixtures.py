"""Committed, precisely owned fixtures for independent publication lease sessions."""

from collections.abc import Generator
from contextlib import contextmanager

from sqlalchemy import delete, event, select
from sqlalchemy.orm import Session
from sqlalchemy.sql import column

from onyx.db.engine.sql_engine import get_sqlalchemy_engine
from onyx.db.models import (
    AmendmentBatch,
    AmendmentSourcePackage,
    AnnexChangeSet,
    AnnexPublicationIntent,
    DocumentSet,
    RegulatoryFilePublication,
    RegulatoryPublicationOrdinal,
    RegulatoryTemporalProjection,
    User,
    UserFile,
)


@contextmanager
def committed_review_session() -> Generator[Session, None, None]:
    created: list[object] = []
    with Session(get_sqlalchemy_engine()) as session:

        def remember(session: Session, *_args: object) -> None:
            created.extend(
                item
                for item in session.new
                if isinstance(item, (DocumentSet, User, UserFile))
            )

        event.listen(session, "before_flush", remember)
        try:
            yield session
        finally:
            session.rollback()
            groups = [item.id for item in created if isinstance(item, DocumentSet)]
            files = [item.id for item in created if isinstance(item, UserFile)]
            users = [item.id for item in created if isinstance(item, User)]
            changes = (
                select(AnnexChangeSet.id)
                .join(AmendmentBatch)
                .where(AmendmentBatch.document_set_id.in_(groups))
            )
            session.execute(
                delete(AnnexPublicationIntent).where(
                    AnnexPublicationIntent.change_set_id.in_(changes)
                )
            )
            session.execute(
                delete(AmendmentBatch).where(AmendmentBatch.document_set_id.in_(groups))
            )
            session.execute(
                delete(RegulatoryTemporalProjection).where(
                    RegulatoryTemporalProjection.user_file_id.in_(files)
                )
            )
            session.execute(
                delete(RegulatoryPublicationOrdinal).where(
                    RegulatoryPublicationOrdinal.user_file_id.in_(files)
                )
            )
            session.execute(
                delete(RegulatoryFilePublication).where(
                    RegulatoryFilePublication.user_file_id.in_(files)
                )
            )
            session.execute(delete(UserFile).where(UserFile.id.in_(files)))
            session.execute(
                delete(AmendmentSourcePackage).where(
                    AmendmentSourcePackage.document_set_id.in_(groups)
                )
            )
            session.execute(delete(DocumentSet).where(DocumentSet.id.in_(groups)))
            session.execute(delete(User).where(column("id").in_(users)))
            session.commit()
