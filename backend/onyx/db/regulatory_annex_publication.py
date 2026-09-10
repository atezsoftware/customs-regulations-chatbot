"""Short scoped database reads for final annex publication preparation."""

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from onyx.db.models import RegulatoryTemporalProjection, SearchSettings, UserFile
from onyx.regulatory.amendments.annexes.models import (
    AnnexChangeDraft,
    AnnexProjectionAccess,
    AnnexTemporalProjection,
)


def load_annex_publication_inputs(
    session: Session, draft: AnnexChangeDraft
) -> tuple[UserFile, list[SearchSettings], AnnexProjectionAccess]:
    from onyx.access.access import get_access_for_user_files
    from onyx.db.regulatory_amendments import get_batch
    from onyx.db.regulatory_annex_changes import validate_annex_review_scope
    from onyx.db.regulatory_annexes import require_annex_file_scope
    from onyx.db.search_settings import get_active_search_settings_list
    from onyx.db.user_file import (
        fetch_document_set_names_for_user_files,
        fetch_persona_ids_for_user_files,
        fetch_user_project_ids_for_user_files,
    )
    from onyx.regulatory.amendments.annexes.config import REGULATORY_ANNEX_ENVIRONMENT

    if draft.user_file_id is None or draft.batch_id is None:
        raise ValueError("publication preparation scope missing")
    batch = get_batch(session, draft.batch_id)
    if batch is None:
        raise ValueError("publication batch missing")
    validate_annex_review_scope(
        session, batch=batch, draft=draft, environment=REGULATORY_ANNEX_ENVIRONMENT
    )
    file = require_annex_file_scope(session, batch.document_set_id, draft.user_file_id)
    settings = get_active_search_settings_list(session)
    if sum(item.status.is_current() for item in settings) != 1:
        raise ValueError("publication requires one current index")
    identifier = str(file.id)
    access = get_access_for_user_files([identifier], session)
    return (
        file,
        settings,
        AnnexProjectionAccess(
            access=access[identifier],
            project_ids=fetch_user_project_ids_for_user_files(
                [identifier], session
            ).get(identifier, []),
            persona_ids=fetch_persona_ids_for_user_files([identifier], session).get(
                identifier, []
            ),
            document_sets=fetch_document_set_names_for_user_files(
                [identifier], session
            ).get(identifier, []),
        ),
    )


def load_file_temporal_bindings(
    session: Session, user_file_id: UUID
) -> list[AnnexTemporalProjection]:
    from onyx.document_index.publication_models import publication_digest

    bindings = []
    for row in session.scalars(
        select(RegulatoryTemporalProjection).where(
            RegulatoryTemporalProjection.user_file_id == user_file_id
        )
    ):
        if publication_digest(row.payload) != row.payload_sha256:
            raise ValueError("temporal binding payload changed")
        bindings.append(AnnexTemporalProjection.model_validate(row.payload))
    return bindings


def publication_input_scope_hash(
    file: UserFile, settings: list[SearchSettings], access: AnnexProjectionAccess
) -> str:
    """Bind every persisted index/file input and current ACL without exposing secrets."""
    from sqlalchemy import inspect

    from onyx.regulatory.amendments.annexes.context_dependencies import context_hash

    def columns(row: UserFile | SearchSettings) -> dict[str, object]:
        return {
            column.key: getattr(row, column.key)
            for column in inspect(type(row)).columns
            if column.key not in ("created_at", "updated_at", "last_accessed_at")
        }

    return context_hash(
        [
            columns(file),
            [columns(item) for item in sorted(settings, key=lambda item: item.id)],
            access.model_dump(mode="json"),
        ]
    )
