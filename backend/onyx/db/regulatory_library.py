"""Read-only files within the same pinned, authorized corpus used by ASv3."""

from collections.abc import Sequence
from uuid import UUID

from sqlalchemy.orm import Session

from onyx.access.access import get_acl_for_user
from onyx.context.search.models import IndexFilters
from onyx.db.asv3_corpus import CorpusScopeUnavailable, resolve_pc_corpus_scope
from onyx.db.document_set import get_document_set_by_id_for_user
from onyx.db.models import User, UserFile
from onyx.db.regulatory_catalog import catalog_file_query
from onyx.error_handling.error_codes import OnyxErrorCode
from onyx.error_handling.exceptions import OnyxError
from shared_configs.contextvars import get_current_tenant_id


def library_scope(session: Session, user: User) -> IndexFilters:
    try:
        scope = resolve_pc_corpus_scope(
            session,
            user=user,
            filters=IndexFilters(
                access_control_list=list(get_acl_for_user(user, session)),
                tenant_id=get_current_tenant_id(),
                regulatory_chunks_only=True,
            ),
        )
    except (CorpusScopeUnavailable, PermissionError) as error:
        raise OnyxError(
            OnyxErrorCode.NOT_FOUND, "Legislation library unavailable"
        ) from error
    set_id = scope.asv3_document_set_id
    corpus = (
        get_document_set_by_id_for_user(session, set_id, user, get_editable=False)
        if set_id is not None
        else None
    )
    if corpus is None or not corpus.is_public:
        raise OnyxError(OnyxErrorCode.NOT_FOUND, "Legislation library unavailable")
    return scope


def scoped_library_file_ids(
    session: Session, document_set_id: int, ids: Sequence[UUID]
) -> set[UUID]:
    if not ids:
        return set()
    query = catalog_file_query(
        q="", folder=None, recursive=True, document_set_id=document_set_id
    )
    return set(
        session.scalars(
            query.with_only_columns(UserFile.id).where(UserFile.id.in_(ids))
        )
    )
