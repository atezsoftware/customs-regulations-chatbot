import json
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, Query, UploadFile
from sqlalchemy.orm import Session

from onyx.auth.permissions import require_permission
from onyx.auth.users import current_curator_or_admin_user
from onyx.background.celery.versioned_apps.client import app as client_app
from onyx.configs.app_configs import DISABLE_VECTOR_DB
from onyx.configs.constants import (
    CELERY_USER_FILE_PROCESSING_TASK_EXPIRES,
    OnyxCeleryPriority,
    OnyxCeleryQueues,
    OnyxCeleryTask,
)
from onyx.db.document_set import (
    check_document_sets_are_public,
    fetch_all_document_sets_for_user,
    fetch_user_file_counts_for_document_sets,
    fetch_user_files_for_document_set,
    get_document_set_by_id,
    get_document_set_by_id_for_user,
    get_user_file_for_document_set_management,
    insert_document_set,
    link_user_file_to_document_set,
    mark_document_set_as_to_be_deleted,
    unlink_user_file_from_document_set,
    update_document_set,
)
from onyx.db.document_set import delete_document_set as db_delete_document_set
from onyx.db.engine.sql_engine import get_session
from onyx.db.enums import Permission, UserFileStatus
from onyx.db.models import DocumentSet as DocumentSetDBModel
from onyx.db.models import User
from onyx.db.projects import upload_files_to_user_files_with_indexing
from onyx.db.regulatory_indexing_jobs import (
    fetch_latest_regulatory_indexing_progress_for_user_files,
)
from onyx.error_handling.error_codes import OnyxErrorCode
from onyx.error_handling.exceptions import OnyxError
from onyx.file_processing.archive_expansion import expand_archive_uploads
from onyx.file_processing.import_capability import ensure_markdown_import_available
from onyx.server.features.document_set.file_models import DocumentSetUserFileSnapshot
from onyx.server.features.document_set.models import (
    CheckDocSetPublicRequest,
    CheckDocSetPublicResponse,
    DocumentSetCreationRequest,
    DocumentSetSummary,
    DocumentSetUpdateRequest,
    IndexChunkedFilesResponse,
)
from onyx.server.features.projects.models import (
    CategorizedFilesSnapshot,
    UserFileSnapshot,
)
from onyx.server.features.projects.user_file_sync import (
    trigger_user_file_metadata_sync,
)
from onyx.utils.variable_functionality import fetch_ee_implementation_or_noop
from shared_configs.contextvars import get_current_tenant_id

router = APIRouter(prefix="/manage")


def _get_editable_document_set_or_raise(
    document_set_id: int, user: User, db_session: Session
) -> DocumentSetDBModel:
    document_set = get_document_set_by_id_for_user(
        db_session=db_session,
        document_set_id=document_set_id,
        user=user,
        get_editable=True,
    )
    if document_set is None:
        raise OnyxError(OnyxErrorCode.NOT_FOUND, "Document set not found")
    return document_set


@router.post("/admin/document-set")
def create_document_set(
    document_set_creation_request: DocumentSetCreationRequest,
    user: User = Depends(current_curator_or_admin_user),
    db_session: Session = Depends(get_session),
    tenant_id: str = Depends(get_current_tenant_id),
) -> int:
    fetch_ee_implementation_or_noop(
        "onyx.db.user_group", "validate_object_creation_for_user", None
    )(
        db_session=db_session,
        user=user,
        target_group_ids=document_set_creation_request.groups,
        object_is_public=document_set_creation_request.is_public,
        object_is_new=True,
    )
    try:
        document_set_db_model, _ = insert_document_set(
            document_set_creation_request=document_set_creation_request,
            user_id=user.id,
            db_session=db_session,
        )
    except Exception as e:
        raise OnyxError(OnyxErrorCode.INVALID_INPUT, str(e)) from e

    if not DISABLE_VECTOR_DB:
        client_app.send_task(
            OnyxCeleryTask.CHECK_FOR_VESPA_SYNC_TASK,
            kwargs={"tenant_id": tenant_id},
            priority=OnyxCeleryPriority.HIGH,
        )

    return document_set_db_model.id


@router.patch("/admin/document-set")
def patch_document_set(
    document_set_update_request: DocumentSetUpdateRequest,
    bg_tasks: BackgroundTasks,
    user: User = Depends(current_curator_or_admin_user),
    db_session: Session = Depends(get_session),
    tenant_id: str = Depends(get_current_tenant_id),
) -> None:
    document_set = get_document_set_by_id(db_session, document_set_update_request.id)
    if document_set is None:
        raise OnyxError(
            OnyxErrorCode.NOT_FOUND,
            f"Document set {document_set_update_request.id} does not exist",
        )

    fetch_ee_implementation_or_noop(
        "onyx.db.user_group", "validate_object_creation_for_user", None
    )(
        db_session=db_session,
        user=user,
        target_group_ids=document_set_update_request.groups,
        object_is_public=document_set_update_request.is_public,
        object_is_owned_by_user=user
        and (document_set.user_id is None or document_set.user_id == user.id),
    )
    try:
        _, _, user_file_ids_to_sync = update_document_set(
            document_set_update_request=document_set_update_request,
            db_session=db_session,
            user=user,
        )
    except Exception as e:
        raise OnyxError(OnyxErrorCode.INVALID_INPUT, str(e)) from e

    for user_file_id in user_file_ids_to_sync:
        trigger_user_file_metadata_sync(user_file_id, tenant_id, bg_tasks)

    if not DISABLE_VECTOR_DB:
        client_app.send_task(
            OnyxCeleryTask.CHECK_FOR_VESPA_SYNC_TASK,
            kwargs={"tenant_id": tenant_id},
            priority=OnyxCeleryPriority.HIGH,
        )


@router.delete("/admin/document-set/{document_set_id}")
def delete_document_set(
    document_set_id: int,
    bg_tasks: BackgroundTasks,
    user: User = Depends(current_curator_or_admin_user),
    db_session: Session = Depends(get_session),
    tenant_id: str = Depends(get_current_tenant_id),
) -> None:
    document_set = get_document_set_by_id(db_session, document_set_id)
    if document_set is None:
        raise OnyxError(
            OnyxErrorCode.NOT_FOUND,
            f"Document set {document_set_id} does not exist",
        )

    # check if the user has "edit" access to the document set.
    # `validate_object_creation_for_user` is poorly named, but this
    # is the right function to use here
    fetch_ee_implementation_or_noop(
        "onyx.db.user_group", "validate_object_creation_for_user", None
    )(
        db_session=db_session,
        user=user,
        object_is_public=document_set.is_public,
        object_is_owned_by_user=user
        and (document_set.user_id is None or document_set.user_id == user.id),
    )

    try:
        user_file_ids_to_sync = mark_document_set_as_to_be_deleted(
            db_session=db_session,
            document_set_id=document_set_id,
            user=user,
        )
    except Exception as e:
        raise OnyxError(OnyxErrorCode.INVALID_INPUT, str(e)) from e

    if DISABLE_VECTOR_DB:
        db_session.refresh(document_set)
        db_delete_document_set(document_set, db_session)
    else:
        client_app.send_task(
            OnyxCeleryTask.CHECK_FOR_VESPA_SYNC_TASK,
            kwargs={"tenant_id": tenant_id},
            priority=OnyxCeleryPriority.HIGH,
        )

    for user_file_id in user_file_ids_to_sync:
        trigger_user_file_metadata_sync(user_file_id, tenant_id, bg_tasks)


@router.get("/admin/document-set/{document_set_id}/files")
def list_document_set_files(
    document_set_id: int,
    user: User = Depends(require_permission(Permission.FULL_ADMIN_PANEL_ACCESS)),
    db_session: Session = Depends(get_session),
) -> list[DocumentSetUserFileSnapshot]:
    _get_editable_document_set_or_raise(document_set_id, user, db_session)
    user_files = fetch_user_files_for_document_set(db_session, document_set_id)
    progress_by_file_id = fetch_latest_regulatory_indexing_progress_for_user_files(
        db_session,
        user_file_ids=[user_file.id for user_file in user_files],
    )
    return [
        DocumentSetUserFileSnapshot.from_user_file(
            user_file,
            progress=progress_by_file_id.get(user_file.id),
        )
        for user_file in user_files
    ]


@router.post("/admin/document-set/{document_set_id}/file/upload")
def upload_document_set_files(
    document_set_id: int,
    bg_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
    temp_id_map: str | None = Form(None),
    user: User = Depends(require_permission(Permission.FULL_ADMIN_PANEL_ACCESS)),
    db_session: Session = Depends(get_session),
) -> CategorizedFilesSnapshot:
    # Markdown (and archives of it) need no source-document parsers, so the
    # lightweight runtime can accept them; other formats are refused per file
    # by categorize_uploaded_files.
    ensure_markdown_import_available()
    _get_editable_document_set_or_raise(document_set_id, user, db_session)

    parsed_temp_id_map: dict[str, str] | None = None
    if temp_id_map:
        try:
            parsed = json.loads(temp_id_map)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            parsed_temp_id_map = {str(key): str(value) for key, value in parsed.items()}

    # Admins bulk-load regulatory sources as a single archive; each entry has to
    # become its own UserFile so it is chunked and cited as a distinct document.
    categorized_files_result = upload_files_to_user_files_with_indexing(
        files=expand_archive_uploads(files),
        project_id=None,
        document_set_id=document_set_id,
        user=user,
        temp_id_map=parsed_temp_id_map,
        db_session=db_session,
        background_tasks=bg_tasks if DISABLE_VECTOR_DB else None,
    )
    return CategorizedFilesSnapshot.from_result(categorized_files_result)


def _enqueue_user_file_indexing(user_file_id: UUID, tenant_id: str) -> None:
    client_app.send_task(
        OnyxCeleryTask.INDEX_SINGLE_USER_FILE,
        kwargs={"user_file_id": str(user_file_id), "tenant_id": tenant_id},
        queue=OnyxCeleryQueues.USER_FILE_PROCESSING,
        priority=OnyxCeleryPriority.HIGH,
        expires=CELERY_USER_FILE_PROCESSING_TASK_EXPIRES,
    )


@router.post("/admin/document-set/{document_set_id}/files/{file_id}/index")
def index_document_set_file(
    document_set_id: int,
    file_id: UUID,
    user: User = Depends(require_permission(Permission.FULL_ADMIN_PANEL_ACCESS)),
    db_session: Session = Depends(get_session),
    tenant_id: str = Depends(get_current_tenant_id),
) -> UserFileSnapshot:
    """Project one reviewed file's chunks into the search index."""

    _get_editable_document_set_or_raise(document_set_id, user, db_session)
    user_file = get_user_file_for_document_set_management(db_session, file_id, user)
    if user_file is None:
        raise OnyxError(OnyxErrorCode.NOT_FOUND, "File not found")
    if user_file.status != UserFileStatus.CHUNKED:
        raise OnyxError(
            OnyxErrorCode.INVALID_INPUT,
            f"Only chunked files can be indexed; this one is {user_file.status.value}",
        )

    _enqueue_user_file_indexing(user_file.id, tenant_id)
    return UserFileSnapshot.from_model(user_file)


@router.post("/admin/document-set/{document_set_id}/index-chunked")
def index_document_set_chunked_files(
    document_set_id: int,
    user: User = Depends(require_permission(Permission.FULL_ADMIN_PANEL_ACCESS)),
    db_session: Session = Depends(get_session),
    tenant_id: str = Depends(get_current_tenant_id),
) -> IndexChunkedFilesResponse:
    """Index every reviewed file in the set, for bulk uploads."""

    _get_editable_document_set_or_raise(document_set_id, user, db_session)
    chunked_files = [
        user_file
        for user_file in fetch_user_files_for_document_set(db_session, document_set_id)
        if user_file.status == UserFileStatus.CHUNKED
    ]
    for user_file in chunked_files:
        _enqueue_user_file_indexing(user_file.id, tenant_id)

    return IndexChunkedFilesResponse(queued=len(chunked_files))


@router.post("/admin/document-set/{document_set_id}/files/{file_id}")
def link_document_set_file(
    document_set_id: int,
    file_id: UUID,
    bg_tasks: BackgroundTasks,
    user: User = Depends(require_permission(Permission.FULL_ADMIN_PANEL_ACCESS)),
    db_session: Session = Depends(get_session),
    tenant_id: str = Depends(get_current_tenant_id),
) -> UserFileSnapshot:
    _get_editable_document_set_or_raise(document_set_id, user, db_session)
    user_file = get_user_file_for_document_set_management(db_session, file_id, user)
    if user_file is None:
        raise OnyxError(OnyxErrorCode.NOT_FOUND, "File not found")

    if (
        link_user_file_to_document_set(db_session, document_set_id, user_file)
        and user_file.needs_document_set_sync
    ):
        trigger_user_file_metadata_sync(user_file.id, tenant_id, bg_tasks)
    return UserFileSnapshot.from_model(user_file)


@router.delete("/admin/document-set/{document_set_id}/files/{file_id}")
def unlink_document_set_file(
    document_set_id: int,
    file_id: UUID,
    bg_tasks: BackgroundTasks,
    user: User = Depends(require_permission(Permission.FULL_ADMIN_PANEL_ACCESS)),
    db_session: Session = Depends(get_session),
    tenant_id: str = Depends(get_current_tenant_id),
) -> None:
    _get_editable_document_set_or_raise(document_set_id, user, db_session)
    user_file = get_user_file_for_document_set_management(db_session, file_id, user)
    if user_file is None:
        raise OnyxError(OnyxErrorCode.NOT_FOUND, "File not found")

    if (
        unlink_user_file_from_document_set(db_session, document_set_id, user_file)
        and user_file.needs_document_set_sync
    ):
        trigger_user_file_metadata_sync(user_file.id, tenant_id, bg_tasks)


"""Endpoints for non-admins"""


@router.get("/document-set")
def list_document_sets_for_user(
    user: User = Depends(require_permission(Permission.BASIC_ACCESS)),
    db_session: Session = Depends(get_session),
    get_editable: bool = Query(
        False, description="If true, return editable document sets"
    ),
) -> list[DocumentSetSummary]:
    document_sets = fetch_all_document_sets_for_user(
        db_session=db_session, user=user, get_editable=get_editable
    )
    file_counts = fetch_user_file_counts_for_document_sets(
        db_session, [document_set.id for document_set in document_sets]
    )
    return [
        DocumentSetSummary.from_model(
            document_set,
            file_count=file_counts.get(document_set.id, 0),
        )
        for document_set in document_sets
    ]


@router.get("/document-set-public")
def document_set_public(
    check_public_request: CheckDocSetPublicRequest,
    _: User = Depends(require_permission(Permission.BASIC_ACCESS)),
    db_session: Session = Depends(get_session),
) -> CheckDocSetPublicResponse:
    is_public = check_document_sets_are_public(
        document_set_ids=check_public_request.document_set_ids, db_session=db_session
    )
    return CheckDocSetPublicResponse(is_public=is_public)
