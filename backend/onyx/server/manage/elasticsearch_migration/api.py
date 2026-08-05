from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from onyx.auth.permissions import require_permission
from onyx.configs.app_configs import ONYX_DISABLE_VESPA
from onyx.db.elasticsearch_migration import (
    get_elasticsearch_migration_state,
    get_elasticsearch_retrieval_state,
    set_enable_elasticsearch_retrieval_with_commit,
)
from onyx.db.engine.sql_engine import get_session
from onyx.db.enums import Permission
from onyx.db.models import User
from onyx.server.manage.elasticsearch_migration.models import (
    ElasticsearchMigrationStatusResponse,
    ElasticsearchRetrievalStatusRequest,
    ElasticsearchRetrievalStatusResponse,
)

admin_router = APIRouter(prefix="/admin/elasticsearch-migration")


@admin_router.get("/status")
def get_elasticsearch_migration_status(
    _: User = Depends(require_permission(Permission.FULL_ADMIN_PANEL_ACCESS)),
    db_session: Session = Depends(get_session),
) -> ElasticsearchMigrationStatusResponse:
    (
        total_chunks_migrated,
        created_at,
        migration_completed_at,
        approx_chunk_count_in_vespa,
    ) = get_elasticsearch_migration_state(db_session)
    return ElasticsearchMigrationStatusResponse(
        total_chunks_migrated=total_chunks_migrated,
        created_at=created_at,
        migration_completed_at=migration_completed_at,
        approx_chunk_count_in_vespa=approx_chunk_count_in_vespa,
    )


@admin_router.get("/retrieval")
def get_elasticsearch_retrieval_status(
    _: User = Depends(require_permission(Permission.FULL_ADMIN_PANEL_ACCESS)),
    db_session: Session = Depends(get_session),
) -> ElasticsearchRetrievalStatusResponse:
    enable_elasticsearch_retrieval = get_elasticsearch_retrieval_state(db_session)
    return ElasticsearchRetrievalStatusResponse(
        enable_elasticsearch_retrieval=enable_elasticsearch_retrieval,
        toggling_retrieval_is_disabled=ONYX_DISABLE_VESPA,
    )


@admin_router.put("/retrieval")
def set_elasticsearch_retrieval_status(
    request: ElasticsearchRetrievalStatusRequest,
    _: User = Depends(require_permission(Permission.FULL_ADMIN_PANEL_ACCESS)),
    db_session: Session = Depends(get_session),
) -> ElasticsearchRetrievalStatusResponse:
    set_enable_elasticsearch_retrieval_with_commit(
        db_session, request.enable_elasticsearch_retrieval
    )
    return ElasticsearchRetrievalStatusResponse(
        enable_elasticsearch_retrieval=request.enable_elasticsearch_retrieval,
        toggling_retrieval_is_disabled=ONYX_DISABLE_VESPA,
    )
