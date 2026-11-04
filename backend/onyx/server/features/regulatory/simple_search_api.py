"""Admin discovery search: shared retrieval followed by a factual source overview."""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from onyx.auth.permissions import require_permission
from onyx.configs.constants import PUBLIC_API_TAGS
from onyx.db.engine.sql_engine import get_session
from onyx.db.enums import Permission
from onyx.db.models import User
from onyx.regulatory.catalog_summary import (
    CATALOG_SUMMARY_MODEL,
    CatalogSummaryUnavailable,
    generate_catalog_summary,
)
from onyx.regulatory.publication_reads import (
    observe_publication_read,
    require_publication_files,
)
from onyx.server.features.regulatory.catalog_api import search_catalog_chunks
from onyx.server.features.regulatory.catalog_models import CatalogSearchRequest
from onyx.server.features.regulatory.simple_search_models import (
    SimpleSearchRequest,
    SimpleSearchResponse,
)
from onyx.server.utils_vector_db import require_vector_db
from onyx.utils.logger import setup_logger

router = APIRouter(prefix="/regulatory/catalog", tags=PUBLIC_API_TAGS)
logger = setup_logger()


@router.post("/simple-search", dependencies=[Depends(require_vector_db)])
def simple_search(
    request: SimpleSearchRequest,
    _: User = Depends(require_permission(Permission.FULL_ADMIN_PANEL_ACCESS)),
    db_session: Session = Depends(get_session),
) -> SimpleSearchResponse:
    observation = observe_publication_read()
    search = search_catalog_chunks(
        CatalogSearchRequest(query=request.query), db_session
    )
    summary: str | None = None
    summary_error: str | None = None
    try:
        summary = generate_catalog_summary(request.query, search, db_session)
    except CatalogSummaryUnavailable as error:
        summary_error = str(error)
    except Exception as error:
        # Provider errors may contain credentials or source text.
        logger.warning("Catalog summary failed: %s", type(error).__name__)
        summary_error = (
            "Nemotron özeti üretilemedi. Arama sonuçlarını inceleyebilirsiniz."
        )
    require_publication_files(observation, tuple(hit.file.id for hit in search.items))
    return SimpleSearchResponse(
        search=search,
        summary=summary,
        summary_model=CATALOG_SUMMARY_MODEL,
        summary_error=summary_error,
    )
