"""Admin discovery search: shared retrieval followed by a factual source overview."""

from uuid import uuid4

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from onyx.auth.permissions import require_permission
from onyx.configs.constants import PUBLIC_API_TAGS
from onyx.context.search.models import IndexFilters
from onyx.db.engine.sql_engine import get_session
from onyx.db.enums import Permission
from onyx.db.models import User
from onyx.regulatory.catalog_query import prepare_catalog_search_query
from onyx.regulatory.catalog_summary import (
    CATALOG_SUMMARY_MODEL,
    CatalogSummaryUnavailable,
    generate_catalog_summary,
)
from onyx.regulatory.catalog_summary_usage import CatalogSummaryUsage
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
from onyx.tracing.framework.create import ensure_trace
from onyx.utils.logger import setup_logger

router = APIRouter(prefix="/regulatory/catalog", tags=PUBLIC_API_TAGS)
logger = setup_logger()


@router.post("/simple-search", dependencies=[Depends(require_vector_db)])
def simple_search(
    request: SimpleSearchRequest,
    _: User = Depends(require_permission(Permission.FULL_ADMIN_PANEL_ACCESS)),
    db_session: Session = Depends(get_session),
) -> SimpleSearchResponse:
    return run_simple_search(request, db_session)


def run_simple_search(
    request: SimpleSearchRequest,
    db_session: Session,
    *,
    scope_filters: IndexFilters | None = None,
    overview_language: str = "tr",
) -> SimpleSearchResponse:
    request_id = uuid4()
    summary: str | None = None
    summary_error: str | None = None
    summary_error_code: str | None = None
    summary_usage: CatalogSummaryUsage | None = None

    def observe_usage(usage: CatalogSummaryUsage) -> None:
        nonlocal summary_usage
        summary_usage = usage

    with ensure_trace(
        "catalog_simple_search",
        trace_id=f"trace_{request_id.hex}",
        group_id=str(request_id),
        metadata={"request_id": str(request_id)},
    ):
        prepared_query = prepare_catalog_search_query(request.query, db_session)
        observation = observe_publication_read()
        search = search_catalog_chunks(
            CatalogSearchRequest(query=prepared_query.query),
            db_session,
            adaptive=True,
            scope_query=request.query,
            **({"scope_filters": scope_filters} if scope_filters is not None else {}),
        )
        try:
            summary = generate_catalog_summary(
                request.query,
                search,
                db_session,
                usage_observer=observe_usage,
                overview=True,
                **(
                    {"overview_language": overview_language}
                    if overview_language != "tr"
                    else {}
                ),
            )
        except CatalogSummaryUnavailable as error:
            summary_error = str(error)
            summary_error_code = error.code.value
        except Exception as error:
            # Provider errors may contain credentials or source text.
            logger.warning(
                "Catalog summary failed request_id=%s error_type=%s",
                request_id,
                type(error).__name__,
            )
            summary_error_code = "provider_error"
            summary_error = (
                "Nemotron özeti üretilemedi. Arama sonuçlarını inceleyebilirsiniz."
            )
        require_publication_files(
            observation, tuple(hit.file.id for hit in search.items)
        )
    logger.info(
        "Catalog simple search request_id=%s summary_error_code=%s "
        "summary_response_id=%s summary_input_tokens=%s summary_output_tokens=%s "
        "summary_provider_cost_usd=%s result_count=%d query_rewritten=%s",
        request_id,
        summary_error_code,
        summary_usage.response_id if summary_usage else None,
        summary_usage.input_tokens if summary_usage else None,
        summary_usage.output_tokens if summary_usage else None,
        summary_usage.provider_cost_usd if summary_usage else None,
        len(search.items),
        prepared_query.rewritten,
    )
    return SimpleSearchResponse(
        search=search,
        summary=summary,
        summary_model=CATALOG_SUMMARY_MODEL,
        summary_error=summary_error,
        search_query=prepared_query.query,
        query_rewritten=prepared_query.rewritten,
        request_id=request_id,
        summary_usage=summary_usage,
        summary_error_code=summary_error_code,
    )
