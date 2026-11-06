"""Request attribution and charged summary failures in admin simple search."""

from collections.abc import Callable
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from onyx.regulatory.catalog_query_models import CatalogQueryPreparation
from onyx.regulatory.catalog_summary import (
    CatalogSummaryRejectionCode,
    CatalogSummaryUnavailable,
)
from onyx.regulatory.catalog_summary_usage import CatalogSummaryUsage
from onyx.server.features.regulatory import simple_search_api as api
from onyx.server.features.regulatory.catalog_models import (
    CatalogFile,
    CatalogSearchHit,
    CatalogSearchPage,
)
from onyx.server.features.regulatory.models import RegulatoryChunkSnapshot
from onyx.server.features.regulatory.simple_search_models import SimpleSearchRequest
from onyx.tracing.framework.create import generation_span, get_current_trace
from onyx.tracing.framework.spans import NoOpSpan


@pytest.fixture
def page() -> CatalogSearchPage:
    now = datetime.now(timezone.utc)
    file_id = uuid4()
    return CatalogSearchPage(
        items=[
            CatalogSearchHit(
                file=CatalogFile(
                    id=file_id,
                    name="test.md",
                    folder_path="",
                    user_id=None,
                    file_id="original",
                    status="COMPLETED",
                    file_type="text/markdown",
                    created_at=now,
                    chunk_count=1,
                    document_sets=[],
                    pending_review=False,
                ),
                chunk=RegulatoryChunkSnapshot(
                    id="stable-source",
                    user_file_id=str(file_id),
                    text="Kaynak metni.",
                    position=1,
                    chunk_type="paragraph",
                    heading_path=[],
                    chunk_metadata={},
                    validity_start_date=None,
                    validity_end_date=None,
                    status="active",
                    source="indexed",
                    supersedes_chunk_id=None,
                    superseded_by_chunk_id=None,
                    created_at=now,
                    updated_at=now,
                ),
                score=0.9,
            )
        ],
        total=1,
        offset=0,
        limit=20,
        has_more=False,
    )


@pytest.mark.parametrize("rejected", [False, True])
def test_trace_and_provider_usage_survive_summary_rejection(
    page: CatalogSearchPage, rejected: bool
) -> None:
    expected = CatalogSummaryUsage(
        response_id="gen-original-response",
        input_tokens=8916,
        output_tokens=126,
        provider_cost_usd=0.0047352,
    )
    trace_ids: list[str] = []

    def check_trace() -> None:
        current = get_current_trace()
        assert current is not None
        trace_ids.append(current.trace_id)

    def prepare(query: str, _: object) -> CatalogQueryPreparation:
        check_trace()
        return CatalogQueryPreparation(query=query, rewritten=False)

    def search(*_: object, **_kwargs: object) -> CatalogSearchPage:
        check_trace()
        return page

    def generate(
        *_: object,
        usage_observer: Callable[[CatalogSummaryUsage], None],
        overview: bool = False,
    ) -> str:
        assert overview is True
        check_trace()
        with generation_span(model=api.CATALOG_SUMMARY_MODEL) as span:
            assert not isinstance(span, NoOpSpan)
            assert span.trace_id == trace_ids[-1]
        usage_observer(expected)
        if rejected:
            raise CatalogSummaryUnavailable(
                "Alıntı seçimi doğrulanamadı.",
                code=CatalogSummaryRejectionCode.UNKNOWN_PASSAGE,
            )
        return "[1] numaralı parça: “Kaynak metni.”"

    assert get_current_trace() is None
    with (
        patch.object(api, "prepare_catalog_search_query", side_effect=prepare),
        patch.object(api, "observe_publication_read"),
        patch.object(api, "search_catalog_chunks", side_effect=search),
        patch.object(api, "generate_catalog_summary", side_effect=generate),
        patch.object(api, "require_publication_files") as publication,
        patch.object(api, "logger") as logger,
    ):
        response = api.simple_search(
            SimpleSearchRequest(query="private original query"),
            MagicMock(),
            MagicMock(),
        )
    assert get_current_trace() is None
    assert trace_ids == [f"trace_{response.request_id.hex}"] * 3
    assert response.summary_usage == expected
    assert response.search is page
    assert response.summary_error_code == ("unknown_passage" if rejected else None)
    assert (response.summary is None) is rejected
    assert bool(response.summary_error) is rejected
    publication.assert_called_once()
    assert "private original query" not in str(logger.mock_calls)
    assert "gen-original-response" in str(logger.mock_calls)


def test_trace_is_closed_when_publication_changes(page: CatalogSearchPage) -> None:
    with (
        patch.object(api, "prepare_catalog_search_query") as prepare,
        patch.object(api, "observe_publication_read"),
        patch.object(api, "search_catalog_chunks", return_value=page),
        patch.object(api, "generate_catalog_summary", return_value="Kaynak [1]."),
        patch.object(
            api, "require_publication_files", side_effect=RuntimeError("changed")
        ),
    ):
        prepare.return_value = CatalogQueryPreparation(query="query", rewritten=False)
        with pytest.raises(RuntimeError, match="changed"):
            api.simple_search(
                SimpleSearchRequest(query="query"), MagicMock(), MagicMock()
            )
    assert get_current_trace() is None
