from datetime import date

import pytest
from pydantic import ValidationError

from onyx.server.features.search.models import SearchRequest
from onyx.tools.constants import REGULATORY_MAX_SEARCH_QUERY_CHARS


def test_search_request_keeps_general_search_defaults() -> None:
    request = SearchRequest(query="ordinary uploaded document")

    assert request.regulatory_chunks_only is False
    assert request.as_of_date is None
    assert request.search_mode == "hybrid"


def test_search_request_accepts_explicit_regulatory_scope() -> None:
    request = SearchRequest.model_validate(
        {
            "query": "Basel Sözleşmesi Madde 8",
            "regulatory_chunks_only": True,
            "as_of_date": "2026-08-02",
            "search_mode": "keyword",
        }
    )

    assert request.regulatory_chunks_only is True
    assert request.as_of_date == date(2026, 8, 2)
    assert request.search_mode == "keyword"


def test_search_request_rejects_oversized_regulatory_fragment() -> None:
    with pytest.raises(ValidationError, match="focused fragment"):
        SearchRequest(
            query="x" * (REGULATORY_MAX_SEARCH_QUERY_CHARS + 1),
            regulatory_chunks_only=True,
        )


def test_search_request_keeps_general_query_length_contract() -> None:
    query = "x" * (REGULATORY_MAX_SEARCH_QUERY_CHARS + 1)

    assert SearchRequest(query=query).query == query
