"""Legal query filters run before retrieval budgets; retained reads stay available."""

from copy import deepcopy
from datetime import date
from typing import Any, cast
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest
from sqlalchemy.orm import Session

from onyx.db import document_lineage, regulatory_public_reads
from onyx.document_index.elasticsearch.client import ElasticsearchIndexClient
from onyx.document_index.elasticsearch.constants import ElasticsearchSearchType
from onyx.document_index.publication_models import PublicationScope, ReadObservation
from onyx.regulatory import publication_reads
from tests.unit.onyx.document_index.elasticsearch.test_document_chunk_serialization import (
    _make_chunk,
)


@pytest.fixture
def reader(monkeypatch: pytest.MonkeyPatch) -> ElasticsearchIndexClient:
    client = ElasticsearchIndexClient.__new__(ElasticsearchIndexClient)
    client._index_name = "lineage-test"
    client._client = MagicMock()
    client._emit_metrics = False
    client._client.search.return_value = {
        "took": 1,
        "timed_out": False,
        "hits": {
            "hits": [
                {
                    "_id": "retained",
                    "_score": 1,
                    "_source": _make_chunk(None, None).model_dump(),
                }
            ]
        },
    }
    observation = ReadObservation(
        scope=PublicationScope(
            tenant_id="test", environment="test", database_identity="test"
        ),
        committed_epoch=0,
    )
    monkeypatch.setattr(
        publication_reads, "observe_publication_read", lambda: observation
    )
    monkeypatch.setattr(
        publication_reads,
        "filter_publication_read",
        lambda _observation, items, *_args, **_kwargs: list(items),
    )
    monkeypatch.setattr(
        regulatory_public_reads, "current_qualified_file_ids", lambda _ids: frozenset()
    )
    return client


@pytest.mark.parametrize(
    "purpose",
    [
        ElasticsearchSearchType.KEYWORD,
        ElasticsearchSearchType.SEMANTIC,
        ElasticsearchSearchType.HYBRID,
    ],
)
def test_search_purpose_injects_confirmed_repeal_before_backend_request(
    reader: ElasticsearchIndexClient,
    monkeypatch: pytest.MonkeyPatch,
    purpose: ElasticsearchSearchType,
) -> None:
    previous = uuid4()
    selector = MagicMock(return_value=frozenset({previous}))
    monkeypatch.setattr(document_lineage, "current_repealed_file_ids", selector)
    body = {"query": {"match": {"content": "kanun"}}, "size": 1}
    original = deepcopy(body)
    reader.search(body, None, search_type=purpose, as_of_date=date(2030, 1, 1))
    selector.assert_called_once_with(as_of_date=date(2030, 1, 1))
    query = cast(MagicMock, reader._client).search.call_args.kwargs["query"]
    assert query["bool"]["must_not"] == [{"terms": {"document_id": [str(previous)]}}]
    assert query["bool"]["must"] == [original["query"]]
    assert cast(MagicMock, reader._client).search.call_args.kwargs["size"] == 1
    assert body == original


@pytest.mark.parametrize(
    "purpose",
    [
        ElasticsearchSearchType.DOC_ID_RETRIEVAL,
        ElasticsearchSearchType.UNKNOWN,
        ElasticsearchSearchType.RANDOM,
    ],
)
def test_retained_citation_and_admin_reads_do_not_apply_current_repeal_gate(
    reader: ElasticsearchIndexClient,
    monkeypatch: pytest.MonkeyPatch,
    purpose: ElasticsearchSearchType,
) -> None:
    selector = MagicMock(
        side_effect=AssertionError(
            "retained read must not consult legal current visibility"
        )
    )
    monkeypatch.setattr(document_lineage, "current_repealed_file_ids", selector)
    body = {"query": {"match_all": {}}, "size": 1}
    result = reader.search(body, None, search_type=purpose)
    assert len(result) == 1
    assert (
        cast(MagicMock, reader._client).search.call_args.kwargs["query"]
        == body["query"]
    )
    selector.assert_not_called()


def test_metadata_database_error_fails_query_before_any_backend_read(
    reader: ElasticsearchIndexClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        document_lineage,
        "current_repealed_file_ids",
        MagicMock(side_effect=RuntimeError("database unavailable")),
    )
    with pytest.raises(RuntimeError, match="database unavailable"):
        reader.search(
            {"query": {"match_all": {}}},
            None,
            search_type=ElasticsearchSearchType.KEYWORD,
        )
    cast(MagicMock, reader._client).search.assert_not_called()


@pytest.mark.parametrize(
    "body",
    [
        {
            "knn": {
                "field": "content_vector",
                "query_vector": [1.0, 0.0],
                "filter": {"term": {"public": True}},
            }
        },
        {
            "knn": [
                {
                    "field": "content_vector",
                    "query_vector": [1.0, 0.0],
                    "filter": [{"term": {"public": True}}],
                }
            ]
        },
        {
            "query": {"match": {"content": "kanun"}},
            "knn": {"field": "content_vector", "query_vector": [1.0, 0.0]},
        },
    ],
)
def test_vector_and_union_lanes_each_keep_acl_and_exclude_repealed_document(
    body: dict[str, Any],
) -> None:
    original = deepcopy(body)
    filtered = ElasticsearchIndexClient._exclude_repealed_files(body, ("old-law",))
    knn = filtered["knn"][0] if isinstance(filtered["knn"], list) else filtered["knn"]
    assert knn["filter"]["bool"]["must_not"] == [
        {"terms": {"document_id": ["old-law"]}}
    ]
    if "query" not in original:
        assert "query" not in filtered
        assert knn["filter"]["bool"]["filter"] == [{"term": {"public": True}}]
    else:
        assert filtered["query"]["bool"]["must_not"] == [
            {"terms": {"document_id": ["old-law"]}}
        ]
    assert original == body


def test_hybrid_fusion_injects_same_prefilter_for_all_ranked_lanes() -> None:
    original = {
        "_onyx_hybrid_fusion": {
            "filters": [{"term": {"public": True}}],
            "subqueries": [],
        }
    }
    filtered = ElasticsearchIndexClient._exclude_repealed_files(original, ("old-law",))
    assert filtered["_onyx_hybrid_fusion"]["filters"] == [
        {"term": {"public": True}},
        {"bool": {"must_not": [{"terms": {"document_id": ["old-law"]}}]}},
    ]
    assert original["_onyx_hybrid_fusion"]["filters"] == [{"term": {"public": True}}]


def test_no_effective_repeal_leaves_existing_query_identical(
    reader: ElasticsearchIndexClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        document_lineage, "current_repealed_file_ids", lambda **_kwargs: frozenset()
    )
    body = {"query": {"match_all": {}}, "size": 1}
    reader.search(
        body,
        None,
        search_type=ElasticsearchSearchType.KEYWORD,
        as_of_date=date(2029, 12, 31),
    )
    assert (
        cast(MagicMock, reader._client).search.call_args.kwargs["query"]
        == body["query"]
    )


def test_repeal_inventory_overflow_fails_instead_of_silently_truncating() -> None:
    session = MagicMock(spec=Session)
    session.scalars.return_value = [UUID(int=index) for index in range(65_536)]
    with pytest.raises(ValueError, match="safe terms-filter limit"):
        document_lineage.repealed_file_ids(session, None, as_of_date=date(2030, 1, 1))
