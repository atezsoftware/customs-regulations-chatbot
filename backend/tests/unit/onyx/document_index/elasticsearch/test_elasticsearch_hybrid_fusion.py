from unittest.mock import MagicMock

from onyx.document_index.elasticsearch.client import ElasticsearchIndexClient


def test_zscore_hybrid_fusion_preserves_weights_and_candidate_union() -> None:
    client = ElasticsearchIndexClient.__new__(ElasticsearchIndexClient)
    client._index_name = "test-index"
    client._client = MagicMock()
    client._client.search.side_effect = [
        {
            "took": 3,
            "timed_out": False,
            "hits": {
                "hits": [
                    {"_id": "a", "_score": 3.0, "_source": {"content": "a"}},
                    {"_id": "b", "_score": 1.0, "_source": {"content": "b"}},
                ]
            },
        },
        {
            "took": 4,
            "timed_out": False,
            "hits": {
                "hits": [
                    {
                        "_id": "b",
                        "_score": 4.0,
                        "_source": {"content": "b"},
                        "highlight": {"content": ["<em>b</em>"]},
                    },
                    {"_id": "c", "_score": 2.0, "_source": {"content": "c"}},
                ]
            },
        },
    ]

    result = client._search_zscore_hybrid(
        {
            "_onyx_zscore_hybrid": {
                "subqueries": [
                    {"knn": {"field": "content_vector", "query_vector": [0.1]}},
                    {"match": {"content": "query"}},
                ],
                "weights": [0.5, 0.5],
                "filters": [{"term": {"hidden": False}}],
                "rank_window_size": 2,
            },
            "timeout": "50s",
            "_source": {"excludes": ["content_vector"]},
            "size": 3,
        }
    )

    hits = result["hits"]["hits"]
    assert [hit["_id"] for hit in hits] == ["a", "b", "c"]
    assert [hit["_score"] for hit in hits] == [0.5, 0.0, -0.5]
    assert hits[1]["highlight"] == {"content": ["<em>b</em>"]}
    assert result["took"] == 7
    assert result["timed_out"] is False

    vector_request = client._client.search.call_args_list[0].kwargs["body"]
    lexical_request = client._client.search.call_args_list[1].kwargs["body"]
    assert vector_request["query"]["knn"]["filter"] == {
        "bool": {"filter": [{"term": {"hidden": False}}]}
    }
    assert lexical_request["query"]["bool"]["filter"] == [{"term": {"hidden": False}}]
