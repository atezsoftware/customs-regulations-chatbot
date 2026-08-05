from unittest.mock import patch

from onyx.document_index.elasticsearch.client import ElasticsearchClient


def test_basic_auth_uses_elasticsearch_client_arguments() -> None:
    with patch("onyx.document_index.elasticsearch.client.Elasticsearch") as mock_client:
        ElasticsearchClient(
            host="elasticsearch-es-http",
            port=9200,
            auth=("elastic", "secret"),
            use_ssl=True,
        )

    assert mock_client.call_args.args == ("https://elasticsearch-es-http:9200",)
    assert mock_client.call_args.kwargs["basic_auth"] == ("elastic", "secret")
