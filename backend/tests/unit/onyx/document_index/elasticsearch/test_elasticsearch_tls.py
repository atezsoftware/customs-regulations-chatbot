import importlib
import os
from unittest.mock import patch

import certifi
import pytest

from onyx.document_index.elasticsearch.client import ElasticsearchClient

_OS_TLS_ENV = (
    "ELASTICSEARCH_VERIFY_CERTS",
    "ELASTICSEARCH_CA_CERTS",
    "ELASTICSEARCH_CLIENT_CERT",
    "ELASTICSEARCH_CLIENT_KEY",
)


# --- client wiring --------------------------------------------------------


def test_client_forwards_tls_kwargs() -> None:
    """The TLS settings must reach the underlying elasticsearch-py client — today
    verify_certs/ca_certs/client_cert/client_key aren't plumbed through at all."""
    with patch("onyx.document_index.elasticsearch.client.Elasticsearch") as mock_os:
        ElasticsearchClient(
            host="h",
            port=9200,
            use_ssl=True,
            verify_certs=True,
            ca_certs="/etc/ssl/os-ca.pem",
            client_cert="/etc/ssl/os-client.crt",
            client_key="/etc/ssl/os-client.key",
        )
        kwargs = mock_os.call_args.kwargs
        assert mock_os.call_args.args == ("https://h:9200",)
        assert kwargs["verify_certs"] is True
        assert kwargs["ca_certs"] == "/etc/ssl/os-ca.pem"
        assert kwargs["client_cert"] == "/etc/ssl/os-client.crt"
        assert kwargs["client_key"] == "/etc/ssl/os-client.key"


def test_client_defaults_to_no_verification() -> None:
    """Back-compat: verification stays off by default so the bundled
    self-signed Elasticsearch keeps working without opt-in config."""
    with patch("onyx.document_index.elasticsearch.client.Elasticsearch") as mock_os:
        ElasticsearchClient()
        assert mock_os.call_args.kwargs["verify_certs"] is False


# --- config validation ----------------------------------------------------


def _clear_os_tls_env() -> None:
    for var in _OS_TLS_ENV:
        os.environ.pop(var, None)


def test_client_cert_without_key_raises() -> None:
    with patch.dict(os.environ, {}, clear=False):
        _clear_os_tls_env()
        os.environ["ELASTICSEARCH_CLIENT_CERT"] = certifi.where()
        import onyx.configs.app_configs as app_configs

        with pytest.raises(ValueError, match="must both be set"):
            importlib.reload(app_configs)
    importlib.reload(app_configs)


def test_nonexistent_ca_raises() -> None:
    with patch.dict(os.environ, {}, clear=False):
        _clear_os_tls_env()
        os.environ["ELASTICSEARCH_CA_CERTS"] = "/no/such/os-ca.pem"
        import onyx.configs.app_configs as app_configs

        with pytest.raises(ValueError, match="does not exist"):
            importlib.reload(app_configs)
    importlib.reload(app_configs)
