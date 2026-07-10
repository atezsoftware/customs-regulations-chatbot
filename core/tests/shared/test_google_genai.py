from __future__ import annotations

from typing import Any

from fs_explorer_shared import google_genai


class _FakeClient:
    calls: list[dict[str, Any]] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.__class__.calls.append(kwargs)


def test_service_account_json_builds_vertex_client(monkeypatch) -> None:
    _FakeClient.calls.clear()
    monkeypatch.setattr("google.genai.Client", _FakeClient)
    monkeypatch.setattr(
        google_genai,
        "_load_service_account_credentials_json",
        lambda raw: ("fake-credentials", "json-project"),
    )
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.delenv("GOOGLE_PROJECT_ID", raising=False)
    monkeypatch.delenv("GCP_PROJECT", raising=False)
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS_JSON", '{"project_id":"x"}')
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "europe-west4")

    client = google_genai.build_genai_client()

    assert isinstance(client, _FakeClient)
    assert _FakeClient.calls == [
        {
            "vertexai": True,
            "credentials": "fake-credentials",
            "project": "json-project",
            "location": "europe-west4",
            "http_options": None,
        }
    ]


def test_parse_credentials_json_accepts_single_quoted_dict_literal() -> None:
    raw = "{'type': 'service_account', 'project_id': 'x'}"

    info = google_genai._parse_credentials_json(raw)

    assert info == {"type": "service_account", "project_id": "x"}


def test_parse_credentials_json_rejects_garbage() -> None:
    import json

    import pytest

    with pytest.raises(json.JSONDecodeError):
        google_genai._parse_credentials_json("not json at all")


def test_discrete_vertex_fields_build_vertex_client(monkeypatch) -> None:
    _FakeClient.calls.clear()
    monkeypatch.setattr("google.genai.Client", _FakeClient)
    monkeypatch.setattr(
        google_genai,
        "_load_service_account_credentials_from_discrete_fields",
        lambda email, key: ("fake-credentials", "discrete-project"),
    )
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS_JSON", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.delenv("GOOGLE_PROJECT_ID", raising=False)
    monkeypatch.delenv("GCP_PROJECT", raising=False)
    monkeypatch.setenv("GOOGLE_VERTEX_CLIENT_EMAIL", "sa@example.iam.gserviceaccount.com")
    monkeypatch.setenv("GOOGLE_VERTEX_PRIVATE_KEY", "-----BEGIN PRIVATE KEY-----\nabc\n-----END PRIVATE KEY-----\n")
    monkeypatch.setenv("GOOGLE_VERTEX_PROJECT", "vertex-project")
    monkeypatch.setenv("GOOGLE_VERTEX_LOCATION", "europe-west4")

    client = google_genai.build_genai_client()

    assert isinstance(client, _FakeClient)
    assert _FakeClient.calls == [
        {
            "vertexai": True,
            "credentials": "fake-credentials",
            "project": "vertex-project",
            "location": "europe-west4",
            "http_options": None,
        }
    ]


def test_normalize_private_key_unescapes_literal_newlines() -> None:
    raw = "-----BEGIN PRIVATE KEY-----\\nabc\\n-----END PRIVATE KEY-----\\n"

    normalized = google_genai._normalize_private_key(raw)

    assert normalized == "-----BEGIN PRIVATE KEY-----\nabc\n-----END PRIVATE KEY-----\n"


def test_normalize_private_key_leaves_real_newlines_untouched() -> None:
    raw = "-----BEGIN PRIVATE KEY-----\nabc\n-----END PRIVATE KEY-----\n"

    assert google_genai._normalize_private_key(raw) == raw


def test_api_key_fallback_builds_developer_client(monkeypatch) -> None:
    _FakeClient.calls.clear()
    monkeypatch.setattr("google.genai.Client", _FakeClient)
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS_JSON", raising=False)
    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

    client = google_genai.build_genai_client()

    assert isinstance(client, _FakeClient)
    assert _FakeClient.calls == [{"api_key": "test-key", "http_options": None}]
