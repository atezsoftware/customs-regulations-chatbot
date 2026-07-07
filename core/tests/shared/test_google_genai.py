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
