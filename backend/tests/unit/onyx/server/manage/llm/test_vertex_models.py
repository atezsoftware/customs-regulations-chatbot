from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import httpx
import pytest
from google import genai
from google.auth.credentials import Credentials as GoogleCredentials
from google.genai import types
from google.oauth2.credentials import Credentials

from onyx.error_handling.exceptions import OnyxError
from onyx.server.manage.llm import api
from onyx.server.manage.llm.models import VertexModelsRequest


@pytest.fixture
def google_requests() -> Iterator[list[httpx.Request]]:
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.headers["authorization"] == "Bearer test-token"
        assert "publishers/google/models" in request.url.path
        if request.url.params.get("pageToken") == "next":
            return httpx.Response(
                200,
                json={
                    "publisherModels": [
                        {
                            "name": "publishers/google/models/gemini-future-flash",
                            "displayName": "Future Flash",
                        },
                        {"name": "publishers/google/models/gemini-current-pro"},
                    ]
                },
            )
        return httpx.Response(
            200,
            json={
                "publisherModels": [
                    {
                        "name": "publishers/google/models/gemini-current-pro",
                        "displayName": "Current Pro",
                    },
                    {"name": "publishers/google/models/text-embedding-005"},
                    {"name": "publishers/google/models/gemini-live-flash"},
                    {"name": "publishers/google/models/gemini-flash-image"},
                    {"name": "publishers/google/models/gemini-flash-tts"},
                ],
                "nextPageToken": "next",
            },
        )

    real_client = genai.Client

    def client_with_transport(
        *,
        vertexai: bool,
        credentials: GoogleCredentials,
        project: str,
        location: str,
        http_options: types.HttpOptions,
    ) -> genai.Client:
        return real_client(
            vertexai=vertexai,
            credentials=credentials,
            project=project,
            location=location,
            http_options=http_options.model_copy(
                update={"client_args": {"transport": httpx.MockTransport(respond)}}
            ),
        )

    with (
        patch.object(genai, "Client", side_effect=client_with_transport),
        patch(
            "google.auth.default",
            return_value=(Credentials("test-token"), "ambient-project"),
        ),
        patch.object(api, "MULTI_TENANT", False),
    ):
        yield requests


def test_refresh_uses_saved_provider_and_reads_all_pages(
    google_requests: list[httpx.Request],
) -> None:
    provider = MagicMock(
        provider="vertex_ai",
        name="Gemini",
        custom_config={
            "vertex_auth_method": "workload_identity",
            "vertex_project": "my-project",
            "vertex_location": "global",
        },
    )
    with (
        patch.object(api, "fetch_existing_llm_provider_by_id", return_value=provider),
        patch.object(api, "sync_vertex_model_configurations") as sync,
    ):
        models = api.get_vertex_available_models(
            VertexModelsRequest(provider_id=7), MagicMock(), MagicMock()
        )
    assert [m.name for m in models] == ["gemini-current-pro", "gemini-future-flash"]
    assert models[1].display_name == "Future Flash"
    assert len(google_requests) == 2
    assert all(
        request.url.host == "aiplatform.googleapis.com" for request in google_requests
    )
    assert [m.name for m in sync.call_args.args[3]] == [
        "gemini-current-pro",
        "gemini-future-flash",
    ]


def test_refresh_rejects_wrong_provider_without_google_request(
    google_requests: list[httpx.Request],
) -> None:
    with patch.object(
        api,
        "fetch_existing_llm_provider_by_id",
        return_value=MagicMock(provider="openai"),
    ):
        with pytest.raises(OnyxError) as error:
            api.get_vertex_available_models(
                VertexModelsRequest(provider_id=7), MagicMock(), MagicMock()
            )
    assert error.value.status_code == 400
    assert google_requests == []


def test_preview_of_edited_connection_does_not_change_saved_models(
    google_requests: list[httpx.Request],
) -> None:
    provider = MagicMock(
        provider="vertex_ai",
        custom_config={
            "vertex_auth_method": "workload_identity",
            "vertex_project": "saved-project",
        },
    )
    with (
        patch.object(api, "fetch_existing_llm_provider_by_id", return_value=provider),
        patch.object(api, "sync_vertex_model_configurations") as sync,
    ):
        models = api.get_vertex_available_models(
            VertexModelsRequest(
                provider_id=7,
                custom_config={
                    "vertex_auth_method": "workload_identity",
                    "vertex_project": "unsaved-project",
                },
            ),
            MagicMock(),
            MagicMock(),
        )
    assert len(models) == 2
    assert len(google_requests) == 2
    sync.assert_not_called()


def test_refresh_reports_google_failure_without_syncing() -> None:
    from google.genai.errors import ClientError

    with (
        patch.object(genai, "Client") as client,
        patch("google.auth.default", return_value=(Credentials("test-token"), None)),
        patch.object(api, "MULTI_TENANT", False),
        patch.object(api, "sync_vertex_model_configurations") as sync,
    ):
        client.return_value.__enter__.return_value.models.list.side_effect = (
            ClientError(
                403,
                {
                    "error": {
                        "message": "private upstream details",
                        "status": "PERMISSION_DENIED",
                    }
                },
            )
        )
        with pytest.raises(OnyxError) as error:
            api.get_vertex_available_models(
                VertexModelsRequest(
                    custom_config={
                        "vertex_auth_method": "workload_identity",
                        "vertex_project": "my-project",
                    }
                ),
                MagicMock(),
                MagicMock(),
            )
    assert "permissions" in error.value.detail.lower()
    assert "private upstream details" not in error.value.detail
    sync.assert_not_called()


def test_refresh_restores_masked_service_account_credentials(
    google_requests: list[httpx.Request],
) -> None:
    provider = MagicMock(
        provider="vertex_ai",
        custom_config={"vertex_credentials": '{"project_id":"saved-project"}'},
    )
    with (
        patch.object(api, "fetch_existing_llm_provider_by_id", return_value=provider),
        patch.object(api, "sync_vertex_model_configurations"),
        patch(
            "google.oauth2.service_account.Credentials.from_service_account_info",
            return_value=Credentials("test-token"),
        ) as parse,
    ):
        models = api.get_vertex_available_models(
            VertexModelsRequest(
                provider_id=7,
                custom_config={
                    "vertex_credentials": "****",
                    "vertex_project": "saved-project",
                    "vertex_location": "europe-west4",
                },
            ),
            MagicMock(),
            MagicMock(),
        )
    assert len(models) == 2
    assert parse.call_args.args[0] == {"project_id": "saved-project"}
    assert all(
        request.url.host == "europe-west4-aiplatform.googleapis.com"
        for request in google_requests
    )


def test_workload_identity_is_rejected_in_multitenant_mode() -> None:
    with patch.object(api, "MULTI_TENANT", True):
        with pytest.raises(OnyxError) as error:
            api.get_vertex_available_models(
                VertexModelsRequest(
                    custom_config={
                        "vertex_auth_method": "workload_identity",
                        "vertex_project": "my-project",
                    }
                ),
                MagicMock(),
                MagicMock(),
            )
    assert error.value.status_code == 400


@pytest.mark.parametrize(
    "location",
    [
        "example.com/anything#",
        "us-central1?x=1",
        "global@evil.com",
        "https://example.com",
    ],
)
def test_invalid_region_cannot_redirect_google_credentials(location: str) -> None:
    with (
        patch.object(api, "MULTI_TENANT", False),
        patch.object(genai, "Client") as client,
    ):
        with pytest.raises(OnyxError) as error:
            api.get_vertex_available_models(
                VertexModelsRequest(
                    custom_config={
                        "vertex_auth_method": "workload_identity",
                        "vertex_project": "my-project",
                        "vertex_location": location,
                    }
                ),
                MagicMock(),
                MagicMock(),
            )
    assert "region" in error.value.detail.lower()
    client.assert_not_called()
