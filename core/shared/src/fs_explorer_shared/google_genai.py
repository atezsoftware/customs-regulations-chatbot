"""Google GenAI client construction shared by API and indexer packages."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

_CLOUD_PLATFORM_SCOPE = "https://www.googleapis.com/auth/cloud-platform"
_DEFAULT_LOCATION = "global"


def _truthy_env(name: str) -> bool:
    value = os.getenv(name)
    return value is not None and value.strip().lower() in {"1", "true", "yes", "on"}


def _google_project_from_env() -> str | None:
    return (
        os.getenv("GOOGLE_CLOUD_PROJECT")
        or os.getenv("GOOGLE_PROJECT_ID")
        or os.getenv("GCP_PROJECT")
    )


def _google_location_from_env() -> str:
    return (
        os.getenv("GOOGLE_CLOUD_LOCATION")
        or os.getenv("GOOGLE_CLOUD_REGION")
        or os.getenv("GCP_REGION")
        or _DEFAULT_LOCATION
    )


def _load_service_account_credentials(path: str) -> tuple[Any, str | None]:
    from google.oauth2 import service_account

    credentials = service_account.Credentials.from_service_account_file(
        path,
        scopes=[_CLOUD_PLATFORM_SCOPE],
    )
    return credentials, getattr(credentials, "project_id", None)


def _load_service_account_credentials_json(raw_json: str) -> tuple[Any, str | None]:
    from google.oauth2 import service_account

    info = json.loads(raw_json)
    credentials = service_account.Credentials.from_service_account_info(
        info,
        scopes=[_CLOUD_PLATFORM_SCOPE],
    )
    return credentials, getattr(credentials, "project_id", None)


def build_genai_client(
    *,
    api_key: str | None = None,
    http_options: Any | None = None,
) -> Any:
    """Build a Google GenAI client from service account/ADC or API key.

    Service-account/Vertex auth is preferred whenever `GOOGLE_APPLICATION_CREDENTIALS`
    is set or `GOOGLE_GENAI_USE_VERTEXAI=true` is configured. API key auth remains as
    a local fallback for older environments.
    """
    from google.genai import Client as GenAIClient

    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    credentials_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    use_vertex = (
        bool(credentials_path)
        or bool(credentials_json)
        or _truthy_env("GOOGLE_GENAI_USE_VERTEXAI")
    )

    if use_vertex:
        credentials = None
        credentials_project = None
        if credentials_path:
            path = Path(credentials_path).expanduser()
            credentials, credentials_project = _load_service_account_credentials(
                str(path)
            )
        elif credentials_json:
            credentials, credentials_project = _load_service_account_credentials_json(
                credentials_json
            )

        project = _google_project_from_env() or credentials_project
        if not project:
            raise ValueError(
                "Google Vertex AI auth requires GOOGLE_CLOUD_PROJECT, "
                "GOOGLE_PROJECT_ID, or a service-account JSON with project_id."
            )

        return GenAIClient(
            vertexai=True,
            credentials=credentials,
            project=project,
            location=_google_location_from_env(),
            http_options=http_options,
        )

    resolved_key = api_key or os.getenv("GOOGLE_API_KEY")
    if resolved_key:
        return GenAIClient(api_key=resolved_key, http_options=http_options)

    raise ValueError(
        "Google GenAI credentials not found. Set GOOGLE_APPLICATION_CREDENTIALS "
        "for service-account auth, or set GOOGLE_API_KEY for API-key auth."
    )
