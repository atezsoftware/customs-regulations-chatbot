"""Google GenAI client construction shared by API and indexer packages."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

_CLOUD_PLATFORM_SCOPE = "https://www.googleapis.com/auth/cloud-platform"
_DEFAULT_LOCATION = "global"
_VERTEX_API_VERSION = "v1"
_DEVELOPER_API_VERSION = "v1beta"


def _truthy_env(name: str) -> bool:
    value = os.getenv(name)
    return value is not None and value.strip().lower() in {"1", "true", "yes", "on"}


def _google_project_from_env() -> str | None:
    return (
        os.getenv("GOOGLE_CLOUD_PROJECT")
        or os.getenv("GOOGLE_PROJECT_ID")
        or os.getenv("GCP_PROJECT")
        or os.getenv("GOOGLE_VERTEX_PROJECT")
    )


def _google_location_from_env() -> str:
    return (
        os.getenv("GOOGLE_CLOUD_LOCATION")
        or os.getenv("GOOGLE_CLOUD_REGION")
        or os.getenv("GCP_REGION")
        or os.getenv("GOOGLE_VERTEX_LOCATION")
        or _DEFAULT_LOCATION
    )


def _normalize_private_key(raw: str) -> str:
    """Unescape a literal `\\n`-encoded PEM into real newlines, if needed.

    Some secret injectors (e.g. a Vault template rendering a JSON field
    verbatim) hand back the key with actual newlines already; others hand
    back a single-line value with literal backslash-n sequences. Only
    unescape when there's no real newline yet, so an already-correct
    multi-line key is left untouched.
    """
    if "\n" not in raw and "\\n" in raw:
        return raw.replace("\\n", "\n")
    return raw


def _load_service_account_credentials_from_discrete_fields(
    client_email: str, private_key: str
) -> tuple[Any, str | None]:
    from google.oauth2 import service_account

    info = {
        "type": "service_account",
        "client_email": client_email,
        "private_key": _normalize_private_key(private_key),
        "private_key_id": os.getenv("GOOGLE_VERTEX_PRIVATE_KEY_ID", ""),
        "token_uri": "https://oauth2.googleapis.com/token",
    }
    project_id = _google_project_from_env()
    if project_id:
        info["project_id"] = project_id
    credentials = service_account.Credentials.from_service_account_info(
        info,
        scopes=[_CLOUD_PLATFORM_SCOPE],
    )
    return credentials, getattr(credentials, "project_id", None)


def _load_service_account_credentials(path: str) -> tuple[Any, str | None]:
    from google.oauth2 import service_account

    credentials = service_account.Credentials.from_service_account_file(
        path,
        scopes=[_CLOUD_PLATFORM_SCOPE],
    )
    return credentials, getattr(credentials, "project_id", None)


def _parse_credentials_json(raw_json: str) -> dict[str, Any]:
    """Parse a service-account payload, tolerating single-quoted dict literals.

    Some secret stores hand back Python `str(dict)`-style values (single
    quotes around keys/strings) instead of strict JSON. `json.loads` rejects
    those with "Expecting property name enclosed in double quotes", so fall
    back to `ast.literal_eval` for that one case before giving up.
    """
    try:
        return json.loads(raw_json)
    except json.JSONDecodeError as json_error:
        import ast

        try:
            info = ast.literal_eval(raw_json)
        except (ValueError, SyntaxError):
            raise json_error from None
        if not isinstance(info, dict):
            raise ValueError(
                "GOOGLE_APPLICATION_CREDENTIALS_JSON did not parse to a JSON object"
            ) from None
        return info


def _load_service_account_credentials_json(raw_json: str) -> tuple[Any, str | None]:
    from google.oauth2 import service_account

    info = _parse_credentials_json(raw_json)
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

    Service-account/Vertex auth is preferred whenever `GOOGLE_APPLICATION_CREDENTIALS`,
    `GOOGLE_APPLICATION_CREDENTIALS_JSON`, or the discrete `GOOGLE_VERTEX_CLIENT_EMAIL`
    + `GOOGLE_VERTEX_PRIVATE_KEY` pair (the shape backend's Vault template injects) is
    set, or `GOOGLE_GENAI_USE_VERTEXAI=true` is configured. API key auth remains as a
    local fallback for older environments.
    """
    from google.genai import Client as GenAIClient
    from google.genai.types import HttpOptions

    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    credentials_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    vertex_client_email = os.getenv("GOOGLE_VERTEX_CLIENT_EMAIL")
    vertex_private_key = os.getenv("GOOGLE_VERTEX_PRIVATE_KEY")
    use_vertex = (
        bool(credentials_path)
        or bool(credentials_json)
        or bool(vertex_client_email and vertex_private_key)
        or _truthy_env("GOOGLE_GENAI_USE_VERTEXAI")
    )

    if use_vertex:
        if http_options is None:
            http_options = HttpOptions(api_version=_VERTEX_API_VERSION)
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
        elif vertex_client_email and vertex_private_key:
            credentials, credentials_project = (
                _load_service_account_credentials_from_discrete_fields(
                    vertex_client_email, vertex_private_key
                )
            )

        project = _google_project_from_env() or credentials_project
        if not project:
            raise ValueError(
                "Google Vertex AI auth requires GOOGLE_CLOUD_PROJECT, "
                "GOOGLE_PROJECT_ID, or a service-account JSON with project_id."
            )

        # Don't forward a caller-supplied http_options here: it's set up for
        # the Gemini Developer API (e.g. api_version="v1beta") and would
        # override the SDK's correct Vertex AI default (api_version
        # "v1beta1"), sending requests to a path that 404s.
        return GenAIClient(
            vertexai=True,
            credentials=credentials,
            project=project,
            location=_google_location_from_env(),
        )

    resolved_key = api_key or os.getenv("GOOGLE_API_KEY")
    if resolved_key:
        if http_options is None:
            http_options = HttpOptions(api_version=_DEVELOPER_API_VERSION)
        return GenAIClient(api_key=resolved_key, http_options=http_options)

    raise ValueError(
        "Google GenAI credentials not found. Set GOOGLE_APPLICATION_CREDENTIALS "
        "for service-account auth, or set GOOGLE_API_KEY for API-key auth."
    )
