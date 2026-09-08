"""Discover Google chat models using the provider's Vertex AI credentials."""

import json
import re

from onyx.error_handling.error_codes import OnyxErrorCode
from onyx.error_handling.exceptions import OnyxError
from onyx.llm.model_capabilities import (
    get_max_input_tokens,
    litellm_thinks_model_supports_image_input,
    model_is_reasoning_model,
)
from onyx.llm.well_known_providers.constants import (
    VERTEX_AUTH_METHOD_KWARG,
    VERTEX_AUTH_METHOD_SERVICE_ACCOUNT,
    VERTEX_AUTH_METHOD_WORKLOAD_IDENTITY,
    VERTEX_CREDENTIALS_FILE_KWARG,
    VERTEX_LOCATION_KWARG,
    VERTEX_PROJECT_KWARG,
    VERTEXAI_PROVIDER_NAME,
)
from onyx.server.manage.llm.models import VertexModelResponse
from shared_configs.configs import MULTI_TENANT


def vertex_connection_settings(config: dict[str, str]) -> tuple[str, str, str, str]:
    """Compare effective connection settings, including legacy/defaulted fields."""
    method = config.get(VERTEX_AUTH_METHOD_KWARG, VERTEX_AUTH_METHOD_SERVICE_ACCOUNT)
    credentials = config.get(VERTEX_CREDENTIALS_FILE_KWARG) or ""
    project = config.get(VERTEX_PROJECT_KWARG, "").strip()
    if method == VERTEX_AUTH_METHOD_SERVICE_ACCOUNT and not project:
        try:
            info = json.loads(credentials)
            if isinstance(info, dict):
                project = str(info.get("project_id") or "").strip()
        except (ValueError, TypeError):
            pass  # Credential validation belongs to discovery.
    return (
        method,
        credentials if method == VERTEX_AUTH_METHOD_SERVICE_ACCOUNT else "",
        project,
        config.get(VERTEX_LOCATION_KWARG, "").strip() or "global",
    )


def discover_vertex_models(custom_config: dict[str, str]) -> list[VertexModelResponse]:
    import google.auth
    import httpx
    from google import genai
    from google.auth.exceptions import GoogleAuthError
    from google.genai import types
    from google.genai.errors import APIError
    from google.oauth2 import service_account

    scopes = ["https://www.googleapis.com/auth/cloud-platform"]
    project = custom_config.get(VERTEX_PROJECT_KWARG, "").strip()
    location = custom_config.get(VERTEX_LOCATION_KWARG, "").strip() or "global"
    # The SDK interpolates this value into the credential-bearing request host.
    if len(location) > 63 or not re.fullmatch(
        r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*", location
    ):
        raise OnyxError(
            OnyxErrorCode.VALIDATION_ERROR, "Invalid Google Cloud region name."
        )
    auth_method = custom_config.get(
        VERTEX_AUTH_METHOD_KWARG, VERTEX_AUTH_METHOD_SERVICE_ACCOUNT
    )
    if auth_method not in (
        VERTEX_AUTH_METHOD_SERVICE_ACCOUNT,
        VERTEX_AUTH_METHOD_WORKLOAD_IDENTITY,
    ) or (auth_method == VERTEX_AUTH_METHOD_WORKLOAD_IDENTITY and MULTI_TENANT):
        raise OnyxError(
            OnyxErrorCode.VALIDATION_ERROR,
            "Unsupported Vertex AI authentication configuration.",
        )
    try:
        if auth_method == VERTEX_AUTH_METHOD_WORKLOAD_IDENTITY:
            credentials, _ = google.auth.default(scopes=scopes)
        else:
            info = json.loads(custom_config.get(VERTEX_CREDENTIALS_FILE_KWARG) or "{}")
            if not isinstance(info, dict):
                raise ValueError("Service account credentials must be a JSON object")
            credentials = service_account.Credentials.from_service_account_info(
                info, scopes=scopes
            )
            project = project or credentials.project_id or ""
        if not project:
            raise OnyxError(
                OnyxErrorCode.VALIDATION_ERROR, "GCP Project ID is required."
            )

        models: dict[str, VertexModelResponse] = {}
        with genai.Client(
            vertexai=True,
            credentials=credentials,
            project=project,
            location=location,
            http_options=types.HttpOptions(
                timeout=30_000,
                retry_options=types.HttpRetryOptions(attempts=1),
            ),
        ) as client:
            # Iteration follows every page; never supplement this result with
            # the bundled catalog, which can include unavailable model IDs.
            for model in client.models.list(
                config={"query_base": True, "page_size": 100}
            ):
                name = (model.name or "").removeprefix("publishers/google/models/")
                if not name.startswith("gemini-") or any(
                    term in name.lower()
                    for term in ("embedding", "image", "tts", "live", "native-audio")
                ):
                    continue
                models.setdefault(
                    name,
                    VertexModelResponse(
                        name=name,
                        display_name=model.display_name or name,
                        max_input_tokens=model.input_token_limit
                        or get_max_input_tokens(name, VERTEXAI_PROVIDER_NAME),
                        supports_image_input=litellm_thinks_model_supports_image_input(
                            name, VERTEXAI_PROVIDER_NAME
                        ),
                        supports_reasoning=model_is_reasoning_model(
                            name, VERTEXAI_PROVIDER_NAME
                        ),
                    ),
                )
        if not models:
            raise OnyxError(
                OnyxErrorCode.BAD_GATEWAY,
                "Google returned no Gemini chat models. Existing models have been retained.",
            )
        return sorted(models.values(), key=lambda model: model.name.lower())
    except (ValueError, KeyError, TypeError) as exc:
        raise OnyxError(
            OnyxErrorCode.VALIDATION_ERROR,
            "Invalid Google service account credentials. Upload a valid JSON key.",
        ) from exc
    except GoogleAuthError as exc:
        raise OnyxError(
            OnyxErrorCode.VALIDATION_ERROR,
            "Google authentication failed. Check the provider credentials or Workload Identity configuration.",
        ) from exc
    except APIError as exc:
        raise OnyxError(
            OnyxErrorCode.BAD_GATEWAY,
            "Google model discovery failed. Check Vertex AI permissions, API access, project and region.",
        ) from exc
    except httpx.HTTPError as exc:
        raise OnyxError(
            OnyxErrorCode.BAD_GATEWAY,
            "Could not reach Google to refresh models. Please try again.",
        ) from exc
