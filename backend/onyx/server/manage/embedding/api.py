import math
from typing import cast

import httpx
from fastapi import APIRouter, Depends
from pydantic import ValidationError
from sqlalchemy.orm import Session

from onyx.auth.permissions import require_permission
from onyx.db.engine.sql_engine import get_session
from onyx.db.enums import Permission
from onyx.db.llm import (
    fetch_existing_embedding_providers,
    remove_embedding_provider,
    upsert_cloud_embedding_provider,
)
from onyx.db.models import User
from onyx.db.search_settings import (
    get_all_search_settings,
    get_current_db_embedding_provider,
    get_embedding_provider_from_provider_type,
)
from onyx.error_handling.error_codes import OnyxErrorCode
from onyx.error_handling.exceptions import OnyxError
from onyx.indexing.models import EmbeddingModelDetail
from onyx.natural_language_processing.constants import (
    OPENROUTER_EMBEDDING_MODELS_URL,
    OPENROUTER_EMBEDDINGS_URL,
)
from onyx.natural_language_processing.search_nlp_models import EmbeddingModel
from onyx.server.manage.embedding.models import (
    CloudEmbeddingProvider,
    CloudEmbeddingProviderCreationRequest,
    OpenRouterEmbeddingModelDetails,
    OpenRouterEmbeddingModelResponse,
    OpenRouterEmbeddingModelsRequest,
    TestEmbeddingRequest,
    TestEmbeddingResponse,
)
from onyx.utils.logger import setup_logger
from shared_configs.configs import MODEL_SERVER_HOST, MODEL_SERVER_PORT
from shared_configs.enums import EmbeddingProvider, EmbedTextType

logger = setup_logger()


admin_router = APIRouter(prefix="/admin/embedding")
basic_router = APIRouter(prefix="/embedding")


def _resolve_embedding_api_key(
    requested_api_key: str | None,
    provider_type: EmbeddingProvider,
    db_session: Session,
) -> str | None:
    if requested_api_key is not None:
        return requested_api_key

    stored_provider = get_embedding_provider_from_provider_type(
        db_session=db_session,
        provider_type=provider_type,
    )
    if stored_provider is None or stored_provider.api_key is None:
        return None

    return stored_provider.api_key.get_value(apply_mask=False)


def _embedding_dimension(embeddings: object) -> int:
    if not isinstance(embeddings, list) or len(embeddings) != 1:
        raise ValueError("Expected exactly one embedding vector.")

    vector = embeddings[0]
    if not isinstance(vector, list) or not vector:
        raise ValueError("The embedding vector must be a non-empty list.")

    for value in vector:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ValueError("The embedding vector contains a non-finite value.")

    return len(vector)


def _get_openrouter_embedding_models_response(api_key: str | None) -> object:
    headers: dict[str, str] = {
        "HTTP-Referer": "https://onyx.app",
        "X-Title": "Onyx",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        response = httpx.get(
            OPENROUTER_EMBEDDING_MODELS_URL,
            headers=headers,
            timeout=10.0,
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.warning(
            "Failed to fetch the OpenRouter embedding catalog", exc_info=True
        )
        raise OnyxError(
            OnyxErrorCode.BAD_GATEWAY,
            "Failed to fetch embedding models from OpenRouter.",
        ) from e


@admin_router.post("/openrouter/available-models")
def get_openrouter_available_embedding_models(
    request: OpenRouterEmbeddingModelsRequest,
    _: User = Depends(require_permission(Permission.FULL_ADMIN_PANEL_ACCESS)),
    db_session: Session = Depends(get_session),
) -> list[OpenRouterEmbeddingModelResponse]:
    api_key = _resolve_embedding_api_key(
        requested_api_key=request.api_key,
        provider_type=EmbeddingProvider.OPENROUTER,
        db_session=db_session,
    )
    payload = _get_openrouter_embedding_models_response(api_key)
    if not isinstance(payload, dict):
        raise OnyxError(
            OnyxErrorCode.VALIDATION_ERROR,
            "OpenRouter returned an invalid embedding model catalog.",
        )

    data = cast(dict[str, object], payload).get("data")
    if not isinstance(data, list) or not data:
        raise OnyxError(
            OnyxErrorCode.VALIDATION_ERROR,
            "OpenRouter returned no embedding models.",
        )

    results: list[OpenRouterEmbeddingModelResponse] = []
    for item in data:
        try:
            model = OpenRouterEmbeddingModelDetails.model_validate(item)
        except ValidationError:
            logger.warning("Skipping an invalid OpenRouter embedding catalog entry")
            continue

        if not model.supports_text_input or not model.produces_embeddings:
            continue

        results.append(
            OpenRouterEmbeddingModelResponse(
                name=model.id,
                display_name=model.display_name,
                context_length=model.context_length or None,
            )
        )

    if not results:
        raise OnyxError(
            OnyxErrorCode.VALIDATION_ERROR,
            "OpenRouter returned no text-capable embedding models.",
        )

    return sorted(results, key=lambda model: model.name)


@admin_router.post("/test-embedding")
def test_embedding_configuration(
    test_llm_request: TestEmbeddingRequest,
    _: User = Depends(require_permission(Permission.FULL_ADMIN_PANEL_ACCESS)),
    db_session: Session = Depends(get_session),
) -> TestEmbeddingResponse:
    try:
        api_key = _resolve_embedding_api_key(
            requested_api_key=test_llm_request.api_key,
            provider_type=test_llm_request.provider_type,
            db_session=db_session,
        )
        api_url = (
            OPENROUTER_EMBEDDINGS_URL
            if test_llm_request.provider_type == EmbeddingProvider.OPENROUTER
            else test_llm_request.api_url
        )
        test_model = EmbeddingModel(
            server_host=MODEL_SERVER_HOST,
            server_port=MODEL_SERVER_PORT,
            api_key=api_key,
            api_url=api_url,
            provider_type=test_llm_request.provider_type,
            model_name=test_llm_request.model_name,
            api_version=test_llm_request.api_version,
            deployment_name=test_llm_request.deployment_name,
            normalize=False,
            query_prefix=None,
            passage_prefix=None,
        )
        embeddings = test_model.encode(
            ["Testing Embedding"], text_type=EmbedTextType.QUERY
        )
        return TestEmbeddingResponse(
            embedding_dimension=_embedding_dimension(embeddings)
        )
    except Exception as e:
        error_msg = "An error occurred while testing your embedding model. Please check your configuration."
        logger.error("%s Error message: %s", error_msg, e, exc_info=True)
        raise OnyxError(OnyxErrorCode.VALIDATION_ERROR, error_msg) from e


@admin_router.get("", response_model=list[EmbeddingModelDetail])
def list_embedding_models(
    _: User = Depends(require_permission(Permission.FULL_ADMIN_PANEL_ACCESS)),
    db_session: Session = Depends(get_session),
) -> list[EmbeddingModelDetail]:
    search_settings = get_all_search_settings(db_session)
    return [EmbeddingModelDetail.from_db_model(setting) for setting in search_settings]


@admin_router.get("/embedding-provider")
def list_embedding_providers(
    _: User = Depends(require_permission(Permission.FULL_ADMIN_PANEL_ACCESS)),
    db_session: Session = Depends(get_session),
) -> list[CloudEmbeddingProvider]:
    return [
        CloudEmbeddingProvider.from_request(embedding_provider_model)
        for embedding_provider_model in fetch_existing_embedding_providers(db_session)
    ]


@admin_router.delete("/embedding-provider/{provider_type}")
def delete_embedding_provider(
    provider_type: EmbeddingProvider,
    _: User = Depends(require_permission(Permission.FULL_ADMIN_PANEL_ACCESS)),
    db_session: Session = Depends(get_session),
) -> None:
    embedding_provider = get_current_db_embedding_provider(db_session=db_session)
    if (
        embedding_provider is not None
        and provider_type == embedding_provider.provider_type
    ):
        raise OnyxError(
            OnyxErrorCode.VALIDATION_ERROR,
            "You can't delete a currently active model",
        )

    remove_embedding_provider(db_session, provider_type=provider_type)


@admin_router.put("/embedding-provider")
def put_cloud_embedding_provider(
    provider: CloudEmbeddingProviderCreationRequest,
    _: User = Depends(require_permission(Permission.FULL_ADMIN_PANEL_ACCESS)),
    db_session: Session = Depends(get_session),
) -> CloudEmbeddingProvider:
    if provider.provider_type == EmbeddingProvider.OPENROUTER:
        provider = provider.model_copy(update={"api_url": OPENROUTER_EMBEDDINGS_URL})
    return upsert_cloud_embedding_provider(db_session, provider)
