from __future__ import annotations

import math
from typing import Any

import httpx
from fastapi import APIRouter, Depends, Response
from sqlalchemy.orm import Session

from onyx.auth.permissions import require_permission
from onyx.cache.interface import (
    CACHE_TRANSIENT_ERRORS,
    CacheLockLostError,
)
from onyx.db.engine.sql_engine import get_session
from onyx.db.enums import Permission
from onyx.db.models import User
from onyx.db.reranking import (
    RerankerRuntimeConfig,
    delete_reranker_configuration,
    get_reranker_configuration,
    get_reranker_configuration_for_update,
    upsert_reranker_configuration,
)
from onyx.error_handling.error_codes import OnyxErrorCode
from onyx.error_handling.exceptions import OnyxError
from onyx.reranking.circuit_breaker import reranker_configuration_fingerprint
from onyx.reranking.constants import (
    RERANK_CONNECT_TIMEOUT_SECONDS,
    RERANK_POOL_TIMEOUT_SECONDS,
    RERANK_READ_TIMEOUT_SECONDS,
    RERANK_WRITE_TIMEOUT_SECONDS,
)
from onyx.reranking.distributed_state import distributed_reranker_attestations
from onyx.reranking.models import RerankError, RerankScore
from onyx.reranking.openrouter import OpenRouterRerankClient
from onyx.reranking.service import invalidate_reranker_circuit
from onyx.server.manage.reranking.models import (
    OpenRouterModelsRequest,
    OpenRouterModelsResponse,
    OpenRouterModelView,
    RerankerConfigUpdate,
    RerankerConfigView,
    RerankerTestRequest,
    RerankerTestResponse,
)
from onyx.utils.encryption import is_masked_credential
from shared_configs.contextvars import get_current_tenant_id
from shared_configs.enums import RerankerProvider

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models?output_modalities=rerank"
_TEST_QUERY = "Onyx reranker configuration test"
_TEST_DOCUMENT = "Configuration test document"
_ATTESTATION_ERRORS = CACHE_TRANSIENT_ERRORS + (CacheLockLostError,)

admin_router = APIRouter(prefix="/admin/reranking")


def _view(config: RerankerRuntimeConfig) -> RerankerConfigView:
    masked_key = (
        config.api_key.get_value(apply_mask=True)
        if config.api_key is not None
        else None
    )
    return RerankerConfigView(
        enabled=config.enabled,
        provider_type=(
            "openrouter"
            if config.provider_type is RerankerProvider.OPENROUTER
            else None
        ),
        model_id=config.model_name,
        api_key_configured=config.api_key is not None,
        masked_api_key=masked_key,
    )


def _provided_api_key(api_key: str | None) -> str | None:
    if api_key is None:
        return None
    if not api_key.strip():
        raise OnyxError(OnyxErrorCode.INVALID_INPUT, "API key cannot be blank.")
    if is_masked_credential(api_key):
        raise OnyxError(
            OnyxErrorCode.INVALID_INPUT,
            "Provide the actual API key, not a masked value.",
        )
    return api_key


def _model_id(model_id: str | None) -> str | None:
    if model_id is None:
        return None
    model_id = model_id.strip()
    if not model_id:
        raise OnyxError(OnyxErrorCode.INVALID_INPUT, "Model ID cannot be blank.")
    return model_id


def _effective_key(*, api_key: str | None, stored: RerankerRuntimeConfig) -> str | None:
    provided_key = _provided_api_key(api_key)
    if provided_key is not None:
        return provided_key
    if stored.api_key is None:
        return None
    return stored.api_key.get_value(apply_mask=False)


def _require_key(api_key: str | None) -> str:
    if api_key is None:
        raise OnyxError(
            OnyxErrorCode.MISSING_REQUIRED_FIELD,
            "Configure an OpenRouter API key first.",
        )
    return api_key


def _invalidate_after_commit(tenant_id: str) -> None:
    invalidate_reranker_circuit(tenant_id)


@admin_router.get("/config")
def get_config(
    _: User = Depends(require_permission(Permission.FULL_ADMIN_PANEL_ACCESS)),
    db_session: Session = Depends(get_session),
) -> RerankerConfigView:
    return _view(get_reranker_configuration(db_session))


@admin_router.put("/config")
def put_config(
    request: RerankerConfigUpdate,
    user: User = Depends(require_permission(Permission.FULL_ADMIN_PANEL_ACCESS)),
    db_session: Session = Depends(get_session),
) -> RerankerConfigView:
    stored = get_reranker_configuration_for_update(db_session)
    provided_key = _provided_api_key(request.api_key)
    effective_key = _effective_key(api_key=provided_key, stored=stored)
    model_id = _model_id(request.model_id)
    if model_id is None:
        raise OnyxError(OnyxErrorCode.MISSING_REQUIRED_FIELD, "Model ID is required.")
    tenant_id = get_current_tenant_id()
    if request.enabled:
        required_key = _require_key(effective_key)
        fingerprint = reranker_configuration_fingerprint(
            model=model_id,
            api_key=required_key,
        )
        try:
            attested = distributed_reranker_attestations.consume(
                tenant_id=tenant_id,
                token=request.test_attestation,
                config_fingerprint=fingerprint,
                config_generation=stored.configuration_generation,
            )
        except _ATTESTATION_ERRORS as error:
            raise OnyxError(
                OnyxErrorCode.SERVICE_UNAVAILABLE,
                "Reranker test verification is temporarily unavailable.",
            ) from error
        if not attested:
            raise OnyxError(
                OnyxErrorCode.INVALID_INPUT,
                "Test this exact API key and model before enabling reranking.",
            )

    config = upsert_reranker_configuration(
        db_session,
        enabled=request.enabled,
        provider_type=RerankerProvider.OPENROUTER,
        model_name=model_id,
        api_key=provided_key,
        updated_by_user_id=user.id,
        commit=False,
    )
    db_session.commit()
    _invalidate_after_commit(tenant_id)
    return _view(config)


@admin_router.delete("/config", status_code=204, response_class=Response)
def delete_config(
    user: User = Depends(require_permission(Permission.FULL_ADMIN_PANEL_ACCESS)),
    db_session: Session = Depends(get_session),
) -> Response:
    delete_reranker_configuration(
        db_session,
        updated_by_user_id=user.id,
        commit=False,
    )
    db_session.commit()
    _invalidate_after_commit(get_current_tenant_id())
    return Response(status_code=204)


@admin_router.post("/test")
def test_config(
    request: RerankerTestRequest,
    _: User = Depends(require_permission(Permission.FULL_ADMIN_PANEL_ACCESS)),
    db_session: Session = Depends(get_session),
) -> RerankerTestResponse:
    stored = get_reranker_configuration(db_session)
    api_key = _require_key(_effective_key(api_key=request.api_key, stored=stored))
    model_id = _model_id(request.model_id) or stored.model_name
    if model_id is None:
        raise OnyxError(
            OnyxErrorCode.MISSING_REQUIRED_FIELD,
            "Select or enter an OpenRouter reranking model first.",
        )
    try:
        scores = OpenRouterRerankClient().rerank(
            api_key=api_key,
            model=model_id,
            query=_TEST_QUERY,
            documents=[_TEST_DOCUMENT],
            top_n=1,
        )
    except RerankError as error:
        raise OnyxError(
            OnyxErrorCode.BAD_GATEWAY,
            "OpenRouter could not complete a private reranking test.",
        ) from error
    if (
        not isinstance(scores, list)
        or len(scores) != 1
        or not isinstance(scores[0], RerankScore)
        or scores[0].index != 0
        or not math.isfinite(scores[0].relevance_score)
    ):
        raise OnyxError(
            OnyxErrorCode.BAD_GATEWAY,
            "OpenRouter could not complete a private reranking test.",
        )

    fingerprint = reranker_configuration_fingerprint(
        model=model_id,
        api_key=api_key,
    )
    try:
        token = distributed_reranker_attestations.issue(
            tenant_id=get_current_tenant_id(),
            config_fingerprint=fingerprint,
            config_generation=stored.configuration_generation,
        )
    except CACHE_TRANSIENT_ERRORS as error:
        raise OnyxError(
            OnyxErrorCode.SERVICE_UNAVAILABLE,
            "Reranker test verification is temporarily unavailable.",
        ) from error
    return RerankerTestResponse(success=True, test_attestation=token)


def _catalog_http_client() -> httpx.Client:
    return httpx.Client(
        timeout=httpx.Timeout(
            connect=RERANK_CONNECT_TIMEOUT_SECONDS,
            read=RERANK_READ_TIMEOUT_SECONDS,
            write=RERANK_WRITE_TIMEOUT_SECONDS,
            pool=RERANK_POOL_TIMEOUT_SECONDS,
        ),
        transport=httpx.HTTPTransport(retries=0),
    )


@admin_router.post("/openrouter-models")
def openrouter_models(
    request: OpenRouterModelsRequest,
    _: User = Depends(require_permission(Permission.FULL_ADMIN_PANEL_ACCESS)),
    db_session: Session = Depends(get_session),
) -> OpenRouterModelsResponse:
    stored = get_reranker_configuration(db_session)
    api_key = _require_key(_effective_key(api_key=request.api_key, stored=stored))
    try:
        with _catalog_http_client() as http:
            response = http.get(
                OPENROUTER_MODELS_URL,
                headers={"Authorization": f"Bearer {api_key}"},
            )
        response.raise_for_status()
        payload: Any = response.json()
    except (httpx.HTTPError, TypeError, ValueError) as error:
        raise OnyxError(
            OnyxErrorCode.BAD_GATEWAY,
            "OpenRouter model catalog is unavailable.",
        ) from error
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise OnyxError(
            OnyxErrorCode.BAD_GATEWAY,
            "OpenRouter returned an invalid model catalog.",
        )
    models = [
        OpenRouterModelView(
            id=item["id"],
            name=item.get("name") or item["id"],
        )
        for item in payload["data"]
        if isinstance(item, dict)
        and isinstance(item.get("id"), str)
        and item["id"]
        and (item.get("name") is None or isinstance(item.get("name"), str))
    ]
    return OpenRouterModelsResponse(models=models)
