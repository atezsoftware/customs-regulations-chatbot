from __future__ import annotations

import json
from collections.abc import Callable, Generator
from typing import Any, cast
from uuid import UUID

import httpx
import pytest
from fastapi.routing import APIRoute
from pydantic import BaseModel
from sqlalchemy.orm import Session

from onyx.db.enums import Permission
from onyx.db.models import User
from onyx.db.reranking import RerankerRuntimeConfig
from onyx.error_handling.exceptions import OnyxError
from onyx.reranking.openrouter import OpenRouterRerankClient
from onyx.utils.sensitive import SensitiveValue
from shared_configs.enums import RerankerProvider

RERANKING_URL = "/admin/reranking"
TEST_MODEL = "voyageai/rerank-2.5"
TEST_KEY = "sk-test-reranking-secret-value"


class _StubUser:
    def __init__(self, *, admin: bool) -> None:
        self.id = UUID("00000000-0000-0000-0000-000000000001")
        self.effective_permissions = (
            [Permission.FULL_ADMIN_PANEL_ACCESS.value] if admin else []
        )


class _Repository:
    def __init__(self) -> None:
        self.enabled = False
        self.provider_type: RerankerProvider | None = None
        self.model_name: str | None = None
        self.api_key: str | None = None
        self.ciphertext: bytes | None = None
        self.committed = False

    def runtime_config(self) -> RerankerRuntimeConfig:
        sensitive_key = (
            SensitiveValue(
                encrypted_bytes=self.api_key.encode(),
                decrypt_fn=lambda value: value.decode(),
            )
            if self.api_key is not None
            else None
        )
        return RerankerRuntimeConfig(
            enabled=self.enabled,
            provider_type=self.provider_type,
            model_name=self.model_name,
            api_key=sensitive_key,
        )

    def get(self, _db_session: object) -> RerankerRuntimeConfig:
        return self.runtime_config()

    def upsert(
        self,
        _db_session: object,
        *,
        enabled: bool,
        provider_type: RerankerProvider | None,
        model_name: str | None,
        api_key: str | None,
        updated_by_user_id: UUID | None = None,  # noqa: ARG002
        commit: bool = True,
    ) -> RerankerRuntimeConfig:
        assert commit is True
        self.enabled = enabled
        self.provider_type = provider_type
        self.model_name = model_name
        if api_key is not None:
            self.api_key = api_key
            self.ciphertext = b"ciphertext:" + api_key.encode()
        self.committed = True
        return self.runtime_config()

    def delete(
        self,
        _db_session: object,
        *,
        updated_by_user_id: UUID | None = None,  # noqa: ARG002
        commit: bool = True,
    ) -> None:
        assert commit is True
        self.enabled = False
        self.provider_type = None
        self.model_name = None
        self.api_key = None
        self.ciphertext = None
        self.committed = True


class _DirectClient:
    def __init__(self, repository: _Repository) -> None:
        self._repository = repository

    def request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        from onyx.server.manage.reranking import api as reranking_api
        from onyx.server.manage.reranking.models import (
            OpenRouterModelsRequest,
            RerankerConfigUpdate,
            RerankerTestRequest,
        )

        payload = kwargs.get("json") or {}
        user = cast(User, _StubUser(admin=True))
        db_session = cast(Session, self._repository)
        try:
            if method == "GET" and path == f"{RERANKING_URL}/config":
                result = reranking_api.get_config(user, db_session)
            elif method == "PUT" and path == f"{RERANKING_URL}/config":
                result = reranking_api.put_config(
                    RerankerConfigUpdate.model_validate(payload),
                    user,
                    db_session,
                )
            elif method == "DELETE" and path == f"{RERANKING_URL}/config":
                result = reranking_api.delete_config(user, db_session)
            elif method == "POST" and path == f"{RERANKING_URL}/test":
                result = reranking_api.test_config(
                    RerankerTestRequest.model_validate(payload),
                    user,
                    db_session,
                )
            elif method == "POST" and path == f"{RERANKING_URL}/openrouter-models":
                result = reranking_api.openrouter_models(
                    OpenRouterModelsRequest.model_validate(payload),
                    user,
                    db_session,
                )
            else:
                return httpx.Response(404)
        except OnyxError as error:
            return httpx.Response(
                error.status_code,
                json=error.error_code.detail(error.detail),
            )
        if isinstance(result, BaseModel):
            return httpx.Response(200, json=result.model_dump(mode="json"))
        return httpx.Response(result.status_code, content=bytes(result.body))

    def get(self, path: str, **kwargs: Any) -> httpx.Response:
        return self.request("GET", path, **kwargs)

    def put(self, path: str, **kwargs: Any) -> httpx.Response:
        return self.request("PUT", path, **kwargs)

    def post(self, path: str, **kwargs: Any) -> httpx.Response:
        return self.request("POST", path, **kwargs)

    def delete(self, path: str, **kwargs: Any) -> httpx.Response:
        return self.request("DELETE", path, **kwargs)


@pytest.fixture
def repository() -> _Repository:
    return _Repository()


def _make_client(
    monkeypatch: pytest.MonkeyPatch,
    repository: _Repository,
    *,
    admin: bool,  # noqa: ARG001
) -> Generator[_DirectClient, None, None]:
    from onyx.server.manage.reranking import api as reranking_api

    monkeypatch.setattr(
        reranking_api,
        "test_attestations",
        reranking_api.RerankerTestAttestationStore(),
    )
    monkeypatch.setattr(reranking_api, "get_reranker_configuration", repository.get)
    monkeypatch.setattr(
        reranking_api, "upsert_reranker_configuration", repository.upsert
    )
    monkeypatch.setattr(
        reranking_api, "delete_reranker_configuration", repository.delete
    )
    yield _DirectClient(repository)


@pytest.fixture
def admin_client(
    monkeypatch: pytest.MonkeyPatch,
    repository: _Repository,
) -> Generator[_DirectClient, None, None]:
    yield from _make_client(monkeypatch, repository, admin=True)


def _put_config(
    client: _DirectClient,
    *,
    enabled: bool,
    model_id: str = TEST_MODEL,
    api_key: str | None = TEST_KEY,
    test_attestation: str | None = None,
) -> httpx.Response:
    payload: dict[str, Any] = {
        "enabled": enabled,
        "provider_type": "openrouter",
        "model_id": model_id,
    }
    if api_key is not None:
        payload["api_key"] = api_key
    if test_attestation is not None:
        payload["test_attestation"] = test_attestation
    return client.put(f"{RERANKING_URL}/config", json=payload)


def _successful_rerank(
    expected_key: str = TEST_KEY,
    expected_model: str = TEST_MODEL,
) -> Callable[..., list[Any]]:
    def rerank(
        self: OpenRouterRerankClient,  # noqa: ARG001
        *,
        api_key: str,
        model: str,
        query: str,
        documents: list[str],
        top_n: int,
    ) -> list[Any]:
        assert (api_key, model) == (expected_key, expected_model)
        assert query == "Onyx reranker configuration test"
        assert documents == ["Configuration test document"]
        assert top_n == 1
        return []

    return rerank


def _test_configuration(
    client: _DirectClient,
    *,
    api_key: str | None = None,
    model_id: str | None = None,
) -> httpx.Response:
    payload: dict[str, str] = {"provider_type": "openrouter"}
    if api_key is not None:
        payload["api_key"] = api_key
    if model_id is not None:
        payload["model_id"] = model_id
    return client.post(f"{RERANKING_URL}/test", json=payload)


@pytest.mark.parametrize(
    ("method", "path", "payload"),
    [
        ("GET", "/config", None),
        (
            "PUT",
            "/config",
            {
                "enabled": False,
                "provider_type": "openrouter",
                "model_id": TEST_MODEL,
            },
        ),
        ("DELETE", "/config", None),
        (
            "POST",
            "/test",
            {"provider_type": "openrouter", "model_id": TEST_MODEL},
        ),
        ("POST", "/openrouter-models", {}),
    ],
)
def test_all_reranking_routes_require_full_admin_access(
    method: str,
    path: str,
    payload: dict[str, Any] | None,
) -> None:
    from onyx.server.manage.reranking.api import admin_router

    del payload
    route = next(
        candidate
        for candidate in admin_router.routes
        if isinstance(candidate, APIRoute)
        and candidate.path == f"{RERANKING_URL}{path}"
        and method in candidate.methods
    )
    required_permissions = {
        getattr(dependency.call, "_required_permission", None)
        for dependency in route.dependant.dependencies
    }
    assert Permission.FULL_ADMIN_PANEL_ACCESS in required_permissions


def test_get_masks_api_key_and_never_returns_plaintext(
    admin_client: _DirectClient,
) -> None:
    assert _put_config(admin_client, enabled=False).status_code == 200

    response = admin_client.get(f"{RERANKING_URL}/config")

    assert response.status_code == 200
    assert response.json() == {
        "enabled": False,
        "provider_type": "openrouter",
        "model_id": TEST_MODEL,
        "api_key_configured": True,
        "masked_api_key": "sk-t...alue",
    }
    assert TEST_KEY not in response.text
    assert "api_key" not in response.json()


def test_put_disabled_with_omitted_key_retains_exact_ciphertext(
    admin_client: _DirectClient,
    repository: _Repository,
) -> None:
    assert _put_config(admin_client, enabled=False).status_code == 200
    before = repository.ciphertext

    response = _put_config(
        admin_client,
        enabled=False,
        model_id="manual/legal-reranker-v2",
        api_key=None,
    )

    assert response.status_code == 200
    assert repository.ciphertext == before
    assert response.json()["model_id"] == "manual/legal-reranker-v2"
    assert response.json()["api_key_configured"] is True


@pytest.mark.parametrize("api_key", ["", "   ", "••••••••••••", "sk-t...alue"])
@pytest.mark.parametrize("endpoint", ["config", "test", "openrouter-models"])
def test_blank_or_masked_unsaved_keys_are_rejected(
    admin_client: _DirectClient,
    api_key: str,
    endpoint: str,
) -> None:
    if endpoint == "config":
        response = _put_config(admin_client, enabled=False, api_key=api_key)
    elif endpoint == "test":
        response = _test_configuration(
            admin_client,
            api_key=api_key,
            model_id=TEST_MODEL,
        )
    else:
        response = admin_client.post(
            f"{RERANKING_URL}/openrouter-models",
            json={"api_key": api_key},
        )
    assert response.status_code == 400


def test_delete_purges_stored_key_and_returns_no_content(
    admin_client: _DirectClient,
    repository: _Repository,
) -> None:
    assert _put_config(admin_client, enabled=False).status_code == 200

    response = admin_client.delete(f"{RERANKING_URL}/config")

    assert response.status_code == 204
    assert response.content == b""
    assert repository.ciphertext is None
    assert admin_client.get(f"{RERANKING_URL}/config").json() == {
        "enabled": False,
        "provider_type": None,
        "model_id": None,
        "api_key_configured": False,
        "masked_api_key": None,
    }


def test_test_endpoint_uses_unsaved_key_and_manual_model_without_persisting(
    admin_client: _DirectClient,
    repository: _Repository,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manual_model = "manual-provider/custom-rerank-model"
    unsaved_key = "sk-unsaved-test-only-secret"
    monkeypatch.setattr(
        OpenRouterRerankClient,
        "rerank",
        _successful_rerank(unsaved_key, manual_model),
    )

    response = _test_configuration(
        admin_client,
        api_key=unsaved_key,
        model_id=manual_model,
    )

    assert response.status_code == 200
    assert response.json()["success"] is True
    assert response.json()["test_attestation"]
    assert repository.api_key is None
    assert repository.model_name is None


def test_enabled_put_requires_attestation_for_exact_effective_key_and_model(
    admin_client: _DirectClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert _put_config(admin_client, enabled=False).status_code == 200
    monkeypatch.setattr(OpenRouterRerankClient, "rerank", _successful_rerank())

    assert _put_config(admin_client, enabled=True, api_key=None).status_code == 400
    token = _test_configuration(admin_client).json()["test_attestation"]
    assert (
        _put_config(
            admin_client,
            enabled=True,
            model_id="different/model",
            api_key=None,
            test_attestation=token,
        ).status_code
        == 400
    )
    assert (
        _put_config(
            admin_client,
            enabled=True,
            api_key="sk-a-different-secret-value",
            test_attestation=token,
        ).status_code
        == 400
    )

    enabled = _put_config(
        admin_client,
        enabled=True,
        api_key=None,
        test_attestation=token,
    )
    assert enabled.status_code == 200
    assert enabled.json()["enabled"] is True


def test_expired_test_attestation_cannot_enable_reranking(
    admin_client: _DirectClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from onyx.server.manage.reranking import api as reranking_api

    now = [100.0]
    monkeypatch.setattr(
        reranking_api,
        "test_attestations",
        reranking_api.RerankerTestAttestationStore(
            ttl_seconds=30,
            clock=lambda: now[0],
        ),
    )
    monkeypatch.setattr(OpenRouterRerankClient, "rerank", _successful_rerank())
    assert _put_config(admin_client, enabled=False).status_code == 200
    token = _test_configuration(admin_client).json()["test_attestation"]

    now[0] = 131.0
    response = _put_config(
        admin_client,
        enabled=True,
        api_key=None,
        test_attestation=token,
    )
    assert response.status_code == 400


def test_attestation_is_tenant_scoped(
    admin_client: _DirectClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from onyx.server.manage.reranking import api as reranking_api

    assert _put_config(admin_client, enabled=False).status_code == 200
    monkeypatch.setattr(OpenRouterRerankClient, "rerank", _successful_rerank())
    tenant = ["tenant-a"]
    monkeypatch.setattr(reranking_api, "get_current_tenant_id", lambda: tenant[0])
    token = _test_configuration(admin_client).json()["test_attestation"]

    tenant[0] = "tenant-b"
    response = _put_config(
        admin_client,
        enabled=True,
        api_key=None,
        test_attestation=token,
    )
    assert response.status_code == 400


def test_catalog_uses_fixed_url_and_body_only_unsaved_key(
    admin_client: _DirectClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[httpx.Request] = []
    unsaved_key = "sk-catalog-unsaved-secret-value"

    def handle_request(
        self: httpx.HTTPTransport,  # noqa: ARG001
        request: httpx.Request,
    ) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={"data": [{"id": TEST_MODEL, "name": "Voyage Rerank 2.5"}]},
            request=request,
        )

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", handle_request)
    response = admin_client.post(
        f"{RERANKING_URL}/openrouter-models",
        json={"api_key": unsaved_key},
    )

    assert response.status_code == 200
    assert response.json() == {
        "models": [{"id": TEST_MODEL, "name": "Voyage Rerank 2.5"}]
    }
    request = requests[0]
    assert request.method == "GET"
    assert str(request.url) == (
        "https://openrouter.ai/api/v1/models?output_modalities=rerank"
    )
    assert request.headers["Authorization"] == f"Bearer {unsaved_key}"
    assert unsaved_key not in str(request.url)


def test_privacy_policy_provider_failure_returns_no_attestation(
    admin_client: _DirectClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_payloads: list[dict[str, Any]] = []

    def handle_request(
        self: httpx.HTTPTransport,  # noqa: ARG001
        request: httpx.Request,
    ) -> httpx.Response:
        request_payloads.append(json.loads(request.content))
        return httpx.Response(400, json={"error": "no ZDR route"}, request=request)

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", handle_request)
    response = _test_configuration(
        admin_client,
        api_key=TEST_KEY,
        model_id=TEST_MODEL,
    )

    assert response.status_code == 502
    assert "test_attestation" not in response.json()
    assert request_payloads[0]["provider"] == {
        "zdr": True,
        "data_collection": "deny",
    }
    assert TEST_KEY not in response.text


def test_put_and_delete_invalidate_caches_after_commit(
    admin_client: _DirectClient,
    repository: _Repository,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from onyx.server.manage.reranking import api as reranking_api

    observed: list[tuple[str, bool, str | None]] = []

    def observe_invalidation(tenant_id: str) -> None:
        assert repository.committed is True
        observed.append((tenant_id, repository.enabled, repository.model_name))

    monkeypatch.setattr(
        reranking_api, "invalidate_reranker_circuit", observe_invalidation
    )
    monkeypatch.setattr(OpenRouterRerankClient, "rerank", _successful_rerank())
    assert _put_config(admin_client, enabled=False).status_code == 200
    token = _test_configuration(admin_client).json()["test_attestation"]

    response = _put_config(
        admin_client,
        enabled=True,
        api_key=None,
        test_attestation=token,
    )
    assert response.status_code == 200
    assert observed[-1][1:] == (True, TEST_MODEL)
    assert (
        _put_config(
            admin_client,
            enabled=True,
            api_key=None,
            test_attestation=token,
        ).status_code
        == 400
    )

    response = admin_client.delete(f"{RERANKING_URL}/config")
    assert response.status_code == 204
    assert observed[-1][1:] == (False, None)
    assert observed[-1][0] == observed[-2][0]
