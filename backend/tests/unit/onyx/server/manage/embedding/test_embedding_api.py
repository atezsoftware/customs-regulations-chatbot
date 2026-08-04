import math
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import Enum

from onyx.db.llm import upsert_cloud_embedding_provider
from onyx.db.models import CloudEmbeddingProvider as CloudEmbeddingProviderModel
from onyx.db.models import SearchSettings
from onyx.error_handling.error_codes import OnyxErrorCode
from onyx.error_handling.exceptions import OnyxError
from onyx.natural_language_processing.constants import (
    OPENROUTER_EMBEDDING_MODELS_URL,
    OPENROUTER_EMBEDDINGS_URL,
)
from onyx.server.manage.embedding.api import (
    get_openrouter_available_embedding_models,
    list_embedding_models,
    list_embedding_providers,
    put_cloud_embedding_provider,
)
from onyx.server.manage.embedding.api import (
    test_embedding_configuration as run_embedding_test,
)
from onyx.server.manage.embedding.models import (
    CloudEmbeddingProviderCreationRequest,
    OpenRouterEmbeddingModelsRequest,
)
from onyx.server.manage.embedding.models import (
    TestEmbeddingRequest as EmbeddingTestRequest,
)
from onyx.utils.encryption import (
    decrypt_bytes_to_string,
    encrypt_string_to_bytes,
    mask_string,
)
from onyx.utils.sensitive import SensitiveValue
from shared_configs.enums import EmbeddingProvider


def _build_sensitive_value(raw_value: str) -> SensitiveValue[str]:
    return SensitiveValue[str](
        encrypted_bytes=encrypt_string_to_bytes(raw_value),
        decrypt_fn=decrypt_bytes_to_string,
    )


def _build_search_settings(raw_api_key: str) -> SimpleNamespace:
    return SimpleNamespace(
        id=7,
        model_name="gemini-embedding-001",
        normalize=False,
        query_prefix="",
        passage_prefix="",
        provider_type=EmbeddingProvider.GOOGLE,
        cloud_provider=SimpleNamespace(
            api_key=_build_sensitive_value(raw_api_key),
            api_url="",
            api_version=None,
            deployment_name=None,
        ),
        api_url="",
    )


def test_list_embedding_models_masks_api_key() -> None:
    raw_api_key = "sk-abcdefghijklmnopqrstuvwxyz1234567890"
    search_settings = _build_search_settings(raw_api_key)

    with patch(
        "onyx.server.manage.embedding.api.get_all_search_settings",
        return_value=[search_settings],
    ):
        response = list_embedding_models(_=MagicMock(), db_session=MagicMock())

    assert len(response) == 1
    assert response[0].api_key == mask_string(raw_api_key)
    assert response[0].api_key != raw_api_key


def test_list_embedding_models_returns_none_for_local_model_api_key() -> None:
    local_search_settings = SimpleNamespace(
        id=1,
        model_name="thenlper/gte-small",
        normalize=False,
        query_prefix="",
        passage_prefix="",
        provider_type=None,
        cloud_provider=None,
        api_url=None,
    )

    with patch(
        "onyx.server.manage.embedding.api.get_all_search_settings",
        return_value=[local_search_settings],
    ):
        response = list_embedding_models(_=MagicMock(), db_session=MagicMock())

    assert len(response) == 1
    assert response[0].api_key is None


def test_list_embedding_providers_uses_sensitive_value_masking_once() -> None:
    raw_api_key = "sk-abcdefghijklmnopqrstuvwxyz1234567890"
    provider_model = SimpleNamespace(
        provider_type=EmbeddingProvider.GOOGLE,
        api_key=_build_sensitive_value(raw_api_key),
        api_url="",
        api_version=None,
        deployment_name=None,
    )

    with patch(
        "onyx.server.manage.embedding.api.fetch_existing_embedding_providers",
        return_value=[provider_model],
    ):
        response = list_embedding_providers(_=MagicMock(), db_session=MagicMock())

    assert len(response) == 1
    assert response[0].api_key == mask_string(raw_api_key)
    assert response[0].api_key != mask_string(mask_string(raw_api_key))


def test_search_settings_api_key_property_returns_raw_value_for_runtime_use() -> None:
    raw_api_key = "sk-runtime-should-use-unmasked-value-1234567890"
    fake_search_settings = SimpleNamespace(
        cloud_provider=SimpleNamespace(api_key=_build_sensitive_value(raw_api_key))
    )

    api_key_property = SearchSettings.__dict__["api_key"]
    assert api_key_property.fget(fake_search_settings) == raw_api_key


def test_openrouter_embedding_provider_uses_varchar_backed_enum() -> None:
    provider_type = CloudEmbeddingProviderModel.__table__.c.provider_type.type

    assert isinstance(provider_type, Enum)
    assert provider_type.native_enum is False
    assert provider_type.length == 50


def test_openrouter_catalog_is_fixed_origin_and_filters_to_text_embeddings() -> None:
    stored_provider = SimpleNamespace(api_key=_build_sensitive_value("stored-key"))
    response = MagicMock()
    response.json.return_value = {
        "data": [
            {
                "id": "vendor/z-model",
                "name": "Z Model",
                "context_length": 8192,
                "architecture": {
                    "input_modalities": ["text"],
                    "output_modalities": ["embeddings"],
                },
            },
            {
                "id": "vendor/a-model",
                "name": "A Model",
                "context_length": 0,
                "architecture": {
                    "input_modalities": ["text", "image"],
                    "output_modalities": ["embeddings"],
                },
            },
            {
                "id": "vendor/image-only",
                "name": "Image Only",
                "architecture": {
                    "input_modalities": ["image"],
                    "output_modalities": ["embeddings"],
                },
            },
            {
                "id": "vendor/chat",
                "name": "Chat",
                "architecture": {
                    "input_modalities": ["text"],
                    "output_modalities": ["text"],
                },
            },
            {"id": "missing-name"},
        ]
    }

    with (
        patch(
            "onyx.server.manage.embedding.api.get_embedding_provider_from_provider_type",
            return_value=stored_provider,
        ),
        patch(
            "onyx.server.manage.embedding.api.httpx.get", return_value=response
        ) as get,
    ):
        results = get_openrouter_available_embedding_models(
            OpenRouterEmbeddingModelsRequest(),
            _=MagicMock(),
            db_session=MagicMock(),
        )

    assert [model.name for model in results] == [
        "vendor/a-model",
        "vendor/z-model",
    ]
    assert results[0].context_length is None
    get.assert_called_once_with(
        OPENROUTER_EMBEDDING_MODELS_URL,
        headers={
            "HTTP-Referer": "https://onyx.app",
            "X-Title": "Onyx",
            "Authorization": "Bearer stored-key",
        },
        timeout=10.0,
    )
    response.raise_for_status.assert_called_once_with()


def test_openrouter_catalog_rejects_catalog_without_text_embeddings() -> None:
    response = MagicMock()
    response.json.return_value = {
        "data": [
            {
                "id": "vendor/image-only",
                "name": "Image Only",
                "architecture": {
                    "input_modalities": ["image"],
                    "output_modalities": ["embeddings"],
                },
            }
        ]
    }

    with (
        patch("onyx.server.manage.embedding.api.httpx.get", return_value=response),
        pytest.raises(OnyxError) as exc_info,
    ):
        get_openrouter_available_embedding_models(
            OpenRouterEmbeddingModelsRequest(api_key="new-key"),
            _=MagicMock(),
            db_session=MagicMock(),
        )

    assert exc_info.value.error_code == OnyxErrorCode.VALIDATION_ERROR


def test_embedding_configuration_reuses_key_and_returns_actual_dimension() -> None:
    stored_provider = SimpleNamespace(api_key=_build_sensitive_value("stored-key"))
    embedding_model = MagicMock()
    embedding_model.encode.return_value = [[0.1, -0.2, 0.3, 0.4]]
    request = EmbeddingTestRequest(
        provider_type=EmbeddingProvider.OPENROUTER,
        model_name="openai/text-embedding-3-small",
        api_url="https://example.invalid/embeddings",
    )

    with (
        patch(
            "onyx.server.manage.embedding.api.get_embedding_provider_from_provider_type",
            return_value=stored_provider,
        ),
        patch(
            "onyx.server.manage.embedding.api.EmbeddingModel",
            return_value=embedding_model,
        ) as embedding_model_cls,
    ):
        result = run_embedding_test(
            request,
            _=MagicMock(),
            db_session=MagicMock(),
        )

    assert result.embedding_dimension == 4
    assert embedding_model_cls.call_args.kwargs["api_key"] == "stored-key"
    assert embedding_model_cls.call_args.kwargs["api_url"] == OPENROUTER_EMBEDDINGS_URL


@pytest.mark.parametrize(
    "embedding_result",
    [
        [],
        [[0.1], [0.2]],
        [[]],
        [[True]],
        [["not-a-number"]],
        [[math.nan]],
        [[math.inf]],
    ],
)
def test_embedding_configuration_rejects_invalid_vector_shape(
    embedding_result: object,
) -> None:
    embedding_model = MagicMock()
    embedding_model.encode.return_value = embedding_result

    with (
        patch(
            "onyx.server.manage.embedding.api.EmbeddingModel",
            return_value=embedding_model,
        ),
        pytest.raises(OnyxError) as exc_info,
    ):
        run_embedding_test(
            EmbeddingTestRequest(
                provider_type=EmbeddingProvider.OPENROUTER,
                api_key="new-key",
                model_name="openai/text-embedding-3-small",
            ),
            _=MagicMock(),
            db_session=MagicMock(),
        )

    assert exc_info.value.error_code == OnyxErrorCode.VALIDATION_ERROR


def test_openrouter_provider_put_replaces_user_url_with_fixed_url() -> None:
    expected_response = MagicMock()
    request = CloudEmbeddingProviderCreationRequest(
        provider_type=EmbeddingProvider.OPENROUTER,
        api_key="new-key",
        api_url="https://example.invalid/embeddings",
    )

    with patch(
        "onyx.server.manage.embedding.api.upsert_cloud_embedding_provider",
        return_value=expected_response,
    ) as upsert:
        response = put_cloud_embedding_provider(
            request,
            _=MagicMock(),
            db_session=MagicMock(),
        )

    assert response is expected_response
    saved_request = upsert.call_args.args[1]
    assert saved_request.provider_type == EmbeddingProvider.OPENROUTER
    assert saved_request.api_url == OPENROUTER_EMBEDDINGS_URL


def test_embedding_provider_upsert_preserves_omitted_api_key() -> None:
    stored_key = _build_sensitive_value("stored-key")
    existing_provider = SimpleNamespace(
        provider_type=EmbeddingProvider.OPENROUTER,
        api_key=stored_key,
        api_url=OPENROUTER_EMBEDDINGS_URL,
        api_version=None,
        deployment_name=None,
    )
    db_session = MagicMock()
    db_session.query.return_value.filter_by.return_value.first.return_value = (
        existing_provider
    )

    response = upsert_cloud_embedding_provider(
        db_session,
        CloudEmbeddingProviderCreationRequest(
            provider_type=EmbeddingProvider.OPENROUTER,
            api_url=OPENROUTER_EMBEDDINGS_URL,
        ),
    )

    assert existing_provider.api_key is stored_key
    assert response.api_key == mask_string("stored-key")
    db_session.commit.assert_called_once_with()
