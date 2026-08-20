import json
from types import SimpleNamespace
from typing import Any, cast

import pytest
from pydantic import ValidationError
from sqlalchemy.orm import Session

from onyx.configs import app_configs
from onyx.db.enums import RegulatoryIndexingStage
from onyx.db.models import LLMProvider, ModelConfiguration
from onyx.llm.constants import LlmProviderNames
from onyx.llm.well_known_providers.constants import (
    VERTEX_AUTH_METHOD_KWARG,
    VERTEX_AUTH_METHOD_SERVICE_ACCOUNT,
    VERTEX_AUTH_METHOD_WORKLOAD_IDENTITY,
    VERTEX_CREDENTIALS_FILE_KWARG,
    VERTEX_LOCATION_KWARG,
    VERTEX_PROJECT_KWARG,
)
from onyx.regulatory.indexing_jobs import configuration
from onyx.regulatory.indexing_jobs.configuration import (
    RegulatoryIndexingConfigurationError,
    compute_regulatory_chunk_generation_hash,
    resolve_regulatory_indexing_snapshot,
    validate_snapshot_for_stage,
)
from onyx.regulatory.indexing_jobs.models import (
    RegulatoryIndexingConfigSnapshot,
    RegulatoryInputHashVersion,
    VertexAuthenticationMode,
    VertexBatchConfig,
)
from onyx.utils.sensitive import SensitiveValue
from shared_configs.enums import EmbeddingProvider

_EMBEDDING_MODEL = "openai/text-embedding-3-large"
_DB_SESSION = cast(Session, SimpleNamespace())


def _decrypt_test_value(value: bytes) -> str:
    return value.decode()


_CONFIGURED_EMBEDDING_KEY = SensitiveValue(
    encrypted_bytes=b"configured",
    decrypt_fn=_decrypt_test_value,
    is_json=False,
)


def _search_settings(
    *,
    provider_type: EmbeddingProvider = EmbeddingProvider.OPENROUTER,
    model_name: str = _EMBEDDING_MODEL,
    model_dim: int = 3072,
    reduced_dimension: int | None = 1024,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=41,
        provider_type=provider_type,
        model_name=model_name,
        model_dim=model_dim,
        reduced_dimension=reduced_dimension,
        final_embedding_dim=reduced_dimension or model_dim,
        index_name="danswer_chunk_v2",
        enable_contextual_rag=True,
        contextual_rag_model_configuration_id=73,
    )


def _vertex_provider(
    *,
    provider: str = LlmProviderNames.VERTEX_AI,
    project: str = "customs-prod",
    location: str = "europe-west4",
    auth_method: str = VERTEX_AUTH_METHOD_SERVICE_ACCOUNT,
    credentials: str | None = '{"type":"service_account","project_id":"customs-prod"}',
) -> SimpleNamespace:
    custom_config = {
        VERTEX_PROJECT_KWARG: project,
        VERTEX_LOCATION_KWARG: location,
        VERTEX_AUTH_METHOD_KWARG: auth_method,
    }
    if credentials is not None:
        custom_config[VERTEX_CREDENTIALS_FILE_KWARG] = credentials
    return SimpleNamespace(provider=provider, custom_config=custom_config)


def _model_configuration(*, provider: SimpleNamespace | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        id=73,
        name="gemini-3.1-flash-lite",
        is_visible=True,
        llm_provider=provider or _vertex_provider(),
    )


def _service_account_model_configuration(
    *,
    credentials: str,
    explicit_project: str | None = None,
) -> ModelConfiguration:
    custom_config = {
        VERTEX_LOCATION_KWARG: "europe-west4",
        VERTEX_AUTH_METHOD_KWARG: VERTEX_AUTH_METHOD_SERVICE_ACCOUNT,
        VERTEX_CREDENTIALS_FILE_KWARG: credentials,
    }
    if explicit_project is not None:
        custom_config[VERTEX_PROJECT_KWARG] = explicit_project
    provider = LLMProvider(
        id=29,
        provider=LlmProviderNames.VERTEX_AI,
        custom_config=custom_config,
    )
    return ModelConfiguration(
        id=73,
        llm_provider_id=29,
        name="gemini-3.1-flash-lite",
        is_visible=True,
        llm_provider=provider,
    )


def _install_admin_configuration(
    monkeypatch: pytest.MonkeyPatch,
    *,
    search_settings: SimpleNamespace | None = None,
    model_configuration: ModelConfiguration | SimpleNamespace | None = None,
    embedding_api_key: SensitiveValue[str] | None = _CONFIGURED_EMBEDDING_KEY,
) -> None:
    monkeypatch.setattr(
        configuration,
        "get_current_search_settings",
        lambda _db_session, **_kwargs: search_settings or _search_settings(),
    )
    monkeypatch.setattr(
        configuration,
        "fetch_model_configuration_by_id",
        lambda _db_session, _model_configuration_id: (
            model_configuration or _model_configuration()
        ),
    )
    monkeypatch.setattr(
        configuration,
        "fetch_embedding_provider",
        lambda _db_session, _provider_type: SimpleNamespace(api_key=embedding_api_key),
    )
    monkeypatch.setattr(
        configuration.app_configs,
        "REGULATORY_INDEXING_GCS_URI",
        "gs://customs-indexing/regulatory",
    )


def test_regulatory_indexing_defaults() -> None:
    assert app_configs.REGULATORY_BATCH_INDEXING_ENABLED is False
    assert app_configs.REGULATORY_INDEXING_MAX_ATTEMPTS == 5
    assert app_configs.REGULATORY_INDEXING_RETRY_BASE_SECONDS == 15
    assert app_configs.REGULATORY_INDEXING_RETRY_MAX_SECONDS == 900
    assert app_configs.REGULATORY_INDEXING_POLL_SECONDS == 30
    assert app_configs.REGULATORY_INDEXING_LEASE_SECONDS == 120
    assert app_configs.REGULATORY_INDEXING_EMBEDDING_REQUEST_SIZE == 64


def test_chunk_generation_hash_is_restart_stable_and_semantics_sensitive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = compute_regulatory_chunk_generation_hash(
        embedding_provider=EmbeddingProvider.OPENROUTER,
        embedding_model_name=_EMBEDDING_MODEL,
    )
    second = compute_regulatory_chunk_generation_hash(
        embedding_provider=EmbeddingProvider.OPENROUTER,
        embedding_model_name=_EMBEDDING_MODEL,
    )

    assert first == second
    assert len(first) == 64

    monkeypatch.setattr(
        configuration,
        "REGULATORY_CHUNKER_CODE_VERSION",
        "test-next-semantics",
    )
    changed = compute_regulatory_chunk_generation_hash(
        embedding_provider=EmbeddingProvider.OPENROUTER,
        embedding_model_name=_EMBEDDING_MODEL,
    )

    assert changed != first


def test_snapshot_is_frozen_json_and_excludes_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_admin_configuration(monkeypatch)

    snapshot = resolve_regulatory_indexing_snapshot(_DB_SESSION)
    dumped = snapshot.model_dump(mode="json")
    encoded = json.dumps(dumped)

    assert dumped["search_settings_id"] == 41
    assert dumped["input_content_hash"] == configuration._READINESS_INPUT_CONTENT_HASH
    assert dumped["input_hash_version"] == "canonical-v2"
    assert len(dumped["chunk_generation_hash"]) == 64
    assert dumped["embedding_provider"] == "openrouter"
    assert dumped["embedding_model_name"] == _EMBEDDING_MODEL
    assert dumped["effective_dimension"] == 1024
    assert dumped["vertex"]["model_configuration_id"] == 73
    assert dumped["vertex"]["authentication_mode"] == "service_account_json"
    assert "credential" not in encoded.lower()
    assert '{"type":"service_account"}' not in encoded
    with pytest.raises(ValidationError):
        snapshot.effective_dimension = 3072


def test_stage_validation_rejects_changed_chunk_generation_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_admin_configuration(monkeypatch)
    snapshot = resolve_regulatory_indexing_snapshot(_DB_SESSION)
    monkeypatch.setattr(
        configuration,
        "REGULATORY_INDEXING_GENERATION_CODE_VERSION",
        "next-generation",
    )

    with pytest.raises(
        RegulatoryIndexingConfigurationError,
        match="Chunk-generation identity",
    ):
        validate_snapshot_for_stage(
            _DB_SESSION,
            snapshot,
            RegulatoryIndexingStage.PREPARING,
        )


def test_unresolved_preparing_snapshot_defers_generation_drift_until_hash_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_admin_configuration(monkeypatch)
    snapshot = resolve_regulatory_indexing_snapshot(_DB_SESSION).model_copy(
        update={"input_hash_version": RegulatoryInputHashVersion.LEGACY_OR_CANONICAL}
    )
    monkeypatch.setattr(
        configuration,
        "REGULATORY_INDEXING_GENERATION_CODE_VERSION",
        "next-generation",
    )

    validate_snapshot_for_stage(
        _DB_SESSION,
        snapshot,
        RegulatoryIndexingStage.PREPARING,
    )

    with pytest.raises(
        RegulatoryIndexingConfigurationError,
        match="Chunk-generation identity",
    ):
        validate_snapshot_for_stage(
            _DB_SESSION,
            snapshot,
            RegulatoryIndexingStage.CONTEXT_SUBMIT,
        )


def test_effective_dimension_uses_native_dimension_without_reduction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_admin_configuration(
        monkeypatch,
        search_settings=_search_settings(reduced_dimension=None),
    )

    snapshot = resolve_regulatory_indexing_snapshot(_DB_SESSION)

    assert snapshot.effective_dimension == 3072


def test_snapshot_rejects_an_inconsistent_effective_dimension() -> None:
    with pytest.raises(ValidationError):
        RegulatoryIndexingConfigSnapshot(
            input_content_hash="1" * 64,
            input_hash_version=RegulatoryInputHashVersion.CANONICAL_V2,
            chunk_generation_hash="2" * 64,
            search_settings_id=41,
            embedding_provider=EmbeddingProvider.OPENROUTER,
            embedding_model_name=_EMBEDDING_MODEL,
            model_dimension=3072,
            reduced_dimension=1024,
            effective_dimension=3072,
            index_name="danswer_chunk_v2",
            prompt_version="contextual-rag-v1",
            prompt_hash="a" * 64,
            vertex=VertexBatchConfig(
                model_configuration_id=73,
                model_name="gemini-3.1-flash-lite",
                project="customs-prod",
                location="europe-west4",
                authentication_mode=VertexAuthenticationMode.WORKLOAD_IDENTITY,
                gcs_uri="gs://customs-indexing/regulatory",
            ),
        )


def test_snapshot_requires_prompt_identity() -> None:
    with pytest.raises(ValidationError):
        RegulatoryIndexingConfigSnapshot.model_validate(
            {
                "search_settings_id": 41,
                "embedding_provider": EmbeddingProvider.OPENROUTER,
                "embedding_model_name": _EMBEDDING_MODEL,
                "model_dimension": 3072,
                "reduced_dimension": 1024,
                "effective_dimension": 1024,
                "index_name": "danswer_chunk_v2",
                "vertex": VertexBatchConfig(
                    model_configuration_id=73,
                    model_name="gemini-3.1-flash-lite",
                    project="customs-prod",
                    location="europe-west4",
                    authentication_mode=VertexAuthenticationMode.WORKLOAD_IDENTITY,
                    gcs_uri="gs://customs-indexing/regulatory",
                ),
            }
        )


def test_snapshot_rejects_nonfinite_retry_policy() -> None:
    with pytest.raises(ValidationError):
        RegulatoryIndexingConfigSnapshot(
            input_content_hash="1" * 64,
            input_hash_version=RegulatoryInputHashVersion.CANONICAL_V2,
            chunk_generation_hash="2" * 64,
            search_settings_id=41,
            embedding_provider=EmbeddingProvider.OPENROUTER,
            embedding_model_name=_EMBEDDING_MODEL,
            model_dimension=3072,
            reduced_dimension=1024,
            effective_dimension=1024,
            index_name="danswer_chunk_v2",
            vertex=VertexBatchConfig(
                model_configuration_id=73,
                model_name="gemini-3.1-flash-lite",
                project="customs-prod",
                location="europe-west4",
                authentication_mode=VertexAuthenticationMode.WORKLOAD_IDENTITY,
                gcs_uri="gs://customs-indexing/regulatory",
            ),
            prompt_version="contextual-rag-v1",
            prompt_hash="a" * 64,
            retry_max_seconds=float("inf"),
        )


@pytest.mark.parametrize(
    ("provider_type", "model_name"),
    [
        (EmbeddingProvider.OPENAI, _EMBEDDING_MODEL),
        (EmbeddingProvider.OPENROUTER, "openai/text-embedding-3-small"),
    ],
)
def test_resolution_rejects_non_production_embedding_contract(
    monkeypatch: pytest.MonkeyPatch,
    provider_type: EmbeddingProvider,
    model_name: str,
) -> None:
    _install_admin_configuration(
        monkeypatch,
        search_settings=_search_settings(
            provider_type=provider_type,
            model_name=model_name,
        ),
    )

    with pytest.raises(RegulatoryIndexingConfigurationError):
        resolve_regulatory_indexing_snapshot(_DB_SESSION)


@pytest.mark.parametrize(
    "provider",
    [
        _vertex_provider(provider=LlmProviderNames.OPENAI),
        _vertex_provider(
            project="",
            auth_method=VERTEX_AUTH_METHOD_WORKLOAD_IDENTITY,
            credentials=None,
        ),
        _vertex_provider(location=""),
        _vertex_provider(credentials=None),
        _vertex_provider(auth_method="unsupported"),
    ],
)
def test_resolution_rejects_invalid_vertex_contract(
    monkeypatch: pytest.MonkeyPatch,
    provider: SimpleNamespace,
) -> None:
    _install_admin_configuration(
        monkeypatch,
        model_configuration=_model_configuration(provider=provider),
    )

    with pytest.raises(RegulatoryIndexingConfigurationError):
        resolve_regulatory_indexing_snapshot(_DB_SESSION)


def test_resolution_accepts_workload_identity_without_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_admin_configuration(
        monkeypatch,
        model_configuration=_model_configuration(
            provider=_vertex_provider(
                auth_method=VERTEX_AUTH_METHOD_WORKLOAD_IDENTITY,
                credentials=None,
            )
        ),
    )

    snapshot = resolve_regulatory_indexing_snapshot(_DB_SESSION)

    assert (
        snapshot.vertex.authentication_mode
        is VertexAuthenticationMode.WORKLOAD_IDENTITY
    )


def test_resolution_derives_project_from_admin_service_account_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_configuration = _service_account_model_configuration(
        credentials=json.dumps(
            {
                "type": "service_account",
                "project_id": "credentials-project",
                "private_key_id": "fixture-only",
            }
        )
    )
    _install_admin_configuration(
        monkeypatch,
        model_configuration=model_configuration,
    )

    snapshot = resolve_regulatory_indexing_snapshot(_DB_SESSION)
    encoded = json.dumps(snapshot.model_dump(mode="json"))

    assert snapshot.vertex.project == "credentials-project"
    assert "private_key_id" not in encoded
    assert "fixture-only" not in encoded


@pytest.mark.parametrize(
    "credentials",
    [
        "{not-valid-json",
        json.dumps({"type": "service_account"}),
        json.dumps({"type": "service_account", "project_id": 123}),
    ],
)
def test_explicit_project_cannot_bypass_invalid_service_account_json(
    monkeypatch: pytest.MonkeyPatch,
    credentials: str,
) -> None:
    _install_admin_configuration(
        monkeypatch,
        model_configuration=_service_account_model_configuration(
            credentials=credentials,
            explicit_project="stale-explicit-project",
        ),
    )

    with pytest.raises(RegulatoryIndexingConfigurationError):
        resolve_regulatory_indexing_snapshot(_DB_SESSION)


def test_stage_validation_detects_search_settings_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = _search_settings()
    _install_admin_configuration(monkeypatch, search_settings=current)
    snapshot = resolve_regulatory_indexing_snapshot(_DB_SESSION)
    current.reduced_dimension = 768
    current.final_embedding_dim = 768

    with pytest.raises(RegulatoryIndexingConfigurationError):
        validate_snapshot_for_stage(
            _DB_SESSION,
            snapshot,
            RegulatoryIndexingStage.EMBEDDING,
        )


def test_embedding_stage_resolves_current_openrouter_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_admin_configuration(monkeypatch)
    snapshot = resolve_regulatory_indexing_snapshot(_DB_SESSION)
    monkeypatch.setattr(
        configuration,
        "fetch_embedding_provider",
        lambda _db_session, _provider_type: SimpleNamespace(api_key=None),
    )

    with pytest.raises(RegulatoryIndexingConfigurationError):
        validate_snapshot_for_stage(
            _DB_SESSION,
            snapshot,
            RegulatoryIndexingStage.EMBEDDING,
        )


def test_context_stage_resolves_current_vertex_authentication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_configuration = _model_configuration()
    _install_admin_configuration(
        monkeypatch,
        model_configuration=model_configuration,
    )
    snapshot = resolve_regulatory_indexing_snapshot(_DB_SESSION)
    model_configuration.llm_provider.custom_config.pop(VERTEX_CREDENTIALS_FILE_KWARG)

    with pytest.raises(RegulatoryIndexingConfigurationError):
        validate_snapshot_for_stage(
            _DB_SESSION,
            snapshot,
            RegulatoryIndexingStage.CONTEXT_SUBMIT,
        )


@pytest.mark.parametrize(
    ("constant_name", "drifted_value"),
    [
        ("_CONTEXTUAL_PROMPT_VERSION", "contextual-rag-v2"),
        ("_CONTEXTUAL_PROMPT_HASH", "b" * 64),
    ],
)
@pytest.mark.parametrize(
    "stage",
    [
        RegulatoryIndexingStage.CONTEXT_SUBMIT,
        RegulatoryIndexingStage.CONTEXT_APPLY,
    ],
)
def test_prompt_dependent_stage_rejects_prompt_drift(
    monkeypatch: pytest.MonkeyPatch,
    constant_name: str,
    drifted_value: str,
    stage: RegulatoryIndexingStage,
) -> None:
    _install_admin_configuration(monkeypatch)
    snapshot = resolve_regulatory_indexing_snapshot(_DB_SESSION)
    monkeypatch.setattr(configuration, constant_name, drifted_value)

    with pytest.raises(RegulatoryIndexingConfigurationError):
        validate_snapshot_for_stage(_DB_SESSION, snapshot, stage)


@pytest.mark.parametrize(
    "stage",
    [
        RegulatoryIndexingStage.CONTEXT_WAIT,
        RegulatoryIndexingStage.EMBEDDING,
        RegulatoryIndexingStage.INDEX_WRITE,
    ],
)
def test_prompt_independent_stage_allows_prompt_drift(
    monkeypatch: pytest.MonkeyPatch,
    stage: RegulatoryIndexingStage,
) -> None:
    _install_admin_configuration(monkeypatch)
    snapshot = resolve_regulatory_indexing_snapshot(_DB_SESSION)
    monkeypatch.setattr(configuration, "_CONTEXTUAL_PROMPT_HASH", "b" * 64)

    validate_snapshot_for_stage(_DB_SESSION, snapshot, stage)


def test_vertex_snapshot_forbids_secret_fields() -> None:
    fields: dict[str, Any] = {
        "model_configuration_id": 73,
        "model_name": "gemini-3.1-flash-lite",
        "project": "customs-prod",
        "location": "europe-west4",
        "authentication_mode": VertexAuthenticationMode.WORKLOAD_IDENTITY,
        "gcs_uri": "gs://customs-indexing/regulatory",
        "credentials": '{"type":"service_account"}',
    }

    with pytest.raises(ValidationError):
        VertexBatchConfig.model_validate(fields)
