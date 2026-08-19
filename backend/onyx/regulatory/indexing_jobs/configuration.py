import json
from hashlib import sha256
from typing import cast

from sqlalchemy.orm import Session

from onyx.configs import app_configs
from onyx.db.enums import RegulatoryIndexingStage
from onyx.db.llm import fetch_embedding_provider, fetch_model_configuration_by_id
from onyx.db.models import LLMProvider, ModelConfiguration, SearchSettings
from onyx.db.search_settings import get_current_search_settings
from onyx.llm.constants import LlmProviderNames
from onyx.llm.well_known_providers.constants import (
    VERTEX_AUTH_METHOD_KWARG,
    VERTEX_AUTH_METHOD_SERVICE_ACCOUNT,
    VERTEX_AUTH_METHOD_WORKLOAD_IDENTITY,
    VERTEX_CREDENTIALS_FILE_KWARG,
    VERTEX_LOCATION_KWARG,
    VERTEX_PROJECT_KWARG,
)
from onyx.prompts.contextual_retrieval import (
    CONTEXTUAL_RAG_PROMPT1,
    CONTEXTUAL_RAG_PROMPT2,
)
from onyx.regulatory.indexing_jobs.models import (
    RegulatoryIndexingConfigSnapshot,
    VertexAuthenticationMode,
    VertexBatchConfig,
)
from shared_configs.enums import EmbeddingProvider

_OPENROUTER_EMBEDDING_MODEL = "openai/text-embedding-3-large"
_CONTEXTUAL_PROMPT_VERSION = "contextual-rag-v1"
_CONTEXTUAL_PROMPT_HASH = sha256(
    f"{CONTEXTUAL_RAG_PROMPT1}\0{CONTEXTUAL_RAG_PROMPT2}".encode()
).hexdigest()
_CONTEXT_STAGES = frozenset(
    {
        RegulatoryIndexingStage.CONTEXT_SUBMIT,
        RegulatoryIndexingStage.CONTEXT_WAIT,
        RegulatoryIndexingStage.CONTEXT_APPLY,
    }
)
_PROMPT_DEPENDENT_STAGES = frozenset(
    {
        RegulatoryIndexingStage.CONTEXT_SUBMIT,
        RegulatoryIndexingStage.CONTEXT_APPLY,
    }
)


class RegulatoryIndexingConfigurationError(ValueError):
    """The active Admin configuration is unsafe for durable indexing."""


def _validate_embedding_contract(search_settings: SearchSettings) -> None:
    if search_settings.provider_type is not EmbeddingProvider.OPENROUTER:
        raise RegulatoryIndexingConfigurationError(
            "Regulatory indexing requires the active OpenRouter embedding provider"
        )
    if search_settings.model_name != _OPENROUTER_EMBEDDING_MODEL:
        raise RegulatoryIndexingConfigurationError(
            "Regulatory indexing requires openai/text-embedding-3-large"
        )
    expected_dimension = search_settings.reduced_dimension or search_settings.model_dim
    if search_settings.final_embedding_dim != expected_dimension:
        raise RegulatoryIndexingConfigurationError(
            "Active SearchSettings has an inconsistent effective dimension"
        )
    if not search_settings.enable_contextual_rag:
        raise RegulatoryIndexingConfigurationError(
            "Contextual retrieval must be enabled for regulatory indexing"
        )
    if search_settings.contextual_rag_model_configuration_id is None:
        raise RegulatoryIndexingConfigurationError(
            "A contextual model must be selected for regulatory indexing"
        )


def _validate_openrouter_credentials(db_session: Session) -> None:
    provider = fetch_embedding_provider(db_session, EmbeddingProvider.OPENROUTER)
    if provider is None or provider.api_key is None:
        raise RegulatoryIndexingConfigurationError(
            "The active OpenRouter embedding provider has no configured credential"
        )
    if not provider.api_key.get_value(apply_mask=False).strip():
        raise RegulatoryIndexingConfigurationError(
            "The active OpenRouter embedding provider has no configured credential"
        )


def _resolve_vertex_config(
    model_configuration: ModelConfiguration,
) -> VertexBatchConfig:
    provider: LLMProvider = model_configuration.llm_provider
    if provider.provider != LlmProviderNames.VERTEX_AI:
        raise RegulatoryIndexingConfigurationError(
            "The contextual model provider must be Vertex AI"
        )
    custom_config = provider.custom_config or {}
    location = (custom_config.get(VERTEX_LOCATION_KWARG) or "").strip()
    gcs_uri = (app_configs.REGULATORY_INDEXING_GCS_URI or "").strip()
    if not location:
        raise RegulatoryIndexingConfigurationError(
            "The Vertex AI location is not configured"
        )
    if not gcs_uri:
        raise RegulatoryIndexingConfigurationError(
            "REGULATORY_INDEXING_GCS_URI is not configured"
        )

    raw_auth_method = custom_config.get(
        VERTEX_AUTH_METHOD_KWARG,
        VERTEX_AUTH_METHOD_SERVICE_ACCOUNT,
    )
    if raw_auth_method == VERTEX_AUTH_METHOD_SERVICE_ACCOUNT:
        raw_credentials = (
            custom_config.get(VERTEX_CREDENTIALS_FILE_KWARG) or ""
        ).strip()
        if not raw_credentials:
            raise RegulatoryIndexingConfigurationError(
                "The Vertex AI service-account credential is not configured"
            )
        try:
            parsed_credentials: object = json.loads(raw_credentials)
        except json.JSONDecodeError as error:
            raise RegulatoryIndexingConfigurationError(
                "The Vertex AI service-account credential is invalid"
            ) from error
        if not isinstance(parsed_credentials, dict):
            raise RegulatoryIndexingConfigurationError(
                "The Vertex AI service-account credential is invalid"
            )
        service_account_info = cast(dict[str, object], parsed_credentials)
        raw_project = service_account_info.get("project_id")
        if not isinstance(raw_project, str) or not raw_project.strip():
            raise RegulatoryIndexingConfigurationError(
                "The Vertex AI service-account project is not configured"
            )
        project = raw_project.strip()
        authentication_mode = VertexAuthenticationMode.SERVICE_ACCOUNT_JSON
    elif raw_auth_method == VERTEX_AUTH_METHOD_WORKLOAD_IDENTITY:
        project = (custom_config.get(VERTEX_PROJECT_KWARG) or "").strip()
        if not project:
            raise RegulatoryIndexingConfigurationError(
                "The Vertex AI project is not configured"
            )
        authentication_mode = VertexAuthenticationMode.WORKLOAD_IDENTITY
    else:
        raise RegulatoryIndexingConfigurationError(
            "The Vertex AI authentication mode is unsupported"
        )

    try:
        return VertexBatchConfig(
            model_configuration_id=model_configuration.id,
            model_name=model_configuration.name,
            project=project,
            location=location,
            authentication_mode=authentication_mode,
            gcs_uri=gcs_uri,
        )
    except ValueError as error:
        raise RegulatoryIndexingConfigurationError(
            "The Vertex AI batch configuration is invalid"
        ) from error


def _get_contextual_model_configuration(
    db_session: Session, search_settings: SearchSettings
) -> ModelConfiguration:
    model_configuration = fetch_model_configuration_by_id(
        db_session,
        search_settings.contextual_rag_model_configuration_id,
    )
    if model_configuration is None:
        raise RegulatoryIndexingConfigurationError(
            "The configured contextual model no longer exists"
        )
    return model_configuration


def resolve_regulatory_indexing_snapshot(
    db_session: Session,
) -> RegulatoryIndexingConfigSnapshot:
    search_settings = get_current_search_settings(db_session)
    _validate_embedding_contract(search_settings)
    _validate_openrouter_credentials(db_session)
    vertex = _resolve_vertex_config(
        _get_contextual_model_configuration(db_session, search_settings)
    )
    return RegulatoryIndexingConfigSnapshot(
        search_settings_id=search_settings.id,
        embedding_provider=EmbeddingProvider.OPENROUTER,
        embedding_model_name=search_settings.model_name,
        model_dimension=search_settings.model_dim,
        reduced_dimension=search_settings.reduced_dimension,
        effective_dimension=search_settings.final_embedding_dim,
        index_name=search_settings.index_name,
        vertex=vertex,
        prompt_version=_CONTEXTUAL_PROMPT_VERSION,
        prompt_hash=_CONTEXTUAL_PROMPT_HASH,
        max_attempts=app_configs.REGULATORY_INDEXING_MAX_ATTEMPTS,
        retry_base_seconds=app_configs.REGULATORY_INDEXING_RETRY_BASE_SECONDS,
        retry_max_seconds=app_configs.REGULATORY_INDEXING_RETRY_MAX_SECONDS,
        poll_seconds=app_configs.REGULATORY_INDEXING_POLL_SECONDS,
        lease_seconds=app_configs.REGULATORY_INDEXING_LEASE_SECONDS,
        embedding_request_size=(app_configs.REGULATORY_INDEXING_EMBEDDING_REQUEST_SIZE),
    )


def _validate_search_settings_snapshot(
    search_settings: SearchSettings,
    snapshot: RegulatoryIndexingConfigSnapshot,
) -> None:
    _validate_embedding_contract(search_settings)
    current_values = (
        search_settings.id,
        search_settings.provider_type,
        search_settings.model_name,
        search_settings.model_dim,
        search_settings.reduced_dimension,
        search_settings.final_embedding_dim,
        search_settings.index_name,
        search_settings.contextual_rag_model_configuration_id,
    )
    snapshot_values = (
        snapshot.search_settings_id,
        snapshot.embedding_provider,
        snapshot.embedding_model_name,
        snapshot.model_dimension,
        snapshot.reduced_dimension,
        snapshot.effective_dimension,
        snapshot.index_name,
        snapshot.vertex.model_configuration_id,
    )
    if current_values != snapshot_values:
        raise RegulatoryIndexingConfigurationError(
            "Active SearchSettings no longer matches the indexing job snapshot"
        )


def validate_snapshot_for_stage(
    db_session: Session,
    snapshot: RegulatoryIndexingConfigSnapshot,
    stage: RegulatoryIndexingStage,
) -> None:
    """Fail closed on Admin configuration drift before a stage executes."""

    if stage in _PROMPT_DEPENDENT_STAGES and (
        snapshot.prompt_version != _CONTEXTUAL_PROMPT_VERSION
        or snapshot.prompt_hash != _CONTEXTUAL_PROMPT_HASH
    ):
        raise RegulatoryIndexingConfigurationError(
            "Contextual prompt identity no longer matches the indexing job snapshot"
        )

    search_settings = get_current_search_settings(db_session)
    _validate_search_settings_snapshot(search_settings, snapshot)

    if stage in _CONTEXT_STAGES or stage is RegulatoryIndexingStage.PREPARING:
        current_vertex = _resolve_vertex_config(
            _get_contextual_model_configuration(db_session, search_settings)
        )
        if current_vertex != snapshot.vertex:
            raise RegulatoryIndexingConfigurationError(
                "Vertex AI configuration no longer matches the indexing job snapshot"
            )
    if stage in {
        RegulatoryIndexingStage.PREPARING,
        RegulatoryIndexingStage.EMBEDDING,
    }:
        _validate_openrouter_credentials(db_session)
