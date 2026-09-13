from __future__ import annotations

import json
import os
import re
from hashlib import sha256
from typing import Literal, TypedDict
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, model_validator
from sqlalchemy import Select, select
from sqlalchemy.orm import Session, joinedload, load_only

from onyx.auth.schemas import UserRole
from onyx.db.llm import (
    can_user_access_llm_provider,
    fetch_user_group_ids,
)
from onyx.db.models import (
    ImageGenerationConfig,
    LLMProvider,
    ModelConfiguration,
    Persona,
    User,
    UserGroup,
)
from onyx.llm.constants import LlmProviderNames
from onyx.llm.well_known_providers.constants import (
    VERTEX_AUTH_METHOD_KWARG,
    VERTEX_AUTH_METHOD_SERVICE_ACCOUNT,
    VERTEX_AUTH_METHOD_WORKLOAD_IDENTITY,
    VERTEX_CREDENTIALS_FILE_KWARG,
    VERTEX_LOCATION_KWARG,
    VERTEX_PROJECT_KWARG,
)
from onyx.regulatory.indexing_jobs.gemini_files_batch import (
    GoogleGeminiFilesBatchGateway,
)
from onyx.regulatory.indexing_jobs.models import (
    VertexAuthenticationMode,
    VertexBatchConfig,
)
from onyx.regulatory.indexing_jobs.vertex_batch import VertexBatchGateway
from onyx.regulatory.labeling.provider import DEFAULT_MODEL
from onyx.tracing.flows import LLMFlow


class LabelingProviderOption(TypedDict):
    id: int
    name: str


LabelingProviderTransport = Literal["gemini_files_v1", "vertex_gcs_v1"]

_FILES_TRANSPORT: LabelingProviderTransport = "gemini_files_v1"
_VERTEX_TRANSPORT: LabelingProviderTransport = "vertex_gcs_v1"
_VERTEX_GCS_URI_ENV = "REGULATORY_LABELING_VERTEX_GCS_URI"
_GCS_BUCKET_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]*[a-z0-9]$")
_INVALID_STAGING_URI_ERROR = (
    "Gemini Batch storage requires a gs://bucket/prefix Cloud Storage location."
)


def _normalize_vertex_staging_uri(value: str) -> str:
    uri = value.strip().rstrip("/")
    try:
        parsed = urlsplit(uri)
    except ValueError:
        raise ValueError(_INVALID_STAGING_URI_ERROR) from None
    prefix = parsed.path.removeprefix("/")
    path_segments = prefix.split("/")
    if (
        parsed.scheme != "gs"
        or not 3 <= len(parsed.netloc) <= 222
        or _GCS_BUCKET_PATTERN.fullmatch(parsed.netloc) is None
        or parsed.query
        or parsed.fragment
        or not prefix
        or any(segment in {"", ".", ".."} for segment in path_segments)
        or any(character.isspace() for character in uri)
        or "\\" in uri
    ):
        raise ValueError(_INVALID_STAGING_URI_ERROR)
    return f"gs://{parsed.netloc}/{prefix}"


def _configured_vertex_staging_uri() -> str:
    value = os.environ.get(_VERTEX_GCS_URI_ENV, "")
    if not value.strip():
        raise ValueError(
            "Gemini Batch storage is not configured. Configure its Cloud Storage "
            "location before starting labeling."
        )
    return _normalize_vertex_staging_uri(value)


def get_labeling_configuration_errors() -> list[str]:
    try:
        _configured_vertex_staging_uri()
    except ValueError as error:
        return [str(error)]
    return []


class LabelingProviderBinding(BaseModel):
    """Non-secret provider identity; rotating a key for the same principal is safe."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_configuration_id: int
    provider_id: int
    project: str
    location: str
    authentication_mode: VertexAuthenticationMode
    credential_identity: str
    model: str = DEFAULT_MODEL
    transport: LabelingProviderTransport = _FILES_TRANSPORT
    staging_uri: str | None = None

    @model_validator(mode="after")
    def validate_transport_configuration(self) -> "LabelingProviderBinding":
        if self.transport == _FILES_TRANSPORT:
            if self.staging_uri is not None:
                raise ValueError("The Gemini Files transport cannot use GCS staging")
            return self
        if self.staging_uri is None:
            raise ValueError("The Vertex transport requires a GCS staging URI")
        normalized = _normalize_vertex_staging_uri(self.staging_uri)
        if normalized != self.staging_uri:
            raise ValueError("The Vertex GCS staging URI must be normalized")
        return self

    @property
    def fingerprint(self) -> str:
        payload = self.model_dump(mode="json")
        if self.transport == _FILES_TRANSPORT and self.staging_uri is None:
            payload.pop("transport")
            payload.pop("staging_uri")
        return sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


def _binding(
    model: ModelConfiguration,
    *,
    transport: LabelingProviderTransport = _FILES_TRANSPORT,
    staging_uri: str | None = None,
) -> LabelingProviderBinding:
    provider = model.llm_provider
    if provider.provider != LlmProviderNames.VERTEX_AI or not model.is_visible:
        raise ValueError("Select an enabled Google model configuration for labeling")
    custom = provider.custom_config or {}
    mode = custom.get(VERTEX_AUTH_METHOD_KWARG, VERTEX_AUTH_METHOD_SERVICE_ACCOUNT)
    if mode == VERTEX_AUTH_METHOD_SERVICE_ACCOUNT:
        try:
            raw: object = json.loads(custom.get(VERTEX_CREDENTIALS_FILE_KWARG, ""))
        except (TypeError, ValueError):
            raise ValueError(
                "The Google service-account configuration is invalid"
            ) from None
        if not isinstance(raw, dict):
            raise ValueError("The Google service-account configuration is invalid")
        project = raw.get("project_id")
        identity = raw.get("client_email")
        private_key = raw.get("private_key")
        if not all(
            isinstance(value, str) and value.strip()
            for value in (project, identity, private_key)
        ):
            raise ValueError("The Google service-account configuration is incomplete")
        assert isinstance(project, str) and isinstance(identity, str)
        authentication_mode = VertexAuthenticationMode.SERVICE_ACCOUNT_JSON
    elif mode == VERTEX_AUTH_METHOD_WORKLOAD_IDENTITY:
        project = custom.get(VERTEX_PROJECT_KWARG, "").strip()
        if not project:
            raise ValueError("The Google workload identity project is not configured")
        identity = "workload_identity"
        authentication_mode = VertexAuthenticationMode.WORKLOAD_IDENTITY
    else:
        raise ValueError("The Google authentication mode is unsupported")
    return LabelingProviderBinding(
        model_configuration_id=model.id,
        provider_id=provider.id,
        project=project.strip(),
        location=custom.get(VERTEX_LOCATION_KWARG, "global").strip() or "global",
        authentication_mode=authentication_mode,
        credential_identity=identity.strip(),
        transport=transport,
        staging_uri=staging_uri,
    )


def _authorized_model(
    db_session: Session,
    model_configuration_id: int,
    user: User | None,
) -> ModelConfiguration:
    model = db_session.scalar(
        _model_statement()
        .where(ModelConfiguration.id == model_configuration_id)
        .execution_options(populate_existing=True)
    )
    if model is None:
        raise ValueError("The labeling model configuration is unavailable")
    if user is not None and not can_user_access_llm_provider(
        model.llm_provider,
        fetch_user_group_ids(db_session, user),
        persona=None,
        is_admin=user.role == UserRole.ADMIN,
    ):
        raise ValueError("The labeling provider is unavailable to this user")
    return model


def _model_statement() -> Select[tuple[ModelConfiguration]]:
    provider = joinedload(ModelConfiguration.llm_provider)
    image_generation_provider_ids = select(ModelConfiguration.llm_provider_id).join(
        ImageGenerationConfig
    )
    return (
        select(ModelConfiguration)
        .where(~ModelConfiguration.llm_provider_id.in_(image_generation_provider_ids))
        .options(
            load_only(
                ModelConfiguration.id,
                ModelConfiguration.llm_provider_id,
                ModelConfiguration.name,
                ModelConfiguration.is_visible,
            ),
            provider.load_only(
                LLMProvider.id,
                LLMProvider.name,
                LLMProvider.provider,
                LLMProvider.is_public,
                LLMProvider.custom_config,
            ),
            provider.selectinload(LLMProvider.groups).load_only(UserGroup.id),
            provider.selectinload(LLMProvider.personas).load_only(Persona.id),
        )
    )


def resolve_labeling_provider_binding(
    db_session: Session,
    model_configuration_id: int,
    *,
    user: User | None = None,
) -> LabelingProviderBinding:
    return _binding(
        _authorized_model(db_session, model_configuration_id, user),
        transport=_VERTEX_TRANSPORT,
        staging_uri=_configured_vertex_staging_uri(),
    )


def get_labeling_provider_options(
    db_session: Session,
    *,
    user: User,
) -> list[LabelingProviderOption]:
    preferred_models = (
        select(ModelConfiguration.id)
        .join(ModelConfiguration.llm_provider)
        .where(
            LLMProvider.provider == LlmProviderNames.VERTEX_AI,
            ModelConfiguration.is_visible.is_(True),
        )
        .distinct(ModelConfiguration.llm_provider_id)
        .order_by(
            ModelConfiguration.llm_provider_id,
            ModelConfiguration.name != DEFAULT_MODEL,
            ModelConfiguration.id,
        )
    )
    models = db_session.scalars(
        _model_statement()
        .where(ModelConfiguration.id.in_(preferred_models))
        .order_by(ModelConfiguration.llm_provider_id)
        .execution_options(populate_existing=True)
    )
    user_group_ids = fetch_user_group_ids(db_session, user)
    options: list[LabelingProviderOption] = []
    for model in models:
        provider = model.llm_provider
        if not can_user_access_llm_provider(
            provider,
            user_group_ids,
            persona=None,
            is_admin=user.role == UserRole.ADMIN,
        ):
            continue
        try:
            _binding(model)
        except ValueError:
            continue
        options.append({"id": model.id, "name": provider.name or "Gemini"})
    return options


def resolve_labeling_gateway(
    db_session: Session,
    model_configuration_id: int,
    *,
    user: User | None = None,
    expected_binding: LabelingProviderBinding | None = None,
) -> VertexBatchGateway:
    """Resolve credentials while the session is open; perform no network I/O here."""
    model = _authorized_model(db_session, model_configuration_id, user)
    transport = (
        expected_binding.transport
        if expected_binding is not None
        else _VERTEX_TRANSPORT
    )
    binding = _binding(
        model,
        transport=transport,
        staging_uri=(
            _configured_vertex_staging_uri() if transport == _VERTEX_TRANSPORT else None
        ),
    )
    if (
        expected_binding is not None
        and binding.fingerprint != expected_binding.fingerprint
    ):
        raise ValueError("The Google provider binding changed after labeling started")
    raw_credentials = (model.llm_provider.custom_config or {}).get(
        VERTEX_CREDENTIALS_FILE_KWARG
    )
    config = VertexBatchConfig(
        model_configuration_id=model.id,
        model_name=DEFAULT_MODEL,
        project=binding.project,
        location=binding.location,
        authentication_mode=binding.authentication_mode,
    )
    if transport == _FILES_TRANSPORT:
        return GoogleGeminiFilesBatchGateway(
            config=config,
            credential_json_provider=lambda: raw_credentials,
            flow=LLMFlow.REGULATORY_LABELING_BATCH,
            max_result_bytes=64 * 1024 * 1024,
            request_timeout_seconds=20,
            max_reconciliation_seconds=180,
        )

    from onyx.regulatory.labeling.vertex_batch import LabelingVertexBatchGateway

    assert binding.staging_uri is not None
    return LabelingVertexBatchGateway(
        config=config,
        staging_uri=binding.staging_uri,
        credential_json_provider=lambda: raw_credentials,
        max_result_bytes=64 * 1024 * 1024,
        request_timeout_seconds=20,
        max_reconciliation_seconds=180,
    )
