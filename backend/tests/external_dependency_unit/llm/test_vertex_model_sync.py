"""Vertex discovery persists a usable catalog across independent DB sessions.

Requires PostgreSQL with CREATE SCHEMA permission. Each test owns its schema;
use CACHE_BACKEND=postgres to exercise cache invalidation without Redis.
"""

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from unittest.mock import patch
from uuid import UUID, uuid4

import httpx
import pytest
from google import genai
from google.auth.credentials import Credentials as GoogleCredentials
from google.genai import types
from google.oauth2.credentials import Credentials
from sqlalchemy import select
from sqlalchemy.schema import CreateSchema, DropSchema

from onyx.auth.schemas import UserRole
from onyx.configs.constants import ANONYMOUS_USER_UUID
from onyx.db.engine.sql_engine import (
    SqlEngine,
    get_session_with_current_tenant,
    get_sqlalchemy_engine,
)
from onyx.db.enums import LLMModelFlowType
from onyx.db.llm import fetch_default_llm_model, sync_vertex_model_configurations
from onyx.db.models import Base, LLMModelFlow, LLMProvider, ModelConfiguration, User
from onyx.llm.well_known_providers.auto_update_service import (
    sync_vertex_models_from_google,
)
from onyx.llm.well_known_providers.vertex_models import discover_vertex_models
from onyx.server.manage.llm.api import list_llm_provider_basics, put_llm_provider
from onyx.server.manage.llm.models import (
    LLMProviderDescriptor,
    LLMProviderResponse,
    LLMProviderUpsertRequest,
    ModelConfigurationUpsertRequest,
    SyncModelEntry,
)
from shared_configs.contextvars import CURRENT_TENANT_ID_CONTEXTVAR


@pytest.fixture
def vertex_database() -> Iterator[None]:
    SqlEngine.init_engine(pool_size=5, max_overflow=2)
    engine = get_sqlalchemy_engine()
    schema = f"test_vertex_sync_{uuid4().hex}"
    table_names = {
        "cache_store",
        "image_generation_config",
        "llm_provider",
        "llm_provider__persona",
        "llm_provider__user_group",
        "model_configuration",
        "llm_model_flow",
        "persona",
        "user",
        "user_group",
    }
    with engine.begin() as connection:
        connection.execute(CreateSchema(schema))
        Base.metadata.create_all(
            connection.execution_options(schema_translate_map={None: schema}),
            tables=[Base.metadata.tables[name] for name in sorted(table_names)],
        )
    token = CURRENT_TENANT_ID_CONTEXTVAR.set(schema)
    try:
        yield
    finally:
        CURRENT_TENANT_ID_CONTEXTVAR.reset(token)
        with engine.begin() as connection:
            connection.execute(DropSchema(schema, cascade=True))


@dataclass
class GoogleCatalog:
    before_response: Callable[[], None] | None = None

    def respond(self, request: httpx.Request) -> httpx.Response:
        assert request.url.host == "aiplatform.googleapis.com"
        assert "publishers/google/models" in request.url.path
        assert request.headers["authorization"] == "Bearer vertex-test-token"
        if self.before_response is not None:
            callback, self.before_response = self.before_response, None
            callback()
        if request.url.params.get("pageToken") == "next":
            return httpx.Response(
                200,
                json={
                    "publisherModels": [
                        {
                            "name": "publishers/google/models/gemini-future-flash",
                            "displayName": "Future Flash",
                        }
                    ]
                },
            )
        return httpx.Response(
            200,
            json={
                "publisherModels": [
                    {
                        "name": "publishers/google/models/gemini-2.5-pro",
                        "displayName": "Gemini 2.5 Pro",
                    },
                    {"name": "publishers/google/models/gemini-flash-image"},
                ],
                "nextPageToken": "next",
            },
        )


@pytest.fixture
def google_catalog() -> Iterator[GoogleCatalog]:
    catalog = GoogleCatalog()
    real_client = genai.Client

    def client_with_transport(
        *,
        vertexai: bool,
        credentials: GoogleCredentials,
        project: str,
        location: str,
        http_options: types.HttpOptions,
    ) -> genai.Client:
        return real_client(
            vertexai=vertexai,
            credentials=credentials,
            project=project,
            location=location,
            http_options=http_options.model_copy(
                update={
                    "client_args": {"transport": httpx.MockTransport(catalog.respond)}
                }
            ),
        )

    with (
        patch.object(genai, "Client", side_effect=client_with_transport),
        patch(
            "google.auth.default",
            return_value=(Credentials("vertex-test-token"), "test-project"),
        ),
    ):
        yield catalog


def _create_provider(*, auto: bool) -> tuple[int, dict[str, str]]:
    config = {
        "vertex_auth_method": "workload_identity",
        "vertex_project": "test-project",
        "vertex_location": "global",
    }
    with get_session_with_current_tenant() as session:
        provider = LLMProvider(
            name="Auto Vertex" if auto else "Manual Vertex",
            provider="vertex_ai",
            custom_config=config,
            is_public=True,
            is_auto_mode=auto,
        )
        session.add(provider)
        session.flush()
        for name, visible in (("gemini-pinned-001", True), ("gemini-2.5-pro", False)):
            model = ModelConfiguration(
                llm_provider_id=provider.id,
                name=name,
                is_visible=visible,
                display_name=name,
                custom_display_name="Custom selected name" if visible else None,
                max_input_tokens=8192,
            )
            session.add(model)
            session.flush()
            session.add(
                LLMModelFlow(
                    model_configuration_id=model.id,
                    llm_model_flow_type=LLMModelFlowType.CHAT,
                    is_default=visible and auto,
                )
            )
        session.commit()
        return provider.id, config


def _user_listing() -> LLMProviderResponse[LLMProviderDescriptor]:
    user = User(id=UUID(ANONYMOUS_USER_UUID), role=UserRole.BASIC)
    with get_session_with_current_tenant() as session:
        return list_llm_provider_basics(user=user, db_session=session)


@pytest.mark.usefixtures("vertex_database", "google_catalog")
def test_auto_sync_updates_user_catalog_preserving_admin_choices() -> None:
    auto_id, _ = _create_provider(auto=True)
    manual_id, _ = _create_provider(auto=False)
    before = _user_listing()
    assert all(
        model.name != "gemini-future-flash"
        for provider in before.providers
        for model in provider.model_configurations
    )

    assert sync_vertex_models_from_google() == {str(auto_id): 1}

    after = _user_listing()
    auto = next(provider for provider in after.providers if provider.id == auto_id)
    models = {model.name: model for model in auto.model_configurations}
    assert {name for name, model in models.items() if model.is_visible} == {
        "gemini-pinned-001",
        "gemini-2.5-pro",
        "gemini-future-flash",
    }
    assert models["gemini-future-flash"].display_name == "Future Flash"
    assert models["gemini-pinned-001"].custom_display_name == "Custom selected name"
    assert models["gemini-pinned-001"].max_input_tokens == 8192
    assert models["gemini-2.5-pro"].max_input_tokens == 8192
    manual = next(provider for provider in after.providers if provider.id == manual_id)
    assert {model.name: model.is_visible for model in manual.model_configurations} == {
        "gemini-pinned-001": True,
        "gemini-2.5-pro": False,
    }
    with get_session_with_current_tenant() as session:
        default = fetch_default_llm_model(session)
        assert default is not None
        assert (default.llm_provider_id, default.name) == (auto_id, "gemini-pinned-001")
        current = session.scalar(
            select(ModelConfiguration).where(
                ModelConfiguration.llm_provider_id == auto_id,
                ModelConfiguration.name == "gemini-2.5-pro",
            )
        )
        assert current is not None
        assert LLMModelFlowType.VISION in current.llm_model_flow_types
        assert LLMModelFlowType.REASONING in current.llm_model_flow_types
    assert sync_vertex_models_from_google() == {str(auto_id): 0}


@pytest.mark.usefixtures("vertex_database", "google_catalog")
def test_auto_save_refreshes_models_and_preserves_pinned_default() -> None:
    provider_id, config = _create_provider(auto=True)
    _user_listing()
    request = LLMProviderUpsertRequest(
        id=provider_id,
        name="Auto Vertex",
        provider="vertex_ai",
        custom_config=config,
        custom_config_changed=False,
        is_auto_mode=True,
        model_configurations=[
            ModelConfigurationUpsertRequest(
                name="gemini-2.5-pro", is_visible=False, max_input_tokens=8192
            )
        ],
    )
    with (
        patch(
            "onyx.server.manage.llm.api.fetch_llm_recommendations_from_github",
            side_effect=AssertionError("Vertex Auto discovery must use Google"),
        ),
        get_session_with_current_tenant() as session,
    ):
        result = put_llm_provider(
            llm_provider_upsert_request=request,
            is_creation=False,
            user=User(id=uuid4(), role=UserRole.ADMIN),
            db_session=session,
        )
    with get_session_with_current_tenant() as session:
        saved_model = session.scalar(
            select(ModelConfiguration).where(
                ModelConfiguration.llm_provider_id == provider_id,
                ModelConfiguration.name == "gemini-future-flash",
            )
        )
        assert saved_model is not None
        saved_model_id = saved_model.id
    models = {model.name: model for model in result.model_configurations}
    assert models["gemini-future-flash"].is_visible
    assert models["gemini-future-flash"].id == saved_model_id
    assert models["gemini-pinned-001"].is_visible
    assert models["gemini-pinned-001"].custom_display_name == "Custom selected name"
    assert models["gemini-pinned-001"].max_input_tokens == 8192
    with get_session_with_current_tenant() as session:
        default = fetch_default_llm_model(session)
        assert default is not None
        assert (default.llm_provider_id, default.name) == (
            provider_id,
            "gemini-pinned-001",
        )
    provider = next(p for p in _user_listing().providers if p.id == provider_id)
    assert any(
        model.name == "gemini-future-flash" and model.is_visible
        for model in provider.model_configurations
    )


@pytest.mark.usefixtures("vertex_database", "google_catalog")
def test_manual_refresh_stores_new_models_without_selecting_them() -> None:
    provider_id, config = _create_provider(auto=False)
    discovered = discover_vertex_models(config)
    with get_session_with_current_tenant() as session:
        assert (
            sync_vertex_model_configurations(
                session,
                provider_id,
                config,
                [SyncModelEntry(**model.model_dump()) for model in discovered],
                require_auto_mode=False,
            )
            == 1
        )
    provider = next(p for p in _user_listing().providers if p.id == provider_id)
    assert {
        model.name: model.is_visible for model in provider.model_configurations
    } == {
        "gemini-pinned-001": True,
        "gemini-2.5-pro": False,
        "gemini-future-flash": False,
    }


@pytest.mark.usefixtures("vertex_database")
@pytest.mark.parametrize("changed_field", ["credentials", "mode"])
def test_inflight_discovery_does_not_overwrite_admin_changes(
    google_catalog: GoogleCatalog, changed_field: str
) -> None:
    provider_id, config = _create_provider(auto=True)

    def edit_provider() -> None:
        with get_session_with_current_tenant() as session:
            provider = session.get(LLMProvider, provider_id)
            assert provider is not None
            if changed_field == "credentials":
                provider.custom_config = {
                    **config,
                    "vertex_credentials": '{"project_id":"changed-project"}',
                }
            else:
                provider.is_auto_mode = False
            session.commit()

    google_catalog.before_response = edit_provider
    assert sync_vertex_models_from_google() == {}
    provider = next(p for p in _user_listing().providers if p.id == provider_id)
    assert {
        model.name: model.is_visible for model in provider.model_configurations
    } == {
        "gemini-pinned-001": True,
        "gemini-2.5-pro": False,
    }
