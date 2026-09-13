"""Labeling provider selection and frozen identities against real PostgreSQL."""

import json
from collections.abc import Callable, Generator
from typing import Any, cast
from unittest.mock import patch
from uuid import uuid4

import pytest
from sqlalchemy import Connection, Engine, event
from sqlalchemy.engine import ExecutionContext
from sqlalchemy.orm import Session

from onyx.db import labeling_configuration
from onyx.db.labeling_configuration import (
    get_labeling_provider_options,
    resolve_labeling_gateway,
    resolve_labeling_provider_binding,
)
from onyx.db.models import (
    ImageGenerationConfig,
    LLMProvider,
    LLMProvider__Persona,
    LLMProvider__UserGroup,
    ModelConfiguration,
    Persona,
    User__UserGroup,
    UserGroup,
    UserRole,
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
from onyx.regulatory.indexing_jobs.models import (
    VertexAuthenticationMode,
    VertexBatchConfig,
)
from onyx.tracing.flows import LLMFlow
from tests.external_dependency_unit.craft.db_helpers import make_user

_PRIVATE_KEY = "labeling-test-private-key-not-a-real-credential"
_ROTATED_KEY = "labeling-test-rotated-key-not-a-real-credential"


@pytest.fixture
def labeling_session(db_session: Session) -> Generator[Session, None, None]:
    # The outer transaction also rolls back if a tested helper starts committing.
    engine = db_session.get_bind()
    assert isinstance(engine, Engine)
    with engine.connect() as connection:
        transaction = connection.begin()
        try:
            with Session(
                bind=connection, join_transaction_mode="create_savepoint"
            ) as session:
                yield session
        finally:
            transaction.rollback()


def _credentials(
    *,
    project: str = "labeling-project",
    principal: str = "labeler@labeling-project.iam.gserviceaccount.com",
    private_key: str = _PRIVATE_KEY,
) -> str:
    return json.dumps(
        {
            "type": "service_account",
            "project_id": project,
            "client_email": principal,
            "private_key": private_key,
            "private_key_id": "test-key-identifier",
            "token_uri": "https://oauth2.googleapis.com/token",
        }
    )


def _model(
    session: Session,
    *,
    is_public: bool = True,
    visible: bool = True,
    provider_type: str = LlmProviderNames.VERTEX_AI,
    name: str = "gemini-2.5-flash",
) -> ModelConfiguration:
    provider = LLMProvider(
        name=f"Labeling test {uuid4().hex}",
        provider=provider_type,
        is_public=is_public,
        custom_config={
            VERTEX_AUTH_METHOD_KWARG: VERTEX_AUTH_METHOD_SERVICE_ACCOUNT,
            VERTEX_CREDENTIALS_FILE_KWARG: _credentials(),
            VERTEX_LOCATION_KWARG: "global",
        },
    )
    model = ModelConfiguration(llm_provider=provider, name=name, is_visible=visible)
    session.add(model)
    session.flush()
    return model


@pytest.mark.parametrize(
    "public,restricted_group,member,restricted_persona,role,allowed",
    [
        (True, False, False, False, UserRole.BASIC, True),
        (False, False, False, False, UserRole.BASIC, False),
        (False, False, False, False, UserRole.ADMIN, True),
        (False, True, True, False, UserRole.BASIC, True),
        (False, True, False, False, UserRole.BASIC, False),
        (False, True, False, False, UserRole.ADMIN, True),
        (True, False, False, True, UserRole.BASIC, False),
        (True, False, False, True, UserRole.ADMIN, False),
        (False, True, True, True, UserRole.BASIC, False),
        (False, True, False, True, UserRole.ADMIN, False),
    ],
)
def test_catalog_and_start_enforce_provider_access_without_persona_context(
    labeling_session: Session,
    public: bool,
    restricted_group: bool,
    member: bool,
    restricted_persona: bool,
    role: UserRole,
    allowed: bool,
) -> None:
    user = make_user(labeling_session, role=role)
    model = _model(labeling_session, is_public=public)
    if restricted_group:
        group = UserGroup(name=f"labeling-group-{uuid4().hex}")
        labeling_session.add(group)
        labeling_session.flush()
        labeling_session.add(
            LLMProvider__UserGroup(
                llm_provider_id=model.llm_provider_id, user_group_id=group.id
            )
        )
        if member:
            labeling_session.add(
                User__UserGroup(user_id=user.id, user_group_id=group.id)
            )
    if restricted_persona:
        persona = Persona(
            name=f"labeling-persona-{uuid4().hex}",
            description="Provider restriction fixture",
            user_id=user.id,
        )
        labeling_session.add(persona)
        labeling_session.flush()
        labeling_session.add(
            LLMProvider__Persona(
                llm_provider_id=model.llm_provider_id, persona_id=persona.id
            )
        )
    labeling_session.flush()
    options = get_labeling_provider_options(labeling_session, user=user)
    assert (model.id in {option["id"] for option in options}) is allowed
    if allowed:
        binding = resolve_labeling_provider_binding(
            labeling_session, model.id, user=user
        )
        assert binding.provider_id == model.llm_provider_id
    else:
        with pytest.raises(ValueError, match="unavailable to this user"):
            resolve_labeling_provider_binding(labeling_session, model.id, user=user)
        with pytest.raises(ValueError, match="unavailable to this user"):
            resolve_labeling_gateway(labeling_session, model.id, user=user)


@pytest.mark.parametrize(
    "provider_type,visible",
    [(LlmProviderNames.VERTEX_AI, False), (LlmProviderNames.OPENAI, True)],
)
def test_unsupported_or_hidden_models_cannot_be_selected_by_id(
    labeling_session: Session, provider_type: str, visible: bool
) -> None:
    user = make_user(labeling_session, role=UserRole.ADMIN)
    model = _model(labeling_session, provider_type=provider_type, visible=visible)
    assert model.id not in {
        option["id"]
        for option in get_labeling_provider_options(labeling_session, user=user)
    }
    with pytest.raises(ValueError, match="enabled Google model"):
        resolve_labeling_provider_binding(labeling_session, model.id, user=user)
    with pytest.raises(ValueError, match="enabled Google model"):
        resolve_labeling_gateway(labeling_session, model.id, user=user)


def test_one_option_per_provider_prefers_labeling_model_without_exposing_credentials(
    labeling_session: Session,
) -> None:
    user = make_user(labeling_session, role=UserRole.BASIC)
    contextual_model = _model(labeling_session)
    labeling_model = ModelConfiguration(
        llm_provider=contextual_model.llm_provider,
        name="gemini-3.8-flash",
        is_visible=True,
    )
    labeling_session.add(labeling_model)
    labeling_session.flush()
    options = get_labeling_provider_options(labeling_session, user=user)
    owned_options = [
        option
        for option in options
        if option["id"] in {contextual_model.id, labeling_model.id}
    ]
    assert owned_options == [
        {"id": labeling_model.id, "name": contextual_model.llm_provider.name}
    ]
    binding = resolve_labeling_provider_binding(
        labeling_session, labeling_model.id, user=user
    )
    serialized = json.dumps(options) + binding.model_dump_json()
    for secret in (_PRIVATE_KEY, "test-key-identifier", "private_key", "token_uri"):
        assert secret not in serialized


def test_image_generation_provider_is_excluded_by_association_not_name(
    labeling_session: Session,
) -> None:
    user = make_user(labeling_session, role=UserRole.ADMIN)
    image_model = _model(labeling_session, name="gemini-3.8-flash")
    labeling_session.add(
        ImageGenerationConfig(
            image_provider_id=f"image-{uuid4().hex}",
            model_configuration_id=image_model.id,
            is_default=False,
        )
    )
    ordinary_model = _model(labeling_session, name="gemini-3.8-flash")
    ordinary_model.llm_provider.name = "Google Image analysis"
    labeling_session.flush()

    option_ids = {
        option["id"]
        for option in get_labeling_provider_options(labeling_session, user=user)
    }
    assert image_model.id not in option_ids
    assert ordinary_model.id in option_ids
    assert (
        resolve_labeling_provider_binding(
            labeling_session, ordinary_model.id, user=user
        ).provider_id
        == ordinary_model.llm_provider_id
    )
    with pytest.raises(ValueError, match="unavailable"):
        resolve_labeling_provider_binding(labeling_session, image_model.id, user=user)
    with pytest.raises(ValueError, match="unavailable"):
        resolve_labeling_gateway(labeling_session, image_model.id, user=user)


def test_labeling_uses_fixed_model_and_current_key_after_same_principal_rotation(
    labeling_session: Session,
) -> None:
    user = make_user(labeling_session, role=UserRole.BASIC)
    model = _model(labeling_session, name="gemini-2.5-pro-contextual-only")
    binding = resolve_labeling_provider_binding(labeling_session, model.id, user=user)
    assert binding.model == "gemini-3.8-flash"
    rotated_credentials = _credentials(private_key=_ROTATED_KEY)
    model.llm_provider.custom_config = {
        **(model.llm_provider.custom_config or {}),
        VERTEX_CREDENTIALS_FILE_KWARG: rotated_credentials,
    }
    labeling_session.flush()
    current = resolve_labeling_provider_binding(labeling_session, model.id, user=user)
    assert current.fingerprint == binding.fingerprint
    with patch.object(
        labeling_configuration, "GoogleGeminiFilesBatchGateway", autospec=True
    ) as gateway_constructor:
        resolve_labeling_gateway(
            labeling_session, model.id, user=user, expected_binding=binding
        )
    arguments = gateway_constructor.call_args.kwargs
    config = arguments["config"]
    assert isinstance(config, VertexBatchConfig)
    assert config.model_name == "gemini-3.8-flash"
    assert config.project == "labeling-project"
    assert config.authentication_mode == VertexAuthenticationMode.SERVICE_ACCOUNT_JSON
    assert arguments["flow"] == LLMFlow.REGULATORY_LABELING_BATCH
    credential_provider = cast(
        Callable[[], str | None], arguments["credential_json_provider"]
    )
    assert credential_provider() == rotated_credentials
    assert _ROTATED_KEY not in current.model_dump_json()


@pytest.mark.parametrize("changed_field", ["principal", "project", "location", "mode"])
def test_frozen_provider_identity_rejects_drift_before_gateway_creation(
    labeling_session: Session, changed_field: str
) -> None:
    user = make_user(labeling_session, role=UserRole.BASIC)
    model = _model(labeling_session)
    binding = resolve_labeling_provider_binding(labeling_session, model.id, user=user)
    custom = dict(model.llm_provider.custom_config or {})
    if changed_field == "principal":
        custom[VERTEX_CREDENTIALS_FILE_KWARG] = _credentials(
            principal="someone-else@labeling-project.iam.gserviceaccount.com"
        )
    elif changed_field == "project":
        custom[VERTEX_CREDENTIALS_FILE_KWARG] = _credentials(project="other-project")
    elif changed_field == "location":
        custom[VERTEX_LOCATION_KWARG] = "europe-west4"
    else:
        custom[VERTEX_AUTH_METHOD_KWARG] = VERTEX_AUTH_METHOD_WORKLOAD_IDENTITY
        custom[VERTEX_PROJECT_KWARG] = "labeling-project"
    model.llm_provider.custom_config = custom
    labeling_session.flush()
    with pytest.raises(ValueError, match="binding changed"):
        resolve_labeling_gateway(
            labeling_session, model.id, user=user, expected_binding=binding
        )


def test_resume_rechecks_revoked_provider_access_even_with_matching_binding(
    labeling_session: Session,
) -> None:
    user = make_user(labeling_session, role=UserRole.BASIC)
    model = _model(labeling_session)
    binding = resolve_labeling_provider_binding(labeling_session, model.id, user=user)
    model.llm_provider.is_public = False
    labeling_session.flush()
    with pytest.raises(ValueError, match="unavailable to this user"):
        resolve_labeling_gateway(
            labeling_session, model.id, user=user, expected_binding=binding
        )


@pytest.mark.parametrize("operation", ["catalog", "binding"])
def test_labeling_resolution_does_not_load_unneeded_models_or_scale_query_count(
    labeling_session: Session, operation: str
) -> None:
    user = make_user(labeling_session, role=UserRole.BASIC)
    selected = _model(labeling_session, name="gemini-3.8-flash")
    unrelated = _model(
        labeling_session, provider_type=LlmProviderNames.OPENAI, name="gpt-5-mini"
    )
    extra_models = [
        ModelConfiguration(
            llm_provider=provider,
            name=f"labeling-perf-{visible}-{index}",
            is_visible=visible,
        )
        for provider, visible in (
            (selected.llm_provider, True),
            (selected.llm_provider, False),
            (unrelated.llm_provider, True),
        )
        for index in range(40)
    ]
    labeling_session.add_all(extra_models)
    labeling_session.flush()
    selected_id = selected.id
    selected_name = selected.llm_provider.name
    unneeded_ids = {model.id for model in extra_models} | {unrelated.id}
    labeling_session.expunge_all()
    statements: list[str] = []
    loaded_model_ids: set[int] = set()

    def record_query(
        _connection: Connection,
        _cursor: Any,
        statement: str,
        _parameters: Any,
        _context: ExecutionContext,
        _executemany: bool,
    ) -> None:
        if statement.lstrip().upper().startswith("SELECT"):
            statements.append(statement)

    def record_load(_session: Session, instance: object) -> None:
        if isinstance(instance, ModelConfiguration):
            loaded_model_ids.add(instance.id)

    bind = labeling_session.get_bind()
    event.listen(bind, "before_cursor_execute", record_query)
    event.listen(labeling_session, "loaded_as_persistent", record_load)
    try:
        if operation == "catalog":
            assert {"id": selected_id, "name": selected_name} in (
                get_labeling_provider_options(labeling_session, user=user)
            )
        else:
            binding = resolve_labeling_provider_binding(
                labeling_session, selected_id, user=user
            )
            assert binding.model_configuration_id == selected_id
    finally:
        event.remove(bind, "before_cursor_execute", record_query)
        event.remove(labeling_session, "loaded_as_persistent", record_load)

    assert selected_id in loaded_model_ids
    assert not loaded_model_ids.intersection(unneeded_ids)
    assert len(statements) <= 8
    assert not any("llm_model_flow" in statement for statement in statements)
