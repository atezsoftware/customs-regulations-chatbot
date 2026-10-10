"""Provider-free reviewer resolution with existing provider and persona rules."""

from typing import cast
from unittest.mock import MagicMock

import pytest
from sqlalchemy.orm import Session

from onyx.auth.schemas import UserRole
from onyx.configs import app_configs
from onyx.db import legal_review_providers as providers
from onyx.db.models import LLMProvider, ModelConfiguration, Persona, User, UserGroup
from onyx.utils.sensitive import make_mock_sensitive_value


def provider(
    *,
    identity: int = 1,
    public: bool = True,
    base: str | None = "https://openrouter.ai/api/v1",
    key: str | None = "openrouter-fixture-key",
    groups: list[int] | None = None,
    personas: list[int] | None = None,
) -> LLMProvider:
    return LLMProvider(
        id=identity,
        name=f"OpenRouter {identity}",
        provider="openrouter",
        api_base=base,
        api_key=make_mock_sensitive_value(key),
        is_public=public,
        groups=[UserGroup(id=value) for value in groups or []],
        personas=[Persona(id=value) for value in personas or []],
    )


@pytest.fixture(autouse=True)
def no_direct_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app_configs, "TYPESAFE_API_KEY", None)
    monkeypatch.setattr(providers, "fetch_user_group_ids", lambda *_args: {3})
    monkeypatch.setattr(providers, "check_llm_cost_limit_for_provider", MagicMock())


def user(*, admin: bool = False) -> User:
    value = MagicMock(spec=User)
    value.role = UserRole.ADMIN if admin else UserRole.BASIC
    return value


def test_direct_key_is_preferred_without_provider_query_or_secret_serialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(app_configs, "TYPESAFE_API_KEY", " direct-fixture-key ")
    session_factory = MagicMock()
    monkeypatch.setattr(providers, "get_session_with_current_tenant", session_factory)
    config = providers.resolve_legal_review_jev(user())
    assert config is not None and config.route == "typesafe"
    assert config.api_key.get_secret_value() == "direct-fixture-key"
    assert "direct-fixture-key" not in repr(config)
    assert "api_key" not in config.model_dump()
    session_factory.assert_called_once_with()
    session_factory.return_value.__enter__.return_value.scalars.assert_not_called()


@pytest.mark.parametrize("direct", [True, False])
def test_cost_limit_guard_is_called_and_refusal_cannot_select_other_route(
    direct: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = MagicMock(spec=Session)
    session.scalars.return_value = [provider()]
    if direct:
        monkeypatch.setattr(app_configs, "TYPESAFE_API_KEY", "direct-fixture-key")
    guard = MagicMock(side_effect=RuntimeError("quota exhausted"))
    monkeypatch.setattr(providers, "check_llm_cost_limit_for_provider", guard)
    with pytest.raises(RuntimeError, match="quota exhausted"):
        providers.resolve_legal_review_jev(
            user(), persona=Persona(id=0), db_session=session
        )
    assert guard.call_count == 1
    assert guard.call_args.kwargs["db_session"] is session
    assert guard.call_args.kwargs["llm_provider_api_key"] == (
        "direct-fixture-key" if direct else "openrouter-fixture-key"
    )
    if direct:
        session.scalars.assert_not_called()


@pytest.mark.parametrize("base", [None, "", "https://openrouter.ai/api/v1/"])
def test_official_openrouter_route_needs_no_chat_model_registration(
    base: str | None,
) -> None:
    row = provider(base=base)
    session = MagicMock(spec=Session)
    session.scalars.return_value = [row]
    config = providers.resolve_legal_review_jev(
        user(), persona=Persona(id=0), db_session=session
    )
    assert config is not None and config.route == "openrouter"
    assert config.provider_name == "OpenRouter 1"
    assert config.api_key.get_secret_value() == "openrouter-fixture-key"
    session.refresh.assert_called_once_with(row, attribute_names=["api_key"])
    cast(MagicMock, row.api_key).get_value.assert_called_once_with(apply_mask=False)


@pytest.mark.parametrize(
    "row",
    [
        provider(public=False, groups=[9]),
        provider(personas=[7]),
        provider(base="https://proxy.example/api/v1"),
        provider(base="http://openrouter.ai/api/v1"),
        provider(base="https://openrouter.ai/api/v1?credential=secret"),
        provider(key=None),
        provider(key=" "),
    ],
)
def test_inaccessible_proxy_or_missing_key_fails_closed(row: LLMProvider) -> None:
    session = MagicMock(spec=Session)
    session.scalars.return_value = [row]
    assert (
        providers.resolve_legal_review_jev(
            user(), persona=Persona(id=0), db_session=session
        )
        is None
    )
    if row.api_key is not None and row.api_base != "https://openrouter.ai/api/v1":
        cast(MagicMock, row.api_key).get_value.assert_not_called()


def test_admin_still_obeys_persona_restriction_and_next_accessible_row_is_used() -> (
    None
):
    blocked = provider(identity=1, public=False, personas=[7])
    allowed = provider(identity=2, public=False, groups=[9], personas=[0])
    session = MagicMock(spec=Session)
    session.scalars.return_value = [blocked, allowed]
    config = providers.resolve_legal_review_jev(
        user(admin=True), persona=Persona(id=0), db_session=session
    )
    assert config is not None and config.provider_name == "OpenRouter 2"
    cast(MagicMock, blocked.api_key).get_value.assert_not_called()
    session.refresh.assert_called_once_with(allowed, attribute_names=["api_key"])


def test_default_persona_is_resolved_in_same_tenant_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = MagicMock(spec=Session)
    row = provider(personas=[0])
    session.scalars.return_value = [row]
    monkeypatch.setattr(
        providers, "get_default_behavior_persona", lambda _session: Persona(id=0)
    )
    config = providers.resolve_legal_review_jev(user(), db_session=session)
    assert config is not None


def test_missing_default_persona_cannot_use_even_public_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = MagicMock(spec=Session)
    monkeypatch.setattr(
        providers, "get_default_behavior_persona", lambda _session: None
    )
    assert providers.resolve_legal_review_jev(user(), db_session=session) is None
    session.scalars.assert_not_called()


def decision_provider(
    *,
    identity: int = 1,
    public: bool = True,
    base: str | None = "https://api.openai.com/v1",
    key: str | None = "openai-fixture-key",
    groups: list[int] | None = None,
    personas: list[int] | None = None,
) -> LLMProvider:
    return LLMProvider(
        id=identity,
        name=f"OpenAI {identity}",
        provider="openai",
        api_base=base,
        api_key=make_mock_sensitive_value(key),
        is_public=public,
        groups=[UserGroup(id=value) for value in groups or []],
        personas=[Persona(id=value) for value in personas or []],
        model_configurations=[
            ModelConfiguration(name="gpt-6-luna", is_visible=True),
            ModelConfiguration(name="gpt-6.1-sol", is_visible=True),
        ],
    )


@pytest.mark.parametrize("base", [None, "", "https://api.openai.com/v1/"])
def test_decision_uses_accessible_official_openai_without_mutating_provider(
    base: str | None,
) -> None:
    row = decision_provider(base=base, public=False, groups=[3], personas=[0])
    session = MagicMock(spec=Session)
    session.scalars.return_value = [row]
    before = (
        row.name,
        row.provider,
        row.api_base,
        row.is_public,
        [(model.name, model.is_visible) for model in row.model_configurations],
    )
    config = providers.resolve_legal_review_decision(
        user(), persona=Persona(id=0), db_session=session
    )
    assert config is not None and config.route == "openai_decisions"
    assert config.provider_name == "OpenAI 1"
    assert config.api_key.get_secret_value() == "openai-fixture-key"
    assert "openai-fixture-key" not in repr(config)
    assert config.model_dump() == {
        "route": "openai_decisions",
        "provider_name": "OpenAI 1",
    }
    session.refresh.assert_called_once_with(row, attribute_names=["api_key"])
    cast(MagicMock, row.api_key).get_value.assert_called_once_with(apply_mask=False)
    assert before == (
        row.name,
        row.provider,
        row.api_base,
        row.is_public,
        [(model.name, model.is_visible) for model in row.model_configurations],
    )
    session.add.assert_not_called()
    session.add_all.assert_not_called()
    session.delete.assert_not_called()
    session.flush.assert_not_called()
    session.commit.assert_not_called()
    statement = session.scalars.call_args.args[0]
    sql = str(statement.compile(compile_kwargs={"literal_binds": True}))
    assert "llm_provider.provider = 'openai'" in sql
    assert "model_configuration.name = 'gpt-6-luna'" in sql
    assert "model_configuration.name = 'gpt-6.1-sol'" in sql
    assert "model_configuration.is_visible = true" in sql


@pytest.mark.parametrize(
    "base",
    [
        "https://proxy.example/v1",
        "http://api.openai.com/v1",
        "https://api.openai.com/v1?credential=secret",
        "https://api.openai.com.attacker.example/v1",
    ],
)
def test_decision_off_domain_provider_is_rejected_before_decryption(base: str) -> None:
    row = decision_provider(base=base)
    session = MagicMock(spec=Session)
    session.scalars.return_value = [row]
    assert (
        providers.resolve_legal_review_decision(
            user(), persona=Persona(id=0), db_session=session
        )
        is None
    )
    session.refresh.assert_not_called()
    cast(MagicMock, row.api_key).get_value.assert_not_called()


@pytest.mark.parametrize(
    "row",
    [decision_provider(public=False, groups=[9]), decision_provider(personas=[7])],
)
def test_decision_inaccessible_provider_cannot_decrypt(row: LLMProvider) -> None:
    session = MagicMock(spec=Session)
    session.scalars.return_value = [row]
    assert (
        providers.resolve_legal_review_decision(
            user(), persona=Persona(id=0), db_session=session
        )
        is None
    )
    session.refresh.assert_not_called()
    cast(MagicMock, row.api_key).get_value.assert_not_called()


@pytest.mark.parametrize(
    "rows", [[], [decision_provider(key=None)], [decision_provider(key=" ")]]
)
def test_decision_missing_provider_or_credential_has_no_alternate_route(
    rows: list[LLMProvider], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(app_configs, "TYPESAFE_API_KEY", "legacy-direct-fixture-key")
    session = MagicMock(spec=Session)
    session.scalars.return_value = rows
    assert (
        providers.resolve_legal_review_decision(
            user(), persona=Persona(id=0), db_session=session
        )
        is None
    )
    cast(MagicMock, providers.check_llm_cost_limit_for_provider).assert_not_called()


def test_decision_cost_limit_refusal_cannot_select_another_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = decision_provider(identity=1)
    second = decision_provider(identity=2)
    session = MagicMock(spec=Session)
    session.scalars.return_value = [first, second]
    guard = MagicMock(side_effect=RuntimeError("quota exhausted"))
    monkeypatch.setattr(providers, "check_llm_cost_limit_for_provider", guard)
    with pytest.raises(RuntimeError, match="quota exhausted"):
        providers.resolve_legal_review_decision(
            user(), persona=Persona(id=0), db_session=session
        )
    guard.assert_called_once()
    assert guard.call_args.kwargs["db_session"] is session
    assert guard.call_args.kwargs["llm_provider_api_key"] == "openai-fixture-key"
    cast(MagicMock, second.api_key).get_value.assert_not_called()


def test_decision_admin_still_obeys_persona_restriction() -> None:
    blocked = decision_provider(identity=1, public=False, personas=[7])
    allowed = decision_provider(identity=2, public=False, groups=[9], personas=[0])
    session = MagicMock(spec=Session)
    session.scalars.return_value = [blocked, allowed]
    config = providers.resolve_legal_review_decision(
        user(admin=True), persona=Persona(id=0), db_session=session
    )
    assert config is not None and config.provider_name == "OpenAI 2"
    cast(MagicMock, blocked.api_key).get_value.assert_not_called()


@pytest.mark.parametrize("available", [False, True])
def test_decision_requires_the_existing_default_persona(
    available: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = MagicMock(spec=Session)
    session.scalars.return_value = [decision_provider(personas=[0])]
    monkeypatch.setattr(
        providers,
        "get_default_behavior_persona",
        lambda _session: Persona(id=0) if available else None,
    )
    assert (
        providers.resolve_legal_review_decision(user(), db_session=session) is not None
    ) is available
    if not available:
        session.scalars.assert_not_called()
