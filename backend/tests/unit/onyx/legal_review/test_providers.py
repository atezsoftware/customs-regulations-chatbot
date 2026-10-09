"""Provider-free JEV resolution with existing provider group and persona rules."""

from typing import cast
from unittest.mock import MagicMock

import pytest
from sqlalchemy.orm import Session

from onyx.auth.schemas import UserRole
from onyx.configs import app_configs
from onyx.db import legal_review_providers as providers
from onyx.db.models import LLMProvider, Persona, User, UserGroup
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
