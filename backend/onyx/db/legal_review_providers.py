"""Resolve actual JEV credentials without changing providers or model visibility."""

from __future__ import annotations

from pydantic import SecretStr
from sqlalchemy import select
from sqlalchemy.orm import Session, load_only, selectinload

from onyx.auth.schemas import UserRole
from onyx.configs import app_configs
from onyx.db.engine.sql_engine import get_session_with_current_tenant
from onyx.db.llm import can_user_access_llm_provider, fetch_user_group_ids
from onyx.db.models import LLMProvider, Persona, User, UserGroup
from onyx.db.persona import get_default_behavior_persona
from onyx.legal_review.models import (
    DIAGNOSIS_MODEL,
    DecisionProviderConfig,
    JevProviderConfig,
)
from onyx.server.usage_limits import check_llm_cost_limit_for_provider
from shared_configs.contextvars import get_current_tenant_id

_OPENROUTER_BASE = "https://openrouter.ai/api/v1"
_OPENAI_BASE = "https://api.openai.com/v1"


def resolve_legal_review_decision(
    user: User,
    *,
    persona: Persona | None = None,
    db_session: Session | None = None,
) -> DecisionProviderConfig | None:
    """Use an accessible configured OpenAI row for the dedicated Decisions API."""
    if db_session is not None:
        return _resolve_decision(db_session, user, persona)
    with get_session_with_current_tenant() as session:
        return _resolve_decision(session, user, persona)


def _resolve_decision(
    session: Session, user: User, persona: Persona | None
) -> DecisionProviderConfig | None:
    persona = persona or get_default_behavior_persona(session)
    if persona is None:
        return None
    providers = session.scalars(
        select(LLMProvider)
        .where(
            LLMProvider.provider == "openai",
            LLMProvider.model_configurations.any(name="gpt-6-luna", is_visible=True),
            LLMProvider.model_configurations.any(name=DIAGNOSIS_MODEL, is_visible=True),
        )
        .options(
            load_only(
                LLMProvider.id,
                LLMProvider.name,
                LLMProvider.api_base,
                LLMProvider.is_public,
            ),
            selectinload(LLMProvider.groups).load_only(UserGroup.id),
            selectinload(LLMProvider.personas).load_only(Persona.id),
        )
        .order_by(LLMProvider.id.asc())
    )
    group_ids = fetch_user_group_ids(session, user)
    for provider in providers:
        if not can_user_access_llm_provider(
            provider, group_ids, persona, is_admin=user.role == UserRole.ADMIN
        ):
            continue
        base = (provider.api_base or "").strip().rstrip("/")
        if base and base != _OPENAI_BASE:
            continue
        session.refresh(provider, attribute_names=["api_key"])
        if provider.api_key is None:
            continue
        key = provider.api_key.get_value(apply_mask=False).strip()
        if key:
            _check_cost_limit(session, key)
            return DecisionProviderConfig(
                api_key=SecretStr(key), provider_name=provider.name
            )
    return None


def resolve_legal_review_jev(
    user: User,
    *,
    persona: Persona | None = None,
    db_session: Session | None = None,
) -> JevProviderConfig | None:
    """Prefer the direct key; otherwise use an accessible official OpenRouter row."""
    direct_key = (app_configs.TYPESAFE_API_KEY or "").strip()
    if direct_key:
        if db_session is not None:
            _check_cost_limit(db_session, direct_key)
        else:
            with get_session_with_current_tenant() as session:
                _check_cost_limit(session, direct_key)
        return JevProviderConfig(route="typesafe", api_key=SecretStr(direct_key))
    if db_session is not None:
        return _resolve_openrouter(db_session, user, persona)
    with get_session_with_current_tenant() as session:
        return _resolve_openrouter(session, user, persona)


def _resolve_openrouter(
    session: Session, user: User, persona: Persona | None
) -> JevProviderConfig | None:
    persona = persona or get_default_behavior_persona(session)
    if persona is None:
        return None
    providers = session.scalars(
        select(LLMProvider)
        .where(LLMProvider.provider == "openrouter")
        .options(
            load_only(
                LLMProvider.id,
                LLMProvider.name,
                LLMProvider.api_base,
                LLMProvider.is_public,
            ),
            selectinload(LLMProvider.groups).load_only(UserGroup.id),
            selectinload(LLMProvider.personas).load_only(Persona.id),
        )
        .order_by(LLMProvider.id.asc())
    )
    group_ids = fetch_user_group_ids(session, user)
    for provider in providers:
        if not can_user_access_llm_provider(
            provider, group_ids, persona, is_admin=user.role == UserRole.ADMIN
        ):
            continue
        base = (provider.api_base or "").strip().rstrip("/")
        if base and base != _OPENROUTER_BASE:
            continue
        # Only decrypt a key after the existing group and persona policy permits it.
        session.refresh(provider, attribute_names=["api_key"])
        if provider.api_key is None:
            continue
        key = provider.api_key.get_value(apply_mask=False).strip()
        if key:
            _check_cost_limit(session, key)
            return JevProviderConfig(
                route="openrouter", api_key=SecretStr(key), provider_name=provider.name
            )
    return None


def _check_cost_limit(session: Session, key: str) -> None:
    check_llm_cost_limit_for_provider(
        db_session=session,
        tenant_id=get_current_tenant_id(),
        llm_provider_api_key=key,
    )
