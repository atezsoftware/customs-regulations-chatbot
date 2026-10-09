"""Resolve the user's preferred Decisions provider under existing access rules."""

from collections.abc import Callable
from urllib.parse import urlsplit

from pydantic import JsonValue
from sqlalchemy.orm import Session

from onyx.asv3.evidence import EvidenceLedger
from onyx.db.engine.sql_engine import get_session_with_current_tenant
from onyx.db.llm import (
    fetch_accessible_llm_provider_by_id,
    fetch_all_accessible_llm_providers,
)
from onyx.db.models import User
from onyx.legal_composite.budget import WorkflowBudget
from onyx.legal_composite.decisions import DecisionsClassifier
from onyx.legal_composite.gateway import BudgetedGateway
from onyx.legal_composite.reviewer import GatewayAnswerReviewer
from onyx.legal_composite.selection import GatewaySourceClassifier, SourceSelector
from onyx.llm.factory import get_llm
from onyx.llm.interfaces import LLMConfig, LLMUserIdentity
from onyx.server.usage_limits import (
    check_llm_cost_limit_for_provider,
    is_onyx_managed_api_key,
)
from onyx.tracing.flows import LLMFlow
from shared_configs.contextvars import get_current_tenant_id


def build_source_selector(
    *,
    session: Session,
    user: User,
    gateway: BudgetedGateway,
    budget: WorkflowBudget,
    ledger: EvidenceLedger,
    check_active: Callable[[], None],
    token_counter: Callable[[str], int] | None,
    run_id: str,
    scope: dict[str, JsonValue],
) -> SourceSelector:
    providers = [
        provider
        for provider in fetch_all_accessible_llm_providers(session, user)
        if (provider.name or "").strip().casefold() == "embedding"
    ]
    if len(providers) == 1:
        public = providers[0]
        models = [
            model
            for model in public.model_configurations
            if model.name == "gpt-6-luna" and model.is_visible
        ]
        if public.provider == "openai" and len(models) == 1:
            provider = fetch_accessible_llm_provider_by_id(session, user, public.id)
            if provider is not None and not is_onyx_managed_api_key(provider.api_key):
                check_llm_cost_limit_for_provider(
                    db_session=session,
                    tenant_id=get_current_tenant_id(),
                    llm_provider_api_key=provider.api_key,
                )
                try:
                    classifier = DecisionsClassifier(
                        config=LLMConfig(
                            model_provider=provider.provider,
                            model_name="gpt-6-luna",
                            temperature=1,
                            api_key=provider.api_key,
                            api_base=provider.api_base,
                            max_input_tokens=min(
                                models[0].max_input_tokens or 128_000, 128_000
                            ),
                        ),
                        budget=budget,
                        ledger=ledger,
                        check_active=check_active,
                        token_counter=token_counter,
                        run_id=run_id,
                        scope=scope,
                        flow=LLMFlow.LEGAL_COMPOSITE_SELECTION,
                        operative_roles=True,
                        max_parallel_batches=4,
                        preserve_finalization_on_timeout=True,
                    )
                    return SourceSelector(classifier, irrelevance_threshold=0.98)
                except ValueError:
                    # Unsupported endpoint configuration cannot forward credentials.
                    pass
    return SourceSelector(GatewaySourceClassifier(gateway), irrelevance_threshold=0.98)


def build_answer_reviewer(
    *,
    session: Session,
    user: User,
    budget: WorkflowBudget,
    ledger: EvidenceLedger,
    check_active: Callable[[], None],
    token_counter: Callable[[str], int] | None,
    run_id: str,
    scope: dict[str, JsonValue],
) -> GatewayAnswerReviewer | None:
    """Use the calibrated single reviewer; failures never select another model."""
    public_providers = fetch_all_accessible_llm_providers(session, user)
    openrouter = [p for p in public_providers if p.provider == "openrouter"]
    named = [p for p in openrouter if (p.name or "").strip().casefold() == "openrouter"]
    preferred = named if len(named) == 1 else openrouter
    if len(preferred) != 1:
        return None
    public = preferred[0]
    model_name = "openai/gpt-5.6-luna"
    models = [
        model
        for model in public.model_configurations
        if model.name == model_name and model.is_visible
    ]
    if len(models) != 1:
        return None
    provider = fetch_accessible_llm_provider_by_id(session, user, public.id)
    if (
        provider is None
        or provider.provider != "openrouter"
        or is_onyx_managed_api_key(provider.api_key)
        or not provider.api_key
    ):
        return None
    if provider.api_base:
        base = urlsplit(provider.api_base)
        if (
            base.scheme != "https"
            or base.hostname != "openrouter.ai"
            or base.username
            or base.password
            or base.port not in (None, 443)
            or base.query
            or base.fragment
            or base.path.rstrip("/") not in {"", "/api", "/api/v1"}
        ):
            return None
    check_llm_cost_limit_for_provider(
        db_session=session,
        tenant_id=get_current_tenant_id(),
        llm_provider_api_key=provider.api_key,
    )
    config = LLMConfig(
        model_provider=provider.provider,
        model_name=model_name,
        temperature=1,
        api_key=provider.api_key,
        api_base=provider.api_base,
        max_input_tokens=min(
            models[0].max_input_tokens or budget.policy.max_context_tokens,
            budget.policy.max_context_tokens,
        ),
    )
    reviewer_user_id = str(user.id)

    def make_gateway() -> BudgetedGateway:
        # Constructed on the calling thread; DB pricing reads never share a Session across workers.
        llm = get_llm(
            provider=config.model_provider,
            model=config.model_name,
            max_input_tokens=config.max_input_tokens,
            deployment_name=None,
            api_key=config.api_key,
            api_base="https://openrouter.ai/api/v1",
            temperature=1,
        )
        with get_session_with_current_tenant() as pricing_session:
            return BudgetedGateway(
                selected_llm=llm,
                research_llm=llm,
                budget=budget,
                ledger=ledger,
                db_session=pricing_session,
                user_identity=LLMUserIdentity(user_id=reviewer_user_id),
                check_active=check_active,
                token_counter=token_counter,
                run_id=run_id,
                scope=scope,
                reserve_finalization=False,
            )

    return GatewayAnswerReviewer(
        config=config,
        gateway_factory=make_gateway,
        budget=budget,
        ledger=ledger,
        check_active=check_active,
    )
