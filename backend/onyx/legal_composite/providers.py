"""Resolve the user's preferred Decisions provider under existing access rules."""

from collections.abc import Callable

from pydantic import JsonValue
from sqlalchemy.orm import Session

from onyx.asv3.evidence import EvidenceLedger
from onyx.db.llm import (
    fetch_accessible_llm_provider_by_id,
    fetch_all_accessible_llm_providers,
)
from onyx.db.models import User
from onyx.legal_composite.budget import WorkflowBudget
from onyx.legal_composite.decisions import DecisionsClassifier
from onyx.legal_composite.gateway import BudgetedGateway
from onyx.legal_composite.selection import GatewaySourceClassifier, SourceSelector
from onyx.llm.interfaces import LLMConfig
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
                    )
                    return SourceSelector(classifier, irrelevance_threshold=0.98)
                except ValueError:
                    # Unsupported endpoint configuration cannot forward credentials.
                    pass
    return SourceSelector(GatewaySourceClassifier(gateway), irrelevance_threshold=0.98)
