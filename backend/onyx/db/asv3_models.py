"""Default-assistant identity for authorized ASv3 model resolution."""

from sqlalchemy import select

from onyx.db.engine.sql_engine import get_session_with_current_tenant
from onyx.db.models import LLMProvider, ModelConfiguration, Persona
from onyx.db.persona import get_default_behavior_persona
from onyx.error_handling.error_codes import OnyxErrorCode
from onyx.error_handling.exceptions import OnyxError
from onyx.server.usage_limits import check_llm_cost_limit_for_provider
from shared_configs.contextvars import get_current_tenant_id


def default_asv3_persona() -> Persona | None:
    with get_session_with_current_tenant() as db_session:
        return get_default_behavior_persona(db_session)


def check_asv3_secondary_model_cost_limit(api_key: str | None) -> None:
    with get_session_with_current_tenant() as db_session:
        check_llm_cost_limit_for_provider(
            db_session=db_session,
            tenant_id=get_current_tenant_id(),
            llm_provider_api_key=api_key,
        )


def resolve_asv3_model_provider_id(provider_type: str, model_name: str) -> int:
    """Select one visible configured model; the model factory still enforces access."""
    with get_session_with_current_tenant() as db_session:
        identifiers = list(
            db_session.scalars(
                select(LLMProvider.id)
                .join(ModelConfiguration)
                .where(
                    LLMProvider.provider == provider_type,
                    ModelConfiguration.name == model_name,
                    ModelConfiguration.is_visible.is_(True),
                )
                .distinct()
                .limit(2)
            )
        )
    if len(identifiers) != 1:
        raise OnyxError(
            OnyxErrorCode.INVALID_INPUT, "The selected language model is unavailable."
        )
    return identifiers[0]
