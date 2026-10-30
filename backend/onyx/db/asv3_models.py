"""Default-assistant identity for authorized ASv3 model resolution."""

from onyx.db.engine.sql_engine import get_session_with_current_tenant
from onyx.db.models import Persona
from onyx.db.persona import get_default_behavior_persona
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
