"""Default-assistant identity for authorized ASv3 model resolution."""

from onyx.db.engine.sql_engine import get_session_with_current_tenant
from onyx.db.models import Persona
from onyx.db.persona import get_default_behavior_persona


def default_asv3_persona() -> Persona | None:
    with get_session_with_current_tenant() as db_session:
        return get_default_behavior_persona(db_session)
