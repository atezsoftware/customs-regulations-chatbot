import hashlib

from onyx.configs.app_configs import POSTGRES_DB, POSTGRES_HOST, POSTGRES_PORT
from onyx.configs.constants import OnyxCeleryQueues


def database_scoped_queue_name(queue: str) -> str:
    database_identity = f"{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
    suffix = hashlib.sha256(database_identity.encode("utf-8")).hexdigest()[:16]
    return f"{queue}_{suffix}"


REGULATORY_AMENDMENT_QUEUE = database_scoped_queue_name(
    OnyxCeleryQueues.REGULATORY_AMENDMENT
)
