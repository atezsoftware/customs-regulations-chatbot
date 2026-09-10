import hashlib
import os

from onyx.background.celery.queue_names import database_scoped_queue_name
from onyx.configs.app_configs import POSTGRES_DB, POSTGRES_HOST, POSTGRES_PORT
from onyx.configs.app_configs import (
    REGULATORY_ANNEX_ENVIRONMENT as REGULATORY_ANNEX_ENVIRONMENT,
)
from onyx.configs.app_configs import (
    REGULATORY_ANNEX_UPDATES_ENABLED as REGULATORY_ANNEX_UPDATES_ENABLED,
)

REGULATORY_ANNEX_WORKER_ENABLED = (
    os.environ.get("REGULATORY_ANNEX_WORKER_ENABLED", "false").lower() == "true"
)

ANNEX_DATABASE_IDENTITY = f"{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"


def source_queue_name(
    *,
    environment: str = REGULATORY_ANNEX_ENVIRONMENT,
    database_identity: str = ANNEX_DATABASE_IDENTITY,
) -> str:
    scope = hashlib.sha256(environment.encode()).hexdigest()[:16]
    return database_scoped_queue_name(
        f"regulatory_annex_sources_{scope}", database_identity=database_identity
    )


def publication_queue_name(
    *,
    environment: str = REGULATORY_ANNEX_ENVIRONMENT,
    database_identity: str = ANNEX_DATABASE_IDENTITY,
) -> str:
    scope = hashlib.sha256(environment.encode()).hexdigest()[:16]
    return database_scoped_queue_name(
        f"regulatory_annex_publication_{scope}", database_identity=database_identity
    )


def analysis_queue_name(
    *,
    environment: str = REGULATORY_ANNEX_ENVIRONMENT,
    database_identity: str = ANNEX_DATABASE_IDENTITY,
) -> str:
    scope = hashlib.sha256(environment.encode()).hexdigest()[:16]
    return database_scoped_queue_name(
        f"regulatory_annex_analysis_{scope}", database_identity=database_identity
    )


def analysis_delivery_queue_name() -> str:
    if REGULATORY_ANNEX_WORKER_ENABLED:
        return analysis_queue_name()
    from onyx.background.celery.queue_names import REGULATORY_AMENDMENT_QUEUE

    return REGULATORY_AMENDMENT_QUEUE
