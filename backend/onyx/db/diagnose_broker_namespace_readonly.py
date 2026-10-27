"""Report effective broker namespaces without credentials or connections."""

import hashlib
import json
import os

from onyx.configs import app_configs

scope = os.environ['DIAGNOSTIC_SCOPE']
assert scope in {'customs-regulations-dev', 'customs-regulations-test'}
assert app_configs.POSTGRES_DB == scope
endpoint = {
    'host': app_configs.REDIS_HOST,
    'port': app_configs.REDIS_PORT,
    'ssl': app_configs.REDIS_SSL,
    'sentinels': app_configs.REDIS_SENTINEL_HOSTS,
    'sentinel_master': app_configs.REDIS_SENTINEL_MASTER_NAME,
}
print('BROKER_NAMESPACE', json.dumps({
    'scope': scope,
    'endpoint_sha256': hashlib.sha256(json.dumps(endpoint, sort_keys=True).encode()).hexdigest(),
    'app_database': app_configs.REDIS_DB_NUMBER,
    'broker_database': app_configs.REDIS_DB_NUMBER_CELERY,
    'result_database': app_configs.REDIS_DB_NUMBER_CELERY_RESULT_BACKEND,
    'sentinel_enabled': bool(app_configs.REDIS_SENTINEL_HOSTS),
    'read_only': True,
    'no_connection_or_inspection': True,
}), flush=True)
