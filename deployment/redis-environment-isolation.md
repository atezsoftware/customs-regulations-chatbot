# Redis environment allocation

All processes in a deployment must use the same `REDIS_DEPLOYMENT_DATABASES`
setting. The three comma-separated numbers select application state/locks,
the Celery broker, and Celery results, in that order. This setting overrides
the three legacy `REDIS_DB_NUMBER*` values so that API, workers, and beat cannot
inherit conflicting database selections from separately managed configuration.
When unset, legacy configuration and defaults remain unchanged.

## DEV allocation

| Redis database | Owner | Purpose |
| --- | --- | --- |
| 4 | customs-regulations-dev | Application state, cache, locks |
| 5 | customs-regulations-dev | Celery queues, delivery bookkeeping, control |
| 6 | customs-regulations-dev | Celery results |

The existing DEV backend deployment runner supplies `4,5,6` to both API and
background Helm releases. These database numbers are reserved for this
deployment and must not be assigned to another environment. The server must
support logical databases; Redis Cluster's DB 0-only configuration is unsuitable.

Before the first move, `backend/scripts/regulatory_redis_isolation.py` checks
for existing connections/data and atomically claims each empty database using
`onyx:deployment-owner`. Existing ownership must match the DEV PostgreSQL
endpoint identity. A partial reservation remains owned and can be retried; no
foreign data is deleted. This marker is an operational allocation record, not
an access-control boundary.

The runner stops the DEV API, drains the DEV workers and beat, and stops the
old background pod before installing the new configuration. Old shared queues
are never copied or purged. Database-backed pending work is recovered by the
new workers; Redis-only cached state starts empty. Existing DEV login sessions
stored in Redis expire from the application's perspective and users log in
again. TEST configuration and data are not migrated.

Release verification checks the selected databases and ownership in both
applications, then checks the local upload/index worker's process, registered
handlers, queues, and responding broker workers. The end-to-end release probe
must also complete normal Markdown upload/indexing and the annex/PDF update.

Kombu's Redis transport scopes control/fanout channels by broker database by
default (`fanout_prefix=True`); do not disable this setting. The deployment
setting also scopes Celery result keys/notifications with `onyx:redis-db:N:`
because Redis Pub/Sub itself is shared across logical databases. Logical databases
share server memory and availability. Infrastructure changes must preserve
this allocation and must not flush databases belonging to other deployments.
