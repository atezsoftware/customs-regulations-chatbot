# Regulatory Production Runbook

This runbook is the deployment contract for the parser-free regulatory application. Production runs
only the digest-pinned `runtime-lite` backend. Source parsing and indexing are a separate, one-shot
operation run from an authorized workstation or controlled import runner.

## 1. Deployment boundary

Production receives:

- `docker-compose.prod.yml`, `docker-compose.regulatory-edge.yml`, and
  `docker-compose.regulatory-prod-lite.yml`;
- exactly one topology overlay: `docker-compose.regulatory-compose-infra.yml` or
  `docker-compose.regulatory-external-infra.yml`;
- one cloud-mode release set whose production handoff contains immutable backend-lite and custom web
  digests; its paired importer digest remains in the separate importer environment;
- production environment configuration and secrets from the approved secret store.

Production does **not** receive or run:

- `docker-compose.regulatory-importer.yml` or `.env.regulatory-importer`;
- the full-runtime importer image;
- source documents or importer manifests;
- primary, document-fetching, document-processing, or generic indexing workers.
- the `inference_model_server` or `indexing_model_server` services, or their dependencies from the
  application services.

The lite backend keeps chat, retrieval, benchmark execution, indexed-file maintenance, and required
lightweight queues. `DOCUMENT_IMPORT_ENABLED=false` is enforced by the overlay. Existing regulatory
chunks remain in PostgreSQL and Elasticsearch and are searchable; upload/indexing API mutations fail
closed.

The default and recommended production model mode is `cloud`. The fixed no-local-model overlay sets
`DISABLE_MODEL_SERVER=true` on both `api_server` and `background`; neither local model service is
started, and no CUDA model image is part of the production release. For a new empty deployment,
cloud embedding is configured in Search Settings from the production Admin UI after startup but
before imports or user traffic. LLM calls may use OpenRouter configured separately under Admin
Language Models. `local` mode, including an `inference_model_server` image, is an explicit opt-in
exception. Compose-managed Redis is changed from the base's ephemeral configuration to AOF
`everysec` on the `regulatory_cache_data` named volume.

## 2. Build and publish outside production

Run this only from a trusted, clean CI checkout after registry authentication. The script proves that
`SOURCE_REVISION` equals checkout `HEAD` and refuses tracked, staged, or untracked changes before any
Docker build. Use the full commit SHA; `latest`, `local`, and branch-style mutable tags are rejected.
The approved pinned base manifests target `linux/amd64`; CI and the production nodes must record that
platform. A multi-architecture release requires separately reviewed index digests and is blocked by
this publisher until then.

```bash
deployment/docker_compose/publish-regulatory-images.sh \
  registry.example.com/team \
  "$GIT_COMMIT_SHA" \
  cloud
```

The default cloud invocation builds and pushes three independent artifacts from the same full Git
revision:

- `regulatory-backend-lite`: the only backend image allowed on the production host;
- `regulatory-web`: the custom frontend, compiled with upstream connections disabled;
- `regulatory-importer`: the full parser runtime allowed only on the import workstation/runner.

Record all three `repository@sha256:...` references printed at the end. Configure only backend and
web digests in production and the importer digest in the separate importer environment. Cloud mode
does not build, publish, or require a model-server image/digest. Do not mix revisions. Registry
retention must keep the complete current and previous known-good release sets. Run the organization's
vulnerability/signature policy before promotion.

For an explicitly approved local-model exception, invoke the same publisher with `local` as the third
argument. That opt-in invocation adds `regulatory-model-server` as a fourth artifact; its digest is
then required by the local-mode preflight, deploy, bundle, and rollback commands.

The backend-lite image is labeled `io.regulatory.role=runtime-lite` and
`io.regulatory.document-import=false`; every published artifact receives the full source revision OCI
label. Promotion tooling must verify the expected role, import capability, and common revision after
pulling.

CI must create the production handoff from the explicit allowlist, not from a repository checkout:

```bash
deployment/docker_compose/build-regulatory-prod-bundle.sh \
  "regulatory-production-${GIT_COMMIT_SHA}.tar.gz" \
  "$GIT_COMMIT_SHA" \
  "$APPROVED_BACKEND_DIGEST" \
  "$APPROVED_WEB_DIGEST"
sha256sum -c "regulatory-production-${GIT_COMMIT_SHA}.tar.gz.sha256"
tar -tzf "regulatory-production-${GIT_COMMIT_SHA}.tar.gz"
```

The builder refuses a dirty checkout or a source revision different from `HEAD`. The bundle includes
a non-secret manifest containing that revision, `model_mode=cloud`, the two production image digests,
and SHA-256 for every deployment file; the adjacent checksum covers the archive and manifest. It
contains only the
production base/edge/topology/lite Compose files, authoritative preflight/deploy scripts,
secret-free production/nginx/role templates, runbook, and nginx templates. It physically excludes
the importer compose/env, importer/publish scripts, Dockerfiles, application source, and source
documents. Production hosts receive this bundle plus approved secrets—not a full repository clone.

CI must sign/attest the checksum and both production image digests with the organization's approved
mechanism. DevOps verifies those signatures/attestations and the image SBOM/vulnerability policy
*before extracting the archive*. Record the bundle SHA-256, source revision, manifest, image
signatures/attestations, and SBOM identifiers in the change record. A checksum alone proves transfer
integrity, not publisher identity.

For a local-model exception, append `"$APPROVED_MODEL_DIGEST"` to the bundle command. The resulting
manifest records `model_mode=local` and the third production image digest.

## 3. Production prerequisites

### Cloud-model configuration gate

Cloud mode removes the model-server containers; it does not automatically change provider settings
stored in PostgreSQL. Complete these application-level steps before production traffic is opened:

1. For a new database with no documents, non-default connectors, or completed user files, deploy the
   parser-free application first. Sign in to production **Admin > Search Settings**, choose
   **OpenRouter**, enter its API key, fetch the embedding models, and select one from the list. The
   application owns the endpoint and derives the vector dimension from its test call; neither value
   is entered by the Admin. Activating the selection is permitted even though
   `DOCUMENT_IMPORT_ENABLED=false`; it promotes the empty cloud index without an indexing worker.
2. Perform that bootstrap before the first import. Run an embedding/retrieval smoke test before user
   traffic. Until activation, Admin remains available but search fails closed with a clear
   local-provider/model-server-disabled error; no local model is started.
3. If production already contains data, do not use the empty bootstrap. Configure the provider and
   complete the required reindex from an authorized import/indexing-capable environment, verify the
   promoted index, and only then cut over to runtime-lite cloud mode.
4. Configure and test OpenRouter for LLM calls separately under **Admin > Language Models** before
   opening user traffic. This LLM step may be performed through the production Admin UI during the
   controlled cutover; it does not perform or replace embedding reindexing.

This branch has no standalone OpenRouter rerank provider/endpoint setting. Do not provision an
`/api/v1/rerank` URL or represent rerank as a DevOps configuration requirement. The existing search
pipeline remains authoritative unless a separately reviewed application feature adds such support.

Before the maintenance window:

1. Choose and record exactly one infrastructure topology:
   - **Compose-managed:** add `docker-compose.regulatory-compose-infra.yml`; PostgreSQL, Elasticsearch,
     Redis, and MinIO run in the project.
   - **External:** add `docker-compose.regulatory-external-infra.yml`; those four local services and
     their `depends_on` edges are removed. Merely changing host variables is forbidden because the
     base file would still start unused local services.
2. Confirm PostgreSQL, Elasticsearch, Redis, object storage, the configured cloud embedding endpoint,
   and the OpenRouter LLM endpoint are reachable over approved egress from the production containers.
   For external Elasticsearch this includes an authenticated cluster-health request and a read against
   the approved live regulatory index/alias; TCP reachability alone is not a pass.
3. Keep PostgreSQL, Elasticsearch, Redis, and object storage off the public internet. Restrict security
   groups/firewalls to the application network and administrative paths.
4. Take coordinated PostgreSQL and Elasticsearch backups/snapshots. Preserve object-store versions or a
   matching backup when source/citation objects are stored there.
5. Pause new benchmark runs and confirm that no external importer is active.
6. Store secrets in the deployment secret manager or a mode-`0600` `.env`; never bake them into an
   image or commit them. Common `.env` contains the restricted application runtime database role—not
   a database owner or migration role—and the API's encryption, tenant/schema, Elasticsearch,
   object-store, model/provider, and TLS settings.
   Merge the non-secret image/topology keys from `env.regulatory-prod.template` into the one full
   mode-`0600` `.env` that also contains all base service secrets/provider configuration. The base
   Compose services hardcode `env_file: .env`; using a separate interpolation-only env file would
   silently omit container configuration and is forbidden.
7. Set `ONYX_BACKEND_LITE_IMAGE` and `ONYX_WEB_SERVER_IMAGE` to the two matching production release
   digests. Leave `ONYX_MODEL_SERVER_IMAGE` unset in cloud mode. A model digest is required only for
   an explicitly approved `local` release. Tag-only references are forbidden.
8. Keep the importer compose file, importer credentials, source mounts, and importer image out of the
   production deployment bundle and registry pull policy for the production host.
9. Create `.env.nginx` from `env.nginx.template`, protect it with mode `0600`, and fill the approved
   domain/TLS proxy values. A missing or unreadable `.env.nginx` is a preflight blocker; do not place
   its contents in the runbook or source control.
10. Create `.env.web` from `env.web.template`, mode `0600`. It is intentionally non-secret and may
    contain only the documented Next.js proxy settings. The production overlay removes common
    `.env` from `web_server`; database, Elasticsearch, object-store, LLM/OpenRouter, encryption/signing,
    and OAuth client secrets in the rendered web environment are blockers.
11. Create `.env.migration` from `env.migration.template` through the secret store, mode `0600`, with
    only the migration role's `POSTGRES_USER`/`POSTGRES_PASSWORD`. Keep it separate from common `.env`.
12. In Compose-managed mode, create `.env.db-admin` from `env.db-admin.template`, mode `0600`, with
    exactly the PostgreSQL bootstrap/owner `POSTGRES_USER`, `POSTGRES_PASSWORD`, and `POSTGRES_DB`.
    It is loaded only by the database container. Do not create or pass this file in external mode.
13. Record the existing immutable Compose project name (normally `onyx`) and pass it explicitly to
    every canonical command. Changing it creates a new set of apparently empty named volumes. Remove
    `COMPOSE_PROFILES` from `.env` and the process environment; only canonical script flags may enable
    a profile.
14. Inventory all running containers on the host, including other Compose project names. Any legacy
    full-backend, importer, indexer, primary, docfetching, docprocessing, or user-file-processing
    container is a blocker until its work is drained and its removal is approved. In cloud mode, any
    inference-model or indexing-model container is also a blocker. Rendered-Compose preflight cannot
    discover an orphan owned by another project.

```bash
docker ps --no-trunc --format '{{.ID}} {{.Image}} {{.Names}} {{.Labels}}'
```

Archive this inventory in the change record and repeat it after rollout. Do not treat a renamed
legacy container as safe; verify its image role labels and running processes.

The production nginx template is not an initial certificate issuer. Before preflight, the mounted
certificate volume must already contain non-empty
`live/$DOMAIN/{fullchain.pem,privkey.pem}`, `options-ssl-nginx.conf`, and `ssl-dhparams.pem`. Verify the
certificate hostname, chain, key match, and expiry (including `openssl x509 -checkend` for the
organization's renewal margin). Provision these from the approved PKI/secret workflow or use a
separately reviewed corporate-ingress overlay. Do not run the repository's legacy
`init-letsencrypt.sh` on production: it downloads mutable remote configuration files. Confirm the
renewal configuration and perform a staged renewal test before relying on the pinned certbot service.

Compose-managed infrastructure additionally requires organization-approved digest locks for
PostgreSQL, Elasticsearch, Redis, MinIO, nginx, and certbot. The repository supplies no arbitrary digest:
DevOps must mirror/approve each artifact and set the six `REGULATORY_*_IMAGE` variables. nginx and
certbot remain Compose services in external-data-infrastructure mode, so their approved digests are
required in both modes. Any rendered tag-only image, especially implicit `certbot:latest`, is a
production blocker.

Compose-managed PostgreSQL, Elasticsearch, Redis, and MinIO are single-host services, not an HA design.
For a serious production SLA, prefer externally managed HA services. Otherwise the service owner
must formally accept and restore-test the documented RPO/RTO, host-loss recovery, coordinated backup,
and capacity/alerting plan before signoff.

On every Compose-managed Elasticsearch host, `vm.max_map_count` must be at least `262144`; preserve the
configured unlimited memlock and `nofile=65536`. Size host/container RAM for the configured 2 GiB JVM
heap plus native memory and filesystem cache, and monitor JVM pressure/GC, cluster status,
unassigned shards, disk watermarks, and data-volume growth. The authenticated healthcheck waits for
at least yellow and is bounded; it never interpolates the admin password into the Compose model.
PostgreSQL is configured with `max_connections=250`. Document the sum of API/background/migration,
importer, monitoring, and administrative pool/concurrency ceilings plus failover headroom; a
calculated maximum at or above 250 blocks rollout. Archive these non-secret checks:

```bash
test "$(sysctl -n vm.max_map_count)" -ge 262144
docker inspect "$(docker compose --project-name "$APPROVED_COMPOSE_PROJECT" \
  --env-file .env -f docker-compose.prod.yml \
  -f docker-compose.regulatory-edge.yml \
  -f docker-compose.regulatory-compose-infra.yml \
  -f docker-compose.regulatory-prod-lite.yml ps -q elasticsearch)" \
  --format '{{json .HostConfig.Ulimits}}'
```

After the managed data services start, archive the authenticated cluster result without placing the
password in the host command or logs:

```bash
docker compose --project-name "$APPROVED_COMPOSE_PROJECT" --env-file .env \
  -f docker-compose.prod.yml \
  -f docker-compose.regulatory-edge.yml \
  -f docker-compose.regulatory-compose-infra.yml \
  -f docker-compose.regulatory-prod-lite.yml \
  exec -T elasticsearch sh -ec \
  'curl --fail --silent --show-error --insecure --user "admin:${ELASTIC_PASSWORD}" \
    "https://127.0.0.1:9200/_cluster/health?wait_for_status=yellow&timeout=10s"'
```

For external basic-auth Elasticsearch, use a mode-`0600` netrc/secret file and the approved CA; do not put
the password in an argument or shell history. The first call must report green/yellow and the second
must prove the approved live index/alias is readable and non-empty:

```bash
test "$(stat -c '%a' "$ELASTICSEARCH_NETRC")" = 600
curl --fail --silent --show-error --netrc-file "$ELASTICSEARCH_NETRC" \
  --cacert "$ELASTICSEARCH_CA_FILE" \
  "$ELASTICSEARCH_URL/_cluster/health?wait_for_status=yellow&timeout=10s" \
  | tee elasticsearch-cluster-health.json
curl --fail --silent --show-error --netrc-file "$ELASTICSEARCH_NETRC" \
  --cacert "$ELASTICSEARCH_CA_FILE" \
  "$ELASTICSEARCH_URL/$APPROVED_REGULATORY_INDEX/_count" \
  | tee elasticsearch-regulatory-index-count.json | jq -e '.count > 0'
```

For IAM or mTLS, use the organization's credential-aware client instead of converting credentials to
basic auth, but archive the same cluster-health and live-index evidence. The authenticated normal-chat
retrieval smoke remains mandatory in every topology.

The current generated `docker-compose.prod.yml` interpolates MinIO/static S3 credentials before
overlays are merged. Therefore external AWS-S3 IAM and `FILE_STORE_BACKEND=postgres` configurations
that legitimately omit those static values are **not** supported by inventing dummy credentials.
They require a reviewed generated-base/template change that removes the irrelevant required
interpolation. Deployment remains blocked until that topology renders without fake secrets.

The production backend needs only the normal runtime connections: PostgreSQL, Elasticsearch, Redis,
object storage, and the approved cloud embedding/LLM provider endpoints. In cloud mode both API and
background must render `DISABLE_MODEL_SERVER=true`. `DISABLE_ONYX_UPSTREAM_CONNECTIONS` and telemetry
disablement are set by the overlay; an egress allowlist should enforce the same boundary at the
network layer.

Compose-managed Redis uses AOF `everysec` on `regulatory_cache_data`; include that volume in host
backup/capacity monitoring. External Redis must provide an approved durability/HA policy. An
ephemeral external broker is a release blocker because benchmark and maintenance messages can be lost
and the lite deployment intentionally has no Beat process to recreate generic ingestion schedules.

## 4. Preflight the rendered deployment

Run only the authoritative preflight wrapper from `deployment/docker_compose`. The approved digest
variables below are non-secret values supplied by release automation. For the recommended cloud
model mode with Compose-managed data services:

```bash
./regulatory-prod-lite-preflight.sh \
  --env-file .env \
  --base-compose docker-compose.prod.yml \
  --project-name "$APPROVED_COMPOSE_PROJECT" \
  --migration-env-file .env.migration \
  --db-admin-env-file .env.db-admin \
  --infra-mode compose-managed \
  --model-mode cloud \
  --expected-image "$APPROVED_BACKEND_DIGEST" \
  --expected-web-image "$APPROVED_WEB_DIGEST"
```

For the recommended cloud model mode with approved external data services, with `COMPOSE_PROFILES`
excluding `local-infra` and `s3-filestore`:

```bash
./regulatory-prod-lite-preflight.sh \
  --env-file .env \
  --base-compose docker-compose.prod.yml \
  --project-name "$APPROVED_COMPOSE_PROJECT" \
  --migration-env-file .env.migration \
  --infra-mode external \
  --model-mode cloud \
  --expected-image "$APPROVED_BACKEND_DIGEST" \
  --expected-web-image "$APPROVED_WEB_DIGEST"
```

For a new empty installation, run this infrastructure preflight before the first deployment, then
complete the Admin Search Settings activation after startup and before traffic or import. For an
existing populated installation, the cloud-provider reindex gate must already be complete before the
cutover preflight. The script selects the fixed no-local-model overlay, proves that both application
services have `DISABLE_MODEL_SERVER=true`, and proves that local inference is inactive. Supplying
`--expected-model-image` in cloud mode is an error.

For an approved local-model exception, change to `--model-mode local` and add
`--expected-model-image "$APPROVED_MODEL_DIGEST"`. The backend, web, and model references must be
from the same local-mode release invocation.

This guard renders Compose, rejects tag-only or malformed application/edge/infrastructure image
references, proves that `api_server` and `background` use the same approved digest, refuses
`MULTI_TENANT=true`, and fails if importer/indexing services appear in the default model. It also
proves that local data services are all present in Compose mode and all absent in external mode.

The overlay resets inherited backend and web build definitions; local mode also resets the model
build. Deployment automation must still use `--no-build` as a second, explicit guard. Missing
backend/web digests, or a missing model digest in local mode, make Compose validation fail instead of
falling back to upstream or `latest` images.

## 5. Migration ownership and rollout

This Compose runbook currently supports only `MULTI_TENANT=false`. The production guard refuses
multi-tenant mode. Plain `alembic upgrade head` is not sufficient there: multi-tenant deployments
need the catalog migration (`alembic -n schema_private upgrade head`) plus controlled migration of
every tenant/shard (`python alembic/run_multitenant_migrations.py`). The regulatory `pg_trgm`
migration has not been approved across tenant search paths, so this release must not be represented as
multi-tenant capable without a dedicated migration test and change plan.

For the supported single-tenant topology, only the profile-gated `regulatory_migration` job owns
schema changes. `api_server` is Uvicorn-only, and the importer never runs or repairs migrations. Store
the migration role's `POSTGRES_USER`/`POSTGRES_PASSWORD` only in the separate mode-`0600`
`.env.migration`; the canonical script injects that file only into the migration job. API/background
keep the DML-only runtime role from common `.env`, and the importer uses a separate least-privilege
write role. Never place migration credentials in common `.env` or an importer environment.

In Compose-managed mode, `.env.db-admin` is used only to bootstrap/start PostgreSQL. An existing
`db_volume` ignores changed `POSTGRES_*` initialization variables: before rollout the DBA must audit
the actual database/schema/table owners and explicitly provision/rotate the runtime and migration
roles and grants. The runtime username must differ from the database owner and migration role; the
migration role may temporarily equal the controlled bootstrap owner only when this is documented and
approved. Verify runtime DML access and denial of schema/extension DDL, then verify migration DDL with
the dedicated role. Do not assume editing an env file changed any existing database principal.

The API runs as UID/GID `1001:1001`, and every Celery child runs as `onyx`; only the supervisor wrapper
stays root long enough to repair the flat log files and drop privileges. For pre-existing named
`api_server_logs`, `background_logs`, or `file-system` volumes, inspect ownership and perform one
reviewed, bounded ownership repair to `1001:1001` before rollout. Do not recursively chown an
unresolved host path or an entire Docker volume root. Archive `id`, writable-log/file-store probes,
and the supervisor process owners in the change record.

Before rollout, the DBA must pre-provision `pg_trgm` in the configured shared migration schema and
verify that `gin_trgm_ops` resolves on the migration search path. The migration role owns application
table/index/function DDL but need not receive extension-superuser authority. Revision
`2010a61d7d88` still executes `CREATE EXTENSION IF NOT EXISTS pg_trgm` and creates GIN trigram indexes;
missing extension objects, wrong extension schema/search path, or insufficient DDL ownership blocks
deployment.

Revision `9ce718a30332` deletes WebSearch, OpenURL, and Python tool rows. Foreign-key cascades remove
their persona associations. An image rollback does not restore those associations, and the migration
downgrade only recreates tool rows. The coordinated PostgreSQL backup and change record are mandatory.

Before stopping the old background container, disable new uploads/import/indexing triggers and record
Celery `active`, `reserved`, and `scheduled` output for every old worker plus broker depth for *every*
declared queue—not only ingestion queues. Primary, docfetching, docprocessing, and
user-file-processing work must be zero. The service owner must explicitly approve any queued
write/delete work that the lite workers will consume, including `connector_deletion`,
`user_file_delete`, metadata sync, and permission upserts. `elasticsearch_migration` has no lite consumer;
any stale depth requires a recorded quarantine/delete decision before cutover and before any future
full-runtime rollback. Never use blanket `celery purge`. Preserve all inspection/depth evidence in the
change record and do not cut over merely because the API is idle.

After queue drain and backup approval, deploy only through the authoritative wrapper. It selects the
fixed overlay chain, validates it, pulls the reviewed images, verifies image role/revision labels,
and enters a bounded maintenance window by stopping nginx/web/API/background before schema changes.
Cloud preflight requires legacy inference/indexing model containers to be drained and stopped first;
the wrapper repeats those stop/state checks as defense in depth. This prevents old API writes during
DDL/data migrations. It then runs the singleton migration, gates API liveness, and starts with
bounded `--no-build --wait`. A migration failure leaves application/edge services stopped while data
services remain intact for investigation:

```bash
./regulatory-prod-lite-deploy.sh deploy \
  --env-file .env \
  --base-compose docker-compose.prod.yml \
  --project-name "$APPROVED_COMPOSE_PROJECT" \
  --migration-env-file .env.migration \
  --db-admin-env-file .env.db-admin \
  --infra-mode compose-managed \
  --model-mode cloud \
  --expected-image "$APPROVED_BACKEND_DIGEST" \
  --expected-web-image "$APPROVED_WEB_DIGEST" \
  --backup-reference "$VERIFIED_BACKUP_REFERENCE" \
  --acknowledge-migration-impact \
  --wait-timeout 900
```

Use the same command with `--infra-mode external` only after removing
`--db-admin-env-file .env.db-admin`; external mode forbids that file/flag. For an approved local-model
exception, use `--model-mode local` and add
`--expected-model-image "$APPROVED_MODEL_DIGEST"`, matching the preflighted topology. An application
rollout must retain the currently approved infrastructure digests.
PostgreSQL/Elasticsearch/Redis/MinIO/nginx/certbot upgrades are separate, separately backed-up changes
and must not be folded into an application cutover.

The regulatory overlay replaces the base API command with Uvicorn-only startup. The canonical deploy
wrapper's one-shot Alembic command is the sole migration owner; ordinary startup and rollback never
run migrations. Do not scale the API during migration. Platforms with multiple replicas must retain
the same singleton migration ownership before rolling replicas.

If the API gate fails, the wrapper leaves `background` stopped. Preserve its output, then use the
same fixed overlay order only for read-only diagnostics; do not retry with ad hoc Compose files.

## 6. Health and smoke checks

First verify the same reviewed topology through the authoritative status command:

```bash
./regulatory-prod-lite-deploy.sh status \
  --env-file .env \
  --base-compose docker-compose.prod.yml \
  --project-name "$APPROVED_COMPOSE_PROJECT" \
  --migration-env-file .env.migration \
  --db-admin-env-file .env.db-admin \
  --infra-mode compose-managed \
  --model-mode cloud \
  --expected-image "$APPROVED_BACKEND_DIGEST" \
  --expected-web-image "$APPROVED_WEB_DIGEST"
```

The background container healthcheck requires the exact expected supervisor program set—four workers
(`regulatory_benchmark`, `user_file_maintenance`, `light`, `monitoring`) plus the log redirector—and
requires all five to be `RUNNING`. `active_queues` must contain only the queues declared by those lite
workers; primary, docfetching, docprocessing, and user-file-processing queues/workers are forbidden.
Cloud preflight treats every running inference/indexing model container as a blocker. Drain and stop
the legacy model service after ownership review before invoking the deploy wrapper. The wrapper's
own stop and post-rollout checks are defense in depth; moving a service behind a profile alone does
not remove an older container. SRE should additionally record Celery `active_queues` from its
approved observability/exec path and verify only the lite queue set; this is a diagnostic check, not
an alternative deploy path.

Use the public frontend/nginx route, not a separately exposed backend port:

```bash
curl --fail --silent --show-error "https://regulatory.example.com/api/health"
curl --fail --silent --show-error "https://regulatory.example.com/api/settings" \
  | jq -e '.document_import_enabled == false'
```

`/api/health` is a shallow process liveness endpoint. It does not prove PostgreSQL, Elasticsearch, Redis,
object store, model-provider, queue, or retrieval readiness. `up --wait` additionally covers the
container healthchecks, but the authenticated functional checks below remain mandatory.

Complete one authenticated application smoke test:

1. Open an existing regulatory directory and ask a known, non-destructive question through the normal
   chat flow.
2. Confirm retrieval returns indexed chunk citations and that expanding a citation opens the chunk,
   not the entire source document.
3. Repeat a known temporal question with current and historical `as_of_date` values and confirm the
   active/superseded chunk changes as expected.
4. Confirm previously indexed files remain visible/searchable to the intended users.
5. Confirm source upload/import controls are absent and an indexing mutation is rejected.
6. Start one small benchmark run only after chat/search succeeds, and confirm the dedicated worker
   completes it.

Review API/background logs for repeated database, Elasticsearch, object-store, cloud embedding,
OpenRouter LLM, or Celery errors before ending the maintenance window.

## 7. Separate importer/indexer operation

Run imports only from an authorized workstation or controlled runner with private connectivity to the
same production PostgreSQL database, Elasticsearch index, and object store. The host needs Docker and the
small compose/env bundle; it pulls the importer digest and does not build an image.

Security requirements:

- use a VPN/private endpoint or tightly firewalled authenticated tunnels; never expose production
  PostgreSQL, Elasticsearch, or object storage publicly;
- use TLS verification with the trusted CA mounted read-only; `verify-full` and IAM must retain the
  real database/service hostname;
- use short-lived, least-privilege credentials where possible. The database role needs application
  data writes but no migration/DDL ownership; Elasticsearch access is limited to the active index and
  object-store access to the configured application prefix;
- copy the production tenant/schema, encryption, active SearchSettings, embedding, and contextual
  retrieval configuration exactly. Temporary AWS credentials include the session token;
- when active SearchSettings use local embeddings, start the matching model-server digest on the
  workstation/private import network and point the importer model/indexing-model hosts to it. The
  alternative is a verified cloud embedding configuration. Never start production's
  `indexing_model_server` profile for an import;
- mount `/imports` read-only and make only `/output` writable; protect `.env.regulatory-importer` with
  mode `0600` and remove/revoke credentials after the run;
- never add Redis, Celery, a public listener, or Alembic execution to the importer job.

Prepare and validate on the import host:

```bash
cp env.regulatory-importer.template .env.regulatory-importer
chmod 600 .env.regulatory-importer
# Fill required values, including the importer digest paired with production.

./regulatory-import-run.sh --help
```

The workstation-only wrapper validates the digest reference and Compose model, pulls the image, and
requires `io.regulatory.role=importer`, `io.regulatory.document-import=true`, and an OCI source
revision exactly equal to the promoted production revision. It is intentionally excluded from the
production bundle.

The importer fails before document writes when its Alembic heads do not exactly match production, the
active Elasticsearch retrieval/index mapping is incompatible or unreachable, inputs are unreadable,
the target user/project is invalid, duplicate names are found, or the manifest path is not writable.

Run an import with a manifest:

```bash
./regulatory-import-run.sh \
  --env-file .env.regulatory-importer \
  --expected-image "$APPROVED_IMPORTER_DIGEST" \
  --expected-revision "$PROMOTED_SOURCE_REVISION" \
  -- \
  /imports/mevzuat.docx \
  --user-email admin@example.com \
  --project-name "Mevzuat" \
  --tenant-id "tenant_schema_from_production" \
  --manifest /output/import-manifest.json
```

Replace `tenant_schema_from_production` with the exact verified production schema. Do not assume
`public`, especially when tenant routing/shards are configured.

Exit code `0` means every file completed, `1` means at least one file failed after preflight, and `2`
means preflight failed. A successful result is also post-verified against the PostgreSQL file/chunk
records and Elasticsearch visibility. Archive the console log and manifest as the import audit record.
Inspect partial writes before retrying; do not use `--allow-duplicate` as an automatic retry switch.

After the import, close tunnels/revoke temporary credentials, then repeat the authenticated production
search and chunk-citation smoke test. No production container restart is required for newly indexed
content.

## 8. Rollback and recovery

Application rollback:

1. Stop `background`.
2. Confirm the previous application image is compatible with the current database migration state.
3. Verify the rollback backend has `io.regulatory.role=runtime-lite` and
   `io.regulatory.document-import=false`. A historical full backend image is forbidden because it
   would restore ingestion/indexing workers.
4. In cloud mode, restore the previous backend-lite and web digests from the same revision. For an
   approved local-mode topology, also restore its matching model digest.
5. Run the authoritative rollback command and all smoke checks. Confirm
   both model-server services remain absent afterward in cloud mode.

```bash
./regulatory-prod-lite-deploy.sh rollback \
  --env-file .env \
  --base-compose docker-compose.prod.yml \
  --project-name "$APPROVED_COMPOSE_PROJECT" \
  --migration-env-file .env.migration \
  --db-admin-env-file .env.db-admin \
  --infra-mode compose-managed \
  --model-mode cloud \
  --expected-image "$PREVIOUS_BACKEND_LITE_DIGEST" \
  --expected-web-image "$PREVIOUS_WEB_DIGEST" \
  --backup-reference "$VERIFIED_BACKUP_REFERENCE" \
  --schema-compatible \
  --wait-timeout 900
```

Use the exact topology modes of the failed release; external mode omits `--db-admin-env-file`. Do not
switch infrastructure/model topology as part of rollback. A local-mode rollback uses
`--model-mode local` and adds `--expected-model-image "$PREVIOUS_MODEL_DIGEST"`.

Changing an image digest does not undo database migrations or imported data. If a migration is not
backward compatible, coordinate a database restore or forward fix; do not improvise a downgrade. If
an import produced incorrect content, use its manifest and the supported admin deletion/re-import
flow. Restore PostgreSQL, Elasticsearch, and object storage only as one coordinated recovery point when
a full data rollback is required.

In particular, rolling back the image does not reconstruct persona/tool associations deleted by
revision `9ce718a30332`. Recovering those relationships requires the coordinated pre-change database
backup or an explicitly reviewed data repair—not merely Alembic downgrade.

Keep the failed and previous digests, deployment logs, migration output, importer manifest, and backup
identifiers in the incident record.
