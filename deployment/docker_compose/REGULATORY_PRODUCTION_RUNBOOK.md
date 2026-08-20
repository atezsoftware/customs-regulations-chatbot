# Regulatory Production Runbook

This runbook is the deployment contract for the parser-free regulatory application. Production runs
only the digest-pinned `runtime-lite` backend. Parser-backed and bulk source imports remain a separate,
one-shot operation run from an authorized workstation or controlled import runner. The explicitly
enabled Markdown-only path is processed durably by the production-lite regulatory indexing worker.

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

The lite backend keeps chat, retrieval, benchmark execution, indexed-file maintenance, Markdown-only
regulatory indexing, and required lightweight queues. `DOCUMENT_IMPORT_ENABLED=false` is enforced by
the overlay while `MARKDOWN_IMPORT_ENABLED=true` permits the parser-free Markdown path. Durable
indexing remains fail-closed until `REGULATORY_BATCH_INDEXING_ENABLED=true` is applied to both API and
background after the prerequisites below. Existing regulatory chunks remain searchable throughout.

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
production base Compose file, authoritative non-root deploy wrapper, the privileged-bundle
installer/entrypoint/manifest, the descriptor-owned snapshot helper, trusted edge/topology/lite
overlays,
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
    full-backend, importer, indexer, primary, docfetching, docprocessing, or generic
    user-file-processing container is a blocker until its work is drained and its removal is approved.
    The only approved production-lite consumer of `user_file_processing` is
    `celery_worker_regulatory_indexing`. In cloud mode, any
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

Run only the authoritative non-root deployment wrapper from the verified release directory. Root
must never execute that wrapper, a script/helper, or a Compose overlay from the extracted handoff,
repository checkout, current working directory, or another operator-writable path. The wrapper rejects ambient
`DOCKER_*`, `COMPOSE_*`, and image interpolation overrides, reads interpolation only from the
selected mode-`0600` environment file, and binds both preflight and rollout to the local
`unix:///var/run/docker.sock` daemon with one explicit environment. Remote Docker contexts are not
supported by this production-lite procedure. The wrapper does not trust the operator's `PATH`,
`HOME`, or Docker CLI plugin discovery: it uses the fixed system path
`/usr/sbin:/usr/bin`, `/usr/bin/docker`, and
`/usr/libexec/docker/cli-plugins/docker-compose` for both phases.

### Install or upgrade the privileged bundle

The only root-executed application code is the fixed entrypoint
`/usr/local/libexec/onyx/regulatory-prod-lite/regulatory-prod-lite-preflight`. It dispatches to one
digest-named release below `releases/` only after recursively checking exact `root:root` ownership,
rejecting symlinks and group/world-writable directories/files, checking the exact member set, and
verifying `REGULATORY_PRIVILEGED_MANIFEST.sha256` plus every listed SHA-256. It then uses
`/usr/bin/python3 -I -S` for the descriptor-owned snapshot helper. The non-root wrapper reads the
same installed digest once and uses that release's trusted overlays for rollout, so preflight and
rollout cannot select different checkout copies.

First verify the production archive's organization signature/attestation and the adjacent archive
SHA-256 through the approved release channel. Extract it as the non-root deployment operator; never
run an installer directly from that extraction or a checkout. Record the reviewed SHA-256 of
`REGULATORY_PRIVILEGED_MANIFEST.sha256` as `APPROVED_PRIVILEGED_MANIFEST_SHA256`. A root custodian then
copies the exact allowlisted files as inert data into the private staging directory named by that
digest. The following is a procedure template: substitute the verified absolute release directory
and literal reviewed digest before the root session, and do not pipe untrusted shell text into it.

```bash
# Run in a controlled root session after release signature and archive-digest verification.
/usr/bin/install -d -o root -g root -m 0755 /var/lib/onyx
/usr/bin/install -d -o root -g root -m 0700 \
  /var/lib/onyx/regulatory-prod-lite-staging
/usr/bin/install -d -o root -g root -m 0700 \
  "/var/lib/onyx/regulatory-prod-lite-staging/$APPROVED_PRIVILEGED_MANIFEST_SHA256"

# Copy, but do not execute, only the manifest's eight files from the verified extraction.
while IFS= read -r _ file; do
  case "$file" in
    regulatory-prod-lite-privileged-entrypoint|regulatory-prod-lite-preflight.sh)
      mode=0755 ;;
    regulatory_readiness_file_snapshot.py|docker-compose.regulatory-edge.yml|\
    docker-compose.regulatory-compose-infra.yml|\
    docker-compose.regulatory-external-infra.yml|\
    docker-compose.no-local-models.yml|docker-compose.regulatory-prod-lite.yml)
      mode=0644 ;;
    *) exit 1 ;;
  esac
  /usr/bin/install -o root -g root -m "$mode" \
    "$VERIFIED_RELEASE_DIR/$file" \
    "/var/lib/onyx/regulatory-prod-lite-staging/$APPROVED_PRIVILEGED_MANIFEST_SHA256/$file"
done <"$VERIFIED_RELEASE_DIR/REGULATORY_PRIVILEGED_MANIFEST.sha256"
/usr/bin/install -o root -g root -m 0644 \
  "$VERIFIED_RELEASE_DIR/REGULATORY_PRIVILEGED_MANIFEST.sha256" \
  "/var/lib/onyx/regulatory-prod-lite-staging/$APPROVED_PRIVILEGED_MANIFEST_SHA256/REGULATORY_PRIVILEGED_MANIFEST.sha256"
```

Bootstrap or upgrade the installer itself only from that same identity-verified artifact. Copy it
to a temporary root-owned file, compare its SHA-256 with the signed outer
`REGULATORY_RELEASE_MANIFEST.txt`, and atomically rename it to the fixed path. Do not grant operators
sudo for this provisioning step and do not invoke the extracted installer with `sudo`.

```bash
/usr/bin/install -o root -g root -m 0755 \
  "$VERIFIED_RELEASE_DIR/install-regulatory-prod-lite-privileged-bundle.sh" \
  /usr/local/sbin/.install-regulatory-prod-lite-privileged-bundle.new
# APPROVED_INSTALLER_SHA256 is copied literally from the already verified outer manifest.
printf '%s  %s\n' "$APPROVED_INSTALLER_SHA256" \
  /usr/local/sbin/.install-regulatory-prod-lite-privileged-bundle.new \
  | /usr/bin/sha256sum --check --strict -
/bin/mv -f /usr/local/sbin/.install-regulatory-prod-lite-privileged-bundle.new \
  /usr/local/sbin/install-regulatory-prod-lite-privileged-bundle
/usr/local/sbin/install-regulatory-prod-lite-privileged-bundle \
  --source-dir "/var/lib/onyx/regulatory-prod-lite-staging/$APPROVED_PRIVILEGED_MANIFEST_SHA256" \
  --expected-manifest-sha256 "$APPROVED_PRIVILEGED_MANIFEST_SHA256"
```

The installer never calls sudo, accepts no destination, refuses execution outside its fixed
root-owned path, validates the private staged source, installs a versioned release through a private
temporary directory, fsyncs it, and atomically updates `current`. Any ownership, mode, symlink,
member, or digest error fails closed before activation. Keep the previous versioned release for a
reviewed rollback; never edit an installed release in place. During an upgrade, install the new
bundle, replace the exact sudoers command below with its new literal digest/arguments using
`visudo -c`, and only then open the change window. The wrapper fails closed while `current` and the
authorized command differ.

Before granting preflight authorization, provision the dedicated Docker CLI configuration at
`/etc/onyx/regulatory-docker`. Both `/etc/onyx` and that directory must be root-owned, must not be
group writable or world accessible, and must not be symlinks. The required `config.json` must be a
root-owned, non-symlink, non-executable regular file, no larger than 1 MiB, with no group write/execute
or world access. A dedicated deployment group may have read/traverse access (for example,
directories mode `0750` and `config.json` mode `0640`) so the non-root rollout can use approved
registry credentials without being able to alter them. For a registry that needs no client
configuration, provision an empty root-owned object:

```bash
/usr/bin/sudo install -d -o root -g onyx-deploy -m 0750 /etc/onyx
/usr/bin/sudo install -d -o root -g onyx-deploy -m 0750 /etc/onyx/regulatory-docker
printf '{}\n' | /usr/bin/sudo tee /etc/onyx/regulatory-docker/config.json >/dev/null
/usr/bin/sudo chown root:onyx-deploy /etc/onyx/regulatory-docker/config.json
/usr/bin/sudo chmod 0640 /etc/onyx/regulatory-docker/config.json
```

Replace `onyx-deploy` with the approved read-only deployment group. Provision authenticated registry
configuration through the organization's secret-safe root workflow; do not print it or derive it
from the operator's home directory.

The wrapper invokes absolute `/usr/bin/sudo -n` (the noninteractive `sudo -n` form) only for the
fixed installed entrypoint. Before the change window, the operator must have one `NOPASSWD` sudoers
rule for the literal full argv emitted by the wrapper: fixed entrypoint, installed bundle digest,
absolute env/base/migration paths, project/topology modes, and approved image digests in their exact
order. Do not use sudoers wildcards. Generate the literal line in the change record, install it with
mode `0440`, and validate it with `/usr/sbin/visudo -c`; authorize neither `/usr/bin/env`, `docker`, a
shell, the installer, nor the deployment wrapper. Do not add `SETENV`: the entrypoint clears ambient
state and fixes `PATH`, `HOME`, Python isolation, Docker config, and the local Docker socket itself.
A missing, stale, broader, or interactive authorization is a release blocker: the
wrapper must fail immediately rather than prompt. Root is required so one descriptor-owning helper
can validate the original numeric `1001:1001` files and hand private `1001:1001` snapshots to the
fixed non-root validation container. Do not use `sudo -E`, print the environment, or run the
deployment itself as root. The approved digest arguments below are non-secret values supplied by
release automation. The exact command both verifies noninteractive privilege and runs the preflight.
Remove any older rule that authorizes a checkout path, `/usr/bin/env`, `/bin/bash`, an operator-derived
`PATH`/`HOME`/Docker config, or wildcard arguments; it must not remain as a broader alternate route.
Test the exact rule noninteractively before the window with the canonical wrapper command below and
`/usr/bin/sudo -n -l`; do not run a separate root preflight by hand.
For the recommended cloud model mode with Compose-managed data services:

```bash
RELEASE_DIR=$(pwd -P)  # verified extracted deployment/docker_compose directory
./regulatory-prod-lite-deploy.sh preflight \
  --env-file "$RELEASE_DIR/.env" \
  --base-compose "$RELEASE_DIR/docker-compose.prod.yml" \
  --project-name "$APPROVED_COMPOSE_PROJECT" \
  --migration-env-file "$RELEASE_DIR/.env.migration" \
  --db-admin-env-file "$RELEASE_DIR/.env.db-admin" \
  --infra-mode compose-managed \
  --model-mode cloud \
  --expected-image "$APPROVED_BACKEND_DIGEST" \
  --expected-web-image "$APPROVED_WEB_DIGEST"
```

For the recommended cloud model mode with approved external data services, with `COMPOSE_PROFILES`
excluding `local-infra` and `s3-filestore`:

```bash
RELEASE_DIR=$(pwd -P)  # verified extracted deployment/docker_compose directory
./regulatory-prod-lite-deploy.sh preflight \
  --env-file "$RELEASE_DIR/.env" \
  --base-compose "$RELEASE_DIR/docker-compose.prod.yml" \
  --project-name "$APPROVED_COMPOSE_PROJECT" \
  --migration-env-file "$RELEASE_DIR/.env.migration" \
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
generic indexing work must be zero. Inspect `user_file_processing` separately: only known Markdown
messages intended for the new durable worker may survive cutover, and the service owner must approve
their tenant/file identities. Unsupported or unidentified messages must be drained or quarantined.
The service owner must explicitly approve any queued
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

Keep `REGULATORY_BATCH_INDEXING_ENABLED=false` through the singleton `alembic upgrade head` and
initial readiness checks. Before enabling it, configure a non-secret `REGULATORY_INDEXING_GCS_URI`,
archive an approved IAM policy/simulator review for the exact runtime identity and workspace (the
read-only probe cannot prove create/delete/cancel permissions), and verify the active OpenRouter
embedding SearchSettings contract. Then change the
flag for both `api_server` and `background` and restart both through the owned deployment workflow;
an API-only or worker-only flag change is forbidden. Repeat worker, Beat, queue, and Markdown-canary
readiness checks after every indexing environment change. To disable the feature, set the flag false
on both processes and restart both after active work is quiesced; recovery then stops claiming stale
jobs, although broker messages already emitted before the restart may still finish.

The shipped backend-lite image contains a read-only readiness command. Run it inside the new
`background` container after the singleton migration and Admin configuration, while the feature flag
is still false. Preserve the operator-approved IAM review as a distinct archived evidence artifact;
readiness hashes its actual bytes without parsing or printing them. Create a separate, non-secret
attestation that binds that digest to the active scope and runtime identity. Mount both files
read-only at `/run/readiness/regulatory-capabilities.json` and
`/run/readiness/regulatory-capability-evidence.json` in `background`. The attestation must use this
exact schema and a `reviewed_at` no more than 24 hours old:

```json
{
  "schema_version": 1,
  "reviewed_at": "2026-08-20T09:00:00+00:00",
  "identity": "regulatory-runtime@example.iam.gserviceaccount.com",
  "evidence_reference": "gs://approved-audit-bucket/change/task-8.json#sha256=bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
  "evidence_sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
  "gcs_uri": "gs://approved-bucket/exact-regulatory-prefix",
  "vertex_project": "approved-project",
  "vertex_location": "approved-location",
  "vertex_model": "approved-model",
  "permissions": [
    "storage.objects.create",
    "storage.objects.get",
    "storage.objects.delete",
    "storage.objects.list",
    "aiplatform.batchPredictionJobs.create",
    "aiplatform.batchPredictionJobs.get",
    "aiplatform.batchPredictionJobs.cancel",
    "aiplatform.batchPredictionJobs.list",
    "aiplatform.models.get"
  ]
}
```

Replace the example digest with the lowercase SHA-256 of the exact archived evidence file bytes. The evidence
reference must point to that result and end in the exact `#sha256=<evidence_sha256>` binding;
readiness rejects a missing, malformed, or mismatched digest. This proves which archived artifact
was reviewed and prevents the attestation from supplying its own unchecked digest. The approved
archive and change-record controls remain responsible for the review's provenance. Never place
credentials, tokens, service-account JSON, document content, or vectors in either file, and never
print the archived evidence during readiness.

Set `REGULATORY_CAPABILITY_ATTESTATION_FILE` and `REGULATORY_CAPABILITY_EVIDENCE_FILE` to distinct
absolute host paths. The canonical overlay mounts both files read-only. Preflight and the in-container
validator require numeric owner/group `1001:1001`, attestation mode `0600`, evidence mode `0400`,
regular files (no symlinks), and bounded sizes. Invoke the canonical preflight through the bounded
`./regulatory-prod-lite-deploy.sh preflight ...` command in section 4; the wrapper uses
noninteractive least-privilege sudo, and direct non-root invocation of the internal preflight stage
fails before Docker is queried:

```bash
evidence_digest=$(sha256sum -- "$REGULATORY_CAPABILITY_EVIDENCE_FILE" | awk '{print $1}')
# Put exactly $evidence_digest in both attestation digest fields before continuing.
sudo chown 1001:1001 "$REGULATORY_CAPABILITY_ATTESTATION_FILE"
sudo chown 1001:1001 "$REGULATORY_CAPABILITY_EVIDENCE_FILE"
sudo chmod 0600 "$REGULATORY_CAPABILITY_ATTESTATION_FILE"
sudo chmod 0400 "$REGULATORY_CAPABILITY_EVIDENCE_FILE"
```

The readiness process itself runs as the application UID/GID, not as the root Supervisor wrapper.
Supervisor creates `/tmp/supervisor.sock` as numeric `1001:1001` with mode `0770`, so that app user
can perform the read-only local status query while no world access is granted.
For the external-infrastructure topology, the exact command is:

```bash
docker compose --project-name "$APPROVED_COMPOSE_PROJECT" --env-file .env \
  -f docker-compose.prod.yml \
  -f docker-compose.regulatory-edge.yml \
  -f docker-compose.regulatory-external-infra.yml \
  -f docker-compose.regulatory-prod-lite.yml \
  exec -T --user 1001:1001 background python /app/scripts/regulatory_indexing_readiness.py \
    --memory-headroom-reviewed \
    --capability-attestation /run/readiness/regulatory-capabilities.json \
    --capability-evidence /run/readiness/regulatory-capability-evidence.json
```

Use the same command with `docker-compose.regulatory-compose-infra.yml` in Compose-managed mode.
Exit `0` means every check passed, exit `1` means not ready, and exit `2` is an invocation
interruption/error. `--json` emits the same redacted result for archival. The command performs only
database reads, exact configured/live Celery queue inspection, GCS list access, Vertex model/batch reads, one
constant-text OpenRouter embedding call, and Elasticsearch mapping reads. It never indexes a probe,
creates/cancels a batch, writes an object, or prints credentials, vectors, or document content.
The GCS and Vertex calls prove only the listed observational operations. Mutation capability is
accepted only from the fresh archived IAM attestation, whose identity must equal the identity used by
both observational probes; missing, stale, wrong-scope, wrong-identity, or incomplete evidence fails
readiness.

Prerequisites are: the migration job completed; the new disabled API/background containers are
running; the dedicated regulatory worker and Beat probe files are healthy; the active Admin Search
Settings select OpenRouter `openai/text-embedding-3-large` with contextual retrieval enabled; the
selected contextual model is Vertex AI with valid batch/GCS access; and the active Elasticsearch
index exists with both dense-vector fields matching the active effective dimension and mapping
attributes. The archived IAM evidence must enumerate GCS object create/get/delete/list and Vertex
batch create/get/cancel/list plus model get for the exact active scope; the command deliberately does
not exercise create/delete/cancel. Before supplying `--memory-headroom-reviewed`, archive the cgroup and per-process RSS
evidence required in section 6 and verify the pod limit plus node headroom; omitting the attestation
fails closed, and any recorded OOM event still fails the check. The repository cannot validate
external Helm/node capacity by itself. The readiness report's `effective_dimension` is authoritative. It is derived from the
active SearchSettings (`reduced_dimension` when set, otherwise the model dimension), is sent to the
OpenRouter probe, and must match the active index mapping. Never substitute a hardcoded `1024` (or
any other dimension) in deployment configuration or acceptance evidence.

The required rollout order is therefore: keep the flag false in both processes; run the singleton
migration; start/restart API and background on the new image with the flag still false; finish Admin
and workspace configuration; obtain readiness exit `0`; set the flag true for both processes;
restart both through the owned deployment workflow; repeat readiness and queue checks; only then run
the disposable canary below. Any failure returns the rollout to the disabled state; do not weaken or
skip a failed check.

The production-lite operator environment contract is exact; keep the values in the approved secret
store/environment, not in source control. Defaults below match the Compose overlay and
`env.prod.template`:

| Variable | Process scope | Default / operator requirement |
| --- | --- | --- |
| `MARKDOWN_IMPORT_ENABLED` | API + background | `true` |
| `MAX_ARCHIVE_COMPRESSION_RATIO` | API | `100` |
| `MAX_ARCHIVE_ENTRIES` | API | `500` |
| `MAX_ARCHIVE_EXPANDED_BYTES` | API | `536870912` |
| `REGULATORY_BATCH_INDEXING_ENABLED` | API + background | `false`; coordinated enable/restart only |
| `REGULATORY_INDEXING_GCS_URI` | background | Required non-secret `gs://...` workspace before enable |
| `REGULATORY_INDEXING_MAX_ATTEMPTS` | background | `5` |
| `REGULATORY_INDEXING_RETRY_BASE_SECONDS` | background | `15` |
| `REGULATORY_INDEXING_RETRY_MAX_SECONDS` | background | `900` |
| `REGULATORY_INDEXING_POLL_SECONDS` | background | `30` |
| `REGULATORY_INDEXING_LEASE_SECONDS` | background | `120` |
| `REGULATORY_INDEXING_EMBEDDING_REQUEST_SIZE` | background | `64` |

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

The machine-readable block below is the canonical production-lite runtime contract and is parsed by
repository tests. It must change with supervisor, Compose health, workflow readiness, or queue wiring.

<!-- production-lite-runtime-contract:start -->
```yaml
supervisor_process_count: 7
workers:
  celery_worker_regulatory_benchmark:
    - regulatory_benchmark
  celery_worker_regulatory_indexing:
    - user_file_processing
    - regulatory_indexing
  celery_worker_user_file_maintenance:
    - user_file_project_sync
    - user_file_delete
  celery_worker_light:
    - vespa_metadata_sync
    - connector_deletion
    - doc_permissions_upsert
    - checkpoint_cleanup
    - index_attempt_cleanup
    - chat_ttl_deletion
  celery_worker_monitoring:
    - monitoring
scheduler:
  name: celery_beat_regulatory_indexing
  tasks:
    - regulatory_indexing_recover_stale
    - monitor_celery_queues
  readiness_file: /tmp/onyx_k8s_regulatoryindexingbeat_readiness.txt
  liveness_file: /tmp/onyx_k8s_regulatoryindexingbeat_liveness.txt
  liveness_max_age_seconds: 150
  probe_marker: pid:instance_uuid
  dispatch_dedup: redis_tenant_entry_utc_slot_set_nx_ex
  claimant_failure_delivery: next_utc_slot
  max_failover_gap_seconds:
    monitor_celery_queues: 10
    regulatory_indexing_recover_stale: 60
  claim_ttl_semantics: stale_key_retention_not_same_slot_takeover
forbidden_queues:
  - primary
  - docfetching
  - docprocessing
  - indexing
  - elasticsearch_migration
operations:
  feature_flag: REGULATORY_BATCH_INDEXING_ENABLED
  default_enabled: false
  required_workspace: REGULATORY_INDEXING_GCS_URI
  migration_before_enable: alembic upgrade head
  restart_after_config_change:
    - api_server
    - background
```
<!-- production-lite-runtime-contract:end -->

The background container healthcheck requires exactly five workers, the dedicated regulatory
indexing Beat, and the log redirector—seven supervisor processes total—and requires every process to
be `RUNNING`. Each pod keeps its Beat shelf at pod-local
`/tmp/regulatory-indexing-beat-schedule`, regenerates every per-tenant entry from PostgreSQL before
readiness and after a restart, and contains only stale regulatory indexing recovery (one minute) and
queue monitoring (ten seconds). It never loads the generic/full-runtime Beat schedule. A corrupt
pod-local shelf is discarded and rebuilt; no shelf is shared between replicas.

Multiple background replicas are safe without a Helm singleton setting. Before dispatch, every Beat
claims a tenant-prefixed Redis key containing the schedule entry and UTC interval slot with atomic
`SET NX EX`. Only the claimant publishes that tenant/slot; followers remain healthy and try later
slots. If a claimant stops after `SET NX EX` but before broker publication, that UTC slot is not
replayed. Deterministic recovery is the next UTC slot: within ten seconds for queue monitoring and
within sixty seconds for stale indexing recovery. The two-interval TTL only bounds stale-key
retention and clock-anomaly impact; it is not a same-slot takeover guarantee. Redis failure prevents
an uncoordinated publish. The PostgreSQL job table—not the Beat shelf or Redis claim—is the durable
indexing scheduler/source of truth.

At each Beat process start, stale readiness and liveness files are removed before Redis/DB waits.
After dependency and schedule initialization both files contain the current supervisor PID plus a
unique instance UUID. Shutdown removes them where Celery can close cleanly. Liveness is rewritten
only after a successful schedule refresh, and Compose/CodeBuild reject a marker for another PID,
different instance markers, or liveness older than 150 seconds. Followers do not need to own a
dispatch slot to remain ready. `active_queues` must
match only the queues declared above. Primary, docfetching, docprocessing, generic indexing, and
Elasticsearch-migration workers/queues are forbidden; `user_file_processing` is required on the
dedicated regulatory indexing worker.

CodeBuild enumerates every non-terminating Ready background pod whose container uses the new image
tag and runs the PID/instance/freshness verifier in each matching replica. Readiness fails when no
matching replica exists or when any matching replica fails its Beat status or probe checks; checking
only the first pod is not accepted.
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
5. While durable indexing is disabled, keep upload controls closed to users and verify no durable job
   was created. After the migration, provider/GCS checks, coordinated flag enable, and process
   restart, upload one approved small Markdown canary; confirm it progresses through the durable job
   stages and becomes searchable. Confirm non-Markdown/parser-backed upload remains rejected.
6. Stop/restart the regulatory indexing worker after a canary reaches a non-terminal stage and verify
   the dedicated Beat recovers the stale job. Confirm queue-depth metrics include
   `regulatory_indexing` and both exact supervisor processes remain ready.
7. Start one small benchmark run only after chat/search succeeds, and confirm the dedicated worker
   completes it.

Use a newly generated, collision-resistant filename, directory, tenant (where applicable), and file
identity for every canary; never reuse, overwrite, rename, or delete an existing production file.
Upload only through the authenticated frontend/nginx route. The Markdown should contain unique,
non-sensitive marker text and at least two canonical chunks. Record the active SearchSettings ID,
reported effective dimension, durable job ID, file ID, tenant, and timestamps, but do not archive
document content or vectors. Observe contextual completion before embedding, hidden Elasticsearch
staging, verification, publication, search visibility, retrieval, and citation through the same
frontend route. Deliver the same job message again to prove idempotency, and perform one controlled
background restart only after recording a non-terminal stage; the stale-recovery Beat must resume it.

For cancellation coverage, use a second new canary and delete it through the frontend while its job
is non-terminal. Confirm the durable job becomes cancelled and its GCS prefix and staged
Elasticsearch chunks are absent using credential-safe list/count diagnostics. Finally delete the
published canary through the frontend, confirm its unique marker is no longer retrievable and no
job-specific GCS objects or Elasticsearch chunks remain, then remove only the disposable local
evidence files. If any prerequisite, frontend route, or cleanup identity is ambiguous, do not start
the canary. Never use direct backend calls, wildcard deletes, index deletion, broker purge, or cleanup
queries that could target pre-existing documents.

Review API/background logs for repeated database, Elasticsearch, object-store, cloud embedding,
OpenRouter LLM, or Celery errors before ending the maintenance window.

The measured cold import of the added worker was approximately `216568 KiB` maximum RSS in the
implementation environment. This repository does not own the external Helm resource values, so this
is evidence rather than a resource limit. Before rollout, compare the background pod's configured
request/limit and available node headroom against the existing workers plus this process. Archive
pre- and post-deploy cgroup `memory.current`, `memory.peak`, `memory.max`, `memory.events`, and per-process
RSS evidence from CodeBuild. Any OOM event, process restart, or unreviewed headroom deficit blocks
readiness; do not invent or silently raise a fixed limit in this repository.

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
