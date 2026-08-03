# Welcome to Onyx

To set up Onyx there are several options, Onyx supports the following for deployment:
1. Quick guided install via the install.sh script
2. Pulling the repo and running `docker compose up -d` from the deployment/docker_compose directory
  - Note, it is recommended to copy over the env.template file to .env and edit the necessary values
3. For large scale deployments leveraging Kubernetes, there are two options, Helm or Terraform.

This README focuses on the easiest guided deployment which is via install.sh.

**For more detailed guides, please refer to the documentation: https://docs.onyx.app/deployment/overview**

## Regulatory production-lite and local imports

The operational handoff for DevOps is in
[`REGULATORY_PRODUCTION_RUNBOOK.md`](REGULATORY_PRODUCTION_RUNBOOK.md).

`docker-compose.regulatory-prod-lite.yml` is an additive overlay for deployments that search an
already-built regulatory index but do not parse source documents on the production host. The lite
image contains chat, search, the API, and lightweight maintenance workers. Its supervisor runs only
the dedicated `regulatory_benchmark`, indexed-file maintenance, `light`, and `monitoring` queues; it
has no generic scheduled-task, Beat, primary, document-processing, or document-fetching worker.
Document-import APIs are disabled, while existing chunks in the production OpenSearch index remain
searchable. The code interpreter is parked behind an opt-in profile.

### Release contract

Builds run on a trusted CI/build runner, never on the production host. Authenticate Docker to the
target registry, then publish all images from the same full commit SHA instead of `latest`:

```bash
deployment/docker_compose/publish-regulatory-images.sh \
  registry.example.com/team \
  "$GIT_COMMIT_SHA"
```

The script publishes four immutable artifacts: runtime-lite backend, custom web, model server, and
the full-runtime one-shot importer. Store `ONYX_BACKEND_LITE_IMAGE`, `ONYX_WEB_SERVER_IMAGE`, and
`ONYX_MODEL_SERVER_IMAGE` in production; store `ONYX_IMPORTER_IMAGE` on the import workstation. The
four digests emitted by one invocation are one release set; do not mix revisions.

The production overlay requires digest-pinned backend, web, and model references and resets their
inherited `build` definitions. It also removes the indexing-model dependency and parks that service
behind an opt-in profile. Use the authoritative preflight/deploy scripts; do not assemble an ad hoc
Compose chain:

```bash
./regulatory-prod-lite-preflight.sh --help
./regulatory-prod-lite-deploy.sh --help
```

The scripts require an explicit infrastructure mode, model mode, expected release digests, bounded
wait, and backup/migration acknowledgements. They enforce `--no-build`; database rollback
compatibility still depends on the migrations in the selected application release. Follow
`REGULATORY_PRODUCTION_RUNBOOK.md` for exact commands.

The base API migration command is supported only for the reviewed single-tenant deployment. The
production guard refuses multi-tenant mode because its catalog and every tenant schema require a
separate migration workflow. Parser packaging and embedding deployment are independent decisions.
Only add `docker-compose.no-local-models.yml` after the active SearchSettings have been verified to use
a reachable cloud embedding provider. Enable the base file's `s3-filestore` profile only when this
deployment owns its MinIO service.

Source-document import is a separate one-shot operation using the full backend runtime. It calls the
same application indexing path as the upload worker, but it does not start Celery or Redis, expose a
port, run Alembic, or run on the production application host. The importer compose file has no build
definition: it pulls the digest-pinned `ONYX_IMPORTER_IMAGE` produced for the same release as the lite
image.

```bash
cp env.regulatory-importer.template .env.regulatory-importer
chmod 600 .env.regulatory-importer
# Fill all required values and set IMPORTER_UID/GID from `id -u` / `id -g`.
./regulatory-import-run.sh --help

# Documents are read-only under /imports and manifests belong under /output.
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

The importer needs reachable TLS-protected endpoints for the same PostgreSQL database, OpenSearch
cluster, and object store used by production. It also needs the same default schema/tenant topology,
encryption secret, SearchSettings, and embedding/contextual-retrieval configuration. Database access
alone is insufficient when originals live in S3 or MinIO. Do not expose these services directly to
the public internet; use private networking, a VPN, or authenticated tunnels. The source directory
and CA bundle are mounted read-only; only the explicitly configured output directory is writable.
The importer verifies its manifest directory before creating any database, object-store, or index
records, so a bad UID/GID or mount permission fails before the import starts.
The completed `.env.regulatory-importer` must never be committed.

If the active SearchSettings use local embeddings, run the matching model-server digest on the import
workstation/private import network and point `IMPORTER_INDEXING_MODEL_SERVER_HOST` to it. Otherwise
use a verified cloud embedding configuration. Never re-enable production's indexing model server for
an import.

The compose file maps `host.docker.internal` to Docker's host gateway for VPN or SSH-tunnel use.
On Linux, a tunnel listening only on host `127.0.0.1` is not reachable from a container. Prefer a
private VPN endpoint. `verify-full` TLS and RDS IAM require the connection hostname to remain the
certificate/token hostname, so `host.docker.internal` is not a valid substitute; use a small local
compose override that maps the real service hostname to `host-gateway`, or connect through private
DNS. Otherwise bind the forward only to the host's Docker-bridge/private interface and firewall it to
the Docker bridge. Never bind PostgreSQL, OpenSearch, or object-store forwards publicly. RDS or
OpenSearch IAM also requires the standard `IMPORTER_AWS_*` credentials; temporary credentials must
include `IMPORTER_AWS_SESSION_TOKEN`.

## install.sh script

```
curl -fsSL https://raw.githubusercontent.com/onyx-dot-app/onyx/main/deployment/docker_compose/install.sh > install.sh && chmod +x install.sh && ./install.sh
```

The script installs the Onyx CLI (`onyx-cli`) and hands over to `onyx-cli deploy install`, which is
where the guided installation lives. Any flags you pass are forwarded to it. If you already have the
CLI (`pip install onyx-cli`), skip the script and run `onyx-cli deploy install` directly.

This provides a guided installation of Onyx via Docker Compose. It will deploy the latest version of Onyx
and set up the volumes to ensure data is persisted across deployments or upgrades.

The deployment files are stored in `~/.config/onyx` (an existing `onyx_data` directory from an older
install is detected and kept in place; `--dir` targets another location). Note that no application
critical data is stored in that directory so even if you delete it, the data needed to restore the app
will not be destroyed.

The data about chats, users, etc. are instead stored as named Docker Volumes. This is managed by Docker
and where it is stored will depend on your Docker setup. You can always delete these as well by running
`onyx-cli deploy uninstall`.

To shut down the deployment without deleting, use `onyx-cli deploy stop`.

### Managing the deployment

Beyond installing, the CLI covers the rest of the lifecycle:

| Command | What it does |
| --- | --- |
| `onyx-cli deploy status` | Installed version, containers, and health (`--json` for scripts) |
| `onyx-cli deploy logs [service...]` | Logs of the deployment's containers |
| `onyx-cli deploy stop` | Stop the containers, keep the data |
| `onyx-cli deploy upgrade [--tag vX.Y.Z]` | Upgrade in place (see below) |
| `onyx-cli deploy uninstall` | Remove the containers, volumes, and deployment directory |

### Upgrading the deployment
Onyx maintains backwards compatibility across all minor versions following SemVer, so upgrading is
`onyx-cli deploy upgrade` (add `--tag vX.Y.Z` to pin a version). It rewrites only IMAGE_TAG, preserves
your .env edits, and backs up hand-edited files.

If you are more comfortable running docker compose commands, you can also run commands directly from
the directory with the docker-compose.yml file. First bring the containers down (`docker compose down`),
verify the version you want in the environment file (see below), (if using `latest` tag, be sure to run
`docker compose pull`) and run `docker compose up` to restart the services on the latest version

### Environment variables
The Docker Compose files try to look for a .env file in the same directory. The installer sets it up
from a file called env.template. Feel free to edit the .env file to customize your deployment. The most
important / common changed values are located near the top of the file. Later `onyx-cli deploy` runs
keep your edits.

IMAGE_TAG is the version of Onyx to run. It is recommended to leave it as latest to get all updates with each redeployment.

Every image publishes a `-dev` twin for each of its tags (e.g. `latest-dev`, `v1.2.3-dev`), so a single
`IMAGE_TAG=latest-dev` selects the dev variant of the whole deployment. Today only the backend image actually differs:
its `-dev` twin adds interactive debugging tools (vim, nano, curl, ps, psql) that the default image leaves out to stay
minimal. The web-server, model-server, and sandbox `-dev` tags are identical to their plain counterparts and exist so
that one version string covers every image.
