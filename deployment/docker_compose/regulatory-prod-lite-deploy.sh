#!/bin/bash -p

set -Eeuo pipefail
umask 077

readonly TRUSTED_SYSTEM_PATH="/usr/sbin:/usr/bin:/sbin:/bin"
readonly ENV_BIN="/usr/bin/env"
readonly BASH_BIN="/bin/bash"
readonly SUDO_BIN="/usr/bin/sudo"
readonly DOCKER_BIN="/usr/bin/docker"
readonly COMPOSE_BIN="/usr/libexec/docker/cli-plugins/docker-compose"
readonly DOCKER_CONFIG_DIR="/etc/onyx/regulatory-docker"
readonly DEPLOYMENT_HOME="/var/empty"
export PATH="$TRUSTED_SYSTEM_PATH"
unset BASH_ENV CDPATH ENV GLOBIGNORE LD_LIBRARY_PATH LD_PRELOAD PYTHONHOME \
  PYTHONPATH SUDO_ASKPASS SUDO_ASKPASS_REQUIRE

SCRIPT_DIR=$(CDPATH='' cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
readonly SCRIPT_DIR
readonly PREFLIGHT="$SCRIPT_DIR/regulatory-prod-lite-preflight.sh"
readonly REGULATORY_OVERLAY="$SCRIPT_DIR/docker-compose.regulatory-prod-lite.yml"
readonly EXTERNAL_INFRA_OVERLAY="$SCRIPT_DIR/docker-compose.regulatory-external-infra.yml"
readonly NO_LOCAL_MODELS_OVERLAY="$SCRIPT_DIR/docker-compose.no-local-models.yml"
readonly EDGE_OVERLAY="$SCRIPT_DIR/docker-compose.regulatory-edge.yml"
readonly COMPOSE_INFRA_OVERLAY="$SCRIPT_DIR/docker-compose.regulatory-compose-infra.yml"
readonly DEPLOYMENT_DOCKER_HOST="unix:///var/run/docker.sock"

usage() {
  cat <<'EOF'
Deploy or inspect the digest-pinned regulatory production-lite stack.

Usage:
  regulatory-prod-lite-deploy.sh deploy \
    --expected-image REPOSITORY@sha256:DIGEST --expected-web-image REPOSITORY@sha256:DIGEST \
    --expected-model-image REPOSITORY@sha256:DIGEST --backup-reference REFERENCE \
    --infra-mode MODE --model-mode MODE --acknowledge-migration-impact [options]

  regulatory-prod-lite-deploy.sh rollback \
    --expected-image REPOSITORY@sha256:DIGEST --expected-web-image REPOSITORY@sha256:DIGEST \
    --expected-model-image REPOSITORY@sha256:DIGEST --backup-reference REFERENCE \
    --infra-mode MODE --model-mode MODE --schema-compatible [options]

  regulatory-prod-lite-deploy.sh status [options]
  regulatory-prod-lite-deploy.sh preflight [--expected-image REFERENCE] [options]

Options:
  --env-file PATH          Compose environment (default: .env beside this script)
  --base-compose PATH      Production base file (default: docker-compose.prod.yml)
  --project-name NAME      Required existing/intended Compose project name (normally: onyx)
  --migration-env-file PATH
                           Required dedicated migration credentials (mode 0600)
  --db-admin-env-file PATH Required only for compose-managed PostgreSQL credentials
  --infra-mode MODE        Required: compose-managed or external
  --model-mode MODE        Required: local or cloud
  --expected-image REF     Reviewed, immutable runtime-lite image reference
  --expected-web-image REF Reviewed, immutable matching web image reference
  --expected-model-image REF
                           Local mode only: reviewed matching model image reference
  --backup-reference REF   Acknowledgement of a verified external DB backup/snapshot
  --acknowledge-migration-impact
                           Confirm extension/DDL authority and reviewed migration data effects
  --schema-compatible      Required for rollback; confirms the older app supports the live schema
  --wait-timeout SECONDS   Bounded Compose health wait (default: 900, maximum: 3600)
  -h, --help               Show this help

Deploy explicitly runs `alembic upgrade head` from the pinned image after the backup acknowledgement,
checks API liveness, then starts with `--no-build --wait`. Multi-tenant deployments are refused because
they require an approved tenant-migration orchestrator. Rollback changes only the application image: it
never downgrades, restores, or otherwise mutates the database schema.
All Docker and Compose commands use one sanitized environment, fixed root-owned Docker CLI
configuration, and the fixed local Docker socket. Ambient Docker, Compose, executable-path, home,
and image overrides cannot cross the boundary. Non-root operators need noninteractive sudo
authorization only for the exact bounded preflight handoff.
EOF
}

die() {
  printf 'Deployment refused: %s\n' "$1" >&2
  exit 1
}

require_value() {
  local option=$1
  local value=${2:-}
  [[ -n "$value" ]] || die "$option requires a value"
}

validate_acknowledgement() {
  local value=$1
  [[ -n "$value" ]] || die "--backup-reference is required"
  ((${#value} <= 256)) || die "--backup-reference is too long"
  [[ "$value" != *$'\n'* && "$value" != *$'\r'* ]] || \
    die "--backup-reference must be a single line"
}

reject_ambient_deployment_overrides() {
  local variable

  while IFS= read -r variable; do
    case "$variable" in
      COMPOSE_PROFILES)
        die "COMPOSE_PROFILES must be unset; topology is selected only by --infra-mode and --model-mode"
        ;;
      COMPOSE_* | DOCKER_* | ONYX_*_IMAGE | REGULATORY_*_IMAGE | \
        BASE_IMAGE_REGISTRY | IMAGE_TAG)
        die "$variable must be unset; deployment inputs come only from the selected environment file and fixed local Docker target"
        ;;
    esac
  done < <(compgen -e)
}

if (($# == 0)); then
  usage >&2
  exit 64
fi

command_name=$1
shift

case "$command_name" in
  deploy | rollback | status | preflight) ;;
  -h | --help)
    usage
    exit 0
    ;;
  *)
    die "unknown command: $command_name"
    ;;
esac

env_file="$SCRIPT_DIR/.env"
base_compose="$SCRIPT_DIR/docker-compose.prod.yml"
expected_image=""
expected_web_image=""
expected_model_image=""
backup_reference=""
schema_compatible=false
wait_timeout=900
infra_mode=""
model_mode=""
migration_impact_acknowledged=false
project_name=""
migration_env_file=""
db_admin_env_file=""

while (($#)); do
  case "$1" in
    --env-file)
      require_value "$1" "${2:-}"
      env_file=$2
      shift 2
      ;;
    --base-compose)
      require_value "$1" "${2:-}"
      base_compose=$2
      shift 2
      ;;
    --project-name)
      require_value "$1" "${2:-}"
      project_name=$2
      shift 2
      ;;
    --migration-env-file)
      require_value "$1" "${2:-}"
      migration_env_file=$2
      shift 2
      ;;
    --db-admin-env-file)
      require_value "$1" "${2:-}"
      db_admin_env_file=$2
      shift 2
      ;;
    --infra-mode)
      require_value "$1" "${2:-}"
      infra_mode=$2
      shift 2
      ;;
    --model-mode)
      require_value "$1" "${2:-}"
      model_mode=$2
      shift 2
      ;;
    --expected-image)
      require_value "$1" "${2:-}"
      expected_image=$2
      shift 2
      ;;
    --expected-web-image)
      require_value "$1" "${2:-}"
      expected_web_image=$2
      shift 2
      ;;
    --expected-model-image)
      require_value "$1" "${2:-}"
      expected_model_image=$2
      shift 2
      ;;
    --backup-reference)
      require_value "$1" "${2:-}"
      backup_reference=$2
      shift 2
      ;;
    --acknowledge-migration-impact)
      migration_impact_acknowledged=true
      shift
      ;;
    --schema-compatible)
      schema_compatible=true
      shift
      ;;
    --wait-timeout)
      require_value "$1" "${2:-}"
      wait_timeout=$2
      shift 2
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      die "unknown option: $1"
      ;;
  esac
done

[[ "$wait_timeout" =~ ^[1-9][0-9]*$ ]] || die "--wait-timeout must be a positive integer"
((wait_timeout <= 3600)) || die "--wait-timeout cannot exceed 3600 seconds"

reject_ambient_deployment_overrides
export HOME="$DEPLOYMENT_HOME"
deployment_environment=(
  "$ENV_BIN" -i
  "PATH=$TRUSTED_SYSTEM_PATH"
  "HOME=$DEPLOYMENT_HOME"
  "TMPDIR=/tmp"
  "LANG=C"
  "LC_ALL=C"
  "DOCKER_HOST=$DEPLOYMENT_DOCKER_HOST"
  "DOCKER_CONFIG=$DOCKER_CONFIG_DIR"
)
readonly -a deployment_environment

preflight_args=(
  --env-file "$env_file"
  --base-compose "$base_compose"
  --project-name "$project_name"
  --migration-env-file "$migration_env_file"
  --infra-mode "$infra_mode"
  --model-mode "$model_mode"
)
if [[ -n "$db_admin_env_file" ]]; then
  preflight_args+=(--db-admin-env-file "$db_admin_env_file")
fi
if [[ -n "$expected_image" ]]; then
  preflight_args+=(--expected-image "$expected_image")
fi
if [[ -n "$expected_web_image" ]]; then
  preflight_args+=(--expected-web-image "$expected_web_image")
fi
if [[ -n "$expected_model_image" ]]; then
  preflight_args+=(--expected-model-image "$expected_model_image")
fi

run_preflight() {
  if ((EUID == 0)); then
    "${deployment_environment[@]}" \
      "$BASH_BIN" -p "$PREFLIGHT" "${preflight_args[@]}"
    return
  fi
  [[ -x "$SUDO_BIN" ]] || \
    die "noninteractive sudo authorization is required for the bounded readiness preflight"
  if ! "$SUDO_BIN" -n -- \
    "${deployment_environment[@]}" \
    "$BASH_BIN" -p "$PREFLIGHT" "${preflight_args[@]}"; then
    die "the bounded readiness preflight failed; noninteractive sudo authorization for this exact handoff is required"
  fi
}

if [[ "$command_name" == "preflight" ]]; then
  run_preflight
  exit 0
fi

run_preflight

compose=(
  "${deployment_environment[@]}" "$COMPOSE_BIN"
  --project-name "$project_name"
  --env-file "$env_file"
  -f "$base_compose"
  -f "$EDGE_OVERLAY"
)
if [[ "$infra_mode" == "compose-managed" ]]; then
  compose+=(-f "$COMPOSE_INFRA_OVERLAY")
else
  compose+=(-f "$EXTERNAL_INFRA_OVERLAY")
fi
if [[ "$model_mode" == "cloud" ]]; then
  compose+=(-f "$NO_LOCAL_MODELS_OVERLAY")
fi
compose+=(-f "$REGULATORY_OVERLAY")

if [[ "$command_name" == "status" ]]; then
  exec "${compose[@]}" ps
fi

[[ -n "$expected_image" ]] || die "--expected-image is required for deploy and rollback"
[[ -n "$expected_web_image" ]] || die "--expected-web-image is required for deploy and rollback"
if [[ "$model_mode" == "local" ]]; then
  [[ -n "$expected_model_image" ]] || \
    die "--expected-model-image is required for local-model deploy and rollback"
elif [[ -n "$expected_model_image" ]]; then
  die "--expected-model-image must be omitted in cloud model mode"
fi
validate_acknowledgement "$backup_reference"

if [[ "$command_name" == "rollback" && "$schema_compatible" != true ]]; then
  die "rollback requires --schema-compatible; database downgrade is never automatic"
fi
if [[ "$command_name" == "deploy" && "$schema_compatible" == true ]]; then
  die "--schema-compatible is only valid for rollback"
fi
if [[ "$command_name" == "deploy" && "$migration_impact_acknowledged" != true ]]; then
  die "deploy requires --acknowledge-migration-impact after reviewing DDL/extension privileges and migration data effects"
fi
if [[ "$command_name" == "rollback" && "$migration_impact_acknowledged" == true ]]; then
  die "--acknowledge-migration-impact is only valid for deploy"
fi

command -v flock >/dev/null 2>&1 || die "flock is required for rollout serialization"
lock_file="${TMPDIR:-/tmp}/regulatory-prod-lite-${project_name}.lock"
exec 9>"$lock_file"
if ! flock -n 9; then
  die "another deployment or rollback is already running for this Compose project"
fi

printf '%s\n' "Pulling the reviewed production application digests."
"${compose[@]}" pull api_server background web_server nginx certbot
if [[ "$model_mode" == "local" ]]; then
  "${compose[@]}" pull inference_model_server
fi

inspect_image_contract() {
  local image_ref=$1
  local expected_role=$2
  local labels

  labels=$("${deployment_environment[@]}" \
    "$DOCKER_BIN" image inspect --format '{{json .Config.Labels}}' "$image_ref") || \
    die "a pulled application image cannot be inspected"
  jq -e --arg role "$expected_role" '
    .["io.regulatory.role"] == $role
    and .["io.regulatory.document-import"] == "false"
    and (
      .["org.opencontainers.image.revision"]
      | type == "string"
        and test("^([0-9a-f]{40}|[0-9a-f]{64})$")
    )
  ' <<<"$labels" >/dev/null || \
    die "a pulled application image does not satisfy its regulatory release contract"
  jq -r '.["org.opencontainers.image.revision"]' <<<"$labels"
}

backend_revision=$(inspect_image_contract "$expected_image" runtime-lite)
web_revision=$(inspect_image_contract "$expected_web_image" web)
[[ "$backend_revision" == "$web_revision" ]] || \
  die "backend and web images were not built from the same source revision"
if [[ "$model_mode" == "local" ]]; then
  model_revision=$(inspect_image_contract "$expected_model_image" model-server)
  [[ "$backend_revision" == "$model_revision" ]] || \
    die "backend and model images were not built from the same source revision"
fi

if ! "${deployment_environment[@]}" "$DOCKER_BIN" run --rm \
  --network none \
  --read-only \
  --cap-drop ALL \
  --security-opt no-new-privileges \
  --tmpfs /tmp:rw,noexec,nosuid,size=16m \
  --env PYTHONDONTWRITEBYTECODE=1 \
  --entrypoint python \
  "$expected_image" \
  -c 'import importlib.util, os; assert os.environ.get("DOCUMENT_IMPORT_ENABLED", "").lower() == "false"; assert all(importlib.util.find_spec(name) is None for name in ("docling", "markitdown", "nvidia", "pypdfium2", "unstructured", "unstructured_client", "playwright", "torch", "triton"))'; then
  die "the backend image failed the parser-free runtime probe"
fi

printf '%s\n' "Stopping any legacy indexing-only model container."
"${compose[@]}" stop indexing_model_server
if ! running_indexer=$("${compose[@]}" ps --status running --quiet indexing_model_server); then
  die "the indexing_model_server state cannot be verified"
fi
if [[ -n "$running_indexer" ]]; then
  die "indexing_model_server is still running after the stop request"
fi

if [[ "$model_mode" == "cloud" ]]; then
  printf '%s\n' "Stopping any legacy local inference-model container."
  "${compose[@]}" stop inference_model_server
  if ! running_inference=$(
    "${compose[@]}" ps --status running --quiet inference_model_server
  ); then
    die "the inference_model_server state cannot be verified"
  fi
  if [[ -n "$running_inference" ]]; then
    die "inference_model_server is still running in cloud model mode"
  fi
fi

printf '%s\n' "Stopping the old background worker before the application transition."
"${compose[@]}" stop background

printf '%s\n' "Entering the maintenance window by stopping ingress, web, and API services."
"${compose[@]}" stop nginx web_server api_server

if [[ "$infra_mode" == "compose-managed" && "$command_name" == "deploy" ]]; then
  printf '%s\n' "Pulling and checking Compose-managed data services after workers are stopped."
  "${compose[@]}" pull relational_db elasticsearch cache minio nginx certbot
  "${compose[@]}" up -d --no-build --wait --wait-timeout "$wait_timeout" \
    relational_db elasticsearch cache minio
fi

if [[ "$command_name" == "deploy" ]]; then
  printf '%s\n' "Backup acknowledgement recorded; applying forward database migrations."
  if ! "${compose[@]}" \
    --profile regulatory-migration \
    run --rm --no-deps --env-from-file "$migration_env_file" regulatory_migration; then
    printf '%s\n' \
      "Migration failed. Application services remain stopped and data services remain intact." >&2
    exit 1
  fi
else
  printf '%s\n' "Schema compatibility acknowledged; no database downgrade or restore will be attempted."
fi

printf '%s\n' "Starting the API liveness gate without host builds."
if ! "${compose[@]}" up -d --no-build --wait --wait-timeout "$wait_timeout" api_server; then
  printf '%s\n' \
    "API liveness gate failed. Ingress, web, and background remain stopped; inspect service logs." >&2
  exit 1
fi

printf '%s\n' "Starting the remaining production stack without host builds."
if ! "${compose[@]}" up -d --no-build --wait --wait-timeout "$wait_timeout"; then
  "${compose[@]}" stop nginx web_server api_server background || true
  printf '%s\n' \
    "Stack health gate failed. Application services were stopped; data services remain intact." >&2
  exit 1
fi
if ! running_indexer=$("${compose[@]}" ps --status running --quiet indexing_model_server); then
  die "the indexing_model_server state cannot be verified after deployment"
fi
if [[ -n "$running_indexer" ]]; then
  die "indexing_model_server became active after deployment"
fi
if [[ "$model_mode" == "cloud" ]]; then
  if ! running_inference=$(
    "${compose[@]}" ps --status running --quiet inference_model_server
  ); then
    die "the inference_model_server state cannot be verified after deployment"
  fi
  if [[ -n "$running_inference" ]]; then
    die "inference_model_server became active in cloud model mode"
  fi
fi
"${compose[@]}" ps
