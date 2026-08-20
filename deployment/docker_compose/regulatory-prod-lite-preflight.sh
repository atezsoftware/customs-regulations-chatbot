#!/bin/bash -p

set -Eeuo pipefail
umask 077

readonly TRUSTED_SYSTEM_PATH="/usr/sbin:/usr/bin:/sbin:/bin"
readonly DOCKER_BIN="/usr/bin/docker"
readonly COMPOSE_BIN="/usr/libexec/docker/cli-plugins/docker-compose"
readonly DOCKER_CONFIG_DIR="/etc/onyx/regulatory-docker"
readonly DEPLOYMENT_DOCKER_HOST="unix:///var/run/docker.sock"
readonly DEPLOYMENT_HOME="/var/empty"
export PATH="$TRUSTED_SYSTEM_PATH"
export HOME="$DEPLOYMENT_HOME"
export TMPDIR="/tmp"
export LANG="C"
export LC_ALL="C"
unset BASH_ENV CDPATH ENV GLOBIGNORE LD_LIBRARY_PATH LD_PRELOAD PYTHONHOME \
  PYTHONPATH SUDO_ASKPASS SUDO_ASKPASS_REQUIRE

readonly MINIMUM_COMPOSE_VERSION="2.24.4"
SCRIPT_DIR=$(CDPATH='' cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
readonly SCRIPT_DIR
readonly DEFAULT_BASE_COMPOSE="$SCRIPT_DIR/docker-compose.prod.yml"
readonly REGULATORY_OVERLAY="$SCRIPT_DIR/docker-compose.regulatory-prod-lite.yml"
readonly EXTERNAL_INFRA_OVERLAY="$SCRIPT_DIR/docker-compose.regulatory-external-infra.yml"
readonly NO_LOCAL_MODELS_OVERLAY="$SCRIPT_DIR/docker-compose.no-local-models.yml"
readonly EDGE_OVERLAY="$SCRIPT_DIR/docker-compose.regulatory-edge.yml"
readonly COMPOSE_INFRA_OVERLAY="$SCRIPT_DIR/docker-compose.regulatory-compose-infra.yml"
readonly READINESS_SNAPSHOT_HELPER="$SCRIPT_DIR/regulatory_readiness_file_snapshot.py"
readonly READINESS_VALIDATION_TIMEOUT_SECONDS=30
readonly READINESS_OWNERSHIP_LABEL="io.regulatory.readiness-preflight-owner"

usage() {
  cat <<'EOF'
Validate a regulatory production-lite deployment without changing Docker state.

Usage:
  regulatory-prod-lite-preflight.sh [options]

Options:
  --env-file PATH          Compose interpolation environment (default: .env beside this script)
  --base-compose PATH      Production base Compose file (default: docker-compose.prod.yml)
  --project-name NAME      Required existing/intended Compose project name (normally: onyx)
  --migration-env-file PATH
                           Required dedicated migration credentials (mode 0600)
  --db-admin-env-file PATH Required only for compose-managed PostgreSQL bootstrap credentials
  --infra-mode MODE        Required: compose-managed or external
  --model-mode MODE        Required: local or cloud
  --expected-image REF     Require the rendered backend image to equal this digest reference
  --expected-web-image REF Require the rendered web image to equal this digest reference
  --expected-model-image REF
                           Local mode only: require the rendered model image digest
  -h, --help               Show this help

The shipped regulatory overlays are fixed and cannot be replaced from the command line. Backend and
web services must use reviewed repository@sha256 digests and have no build definitions. Import and
indexing workers, ambiguous local/external infrastructure, and generic multi-tenant migrations are
rejected. Descriptor-validated host snapshots are passed to one disposable, read-only, no-network
backend container for an independently bounded digest validation.
EOF
}

die() {
  printf 'Preflight failed: %s\n' "$1" >&2
  exit 1
}

require_value() {
  local option=$1
  local value=${2:-}
  [[ -n "$value" ]] || die "$option requires a value"
}

is_digest_reference() {
  local image_ref=$1
  local repository=${image_ref%@*}
  local final_component=${repository##*/}

  [[ "$image_ref" =~ ^[^[:space:]@]+@sha256:[0-9a-f]{64}$ ]] && \
    [[ "$final_component" != *:* ]]
}

version_at_least() {
  local actual=${1#v}
  local required=$2
  local actual_major actual_minor actual_patch
  local required_major required_minor required_patch

  actual=${actual%%[-+]*}
  IFS=. read -r actual_major actual_minor actual_patch _ <<<"$actual"
  IFS=. read -r required_major required_minor required_patch <<<"$required"

  [[ "$actual_major" =~ ^[0-9]+$ ]] || return 1
  [[ "$actual_minor" =~ ^[0-9]+$ ]] || return 1
  [[ "$actual_patch" =~ ^[0-9]+$ ]] || return 1

  (( actual_major > required_major )) && return 0
  (( actual_major < required_major )) && return 1
  (( actual_minor > required_minor )) && return 0
  (( actual_minor < required_minor )) && return 1
  (( actual_patch >= required_patch ))
}

clear_ambient_deployment_overrides() {
  local variable

  while IFS= read -r variable; do
    case "$variable" in
      COMPOSE_* | DOCKER_* | ONYX_*_IMAGE | REGULATORY_*_IMAGE | \
        BASE_IMAGE_REGISTRY | IMAGE_TAG)
        unset -v "$variable"
        ;;
    esac
  done < <(compgen -e)
  export DOCKER_HOST="$DEPLOYMENT_DOCKER_HOST"
  export DOCKER_CONFIG="$DOCKER_CONFIG_DIR"
}

validate_root_owned_directory() {
  local path=$1
  local description=$2
  local owner mode canonical

  [[ -d "$path" && ! -L "$path" ]] || \
    die "$description must be a non-symlink directory"
  canonical=$(/usr/bin/readlink -f -- "$path") || \
    die "$description cannot be canonicalized"
  [[ "$canonical" == "$path" ]] || \
    die "$description must not traverse symlinks"
  owner=$(/usr/bin/stat -c '%u' -- "$path") || \
    die "$description owner cannot be read"
  mode=$(/usr/bin/stat -c '%a' -- "$path") || \
    die "$description mode cannot be read"
  [[ "$owner" == "0" ]] || die "$description must be owned by root"
  [[ "$mode" =~ ^[0-7]{3,4}$ ]] || die "$description mode is invalid"
  (( (8#$mode & 8#027) == 0 )) || \
    die "$description must not be group writable or world accessible"
}

validate_root_owned_file() {
  local path=$1
  local description=$2
  local maximum_size=${3:-1048576}
  local owner mode size

  [[ -f "$path" && ! -L "$path" ]] || \
    die "$description must be a non-symlink regular file"
  owner=$(/usr/bin/stat -c '%u' -- "$path") || \
    die "$description owner cannot be read"
  mode=$(/usr/bin/stat -c '%a' -- "$path") || \
    die "$description mode cannot be read"
  size=$(/usr/bin/stat -c '%s' -- "$path") || \
    die "$description size cannot be read"
  [[ "$owner" == "0" ]] || die "$description must be owned by root"
  [[ "$mode" =~ ^[0-7]{3,4}$ ]] || die "$description mode is invalid"
  (( (8#$mode & 8#137) == 0 )) || \
    die "$description permits unsafe group/world access or execution"
  if ((maximum_size > 0)); then
    ((size <= maximum_size)) || die "$description exceeds the 1 MiB limit"
  fi
}

validate_root_owned_executable() {
  local path=$1
  local description=$2
  local owner mode

  [[ -f "$path" && ! -L "$path" ]] || \
    die "$description must be a non-symlink regular file"
  owner=$(/usr/bin/stat -c '%u' -- "$path") || \
    die "$description owner cannot be read"
  mode=$(/usr/bin/stat -c '%a' -- "$path") || \
    die "$description mode cannot be read"
  [[ "$owner" == "0" ]] || die "$description must be owned by root"
  [[ "$mode" =~ ^[0-7]{3,4}$ ]] || die "$description mode is invalid"
  (( (8#$mode & 8#022) == 0 )) || \
    die "$description must not be group/world writable"
  [[ -x "$path" ]] || die "$description must be executable"
}

env_file="$SCRIPT_DIR/.env"
base_compose="$DEFAULT_BASE_COMPOSE"
expected_image=""
expected_web_image=""
infra_mode=""
model_mode=""
expected_model_image=""
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
    -h | --help)
      usage
      exit 0
      ;;
    *)
      die "unknown option: $1"
      ;;
  esac
done

((EUID == 0)) || \
  die "this bounded readiness preflight must be invoked as root; use the documented sudo command"
clear_ambient_deployment_overrides
[[ "$project_name" =~ ^[a-z0-9][a-z0-9_-]*$ ]] || \
  die "--project-name is required and must be an explicit lowercase Compose project name"
[[ -n "$migration_env_file" ]] || die "--migration-env-file is required"

validate_root_owned_directory "${DOCKER_CONFIG_DIR%/*}" \
  "the Docker CLI configuration parent directory"
validate_root_owned_directory "$DOCKER_CONFIG_DIR" \
  "the Docker CLI configuration directory"
validate_root_owned_file "$DOCKER_CONFIG_DIR/config.json" \
  "the Docker CLI configuration file"
validate_root_owned_executable "$DOCKER_BIN" "the Docker CLI"
validate_root_owned_executable "$COMPOSE_BIN" "the Docker Compose CLI"
command -v jq >/dev/null 2>&1 || die "jq is required for fail-closed Compose validation"
command -v readlink >/dev/null 2>&1 || die "readlink is required for environment-path validation"
command -v stat >/dev/null 2>&1 || die "stat is required for environment permission validation"
command -v python3 >/dev/null 2>&1 || die "python3 is required for readiness snapshot validation"
command -v timeout >/dev/null 2>&1 || die "timeout is required for bounded readiness validation"
[[ -r "$READINESS_SNAPSHOT_HELPER" && -f "$READINESS_SNAPSHOT_HELPER" ]] || \
  die "the readiness snapshot helper is unavailable"
[[ -r "$env_file" && -f "$env_file" ]] || die "the environment file is not a readable regular file"
[[ -r "$migration_env_file" && -f "$migration_env_file" ]] || \
  die "the migration environment file is not a readable regular file"
[[ -r "$base_compose" && -f "$base_compose" ]] || die "the base Compose file is not readable"
[[ -r "$REGULATORY_OVERLAY" && -f "$REGULATORY_OVERLAY" ]] || die "the regulatory overlay is not readable"
[[ -r "$EDGE_OVERLAY" && -f "$EDGE_OVERLAY" ]] || die "the shipped edge-image overlay is not readable"

env_file_mode=$(stat -c '%a' -- "$env_file") || die "the environment file mode cannot be read"
[[ "$env_file_mode" == "600" ]] || die "the production environment file must have mode 0600"
migration_env_file_mode=$(stat -c '%a' -- "$migration_env_file") || \
  die "the migration environment file mode cannot be read"
[[ "$migration_env_file_mode" == "600" ]] || \
  die "the migration environment file must have mode 0600"

canonical_migration_env=$(readlink -f -- "$migration_env_file") || \
  die "the migration environment file cannot be canonicalized"
canonical_service_migration_env=$(readlink -f -- "$(dirname -- "$base_compose")/.env.migration") || \
  die "the service migration environment path cannot be canonicalized"
[[ "$canonical_migration_env" == "$canonical_service_migration_env" ]] || \
  die "--migration-env-file must be the .env.migration beside the base Compose file"

if ! awk -F= '
  /^[[:space:]]*($|#)/ { next }
  {
    key = $1
    gsub(/^[[:space:]]+|[[:space:]]+$/, "", key)
    if (key != "POSTGRES_USER" && key != "POSTGRES_PASSWORD") { exit 1 }
    count[key]++
    value = substr($0, index($0, "=") + 1)
    gsub(/^[[:space:]]+|[[:space:]]+$/, "", value)
    if (value == "" || value == "\"\"") { exit 1 }
  }
  END {
    if (count["POSTGRES_USER"] != 1 || count["POSTGRES_PASSWORD"] != 1) { exit 1 }
  }
' "$migration_env_file"; then
  die "the migration environment may contain exactly one non-empty POSTGRES_USER and POSTGRES_PASSWORD"
fi

if [[ -n "${COMPOSE_PROFILES:-}" ]] || \
  grep -Eq '^[[:space:]]*COMPOSE_PROFILES[[:space:]]*=' "$env_file"; then
  die "COMPOSE_PROFILES must be unset; topology is selected only by --infra-mode and --model-mode"
fi
if grep -Eq '^[[:space:]]*POSTGRES_MIGRATION_[A-Za-z0-9_]*[[:space:]]*=' "$env_file"; then
  die "migration credentials must not be present in the common production environment"
fi
if grep -Eq '^[[:space:]]*POSTGRES_(ADMIN|DB_ADMIN)_[A-Za-z0-9_]*[[:space:]]*=' "$env_file"; then
  die "database-admin credentials must not be present in the common production environment"
fi

if grep -Eq 'path:[[:space:]]*\.env([[:space:]]|$)' "$base_compose"; then
  canonical_env_file=$(readlink -f -- "$env_file") || \
    die "the selected environment file cannot be canonicalized"
  canonical_service_env=$(readlink -f -- "$(dirname -- "$base_compose")/.env") || \
    die "the base service environment path cannot be canonicalized"
  [[ "$canonical_env_file" == "$canonical_service_env" ]] || \
    die "--env-file must be the same .env file loaded by the base Compose services"
fi

if grep -Eq 'path:[[:space:]]*\.env\.nginx([[:space:]]|$)' "$base_compose"; then
  nginx_env_file=$(dirname -- "$base_compose")/.env.nginx
  [[ -r "$nginx_env_file" && -f "$nginx_env_file" ]] || \
    die ".env.nginx must be provisioned as a readable regular file beside the selected base Compose file"
  nginx_env_file_mode=$(stat -c '%a' -- "$nginx_env_file") || \
    die "the nginx environment file mode cannot be read"
  [[ "$nginx_env_file_mode" == "600" ]] || \
    die "the nginx environment file must have mode 0600"
fi

web_env_file=$(dirname -- "$base_compose")/.env.web
[[ -r "$web_env_file" && -f "$web_env_file" ]] || \
  die ".env.web must be provisioned as a readable regular file beside the base Compose file"
web_env_file_mode=$(stat -c '%a' -- "$web_env_file") || \
  die "the web environment file mode cannot be read"
[[ "$web_env_file_mode" == "600" ]] || die "the web environment file must have mode 0600"
if ! awk -F= '
  /^[[:space:]]*($|#)/ { next }
  {
    key = $1
    gsub(/^[[:space:]]+|[[:space:]]+$/, "", key)
    if (key != "INTERNAL_URL" && key != "DISABLE_ONYX_UPSTREAM_CONNECTIONS" && key != "NEXT_PUBLIC_DISABLE_ONYX_UPSTREAM_CONNECTIONS" && key != "ENABLE_PAID_ENTERPRISE_EDITION_FEATURES" && key != "LICENSE_ENFORCEMENT_ENABLED") { exit 1 }
  }
' "$web_env_file"; then
  die "the web environment contains a key outside the approved non-secret allowlist"
fi

case "$infra_mode" in
  compose-managed)
    [[ -r "$COMPOSE_INFRA_OVERLAY" && -f "$COMPOSE_INFRA_OVERLAY" ]] || \
      die "the shipped Compose-managed infrastructure overlay is not readable"
    [[ -n "$db_admin_env_file" ]] || \
      die "--db-admin-env-file is required in compose-managed mode"
    [[ -r "$db_admin_env_file" && -f "$db_admin_env_file" ]] || \
      die "the database-admin environment file is not a readable regular file"
    db_admin_env_file_mode=$(stat -c '%a' -- "$db_admin_env_file") || \
      die "the database-admin environment file mode cannot be read"
    [[ "$db_admin_env_file_mode" == "600" ]] || \
      die "the database-admin environment file must have mode 0600"
    canonical_db_admin_env=$(readlink -f -- "$db_admin_env_file") || \
      die "the database-admin environment file cannot be canonicalized"
    canonical_service_db_admin_env=$(readlink -f -- "$(dirname -- "$base_compose")/.env.db-admin") || \
      die "the service database-admin environment path cannot be canonicalized"
    [[ "$canonical_db_admin_env" == "$canonical_service_db_admin_env" ]] || \
      die "--db-admin-env-file must be the .env.db-admin beside the base Compose file"
    if ! awk -F= '
      /^[[:space:]]*($|#)/ { next }
      {
        key = $1
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", key)
        if (key != "POSTGRES_USER" && key != "POSTGRES_PASSWORD" && key != "POSTGRES_DB") { exit 1 }
        count[key]++
        value = substr($0, index($0, "=") + 1)
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", value)
        if (value == "" || value == "\"\"") { exit 1 }
      }
      END {
        if (count["POSTGRES_USER"] != 1 || count["POSTGRES_PASSWORD"] != 1 || count["POSTGRES_DB"] != 1) { exit 1 }
      }
    ' "$db_admin_env_file"; then
      die "the database-admin environment may contain exactly POSTGRES_USER, POSTGRES_PASSWORD, and POSTGRES_DB"
    fi
    ;;
  external)
    [[ -r "$EXTERNAL_INFRA_OVERLAY" && -f "$EXTERNAL_INFRA_OVERLAY" ]] || \
      die "the shipped external-infrastructure overlay is not readable"
    [[ -z "$db_admin_env_file" ]] || \
      die "--db-admin-env-file is invalid in external infrastructure mode"
    ;;
  *)
    die "--infra-mode must explicitly be compose-managed or external"
    ;;
esac

case "$model_mode" in
  local) ;;
  cloud)
    [[ -r "$NO_LOCAL_MODELS_OVERLAY" && -f "$NO_LOCAL_MODELS_OVERLAY" ]] || \
      die "the shipped no-local-models overlay is not readable"
    ;;
  *)
    die "--model-mode must explicitly be local or cloud"
    ;;
esac

if [[ "$model_mode" == "cloud" && -n "$expected_model_image" ]]; then
  die "--expected-model-image must be omitted in cloud model mode"
fi

if [[ -n "$expected_image" ]] && ! is_digest_reference "$expected_image"; then
  die "--expected-image must be an immutable repository@sha256 reference"
fi
if [[ -n "$expected_web_image" ]] && ! is_digest_reference "$expected_web_image"; then
  die "--expected-web-image must be an immutable repository@sha256 reference"
fi
if [[ -n "$expected_model_image" ]] && ! is_digest_reference "$expected_model_image"; then
  die "--expected-model-image must be an immutable repository@sha256 reference"
fi

compose_version=$("$COMPOSE_BIN" version --short 2>/dev/null) || \
  die "Docker Compose v2 is unavailable"
version_at_least "$compose_version" "$MINIMUM_COMPOSE_VERSION" || \
  die "Docker Compose $MINIMUM_COMPOSE_VERSION or newer is required for !reset"

config_file=$(mktemp "${TMPDIR:-/tmp}/regulatory-prod-lite-config.XXXXXX")
config_error_file=$(mktemp "${TMPDIR:-/tmp}/regulatory-prod-lite-config-error.XXXXXX")
active_services_file=$(mktemp "${TMPDIR:-/tmp}/regulatory-prod-lite-services.XXXXXX")
profiled_config_file=$(mktemp "${TMPDIR:-/tmp}/regulatory-prod-lite-profiled-config.XXXXXX")
container_inventory_file=$(mktemp "${TMPDIR:-/tmp}/regulatory-prod-lite-containers.XXXXXX")
snapshot_directory=""
readiness_cidfile=""
readiness_ownership_token=""
chmod 600 \
  "$config_file" \
  "$config_error_file" \
  "$active_services_file" \
  "$profiled_config_file" \
  "$container_inventory_file"
cleanup() {
  local primary_status=$?
  local final_status=$primary_status
  local cleanup_failed=false
  local readiness_cid=""
  local label_owned_cid=""
  local matching_containers=""
  local observed_ownership_token=""

  trap - EXIT HUP INT TERM
  set +e

  if [[ -n "$readiness_cidfile" && -f "$readiness_cidfile" && ! -L "$readiness_cidfile" ]]; then
    readiness_cid=$(<"$readiness_cidfile")
    if [[ ! "$readiness_cid" =~ ^[0-9a-f]{64}$ ]]; then
      readiness_cid=""
    fi
  fi

  if [[ -n "$readiness_ownership_token" ]]; then
    if ! matching_containers=$(timeout --foreground --kill-after=2s 10s \
      "$DOCKER_BIN" container ls --all --quiet --no-trunc \
      --filter "label=$READINESS_OWNERSHIP_LABEL=$readiness_ownership_token" 2>/dev/null); then
      printf '%s\n' \
        "Preflight cleanup failed: label-owned readiness containers could not be queried" >&2
      cleanup_failed=true
    elif [[ -n "$matching_containers" && \
      ! "$matching_containers" =~ ^[0-9a-f]{64}$ ]]; then
      printf '%s\n' \
        "Preflight cleanup failed: label-owned readiness container identity is ambiguous" >&2
      cleanup_failed=true
    else
      label_owned_cid=$matching_containers
    fi
  fi

  if [[ "$cleanup_failed" == false && -n "$readiness_cid" && \
    -n "$label_owned_cid" && "$readiness_cid" != "$label_owned_cid" ]]; then
    printf '%s\n' \
      "Preflight cleanup failed: cidfile and ownership label identify different containers" >&2
    cleanup_failed=true
  fi

  if [[ "$cleanup_failed" == false && -z "$label_owned_cid" && -n "$readiness_cid" ]]; then
    if ! matching_containers=$(timeout --foreground --kill-after=2s 10s \
      "$DOCKER_BIN" container ls --all --quiet --no-trunc \
      --filter "id=$readiness_cid" 2>/dev/null); then
      printf '%s\n' \
        "Preflight cleanup failed: readiness container identity could not be queried" >&2
      cleanup_failed=true
    elif [[ -n "$matching_containers" ]]; then
      printf '%s\n' \
        "Preflight cleanup failed: cidfile container is not bound to the private ownership label; refusing removal" >&2
      cleanup_failed=true
    fi
  fi

  if [[ "$cleanup_failed" == false && -n "$label_owned_cid" ]]; then
    if ! observed_ownership_token=$(timeout --foreground --kill-after=2s 10s \
      "$DOCKER_BIN" inspect --type container \
      --format '{{ index .Config.Labels "io.regulatory.readiness-preflight-owner" }}' \
      "$label_owned_cid" 2>/dev/null); then
      printf '%s\n' \
        "Preflight cleanup failed: readiness container ownership label could not be read" >&2
      cleanup_failed=true
    elif [[ "$observed_ownership_token" != "$readiness_ownership_token" ]]; then
      printf '%s\n' \
        "Preflight cleanup failed: readiness container ownership label does not match; refusing removal" >&2
      cleanup_failed=true
    elif ! timeout --foreground --kill-after=2s 10s \
      "$DOCKER_BIN" rm -f "$label_owned_cid" >/dev/null 2>&1; then
      printf '%s\n' \
        "Preflight cleanup failed: could not remove the owned readiness container" >&2
      cleanup_failed=true
    fi
  fi

  if [[ "$cleanup_failed" == false && -n "$readiness_ownership_token" ]]; then
    if ! matching_containers=$(timeout --foreground --kill-after=2s 10s \
      "$DOCKER_BIN" container ls --all --quiet --no-trunc \
      --filter "label=$READINESS_OWNERSHIP_LABEL=$readiness_ownership_token" 2>/dev/null); then
      printf '%s\n' \
        "Preflight cleanup failed: label-owned readiness container removal could not be verified" >&2
      cleanup_failed=true
    elif [[ -n "$matching_containers" ]]; then
      printf '%s\n' \
        "Preflight cleanup failed: a label-owned readiness container still exists" >&2
      cleanup_failed=true
    fi
  fi
  if [[ -n "$snapshot_directory" ]]; then
    if ! rm -f -- \
      "$snapshot_directory/regulatory-capabilities.json" \
      "$snapshot_directory/regulatory-capability-evidence.json" \
      "$snapshot_directory/readiness.cid"; then
      printf '%s\n' "Preflight cleanup failed: private readiness files could not be removed" >&2
      cleanup_failed=true
    fi
    if ! rmdir -- "$snapshot_directory" 2>/dev/null; then
      printf '%s\n' "Preflight cleanup failed: the private readiness directory could not be removed" >&2
      cleanup_failed=true
    fi
  fi
  if ! rm -f -- \
    "$config_file" \
    "$config_error_file" \
    "$active_services_file" \
    "$profiled_config_file" \
    "$container_inventory_file"; then
    printf '%s\n' "Preflight cleanup failed: temporary validation files could not be removed" >&2
    cleanup_failed=true
  fi

  if [[ "$cleanup_failed" == true && "$primary_status" -eq 0 ]]; then
    final_status=1
  fi
  exit "$final_status"
}
trap cleanup EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

snapshot_directory=$(mktemp -d "${TMPDIR:-/tmp}/regulatory-readiness-snapshot.XXXXXX")
chmod 700 "$snapshot_directory"
readiness_cidfile="$snapshot_directory/readiness.cid"
readiness_ownership_token=$(python3 -c \
  'import secrets; print(secrets.token_hex(32))') || \
  die "a private readiness container ownership token could not be created"
[[ "$readiness_ownership_token" =~ ^[0-9a-f]{64}$ ]] || \
  die "the private readiness container ownership token is invalid"

if ! "$DOCKER_BIN" ps \
  --format '{{.ID}}\t{{.Names}}\t{{.Label "com.docker.compose.service"}}' \
  >"$container_inventory_file" 2>"$config_error_file"; then
  die "running container inventory cannot be read"
fi

suspicious_containers=()
while IFS=$'\t' read -r container_id container_name compose_service; do
  [[ -n "$container_name" ]] || continue
  identity=${container_name,,}:${compose_service,,}
  suspicious_container=false
  case "$identity" in
    *indexing_model_server* | *docprocessing* | *docfetching* | *indexing_worker* | \
      *user_file_processing* | *regulatory_importer* | *document_importer* | \
      *celery_worker_primary* | *:importer)
      suspicious_container=true
      ;;
  esac
  container_image_id=$("$DOCKER_BIN" inspect --format '{{.Image}}' "$container_id" 2>/dev/null || true)
  container_role=$("$DOCKER_BIN" image inspect \
    --format '{{index .Config.Labels "io.regulatory.role"}}' \
    "$container_image_id" 2>/dev/null || true)
  if [[ "$container_role" == "importer" ]]; then
    suspicious_container=true
  fi
  if [[ "$model_mode" == "cloud" ]]; then
    case "$identity:$container_role" in
      *inference_model_server* | *indexing_model_server* | *model-server*)
        suspicious_container=true
        ;;
    esac
  fi
  if [[ "$compose_service" == "background" ]]; then
    if [[ "$container_role" != "runtime-lite" ]]; then
      suspicious_container=true
    fi
  fi
  if [[ "$suspicious_container" == true ]]; then
    suspicious_containers+=("$container_name")
  fi
done <"$container_inventory_file"

if ((${#suspicious_containers[@]})); then
  die "running ingestion/indexer or forbidden local-model containers detected (${suspicious_containers[*]}); stop the named containers manually after ownership review"
fi

compose=(
  "$COMPOSE_BIN"
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

if ! "${compose[@]}" config --format json >"$config_file" 2>"$config_error_file"; then
  die "the merged Compose configuration is invalid (details withheld because rendered errors can contain secrets)"
fi
if ! "${compose[@]}" config --services >"$active_services_file" 2>"$config_error_file"; then
  die "the active Compose service set cannot be resolved (details withheld because rendered errors can contain secrets)"
fi
if ! "${compose[@]}" \
  --profile local-models \
  --profile indexing-model-server \
  --profile regulatory-migration \
  config --format json >"$profiled_config_file" 2>"$config_error_file"; then
  die "the profiled model service contract cannot be resolved (details withheld because rendered errors can contain secrets)"
fi

expected_active_services=(api_server background web_server nginx certbot)
if [[ "$model_mode" == "local" ]]; then
  expected_active_services+=(inference_model_server)
fi
if [[ "$infra_mode" == "compose-managed" ]]; then
  expected_active_services+=(relational_db elasticsearch cache minio)
fi

declare -A expected_active_lookup=()
declare -A observed_active_lookup=()
for service in "${expected_active_services[@]}"; do
  expected_active_lookup["$service"]=1
done
unexpected_active_services=()
while IFS= read -r service; do
  [[ -n "$service" ]] || continue
  observed_active_lookup["$service"]=1
  if [[ -z "${expected_active_lookup[$service]+present}" ]]; then
    unexpected_active_services+=("$service")
  fi
done <"$active_services_file"
if ((${#unexpected_active_services[@]})); then
  die "unexpected default-active services detected (${unexpected_active_services[*]}); review profiles and topology"
fi
missing_active_services=()
for service in "${expected_active_services[@]}"; do
  if [[ -z "${observed_active_lookup[$service]+present}" ]]; then
    missing_active_services+=("$service")
  fi
done
if ((${#missing_active_services[@]})); then
  die "required default-active services are missing (${missing_active_services[*]})"
fi

jq -e '
  .services as $services
  | ($services.api_server // null) as $api
  | ($services.background // null) as $background
  | ($services.web_server // null) as $web
  | ($api != null and $background != null and $web != null)
    and ($api.build == null and $background.build == null)
    and ($web.build == null)
    and (($api.user // "") == "1001:1001")
    and ($api.image == $background.image)
    and ($api.image | type == "string")
    and ($api.image | test("^[^[:space:]@]+@sha256:[0-9a-f]{64}$"))
    and ($api.image | split("@")[0] | split("/")[-1] | contains(":") | not)
    and ($web.image | type == "string")
    and ($web.image | test("^[^[:space:]@]+@sha256:[0-9a-f]{64}$"))
    and ($web.image | split("@")[0] | split("/")[-1] | contains(":") | not)
    and (($api.environment.DOCUMENT_IMPORT_ENABLED // "") | tostring | ascii_downcase == "false")
    and (($background.environment.DOCUMENT_IMPORT_ENABLED // "") | tostring | ascii_downcase == "false")
    and (($api.environment.ENABLE_PAID_ENTERPRISE_EDITION_FEATURES // "") | tostring | ascii_downcase == "true")
    and (($background.environment.ENABLE_PAID_ENTERPRISE_EDITION_FEATURES // "") | tostring | ascii_downcase == "true")
    and (($api.environment.LICENSE_ENFORCEMENT_ENABLED // "") | tostring | ascii_downcase == "false")
    and (($background.environment.LICENSE_ENFORCEMENT_ENABLED // "") | tostring | ascii_downcase == "false")
    and (($api.pull_policy // "") == "always")
    and (($background.pull_policy // "") == "always")
    and (($web.pull_policy // "") == "always")
    and (
      [
        ($web.environment // {} | keys[])
        | ascii_upcase
        | select(test("POSTGRES_PASSWORD|OPENAI|OPENROUTER|SECRET|ENCRYPTION"))
      ]
      | length == 0
    )
    and (($api.command // []) | tostring | ascii_downcase | contains("uvicorn"))
    and (($api.command // []) | tostring | ascii_downcase | contains("alembic") | not)
    and (($background.depends_on // {}) | has("indexing_model_server") | not)
    and (
      ($services.indexing_model_server // null) == null
      or (($services.indexing_model_server.profiles // []) | length > 0)
    )
    and (
      [
        "importer",
        "regulatory_importer",
        "document_importer",
        "indexer",
        "document_indexer",
        "indexing_worker",
        "docfetching",
        "docprocessing"
      ]
      | all(. as $forbidden | ($services | has($forbidden) | not))
    )
' "$config_file" >/dev/null || \
  die "image/build/import or production service invariants are invalid"

jq -e '
  [
    (.services.background.volumes // [])[]
    | select(.target == "/run/readiness/regulatory-capabilities.json")
  ] as $mounts
  | ($mounts | length == 1)
    and ($mounts[0].type == "bind")
    and ($mounts[0].read_only == true)
    and (($mounts[0].source // "") | type == "string" and length > 0)
' "$config_file" >/dev/null || \
  die "the readiness capability attestation bind mount must be unique and read-only"

attestation_path=$(jq -er '
  .services.background.volumes[]
  | select(.target == "/run/readiness/regulatory-capabilities.json")
  | .source
' "$config_file") || die "the readiness capability attestation source cannot be resolved"

evidence_mount_count=$(jq '[
    (.services.background.volumes // [])[]
    | select(.type == "bind")
    | select(.target == "/run/readiness/regulatory-capability-evidence.json")
    | select(.read_only == true)
  ] | length' "$config_file") || die "the readiness capability evidence mount cannot be validated"
[[ "$evidence_mount_count" == "1" ]] || \
  die "the readiness archived capability evidence bind mount must be unique and read-only"
evidence_path=$(jq -er '
  .services.background.volumes[]
  | select(.type == "bind" and .target == "/run/readiness/regulatory-capability-evidence.json" and .read_only == true)
  | .source
' "$config_file") || die "the readiness capability evidence source cannot be resolved"
[[ "$evidence_path" != "$attestation_path" ]] || \
  die "the readiness attestation and archived IAM evidence must be distinct files"

if [[ "$model_mode" == "local" ]]; then
  jq -e '
    .services.inference_model_server as $model
    | .services.indexing_model_server as $indexing_model
    | ($model != null and $indexing_model != null)
      and ($model.build == null and $indexing_model.build == null)
      and ($model.image == $indexing_model.image)
      and ($model.image | type == "string")
      and ($model.image | test("^[^[:space:]@]+@sha256:[0-9a-f]{64}$"))
      and ($model.image | split("@")[0] | split("/")[-1] | contains(":") | not)
      and (($model.pull_policy // "") == "always")
      and (($indexing_model.pull_policy // "") == "always")
  ' "$profiled_config_file" >/dev/null || \
    die "local model services must use one approved immutable image without host builds"
fi

jq -e '
  .services.api_server as $api
  | .services.regulatory_migration as $migration
  | ($migration != null)
    and ($migration.image == $api.image)
    and ($migration.build == null)
    and (($migration.pull_policy // "") == "always")
    and (($migration.profiles // []) | index("regulatory-migration") != null)
    and (
      ($migration.command // [])
      | if type == "array" then join(" ") else tostring end
      | ascii_downcase
      | contains("alembic upgrade head")
    )
    and (($migration.command // []) | tostring | ascii_downcase | contains("uvicorn") | not)
    and (($migration.environment.DOCUMENT_IMPORT_ENABLED // "") | tostring | ascii_downcase == "false")
    and (($migration.environment.MULTI_TENANT // "false") | tostring | ascii_downcase != "true")
    and (($migration.environment.POSTGRES_USER // "") | length > 0)
    and (($migration.environment.POSTGRES_PASSWORD // "") | length > 0)
    and ($migration.environment.POSTGRES_USER != $api.environment.POSTGRES_USER)
    and ($migration.environment.POSTGRES_PASSWORD != $api.environment.POSTGRES_PASSWORD)
    and ($migration.environment.POSTGRES_HOST == $api.environment.POSTGRES_HOST)
    and ($migration.environment.POSTGRES_PORT == $api.environment.POSTGRES_PORT)
    and ($migration.environment.POSTGRES_DB == $api.environment.POSTGRES_DB)
' "$profiled_config_file" >/dev/null || \
  die "the profile-gated migration service is missing or does not isolate migration credentials"

jq -e '
  def immutable_image:
    . != null
    and .build == null
    and (.image | type == "string")
    and (.image | test("^[^[:space:]@]+@sha256:[0-9a-f]{64}$"))
    and (.image | split("@")[0] | split("/")[-1] | contains(":") | not)
    and ((.pull_policy // "") == "always");
  (.services.nginx | immutable_image)
  and (.services.certbot | immutable_image)
' "$config_file" >/dev/null || \
  die "nginx and certbot must use approved immutable images without host builds"

if jq -e '
  any(
    .services.api_server.environment.MULTI_TENANT,
    .services.background.environment.MULTI_TENANT;
    (tostring | ascii_downcase) == "true"
  )
' "$config_file" >/dev/null; then
  die "MULTI_TENANT=true is unsupported here: use approved schema_private and per-tenant migration orchestration, never bare alembic upgrade head"
fi

if grep -Fxq 'indexing_model_server' "$active_services_file"; then
  die "indexing_model_server is active; production-lite permits it only behind a disabled opt-in profile"
fi
if grep -Fxq 'code-interpreter' "$active_services_file"; then
  die "code-interpreter is active; production-lite does not permit this opt-in service"
fi

if [[ "$model_mode" == "local" ]]; then
  grep -Fxq 'inference_model_server' "$active_services_file" || \
    die "local model mode requires inference_model_server to be active"
  jq -e '
    ((.services.api_server.environment.DISABLE_MODEL_SERVER // "false") | tostring | ascii_downcase != "true")
    and ((.services.background.environment.DISABLE_MODEL_SERVER // "false") | tostring | ascii_downcase != "true")
  ' "$config_file" >/dev/null || \
    die "local model mode cannot disable the model server"
else
  if grep -Fxq 'inference_model_server' "$active_services_file"; then
    die "cloud model mode must not activate inference_model_server"
  fi
  jq -e '
    ((.services.api_server.environment.DISABLE_MODEL_SERVER // "") | tostring | ascii_downcase == "true")
    and ((.services.background.environment.DISABLE_MODEL_SERVER // "") | tostring | ascii_downcase == "true")
  ' "$config_file" >/dev/null || \
    die "cloud model mode requires the no-local-models application settings"
fi

local_services=(relational_db elasticsearch cache minio)
if [[ "$infra_mode" == "compose-managed" ]]; then
  for service in "${local_services[@]}"; do
    grep -Fxq "$service" "$active_services_file" || \
      die "compose-managed mode requires the local $service service to be active"
  done
  jq -e '
    def immutable_image:
      . != null
      and .build == null
      and (.image | type == "string")
      and (.image | test("^[^[:space:]@]+@sha256:[0-9a-f]{64}$"))
      and (.image | split("@")[0] | split("/")[-1] | contains(":") | not)
      and ((.pull_policy // "") == "always");
    (.services.relational_db | immutable_image)
    and (.services.elasticsearch | immutable_image)
    and (.services.cache | immutable_image)
    and (.services.minio | immutable_image)
  ' "$config_file" >/dev/null || \
    die "Compose-managed data services must use approved immutable images without host builds"
  jq -e '
    .services.api_server.environment as $runtime
    | .services.relational_db.environment as $admin
    | (($admin.POSTGRES_USER // "") | length > 0)
    and (($admin.POSTGRES_PASSWORD // "") | length > 0)
    and (($admin.POSTGRES_DB // "") | length > 0)
    and ($admin.POSTGRES_USER != $runtime.POSTGRES_USER)
    and (
      [$runtime | keys[] | select(test("^POSTGRES_(ADMIN|DB_ADMIN|MIGRATION)_"))]
      | length == 0
    )
  ' "$config_file" >/dev/null || \
    die "Compose-managed PostgreSQL must isolate admin credentials from the runtime account"
  jq -e '
    .services.api_server.environment as $environment
    | $environment.POSTGRES_HOST == "relational_db"
      and $environment.ELASTICSEARCH_HOST == "elasticsearch"
      and $environment.REDIS_HOST == "cache"
      and ($environment.S3_ENDPOINT_URL | startswith("http://minio:"))
  ' "$config_file" >/dev/null || \
    die "compose-managed mode cannot mix external database, search, cache, or object-store endpoints"
else
  for service in "${local_services[@]}"; do
    if grep -Fxq "$service" "$active_services_file"; then
      die "external mode must not activate the local $service service"
    fi
  done
  jq -e '
    .services.api_server as $api
    | .services.background as $background
    | ($api.environment.POSTGRES_HOST != null and $api.environment.POSTGRES_HOST != "relational_db")
      and ($api.environment.ELASTICSEARCH_HOST != null and $api.environment.ELASTICSEARCH_HOST != "elasticsearch")
      and ($api.environment.REDIS_HOST != null and $api.environment.REDIS_HOST != "cache")
      and (
        (($api.environment.FILE_STORE_BACKEND // "s3") | ascii_downcase) != "s3"
        or (($api.environment.S3_ENDPOINT_URL // "") | startswith("http://minio:") | not)
      )
      and (($api.depends_on // {}) | has("relational_db") | not)
      and (($api.depends_on // {}) | has("elasticsearch") | not)
      and (($api.depends_on // {}) | has("cache") | not)
      and (($api.depends_on // {}) | has("minio") | not)
      and (($background.depends_on // {}) | has("relational_db") | not)
      and (($background.depends_on // {}) | has("elasticsearch") | not)
      and (($background.depends_on // {}) | has("cache") | not)
  ' "$config_file" >/dev/null || \
    die "external mode requires external endpoints and fully reset local-infrastructure dependencies"
fi

if [[ -n "$expected_image" ]]; then
  jq -e --arg expected "$expected_image" \
    '.services.api_server.image == $expected and .services.background.image == $expected' \
    "$config_file" >/dev/null || die "the rendered backend digest does not match --expected-image"
fi
if [[ -n "$expected_web_image" ]]; then
  jq -e --arg expected "$expected_web_image" \
    '.services.web_server.image == $expected' \
    "$config_file" >/dev/null || die "the rendered web digest does not match --expected-web-image"
fi
if [[ "$model_mode" == "local" && -n "$expected_model_image" ]]; then
  jq -e --arg expected "$expected_model_image" \
    '.services.inference_model_server.image == $expected and .services.indexing_model_server.image == $expected' \
    "$profiled_config_file" >/dev/null || die "the rendered model digest does not match --expected-model-image"
fi

background_image=$(jq -er '.services.background.image' "$config_file") || \
  die "the rendered background image cannot be resolved for readiness-file validation"
python3 "$READINESS_SNAPSHOT_HELPER" \
  --attestation "$attestation_path" \
  --evidence "$evidence_path" \
  --snapshot-directory "$snapshot_directory" || \
  die "the readiness capability sources failed secure descriptor validation"

if ! timeout \
  --foreground \
  --kill-after=5s \
  "${READINESS_VALIDATION_TIMEOUT_SECONDS}s" \
  "$DOCKER_BIN" run \
  --cidfile "$readiness_cidfile" \
  --label "$READINESS_OWNERSHIP_LABEL=$readiness_ownership_token" \
  --rm \
  --pull never \
  --network none \
  --read-only \
  --cap-drop ALL \
  --security-opt no-new-privileges \
  --pids-limit 64 \
  --user 1001:1001 \
  --mount "type=bind,source=$snapshot_directory/regulatory-capabilities.json,target=/run/readiness/regulatory-capabilities.json,readonly" \
  --mount "type=bind,source=$snapshot_directory/regulatory-capability-evidence.json,target=/run/readiness/regulatory-capability-evidence.json,readonly" \
  --entrypoint /usr/local/bin/python \
  "$background_image" \
  /app/scripts/regulatory_indexing_readiness.py \
  --validate-capability-snapshots-only \
  --capability-attestation /run/readiness/regulatory-capabilities.json \
  --capability-evidence /run/readiness/regulatory-capability-evidence.json \
  >"$config_error_file" 2>&1; then
  die "the readiness capability snapshot validation timed out or failed"
fi

printf '%s\n' \
  "Regulatory production-lite preflight passed. No managed service or persistent Docker state was changed."
