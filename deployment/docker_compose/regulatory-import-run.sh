#!/usr/bin/env bash

set -Eeuo pipefail

readonly SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly IMPORTER_COMPOSE="$SCRIPT_DIR/docker-compose.regulatory-importer.yml"

usage() {
  cat <<'EOF'
Run the digest-pinned regulatory importer after verifying its release labels.

Usage:
  regulatory-import-run.sh \
    --env-file PATH \
    --expected-image REPOSITORY@sha256:DIGEST \
    --expected-revision FULL_GIT_SHA \
    -- IMPORTER_ARGUMENTS...

This workstation-only wrapper validates Compose, pulls and inspects the importer image,
then invokes the one-shot importer. It must never be copied to a production host.
EOF
}

die() {
  printf 'Importer refused: %s\n' "$1" >&2
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

env_file="$SCRIPT_DIR/.env.regulatory-importer"
expected_image=""
expected_revision=""

while (($#)); do
  case "$1" in
    --env-file)
      require_value "$1" "${2:-}"
      env_file=$2
      shift 2
      ;;
    --expected-image)
      require_value "$1" "${2:-}"
      expected_image=$2
      shift 2
      ;;
    --expected-revision)
      require_value "$1" "${2:-}"
      expected_revision=$2
      shift 2
      ;;
    --)
      shift
      break
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

[[ -n "$expected_image" ]] || die "--expected-image is required"
[[ -n "$expected_revision" ]] || die "--expected-revision is required"
(($#)) || die "importer arguments are required after --"
is_digest_reference "$expected_image" || \
  die "--expected-image must be an immutable repository@sha256 reference"
[[ "$expected_revision" =~ ^([0-9a-f]{40}|[0-9a-f]{64})$ ]] || \
  die "--expected-revision must be a full lowercase Git commit SHA"

command -v docker >/dev/null 2>&1 || die "docker is not installed"
command -v jq >/dev/null 2>&1 || die "jq is required for fail-closed image validation"
[[ -r "$env_file" && -f "$env_file" ]] || die "the importer env file is not a readable regular file"
[[ -r "$IMPORTER_COMPOSE" && -f "$IMPORTER_COMPOSE" ]] || die "the importer Compose file is unavailable"

env_mode=$(stat -c '%a' "$env_file") || die "the importer env-file permissions cannot be inspected"
(( (8#$env_mode & 8#077) == 0 )) || die "the importer env file must not be accessible by group or others"

compose=(
  docker compose
  --env-file "$env_file"
  -f "$IMPORTER_COMPOSE"
  --profile importer
)

"${compose[@]}" config --quiet || die "the importer Compose configuration is invalid"
rendered_image=$("${compose[@]}" config --format json | jq -er '.services.importer.image') || \
  die "the rendered importer image cannot be resolved"
[[ "$rendered_image" == "$expected_image" ]] || \
  die "the rendered importer digest does not match --expected-image"

"${compose[@]}" pull importer
labels=$(docker image inspect --format '{{json .Config.Labels}}' "$expected_image") || \
  die "the pulled importer image cannot be inspected"
jq -e --arg revision "$expected_revision" '
  .["io.regulatory.role"] == "importer"
  and .["io.regulatory.document-import"] == "true"
  and .["org.opencontainers.image.revision"] == $revision
' <<<"$labels" >/dev/null || \
  die "the importer image role, import capability, or source revision is invalid"

exec "${compose[@]}" run --rm --no-deps importer "$@"
