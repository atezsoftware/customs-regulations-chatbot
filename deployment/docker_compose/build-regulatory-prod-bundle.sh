#!/bin/sh

set -eu

usage() {
  cat <<'EOF'
Create an allowlisted, checksummed regulatory production deployment bundle.

Usage:
  build-regulatory-prod-bundle.sh OUTPUT.tar.gz SOURCE_REVISION BACKEND_DIGEST WEB_DIGEST [MODEL_DIGEST]

SOURCE_REVISION must be the full clean-checkout HEAD. All image references must
be immutable repository@sha256 digests from that revision's release set. The
absence of MODEL_DIGEST creates the approved cloud-model bundle; supplying it
creates a local-model bundle. The bundle intentionally excludes importer
compose/env files, image-publish tooling, application source, Dockerfiles, and
source documents.
EOF
}

die() {
  printf 'Bundle creation failed: %s\n' "$1" >&2
  exit 1
}

is_digest_reference() {
  image_ref=$1
  repository=${image_ref%@*}
  final_component=${repository##*/}

  printf '%s\n' "$image_ref" \
    | grep -Eq '^[^[:space:]@]+@sha256:[0-9a-f]{64}$' \
    && case "$final_component" in *:*) return 1 ;; *) return 0 ;; esac
}

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
  usage
  exit 0
fi
if [ "$#" -lt 4 ] || [ "$#" -gt 5 ]; then
  usage >&2
  exit 64
fi

script_dir=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(CDPATH='' cd -- "$script_dir/../.." && pwd)
output=$1
source_revision=$2
backend_image=$3
web_image=$4
model_image=${5:-}
model_mode=local
if [ "$#" -eq 4 ]; then
  model_image=
  model_mode=cloud
fi

printf '%s\n' "$source_revision" | grep -Eq '^([0-9a-f]{40}|[0-9a-f]{64})$' \
  || die "SOURCE_REVISION must be a full lowercase Git object ID"
is_digest_reference "$backend_image" || die "BACKEND_DIGEST is not an immutable digest reference"
is_digest_reference "$web_image" || die "WEB_DIGEST is not an immutable digest reference"
if [ "$model_mode" = "local" ]; then
  is_digest_reference "$model_image" || die "MODEL_DIGEST is not an immutable digest reference"
fi

case "$output" in
  /*) ;;
  *) output=$(pwd)/$output ;;
esac
checksum="$output.sha256"

[ ! -e "$output" ] || die "refusing to overwrite existing bundle: $output"
[ ! -e "$checksum" ] || die "refusing to overwrite existing checksum: $checksum"
output_parent=$(dirname -- "$output")
[ -d "$output_parent" ] && [ -w "$output_parent" ] \
  || die "bundle output directory must already exist and be writable: $output_parent"

command -v git >/dev/null 2>&1 || die "git is required for provenance validation"
command -v sha256sum >/dev/null 2>&1 || die "sha256sum is required"

cd "$repo_root"
head_revision=$(git rev-parse --verify HEAD) || die "cannot resolve checkout HEAD"
[ "$head_revision" = "$source_revision" ] \
  || die "SOURCE_REVISION does not equal checkout HEAD"
git diff --quiet --no-ext-diff \
  || die "checkout has unstaged tracked changes"
git diff --cached --quiet --no-ext-diff \
  || die "checkout has staged changes"
[ -z "$(git ls-files --others --exclude-standard)" ] \
  || die "checkout has untracked files"

bundle_files='deployment/docker_compose/docker-compose.prod.yml
deployment/docker_compose/docker-compose.regulatory-edge.yml
deployment/docker_compose/docker-compose.regulatory-compose-infra.yml
deployment/docker_compose/docker-compose.regulatory-external-infra.yml
deployment/docker_compose/docker-compose.no-local-models.yml
deployment/docker_compose/docker-compose.regulatory-prod-lite.yml
deployment/docker_compose/env.regulatory-prod.template
deployment/docker_compose/env.db-admin.template
deployment/docker_compose/env.migration.template
deployment/docker_compose/env.web.template
deployment/docker_compose/env.nginx.template
deployment/docker_compose/regulatory-prod-lite-preflight.sh
deployment/docker_compose/regulatory-prod-lite-deploy.sh
deployment/docker_compose/REGULATORY_PRODUCTION_RUNBOOK.md
deployment/data/nginx/app.conf.template
deployment/data/nginx/app.conf.template.no-letsencrypt
deployment/data/nginx/app.conf.template.prod
deployment/data/nginx/mcp.conf.inc.template
deployment/data/nginx/mcp_upstream.conf.inc.template
deployment/data/nginx/run-nginx.sh'

for file in $bundle_files; do
  [ -f "$file" ] || die "allowlisted file is missing: $file"
  git ls-files --error-unmatch -- "$file" >/dev/null 2>&1 \
    || die "allowlisted file is not tracked by the release commit: $file"
done

umask 077
staging_root=$(mktemp -d "${TMPDIR:-/tmp}/regulatory-prod-bundle.XXXXXX") \
  || die "cannot create a private staging directory"
trap 'rm -rf -- "$staging_root"' EXIT HUP INT TERM

for file in $bundle_files; do
  mkdir -p -- "$staging_root/$(dirname -- "$file")"
  cp -p -- "$file" "$staging_root/$file"
done

manifest="$staging_root/deployment/docker_compose/REGULATORY_RELEASE_MANIFEST.txt"
{
  printf 'format=regulatory-production-bundle-v2\n'
  printf 'source_revision=%s\n' "$source_revision"
  printf 'release_platform=linux/amd64\n'
  printf 'model_mode=%s\n' "$model_mode"
  printf 'backend_image=%s\n' "$backend_image"
  printf 'web_image=%s\n' "$web_image"
  if [ "$model_mode" = "local" ]; then
    printf 'model_image=%s\n' "$model_image"
  fi
  printf 'file_sha256:\n'
  for file in $bundle_files; do
    file_hash=$(sha256sum -- "$file" | awk '{print $1}')
    printf '%s  %s\n' "$file_hash" "$file"
  done
} >"$manifest"

tar -czf "$output" -C "$staging_root" deployment
output_name=$(basename -- "$output")
(
  cd "$output_parent"
  sha256sum -- "$output_name" >"$output_name.sha256"
)

printf 'Created allowlisted regulatory production bundle: %s\n' "$output"
printf 'Created bundle checksum: %s\n' "$checksum"
printf '%s\n' \
  'CI must sign/attest both the bundle checksum and release images before promotion.'
