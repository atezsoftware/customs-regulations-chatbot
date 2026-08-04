#!/bin/sh

set -eu

usage() {
  cat <<'EOF'
Build and publish the regulatory runtime, web, and one-shot importer images.
The CUDA-backed model-server artifact is opt-in and is never built for the
default cloud-model release.

Usage:
  publish-regulatory-images.sh REGISTRY_PREFIX RELEASE_TAG [cloud|local]

Example:
  publish-regulatory-images.sh registry.example.com/team "$GIT_COMMIT_SHA" cloud

Run this on a trusted CI/build runner after authenticating Docker to the registry.
The script prints immutable digest references for the production and importer env files.

Optional environment variables:
  LITE_IMAGE_REPOSITORY      Override REGISTRY_PREFIX/regulatory-backend-lite
  WEB_IMAGE_REPOSITORY       Override REGISTRY_PREFIX/regulatory-web
  MODEL_IMAGE_REPOSITORY     Override REGISTRY_PREFIX/regulatory-model-server
  IMPORTER_IMAGE_REPOSITORY  Override REGISTRY_PREFIX/regulatory-importer
  BASE_IMAGE_REGISTRY        Registry prefix for Dockerfile base-image mirrors
  SOURCE_REVISION            Full Git commit SHA when RELEASE_TAG is not a SHA
  RELEASE_PLATFORM           Release platform (only linux/amd64 is approved)
  SENTRY_AUTH_TOKEN          Optional BuildKit secret for the web build
EOF
}

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
  usage
  exit 0
fi

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
  usage >&2
  exit 64
fi

registry_prefix=${1%/}
release_tag=$2
model_mode=${3:-cloud}
source_revision=${SOURCE_REVISION:-$release_tag}
release_platform=${RELEASE_PLATFORM:-linux/amd64}

case "$model_mode" in
  cloud | local) ;;
  *)
    echo "MODEL_MODE must be cloud or local" >&2
    exit 64
    ;;
esac

case "$registry_prefix" in
  "" | *://* | *[!A-Za-z0-9._:/-]*)
    echo "REGISTRY_PREFIX must be a registry host/path without a URL scheme" >&2
    exit 64
    ;;
esac

case "$release_tag" in
  "" | latest | local | dev | edge | main | master | *[!A-Za-z0-9_.-]*)
    echo "RELEASE_TAG must be an immutable release identifier, such as a Git commit SHA" >&2
    exit 64
    ;;
  [A-Za-z0-9_]*) ;;
  *)
    echo "RELEASE_TAG must start with an alphanumeric character or underscore" >&2
    exit 64
    ;;
esac

if [ "${#release_tag}" -gt 128 ]; then
  echo "RELEASE_TAG exceeds Docker's 128-character tag limit" >&2
  exit 64
fi

case "${#source_revision}" in
  40 | 64) ;;
  *)
    echo "SOURCE_REVISION must be a full 40- or 64-character Git commit SHA" >&2
    exit 64
    ;;
esac
case "$source_revision" in
  *[!0-9a-f]*)
    echo "SOURCE_REVISION must be a lowercase hexadecimal Git commit SHA" >&2
    exit 64
    ;;
esac
if [ "$release_platform" != "linux/amd64" ]; then
  echo "RELEASE_PLATFORM must be linux/amd64 for the approved pinned base manifests" >&2
  exit 64
fi

script_dir=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(CDPATH='' cd -- "$script_dir/../.." && pwd)
backend_dir=$(CDPATH='' cd -- "$script_dir/../../backend" && pwd)
web_dir=$(CDPATH='' cd -- "$script_dir/../../web" && pwd)

command -v git >/dev/null 2>&1 || {
  echo "git is required to verify release provenance" >&2
  exit 1
}
checkout_root=$(git -C "$repo_root" rev-parse --show-toplevel 2>/dev/null) || {
  echo "Release source is not a Git checkout" >&2
  exit 1
}
if [ "$checkout_root" != "$repo_root" ]; then
  echo "Release script must run from this repository's canonical checkout" >&2
  exit 1
fi
checkout_revision=$(git -C "$repo_root" rev-parse HEAD 2>/dev/null) || {
  echo "Could not resolve the checked-out Git revision" >&2
  exit 1
}
if [ "$checkout_revision" != "$source_revision" ]; then
  echo "SOURCE_REVISION does not match the checked-out Git HEAD" >&2
  exit 1
fi
if ! git -C "$repo_root" diff --quiet --no-ext-diff --; then
  echo "Tracked working-tree changes are forbidden in a release build" >&2
  exit 1
fi
if ! git -C "$repo_root" diff --cached --quiet --no-ext-diff --; then
  echo "Staged-but-uncommitted changes are forbidden in a release build" >&2
  exit 1
fi
untracked_files=$(git -C "$repo_root" ls-files --others --exclude-standard)
if [ -n "$untracked_files" ]; then
  echo "Untracked files are forbidden in a release build" >&2
  exit 1
fi

lite_repository=${LITE_IMAGE_REPOSITORY:-$registry_prefix/regulatory-backend-lite}
web_repository=${WEB_IMAGE_REPOSITORY:-$registry_prefix/regulatory-web}
importer_repository=${IMPORTER_IMAGE_REPOSITORY:-$registry_prefix/regulatory-importer}
lite_tagged=$lite_repository:$release_tag
web_tagged=$web_repository:$release_tag
importer_tagged=$importer_repository:$release_tag
if [ "$model_mode" = "local" ]; then
  model_repository=${MODEL_IMAGE_REPOSITORY:-$registry_prefix/regulatory-model-server}
  model_tagged=$model_repository:$release_tag
fi

build_image() {
  dockerfile=$1
  target=$2
  image_ref=$3
  context=$4
  role=$5
  document_import=$6

  set -- docker build \
    --pull \
    --platform "$release_platform" \
    --file "$dockerfile" \
    --target "$target" \
    --build-arg "ONYX_VERSION=$release_tag" \
    --build-arg "SOURCE_REVISION=$source_revision" \
    --label "org.opencontainers.image.version=$release_tag" \
    --label "org.opencontainers.image.revision=$source_revision" \
    --label "io.regulatory.role=$role" \
    --label "io.regulatory.document-import=$document_import"

  if [ -n "${BASE_IMAGE_REGISTRY:-}" ]; then
    set -- "$@" --build-arg "BASE_IMAGE_REGISTRY=$BASE_IMAGE_REGISTRY"
  fi

  set -- "$@" --tag "$image_ref" "$context"
  "$@"
}

build_web_image() {
  set -- docker build \
    --pull \
    --platform "$release_platform" \
    --file "$web_dir/Dockerfile" \
    --target runner \
    --build-arg "ONYX_VERSION=$release_tag" \
    --build-arg "SOURCE_REVISION=$source_revision" \
    --build-arg "SENTRY_RELEASE=$release_tag" \
    --build-arg "NEXT_PUBLIC_DISABLE_ONYX_UPSTREAM_CONNECTIONS=true" \
    --build-arg "NEXT_PUBLIC_DISABLE_LOGOUT=${NEXT_PUBLIC_DISABLE_LOGOUT:-}" \
    --build-arg "NEXT_PUBLIC_FORGOT_PASSWORD_ENABLED=${NEXT_PUBLIC_FORGOT_PASSWORD_ENABLED:-}" \
    --build-arg "NEXT_PUBLIC_THEME=${NEXT_PUBLIC_THEME:-}" \
    --build-arg "NEXT_PUBLIC_DO_NOT_USE_TOGGLE_OFF_DANSWER_POWERED=${NEXT_PUBLIC_DO_NOT_USE_TOGGLE_OFF_DANSWER_POWERED:-false}" \
    --build-arg "NEXT_PUBLIC_CUSTOM_REFRESH_URL=${NEXT_PUBLIC_CUSTOM_REFRESH_URL:-}" \
    --build-arg "NEXT_PUBLIC_POSTHOG_KEY=${NEXT_PUBLIC_POSTHOG_KEY:-}" \
    --build-arg "NEXT_PUBLIC_POSTHOG_HOST=${NEXT_PUBLIC_POSTHOG_HOST:-}" \
    --build-arg "NEXT_PUBLIC_CLOUD_ENABLED=${NEXT_PUBLIC_CLOUD_ENABLED:-}" \
    --build-arg "NEXT_PUBLIC_SENTRY_DSN=${NEXT_PUBLIC_SENTRY_DSN:-}" \
    --build-arg "NEXT_PUBLIC_GTM_ENABLED=${NEXT_PUBLIC_GTM_ENABLED:-}" \
    --build-arg "NEXT_PUBLIC_INCLUDE_ERROR_POPUP_SUPPORT_LINK=${NEXT_PUBLIC_INCLUDE_ERROR_POPUP_SUPPORT_LINK:-}" \
    --build-arg "NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY=${NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY:-}" \
    --build-arg "NEXT_PUBLIC_RECAPTCHA_SITE_KEY=${NEXT_PUBLIC_RECAPTCHA_SITE_KEY:-}" \
    --build-arg "WEB_FRAME_PROTECTION_ENABLED=${WEB_FRAME_PROTECTION_ENABLED:-true}" \
    --build-arg "NODE_OPTIONS=${NODE_OPTIONS:---max-old-space-size=4096}" \
    --label "org.opencontainers.image.version=$release_tag" \
    --label "org.opencontainers.image.revision=$source_revision" \
    --label "io.regulatory.role=web" \
    --label "io.regulatory.document-import=false"

  if [ -n "${BASE_IMAGE_REGISTRY:-}" ]; then
    set -- "$@" --build-arg "BASE_IMAGE_REGISTRY=$BASE_IMAGE_REGISTRY"
  fi
  if [ -n "${SENTRY_AUTH_TOKEN:-}" ]; then
    set -- "$@" --secret id=sentry_auth_token,env=SENTRY_AUTH_TOKEN
  fi

  set -- "$@" --tag "$web_tagged" "$web_dir"
  "$@"
}

resolve_digest_ref() {
  repository=$1
  tagged_ref=$2
  digest_ref=$(docker image inspect \
    --format '{{range .RepoDigests}}{{println .}}{{end}}' \
    "$tagged_ref" | awk -v prefix="$repository@" '
      index($0, prefix) == 1 || index($0, "docker.io/" prefix) == 1 { print; exit }
    ')

  if [ -z "$digest_ref" ]; then
    echo "Could not resolve a pushed digest for $tagged_ref" >&2
    exit 1
  fi

  digest=${digest_ref#*@}
  hash=${digest#sha256:}
  if [ "$hash" = "$digest" ] || [ "${#hash}" -ne 64 ]; then
    echo "Registry returned an invalid sha256 digest for $tagged_ref" >&2
    exit 1
  fi

  case "$hash" in
    *[!0-9a-f]*)
      echo "Registry returned an invalid sha256 digest for $tagged_ref" >&2
      exit 1
      ;;
  esac

  printf '%s\n' "$repository@$digest"
}

echo "Building runtime-lite image: $lite_tagged"
build_image "$backend_dir/Dockerfile.runtime-lite" runtime-lite "$lite_tagged" "$backend_dir" runtime-lite false

echo "Building matching web image: $web_tagged"
build_web_image

if [ "$model_mode" = "local" ]; then
  echo "Building matching model-server image: $model_tagged"
  build_image "$backend_dir/Dockerfile.model_server" final "$model_tagged" "$backend_dir" model-server false
fi

echo "Building one-shot importer image: $importer_tagged"
build_image "$backend_dir/Dockerfile" runtime "$importer_tagged" "$backend_dir" importer true

docker push "$lite_tagged"
docker push "$web_tagged"
if [ "$model_mode" = "local" ]; then
  docker push "$model_tagged"
fi
docker push "$importer_tagged"

lite_digest_ref=$(resolve_digest_ref "$lite_repository" "$lite_tagged")
web_digest_ref=$(resolve_digest_ref "$web_repository" "$web_tagged")
importer_digest_ref=$(resolve_digest_ref "$importer_repository" "$importer_tagged")
if [ "$model_mode" = "local" ]; then
  model_digest_ref=$(resolve_digest_ref "$model_repository" "$model_tagged")
fi

cat <<EOF

Published release $release_tag. Store these immutable references in deployment configuration:
REGULATORY_SOURCE_REVISION=$source_revision
REGULATORY_RELEASE_PLATFORM=$release_platform
REGULATORY_MODEL_MODE=$model_mode
ONYX_BACKEND_LITE_IMAGE=$lite_digest_ref
ONYX_WEB_SERVER_IMAGE=$web_digest_ref
ONYX_IMPORTER_IMAGE=$importer_digest_ref
EOF

if [ "$model_mode" = "local" ]; then
  printf 'ONYX_MODEL_SERVER_IMAGE=%s\n' "$model_digest_ref"
fi
