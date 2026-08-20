#!/bin/bash -p

set -Eeuo pipefail
umask 077

readonly TRUSTED_SYSTEM_PATH="/usr/sbin:/usr/bin"
readonly INSTALLER_PATH="/usr/local/sbin/install-regulatory-prod-lite-privileged-bundle"
readonly STAGING_ROOT="/var/lib/onyx/regulatory-prod-lite-staging"
readonly SYSTEM_LIBEXEC_ROOT="/usr/local/libexec"
readonly ONYX_LIBEXEC_ROOT="$SYSTEM_LIBEXEC_ROOT/onyx"
readonly INSTALL_ROOT="/usr/local/libexec/onyx/regulatory-prod-lite"
readonly RELEASES_ROOT="$INSTALL_ROOT/releases"
readonly ENTRYPOINT_PATH="$INSTALL_ROOT/regulatory-prod-lite-preflight"
readonly INSTALL_LOCK="/run/lock/onyx-regulatory-prod-lite-install.lock"
readonly MANIFEST_NAME="REGULATORY_PRIVILEGED_MANIFEST.sha256"
readonly MAXIMUM_BUNDLE_FILE_SIZE=1048576
readonly -a RELEASE_FILES=(
  regulatory-prod-lite-privileged-entrypoint
  regulatory-prod-lite-preflight.sh
  regulatory_readiness_file_snapshot.py
  docker-compose.regulatory-edge.yml
  docker-compose.regulatory-compose-infra.yml
  docker-compose.regulatory-external-infra.yml
  docker-compose.no-local-models.yml
  docker-compose.regulatory-prod-lite.yml
)
readonly -a INSTALLER_ANCESTORS=(
  /
  /usr
  /usr/bin
  /usr/sbin
  /usr/local
  /usr/local/sbin
)
readonly -a STAGING_ANCESTORS=(/ /var /var/lib /var/lib/onyx)
readonly -a INSTALL_ANCESTORS=(/ /usr /usr/local)

export PATH="$TRUSTED_SYSTEM_PATH"
export HOME="/var/empty"
export TMPDIR="/tmp"
export LANG="C"
export LC_ALL="C"
unset BASH_ENV CDPATH ENV GLOBIGNORE LD_LIBRARY_PATH LD_PRELOAD PYTHONHOME \
  PYTHONPATH SUDO_ASKPASS SUDO_ASKPASS_REQUIRE

usage() {
  cat <<'EOF'
Install one reviewed privileged regulatory preflight bundle.

Usage:
  install-regulatory-prod-lite-privileged-bundle \
    --source-dir /var/lib/onyx/regulatory-prod-lite-staging/SHA256 \
    --expected-manifest-sha256 SHA256

This installer must first be provisioned at its fixed root-owned path from an
identity-verified production artifact. It never invokes sudo and refuses a
checkout, symlinked source, mutable source, or caller-selected destination.
EOF
}

die() {
  printf 'Privileged bundle installation failed: %s\n' "$1" >&2
  exit 1
}

validate_directory() {
  local path=$1
  local description=$2
  local owner mode

  [[ ! -L "$path" ]] || die "$description must not be a symlink"
  [[ -d "$path" ]] || die "$description must be a directory"
  owner=$(/usr/bin/stat -c '%u:%g' -- "$path") || \
    die "$description ownership cannot be read"
  mode=$(/usr/bin/stat -c '%a' -- "$path") || \
    die "$description mode cannot be read"
  [[ "$owner" == "0:0" ]] || die "$description must be owned by root:root"
  [[ "$mode" =~ ^[0-7]{3,4}$ ]] || die "$description mode is invalid"
  (( (8#$mode & 8#022) == 0 )) || \
    die "$description must not be group/world writable"
}

ensure_directory() {
  local path=$1
  local description=$2

  if [[ -e "$path" || -L "$path" ]]; then
    validate_directory "$path" "$description"
    return
  fi
  /usr/bin/install -d -o root -g root -m 0755 "$path"
  validate_directory "$path" "$description"
}

validate_file() {
  local path=$1
  local description=$2
  local owner mode size

  [[ ! -L "$path" ]] || die "$description must not be a symlink"
  [[ -f "$path" ]] || die "$description must be a regular file"
  owner=$(/usr/bin/stat -c '%u:%g' -- "$path") || \
    die "$description ownership cannot be read"
  mode=$(/usr/bin/stat -c '%a' -- "$path") || \
    die "$description mode cannot be read"
  size=$(/usr/bin/stat -c '%s' -- "$path") || \
    die "$description size cannot be read"
  [[ "$owner" == "0:0" ]] || die "$description must be owned by root:root"
  [[ "$mode" =~ ^[0-7]{3,4}$ ]] || die "$description mode is invalid"
  (( (8#$mode & 8#022) == 0 )) || \
    die "$description must not be group/world writable"
  ((size <= MAXIMUM_BUNDLE_FILE_SIZE)) || \
    die "$description exceeds the 1 MiB limit"
}

validate_manifest_shape() {
  local manifest=$1
  local index=0
  local hash filename extra

  while read -r hash filename extra; do
    [[ -z "$extra" ]] || die "the privileged manifest is malformed"
    [[ "$hash" =~ ^[0-9a-f]{64}$ ]] || \
      die "the privileged manifest contains an invalid digest"
    ((index < ${#RELEASE_FILES[@]})) || \
      die "the privileged manifest contains unexpected files"
    [[ "$filename" == "${RELEASE_FILES[$index]}" ]] || \
      die "the privileged manifest file set or ordering is invalid"
    ((index += 1))
  done <"$manifest"
  ((index == ${#RELEASE_FILES[@]})) || \
    die "the privileged manifest is incomplete"
}

((EUID == 0)) || die "this installer must be run as root:root"
[[ "$(/usr/bin/id -g)" == "0" ]] || die "this installer must be run as root:root"

installer_canonical=$(/usr/bin/readlink -f -- "${BASH_SOURCE[0]}") || \
  die "the installer cannot be canonicalized"
[[ "$installer_canonical" == "$INSTALLER_PATH" ]] || \
  die "this installer must run only from its fixed root-owned path"
validate_file "$INSTALLER_PATH" "the privileged bundle installer"
[[ -x "$INSTALLER_PATH" ]] || die "the privileged bundle installer must be executable"
for path in "${INSTALLER_ANCESTORS[@]}"; do
  validate_directory "$path" "privileged installer ancestor $path"
done

source_directory=""
expected_manifest_sha256=""
while (($#)); do
  case "$1" in
    --source-dir)
      [[ -n "${2:-}" ]] || die "--source-dir requires a value"
      source_directory=$2
      shift 2
      ;;
    --expected-manifest-sha256)
      [[ -n "${2:-}" ]] || die "--expected-manifest-sha256 requires a value"
      expected_manifest_sha256=$2
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

[[ "$expected_manifest_sha256" =~ ^[0-9a-f]{64}$ ]] || \
  die "--expected-manifest-sha256 must be a lowercase SHA-256"
[[ "$source_directory" == "$STAGING_ROOT/$expected_manifest_sha256" ]] || \
  die "--source-dir must be the digest-named directory below the fixed staging root"

for path in "${STAGING_ANCESTORS[@]}"; do
  validate_directory "$path" "privileged staging ancestor $path"
done
validate_directory "$STAGING_ROOT" "the privileged staging root"
validate_directory "$source_directory" "the privileged staged bundle"
manifest="$source_directory/$MANIFEST_NAME"
validate_file "$manifest" "the privileged staged manifest"
validate_manifest_shape "$manifest"
for filename in "${RELEASE_FILES[@]}"; do
  validate_file "$source_directory/$filename" "staged privileged file $filename"
done

observed_members=$(
  /usr/bin/find "$source_directory" -mindepth 1 -maxdepth 1 -printf '%f\n' \
    | /usr/bin/sort
) || die "the staged privileged bundle cannot be enumerated"
expected_members=$(
  printf '%s\n' "${RELEASE_FILES[@]}" "$MANIFEST_NAME" | /usr/bin/sort
)
[[ "$observed_members" == "$expected_members" ]] || \
  die "the staged privileged bundle contains unexpected or missing members"

actual_manifest_sha256=$(/usr/bin/sha256sum -- "$manifest") || \
  die "the staged privileged manifest digest cannot be calculated"
actual_manifest_sha256=${actual_manifest_sha256%% *}
[[ "$actual_manifest_sha256" == "$expected_manifest_sha256" ]] || \
  die "the staged privileged manifest is not the reviewed manifest"
(
  cd -- "$source_directory"
  /usr/bin/sha256sum --check --strict --status "$MANIFEST_NAME"
) || die "the staged privileged file digest validation failed"

exec 9>"$INSTALL_LOCK"
/usr/bin/flock --exclusive 9 || die "the privileged bundle install lock cannot be acquired"

for path in "${INSTALL_ANCESTORS[@]}"; do
  validate_directory "$path" "privileged install directory $path"
done
ensure_directory "$SYSTEM_LIBEXEC_ROOT" \
  "privileged install directory $SYSTEM_LIBEXEC_ROOT"
ensure_directory "$ONYX_LIBEXEC_ROOT" \
  "privileged install directory $ONYX_LIBEXEC_ROOT"
ensure_directory "$INSTALL_ROOT" "privileged install directory $INSTALL_ROOT"
ensure_directory "$RELEASES_ROOT" "privileged install directory $RELEASES_ROOT"

release_directory="$RELEASES_ROOT/$expected_manifest_sha256"
temporary_release="$RELEASES_ROOT/.install-$expected_manifest_sha256-$$"
temporary_entrypoint="$INSTALL_ROOT/.entrypoint-$expected_manifest_sha256-$$"
temporary_current="$INSTALL_ROOT/.current-$expected_manifest_sha256-$$"
cleanup() {
  local status=$?
  trap - EXIT HUP INT TERM
  /bin/rm -rf -- "$temporary_release"
  /bin/rm -f -- "$temporary_entrypoint" "$temporary_current"
  exit "$status"
}
trap cleanup EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

if [[ ! -e "$release_directory" ]]; then
  /usr/bin/install -d -o root -g root -m 0755 "$temporary_release"
  for filename in "${RELEASE_FILES[@]}"; do
    mode=0644
    case "$filename" in
      regulatory-prod-lite-privileged-entrypoint | regulatory-prod-lite-preflight.sh)
        mode=0755
        ;;
    esac
    /usr/bin/install -o root -g root -m "$mode" \
      "$source_directory/$filename" "$temporary_release/$filename"
    /usr/bin/sync -f "$temporary_release/$filename"
  done
  /usr/bin/install -o root -g root -m 0644 "$manifest" \
    "$temporary_release/$MANIFEST_NAME"
  /usr/bin/sync -f "$temporary_release/$MANIFEST_NAME"
  /usr/bin/sync -f "$temporary_release"
  /bin/mv -- "$temporary_release" "$release_directory"
  /usr/bin/sync -f "$RELEASES_ROOT"
fi

validate_directory "$release_directory" "the installed privileged bundle release"
validate_file "$release_directory/$MANIFEST_NAME" \
  "the installed privileged bundle manifest"
installed_manifest_sha256=$(
  /usr/bin/sha256sum -- "$release_directory/$MANIFEST_NAME"
) || die "the installed privileged manifest digest cannot be calculated"
installed_manifest_sha256=${installed_manifest_sha256%% *}
[[ "$installed_manifest_sha256" == "$expected_manifest_sha256" ]] || \
  die "the installed privileged manifest does not match its release directory"
for filename in "${RELEASE_FILES[@]}"; do
  validate_file "$release_directory/$filename" \
    "installed privileged file $filename"
done
installed_members=$(
  /usr/bin/find "$release_directory" -mindepth 1 -maxdepth 1 -printf '%f\n' \
    | /usr/bin/sort
) || die "the installed privileged bundle cannot be enumerated"
[[ "$installed_members" == "$expected_members" ]] || \
  die "the installed privileged bundle contains unexpected or missing members"
(
  cd -- "$release_directory"
  /usr/bin/sha256sum --check --strict --status "$MANIFEST_NAME"
) || die "the installed privileged file digest validation failed"

/usr/bin/install -o root -g root -m 0755 \
  "$release_directory/regulatory-prod-lite-privileged-entrypoint" \
  "$temporary_entrypoint"
/usr/bin/sync -f "$temporary_entrypoint"
/bin/mv -f -- "$temporary_entrypoint" "$ENTRYPOINT_PATH"
/usr/bin/sync -f "$INSTALL_ROOT"

printf '%s\n' "$expected_manifest_sha256" >"$temporary_current"
/bin/chown root:root "$temporary_current"
/bin/chmod 0644 "$temporary_current"
/usr/bin/sync -f "$temporary_current"
/bin/mv -f -- "$temporary_current" "$INSTALL_ROOT/current"
/usr/bin/sync -f "$INSTALL_ROOT"

trap - EXIT HUP INT TERM
printf 'Installed privileged regulatory preflight bundle: %s\n' \
  "$expected_manifest_sha256"
