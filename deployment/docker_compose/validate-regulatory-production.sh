#!/bin/sh

set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

printf '%s\n' \
  "validate-regulatory-production.sh is a compatibility alias; use regulatory-prod-lite-preflight.sh." >&2
exec "$script_dir/regulatory-prod-lite-preflight.sh" "$@"
