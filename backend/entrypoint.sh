#!/usr/bin/env bash
set -euo pipefail

if [[ -f /vault/secrets/config ]]; then
  # shellcheck disable=SC1091
  source /vault/secrets/config
fi

if [[ -z "${DATABASE_URL:-}" ]] && [[ -n "${DB_HOST:-}" ]] && [[ -n "${DB_USER:-}" ]] && [[ -n "${DB_NAME:-}" ]]; then
  export DATABASE_URL="postgresql://${DB_USER}:${DB_PASSWORD:-}@${DB_HOST}:${DB_PORT:-5432}/${DB_NAME}"
fi

if [[ -z "${DATABASE_URL:-}" ]]; then
  echo "DATABASE_URL (or DB_HOST/DB_USER/DB_NAME) is required." >&2
  exit 1
fi

wait_for_database() {
  echo "Waiting for database..."
  local attempt=1
  local max_attempts=30
  while (( attempt <= max_attempts )); do
    if (
      cd /app/db
      node <<'NODE'
const {Client} = require('pg');
const client = new Client({connectionString: process.env.DATABASE_URL});
client
  .connect()
  .then(() => client.end())
  .then(() => process.exit(0))
  .catch(() => process.exit(1));
NODE
    ); then
      echo "Database is ready."
      return 0
    fi
    echo "Attempt ${attempt}/${max_attempts}: database not ready, retrying..."
    attempt=$((attempt + 1))
    sleep 2
  done
  echo "Database unreachable." >&2
  exit 1
}

run_migrations() {
  echo "Running database migrations..."
  (
    cd /app/db
    npm run migrate:up
  )
  echo "Migrations complete."
}

wait_for_database
run_migrations

echo "Starting backend..."
exec node /app/backend/dist/index.js
