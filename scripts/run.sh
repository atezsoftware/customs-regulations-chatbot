#!/usr/bin/env bash
# Single entrypoint to bring up any subset of this repo's apps (db,
# core-api, core-indexer, backend, frontend) in a chosen environment
# (dev, test, prod).
#
# Usage:
#   scripts/run.sh --env dev --apps all
#   scripts/run.sh --env dev --apps db,backend
#   scripts/run.sh --env test --apps db
#   scripts/run.sh --env prod --apps backend
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENVIRONMENT="dev"
APPS="all"

usage() {
  cat <<EOF
Usage: $0 --env <dev|test|prod> --apps <all|db,core-api,core-indexer,backend,frontend>

  --env   Environment to run (default: dev)
  --apps  Comma-separated list of apps to start, or "all" (default: all)

Examples:
  $0 --env dev --apps all
  $0 --env dev --apps db,backend
  $0 --env test --apps db
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env) ENVIRONMENT="$2"; shift 2 ;;
    --apps) APPS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

case "$ENVIRONMENT" in
  dev|test|prod) ;;
  *) echo "Invalid --env '$ENVIRONMENT' (expected dev, test or prod)" >&2; exit 1 ;;
esac

if [[ "$APPS" == "all" ]]; then
  APPS="db,core-api,core-indexer,backend,frontend"
fi
IFS=',' read -ra APP_LIST <<< "$APPS"
for app in "${APP_LIST[@]}"; do
  case "$app" in
    db|core-api|core-indexer|backend|frontend) ;;
    *) echo "Invalid app '$app' (expected db, core-api, core-indexer, backend or frontend)" >&2; exit 1 ;;
  esac
done

PIDS=()
MIGRATIONS_RAN=0
cleanup() {
  if [[ ${#PIDS[@]} -gt 0 ]]; then
    echo
    echo "Stopping apps..."
    kill "${PIDS[@]}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

is_local_host() {
  case "${1:-}" in
    ""|localhost|127.0.0.1|::1) return 0 ;;
    *) return 1 ;;
  esac
}

resolved_db_host() {
  if [[ -n "${DB_HOST:-}" ]]; then
    printf '%s\n' "$DB_HOST"
    return
  fi

  if [[ -n "${DATABASE_URL:-}" ]]; then
    local after_scheme="${DATABASE_URL#*://}"
    local after_creds="${after_scheme#*@}"
    printf '%s\n' "${after_creds%%[:/?]*}"
  fi
}

database_is_remote() {
  local host
  host="$(resolved_db_host)"
  [[ -n "$host" ]] && ! is_local_host "$host"
}

load_env() {
  local root_env="$ROOT_DIR/.env.$ENVIRONMENT"
  local db_env="$ROOT_DIR/db/.env.$ENVIRONMENT"

  set -a
  [[ -f "$root_env" ]] && source "$root_env"
  [[ -f "$db_env" ]] && source "$db_env"
  set +a

  export APP_ENV="$ENVIRONMENT"
  if [[ -z "${DATABASE_URL:-}" ]]; then
    if [[ -n "${DB_USER:-}" && -n "${DB_HOST:-}" && -n "${DB_NAME:-}" ]]; then
      export DATABASE_URL="postgresql://${DB_USER}:${DB_PASSWORD:-}@${DB_HOST}:${DB_PORT:-5432}/${DB_NAME}"
    elif [[ -n "${POSTGRES_USER:-}" && -n "${POSTGRES_DB:-}" ]]; then
      export DATABASE_URL="postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD:-}@localhost:${POSTGRES_PORT:-5432}/${POSTGRES_DB}"
    fi
  fi
}

run_migrations() {
  if [[ "$ENVIRONMENT" == "prod" ]]; then
    echo "==> [db:prod] skipping auto-migration (no host port exposed by design)."
    echo "    Run migrations from a host on the same docker network, e.g.:"
    echo "    cd db && DATABASE_URL=postgresql://user:pass@postgres:5432/db npm run migrate:up"
    return
  fi

  if [[ "$MIGRATIONS_RAN" == "1" ]]; then
    return
  fi

  load_env
  if database_is_remote; then
    if [[ -z "${DATABASE_URL:-}" ]]; then
      echo "Remote DB is configured, but DATABASE_URL is empty." >&2
      exit 1
    fi
    echo "==> [db:$ENVIRONMENT] running migrations against remote Postgres ($(resolved_db_host))"
    (
      cd "$ROOT_DIR/db"
      [[ -d node_modules ]] || npm install
      npm run migrate:up
    )
    MIGRATIONS_RAN=1
    return
  fi

  local env_file="$ROOT_DIR/db/.env.$ENVIRONMENT"
  if [[ ! -f "$env_file" ]]; then
    echo "Missing $env_file. Copy db/.env.$ENVIRONMENT.example to $env_file and fill it in." >&2
    exit 1
  fi

  # shellcheck disable=SC1090
  source "$env_file"
  local port="${POSTGRES_PORT:-5432}"
  echo "==> [db:$ENVIRONMENT] running migrations"
  (
    cd "$ROOT_DIR/db"
    [[ -d node_modules ]] || npm install
    DATABASE_URL="postgresql://$POSTGRES_USER:$POSTGRES_PASSWORD@localhost:$port/$POSTGRES_DB" npm run migrate:up
  )
  MIGRATIONS_RAN=1
}

run_db() {
  load_env
  if database_is_remote; then
    echo "==> [db:$ENVIRONMENT] remote Postgres configured ($(resolved_db_host)); skipping local docker postgres."
    run_migrations
    return
  fi

  echo "==> [db:$ENVIRONMENT] starting Postgres via docker compose"
  local env_file="$ROOT_DIR/db/.env.$ENVIRONMENT"
  if [[ ! -f "$env_file" ]]; then
    echo "Missing $env_file. Copy db/.env.$ENVIRONMENT.example to $env_file and fill it in." >&2
    exit 1
  fi
  (
    cd "$ROOT_DIR/db"
    docker compose -f docker-compose.yml -f "docker-compose.$ENVIRONMENT.yml" --env-file ".env.$ENVIRONMENT" up -d
  )

  if [[ "$ENVIRONMENT" == "prod" ]]; then
    run_migrations
    return
  fi

  # shellcheck disable=SC1090
  source "$env_file"
  local port="${POSTGRES_PORT:-5432}"
  echo "==> [db:$ENVIRONMENT] waiting for Postgres to become ready on port $port"
  for _ in $(seq 1 30); do
    if docker compose -f "$ROOT_DIR/db/docker-compose.yml" -f "$ROOT_DIR/db/docker-compose.$ENVIRONMENT.yml" \
        --env-file "$env_file" exec -T postgres pg_isready -U "$POSTGRES_USER" -d "$POSTGRES_DB" >/dev/null 2>&1; then
      break
    fi
    sleep 1
  done

  run_migrations
}

run_core_api() {
  echo "==> [core-api:$ENVIRONMENT] chat/agent API (no Docling/langextract)"
  load_env
  case "$ENVIRONMENT" in
    dev)
      (
        cd "$ROOT_DIR/core"
        command -v uv >/dev/null 2>&1 || { echo "Cannot start core-api: uv is not available." >&2; exit 1; }
        uv run --package fs-explorer-api uvicorn \
          fs_explorer_api.server:app \
          --host 127.0.0.1 \
          --port "${CORE_PORT:-8000}" \
          --reload \
          --reload-dir "$ROOT_DIR/core/api/src" \
          --reload-dir "$ROOT_DIR/core/shared/src"
      ) &
      PIDS+=($!)
      ;;
    test|prod)
      (
        cd "$ROOT_DIR/core"
        docker compose -f docker-compose.yml -f "docker-compose.$ENVIRONMENT.yml" up -d --build core-api
      )
      ;;
  esac
}

run_core_indexer() {
  echo "==> [core-indexer:$ENVIRONMENT] Docling/langextract indexing service"
  load_env
  case "$ENVIRONMENT" in
    dev)
      (
        cd "$ROOT_DIR/core"
        command -v uv >/dev/null 2>&1 || { echo "Cannot start core-indexer: uv is not available." >&2; exit 1; }
        uv run --package fs-explorer-indexer uvicorn \
          fs_explorer_indexer.indexer_server:app \
          --host 127.0.0.1 \
          --port "${CORE_INDEXER_PORT:-8001}" \
          --reload \
          --reload-dir "$ROOT_DIR/core/indexer/src" \
          --reload-dir "$ROOT_DIR/core/shared/src"
      ) &
      PIDS+=($!)
      ;;
    test|prod)
      (
        cd "$ROOT_DIR/core"
        docker compose -f docker-compose.yml -f "docker-compose.$ENVIRONMENT.yml" up -d --build core-indexer
      )
      ;;
  esac
}

run_backend() {
  echo "==> [backend:$ENVIRONMENT] LoopBack 4 API"
  load_env
  case "$ENVIRONMENT" in
    dev|test)
      run_migrations
      (
        cd "$ROOT_DIR/backend"
        [[ -d node_modules ]] || npm install
        case "$ENVIRONMENT" in
          dev) npm run dev ;;
          test) NODE_ENV=test npm run build && NODE_ENV=test npm start ;;
        esac
      ) &
      PIDS+=($!)
      ;;
    prod)
      (
        cd "$ROOT_DIR/backend"
        docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build
      )
      ;;
  esac
}

run_frontend() {
  if [[ ! -f "$ROOT_DIR/frontend/package.json" ]]; then
    echo "==> [frontend] not scaffolded yet — skipping."
    return
  fi
  load_env
  echo "==> [frontend:$ENVIRONMENT] web app"
  (
    cd "$ROOT_DIR/frontend"
    [[ -d node_modules ]] || npm install
    case "$ENVIRONMENT" in
      dev) npm run dev ;;
      test) npm run build ;;
      prod) npm run build && npm run preview -- --host 0.0.0.0 ;;
    esac
  ) &
  PIDS+=($!)
}

for app in "${APP_LIST[@]}"; do
  case "$app" in
    db) run_db ;;
    core-api) run_core_api ;;
    core-indexer) run_core_indexer ;;
    backend) run_backend ;;
    frontend) run_frontend ;;
  esac
done

if [[ ${#PIDS[@]} -gt 0 ]]; then
  echo
  echo "Running: ${APP_LIST[*]} (env: $ENVIRONMENT). Press Ctrl+C to stop."
  wait
fi
