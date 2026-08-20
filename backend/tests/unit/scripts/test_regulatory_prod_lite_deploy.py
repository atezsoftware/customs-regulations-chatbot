from __future__ import annotations

import fcntl
import hashlib
import json
import os
import pwd
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[4]
_COMPOSE_ROOT = _REPO_ROOT / "deployment" / "docker_compose"
_PREFLIGHT = _COMPOSE_ROOT / "regulatory-prod-lite-preflight.sh"
_DEPLOY = _COMPOSE_ROOT / "regulatory-prod-lite-deploy.sh"
_SNAPSHOT_HELPER = _COMPOSE_ROOT / "regulatory_readiness_file_snapshot.py"
_PRIVILEGED_ENTRYPOINT = _COMPOSE_ROOT / "regulatory-prod-lite-privileged-entrypoint"
_PRIVILEGED_INSTALLER = (
    _COMPOSE_ROOT / "install-regulatory-prod-lite-privileged-bundle.sh"
)
_PRIVILEGED_MANIFEST = _COMPOSE_ROOT / "REGULATORY_PRIVILEGED_MANIFEST.sha256"
_RUNBOOK = _COMPOSE_ROOT / "REGULATORY_PRODUCTION_RUNBOOK.md"
_PRIVILEGED_RELEASE_FILES = (
    "regulatory-prod-lite-privileged-entrypoint",
    "regulatory-prod-lite-preflight.sh",
    "regulatory_readiness_file_snapshot.py",
    "docker-compose.regulatory-edge.yml",
    "docker-compose.regulatory-compose-infra.yml",
    "docker-compose.regulatory-external-infra.yml",
    "docker-compose.no-local-models.yml",
    "docker-compose.regulatory-prod-lite.yml",
)
_IMAGE = "registry.example.com/team/regulatory-backend-lite@sha256:" + "a" * 64
_WEB_IMAGE = "registry.example.com/team/regulatory-web@sha256:" + "b" * 64
_MODEL_IMAGE = "registry.example.com/team/regulatory-model@sha256:" + "c" * 64
_POSTGRES_IMAGE = "registry.example.com/mirror/postgres@sha256:" + "1" * 64
_ELASTICSEARCH_IMAGE = "registry.example.com/mirror/elasticsearch@sha256:" + "2" * 64
_REDIS_IMAGE = "registry.example.com/mirror/redis@sha256:" + "3" * 64
_MINIO_IMAGE = "registry.example.com/mirror/minio@sha256:" + "4" * 64
_NGINX_IMAGE = "registry.example.com/mirror/nginx@sha256:" + "5" * 64
_CERTBOT_IMAGE = "registry.example.com/mirror/certbot@sha256:" + "6" * 64
_MANAGED_SERVICES = (
    "api_server",
    "background",
    "web_server",
    "inference_model_server",
    "nginx",
    "certbot",
    "relational_db",
    "elasticsearch",
    "cache",
    "minio",
)
_ATTESTATION_TARGET = "/run/readiness/regulatory-capabilities.json"
_EVIDENCE_TARGET = "/run/readiness/regulatory-capability-evidence.json"


def _config(
    *,
    build: object | None = None,
    extra_service: str | None = None,
    multi_tenant: bool = False,
    indexing_dependency: bool = False,
    external_infra: bool = False,
    cloud_models: bool = False,
    api_command: list[str] | None = None,
) -> str:
    environment = {
        "DOCUMENT_IMPORT_ENABLED": "false",
        "MULTI_TENANT": str(multi_tenant).lower(),
        "POSTGRES_HOST": "db.prod.internal" if external_infra else "relational_db",
        "ELASTICSEARCH_HOST": "search.prod.internal"
        if external_infra
        else "elasticsearch",
        "REDIS_HOST": "redis.prod.internal" if external_infra else "cache",
        "S3_ENDPOINT_URL": (
            "https://objects.prod.internal" if external_infra else "http://minio:9000"
        ),
        "DISABLE_MODEL_SERVER": str(cloud_models).lower(),
        "ENABLE_PAID_ENTERPRISE_EDITION_FEATURES": "true",
        "LICENSE_ENFORCEMENT_ENABLED": "false",
        "POSTGRES_USER": "onyx_runtime",
        "POSTGRES_PASSWORD": "runtime-secret",
        "POSTGRES_PORT": "5432",
        "POSTGRES_DB": "postgres",
    }
    backend: dict[str, object] = {
        "image": _IMAGE,
        "environment": environment,
        "pull_policy": "always",
    }
    api = {
        **backend,
        "user": "1001:1001",
        "command": api_command
        or ["uvicorn", "onyx.main:app", "--host", "0.0.0.0", "--port", "8080"],
    }
    web: dict[str, object] = {
        "image": _WEB_IMAGE,
        "pull_policy": "always",
        "environment": {"INTERNAL_URL": "http://api_server:8080"},
    }
    if build is not None:
        backend["build"] = build
        web["build"] = build
    services: dict[str, object] = {
        "api_server": api,
        "background": dict(backend),
        "web_server": web,
        "inference_model_server": {
            "image": _MODEL_IMAGE,
            "pull_policy": "always",
            **({"profiles": ["local-models"]} if cloud_models else {}),
        },
        "indexing_model_server": {
            "image": _MODEL_IMAGE,
            "pull_policy": "always",
            "profiles": ["local-models" if cloud_models else "indexing-model-server"],
        },
        "relational_db": {
            "image": _POSTGRES_IMAGE,
            "pull_policy": "always",
            "environment": {
                "POSTGRES_USER": "onyx_db_admin",
                "POSTGRES_PASSWORD": "db-admin-secret",
                "POSTGRES_DB": "postgres",
            },
        },
        "elasticsearch": {"image": _ELASTICSEARCH_IMAGE, "pull_policy": "always"},
        "cache": {"image": _REDIS_IMAGE, "pull_policy": "always"},
        "minio": {"image": _MINIO_IMAGE, "pull_policy": "always"},
        "nginx": {"image": _NGINX_IMAGE, "pull_policy": "always"},
        "certbot": {"image": _CERTBOT_IMAGE, "pull_policy": "always"},
        "regulatory_migration": {
            "image": _IMAGE,
            "environment": {
                **environment,
                "POSTGRES_USER": "onyx_migrator",
                "POSTGRES_PASSWORD": "migration-secret",
            },
            "command": ["alembic", "upgrade", "head"],
            "profiles": ["regulatory-migration"],
            "pull_policy": "always",
        },
    }
    if indexing_dependency:
        services["background"] = {
            **dict(backend),
            "depends_on": {"indexing_model_server": {"condition": "service_started"}},
        }
        services["indexing_model_server"] = {
            "image": _MODEL_IMAGE,
            "pull_policy": "always",
            "profiles": ["indexing-model-server"],
        }
    if extra_service is not None:
        services[extra_service] = {"image": "example.invalid/forbidden:latest"}
    background = services["background"]
    assert isinstance(background, dict)
    background["volumes"] = [
        {
            "type": "bind",
            "source": "/tmp/regulatory-capabilities.json",
            "target": _ATTESTATION_TARGET,
            "read_only": True,
        },
        {
            "type": "bind",
            "source": "/tmp/regulatory-capability-evidence.json",
            "target": _EVIDENCE_TARGET,
            "read_only": True,
        },
    ]
    return json.dumps({"services": services})


def _fake_docker(
    tmp_path: Path,
    *,
    config: str | None = None,
    version: str = "2.40.3",
    active_services: tuple[str, ...] = _MANAGED_SERVICES,
    backend_role: str = "runtime-lite",
    fail_match: str = "",
    running_inventory: str = "",
    backend_revision: str = "d" * 40,
    web_revision: str = "d" * 40,
    model_revision: str = "d" * 40,
    attestation_owner: str = "1001",
    attestation_mode: int = 0o600,
    evidence_owner: str = "1001",
    evidence_mode: int = 0o400,
    fake_timeout_exit: str = "",
    use_real_snapshot_helper: bool = False,
    use_real_timeout: bool = False,
    real_runtime_image: str = "",
    real_runtime_use_setpriv_nnp: bool = False,
) -> tuple[dict[str, str], Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log_path = tmp_path / "docker.log"
    environment_log_path = tmp_path / "docker-environment.log"
    sudo_log_path = tmp_path / "sudo.log"
    timeout_log = tmp_path / "timeout.log"
    readiness_state_path = tmp_path / "readiness-container.state"
    readiness_label_path = tmp_path / "readiness-container.label"
    readiness_cid = "9" * 64
    config_path = tmp_path / "config.json"
    evidence_path = tmp_path / "regulatory-capability-evidence.json"
    evidence_bytes = b'{"approved":true}\n'
    evidence_path.write_bytes(evidence_bytes)
    evidence_path.chmod(evidence_mode)
    evidence_digest = hashlib.sha256(evidence_bytes).hexdigest()
    attestation_path = tmp_path / "regulatory-capabilities.json"
    attestation_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "evidence_reference": (f"archive://TASK-8#sha256={evidence_digest}"),
                "evidence_sha256": evidence_digest,
            }
        ),
        encoding="utf-8",
    )
    attestation_path.chmod(attestation_mode)
    rendered_config = json.loads(config or _config())
    background = rendered_config["services"]["background"]
    for volume in background.get("volumes", []):
        if volume.get("target") == _ATTESTATION_TARGET:
            volume["source"] = str(attestation_path)
        if volume.get("target") == _EVIDENCE_TARGET:
            volume["source"] = str(evidence_path)
    config_path.write_text(json.dumps(rendered_config), encoding="utf-8")
    services_path = tmp_path / "services.txt"
    services_path.write_text("\n".join(active_services) + "\n", encoding="utf-8")
    inventory_path = tmp_path / "inventory.txt"
    inventory_path.write_text(running_inventory, encoding="utf-8")
    base_path = tmp_path / "docker-compose.prod.yml"
    base_path.write_text("services: {}\n", encoding="utf-8")
    fake_environment_path = tmp_path / "fake-environment"
    fake_environment_defaults = {
        "FAKE_COMPOSE_CONFIG": str(config_path),
        "FAKE_COMPOSE_SERVICES": str(services_path),
        "FAKE_COMPOSE_VERSION": version,
        "FAKE_BACKEND_ROLE": backend_role,
        "FAKE_BASE_COMPOSE": str(base_path),
        "FAKE_MIGRATION_ENV": str(tmp_path / ".env.migration"),
        "FAKE_DB_ADMIN_ENV": str(tmp_path / ".env.db-admin"),
        "FAKE_DOCKER_FAIL_MATCH": fail_match,
        "FAKE_DOCKER_INVENTORY": str(inventory_path),
        "FAKE_DOCKER_LOG": str(log_path),
        "FAKE_BACKEND_REVISION": backend_revision,
        "FAKE_WEB_REVISION": web_revision,
        "FAKE_MODEL_REVISION": model_revision,
        "FAKE_ATTESTATION_PATH": str(attestation_path),
        "FAKE_EVIDENCE_PATH": str(evidence_path),
        "FAKE_ATTESTATION_OWNER": attestation_owner,
        "FAKE_ATTESTATION_MODE": f"{attestation_mode:o}",
        "FAKE_EVIDENCE_OWNER": evidence_owner,
        "FAKE_EVIDENCE_MODE": f"{evidence_mode:o}",
        "FAKE_TIMEOUT_EXIT": fake_timeout_exit,
        "FAKE_TIMEOUT_LOG": str(timeout_log),
        "FAKE_READINESS_CID": readiness_cid,
        "FAKE_READINESS_STATE": str(readiness_state_path),
        "FAKE_READINESS_LABEL": str(readiness_label_path),
        "FAKE_CLEANUP_LABEL_OVERRIDE": "",
        "FAKE_DOCKER_RM_FAIL": "false",
        "FAKE_READINESS_PERSISTS": "false",
        "FAKE_PRE_CID_FAILURE": "false",
        "FAKE_INVALID_CIDFILE": "false",
        "FAKE_LABEL_QUERY_RESULT": "",
        "FAKE_REAL_RUNTIME_IMAGE": real_runtime_image,
        "FAKE_REAL_RUNTIME_USE_SETPRIV_NNP": str(real_runtime_use_setpriv_nnp).lower(),
        "FAKE_REAL_RUNTIME_FORCE_TIMEOUT": "false",
        "FAKE_REAL_RUNTIME_PRE_CID_FAILURE": "false",
        "FAKE_REAL_DOCKER": "/usr/bin/docker",
        "FAKE_OUTER_TMPDIR_SOURCE": "",
        "_IMAGE": _IMAGE,
        "_MODEL_IMAGE": _MODEL_IMAGE,
        "_WEB_IMAGE": _WEB_IMAGE,
    }
    fake_environment_path.write_text(
        "".join(
            f"if [[ -z ${{{key}+x}} ]]; then export {key}={shlex.quote(value)}; fi\n"
            for key, value in fake_environment_defaults.items()
        ),
        encoding="utf-8",
    )
    fake_environment_path.chmod(0o600)
    docker = bin_dir / "docker"
    docker.write_text(
        """#!/usr/bin/env bash
set -eu
source "__FAKE_ENVIRONMENT__"
printf '%s\\n' "$*" >>"$FAKE_DOCKER_LOG"
printf 'DOCKER_HOST=%s DOCKER_CONTEXT=%s ONYX_BACKEND_LITE_IMAGE=%s AMBIENT_MARKER=%s PATH=%s HOME=%s DOCKER_CONFIG=%s BASH_ENV=%s ENV=%s PYTHONPATH=%s SUDO_ASKPASS=%s\\n' \
  "${DOCKER_HOST-<unset>}" \
  "${DOCKER_CONTEXT-<unset>}" \
  "${ONYX_BACKEND_LITE_IMAGE-<unset>}" \
  "${REGULATORY_AMBIENT_MARKER-<unset>}" \
  "${PATH-<unset>}" \
  "${HOME-<unset>}" \
  "${DOCKER_CONFIG-<unset>}" \
  "${BASH_ENV-<unset>}" \
  "${ENV-<unset>}" \
  "${PYTHONPATH-<unset>}" \
  "${SUDO_ASKPASS-<unset>}" \
  >>"__FAKE_DOCKER_ENVIRONMENT_LOG__"
if [[ "${1:-} ${2:-} ${3:-}" == "compose version --short" ]]; then
  printf '%s\\n' "$FAKE_COMPOSE_VERSION"
  exit 0
fi
if [[ "${1:-}" == "ps" ]]; then
  command cat "$FAKE_DOCKER_INVENTORY"
  exit 0
fi
if [[ "${1:-} ${2:-}" == "container ls" ]]; then
  if [[ -n "$FAKE_REAL_RUNTIME_IMAGE" ]]; then
    exec "$FAKE_REAL_DOCKER" "$@"
  fi
  if [[ " $* " == *" --filter label=io.regulatory.readiness-preflight-owner="* && \
        -n "$FAKE_LABEL_QUERY_RESULT" ]]; then
    printf '%s\\n' "$FAKE_LABEL_QUERY_RESULT"
    exit 0
  fi
  if [[ -f "$FAKE_READINESS_STATE" ]]; then
    printf '%s\\n' "$FAKE_READINESS_CID"
  fi
  exit 0
fi
if [[ "${1:-}" == "inspect" && " $* " == *" --type container "* ]]; then
  if [[ -n "$FAKE_REAL_RUNTIME_IMAGE" ]]; then
    exec "$FAKE_REAL_DOCKER" "$@"
  fi
  [[ -f "$FAKE_READINESS_STATE" ]] || exit 1
  if [[ -n "$FAKE_CLEANUP_LABEL_OVERRIDE" ]]; then
    printf '%s\\n' "$FAKE_CLEANUP_LABEL_OVERRIDE"
  else
    command cat "$FAKE_READINESS_LABEL"
  fi
  exit 0
fi
if [[ "${1:-}" == "inspect" ]]; then
  printf '%s\\n' 'sha256:7777777777777777777777777777777777777777777777777777777777777777'
  exit 0
fi
if [[ "${1:-}" == "rm" ]]; then
  if [[ -n "$FAKE_REAL_RUNTIME_IMAGE" ]]; then
    exec "$FAKE_REAL_DOCKER" "$@"
  fi
  if [[ "$FAKE_DOCKER_RM_FAIL" == "true" ]]; then
    exit 55
  fi
  if [[ "${3:-}" == "$FAKE_READINESS_CID" ]]; then
    command rm -f -- "$FAKE_READINESS_STATE" "$FAKE_READINESS_LABEL"
  fi
  exit 0
fi
if [[ "${1:-}" == "run" && " $* " == *" --validate-capability-snapshots-only "* ]]; then
  if [[ -n "$FAKE_REAL_RUNTIME_IMAGE" ]]; then
    arguments=("$@")
    if [[ -n "$FAKE_OUTER_TMPDIR_SOURCE" ]]; then
      for index in "${!arguments[@]}"; do
        case "${arguments[$index]}" in
          type=bind,source=/tmp/*)
            source_and_rest=${arguments[$index]#type=bind,source=/tmp/}
            arguments[$index]="type=bind,source=$FAKE_OUTER_TMPDIR_SOURCE/$source_and_rest"
            ;;
        esac
      done
    fi
    if [[ "$FAKE_REAL_RUNTIME_PRE_CID_FAILURE" == "true" ]]; then
      ownership_token=""
      for argument in "${arguments[@]}"; do
        case "$argument" in
          io.regulatory.readiness-preflight-owner=*)
            ownership_token=${argument#*=}
            ;;
        esac
      done
      [[ "$ownership_token" =~ ^[0-9a-f]{64}$ ]] || exit 57
      "$FAKE_REAL_DOCKER" create \
        --label "io.regulatory.readiness-preflight-owner=$ownership_token" \
        --user 1001:1001 \
        --entrypoint /bin/sleep \
        "$FAKE_REAL_RUNTIME_IMAGE" 60 >/dev/null
      exit 42
    fi
    for index in "${!arguments[@]}"; do
      if [[ "${arguments[$index]}" == "$_IMAGE" ]]; then
        arguments[$index]="$FAKE_REAL_RUNTIME_IMAGE"
      fi
    done
    if [[ "$FAKE_REAL_RUNTIME_FORCE_TIMEOUT" == "true" ]]; then
      adapted=()
      skip_next=false
      for index in "${!arguments[@]}"; do
        if [[ "$skip_next" == "true" ]]; then
          skip_next=false
          continue
        fi
        if [[ "${arguments[$index]}" == "--security-opt" && \
              "${arguments[$((index + 1))]:-}" == "no-new-privileges" && \
              "$FAKE_REAL_RUNTIME_USE_SETPRIV_NNP" == "true" ]]; then
          skip_next=true
          continue
        fi
        if [[ "${arguments[$index]}" == "--entrypoint" ]]; then
          adapted+=("--entrypoint")
          if [[ "$FAKE_REAL_RUNTIME_USE_SETPRIV_NNP" == "true" ]]; then
            adapted+=("/usr/bin/setpriv")
          else
            adapted+=("/bin/sh")
          fi
          skip_next=true
          continue
        fi
        adapted+=("${arguments[$index]}")
        if [[ "${arguments[$index]}" == "$FAKE_REAL_RUNTIME_IMAGE" ]]; then
          if [[ "$FAKE_REAL_RUNTIME_USE_SETPRIV_NNP" == "true" ]]; then
            adapted+=("--no-new-privs" "/bin/sh" "-c" 'trap "" TERM; sleep 60')
          else
            adapted+=("-c" 'trap "" TERM; sleep 60')
          fi
          break
        fi
      done
      exec "$FAKE_REAL_DOCKER" "${adapted[@]}"
    fi
    if [[ "$FAKE_REAL_RUNTIME_USE_SETPRIV_NNP" == "true" ]]; then
      adapted=()
      skip_next=false
      for index in "${!arguments[@]}"; do
        if [[ "$skip_next" == "true" ]]; then
          skip_next=false
          continue
        fi
        if [[ "${arguments[$index]}" == "--security-opt" && \
              "${arguments[$((index + 1))]:-}" == "no-new-privileges" ]]; then
          skip_next=true
          continue
        fi
        if [[ "${arguments[$index]}" == "/usr/local/bin/python" && \
              "${arguments[$((index - 1))]:-}" == "--entrypoint" ]]; then
          adapted+=("/usr/bin/setpriv")
          continue
        fi
        adapted+=("${arguments[$index]}")
        if [[ "${arguments[$index]}" == "$FAKE_REAL_RUNTIME_IMAGE" ]]; then
          adapted+=("--no-new-privs" "/usr/local/bin/python")
        fi
      done
      arguments=("${adapted[@]}")
    fi
    exec "$FAKE_REAL_DOCKER" "${arguments[@]}"
  fi
  arguments=("$@")
  cidfile=""
  ownership_token=""
  for index in "${!arguments[@]}"; do
    case "${arguments[$index]}" in
      --cidfile) cidfile=${arguments[$((index + 1))]:-} ;;
      io.regulatory.readiness-preflight-owner=*)
        ownership_token=${arguments[$index]#*=}
        ;;
    esac
  done
  [[ -n "$cidfile" && -n "$ownership_token" ]] || exit 56
  printf '%s\\n' "$ownership_token" >"$FAKE_READINESS_LABEL"
  : >"$FAKE_READINESS_STATE"
  if [[ "$FAKE_PRE_CID_FAILURE" == "true" ]]; then
    exit 42
  fi
  if [[ "$FAKE_INVALID_CIDFILE" == "true" ]]; then
    printf '%s\\n' 'invalid-private-cid' >"$cidfile"
  else
    printf '%s\\n' "$FAKE_READINESS_CID" >"$cidfile"
  fi
  if [[ -z "$FAKE_TIMEOUT_EXIT" && "$FAKE_READINESS_PERSISTS" != "true" ]]; then
    command rm -f -- "$FAKE_READINESS_STATE" "$FAKE_READINESS_LABEL"
  fi
  exit 0
fi
if [[ " $* " == *" config --services "* ]]; then
  command cat "$FAKE_COMPOSE_SERVICES"
  exit 0
fi
for argument in "$@"; do
  if [[ "$argument" == "config" ]]; then
    command cat "$FAKE_COMPOSE_CONFIG"
    exit 0
  fi
done
if [[ "${1:-} ${2:-}" == "image inspect" ]]; then
  role="$FAKE_BACKEND_ROLE"
  revision="$FAKE_BACKEND_REVISION"
  case " $* " in
    *" $_WEB_IMAGE "*) role=web; revision="$FAKE_WEB_REVISION" ;;
    *" $_MODEL_IMAGE "*) role=model-server; revision="$FAKE_MODEL_REVISION" ;;
  esac
  if [[ " $* " == *" io.regulatory.role "* ]]; then
    printf '%s\\n' "$role"
  else
    printf '{"io.regulatory.role":"%s","io.regulatory.document-import":"false","org.opencontainers.image.revision":"%s"}\\n' "$role" "$revision"
  fi
  exit 0
fi
if [[ -n "$FAKE_DOCKER_FAIL_MATCH" && " $* " == *"$FAKE_DOCKER_FAIL_MATCH"* ]]; then
  exit 42
fi
exit 0
""".replace("__FAKE_ENVIRONMENT__", str(fake_environment_path)).replace(
            "__FAKE_DOCKER_ENVIRONMENT_LOG__", str(environment_log_path)
        ),
        encoding="utf-8",
    )
    docker.chmod(0o755)
    fake_compose = bin_dir / "docker-compose"
    fake_compose.write_text(
        f'#!/bin/bash\nexec {shlex.quote(str(docker))} compose "$@"\n',
        encoding="utf-8",
    )
    fake_compose.chmod(0o755)
    fake_sudo = bin_dir / "sudo"
    fake_sudo.write_text(
        """#!/usr/bin/env bash
set -eu
printf '%s\\n' "$*" >>"__FAKE_SUDO_LOG__"
if [[ "${1:-}" != "-n" ]]; then
  exit 88
fi
shift
if [[ "${1:-}" == "--" ]]; then
  shift
fi
if [[ "${FAKE_SUDO_AUTHORIZATION_FAIL:-false}" == "true" ]]; then
  exit 89
fi
if ((EUID == 0)); then
  exec /usr/bin/env -i PATH=/usr/bin:/bin HOME=/root LANG=C LC_ALL=C "$@"
fi
exec /usr/bin/unshare --user --map-root-user \
  /usr/bin/env -i PATH=/usr/bin:/bin HOME=/root LANG=C LC_ALL=C "$@"
""".replace("__FAKE_SUDO_LOG__", str(sudo_log_path)),
        encoding="utf-8",
    )
    fake_sudo.chmod(0o755)
    if not use_real_snapshot_helper:
        fake_python = bin_dir / "python3"
        fake_python.write_text(
            """#!/usr/bin/env bash
set -eu
source "__FAKE_ENVIRONMENT__"
isolated_args=()
while [[ "${1:-}" == "-I" || "${1:-}" == "-S" ]]; do
  isolated_args+=("$1")
  shift
done
if [[ "${1:-}" == *"regulatory_readiness_file_snapshot.py" ]]; then
  shift
  attestation=""
  evidence=""
  snapshot_directory=""
  while (($#)); do
    case "$1" in
      --attestation) attestation=$2; shift 2 ;;
      --evidence) evidence=$2; shift 2 ;;
      --snapshot-directory) snapshot_directory=$2; shift 2 ;;
      *) exit 2 ;;
    esac
  done
  if [[ "$FAKE_ATTESTATION_OWNER" != "1001" || "$FAKE_ATTESTATION_MODE" != "600" ]]; then
    exit 1
  fi
  if [[ "$FAKE_EVIDENCE_OWNER" != "1001" || "$FAKE_EVIDENCE_MODE" != "400" ]]; then
    exit 1
  fi
  command cp -- "$attestation" "$snapshot_directory/regulatory-capabilities.json"
  command cp -- "$evidence" "$snapshot_directory/regulatory-capability-evidence.json"
  command chmod 0600 \
    "$snapshot_directory/regulatory-capabilities.json" \
    "$snapshot_directory/regulatory-capability-evidence.json"
  exit 0
fi
exec /usr/bin/python3 "${isolated_args[@]}" "$@"
""".replace("__FAKE_ENVIRONMENT__", str(fake_environment_path)),
            encoding="utf-8",
        )
        fake_python.chmod(0o755)
    if not use_real_timeout:
        fake_timeout = bin_dir / "timeout"
        fake_timeout.write_text(
            """#!/usr/bin/env bash
set -eu
source "__FAKE_ENVIRONMENT__"
printf '%s\n' "$*" >>"$FAKE_TIMEOUT_LOG"
while (($#)); do
  case "$1" in
    --foreground | --kill-after=*) shift ;;
    *s) shift; break ;;
    *) exit 2 ;;
  esac
done
if [[ -n "$FAKE_TIMEOUT_EXIT" && "${1##*/} ${2:-}" == "docker run" ]]; then
  "$@"
  exit "$FAKE_TIMEOUT_EXIT"
fi
exec "$@"
""".replace("__FAKE_ENVIRONMENT__", str(fake_environment_path)),
            encoding="utf-8",
        )
        fake_timeout.chmod(0o755)
    fake_stat = bin_dir / "stat"
    fake_stat.write_text(
        """#!/usr/bin/env bash
set -eu
source "__FAKE_ENVIRONMENT__"
last=${!#}
if [[ "$last" == "$FAKE_ATTESTATION_PATH" && ( "${2:-}" == "%u" || "${2:-}" == "%g" ) ]]; then
  printf '%s\\n' "$FAKE_ATTESTATION_OWNER"
  exit 0
fi
if [[ "$last" == "$FAKE_EVIDENCE_PATH" && ( "${2:-}" == "%u" || "${2:-}" == "%g" ) ]]; then
  printf '%s\\n' "$FAKE_EVIDENCE_OWNER"
  exit 0
fi
exec /usr/bin/stat "$@"
""".replace("__FAKE_ENVIRONMENT__", str(fake_environment_path)),
        encoding="utf-8",
    )
    fake_stat.chmod(0o755)
    env_file = tmp_path / ".env"
    env_file.write_text("# fake production environment\n", encoding="utf-8")
    env_file.chmod(0o600)
    migration_env_file = tmp_path / ".env.migration"
    migration_env_file.write_text(
        "POSTGRES_USER=onyx_migrator\nPOSTGRES_PASSWORD=migration-secret\n",
        encoding="utf-8",
    )
    migration_env_file.chmod(0o600)
    db_admin_env_file = tmp_path / ".env.db-admin"
    db_admin_env_file.write_text(
        "POSTGRES_USER=onyx_db_admin\n"
        "POSTGRES_PASSWORD=db-admin-secret\n"
        "POSTGRES_DB=postgres\n",
        encoding="utf-8",
    )
    db_admin_env_file.chmod(0o600)
    web_env_file = tmp_path / ".env.web"
    web_env_file.write_text(
        "INTERNAL_URL=http://api_server:8080\n"
        "DISABLE_ONYX_UPSTREAM_CONNECTIONS=true\n"
        "NEXT_PUBLIC_DISABLE_ONYX_UPSTREAM_CONNECTIONS=true\n",
        encoding="utf-8",
    )
    web_env_file.chmod(0o600)
    trusted_docker_config = tmp_path / "trusted-docker-config"
    trusted_docker_config.mkdir(mode=0o700)
    trusted_docker_config_file = trusted_docker_config / "config.json"
    trusted_docker_config_file.write_text("{}\n", encoding="utf-8")
    trusted_docker_config_file.chmod(0o600)

    # Production paths are compile-time constants, not environment overrides. Unit
    # tests exercise an isolated copy whose constants point at fixture-owned tools.
    test_compose_root = tmp_path / "docker-compose-bundle"
    shutil.copytree(_COMPOSE_ROOT, test_compose_root)
    installed_bundle_root = tmp_path / "installed-regulatory-bundle"
    script_replacements = {
        'readonly TRUSTED_SYSTEM_PATH="/usr/sbin:/usr/bin"': (
            "readonly TRUSTED_SYSTEM_PATH="
            + shlex.quote(f"{bin_dir}:/usr/sbin:/usr/bin:/sbin:/bin")
        ),
        'readonly DOCKER_BIN="/usr/bin/docker"': (
            f"readonly DOCKER_BIN={shlex.quote(str(docker))}"
        ),
        'readonly COMPOSE_BIN="/usr/libexec/docker/cli-plugins/docker-compose"': (
            f"readonly COMPOSE_BIN={shlex.quote(str(fake_compose))}"
        ),
        'readonly SUDO_BIN="/usr/bin/sudo"': (
            f"readonly SUDO_BIN={shlex.quote(str(fake_sudo))}"
        ),
        'readonly DOCKER_CONFIG_DIR="/etc/onyx/regulatory-docker"': (
            f"readonly DOCKER_CONFIG_DIR={shlex.quote(str(trusted_docker_config))}"
        ),
        'readonly PYTHON_BIN="/usr/bin/python3"': (
            f"readonly PYTHON_BIN={shlex.quote(str(bin_dir / 'python3'))}"
        ),
        'readonly PRIVILEGED_BUNDLE_ROOT="/usr/local/libexec/onyx/regulatory-prod-lite"': (
            "readonly PRIVILEGED_BUNDLE_ROOT=" + shlex.quote(str(installed_bundle_root))
        ),
        'readonly INSTALL_ROOT="/usr/local/libexec/onyx/regulatory-prod-lite"': (
            f"readonly INSTALL_ROOT={shlex.quote(str(installed_bundle_root))}"
        ),
        "readonly -a SYSTEM_ANCESTORS=(\n"
        "  /\n"
        "  /usr\n"
        "  /usr/bin\n"
        "  /usr/sbin\n"
        "  /usr/local\n"
        "  /usr/local/libexec\n"
        "  /usr/local/libexec/onyx\n"
        ")": (
            "readonly -a SYSTEM_ANCESTORS=("
            + shlex.quote(str(installed_bundle_root.parent))
            + ")"
        ),
    }
    for script_name in (
        "regulatory-prod-lite-deploy.sh",
        "regulatory-prod-lite-preflight.sh",
        "regulatory-prod-lite-privileged-entrypoint",
    ):
        script_path = test_compose_root / script_name
        script_text = script_path.read_text(encoding="utf-8")
        for production_value, test_value in script_replacements.items():
            script_text = script_text.replace(production_value, test_value)
        script_path.write_text(script_text, encoding="utf-8")

    releases_root = installed_bundle_root / "releases"
    releases_root.mkdir(parents=True, mode=0o755)
    installed_bundle_root.chmod(0o755)
    pending_release = releases_root / ".pending"
    pending_release.mkdir(mode=0o755)
    manifest_lines: list[str] = []
    for file_name in _PRIVILEGED_RELEASE_FILES:
        source = test_compose_root / file_name
        destination = pending_release / file_name
        shutil.copy2(source, destination)
        destination.chmod(
            0o755
            if file_name
            in {
                "regulatory-prod-lite-privileged-entrypoint",
                "regulatory-prod-lite-preflight.sh",
            }
            else 0o644
        )
        manifest_lines.append(
            f"{hashlib.sha256(destination.read_bytes()).hexdigest()}  {file_name}\n"
        )
    manifest_path = pending_release / "REGULATORY_PRIVILEGED_MANIFEST.sha256"
    manifest_path.write_text("".join(manifest_lines), encoding="utf-8")
    manifest_path.chmod(0o644)
    bundle_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    release_path = releases_root / bundle_digest
    pending_release.rename(release_path)
    installed_entrypoint = installed_bundle_root / "regulatory-prod-lite-preflight"
    shutil.copy2(
        release_path / "regulatory-prod-lite-privileged-entrypoint",
        installed_entrypoint,
    )
    installed_entrypoint.chmod(0o755)
    current_release = installed_bundle_root / "current"
    current_release.write_text(f"{bundle_digest}\n", encoding="utf-8")
    current_release.chmod(0o644)
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}:{env['PATH']}",
            "FAKE_COMPOSE_CONFIG": str(config_path),
            "FAKE_COMPOSE_SERVICES": str(services_path),
            "FAKE_COMPOSE_VERSION": version,
            "FAKE_BACKEND_ROLE": backend_role,
            "FAKE_BASE_COMPOSE": str(base_path),
            "FAKE_MIGRATION_ENV": str(migration_env_file),
            "FAKE_DB_ADMIN_ENV": str(db_admin_env_file),
            "FAKE_DOCKER_FAIL_MATCH": fail_match,
            "FAKE_DOCKER_INVENTORY": str(inventory_path),
            "FAKE_DOCKER_LOG": str(log_path),
            "FAKE_BACKEND_REVISION": backend_revision,
            "FAKE_WEB_REVISION": web_revision,
            "FAKE_MODEL_REVISION": model_revision,
            "FAKE_ATTESTATION_PATH": str(attestation_path),
            "FAKE_EVIDENCE_PATH": str(evidence_path),
            "FAKE_ATTESTATION_OWNER": attestation_owner,
            "FAKE_ATTESTATION_MODE": f"{attestation_mode:o}",
            "FAKE_EVIDENCE_OWNER": evidence_owner,
            "FAKE_EVIDENCE_MODE": f"{evidence_mode:o}",
            "FAKE_TIMEOUT_EXIT": fake_timeout_exit,
            "FAKE_TIMEOUT_LOG": str(timeout_log),
            "FAKE_READINESS_CID": readiness_cid,
            "FAKE_READINESS_STATE": str(readiness_state_path),
            "FAKE_READINESS_LABEL": str(readiness_label_path),
            "FAKE_CLEANUP_LABEL_OVERRIDE": "",
            "FAKE_DOCKER_RM_FAIL": "false",
            "FAKE_READINESS_PERSISTS": "false",
            "FAKE_PRE_CID_FAILURE": "false",
            "FAKE_INVALID_CIDFILE": "false",
            "FAKE_LABEL_QUERY_RESULT": "",
            "FAKE_SUDO_AUTHORIZATION_FAIL": "false",
            "FAKE_DOCKER_ENVIRONMENT_LOG": str(environment_log_path),
            "FAKE_SUDO_LOG": str(sudo_log_path),
            "FAKE_REAL_RUNTIME_IMAGE": real_runtime_image,
            "FAKE_REAL_RUNTIME_USE_SETPRIV_NNP": str(
                real_runtime_use_setpriv_nnp
            ).lower(),
            "FAKE_REAL_RUNTIME_FORCE_TIMEOUT": "false",
            "FAKE_REAL_RUNTIME_PRE_CID_FAILURE": "false",
            "FAKE_REAL_DOCKER": "/usr/bin/docker",
            "FAKE_OUTER_TMPDIR_SOURCE": "",
            "FAKE_DEPLOY_SCRIPT": str(
                test_compose_root / "regulatory-prod-lite-deploy.sh"
            ),
            "FAKE_PREFLIGHT_SCRIPT": str(installed_entrypoint),
            "FAKE_PRIVILEGED_BUNDLE_DIGEST": bundle_digest,
            "FAKE_PRIVILEGED_BUNDLE_ROOT": str(installed_bundle_root),
            "FAKE_TRUSTED_COMMAND_PATH": (f"{bin_dir}:/usr/sbin:/usr/bin:/sbin:/bin"),
            "FAKE_TRUSTED_DOCKER_CONFIG": str(trusted_docker_config),
            "_IMAGE": _IMAGE,
            "_MODEL_IMAGE": _MODEL_IMAGE,
            "_WEB_IMAGE": _WEB_IMAGE,
        }
    )
    return env, env_file


def _run(
    script: Path,
    args: list[str],
    env: dict[str, str],
    *,
    preflight_as_root: bool = True,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    is_preflight = script == _PREFLIGHT
    if script == _DEPLOY and "FAKE_DEPLOY_SCRIPT" in env:
        script = Path(env["FAKE_DEPLOY_SCRIPT"])
    elif is_preflight and "FAKE_PREFLIGHT_SCRIPT" in env:
        script = Path(env["FAKE_PREFLIGHT_SCRIPT"])
        args = ["--bundle-digest", env["FAKE_PRIVILEGED_BUNDLE_DIGEST"], *args]
    command = [str(script), *args]
    if is_preflight:
        if preflight_as_root and os.geteuid() != 0:
            command = ["unshare", "--user", "--map-root-user", *command]
        elif not preflight_as_root and os.geteuid() == 0:
            command = [
                "setpriv",
                "--reuid=65534",
                "--regid=65534",
                "--clear-groups",
                *command,
            ]
    return subprocess.run(
        command,
        cwd=cwd or script.parent,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _subordinate_id_start(path: Path, identity: int) -> int | None:
    identity_names = {str(identity), pwd.getpwuid(identity).pw_name}
    for line in path.read_text(encoding="utf-8").splitlines():
        name, start, length = line.split(":")
        if name in identity_names and int(length) > 1001:
            return int(start)
    return None


def _prepare_privileged_installer_fixture(
    tmp_path: Path,
) -> tuple[Path, Path, Path, str]:
    tmp_path.chmod(0o700)
    fixed_sbin = tmp_path / "fixed-sbin"
    fixed_sbin.mkdir(mode=0o755)
    staging_root = tmp_path / "root-staging"
    staging_root.mkdir(mode=0o755)
    system_libexec = tmp_path / "system-libexec"
    install_root = system_libexec / "onyx" / "regulatory-prod-lite"
    installer_path = fixed_sbin / "install-regulatory-prod-lite-privileged-bundle"
    lock_path = tmp_path / "install.lock"

    installer_text = _PRIVILEGED_INSTALLER.read_text(encoding="utf-8")
    replacements = {
        'readonly INSTALLER_PATH="/usr/local/sbin/install-regulatory-prod-lite-privileged-bundle"': (
            f"readonly INSTALLER_PATH={shlex.quote(str(installer_path))}"
        ),
        'readonly STAGING_ROOT="/var/lib/onyx/regulatory-prod-lite-staging"': (
            f"readonly STAGING_ROOT={shlex.quote(str(staging_root))}"
        ),
        'readonly SYSTEM_LIBEXEC_ROOT="/usr/local/libexec"': (
            f"readonly SYSTEM_LIBEXEC_ROOT={shlex.quote(str(system_libexec))}"
        ),
        'readonly INSTALL_ROOT="/usr/local/libexec/onyx/regulatory-prod-lite"': (
            f"readonly INSTALL_ROOT={shlex.quote(str(install_root))}"
        ),
        'readonly INSTALL_LOCK="/run/lock/onyx-regulatory-prod-lite-install.lock"': (
            f"readonly INSTALL_LOCK={shlex.quote(str(lock_path))}"
        ),
        "readonly -a INSTALLER_ANCESTORS=(\n"
        "  /\n"
        "  /usr\n"
        "  /usr/bin\n"
        "  /usr/sbin\n"
        "  /usr/local\n"
        "  /usr/local/sbin\n"
        ")": (
            "readonly -a INSTALLER_ANCESTORS=("
            f"{shlex.quote(str(tmp_path))} {shlex.quote(str(fixed_sbin))})"
        ),
        "readonly -a STAGING_ANCESTORS=(/ /var /var/lib /var/lib/onyx)": (
            f"readonly -a STAGING_ANCESTORS=({shlex.quote(str(tmp_path))})"
        ),
        "readonly -a INSTALL_ANCESTORS=(/ /usr /usr/local)": (
            f"readonly -a INSTALL_ANCESTORS=({shlex.quote(str(tmp_path))})"
        ),
    }
    for production_value, test_value in replacements.items():
        assert production_value in installer_text
        installer_text = installer_text.replace(production_value, test_value)
    installer_path.write_text(installer_text, encoding="utf-8")
    installer_path.chmod(0o755)

    pending_stage = staging_root / ".pending"
    pending_stage.mkdir(mode=0o755)
    manifest_lines: list[str] = []
    for file_name in _PRIVILEGED_RELEASE_FILES:
        source_bytes = (_COMPOSE_ROOT / file_name).read_bytes()
        if file_name == "regulatory-prod-lite-privileged-entrypoint":
            entrypoint_text = source_bytes.decode()
            entrypoint_text = entrypoint_text.replace(
                'readonly INSTALL_ROOT="/usr/local/libexec/onyx/regulatory-prod-lite"',
                f"readonly INSTALL_ROOT={shlex.quote(str(install_root))}",
            )
            entrypoint_text = entrypoint_text.replace(
                "readonly -a SYSTEM_ANCESTORS=(\n"
                "  /\n"
                "  /usr\n"
                "  /usr/bin\n"
                "  /usr/sbin\n"
                "  /usr/local\n"
                "  /usr/local/libexec\n"
                "  /usr/local/libexec/onyx\n"
                ")",
                f"readonly -a SYSTEM_ANCESTORS=({shlex.quote(str(tmp_path))})",
            )
            source_bytes = entrypoint_text.encode()
        destination = pending_stage / file_name
        destination.write_bytes(source_bytes)
        destination.chmod(
            0o755
            if file_name
            in {
                "regulatory-prod-lite-privileged-entrypoint",
                "regulatory-prod-lite-preflight.sh",
            }
            else 0o644
        )
        manifest_lines.append(
            f"{hashlib.sha256(source_bytes).hexdigest()}  {file_name}\n"
        )
    manifest = pending_stage / "REGULATORY_PRIVILEGED_MANIFEST.sha256"
    manifest.write_text("".join(manifest_lines), encoding="utf-8")
    manifest.chmod(0o644)
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    stage = staging_root / digest
    pending_stage.rename(stage)
    return installer_path, stage, install_root, digest


def _run_privileged_installer(
    installer_path: Path,
    stage: Path,
    digest: str,
) -> subprocess.CompletedProcess[str]:
    command = [
        str(installer_path),
        "--source-dir",
        str(stage),
        "--expected-manifest-sha256",
        digest,
    ]
    if os.geteuid() != 0:
        command = ["/usr/bin/unshare", "--user", "--map-root-user", *command]
    return subprocess.run(
        command,
        cwd=installer_path.parent,
        capture_output=True,
        text=True,
        check=False,
    )


def _run_snapshot_helper_as_namespace_root(
    attestation_path: Path,
    evidence_path: Path,
    snapshot_directory: Path,
) -> subprocess.CompletedProcess[str]:
    probe = """
import json
import os
from pathlib import Path
import subprocess
import sys

helper, attestation, evidence, snapshots = sys.argv[1:]
attestation_path = Path(attestation)
evidence_path = Path(evidence)
snapshot_directory = Path(snapshots)
os.chown(attestation_path, 1001, 1001)
os.chown(evidence_path, 1001, 1001)
result = subprocess.run(
    [
        sys.executable,
        helper,
        "--attestation",
        attestation,
        "--evidence",
        evidence,
        "--snapshot-directory",
        snapshots,
    ],
    capture_output=True,
    text=True,
    check=False,
)
observed = {}
for path in snapshot_directory.iterdir():
    metadata = path.stat()
    observed[path.name] = {
        "uid": metadata.st_uid,
        "gid": metadata.st_gid,
        "mode": metadata.st_mode & 0o777,
        "matches_attestation": path.read_bytes() == attestation_path.read_bytes(),
        "matches_evidence": path.read_bytes() == evidence_path.read_bytes(),
    }
print(json.dumps({
    "returncode": result.returncode,
    "stdout": result.stdout,
    "stderr": result.stderr,
    "snapshots": observed,
}))
"""
    command = [sys.executable, "-c", probe]
    if os.geteuid() != 0:
        subordinate_uid = _subordinate_id_start(Path("/etc/subuid"), os.geteuid())
        subordinate_gid = _subordinate_id_start(Path("/etc/subgid"), os.getegid())
        if subordinate_uid is None or subordinate_gid is None:
            pytest.skip("root handoff test requires subordinate uid/gid mappings")
        command = [
            "unshare",
            "--user",
            f"--map-users=0:{os.geteuid()}:1",
            f"--map-users=1001:{subordinate_uid}:1",
            f"--map-groups=0:{os.getegid()}:1",
            f"--map-groups=1001:{subordinate_gid}:1",
            *command,
        ]
    return subprocess.run(
        [
            *command,
            str(_SNAPSHOT_HELPER),
            str(attestation_path),
            str(evidence_path),
            str(snapshot_directory),
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def test_readiness_snapshot_helper_copies_only_descriptor_validated_bytes(
    tmp_path: Path,
) -> None:
    attestation_bytes = b'{"schema_version":1,"marker":"never-print-attestation"}\n'
    evidence_bytes = b'{"marker":"never-print-evidence"}\n'
    attestation_path = tmp_path / "capability-attestation.json"
    attestation_path.write_bytes(attestation_bytes)
    attestation_path.chmod(0o600)
    evidence_path = tmp_path / "capability-evidence.json"
    evidence_path.write_bytes(evidence_bytes)
    evidence_path.chmod(0o400)
    sibling_secret = tmp_path / "unrelated-secret"
    sibling_secret.write_text("must-not-be-read", encoding="utf-8")
    sibling_secret.chmod(0o600)
    snapshot_directory = tmp_path / "snapshots"
    snapshot_directory.mkdir(mode=0o700)

    result = _run_snapshot_helper_as_namespace_root(
        attestation_path,
        evidence_path,
        snapshot_directory,
    )

    assert result.returncode == 0, result.stderr
    observed = json.loads(result.stdout)
    assert observed["returncode"] == 0, observed["stderr"]
    assert observed["stdout"] == ""
    assert "never-print" not in observed["stdout"]
    assert "never-print" not in observed["stderr"]
    assert set(observed["snapshots"]) == {
        "regulatory-capabilities.json",
        "regulatory-capability-evidence.json",
    }
    assert observed["snapshots"]["regulatory-capabilities.json"] == {
        "uid": 1001,
        "gid": 1001,
        "mode": 0o600,
        "matches_attestation": True,
        "matches_evidence": False,
    }
    assert observed["snapshots"]["regulatory-capability-evidence.json"] == {
        "uid": 1001,
        "gid": 1001,
        "mode": 0o600,
        "matches_attestation": False,
        "matches_evidence": True,
    }
    assert sibling_secret.read_text(encoding="utf-8") == "must-not-be-read"


def test_readiness_snapshot_helper_rejects_final_component_symlink(
    tmp_path: Path,
) -> None:
    target_path = tmp_path / "attestation-target.json"
    target_path.write_bytes(b'{"marker":"symlink-target-never-print"}\n')
    target_path.chmod(0o600)
    attestation_path = tmp_path / "capability-attestation.json"
    attestation_path.symlink_to(target_path)
    evidence_path = tmp_path / "capability-evidence.json"
    evidence_path.write_bytes(b'{"approved":true}\n')
    evidence_path.chmod(0o400)
    snapshot_directory = tmp_path / "snapshots"
    snapshot_directory.mkdir(mode=0o700)

    result = _run_snapshot_helper_as_namespace_root(
        attestation_path,
        evidence_path,
        snapshot_directory,
    )

    assert result.returncode == 0, result.stderr
    observed = json.loads(result.stdout)
    assert observed["returncode"] == 1
    assert "secure descriptor validation" in observed["stderr"]
    assert "symlink-target-never-print" not in observed["stdout"]
    assert "symlink-target-never-print" not in observed["stderr"]
    assert observed["snapshots"] == {}


def test_readiness_snapshot_helper_rejects_non_root_operator(tmp_path: Path) -> None:
    if os.geteuid() == 0:
        pytest.skip(
            "non-root helper contract is covered by the privileged runtime test"
        )
    attestation_path = tmp_path / "capability-attestation.json"
    attestation_path.write_bytes(b'{"schema_version":1}\n')
    attestation_path.chmod(0o600)
    evidence_path = tmp_path / "capability-evidence.json"
    evidence_path.write_bytes(b'{"approved":true}\n')
    evidence_path.chmod(0o400)
    snapshot_directory = tmp_path / "snapshots"
    snapshot_directory.mkdir(mode=0o700)

    result = subprocess.run(
        [
            sys.executable,
            str(_SNAPSHOT_HELPER),
            "--attestation",
            str(attestation_path),
            "--evidence",
            str(evidence_path),
            "--snapshot-directory",
            str(snapshot_directory),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "must run as root" in result.stderr
    assert list(snapshot_directory.iterdir()) == []


def _deploy_args(env_file: Path, env: dict[str, str]) -> list[str]:
    return [
        "--env-file",
        str(env_file),
        "--base-compose",
        env["FAKE_BASE_COMPOSE"],
        "--project-name",
        "onyx",
        "--migration-env-file",
        env["FAKE_MIGRATION_ENV"],
        "--db-admin-env-file",
        env["FAKE_DB_ADMIN_ENV"],
        "--infra-mode",
        "compose-managed",
        "--model-mode",
        "local",
        "--expected-image",
        _IMAGE,
        "--expected-web-image",
        _WEB_IMAGE,
        "--expected-model-image",
        _MODEL_IMAGE,
        "--backup-reference",
        "snapshot-2026-08-03T04:00Z",
        "--acknowledge-migration-impact",
    ]


def _preflight_args(env_file: Path, env: dict[str, str]) -> list[str]:
    return [
        "--env-file",
        str(env_file),
        "--base-compose",
        env["FAKE_BASE_COMPOSE"],
        "--project-name",
        "onyx",
        "--migration-env-file",
        env["FAKE_MIGRATION_ENV"],
        "--db-admin-env-file",
        env["FAKE_DB_ADMIN_ENV"],
        "--infra-mode",
        "compose-managed",
        "--model-mode",
        "local",
        "--expected-image",
        _IMAGE,
        "--expected-web-image",
        _WEB_IMAGE,
        "--expected-model-image",
        _MODEL_IMAGE,
    ]


def _cloud_deploy_args(env_file: Path, env: dict[str, str]) -> list[str]:
    return [
        "--env-file",
        str(env_file),
        "--base-compose",
        env["FAKE_BASE_COMPOSE"],
        "--project-name",
        "onyx",
        "--migration-env-file",
        env["FAKE_MIGRATION_ENV"],
        "--infra-mode",
        "external",
        "--model-mode",
        "cloud",
        "--expected-image",
        _IMAGE,
        "--expected-web-image",
        _WEB_IMAGE,
        "--backup-reference",
        "snapshot-2026-08-03T04:00Z",
        "--acknowledge-migration-impact",
    ]


def test_preflight_accepts_digest_pinned_parser_free_runtime(tmp_path: Path) -> None:
    env, env_file = _fake_docker(tmp_path)

    result = _run(
        _PREFLIGHT,
        [
            "--env-file",
            str(env_file),
            "--base-compose",
            env["FAKE_BASE_COMPOSE"],
            "--project-name",
            "onyx",
            "--migration-env-file",
            env["FAKE_MIGRATION_ENV"],
            "--db-admin-env-file",
            env["FAKE_DB_ADMIN_ENV"],
            "--infra-mode",
            "compose-managed",
            "--model-mode",
            "local",
            "--expected-image",
            _IMAGE,
            "--expected-web-image",
            _WEB_IMAGE,
            "--expected-model-image",
            _MODEL_IMAGE,
        ],
        env,
    )

    assert result.returncode == 0, result.stderr
    assert "No managed service or persistent Docker state was changed" in result.stdout


def test_preflight_rejects_non_root_operator_before_docker(tmp_path: Path) -> None:
    env, env_file = _fake_docker(tmp_path)

    result = _run(
        _PREFLIGHT,
        [
            "--env-file",
            str(env_file),
            "--base-compose",
            env["FAKE_BASE_COMPOSE"],
            "--project-name",
            "onyx",
            "--migration-env-file",
            env["FAKE_MIGRATION_ENV"],
            "--db-admin-env-file",
            env["FAKE_DB_ADMIN_ENV"],
            "--infra-mode",
            "compose-managed",
            "--model-mode",
            "local",
        ],
        env,
        preflight_as_root=False,
    )

    assert result.returncode == 1
    assert "must run as root:root" in result.stderr
    docker_log = Path(env["FAKE_DOCKER_LOG"])
    assert not docker_log.exists() or docker_log.read_text(encoding="utf-8") == ""


def test_preflight_delegates_capability_files_to_no_network_secure_reader(
    tmp_path: Path,
) -> None:
    env, env_file = _fake_docker(tmp_path)

    result = _run(
        _PREFLIGHT,
        [
            "--env-file",
            str(env_file),
            "--base-compose",
            env["FAKE_BASE_COMPOSE"],
            "--project-name",
            "onyx",
            "--migration-env-file",
            env["FAKE_MIGRATION_ENV"],
            "--db-admin-env-file",
            env["FAKE_DB_ADMIN_ENV"],
            "--infra-mode",
            "compose-managed",
            "--model-mode",
            "local",
            "--expected-image",
            _IMAGE,
            "--expected-web-image",
            _WEB_IMAGE,
            "--expected-model-image",
            _MODEL_IMAGE,
        ],
        env,
    )

    assert result.returncode == 0, result.stderr
    docker_log = Path(env["FAKE_DOCKER_LOG"]).read_text(encoding="utf-8")
    timeout_log = Path(env["FAKE_TIMEOUT_LOG"]).read_text(encoding="utf-8")
    assert (
        "--foreground --kill-after=5s 30s "
        f"{Path(env['FAKE_DOCKER_LOG']).parent / 'bin' / 'docker'} run" in timeout_log
    )
    readiness_command = next(
        command
        for command in docker_log.splitlines()
        if "--validate-capability-snapshots-only" in command
    )
    assert "run --cidfile " in readiness_command
    assert "--label io.regulatory.readiness-preflight-owner=" in readiness_command
    ownership_token = readiness_command.split(
        "--label io.regulatory.readiness-preflight-owner=", 1
    )[1].split()[0]
    assert len(ownership_token) == 64
    assert all(character in "0123456789abcdef" for character in ownership_token)
    assert " --name " not in readiness_command
    assert "--rm --pull never --network none --read-only" in docker_log
    assert "--user 1001:1001" in docker_log
    assert _IMAGE in docker_log
    assert "/app/scripts/regulatory_indexing_readiness.py" in docker_log
    assert "--validate-capability-snapshots-only" in docker_log
    assert env["FAKE_ATTESTATION_PATH"] not in readiness_command
    assert env["FAKE_EVIDENCE_PATH"] not in readiness_command


def test_preflight_times_out_readiness_container_and_cleans_snapshot(
    tmp_path: Path,
) -> None:
    env, env_file = _fake_docker(tmp_path, fake_timeout_exit="124")
    env["TMPDIR"] = str(tmp_path)

    result = _run(
        _PREFLIGHT,
        [
            "--env-file",
            str(env_file),
            "--base-compose",
            env["FAKE_BASE_COMPOSE"],
            "--project-name",
            "onyx",
            "--migration-env-file",
            env["FAKE_MIGRATION_ENV"],
            "--db-admin-env-file",
            env["FAKE_DB_ADMIN_ENV"],
            "--infra-mode",
            "compose-managed",
            "--model-mode",
            "local",
        ],
        env,
    )

    assert result.returncode == 1
    assert "timed out or failed" in result.stderr
    timeout_log = Path(env["FAKE_TIMEOUT_LOG"]).read_text(encoding="utf-8")
    assert (
        "--foreground --kill-after=5s 30s "
        f"{Path(env['FAKE_DOCKER_LOG']).parent / 'bin' / 'docker'} run" in timeout_log
    )
    docker_log = Path(env["FAKE_DOCKER_LOG"]).read_text(encoding="utf-8")
    assert " --name " not in docker_log
    assert f"rm -f {env['FAKE_READINESS_CID']}" in docker_log
    assert not Path(env["FAKE_READINESS_STATE"]).exists()
    assert not list(tmp_path.glob("regulatory-readiness-snapshot.*"))


@pytest.mark.parametrize("invalid_cidfile", [False, True])
def test_preflight_recovers_owned_container_when_cidfile_is_unavailable(
    tmp_path: Path,
    invalid_cidfile: bool,
) -> None:
    env, env_file = _fake_docker(tmp_path)
    env["TMPDIR"] = str(tmp_path)
    env["FAKE_PRE_CID_FAILURE"] = str(not invalid_cidfile).lower()
    env["FAKE_INVALID_CIDFILE"] = str(invalid_cidfile).lower()
    if invalid_cidfile:
        env["FAKE_READINESS_PERSISTS"] = "true"

    result = _run(
        _PREFLIGHT,
        [
            "--env-file",
            str(env_file),
            "--base-compose",
            env["FAKE_BASE_COMPOSE"],
            "--project-name",
            "onyx",
            "--migration-env-file",
            env["FAKE_MIGRATION_ENV"],
            "--db-admin-env-file",
            env["FAKE_DB_ADMIN_ENV"],
            "--infra-mode",
            "compose-managed",
            "--model-mode",
            "local",
        ],
        env,
    )

    assert result.returncode == (0 if invalid_cidfile else 1)
    if not invalid_cidfile:
        assert "timed out or failed" in result.stderr
    docker_log = Path(env["FAKE_DOCKER_LOG"]).read_text(encoding="utf-8")
    assert (
        "container ls --all --quiet --no-trunc --filter "
        "label=io.regulatory.readiness-preflight-owner="
    ) in docker_log
    assert f"rm -f {env['FAKE_READINESS_CID']}" in docker_log
    assert not Path(env["FAKE_READINESS_STATE"]).exists()
    assert not list(tmp_path.glob("regulatory-readiness-snapshot.*"))


def test_preflight_refuses_ambiguous_label_fallback_without_removing_anything(
    tmp_path: Path,
) -> None:
    env, env_file = _fake_docker(tmp_path)
    env["TMPDIR"] = str(tmp_path)
    env["FAKE_PRE_CID_FAILURE"] = "true"
    env["FAKE_LABEL_QUERY_RESULT"] = f"{'8' * 64}\n{'9' * 64}"

    result = _run(
        _PREFLIGHT,
        [
            "--env-file",
            str(env_file),
            "--base-compose",
            env["FAKE_BASE_COMPOSE"],
            "--project-name",
            "onyx",
            "--migration-env-file",
            env["FAKE_MIGRATION_ENV"],
            "--db-admin-env-file",
            env["FAKE_DB_ADMIN_ENV"],
            "--infra-mode",
            "compose-managed",
            "--model-mode",
            "local",
        ],
        env,
    )

    assert result.returncode == 1
    assert "timed out or failed" in result.stderr
    assert "identity is ambiguous" in result.stderr
    docker_log = Path(env["FAKE_DOCKER_LOG"]).read_text(encoding="utf-8")
    assert "rm -f" not in docker_log
    assert Path(env["FAKE_READINESS_STATE"]).exists()


def test_preflight_refuses_to_remove_container_with_mismatched_ownership_label(
    tmp_path: Path,
) -> None:
    env, env_file = _fake_docker(tmp_path, fake_timeout_exit="124")
    env["TMPDIR"] = str(tmp_path)
    env["FAKE_CLEANUP_LABEL_OVERRIDE"] = "unrelated-owner-token"

    result = _run(
        _PREFLIGHT,
        [
            "--env-file",
            str(env_file),
            "--base-compose",
            env["FAKE_BASE_COMPOSE"],
            "--project-name",
            "onyx",
            "--migration-env-file",
            env["FAKE_MIGRATION_ENV"],
            "--db-admin-env-file",
            env["FAKE_DB_ADMIN_ENV"],
            "--infra-mode",
            "compose-managed",
            "--model-mode",
            "local",
        ],
        env,
    )

    assert result.returncode == 1
    assert "timed out or failed" in result.stderr
    assert "ownership label does not match" in result.stderr
    docker_log = Path(env["FAKE_DOCKER_LOG"]).read_text(encoding="utf-8")
    assert f"rm -f {env['FAKE_READINESS_CID']}" not in docker_log
    assert Path(env["FAKE_READINESS_STATE"]).exists()


def test_preflight_surfaces_exact_container_removal_failure(tmp_path: Path) -> None:
    env, env_file = _fake_docker(tmp_path)
    env["TMPDIR"] = str(tmp_path)
    env["FAKE_DOCKER_RM_FAIL"] = "true"
    env["FAKE_READINESS_PERSISTS"] = "true"

    result = _run(
        _PREFLIGHT,
        [
            "--env-file",
            str(env_file),
            "--base-compose",
            env["FAKE_BASE_COMPOSE"],
            "--project-name",
            "onyx",
            "--migration-env-file",
            env["FAKE_MIGRATION_ENV"],
            "--db-admin-env-file",
            env["FAKE_DB_ADMIN_ENV"],
            "--infra-mode",
            "compose-managed",
            "--model-mode",
            "local",
        ],
        env,
    )

    assert result.returncode == 1
    assert "timed out or failed" not in result.stderr
    assert "could not remove the owned readiness container" in result.stderr
    docker_log = Path(env["FAKE_DOCKER_LOG"]).read_text(encoding="utf-8")
    assert f"rm -f {env['FAKE_READINESS_CID']}" in docker_log
    assert Path(env["FAKE_READINESS_STATE"]).exists()


def test_preflight_normal_cleanup_leaves_no_container_or_private_files(
    tmp_path: Path,
) -> None:
    env, env_file = _fake_docker(tmp_path)
    env["TMPDIR"] = str(tmp_path)

    result = _run(
        _PREFLIGHT,
        [
            "--env-file",
            str(env_file),
            "--base-compose",
            env["FAKE_BASE_COMPOSE"],
            "--project-name",
            "onyx",
            "--migration-env-file",
            env["FAKE_MIGRATION_ENV"],
            "--db-admin-env-file",
            env["FAKE_DB_ADMIN_ENV"],
            "--infra-mode",
            "compose-managed",
            "--model-mode",
            "local",
        ],
        env,
    )

    assert result.returncode == 0, result.stderr
    assert not Path(env["FAKE_READINESS_STATE"]).exists()
    assert not list(tmp_path.glob("regulatory-readiness-snapshot.*"))
    docker_log = Path(env["FAKE_DOCKER_LOG"]).read_text(encoding="utf-8")
    assert (
        f"container ls --all --quiet --no-trunc --filter id={env['FAKE_READINESS_CID']}"
        in docker_log
    )


@pytest.mark.parametrize(
    ("read_only", "owner", "expected_error"),
    [
        (False, "1001", "attestation bind mount"),
        (True, "1000", "failed secure descriptor validation"),
    ],
)
def test_preflight_rejects_untrusted_readiness_attestation_mount(
    tmp_path: Path,
    read_only: bool,
    owner: str,
    expected_error: str,
) -> None:
    config = json.loads(_config())
    config["services"]["background"]["volumes"][0]["read_only"] = read_only
    env, env_file = _fake_docker(
        tmp_path,
        config=json.dumps(config),
        attestation_owner=owner,
    )

    result = _run(
        _PREFLIGHT,
        [
            "--env-file",
            str(env_file),
            "--base-compose",
            env["FAKE_BASE_COMPOSE"],
            "--project-name",
            "onyx",
            "--migration-env-file",
            env["FAKE_MIGRATION_ENV"],
            "--db-admin-env-file",
            env["FAKE_DB_ADMIN_ENV"],
            "--infra-mode",
            "compose-managed",
            "--model-mode",
            "local",
        ],
        env,
    )

    assert result.returncode == 1
    assert expected_error in result.stderr


def test_preflight_rejects_missing_archived_capability_evidence_mount(
    tmp_path: Path,
) -> None:
    config = json.loads(_config())
    config["services"]["background"]["volumes"] = [
        volume
        for volume in config["services"]["background"]["volumes"]
        if volume["target"] != "/run/readiness/regulatory-capability-evidence.json"
    ]
    env, env_file = _fake_docker(tmp_path, config=json.dumps(config))

    result = _run(
        _PREFLIGHT,
        [
            "--env-file",
            str(env_file),
            "--base-compose",
            env["FAKE_BASE_COMPOSE"],
            "--project-name",
            "onyx",
            "--migration-env-file",
            env["FAKE_MIGRATION_ENV"],
            "--db-admin-env-file",
            env["FAKE_DB_ADMIN_ENV"],
            "--infra-mode",
            "compose-managed",
            "--model-mode",
            "local",
        ],
        env,
    )

    assert result.returncode == 1
    assert "archived capability evidence bind mount" in result.stderr


@pytest.mark.parametrize(
    ("read_only", "owner", "mode", "expected_error"),
    [
        (False, "1001", 0o400, "archived capability evidence bind mount"),
        (True, "1000", 0o400, "failed secure descriptor validation"),
        (True, "1001", 0o600, "failed secure descriptor validation"),
    ],
)
def test_preflight_rejects_untrusted_archived_capability_evidence(
    tmp_path: Path,
    read_only: bool,
    owner: str,
    mode: int,
    expected_error: str,
) -> None:
    config = json.loads(_config())
    config["services"]["background"]["volumes"][1]["read_only"] = read_only
    env, env_file = _fake_docker(
        tmp_path,
        config=json.dumps(config),
        evidence_owner=owner,
        evidence_mode=mode,
    )

    result = _run(
        _PREFLIGHT,
        [
            "--env-file",
            str(env_file),
            "--base-compose",
            env["FAKE_BASE_COMPOSE"],
            "--project-name",
            "onyx",
            "--migration-env-file",
            env["FAKE_MIGRATION_ENV"],
            "--db-admin-env-file",
            env["FAKE_DB_ADMIN_ENV"],
            "--infra-mode",
            "compose-managed",
            "--model-mode",
            "local",
        ],
        env,
    )

    assert result.returncode == 1
    assert expected_error in result.stderr


def test_preflight_accepts_unambiguous_external_infrastructure(tmp_path: Path) -> None:
    env, env_file = _fake_docker(
        tmp_path,
        config=_config(external_infra=True),
        active_services=(
            "api_server",
            "background",
            "web_server",
            "inference_model_server",
            "nginx",
            "certbot",
        ),
    )

    result = _run(
        _PREFLIGHT,
        [
            "--env-file",
            str(env_file),
            "--base-compose",
            env["FAKE_BASE_COMPOSE"],
            "--project-name",
            "onyx",
            "--migration-env-file",
            env["FAKE_MIGRATION_ENV"],
            "--infra-mode",
            "external",
            "--model-mode",
            "local",
        ],
        env,
    )

    assert result.returncode == 0, result.stderr


def test_preflight_accepts_cloud_models_without_activating_model_containers(
    tmp_path: Path,
) -> None:
    env, env_file = _fake_docker(
        tmp_path,
        config=_config(external_infra=True, cloud_models=True),
        active_services=("api_server", "background", "web_server", "nginx", "certbot"),
    )

    result = _run(
        _PREFLIGHT,
        [
            "--env-file",
            str(env_file),
            "--base-compose",
            env["FAKE_BASE_COMPOSE"],
            "--project-name",
            "onyx",
            "--migration-env-file",
            env["FAKE_MIGRATION_ENV"],
            "--infra-mode",
            "external",
            "--model-mode",
            "cloud",
        ],
        env,
    )

    assert result.returncode == 0, result.stderr


def test_preflight_rejects_model_digest_in_cloud_mode(tmp_path: Path) -> None:
    env, env_file = _fake_docker(
        tmp_path,
        config=_config(external_infra=True, cloud_models=True),
        active_services=("api_server", "background", "web_server", "nginx", "certbot"),
    )

    result = _run(
        _PREFLIGHT,
        [
            "--env-file",
            str(env_file),
            "--base-compose",
            env["FAKE_BASE_COMPOSE"],
            "--project-name",
            "onyx",
            "--migration-env-file",
            env["FAKE_MIGRATION_ENV"],
            "--infra-mode",
            "external",
            "--model-mode",
            "cloud",
            "--expected-model-image",
            _MODEL_IMAGE,
        ],
        env,
    )

    assert result.returncode != 0
    assert "must be omitted in cloud model mode" in result.stderr


@pytest.mark.parametrize(
    ("config", "version"),
    [
        (_config(build={"context": "../../backend"}), "2.40.3"),
        (_config(extra_service="importer"), "2.40.3"),
        (_config(multi_tenant=True), "2.40.3"),
        (_config(indexing_dependency=True), "2.40.3"),
        (
            _config(
                api_command=[
                    "/bin/sh",
                    "-c",
                    "alembic upgrade head && uvicorn onyx.main:app",
                ]
            ),
            "2.40.3",
        ),
        (_config(), "2.23.3"),
    ],
)
def test_preflight_fails_closed_for_unsafe_release_shape(
    tmp_path: Path, config: str, version: str
) -> None:
    env, env_file = _fake_docker(tmp_path, config=config, version=version)

    result = _run(
        _PREFLIGHT,
        [
            "--env-file",
            str(env_file),
            "--base-compose",
            env["FAKE_BASE_COMPOSE"],
            "--project-name",
            "onyx",
            "--migration-env-file",
            env["FAKE_MIGRATION_ENV"],
            "--db-admin-env-file",
            env["FAKE_DB_ADMIN_ENV"],
            "--infra-mode",
            "compose-managed",
            "--model-mode",
            "local",
        ],
        env,
    )

    assert result.returncode != 0
    log = (tmp_path / "docker.log").read_text(encoding="utf-8")
    assert " pull" not in log
    assert " up " not in log
    assert " run " not in log


def test_preflight_rejects_default_active_indexing_model_service(
    tmp_path: Path,
) -> None:
    config = json.loads(_config())
    config["services"]["indexing_model_server"] = {
        "image": _MODEL_IMAGE,
        "pull_policy": "always",
        "profiles": ["indexing-model-server"],
    }
    env, env_file = _fake_docker(
        tmp_path,
        config=json.dumps(config),
        active_services=(*_MANAGED_SERVICES, "indexing_model_server"),
    )

    result = _run(
        _PREFLIGHT,
        [
            "--env-file",
            str(env_file),
            "--base-compose",
            env["FAKE_BASE_COMPOSE"],
            "--project-name",
            "onyx",
            "--migration-env-file",
            env["FAKE_MIGRATION_ENV"],
            "--db-admin-env-file",
            env["FAKE_DB_ADMIN_ENV"],
            "--infra-mode",
            "compose-managed",
            "--model-mode",
            "local",
        ],
        env,
    )

    assert result.returncode != 0
    assert "indexing_model_server" in result.stderr
    assert "unexpected default-active services" in result.stderr


def test_preflight_rejects_running_legacy_indexer_by_container_name(
    tmp_path: Path,
) -> None:
    env, env_file = _fake_docker(
        tmp_path,
        running_inventory=(
            "deadbeef1234\told-project-indexing_model_server-1\tindexing_model_server\n"
        ),
    )

    result = _run(
        _PREFLIGHT,
        [
            "--env-file",
            str(env_file),
            "--base-compose",
            env["FAKE_BASE_COMPOSE"],
            "--project-name",
            "onyx",
            "--migration-env-file",
            env["FAKE_MIGRATION_ENV"],
            "--db-admin-env-file",
            env["FAKE_DB_ADMIN_ENV"],
            "--infra-mode",
            "compose-managed",
            "--model-mode",
            "local",
        ],
        env,
    )

    assert result.returncode != 0
    assert "old-project-indexing_model_server-1" in result.stderr
    assert "stop the named containers manually" in result.stderr


def test_cloud_preflight_rejects_running_legacy_inference_model(
    tmp_path: Path,
) -> None:
    env, env_file = _fake_docker(
        tmp_path,
        config=_config(external_infra=True, cloud_models=True),
        active_services=("api_server", "background", "web_server", "nginx", "certbot"),
        running_inventory=(
            "feedface1234\told-project-inference_model_server-1\t"
            "inference_model_server\n"
        ),
    )

    result = _run(
        _PREFLIGHT,
        [
            "--env-file",
            str(env_file),
            "--base-compose",
            env["FAKE_BASE_COMPOSE"],
            "--project-name",
            "onyx",
            "--migration-env-file",
            env["FAKE_MIGRATION_ENV"],
            "--infra-mode",
            "external",
            "--model-mode",
            "cloud",
        ],
        env,
    )

    assert result.returncode != 0
    assert "old-project-inference_model_server-1" in result.stderr
    assert "forbidden local-model" in result.stderr


def test_preflight_checks_background_container_image_id_not_mutable_tag(
    tmp_path: Path,
) -> None:
    env, env_file = _fake_docker(
        tmp_path,
        backend_role="importer",
        running_inventory="cafebabe1234\tlegacy-background-1\tbackground\n",
    )

    result = _run(
        _PREFLIGHT,
        [
            "--env-file",
            str(env_file),
            "--base-compose",
            env["FAKE_BASE_COMPOSE"],
            "--project-name",
            "onyx",
            "--migration-env-file",
            env["FAKE_MIGRATION_ENV"],
            "--db-admin-env-file",
            env["FAKE_DB_ADMIN_ENV"],
            "--infra-mode",
            "compose-managed",
            "--model-mode",
            "local",
        ],
        env,
    )

    assert result.returncode != 0
    assert "legacy-background-1" in result.stderr
    docker_log = (tmp_path / "docker.log").read_text(encoding="utf-8")
    assert "inspect --format {{.Image}} cafebabe1234" in docker_log
    assert "legacy/model:latest" not in docker_log


def test_preflight_rejects_standalone_importer_service_and_image_role(
    tmp_path: Path,
) -> None:
    env, env_file = _fake_docker(
        tmp_path,
        backend_role="importer",
        running_inventory="facefeed1234\tregulatory-importer-run-1\timporter\n",
    )

    result = _run(
        _PREFLIGHT,
        [
            "--env-file",
            str(env_file),
            "--base-compose",
            env["FAKE_BASE_COMPOSE"],
            "--project-name",
            "onyx",
            "--migration-env-file",
            env["FAKE_MIGRATION_ENV"],
            "--db-admin-env-file",
            env["FAKE_DB_ADMIN_ENV"],
            "--infra-mode",
            "compose-managed",
            "--model-mode",
            "local",
        ],
        env,
    )

    assert result.returncode != 0
    assert "regulatory-importer-run-1" in result.stderr
    docker_log = (tmp_path / "docker.log").read_text(encoding="utf-8")
    assert "inspect --format {{.Image}} facefeed1234" in docker_log


def test_preflight_explains_missing_nginx_environment_file(tmp_path: Path) -> None:
    env, env_file = _fake_docker(tmp_path)
    base_compose = Path(env["FAKE_BASE_COMPOSE"])
    base_compose.write_text(
        "services:\n  nginx:\n    env_file:\n      - path: .env.nginx\n",
        encoding="utf-8",
    )

    result = _run(
        _PREFLIGHT,
        [
            "--env-file",
            str(env_file),
            "--base-compose",
            str(base_compose),
            "--project-name",
            "onyx",
            "--migration-env-file",
            env["FAKE_MIGRATION_ENV"],
            "--db-admin-env-file",
            env["FAKE_DB_ADMIN_ENV"],
            "--infra-mode",
            "compose-managed",
            "--model-mode",
            "local",
        ],
        env,
    )

    assert result.returncode != 0
    assert ".env.nginx must be provisioned" in result.stderr


def test_preflight_rejects_interpolation_env_different_from_service_env(
    tmp_path: Path,
) -> None:
    env, _env_file = _fake_docker(tmp_path)
    base_compose = Path(env["FAKE_BASE_COMPOSE"])
    base_compose.write_text(
        "services:\n  api_server:\n    env_file:\n      - path: .env\n",
        encoding="utf-8",
    )
    alternate_env = tmp_path / "alternate.env"
    alternate_env.write_text("# not the service env\n", encoding="utf-8")
    alternate_env.chmod(0o600)

    result = _run(
        _PREFLIGHT,
        [
            "--env-file",
            str(alternate_env),
            "--base-compose",
            str(base_compose),
            "--project-name",
            "onyx",
            "--migration-env-file",
            env["FAKE_MIGRATION_ENV"],
            "--db-admin-env-file",
            env["FAKE_DB_ADMIN_ENV"],
            "--infra-mode",
            "compose-managed",
            "--model-mode",
            "local",
        ],
        env,
    )

    assert result.returncode != 0
    assert "same .env file loaded by the base" in result.stderr


def test_preflight_requires_private_environment_permissions(tmp_path: Path) -> None:
    env, env_file = _fake_docker(tmp_path)
    env_file.chmod(0o640)

    result = _run(
        _PREFLIGHT,
        [
            "--env-file",
            str(env_file),
            "--base-compose",
            env["FAKE_BASE_COMPOSE"],
            "--project-name",
            "onyx",
            "--migration-env-file",
            env["FAKE_MIGRATION_ENV"],
            "--db-admin-env-file",
            env["FAKE_DB_ADMIN_ENV"],
            "--infra-mode",
            "compose-managed",
            "--model-mode",
            "local",
        ],
        env,
    )

    assert result.returncode != 0
    assert "mode 0600" in result.stderr


def test_preflight_rejects_mixed_external_and_local_infrastructure(
    tmp_path: Path,
) -> None:
    env, env_file = _fake_docker(tmp_path, config=_config(external_infra=True))

    result = _run(
        _PREFLIGHT,
        [
            "--env-file",
            str(env_file),
            "--base-compose",
            env["FAKE_BASE_COMPOSE"],
            "--project-name",
            "onyx",
            "--migration-env-file",
            env["FAKE_MIGRATION_ENV"],
            "--infra-mode",
            "external",
            "--model-mode",
            "local",
        ],
        env,
    )

    assert result.returncode != 0
    assert "unexpected default-active services" in result.stderr
    assert "relational_db" in result.stderr


def test_deploy_never_builds_and_runs_migration_before_start(tmp_path: Path) -> None:
    env, env_file = _fake_docker(tmp_path)

    result = _run(_DEPLOY, ["deploy", *_deploy_args(env_file, env)], env)

    assert result.returncode == 0, result.stderr
    commands = (tmp_path / "docker.log").read_text(encoding="utf-8").splitlines()
    pull_index = next(
        i
        for i, command in enumerate(commands)
        if " pull api_server background web_server" in command
    )
    probe_index = next(
        i
        for i, command in enumerate(commands)
        if command.startswith("run --rm --network none")
    )
    indexing_stop_index = next(
        i
        for i, command in enumerate(commands)
        if " stop indexing_model_server" in command
    )
    stop_index = next(
        i for i, command in enumerate(commands) if " stop background" in command
    )
    application_stop_index = next(
        i
        for i, command in enumerate(commands)
        if " stop nginx web_server api_server" in command
    )
    infra_index = next(
        i
        for i, command in enumerate(commands)
        if " up -d --no-build" in command
        and " relational_db elasticsearch cache minio" in command
    )
    migration_index = next(
        i
        for i, command in enumerate(commands)
        if " --profile regulatory-migration run --rm --no-deps --env-from-file "
        in command
        and command.endswith(" regulatory_migration")
    )
    api_gate_index = next(
        i
        for i, command in enumerate(commands)
        if " up -d --no-build --wait --wait-timeout 900 api_server" in command
    )
    full_up_index = next(
        i
        for i, command in enumerate(commands)
        if command.endswith(" up -d --no-build --wait --wait-timeout 900")
    )
    assert (
        pull_index
        < probe_index
        < indexing_stop_index
        < stop_index
        < application_stop_index
        < infra_index
        < migration_index
        < api_gate_index
        < full_up_index
    )
    assert all("docker build" not in command for command in commands)


def test_deploy_rejects_operator_compose_profiles_before_bounded_sudo(
    tmp_path: Path,
) -> None:
    env, env_file = _fake_docker(tmp_path)
    env["COMPOSE_PROFILES"] = "local-infra"

    result = _run(_DEPLOY, ["deploy", *_deploy_args(env_file, env)], env)

    assert result.returncode == 1
    assert "COMPOSE_PROFILES must be unset" in result.stderr
    docker_log = Path(env["FAKE_DOCKER_LOG"])
    assert not docker_log.exists() or docker_log.read_text(encoding="utf-8") == ""


@pytest.mark.parametrize(
    "variable",
    [
        "DOCKER_HOST",
        "DOCKER_CONTEXT",
        "DOCKER_CONFIG",
        "ONYX_BACKEND_LITE_IMAGE",
        "ONYX_WEB_SERVER_IMAGE",
        "REGULATORY_POSTGRES_IMAGE",
    ],
)
def test_deploy_rejects_dangerous_ambient_target_overrides_before_sudo(
    tmp_path: Path,
    variable: str,
) -> None:
    env, env_file = _fake_docker(tmp_path)
    env[variable] = "operator-ambient-override"

    result = _run(_DEPLOY, ["deploy", *_deploy_args(env_file, env)], env)

    assert result.returncode == 1
    assert variable in result.stderr
    assert "must be unset" in result.stderr
    assert not Path(env["FAKE_SUDO_LOG"]).exists()
    assert not Path(env["FAKE_DOCKER_LOG"]).exists()


def test_deploy_uses_identical_sanitized_environment_before_and_after_sudo(
    tmp_path: Path,
) -> None:
    env, env_file = _fake_docker(tmp_path)
    env["REGULATORY_AMBIENT_MARKER"] = "must-not-cross-boundary"

    result = _run(_DEPLOY, ["deploy", *_deploy_args(env_file, env)], env)

    assert result.returncode == 0, result.stderr
    observed_environments = (
        Path(env["FAKE_DOCKER_ENVIRONMENT_LOG"])
        .read_text(encoding="utf-8")
        .splitlines()
    )
    assert len(observed_environments) > 10
    assert set(observed_environments) == {
        "DOCKER_HOST=unix:///var/run/docker.sock "
        "DOCKER_CONTEXT=<unset> "
        "ONYX_BACKEND_LITE_IMAGE=<unset> "
        "AMBIENT_MARKER=<unset> "
        f"PATH={env['FAKE_TRUSTED_COMMAND_PATH']} "
        "HOME=/var/empty "
        f"DOCKER_CONFIG={env['FAKE_TRUSTED_DOCKER_CONFIG']} "
        "BASH_ENV=<unset> ENV=<unset> PYTHONPATH=<unset> SUDO_ASKPASS=<unset>"
    }


def test_deploy_ignores_hostile_command_and_home_paths_across_root_boundary(
    tmp_path: Path,
) -> None:
    env, env_file = _fake_docker(tmp_path)
    trusted_bin = Path(env["FAKE_DOCKER_LOG"]).parent / "bin"
    hostile_bin = tmp_path / "hostile-bin"
    hostile_bin.mkdir()
    hostile_home = tmp_path / "hostile-home"
    hostile_plugin_dir = hostile_home / ".docker" / "cli-plugins"
    hostile_plugin_dir.mkdir(parents=True)
    marker = tmp_path / "hostile-executable-ran"

    def write_forwarder(name: str, target: Path) -> None:
        executable = hostile_bin / name
        executable.write_text(
            "#!/bin/bash\n"
            f"printf '%s\\n' {shlex.quote(name)} >>{shlex.quote(str(marker))}\n"
            f'exec {shlex.quote(str(target))} "$@"\n',
            encoding="utf-8",
        )
        executable.chmod(0o755)

    write_forwarder("bash", Path("/bin/bash"))
    write_forwarder("sudo", trusted_bin / "sudo")
    write_forwarder("docker-compose", trusted_bin / "docker-compose")
    write_forwarder("python3", trusted_bin / "python3")
    write_forwarder("timeout", trusted_bin / "timeout")
    write_forwarder("stat", trusted_bin / "stat")
    hostile_plugin = hostile_plugin_dir / "docker-compose"
    hostile_plugin.write_text(
        "#!/bin/bash\n"
        f"printf '%s\\n' docker-compose-plugin >>{shlex.quote(str(marker))}\n"
        f'exec {shlex.quote(str(trusted_bin / "docker"))} compose "$@"\n',
        encoding="utf-8",
    )
    hostile_plugin.chmod(0o755)
    hostile_docker = hostile_bin / "docker"
    hostile_docker.write_text(
        "#!/bin/bash\n"
        f"printf '%s\\n' docker >>{shlex.quote(str(marker))}\n"
        "if [[ ${1:-} == compose ]]; then\n"
        "  shift\n"
        f'  exec {shlex.quote(str(hostile_plugin))} "$@"\n'
        "fi\n"
        f'exec {shlex.quote(str(trusted_bin / "docker"))} "$@"\n',
        encoding="utf-8",
    )
    hostile_docker.chmod(0o755)
    bash_env = tmp_path / "hostile-bash-env"
    bash_env.write_text(
        f"printf '%s\\n' BASH_ENV >>{shlex.quote(str(marker))}\n",
        encoding="utf-8",
    )

    env.update(
        {
            "PATH": f"{hostile_bin}:{env['PATH']}",
            "HOME": str(hostile_home),
            "BASH_ENV": str(bash_env),
            "ENV": str(tmp_path / "hostile-env"),
            "PYTHONPATH": str(tmp_path / "hostile-python"),
            "SUDO_ASKPASS": str(hostile_bin / "sudo-askpass"),
        }
    )

    result = _run(_DEPLOY, ["deploy", *_deploy_args(env_file, env)], env)

    assert result.returncode == 0, result.stderr
    assert not marker.exists()
    sudo_log = Path(env["FAKE_SUDO_LOG"]).read_text(encoding="utf-8")
    assert sudo_log.startswith(f"-n -- {env['FAKE_PREFLIGHT_SCRIPT']} --bundle-digest ")
    assert "/usr/bin/env" not in sudo_log
    assert "/bin/bash" not in sudo_log
    assert str(hostile_bin) not in sudo_log
    assert str(hostile_home) not in sudo_log
    assert "BASH_ENV" not in sudo_log
    assert "PYTHONPATH" not in sudo_log
    assert "SUDO_ASKPASS" not in sudo_log


def test_preflight_rejects_writable_fixed_docker_config_before_docker(
    tmp_path: Path,
) -> None:
    env, env_file = _fake_docker(tmp_path)
    Path(env["FAKE_TRUSTED_DOCKER_CONFIG"]).chmod(0o770)

    result = _run(
        _PREFLIGHT,
        [
            "--env-file",
            str(env_file),
            "--base-compose",
            env["FAKE_BASE_COMPOSE"],
            "--project-name",
            "onyx",
            "--migration-env-file",
            env["FAKE_MIGRATION_ENV"],
            "--db-admin-env-file",
            env["FAKE_DB_ADMIN_ENV"],
            "--infra-mode",
            "compose-managed",
            "--model-mode",
            "local",
        ],
        env,
    )

    assert result.returncode == 1
    assert "Docker CLI configuration directory" in result.stderr
    assert "must not be group writable or world accessible" in result.stderr
    assert not Path(env["FAKE_DOCKER_LOG"]).exists()


def test_deployment_scripts_pin_privileged_executable_boundaries() -> None:
    deploy_source = _DEPLOY.read_text(encoding="utf-8")
    preflight_source = _PREFLIGHT.read_text(encoding="utf-8")

    for source in (deploy_source, preflight_source):
        assert source.startswith("#!/bin/bash -p\n")
        assert 'readonly TRUSTED_SYSTEM_PATH="/usr/sbin:/usr/bin"' in source
        assert 'readonly DOCKER_BIN="/usr/bin/docker"' in source
        assert (
            'readonly COMPOSE_BIN="/usr/libexec/docker/cli-plugins/docker-compose"'
            in source
        )
        assert 'readonly DOCKER_CONFIG_DIR="/etc/onyx/regulatory-docker"' in source
        assert "$HOME/.docker" not in source
        assert "docker compose" not in source
    assert 'readonly SUDO_BIN="/usr/bin/sudo"' in deploy_source
    assert '"$SUDO_BIN" -n --' in deploy_source
    assert (
        'readonly PRIVILEGED_BUNDLE_ROOT="/usr/local/libexec/onyx/regulatory-prod-lite"'
        in deploy_source
    )
    assert '"$SUDO_BIN" -n -- "$PRIVILEGED_PREFLIGHT"' in deploy_source
    assert '"$BASH_BIN" -p "$PREFLIGHT"' not in deploy_source
    assert '"$SCRIPT_DIR/regulatory-prod-lite-preflight.sh"' not in deploy_source


def test_privileged_bundle_sources_define_digest_bound_installed_boundary() -> None:
    assert _PRIVILEGED_ENTRYPOINT.is_file()
    assert _PRIVILEGED_INSTALLER.is_file()
    assert _PRIVILEGED_MANIFEST.is_file()

    entrypoint = _PRIVILEGED_ENTRYPOINT.read_text(encoding="utf-8")
    installer = _PRIVILEGED_INSTALLER.read_text(encoding="utf-8")
    preflight = _PREFLIGHT.read_text(encoding="utf-8")

    assert entrypoint.startswith("#!/bin/bash -p\n")
    assert 'readonly INSTALL_ROOT="/usr/local/libexec/onyx/regulatory-prod-lite"' in (
        entrypoint
    )
    assert "--bundle-digest" in entrypoint
    assert "sha256sum" in entrypoint
    assert "must be owned by root:root" in entrypoint
    assert "must not be a symlink" in entrypoint
    assert installer.startswith("#!/bin/bash -p\n")
    assert "must be run as root" in installer
    assert "--expected-manifest-sha256" in installer
    assert 'readonly PYTHON_BIN="/usr/bin/python3"' in preflight
    assert preflight.count('"$PYTHON_BIN" -I -S') == 2

    manifest_entries = []
    for line in _PRIVILEGED_MANIFEST.read_text(encoding="utf-8").splitlines():
        digest, file_name = line.split("  ", maxsplit=1)
        manifest_entries.append(file_name)
        assert (
            digest
            == hashlib.sha256((_COMPOSE_ROOT / file_name).read_bytes()).hexdigest()
        )
    assert manifest_entries == list(_PRIVILEGED_RELEASE_FILES)


def test_privileged_installer_atomically_installs_reviewed_digest_bundle(
    tmp_path: Path,
) -> None:
    installer, stage, install_root, digest = _prepare_privileged_installer_fixture(
        tmp_path
    )

    result = _run_privileged_installer(installer, stage, digest)

    assert result.returncode == 0, result.stderr
    assert (install_root / "current").read_text(encoding="utf-8") == f"{digest}\n"
    release = install_root / "releases" / digest
    assert release.is_dir()
    assert {path.name for path in release.iterdir()} == {
        *_PRIVILEGED_RELEASE_FILES,
        "REGULATORY_PRIVILEGED_MANIFEST.sha256",
    }
    assert (install_root / "regulatory-prod-lite-preflight").stat().st_mode & 0o777 == (
        0o755
    )
    assert not list((install_root / "releases").glob(".install-*"))
    assert not list(install_root.glob(".current-*"))
    assert not list(install_root.glob(".entrypoint-*"))


@pytest.mark.parametrize(
    ("mutation", "diagnostic"),
    [
        ("symlink", "must not be a symlink"),
        ("writable", "must not be group/world writable"),
        ("directory-writable", "must not be group/world writable"),
    ],
)
def test_privileged_installer_rejects_unsafe_staged_files_without_activation(
    tmp_path: Path,
    mutation: str,
    diagnostic: str,
) -> None:
    installer, stage, install_root, digest = _prepare_privileged_installer_fixture(
        tmp_path
    )
    helper = stage / "regulatory_readiness_file_snapshot.py"
    if mutation == "symlink":
        helper.unlink()
        helper.symlink_to(_SNAPSHOT_HELPER)
    else:
        helper.chmod(0o666)

    result = _run_privileged_installer(installer, stage, digest)

    assert result.returncode == 1
    assert diagnostic in result.stderr
    assert not install_root.exists()


def test_privileged_installer_refuses_checkout_execution_before_copying(
    tmp_path: Path,
) -> None:
    command = [str(_PRIVILEGED_INSTALLER), "--help"]
    if os.geteuid() != 0:
        command = ["/usr/bin/unshare", "--user", "--map-root-user", *command]

    result = subprocess.run(
        command,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "only from its fixed root-owned path" in result.stderr


def test_deploy_uses_installed_bundle_when_checkout_security_files_are_hostile(
    tmp_path: Path,
) -> None:
    env, env_file = _fake_docker(tmp_path)
    checkout_root = Path(env["FAKE_DEPLOY_SCRIPT"]).parent
    marker = tmp_path / "checkout-security-code-ran"
    (checkout_root / "regulatory-prod-lite-preflight.sh").write_text(
        f"#!/bin/bash\nprintf owned >{shlex.quote(str(marker))}\nexit 91\n",
        encoding="utf-8",
    )
    (checkout_root / "regulatory_readiness_file_snapshot.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('owned')\n",
        encoding="utf-8",
    )
    (checkout_root / "docker-compose.regulatory-prod-lite.yml").write_text(
        "this is deliberately not a Compose document\n",
        encoding="utf-8",
    )
    for path in (
        checkout_root / "regulatory-prod-lite-preflight.sh",
        checkout_root / "regulatory_readiness_file_snapshot.py",
        checkout_root / "docker-compose.regulatory-prod-lite.yml",
    ):
        path.chmod(0o777)

    result = _run(_DEPLOY, ["deploy", *_deploy_args(env_file, env)], env)

    assert result.returncode == 0, result.stderr
    assert not marker.exists()
    docker_log = Path(env["FAKE_DOCKER_LOG"]).read_text(encoding="utf-8")
    installed_root = env["FAKE_PRIVILEGED_BUNDLE_ROOT"]
    assert f"{installed_root}/releases/" in docker_log
    assert str(checkout_root / "docker-compose.regulatory-prod-lite.yml") not in (
        docker_log
    )


def test_privileged_preflight_isolated_python_ignores_hostile_cwd(
    tmp_path: Path,
) -> None:
    env, env_file = _fake_docker(tmp_path)
    hostile_cwd = tmp_path / "hostile-cwd"
    hostile_cwd.mkdir()
    marker = tmp_path / "hostile-secrets-module-ran"
    (hostile_cwd / "secrets.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('owned')\n"
        "def token_hex(_: int) -> str:\n"
        "    return '0' * 64\n",
        encoding="utf-8",
    )

    result = _run(
        _PREFLIGHT,
        _preflight_args(env_file, env),
        env,
        cwd=hostile_cwd,
    )

    assert result.returncode == 0, result.stderr
    assert not marker.exists()


@pytest.mark.parametrize(
    ("mutation", "diagnostic"),
    [
        ("symlink", "must not be a symlink"),
        ("writable", "must not be group/world writable"),
    ],
)
def test_privileged_preflight_rejects_mutable_or_symlinked_installed_overlay(
    tmp_path: Path,
    mutation: str,
    diagnostic: str,
) -> None:
    env, env_file = _fake_docker(tmp_path)
    release = (
        Path(env["FAKE_PRIVILEGED_BUNDLE_ROOT"])
        / "releases"
        / env["FAKE_PRIVILEGED_BUNDLE_DIGEST"]
    )
    overlay = release / "docker-compose.regulatory-prod-lite.yml"
    if mutation == "directory-writable":
        release.parent.chmod(0o777)
    elif mutation == "symlink":
        overlay.unlink()
        overlay.symlink_to(
            Path(env["FAKE_DEPLOY_SCRIPT"]).parent
            / "docker-compose.regulatory-prod-lite.yml"
        )
    else:
        overlay.chmod(0o666)

    result = _run(_PREFLIGHT, _preflight_args(env_file, env), env)

    assert result.returncode == 1
    assert diagnostic in result.stderr
    assert not Path(env["FAKE_DOCKER_LOG"]).exists()


def test_privileged_preflight_rejects_non_root_owned_installed_helper(
    tmp_path: Path,
) -> None:
    if os.geteuid() == 0:
        pytest.skip("the namespace ownership probe is intended for a non-root runner")
    subordinate_uid = _subordinate_id_start(Path("/etc/subuid"), os.geteuid())
    subordinate_gid = _subordinate_id_start(Path("/etc/subgid"), os.getegid())
    if subordinate_uid is None or subordinate_gid is None:
        pytest.skip("root boundary test requires subordinate uid/gid mappings")
    env, env_file = _fake_docker(tmp_path)
    release = (
        Path(env["FAKE_PRIVILEGED_BUNDLE_ROOT"])
        / "releases"
        / env["FAKE_PRIVILEGED_BUNDLE_DIGEST"]
    )
    helper = release / "regulatory_readiness_file_snapshot.py"
    entrypoint = Path(env["FAKE_PREFLIGHT_SCRIPT"])
    command = [
        "/usr/bin/unshare",
        "--user",
        f"--map-users=0:{os.geteuid()}:1",
        f"--map-users=1001:{subordinate_uid}:1",
        f"--map-groups=0:{os.getegid()}:1",
        f"--map-groups=1001:{subordinate_gid}:1",
        "/bin/bash",
        "-c",
        'chown 1001:1001 "$1" && shift && exec "$@"',
        "ownership-probe",
        str(helper),
        str(entrypoint),
        "--bundle-digest",
        env["FAKE_PRIVILEGED_BUNDLE_DIGEST"],
        *_preflight_args(env_file, env),
    ]

    result = subprocess.run(
        command,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1, result.stderr
    assert "must be owned by root:root" in result.stderr
    assert not Path(env["FAKE_DOCKER_LOG"]).exists()


def test_deploy_uses_noninteractive_sudo_and_controls_authorization_failure(
    tmp_path: Path,
) -> None:
    env, env_file = _fake_docker(tmp_path)
    env["FAKE_SUDO_AUTHORIZATION_FAIL"] = "true"

    result = _run(_DEPLOY, ["deploy", *_deploy_args(env_file, env)], env)

    assert result.returncode == 1
    assert "noninteractive sudo authorization" in result.stderr
    sudo_log = Path(env["FAKE_SUDO_LOG"]).read_text(encoding="utf-8")
    assert sudo_log.startswith(f"-n -- {env['FAKE_PREFLIGHT_SCRIPT']} --bundle-digest ")
    assert "/usr/bin/env" not in sudo_log
    assert "/bin/bash" not in sudo_log
    assert not Path(env["FAKE_DOCKER_LOG"]).exists()


def test_runbook_requires_canonical_noninteractive_least_privilege_preflight() -> None:
    runbook = _RUNBOOK.read_text(encoding="utf-8")

    assert "./regulatory-prod-lite-deploy.sh preflight" in runbook
    assert "`sudo -n`" in runbook
    assert "`NOPASSWD` sudoers" in runbook
    assert "Do not use sudoers wildcards" in runbook
    assert "authorize neither `/usr/bin/env`, `docker`, a" in runbook
    assert "`/usr/bin/sudo -n`" in runbook
    assert "`/etc/onyx/regulatory-docker`" in runbook
    assert "root-owned" in runbook
    assert "/usr/local/libexec/onyx/regulatory-prod-lite" in runbook
    assert "/usr/local/sbin/install-regulatory-prod-lite-privileged-bundle" in (runbook)
    assert "run an installer directly from that extraction or a checkout" in runbook
    assert "sudo for this provisioning step" in runbook
    assert "`/usr/bin/python3 -I -S`" in runbook
    assert "$HOME/.docker" not in runbook
    assert "sudo -- ./regulatory-prod-lite-preflight.sh" not in runbook


def test_cloud_deploy_never_pulls_or_inspects_model_image(tmp_path: Path) -> None:
    env, env_file = _fake_docker(
        tmp_path,
        config=_config(external_infra=True, cloud_models=True),
        active_services=("api_server", "background", "web_server", "nginx", "certbot"),
    )

    result = _run(_DEPLOY, ["deploy", *_cloud_deploy_args(env_file, env)], env)

    assert result.returncode == 0, result.stderr
    commands = (tmp_path / "docker.log").read_text(encoding="utf-8").splitlines()
    assert not any(" pull inference_model_server" in command for command in commands)
    assert not any(_MODEL_IMAGE in command for command in commands)
    assert any(" stop inference_model_server" in command for command in commands)


def test_deploy_requires_backup_acknowledgement_before_pull(tmp_path: Path) -> None:
    env, env_file = _fake_docker(tmp_path)
    args = _deploy_args(env_file, env)
    backup_option = args.index("--backup-reference")
    del args[backup_option : backup_option + 2]

    result = _run(_DEPLOY, ["deploy", *args], env)

    assert result.returncode != 0
    log = (tmp_path / "docker.log").read_text(encoding="utf-8")
    assert " pull" not in log
    assert " run " not in log
    assert " up " not in log


def test_deploy_requires_explicit_migration_impact_acknowledgement(
    tmp_path: Path,
) -> None:
    env, env_file = _fake_docker(tmp_path)
    args = _deploy_args(env_file, env)
    args.remove("--acknowledge-migration-impact")

    result = _run(_DEPLOY, ["deploy", *args], env)

    assert result.returncode != 0
    assert "--acknowledge-migration-impact" in result.stderr
    log = (tmp_path / "docker.log").read_text(encoding="utf-8")
    assert " pull" not in log


def test_deploy_refuses_concurrent_rollout_for_same_project(tmp_path: Path) -> None:
    env, env_file = _fake_docker(tmp_path)
    env["TMPDIR"] = str(tmp_path)
    args = _deploy_args(env_file, env)
    project_value = args.index("--project-name") + 1
    args[project_value] = "onyx-lock-test"
    lock_path = tmp_path / "regulatory-prod-lite-onyx-lock-test.lock"

    with lock_path.open("w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = _run(_DEPLOY, ["deploy", *args], env)

    assert result.returncode != 0
    assert "already running" in result.stderr
    docker_log = (tmp_path / "docker.log").read_text(encoding="utf-8")
    assert " pull" not in docker_log


def test_bad_runtime_contract_does_not_stop_running_worker(tmp_path: Path) -> None:
    env, env_file = _fake_docker(tmp_path, backend_role="importer")

    result = _run(_DEPLOY, ["deploy", *_deploy_args(env_file, env)], env)

    assert result.returncode != 0
    log = (tmp_path / "docker.log").read_text(encoding="utf-8")
    assert " pull api_server background web_server" in log
    assert " stop background" not in log
    assert "alembic upgrade" not in log


def test_mismatched_source_revisions_do_not_stop_running_worker(tmp_path: Path) -> None:
    env, env_file = _fake_docker(tmp_path, web_revision="e" * 40)

    result = _run(_DEPLOY, ["deploy", *_deploy_args(env_file, env)], env)

    assert result.returncode != 0
    assert "same source revision" in result.stderr
    log = (tmp_path / "docker.log").read_text(encoding="utf-8")
    assert " stop background" not in log


def test_failed_migration_leaves_background_stopped(tmp_path: Path) -> None:
    env, env_file = _fake_docker(
        tmp_path,
        fail_match=" --profile regulatory-migration run --rm --no-deps --env-from-file ",
    )

    result = _run(_DEPLOY, ["deploy", *_deploy_args(env_file, env)], env)

    assert result.returncode != 0
    log = (tmp_path / "docker.log").read_text(encoding="utf-8")
    assert " stop background" in log
    assert "regulatory_migration" in log
    assert " up -d --no-build --wait --wait-timeout 900 api_server" not in log
    assert "Application services remain stopped" in result.stderr


def test_failed_api_liveness_gate_leaves_background_stopped(tmp_path: Path) -> None:
    env, env_file = _fake_docker(
        tmp_path,
        fail_match=" up -d --no-build --wait --wait-timeout 900 api_server ",
    )

    result = _run(_DEPLOY, ["deploy", *_deploy_args(env_file, env)], env)

    assert result.returncode != 0
    commands = (tmp_path / "docker.log").read_text(encoding="utf-8").splitlines()
    assert any(" stop background" in command for command in commands)
    assert not any(
        command.endswith(" up -d --no-build --wait --wait-timeout 900")
        for command in commands
    )
    assert "background remain stopped" in result.stderr


def test_rollback_requires_schema_compatibility_and_never_migrates(
    tmp_path: Path,
) -> None:
    env, env_file = _fake_docker(tmp_path)
    deploy_args = _deploy_args(env_file, env)
    deploy_args.remove("--acknowledge-migration-impact")
    base_args = ["rollback", *deploy_args]

    refused = _run(_DEPLOY, base_args, env)
    assert refused.returncode != 0
    before = (tmp_path / "docker.log").read_text(encoding="utf-8")
    assert " pull" not in before

    accepted = _run(_DEPLOY, [*base_args, "--schema-compatible"], env)

    assert accepted.returncode == 0, accepted.stderr
    after = (tmp_path / "docker.log").read_text(encoding="utf-8")
    assert " pull api_server background web_server" in after
    assert " stop background" in after
    assert " up -d --no-build --wait --wait-timeout 900 api_server" in after
    assert " up -d --no-build --wait --wait-timeout 900" in after
    assert "alembic upgrade" not in after
