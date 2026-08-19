from __future__ import annotations

import fcntl
import json
import os
import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[4]
_COMPOSE_ROOT = _REPO_ROOT / "deployment" / "docker_compose"
_PREFLIGHT = _COMPOSE_ROOT / "regulatory-prod-lite-preflight.sh"
_DEPLOY = _COMPOSE_ROOT / "regulatory-prod-lite-deploy.sh"
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
) -> tuple[dict[str, str], Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log_path = tmp_path / "docker.log"
    config_path = tmp_path / "config.json"
    attestation_path = tmp_path / "regulatory-capabilities.json"
    attestation_path.write_text("{}\n", encoding="utf-8")
    attestation_path.chmod(attestation_mode)
    evidence_path = tmp_path / "regulatory-capability-evidence.json"
    evidence_path.write_text('{"approved":true}\n', encoding="utf-8")
    evidence_path.chmod(evidence_mode)
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
    docker = bin_dir / "docker"
    docker.write_text(
        """#!/usr/bin/env bash
set -eu
printf '%s\\n' "$*" >>"$FAKE_DOCKER_LOG"
if [[ "${1:-} ${2:-} ${3:-}" == "compose version --short" ]]; then
  printf '%s\\n' "$FAKE_COMPOSE_VERSION"
  exit 0
fi
if [[ "${1:-}" == "ps" ]]; then
  command cat "$FAKE_DOCKER_INVENTORY"
  exit 0
fi
if [[ "${1:-}" == "inspect" ]]; then
  printf '%s\\n' 'sha256:7777777777777777777777777777777777777777777777777777777777777777'
  exit 0
fi
if [[ "${1:-}" == "run" && " $* " == *" --validate-capability-files-only "* ]]; then
  if [[ "$FAKE_ATTESTATION_OWNER" != "1001" || "$FAKE_ATTESTATION_MODE" != "600" ]]; then
    exit 42
  fi
  if [[ "$FAKE_EVIDENCE_OWNER" != "1001" || "$FAKE_EVIDENCE_MODE" != "400" ]]; then
    exit 42
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
""",
        encoding="utf-8",
    )
    docker.chmod(0o755)
    fake_stat = bin_dir / "stat"
    fake_stat.write_text(
        """#!/usr/bin/env bash
set -eu
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
""",
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
            "_MODEL_IMAGE": _MODEL_IMAGE,
            "_WEB_IMAGE": _WEB_IMAGE,
        }
    )
    return env, env_file


def _run(
    script: Path, args: list[str], env: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(script), *args],
        cwd=_COMPOSE_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


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
    assert "run --rm --pull never --network none --read-only" in docker_log
    assert "--user 1001:1001" in docker_log
    assert _IMAGE in docker_log
    assert "/app/scripts/regulatory_indexing_readiness.py" in docker_log
    assert "--validate-capability-files-only" in docker_log


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
