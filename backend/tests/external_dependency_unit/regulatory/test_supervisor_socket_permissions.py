from __future__ import annotations

import configparser
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.unit.scripts import test_regulatory_prod_lite_deploy as prod_lite_deploy

_REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
_BACKEND_ROOT = _REPOSITORY_ROOT / "backend"
_RUNTIME_DOCKERFILE = _BACKEND_ROOT / "Dockerfile.runtime-lite"
_SUPERVISOR_CONFIG = _BACKEND_ROOT / "supervisord-lite.conf"


@pytest.fixture(scope="module")
def runtime_lite_supervisor_image() -> Iterator[str]:
    fingerprint = hashlib.sha256()
    for path in (
        _RUNTIME_DOCKERFILE,
        _BACKEND_ROOT / "requirements" / "runtime.txt",
        _BACKEND_ROOT / "requirements" / "ee.txt",
        _BACKEND_ROOT / "scripts" / "regulatory_indexing_readiness.py",
    ):
        fingerprint.update(path.read_bytes())
    image = f"onyx-task8-runtime-lite-supervisor:{fingerprint.hexdigest()[:16]}"
    existing = subprocess.run(
        ["docker", "image", "inspect", image],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    built_for_test = existing.returncode != 0
    if built_for_test:
        build = subprocess.run(
            [
                "docker",
                "build",
                "--target",
                "runtime-lite",
                "--tag",
                image,
                "--file",
                str(_RUNTIME_DOCKERFILE),
                str(_BACKEND_ROOT),
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=900,
        )
        assert build.returncode == 0, build.stderr

    try:
        yield image
    finally:
        if built_for_test:
            subprocess.run(
                ["docker", "image", "rm", image],
                capture_output=True,
                text=True,
                check=False,
                timeout=60,
            )


@pytest.fixture(scope="module")
def preflight_operator_image(
    runtime_lite_supervisor_image: str,
) -> Iterator[str]:
    fingerprint = hashlib.sha256(
        f"{runtime_lite_supervisor_image}:trusted-cli-v1".encode()
    ).hexdigest()[:16]
    image = f"onyx-task8-preflight-operator:{fingerprint}"
    existing = subprocess.run(
        ["docker", "image", "inspect", image],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    built_for_test = existing.returncode != 0
    if built_for_test:
        build = subprocess.run(
            ["docker", "build", "--tag", image, "-"],
            input=(
                f"FROM {runtime_lite_supervisor_image}\n"
                "USER 0:0\n"
                "RUN apt-get update && apt-get install -y --no-install-recommends jq "
                "&& rm -rf /var/lib/apt/lists/* "
                "&& install -d -o root -g root -m 0750 /etc/onyx "
                "/etc/onyx/regulatory-docker "
                "&& install -d -o root -g root -m 0755 "
                "/usr/libexec/docker/cli-plugins "
                "&& printf '{}\\n' >/etc/onyx/regulatory-docker/config.json "
                "&& chmod 0640 /etc/onyx/regulatory-docker/config.json "
                "&& if [ ! -e /usr/bin/python3 ]; then "
                "ln -s /usr/local/bin/python3 /usr/bin/python3; fi\n"
            ),
            capture_output=True,
            text=True,
            check=False,
            timeout=300,
        )
        assert build.returncode == 0, build.stderr

    try:
        yield image
    finally:
        if built_for_test:
            subprocess.run(
                ["docker", "image", "rm", image],
                capture_output=True,
                text=True,
                check=False,
                timeout=60,
            )


def test_runtime_supervisor_socket_enforces_canonical_numeric_ownership(
    runtime_lite_supervisor_image: str,
    request: pytest.FixtureRequest,
) -> None:
    parser = configparser.ConfigParser(interpolation=None)
    parser.read(_SUPERVISOR_CONFIG, encoding="utf-8")
    socket_config = parser["unix_http_server"]
    assert socket_config["chown"] == "1001:1001"
    assert int(socket_config["chmod"], 8) == 0o770

    probe_root = Path(
        tempfile.mkdtemp(prefix=".task8-supervisor-", dir=_REPOSITORY_ROOT)
    )
    request.addfinalizer(lambda: shutil.rmtree(probe_root, ignore_errors=True))
    executable_config = probe_root / "supervisord.conf"
    executable_config.write_text(
        "\n".join(
            [
                "[supervisord]",
                "nodaemon=true",
                "user=root",
                "pidfile=/tmp/supervisord.pid",
                "logfile=/tmp/supervisord.log",
                "childlogdir=/tmp",
                "",
                "[unix_http_server]",
                "file=/tmp/supervisor.sock",
                f"chmod={socket_config['chmod']}",
                f"chown={socket_config['chown']}",
                "",
                "[supervisorctl]",
                "serverurl=unix:///tmp/supervisor.sock",
                "",
                "[rpcinterface:supervisor]",
                "supervisor.rpcinterface_factory = supervisor.rpcinterface:make_main_rpcinterface",
                "",
                "[program:permission_probe]",
                "user=onyx",
                "command=/bin/sleep 30",
                "autorestart=false",
            ]
        ),
        encoding="utf-8",
    )
    probe_script = probe_root / "probe.py"
    probe_script.write_text(
        """
import os
import pathlib
import subprocess
import time

CONFIG = "/probe/supervisord.conf"
SOCKET = pathlib.Path("/tmp/supervisor.sock")
PYTHON = "/usr/local/bin/python"
READINESS = "/app/scripts/regulatory_indexing_readiness.py"
SUPERVISORCTL = "/usr/local/bin/supervisorctl"
ATTESTATION = pathlib.Path("/tmp/capability-attestation.json")
EVIDENCE = pathlib.Path("/tmp/capability-evidence.json")


def run_as(
    uid: int,
    gid: int,
    command: list[str],
    *,
    timeout: float = 5,
) -> subprocess.CompletedProcess[str]:
    def drop_identity() -> None:
        os.setgroups([])
        os.setgid(gid)
        os.setuid(uid)

    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
        preexec_fn=drop_identity,
        timeout=timeout,
    )


def query_as(uid: int, gid: int) -> subprocess.CompletedProcess[str]:
    return run_as(uid, gid, [SUPERVISORCTL, "-c", CONFIG, "status"])


ATTESTATION.write_bytes(b"{}\\n")
EVIDENCE.write_bytes(b"archived-evidence-never-print\\n")
ATTESTATION.chmod(0o600)
EVIDENCE.chmod(0o400)
os.chown(ATTESTATION, 1001, 1001)
os.chown(EVIDENCE, 1001, 1001)
file_validation = run_as(
    1001,
    1001,
    [
        PYTHON,
        READINESS,
        "--validate-capability-files-only",
        "--capability-attestation",
        str(ATTESTATION),
        "--capability-evidence",
        str(EVIDENCE),
    ],
    timeout=20,
)
if file_validation.returncode != 0 or "READY" not in file_validation.stdout:
    raise RuntimeError("Runtime image could not securely validate capability files")
if "never-print" in file_validation.stdout or "never-print" in file_validation.stderr:
    raise RuntimeError("Runtime readiness emitted capability evidence contents")


supervisor = subprocess.Popen(
    ["/usr/local/bin/supervisord", "-c", CONFIG],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)
try:
    deadline = time.monotonic() + 5
    while not SOCKET.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    if not SOCKET.exists():
        raise RuntimeError("Supervisor did not create the canonical socket")

    metadata = SOCKET.stat()
    if (metadata.st_uid, metadata.st_gid) != (1001, 1001):
        raise RuntimeError("Supervisor created the socket with wrong ownership")
    if metadata.st_mode & 0o777 != 0o770:
        raise RuntimeError("Supervisor created the socket with wrong mode")

    application_user = query_as(1001, 1001)
    if application_user.returncode != 0 or "permission_probe" not in application_user.stdout:
        raise RuntimeError("Application UID/GID could not query Supervisor")

    unrelated_user = query_as(2002, 2002)
    if unrelated_user.returncode == 0:
        raise RuntimeError("Unrelated UID/GID unexpectedly queried Supervisor")
finally:
    supervisor.terminate()
    try:
        supervisor.wait(timeout=5)
    except subprocess.TimeoutExpired:
        supervisor.kill()
        supervisor.wait(timeout=5)
""".strip()
        + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--pull",
            "never",
            "--network",
            "none",
            "--read-only",
            "--pids-limit",
            "64",
            "--memory",
            "512m",
            "--cpus",
            "1",
            "--user",
            "0:0",
            "--cap-drop",
            "ALL",
            "--cap-add",
            "CHOWN",
            "--cap-add",
            "SETUID",
            "--cap-add",
            "SETGID",
            "--cap-add",
            "KILL",
            "--tmpfs",
            "/tmp:rw,nosuid,nodev,noexec,size=16m,mode=1777",
            "--mount",
            f"type=bind,source={executable_config},target=/probe/supervisord.conf,readonly",
            "--mount",
            f"type=bind,source={probe_script},target=/probe/probe.py,readonly",
            "--entrypoint",
            "/usr/local/bin/python",
            runtime_lite_supervisor_image,
            "/probe/probe.py",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=45,
    )

    assert result.returncode == 0, result.stderr


def _run_preflight_as_distinct_operator_and_root(
    env_file: Path,
    env: dict[str, str],
    operator_image: str,
    probe_root: Path,
) -> subprocess.CompletedProcess[str]:
    command = [
        str(prod_lite_deploy._PREFLIGHT),
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
    ]
    driver = probe_root / "preflight-operator.py"
    driver.write_text(
        """
import json
import os
from pathlib import Path
import subprocess
import sys

attestation, evidence, *command = sys.argv[1:]


def run_as(uid=None, gid=None):
    def drop_identity():
        assert uid is not None and gid is not None
        os.setgroups([])
        os.setgid(gid)
        os.setuid(uid)

    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
        preexec_fn=drop_identity if uid is not None else None,
        timeout=90,
    )


non_root = run_as(2002, 2002)
if non_root.returncode != 1 or "must be invoked as root" not in non_root.stderr:
    raise RuntimeError("Distinct non-root operator did not fail closed before Docker")

os.chown(attestation, 1001, 1001)
os.chown(evidence, 1001, 1001)
attestation_metadata = Path(attestation).stat()
evidence_metadata = Path(evidence).stat()
root = run_as()
print(json.dumps({
    "non_root_returncode": non_root.returncode,
    "non_root_stderr": non_root.stderr,
    "source_owners": [
        [attestation_metadata.st_uid, attestation_metadata.st_gid],
        [evidence_metadata.st_uid, evidence_metadata.st_gid],
    ],
    "root_returncode": root.returncode,
    "root_stdout": root.stdout,
    "root_stderr": root.stderr,
}))
""".strip()
        + "\n",
        encoding="utf-8",
    )
    driver.chmod(0o600)

    operator_environment = probe_root / "operator.env"
    selected_environment = {
        key: value
        for key, value in env.items()
        if key.startswith("FAKE_") or key in {"_IMAGE", "_MODEL_IMAGE", "_WEB_IMAGE"}
    }
    selected_environment["PATH"] = (
        f"{Path(env['FAKE_DOCKER_LOG']).parent / 'bin'}:/usr/local/bin:/usr/bin:/bin"
    )
    selected_environment["TMPDIR"] = env["TMPDIR"]
    selected_environment["FAKE_REAL_DOCKER"] = "/usr/libexec/regulatory-test-docker"
    selected_environment["FAKE_OUTER_TMPDIR_SOURCE"] = str(probe_root)
    operator_environment.write_text(
        "".join(f"{key}={value}\n" for key, value in selected_environment.items()),
        encoding="utf-8",
    )
    operator_environment.chmod(0o600)

    docker_binary = Path(shutil.which("docker") or "")
    assert docker_binary.is_file()
    operator_tools = probe_root / "operator-tools"
    operator_tools.mkdir(mode=0o700)
    real_docker_source = operator_tools / "docker-real"
    shutil.copy2(docker_binary, real_docker_source)
    fake_docker_source = operator_tools / "docker"
    fixture_docker = Path(env["FAKE_DOCKER_LOG"]).parent / "bin" / "docker"
    fake_environment_source = operator_tools / "fake-environment"
    fixture_environment = Path(env["FAKE_DOCKER_LOG"]).parent / "fake-environment"
    shutil.copy2(fixture_environment, fake_environment_source)
    fake_docker_source.write_text(
        fixture_docker.read_text(encoding="utf-8").replace(
            f'source "{fixture_environment}"',
            'source "/usr/libexec/regulatory-test-environment"',
        ),
        encoding="utf-8",
    )
    fake_compose_source = operator_tools / "docker-compose"
    fake_compose_source.write_text(
        '#!/bin/bash\nexec /usr/bin/docker compose "$@"\n',
        encoding="utf-8",
    )
    prepare_tools = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--pull",
            "never",
            "--network",
            "none",
            "--mount",
            f"type=bind,source={operator_tools},target=/tools",
            "--entrypoint",
            "/bin/sh",
            operator_image,
            "-c",
            "chown 0:0 /tools/* && chmod 0755 /tools/docker /tools/docker-compose /tools/docker-real && chmod 0600 /tools/fake-environment",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert prepare_tools.returncode == 0, prepare_tools.stderr

    return subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--pull",
            "never",
            "--network",
            "none",
            "--read-only",
            "--pids-limit",
            "128",
            "--memory",
            "512m",
            "--cpus",
            "1",
            "--user",
            "0:0",
            "--cap-drop",
            "ALL",
            "--cap-add",
            "CHOWN",
            "--cap-add",
            "DAC_OVERRIDE",
            "--cap-add",
            "FOWNER",
            "--cap-add",
            "SETUID",
            "--cap-add",
            "SETGID",
            "--mount",
            f"type=bind,source={probe_root},target=/tmp",
            "--env-file",
            str(operator_environment),
            "--env",
            "PYTHONDONTWRITEBYTECODE=1",
            "--mount",
            "type=bind,source=/var/run/docker.sock,target=/var/run/docker.sock",
            "--mount",
            f"type=bind,source={_REPOSITORY_ROOT},target={_REPOSITORY_ROOT}",
            "--mount",
            f"type=bind,source={fake_docker_source},target=/usr/bin/docker,readonly",
            "--mount",
            f"type=bind,source={fake_compose_source},target=/usr/libexec/docker/cli-plugins/docker-compose,readonly",
            "--mount",
            f"type=bind,source={real_docker_source},target=/usr/libexec/regulatory-test-docker,readonly",
            "--mount",
            f"type=bind,source={fake_environment_source},target=/usr/libexec/regulatory-test-environment,readonly",
            "--entrypoint",
            "/usr/local/bin/python",
            operator_image,
            str(driver),
            env["FAKE_ATTESTATION_PATH"],
            env["FAKE_EVIDENCE_PATH"],
            *command,
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=150,
    )


@pytest.fixture(scope="module")
def real_docker_needs_setpriv_nnp(
    runtime_lite_supervisor_image: str,
) -> bool:
    docker_nnp = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--pull",
            "never",
            "--network",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--entrypoint",
            "/bin/true",
            runtime_lite_supervisor_image,
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    if docker_nnp.returncode == 0:
        return False

    assert "operation not permitted" in (docker_nnp.stdout + docker_nnp.stderr)
    setpriv_nnp = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--pull",
            "never",
            "--network",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--entrypoint",
            "/usr/bin/setpriv",
            runtime_lite_supervisor_image,
            "--no-new-privs",
            "/bin/sh",
            "-c",
            "grep -Eq '^NoNewPrivs:[[:space:]]+1$' /proc/self/status",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert setpriv_nnp.returncode == 0, setpriv_nnp.stderr
    return True


@pytest.mark.parametrize("scenario", ["normal", "symlink", "timeout", "pre-cid"])
def test_preflight_real_runtime_enforces_source_and_cleanup_contract(
    runtime_lite_supervisor_image: str,
    preflight_operator_image: str,
    real_docker_needs_setpriv_nnp: bool,
    request: pytest.FixtureRequest,
    scenario: str,
) -> None:
    docker_visible_temp = Path(
        tempfile.mkdtemp(prefix=".task8-preflight-", dir=_REPOSITORY_ROOT)
    )
    request.addfinalizer(lambda: shutil.rmtree(docker_visible_temp, ignore_errors=True))
    fake_root = docker_visible_temp / "fixture"
    fake_root.mkdir()
    env, env_file = prod_lite_deploy._fake_docker(
        fake_root,
        use_real_snapshot_helper=True,
        use_real_timeout=True,
        real_runtime_image=runtime_lite_supervisor_image,
        real_runtime_use_setpriv_nnp=real_docker_needs_setpriv_nnp,
    )
    env["TMPDIR"] = str(docker_visible_temp)
    env["FAKE_REAL_RUNTIME_FORCE_TIMEOUT"] = str(scenario == "timeout").lower()
    env["FAKE_REAL_RUNTIME_PRE_CID_FAILURE"] = str(scenario == "pre-cid").lower()
    unrelated_container = ""
    if scenario == "pre-cid":
        unrelated = subprocess.run(
            [
                "docker",
                "create",
                "--label",
                f"io.regulatory.readiness-preflight-owner={'0' * 64}",
                "--user",
                "1001:1001",
                "--entrypoint",
                "/bin/true",
                runtime_lite_supervisor_image,
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        assert unrelated.returncode == 0, unrelated.stderr
        unrelated_container = unrelated.stdout.strip()
        assert re.fullmatch(r"[0-9a-f]{64}", unrelated_container)
        request.addfinalizer(
            lambda: subprocess.run(
                ["docker", "rm", "-f", unrelated_container],
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
            )
        )
    evidence_path = Path(env["FAKE_EVIDENCE_PATH"])
    evidence_bytes = b'{"approved":true,"marker":"runtime-evidence-never-print"}\n'
    evidence_path.chmod(0o600)
    evidence_path.write_bytes(evidence_bytes)
    evidence_path.chmod(0o400)
    evidence_digest = hashlib.sha256(evidence_bytes).hexdigest()
    attestation_path = Path(env["FAKE_ATTESTATION_PATH"])
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
    attestation_path.chmod(0o600)
    if scenario == "symlink":
        target_path = fake_root / "attestation-target.json"
        attestation_path.replace(target_path)
        attestation_path.symlink_to(target_path)

    docker_log_path = Path(env["FAKE_DOCKER_LOG"])
    docker_log_path.touch(mode=0o666)
    docker_log_path.chmod(0o666)

    result = _run_preflight_as_distinct_operator_and_root(
        env_file,
        env,
        preflight_operator_image,
        docker_visible_temp,
    )

    assert result.returncode == 0, result.stderr
    observed = json.loads(result.stdout)
    assert observed["non_root_returncode"] == 1
    assert "must be invoked as root" in observed["non_root_stderr"]
    assert observed["source_owners"] == [[1001, 1001], [1001, 1001]]
    combined_output = result.stdout + result.stderr
    assert "runtime-evidence-never-print" not in combined_output
    docker_log = docker_log_path.read_text(encoding="utf-8")
    validation_runs = [
        command
        for command in docker_log.splitlines()
        if "--validate-capability-snapshots-only" in command
    ]
    if scenario == "symlink":
        assert observed["root_returncode"] == 1
        assert "sources failed secure descriptor validation" in observed["root_stderr"]
        assert validation_runs == []
    elif scenario in {"timeout", "pre-cid"}:
        assert observed["root_returncode"] == 1
        assert "snapshot validation timed out or failed" in observed["root_stderr"]
        assert len(validation_runs) == 1
        if scenario == "pre-cid":
            assert (
                "--filter label=io.regulatory.readiness-preflight-owner=" in docker_log
            )
            ownership_token = re.search(
                r"--label io\.regulatory\.readiness-preflight-owner=([0-9a-f]{64})",
                validation_runs[0],
            )
            assert ownership_token is not None
            owned_residue = subprocess.run(
                [
                    "docker",
                    "container",
                    "ls",
                    "--all",
                    "--quiet",
                    "--no-trunc",
                    "--filter",
                    "label=io.regulatory.readiness-preflight-owner="
                    f"{ownership_token.group(1)}",
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
            )
            assert owned_residue.returncode == 0, owned_residue.stderr
            assert owned_residue.stdout == ""
            unrelated_still_exists = subprocess.run(
                ["docker", "container", "inspect", unrelated_container],
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
            )
            assert unrelated_still_exists.returncode == 0
        removed_container = re.search(
            r"^rm -f ([0-9a-f]{64})$", docker_log, re.MULTILINE
        )
        assert removed_container is not None
        inspect_removed = subprocess.run(
            ["docker", "container", "inspect", removed_container.group(1)],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        assert inspect_removed.returncode != 0
    else:
        assert observed["root_returncode"] == 0, observed["root_stderr"]
        assert len(validation_runs) == 1
        assert "--network none --read-only" in validation_runs[0]
        assert "--user 1001:1001" in validation_runs[0]
    assert not list(docker_visible_temp.glob("regulatory-readiness-snapshot.*"))
