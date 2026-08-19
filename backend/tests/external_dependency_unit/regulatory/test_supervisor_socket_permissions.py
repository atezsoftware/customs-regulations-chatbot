from __future__ import annotations

import configparser
import hashlib
import json
import os
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


def _run_preflight_with_canonical_source_owner(
    env_file: Path,
    env: dict[str, str],
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
    if (os.geteuid(), os.getegid()) != (1001, 1001) and os.geteuid() != 0:
        unshare = shutil.which("unshare")
        assert unshare is not None, "numeric-owner preflight proof requires unshare"
        command = [
            unshare,
            "--user",
            f"--map-users={os.geteuid()},1001,1",
            f"--map-groups={os.getegid()},1001,1",
            *command,
        ]

    return subprocess.run(
        command,
        cwd=prod_lite_deploy._COMPOSE_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=90,
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


@pytest.mark.parametrize("attestation_symlink", [False, True])
def test_preflight_real_runtime_validates_snapshots_and_rejects_host_symlink(
    tmp_path: Path,
    runtime_lite_supervisor_image: str,
    real_docker_needs_setpriv_nnp: bool,
    request: pytest.FixtureRequest,
    attestation_symlink: bool,
) -> None:
    docker_visible_temp = Path(
        tempfile.mkdtemp(prefix=".task8-preflight-", dir=_REPOSITORY_ROOT)
    )
    request.addfinalizer(lambda: shutil.rmtree(docker_visible_temp, ignore_errors=True))
    env, env_file = prod_lite_deploy._fake_docker(
        tmp_path,
        use_real_snapshot_helper=True,
        use_real_timeout=True,
        real_runtime_image=runtime_lite_supervisor_image,
        real_runtime_use_setpriv_nnp=real_docker_needs_setpriv_nnp,
    )
    env["TMPDIR"] = str(docker_visible_temp)
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
    if attestation_symlink:
        target_path = tmp_path / "attestation-target.json"
        attestation_path.replace(target_path)
        attestation_path.symlink_to(target_path)

    if os.geteuid() == 0:
        os.chown(attestation_path, 1001, 1001)
        os.chown(evidence_path, 1001, 1001)

    result = _run_preflight_with_canonical_source_owner(env_file, env)

    combined_output = result.stdout + result.stderr
    assert "runtime-evidence-never-print" not in combined_output
    docker_log = Path(env["FAKE_DOCKER_LOG"]).read_text(encoding="utf-8")
    validation_runs = [
        command
        for command in docker_log.splitlines()
        if "--validate-capability-snapshots-only" in command
    ]
    if attestation_symlink:
        assert result.returncode == 1
        assert "sources failed secure descriptor validation" in result.stderr
        assert validation_runs == []
    else:
        assert result.returncode == 0, result.stderr
        assert len(validation_runs) == 1
        assert "--network none --read-only" in validation_runs[0]
    assert not list(docker_visible_temp.glob("regulatory-readiness-snapshot.*"))
