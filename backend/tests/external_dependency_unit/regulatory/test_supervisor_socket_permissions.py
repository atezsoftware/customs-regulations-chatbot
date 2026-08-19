from __future__ import annotations

import configparser
import hashlib
import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

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
