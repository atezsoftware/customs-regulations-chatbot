"""One nonblocking DEV stack probe with an owned temporary binary only."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import io
import json
import re
import subprocess
import urllib.request
import zipfile
from pathlib import Path

IMAGE = "255114580789.dkr.ecr.eu-central-1.amazonaws.com/customs-regulations-backend-lite-dev:758bfd31e91e1ab49538aa54ffcb31b790abbc44"
WHEEL_URL = "https://files.pythonhosted.org/packages/78/1f/651389a861f6148754443834a50dd576993257af5f645d4316ab94505267/py_spy-0.4.3-py2.py3-none-manylinux1_x86_64.manylinux_2_5_x86_64.whl"
WHEEL_SHA = "8a4277225f470d394a85cca0cdd944bef87f0097b6f20150a6dd3ac2e92806b8"
BINARY_SHA = "a41205d4925beafd108e1ace85e66c0a1a7adbfcbebcd4441f07848b608886c0"


def exact_worker_argv(data: bytes) -> bool:
    args = data.rstrip(b"\0").split(b"\0")
    prefix = [
        b"/usr/local/bin/python",
        b"/usr/local/bin/celery",
        b"-A",
        b"onyx.background.celery.versioned_apps.user_file_processing",
        b"worker",
    ]
    return (
        args[:5] == prefix
        and len(args) == 8
        and args[5].startswith(b"--hostname=user_file_processing@")
        and args[6] == b"-Q"
        and args[7]
        == b"user_file_processing,user_file_project_sync,user_file_delete,user_file_port"
    )


def remote_probe() -> None:
    import hashlib
    import json
    import os
    import platform
    import secrets
    import stat
    import subprocess
    import sys
    from pathlib import Path

    expected_sha = "a41205d4925beafd108e1ace85e66c0a1a7adbfcbebcd4441f07848b608886c0"
    expected_size = 8094072
    run_id = sys.argv[1]
    if not run_id.isdecimal() or len(run_id) > 20:
        raise SystemExit("Invalid run identity")
    if platform.machine() != "x86_64":
        raise SystemExit("BLOCKED unsupported architecture")
    pids = []
    processes = list(Path("/proc").iterdir())
    if len(processes) > 8192:
        raise SystemExit("BLOCKED process inventory bound")
    for path in processes:
        if not path.name.isdecimal():
            continue
        try:
            with (path / "cmdline").open("rb") as source:
                command = source.read(8193)
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if len(command) <= 8192 and exact_worker_argv(command):
            pids.append(int(path.name))
    if len(pids) != 1:
        raise SystemExit("BLOCKED exact worker PID is not unique")
    pid = pids[0]

    def status_fields(path: Path) -> dict[str, str]:
        with path.open() as source:
            value = source.read(32769)
        if len(value) > 32768:
            raise ValueError("Process status bound")
        return dict(line.split(":", 1) for line in value.splitlines() if ":" in line)

    worker = status_fields(Path(f"/proc/{pid}/status"))
    caller = status_fields(Path("/proc/self/status"))
    uids = [int(value) for value in worker["Uid"].split()]
    gids = [int(value) for value in worker["Gid"].split()]
    if len(set(uids)) != 1 or len(set(gids)) != 1:
        raise SystemExit("BLOCKED mixed worker identities")
    uid, gid = uids[0], gids[0]
    if os.geteuid() not in {0, uid} or (os.geteuid() != 0 and os.getegid() != gid):
        raise SystemExit("BLOCKED caller identity")
    if os.geteuid() != 0 and os.getgroups():
        raise SystemExit("BLOCKED cannot drop supplementary groups")
    try:
        with Path("/proc/sys/kernel/yama/ptrace_scope").open() as source:
            yama = source.read(32).strip()
        if not yama.isdecimal():
            yama = "unavailable"
    except (FileNotFoundError, PermissionError):
        yama = "unavailable"
    preflight = {
        "pid": pid,
        "worker_uid": uid,
        "worker_gid": gid,
        "caller_uid": os.geteuid(),
        "caller_gid": os.getegid(),
        "caller_seccomp": caller.get("Seccomp", "unavailable").strip(),
        "worker_seccomp": worker.get("Seccomp", "unavailable").strip(),
        "ptrace_scope": yama,
        "cap_eff": caller.get("CapEff", "unavailable").strip(),
    }
    data = sys.stdin.buffer.read(expected_size + 1)
    if len(data) != expected_size or hashlib.sha256(data).hexdigest() != expected_sha:
        raise SystemExit("BLOCKED binary identity")
    path = Path("/tmp") / ("intake-pyspy-" + run_id + "-" + secrets.token_hex(16))
    descriptor = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o700
    )
    owned = os.fstat(descriptor)
    cleanup = False
    result = None
    failure = None
    try:
        with os.fdopen(descriptor, "wb") as target:
            target.write(data)
            target.flush()
            if os.geteuid() == 0:
                os.fchown(target.fileno(), uid, gid)
            os.fchmod(target.fileno(), 0o700)
        with path.open("rb") as binary:
            if (
                hashlib.sha256(binary.read(expected_size + 1)).hexdigest()
                != expected_sha
            ):
                raise ValueError("Copied binary identity")
        drop_identity = os.geteuid() == 0
        with Path(f"/proc/{pid}/cmdline").open("rb") as source:
            if not exact_worker_argv(source.read(8193)):
                raise ValueError("Worker PID identity changed")
        result = subprocess.run(
            [str(path), "dump", "--nonblocking", "--json", "--pid", str(pid)],
            capture_output=True,
            timeout=15,
            check=False,
            user=uid if drop_identity else None,
            group=gid if drop_identity else None,
            extra_groups=[] if drop_identity else None,
        )
    except subprocess.TimeoutExpired:
        failure = "BLOCKED profiler timeout; only profiler child terminated"
    except PermissionError:
        failure = "BLOCKED existing policy denied probe"
    finally:
        current = path.lstat()
        if stat.S_ISREG(current.st_mode) and (current.st_dev, current.st_ino) == (
            owned.st_dev,
            owned.st_ino,
        ):
            path.unlink()
            cleanup = True
    report: dict[str, object] = {
        "preflight": preflight,
        "ephemeral_binary_write": True,
        "owned_binary_removed": cleanup,
        "worker_signals_sent": 0,
        "permission_changes": 0,
        "application_db_provider_writes": 0,
        "nonblocking": True,
        "locals_requested": False,
        "probe_uid": uid,
        "probe_gid": gid,
    }
    if failure:
        report["status"] = failure
    elif result is None or result.returncode:
        report["status"] = "BLOCKED profiler failed; no fallback"
        report["exit_code"] = result.returncode if result else None
        report["diagnostic"] = (
            result.stderr[:4096].decode(errors="replace") if result else ""
        )
    else:
        if len(result.stdout) > 512 * 1024:
            raise SystemExit("BLOCKED stack response bound")
        traces = json.loads(result.stdout)
        if not isinstance(traces, list) or len(traces) > 256:
            raise SystemExit("BLOCKED thread response bound")
        threads = []
        for trace in traces:
            frames = trace.get("frames", [])
            if not isinstance(frames, list) or len(frames) > 256:
                raise SystemExit("BLOCKED frame response bound")
            if any(frame.get("locals") for frame in frames):
                raise SystemExit("BLOCKED unexpected locals")
            threads.append(
                {
                    "thread_id": trace.get("thread_id"),
                    "active": trace.get("active"),
                    "owns_gil": trace.get("owns_gil"),
                    "frames": [
                        {key: frame.get(key) for key in ("name", "filename", "line")}
                        for frame in frames
                    ],
                }
            )
        report["status"] = "stack_snapshot"
        report["threads"] = threads
    encoded = json.dumps(report)
    if len(encoded.encode()) > 512 * 1024:
        raise SystemExit("BLOCKED sanitized report bound")
    print(encoded)


def sanitized(text: str) -> str:
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)
    text = re.sub(r"(?i)\bBearer\s+[^\s\"'<>]+", "Bearer [REDACTED]", text)
    text = re.sub(
        r"(?i)([\"']?(?:authorization|api[_-]?key|access[_-]?token|refresh[_-]?token|password|secret(?:_access_key)?|private_key|client_secret|credential)[\"']?\s*[:=]\s*)(?:[\"'][^\"']*[\"']|[^\s,;}]+)",
        r"\1[REDACTED]",
        text,
    )
    return re.sub(r"(https?://[^\s?\"']+)\?[^\s\"']+", r"\1?[REDACTED_QUERY]", text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pods", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--remaining-files", type=int, required=True)
    args = parser.parse_args()
    if args.remaining_files == 0:
        print("SKIPPED all28 confirmed complete by dispatching operator")
        return
    if args.remaining_files not in {1, 2} or not re.fullmatch(
        r"[0-9]{1,20}", args.run_id
    ):
        raise SystemExit(
            "Fresh operator confirmation of one or two remaining files required"
        )
    if args.pods.stat().st_size > 1024 * 1024:
        raise SystemExit("Pod inventory bound")
    targets: list[tuple[str, str]] = []
    for pod in json.loads(args.pods.read_text())["items"]:
        if (
            pod["metadata"].get("labels", {}).get("app")
            != "dev-customs-regulations-background"
            or pod["metadata"].get("deletionTimestamp")
            or pod["status"]["phase"] != "Running"
        ):
            continue
        ready = {
            item["name"]
            for item in pod["status"].get("containerStatuses", [])
            if item.get("ready")
        }
        for container in pod["spec"]["containers"]:
            if container["image"] == IMAGE and container["name"] in ready:
                targets.append((pod["metadata"]["name"], container["name"]))
    if len(targets) != 1:
        raise SystemExit("Exactly one Ready758 DEV background target required")
    with urllib.request.urlopen(WHEEL_URL, timeout=20) as response:
        if not response.geturl().startswith("https://files.pythonhosted.org/"):
            raise SystemExit("Unexpected wheel origin")
        wheel = response.read(2936518)
    if len(wheel) != 2936517 or hashlib.sha256(wheel).hexdigest() != WHEEL_SHA:
        raise SystemExit("Pinned wheel identity changed")
    with zipfile.ZipFile(io.BytesIO(wheel)) as archive:
        name = "py_spy-0.4.3.data/scripts/py-spy"
        if archive.getinfo(name).file_size != 8094072:
            raise SystemExit("Pinned binary size changed")
        binary = archive.read(name)
    if hashlib.sha256(binary).hexdigest() != BINARY_SHA:
        raise SystemExit("Pinned binary hash changed")
    program = (
        inspect.getsource(exact_worker_argv)
        + "\n"
        + inspect.getsource(remote_probe)
        + "\nremote_probe()\n"
    )
    pod, container = targets[0]
    result = subprocess.run(
        [
            "kubectl",
            "exec",
            "-i",
            pod,
            "-n",
            "customs-regulations-dev",
            "-c",
            container,
            "--",
            "python",
            "-c",
            program,
            args.run_id,
        ],
        input=binary,
        capture_output=True,
        timeout=45,
        check=False,
    )
    if len(result.stdout) > 512 * 1024 or len(result.stderr) > 8192:
        raise SystemExit("Remote response bound")
    print("TARGET", pod, container, IMAGE)
    print(
        "EPHEMERAL_PROBE_NOT_FULLY_READ_ONLY remaining_confirmation="
        + str(args.remaining_files)
    )
    print(sanitized(result.stdout.decode(errors="replace")))
    if result.returncode:
        print(sanitized(result.stderr.decode(errors="replace")))
        raise SystemExit("BLOCKED remote probe failed; no fallback")


if __name__ == "__main__":
    main()
