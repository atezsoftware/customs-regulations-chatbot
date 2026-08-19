from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

BEAT_PROCESS_NAME = "celery_beat_regulatory_indexing"
BEAT_READINESS_PATH = Path(
    "/tmp/onyx_k8s_regulatoryindexingbeat_readiness.txt"  # noqa: S108 — supervised probe contract
)
BEAT_LIVENESS_PATH = Path(
    "/tmp/onyx_k8s_regulatoryindexingbeat_liveness.txt"  # noqa: S108 — supervised probe contract
)

# Scheduler tenant refresh runs once a minute. Two missed refreshes plus one
# 30-second Compose health interval is the maximum accepted liveness age.
BEAT_LIVENESS_MAX_AGE_SECONDS = 150

_MARKER_PATTERN = re.compile(r"^(?P<pid>[1-9][0-9]*):(?P<instance>[0-9a-f]{32})$")
_PID_PATTERN = re.compile(r"\bpid\s+(?P<pid>[1-9][0-9]*),")


class BeatProbeError(RuntimeError):
    pass


def _running_pid(status_text: str) -> int:
    for line in status_text.splitlines():
        fields = line.split()
        if not fields or fields[0] != BEAT_PROCESS_NAME:
            continue
        if len(fields) < 2 or fields[1] != "RUNNING":
            raise BeatProbeError(f"{BEAT_PROCESS_NAME} is not RUNNING")
        match = _PID_PATTERN.search(line)
        if match is None:
            raise BeatProbeError(f"{BEAT_PROCESS_NAME} has no supervisor PID")
        return int(match.group("pid"))
    raise BeatProbeError(f"{BEAT_PROCESS_NAME} is absent from supervisor status")


def _read_marker(path: Path, probe: str) -> re.Match[str]:
    try:
        marker = path.read_text(encoding="utf-8").strip()
    except OSError as error:
        raise BeatProbeError(f"{probe} probe cannot be read: {path}") from error
    match = _MARKER_PATTERN.fullmatch(marker)
    if match is None:
        raise BeatProbeError(f"{probe} probe has an invalid process marker")
    return match


def validate_regulatory_indexing_beat(
    status_text: str,
    *,
    readiness_path: Path = BEAT_READINESS_PATH,
    liveness_path: Path = BEAT_LIVENESS_PATH,
    now: float | None = None,
    max_liveness_age_seconds: int = BEAT_LIVENESS_MAX_AGE_SECONDS,
) -> None:
    supervisor_pid = _running_pid(status_text)
    readiness = _read_marker(readiness_path, "readiness")
    liveness = _read_marker(liveness_path, "liveness")
    if readiness.group(0) != liveness.group(0):
        raise BeatProbeError("readiness and liveness identify different instances")
    if int(readiness.group("pid")) != supervisor_pid:
        raise BeatProbeError(
            "probe marker does not identify the current supervisor PID"
        )

    checked_at = time.time() if now is None else now
    try:
        liveness_age = checked_at - liveness_path.stat().st_mtime
    except OSError as error:
        raise BeatProbeError(
            f"liveness probe cannot be inspected: {liveness_path}"
        ) from error
    if liveness_age < 0 or liveness_age > max_liveness_age_seconds:
        raise BeatProbeError(
            "liveness probe is stale: "
            f"age={liveness_age:.1f}s max={max_liveness_age_seconds}s"
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--status-text")
    parser.add_argument("--readiness-path", type=Path, default=BEAT_READINESS_PATH)
    parser.add_argument("--liveness-path", type=Path, default=BEAT_LIVENESS_PATH)
    parser.add_argument(
        "--max-liveness-age-seconds",
        type=int,
        default=BEAT_LIVENESS_MAX_AGE_SECONDS,
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    status_text = args.status_text
    if status_text is None:
        result = subprocess.run(
            [
                "supervisorctl",
                "-c",
                "/etc/supervisor/conf.d/supervisord.conf",
                "status",
                BEAT_PROCESS_NAME,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            print(result.stdout + result.stderr, file=sys.stderr, end="")
            return 1
        status_text = result.stdout

    try:
        validate_regulatory_indexing_beat(
            status_text,
            readiness_path=args.readiness_path,
            liveness_path=args.liveness_path,
            max_liveness_age_seconds=args.max_liveness_age_seconds,
        )
    except BeatProbeError as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
