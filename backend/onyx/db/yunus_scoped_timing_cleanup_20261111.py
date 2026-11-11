"""Stop only the known expired diagnostic; source and labeling workers are separate."""

import json
import os
import signal
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import cast


def main() -> None:
    lower = datetime(2026, 11, 11, 9, 48, 42, tzinfo=timezone.utc).timestamp()
    upper = datetime(2026, 11, 11, 9, 48, 55, tzinfo=timezone.utc).timestamp()
    boot = next(
        int(line.split()[1])
        for line in Path("/proc/stat").read_text().splitlines()
        if line.startswith("btime ")
    )
    tick_rate = os.sysconf("SC_CLK_TCK")
    matches = []
    for directory in Path("/proc").iterdir():
        if not directory.name.isdigit() or int(directory.name) == os.getpid():
            continue
        try:
            args = directory.joinpath("cmdline").read_bytes().split(b"\0")
            args = [value.decode() for value in args if value]
            if len(args) != 2 or Path(args[0]).name != "python" or args[1] != "-":
                continue
            fields = directory.joinpath("stat").read_text().rsplit(") ", 1)[1].split()
            started = boot + int(fields[19]) / tick_rate
            if lower <= started <= upper:
                matches.append((int(directory.name), started))
        except (FileNotFoundError, ProcessLookupError):
            continue
    if len(matches) > 1:
        raise ValueError(
            "More than one diagnostic matches the exact start window; nothing stopped"
        )
    print("SCOPED_EXPIRED_PROCESS_INVENTORY", json.dumps(matches), flush=True)
    open_pid = cast(Callable[[int], int], getattr(os, "pidfd_open"))
    send_signal = cast(Callable[[int, int], None], getattr(signal, "pidfd_send_signal"))
    for pid, started in matches:
        descriptor = open_pid(pid)
        try:
            fields = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
            if boot + int(fields[19]) / tick_rate != started:
                raise ValueError("Diagnostic process identity changed")
            send_signal(descriptor, signal.SIGTERM)
        finally:
            os.close(descriptor)
        print("SCOPED_EXPIRED_PROCESS_TERMINATED", pid, flush=True)


if __name__ == "__main__":
    main()
