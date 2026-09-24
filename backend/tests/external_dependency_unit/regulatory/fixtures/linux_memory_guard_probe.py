"""Synthetic Linux-only checks; no application data, network, or database access."""

import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from onyx.regulatory.amendments.memory_budget import (
    MIB,
    MemoryPolicy,
    ResourcePressure,
    read_memory,
)
from onyx.regulatory.amendments.supervision import supervise


class LinuxMemoryGuardProbe(unittest.TestCase):
    def test_memory_pressure_stops_before_container_oom(self) -> None:
        reading = read_memory()
        assert reading is not None and reading.limit == 256 * MIB
        events = Path("/sys/fs/cgroup/memory.events")
        before = events.read_text()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ResourcePressure) as raised:
                supervise(
                    [
                        sys.executable,
                        "-c",
                        "import time; blocks=[]\nfor i in range(384):\n blocks.append(bytearray(1024*1024)); time.sleep(.01)",
                    ],
                    lock_path=Path(directory) / "slot",
                    policy=MemoryPolicy(reserve_bytes=32 * MIB, item_bytes=8 * MIB),
                )
            assert raised.exception.started
        after = events.read_text()

        def oom_count(text: str) -> dict[str, int]:
            return {
                k: int(v)
                for k, v in (line.split() for line in text.splitlines())
                if k.startswith("oom")
            }

        assert oom_count(before) == oom_count(after), (before, after)
        print("Container pressure stopped child; OOM counters unchanged.", flush=True)

    def test_parent_death_terminates_analysis_child(self) -> None:
        from onyx.regulatory.amendments.supervision import protect_parent_lifetime

        assert callable(protect_parent_lifetime)
        with tempfile.TemporaryDirectory() as directory:
            pidfile = Path(directory) / "child.pid"
            child_code = (
                "from onyx.regulatory.amendments.supervision import protect_parent_lifetime; "
                "protect_parent_lifetime(); import os,time; "
                f'open({str(pidfile)!r}, "w").write(str(os.getpid())); time.sleep(30)'
            )
            parent_code = (
                "from onyx.regulatory.amendments.supervision import supervise; "
                "from onyx.regulatory.amendments.memory_budget import MemoryPolicy,MIB; "
                "from pathlib import Path; import sys; "
                f'supervise([sys.executable,"-c",{child_code!r}], '
                f"lock_path=Path({str(Path(directory) / 'slot')!r}), "
                "policy=MemoryPolicy(reserve_bytes=32*MIB,item_bytes=8*MIB))"
            )
            parent = subprocess.Popen([sys.executable, "-c", parent_code])
            child_pid = None
            try:
                deadline = time.monotonic() + 5
                while not pidfile.exists() and time.monotonic() < deadline:
                    time.sleep(0.02)
                assert pidfile.exists(), "child did not start"
                child_pid = int(pidfile.read_text())
                parent.kill()
                parent.wait(timeout=2)
                deadline = time.monotonic() + 2
                while time.monotonic() < deadline:
                    try:
                        state = Path(f"/proc/{child_pid}/stat").read_text().split()[2]
                    except FileNotFoundError:
                        break
                    if state == "Z":
                        break
                    time.sleep(0.02)
                else:
                    self.fail("child survived its supervisor")
                print("Kernel parent-death signal stopped orphan child.", flush=True)
            finally:
                if parent.poll() is None:
                    parent.kill()
                    parent.wait(timeout=2)
                if child_pid is not None:
                    try:
                        os.kill(child_pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass


if __name__ == "__main__":
    unittest.main(verbosity=2)
