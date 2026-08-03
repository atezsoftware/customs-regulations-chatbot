import os
import subprocess
import sys
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parents[4]
_SUPERVISORD_CONFIG = _BACKEND_ROOT / "supervisord.conf"


def test_supervisord_starts_watchdog_as_module() -> None:
    config = _SUPERVISORD_CONFIG.read_text()

    assert "command=python -m onyx.utils.supervisord_watchdog\n" in config
    assert "command=python onyx/utils/supervisord_watchdog.py\n" not in config


def test_watchdog_module_entrypoint_does_not_shadow_stdlib_modules() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_BACKEND_ROOT)

    result = subprocess.run(
        [sys.executable, "-m", "onyx.utils.supervisord_watchdog", "--help"],
        cwd=_BACKEND_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Supervisord Watchdog" in result.stdout
