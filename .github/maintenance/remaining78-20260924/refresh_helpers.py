"""Replace only reviewed task helpers; never launch, stop or retry a worker."""

import hashlib
import json
import os
from pathlib import Path
from typing import Any

BASELINE = {
    "repair_server.py": "cead069eb38b2d9d11eaff4ab19e51ef978f4ebaec5fdb5b3d9f3369bdfd5dba",
    "backpressure.py": "468fcae849e25c8172f0a886b1a492a769d5068fb6ff34902007c3f9357ef8e5",
    "supervisor.py": "40ed4b9be7309278293198391d1ca63fb669d046dcdfc5798e073c25344b5c3d",
    "file-plan.json": "aa19e9b3f169d85858ba87a30edc570a057383307803f2a0041863fdd081efae",
}
REPLACEMENTS = ("bounded_inventory.py", "backpressure.py", "repair_server.py")


def install(
    directory: Path, incoming: Path, *, process_root: Path = Path("/proc")
) -> dict[str, Any]:
    paths = [
        directory / "STOP",
        directory.parent / "dev-remaining78-20260924" / "STOP",
        directory.parent / "dev-remaining78-20260924-resume1" / "STOP",
    ]
    if any(path.exists() for path in paths):
        raise ValueError("STOP is present; refusing helper refresh")
    receipt = json.loads((directory / "launch.json").read_text())
    command = process_root / "343" / "cmdline"
    if (
        receipt.get("pid") != 343
        or not command.exists()
        or str(directory / "supervisor.py").encode()
        not in command.read_bytes().split(b"\x00")
    ):
        raise ValueError("reviewed supervisor343 is no longer running")
    hashes = {
        name: hashlib.sha256((incoming / name).read_bytes()).hexdigest()
        for name in REPLACEMENTS
    }
    for name, expected in BASELINE.items():
        actual = hashlib.sha256((directory / name).read_bytes()).hexdigest()
        if actual != expected and actual != hashes.get(name):
            raise ValueError("reviewed task helper changed: " + name)
    bounded = directory / "bounded_inventory.py"
    if (
        bounded.exists()
        and hashlib.sha256(bounded.read_bytes()).hexdigest() != hashes[bounded.name]
    ):
        raise ValueError("reviewed bounded helper changed")
    for name in REPLACEMENTS:
        target = directory / name
        if (
            target.exists()
            and hashlib.sha256(target.read_bytes()).hexdigest() == hashes[name]
        ):
            continue
        temporary = target.with_suffix(".refresh-tmp")
        with temporary.open("xb") as stream:
            stream.write((incoming / name).read_bytes())
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(target)
    result = {
        "pid": 343,
        "helper_version": "bounded-inventory-v1",
        "installed_sha256": hashes,
    }
    (directory / "helper-refresh.json").write_text(json.dumps(result))
    return result


def main() -> None:
    if (
        not os.environ.get("KUBERNETES_SERVICE_HOST")
        or os.environ.get("POSTGRES_DB") != "customs-regulations-dev"
    ):
        raise ValueError("existing DEV project container required")
    import psycopg2

    with psycopg2.connect(
        host=os.environ["POSTGRES_HOST"],
        port=os.environ.get("POSTGRES_PORT", "5432"),
        dbname=os.environ["POSTGRES_DB"],
        user=os.environ.get("POSTGRES_USER", "postgres"),
        password=os.environ["POSTGRES_PASSWORD"],
        options="-c default_transaction_read_only=on -c statement_timeout=15000",
    ) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT current_database()")
            if cursor.fetchone() != ("customs-regulations-dev",):
                raise ValueError("DEV database identity changed")
            cursor.execute(
                "SELECT value FROM public.key_value_store WHERE key=%s",
                ("regulatory_maintenance:remaining78-20260924-resume2:status",),
            )
            record = cursor.fetchone()
            if (
                not record
                or record[0].get("pid") != 343
                or record[0].get("state") != "running"
            ):
                raise ValueError("reviewed supervisor state changed")
            cursor.execute(
                "SELECT value FROM public.key_value_store WHERE key=%s",
                ("regulatory_maintenance:remaining78-20260924-resume2:control",),
            )
            control = cursor.fetchone()
            if control and control[0].get("stop") is True:
                raise ValueError("STOP control is set")
    incoming = Path(__file__).resolve().parent
    print(json.dumps(install(incoming.parent, incoming)))


if __name__ == "__main__":
    main()
