"""Observe the scoped annex worker without task submission or data changes."""

import json
import re
import subprocess

from onyx.background.celery.versioned_apps.regulatory_annex import app
from onyx.regulatory.amendments.annexes.config import source_queue_name


def main() -> None:
    print("SOURCE_QUEUE", source_queue_name(), flush=True)
    for method in ("active", "reserved", "active_queues"):
        inspector = app.control.inspect(timeout=5)
        result = getattr(inspector, method)()
        bounded = {}
        for worker, items in (result or {}).items():
            if method == "active_queues":
                bounded[worker] = [
                    item.get("name")
                    for item in items
                    if str(item.get("name", "")).startswith("regulatory_annex_")
                ]
            else:
                bounded[worker] = [
                    {key: item.get(key) for key in ("id", "name", "time_start")}
                    for item in items
                    if item.get("name")
                    in {
                        "acquire_amendment_sources",
                        "regulatory_document_intake_prepare",
                    }
                ]
        print("WORKER_" + method.upper(), json.dumps(bounded, default=str), flush=True)
    result = subprocess.run(
        ["tail", "-n", "250", "/var/log/onyx/celery_worker_regulatory_annex.log"],
        capture_output=True,
        text=True,
        check=False,
        timeout=5,
    )
    for line in result.stdout.splitlines():
        if re.search(
            r"api.key|authorization|bearer|password|secret|token=", line, re.I
        ):
            print("[credential-bearing line omitted]")
        else:
            print("ANNEX_LOG", line[:500])


if __name__ == "__main__":
    main()
