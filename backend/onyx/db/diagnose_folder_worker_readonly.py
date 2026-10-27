"""Observe the scoped annex worker without task submission or data changes."""

import json
import re
import subprocess

from onyx.background.celery.versioned_apps.regulatory_annex import app
from onyx.configs import app_configs
from onyx.regulatory.amendments.annexes import config as annex_config
from onyx.regulatory.amendments.annexes.config import source_queue_name

OWNED_FILES = (
    "10e927dc-a124-4c6a-85b1-e3619b61b6d3",
    "a3c6e8cf-93a9-460b-9edf-a9cdd4ba05a9",
)


def main() -> None:
    print("INTAKE_RECOVERY_GATE", json.dumps({
        "enabled": annex_config.REGULATORY_ANNEX_WORKER_ENABLED,
        "environment": annex_config.REGULATORY_ANNEX_ENVIRONMENT,
        "database_identity": annex_config.ANNEX_DATABASE_IDENTITY,
    }), flush=True)
    control_tail = subprocess.run(
        ["tail", "-n", "800", "/var/log/onyx/celery_worker_light.log"],
        check=False, capture_output=True, text=True,
    )
    control_lines = [line for line in control_tail.stdout.splitlines() if "recover_document_intakes" in line]
    print("INTAKE_RECOVERY_LIGHT_LOG_COUNTS", json.dumps({
        "received": sum("received" in line.lower() for line in control_lines),
        "succeeded": sum("succeeded" in line.lower() for line in control_lines),
        "failed": sum("failed" in line.lower() or "scope mismatch" in line.lower() for line in control_lines),
    }), flush=True)
    registered = app.control.inspect(timeout=5).registered() or {}
    print(
        "INTAKE_RECOVERY_REGISTERED",
        json.dumps({
            worker: any(str(name).split(" ", 1)[0] == "recover_document_intakes" for name in tasks)
            for worker, tasks in registered.items()
            if worker.startswith("light@")
        }),
        flush=True,
    )
    print(
        "CONTEXTUAL_RUNTIME_LIMITS",
        json.dumps(
            {
                name: getattr(app_configs, name)
                for name in (
                    "REGULATORY_BATCH_INDEXING_ENABLED",
                    "ENABLE_CONTEXTUAL_RAG",
                    "CONTEXTUAL_RAG_MAX_WORKERS",
                    "CONTEXTUAL_RAG_MAX_RETRIES",
                    "CONTEXTUAL_RAG_LLM_TIMEOUT_SECONDS",
                    "CONTEXTUAL_RAG_RETRY_BASE_SECONDS",
                    "CONTEXTUAL_RAG_RETRY_MAX_SECONDS",
                    "USE_DOCUMENT_SUMMARY",
                    "USE_CHUNK_SUMMARY",
                    "MAX_TOKENS_FOR_FULL_INCLUSION",
                )
            }
        ),
        flush=True,
    )
    print("SOURCE_QUEUE", source_queue_name(), flush=True)
    for method in ("active", "reserved", "active_queues"):
        inspector = app.control.inspect(timeout=5)
        result = getattr(inspector, method)()
        bounded = {}
        for worker, items in (result or {}).items():
            if method in {"active", "reserved"}:
                for item in items:
                    if item.get("name") != "index_single_user_file":
                        continue
                    arguments = json.dumps(
                        [item.get("args"), item.get("kwargs")], default=str
                    )
                    owned = [
                        identifier
                        for identifier in OWNED_FILES
                        if identifier in arguments
                    ]
                    if owned:
                        print(
                            "OWNED_INDEX_WORKER",
                            json.dumps(
                                {
                                    "worker": worker,
                                    "inspection": method,
                                    "file_ids": owned,
                                    **{
                                        key: item.get(key)
                                        for key in ("id", "name", "time_start")
                                    },
                                },
                                default=str,
                            ),
                            flush=True,
                        )
            if method == "active_queues":
                if worker.startswith("light@"):
                    print(
                        "INTAKE_RECOVERY_CONTROL_QUEUE",
                        json.dumps({"worker": worker, "consumes_checkpoint_cleanup": any(item.get("name") == "checkpoint_cleanup" for item in items)}),
                        flush=True,
                    )
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
    result = subprocess.run(
        ["tail", "-n", "160", "/var/log/onyx/celery_worker_user_file_processing.log"],
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
        elif re.search(
            r"context|llm|litellm|retry|429|timeout|index_single|3521feda|a3c6e8cf|10e927dc",
            line,
            re.I,
        ):
            print("INDEX_PROGRESS_LOG", line[:500])


if __name__ == "__main__":
    main()
