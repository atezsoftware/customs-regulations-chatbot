"""Fixed DEV cutover probe. Database transactions end before index I/O."""

import argparse
import json
import os
import re
import sys
import time
from typing import Any

from sqlalchemy import text

from onyx.db.engine.sql_engine import SqlEngine, get_session_with_tenant
from shared_configs.configs import MULTI_TENANT


class CutoverRefusal(RuntimeError):
    """A fixed diagnostic that is safe to expose in runner logs."""


def configured_indices() -> list[str]:
    if MULTI_TENANT or os.environ.get("POSTGRES_DB") != "customs-regulations-dev":
        raise CutoverRefusal("dev_database_required")
    with (
        SqlEngine.scoped_engine(pool_size=2, max_overflow=0),
        get_session_with_tenant(tenant_id="public") as session,
    ):
        session.execute(text("SET TRANSACTION READ ONLY"))
        if (
            session.execute(text("SELECT current_database()")).scalar_one()
            != "customs-regulations-dev"
        ):
            raise CutoverRefusal("dev_database_required")
        rows = (
            session.execute(
                text(
                    "SELECT index_name FROM search_settings WHERE status IN ('PRESENT', 'FUTURE') ORDER BY index_name"
                )
            )
            .scalars()
            .all()
        )
        names = [str(row) for row in rows]
    if (
        not names
        or len(names) != len(set(names))
        or any(
            not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", name) or "_dev_" not in name
            for name in names
        )
    ):
        raise CutoverRefusal("exact_dev_indices_required")
    return names


def inspect_indices(
    client: Any, names: list[str], *, allow_write_block: bool = True
) -> dict[str, str]:
    result: dict[str, str] = {}
    for name in names:
        metadata = client.indices.get(index=name, flat_settings=True)
        if set(metadata) != {name}:
            raise CutoverRefusal("physical_index_required")
        item = metadata[name]
        settings = item["settings"]
        if (
            not allow_write_block
            and str(settings.get("index.blocks.write", "false")).lower() == "true"
        ):
            raise CutoverRefusal("existing_write_block_requires_review")
        if (
            item.get("aliases")
            or item.get("data_stream")
            or any(key.startswith("index.lifecycle.") for key in settings)
        ):
            raise CutoverRefusal("unmanaged_exact_index_required")
        result[name] = settings["index.uuid"]
    return result


def validate_scope_evidence(
    client: Any, evidence: dict[str, Any], names: list[str]
) -> None:
    if (
        evidence.get("mode") not in {"dev-dedicated", "shared-scoped"}
        or evidence.get("expires_at", 0) <= time.time()
        or not evidence.get("evidence_ref")
        or evidence.get("indices") != names
        or evidence.get("cluster_uuid") != client.info()["cluster_uuid"]
    ):
        raise CutoverRefusal("authoritative_DEV_ES_scope_evidence_required")
    if evidence["mode"] == "shared-scoped" and (
        evidence.get("producer_inventory_complete") is not True
        or evidence.get("index_lifecycle_idle") is not True
        or not isinstance(evidence.get("task_ids"), list)
    ):
        raise CutoverRefusal("authoritative_shared_scope_evidence_required")


def drain_server_work(client: Any, evidence: dict[str, Any]) -> None:
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        if evidence["mode"] == "dev-dedicated":
            # Only a positively identified DEV-exclusive cluster permits this inventory.
            response = client.tasks.list(
                actions=[
                    "indices:data/write/*",
                    "indices:admin/create",
                    "indices:admin/delete",
                    "indices:admin/rollover",
                    "indices:admin/mapping/put",
                    "indices:admin/settings/update",
                ],
                detailed=False,
            )
            if response.get("node_failures") or response.get("task_failures"):
                raise CutoverRefusal("server_task_inventory_incomplete")
            active = any(
                node.get("tasks") for node in response.get("nodes", {}).values()
            )
            pending = client.cluster.pending_tasks().get("tasks", [])
            if not active and not pending:
                return
        else:
            task_ids = evidence["task_ids"]
            if any(
                not re.fullmatch(r"[A-Za-z0-9_-]+:[0-9]+", task_id)
                for task_id in task_ids
            ):
                raise CutoverRefusal("exact_DEV_task_ids_required")
            if all(
                client.tasks.get(task_id=task_id).get("completed") is True
                for task_id in task_ids
            ):
                return
        time.sleep(2)
    raise CutoverRefusal("server_work_not_drained")


def operate(
    phase: str, expected: dict[str, str] | None, evidence: dict[str, Any]
) -> dict[str, Any]:
    from onyx.document_index.elasticsearch.client import ElasticsearchClient

    names = configured_indices()
    with ElasticsearchClient() as wrapper:
        client = wrapper.publication_client()
        if phase == "inventory":
            return {
                "cluster_uuid": client.info()["cluster_uuid"],
                "indices": inspect_indices(client, names, allow_write_block=False),
            }
        validate_scope_evidence(client, evidence, names)
        current = inspect_indices(client, names, allow_write_block=phase != "inspect")
        if phase == "inspect":
            return current
        if not expected or current != expected:
            raise CutoverRefusal("physical_index_changed")
        drain_server_work(client, evidence)
        for name in names:
            if phase == "block":
                response = client.indices.add_block(
                    index=name, block="write", timeout="60s", master_timeout="60s"
                )
                indices = response.get("indices", [])
                if (
                    response.get("acknowledged") is not True
                    or response.get("shards_acknowledged") is not True
                    or len(indices) != 1
                    or indices[0].get("name") != name
                    or indices[0].get("blocked") is not True
                ):
                    raise CutoverRefusal("write_barrier_unacknowledged")
            elif phase == "unblock":
                response = client.indices.put_settings(
                    index=name,
                    settings={"index.blocks.write": None},
                    timeout="60s",
                    master_timeout="60s",
                )
                if response.get("acknowledged") is not True:
                    raise CutoverRefusal("unblock_unacknowledged")
            else:
                raise CutoverRefusal("unsupported_phase")
        if inspect_indices(client, names) != expected:
            raise CutoverRefusal("physical_index_changed")
        return current


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("inventory", "inspect", "block", "unblock"))
    args = parser.parse_args()
    payload = json.load(sys.stdin)
    print(
        json.dumps(
            operate(args.phase, payload.get("indices"), payload.get("scope", {})),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except CutoverRefusal as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
    except Exception:
        print("DEV_CUTOVER_INDEX_REFUSED", file=sys.stderr)
        sys.exit(1)
