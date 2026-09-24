"""One-shot DEV repair proposal. No local execution or automatic failed-file retry."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast
from uuid import UUID, uuid4

from backpressure import GuardedClient
from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan
from image_correction import FILE_ID as IMAGE_FILE_ID
from image_correction import PARENTS, corrected_rows, prepare_image_correction
from scripts.prepare_regulatory_publication_baselines import read_file_inventory

from onyx.db.engine.sql_engine import SqlEngine
from onyx.db.regulatory_maintenance import claim_regulatory_maintenance
from onyx.db.regulatory_publication import PublicationStore
from onyx.db.regulatory_publication_baseline import (
    baseline_audit_inputs,
    baseline_audit_settings,
)
from onyx.db.regulatory_writer_publication import load_owned_writer_inputs
from onyx.document_index.elasticsearch.client import ElasticsearchClient
from onyx.document_index.publication_models import (
    PublicationScope,
    publication_digest,
    publication_source,
    publication_streaming_digest,
)
from onyx.file_store.file_store import get_default_file_store
from onyx.key_value_store.factory import get_kv_store
from onyx.key_value_store.interface import KvKeyNotFoundError
from onyx.regulatory.amendments.annexes import config
from onyx.regulatory.amendments.annexes.models import AnnexCanonicalSnapshot
from onyx.regulatory.amendments.annexes.publication_execution import (
    LEASE_TTL,
    publication_heartbeat,
)
from onyx.regulatory.amendments.annexes.selective_impact import (
    recover_source_membership,
    source_ids,
)
from onyx.regulatory.publication_baseline import (
    audit_baseline_inventory,
    observed_index_snapshot,
    prepare_owned_baseline,
)
from onyx.regulatory.restored_index_evidence import verified_restored_index_source
from onyx.regulatory.source_metadata_repair import repair_canonical_source_metadata
from onyx.regulatory.writer_projection import prepare_owned_correction
from onyx.regulatory.writer_publication import execute_writer_publication
from onyx.utils.variable_functionality import set_is_ee_based_on_env_variable
from shared_configs.contextvars import CURRENT_TENANT_ID_CONTEXTVAR

DATABASE = "customs-regulations-dev"
INDEX = "danswer_chunk_dev_gemini_embedding_2_1024"
INDEX_UUID = "q8lSz7g2Rvq7739qGJz6jg"
OLD_UUID = "umb9_zfJRgW_MYV9xqClsg"
PREFIX = "regulatory_maintenance:remaining78-20260924-resume1:"
OLD_PREFIX = "regulatory_maintenance:remaining78-20260924:"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def protected_canonical(rows: list[AnnexCanonicalSnapshot]) -> str:
    return publication_digest(
        [
            row.model_dump(mode="json", exclude={"heading_path", "metadata"})
            for row in rows
            if row.id not in PARENTS
        ]
    )


def source_fingerprint(hits: list[dict[str, Any]]) -> str:
    sources = [
        publication_source(json.dumps(hit["_source"]))
        for hit in hits
        if hit["_source"].get("regulatory_chunk_id") not in PARENTS
    ]
    return publication_digest(sorted(sources, key=lambda row: row["chunk_index"]))


def inventory(client: Elasticsearch, file_id: UUID) -> list[dict[str, Any]]:
    info = client.indices.get(index=INDEX)
    if set(info) != {INDEX} or info[INDEX]["settings"]["index"]["uuid"] != INDEX_UUID:
        raise ValueError("DEV physical index identity changed")
    return list(
        scan(
            client,
            index=INDEX,
            query={
                "seq_no_primary_term": True,
                "query": {"term": {"document_id": str(file_id)}},
            },
        )
    )


def process_file(client: Elasticsearch, plan: dict[str, Any]) -> dict[str, Any]:
    file_id = UUID(plan["file_id"])
    before = baseline_audit_inputs("public", file_id)
    if before.pending_manifest or before.gate_closed:
        if str(file_id) != "12328001-2f21-407e-b230-7d50a8dbf28f":
            raise ValueError("unreviewed pending publication requires inspection")
        from onyx.db.regulatory_writer_publication import pending_writer_manifest

        scope = PublicationScope(
            tenant_id="public",
            environment=config.REGULATORY_ANNEX_ENVIRONMENT,
            database_identity=config.ANNEX_DATABASE_IDENTITY,
        )
        authority = PublicationStore(scope)
        owner = authority.acquire(file_id, owner_id=uuid4(), ttl=LEASE_TTL)
        try:
            with publication_heartbeat(owner):
                frozen = pending_writer_manifest(owner)
                if (
                    frozen is None
                    or str(frozen.id) != plan.get("reviewed_pending_id")
                    or frozen.kind != "baseline"
                    or frozen.canonical_after is None
                ):
                    raise ValueError("reviewed recovery manifest changed")
                if (
                    publication_streaming_digest(frozen.model_dump(mode="json"))
                    != plan["reviewed_pending_sha"]
                ):
                    raise ValueError("reviewed frozen manifest checksum changed")
                if protected_canonical(frozen.canonical_after) != protected_canonical(
                    before.canonical
                ):
                    raise ValueError("recovery would change protected source")
                execute_writer_publication(
                    owner,
                    cast(
                        Elasticsearch,
                        GuardedClient(client, lambda: authority.reservations(owner)),
                    ),
                )
        finally:
            authority.release(owner)
        result = process_file(client, plan)
        result["recovered_manifest_id"] = str(frozen.id)
        return result

    settings = baseline_audit_settings("public", DATABASE)
    current = [item for item in settings if item.status.is_current()]
    if len(settings) != 1 or len(current) != 1 or current[0].index_name != INDEX:
        raise ValueError("DEV active model changed")
    index = observed_index_snapshot(current[0], INDEX_UUID)
    initial_hits = inventory(client, file_id)
    before_hash = protected_canonical(before.canonical)
    source_hash = source_fingerprint(initial_hits)
    if len(before.canonical) != plan["canonical_count"]:
        raise ValueError("canonical count differs from approved selection")
    # A previously finished file is only skipped after full evidence validation.
    if initial_hits and all(
        (hit["_source"].get("publication_evidence") or {})
        .get("index", {})
        .get("index_uuid")
        == INDEX_UUID
        for hit in initial_hits
    ):
        audit = audit_baseline_inventory(
            before.canonical,
            read_file_inventory(client, index, "public", file_id, hits=initial_hits),
            before.bindings,
        )
        if audit.state == "ready":
            return {
                "file_id": str(file_id),
                "state": "ready",
                "already_ready": True,
                "indexed_count": audit.indexed_count,
                "new_indexed_count": 0,
            }
    scope = PublicationScope(
        tenant_id="public",
        environment=config.REGULATORY_ANNEX_ENVIRONMENT,
        database_identity=config.ANNEX_DATABASE_IDENTITY,
    )
    authority = PublicationStore(scope)
    owner = authority.acquire(file_id, owner_id=uuid4(), ttl=LEASE_TTL)
    try:
        client = cast(
            Elasticsearch, GuardedClient(client, lambda: authority.reservations(owner))
        )
        with publication_heartbeat(owner) as lost:
            inputs = load_owned_writer_inputs(owner)
            if inputs.canonical != before.canonical:
                raise ValueError("canonical changed while acquiring ownership")
            fresh_hits = inventory(client, file_id)
            if str(file_id) == IMAGE_FILE_ID and publication_digest(
                sorted(initial_hits, key=lambda h: h["_id"])
            ) != publication_digest(sorted(fresh_hits, key=lambda h: h["_id"])):
                raise ValueError("image evidence changed while acquiring ownership")
            if source_fingerprint(fresh_hits) != source_hash:
                raise ValueError("index changed while acquiring ownership")
            manifest = None
            action = plan["action"]
            if action == "missing":
                if fresh_hits or inputs.bindings:
                    raise ValueError("missing-only repair refuses existing projections")
                # Existing canonical units are authoritative: never parse the raw file.
                manifest = prepare_owned_correction(
                    owner,
                    client,
                    inputs,
                    inputs.canonical,
                    changed_id=None,
                    target_settings_ids={current[0].id},
                )
                # Initial qualification uses bounded bulk with identical fencing/proofs.
                manifest = manifest.model_copy(update={"kind": "baseline"})
                if manifest.canonical_after != inputs.canonical:
                    raise ValueError(
                        "missing projection preparation changed canonical records"
                    )
            elif action == "metadata":
                expected = {
                    hit["_source"]["regulatory_chunk_id"]: hit["_source"].get(
                        "heading_path"
                    )
                    or []
                    for hit in fresh_hits
                }
                recovered = recover_source_membership(inputs.canonical)
                known_ids = {row.id for row in inputs.canonical}
                images = [
                    row
                    for row in inputs.canonical
                    if row.metadata.get("bound_to_regulatory_chunk_id")
                    and row.id not in recovered
                    and not set(source_ids(row)) <= known_ids
                ]
                needs_source = bool(images) or any(
                    row.id in expected and row.heading_path != expected[row.id]
                    for row in inputs.canonical
                )
                markdown = None
                assets: dict[str, str] = {}
                if needs_source:
                    if not inputs.file.name.lower().endswith(".md"):
                        raise ValueError(
                            "source metadata repair requires the original Markdown"
                        )
                    store = get_default_file_store()
                    with store.read_file(inputs.file.file_id, mode="rb") as source:
                        markdown = source.read()
                    for row in images:
                        asset = row.metadata.get("image_file_id")
                        if isinstance(asset, str):
                            with store.read_file(asset, mode="rb") as image:
                                assets[asset] = hashlib.sha256(image.read()).hexdigest()
                if str(file_id) == IMAGE_FILE_ID:
                    if markdown is None:
                        raise ValueError("image correction requires original Markdown")
                    after = corrected_rows(inputs.canonical, markdown, assets, expected)
                    backup = {
                        "canonical": [
                            row.model_dump(mode="json")
                            for row in inputs.canonical
                            if row.id in PARENTS
                        ],
                        "index_hits": [
                            hit
                            for hit in fresh_hits
                            if hit["_source"].get("regulatory_chunk_id") in PARENTS
                        ],
                    }
                    kv = get_kv_store()
                    kv.store(PREFIX + "image_backup", backup)
                    if kv.load(PREFIX + "image_backup", refresh_cache=True) != backup:
                        raise ValueError("two-image archive readback differs")
                    manifest = prepare_image_correction(owner, client, inputs, after)
                else:
                    after = repair_canonical_source_metadata(
                        inputs.canonical,
                        markdown=markdown,
                        asset_sha256=assets,
                        indexed_headings=expected,
                    )
                    manifest = prepare_owned_baseline(
                        owner, client, inputs, canonical_after=after
                    )
            elif action == "restored_index":
                by_ordinal = {
                    binding.projection.ordinal: binding
                    for binding in inputs.bindings
                    if binding.index.index_uuid == INDEX_UUID
                }
                if (
                    len(by_ordinal) != len(fresh_hits)
                    or len(by_ordinal) != plan["canonical_count"]
                ):
                    raise ValueError(
                        "restored binding coverage differs from search inventory"
                    )
                replacements = []
                for hit in fresh_hits:
                    original = hit["_source"]
                    if (
                        original.get("publication_scope") != authority.scope_key
                        or original.get("publication_floor", owner.fencing_token)
                        >= owner.fencing_token
                    ):
                        raise ValueError(
                            "restored evidence crosses publication ownership"
                        )
                    repaired = verified_restored_index_source(
                        original,
                        by_ordinal[original["chunk_index"]],
                        old_index_uuid=OLD_UUID,
                    )
                    if original != repaired:
                        replacements.extend(
                            [
                                {
                                    "update": {
                                        "_index": INDEX,
                                        "_id": hit["_id"],
                                        "if_seq_no": hit["_seq_no"],
                                        "if_primary_term": hit["_primary_term"],
                                    }
                                },
                                {
                                    "doc": {
                                        "publication_evidence": repaired[
                                            "publication_evidence"
                                        ],
                                        "publication_payload": repaired[
                                            "publication_payload"
                                        ],
                                    }
                                },
                            ]
                        )
                if replacements:
                    if lost.is_set():
                        raise ValueError(
                            "ownership heartbeat lost before identity repair"
                        )
                    authority.reservations(owner)
                    response = client.bulk(operations=replacements, refresh="wait_for")
                    if response.get("errors"):
                        raise ValueError(
                            "index changed during identity repair; inspect before retry"
                        )
            else:
                raise ValueError("unrecognized repair action")
            if manifest is not None:
                if any(
                    item.index_name != INDEX or item.index_uuid != INDEX_UUID
                    for item in manifest.indexes
                ):
                    raise ValueError(
                        "prepared manifest differs from the approved DEV index"
                    )
                if lost.is_set():
                    raise ValueError("ownership heartbeat lost before publication")
                execute_writer_publication(owner, client, manifest)
        after_inputs = baseline_audit_inputs("public", file_id)
        after_hits = inventory(client, file_id)
        if str(file_id) == IMAGE_FILE_ID:
            if manifest is None or after_inputs.canonical != manifest.canonical_after:
                raise ValueError(
                    "image correction differs from approved frozen manifest"
                )
            original_rows = {row.id: row for row in before.canonical}
            for row in after_inputs.canonical:
                if row.id in PARENTS:
                    old = original_rows[row.id]
                    if row.model_dump(
                        exclude={"text", "heading_path", "metadata"}
                    ) != old.model_dump(exclude={"text", "heading_path", "metadata"}):
                        raise ValueError("image identity or legal dates changed")
                    for key in ("image_file_id", "image_order", "image_alt"):
                        if row.metadata.get(key) != old.metadata.get(key):
                            raise ValueError("image asset identity changed")
        if protected_canonical(after_inputs.canonical) != before_hash:
            raise ValueError("canonical text, identities, ordering or dates changed")
        if plan["action"] == "missing" and after_inputs.canonical != before.canonical:
            raise ValueError("missing projection repair changed canonical metadata")
        if (
            plan["action"] != "missing"
            and source_fingerprint(after_hits) != source_hash
        ):
            raise ValueError("existing search payload or vectors changed")
        after = audit_baseline_inventory(
            after_inputs.canonical,
            read_file_inventory(client, index, "public", file_id, hits=after_hits),
            after_inputs.bindings,
        )
        if (
            after.state != "ready"
            or after_inputs.pending_manifest
            or after_inputs.gate_closed
        ):
            raise ValueError(
                "post-repair file is not ready: " + after.model_dump_json()
            )
        return {
            "file_id": str(file_id),
            "state": "ready",
            "action": plan["action"],
            "indexed_count": after.indexed_count,
            "new_indexed_count": after.indexed_count
            if plan["action"] == "missing"
            else 0,
            "protected_canonical_sha256": before_hash,
            "before_source_sha256": source_hash,
            "after_source_sha256": source_fingerprint(after_hits),
            "audit": after.model_dump(mode="json"),
            "corrected_image_ids": sorted(PARENTS)
            if str(file_id) == IMAGE_FILE_ID
            else [],
        }
    finally:
        try:
            authority.release(owner)
        except ValueError:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if not args.apply or not os.environ.get("KUBERNETES_SERVICE_HOST"):
        parser.error("explicit --apply and the DEV project container are required")
    if config.REGULATORY_ANNEX_ENVIRONMENT != "dev":
        parser.error("the publication environment must be DEV")
    plans = json.loads(args.plan.read_text())
    ids = [str(UUID(item["file_id"])) for item in plans]
    if not ids or len(set(ids)) != len(ids) or len(ids) > 78:
        raise ValueError("invalid explicit remaining-file selection")
    set_is_ee_based_on_env_variable()
    os.nice(10)
    SqlEngine.init_engine(pool_size=3, max_overflow=0)
    CURRENT_TENANT_ID_CONTEXTVAR.set("public")
    baseline_audit_settings("public", DATABASE)
    old_state = get_kv_store().load(OLD_PREFIX + "status", refresh_cache=True)
    old_control = get_kv_store().load(OLD_PREFIX + "control", refresh_cache=True)
    if (
        not isinstance(old_state, dict)
        or cast(dict[str, Any], old_state).get("state") != "stopped"
    ):
        raise ValueError("previous process has not reached its stopped boundary")
    if (
        not isinstance(old_control, dict)
        or cast(dict[str, Any], old_control).get("requested_by") != "agent"
    ):
        raise ValueError("a user stop requires user-directed resumption")
    claim_regulatory_maintenance(
        tenant_id="public",
        key=PREFIX + "claim",
        expected_database=DATABASE,
        selection_sha256=publication_digest(plans),
    )
    kv = get_kv_store()
    try:
        kv.load(PREFIX + "status", refresh_cache=True)
    except KvKeyNotFoundError:
        pass
    else:
        raise ValueError(
            "a launch record already exists; inspect rather than repeat dispatch"
        )
    state: dict[str, Any] = {
        "state": "running",
        "selected": len(plans),
        "processed": 0,
        "ready": 0,
        "failed": 0,
        "indexed_count": 0,
        "new_indexed_count": 0,
        "pid": os.getpid(),
    }
    lock = threading.Lock()
    stop = threading.Event()

    def persist_status() -> None:
        with lock:
            snapshot = {**state, "heartbeat_at": now()}
        kv.store(PREFIX + "status", snapshot)

    def heartbeat() -> None:
        CURRENT_TENANT_ID_CONTEXTVAR.set("public")
        while not stop.wait(30):
            try:
                persist_status()
            except Exception:
                # Local receipts survive a temporary status-store outage.
                continue

    with args.output.open("x") as report, ElasticsearchClient() as transport:
        persist_status()
        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
        try:
            for item in plans:
                try:
                    control: Any = kv.load(PREFIX + "control", refresh_cache=True)
                except KvKeyNotFoundError:
                    control = None
                if (
                    (args.output.parent / "STOP").exists()
                    or (
                        args.output.parent.parent / "dev-remaining78-20260924" / "STOP"
                    ).exists()
                    or isinstance(control, dict)
                    and cast(dict[str, Any], control).get("stop") is True
                ):
                    with lock:
                        state["state"] = "stopped"
                    return
                with lock:
                    state.update(current_file=item["file_id"], file_started_at=now())
                persist_status()
                report.write(
                    json.dumps(
                        {"state": "started", "file_id": item["file_id"], "at": now()}
                    )
                    + "\n"
                )
                report.flush()
                try:
                    record: dict[str, Any] = process_file(
                        transport.publication_client(), item
                    )
                except Exception as error:
                    record = {
                        "file_id": item["file_id"],
                        "state": "failed",
                        "error_type": type(error).__name__,
                        "detail": str(error)[:1000],
                        "traceback": traceback.format_exc(limit=12),
                    }
                record["at"] = now()
                report.write(json.dumps(record, ensure_ascii=False) + "\n")
                report.flush()
                os.fsync(report.fileno())
                kv.store(PREFIX + "file:" + item["file_id"], record)
                with lock:
                    state["processed"] += 1
                    state["ready"] += int(record["state"] == "ready")
                    state["failed"] += int(record["state"] != "ready")
                    state["indexed_count"] += record.get("indexed_count", 0)
                    state["new_indexed_count"] += record.get("new_indexed_count", 0)
                persist_status()
            with lock:
                state["state"] = (
                    "complete_with_errors" if state["failed"] else "complete"
                )
            report.write(json.dumps({"state": state["state"], "at": now()}) + "\n")
        except BaseException:
            with lock:
                state["state"] = "operator_error"
            raise
        finally:
            stop.set()
            thread.join(timeout=5)
            persist_status()


if __name__ == "__main__":
    main()
