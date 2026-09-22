"""Audit existing DEV search evidence, optionally prepare owned resumable baselines.

Default is read-only. Reports contain identities and hashes, never vector payloads.
Run with the deployed application's environment for --apply so ownership scope
matches the normal workers. No embedding or context provider is called.
"""

import argparse
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan

from onyx.db.engine.sql_engine import SqlEngine
from onyx.db.regulatory_publication import PublicationStore
from onyx.db.regulatory_publication_baseline import (
    BaselineAuditInputs,
    baseline_audit_inputs,
    baseline_audit_inputs_batch,
    baseline_audit_settings,
    baseline_file_ids,
)
from onyx.document_index.elasticsearch.client import ElasticsearchClient
from onyx.document_index.elasticsearch.publication import indexed_evidence_from_source
from onyx.document_index.elasticsearch.schema import get_elasticsearch_doc_chunk_id
from onyx.document_index.interfaces_new import TenantState
from onyx.document_index.publication_models import (
    IndexedProjectionEvidence,
    PublicationIndexSnapshot,
    PublicationScope,
)
from onyx.regulatory.amendments.annexes import config
from onyx.regulatory.amendments.annexes.publication_execution import LEASE_TTL
from onyx.regulatory.publication_baseline import (
    audit_baseline_inventory,
    ensure_owned_baseline,
    observed_index_snapshot,
)
from onyx.regulatory.writer_publication import recover_owned_writer_before_next
from onyx.utils.logger import setup_logger
from shared_configs.contextvars import CURRENT_TENANT_ID_CONTEXTVAR

logger = setup_logger()


def read_file_inventory(
    client: Elasticsearch,
    index: PublicationIndexSnapshot,
    tenant: str,
    file_id: UUID,
    *,
    hits: list[dict[str, Any]] | None = None,
    verify_index: bool = True,
) -> list[IndexedProjectionEvidence]:
    filters: list[dict[str, Any]] = [{"term": {"document_id": str(file_id)}}]
    if index.multitenant:
        filters.append({"term": {"tenant_id": tenant}})
    result = []
    for hit in (
        hits
        if hits is not None
        else scan(
            client,
            index=index.index_name,
            query={"query": {"bool": {"filter": filters}}},
        )
    ):
        source = hit["_source"]
        if source.get("publication_tombstone") is True:
            continue
        ordinal = source.get("chunk_index")
        if type(ordinal) is not int or hit["_id"] != get_elasticsearch_doc_chunk_id(
            TenantState(tenant_id=tenant, multitenant=index.multitenant),
            str(file_id),
            ordinal,
        ):
            raise ValueError("baseline indexed identity mismatch")
        stored = source.get("publication_evidence")
        snapshot = (
            PublicationIndexSnapshot.model_validate(stored["index"])
            if stored
            else index
        )
        if not snapshot.matches_temporal_index(index):
            raise ValueError("baseline physical index or model differs")
        result.append(indexed_evidence_from_source(snapshot, source))
    if verify_index:
        actual = client.indices.get(index=index.index_name)
        if (
            set(actual) != {index.index_name}
            or actual[index.index_name]["settings"]["index"]["uuid"] != index.index_uuid
        ):
            raise ValueError("baseline physical index changed during audit")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tenant", default="public")
    parser.add_argument(
        "--expected-database", required=True, choices=["customs-regulations-dev"]
    )
    parser.add_argument("--expected-index-uuid", required=True)
    parser.add_argument("--file-id", type=UUID, action="append")
    parser.add_argument("--after-file-id", type=UUID)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=4)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.apply and not args.file_id and args.limit is None:
        parser.error("--apply requires an explicit file list or bounded --limit")
    SqlEngine.init_engine(pool_size=args.workers, max_overflow=0)
    CURRENT_TENANT_ID_CONTEXTVAR.set(args.tenant)
    settings = baseline_audit_settings(args.tenant, args.expected_database)
    counts: Counter[str] = Counter()
    with ElasticsearchClient() as transport, args.output.open("a") as report:
        client = transport.publication_client()
        indexes = []
        current_uuids: set[str] = set()
        for setting in settings:
            if not client.indices.exists(index=setting.index_name):
                if setting.status.is_current():
                    raise ValueError("baseline current index unavailable")
                continue
            physical = client.indices.get(index=setting.index_name)
            if set(physical) != {setting.index_name}:
                raise ValueError("baseline requires a concrete index")
            index = observed_index_snapshot(
                setting, physical[setting.index_name]["settings"]["index"]["uuid"]
            )
            if (
                setting.status.is_current()
                and index.index_uuid != args.expected_index_uuid
            ):
                raise ValueError("baseline current physical index UUID mismatch")
            indexes.append(index)
            if setting.status.is_current():
                current_uuids.add(index.index_uuid)
        identifiers = args.file_id or baseline_file_ids(
            args.tenant, after=args.after_file_id, limit=args.limit
        )
        authority = PublicationStore(
            PublicationScope(
                tenant_id=args.tenant,
                environment=config.REGULATORY_ANNEX_ENVIRONMENT,
                database_identity=config.ANNEX_DATABASE_IDENTITY,
            )
        )

        def process_file(
            file_id: UUID,
            frozen_inputs: BaselineAuditInputs | None = None,
            inventories: dict[str, list[IndexedProjectionEvidence]] | None = None,
        ) -> dict[str, Any]:
            record: dict[str, Any] = {
                "file_id": str(file_id),
                "database": args.expected_database,
                "applied": False,
                "read_mode": "batch" if frozen_inputs is not None else "single",
            }
            try:
                inputs = frozen_inputs or baseline_audit_inputs(args.tenant, file_id)
                resumed_from = None
                if inputs.pending_manifest or inputs.gate_closed:
                    if (
                        not args.apply
                        or inputs.manifest is None
                        or inputs.manifest.kind != "baseline"
                    ):
                        return {
                            **record,
                            "state": "pending",
                            "pending_kind": inputs.manifest.kind
                            if inputs.manifest
                            else None,
                        }
                    resumed_from = inputs.manifest
                    owner = authority.acquire(file_id, owner_id=uuid4(), ttl=LEASE_TTL)
                    try:
                        from onyx.db.regulatory_writer_publication import (
                            pending_writer_manifest,
                        )

                        pending = pending_writer_manifest(owner)
                        if pending != resumed_from:
                            raise ValueError("baseline recovery changed after audit")
                        owner = recover_owned_writer_before_next(owner)
                        record["applied"] = True
                    finally:
                        authority.release(owner)
                    inputs = baseline_audit_inputs(args.tenant, file_id)
                    record["resumed"] = True
                audits = {}
                for index in indexes:
                    audits[index.index_uuid] = audit_baseline_inventory(
                        inputs.canonical,
                        inventories[index.index_uuid]
                        if inventories is not None
                        else read_file_inventory(client, index, args.tenant, file_id),
                        [
                            b
                            for b in inputs.bindings
                            if b.index.index_uuid == index.index_uuid
                        ],
                        require_complete=index.index_uuid in current_uuids,
                    )
                record["indexes"] = {
                    key: value.model_dump(mode="json") for key, value in audits.items()
                }
                state = (
                    "pending"
                    if inputs.pending_manifest or inputs.gate_closed
                    else "unresolved"
                    if any(a.state == "unresolved" for a in audits.values())
                    else "ready"
                    if all(a.state in {"ready", "unindexed"} for a in audits.values())
                    else "legacy"
                )
                record["state"] = state
                if resumed_from is not None:
                    for index in indexes:
                        retained = [
                            b
                            for b in resumed_from.bindings
                            if b.index.index_uuid == index.index_uuid
                        ]
                        from onyx.document_index.publication_models import (
                            ObservedPublicationProjection,
                        )

                        frozen = [
                            IndexedProjectionEvidence(
                                index=b.index,
                                source_json=b.projection.source_json,
                                frozen_projection=None
                                if isinstance(
                                    b.projection, ObservedPublicationProjection
                                )
                                else b.projection,
                                observed_projection=b.projection
                                if isinstance(
                                    b.projection, ObservedPublicationProjection
                                )
                                else None,
                                payload_sha256=None,
                            )
                            for b in retained
                        ]
                        expected = audit_baseline_inventory(
                            inputs.canonical,
                            frozen,
                            retained,
                            require_complete=index.index_uuid in current_uuids,
                        )
                        actual = audits[index.index_uuid]
                        if (expected.source_sha256, expected.vectors_sha256) != (
                            actual.source_sha256,
                            actual.vectors_sha256,
                        ):
                            raise ValueError(
                                "baseline recovery source/vector verification failed"
                            )
                if args.apply and state == "legacy":
                    owner = authority.acquire(file_id, owner_id=uuid4(), ttl=LEASE_TTL)
                    try:
                        owner = recover_owned_writer_before_next(owner)
                        owner = ensure_owned_baseline(owner, client)
                        record["applied"] = True
                    finally:
                        authority.release(owner)
                    after = baseline_audit_inputs(args.tenant, file_id)
                    checks = {}
                    for index in indexes:
                        verified = audit_baseline_inventory(
                            after.canonical,
                            read_file_inventory(client, index, args.tenant, file_id),
                            [
                                b
                                for b in after.bindings
                                if b.index.index_uuid == index.index_uuid
                            ],
                            require_complete=index.index_uuid in current_uuids,
                        )
                        prior = audits[index.index_uuid]
                        if verified.state not in {"ready", "unindexed"} or (
                            verified.source_sha256,
                            verified.vectors_sha256,
                        ) != (prior.source_sha256, prior.vectors_sha256):
                            raise ValueError(
                                "baseline source/vector preservation verification failed"
                            )
                        checks[index.index_uuid] = verified.model_dump(mode="json")
                    record.update(state="ready", after=checks)
            except Exception as error:
                # Full trace belongs to protected service logs; never dump request credentials.
                logger.exception("Publication baseline failed for file %s", file_id)
                record.update(state="error", error_type=type(error).__name__)
            return record

        def record_result(record: dict[str, Any]) -> None:
            counts[str(record["state"])] += 1
            report.write(json.dumps(record, ensure_ascii=False) + "\n")
            report.flush()
            print(
                json.dumps(
                    {
                        "file_id": record["file_id"],
                        "state": record["state"],
                        "processed": sum(counts.values()),
                    }
                ),
                flush=True,
            )

        def process_group(file_ids: list[UUID]) -> list[dict[str, Any]]:
            try:
                inputs_by_file = baseline_audit_inputs_batch(args.tenant, file_ids)
                inventories: dict[UUID, dict[str, list[IndexedProjectionEvidence]]] = {
                    identifier: {} for identifier in file_ids
                }
                for index in indexes:
                    queries: list[dict[str, Any]] = []
                    for file_id in file_ids:
                        filters: list[dict[str, Any]] = [
                            {"term": {"document_id": str(file_id)}}
                        ]
                        if index.multitenant:
                            filters.append({"term": {"tenant_id": args.tenant}})
                        queries.extend(
                            [
                                {"index": index.index_name},
                                {
                                    "size": 1000,
                                    "track_total_hits": True,
                                    "query": {"bool": {"filter": filters}},
                                },
                            ]
                        )
                    responses = client.msearch(searches=queries)["responses"]
                    if len(responses) != len(file_ids):
                        raise ValueError("baseline batch inventory incomplete")
                    for file_id, response in zip(file_ids, responses):
                        if (
                            response.get("error")
                            or response.get("timed_out")
                            or response.get("_shards", {}).get("failed")
                        ):
                            raise ValueError("baseline batch search failed")
                        hits = response["hits"]["hits"]
                        complete = response["hits"]["total"]["value"] == len(hits)
                        inventories[file_id][index.index_uuid] = read_file_inventory(
                            client,
                            index,
                            args.tenant,
                            file_id,
                            hits=hits if complete else None,
                            verify_index=False,
                        )
                    physical = client.indices.get(index=index.index_name)
                    if (
                        set(physical) != {index.index_name}
                        or physical[index.index_name]["settings"]["index"]["uuid"]
                        != index.index_uuid
                    ):
                        raise ValueError("baseline index changed during batched read")
                return [
                    process_file(file_id, inputs_by_file[file_id], inventories[file_id])
                    for file_id in file_ids
                ]
            except Exception:
                # Isolate a failed file; a single corrupt record must not hide its neighbors.
                return [process_file(file_id) for file_id in file_ids]

        if args.apply or args.file_id:
            for file_id in identifiers:
                record_result(process_file(file_id))
        else:
            groups = [
                identifiers[offset : offset + 32]
                for offset in range(0, len(identifiers), 32)
            ]
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                for records in pool.map(process_group, groups):
                    for record in records:
                        record_result(record)
        report.write(
            json.dumps(
                {
                    "summary": dict(counts),
                    "database": args.expected_database,
                    "index_uuid": args.expected_index_uuid,
                }
            )
            + "\n"
        )
    if counts["unresolved"] or counts["error"] or counts["pending"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
