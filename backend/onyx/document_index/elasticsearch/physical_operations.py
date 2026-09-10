"""Bounded single-tenant physical replacement with retained exact-UUID authority."""

from typing import Any, Literal
from uuid import UUID

from elastic_transport import ObjectApiResponse
from elasticsearch import Elasticsearch
from pydantic import JsonValue

from onyx.db.engine.sql_engine import get_session_with_current_tenant
from onyx.db.regulatory_physical_indexes import (
    PhysicalIndexOperation,
    advance_physical_operation,
    claim_physical_operation,
    pending_physical_operation,
    release_physical_operation,
    require_physical_index_available,
)
from onyx.document_index.publication_models import PublicationScope
from onyx.regulatory.amendments.annexes import config
from shared_configs.contextvars import get_current_tenant_id

_OPERATION_META = "onyx_physical_operation"


def physical_scope() -> PublicationScope:
    return PublicationScope(
        tenant_id=get_current_tenant_id(),
        environment=config.REGULATORY_ANNEX_ENVIRONMENT,
        database_identity=config.ANNEX_DATABASE_IDENTITY,
    )


def require_fresh_index_name(index_name: str) -> None:
    with get_session_with_current_tenant() as session:
        require_physical_index_available(session, index_name)


def concrete_index_uuid(client: Elasticsearch, index_name: str) -> str | None:
    if not client.indices.exists(index=index_name):
        return None
    response = client.indices.get(index=index_name)
    if set(response) != {index_name}:
        raise ValueError("physical operation requires an exact concrete index")
    return str(response[index_name]["settings"]["index"]["uuid"])


def run_empty_physical_operation(
    client: Elasticsearch,
    *,
    index_name: str,
    multitenant: bool,
    operation: Literal["delete", "recreate"],
    mappings: dict[str, Any] | None = None,
    settings: dict[str, Any] | None = None,
    discard_search_settings_id: int | None = None,
) -> str | None:
    if multitenant:
        raise ValueError(
            "destructive shared physical-index operations require global authority"
        )
    scope = physical_scope()
    request: dict[str, JsonValue] = {
        "mappings": mappings or {},
        "settings": settings or {},
        "discard_search_settings_id": discard_search_settings_id,
    }
    pending = pending_physical_operation(scope, index_name)
    original_uuid = (
        pending.index_uuid if pending else concrete_index_uuid(client, index_name)
    )
    if original_uuid is None:
        raise ValueError("physical operation has no retained original UUID")
    authority = claim_physical_operation(
        scope,
        index_name=index_name,
        index_uuid=original_uuid,
        operation=operation,
        request=request,
        discard_search_settings_id=discard_search_settings_id,
    )
    try:
        if authority.phase == "prepared":
            if concrete_index_uuid(client, index_name) != authority.index_uuid:
                raise ValueError("physical operation original UUID changed")
            block = client.indices.add_block(index=index_name, block="write")
            if not block.get("acknowledged") or not block.get("shards_acknowledged"):
                raise ValueError(
                    "physical operation write barrier was not fully acknowledged"
                )
            if client.count(index=index_name, query={"match_all": {}})["count"] != 0:
                raise ValueError("physical operation cannot remove a nonempty index")
            authority = advance_physical_operation(
                authority, expected_phase="prepared", phase="deleting"
            )
            # An ambiguous DELETE is never reposted. A late metadata request could delete a reused name.
            deleted = client.options(opaque_id=str(authority.id)).indices.delete(
                index=index_name
            )
            if (
                not deleted.get("acknowledged")
                or concrete_index_uuid(client, index_name) is not None
            ):
                raise ValueError(
                    "physical deletion has no positively established terminal outcome"
                )
            authority = advance_physical_operation(
                authority,
                expected_phase="deleting",
                phase="deleted",
                terminal_evidence=_delete_acknowledgement(authority, deleted),
            )
        if authority.phase == "deleted" and operation == "delete":
            advance_physical_operation(
                authority, expected_phase="deleted", phase="complete"
            )
            return None
        if authority.phase == "deleted":
            authority = advance_physical_operation(
                authority, expected_phase="deleted", phase="creating"
            )
        if authority.phase == "creating":
            result_uuid = _create_owned_index(client, authority)
            advance_physical_operation(
                authority,
                expected_phase="creating",
                phase="complete",
                result_index_uuid=result_uuid,
            )
            return result_uuid
        raise ValueError("physical operation requires reconciliation")
    finally:
        release_physical_operation(authority)


def _delete_acknowledgement(
    operation: PhysicalIndexOperation, response: ObjectApiResponse[Any]
) -> dict[str, JsonValue]:
    if response.body.get("acknowledged") is not True or response.meta.headers.get(
        "x-opaque-id"
    ) != str(operation.id):
        raise ValueError(
            "physical recovery requires the correlated terminal acknowledgement of the original DELETE"
        )
    return {
        "version": 1,
        "operation_id": str(operation.id),
        "owner_id": str(operation.owner_id),
        "index_name": operation.index_name,
        "original_index_uuid": operation.index_uuid,
        "response": dict(response.body),
        "opaque_id": response.meta.headers.get("x-opaque-id"),
    }


def reconcile_physical_delete_after_writer_exit(
    client: Elasticsearch,
    *,
    operation_id: UUID,
    index_name: str,
    multitenant: bool,
    original_response: ObjectApiResponse[Any],
    writer_exit_evidence_sha256: str,
) -> None:
    """Operator-only recovery using an archived original ACK and verified writer exit.

    The caller must retain evidence that this operation's original writer terminated.
    Its correlated DELETE acknowledgement proves that request completed. Neither time
    passage nor an empty task snapshot can substitute for either artifact. Without the
    original terminal response this interface deliberately leaves the name reserved.
    """
    if multitenant:
        raise ValueError(
            "destructive shared physical-index recovery requires global authority"
        )
    if len(writer_exit_evidence_sha256) != 64 or any(
        value not in "0123456789abcdef" for value in writer_exit_evidence_sha256
    ):
        raise ValueError("physical recovery requires retained writer-exit evidence")
    operation = pending_physical_operation(physical_scope(), index_name)
    if (
        operation is None
        or operation.id != operation_id
        or operation.phase != "deleting"
    ):
        raise ValueError(
            "physical recovery does not match the retained indeterminate DELETE"
        )
    evidence = _delete_acknowledgement(operation, original_response)
    if concrete_index_uuid(client, index_name) is not None:
        raise ValueError(
            "physical recovery cannot release an existing or reused index name"
        )
    evidence["writer_exit_evidence_sha256"] = writer_exit_evidence_sha256
    operation = advance_physical_operation(
        operation,
        expected_phase="deleting",
        phase="deleted",
        terminal_evidence=evidence,
    )
    if operation.operation == "delete":
        advance_physical_operation(
            operation, expected_phase="deleted", phase="complete"
        )
    else:
        release_physical_operation(operation)


def _create_owned_index(
    client: Elasticsearch, operation: PhysicalIndexOperation
) -> str:
    desired = operation.request["mappings"]
    settings = operation.request["settings"]
    if not isinstance(desired, dict) or not isinstance(settings, dict):
        raise ValueError("physical recreation request is incomplete")
    marker = {"id": str(operation.id), "request_sha256": operation.request_sha256}
    mappings = dict(desired)
    metadata = mappings.get("_meta", {})
    if not isinstance(metadata, dict):
        raise ValueError("physical recreation mapping metadata is invalid")
    mappings["_meta"] = {**metadata, _OPERATION_META: marker}
    if concrete_index_uuid(client, operation.index_name) is None:
        # Identical create requests are atomic at the retained name; no duplicate may delete it.
        response = client.indices.create(
            index=operation.index_name, mappings=mappings, settings=settings
        )
        if not response.get("acknowledged") or not response.get("shards_acknowledged"):
            raise ValueError("physical recreation acknowledgement is incomplete")
    actual = client.indices.get_mapping(index=operation.index_name)[
        operation.index_name
    ]["mappings"]
    if actual.get("_meta", {}).get(_OPERATION_META) != marker:
        raise ValueError("physical recreation UUID belongs to a different operation")
    for key, value in desired.items():
        mismatch = _mapping_mismatch(actual.get(key), value, key)
        if key != "_meta" and mismatch is not None:
            raise ValueError(
                f"physical recreation mapping does not match retained input: {mismatch}"
            )
    result = concrete_index_uuid(client, operation.index_name)
    if result is None:
        raise ValueError("physical recreation has no terminal UUID")
    return result


def _mapping_mismatch(actual: JsonValue, expected: JsonValue, path: str) -> str | None:
    """ES adds mapping defaults; every explicitly requested value must remain present."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return path
        for key, value in expected.items():
            # ES 8.17 omits these explicit default mapping parameters on GET.
            defaults: dict[str, JsonValue] = {
                "store": False,
                "index": True,
                "doc_values": True,
            }
            if (
                key not in actual
                and "type" in expected
                and key in defaults
                and value == defaults[key]
            ):
                continue
            mismatch = _mapping_mismatch(actual.get(key), value, f"{path}.{key}")
            if mismatch is not None:
                return mismatch
        return None
    return None if actual == expected else f"{path} ({actual!r} != {expected!r})"
