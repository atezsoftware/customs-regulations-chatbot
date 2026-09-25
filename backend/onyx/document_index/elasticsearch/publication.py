"""Update-only fenced projection writes with permanent reservation tombstones.

Initialize may only create hidden empty tombstones. Seal covers EVERY durable
reservation, including abandoned planned IDs. Each token then accepts one frozen
final operation per ID. Verification requires every reservation finalized before
PG can open the gate. Never physically delete these records or mix legacy writers.
"""

import json
from collections.abc import Callable, Mapping
from typing import Any, cast

from elasticsearch import ConflictError, Elasticsearch
from elasticsearch.helpers import scan
from pydantic import JsonValue

from onyx.document_index.elasticsearch.schema import get_elasticsearch_doc_chunk_id
from onyx.document_index.interfaces_new import (
    DocumentChunkVerificationError,
    TenantState,
)
from onyx.document_index.publication_models import (
    OBSERVED_MUTABLE_SOURCE_FIELDS,
    FileReservations,
    FrozenPublicationProjection,
    IndexedProjectionEvidence,
    ObservedPublicationProjection,
    PublicationIndexSnapshot,
    PublicationProjection,
    PublicationVerification,
    RetainedPublicationProjection,
    publication_digest,
    publication_list_digest,
    publication_source,
)

# Checks run atomically against stored ownership, including metadata-only writes.
_SCOPE = """
if (ctx._source.document_id != params.file || ctx._source.chunk_index != params.ordinal ||
    (ctx._source.tenant_id != params.tenant) ||
    (ctx._source.publication_scope != null && ctx._source.publication_scope != params.scope)) {
    throw new IllegalArgumentException('publication scope conflict');
}
"""
_SEAL = (
    _SCOPE
    + """
if (ctx._source.publication_floor != null && ctx._source.publication_floor > params.token) {
    throw new IllegalArgumentException('stale publication ownership');
}
ctx._source.publication_scope = params.scope;
ctx._source.publication_floor = params.token;
"""
)
_WRITE = (
    _SCOPE
    + """
if (ctx._source.publication_floor != params.token) {
    throw new IllegalArgumentException('stale or unsealed publication ownership');
}
if (ctx._source.publication_token == params.token) {
    if (ctx._source.publication_operation != params.operation) {
        throw new IllegalArgumentException('different equal-token publication payload');
    }
    ctx.op = 'none';
} else {
    if (params.containsKey('observation') && params.observation != null &&
        ctx._source.publication_tombstone != true) {
        def original = new HashMap();
        for (def entry : ctx._source.entrySet()) {
            if (!entry.getKey().startsWith('publication_') &&
                !params.observation_mutable.contains(entry.getKey())) {
                original.put(entry.getKey(), entry.getValue());
            }
        }
        def receipt = ctx._source.publication_evidence;
        if (receipt != null && receipt.kind == 'observed-v1' &&
            receipt.observation.heading_repair != null) {
            def prior = receipt.observation.heading_repair;
            def repair = params.observation_heading_repair;
            if (repair == null || prior.canonical_chunk_id != repair.canonical_chunk_id ||
                receipt.observation.observed_immutable_sha256 != params.observation_immutable ||
                prior.original_heading_present != repair.original_heading_present ||
                (prior.original_heading_path == null ? repair.original_heading_path != null :
                 !prior.original_heading_path.equals(repair.original_heading_path)) ||
                !prior.corrected_heading_path.equals(original.get('heading_path'))) {
                throw new IllegalArgumentException('observed heading repair authority changed');
            }
            if (prior.original_heading_present) {
                original.put('heading_path', prior.original_heading_path);
            } else {
                original.remove('heading_path');
            }
        }
        if (!original.equals(params.observation) ||
            (ctx._source.publication_evidence != null &&
             ctx._source.publication_evidence.kind != 'observed-v1')) {
            throw new IllegalArgumentException('observed source changed before publication');
        }
    }
    if (params.base != null) {
        if (ctx._source.publication_payload != params.base) {
            throw new IllegalArgumentException('metadata base payload changed');
        }
        def actual = new HashMap(ctx._source);
        actual.remove('publication_floor');
        actual.remove('publication_token');
        actual.remove('publication_payload');
        actual.remove('publication_operation');
        if (!actual.equals(params.previous)) {
            throw new IllegalArgumentException('metadata base content/evidence changed');
        }
    }
    ctx._source = params.source;
}
"""
)

_RETAIN = (
    _SCOPE
    + """
if (ctx._source.publication_floor != params.token) {
    throw new IllegalArgumentException('stale or unsealed preservation');
}
def actual = new HashMap(ctx._source);
for (def key : params.controls) { actual.remove(key); }
if (!actual.equals(params.previous) && !actual.equals(params.updated)) {
    throw new IllegalArgumentException('retained source changed');
}
if (ctx._source.publication_token == params.token &&
    ctx._source.publication_operation != params.operation) {
    throw new IllegalArgumentException('different equal-token preservation');
}
if (params.updated.containsKey('validity_end_date')) {
    ctx._source.validity_end_date = params.updated.validity_end_date;
}
ctx._source.publication_token = params.token;
ctx._source.publication_payload = params.payload;
ctx._source.publication_operation = params.operation;
"""
)

_RETENTION_CONTROLS = (
    "publication_floor",
    "publication_token",
    "publication_payload",
    "publication_operation",
    "publication_scope",
)


class FencedPublicationIndex:
    def __init__(
        self, client: Elasticsearch, snapshot: PublicationIndexSnapshot
    ) -> None:
        self.client = client
        self.snapshot = snapshot

    def _check_index(self) -> None:
        indices = self.client.indices.get(index=self.snapshot.index_name)
        if (
            set(indices) != {self.snapshot.index_name}
            or indices[self.snapshot.index_name]["settings"]["index"]["uuid"]
            != self.snapshot.index_uuid
        ):
            raise ValueError("publication index identity changed")
        properties = indices[self.snapshot.index_name]["mappings"]["properties"]
        if ("tenant_id" in properties) != self.snapshot.multitenant:
            raise ValueError("publication index tenant mode mismatch")
        vector = properties["content_vector"]
        if vector["dims"] != self.snapshot.vector_dimension:
            raise ValueError("publication index vector dimension changed")

    def _params(
        self, reservations: FileReservations, ordinal: int, *, require_gate: bool = True
    ) -> dict[str, JsonValue]:
        if (
            require_gate and not reservations.gate_closed
        ) or ordinal not in reservations.ordinals:
            raise ValueError("publication requires gated reserved ordinal")
        owner = reservations.ownership
        if not self.snapshot.multitenant and owner.scope.tenant_id != "public":
            raise ValueError("single-tenant index requires public tenant authority")
        return {
            "file": str(owner.user_file_id),
            "ordinal": ordinal,
            "tenant": owner.scope.tenant_id if self.snapshot.multitenant else None,
            "scope": publication_digest(owner.scope.model_dump(mode="json")),
            "token": owner.fencing_token,
        }

    def _id(self, reservations: FileReservations, ordinal: int) -> str:
        return get_elasticsearch_doc_chunk_id(
            TenantState(
                tenant_id=reservations.ownership.scope.tenant_id,
                multitenant=self.snapshot.multitenant,
            ),
            str(reservations.ownership.user_file_id),
            ordinal,
        )

    def _empty(
        self, reservations: FileReservations, ordinal: int
    ) -> dict[str, JsonValue]:
        params = self._params(reservations, ordinal)
        source: dict[str, JsonValue] = {
            "document_id": params["file"],
            "chunk_index": ordinal,
            "hidden": True,
            "publication_tombstone": True,
            "publication_scope": params["scope"],
            "publication_floor": 0,
        }
        if params["tenant"] is not None:
            source["tenant_id"] = params["tenant"]
        return source

    def initialize(self, reservations: FileReservations) -> None:
        """Create-only nonsearchable placeholders; safe even if this call arrives late."""
        self._check_index()
        for ordinal in reservations.ordinals:
            try:
                self.client.create(
                    index=self.snapshot.index_name,
                    id=self._id(reservations, ordinal),
                    document=self._empty(reservations, ordinal),
                )
            except ConflictError:
                # Seal performs the atomic tenant/file check on existing/legacy entries.
                continue

    def seal(self, reservations: FileReservations) -> None:
        """Adopt/seal complete committed PG inventory, preserving existing ordinals."""
        self.initialize(reservations)
        for ordinal in reservations.ordinals:
            self.client.update(
                index=self.snapshot.index_name,
                id=self._id(reservations, ordinal),
                script={
                    "lang": "painless",
                    "source": _SEAL,
                    "params": self._params(reservations, ordinal),
                },
                retry_on_conflict=3,
            )

    def _source(
        self,
        reservations: FileReservations,
        projection: PublicationProjection | None,
        ordinal: int,
    ) -> dict[str, JsonValue]:
        params = self._params(reservations, ordinal)
        if projection is None:
            payload = self._empty(reservations, ordinal)
        else:
            payload = cast(dict[str, JsonValue], json.loads(projection.source_json))
            if (
                payload.get("document_id") != params["file"]
                or payload.get("tenant_id") != params["tenant"]
            ):
                raise ValueError("projection tenant/file scope mismatch")
            vector = payload.get("content_vector")
            if (
                not isinstance(vector, list)
                or len(vector) != self.snapshot.vector_dimension
                or any(
                    not isinstance(value, (int, float)) or isinstance(value, bool)
                    for value in vector
                )
            ):
                raise ValueError("projection vector dimension/content mismatch")
            if isinstance(projection, ObservedPublicationProjection):
                if not projection.accepted_by(self.snapshot):
                    raise ValueError("observation crosses physical index")
                # Revalidate model_copy results before any transport mutation.
                ObservedPublicationProjection.model_validate(projection.model_dump())
                payload["publication_evidence"] = {
                    "kind": "observed-v1",
                    "observation": projection.model_dump(
                        mode="json", exclude={"source_json"}
                    ),
                    "index": self.snapshot.model_dump(mode="json"),
                }
            else:
                config = cast(JsonValue, json.loads(projection.embedding_config_json))
                if not self.snapshot.accepts_encoder_configuration(config):
                    raise ValueError("projection model configuration mismatch")
                payload["publication_evidence"] = {
                    "context_projection_id": projection.context_projection_id,
                    "embedding_inputs": list(projection.embedding_inputs),
                    "embedding_config": config,
                    "index": self.snapshot.model_dump(mode="json"),
                }
            payload["publication_tombstone"] = False
        payload["publication_scope"] = params["scope"]
        payload["publication_floor"] = params["token"]
        payload["publication_token"] = params["token"]
        # This digest is compared only for retry identity; verification reads actual source.
        payload["publication_payload"] = publication_digest(payload)
        return payload

    def _write(
        self,
        reservations: FileReservations,
        ordinal: int,
        source: dict[str, JsonValue],
        *,
        base: str | None = None,
        previous: dict[str, JsonValue] | None = None,
    ) -> None:
        self._check_index()
        params = self._params(reservations, ordinal)
        operation_payload: dict[str, JsonValue] = {"source": source, "base": base}
        if previous is not None:
            operation_payload["previous"] = previous
        operation = publication_digest(operation_payload)
        source["publication_operation"] = operation
        params.update(
            {
                "source": source,
                "operation": operation,
                "base": base,
                "previous": previous,
            }
        )
        self._guard_observation(params, source)
        # Intentionally no upsert: an unseen/uninitialized ID cannot become live.
        self.client.update(
            index=self.snapshot.index_name,
            id=self._id(reservations, ordinal),
            script={"lang": "painless", "source": _WRITE, "params": params},
            retry_on_conflict=3,
        )

    @staticmethod
    def _guard_observation(
        params: dict[str, JsonValue], source: dict[str, JsonValue]
    ) -> None:
        evidence = source.get("publication_evidence")
        if isinstance(evidence, dict) and evidence.get("kind") == "observed-v1":
            observed = ObservedPublicationProjection.model_validate(
                {
                    **cast(dict[str, JsonValue], evidence["observation"]),
                    "source_json": json.dumps(publication_source(json.dumps(source))),
                }
            )
            original = publication_source(observed.source_json)
            repair = observed.heading_repair
            if repair is not None:
                if repair.original_heading_present:
                    original["heading_path"] = cast(
                        JsonValue, repair.original_heading_path
                    )
                else:
                    original.pop("heading_path", None)
            params["observation_heading_repair"] = (
                repair.model_dump(mode="json") if repair else None
            )
            params["observation_immutable"] = observed.observed_immutable_sha256
            params["observation"] = {
                key: value
                for key, value in original.items()
                if key not in OBSERVED_MUTABLE_SOURCE_FIELDS
            }
            params["observation_mutable"] = sorted(OBSERVED_MUTABLE_SOURCE_FIELDS)

    def upsert(
        self, reservations: FileReservations, projection: PublicationProjection
    ) -> None:
        self._write(
            reservations,
            projection.ordinal,
            self._source(reservations, projection, projection.ordinal),
        )

    def tombstone(self, reservations: FileReservations, ordinal: int) -> None:
        self._write(reservations, ordinal, self._source(reservations, None, ordinal))

    def update_metadata(
        self,
        reservations: FileReservations,
        *,
        previous: FrozenPublicationProjection,
        updated: FrozenPublicationProjection,
        previous_payload_sha256: str,
    ) -> None:
        """Frozen full result, targeted metadata only; retains caller's original token."""
        before = cast(dict[str, JsonValue], json.loads(previous.source_json))
        after = cast(dict[str, JsonValue], json.loads(updated.source_json))
        allowed = {
            "hidden",
            "document_sets",
            "user_projects",
            "personas",
            "public",
            "access_control_list",
            "validity_start_date",
            "validity_end_date",
        }
        if (
            previous.ordinal != updated.ordinal
            or previous.context_projection_id != updated.context_projection_id
            or previous.embedding_inputs != updated.embedding_inputs
            or previous.embedding_config_json != updated.embedding_config_json
            or {key: value for key, value in before.items() if key not in allowed}
            != {key: value for key, value in after.items() if key not in allowed}
        ):
            raise ValueError(
                "metadata update changes frozen embedding/content identity"
            )
        # Bind the supplied previous projection to the actual stored content and
        # encoder evidence atomically in ES. A separately valid digest is insufficient.
        previous_source = self._source(reservations, previous, previous.ordinal)
        for field in ("publication_floor", "publication_token", "publication_payload"):
            previous_source.pop(field)
        self._write(
            reservations,
            updated.ordinal,
            self._source(reservations, updated, updated.ordinal),
            base=previous_payload_sha256,
            previous=previous_source,
        )

    def verify(
        self,
        reservations: FileReservations,
        projections: tuple[PublicationProjection, ...],
        retained: tuple[RetainedPublicationProjection, ...] = (),
    ) -> PublicationVerification:
        self._check_index()
        live = {projection.ordinal: projection for projection in projections}
        preserved = {
            json.loads(item.source_json)["chunk_index"]: item for item in retained
        }
        if (
            len(preserved) != len(retained)
            or set(preserved) & set(live)
            or not set(preserved).issubset(reservations.ordinals)
        ):
            raise DocumentChunkVerificationError(
                "duplicate or unreserved retained identity"
            )
        if len(live) != len(projections) or not set(live).issubset(
            reservations.ordinals
        ):
            raise DocumentChunkVerificationError(
                "duplicate or unreserved projection identity"
            )
        self.client.indices.refresh(index=self.snapshot.index_name)
        query: dict[str, JsonValue] = {
            "bool": {
                "filter": [
                    {"term": {"document_id": str(reservations.ownership.user_file_id)}}
                ]
            }
        }
        if self.snapshot.multitenant:
            query = {
                "bool": {
                    "filter": [
                        query,
                        {"term": {"tenant_id": reservations.ownership.scope.tenant_id}},
                    ]
                }
            }
        expected_ids = {
            self._id(reservations, ordinal): ordinal
            for ordinal in reservations.ordinals
        }

        def expected_source(ordinal: int) -> dict[str, JsonValue]:
            return (
                self._retained_source(reservations, preserved[ordinal])
                if ordinal in preserved
                else self._source(reservations, live.get(ordinal), ordinal)
            )

        seen: set[str] = set()
        # Keep only identities across pages; full decoded vectors are compared per hit.
        for hit in scan(
            self.client, index=self.snapshot.index_name, query={"query": query}, size=64
        ):
            identifier = hit["_id"]
            if identifier not in expected_ids or identifier in seen:
                raise DocumentChunkVerificationError(
                    "missing/extra/duplicate reserved file projection"
                )
            ordinal = expected_ids[identifier]
            actual = dict(hit["_source"])
            operation = actual.pop("publication_operation", None)
            if not operation or actual != expected_source(ordinal):
                raise DocumentChunkVerificationError(
                    f"exact projection/tombstone verification failed for ordinal {ordinal}"
                )
            seen.add(identifier)
        if seen != expected_ids.keys():
            raise DocumentChunkVerificationError(
                "missing/extra reserved file projection"
            )
        return PublicationVerification(
            reservations=reservations,
            index=self.snapshot,
            live_ordinals=tuple(sorted(set(live) | set(preserved))),
            canonical_chunk_ids=frozenset(
                json.loads(item.source_json)["regulatory_chunk_id"]
                for item in (*projections, *retained)
            ),
            manifest_sha256=publication_list_digest(
                expected_source(ordinal) for ordinal in reservations.ordinals
            ),
        )

    def _retained_source(
        self, reservations: FileReservations, retained: RetainedPublicationProjection
    ) -> dict[str, JsonValue]:
        if not self.snapshot.matches_temporal_index(retained.evidence.index):
            raise ValueError("retention crosses physical index")
        source = cast(dict[str, JsonValue], json.loads(retained.source_json))
        ordinal = source["chunk_index"]
        if not isinstance(ordinal, int):
            raise ValueError("retained ordinal missing")
        params = self._params(reservations, ordinal)
        if (
            source.get("document_id") != params["file"]
            or source.get("tenant_id") != params["tenant"]
        ):
            raise ValueError("retained tenant/file mismatch")
        for key in _RETENTION_CONTROLS:
            source.pop(key, None)
        source.update(
            publication_scope=params["scope"],
            publication_floor=params["token"],
            publication_token=params["token"],
        )
        source["publication_payload"] = publication_digest(source)
        return source

    def retain(
        self, reservations: FileReservations, retained: RetainedPublicationProjection
    ) -> None:
        """CAS preserves vectors and existing evidence; never manufactures provenance."""
        self._check_index()
        source = self._retained_source(reservations, retained)
        ordinal = cast(int, source["chunk_index"])
        previous = {
            k: v
            for k, v in json.loads(retained.evidence.source_json).items()
            if k not in _RETENTION_CONTROLS
        }
        updated = {k: v for k, v in source.items() if k not in _RETENTION_CONTROLS}
        operation = publication_digest({"retained": source})
        params = self._params(reservations, ordinal)
        params.update(
            previous=previous,
            updated=updated,
            controls=list(_RETENTION_CONTROLS),
            operation=operation,
            payload=source["publication_payload"],
        )
        self.client.update(
            index=self.snapshot.index_name,
            id=self._id(reservations, ordinal),
            script={"lang": "painless", "source": _RETAIN, "params": params},
            retry_on_conflict=3,
        )

    def read_evidence(
        self, reservations: FileReservations, ordinal: int
    ) -> IndexedProjectionEvidence:
        """Fetch actual stored inputs/vectors; absent legacy provenance stays unverified.

        The raw source is available for the existing reconstructable-legacy proof
        path. This method never labels a newly regenerated old input as verified.
        """
        self._check_index()
        self._params(reservations, ordinal, require_gate=False)
        result = self.client.get(
            index=self.snapshot.index_name, id=self._id(reservations, ordinal)
        )
        source = dict(result["_source"])
        return self._evidence_from_source(reservations, ordinal, source)

    def _evidence_from_source(
        self, reservations: FileReservations, ordinal: int, source: dict[str, Any]
    ) -> IndexedProjectionEvidence:
        params = self._params(reservations, ordinal, require_gate=False)
        if (
            source.get("document_id") != params["file"]
            or source.get("chunk_index") != ordinal
            or source.get("tenant_id") != params["tenant"]
            or source.get("publication_scope", params["scope"]) != params["scope"]
        ):
            raise ValueError("indexed evidence scope mismatch")
        return indexed_evidence_from_source(self.snapshot, source)

    def existing_ordinals(self, reservations: FileReservations) -> tuple[int, ...]:
        """Identify legacy physical IDs without inventing embedding provenance."""
        self._check_index()
        self.client.indices.refresh(index=self.snapshot.index_name)
        filters: list[dict[str, JsonValue]] = [
            {"term": {"document_id": str(reservations.ownership.user_file_id)}}
        ]
        if self.snapshot.multitenant:
            filters.append(
                {"term": {"tenant_id": reservations.ownership.scope.tenant_id}}
            )
        ordinals: set[int] = set()
        for hit in scan(
            self.client,
            index=self.snapshot.index_name,
            query={
                "_source": ["chunk_index", "publication_*"],
                "query": {"bool": {"filter": filters}},
            },
        ):
            source = hit["_source"]
            ordinal = source.get("chunk_index")
            if (
                type(ordinal) is not int
                or ordinal < 0
                or ordinal >= 2**63
                or hit["_id"] != self._id(reservations, ordinal)
            ):
                raise ValueError("legacy indexed projection identity is invalid")
            if ordinal not in reservations.ordinals and any(
                key.startswith("publication_") for key in source
            ):
                raise ValueError(
                    "protected indexed projection has no durable reservation"
                )
            ordinals.add(ordinal)
        return tuple(sorted(ordinals))

    def inventory_evidence(
        self, reservations: FileReservations
    ) -> tuple[IndexedProjectionEvidence, ...]:
        """Freeze actual live inventory, including historical/expired projections."""
        self._check_index()
        self.client.indices.refresh(index=self.snapshot.index_name)
        filters: list[dict[str, JsonValue]] = [
            {"term": {"document_id": str(reservations.ownership.user_file_id)}}
        ]
        if self.snapshot.multitenant:
            filters.append(
                {"term": {"tenant_id": reservations.ownership.scope.tenant_id}}
            )
        inventory: list[IndexedProjectionEvidence] = []
        for hit in scan(
            self.client,
            index=self.snapshot.index_name,
            query={"query": {"bool": {"filter": filters}}},
        ):
            source = hit["_source"]
            ordinal = source.get("chunk_index")
            if (
                not isinstance(ordinal, int)
                or ordinal not in reservations.ordinals
                or hit["_id"] != self._id(reservations, ordinal)
            ):
                raise ValueError(
                    "indexed evidence contains an unreserved/foreign projection"
                )
            if source.get("publication_tombstone") is True:
                continue
            stored = source.get("publication_evidence")
            adapter = self
            if stored is not None:
                previous = PublicationIndexSnapshot.model_validate(stored["index"])
                if (previous.index_name, previous.index_uuid) != (
                    self.snapshot.index_name,
                    self.snapshot.index_uuid,
                ):
                    raise ValueError(
                        "indexed evidence physical index identity mismatch"
                    )
                adapter = FencedPublicationIndex(self.client, previous)
            inventory.append(
                adapter._evidence_from_source(reservations, ordinal, dict(source))
            )
        self._check_index()
        return tuple(
            sorted(
                inventory, key=lambda item: json.loads(item.source_json)["chunk_index"]
            )
        )

    def publish_inventory(
        self,
        reservations: FileReservations,
        projections: tuple[PublicationProjection, ...],
        *,
        before_batch: Callable[[], object],
    ) -> None:
        """Batch the same create/seal/write scripts; preserve every fencing check.

        The complete reserved inventory is fenced, including retained vectors and
        abandoned ordinals. Only the caller's frozen projections supply content.
        Any partial failure leaves the gate closed for an idempotent retry.
        """
        live = {projection.ordinal: projection for projection in projections}
        if len(live) != len(projections) or not set(live).issubset(
            reservations.ordinals
        ):
            raise ValueError("bulk publication requires unique reserved projections")
        for phase in ("create", "seal", "write"):
            for offset in range(0, len(reservations.ordinals), 64):
                before_batch()
                self._check_index()
                operations: list[Mapping[str, Any]] = []
                for ordinal in reservations.ordinals[offset : offset + 64]:
                    identity = {
                        "_index": self.snapshot.index_name,
                        "_id": self._id(reservations, ordinal),
                    }
                    if phase == "create":
                        operations.extend(
                            [{"create": identity}, self._empty(reservations, ordinal)]
                        )
                        continue
                    params = self._params(reservations, ordinal)
                    script = _SEAL
                    if phase == "write":
                        source = self._source(reservations, live.get(ordinal), ordinal)
                        operation = publication_digest({"source": source, "base": None})
                        source["publication_operation"] = operation
                        params.update(
                            source=source, operation=operation, base=None, previous=None
                        )
                        self._guard_observation(params, source)
                        script = _WRITE
                    operations.extend(
                        [
                            {"update": {**identity, "retry_on_conflict": 3}},
                            {
                                "script": {
                                    "lang": "painless",
                                    "source": script,
                                    "params": params,
                                }
                            },
                        ]
                    )
                response = self.client.bulk(operations=operations)
                items = response.get("items", [])
                if len(items) != len(operations) // 2:
                    raise ValueError("bulk publication returned incomplete results")
                for item in items:
                    outcome = item["create" if phase == "create" else "update"]
                    status = outcome.get("status", 500)
                    if not (200 <= status < 300 or phase == "create" and status == 409):
                        raise ValueError(
                            f"bulk publication {phase} failed (status {status})"
                        )


def indexed_evidence_from_source(
    snapshot: PublicationIndexSnapshot, source: dict[str, Any]
) -> IndexedProjectionEvidence:
    """Validate stored content/provenance; callers establish file and tenant scope."""
    ordinal = source.get("chunk_index")
    if type(ordinal) is not int:
        raise ValueError("indexed evidence ordinal missing")
    frozen = None
    digest = None
    evidence = source.get("publication_evidence")
    if evidence is not None:
        stored = dict(source)
        digest = stored.pop("publication_payload", None)
        stored.pop("publication_operation", None)
        # Sealing advances the floor while retaining the previous frozen payload.
        stored["publication_floor"] = stored.get("publication_token")
        if (
            publication_digest(stored) != digest
            or PublicationIndexSnapshot.model_validate(evidence["index"]) != snapshot
        ):
            raise DocumentChunkVerificationError(
                "stored embedding evidence identity mismatch"
            )
        if evidence.get("kind") == "observed-v1":
            observed = ObservedPublicationProjection.model_validate(
                {
                    **evidence["observation"],
                    "source_json": json.dumps(publication_source(json.dumps(source))),
                }
            )
            if not observed.accepted_by(snapshot) or publication_source(
                observed.source_json
            ) != publication_source(json.dumps(source)):
                raise DocumentChunkVerificationError(
                    "stored observation source mismatch"
                )
            return IndexedProjectionEvidence(
                index=snapshot,
                source_json=json.dumps(source),
                frozen_projection=None,
                payload_sha256=digest,
                observed_projection=observed,
            )
        frozen = FrozenPublicationProjection(
            ordinal=ordinal,
            context_projection_id=evidence["context_projection_id"],
            source_json=json.dumps(
                {
                    key: value
                    for key, value in source.items()
                    if not key.startswith("publication_")
                }
            ),
            embedding_inputs=tuple(evidence["embedding_inputs"]),
            embedding_config_json=json.dumps(evidence["embedding_config"]),
        )
        if not snapshot.accepts_encoder_configuration(evidence["embedding_config"]):
            raise DocumentChunkVerificationError(
                "stored embedding configuration mismatch"
            )
    return IndexedProjectionEvidence(
        index=snapshot,
        source_json=json.dumps(source),
        frozen_projection=frozen,
        payload_sha256=digest,
    )
