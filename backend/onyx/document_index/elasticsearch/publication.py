"""Update-only fenced projection writes with permanent reservation tombstones.

Initialize may only create hidden empty tombstones. Seal covers EVERY durable
reservation, including abandoned planned IDs. Each token then accepts one frozen
final operation per ID. Verification requires every reservation finalized before
PG can open the gate. Never physically delete these records or mix legacy writers.
"""

import json
from typing import cast

from elasticsearch import ConflictError, Elasticsearch
from elasticsearch.helpers import scan
from pydantic import JsonValue

from onyx.document_index.elasticsearch.schema import get_elasticsearch_doc_chunk_id
from onyx.document_index.interfaces_new import (
    DocumentChunkVerificationError,
    TenantState,
)
from onyx.document_index.publication_models import (
    FileReservations,
    FrozenPublicationProjection,
    IndexedProjectionEvidence,
    PublicationIndexSnapshot,
    PublicationVerification,
    publication_digest,
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
    if (params.base != null && ctx._source.publication_payload != params.base) {
        throw new IllegalArgumentException('metadata base payload changed');
    }
    ctx._source = params.source;
}
"""
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
        projection: FrozenPublicationProjection | None,
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
            config = cast(JsonValue, json.loads(projection.embedding_config_json))
            if publication_digest(config) != self.snapshot.embedding_config_sha256:
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
    ) -> None:
        self._check_index()
        params = self._params(reservations, ordinal)
        operation = publication_digest({"source": source, "base": base})
        source["publication_operation"] = operation
        params.update({"source": source, "operation": operation, "base": base})
        # Intentionally no upsert: an unseen/uninitialized ID cannot become live.
        self.client.update(
            index=self.snapshot.index_name,
            id=self._id(reservations, ordinal),
            script={"lang": "painless", "source": _WRITE, "params": params},
            retry_on_conflict=3,
        )

    def upsert(
        self, reservations: FileReservations, projection: FrozenPublicationProjection
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
        self._write(
            reservations,
            updated.ordinal,
            self._source(reservations, updated, updated.ordinal),
            base=previous_payload_sha256,
        )

    def verify(
        self,
        reservations: FileReservations,
        projections: tuple[FrozenPublicationProjection, ...],
    ) -> PublicationVerification:
        self._check_index()
        live = {projection.ordinal: projection for projection in projections}
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
            self._id(reservations, ordinal) for ordinal in reservations.ordinals
        }
        sources: dict[str, dict[str, JsonValue]] = {}
        # Stop on an extra hit; storage remains bounded by the frozen inventory.
        for hit in scan(
            self.client, index=self.snapshot.index_name, query={"query": query}
        ):
            if hit["_id"] not in expected_ids:
                raise DocumentChunkVerificationError(
                    "missing/extra reserved file projection"
                )
            sources[hit["_id"]] = hit["_source"]
        if set(sources) != expected_ids:
            raise DocumentChunkVerificationError(
                "missing/extra reserved file projection"
            )
        manifest: list[JsonValue] = []
        for ordinal in reservations.ordinals:
            expected = self._source(reservations, live.get(ordinal), ordinal)
            actual = dict(sources[self._id(reservations, ordinal)])
            operation = actual.pop("publication_operation", None)
            if not operation or actual != expected:
                raise DocumentChunkVerificationError(
                    f"exact projection/tombstone verification failed for ordinal {ordinal}"
                )
            manifest.append(expected)
        return PublicationVerification(
            reservations=reservations,
            index=self.snapshot,
            live_ordinals=tuple(sorted(live)),
            canonical_chunk_ids=frozenset(
                json.loads(item.source_json)["regulatory_chunk_id"]
                for item in projections
            ),
            manifest_sha256=publication_digest(manifest),
        )

    def read_evidence(
        self, reservations: FileReservations, ordinal: int
    ) -> IndexedProjectionEvidence:
        """Fetch actual stored inputs/vectors; absent legacy provenance stays unverified.

        The raw source is available for the existing reconstructable-legacy proof
        path. This method never labels a newly regenerated old input as verified.
        """
        self._check_index()
        params = self._params(reservations, ordinal, require_gate=False)
        result = self.client.get(
            index=self.snapshot.index_name, id=self._id(reservations, ordinal)
        )
        source = dict(result["_source"])
        if (
            source.get("document_id") != params["file"]
            or source.get("chunk_index") != ordinal
            or source.get("tenant_id") != params["tenant"]
            or source.get("publication_scope", params["scope"]) != params["scope"]
        ):
            raise ValueError("indexed evidence scope mismatch")
        frozen = None
        digest = None
        evidence = source.get("publication_evidence")
        if evidence is not None:
            stored = dict(source)
            digest = stored.pop("publication_payload", None)
            stored.pop("publication_operation", None)
            # Sealing advances the floor while retaining the previous frozen payload.
            stored["publication_floor"] = stored.get("publication_token")
            if publication_digest(stored) != digest or evidence[
                "index"
            ] != self.snapshot.model_dump(mode="json"):
                raise DocumentChunkVerificationError(
                    "stored embedding evidence identity mismatch"
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
            if (
                publication_digest(evidence["embedding_config"])
                != self.snapshot.embedding_config_sha256
            ):
                raise DocumentChunkVerificationError(
                    "stored embedding configuration mismatch"
                )
        return IndexedProjectionEvidence(
            index=self.snapshot,
            source_json=json.dumps(source),
            frozen_projection=frozen,
            payload_sha256=digest,
        )
