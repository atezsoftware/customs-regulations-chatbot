"""Large-file verification stays exact without retaining decoded vector inventories."""

import json
import weakref
from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from pydantic import JsonValue

from onyx.document_index.elasticsearch import publication
from onyx.document_index.interfaces_new import DocumentChunkVerificationError
from onyx.document_index.publication_models import (
    FileReservations,
    IndexedProjectionEvidence,
    ObservedPublicationProjection,
    RetainedPublicationProjection,
    publication_digest,
    publication_list_digest,
)
from tests.unit.onyx.document_index.elasticsearch.test_observed_publication import (
    observed,
)
from tests.unit.onyx.regulatory.indexing_jobs.owned_publication_test_helpers import (
    OwnedAuthority,
)


class TrackedVector(list[float]):
    pass


@pytest.mark.parametrize(
    "values",
    [
        [],
        [None],
        [
            {"z": [1, -0.0, 1e-17], "a": 'Türkçe \\ "\n'},
            True,
            None,
            {"nested": {"b": 2, "a": 1}},
        ],
    ],
)
def test_streamed_list_digest_is_byte_compatible(values: list[JsonValue]) -> None:
    assert publication_list_digest(iter(values)) == publication_digest(values)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_streamed_list_digest_rejects_nonfinite_values(value: float) -> None:
    with pytest.raises(ValueError):
        publication_list_digest(iter([{"vector": [value]}]))


def test_manifest_digest_preserves_nested_json_without_whole_manifest_buffer() -> None:
    import tracemalloc

    from onyx.document_index.publication_models import publication_streaming_digest

    payload: JsonValue = {
        "bindings": [
            {"source": "Türkçe " * 20000, "vector": [1.0, -0.0]} for _ in range(32)
        ],
        "scope": {"tenant": "public"},
    }
    expected = publication_digest(payload)
    tracemalloc.start()
    try:
        actual = publication_streaming_digest(payload)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert actual == expected
    assert peak < 2_000_000, "digest materializes the complete JSON manifest"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_streamed_manifest_digest_rejects_nonfinite_values(value: float) -> None:
    from onyx.document_index.publication_models import publication_streaming_digest

    with pytest.raises(ValueError):
        publication_streaming_digest({"bindings": [{"vector": [value]}]})


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "missing",
        "extra",
        "duplicate",
        "vector",
        "owner",
        "operation",
        "tombstone",
    ],
)
@pytest.mark.parametrize("retain_last", [False, True])
def test_full_verification_is_bounded_and_preserves_ordered_proof(
    monkeypatch: pytest.MonkeyPatch, fault: str | None, retain_last: bool
) -> None:
    authority = OwnedAuthority(uuid4(), tenant_id="public")
    reservations = FileReservations(
        ownership=authority.owner, ordinals=tuple(range(32)), gate_closed=True
    )
    index = observed().observed_index
    projections = []
    for ordinal in range(31):
        source = json.loads(observed().source_json)
        source.update(
            document_id=str(authority.owner.user_file_id), chunk_index=ordinal
        )
        projections.append(
            ObservedPublicationProjection.observe(
                context_projection_id=str(ordinal),
                source_json=json.dumps(source),
                observed_index=index,
            )
        )
    client = MagicMock()
    client.indices.get.return_value = {
        index.index_name: {
            "settings": {"index": {"uuid": index.index_uuid}},
            "mappings": {"properties": {"content_vector": {"dims": 2}}},
        }
    }
    adapter = publication.FencedPublicationIndex(client, index)
    expected = [
        adapter._source(reservations, projections[i] if i < 31 else None, i)
        for i in range(32)
    ]
    retained = ()
    if retain_last:
        projection = projections.pop()
        source_json = json.dumps(expected[30])
        retained = (
            RetainedPublicationProjection(
                evidence=IndexedProjectionEvidence(
                    index=index,
                    source_json=source_json,
                    frozen_projection=None,
                    observed_projection=projection,
                    payload_sha256=None,
                ),
                source_json=source_json,
            ),
        )
        expected[30] = adapter._retained_source(reservations, retained[0])
    legacy_digest = publication_digest(expected)
    serialized = [json.dumps(source) for source in expected]
    del expected
    vectors: list[weakref.ReferenceType[TrackedVector]] = []

    def scan(_client: object, **kwargs: Any) -> Iterator[dict[str, Any]]:
        assert "_source" not in kwargs["query"], "exact verification needs every field"
        order = list(reversed(range(32)))
        if fault == "missing":
            order.pop()
        if fault == "duplicate":
            order.append(0)
        for ordinal in order:
            source = json.loads(serialized[ordinal])
            source["publication_operation"] = "frozen-operation"
            if "content_vector" in source:
                source["content_vector"] = TrackedVector(source["content_vector"])
                vectors.append(weakref.ref(source["content_vector"]))
            if ordinal == 0:
                if fault == "vector":
                    source["content_vector"][0] += 0.01
                elif fault == "owner":
                    source["publication_token"] -= 1
                elif fault == "operation":
                    del source["publication_operation"]
            if ordinal == 31 and fault == "tombstone":
                source["publication_tombstone"] = False
            yield {
                "_id": "foreign"
                if fault == "extra" and ordinal == 0
                else adapter._id(reservations, ordinal),
                "_source": source,
            }
            if fault is None:
                assert sum(ref() is not None for ref in vectors) <= 2, (
                    "verification retains the decoded vector inventory"
                )

    monkeypatch.setattr(publication, "scan", scan)
    if fault is not None:
        with pytest.raises(DocumentChunkVerificationError):
            adapter.verify(reservations, tuple(projections), retained)
    else:
        proof = adapter.verify(reservations, tuple(projections), retained)
        assert proof.manifest_sha256 == legacy_digest
        assert proof.live_ordinals == tuple(range(31))
