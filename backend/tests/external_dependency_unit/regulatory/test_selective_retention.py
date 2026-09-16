import json
from datetime import timedelta
from uuid import UUID, uuid4

import pytest
from elasticsearch import BadRequestError, Elasticsearch

from onyx.document_index.publication_models import RetainedPublicationProjection
from tests.external_dependency_unit.regulatory.test_publication_primitives import (
    adapter_for,
    frozen_projection,
    store,
)
from tests.external_dependency_unit.regulatory.test_publication_primitives import (
    es as es,
)
from tests.external_dependency_unit.regulatory.test_publication_primitives import (
    owned_file as owned_file,
)


def test_legacy_vector_survives_closing_retry_and_new_fence(
    owned_file: UUID, es: tuple[Elasticsearch, str]
) -> None:
    authority = store()
    owner = authority.acquire(owned_file, owner_id=uuid4(), ttl=timedelta(minutes=2))
    adapter = adapter_for(es)
    baseline = authority.reservations(owner)
    original = frozen_projection(owned_file, 0)
    source = json.loads(original.source_json)
    source.update(title="Retained title", title_vector=[0.6, 0.7, 0.8])
    es[0].index(index=es[1], id=adapter._id(baseline, 0), document=source, refresh=True)
    evidence = adapter.read_evidence(baseline, 0)
    assert evidence.frozen_projection is None
    closed = dict(source, validity_end_date=1800000000)
    retained = RetainedPublicationProjection(
        evidence=evidence, source_json=json.dumps(closed)
    )
    authority.close_gate(owner)
    reservations = authority.reservations(owner)
    adapter.seal(reservations)
    adapter.retain(reservations, retained)
    adapter.retain(reservations, retained)
    adapter.tombstone(reservations, 1_000_000_042)
    proof = adapter.verify(reservations, (), (retained,))
    assert proof.live_ordinals == (0,)
    actual = adapter.read_evidence(reservations, 0)
    saved = json.loads(actual.source_json)
    assert saved["content_vector"] == source["content_vector"]
    assert saved["title_vector"] == source["title_vector"]
    assert "publication_evidence" not in saved
    assert actual.frozen_projection is None
    authority.release(owner)
    successor = authority.acquire(
        owned_file, owner_id=uuid4(), ttl=timedelta(minutes=2)
    )
    successor_reservations = authority.reservations(successor)
    adapter.seal(successor_reservations)
    with pytest.raises(BadRequestError):
        adapter.retain(reservations, retained)
    adapter.retain(successor_reservations, retained)
    adapter.tombstone(successor_reservations, 1_000_000_042)
    adapter.verify(successor_reservations, (), (retained,))
    authority.release(successor)


def test_retention_rejects_content_vector_or_scope_replacement(
    owned_file: UUID, es: tuple[Elasticsearch, str]
) -> None:
    authority = store()
    owner = authority.acquire(owned_file, owner_id=uuid4(), ttl=timedelta(minutes=2))
    adapter = adapter_for(es)
    reservations = authority.reservations(owner)
    source = json.loads(frozen_projection(owned_file, 0).source_json)
    es[0].index(
        index=es[1], id=adapter._id(reservations, 0), document=source, refresh=True
    )
    evidence = adapter.read_evidence(reservations, 0)
    for field, value in [
        ("content", "invented"),
        ("content_vector", [0.4, 0.5, 0.6]),
        ("document_id", str(uuid4())),
    ]:
        with pytest.raises(ValueError, match="content or vector"):
            RetainedPublicationProjection(
                evidence=evidence, source_json=json.dumps({**source, field: value})
            )
    retained = RetainedPublicationProjection(
        evidence=evidence, source_json=json.dumps(source)
    )
    authority.close_gate(owner)
    reservations = authority.reservations(owner)
    adapter.seal(reservations)
    es[0].update(
        index=es[1],
        id=adapter._id(reservations, 0),
        doc={"content": "concurrent change"},
    )
    with pytest.raises(BadRequestError):
        adapter.retain(reservations, retained)
    authority.release(owner)
