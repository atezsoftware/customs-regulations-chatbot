"""Ordinal discovery must preserve ownership guards without downloading vectors."""

from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from onyx.document_index.elasticsearch import publication
from onyx.document_index.publication_models import FileReservations
from tests.unit.onyx.document_index.elasticsearch.test_observed_publication import (
    observed,
)
from tests.unit.onyx.regulatory.indexing_jobs.owned_publication_test_helpers import (
    OwnedAuthority,
)


@pytest.mark.parametrize(
    "source,malformed_id,reason",
    [
        ({"chunk_index": 5}, False, None),
        ({"chunk_index": 5, "publication_floor": 0}, False, "durable reservation"),
        (
            {"chunk_index": 5, "publication_future_marker": None},
            False,
            "durable reservation",
        ),
        ({"chunk_index": True}, False, "identity is invalid"),
        ({"chunk_index": -1}, False, "identity is invalid"),
        ({"chunk_index": 2**63}, False, "identity is invalid"),
        ({}, False, "identity is invalid"),
        ({"chunk_index": 5}, True, "identity is invalid"),
    ],
)
def test_ordinal_discovery_requests_only_identity_and_all_publication_controls(
    monkeypatch: pytest.MonkeyPatch,
    source: dict[str, Any],
    malformed_id: bool,
    reason: str | None,
) -> None:
    authority = OwnedAuthority(uuid4(), tenant_id="public")
    reservations = FileReservations(
        ownership=authority.owner, ordinals=(), gate_closed=False
    )
    index = observed().observed_index
    client = MagicMock()
    client.indices.get.return_value = {
        index.index_name: {
            "settings": {"index": {"uuid": index.index_uuid}},
            "mappings": {"properties": {"content_vector": {"dims": 2}}},
        }
    }
    adapter = publication.FencedPublicationIndex(client, index)

    def scan(
        _client: object, *, index: str, query: dict[str, Any]
    ) -> Iterator[dict[str, Any]]:
        assert index == adapter.snapshot.index_name
        assert query.get("_source") == ["chunk_index", "publication_*"], (
            "ordinal discovery must not transfer content or vectors"
        )
        return iter(
            [
                {
                    "_id": "foreign" if malformed_id else adapter._id(reservations, 5),
                    "_source": source,
                }
            ]
        )

    monkeypatch.setattr(publication, "scan", scan)
    if reason is None:
        assert adapter.existing_ordinals(reservations) == (5,)
    else:
        with pytest.raises(ValueError, match=reason):
            adapter.existing_ordinals(reservations)
