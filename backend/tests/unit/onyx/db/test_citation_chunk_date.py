"""Exact citation date selection never invents authority outside retained windows."""

import json
from datetime import date
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from sqlalchemy.orm import Session

from onyx.db import regulatory_public_reads as reads
from onyx.db.models import RegulatoryChunk
from onyx.document_index.publication_models import (
    FrozenPublicationProjection,
    PublicationIndexSnapshot,
)
from onyx.regulatory.amendments.annexes.models import AnnexTemporalProjection


@pytest.fixture
def binding() -> AnnexTemporalProjection:
    return AnnexTemporalProjection(
        id=uuid4(),
        index=PublicationIndexSnapshot(
            index_name="current-index",
            index_uuid="physical-one",
            search_settings_id=9,
            model_provider="fixture",
            model_name="fixture",
            vector_dimension=3,
            embedding_config_sha256="0" * 64,
            multitenant=False,
        ),
        projection=FrozenPublicationProjection(
            ordinal=4,
            context_projection_id="context",
            source_json=json.dumps(
                {
                    "document_id": str(uuid4()),
                    "chunk_index": 4,
                    "regulatory_chunk_id": "canonical",
                    "content": "5%",
                    "content_vector": [0.1, 0.2, 0.3],
                    "source_type": "file",
                    "public": False,
                    "access_control_list": [],
                    "global_boost": 1,
                    "semantic_identifier": "fixture",
                    "blurb": "5%",
                    "doc_summary": "",
                    "chunk_context": "",
                }
            ),
            embedding_inputs=("5%",),
            embedding_config_json="{}",
        ),
        canonical_base_sha256="0" * 64,
        derived_role="canonical",
        dependency_ids=[],
        representation_text="5%",
        reference_date=None,
        effective_start=None,
        effective_end=None,
        semantic_position=0,
    )


@pytest.mark.parametrize(
    "start,end,reference,canonical_start,canonical_end,expected",
    [
        (
            date(2020, 1, 1),
            date(2026, 9, 10),
            date(2025, 1, 1),
            None,
            None,
            date(2025, 1, 1),
        ),
        (
            date(2020, 1, 1),
            date(2026, 9, 10),
            date(2026, 9, 10),
            None,
            None,
            date(2020, 1, 1),
        ),
        (None, date(2026, 9, 10), None, None, None, date(2026, 9, 9)),
        (None, None, None, date(2025, 1, 1), date(2026, 1, 1), date(2025, 1, 1)),
        (
            date(2020, 1, 1),
            None,
            date(2020, 1, 1),
            date(2025, 1, 1),
            None,
            date(2025, 1, 1),
        ),
        (date.max, None, None, None, None, date.max),
        (None, date.min, None, None, None, None),
        (date(2026, 9, 10), date(2026, 9, 10), None, None, None, None),
        (date(2026, 9, 11), date(2026, 9, 10), None, None, None, None),
        (date(2020, 1, 1), date(2021, 1, 1), None, date(2022, 1, 1), None, None),
    ],
)
def test_citation_interval_edges(
    binding: AnnexTemporalProjection,
    monkeypatch: pytest.MonkeyPatch,
    start: date | None,
    end: date | None,
    reference: date | None,
    canonical_start: date | None,
    canonical_end: date | None,
    expected: date | None,
) -> None:
    identifier = uuid4()
    candidate = binding.model_copy(
        update={
            "effective_start": start,
            "effective_end": end,
            "reference_date": reference,
        }
    )
    payload = candidate.model_dump_json()
    monkeypatch.setattr(
        reads, "load_file_temporal_bindings", lambda *_args, **_kwargs: [candidate]
    )
    session = MagicMock(spec=Session)
    session.get.return_value = RegulatoryChunk(
        id="canonical",
        user_file_id=identifier,
        validity_start_date=canonical_start,
        validity_end_date=canonical_end,
    )

    def resolve() -> date | None:
        return reads.citation_chunk_as_of_date(
            session,
            document_id=str(identifier),
            chunk_id=4,
            search_settings_id=9,
            index_name="current-index",
        )

    if expected is None:
        with pytest.raises(ValueError, match="no valid retained legal interval"):
            resolve()
    else:
        assert resolve() == expected
    assert candidate.model_dump_json() == payload


def test_citation_requires_unique_current_settings_binding(
    binding: AnnexTemporalProjection, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = MagicMock(spec=Session)
    identifier = str(uuid4())
    candidates = [binding]
    monkeypatch.setattr(
        reads, "load_file_temporal_bindings", lambda *_args, **_kwargs: candidates
    )
    assert (
        reads.citation_chunk_as_of_date(
            session,
            document_id=identifier,
            chunk_id=4,
            search_settings_id=10,
            index_name="current-index",
        )
        is None
    )
    assert (
        reads.citation_chunk_as_of_date(
            session,
            document_id=identifier,
            chunk_id=4,
            search_settings_id=9,
            index_name="different-index",
        )
        is None
    )
    candidates.append(
        binding.model_copy(
            update={
                "id": uuid4(),
                "index": binding.index.model_copy(
                    update={"index_uuid": "other-physical"}
                ),
            }
        )
    )
    with pytest.raises(ValueError, match="ambiguous"):
        reads.citation_chunk_as_of_date(
            session,
            document_id=identifier,
            chunk_id=4,
            search_settings_id=9,
            index_name="current-index",
        )
    session.get.assert_not_called()


@pytest.mark.parametrize("document_id", ["legacy-document", str(uuid4())])
def test_unqualified_legacy_identity_retains_default_date(
    monkeypatch: pytest.MonkeyPatch, document_id: str
) -> None:
    monkeypatch.setattr(
        reads, "load_file_temporal_bindings", lambda *_args, **_kwargs: []
    )
    assert (
        reads.citation_chunk_as_of_date(
            MagicMock(spec=Session),
            document_id=document_id,
            chunk_id=4,
            search_settings_id=9,
            index_name="current-index",
        )
        is None
    )
