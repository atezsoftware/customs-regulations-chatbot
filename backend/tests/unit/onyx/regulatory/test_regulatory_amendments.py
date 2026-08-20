from unittest.mock import MagicMock
from uuid import UUID

import pytest
from sqlalchemy.orm import Session

from onyx.db.regulatory_amendments import approve_amendment_proposal
from onyx.regulatory.amendments import candidate_finder


def test_amendment_candidate_queries_exclude_hierarchical_aggregates() -> None:
    for statement in (
        candidate_finder._TEXT_TRGM_SQL,
        candidate_finder._HEADING_TRGM_SQL,
        candidate_finder._STRUCTURED_SQL,
    ):
        assert "chunk_type IS DISTINCT FROM 'hierarchical_aggregate'" in str(statement)


def test_amendment_approval_rejects_derived_aggregate_target() -> None:
    proposal = MagicMock()
    proposal.id = 41
    proposal.status = "pending"
    proposal.old_chunk_id = "aggregate-id"
    proposal.new_chunk_draft = {
        "user_file_id": str(UUID("00000000-0000-0000-0000-000000000123")),
        "position": 3,
        "text": "Yeni metin",
    }
    aggregate = MagicMock()
    aggregate.status = "active"
    aggregate.chunk_type = "hierarchical_aggregate"
    aggregate.chunk_metadata = {"chunk_variant": "hierarchical_aggregate"}
    db_session = MagicMock(spec=Session)
    db_session.get.return_value = aggregate

    with pytest.raises(
        ValueError, match="Derived aggregate chunks cannot be amended directly"
    ):
        approve_amendment_proposal(
            db_session,
            proposal,
            decided_by=None,
        )

    db_session.add.assert_not_called()
