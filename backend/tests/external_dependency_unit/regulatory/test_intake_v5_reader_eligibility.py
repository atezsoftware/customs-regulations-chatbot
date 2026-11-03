"""Real owned v5 sources retain native reader checks without mixed-profile lineage."""

from typing import cast
from uuid import UUID

import pytest
from sqlalchemy.orm import Session

from onyx.db.models import RegulatoryChunk
from onyx.db.regulatory_labeling_review_authority import labeling_review_authority
from tests.external_dependency_unit.regulatory import (
    test_intake_labeling_review_authority as fixtures,
)
from tests.external_dependency_unit.regulatory.test_labeling_jobs import LabelingData
from tests.external_dependency_unit.regulatory.test_labeling_jobs import (
    labeling_data as labeling_data,
)
from tests.external_dependency_unit.regulatory.test_labeling_jobs import (
    labeling_database as labeling_database,
)


@pytest.mark.parametrize("uniform", [True, False])
def test_native_v5_reader_requires_current_owned_uniform_policy(
    labeling_data: LabelingData, monkeypatch: pytest.MonkeyPatch, uniform: bool
) -> None:
    original = fixtures.seed_v4_owned_intake_source

    def seed(session: Session, data: LabelingData, *, raw_sha256: str) -> UUID:
        asset = original(session, data, raw_sha256=raw_sha256)
        for identifier in (
            (data.first_id, data.second_id) if uniform else (data.first_id,)
        ):
            row = session.get_one(RegulatoryChunk, identifier)
            row.chunk_metadata = {
                **row.chunk_metadata,
                "intake_chunking_version": "5",
                "intake_preparation_policy": {
                    **row.chunk_metadata["intake_preparation_policy"],
                    "chunking_version": "5",
                },
            }
        session.flush()
        return asset

    monkeypatch.setattr(fixtures, "seed_v4_owned_intake_source", seed)
    with Session(labeling_data.database.engine) as session:
        if not uniform:
            with pytest.raises(ValueError, match="one frozen whole-table profile"):
                fixtures._prepared_run(session, labeling_data, reader=True)
        else:
            run = fixtures._prepared_run(session, labeling_data, reader=True)
            authority = labeling_review_authority(session, run.id)
            assert (
                authority is not None
                and len(cast(list[object], authority["structural_readers"])) == 1
            )
