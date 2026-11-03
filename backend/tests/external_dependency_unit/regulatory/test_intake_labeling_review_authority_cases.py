"""Current reader authority rejects drift while preserving partial editing."""

from typing import cast

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from onyx.db.models import (
    RegulatoryDerivedLabelProjection,
    RegulatoryLabelingItem,
)
from onyx.db.regulatory_labeling_readers import STRUCTURAL_READER_KEY
from onyx.db.regulatory_labeling_review_authority import labeling_review_authority
from tests.external_dependency_unit.regulatory.test_intake_labeling_review_authority import (
    _prepared_run,
)
from tests.external_dependency_unit.regulatory.test_labeling_jobs import LabelingData
from tests.external_dependency_unit.regulatory.test_labeling_jobs import (
    labeling_data as labeling_data,
)
from tests.external_dependency_unit.regulatory.test_labeling_jobs import (
    labeling_database as labeling_database,
)


@pytest.mark.parametrize(
    "tamper",
    [
        "labels",
        "provenance",
        "proof",
        "child_context",
        "child_context_text",
        "child_output",
    ],
)
def test_reader_authority_rejects_proof_projection_or_current_child_tampering(
    labeling_data: LabelingData, tamper: str
) -> None:
    with Session(labeling_data.database.engine) as session:
        run = _prepared_run(session, labeling_data, reader=True)
        assert labeling_review_authority(session, run.id) is not None
        target = session.scalar(
            select(RegulatoryDerivedLabelProjection).where(
                RegulatoryDerivedLabelProjection.run_id == run.id,
                RegulatoryDerivedLabelProjection.regulatory_chunk_id
                == labeling_data.first_id,
            )
        )
        assert target is not None
        if tamper == "labels":
            target.labels = ["customs_value"]
        elif tamper == "provenance":
            provenance: dict[str, object] = {"origin": []}
            target.provenance = provenance
        elif tamper == "proof":
            target.source_snapshot = {
                **target.source_snapshot,
                STRUCTURAL_READER_KEY: None,
            }
        else:
            child = session.scalar(
                select(RegulatoryLabelingItem).where(
                    RegulatoryLabelingItem.run_id == run.id,
                    RegulatoryLabelingItem.regulatory_chunk_id
                    == labeling_data.second_id,
                )
            )
            assert child is not None
            if tamper == "child_context_text":
                assert child.context_snapshot is not None
                child.context_snapshot += " changed context"
            elif tamper == "child_output":
                assignments: list[dict[str, object]] = [
                    {"label_id": "origin", "evidence_quote": "not in canonical target"}
                ]
                child.assignments = assignments
            else:
                child.source_snapshot = {
                    **child.source_snapshot,
                    "context_member_ids": [],
                }
        session.flush()
        with pytest.raises(ValueError):
            labeling_review_authority(session, run.id)


def test_partial_unresolved_reader_retains_truthful_edit_authority(
    labeling_data: LabelingData,
) -> None:
    with Session(labeling_data.database.engine) as session:
        run = _prepared_run(session, labeling_data, reader=True)
        target = session.scalar(
            select(RegulatoryDerivedLabelProjection).where(
                RegulatoryDerivedLabelProjection.run_id == run.id,
                RegulatoryDerivedLabelProjection.regulatory_chunk_id
                == labeling_data.first_id,
            )
        )
        child = session.scalar(
            select(RegulatoryLabelingItem).where(
                RegulatoryLabelingItem.run_id == run.id,
                RegulatoryLabelingItem.regulatory_chunk_id == labeling_data.second_id,
            )
        )
        assert target is not None and child is not None
        run.status = "completed_with_errors"
        child.status, child.error = "failed", "Model abstained: insufficient evidence"
        child.labels, child.assignments = [], []
        target.resolution, target.unresolved_reason = (
            "unresolved",
            "source_label_result_unavailable",
        )
        target.labels, target.provenance = [], {}
        session.flush()
        partial = labeling_review_authority(session, run.id)
        assert partial is not None
        assert (
            cast(list[dict[str, object]], partial["structural_readers"])[0][
                "resolution"
            ]
            == "unresolved"
        )
        child.error = "Different failed result"
        session.flush()
        assert partial != labeling_review_authority(session, run.id)
        run.status = "completed"
        session.flush()
        with pytest.raises(ValueError):
            labeling_review_authority(session, run.id)
