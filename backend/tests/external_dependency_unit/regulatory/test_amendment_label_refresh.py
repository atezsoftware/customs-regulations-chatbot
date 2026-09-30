from uuid import uuid4

import pytest
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from onyx.db import regulatory_labeling
from onyx.db.labeling_configuration import resolve_labeling_provider_binding
from onyx.db.models import (
    AmendmentBatch,
    AmendmentProposal,
    DocumentSet__UserFile,
    RegulatoryAmendmentLabelRefresh,
    RegulatoryChunk,
    RegulatoryLabelingItem,
    RegulatoryLabelingRun,
    RegulatoryLabelSettings,
    User,
    UserFile,
)
from onyx.db.regulatory_amendments import finalize_amendment_proposal_projection
from onyx.db.regulatory_label_refresh import (
    list_refreshes,
    reconcile_finished_refreshes,
    record_published_amendment_refresh,
    refresh_snapshot,
    retry_failed_refresh,
    start_next_refresh,
)
from tests.external_dependency_unit.regulatory.test_labeling_jobs import LabelingData
from tests.external_dependency_unit.regulatory.test_labeling_jobs import (
    labeling_data as labeling_data,
)
from tests.external_dependency_unit.regulatory.test_labeling_jobs import (
    labeling_database as labeling_database,
)


def test_published_amendment_relabels_affected_file_without_duplicating_intent(
    labeling_data: LabelingData,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("REGULATORY_AMENDMENT_LABEL_REFRESH_ENABLED", raising=False)
    extra_file_id = uuid4()
    extra_chunk_id = str(uuid4())
    new_chunk_id = str(uuid4())
    with Session(labeling_data.database.engine) as session:
        user = session.get(User, labeling_data.user_id)
        settings = session.get(RegulatoryLabelSettings, 1)
        assert user is not None and settings is not None
        original_taxonomy_id = settings.taxonomy_id
        settings.taxonomy_id = labeling_data.taxonomy_id
        binding = resolve_labeling_provider_binding(
            session, labeling_data.model_id, user=user
        )
        template, _ = regulatory_labeling.create_labeling_run(
            session,
            document_set_id=labeling_data.document_set_id,
            taxonomy=settings.taxonomy,
            model_configuration_id=labeling_data.model_id,
            model="gemini-3.8-flash",
            provider_binding=binding.model_dump(mode="json"),
            requested_by_id=user.id,
            idempotency_key=uuid4(),
            uses_current_labels=True,
        )
        template.status = "completed"
        template.stage = "finished"
        session.add(
            UserFile(
                id=extra_file_id,
                user_id=user.id,
                file_id=str(uuid4()),
                name="Other regulation",
                file_type="text/markdown",
            )
        )
        session.flush()
        session.add(
            DocumentSet__UserFile(
                document_set_id=labeling_data.document_set_id,
                user_file_id=extra_file_id,
            )
        )
        session.add_all(
            [
                RegulatoryChunk(
                    id=extra_chunk_id,
                    user_file_id=extra_file_id,
                    text="Unrelated source chunk that must not be relabeled.",
                    position=0,
                    projection_ordinal=0,
                    heading_path=["Other regulation"],
                    chunk_type="article",
                    chunk_metadata={"chunk_variant": "atomic"},
                    source="indexed",
                    status="active",
                ),
                RegulatoryChunk(
                    id=new_chunk_id,
                    user_file_id=labeling_data.file_id,
                    text="A newly inserted customs rule requiring a fresh label.",
                    position=4,
                    projection_ordinal=4,
                    heading_path=["Customs regulation", "New article"],
                    chunk_type="article",
                    chunk_metadata={"chunk_variant": "atomic"},
                    source="amendment",
                    status="active",
                ),
            ]
        )
        batch = AmendmentBatch(
            document_set_id=labeling_data.document_set_id,
            raw_text="Add a customs rule.",
            user_file_ids=[str(labeling_data.file_id)],
            status="analyzed",
        )
        session.add(batch)
        session.flush()
        proposal = AmendmentProposal(
            batch_id=batch.id,
            instruction_index=0,
            instruction_text=batch.raw_text,
            status="approving",
            applied_new_chunk_id=new_chunk_id,
            applied_new_chunk_ids=[new_chunk_id],
        )
        session.add(proposal)
        session.commit()

        try:
            assert finalize_amendment_proposal_projection(
                session, proposal_id=proposal.id, succeeded=True
            )
            record_published_amendment_refresh(
                session, proposal_id=proposal.id, user_file_id=labeling_data.file_id
            )
            assert (
                session.scalar(
                    select(RegulatoryAmendmentLabelRefresh.id).where(
                        RegulatoryAmendmentLabelRefresh.proposal_id == proposal.id
                    )
                )
                is None
            )
            monkeypatch.setenv("REGULATORY_AMENDMENT_LABEL_REFRESH_ENABLED", "true")
            record_published_amendment_refresh(
                session, proposal_id=proposal.id, user_file_id=labeling_data.file_id
            )
            record_published_amendment_refresh(
                session, proposal_id=proposal.id, user_file_id=labeling_data.file_id
            )
            session.commit()
            intents = session.scalars(
                select(RegulatoryAmendmentLabelRefresh).where(
                    RegulatoryAmendmentLabelRefresh.proposal_id == proposal.id
                )
            ).all()
            assert len(intents) == 1
            assert intents[0].new_chunk_ids == [new_chunk_id]

            run_id = start_next_refresh(session)
            assert run_id is not None
            session.commit()
            run = session.get(RegulatoryLabelingRun, run_id)
            assert run is not None
            assert run.file_ids == [str(labeling_data.file_id)]
            item_ids = set(
                session.scalars(
                    select(RegulatoryLabelingItem.regulatory_chunk_id).where(
                        RegulatoryLabelingItem.run_id == run_id
                    )
                ).all()
            )
            assert new_chunk_id in item_ids
            assert labeling_data.first_id in item_ids
            assert extra_chunk_id not in item_ids
            assert intents[0].status == "running"

            later = AmendmentProposal(
                batch_id=batch.id,
                instruction_index=1,
                instruction_text="A later change to the same source.",
                status="approved",
                applied_new_chunk_id=new_chunk_id,
                applied_new_chunk_ids=[new_chunk_id],
            )
            session.add(later)
            session.flush()
            record_published_amendment_refresh(
                session, proposal_id=later.id, user_file_id=labeling_data.file_id
            )
            assert start_next_refresh(session) is None
            later_intent = session.scalar(
                select(RegulatoryAmendmentLabelRefresh).where(
                    RegulatoryAmendmentLabelRefresh.proposal_id == later.id
                )
            )
            assert later_intent is not None and later_intent.status == "pending"

            run.status = "completed"
            run.stage = "finished"
            session.flush()
            assert reconcile_finished_refreshes(session) == 1
            assert intents[0].status == "completed"
            later_intent.next_retry_at = None
            later_run_id = start_next_refresh(session)
            assert later_run_id is not None and later_run_id != run_id
            assert later_intent.run_id == later_run_id
            for attempt in range(1, 4):
                failed_run = session.get(RegulatoryLabelingRun, later_intent.run_id)
                assert failed_run is not None
                failed_run.status = "failed"
                failed_run.stage = "finished"
                session.flush()
                assert reconcile_finished_refreshes(session) == 1
                assert later_intent.attempt_count == attempt
                if attempt < 3:
                    assert later_intent.status == "pending"
                    later_intent.next_retry_at = None
                    assert start_next_refresh(session) is not None
                else:
                    assert later_intent.status == "failed"
            visible = list_refreshes(
                session, document_set_id=labeling_data.document_set_id
            )
            assert {row.id for row in visible} == {intents[0].id, later_intent.id}
            assert refresh_snapshot(later_intent).status == "failed"
            retried = retry_failed_refresh(
                session,
                document_set_id=labeling_data.document_set_id,
                refresh_id=later_intent.id,
            )
            assert retried is not None
            assert retried.status == "pending"
            assert retried.attempt_count == 0
            assert retried.run_id is None
            assert proposal.status == "approved"
        finally:
            session.rollback()
            settings = session.get(RegulatoryLabelSettings, 1)
            assert settings is not None
            settings.taxonomy_id = original_taxonomy_id
            session.execute(
                delete(RegulatoryChunk).where(
                    RegulatoryChunk.user_file_id == extra_file_id
                )
            )
            session.execute(delete(UserFile).where(UserFile.id == extra_file_id))
            session.commit()
