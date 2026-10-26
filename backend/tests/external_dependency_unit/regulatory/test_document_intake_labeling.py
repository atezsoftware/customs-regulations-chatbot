"""Intake-only labeling performance policy is fenced by durable current scope."""

from collections.abc import Iterator
from contextlib import contextmanager
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from sqlalchemy import update
from sqlalchemy.orm import Session

from onyx.db import document_intake as intake
from onyx.db import document_intake_edit as editing
from onyx.db import document_intake_reprepare as reprepare
from onyx.db import regulatory_labeling as labeling
from onyx.db.document_intake_labeling import is_current_intake_labeling_run
from onyx.db.enums import UserFileStatus
from onyx.db.labeling_configuration import LabelingProviderBinding
from onyx.db.models import (
    DocumentSet,
    ModelConfiguration,
    RegulatoryDocumentIntake,
    RegulatoryDocumentIntakeFile,
    RegulatoryLabelingRun,
    User,
    UserFile,
)
from onyx.regulatory.indexing_jobs.models import VertexAuthenticationMode
from onyx.regulatory.labeling import orchestrator
from tests.external_dependency_unit.regulatory.test_document_intake import (
    _admitted,
    _completed_labels,
)
from tests.external_dependency_unit.regulatory.test_document_intake import (
    intake_owner as intake_owner,
)
from tests.external_dependency_unit.regulatory.test_document_intake import (
    intake_session as intake_session,
)


def _current_run(
    session: Session, owner: tuple[User, DocumentSet, ModelConfiguration]
) -> tuple[
    RegulatoryDocumentIntake,
    list[RegulatoryDocumentIntakeFile],
    RegulatoryLabelingRun,
]:
    user, target, model = owner
    batch, members = _admitted(session, user, target, model)
    run = _completed_labels(session, batch, members)
    definition: dict[str, object] = {
        "name": "Intake labels",
        "labels": [{"id": "customs", "name": "Gümrük", "description": "Gümrük"}],
    }
    run.taxonomy.definition = definition
    binding = LabelingProviderBinding(
        model_configuration_id=model.id,
        provider_id=model.llm_provider.id,
        project="intake-policy-test",
        location="global",
        authentication_mode=VertexAuthenticationMode.WORKLOAD_IDENTITY,
        credential_identity="fixture",
        model="gemini-3.5-flash",
        transport="vertex_direct_v1",
    )
    run.provider_binding = binding.model_dump(mode="json")
    run.requested_by_id = user.id
    run.status, run.stage = "queued", "preparing"
    for member in members:
        member.status = "labeling"
    session.flush()
    return batch, members, run


@pytest.mark.parametrize("scope", ["current", "ordinary", "broadened", "published"])
def test_only_current_unpublished_intake_gets_direct_policy(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    monkeypatch: pytest.MonkeyPatch,
    scope: str,
) -> None:
    batch, members, run = _current_run(intake_session, intake_owner)
    if scope == "ordinary":
        batch.labeling_run_id = None
    elif scope == "broadened":
        run.file_ids = [*run.file_ids, str(uuid4())]
    elif scope == "published":
        intake_session.get_one(
            UserFile, members[0].user_file_id
        ).status = UserFileStatus.COMPLETED
    intake_session.flush()
    assert is_current_intake_labeling_run(intake_session, run) == (scope == "current")
    lease = labeling.claim_run(
        intake_session, run_id=run.id, expected_generation=0, lease_seconds=300
    )
    assert lease is not None

    @contextmanager
    def session_scope() -> Iterator[Session]:
        yield intake_session

    resolver = MagicMock()
    grouping = MagicMock(wraps=orchestrator.shard_by_count_and_bytes)
    monkeypatch.setattr(orchestrator, "get_session_with_current_tenant", session_scope)
    monkeypatch.setattr(orchestrator, "resolve_labeling_gateway", resolver)
    monkeypatch.setattr(orchestrator, "shard_by_count_and_bytes", grouping)
    orchestrator._gateway_for_claimed_run(lease)
    assert resolver.call_args.kwargs["direct_max_concurrency"] == (
        4 if scope == "current" else 1
    )
    orchestrator._prepare(lease, "public")
    assert grouping.call_args.kwargs["item_limit"] == (8 if scope == "current" else 64)


def test_completed_previous_run_loses_intake_policy_when_replaced(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
) -> None:
    batch, members, previous = _current_run(intake_session, intake_owner)
    previous.status = "completed"
    current = _completed_labels(intake_session, batch, members)
    current.status = "queued"
    current.requested_by_id = intake_owner[0].id
    intake_session.flush()
    assert not is_current_intake_labeling_run(intake_session, previous)
    assert is_current_intake_labeling_run(intake_session, current)


def test_association_check_reads_database_instead_of_cached_batch(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
) -> None:
    batch, _members, run = _current_run(intake_session, intake_owner)
    assert is_current_intake_labeling_run(intake_session, run)
    intake_session.execute(
        update(RegulatoryDocumentIntake)
        .where(RegulatoryDocumentIntake.id == batch.id)
        .values(labeling_run_id=None)
        .execution_options(synchronize_session=False)
    )
    assert batch.labeling_run_id == run.id
    assert not is_current_intake_labeling_run(intake_session, run)


def test_active_lease_prevents_edit_or_reprepare_reassociation(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
) -> None:
    batch, members, run = _current_run(intake_session, intake_owner)
    lease = labeling.claim_run(
        intake_session, run_id=run.id, expected_generation=0, lease_seconds=300
    )
    assert lease is not None
    batch.status = "partial"
    intake_session.flush()
    assert editing.edit_review_digest(intake_session, batch) is None
    with pytest.raises(intake.IntakeConflictError, match="active.*labeling"):
        reprepare.reprepare_snapshot(intake_session, batch)
    assert batch.labeling_run_id == run.id
    assert {str(member.user_file_id) for member in members} == set(run.file_ids)
    batch.status = "preparing"
    intake_session.flush()
    assert is_current_intake_labeling_run(intake_session, run)
