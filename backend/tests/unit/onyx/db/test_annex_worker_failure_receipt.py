"""Worker failure receipts bind the current lease and exact private canary owner."""

import json
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest

from onyx.db import regulatory_amendments as jobs
from onyx.db import regulatory_annex_acceptance as acceptance
from onyx.db.models import AmendmentBatch, AmendmentSourcePackage, DocumentSet, KVStore


def batch_fixture() -> AmendmentBatch:
    return AmendmentBatch(
        id=44,
        document_set_id=18,
        created_by=uuid4(),
        source_package_id=uuid4(),
        user_file_ids=[str(uuid4())],
        stage="segmenting",
        status="analyzing",
        lease_generation=3,
    )


@pytest.mark.parametrize("current", [True, False])
def test_failure_receipt_is_atomic_with_current_lease_only(current: bool) -> None:
    batch = batch_fixture()
    session = MagicMock()
    session.scalar.return_value = batch
    error = RuntimeError("NEVER_EMIT_secret_or_source")
    result = jobs.mark_batch_failed(
        session,
        batch_id=44,
        lease_generation=3 if current else 2,
        error_message="Analysis failed. Retry to resume from the last completed instruction.",
        failure=error,
    )
    assert result is current
    if not current:
        session.add.assert_not_called()
        session.commit.assert_not_called()
        session.rollback.assert_called_once()
        return
    row = session.add.call_args.args[0]
    assert row.key == "regulatory_amendment_failure:44:3"
    assert row.value["batch_id"] == 44
    assert row.value["lease_generation"] == 3
    assert row.value["document_set_id"] == 18
    assert json.loads(row.value["detail"])["exceptions"][0]["type"] == "RuntimeError"
    assert "NEVER_EMIT" not in json.dumps(row.value)
    assert (
        batch.error_message
        == "Analysis failed. Retry to resume from the last completed instruction."
    )
    assert batch.status == "failed"
    session.commit.assert_called_once()


@pytest.mark.parametrize(
    "mismatch",
    [
        None,
        "owner",
        "scope",
        "package",
        "file",
        "lease",
        "saved_run",
        "public",
        "receipt",
        "package_owner",
        "no_receipt",
        "not_failed",
    ],
)
def test_canary_reads_only_exact_owned_failed_lease(
    monkeypatch: pytest.MonkeyPatch, mismatch: str | None
) -> None:
    batch = batch_fixture()
    batch.status = "failed"
    assert batch.created_by is not None
    run = acceptance.CanaryRun(
        release_sha="a" * 40,
        user_id=batch.created_by,
        file_id=UUID(batch.user_file_ids[0]),
        document_set_id=18,
        batch_id=44,
        package_id=batch.source_package_id,
    )
    saved = KVStore(key=run.key, value=run.model_dump(mode="json"))
    scope = DocumentSet(id=18, name=run.name, user_id=run.user_id, is_public=False)
    package = AmendmentSourcePackage(
        id=run.package_id, document_set_id=18, created_by=run.user_id
    )
    detail = '{"stage":"source_review","exceptions":[]}'
    receipt_payload = {
        "batch_id": 44,
        "lease_generation": 3,
        "document_set_id": 18,
        "created_by": str(run.user_id),
        "source_package_id": str(run.package_id),
        "user_file_ids": [str(run.file_id)],
        "detail": detail,
    }
    receipt = KVStore(key="regulatory_amendment_failure:44:3", value=receipt_payload)
    if mismatch == "owner":
        batch.created_by = uuid4()
    if mismatch == "scope":
        batch.document_set_id = 19
    if mismatch == "package":
        batch.source_package_id = uuid4()
    if mismatch == "file":
        batch.user_file_ids = [str(uuid4())]
    if mismatch == "lease":
        receipt_payload["lease_generation"] = 2
    if mismatch == "saved_run":
        saved.value = {**run.model_dump(mode="json"), "run_id": str(uuid4())}
    if mismatch == "public":
        scope.is_public = True
    if mismatch == "receipt":
        receipt_payload["created_by"] = str(uuid4())
    if mismatch == "package_owner":
        package.created_by = uuid4()
    if mismatch == "not_failed":
        batch.status = "analyzing"
    session = MagicMock()
    rows = {
        (KVStore, run.key): saved,
        (AmendmentBatch, 44): batch,
        (DocumentSet, 18): scope,
        (KVStore, receipt.key): receipt,
        (AmendmentSourcePackage, run.package_id): package,
    }
    if mismatch == "no_receipt":
        del rows[KVStore, receipt.key]
    session.get.side_effect = lambda model, key: rows.get((model, key))
    session.scalar.return_value = scope
    factory = MagicMock()
    factory.return_value.__enter__.return_value = session
    monkeypatch.setattr(acceptance, "get_session_with_current_tenant", factory)
    if mismatch in {"no_receipt", "not_failed"}:
        assert acceptance.read_canary_worker_failure(run) is None
    elif mismatch:
        with pytest.raises(ValueError, match="canary_worker_failure"):
            acceptance.read_canary_worker_failure(run)
    else:
        assert acceptance.read_canary_worker_failure(run) == (receipt.key, detail)
        retained = acceptance.retained_canary_audit(run)
        assert (
            acceptance.RetainedArtifact(kind="worker_failure_receipt", id=receipt.key)
            in retained
        )
    session.add.assert_not_called()
    session.commit.assert_not_called()
