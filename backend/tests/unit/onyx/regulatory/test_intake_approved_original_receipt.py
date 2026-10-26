from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from sqlalchemy.orm import Session

from onyx.db import document_intake as intake
from onyx.db import document_intake_original_ingestion as authority
from onyx.db.enums import UserFileStatus
from onyx.db.models import (
    AmendmentSourcePackage,
    RegulatoryDocumentIntake,
    RegulatoryFilePublication,
    RegulatorySourceAsset,
    UserFile,
)
from onyx.file_processing.original_ingestion import OriginalIngestionReceipt


def _approved(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    MagicMock,
    OriginalIngestionReceipt,
    dict[str, SimpleNamespace],
    MagicMock,
]:
    original = OriginalIngestionReceipt(
        file_id="owned-original",
        raw_sha256="a" * 64,
        documents_sha256="b" * 64,
        plaintext_sha256="c" * 64,
        canonical_sha256="d" * 64,
        generation_hash="e" * 64,
    )
    user_id, file_id, batch_id, package_id, asset_id = (uuid4() for _ in range(5))
    state = {
        "member": SimpleNamespace(
            id=uuid4(),
            intake_id=batch_id,
            user_file_id=file_id,
            source_asset_id=asset_id,
            status="completed",
            approval_attempt_id=uuid4(),
        ),
        "batch": SimpleNamespace(
            id=batch_id,
            status="completed",
            stage="completed",
            review_sha256="f" * 64,
            source_package_id=package_id,
            created_by=user_id,
            document_set_id=15,
        ),
        "file": SimpleNamespace(
            id=file_id,
            user_id=user_id,
            status=UserFileStatus.COMPLETED,
            file_id=original.file_id,
            regulatory_chunk_generation_hash=original.generation_hash,
        ),
        "publication": SimpleNamespace(
            gate_closed=False,
            epoch=1,
            writer_manifest=None,
            original_ingestion_receipt=original.model_dump(mode="json"),
        ),
        "asset": SimpleNamespace(package_id=package_id, sha256=original.raw_sha256),
        "package": SimpleNamespace(
            status="ready", issues=[], created_by=user_id, document_set_id=15
        ),
        "chunk": SimpleNamespace(
            user_file_id=file_id,
            source="indexed",
            supersedes_chunk_id=None,
            superseded_by_chunk_id=None,
            projection_ordinal=42,
        ),
    }
    session = MagicMock(spec=Session)
    session.scalar.return_value = state["member"]
    session.scalars.return_value = [state["member"]]
    session.execute.return_value.all.return_value = [(state["chunk"], 42)]
    entities = {
        RegulatoryDocumentIntake: "batch",
        UserFile: "file",
        RegulatoryFilePublication: "publication",
        RegulatorySourceAsset: "asset",
        AmendmentSourcePackage: "package",
    }
    session.get.side_effect = lambda model, *_args, **_kwargs: state[entities[model]]
    resolver = MagicMock(return_value=original)
    monkeypatch.setattr(authority, "reprepared_original_receipt", resolver)
    monkeypatch.setattr(
        intake, "review_digest", lambda *_args: state["batch"].review_sha256
    )
    return session, original, state, resolver


def test_approved_review_derives_only_final_canonical_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, original, state, resolver = _approved(monkeypatch)
    effective = original.model_copy(
        update={"documents_sha256": "1" * 64, "canonical_sha256": "2" * 64}
    )
    resolver.return_value = effective
    frozen = deepcopy(state["publication"].original_ingestion_receipt)
    result = authority.approved_intake_original_receipt(
        session, state["file"].id, original, canonical_sha256="3" * 64
    )
    assert result == effective.model_copy(update={"canonical_sha256": "3" * 64})
    assert result is not original and result is not effective
    assert state["publication"].original_ingestion_receipt == frozen
    assert effective.canonical_sha256 == "2" * 64
    session.add.assert_not_called()
    session.flush.assert_not_called()
    session.commit.assert_not_called()


@pytest.mark.parametrize(
    ("entity", "field", "value"),
    [
        ("batch", "status", "ready"),
        ("batch", "stage", "indexing"),
        ("batch", "review_sha256", None),
        ("member", "status", "indexing"),
        ("member", "approval_attempt_id", None),
        ("file", "status", UserFileStatus.CHUNKED),
        ("file", "file_id", "foreign-original"),
        ("file", "user_id", uuid4()),
        ("file", "regulatory_chunk_generation_hash", "9" * 64),
        ("publication", "gate_closed", True),
        ("publication", "epoch", 0),
        ("publication", "writer_manifest", {"pending": True}),
        ("publication", "original_ingestion_receipt", {}),
        ("asset", "package_id", uuid4()),
        ("asset", "sha256", "9" * 64),
        ("package", "status", "failed"),
        ("package", "issues", ["unresolved"]),
        ("package", "created_by", uuid4()),
        ("package", "document_set_id", -1),
        ("chunk", "source", "amendment"),
        ("chunk", "supersedes_chunk_id", "previous"),
        ("chunk", "superseded_by_chunk_id", "next"),
        ("chunk", "projection_ordinal", 43),
    ],
)
def test_unapproved_or_changed_proof_fails_closed(
    monkeypatch: pytest.MonkeyPatch, entity: str, field: str, value: object
) -> None:
    session, original, state, _resolver = _approved(monkeypatch)
    setattr(state[entity], field, value)
    assert (
        authority.approved_intake_original_receipt(
            session, state["file"].id, original, canonical_sha256="3" * 64
        )
        is None
    )


@pytest.mark.parametrize("field", ["file_id", "raw_sha256", "plaintext_sha256"])
def test_reprepare_must_retain_original_source(
    monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    session, original, state, resolver = _approved(monkeypatch)
    resolver.return_value = original.model_copy(update={field: "changed"})
    assert (
        authority.approved_intake_original_receipt(
            session, state["file"].id, original, canonical_sha256="3" * 64
        )
        is None
    )


@pytest.mark.parametrize(
    "error", [ValueError("invalid receipt"), intake.IntakeConflictError("changed")]
)
def test_invalid_audit_or_review_fails_closed(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    session, original, state, resolver = _approved(monkeypatch)
    resolver.side_effect = error
    assert (
        authority.approved_intake_original_receipt(
            session, state["file"].id, original, canonical_sha256="3" * 64
        )
        is None
    )


def test_unreviewed_canonical_change_and_missing_ordinal_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, original, state, _resolver = _approved(monkeypatch)
    monkeypatch.setattr(intake, "review_digest", lambda *_args: "changed")
    assert (
        authority.approved_intake_original_receipt(
            session, state["file"].id, original, canonical_sha256="3" * 64
        )
        is None
    )
    monkeypatch.setattr(
        intake, "review_digest", lambda *_args: state["batch"].review_sha256
    )
    session.execute.return_value.all.return_value = [(state["chunk"], None)]
    assert (
        authority.approved_intake_original_receipt(
            session, state["file"].id, original, canonical_sha256="3" * 64
        )
        is None
    )
