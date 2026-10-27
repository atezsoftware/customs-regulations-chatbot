"""A formatter transition retains raw originals and an independently proven artifact."""

from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from io import BytesIO
from types import SimpleNamespace
from typing import Any
from uuid import UUID, uuid4
from zipfile import ZipFile

import pytest
from sqlalchemy import select
from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm import Session

from onyx.connectors.models import Document
from onyx.db import document_intake as intake
from onyx.db import document_intake_normalization as normalization
from onyx.db import document_intake_reprepare as repository
from onyx.db.models import (
    DocumentSet,
    ModelConfiguration,
    RegulatoryChunk,
    RegulatoryDocumentIntake,
    RegulatoryDocumentIntakeReprepare,
    RegulatoryFilePublication,
    RegulatoryLabelingRun,
    RegulatorySourceAsset,
    User,
    UserFile,
)
from onyx.file_processing.original_ingestion import (
    LoadedUserFileDocuments,
    OriginalIngestionReceipt,
)
from onyx.file_store.postgres_file_store import PostgresBackedFileStore
from onyx.regulatory import writer_publication
from onyx.regulatory.amendments.annexes import job as source_job
from onyx.regulatory.document_intake import job as intake_job
from onyx.regulatory.document_intake import preparation, reprepare_job
from tests.external_dependency_unit.regulatory.test_document_intake import _ENVIRONMENT
from tests.external_dependency_unit.regulatory.test_document_intake import (
    intake_owner as intake_owner,
)
from tests.external_dependency_unit.regulatory.test_document_intake import (
    intake_session as intake_session,
)
from tests.external_dependency_unit.regulatory.test_document_intake_markdown import (
    native_markdown_runtime as native_markdown_runtime,
)
from tests.external_dependency_unit.regulatory.test_document_intake_reprepare import (
    _request,
)


def _transition_fixture(
    session: Session,
    owner: tuple[User, DocumentSet, ModelConfiguration],
    store: PostgresBackedFileStore,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[RegulatoryDocumentIntake, UUID]:
    from onyx.db.document_intake_sources import create_zip_source_package

    user, _target, model = owner
    contents = {
        "main.html": (
            "<main><h1>Başvuru Tebliği</h1><h2>MADDE 1 — Başvuru usulü</h2>"
            "<p>Başvuru sırasında EK-1 ve EK-2 belgeleri sunulur.</p>"
            "<p><a href='ekler/ek-b.md'>Ekler için tıklayınız: EK-1</a></p>"
            "<p><a href='ekler/a1.md'>EK-2 için tıklayınız</a></p></main>"
        ).encode(),
        "ekler/ek-b.md": "# EK-1\n\nBirinci ekin yasal içeriği.\n".encode(),
        "ekler/a1.md": "# EK-2\n\nİkinci ekin yasal içeriği.\n".encode(),
    }
    output = BytesIO()
    with ZipFile(output, "w") as archive:
        for path, raw in contents.items():
            archive.writestr(path, raw)
    package, _ = create_zip_source_package(
        session,
        user=user,
        environment=_ENVIRONMENT,
        idempotency_key=uuid4(),
        display_name="normalization.zip",
        content=output.getvalue(),
        store=store,
    )
    package_id = package.id
    session.commit()
    source_job.run_source_package(package_id=package_id, environment=_ENVIRONMENT)
    batch, _ = intake.create_intake(
        session,
        source_package_id=package_id,
        idempotency_key=uuid4(),
        labeling_model_configuration_id=model.id,
        user=user,
        environment=_ENVIRONMENT,
    )
    batch_id, model_id = batch.id, model.id
    session.commit()
    monkeypatch.setattr(
        intake,
        "resolve_intake_labeling_provider_binding",
        lambda *_args, **_kwargs: SimpleNamespace(
            model="gemini-3.5-flash",
            model_dump=lambda **_options: {"model_configuration_id": model_id},
        ),
    )
    monkeypatch.setattr(
        "onyx.regulatory.document_intake.metadata_job.run_intake_metadata",
        lambda **_kwargs: None,
    )
    current_formatter = preparation.prepare_intake_text

    def previous_formatter(text: str, **kwargs: Any) -> str:
        return current_formatter(text, **kwargs).replace(
            "[EK-1](ekler/ek-b.md)\n[EK-2](ekler/a1.md)",
            "[EK-1](ekler/ek-b.md), [EK-1](ekler/a1.md)\nEK-2 için tıklayınız",
        )

    with monkeypatch.context() as initial:
        initial.setattr(preparation, "prepare_intake_text", previous_formatter)
        run_id = intake_job.run_document_intake(intake_id=batch_id, tenant_id="public")
    assert run_id is not None
    session.expire_all()
    batch = session.get_one(RegulatoryDocumentIntake, batch_id)
    run = session.get_one(RegulatoryLabelingRun, run_id)
    run.status, run.stage = "failed", "finished"
    batch.status, batch.stage = "partial", "labeling"
    session.commit()
    connection = session.connection()

    @contextmanager
    def shared_session(**_kwargs: object) -> Iterator[Session]:
        with Session(
            connection, join_transaction_mode="create_savepoint", expire_on_commit=False
        ) as shared:
            yield shared

    monkeypatch.setattr(repository, "get_session_with_tenant", shared_session)
    monkeypatch.setattr(
        reprepare_job, "get_session_with_current_tenant", shared_session
    )
    monkeypatch.setattr(reprepare_job, "get_default_file_store", lambda: store)
    monkeypatch.setattr(normalization, "get_default_file_store", lambda: store)
    monkeypatch.setattr(reprepare_job, "run_intake_metadata", lambda **_kwargs: None)
    return batch, run_id


def _prepared_transition(
    session: Session,
    owner: tuple[User, DocumentSet, ModelConfiguration],
    store: PostgresBackedFileStore,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[RegulatoryDocumentIntake, RegulatoryDocumentIntakeReprepare]:
    batch, _run_id = _transition_fixture(session, owner, store, monkeypatch)
    request = _request(session, batch)
    assert (
        reprepare_job.run_document_intake_reprepare(
            intake_id=batch.id, request_id=request.id, tenant_id="public"
        )
        is not None
    )
    session.expire_all()
    return session.get_one(RegulatoryDocumentIntake, batch.id), session.get_one(
        RegulatoryDocumentIntakeReprepare, request.id
    )


def test_changed_navigation_reprepare_seals_artifact_preserves_sources_and_survives_future_formatter(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    native_markdown_runtime: PostgresBackedFileStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = intake_session
    batch, _old_run = _transition_fixture(
        session, intake_owner, native_markdown_runtime, monkeypatch
    )
    assets_before = [
        tuple(row) for row in session.execute(select(RegulatorySourceAsset.__table__))
    ]
    members_before = [
        (
            m.id,
            m.user_file_id,
            m.source_asset_id,
            m.parent_document_id,
            deepcopy(m.relationships),
        )
        for m in intake.intake_files(session, batch)
    ]
    receipts = {
        m.user_file_id: deepcopy(
            session.get_one(
                RegulatoryFilePublication, m.user_file_id
            ).original_ingestion_receipt
        )
        for m in intake.intake_files(session, batch)
    }
    request = _request(session, batch)
    assert (
        reprepare_job.run_document_intake_reprepare(
            intake_id=batch.id, request_id=request.id, tenant_id="public"
        )
        is not None
    )
    session.expire_all()
    request = session.get_one(RegulatoryDocumentIntakeReprepare, request.id)
    transitions = [
        entry
        for entry in request.documents.values()
        if "normalization_derivation" in entry
    ]
    assert len(transitions) == 1
    entry = transitions[0]
    assert (
        entry["source_receipt"]["plaintext_sha256"]
        != entry["after_receipt"]["plaintext_sha256"]
    )
    assert entry["normalization_derivation"]["formatter_version"] == 2
    assert entry["before_revision_ids"] and entry["after_revision_ids"]
    assert [
        tuple(row) for row in session.execute(select(RegulatorySourceAsset.__table__))
    ] == assets_before
    assert [
        (
            m.id,
            m.user_file_id,
            m.source_asset_id,
            m.parent_document_id,
            m.role,
            m.relationships,
        )[:4]
        + (deepcopy(m.relationships),)
        for m in intake.intake_files(session, batch)
    ] == members_before
    assert {
        m.user_file_id: session.get_one(
            RegulatoryFilePublication, m.user_file_id
        ).original_ingestion_receipt
        for m in intake.intake_files(session, batch)
    } == receipts
    main_id = UUID(entry["user_file_id"])
    rows = list(
        session.scalars(
            select(RegulatoryChunk).where(RegulatoryChunk.user_file_id == main_id)
        )
    )
    assert any("[EK-2](ekler/a1.md)" in row.text for row in rows)
    assert not any("tıklayınız" in row.text for row in rows)
    monkeypatch.setattr(
        preparation,
        "load_verified_intake_documents",
        lambda **_: pytest.fail(
            "Historical artifact must not execute a future formatter"
        ),
    )
    monkeypatch.setattr(
        normalization,
        "load_verified_intake_documents",
        lambda **_: pytest.fail(
            "Historical artifact must not execute a future formatter"
        ),
    )
    original = OriginalIngestionReceipt.model_validate(receipts[main_id])
    assert (
        repository.reprepared_original_receipt(session, main_id, original).model_dump(
            mode="json"
        )
        == entry["after_receipt"]
    )
    session.get_one(UserFile, main_id).name = "Kaynakla doğrulanmış yeni görünen başlık"
    session.flush()
    assert (
        repository.reprepared_original_receipt(
            session, main_id, original
        ).plaintext_sha256
        == entry["after_receipt"]["plaintext_sha256"]
    )


@pytest.mark.parametrize(
    "change",
    [
        "artifact",
        "witness",
        "old_receipt",
        "prepared_sha",
        "relationships",
        "source_pointer",
        "raw_blob",
        "prepared_blob",
        "coherent_forgery",
    ],
)
def test_retained_normalization_forgery_or_source_drift_is_rejected(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    native_markdown_runtime: PostgresBackedFileStore,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    session = intake_session
    batch, request = _prepared_transition(
        session, intake_owner, native_markdown_runtime, monkeypatch
    )
    entry = next(
        entry
        for entry in request.documents.values()
        if "normalization_derivation" in entry
    )
    file_id = UUID(entry["user_file_id"])
    original = OriginalIngestionReceipt.model_validate(entry["source_receipt"])
    member = next(
        m for m in intake.intake_files(session, batch) if m.user_file_id == file_id
    )
    altered = deepcopy(request.documents)
    changed = altered[str(member.id)]
    asset = session.get_one(RegulatorySourceAsset, member.source_asset_id)
    if change in {"prepared_sha", "source_pointer"}:
        with pytest.raises(DBAPIError, match="source assets are immutable"):
            with session.begin_nested():
                if change == "prepared_sha":
                    asset.text_sha256 = "0" * 64
                else:
                    asset.text_file_id = "foreign-prepared-source"
                session.flush()
        return
    if change == "artifact":
        changed["normalization_derivation"]["normalized_documents"][0]["sections"][0][
            "text"
        ] += " FORGED"
    elif change == "witness":
        changed["normalization_derivation"]["witness_sha256"] = "0" * 64
    elif change == "old_receipt":
        changed["source_receipt"]["plaintext_sha256"] = "0" * 64
    elif change == "relationships":
        member.relationships = []
    elif change in {"raw_blob", "prepared_blob"}:
        target = asset.file_id if change == "raw_blob" else asset.text_file_id
        read = native_markdown_runtime.read_file
        monkeypatch.setattr(
            native_markdown_runtime,
            "read_file",
            lambda key: BytesIO(b"CORRUPT") if key == target else read(key),
        )
    elif change == "coherent_forgery":
        from onyx.document_index.publication_models import publication_digest

        artifact = changed["normalization_derivation"]
        artifact["normalized_documents"][0]["sections"][0]["text"] += " FORGED"
        forged = LoadedUserFileDocuments(
            [
                Document.model_validate(value)
                for value in artifact["normalized_documents"]
            ],
            file_id=original.file_id,
            raw_sha256=original.raw_sha256,
        )
        for key in ("documents_sha256", "plaintext_sha256"):
            digest = getattr(forged.original_extraction, key)
            artifact[f"normalized_{key}"] = digest
            changed["after_receipt"][key] = digest
        artifact["witness_sha256"] = publication_digest(
            {key: value for key, value in artifact.items() if key != "witness_sha256"}
        )
    request.documents = altered
    session.flush()
    with pytest.raises(ValueError):
        repository.reprepared_original_receipt(session, file_id, original)


def test_caller_forged_reprepare_text_fails_before_canonical_changes(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    native_markdown_runtime: PostgresBackedFileStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = intake_session
    batch, _run = _transition_fixture(
        session, intake_owner, native_markdown_runtime, monkeypatch
    )
    before = [tuple(row) for row in session.execute(select(RegulatoryChunk.__table__))]
    request = _request(session, batch)
    original = writer_publication.reprepare_owned_intake_file

    def forged(
        file_id: UUID, tenant_id: str, documents: list[Document], **kwargs: Any
    ) -> bool:
        assert isinstance(documents, LoadedUserFileDocuments)
        copies = [doc.model_copy(deep=True) for doc in documents]
        copies[0].sections[0].text = (
            copies[0].sections[0].materialize_text() + " FORGED LAW"
        )
        return original(
            file_id,
            tenant_id,
            LoadedUserFileDocuments(
                copies,
                file_id=documents.original_extraction.file_id,
                raw_sha256=documents.original_extraction.raw_sha256,
            ),
            **kwargs,
        )

    monkeypatch.setattr(writer_publication, "reprepare_owned_intake_file", forged)
    with pytest.raises(ValueError, match="verified source formatter"):
        reprepare_job.run_document_intake_reprepare(
            intake_id=batch.id, request_id=request.id, tenant_id="public"
        )
    session.expire_all()
    assert [
        tuple(row) for row in session.execute(select(RegulatoryChunk.__table__))
    ] == before


def test_plaintext_failure_after_committed_reprepare_restores_exact_artifact_without_rechunking(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    native_markdown_runtime: PostgresBackedFileStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from onyx.file_store import utils

    session = intake_session
    batch, _run = _transition_fixture(
        session, intake_owner, native_markdown_runtime, monkeypatch
    )
    request = _request(session, batch)
    real_store = utils.store_user_file_plaintext
    calls = 0

    def failing(file_id: UUID, text: str) -> bool:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError("Injected failure after canonical commit")
        return real_store(file_id, text)

    monkeypatch.setattr(utils, "store_user_file_plaintext", failing)
    with pytest.raises(ValueError, match="after canonical commit"):
        reprepare_job.run_document_intake_reprepare(
            intake_id=batch.id, request_id=request.id, tenant_id="public"
        )
    session.expire_all()
    request = session.get_one(RegulatoryDocumentIntakeReprepare, request.id)
    done = {
        UUID(entry["user_file_id"])
        for entry in request.documents.values()
        if entry["status"] == "prepared"
    }
    assert len(done) == 1
    original = writer_publication.reprepare_owned_intake_file

    def remaining(
        file_id: UUID, tenant_id: str, documents: list[Document], **kwargs: Any
    ) -> bool:
        assert file_id not in done
        return original(file_id, tenant_id, documents, **kwargs)

    monkeypatch.setattr(writer_publication, "reprepare_owned_intake_file", remaining)
    assert (
        reprepare_job.run_document_intake_reprepare(
            intake_id=batch.id, request_id=request.id, tenant_id="public"
        )
        is not None
    )
    session.expire_all()
    assert (
        session.get_one(RegulatoryDocumentIntakeReprepare, request.id).status
        == "prepared"
    )
