"""Storage-only Markdown drafts and exact explicit attachment admission."""

import json
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from onyx.db import document_intake as intake
from onyx.db import markdown_drafts as drafts
from onyx.db.amendment_sources import list_source_assets
from onyx.db.models import (
    DocumentSet,
    ModelConfiguration,
    RegulatoryChunk,
    RegulatoryDocumentIntake,
    RegulatoryLabelingRun,
    RegulatoryMarkdownDraftDocument,
    User,
    UserFile,
)
from onyx.file_store.postgres_file_store import PostgresBackedFileStore
from onyx.regulatory.document_intake import job as intake_job
from shared_configs.contextvars import get_current_tenant_id
from tests.external_dependency_unit.regulatory.test_document_intake import (
    _ENVIRONMENT,
)
from tests.external_dependency_unit.regulatory.test_document_intake import (
    intake_owner as intake_owner,
)
from tests.external_dependency_unit.regulatory.test_document_intake import (
    intake_session as intake_session,
)
from tests.external_dependency_unit.regulatory.test_document_intake_markdown import (
    native_markdown_runtime as native_markdown_runtime,
)
from tests.external_dependency_unit.regulatory.test_labeling_jobs import (
    LabelingDatabase,
)


def test_actual_main_and_numeric_annex_use_lossless_native_chunking(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    native_markdown_runtime: PostgresBackedFileStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, store = intake_session, native_markdown_runtime
    owner, _, model = intake_owner
    fixtures = (
        Path(__file__).parents[2] / "unit/onyx/regulatory/fixtures/md_annex_navigation"
    )
    originals = [
        (name, (fixtures / name).read_bytes())
        for name in ("main.md", "20220618-8-1 (1).md")
    ]
    draft = drafts.create_draft(
        session, user=owner, environment=_ENVIRONMENT, idempotency_key=uuid4()
    )
    drafts.add_documents(
        session,
        draft,
        files=originals[:1],
        expected_revision=0,
        idempotency_key=uuid4(),
        store=store,
    )
    root = drafts.draft_documents(session, draft)[0]
    drafts.add_documents(
        session,
        draft,
        files=originals[1:],
        expected_revision=1,
        idempotency_key=uuid4(),
        parent_document_id=root.id,
        store=store,
    )
    package = drafts.prepare_draft(session, draft, expected_revision=2, store=store)
    batch, _ = intake.create_intake(
        session,
        source_package_id=package.id,
        idempotency_key=uuid4(),
        labeling_model_configuration_id=model.id,
        user=owner,
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
    assert (
        intake_job.run_document_intake(
            intake_id=batch_id, tenant_id=get_current_tenant_id()
        )
        is not None
    )
    session.expire_all()
    batch = session.get_one(RegulatoryDocumentIntake, batch_id)
    members = intake.intake_files(session, batch)
    assert len(members) == 2 and all(
        member.status in {"chunked", "labeling", "ready"} for member in members
    ), (batch.error, [(member.status, member.error) for member in members])
    root_member = next(member for member in members if member.role == "root")
    child = next(member for member in members if member.role == "annex")
    assert child.parent_document_id == root_member.id
    for member in members:
        rows = list(
            session.scalars(
                select(RegulatoryChunk).where(
                    RegulatoryChunk.user_file_id == member.user_file_id
                )
            )
        )
        assert rows
        delivered = "\n".join(row.text for row in rows)
        if member is root_member:
            assert "kapsamaz." in delivered and "MADDE 45" in delivered
            assert (
                "https://www.resmigazete.gov.tr/eskiler/2022/06/20220618-8-1.pdf"
                in delivered
            )
        original = originals[0 if member is root_member else 1][1]
        user_file = session.get_one(UserFile, member.user_file_id)
        with store.read_file(user_file.file_id) as stream:
            assert stream.read() == original


def test_draft_storage_freeze_and_native_preparation(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    native_markdown_runtime: PostgresBackedFileStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, store = intake_session, native_markdown_runtime
    owner, _, model = intake_owner
    before = {
        table: set(session.scalars(select(table.id)))
        for table in (
            UserFile,
            RegulatoryChunk,
            RegulatoryDocumentIntake,
            RegulatoryLabelingRun,
        )
    }
    draft = drafts.create_draft(
        session, user=owner, environment=_ENVIRONMENT, idempotency_key=uuid4()
    )
    root = b"\xef\xbb\xbf# Main\n\nMADDE 1 - STAGED_ROOT authentic rule.\n"
    other = b"# Independent\n\nMADDE 1 - INDEPENDENT_ROOT rule.\n"
    child = b"# Annex\n\nMADDE 1 - STAGED_ANNEX attachment.\n"
    key = uuid4()
    drafts.add_documents(
        session,
        draft,
        files=[("same.md", root), ("same.md", other)],
        expected_revision=0,
        idempotency_key=key,
        store=store,
    )
    docs = drafts.draft_documents(session, draft)
    parent = next(doc for doc in docs if doc.sha256 == sha256(root).hexdigest())
    drafts.add_documents(
        session,
        draft,
        files=[("annex.md", child)],
        parent_document_id=parent.id,
        expected_revision=1,
        idempotency_key=uuid4(),
        store=store,
    )
    # Lost-response replay survives later mutations, while changing a receipt is rejected.
    drafts.add_documents(
        session,
        draft,
        files=[("same.md", root), ("same.md", other)],
        expected_revision=0,
        idempotency_key=key,
        store=store,
    )
    assert draft.revision == 2
    with pytest.raises(intake.IntakeConflictError):
        drafts.add_documents(
            session,
            draft,
            files=[("changed.md", other)],
            expected_revision=0,
            idempotency_key=key,
            store=store,
        )
    for table, ids in before.items():
        assert set(session.scalars(select(table.id))) == ids
    package = drafts.prepare_draft(session, draft, expected_revision=2, store=store)
    assert package.status == "ready" and draft.status == "prepared"
    assert (
        drafts.prepare_draft(session, draft, expected_revision=2, store=store).id
        == package.id
    )
    assert set(session.scalars(select(UserFile.id))) == before[UserFile]
    assets = list_source_assets(session, package.id)
    originals = {sha256(raw).hexdigest(): raw for raw in (root, other, child)}
    for asset in assets:
        with store.read_file(asset.file_id) as stream:
            assert stream.read() == originals[asset.sha256]
    assert package.manifest_file_id is not None
    with store.read_file(package.manifest_file_id) as stream:
        manifest = json.load(stream)
    assert len(manifest["intake_graph"]["root_hashes"]) == 2
    assert manifest["intake_graph"]["primary_parent_by_hash"] == {
        sha256(child).hexdigest(): sha256(root).hexdigest()
    }
    assert (
        manifest["links"][0]["source_field"]
        == f"markdown-draft:{parent.id}:{next(doc.id for doc in drafts.draft_documents(session, draft) if doc.parent_document_id)}"
    )
    assert manifest["links"][0]["original_url"] is None
    batch, _ = intake.create_intake(
        session,
        source_package_id=package.id,
        idempotency_key=uuid4(),
        labeling_model_configuration_id=model.id,
        user=owner,
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
    assert (
        intake_job.run_document_intake(
            intake_id=batch_id, tenant_id=get_current_tenant_id()
        )
        is not None
    )
    session.expire_all()
    members = intake.intake_files(session, batch)
    assert len(members) == 3
    assert sum(member.role == "root" for member in members) == 2
    annex = next(member for member in members if member.role == "annex")
    assert annex.parent_document_id is not None
    assert (
        session.scalar(
            select(RegulatoryChunk).where(
                RegulatoryChunk.user_file_id == annex.user_file_id
            )
        )
        is not None
    )


def test_draft_scope_mutation_rejections_and_subtree_cleanup(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    native_markdown_runtime: PostgresBackedFileStore,
) -> None:
    from onyx.auth.schemas import UserRole
    from onyx.db.models import FileRecord
    from tests.external_dependency_unit.regulatory.test_document_intake import (
        create_test_user,
    )

    session, store = intake_session, native_markdown_runtime
    owner, _, _ = intake_owner
    key = uuid4()
    draft = drafts.create_draft(
        session, user=owner, environment=_ENVIRONMENT, idempotency_key=key
    )
    assert (
        drafts.create_draft(
            session, user=owner, environment=_ENVIRONMENT, idempotency_key=key
        ).id
        == draft.id
    )
    outsider = create_test_user(session, "other-draft-owner", UserRole.ADMIN)
    for user, environment in ((outsider, _ENVIRONMENT), (owner, "another-environment")):
        with pytest.raises(ValueError):
            drafts.get_draft(session, draft.id, user=user, environment=environment)
        assert drafts.list_drafts(session, user=user, environment=environment) == []
    with pytest.raises(ValueError):
        drafts.prepare_draft(session, draft, expected_revision=0, store=store)
    record_ids = set(session.scalars(select(FileRecord.file_id)))
    invalid_batches = [
        [("root.md", b"# Good root"), ("binary.md", b"%PDF-1.7 not markdown")],
        [("utf.md", b"\xff\xfe")],
        [("empty.md", b" ")],
        [("image.md", b"# Root\n![image](x.png)")],
        [("html.md", b'<iframe src="https://example.com"></iframe>')],
        [("danger.md", b"[unsafe](file:///etc/passwd)")],
        [("duplicate.md", b"# Same"), ("same.md", b"# Same")],
        [("a.txt", b"# Wrong extension")],
        [("big.md", b"x" * (2 * 1024 * 1024 + 1))],
        [("../unsafe.md", b"# Traversal")],
        [("first.md", b"# first")] * 22,
    ]
    for files in invalid_batches:
        with pytest.raises(ValueError):
            drafts.add_documents(
                session,
                draft,
                files=files,
                expected_revision=0,
                idempotency_key=uuid4(),
                store=store,
            )
        assert draft.revision == 0 and drafts.draft_documents(session, draft) == []
        assert set(session.scalars(select(FileRecord.file_id))) == record_ids
    with pytest.raises(ValueError):
        drafts.add_documents(
            session,
            draft,
            files=[("annex.md", b"# Annex")],
            expected_revision=0,
            idempotency_key=uuid4(),
            parent_document_id=uuid4(),
            store=store,
        )
    drafts.add_documents(
        session,
        draft,
        files=[("root.md", b"# Main")],
        expected_revision=0,
        idempotency_key=uuid4(),
        store=store,
    )
    root = drafts.draft_documents(session, draft)[0]
    other = drafts.create_draft(
        session, user=owner, environment=_ENVIRONMENT, idempotency_key=uuid4()
    )
    with pytest.raises(ValueError):
        drafts.add_documents(
            session,
            other,
            files=[("child.md", b"# wrong draft")],
            expected_revision=0,
            idempotency_key=uuid4(),
            parent_document_id=root.id,
            store=store,
        )
    with pytest.raises(intake.IntakeConflictError):
        drafts.add_documents(
            session,
            draft,
            files=[("child.md", b"# stale")],
            expected_revision=0,
            idempotency_key=uuid4(),
            store=store,
        )
    with pytest.raises(ValueError):
        drafts.add_documents(
            session,
            draft,
            files=[("root-again.md", b"# Main")],
            expected_revision=1,
            idempotency_key=uuid4(),
            store=store,
        )
    drafts.add_documents(
        session,
        draft,
        files=[("child.md", b"# Child")],
        expected_revision=1,
        idempotency_key=uuid4(),
        parent_document_id=root.id,
        store=store,
    )
    child = next(
        row for row in drafts.draft_documents(session, draft) if row.parent_document_id
    )
    drafts.add_documents(
        session,
        draft,
        files=[("grandchild.md", b"# Grandchild")],
        expected_revision=2,
        idempotency_key=uuid4(),
        parent_document_id=child.id,
        store=store,
    )
    file_ids = {row.file_id for row in drafts.draft_documents(session, draft)}
    drafts.remove_document(session, draft, root.id, expected_revision=3)
    assert draft.revision == 4 and drafts.draft_documents(session, draft) == []
    session.commit()
    drafts.cleanup_removed_documents(session, draft, store=store)
    assert not file_ids.intersection(session.scalars(select(FileRecord.file_id)))
    drafts.cleanup_removed_documents(session, draft, store=store)


def test_prepared_draft_is_immutable_and_original_corruption_rejected(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    native_markdown_runtime: PostgresBackedFileStore,
) -> None:
    from io import BytesIO

    from onyx.configs.constants import FileOrigin

    session, store = intake_session, native_markdown_runtime
    owner, _, _ = intake_owner
    draft = drafts.create_draft(
        session, user=owner, environment=_ENVIRONMENT, idempotency_key=uuid4()
    )
    drafts.add_documents(
        session,
        draft,
        files=[("root.md", b"# Original root")],
        expected_revision=0,
        idempotency_key=uuid4(),
        store=store,
    )
    root = drafts.draft_documents(session, draft)[0]
    store.save_file(
        BytesIO(b"# Corruption"),
        "root.md",
        FileOrigin.OTHER,
        "text/markdown",
        file_id=root.file_id,
    )
    with pytest.raises(ValueError, match="original changed"):
        drafts.prepare_draft(session, draft, expected_revision=1, store=store)
    assert draft.status == "draft" and draft.source_package_id is None
    store.save_file(
        BytesIO(b"# Original root"),
        "root.md",
        FileOrigin.OTHER,
        "text/markdown",
        file_id=root.file_id,
    )
    package = drafts.prepare_draft(session, draft, expected_revision=1, store=store)
    assert (
        drafts.prepare_draft(session, draft, expected_revision=1, store=store).id
        == package.id
    )
    for mutation in (
        lambda: drafts.add_documents(
            session,
            draft,
            files=[("late.md", b"# Late")],
            expected_revision=2,
            idempotency_key=uuid4(),
            store=store,
        ),
        lambda: drafts.remove_document(session, draft, root.id, expected_revision=2),
    ):
        with pytest.raises(intake.IntakeConflictError):
            mutation()


def test_partial_storage_failure_rolls_back_every_postgres_blob(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    native_markdown_runtime: PostgresBackedFileStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from onyx.db.models import FileRecord

    session, store = intake_session, native_markdown_runtime
    owner, _, _ = intake_owner
    draft = drafts.create_draft(
        session, user=owner, environment=_ENVIRONMENT, idempotency_key=uuid4()
    )
    session.commit()
    draft_id = draft.id
    before = set(session.scalars(select(FileRecord.file_id)))
    original_save = store.save_file
    calls = 0

    def fail_second(*args: Any, **kwargs: Any) -> str:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("controlled storage failure")
        return original_save(*args, **kwargs)

    monkeypatch.setattr(store, "save_file", fail_second)
    with pytest.raises(RuntimeError, match="controlled storage failure"):
        drafts.add_documents(
            session,
            draft,
            files=[("first.md", b"# first"), ("second.md", b"# second")],
            expected_revision=0,
            idempotency_key=uuid4(),
            store=store,
        )
    session.rollback()
    draft = drafts.get_draft(session, draft_id, user=owner, environment=_ENVIRONMENT)
    assert draft.revision == 0 and drafts.draft_documents(session, draft) == []
    assert set(session.scalars(select(FileRecord.file_id))) == before


def test_frozen_upload_identity_and_explicit_parent_provenance_cannot_change(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    native_markdown_runtime: PostgresBackedFileStore,
) -> None:
    from onyx.db.document_intake_graph import require_frozen_zip_intake_graph
    from onyx.regulatory.amendments.annexes.models import SourceLink
    from onyx.regulatory.document_intake.source_graph import (
        MARKDOWN_DRAFT_INPUT_KIND,
        parse_intake_graph,
    )

    session, store = intake_session, native_markdown_runtime
    owner, _, _ = intake_owner
    draft = drafts.create_draft(
        session, user=owner, environment=_ENVIRONMENT, idempotency_key=uuid4()
    )
    drafts.add_documents(
        session,
        draft,
        files=[("root.md", b"# Root")],
        expected_revision=0,
        idempotency_key=uuid4(),
        store=store,
    )
    root = drafts.draft_documents(session, draft)[0]
    drafts.add_documents(
        session,
        draft,
        files=[("child.md", b"# Child")],
        expected_revision=1,
        idempotency_key=uuid4(),
        parent_document_id=root.id,
        store=store,
    )
    package = drafts.prepare_draft(session, draft, expected_revision=2, store=store)
    assets = list_source_assets(session, package.id)
    assert package.manifest_file_id is not None
    with store.read_file(package.manifest_file_id) as stream:
        manifest_bytes = stream.read()
    manifest = json.loads(manifest_bytes)
    links = [SourceLink.model_validate(value) for value in manifest["links"]]
    graph = parse_intake_graph(
        manifest,
        input_kind=MARKDOWN_DRAFT_INPUT_KIND,
        asset_hashes=[row.sha256 for row in assets],
        links=links,
    )
    require_frozen_zip_intake_graph(
        session,
        package,
        assets,
        links,
        intake_graph=graph,
        manifest_bytes=manifest_bytes,
    )
    fabricated = [
        link.model_copy(update={"original_url": "child.md"}) for link in links
    ]
    with pytest.raises(ValueError, match="authenticated upload provenance"):
        parse_intake_graph(
            manifest,
            input_kind=MARKDOWN_DRAFT_INPUT_KIND,
            asset_hashes=[row.sha256 for row in assets],
            links=fabricated,
        )
    original_file_id = assets[0].file_id
    assets[0].file_id = "changed-source-original"
    with pytest.raises(ValueError, match="storage identity changed"):
        require_frozen_zip_intake_graph(
            session,
            package,
            assets,
            links,
            intake_graph=graph,
            manifest_bytes=manifest_bytes,
        )
    assets[0].file_id = original_file_id
    with pytest.raises(ValueError, match="frozen source manifest"):
        require_frozen_zip_intake_graph(
            session,
            package,
            assets,
            links,
            intake_graph=graph,
            manifest_bytes=manifest_bytes + b" ",
        )


@pytest.mark.parametrize(
    "mode", ["same-upload", "different-upload", "upload-vs-prepare"]
)
def test_concurrent_mutations_use_one_locked_revision(
    intake_database: LabelingDatabase,
    mode: str,
) -> None:
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    from onyx.auth.schemas import UserRole
    from onyx.db.models import RegulatoryMarkdownDraftUpload
    from tests.external_dependency_unit.regulatory.test_document_intake import (
        create_test_user,
    )

    engine = intake_database.engine
    store = PostgresBackedFileStore()
    with Session(engine) as setup:
        owner = create_test_user(setup, f"race-{uuid4()}", UserRole.ADMIN)
        target = setup.scalar(
            select(DocumentSet).where(DocumentSet.name == "PC Külliyatı")
        )
        if target is None:
            setup.add(
                DocumentSet(
                    name="PC Külliyatı",
                    user_id=owner.id,
                    is_public=True,
                    is_up_to_date=True,
                )
            )
            setup.flush()
        draft = drafts.create_draft(
            setup, user=owner, environment=_ENVIRONMENT, idempotency_key=uuid4()
        )
        if mode == "upload-vs-prepare":
            drafts.add_documents(
                setup,
                draft,
                files=[("initial.md", b"# Initial")],
                expected_revision=0,
                idempotency_key=uuid4(),
                store=store,
            )
        setup.commit()
        owner_id, draft_id = owner.id, draft.id
    barrier = Barrier(2, timeout=15)
    key = uuid4()

    def mutate(index: int) -> str:
        with Session(engine) as session:
            user = session.get_one(User, owner_id)
            current = drafts.get_draft(
                session, draft_id, user=user, environment=_ENVIRONMENT
            )
            barrier.wait()
            try:
                if mode == "upload-vs-prepare" and index == 1:
                    drafts.prepare_draft(
                        session, current, expected_revision=1, store=store
                    )
                else:
                    raw = (
                        b"# Different second"
                        if mode == "different-upload" and index == 1
                        else b"# Same upload"
                    )
                    drafts.add_documents(
                        session,
                        current,
                        files=[("race.md", raw)],
                        expected_revision=1 if mode == "upload-vs-prepare" else 0,
                        idempotency_key=key,
                        store=store,
                    )
                session.commit()
                return "success"
            except intake.IntakeConflictError:
                session.rollback()
                return "conflict"

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(mutate, index) for index in range(2)]
        results = [future.result(timeout=30) for future in futures]
    assert results.count("success") == (2 if mode == "same-upload" else 1)
    assert results.count("conflict") == (0 if mode == "same-upload" else 1)
    with Session(engine) as session:
        current = drafts.get_draft(
            session,
            draft_id,
            user=session.get_one(User, owner_id),
            environment=_ENVIRONMENT,
        )
        assert current.revision == (2 if mode == "upload-vs-prepare" else 1)
        documents = drafts.draft_documents(session, current)
        assert len(documents) == (
            1
            if current.status == "prepared"
            else 2
            if mode == "upload-vs-prepare"
            else 1
        )
        receipts = list(
            session.scalars(
                select(RegulatoryMarkdownDraftUpload).where(
                    RegulatoryMarkdownDraftUpload.draft_id == draft_id
                )
            )
        )
        assert len(receipts) == len(documents)


def test_http_draft_boundary_never_starts_chunking_or_indexing(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    native_markdown_runtime: PostgresBackedFileStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from onyx.db.engine.sql_engine import get_session
    from onyx.error_handling.exceptions import register_onyx_exception_handlers
    from onyx.server.features.regulatory import document_intake_api as api

    session, store = intake_session, native_markdown_runtime
    owner, _, _ = intake_owner
    app = FastAPI()
    register_onyx_exception_handlers(app)
    app.include_router(api.router, prefix="/api/regulatory")
    app.dependency_overrides[api._admin] = lambda: owner
    app.dependency_overrides[get_session] = lambda: session
    monkeypatch.setattr(api.config, "REGULATORY_ANNEX_UPDATES_ENABLED", True)
    monkeypatch.setattr(api.config, "REGULATORY_ANNEX_ENVIRONMENT", _ENVIRONMENT)
    monkeypatch.setattr(drafts, "get_default_file_store", lambda: store)
    before = {
        table: set(session.scalars(select(table.id)))
        for table in (
            UserFile,
            RegulatoryChunk,
            RegulatoryDocumentIntake,
            RegulatoryLabelingRun,
        )
    }
    base = "/api/regulatory/document-intake/markdown-drafts"
    with TestClient(app) as client:
        key = str(uuid4())
        created = client.post(base, json={"idempotency_key": key})
        assert created.status_code == 200, created.text
        snapshot = created.json()
        draft_url = f"{base}/{snapshot['id']}"
        assert (
            client.post(base, json={"idempotency_key": key}).json()["id"]
            == snapshot["id"]
        )
        assert client.get(base).json()[0]["id"] == snapshot["id"]
        assert client.get(draft_url).json()["documents"] == []
        root_response = client.post(
            f"{draft_url}/documents",
            files=[("files", ("main.md", b"# Main", "text/markdown"))],
            data={"idempotency_key": str(uuid4()), "expected_revision": "0"},
        )
        assert root_response.status_code == 200, root_response.text
        root_id = root_response.json()["documents"][0]["id"]
        response = client.post(
            f"{draft_url}/documents",
            files=[("files", ("annex.md", b"# Annex", "text/markdown"))],
            data={
                "idempotency_key": str(uuid4()),
                "expected_revision": "1",
                "parent_document_id": root_id,
            },
        )
        assert response.status_code == 200, response.text
        assert response.json()["revision"] == 2
        stale = client.delete(f"{draft_url}/documents/{root_id}?expected_revision=1")
        assert stale.status_code == 409
        invalid = client.post(
            f"{draft_url}/documents",
            files=[("files", ("bad.md", b"\xff", "text/markdown"))],
            data={"idempotency_key": str(uuid4()), "expected_revision": "2"},
        )
        assert invalid.status_code == 400
        assert "UTF-8" in invalid.json()["detail"]
        large = client.post(
            f"{draft_url}/documents",
            files=[
                ("files", ("big.md", b"x" * (2 * 1024 * 1024 + 1), "text/markdown"))
            ],
            data={"idempotency_key": str(uuid4()), "expected_revision": "2"},
        )
        assert large.status_code == 413
        prepared = client.post(f"{draft_url}/prepare", json={"expected_revision": 2})
        assert prepared.status_code == 200, prepared.text
        assert prepared.json()["status"] == "ready"
        assert (
            client.post(f"{draft_url}/prepare", json={"expected_revision": 2}).json()[
                "id"
            ]
            == prepared.json()["id"]
        )
        assert client.get(draft_url).json()["status"] == "prepared"
        assert (
            client.delete(
                f"{draft_url}/documents/{root_id}?expected_revision=3"
            ).status_code
            == 409
        )
        assert client.get(f"{base}/{uuid4()}").status_code == 404
    for table, ids in before.items():
        assert set(session.scalars(select(table.id))) == ids


def test_snapshot_revision_and_document_tree_share_one_locked_state(
    intake_database: LabelingDatabase, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sqlalchemy.exc import OperationalError

    from onyx.auth.schemas import UserRole
    from onyx.db.models import RegulatoryMarkdownDraft
    from tests.external_dependency_unit.regulatory.test_document_intake import (
        create_test_user,
    )

    engine = intake_database.engine
    with Session(engine) as setup:
        owner = create_test_user(setup, f"snapshot-{uuid4()}", UserRole.ADMIN)
        if (
            setup.scalar(select(DocumentSet).where(DocumentSet.name == "PC Külliyatı"))
            is None
        ):
            setup.add(
                DocumentSet(
                    name="PC Külliyatı",
                    user_id=owner.id,
                    is_public=True,
                    is_up_to_date=True,
                )
            )
            setup.flush()
        draft = drafts.create_draft(
            setup, user=owner, environment=_ENVIRONMENT, idempotency_key=uuid4()
        )
        setup.commit()
        owner_id, draft_id = owner.id, draft.id
    original_documents = drafts.draft_documents
    locked = False

    def concurrent_writer(
        session: Session, draft: RegulatoryMarkdownDraft
    ) -> list[RegulatoryMarkdownDraftDocument]:
        nonlocal locked
        with Session(engine) as writer:
            try:
                current = writer.scalars(
                    select(RegulatoryMarkdownDraft)
                    .where(RegulatoryMarkdownDraft.id == draft_id)
                    .with_for_update(nowait=True)
                ).one()
            except OperationalError as error:
                assert getattr(error.orig, "pgcode", None) == "55P03"
                locked = True
                writer.rollback()
            else:
                # Without a snapshot lock, an upload can commit between revision and tree reads.
                monkeypatch.setattr(drafts, "draft_documents", original_documents)
                drafts.add_documents(
                    writer,
                    current,
                    files=[("concurrent.md", b"# Concurrent")],
                    expected_revision=0,
                    idempotency_key=uuid4(),
                    store=PostgresBackedFileStore(),
                )
                writer.commit()
        return original_documents(session, draft)

    monkeypatch.setattr(drafts, "draft_documents", concurrent_writer)
    with Session(engine) as reader:
        current = drafts.get_draft(
            reader,
            draft_id,
            user=reader.get_one(User, owner_id),
            environment=_ENVIRONMENT,
        )
        snapshot = drafts.draft_snapshot(reader, current)
        assert snapshot.revision == 0 and snapshot.documents == []
        assert locked


def test_removed_original_cleanup_is_retried_on_draft_resume(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    native_markdown_runtime: PostgresBackedFileStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from onyx.db.engine.sql_engine import get_session
    from onyx.db.models import FileRecord
    from onyx.error_handling.exceptions import register_onyx_exception_handlers
    from onyx.server.features.regulatory import document_intake_api as api

    session, store = intake_session, native_markdown_runtime
    owner, _, _ = intake_owner
    draft = drafts.create_draft(
        session, user=owner, environment=_ENVIRONMENT, idempotency_key=uuid4()
    )
    drafts.add_documents(
        session,
        draft,
        files=[("remove.md", b"# Remove")],
        expected_revision=0,
        idempotency_key=uuid4(),
        store=store,
    )
    document = drafts.draft_documents(session, draft)[0]
    draft_id, document_id, file_id = draft.id, document.id, document.file_id
    session.commit()
    original_delete = store.delete_file
    attempts = 0

    def fail_first(
        file_id: str, error_on_missing: bool = True, db_session: Session | None = None
    ) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("temporary deletion outage")
        original_delete(
            file_id, error_on_missing=error_on_missing, db_session=db_session
        )

    monkeypatch.setattr(store, "delete_file", fail_first)
    monkeypatch.setattr(drafts, "get_default_file_store", lambda: store)
    monkeypatch.setattr(api.config, "REGULATORY_ANNEX_UPDATES_ENABLED", True)
    monkeypatch.setattr(api.config, "REGULATORY_ANNEX_ENVIRONMENT", _ENVIRONMENT)
    app = FastAPI()
    register_onyx_exception_handlers(app)
    app.include_router(api.router, prefix="/api/regulatory")
    app.dependency_overrides[api._admin] = lambda: owner
    app.dependency_overrides[get_session] = lambda: session
    base = f"/api/regulatory/document-intake/markdown-drafts/{draft_id}"
    with TestClient(app) as client:
        response = client.delete(f"{base}/documents/{document_id}?expected_revision=1")
        assert response.status_code == 200, response.text
        assert response.json()["documents"] == []
        assert file_id in set(session.scalars(select(FileRecord.file_id)))
        assert client.get(base).status_code == 200
        assert file_id not in set(session.scalars(select(FileRecord.file_id)))
    assert attempts == 2


def test_external_blob_rollback_deletes_new_objects_and_committed_metadata(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from unittest.mock import MagicMock

    from onyx.db.models import FileRecord
    from onyx.file_store.file_store import S3BackedFileStore

    session = intake_session
    owner, _, _ = intake_owner
    draft = drafts.create_draft(
        session, user=owner, environment=_ENVIRONMENT, idempotency_key=uuid4()
    )
    session.commit()
    draft_id = draft.id
    before = set(session.scalars(select(FileRecord.file_id)))
    store = S3BackedFileStore(bucket_name="draft-test-no-network")
    client = MagicMock()
    client.put_object.side_effect = [{}, RuntimeError("controlled second put failure")]
    monkeypatch.setattr(store, "_get_s3_client", lambda: client)
    with pytest.raises(RuntimeError, match="controlled second put failure"):
        drafts.add_documents(
            session,
            draft,
            files=[("first.md", b"# first"), ("second.md", b"# second")],
            expected_revision=0,
            idempotency_key=uuid4(),
            store=store,
        )
    # The first external object's metadata committed independently of the draft.
    assert len(set(session.scalars(select(FileRecord.file_id))) - before) == 1
    session.rollback()
    client.delete_object.assert_called_once()
    assert set(session.scalars(select(FileRecord.file_id))) == before
    current = drafts.get_draft(session, draft_id, user=owner, environment=_ENVIRONMENT)
    assert current.revision == 0 and drafts.draft_documents(session, current) == []


@pytest.mark.parametrize(
    ("literal_url", "annex_name"),
    [("ekler/ek-1.md", "ek-1.md"), ("ekler/ek%201.md#madde-1", "ek 1.md")],
)
def test_staged_attachment_body_paths_resolve_unique_family_basename_without_rewriting(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    native_markdown_runtime: PostgresBackedFileStore,
    literal_url: str,
    annex_name: str,
) -> None:
    session, store = intake_session, native_markdown_runtime
    owner, _, _ = intake_owner
    draft = drafts.create_draft(
        session, user=owner, environment=_ENVIRONMENT, idempotency_key=uuid4()
    )
    root_bytes = b"\xef\xbb\xbf" + f"# Ana\n\n[Ek]({literal_url})\n".encode()
    annex_bytes = b"\xef\xbb\xbf# Annex\n\nMADDE 1 - Exact original text."
    drafts.add_documents(
        session,
        draft,
        files=[("ana.md", root_bytes)],
        expected_revision=0,
        idempotency_key=uuid4(),
        store=store,
    )
    root = drafts.draft_documents(session, draft)[0]
    drafts.add_documents(
        session,
        draft,
        files=[(annex_name, annex_bytes)],
        expected_revision=1,
        idempotency_key=uuid4(),
        parent_document_id=root.id,
        store=store,
    )
    package = drafts.prepare_draft(session, draft, expected_revision=2, store=store)
    assert package.manifest_file_id is not None
    with store.read_file(package.manifest_file_id) as stream:
        manifest = json.load(stream)
    body_link = next(
        link
        for link in manifest["links"]
        if link["source_field"].startswith("markdown:")
    )
    assert body_link["original_url"] == literal_url
    assert body_link["target_asset_hash"] == sha256(annex_bytes).hexdigest()
    assert body_link["parent_asset_hash"] == sha256(root_bytes).hexdigest()
    for asset in list_source_assets(session, package.id):
        with store.read_file(asset.file_id) as stream:
            raw = stream.read()
        assert raw == (
            root_bytes
            if asset.sha256 == sha256(root_bytes).hexdigest()
            else annex_bytes
        )
    assert draft.status == "prepared"


@pytest.mark.parametrize(
    ("literal_url", "annex_names"),
    [
        ("ekler/missing.md", ["known.md"]),
        ("ekler/ek.md", ["ek.md", "ek.md"]),
        ("../ek.md", ["ek.md"]),
        ("%2e%2e/ek.md", ["ek.md"]),
        ("safe/%2e%2e/%2e%2e/ek.md", ["ek.md"]),
    ],
)
def test_staged_basename_resolution_rejects_missing_ambiguous_and_traversing_targets(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    native_markdown_runtime: PostgresBackedFileStore,
    literal_url: str,
    annex_names: list[str],
) -> None:
    session, store = intake_session, native_markdown_runtime
    owner, _, _ = intake_owner
    draft = drafts.create_draft(
        session, user=owner, environment=_ENVIRONMENT, idempotency_key=uuid4()
    )
    drafts.add_documents(
        session,
        draft,
        files=[("ana.md", f"# Ana\n\n[Ek]({literal_url})".encode())],
        expected_revision=0,
        idempotency_key=uuid4(),
        store=store,
    )
    root = drafts.draft_documents(session, draft)[0]
    drafts.add_documents(
        session,
        draft,
        files=[
            (name, f"# Annex {index}".encode())
            for index, name in enumerate(annex_names)
        ],
        expected_revision=1,
        idempotency_key=uuid4(),
        parent_document_id=root.id,
        store=store,
    )
    with pytest.raises(ValueError):
        drafts.prepare_draft(session, draft, expected_revision=2, store=store)
    assert draft.status == "draft" and draft.source_package_id is None


def test_staged_explicit_attachment_survives_metadata_labeling_review_and_approval(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    native_markdown_runtime: PostgresBackedFileStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from onyx.db.enums import UserFileStatus
    from onyx.db.models import (
        RegulatoryFilePublication,
        RegulatoryTemporalProjection,
        UserFileProjectionRepair,
    )
    from onyx.regulatory.document_intake.metadata import METADATA_AUDIT_KEY
    from onyx.regulatory.labeling import orchestrator
    from tests.external_dependency_unit.regulatory.test_document_intake_zip import (
        _inference_runtime,
    )

    session, store = intake_session, native_markdown_runtime
    owner, _, model = intake_owner
    sdk = _inference_runtime(session, model, store, monkeypatch)
    originals = {
        "ana.md": b"\xef\xbb\xbf# Ana Mevzuat\n\nMADDE 1 - STAGED_PARENT_OPERATIVE beyan edilir.\n",
        "bagimsiz.md": b"# Independent\n\nMADDE 1 - STAGED_OTHER_OPERATIVE otuz gun uygulanir.\n",
        "ek.md": b"\xef\xbb\xbf# EK-1\n\nMADDE 1 - STAGED_ANNEX_OPERATIVE fatura ibraz edilir.\n",
    }
    draft = drafts.create_draft(
        session, user=owner, environment=_ENVIRONMENT, idempotency_key=uuid4()
    )
    drafts.add_documents(
        session,
        draft,
        files=[(name, originals[name]) for name in ("ana.md", "bagimsiz.md")],
        expected_revision=0,
        idempotency_key=uuid4(),
        store=store,
    )
    selected_parent = next(
        row
        for row in drafts.draft_documents(session, draft)
        if row.display_name == "ana.md"
    )
    drafts.add_documents(
        session,
        draft,
        files=[("ek.md", originals["ek.md"])],
        expected_revision=1,
        idempotency_key=uuid4(),
        parent_document_id=selected_parent.id,
        store=store,
    )
    package = drafts.prepare_draft(session, draft, expected_revision=2, store=store)
    assert package.manifest_file_id is not None
    with store.read_file(package.manifest_file_id) as stream:
        source_manifest = json.load(stream)
    assert len(source_manifest["links"]) == 1
    source_relationship = source_manifest["links"][0]
    assert source_relationship["original_url"] is None
    assert source_relationship["source_field"].startswith(
        f"markdown-draft:{selected_parent.id}:"
    )
    batch, _ = intake.create_intake(
        session,
        source_package_id=package.id,
        idempotency_key=uuid4(),
        labeling_model_configuration_id=model.id,
        user=owner,
        environment=_ENVIRONMENT,
    )
    batch_id = batch.id
    session.commit()
    assert (
        intake_job.run_document_intake(
            intake_id=batch_id, tenant_id=get_current_tenant_id()
        )
        is not None
    )
    session.expire_all()
    batch = session.get_one(RegulatoryDocumentIntake, batch_id)
    assert batch.labeling_run_id is not None
    run_id = batch.labeling_run_id
    for _step in range(20):
        session.expire_all()
        run = session.get_one(RegulatoryLabelingRun, run_id)
        if run.status not in {"queued", "running"}:
            break
        orchestrator.run_labeling_step(
            run.id, run.lease_generation, get_current_tenant_id()
        )
    session.expire_all()
    run = session.get_one(RegulatoryLabelingRun, run_id)
    assert run.status == "completed", run.error
    intake.refresh_intake(session, batch)
    assert batch.status == "ready" and batch.stage == "review" and batch.review_sha256
    members = intake.intake_files(session, batch)
    assets = {
        asset.id: asset
        for asset in list_source_assets(session, batch.source_package_id)
    }
    by_name = {
        assets[member.source_asset_id].display_name: member for member in members
    }
    assert {name for name, member in by_name.items() if member.role == "root"} == {
        "ana.md",
        "bagimsiz.md",
    }
    parent, annex = by_name["ana.md"], by_name["ek.md"]
    assert annex.parent_document_id == parent.id and annex.relationships == [
        source_relationship
    ]
    new_file_ids = {member.user_file_id for member in members}
    assert set(run.file_ids) == {str(file_id) for file_id in new_file_ids}
    chunks = list(
        session.scalars(
            select(RegulatoryChunk).where(
                RegulatoryChunk.user_file_id.in_(new_file_ids)
            )
        )
    )
    assert chunks and all(
        METADATA_AUDIT_KEY in chunk.chunk_metadata for chunk in chunks
    )
    assert all(
        "STAGED_ANNEX_OPERATIVE" not in chunk.text
        for chunk in chunks
        if chunk.user_file_id == parent.user_file_id
    )
    assert any(
        "STAGED_ANNEX_OPERATIVE" in chunk.text
        for chunk in chunks
        if chunk.user_file_id == annex.user_file_id
    )
    requests = [
        json.loads(call.kwargs["contents"][0].parts[0].text)
        for call in sdk.models.generate_content.call_args_list
    ]
    assert any("target_chunk_ids" in request for request in requests)
    assert any("target" in request for request in requests)
    assert not list(
        session.scalars(
            select(UserFileProjectionRepair).where(
                UserFileProjectionRepair.user_file_id.in_(new_file_ids)
            )
        )
    )
    assert not list(
        session.scalars(
            select(RegulatoryTemporalProjection).where(
                RegulatoryTemporalProjection.user_file_id.in_(new_file_ids)
            )
        )
    )
    for member in members:
        file = session.get_one(UserFile, member.user_file_id)
        asset = assets[member.source_asset_id]
        assert file.status == UserFileStatus.CHUNKED and file.file_id != asset.file_id
        for original_file_id in (file.file_id, asset.file_id):
            with store.read_file(original_file_id) as stream:
                assert stream.read() == originals[asset.display_name]
        publication = session.get_one(RegulatoryFilePublication, member.user_file_id)
        assert (
            publication.epoch == 0
            and publication.writer_manifest_sha256 is None
            and not publication.gate_closed
        )
    review = intake.intake_snapshot(session, batch)
    annex_review = next(
        document for document in review["documents"] if document["id"] == str(annex.id)
    )
    assert annex_review["parent_document_id"] == str(parent.id)
    reviewed_hash = batch.review_sha256
    deliveries = intake.approve_intake(session, batch, reviewed_hash)
    session.commit()
    session.expire_all()
    batch = session.get_one(RegulatoryDocumentIntake, batch_id)
    assert batch.status == "approving" and batch.stage == "indexing"
    assert {file_id for file_id, _attempt in deliveries} == new_file_ids
    assert (
        len(
            list(
                session.scalars(
                    select(UserFileProjectionRepair).where(
                        UserFileProjectionRepair.user_file_id.in_(new_file_ids)
                    )
                )
            )
        )
        == 3
    )
    approved_annex = next(
        member
        for member in intake.intake_files(session, batch)
        if member.id == annex.id
    )
    assert (
        approved_annex.parent_document_id == parent.id
        and approved_annex.relationships == [source_relationship]
    )
    assert intake.approve_intake(session, batch, reviewed_hash) == []
