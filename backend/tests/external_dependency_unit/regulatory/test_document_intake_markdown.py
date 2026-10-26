"""Native Markdown source preparation and separate canonical admission in PostgreSQL."""

import json
from collections.abc import Generator
from contextlib import contextmanager
from hashlib import sha256
from io import BytesIO
from types import SimpleNamespace
from uuid import uuid4
from zipfile import ZipFile

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from onyx.db import document_intake as repository
from onyx.db import regulatory_publication, regulatory_writer_publication
from onyx.db.amendment_sources import list_source_assets
from onyx.db.document_intake_sources import create_markdown_source_package
from onyx.db.models import (
    AmendmentBatch,
    AmendmentSourcePackage,
    DocumentSet,
    ModelConfiguration,
    RegulatoryChunk,
    RegulatoryLabelingItem,
    RegulatoryLabelingRun,
    RegulatorySourceAsset,
    User,
    UserFile,
)
from onyx.file_processing import extract_file_text, user_file_loader
from onyx.file_store import postgres_file_store
from onyx.file_store.postgres_file_store import PostgresBackedFileStore
from onyx.indexing.embedder import DefaultIndexingEmbedder
from onyx.regulatory.amendments import analysis_llm, pdf_vision
from onyx.regulatory.amendments.annexes import job as source_job
from onyx.regulatory.amendments.annexes import source_images
from onyx.regulatory.document_intake import job as intake_job
from onyx.regulatory.document_intake.markdown_package import MarkdownPackageError
from tests.external_dependency_unit.regulatory.test_document_intake import (
    _ENVIRONMENT,
)
from tests.external_dependency_unit.regulatory.test_document_intake import (
    intake_owner as intake_owner,
)
from tests.external_dependency_unit.regulatory.test_document_intake import (
    intake_session as intake_session,
)
from tests.external_dependency_unit.regulatory.test_document_intake_job import (
    _CharacterTokenizer,
)


def _markdown_input(
    packaged: bool, markdown_headings: bool = False
) -> tuple[str, bytes, dict[str, bytes]]:
    root = "# Ana belge\n\nMADDE 1 — ROOT_NATIVE_MARKDOWN elektronik beyan kuralıdır.\n"
    if markdown_headings:
        root = (
            "# Ana belge\n\n## MADDE 1 — Başvuru usulü\n\n"
            "ROOT_NATIVE_MARKDOWN elektronik beyan kuralıdır.\n\n"
            "## MADDE 2 — Belge düzeni\n\n"
            "ROOT_OPERATIVE_TWO ticari fatura ibraz edilir.\n\n"
            "## MADDE 3 — Yürürlük\n\n"
            "ROOT_OPERATIVE_THREE bu kanun 01/01/2027 tarihinde yürürlüğe girer.\n"
        )
    if not packaged:
        raw = root.encode()
        return "ana.md", raw, {sha256(raw).hexdigest(): raw}
    root += (
        "\n[Ekler için tıklayınız](ekler/ek-1.md)\n"
        "\n[EK-2 belge listesi](ekler/ek-2.md)\n"
        "\n[Resmî atıf](https://example.gov/reference.html)\n"
    )
    documents = {
        "ana.md": root.encode(),
        "ekler/ek-1.md": (
            "# EK-1\n\n1. ANNEX_ONE_NATIVE ticari fatura sunulur.\n"
        ).encode(),
        "ekler/ek-2.md": (
            "# EK-2\n\n| Kod | Belge |\n| --- | --- |\n| 0101 | ANNEX_TWO_NATIVE menşe belgesi |\n"
        ).encode(),
    }
    manifest = {
        "format": "legislation-markdown-v1",
        "main": "ana.md",
        "documents": [
            {"path": "ana.md", "title": "Ana belge"},
            {"path": "ekler/ek-1.md", "title": "EK-1", "parent": "ana.md"},
            {"path": "ekler/ek-2.md", "title": "EK-2", "parent": "ana.md"},
        ],
    }
    archive = BytesIO()
    with ZipFile(archive, "w") as zipped:
        zipped.writestr("manifest.json", json.dumps(manifest))
        for path, content in documents.items():
            zipped.writestr(path, content)
    return (
        "mevzuat.zip",
        archive.getvalue(),
        {sha256(content).hexdigest(): content for content in documents.values()},
    )


@pytest.fixture
def native_markdown_runtime(
    intake_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> PostgresBackedFileStore:
    connection = intake_session.connection()

    @contextmanager
    def shared_session(**_kwargs: object) -> Generator[Session, None, None]:
        with Session(
            connection,
            join_transaction_mode="create_savepoint",
            expire_on_commit=False,
        ) as session:
            yield session

    @contextmanager
    def optional_session(
        db_session: Session | None = None,
    ) -> Generator[Session, None, None]:
        if db_session is not None:
            yield db_session
        else:
            with shared_session() as session:
                yield session

    for module, method in (
        (source_job, "get_session_with_current_tenant"),
        (intake_job, "get_session_with_current_tenant"),
        (regulatory_publication, "get_session_with_tenant"),
        (regulatory_writer_publication, "get_session_with_tenant"),
        (postgres_file_store, "get_session_with_current_tenant"),
    ):
        monkeypatch.setattr(module, method, shared_session)
    monkeypatch.setattr(
        postgres_file_store, "get_session_with_current_tenant_if_none", optional_session
    )
    store = PostgresBackedFileStore()
    monkeypatch.setattr(source_job, "get_default_file_store", lambda: store)
    monkeypatch.setattr(intake_job, "get_default_file_store", lambda: store)
    monkeypatch.setattr(
        DefaultIndexingEmbedder,
        "from_db_search_settings",
        lambda **_kwargs: SimpleNamespace(
            embedding_model=SimpleNamespace(tokenizer=_CharacterTokenizer())
        ),
    )

    def forbidden_conversion(*_args: object, **_kwargs: object) -> None:
        pytest.fail("Supplied Markdown must not enter extraction, conversion or vision")

    for module, method in (
        (source_job, "acquire_source_package"),
        (source_job, "download_source"),
        (analysis_llm, "get_amendment_analysis_llm"),
        (pdf_vision, "prepare_pdf_source"),
        (source_images, "prepare_image_source"),
        (extract_file_text, "get_markitdown_converter"),
        (user_file_loader, "load_user_file_documents"),
    ):
        monkeypatch.setattr(module, method, forbidden_conversion)
    return store


@pytest.mark.parametrize("packaged", [False, True], ids=["single-md", "three-md-zip"])
@pytest.mark.parametrize(
    "markdown_headings",
    [False, True],
    ids=["inline-articles", "markdown-article-headings"],
)
def test_native_markdown_source_and_independent_intakes_skip_conversion(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    native_markdown_runtime: PostgresBackedFileStore,
    monkeypatch: pytest.MonkeyPatch,
    packaged: bool,
    markdown_headings: bool,
) -> None:
    owner, target, model = intake_owner
    store = native_markdown_runtime
    display_name, content, originals = _markdown_input(packaged, markdown_headings)
    existing_files = set(intake_session.scalars(select(UserFile.id)))
    amendments = list(intake_session.scalars(select(AmendmentBatch.id)))
    key = uuid4()
    package, created = create_markdown_source_package(
        intake_session,
        user=owner,
        environment=_ENVIRONMENT,
        idempotency_key=key,
        display_name=display_name,
        content=content,
        store=store,
    )
    assert created and package.document_set_id == target.id
    assert package.input_spec["preparation_method"] == "supplied_markdown"
    assert package.input_spec["kind"] == "legislation_markdown_v1"
    repeated, created_again = create_markdown_source_package(
        intake_session,
        user=owner,
        environment=_ENVIRONMENT,
        idempotency_key=key,
        display_name=display_name,
        content=content,
        store=store,
    )
    assert repeated.id == package.id and not created_again
    package_id = package.id
    intake_session.commit()
    source_job.run_source_package(package_id=package_id, environment=_ENVIRONMENT)
    source_job.run_source_package(package_id=package_id, environment=_ENVIRONMENT)
    intake_session.expire_all()
    package = intake_session.get_one(AmendmentSourcePackage, package_id)
    assert package.status == "ready" and not package.issues
    assets = list_source_assets(intake_session, package_id)
    assert len(assets) == (3 if packaged else 1)
    assert {asset.mime_type for asset in assets} == {"text/markdown"}
    assert {asset.sha256 for asset in assets} == set(originals)
    for asset in assets:
        with store.read_file(asset.file_id) as stream:
            assert stream.read() == originals[asset.sha256]
        assert asset.text_file_id is not None
        with store.read_file(asset.text_file_id) as stream:
            assert stream.read() == originals[asset.sha256]

    batch, _created = repository.create_intake(
        intake_session,
        source_package_id=package_id,
        idempotency_key=uuid4(),
        labeling_model_configuration_id=model.id,
        user=owner,
        environment=_ENVIRONMENT,
    )
    batch_id, model_id = batch.id, model.id
    intake_session.commit()
    monkeypatch.setattr(
        repository,
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
    run_id = intake_job.run_document_intake(intake_id=batch_id, tenant_id="public")
    assert run_id is not None
    intake_session.expire_all()
    batch = intake_session.get_one(type(batch), batch_id)
    members = repository.intake_files(intake_session, batch)
    new_files = {member.user_file_id for member in members}
    assert len(members) == len(originals)
    assert not new_files.intersection(existing_files)
    root = next(member for member in members if member.role == "root")
    annexes = [member for member in members if member.role == "annex"]
    assert len(annexes) == (2 if packaged else 0)
    assert all(member.parent_document_id == root.id for member in annexes)
    assert all(member.relationships for member in annexes)
    snapshot = repository.intake_snapshot(intake_session, batch)
    document_views = {document["id"]: document for document in snapshot["documents"]}
    root_links = document_views[str(root.id)]["document_links"]
    assert {link["target_document_id"] for link in root_links} == {
        str(annex.id) for annex in annexes
    }
    assert {link["href"] for link in root_links} == (
        {"ekler/ek-1.md", "ekler/ek-2.md"} if packaged else set()
    )
    assert all(
        link["target_document_id"] in document_views
        for document in snapshot["documents"]
        for link in document["document_links"]
    )
    chunks_by_file: dict[str, str] = {}
    for member in members:
        file = intake_session.get_one(UserFile, member.user_file_id)
        asset = intake_session.get_one(RegulatorySourceAsset, member.source_asset_id)
        assert file.content_type == "text/markdown" and file.file_id != asset.file_id
        with store.read_file(file.file_id) as stream:
            assert stream.read() == originals[asset.sha256]
        chunks_by_file[str(file.id)] = "\n".join(
            intake_session.scalars(
                select(RegulatoryChunk.text).where(
                    RegulatoryChunk.user_file_id == file.id
                )
            )
        )
    assert "ROOT_NATIVE_MARKDOWN" in chunks_by_file[str(root.user_file_id)]
    assert "ANNEX_ONE_NATIVE" not in chunks_by_file[str(root.user_file_id)]
    if markdown_headings:
        atomic_rows = [
            row
            for row in intake_session.scalars(
                select(RegulatoryChunk).where(
                    RegulatoryChunk.user_file_id == root.user_file_id
                )
            )
            if row.chunk_metadata.get("chunk_variant") == "atomic"
        ]
        assert len(atomic_rows) == 3
        for heading, body in (
            ("## MADDE 1 — Başvuru usulü", "ROOT_NATIVE_MARKDOWN"),
            ("## MADDE 2 — Belge düzeni", "ROOT_OPERATIVE_TWO"),
            ("## MADDE 3 — Yürürlük", "ROOT_OPERATIVE_THREE"),
        ):
            assert any(heading in row.text and body in row.text for row in atomic_rows)
        assert all(
            any(
                marker in row.text
                for marker in (
                    "ROOT_NATIVE_MARKDOWN",
                    "ROOT_OPERATIVE_TWO",
                    "ROOT_OPERATIVE_THREE",
                )
            )
            for row in atomic_rows
        )
    if packaged:
        assert "tıklayınız" not in chunks_by_file[str(root.user_file_id)]
        assert "[EK-1](ekler/ek-1.md)" in chunks_by_file[str(root.user_file_id)]
        assert any(
            "Ekler için tıklayınız" in relationship.get("label", "")
            for relationship in root.relationships
        )
        assert any("ANNEX_ONE_NATIVE" in text for text in chunks_by_file.values())
        assert any("ANNEX_TWO_NATIVE" in text for text in chunks_by_file.values())
    run = intake_session.get_one(RegulatoryLabelingRun, run_id)
    assert run.model == "gemini-3.5-flash"
    assert set(run.file_ids) == {str(identifier) for identifier in new_files}
    items = list(
        intake_session.scalars(
            select(RegulatoryLabelingItem).where(
                RegulatoryLabelingItem.run_id == run_id
            )
        )
    )
    assert items and {item.user_file_id for item in items} == new_files
    assert list(intake_session.scalars(select(AmendmentBatch.id))) == amendments


def test_declared_but_unlinked_markdown_annex_fails_before_admission(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    native_markdown_runtime: PostgresBackedFileStore,
) -> None:
    owner, _target, model = intake_owner
    archive = BytesIO()
    with ZipFile(archive, "w") as zipped:
        zipped.writestr(
            "manifest.json",
            json.dumps(
                {
                    "format": "legislation-markdown-v1",
                    "main": "ana.md",
                    "documents": [
                        {"path": "ana.md", "title": "Ana belge"},
                        {
                            "path": "ek.md",
                            "title": "Eksik bağlantılı ek",
                            "parent": "ana.md",
                        },
                    ],
                }
            ),
        )
        zipped.writestr("ana.md", "# Ana belge\n\nMADDE 1 — Ek bağlantısı yok.\n")
        zipped.writestr("ek.md", "# Ek belge\n\n1. Bağımsız ek içeriği.\n")
    files_before = set(intake_session.scalars(select(UserFile.id)))
    package, _created = create_markdown_source_package(
        intake_session,
        user=owner,
        environment=_ENVIRONMENT,
        idempotency_key=uuid4(),
        display_name="unlinked.zip",
        content=archive.getvalue(),
        store=native_markdown_runtime,
    )
    package_id = package.id
    intake_session.commit()
    with pytest.raises(MarkdownPackageError):
        source_job.run_source_package(package_id=package_id, environment=_ENVIRONMENT)
    intake_session.expire_all()
    package = intake_session.get_one(AmendmentSourcePackage, package_id)
    assert package.status == "failed" and not list_source_assets(
        intake_session, package_id
    )
    assert len(package.issues) == 1
    assert package.issues[0]["code"] == "invalid_markdown_package"
    assert package.issues[0]["retryable"] is False
    assert package.issues[0]["failure_detail"]
    with pytest.raises(ValueError, match="complete|verified"):
        repository.create_intake(
            intake_session,
            source_package_id=package_id,
            idempotency_key=uuid4(),
            labeling_model_configuration_id=model.id,
            user=owner,
            environment=_ENVIRONMENT,
        )
    assert set(intake_session.scalars(select(UserFile.id))) == files_before
