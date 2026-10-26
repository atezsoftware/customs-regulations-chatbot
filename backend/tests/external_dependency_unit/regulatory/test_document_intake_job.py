"""Actual canonical preparation and scoped label admission against local PostgreSQL."""

import json
from collections.abc import Generator
from contextlib import contextmanager
from hashlib import sha256
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from onyx.configs.constants import FileOrigin
from onyx.db import document_intake as repository
from onyx.db import regulatory_publication, regulatory_writer_publication
from onyx.db.enums import UserFileStatus
from onyx.db.models import (
    AmendmentBatch,
    AmendmentSourcePackage,
    DocumentSet,
    ModelConfiguration,
    RegulatoryChunk,
    RegulatoryFilePublication,
    RegulatoryLabelingItem,
    RegulatoryLabelingRun,
    RegulatorySourceAsset,
    User,
    UserFile,
)
from onyx.file_store import postgres_file_store
from onyx.file_store.postgres_file_store import PostgresBackedFileStore
from onyx.indexing.embedder import DefaultIndexingEmbedder
from onyx.natural_language_processing.utils import BaseTokenizer
from onyx.regulatory.amendments.annexes.models import SourceLink
from onyx.regulatory.document_intake import job
from tests.external_dependency_unit.regulatory.test_document_intake import (
    intake_owner as intake_owner,
)
from tests.external_dependency_unit.regulatory.test_document_intake import (
    intake_session as intake_session,
)


class _CharacterTokenizer(BaseTokenizer):
    def encode(self, string: str) -> list[int]:
        return [ord(character) for character in string]

    def tokenize(self, string: str) -> list[str]:
        return list(string)

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(token) for token in tokens)


def _verified_html_package(
    session: Session, owner: User, target: DocumentSet
) -> AmendmentSourcePackage:
    package = AmendmentSourcePackage(
        document_set_id=target.id,
        created_by=owner.id,
        environment="document-intake-regression",
        idempotency_key=str(uuid4()),
        request_hash=uuid4().hex * 2,
        input_spec={"url": "https://example.gov/new-law.html"},
        status="processing",
    )
    session.add(package)
    session.flush()
    store = PostgresBackedFileStore()
    assets: list[RegulatorySourceAsset] = []
    for name, source, prepared in (
        (
            "new-law.html",
            "<main><h1>Yeni Belge</h1><p>MADDE 1 - ROOT_ONLY_RULE.</p>"
            '<a href="EK-1.html">Ekleri için tıklayınız</a></main>',
            "Yeni Belge\nMADDE 1 - ROOT_ONLY_RULE.\nEkleri için tıklayınız",
        ),
        (
            "EK-1.html",
            "<main><h1>EK-1</h1><table><tr><th>GTIP</th><th>Belge</th></tr>"
            "<tr><td>0101</td><td>ANNEX_ONLY_CERTIFICATE</td></tr></table></main>",
            "EK-1\nGTIP\nBelge\n0101\nANNEX_ONLY_CERTIFICATE",
        ),
    ):
        raw, text = source.encode(), prepared.encode()
        file_id = store.save_file(
            BytesIO(raw), name, FileOrigin.OTHER, "text/html", db_session=session
        )
        text_id = store.save_file(
            BytesIO(text),
            "extracted.txt",
            FileOrigin.OTHER,
            "text/plain",
            db_session=session,
        )
        asset = RegulatorySourceAsset(
            package_id=package.id,
            sha256=sha256(raw).hexdigest(),
            file_id=file_id,
            text_file_id=text_id,
            text_sha256=sha256(text).hexdigest(),
            mime_type="text/html",
            display_name=name,
            byte_count=len(raw),
            original_url=f"https://example.gov/{name}",
            final_url=f"https://example.gov/{name}",
        )
        assets.append(asset)
    session.add_all(assets)
    session.flush()
    link = SourceLink(
        parent_asset_hash=assets[0].sha256,
        target_asset_hash=assets[1].sha256,
        parent_url=assets[0].original_url,
        requested_url=assets[1].original_url,
        original_url="EK-1.html",
        final_url=assets[1].final_url,
        source_field="html:0",
        label="Ekleri için tıklayınız",
        kind="url",
    )
    manifest = json.dumps(
        {
            "status": "ready",
            "assets": [
                {"sha256": asset.sha256, "mime_type": asset.mime_type}
                for asset in assets
            ],
            "links": [link.model_dump(mode="json")],
            "issues": [],
        }
    ).encode()
    package.manifest_file_id = store.save_file(
        BytesIO(manifest),
        "source-manifest.json",
        FileOrigin.OTHER,
        "application/json",
        db_session=session,
    )
    package.manifest_sha256 = sha256(manifest).hexdigest()
    package.asset_count = len(assets)
    package.total_bytes = sum(asset.byte_count for asset in assets)
    package.status = "ready"
    session.flush()
    return package


def test_job_chunks_independent_originals_and_freezes_only_new_label_scope(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner, target, model = intake_owner
    package = _verified_html_package(intake_session, owner, target)
    amendment_ids = list(intake_session.scalars(select(AmendmentBatch.id)))
    batch, _ = repository.create_intake(
        intake_session,
        source_package_id=package.id,
        idempotency_key=uuid4(),
        labeling_model_configuration_id=model.id,
        user=owner,
        environment="document-intake-regression",
    )
    intake_id, model_id = batch.id, model.id
    intake_session.commit()
    connection = intake_session.connection()

    @contextmanager
    def shared_connection_session(**_kwargs: object) -> Generator[Session, None, None]:
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
            with shared_connection_session() as session:
                yield session

    monkeypatch.setattr(
        job, "get_session_with_current_tenant", shared_connection_session
    )
    monkeypatch.setattr(
        regulatory_publication, "get_session_with_tenant", shared_connection_session
    )
    monkeypatch.setattr(
        regulatory_writer_publication,
        "get_session_with_tenant",
        shared_connection_session,
    )
    monkeypatch.setattr(
        postgres_file_store,
        "get_session_with_current_tenant",
        shared_connection_session,
    )
    monkeypatch.setattr(
        postgres_file_store, "get_session_with_current_tenant_if_none", optional_session
    )
    monkeypatch.setattr(
        DefaultIndexingEmbedder,
        "from_db_search_settings",
        lambda **_kwargs: SimpleNamespace(
            embedding_model=SimpleNamespace(tokenizer=_CharacterTokenizer())
        ),
    )
    monkeypatch.setattr(
        repository,
        "resolve_intake_labeling_provider_binding",
        lambda *_args, **_kwargs: SimpleNamespace(
            model="gemini-3.5-flash",
            model_dump=lambda **_options: {"model_configuration_id": model_id},
        ),
    )
    metadata = MagicMock()
    monkeypatch.setattr(
        "onyx.regulatory.document_intake.metadata_job.run_intake_metadata", metadata
    )
    run_id = job.run_document_intake(intake_id=intake_id, tenant_id="public")
    metadata.assert_called_once()
    assert metadata.call_args.kwargs["intake_id"] == intake_id
    assert run_id is not None
    intake_session.expire_all()
    batch = intake_session.get_one(type(batch), intake_id)
    members = repository.intake_files(intake_session, batch)
    assert len(members) == 2
    assert {member.status for member in members} == {"labeling"}
    new_files = {member.user_file_id for member in members}
    run = intake_session.get_one(RegulatoryLabelingRun, run_id)
    assert set(run.file_ids) == {str(identifier) for identifier in new_files}
    assert run.status == "queued"
    items = list(
        intake_session.scalars(
            select(RegulatoryLabelingItem).where(
                RegulatoryLabelingItem.run_id == run_id
            )
        )
    )
    assert items and {item.user_file_id for item in items} == new_files
    root = next(member for member in members if member.role == "root")
    annex = next(member for member in members if member.role == "annex")
    assert annex.parent_document_id == root.id
    texts: dict[UUID, str] = {}
    for identifier in new_files:
        texts[identifier] = "\n".join(
            intake_session.scalars(
                select(RegulatoryChunk.text).where(
                    RegulatoryChunk.user_file_id == identifier
                )
            )
        )
        file = intake_session.get_one(UserFile, identifier)
        assert file.status == UserFileStatus.CHUNKED
        member = next(item for item in members if item.user_file_id == identifier)
        asset = intake_session.get_one(RegulatorySourceAsset, member.source_asset_id)
        assert file.file_id != asset.file_id
        publication = intake_session.get_one(RegulatoryFilePublication, identifier)
        assert publication.original_ingestion_receipt is not None
        assert publication.original_ingestion_receipt["file_id"] == file.file_id
        assert publication.original_ingestion_receipt["raw_sha256"] == asset.sha256
        assert publication.epoch == 0
        assert publication.writer_manifest is None
    assert "ROOT_ONLY_RULE" in texts[root.user_file_id]
    assert "ANNEX_ONLY_CERTIFICATE" not in texts[root.user_file_id]
    assert "ANNEX_ONLY_CERTIFICATE" in texts[annex.user_file_id]
    assert "GTIP\tBelge" in texts[annex.user_file_id]
    assert list(intake_session.scalars(select(AmendmentBatch.id))) == amendment_ids
