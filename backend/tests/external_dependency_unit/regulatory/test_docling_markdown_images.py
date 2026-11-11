"""Actual Docling originals stage and prepare into two owner-scoped files, never index."""

from pathlib import Path
from types import SimpleNamespace
from typing import cast
from uuid import uuid4

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from onyx.access.access import user_can_access_chat_file
from onyx.db import document_intake as intake
from onyx.db import markdown_drafts as drafts
from onyx.db.document_intake_metadata import load_metadata_work, metadata_chunks
from onyx.db.document_intake_metadata_checkpoint import _source_snapshot
from onyx.db.models import (
    DocumentSet,
    ModelConfiguration,
    RegulatoryChunk,
    RegulatoryDocumentIntakeFile,
    RegulatoryIndexingJob,
    User,
    UserFile,
)
from onyx.file_store.postgres_file_store import PostgresBackedFileStore
from onyx.indexing.embedder import DefaultIndexingEmbedder
from onyx.llm.interfaces import LLM
from onyx.natural_language_processing.utils import get_tokenizer
from onyx.regulatory.amendments import analysis_llm
from onyx.regulatory.amendments.annexes import source_images
from onyx.regulatory.amendments.annexes.models import AcquiredAsset
from onyx.regulatory.document_intake import job as intake_job
from onyx.regulatory.document_intake.metadata import build_metadata_evidence_catalog
from shared_configs.contextvars import get_current_tenant_id
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

FIXTURES = Path(__file__).parents[2] / "unit/onyx/regulatory/fixtures/docling_737039bf"
MAIN = "dem__ryolu_altyapi_er______m_ve_kapas__te_tahs__s_y__netmel______.md"
ANNEX = "20230928-1-1.md"


def test_actual_docling_pair_preserves_images_sources_and_parent_before_metadata(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    native_markdown_runtime: PostgresBackedFileStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, store = (intake_session, native_markdown_runtime)
    owner, _target, model = intake_owner
    existing_files = set(session.scalars(select(UserFile.id)))
    existing_index_jobs = session.scalar(
        select(func.count()).select_from(RegulatoryIndexingJob)
    )
    draft = drafts.create_draft(
        session, user=owner, environment=_ENVIRONMENT, idempotency_key=uuid4()
    )

    def forbidden_model(**_kwargs: object) -> LLM:
        pytest.fail("Staging attempted image transcription")

    monkeypatch.setattr(analysis_llm, "get_amendment_analysis_llm", forbidden_model)
    drafts.add_documents(
        session,
        draft,
        files=[(MAIN, (FIXTURES / MAIN).read_bytes())],
        expected_revision=0,
        idempotency_key=uuid4(),
        store=store,
    )
    root = drafts.draft_documents(session, draft)[0]
    drafts.add_documents(
        session,
        draft,
        files=[(ANNEX, (FIXTURES / ANNEX).read_bytes())],
        expected_revision=1,
        idempotency_key=uuid4(),
        parent_document_id=root.id,
        store=store,
    )
    assert set(session.scalars(select(UserFile.id))) == existing_files
    package = drafts.prepare_draft(session, draft, expected_revision=2, store=store)
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
    token = intake.claim_intake(session, batch_id)
    assert token is not None
    session.commit()
    tokenizer = get_tokenizer("gemini-embedding-2", "google")
    monkeypatch.setattr(
        DefaultIndexingEmbedder,
        "from_db_search_settings",
        lambda **_kwargs: SimpleNamespace(
            embedding_model=SimpleNamespace(tokenizer=tokenizer)
        ),
    )
    calls = []

    def transcribe(
        asset: AcquiredAsset, *, llm: LLM | None, deadline: float
    ) -> AcquiredAsset:
        assert llm is not None and deadline > 0
        calls.append(asset.sha256)
        assert (
            asset.sha256
            == "8ed2e7123d314a1dd4669ff7952d3879ff82504ccd0c249d106937bbb203a8cd"
        )
        return asset.model_copy(
            update={"text": (FIXTURES / "diagram-transcript.md").read_text()}
        )

    monkeypatch.setattr(
        analysis_llm,
        "get_amendment_analysis_llm",
        lambda **_kwargs: cast(
            LLM,
            SimpleNamespace(
                config=SimpleNamespace(
                    model_provider="fixture", model_name="verified-diagram"
                )
            ),
        ),
    )
    monkeypatch.setattr(source_images, "prepare_image_source", transcribe)
    intake_job._prepare_intake_sources(
        intake_id=batch_id,
        tenant_id=get_current_tenant_id(),
        token=token,
        check_owner=lambda: None,
    )
    session.expire_all()
    members = list(
        session.scalars(
            select(RegulatoryDocumentIntakeFile).where(
                RegulatoryDocumentIntakeFile.intake_id == batch_id
            )
        )
    )
    assert len(members) == 2
    parent = next((m for m in members if m.role == "root"))
    child = next((m for m in members if m.role == "annex"))
    assert child.parent_document_id == parent.id and child.relationships
    assert set(session.scalars(select(UserFile.id))) - existing_files == {
        parent.user_file_id,
        child.user_file_id,
    }
    images = []
    for member, name in [(parent, MAIN), (child, ANNEX)]:
        file = session.get_one(UserFile, member.user_file_id)
        with store.read_file(file.file_id) as raw:
            assert raw.read() == (FIXTURES / name).read_bytes()
        rows = list(
            session.scalars(
                select(RegulatoryChunk)
                .where(RegulatoryChunk.user_file_id == file.id)
                .order_by(RegulatoryChunk.position, RegulatoryChunk.id)
            )
        )
        assert rows
        assert not any(("data:image/png;base64" in row.text for row in rows))
        targets = metadata_chunks(rows)
        assert targets and build_metadata_evidence_catalog(targets)
        for row in rows:
            if row.chunk_metadata.get("image_file_id"):
                assert member is child
                images.append(row)
        if member is parent:
            assert any(("MADDE 35" in row.text for row in rows))
    assert images
    image_ids = {r.chunk_metadata["image_file_id"] for r in images}
    assert len(image_ids) == 1 and len(calls) == 1
    delivered = "\n".join((row.text for row in images))
    for marker in [
        "15 Aralık",
        "15 Nisan",
        "Nihai Orer",
        "Tren çizgisi için son başvuru tarihi",
    ]:
        assert marker in delivered
    for image_id in image_ids:
        assert user_can_access_chat_file(image_id, owner, session)
    batch, metadata_work = load_metadata_work(session, batch_id, token)
    assert {member_id for member_id, _chunks in metadata_work} == {parent.id, child.id}
    snapshot, requests, completed = _source_snapshot(session, batch)
    assert set(snapshot["documents"]) == {str(parent.id), str(child.id)}
    assert set(requests) == {str(parent.id), str(child.id)} and not completed
    monkeypatch.setattr(analysis_llm, "get_amendment_analysis_llm", forbidden_model)
    intake_job._prepare_intake_sources(
        intake_id=batch_id,
        tenant_id=get_current_tenant_id(),
        token=token,
        check_owner=lambda: None,
    )
    assert len(calls) == 1
    assert (
        session.scalar(select(func.count()).select_from(RegulatoryIndexingJob))
        == existing_index_jobs
    )
