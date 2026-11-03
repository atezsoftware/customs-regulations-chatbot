"""Verified real matrix extraction reaches canonical v5 intact through normal storage."""

import json
from functools import partial
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from onyx.configs.constants import FileOrigin
from onyx.db import document_intake as intake
from onyx.db.document_intake_edit_budget import validate_intake_edit_budgets
from onyx.db.models import (
    AmendmentSourcePackage,
    DocumentSet,
    ModelConfiguration,
    RegulatoryChunk,
    RegulatorySourceAsset,
    SearchSettings,
    User,
    UserFile,
)
from onyx.file_store.postgres_file_store import PostgresBackedFileStore
from onyx.natural_language_processing.utils import TiktokenTokenizer
from onyx.regulatory.document_intake.models import (
    IntakePreparationPolicy,
    IntakeSourceAsset,
)
from onyx.regulatory.document_intake.preparation import load_verified_intake_documents
from onyx.regulatory.indexing import documents_to_regulatory_chunks
from onyx.regulatory.indexing_jobs.configuration import (
    compute_regulatory_chunk_generation_hash,
)
from tests.external_dependency_unit.regulatory.test_document_intake import _ENVIRONMENT
from tests.external_dependency_unit.regulatory.test_document_intake import (
    intake_owner as intake_owner,
)
from tests.external_dependency_unit.regulatory.test_document_intake import (
    intake_session as intake_session,
)
from tests.external_dependency_unit.regulatory.test_document_intake_preparation_policy import (
    google_settings as google_settings,
)


def test_real_retained_matrix_is_admitted_frozen_and_stored_without_date_clause_fragments(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    google_settings: SearchSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = intake_session
    owner, target, model = intake_owner
    fixture = json.loads(
        (
            Path(__file__).parents[2]
            / "unit/onyx/regulatory/fixtures/rg_matrix_retained_pages.json"
        ).read_text()
    )
    source = (
        "# EK V\n\n(3) Çapraz kümülasyon, 2012 kurallarına (C) göre mümkündür.\n\n"
        + "\n\n".join(page["text"] for page in fixture["pages"])
    )
    raw = source.encode()
    package = AmendmentSourcePackage(
        document_set_id=target.id,
        created_by=owner.id,
        environment=_ENVIRONMENT,
        idempotency_key=str(uuid4()),
        request_hash=sha256(raw).hexdigest(),
        input_spec={"url": "https://example.gov/verified-annex.md"},
        status="ready",
        asset_count=1,
        total_bytes=len(raw),
        issues=[],
    )
    session.add(package)
    session.flush()
    store = PostgresBackedFileStore()
    file_id = store.save_file(
        BytesIO(raw), "annex.md", FileOrigin.OTHER, "text/markdown", db_session=session
    )
    asset = RegulatorySourceAsset(
        package_id=package.id,
        sha256=sha256(raw).hexdigest(),
        file_id=file_id,
        text_file_id=file_id,
        text_sha256=sha256(raw).hexdigest(),
        mime_type="text/markdown",
        display_name="annex.md",
        byte_count=len(raw),
    )
    session.add(asset)
    session.flush()
    manifest = json.dumps(
        {
            "status": "ready",
            "assets": [{"sha256": asset.sha256}],
            "links": [],
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
    session.flush()
    batch, created = intake.create_intake(
        session,
        source_package_id=package.id,
        idempotency_key=uuid4(),
        labeling_model_configuration_id=model.id,
        user=owner,
        environment=_ENVIRONMENT,
    )
    assert created
    policy = IntakePreparationPolicy.model_validate(batch.preparation_policy)
    assert policy.chunking_version == "5"
    token = intake.claim_intake(session, batch.id)
    assert token is not None
    member = intake.admit_intake_assets(
        session, batch, [asset], [], root_hash=asset.sha256
    )[0]
    monkeypatch.setattr(
        store, "read_file", partial(store.read_file, db_session=session)
    )
    documents = load_verified_intake_documents(
        package_id=package.id,
        asset=IntakeSourceAsset.model_validate(asset),
        user_file_id=member.user_file_id,
        links=[],
        package_assets=[IntakeSourceAsset.model_validate(asset)],
        store=store,
        chunking_version=policy.chunking_version,
        preparation_policy=policy,
    )
    assert documents.original_extraction.raw_sha256 == asset.sha256
    chunks = documents_to_regulatory_chunks(
        documents,
        session,
        TiktokenTokenizer("text-embedding-3-large"),
        enable_contextual_rag=True,
    )
    assert chunks
    intake.mark_intake_file_prepared(session, batch.id, token, member.id)
    session.flush()
    rows = list(
        session.scalars(
            select(RegulatoryChunk)
            .where(RegulatoryChunk.user_file_id == member.user_file_id)
            .order_by(RegulatoryChunk.position)
        )
    )
    tables = [row for row in rows if row.chunk_type == "table"]
    assert len(tables) == 4
    assert not any(
        row.chunk_type == "clause" and row.text.startswith(("(R)", "(C)"))
        for row in rows
    )
    for row in tables:
        assert "Tablo 2:" in row.heading_path[-1]
        assert not any("(3) Çapraz" in part for part in row.heading_path)
        assert row.chunk_metadata["intake_chunking_version"] == "5"
        assert (
            row.chunk_metadata["intake_logical_table"]["coordinate_system"]
            == "retained_pipe_cells"
        )
        assert row.text in source
    file = session.get_one(UserFile, member.user_file_id)
    file.regulatory_chunk_generation_hash = compute_regulatory_chunk_generation_hash(
        embedding_provider=google_settings.provider_type,
        embedding_model_name=google_settings.model_name,
        enable_contextual_rag=True,
        intake_chunking_version="5",
        intake_preparation_policy=policy,
    )
    session.flush()
    validate_intake_edit_budgets(session, {file.id: file}, {file.id: rows})
    assert member.status == "chunked"
    assert sum("(R)" in row.text for row in tables) == 4
