"""Real intake writer keeps legal headings with body under the embedding budget."""

import re
from types import SimpleNamespace
from uuid import uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from onyx.db import document_intake as repository
from onyx.db.document_intake_sources import create_markdown_source_package
from onyx.db.models import (
    DocumentSet,
    ModelConfiguration,
    RegulatoryChunk,
    RegulatoryFilePublication,
    RegulatoryLabelingRun,
    User,
    UserFile,
)
from onyx.db.search_settings import get_current_search_settings
from onyx.file_store.postgres_file_store import PostgresBackedFileStore
from onyx.indexing.contextual_settings import effective_contextual_rag_enabled
from onyx.indexing.embedder import DefaultIndexingEmbedder
from onyx.natural_language_processing.utils import TiktokenTokenizer
from onyx.regulatory.amendments.annexes import job as source_job
from onyx.regulatory.document_intake import job as intake_job
from onyx.regulatory.document_intake.chunking_profile import (
    INTAKE_CHUNKING_CODE_VERSION,
)
from onyx.regulatory.indexing_jobs.configuration import (
    compute_regulatory_chunk_generation_hash,
)
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


def test_long_native_markdown_uses_real_tokenizer_without_heading_only_chunks(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    native_markdown_runtime: PostgresBackedFileStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner, _target, model = intake_owner
    tokenizer = TiktokenTokenizer("text-embedding-3-large")
    monkeypatch.setattr(
        DefaultIndexingEmbedder,
        "from_db_search_settings",
        lambda **_kwargs: SimpleNamespace(
            embedding_model=SimpleNamespace(tokenizer=tokenizer)
        ),
    )
    monkeypatch.setattr("onyx.regulatory.indexing.DOC_EMBEDDING_CONTEXT_SIZE", 512)
    bodies = [
        (
            f"MADDE_{index}_GOVDE: Başvuru sahibi ithalat beyannamesini ve ticari "
            "faturayı birlikte sunar. Eksik belge tespit edilirse eksiklik açıkça "
            "bildirilir; tamamlanmadan işlem sonuçlandırılmaz. "
        )
        * 8
        for index in range(1, 4)
    ]
    article_text = "\n\n".join(
        f"## MADDE {index} — Başvuru koşulları {index}\n\n{body}"
        for index, body in enumerate(bodies, 1)
    )
    raw = ("# Uzun madde bütçe testi\n\n" + article_text).encode()
    package, _created = create_markdown_source_package(
        intake_session,
        user=owner,
        environment=_ENVIRONMENT,
        idempotency_key=uuid4(),
        display_name="uzun-maddeler.md",
        content=raw,
        store=native_markdown_runtime,
    )
    package_id = package.id
    intake_session.commit()
    source_job.run_source_package(package_id=package_id, environment=_ENVIRONMENT)
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
    intake_session.expire_all()
    batch = intake_session.get_one(type(batch), batch_id)
    member = repository.intake_files(intake_session, batch)[0]
    rows = list(
        intake_session.scalars(
            select(RegulatoryChunk)
            .where(RegulatoryChunk.user_file_id == member.user_file_id)
            .order_by(RegulatoryChunk.position)
        )
    )
    atomic = [row for row in rows if row.chunk_metadata["chunk_variant"] == "atomic"]
    assert len(atomic) > 3
    assert all(len(tokenizer.encode(row.text)) <= 312 for row in rows)
    assert all("intake_chunking_version" in row.chunk_metadata for row in rows)
    assert all(
        not (row.text.startswith("##") and "\n" not in row.text) for row in atomic
    )
    assert re.sub(r"\s+", "", "".join(row.text for row in atomic)) == re.sub(
        r"\s+", "", article_text
    )
    for index in range(1, 4):
        article = [
            row for row in atomic if row.chunk_metadata["article_no"] == str(index)
        ]
        first = article[0]
        assert f"## MADDE {index}" in first.text
        assert f"MADDE_{index}_GOVDE" in first.text
        for row in article:
            assert row.chunk_metadata["article_title"] == f"Başvuru koşulları {index}"
            assert row.heading_path[-1] == f"MADDE {index} - Başvuru koşulları {index}"
    file = intake_session.get_one(UserFile, member.user_file_id)
    with native_markdown_runtime.read_file(file.file_id) as stream:
        assert stream.read() == raw
    receipt = intake_session.get_one(
        RegulatoryFilePublication, file.id
    ).original_ingestion_receipt
    assert receipt is not None
    assert receipt["generation_hash"] == file.regulatory_chunk_generation_hash
    setting = get_current_search_settings(intake_session)
    assert receipt["generation_hash"] == compute_regulatory_chunk_generation_hash(
        embedding_provider=setting.provider_type,
        embedding_model_name=setting.model_name,
        enable_contextual_rag=effective_contextual_rag_enabled(setting),
        intake_chunking_version=INTAKE_CHUNKING_CODE_VERSION,
    )
    assert receipt["generation_hash"] != compute_regulatory_chunk_generation_hash(
        embedding_provider=setting.provider_type,
        embedding_model_name=setting.model_name,
        enable_contextual_rag=effective_contextual_rag_enabled(setting),
    )
    run = intake_session.get_one(RegulatoryLabelingRun, run_id)
    assert run.file_ids == [str(file.id)]
