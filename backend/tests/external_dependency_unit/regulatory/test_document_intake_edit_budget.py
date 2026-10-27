"""Intake corrections keep the frozen content reserve before any paid relabel work."""

from uuid import uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from onyx.db import document_intake as intake
from onyx.db import document_intake_edit as editing
from onyx.db import document_intake_edit_budget as budget
from onyx.db.models import (
    CloudEmbeddingProvider,
    DocumentSet,
    ModelConfiguration,
    RegulatoryChunk,
    RegulatoryDocumentIntake,
    RegulatoryDocumentIntakeFile,
    RegulatoryLabelingRun,
    User,
    UserFile,
)
from onyx.db.search_settings import get_current_search_settings
from onyx.indexing.chunker import DEFAULT_CONTEXTUAL_RAG_RESERVED_TOKENS
from onyx.indexing.embedder import DefaultIndexingEmbedder
from onyx.natural_language_processing.utils import BaseTokenizer
from onyx.regulatory.indexing_jobs.configuration import (
    compute_regulatory_chunk_generation_hash,
)
from shared_configs.configs import DOC_EMBEDDING_CONTEXT_SIZE
from shared_configs.enums import EmbeddingProvider
from tests.external_dependency_unit.regulatory.test_document_intake import (
    intake_owner as intake_owner,
)
from tests.external_dependency_unit.regulatory.test_document_intake import (
    intake_session as intake_session,
)
from tests.external_dependency_unit.regulatory.test_document_intake_bulk_edit import (
    _snapshot,
)
from tests.external_dependency_unit.regulatory.test_document_intake_edit import (
    _ready,
    _root_chunk,
)

_REAL_TOKENIZER = budget._embedding_tokenizer


def _prepared(
    session: Session,
    owner: tuple[User, DocumentSet, ModelConfiguration],
    monkeypatch: pytest.MonkeyPatch,
    *,
    version: str = "2",
    aggregate: bool = False,
) -> tuple[RegulatoryDocumentIntake, list[RegulatoryDocumentIntakeFile], BaseTokenizer]:
    settings = get_current_search_settings(session)
    if session.get(CloudEmbeddingProvider, EmbeddingProvider.OPENROUTER) is None:
        session.add(CloudEmbeddingProvider(provider_type=EmbeddingProvider.OPENROUTER))
        session.flush()
    settings.provider_type = EmbeddingProvider.OPENROUTER
    settings.model_name = "openai/text-embedding-3-large"
    settings.enable_contextual_rag = True
    session.flush()
    batch, members = _ready(
        session, owner, monkeypatch, aggregate=aggregate, chunking_version=version
    )
    monkeypatch.setattr(budget, "_embedding_tokenizer", _REAL_TOKENIZER)

    def forbidden(*_args: object, **_kwargs: object) -> None:
        pytest.fail("Intake edit preflight must not call an embedding provider")

    monkeypatch.setattr(DefaultIndexingEmbedder, "embed_chunks", forbidden)
    return batch, members, _REAL_TOKENIZER(settings)


@pytest.mark.parametrize("version", ["1", "2"])
def test_edit_accepts_exact_frozen_content_boundary_with_real_indexing_tokenizer(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    monkeypatch: pytest.MonkeyPatch,
    version: str,
) -> None:
    batch, members, tokenizer = _prepared(
        intake_session, intake_owner, monkeypatch, version=version
    )
    root, chunk = _root_chunk(intake_session, members)
    limit = DOC_EMBEDDING_CONTEXT_SIZE - DEFAULT_CONTEXTUAL_RAG_RESERVED_TOKENS
    text = " a" * limit
    assert len(tokenizer.encode(text)) == limit == 312
    assert batch.review_sha256
    run_id = editing.edit_intake_chunk(
        intake_session,
        batch,
        document_id=root.id,
        chunk_id=chunk.id,
        expected_review_sha256=batch.review_sha256,
        updates={"text": text},
    )
    assert run_id and chunk.text == text
    assert intake_session.get_one(RegulatoryLabelingRun, run_id).completed_chunks == 1


@pytest.mark.parametrize("version", ["1", "2"])
def test_one_over_budget_edit_rejects_all_files_before_binding_or_a_new_run(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    monkeypatch: pytest.MonkeyPatch,
    version: str,
) -> None:
    batch, members, tokenizer = _prepared(
        intake_session, intake_owner, monkeypatch, version=version
    )
    root, first = _root_chunk(intake_session, members)
    annex = next(member for member in members if member.role == "annex")
    last = intake_session.scalars(
        select(RegulatoryChunk).where(
            RegulatoryChunk.user_file_id == annex.user_file_id
        )
    ).one()
    limit = DOC_EMBEDDING_CONTEXT_SIZE - DEFAULT_CONTEXTUAL_RAG_RESERVED_TOKENS
    text = " a" * (limit + 1)
    assert len(tokenizer.encode(text)) == limit + 1 == 313

    def forbidden(*_args: object, **_kwargs: object) -> None:
        pytest.fail("Oversized edits must fail before a labeling binding is resolved")

    monkeypatch.setattr(editing, "resolve_intake_labeling_provider_binding", forbidden)
    assert batch.review_sha256
    before = _snapshot(intake_session)
    with pytest.raises(ValueError, match=rf"{last.id}.*313.*312"):
        editing.edit_intake_chunks(
            intake_session,
            batch,
            expected_review_sha256=batch.review_sha256,
            edits=[
                editing.IntakeChunkEdit(
                    root.id,
                    first.id,
                    {"text": "Valid source correction must also roll back"},
                ),
                editing.IntakeChunkEdit(annex.id, last.id, {"text": text}),
            ],
        )
    assert _snapshot(intake_session) == before


def test_rebuilt_aggregate_is_preflighted_even_when_each_edited_source_fits(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch, members, tokenizer = _prepared(
        intake_session, intake_owner, monkeypatch, aggregate=True
    )
    root, first = _root_chunk(intake_session, members)
    second = intake_session.scalars(
        select(RegulatoryChunk).where(
            RegulatoryChunk.user_file_id == root.user_file_id,
            RegulatoryChunk.position == 1,
        )
    ).one()
    aggregate = intake_session.scalars(
        select(RegulatoryChunk).where(
            RegulatoryChunk.user_file_id == root.user_file_id,
            RegulatoryChunk.chunk_type == "hierarchical_aggregate",
        )
    ).one()
    text = " a" * 180
    assert len(tokenizer.encode(text)) == 180
    assert batch.review_sha256
    before = _snapshot(intake_session)
    with pytest.raises(ValueError, match=rf"{aggregate.id}.*312"):
        editing.edit_intake_chunks(
            intake_session,
            batch,
            expected_review_sha256=batch.review_sha256,
            edits=[
                editing.IntakeChunkEdit(root.id, first.id, {"text": text}),
                editing.IntakeChunkEdit(root.id, second.id, {"text": text}),
            ],
        )
    assert _snapshot(intake_session) == before


@pytest.mark.parametrize(
    "drift",
    ["missing_generation", "unknown_generation", "changed_settings", "mixed_versions"],
)
def test_preflight_rejects_generation_or_setting_drift_without_mutating_review(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
) -> None:
    batch, members, _tokenizer = _prepared(intake_session, intake_owner, monkeypatch)
    root, chunk = _root_chunk(intake_session, members)
    file = intake_session.get_one(UserFile, root.user_file_id)
    if drift == "missing_generation":
        file.regulatory_chunk_generation_hash = None
    elif drift == "unknown_generation":
        file.regulatory_chunk_generation_hash = uuid4().hex * 2
    elif drift == "changed_settings":
        get_current_search_settings(
            intake_session
        ).model_name = "openai/text-embedding-3-small"
    else:
        chunk.chunk_metadata = {**chunk.chunk_metadata, "intake_chunking_version": "1"}
    intake_session.flush()
    batch.review_sha256 = intake.review_digest(intake_session, batch)
    receipt = batch.review_sha256
    assert receipt
    before = _snapshot(intake_session)
    with pytest.raises(intake.IntakeConflictError, match="generation|versions"):
        editing.edit_intake_chunk(
            intake_session,
            batch,
            document_id=root.id,
            chunk_id=chunk.id,
            expected_review_sha256=receipt,
            updates={
                "text": "Small correction which must retain the frozen generation"
            },
        )
    assert _snapshot(intake_session) == before


@pytest.mark.parametrize("frozen_contextual", [True, False])
def test_intake_resolver_context_policy_stays_true_in_a_false_workspace(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    monkeypatch: pytest.MonkeyPatch,
    frozen_contextual: bool,
) -> None:
    batch, members, tokenizer = _prepared(intake_session, intake_owner, monkeypatch)
    root, chunk = _root_chunk(intake_session, members)
    setting = get_current_search_settings(intake_session)
    setting.enable_contextual_rag = False
    monkeypatch.setattr(
        budget, "effective_contextual_rag_enabled", lambda _setting: False
    )
    if not frozen_contextual:
        intake_session.get_one(
            UserFile, root.user_file_id
        ).regulatory_chunk_generation_hash = compute_regulatory_chunk_generation_hash(
            embedding_provider=setting.provider_type,
            embedding_model_name=setting.model_name,
            enable_contextual_rag=False,
            intake_chunking_version="2",
        )
    intake_session.flush()
    batch.review_sha256 = intake.review_digest(intake_session, batch)
    receipt = batch.review_sha256
    assert receipt
    text = " a" * 313
    assert len(tokenizer.encode(text)) == 313
    before = _snapshot(intake_session)
    expected = ValueError if frozen_contextual else intake.IntakeConflictError
    match = rf"{chunk.id}.*313.*312" if frozen_contextual else "generation"
    with pytest.raises(expected, match=match):
        editing.edit_intake_chunk(
            intake_session,
            batch,
            document_id=root.id,
            chunk_id=chunk.id,
            expected_review_sha256=receipt,
            updates={"text": text},
        )
    assert _snapshot(intake_session) == before
