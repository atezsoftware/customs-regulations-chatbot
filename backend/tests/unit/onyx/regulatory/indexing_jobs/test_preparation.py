import datetime
from collections.abc import Callable
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest
from sqlalchemy.orm import Session

from onyx.configs.constants import DocumentSource
from onyx.connectors.models import Document, TextSection
from onyx.db import regulatory_indexing_jobs as indexing_job_repository
from onyx.db.enums import RegulatoryIndexingItemStatus
from onyx.db.models import (
    RegulatoryChunk,
    RegulatoryIndexingItem,
    RegulatoryIndexingJob,
)
from onyx.natural_language_processing.utils import BaseTokenizer
from onyx.regulatory.chunker import RegulatoryChunk as ChunkerRegulatoryChunk
from onyx.regulatory.chunker import RegulatoryChunker
from onyx.regulatory.indexing import (
    REGULATORY_MAX_CHUNK_CHARS,
    documents_to_regulatory_chunks,
)
from onyx.regulatory.indexing_jobs import preparation
from onyx.regulatory.indexing_jobs.contextual import contextualized_embedding_text
from onyx.regulatory.indexing_jobs.models import (
    RegulatoryIndexingConfigSnapshot,
    RegulatoryInputHashVersion,
    VertexAuthenticationMode,
    VertexBatchConfig,
)
from shared_configs.enums import EmbeddingProvider


class _CharacterTokenizer(BaseTokenizer):
    def encode(self, string: str) -> list[int]:
        return [ord(character) for character in string]

    def tokenize(self, string: str) -> list[str]:
        return list(string)

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(token) for token in tokens)


def _document(user_file_id: UUID, text: str) -> Document:
    return Document(
        id=str(user_file_id),
        source=DocumentSource.USER_FILE,
        semantic_identifier="gumruk-yonetmeligi.md",
        metadata={},
        sections=[TextSection(text=text)],
    )


def _snapshot(
    *, input_content_hash: str = "1" * 64
) -> RegulatoryIndexingConfigSnapshot:
    return RegulatoryIndexingConfigSnapshot(
        input_content_hash=input_content_hash,
        input_hash_version=RegulatoryInputHashVersion.CANONICAL_V2,
        chunk_generation_hash="2" * 64,
        search_settings_id=41,
        embedding_provider=EmbeddingProvider.OPENROUTER,
        embedding_model_name="openai/text-embedding-3-large",
        model_dimension=3072,
        reduced_dimension=1024,
        effective_dimension=1024,
        index_name="danswer_chunk_v2",
        vertex=VertexBatchConfig(
            model_configuration_id=73,
            model_name="gemini-3.1-flash-lite",
            project="customs-prod",
            location="europe-west4",
            authentication_mode=VertexAuthenticationMode.WORKLOAD_IDENTITY,
            gcs_uri="gs://customs-indexing/regulatory",
        ),
        prompt_version="contextual-rag-v1",
        prompt_hash="a" * 64,
    )


def test_versioned_input_hash_preserves_legacy_recovery_and_v2_metadata() -> None:
    user_file_id = uuid4()
    document = _document(user_file_id, "MADDE 1 - Değişmeyen hüküm.")
    first_load = document.model_copy(
        update={
            "doc_updated_at": datetime.datetime(
                2026, 8, 20, 7, 0, tzinfo=datetime.timezone.utc
            )
        }
    )
    recovered_load = document.model_copy(
        update={
            "doc_updated_at": datetime.datetime(
                2026, 8, 20, 7, 5, tzinfo=datetime.timezone.utc
            )
        }
    )
    metadata_changed = recovered_load.model_copy(
        update={"metadata": {"regulation_number": "2026/1"}}
    )

    canonical_hash = preparation.regulatory_documents_content_hash(
        [first_load], RegulatoryInputHashVersion.CANONICAL_V2
    )
    assert (
        preparation.regulatory_documents_content_hash(
            [recovered_load], RegulatoryInputHashVersion.CANONICAL_V2
        )
        == canonical_hash
    )
    assert (
        preparation.regulatory_documents_content_hash(
            [metadata_changed], RegulatoryInputHashVersion.CANONICAL_V2
        )
        != canonical_hash
    )

    legacy_hash = preparation.regulatory_documents_content_hash(
        [first_load], RegulatoryInputHashVersion.LEGACY_V1
    )
    assert (
        preparation.regulatory_documents_content_hash(
            [metadata_changed], RegulatoryInputHashVersion.LEGACY_V1
        )
        == legacy_hash
    )


def test_chunk_row_v3_hash_is_ordered_and_metadata_sensitive() -> None:
    user_file_id = uuid4()
    rows = [
        cast(
            RegulatoryChunk,
            SimpleNamespace(
                id="chunk-b",
                user_file_id=user_file_id,
                position=2,
                text="MADDE 2",
                chunk_type="article",
                heading_path=["BİRİNCİ BÖLÜM", "MADDE 2"],
                chunk_metadata={"article_no": "2"},
                validity_start_date=datetime.date(2026, 1, 1),
                validity_end_date=None,
                status="active",
                source="indexed",
                supersedes_chunk_id=None,
                superseded_by_chunk_id=None,
            ),
        ),
        cast(
            RegulatoryChunk,
            SimpleNamespace(
                id="chunk-a",
                user_file_id=user_file_id,
                position=1,
                text="MADDE 1",
                chunk_type="article",
                heading_path=["BİRİNCİ BÖLÜM", "MADDE 1"],
                chunk_metadata={"article_no": "1"},
                validity_start_date=None,
                validity_end_date=None,
                status="active",
                source="indexed",
                supersedes_chunk_id=None,
                superseded_by_chunk_id=None,
            ),
        ),
    ]

    first = preparation.regulatory_chunks_content_hash(rows)
    assert preparation.regulatory_chunks_content_hash(list(reversed(rows))) == first
    rows[0].chunk_metadata = {"article_no": "2", "version": "changed"}
    assert preparation.regulatory_chunks_content_hash(rows) != first


def test_unresolved_compatibility_hash_resolves_exact_legacy_algorithm() -> None:
    user_file_id = uuid4()
    document = _document(user_file_id, "MADDE 1 - Eski kimlikli hüküm.")
    legacy_hash = preparation.regulatory_documents_content_hash(
        [document], RegulatoryInputHashVersion.LEGACY_V1
    )

    assert (
        preparation.resolve_regulatory_documents_input_hash_version(
            [document],
            persisted_content_hash=legacy_hash,
            declared_version=RegulatoryInputHashVersion.LEGACY_OR_CANONICAL,
        )
        is RegulatoryInputHashVersion.LEGACY_V1
    )


def test_unresolved_compatibility_hash_resolves_exact_canonical_algorithm() -> None:
    user_file_id = uuid4()
    document = _document(user_file_id, "MADDE 1 - Metadata duyarlı hüküm.")
    document = document.model_copy(update={"metadata": {"regulation_number": "2026/7"}})
    canonical_hash = preparation.regulatory_documents_content_hash(
        [document], RegulatoryInputHashVersion.CANONICAL_V2
    )

    assert (
        preparation.resolve_regulatory_documents_input_hash_version(
            [document],
            persisted_content_hash=canonical_hash,
            declared_version=RegulatoryInputHashVersion.LEGACY_OR_CANONICAL,
        )
        is RegulatoryInputHashVersion.CANONICAL_V2
    )


def test_unresolved_compatibility_hash_fails_closed_on_ambiguous_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = _document(uuid4(), "MADDE 1 - Çakışma koruması.")
    persisted_hash = "a" * 64
    monkeypatch.setattr(
        preparation,
        "regulatory_documents_content_hash",
        lambda _documents, _version: persisted_hash,
    )

    with pytest.raises(ValueError, match="uniquely resolve"):
        preparation.resolve_regulatory_documents_input_hash_version(
            [document],
            persisted_content_hash=persisted_hash,
            declared_version=RegulatoryInputHashVersion.LEGACY_OR_CANONICAL,
        )


def test_unresolved_compatibility_hash_fails_closed_when_neither_matches() -> None:
    document = _document(uuid4(), "MADDE 1 - Eşleşmeyen içerik.")

    with pytest.raises(ValueError, match="uniquely resolve"):
        preparation.resolve_regulatory_documents_input_hash_version(
            [document],
            persisted_content_hash="f" * 64,
            declared_version=RegulatoryInputHashVersion.LEGACY_OR_CANONICAL,
        )


def test_new_preparation_snapshots_use_canonical_v2_hashing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user_file_id = uuid4()
    document = _document(user_file_id, "MADDE 1 - Metadata-sensitive hüküm.")
    expected_hash = preparation.regulatory_documents_content_hash(
        [document], RegulatoryInputHashVersion.CANONICAL_V2
    )
    captured_snapshot_kwargs: list[dict[str, object]] = []
    snapshot = _snapshot(input_content_hash=expected_hash)
    job = cast(
        RegulatoryIndexingJob,
        SimpleNamespace(
            id=uuid4(),
            user_file_id=user_file_id,
            content_hash=expected_hash,
            chunk_generation_hash=snapshot.chunk_generation_hash,
            lease_generation=0,
            config_snapshot=snapshot.model_dump(mode="json"),
        ),
    )

    def resolve_snapshot(_session: Session, **kwargs: object) -> object:
        captured_snapshot_kwargs.append(kwargs)
        return snapshot

    monkeypatch.setattr(
        preparation, "resolve_regulatory_indexing_snapshot", resolve_snapshot
    )
    monkeypatch.setattr(
        preparation,
        "create_or_get_regulatory_indexing_job",
        lambda *_args, **_kwargs: job,
    )
    monkeypatch.setattr(
        preparation,
        "claim_regulatory_indexing_job",
        lambda *_args, **_kwargs: False,
    )

    assert (
        preparation.prepare_regulatory_indexing_job(
            user_file_id,
            [document],
            "tenant-a",
            cast(Session, SimpleNamespace()),
        )
        == job.id
    )
    assert captured_snapshot_kwargs == [
        {
            "input_content_hash": expected_hash,
            "input_hash_version": RegulatoryInputHashVersion.CANONICAL_V2,
        }
    ]


def test_duplicate_delivery_does_not_prepare_an_active_older_chunk_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user_file_id = uuid4()
    document = _document(user_file_id, "MADDE 1 - Sürüm geçişi hükmü.")
    content_hash = preparation.regulatory_documents_content_hash(
        [document], RegulatoryInputHashVersion.CANONICAL_V2
    )
    current_snapshot = _snapshot(input_content_hash=content_hash)
    older_snapshot = current_snapshot.model_copy(
        update={"chunk_generation_hash": "3" * 64}
    )
    active_job = cast(
        RegulatoryIndexingJob,
        SimpleNamespace(
            id=uuid4(),
            user_file_id=user_file_id,
            content_hash=content_hash,
            chunk_generation_hash=older_snapshot.chunk_generation_hash,
            lease_generation=4,
            config_snapshot=older_snapshot.model_dump(mode="json"),
        ),
    )

    monkeypatch.setattr(
        preparation,
        "resolve_regulatory_indexing_snapshot",
        lambda *_args, **_kwargs: current_snapshot,
    )
    monkeypatch.setattr(
        preparation,
        "create_or_get_regulatory_indexing_job",
        lambda *_args, **_kwargs: active_job,
    )
    monkeypatch.setattr(
        preparation,
        "claim_regulatory_indexing_job",
        lambda *_args, **_kwargs: pytest.fail(
            "an older active generation must be recovered, not re-prepared"
        ),
    )
    monkeypatch.setattr(
        preparation,
        "documents_to_regulatory_chunks",
        lambda **_kwargs: pytest.fail("canonical chunks must remain untouched"),
    )

    assert (
        preparation.prepare_regulatory_indexing_job(
            user_file_id,
            [document],
            "tenant-a",
            cast(Session, SimpleNamespace()),
        )
        == active_job.id
    )


def test_public_boundary_uses_the_canonical_regulatory_chunker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user_file_id = uuid4()
    text = """GÜMRÜK YÖNETMELİĞİ

BİRİNCİ BÖLÜM

MADDE 12 - (1) Transit rejiminde teminat aranır.

(2) Teminatın kapsamı idarece belirlenir.
"""
    expected = RegulatoryChunker(max_chunk_chars=REGULATORY_MAX_CHUNK_CHARS).chunk_text(
        text,
        source_path=str(user_file_id),
        source_file="gumruk-yonetmeligi.md",
    )
    captured_chunks: list[ChunkerRegulatoryChunk] = []

    def replace_chunks(
        _db_session: Session,
        persisted_user_file_id: UUID,
        chunks: list[ChunkerRegulatoryChunk],
    ) -> list[RegulatoryChunk]:
        assert persisted_user_file_id == user_file_id
        captured_chunks.extend(chunks)
        return cast(
            list[RegulatoryChunk],
            [
                SimpleNamespace(
                    id=f"rc_{index}",
                    position=chunk.metadata.chunk_order,
                    heading_path=list(chunk.metadata.heading_path),
                    validity_start_date=None,
                    validity_end_date=None,
                )
                for index, chunk in enumerate(chunks)
            ],
        )

    monkeypatch.setattr(
        "onyx.regulatory.indexing.replace_indexed_chunks_for_file", replace_chunks
    )
    db_session = MagicMock(spec=Session)

    actual = documents_to_regulatory_chunks(
        documents=[_document(user_file_id, text)],
        db_session=db_session,
        tokenizer=_CharacterTokenizer(),
        enable_contextual_rag=False,
    )

    assert [chunk.content for chunk in actual] == [
        chunk.text for chunk in expected.chunks
    ]
    assert [chunk.heading_path for chunk in actual] == [
        list(chunk.metadata.heading_path) for chunk in expected.chunks
    ]
    assert [chunk.text for chunk in captured_chunks] == [
        chunk.text for chunk in expected.chunks
    ]
    db_session.flush.assert_called_once_with()


def test_public_boundary_aggregates_same_file_documents_before_one_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user_file_id = uuid4()
    replacement_batches: list[list[ChunkerRegulatoryChunk]] = []

    def replace_chunks(
        _db_session: Session,
        persisted_user_file_id: UUID,
        chunks: list[ChunkerRegulatoryChunk],
    ) -> list[RegulatoryChunk]:
        assert persisted_user_file_id == user_file_id
        replacement_batches.append(chunks)
        return cast(
            list[RegulatoryChunk],
            [
                SimpleNamespace(
                    id=f"rc_{index}",
                    position=chunk.metadata.chunk_order,
                    heading_path=list(chunk.metadata.heading_path),
                    validity_start_date=None,
                    validity_end_date=None,
                )
                for index, chunk in enumerate(chunks)
            ],
        )

    monkeypatch.setattr(
        "onyx.regulatory.indexing.replace_indexed_chunks_for_file", replace_chunks
    )
    db_session = MagicMock(spec=Session)

    actual = documents_to_regulatory_chunks(
        documents=[
            _document(user_file_id, "MADDE 1 - Birinci dosya bölümü."),
            _document(user_file_id, "MADDE 2 - İkinci dosya bölümü."),
        ],
        db_session=db_session,
        tokenizer=_CharacterTokenizer(),
        enable_contextual_rag=False,
    )

    assert len(replacement_batches) == 1
    assert [chunk.content for chunk in actual] == [
        row.text for row in replacement_batches[0]
    ]
    assert "Birinci dosya bölümü" in "\n".join(chunk.content for chunk in actual)
    assert "İkinci dosya bölümü" in "\n".join(chunk.content for chunk in actual)


def test_legacy_chunker_keeps_different_user_files_as_separate_replacements(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_file_id = uuid4()
    second_file_id = uuid4()
    replaced_file_ids: list[UUID] = []

    def replace_chunks(
        _db_session: Session,
        user_file_id: UUID,
        chunks: list[ChunkerRegulatoryChunk],
    ) -> list[RegulatoryChunk]:
        replaced_file_ids.append(user_file_id)
        return cast(
            list[RegulatoryChunk],
            [
                SimpleNamespace(
                    id=f"{user_file_id}-{index}",
                    position=chunk.metadata.chunk_order,
                    heading_path=list(chunk.metadata.heading_path),
                    validity_start_date=None,
                    validity_end_date=None,
                )
                for index, chunk in enumerate(chunks)
            ],
        )

    monkeypatch.setattr(
        "onyx.regulatory.indexing.replace_indexed_chunks_for_file", replace_chunks
    )

    documents_to_regulatory_chunks(
        documents=[
            _document(first_file_id, "MADDE 1 - Birinci dosya."),
            _document(second_file_id, "MADDE 1 - İkinci dosya."),
        ],
        db_session=MagicMock(spec=Session),
        tokenizer=_CharacterTokenizer(),
        enable_contextual_rag=False,
    )

    assert replaced_file_ids == [first_file_id, second_file_id]


@pytest.mark.parametrize(
    ("row_count", "row_text"),
    [
        (1, "MADDE 1 - Kısa ve kendi bağlamını taşıyan hüküm."),
        (2, "MADDE 1 - " + "kesintisiz" * 60),
    ],
)
def test_context_ineligible_item_is_skipped_but_keeps_original_embedding_text(
    monkeypatch: pytest.MonkeyPatch,
    row_count: int,
    row_text: str,
) -> None:
    user_file_id = uuid4()
    job_id = uuid4()
    content_hash = preparation.regulatory_documents_content_hash(
        [_document(user_file_id, row_text)],
        RegulatoryInputHashVersion.CANONICAL_V2,
    )
    snapshot = _snapshot(input_content_hash=content_hash)
    job = cast(
        RegulatoryIndexingJob,
        SimpleNamespace(
            id=job_id,
            user_file_id=user_file_id,
            content_hash=content_hash,
            chunk_generation_hash=snapshot.chunk_generation_hash,
            lease_generation=0,
            config_snapshot=snapshot.model_dump(mode="json"),
        ),
    )
    rows = cast(
        list[RegulatoryChunk],
        [
            SimpleNamespace(
                id=f"rc_{index}",
                user_file_id=user_file_id,
                position=index,
                text=row_text if index == 0 else "MADDE 2 - Kısa hüküm.",
                heading_path=[f"MADDE {index + 1}"],
                validity_start_date=None,
                validity_end_date=None,
            )
            for index in range(row_count)
        ],
    )
    prepared_items: list[indexing_job_repository.RegulatoryIndexingPreparedItem] = []

    monkeypatch.setattr(
        preparation,
        "resolve_regulatory_indexing_snapshot",
        lambda _session, **_kwargs: snapshot,
    )
    monkeypatch.setattr(
        preparation,
        "get_tokenizer",
        lambda _model_name, _provider: _CharacterTokenizer(),
    )
    monkeypatch.setattr(
        preparation,
        "create_or_get_regulatory_indexing_job",
        lambda *_args, **_kwargs: job,
    )
    monkeypatch.setattr(
        preparation,
        "claim_regulatory_indexing_job",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        preparation,
        "get_regulatory_indexing_job",
        lambda *_args, **_kwargs: job,
    )
    monkeypatch.setattr(
        preparation, "documents_to_regulatory_chunks", lambda **_kwargs: []
    )
    monkeypatch.setattr(
        preparation,
        "get_chunks_for_file",
        lambda *_args, **_kwargs: rows,
    )

    def persist_preparation(
        _session: Session,
        **kwargs: object,
    ) -> bool:
        callback = cast(
            Callable[[], list[indexing_job_repository.RegulatoryIndexingPreparedItem]],
            kwargs["prepare_items"],
        )
        prepared_items.extend(callback())
        return True

    monkeypatch.setattr(
        indexing_job_repository,
        "persist_regulatory_indexing_preparation",
        persist_preparation,
    )

    preparation.prepare_regulatory_indexing_job(
        user_file_id=user_file_id,
        documents=[_document(user_file_id, row_text)],
        tenant_id="tenant-a",
        db_session=cast(Session, SimpleNamespace()),
    )

    first_spec = next(
        item for item in prepared_items if item.regulatory_chunk_id == rows[0].id
    )
    first_item = cast(
        RegulatoryIndexingItem,
        SimpleNamespace(
            regulatory_chunk_id=first_spec.regulatory_chunk_id,
            status=(
                RegulatoryIndexingItemStatus.SKIPPED.value
                if first_spec.skip_context
                else RegulatoryIndexingItemStatus.PENDING.value
            ),
            context=None,
        ),
    )
    assert first_spec.skip_context
    assert first_item.status == RegulatoryIndexingItemStatus.SKIPPED.value
    assert contextualized_embedding_text(rows[0], first_item) == row_text


def test_duplicate_preparation_does_not_replace_successful_item_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user_file_id = uuid4()
    job_id = uuid4()
    document = _document(user_file_id, "MADDE 1 - Transit hükmü.")
    content_hash = preparation.regulatory_documents_content_hash(
        [document], RegulatoryInputHashVersion.CANONICAL_V2
    )
    snapshot = _snapshot(input_content_hash=content_hash)
    job = cast(
        RegulatoryIndexingJob,
        SimpleNamespace(
            id=job_id,
            user_file_id=user_file_id,
            content_hash=content_hash,
            chunk_generation_hash=snapshot.chunk_generation_hash,
            lease_generation=0,
            config_snapshot=snapshot.model_dump(mode="json"),
        ),
    )
    rows = cast(
        list[RegulatoryChunk],
        [
            SimpleNamespace(
                id=f"rc_{index}",
                user_file_id=user_file_id,
                position=index,
                text=f"MADDE {index + 1} - Transit hükmü.",
                heading_path=[f"MADDE {index + 1}"],
                validity_start_date=None,
                validity_end_date=None,
            )
            for index in range(2)
        ],
    )
    raw_item = SimpleNamespace(
        id=uuid4(),
        job_id=job_id,
        regulatory_chunk_id=rows[0].id,
        request_hash="",
        status=RegulatoryIndexingItemStatus.PENDING.value,
        context=None,
    )
    item = cast(RegulatoryIndexingItem, raw_item)
    claims = iter([True, False])
    replacement_calls = 0

    monkeypatch.setattr(
        preparation,
        "resolve_regulatory_indexing_snapshot",
        lambda _session, **_kwargs: snapshot,
    )
    monkeypatch.setattr(
        preparation,
        "get_tokenizer",
        lambda _model_name, _provider: _CharacterTokenizer(),
    )
    monkeypatch.setattr(
        preparation,
        "create_or_get_regulatory_indexing_job",
        lambda *_args, **_kwargs: job,
    )
    monkeypatch.setattr(
        preparation,
        "claim_regulatory_indexing_job",
        lambda *_args, **_kwargs: next(claims),
    )
    monkeypatch.setattr(
        preparation,
        "get_regulatory_indexing_job",
        lambda *_args, **_kwargs: job,
    )

    def replace_chunks(**_kwargs: object) -> list[object]:
        nonlocal replacement_calls
        replacement_calls += 1
        return []

    monkeypatch.setattr(preparation, "documents_to_regulatory_chunks", replace_chunks)
    monkeypatch.setattr(
        preparation,
        "get_chunks_for_file",
        lambda *_args, **_kwargs: rows,
    )

    monkeypatch.setattr(
        indexing_job_repository,
        "persist_regulatory_indexing_preparation",
        lambda _session, **kwargs: bool(kwargs["prepare_items"]()),
    )
    first_job_id = preparation.prepare_regulatory_indexing_job(
        user_file_id, [document], "tenant-a", cast(Session, SimpleNamespace())
    )
    raw_item.status = RegulatoryIndexingItemStatus.CONTEXT_READY.value
    raw_item.context = {"contextual_text": "Başarıyla üretilen bağlam."}
    second_job_id = preparation.prepare_regulatory_indexing_job(
        user_file_id, [document], "tenant-a", cast(Session, SimpleNamespace())
    )

    assert first_job_id == second_job_id == job_id
    assert replacement_calls == 1
    assert item.status == RegulatoryIndexingItemStatus.CONTEXT_READY.value
    assert item.context == {"contextual_text": "Başarıyla üretilen bağlam."}


def test_claimed_preparation_resumes_without_claiming_again_and_advances_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user_file_id = uuid4()
    job_id = uuid4()
    document = _document(user_file_id, "MADDE 1 - Kurtarılan hazırlık.")
    content_hash = preparation.regulatory_documents_content_hash(
        [document], RegulatoryInputHashVersion.CANONICAL_V2
    )
    snapshot = _snapshot(input_content_hash=content_hash).model_copy(
        update={"input_hash_version": RegulatoryInputHashVersion.LEGACY_OR_CANONICAL}
    )
    job = cast(
        RegulatoryIndexingJob,
        SimpleNamespace(
            id=job_id,
            user_file_id=user_file_id,
            content_hash=content_hash,
            chunk_generation_hash=snapshot.chunk_generation_hash,
            lease_generation=3,
            status="RUNNING",
            stage="PREPARING",
            config_snapshot=snapshot.model_dump(mode="json"),
        ),
    )
    row = cast(
        RegulatoryChunk,
        SimpleNamespace(
            id="rc_recovered",
            user_file_id=user_file_id,
            position=0,
            text="MADDE 1 - Kurtarılan hazırlık.",
            heading_path=["MADDE 1"],
            validity_start_date=None,
            validity_end_date=None,
        ),
    )
    tokenizer_calls: list[tuple[str | None, object]] = []
    atomic_calls: list[tuple[UUID, int, str]] = []

    monkeypatch.setattr(
        preparation,
        "get_regulatory_indexing_job",
        lambda _session, persisted_job_id: job if persisted_job_id == job_id else None,
        raising=False,
    )
    monkeypatch.setattr(
        preparation,
        "claim_regulatory_indexing_job",
        lambda *_args, **_kwargs: pytest.fail("claimed recovery must not claim again"),
    )

    def get_distinct_tokenizer(
        model_name: str | None, provider: object
    ) -> _CharacterTokenizer:
        tokenizer_calls.append((model_name, provider))
        return _CharacterTokenizer()

    monkeypatch.setattr(preparation, "get_tokenizer", get_distinct_tokenizer)
    monkeypatch.setattr(
        preparation, "documents_to_regulatory_chunks", lambda **_kwargs: []
    )
    monkeypatch.setattr(
        preparation,
        "get_chunks_for_file",
        lambda *_args, **_kwargs: [row],
    )

    def persist_atomically(
        _session: Session,
        *,
        job_id: UUID,
        expected_generation: int,
        prepare_items: Callable[
            [], list[indexing_job_repository.RegulatoryIndexingPreparedItem]
        ],
        resolved_input_hash_version: str,
        now: object,
    ) -> bool:
        del now
        atomic_calls.append((job_id, expected_generation, resolved_input_hash_version))
        prepared_items = prepare_items()
        assert len(prepared_items) == 1
        return True

    monkeypatch.setattr(
        indexing_job_repository,
        "persist_regulatory_indexing_preparation",
        persist_atomically,
        raising=False,
    )
    result = preparation.prepare_claimed_regulatory_indexing_job(
        job_id=job_id,
        expected_generation=3,
        documents=[document],
        tenant_id="tenant-a",
        db_session=cast(Session, SimpleNamespace()),
    )

    assert result == job_id
    assert atomic_calls == [(job_id, 3, RegulatoryInputHashVersion.CANONICAL_V2.value)]
    assert tokenizer_calls == [
        (snapshot.embedding_model_name, snapshot.embedding_provider),
    ]
