from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest
from sqlalchemy.orm import Session

from onyx.configs.constants import DocumentSource
from onyx.connectors.models import Document, TextSection
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


def _snapshot() -> RegulatoryIndexingConfigSnapshot:
    return RegulatoryIndexingConfigSnapshot(
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
    job = cast(
        RegulatoryIndexingJob,
        SimpleNamespace(
            id=job_id,
            user_file_id=user_file_id,
            lease_generation=0,
            config_snapshot=_snapshot().model_dump(mode="json"),
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
            )
            for index in range(row_count)
        ],
    )
    items: dict[str, RegulatoryIndexingItem] = {}
    skipped_item_ids: list[UUID] = []

    monkeypatch.setattr(
        preparation,
        "resolve_regulatory_indexing_snapshot",
        lambda _session: _snapshot(),
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
        preparation, "documents_to_regulatory_chunks", lambda **_kwargs: []
    )
    monkeypatch.setattr(
        preparation,
        "get_chunks_for_file",
        lambda *_args, **_kwargs: rows,
    )

    def create_item(*_args: object, **kwargs: object) -> RegulatoryIndexingItem:
        row_id = cast(str, kwargs["regulatory_chunk_id"])
        item = cast(
            RegulatoryIndexingItem,
            SimpleNamespace(
                id=uuid4(),
                job_id=job_id,
                regulatory_chunk_id=row_id,
                request_hash=kwargs["request_hash"],
                status=RegulatoryIndexingItemStatus.PENDING.value,
                context=None,
            ),
        )
        items[row_id] = item
        return item

    monkeypatch.setattr(
        preparation, "create_or_get_regulatory_indexing_item", create_item
    )

    def persist_skipped(*_args: object, **kwargs: object) -> bool:
        item_id = cast(UUID, kwargs["item_id"])
        skipped_item_ids.append(item_id)
        for item in items.values():
            if item.id == item_id:
                item.status = RegulatoryIndexingItemStatus.SKIPPED.value
        return True

    monkeypatch.setattr(
        preparation, "persist_regulatory_indexing_item_skipped", persist_skipped
    )
    monkeypatch.setattr(
        preparation,
        "advance_regulatory_indexing_job",
        lambda *_args, **_kwargs: True,
    )

    preparation.prepare_regulatory_indexing_job(
        user_file_id=user_file_id,
        documents=[_document(user_file_id, row_text)],
        tenant_id="tenant-a",
        db_session=cast(Session, SimpleNamespace()),
    )

    first_item = items[rows[0].id]
    assert first_item.id in skipped_item_ids
    assert first_item.status == RegulatoryIndexingItemStatus.SKIPPED.value
    assert contextualized_embedding_text(rows[0], first_item) == row_text


def test_duplicate_preparation_does_not_replace_successful_item_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user_file_id = uuid4()
    job_id = uuid4()
    job = cast(
        RegulatoryIndexingJob,
        SimpleNamespace(
            id=job_id,
            user_file_id=user_file_id,
            lease_generation=0,
            config_snapshot=_snapshot().model_dump(mode="json"),
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
        lambda _session: _snapshot(),
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

    def create_item(*_args: object, **kwargs: object) -> RegulatoryIndexingItem:
        raw_item.regulatory_chunk_id = kwargs["regulatory_chunk_id"]
        raw_item.request_hash = kwargs["request_hash"]
        return item

    monkeypatch.setattr(
        preparation, "create_or_get_regulatory_indexing_item", create_item
    )
    monkeypatch.setattr(
        preparation,
        "persist_regulatory_indexing_item_skipped",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        preparation,
        "advance_regulatory_indexing_job",
        lambda *_args, **_kwargs: True,
    )
    document = _document(user_file_id, "MADDE 1 - Transit hükmü.")

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
