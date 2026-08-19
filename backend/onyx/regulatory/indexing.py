"""Bridges the RegulatoryChunker into Onyx's indexing pipeline.

`RegulatoryIndexingChunker` is a drop-in replacement for
`onyx.indexing.chunker.Chunker` used on the user-file indexing path: instead
of token-window sentence chunking, it splits each document along its
regulatory structure (madde / fıkra / bent / table), persists the chunks to
the `regulatory_chunk` Postgres table (source of truth), and emits one
`DocAwareChunk` per row so the existing embedding + Elasticsearch write path is
reused unchanged.

Rows are written in the same DB session/transaction the indexing pipeline
uses, so a pipeline failure rolls back the chunk rows together with the rest
of the indexing bookkeeping.
"""

from collections.abc import Sequence
from uuid import UUID

from chonkie import SentenceChunker
from sqlalchemy.orm import Session

from onyx.configs.app_configs import BLURB_SIZE
from onyx.connectors.models import Document, IndexingDocument
from onyx.db.regulatory_chunks import replace_indexed_chunks_for_file
from onyx.indexing.chunker import DEFAULT_CONTEXTUAL_RAG_RESERVED_TOKENS
from onyx.indexing.chunking import extract_blurb
from onyx.indexing.indexing_heartbeat import IndexingHeartbeatInterface
from onyx.indexing.models import DocAwareChunk
from onyx.natural_language_processing.utils import BaseTokenizer
from onyx.regulatory.chunker import ChunkedDocument, RegulatoryChunker
from onyx.regulatory.contextual import (
    MIN_CONTEXTUAL_RAG_RESERVED_TOKENS,
    contextual_reserve_for_embedding_text,
)
from onyx.utils.logger import setup_logger
from shared_configs.configs import DOC_EMBEDDING_CONTEXT_SIZE

logger = setup_logger()

# Character budget per chunk. The chunker splits oversized articles/tables
# itself, so this is a hard cap, not a target.
REGULATORY_MAX_CHUNK_CHARS = 2400
REGULATORY_CONTEXTUAL_INITIAL_MAX_CHUNK_CHARS = 1200
REGULATORY_CONTEXTUAL_MIN_MAX_CHUNK_CHARS = 500


def _as_indexing_document(document: Document) -> IndexingDocument:
    if isinstance(document, IndexingDocument):
        return document
    return IndexingDocument(
        **document.model_dump(),
        processed_sections=[section.model_copy() for section in document.sections],
    )


def document_text_for_regulatory_indexing(document: Document) -> str:
    """Return the exact ordered text consumed by the regulatory chunker."""

    indexing_document = _as_indexing_document(document)
    parts: list[str] = []
    sections = indexing_document.processed_sections or indexing_document.sections
    for section in sections:
        section_text = getattr(section, "text", None)
        if section_text and section_text.strip():
            parts.append(section_text.strip())
    return "\n\n".join(parts)


def documents_to_regulatory_chunks(
    documents: Sequence[Document],
    db_session: Session,
    tokenizer: BaseTokenizer,
    *,
    callback: IndexingHeartbeatInterface | None = None,
    enable_contextual_rag: bool = True,
) -> list[DocAwareChunk]:
    """Persist canonical chunks through the maintained indexing chunker boundary."""

    indexing_documents = [_as_indexing_document(document) for document in documents]
    return RegulatoryIndexingChunker(
        db_session=db_session,
        tokenizer=tokenizer,
        callback=callback,
        enable_contextual_rag=enable_contextual_rag,
    ).chunk(indexing_documents)


class RegulatoryIndexingChunker:
    """Structure-aware chunker for the user-file indexing pipeline.

    Interface-compatible with `onyx.indexing.chunker.Chunker` for the
    attributes the pipeline actually uses: `chunk()` and `chunk_token_limit`.
    """

    def __init__(
        self,
        db_session: Session,
        tokenizer: BaseTokenizer,
        callback: IndexingHeartbeatInterface | None = None,
        enable_contextual_rag: bool = False,
    ) -> None:
        self.db_session = db_session
        self.chunk_token_limit = DOC_EMBEDDING_CONTEXT_SIZE
        self.callback = callback
        self.tokenizer = tokenizer
        self.enable_contextual_rag = enable_contextual_rag
        self._regulatory_chunker = RegulatoryChunker(
            max_chunk_chars=(
                REGULATORY_CONTEXTUAL_INITIAL_MAX_CHUNK_CHARS
                if enable_contextual_rag
                else REGULATORY_MAX_CHUNK_CHARS
            )
        )

        def token_counter(text: str) -> int:
            return len(tokenizer.encode(text))

        self._blurb_splitter = SentenceChunker(
            tokenizer_or_token_counter=token_counter,
            chunk_size=BLURB_SIZE,
            chunk_overlap=0,
            return_type="texts",
        )

    def chunk(self, documents: list[IndexingDocument]) -> list[DocAwareChunk]:
        final_chunks: list[DocAwareChunk] = []
        for document in documents:
            if self.callback and self.callback.should_stop():
                raise RuntimeError("RegulatoryIndexingChunker: Stop signal detected")

            chunks = self._chunk_single_document(document)
            final_chunks.extend(chunks)

            if self.callback:
                self.callback.progress("RegulatoryIndexingChunker.chunk", len(chunks))
        return final_chunks

    def _chunk_single_document(self, document: IndexingDocument) -> list[DocAwareChunk]:
        # User-file documents are stamped with the user_file UUID as their id
        # (see _load_user_file_documents).
        user_file_id = UUID(document.id)

        text = document_text_for_regulatory_indexing(document)
        source_file = document.semantic_identifier or document.id
        chunked, contextual_rag_reserved_tokens = self._chunk_with_embedding_budget(
            text,
            source_path=str(user_file_id),
            source_file=source_file,
        )
        for warning in chunked.warnings:
            logger.warning(
                "RegulatoryChunker warning for user_file=%s: %s", user_file_id, warning
            )

        rows = replace_indexed_chunks_for_file(
            self.db_session, user_file_id, chunked.chunks
        )
        # Assign ids before the session commits so the emitted DocAwareChunks
        # can reference them.
        self.db_session.flush()

        doc_chunks: list[DocAwareChunk] = []
        for row, chunk, reserved_tokens in zip(
            rows,
            chunked.chunks,
            contextual_rag_reserved_tokens,
        ):
            doc_chunks.append(
                DocAwareChunk(
                    source_document=document,
                    chunk_id=row.position,
                    blurb=extract_blurb(chunk.text, self._blurb_splitter),
                    content=chunk.text,
                    source_links={0: ""},
                    image_file_id=None,
                    section_continuation=False,
                    title_prefix="",
                    metadata_suffix_semantic="",
                    metadata_suffix_keyword="",
                    mini_chunk_texts=None,
                    large_chunk_id=None,
                    doc_summary="",
                    chunk_context="",
                    contextual_rag_reserved_tokens=reserved_tokens,
                    regulatory_chunk_id=row.id,
                    heading_path=list(row.heading_path),
                    validity_start_date=row.validity_start_date,
                    validity_end_date=row.validity_end_date,
                )
            )
        return doc_chunks

    def _chunk_with_embedding_budget(
        self,
        text: str,
        *,
        source_path: str,
        source_file: str,
    ) -> tuple[ChunkedDocument, list[int]]:
        """Keep legal text intact while reserving room for contextual summaries."""

        max_chunk_chars = self._regulatory_chunker.max_chunk_chars
        while True:
            chunked = RegulatoryChunker(max_chunk_chars=max_chunk_chars).chunk_text(
                text,
                source_path=source_path,
                source_file=source_file,
            )
            if not self.enable_contextual_rag or not chunked.chunks:
                return chunked, [0] * len(chunked.chunks)

            largest_chunk = max(
                chunked.chunks,
                key=lambda chunk: len(self.tokenizer.encode(chunk.text)),
            )
            largest_chunk_tokens = len(self.tokenizer.encode(largest_chunk.text))
            # A one-chunk document already contains its own complete context.
            if (
                len(chunked.chunks) == 1
                and largest_chunk_tokens <= self.chunk_token_limit
            ):
                return chunked, [0]

            content_budget = (
                self.chunk_token_limit - DEFAULT_CONTEXTUAL_RAG_RESERVED_TOKENS
            )
            if largest_chunk_tokens <= content_budget:
                return chunked, [DEFAULT_CONTEXTUAL_RAG_RESERVED_TOKENS] * len(
                    chunked.chunks
                )

            if max_chunk_chars <= REGULATORY_CONTEXTUAL_MIN_MAX_CHUNK_CHARS:
                reserves = [
                    contextual_reserve_for_embedding_text(
                        item.text,
                        tokenizer=self.tokenizer,
                        embedding_token_limit=self.chunk_token_limit,
                        requested_reserve=DEFAULT_CONTEXTUAL_RAG_RESERVED_TOKENS,
                    )
                    for item in chunked.chunks
                ]
                eligible_count = sum(reserve > 0 for reserve in reserves)
                logger.warning(
                    "Regulatory contextual eligibility for user_file=%s at the "
                    "minimum structural chunk size: eligible=%d ineligible=%d "
                    "largest_content_tokens=%d embedding_limit=%d "
                    "minimum_useful_reserve=%d",
                    source_path,
                    eligible_count,
                    len(reserves) - eligible_count,
                    largest_chunk_tokens,
                    self.chunk_token_limit,
                    MIN_CONTEXTUAL_RAG_RESERVED_TOKENS,
                )
                return chunked, reserves

            scaled_limit = int(
                max_chunk_chars * content_budget / largest_chunk_tokens * 0.9
            )
            max_chunk_chars = max(
                REGULATORY_CONTEXTUAL_MIN_MAX_CHUNK_CHARS,
                min(max_chunk_chars - 1, scaled_limit),
            )
