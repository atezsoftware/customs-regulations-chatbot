"""PRESENT -> FUTURE chunk copy for the reindex port.

Reads a document's existing chunks from the PRESENT Elasticsearch index via the PIT
scan, re-embeds them under the FUTURE model, and writes them to the FUTURE index
create-only (it never overwrites a chunk a live/forward writer already owns, so a
stale backlog write can't clobber a fresher one). The port Celery task drives this per
batch and owns lifecycle (cursor, stall); keeping the Elasticsearch specifics here
keeps them off the generic docprocessing worker.

Contextual-RAG-ON AUGMENTATION re-embeds one document per page — its per-chunk LLM
re-enrichment is the slow, unheartbeated phase, and needs a doc's chunks complete
(they span PIT pages) to rebuild the doc text. A contextual MODEL_ONLY port also
reassembles documents so it can reject provably eligible regulatory chunks whose
stored enrichment is empty before writing them. Other MODEL_ONLY and RAG-off paths
stream PIT pages.
"""

from collections import defaultdict
from collections.abc import Callable, Iterable

from onyx.configs.app_configs import USE_CHUNK_SUMMARY, USE_DOCUMENT_SUMMARY
from onyx.db.models import SearchSettings
from onyx.document_index.elasticsearch.client import ElasticsearchIndexClient
from onyx.document_index.elasticsearch.elasticsearch_document_index import (
    ElasticsearchDocumentIndex,
)
from onyx.document_index.elasticsearch.schema import (
    DocumentChunk,
    DocumentChunkWithoutVectors,
)
from onyx.document_index.factory import build_elasticsearch_document_index
from onyx.indexing.chunker import DEFAULT_CONTEXTUAL_RAG_RESERVED_TOKENS
from onyx.indexing.contextual_settings import (
    effective_contextual_rag_enabled,
    require_contextual_rag_llm,
)
from onyx.indexing.embedder import DefaultIndexingEmbedder, IndexingEmbedder
from onyx.indexing.port_reembed import (
    AugmentationReembedContext,
    ReembedStrategy,
    _bare_contents,
    re_embed_chunks,
    select_reembed_strategy,
)
from onyx.natural_language_processing.utils import BaseTokenizer, get_tokenizer
from onyx.regulatory.contextual import (
    context_reference_date,
    contextual_reserve_for_embedding_text,
    validity_window_contains,
)
from shared_configs.configs import DOC_EMBEDDING_CONTEXT_SIZE

# Cap per bulk write so it can't run long unheartbeated and get a live port stall-failed.
_PORT_WRITE_PAGE_SIZE = 1000


class RegulatoryContextualPortError(RuntimeError):
    """A port would copy regulatory chunks without required contextual evidence."""


def _has_visible_structural_peer(
    target: DocumentChunkWithoutVectors,
    regulatory_chunks: list[DocumentChunkWithoutVectors],
) -> bool:
    """Return whether a source-backed contextual peer is provably available.

    The persisted Elasticsearch projection does not retain the chunker's contextual
    reserve or logical ``position``. Distinct non-empty heading paths are therefore
    the narrow invariant we can prove without guessing: they cannot be two temporal
    versions of the same structural slot, and an unstructured/legacy row is not
    treated as eligible merely because another row shares its document id.
    """

    target_heading = tuple(target.heading_path or ())
    if not target_heading:
        return False
    reference_date = context_reference_date(
        target.validity_start_date,
        target.validity_end_date,
    )
    for candidate in regulatory_chunks:
        if candidate.regulatory_chunk_id == target.regulatory_chunk_id:
            continue
        candidate_heading = tuple(candidate.heading_path or ())
        if not candidate_heading or candidate_heading == target_heading:
            continue
        if validity_window_contains(
            candidate.validity_start_date,
            candidate.validity_end_date,
            reference_date,
        ):
            return True
    return False


def _validate_regulatory_contextual_page(
    stored_chunks: list[DocumentChunkWithoutVectors],
    reembedded_chunks: list[DocumentChunk],
    tokenizer: BaseTokenizer,
) -> None:
    """Fail before writing a provably eligible regulatory chunk with no context.

    A MODEL_ONLY port has no persisted eligibility bit. We reconstruct only what is
    defensible from the complete document page: a temporally visible, structurally
    distinct peer and enough embedding capacity under the same reserve function used
    by regulatory indexing. One-chunk, oversized, and non-structural documents remain
    valid with empty contextual fields.
    """

    if not USE_CHUNK_SUMMARY and not USE_DOCUMENT_SUMMARY:
        return

    regulatory_chunks = [
        chunk for chunk in stored_chunks if chunk.regulatory_chunk_id is not None
    ]
    if len(regulatory_chunks) < 2:
        return

    structurally_eligible_identities = {
        (chunk.document_id, chunk.chunk_index)
        for chunk in regulatory_chunks
        if _has_visible_structural_peer(chunk, regulatory_chunks)
    }
    if not structurally_eligible_identities:
        return

    reembedded_by_identity = {
        (chunk.document_id, chunk.chunk_index): chunk for chunk in reembedded_chunks
    }
    bare_contents = _bare_contents(stored_chunks)
    incomplete_count = 0
    eligible_count = 0
    for stored, bare_content in zip(stored_chunks, bare_contents):
        if (
            stored.document_id,
            stored.chunk_index,
        ) not in structurally_eligible_identities:
            continue
        reserve = contextual_reserve_for_embedding_text(
            bare_content,
            tokenizer=tokenizer,
            embedding_token_limit=stored.max_chunk_size,
            requested_reserve=DEFAULT_CONTEXTUAL_RAG_RESERVED_TOKENS,
        )
        if reserve == 0:
            continue
        eligible_count += 1
        reembedded = reembedded_by_identity.get(
            (stored.document_id, stored.chunk_index)
        )
        if reembedded is None:
            raise RegulatoryContextualPortError(
                "Regulatory contextual port validation could not match a "
                "re-embedded chunk by identity"
            )
        required_context = (
            reembedded.chunk_context if USE_CHUNK_SUMMARY else reembedded.doc_summary
        )
        if not required_context.strip():
            incomplete_count += 1

    if incomplete_count:
        raise RegulatoryContextualPortError(
            "Regulatory contextual port is incomplete: "
            f"{incomplete_count}/{eligible_count} provably eligible chunks "
            "lack required generated context"
        )


def _build_augmentation_ctx(
    future_search_settings: SearchSettings,
) -> AugmentationReembedContext:
    """Prepare the AUGMENTATION inputs while a DB session is available. The FUTURE
    embedding tokenizer is always resolved (reproduces the chunker's metadata-tail skip);
    for FUTURE-RAG-on we also resolve the contextual LLM/tokenizer and the same token
    budgets the chunker uses."""
    future_embedding_tokenizer = get_tokenizer(
        model_name=future_search_settings.model_name,
        provider_type=future_search_settings.provider_type,
    )
    if not effective_contextual_rag_enabled(future_search_settings):
        return AugmentationReembedContext(
            future_enable_contextual_rag=False,
            future_embedding_tokenizer=future_embedding_tokenizer,
        )

    llm = require_contextual_rag_llm(future_search_settings)
    assert llm is not None, "contextual port context built while disabled"
    tokenizer = get_tokenizer(
        model_name=llm.config.model_name,
        provider_type=llm.config.model_provider,
    )
    return AugmentationReembedContext(
        future_enable_contextual_rag=True,
        future_embedding_tokenizer=future_embedding_tokenizer,
        llm=llm,
        tokenizer=tokenizer,
        # The same *2 fudge factor over the chunk size that the indexing
        # pipeline applies to absorb embedder-vs-LLM tokenizer drift.
        chunk_token_limit=DOC_EMBEDDING_CONTEXT_SIZE * 2,
        contextual_rag_reserved_tokens=DEFAULT_CONTEXTUAL_RAG_RESERVED_TOKENS,
    )


def copy_present_chunks_to_future(
    present_client: ElasticsearchIndexClient,
    future_index: ElasticsearchDocumentIndex,
    doc_ids: list[str],
    strategy: ReembedStrategy,
    embedder: IndexingEmbedder,
    present_tokenizer: BaseTokenizer,
    augmentation_ctx: AugmentationReembedContext | None = None,
    require_contextual_regulatory_completeness: bool = False,
    surviving_doc_ids: Callable[[], set[str]] | None = None,
    should_abort: Callable[[], bool] | None = None,
) -> tuple[int, bool]:
    """Port one batch PRESENT -> FUTURE; returns (chunks written, aborted).
    aborted=True means should_abort stopped the copy mid-batch, so the caller must not
    advance its cursor past this partial batch.

    should_abort brackets each re-embed and precedes each write — it aborts a cancelled
    attempt and heartbeats so a slow-but-live port isn't stall-failed. surviving_doc_ids
    drops chunks of docs deleted mid-batch (no resurrection)."""
    pages: Iterable[list[DocumentChunkWithoutVectors]]
    # Contextual RAG-on AUGMENTATION: buffer to reassemble each doc (chunks span PIT pages), then
    # re-embed one doc per page so the unheartbeated per-chunk LLM re-enrichment is bounded
    # to a single doc, not a whole batch (kept under the stall threshold). Others stream PIT pages.
    rag_on_augmentation = (
        strategy is ReembedStrategy.AUGMENTATION
        and augmentation_ctx is not None
        and augmentation_ctx.future_enable_contextual_rag
    )
    # Context validation also needs one complete document at a time. PIT pages
    # can split a document, so reassemble before deciding eligibility; otherwise
    # the first half could evade the guard or be judged against an incomplete
    # structural/temporal peer set.
    validate_model_only_context = (
        strategy is ReembedStrategy.MODEL_ONLY
        and require_contextual_regulatory_completeness
    )
    if rag_on_augmentation or validate_model_only_context:
        by_doc: dict[str, list[DocumentChunkWithoutVectors]] = defaultdict(list)
        for page in present_client.iter_chunks_for_doc_ids(doc_ids):
            for chunk in page:
                by_doc[chunk.document_id].append(chunk)
        pages = list(by_doc.values())
    else:
        pages = present_client.iter_chunks_for_doc_ids(doc_ids)

    chunks_written = 0
    for page_chunks in pages:
        # Heartbeat before re_embed (the longest gap), not just before writes; also
        # skips a needless re_embed on cancel.
        if should_abort is not None and should_abort():
            return chunks_written, True
        reembedded = re_embed_chunks(
            page_chunks,
            strategy,
            embedder,
            augmentation_ctx=augmentation_ctx,
            present_tokenizer=present_tokenizer,
        )
        if not reembedded:
            continue
        if validate_model_only_context:
            _validate_regulatory_contextual_page(
                page_chunks,
                reembedded,
                present_tokenizer,
            )
        # Mark these as port writes so the orphan sweep can delete a resurrected doc
        # (create-only re-add after a concurrent delete) without touching a legitimately
        # re-added one, whose forward-written chunks are unmarked. DocumentChunk is
        # frozen, so rebuild via model_copy rather than mutating.
        reembedded = [
            chunk.model_copy(update={"written_by_port": True}) for chunk in reembedded
        ]
        # Stop writing the instant the attempt is cancelled (e.g. by a deletion).
        if should_abort is not None and should_abort():
            return chunks_written, True
        # Heartbeat before each sub-page write.
        for i in range(0, len(reembedded), _PORT_WRITE_PAGE_SIZE):
            if should_abort is not None and should_abort():
                return chunks_written, True
            sub = reembedded[i : i + _PORT_WRITE_PAGE_SIZE]
            # Drop chunks of docs deleted mid-batch, re-checked immediately before each
            # write (not once per page): a doc's chunks can span several sub-pages, and a
            # doc deleted between writes would otherwise be create-only resurrected.
            if surviving_doc_ids is not None:
                surviving = surviving_doc_ids()
                sub = [c for c in sub if c.document_id in surviving]
                if not sub:
                    continue
            future_index.index_raw_chunks(sub, use_create_only=True)
            chunks_written += len(sub)
    return chunks_written, False


class PortCopier:
    """Resolves the Elasticsearch handles, reembed strategy, and embedder once so
    copy_doc_batch runs with no DB session held. Build it while the search
    settings are session-attached: the FUTURE provider credentials lazy-load,
    and the AUGMENTATION contextual LLM/model-config resolution needs a session.
    """

    def __init__(
        self,
        present_search_settings: SearchSettings,
        future_search_settings: SearchSettings,
    ) -> None:
        self._strategy = select_reembed_strategy(
            present_search_settings, future_search_settings
        )
        self._present_client = ElasticsearchIndexClient(
            index_name=present_search_settings.index_name
        )
        self._future_index = build_elasticsearch_document_index(future_search_settings)
        self._embedder = DefaultIndexingEmbedder.from_db_search_settings(
            future_search_settings
        )
        # The PRESENT model's tokenizer (what indexing used) — MODEL_ONLY needs it to
        # reproduce the metadata-tail skip; the FUTURE embedder's would flip it.
        self._present_tokenizer = get_tokenizer(
            model_name=present_search_settings.model_name,
            provider_type=present_search_settings.provider_type,
        )
        self._augmentation_ctx: AugmentationReembedContext | None = None
        if self._strategy is ReembedStrategy.AUGMENTATION:
            self._augmentation_ctx = _build_augmentation_ctx(future_search_settings)
        self._require_contextual_regulatory_completeness = (
            effective_contextual_rag_enabled(future_search_settings)
        ) and (USE_CHUNK_SUMMARY or USE_DOCUMENT_SUMMARY)

    def delete_port_written(self, document_ids: list[str]) -> int:
        """Delete only the port-written chunks of these docs from the target index —
        used by the orphan sweep to remove a resurrected doc while leaving a
        legitimately re-added one (unmarked chunks) intact. Returns chunks deleted."""
        return self._future_index.delete_port_written_chunks(document_ids)

    def copy_doc_batch(
        self,
        doc_ids: list[str],
        surviving_doc_ids: Callable[[], set[str]] | None = None,
        should_abort: Callable[[], bool] | None = None,
    ) -> tuple[int, bool]:
        return copy_present_chunks_to_future(
            present_client=self._present_client,
            future_index=self._future_index,
            doc_ids=doc_ids,
            strategy=self._strategy,
            embedder=self._embedder,
            present_tokenizer=self._present_tokenizer,
            augmentation_ctx=self._augmentation_ctx,
            require_contextual_regulatory_completeness=(
                self._require_contextual_regulatory_completeness
            ),
            surviving_doc_ids=surviving_doc_ids,
            should_abort=should_abort,
        )
