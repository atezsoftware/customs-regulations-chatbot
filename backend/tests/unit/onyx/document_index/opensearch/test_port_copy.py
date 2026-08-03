"""Unit coverage for the reindex-port copier's mid-batch deletion guard.

`copy_present_chunks_to_future` re-checks document existence right before the
create-only write (after the slow re-embed) so a doc deleted while the batch was
being read/embedded is not resurrected into the FUTURE index.
"""

import datetime
from typing import cast
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from onyx.configs.constants import DocumentSource
from onyx.document_index.interfaces_new import TenantState
from onyx.document_index.opensearch.port_copy import (
    RegulatoryContextualPortError,
    copy_present_chunks_to_future,
)
from onyx.document_index.opensearch.schema import (
    DocumentChunk,
    DocumentChunkWithoutVectors,
)
from onyx.indexing.port_reembed import ReembedStrategy
from onyx.natural_language_processing.utils import BaseTokenizer
from shared_configs.configs import POSTGRES_DEFAULT_SCHEMA


class _Chunk(BaseModel):
    # Frozen like the real DocumentChunk, so the copier must mark via model_copy.
    model_config = {"frozen": True}
    document_id: str
    written_by_port: bool | None = None


def _chunk(doc_id: str) -> _Chunk:
    return _Chunk(document_id=doc_id)


def _passthrough_reembed(page: list, *_: object, **__: object) -> list:
    # re-embed stub: return the page unchanged (preserves document_ids).
    return list(page)


class _CharTokenizer:
    def encode(self, text: str) -> list[int]:
        return [ord(character) for character in text]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(token) for token in tokens)


_TOKENIZER = cast(BaseTokenizer, _CharTokenizer())


def _regulatory_chunk(
    *,
    chunk_index: int,
    heading_path: list[str] | None,
    content: str = "Short legal provision",
    max_chunk_size: int = 512,
    doc_summary: str = "",
    chunk_context: str = "",
    regulatory_chunk_id: str | None = None,
    validity_start_date: datetime.datetime | None = None,
    validity_end_date: datetime.datetime | None = None,
) -> DocumentChunkWithoutVectors:
    return DocumentChunkWithoutVectors(
        document_id="regulatory-doc",
        chunk_index=chunk_index,
        max_chunk_size=max_chunk_size,
        title=None,
        content=content,
        source_type=DocumentSource.USER_FILE.value,
        public=True,
        access_control_list=[],
        global_boost=0,
        semantic_identifier="Regulation",
        blurb="Legal provision",
        doc_summary=doc_summary,
        chunk_context=chunk_context,
        regulatory_chunk_id=(
            regulatory_chunk_id
            if regulatory_chunk_id is not None
            else f"regulatory-{chunk_index}"
        ),
        heading_path=heading_path,
        validity_start_date=validity_start_date,
        validity_end_date=validity_end_date,
        tenant_id=TenantState(
            tenant_id=POSTGRES_DEFAULT_SCHEMA,
            multitenant=False,
        ),
    )


def _embedded_passthrough(
    page: list[DocumentChunkWithoutVectors],
    *_: object,
    **__: object,
) -> list[DocumentChunk]:
    return [
        DocumentChunk(
            **chunk.model_dump(),
            content_vector=[0.1, 0.2],
            title_vector=None,
        )
        for chunk in page
    ]


@patch("onyx.document_index.opensearch.port_copy.re_embed_chunks")
def test_copier_drops_docs_deleted_mid_batch(mock_reembed: MagicMock) -> None:
    present_client = MagicMock()
    present_client.iter_chunks_for_doc_ids.return_value = [
        [_chunk("doc_a"), _chunk("doc_b")]
    ]
    mock_reembed.side_effect = _passthrough_reembed
    future_index = MagicMock()

    written, aborted = copy_present_chunks_to_future(
        present_client=present_client,
        future_index=future_index,
        doc_ids=["doc_a", "doc_b"],
        strategy=ReembedStrategy.MODEL_ONLY,
        embedder=MagicMock(),
        present_tokenizer=MagicMock(),
        surviving_doc_ids=lambda: {"doc_a"},  # doc_b deleted mid-batch
    )

    assert written == 1
    assert aborted is False
    (written_chunks,), _ = future_index.index_raw_chunks.call_args
    assert [c.document_id for c in written_chunks] == ["doc_a"]


@patch("onyx.document_index.opensearch.port_copy.re_embed_chunks")
def test_copier_skips_write_when_whole_batch_deleted(mock_reembed: MagicMock) -> None:
    present_client = MagicMock()
    present_client.iter_chunks_for_doc_ids.return_value = [[_chunk("doc_a")]]
    mock_reembed.side_effect = _passthrough_reembed
    future_index = MagicMock()

    written, aborted = copy_present_chunks_to_future(
        present_client=present_client,
        future_index=future_index,
        doc_ids=["doc_a"],
        strategy=ReembedStrategy.MODEL_ONLY,
        embedder=MagicMock(),
        present_tokenizer=MagicMock(),
        surviving_doc_ids=lambda: set(),  # everything deleted
    )

    assert written == 0
    assert aborted is False
    future_index.index_raw_chunks.assert_not_called()


@patch("onyx.document_index.opensearch.port_copy._PORT_WRITE_PAGE_SIZE", 2)
@patch("onyx.document_index.opensearch.port_copy.re_embed_chunks")
def test_copier_rechecks_survival_before_each_sub_page(mock_reembed: MagicMock) -> None:
    """Survival is re-checked before EACH sub-page write, not once per page: a doc whose
    chunks span several sub-pages and is deleted between writes must not have its later
    sub-pages create-only resurrected into FUTURE."""
    present_client = MagicMock()
    # one page of 4 doc_a chunks -> 2 sub-pages at the patched _PORT_WRITE_PAGE_SIZE=2
    present_client.iter_chunks_for_doc_ids.return_value = [
        [_chunk("doc_a"), _chunk("doc_a"), _chunk("doc_a"), _chunk("doc_a")]
    ]
    mock_reembed.side_effect = _passthrough_reembed
    future_index = MagicMock()
    # survives the first sub-page's check, deleted before the second
    surviving = MagicMock(side_effect=[{"doc_a"}, set()])

    written, aborted = copy_present_chunks_to_future(
        present_client=present_client,
        future_index=future_index,
        doc_ids=["doc_a"],
        strategy=ReembedStrategy.MODEL_ONLY,
        embedder=MagicMock(),
        present_tokenizer=MagicMock(),
        surviving_doc_ids=surviving,
    )

    assert aborted is False
    assert written == 2  # only the first sub-page; the second dropped post-deletion
    future_index.index_raw_chunks.assert_called_once()
    assert surviving.call_count == 2  # re-checked per sub-page, not once for the page


@patch("onyx.document_index.opensearch.port_copy.re_embed_chunks")
def test_copier_aborts_write_when_cancelled_mid_batch(mock_reembed: MagicMock) -> None:
    # Two pages; the attempt is cancelled after the first page is written.
    present_client = MagicMock()
    present_client.iter_chunks_for_doc_ids.return_value = [
        [_chunk("doc_a")],
        [_chunk("doc_b")],
    ]
    mock_reembed.side_effect = _passthrough_reembed
    future_index = MagicMock()

    # should_abort is polled three times per page (loop-top heartbeat, post-re-embed,
    # and before the sub-page write): allow all of page 1's polls, then cancel at
    # page 2's loop-top poll.
    aborts = iter([False, False, False, True])

    written, aborted = copy_present_chunks_to_future(
        present_client=present_client,
        future_index=future_index,
        doc_ids=["doc_a", "doc_b"],
        strategy=ReembedStrategy.MODEL_ONLY,
        embedder=MagicMock(),
        present_tokenizer=MagicMock(),
        should_abort=lambda: next(aborts),
    )

    # only the first page was written; the second is skipped by the abort.
    assert written == 1
    assert aborted is True
    future_index.index_raw_chunks.assert_called_once()
    (written_chunks,), _ = future_index.index_raw_chunks.call_args
    assert [c.document_id for c in written_chunks] == ["doc_a"]


def _aug_ctx(rag_on: bool) -> MagicMock:
    ctx = MagicMock()
    ctx.future_enable_contextual_rag = rag_on
    return ctx


@patch("onyx.document_index.opensearch.port_copy.re_embed_chunks")
def test_rag_on_augmentation_reembeds_one_page_per_document(
    mock_reembed: MagicMock,
) -> None:
    # RAG-on AUGMENTATION re-embeds one doc per page (bounds the unheartbeated LLM
    # phase); a doc's chunks can span PIT pages, so they're reassembled first.
    present_client = MagicMock()
    present_client.iter_chunks_for_doc_ids.return_value = [
        [_chunk("doc_a"), _chunk("doc_b")],
        [_chunk("doc_a")],  # doc_a's chunks span two PIT pages
    ]
    future_index = MagicMock()

    events: list[tuple] = []

    def _reembed(page: list, *_: object, **__: object) -> list:
        events.append(("reembed", sorted({c.document_id for c in page})))
        return list(page)

    def _should_abort() -> bool:
        events.append(("heartbeat",))
        return False

    mock_reembed.side_effect = _reembed

    written, aborted = copy_present_chunks_to_future(
        present_client=present_client,
        future_index=future_index,
        doc_ids=["doc_a", "doc_b"],
        strategy=ReembedStrategy.AUGMENTATION,
        embedder=MagicMock(),
        present_tokenizer=MagicMock(),
        augmentation_ctx=_aug_ctx(rag_on=True),
        should_abort=_should_abort,
    )

    assert written == 3
    assert aborted is False

    # One re_embed per document (not per batch), each with all the doc's chunks.
    reembed_calls = [e for e in events if e[0] == "reembed"]
    assert [e[1] for e in reembed_calls] == [["doc_a"], ["doc_b"]]

    # A heartbeat precedes each re_embed, bracketing the slow phase.
    for i, e in enumerate(events):
        if e[0] == "reembed":
            assert events[i - 1] == ("heartbeat",)


@patch("onyx.document_index.opensearch.port_copy.re_embed_chunks")
def test_rag_off_augmentation_streams_per_pit_page(mock_reembed: MagicMock) -> None:
    # RAG-off AUGMENTATION has no LLM step, so it streams PIT pages as-is rather than
    # paying the per-document buffering cost.
    present_client = MagicMock()
    present_client.iter_chunks_for_doc_ids.return_value = [
        [_chunk("doc_a"), _chunk("doc_b")],
        [_chunk("doc_a")],  # doc_a spans PIT pages; not reassembled when RAG is off
    ]
    mock_reembed.side_effect = _passthrough_reembed
    future_index = MagicMock()

    written, aborted = copy_present_chunks_to_future(
        present_client=present_client,
        future_index=future_index,
        doc_ids=["doc_a", "doc_b"],
        strategy=ReembedStrategy.AUGMENTATION,
        embedder=MagicMock(),
        present_tokenizer=MagicMock(),
        augmentation_ctx=_aug_ctx(rag_on=False),
    )

    assert written == 3
    assert aborted is False
    # Streamed one re_embed per PIT page (not reassembled per document).
    reembed_pages = [call.args[0] for call in mock_reembed.call_args_list]
    doc_sets = [sorted({c.document_id for c in page}) for page in reembed_pages]
    assert doc_sets == [["doc_a", "doc_b"], ["doc_a"]]


@patch("onyx.document_index.opensearch.port_copy.re_embed_chunks")
def test_copier_writes_all_without_filter(mock_reembed: MagicMock) -> None:
    present_client = MagicMock()
    present_client.iter_chunks_for_doc_ids.return_value = [
        [_chunk("doc_a"), _chunk("doc_b")]
    ]
    mock_reembed.side_effect = _passthrough_reembed
    future_index = MagicMock()

    written, aborted = copy_present_chunks_to_future(
        present_client=present_client,
        future_index=future_index,
        doc_ids=["doc_a", "doc_b"],
        strategy=ReembedStrategy.MODEL_ONLY,
        embedder=MagicMock(),
        present_tokenizer=MagicMock(),
    )

    assert written == 2
    assert aborted is False
    future_index.index_raw_chunks.assert_called_once()


@patch("onyx.document_index.opensearch.port_copy.USE_DOCUMENT_SUMMARY", False)
@patch("onyx.document_index.opensearch.port_copy.USE_CHUNK_SUMMARY", True)
@patch("onyx.document_index.opensearch.port_copy.re_embed_chunks")
def test_contextual_guard_reassembles_split_document_and_fails_before_write(
    mock_reembed: MagicMock,
) -> None:
    first = _regulatory_chunk(chunk_index=0, heading_path=["MADDE 1"])
    second = _regulatory_chunk(chunk_index=1, heading_path=["MADDE 2"])
    present_client = MagicMock()
    present_client.iter_chunks_for_doc_ids.return_value = [[first], [second]]
    mock_reembed.side_effect = _embedded_passthrough
    future_index = MagicMock()

    with pytest.raises(RegulatoryContextualPortError, match="2/2"):
        copy_present_chunks_to_future(
            present_client=present_client,
            future_index=future_index,
            doc_ids=["regulatory-doc"],
            strategy=ReembedStrategy.MODEL_ONLY,
            embedder=MagicMock(),
            present_tokenizer=_TOKENIZER,
            require_contextual_regulatory_completeness=True,
        )

    reembedded_page = mock_reembed.call_args.args[0]
    assert [chunk.chunk_index for chunk in reembedded_page] == [0, 1]
    future_index.index_raw_chunks.assert_not_called()


@pytest.mark.parametrize(
    "chunks",
    [
        # A one-chunk document has no additional document context to generate.
        [_regulatory_chunk(chunk_index=0, heading_path=["MADDE 1"])],
        # Distinct structural chunks with too little embedding capacity are not
        # eligible for contextual output.
        [
            _regulatory_chunk(
                chunk_index=0,
                heading_path=["MADDE 1"],
                content="x" * 30,
                max_chunk_size=40,
            ),
            _regulatory_chunk(
                chunk_index=1,
                heading_path=["MADDE 2"],
                content="x" * 30,
                max_chunk_size=40,
            ),
        ],
        # Legacy regulatory rows without structural provenance are not guessed
        # eligible solely from a shared document id.
        [
            _regulatory_chunk(chunk_index=0, heading_path=None),
            _regulatory_chunk(chunk_index=1, heading_path=None),
        ],
        # Non-overlapping legal versions are never peers in the same temporal
        # snapshot and therefore do not make one another context-eligible.
        [
            _regulatory_chunk(
                chunk_index=0,
                heading_path=["MADDE 1"],
                validity_end_date=datetime.datetime(
                    2020, 1, 1, tzinfo=datetime.timezone.utc
                ),
            ),
            _regulatory_chunk(
                chunk_index=1,
                heading_path=["MADDE 2"],
                validity_start_date=datetime.datetime(
                    2020, 1, 1, tzinfo=datetime.timezone.utc
                ),
            ),
        ],
        # Ordinary chunks remain outside the regulatory guard.
        [
            _regulatory_chunk(
                chunk_index=0,
                heading_path=None,
                regulatory_chunk_id=None,
            ),
            _regulatory_chunk(
                chunk_index=1,
                heading_path=None,
                regulatory_chunk_id=None,
            ),
        ],
    ],
    ids=["one_chunk", "oversized", "nonstructural", "temporal_versions", "ordinary"],
)
@patch("onyx.document_index.opensearch.port_copy.USE_DOCUMENT_SUMMARY", False)
@patch("onyx.document_index.opensearch.port_copy.USE_CHUNK_SUMMARY", True)
@patch("onyx.document_index.opensearch.port_copy.re_embed_chunks")
def test_contextual_guard_preserves_legitimate_empty_context(
    mock_reembed: MagicMock,
    chunks: list[DocumentChunkWithoutVectors],
) -> None:
    present_client = MagicMock()
    present_client.iter_chunks_for_doc_ids.return_value = [chunks]
    mock_reembed.side_effect = _embedded_passthrough
    future_index = MagicMock()

    written, aborted = copy_present_chunks_to_future(
        present_client=present_client,
        future_index=future_index,
        doc_ids=["regulatory-doc"],
        strategy=ReembedStrategy.MODEL_ONLY,
        embedder=MagicMock(),
        present_tokenizer=_TOKENIZER,
        require_contextual_regulatory_completeness=True,
    )

    assert written == len(chunks)
    assert aborted is False
    future_index.index_raw_chunks.assert_called_once()


@patch("onyx.document_index.opensearch.port_copy.USE_DOCUMENT_SUMMARY", False)
@patch("onyx.document_index.opensearch.port_copy.USE_CHUNK_SUMMARY", True)
@patch("onyx.document_index.opensearch.port_copy.re_embed_chunks")
def test_contextual_guard_accepts_generated_chunk_context(
    mock_reembed: MagicMock,
) -> None:
    context = "\n\nGenerated source context"
    chunks = [
        _regulatory_chunk(
            chunk_index=0,
            heading_path=["MADDE 1"],
            content=f"First provision{context}",
            chunk_context=context,
        ),
        _regulatory_chunk(
            chunk_index=1,
            heading_path=["MADDE 2"],
            content=f"Second provision{context}",
            chunk_context=context,
        ),
    ]
    present_client = MagicMock()
    present_client.iter_chunks_for_doc_ids.return_value = [chunks]
    mock_reembed.side_effect = _embedded_passthrough
    future_index = MagicMock()

    written, aborted = copy_present_chunks_to_future(
        present_client=present_client,
        future_index=future_index,
        doc_ids=["regulatory-doc"],
        strategy=ReembedStrategy.MODEL_ONLY,
        embedder=MagicMock(),
        present_tokenizer=_TOKENIZER,
        require_contextual_regulatory_completeness=True,
    )

    assert written == 2
    assert aborted is False
    future_index.index_raw_chunks.assert_called_once()


@patch("onyx.document_index.opensearch.port_copy.USE_DOCUMENT_SUMMARY", True)
@patch("onyx.document_index.opensearch.port_copy.USE_CHUNK_SUMMARY", False)
@patch("onyx.document_index.opensearch.port_copy.re_embed_chunks")
def test_contextual_guard_uses_configured_document_summary_field(
    mock_reembed: MagicMock,
) -> None:
    chunks = [
        _regulatory_chunk(chunk_index=0, heading_path=["MADDE 1"]),
        _regulatory_chunk(chunk_index=1, heading_path=["MADDE 2"]),
    ]
    present_client = MagicMock()
    present_client.iter_chunks_for_doc_ids.return_value = [chunks]
    mock_reembed.side_effect = _embedded_passthrough
    future_index = MagicMock()

    with pytest.raises(RegulatoryContextualPortError, match="2/2"):
        copy_present_chunks_to_future(
            present_client=present_client,
            future_index=future_index,
            doc_ids=["regulatory-doc"],
            strategy=ReembedStrategy.MODEL_ONLY,
            embedder=MagicMock(),
            present_tokenizer=_TOKENIZER,
            require_contextual_regulatory_completeness=True,
        )

    future_index.index_raw_chunks.assert_not_called()


@patch("onyx.document_index.opensearch.port_copy.USE_DOCUMENT_SUMMARY", False)
@patch("onyx.document_index.opensearch.port_copy.USE_CHUNK_SUMMARY", True)
@patch("onyx.document_index.opensearch.port_copy.re_embed_chunks")
def test_augmentation_uses_its_exact_future_eligibility_guard_only(
    mock_reembed: MagicMock,
) -> None:
    """The MODEL_ONLY inference must not second-guess AUGMENTATION.

    AUGMENTATION validates its generated regulatory context inside
    ``re_embed_chunks`` using the FUTURE tokenizer/reserve. This copier-level
    guard intentionally leaves that result alone even when its source fields are
    empty in this stub.
    """

    chunks = [
        _regulatory_chunk(chunk_index=0, heading_path=["MADDE 1"]),
        _regulatory_chunk(chunk_index=1, heading_path=["MADDE 2"]),
    ]
    present_client = MagicMock()
    present_client.iter_chunks_for_doc_ids.return_value = [[chunks[0]], [chunks[1]]]
    mock_reembed.side_effect = _embedded_passthrough
    future_index = MagicMock()

    written, aborted = copy_present_chunks_to_future(
        present_client=present_client,
        future_index=future_index,
        doc_ids=["regulatory-doc"],
        strategy=ReembedStrategy.AUGMENTATION,
        embedder=MagicMock(),
        present_tokenizer=_TOKENIZER,
        augmentation_ctx=_aug_ctx(rag_on=True),
        require_contextual_regulatory_completeness=True,
    )

    assert written == 2
    assert aborted is False
    future_index.index_raw_chunks.assert_called_once()
