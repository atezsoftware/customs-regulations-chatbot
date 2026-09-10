from datetime import date
from uuid import uuid4

from onyx.db.enums import RegulatoryIndexingItemStatus
from onyx.db.models import RegulatoryChunk, RegulatoryIndexingItem
from onyx.regulatory.amendments.annexes.publication_representations import _snapshot
from onyx.regulatory.indexing_jobs import publisher
from onyx.regulatory.indexing_jobs.projection_identity import DurableProjectionInput
from tests.unit.onyx.regulatory.indexing_jobs.test_publisher import (
    _snapshot as configuration_snapshot,
)


def test_durable_publication_counts_distinct_dated_projections_for_one_canonical() -> (
    None
):
    job_id, file_id = uuid4(), uuid4()
    row = RegulatoryChunk(
        id=str(uuid4()),
        user_file_id=file_id,
        position=0,
        projection_ordinal=7,
        text="Same legal text",
        chunk_type="article",
        status="active",
        heading_path=[],
        chunk_metadata={},
        source="original",
    )
    items = []
    for ordinal, start, end in [
        (7, None, date(2030, 1, 1)),
        (12, date(2030, 1, 1), None),
    ]:
        item = RegulatoryIndexingItem(
            id=uuid4(),
            job_id=job_id,
            regulatory_chunk_id=row.id,
            status=RegulatoryIndexingItemStatus.EMBEDDED.value,
            request_hash=str(ordinal),
            vector=[float(ordinal), 0.2, 0.3],
        )
        # Persisted projection identity must remain separate from the canonical ID.
        item.projection_id = uuid4()
        item.projection_ordinal = ordinal
        item.effective_start = start
        item.effective_end = end
        item.projection_input = DurableProjectionInput(
            representation=_snapshot(row),
            context_rows=[_snapshot(row)],
            reference_date=start,
        ).model_dump(mode="json")
        items.append(item)
    ordered = publisher._ordered_projection(
        job_id=job_id,
        user_file_id=file_id,
        rows=[row],
        items=items,
        expected_dimension=3,
    )
    assert [(current.id, item.projection_ordinal) for current, item in ordered] == [
        (row.id, 7),
        (row.id, 12),
    ]
    assert [item.vector for _, item in ordered] == [[7.0, 0.2, 0.3], [12.0, 0.2, 0.3]]

    counts = publisher._expected_verification(
        job_id=job_id,
        user_file_id=file_id,
        rows=[row],
        items=items,
        snapshot=configuration_snapshot(),
    )
    assert counts.canonical_chunk_count == 1
    assert counts.embedded_item_count == 2
    verification = publisher._verification_request(
        expected=counts, rows=[row], items=items, hidden=True
    )
    assert [
        (entry.chunk_index, entry.regulatory_chunk_id)
        for entry in verification.expected_chunks
    ] == [(7, row.id), (12, row.id)]


def test_dated_items_share_one_identical_context_transport_request() -> None:
    from onyx.db.models import RegulatoryIndexingJob
    from onyx.regulatory.indexing_jobs.contextual import (
        ContextualRequestFactory,
        build_contextual_requests,
    )
    from tests.unit.onyx.regulatory.indexing_jobs.test_contextual import (
        _CharacterTokenizer,
    )

    job = RegulatoryIndexingJob(
        id=uuid4(),
        user_file_id=uuid4(),
        config_snapshot={
            "vertex": {"model_name": "gemini-3.1-flash-lite"},
        },
    )
    rows = [
        RegulatoryChunk(
            id=str(uuid4()),
            user_file_id=job.user_file_id,
            position=position,
            projection_ordinal=position,
            text=f"Article {position}. Legal text.",
            chunk_type="article",
            status="active",
            heading_path=[],
            chunk_metadata={},
            source="original",
        )
        for position in range(2)
    ]
    tokenizer = _CharacterTokenizer()
    request = ContextualRequestFactory(
        job=job,
        rows=rows,
        embedding_tokenizer=tokenizer,
        contextual_tokenizer=tokenizer,
    ).request(rows[0])
    items = [
        RegulatoryIndexingItem(
            id=uuid4(),
            job_id=job.id,
            regulatory_chunk_id=rows[0].id,
            projection_id=uuid4(),
            projection_ordinal=ordinal,
            projection_input=DurableProjectionInput(
                representation=_snapshot(rows[0]),
                context_rows=[_snapshot(row) for row in rows],
                reference_date=start,
            ).model_dump(mode="json"),
            status="PENDING",
            request_hash=request.request_hash,
        )
        for ordinal, start in [(0, None), (5, date(2030, 1, 1))]
    ]
    requests = build_contextual_requests(
        job,
        rows,
        items,
        embedding_tokenizer=tokenizer,
        contextual_tokenizer=tokenizer,
    )
    assert len(requests) == 1
    assert requests[0].request_hash == request.request_hash
    assert len({item.projection_id for item in items}) == 2


def test_durable_fitted_checkpoint_is_not_relabelled_as_raw_model_output() -> None:
    import pytest

    from onyx.db.models import RegulatoryIndexingJob
    from onyx.regulatory.amendments.annexes.context_dependencies import context_hash
    from onyx.regulatory.contextual import fit_context_fields_to_embedding_budget
    from onyx.regulatory.indexing_jobs.contextual import ContextualRequestFactory
    from onyx.regulatory.indexing_jobs.embedding_receipts import DurableEmbeddingReceipt
    from onyx.regulatory.indexing_jobs.projection_preparation import (
        freeze_durable_item_context,
    )
    from shared_configs.configs import DOC_EMBEDDING_CONTEXT_SIZE
    from tests.unit.onyx.regulatory.indexing_jobs.test_contextual import (
        _CharacterTokenizer,
    )

    job = RegulatoryIndexingJob(
        id=uuid4(),
        user_file_id=uuid4(),
        config_snapshot={"vertex": {"model_name": "gemini-3.1-flash-lite"}},
    )
    rows = [
        RegulatoryChunk(
            id=str(uuid4()),
            user_file_id=job.user_file_id,
            position=position,
            projection_ordinal=position,
            text=f"Article {position}. Legal text.",
            chunk_type="article",
            status="active",
            heading_path=[],
            chunk_metadata={},
            source="original",
        )
        for position in range(2)
    ]
    tokenizer = _CharacterTokenizer()
    factory = ContextualRequestFactory(
        job=job,
        rows=rows,
        embedding_tokenizer=tokenizer,
        contextual_tokenizer=tokenizer,
    )
    request = factory.request(rows[0])
    fitted, _ = fit_context_fields_to_embedding_budget(
        title_prefix="",
        content=rows[0].text,
        metadata_suffix="",
        doc_summary="Retained context",
        chunk_context="",
        tokenizer=tokenizer,
        embedding_token_limit=DOC_EMBEDDING_CONTEXT_SIZE,
    )
    item = RegulatoryIndexingItem(
        id=uuid4(),
        job_id=job.id,
        regulatory_chunk_id=rows[0].id,
        request_hash=request.request_hash,
        status="CONTEXT_READY",
        context={
            "contextual_text": fitted,
            "context_input": factory.request_provenance(rows[0], request),
        },
    )
    text = fitted + rows[0].text
    receipt = DurableEmbeddingReceipt(
        texts=[text],
        configuration={"dimension": 3},
        source_text_sha256=context_hash(text),
    )
    view = freeze_durable_item_context(
        job=job,
        rows=rows,
        row=rows[0],
        item=item,
        embedding_tokenizer=tokenizer,
        contextual_tokenizer=tokenizer,
        receipt=receipt,
    )
    assert view.calls[0].stage == "durable_fitted_checkpoint"
    assert view.calls[0].output == fitted
    assert view.projections[0].embedding_texts == [text]
    assert item.context is not None
    assert "raw_contextual_text" not in item.context
    item.context = dict[str, object](contextual_text=fitted)
    with pytest.raises(ValueError, match="context checkpoint proof"):
        freeze_durable_item_context(
            job=job,
            rows=rows,
            row=rows[0],
            item=item,
            embedding_tokenizer=tokenizer,
            contextual_tokenizer=tokenizer,
            receipt=receipt,
        )
