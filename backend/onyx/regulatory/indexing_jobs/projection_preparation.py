"""Prepare dated durable requests from retained canonical and contextual history."""

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from onyx.db.models import RegulatoryChunk, RegulatoryIndexingItem
    from onyx.regulatory.amendments.annexes.models import PreparedContextView
    from onyx.regulatory.indexing_jobs.embedding_receipts import DurableEmbeddingReceipt
from collections.abc import Mapping
from datetime import date, timedelta
from uuid import uuid5

from onyx.db.models import RegulatoryIndexingJob
from onyx.db.regulatory_indexing_jobs import RegulatoryIndexingPreparedItem
from onyx.db.regulatory_publication import PublicationStore
from onyx.db.regulatory_writer_publication import OwnedWriterInputs
from onyx.document_index.publication_models import FileOwnership
from onyx.natural_language_processing.utils import BaseTokenizer
from onyx.regulatory.amendments.annexes.context_dependencies import (
    context_hash,
    effective_context_rows,
)
from onyx.regulatory.amendments.annexes.publication_representations import _snapshot
from onyx.regulatory.amendments.annexes.staging import canonical_snapshot_rows
from onyx.regulatory.indexing_jobs.contextual import ContextualRequestFactory
from onyx.regulatory.indexing_jobs.projection_identity import DurableProjectionInput
from onyx.regulatory.indexing_jobs.vertex_batch import VertexBatchRequest


def prepare_owned_durable_items(
    *,
    owner: FileOwnership,
    job: RegulatoryIndexingJob,
    inputs: OwnedWriterInputs,
    embedding_tokenizer: BaseTokenizer,
    contextual_tokenizer: BaseTokenizer,
    reference_dates: Mapping[tuple[str, date | None, date | None], date] | None = None,
) -> list[RegulatoryIndexingPreparedItem]:
    authority = PublicationStore(owner.scope)
    if owner.user_file_id != job.user_file_id:
        raise ValueError("durable preparation ownership differs from job")
    prior = [
        binding
        for binding in inputs.bindings
        if binding.index.search_settings_id == job.search_settings_id
    ]
    if not prior:
        current_names = {
            settings.index_name
            for settings in inputs.settings
            if settings.status.is_current()
        }
        prior = [
            binding
            for binding in inputs.bindings
            if binding.index.index_name in current_names
        ]
    boundaries = sorted(
        {
            date.min,
            date.max,
            *(
                value
                for row in inputs.canonical
                for value in (row.validity_start_date, row.validity_end_date)
                if value is not None
            ),
            *(
                value
                for binding in prior
                for value in (binding.effective_start, binding.effective_end)
                if value is not None
            ),
        }
    )
    prepared: list[RegulatoryIndexingPreparedItem] = []
    used: set[int] = set()
    for start, end in zip(boundaries, boundaries[1:]):
        when = (
            start
            if start != date.min
            else end - timedelta(days=1)
            if end != date.max
            else date(2000, 1, 1)
        )
        lower, upper = (
            None if start == date.min else start,
            None if end == date.max else end,
        )
        rows = canonical_snapshot_rows(inputs.canonical)
        by_id = {row.id: row for row in rows}
        matching = {}
        for binding in prior:
            if (
                binding.effective_start is None or binding.effective_start <= when
            ) and (binding.effective_end is None or when < binding.effective_end):
                canonical_id = json.loads(binding.projection.source_json)[
                    "regulatory_chunk_id"
                ]
                row = by_id[canonical_id]
                row.text = binding.representation_text
                row.chunk_metadata = dict(binding.representation_metadata)
                row.position = binding.semantic_position
                matching[canonical_id] = binding
        factory = ContextualRequestFactory(
            job=job,
            rows=rows,
            embedding_tokenizer=embedding_tokenizer,
            contextual_tokenizer=contextual_tokenizer,
            reference_date_override=when,
        )
        frozen_rows = [_snapshot(row) for row in rows]
        for row in effective_context_rows(rows, when):
            reference = (reference_dates or {}).get((row.id, lower, upper), when)
            if not start <= reference < end:
                raise ValueError(
                    "durable reference checkpoint is outside its legal window"
                )
            item_factory = (
                factory
                if reference == when
                else ContextualRequestFactory(
                    job=job,
                    rows=rows,
                    embedding_tokenizer=embedding_tokenizer,
                    contextual_tokenizer=contextual_tokenizer,
                    reference_date_override=reference,
                )
            )
            previous = matching.get(row.id)
            projection_id = uuid5(job.id, f"{row.id}:{lower}:{upper}")
            ordinal = (
                previous.projection.ordinal
                if previous is not None
                else row.projection_ordinal
            )
            if ordinal in used:
                ordinal = authority.allocate(owner, f"durable:{job.id}:{projection_id}")
            used.add(ordinal)
            reserve = item_factory.reserve(row)
            request = (
                item_factory.request(row)
                if reserve
                else VertexBatchRequest(
                    prompt=f"Context skipped for canonical chunk {row.id}"
                )
            )
            prepared.append(
                RegulatoryIndexingPreparedItem(
                    regulatory_chunk_id=row.id,
                    request_hash=request.request_hash,
                    skip_context=reserve == 0,
                    source_snapshot=item_factory.source_snapshot(row),
                    context_input=item_factory.request_provenance(row, request),
                    projection_id=projection_id,
                    projection_ordinal=ordinal,
                    effective_start=lower,
                    effective_end=upper,
                    projection_input=DurableProjectionInput(
                        representation=_snapshot(row),
                        context_rows=frozen_rows,
                        reference_date=reference,
                        canonical_revision_id=inputs.revisions[previous.id]
                        if previous is not None
                        else inputs.canonical_revisions[row.id],
                        canonical_base_sha256=previous.canonical_base_sha256
                        if previous is not None
                        else context_hash(row.text),
                    ),
                )
            )
    return prepared


def freeze_durable_item_context(
    *,
    job: RegulatoryIndexingJob,
    rows: list["RegulatoryChunk"],
    row: "RegulatoryChunk",
    item: "RegulatoryIndexingItem",
    embedding_tokenizer: BaseTokenizer,
    contextual_tokenizer: BaseTokenizer,
    receipt: "DurableEmbeddingReceipt",
) -> "PreparedContextView":
    """Retain verified fitted checkpoints without manufacturing raw model output."""
    from onyx.llm.constants import LlmProviderNames
    from onyx.regulatory.amendments.annexes.context_dependencies import (
        canonical_dependency_ids,
    )
    from onyx.regulatory.amendments.annexes.models import (
        ContextGenerationCall,
        FrozenContextProjection,
        PreparedContextView,
    )
    from onyx.regulatory.contextual import fit_context_fields_to_embedding_budget
    from onyx.regulatory.indexing_jobs.contextual import (
        _contextual_model_name,
        _contextual_safe_input_limit,
        _document_context_utf8_byte_limit,
    )
    from onyx.regulatory.indexing_jobs.projection_identity import projection_input
    from shared_configs.configs import DOC_EMBEDDING_CONTEXT_SIZE

    frozen = projection_input(item)
    if frozen is not None:
        if _snapshot(row) != frozen.representation:
            raise ValueError("durable context checkpoint representation changed")
        rows = canonical_snapshot_rows(frozen.context_rows)
    factory = ContextualRequestFactory(
        job=job,
        rows=rows,
        embedding_tokenizer=embedding_tokenizer,
        contextual_tokenizer=contextual_tokenizer,
        reference_date_override=frozen.reference_date if frozen else None,
    )
    source = factory.source_snapshot(row)
    request = (
        factory.request(row)
        if factory.reserve(row)
        else VertexBatchRequest(prompt=f"Context skipped for canonical chunk {row.id}")
    )
    provenance = factory.request_provenance(row, request)
    checkpoint = item.context or {}
    if (
        checkpoint.get("context_input") != provenance
        or item.request_hash != request.request_hash
    ):
        raise ValueError("durable context checkpoint proof is unavailable")
    calls: list[ContextGenerationCall] = []
    config: dict[str, str | int | float | bool | None] = {
        "model_provider": LlmProviderNames.VERTEX_AI.value,
        "model_name": _contextual_model_name(job),
        "temperature": 0,
        "max_output_tokens": 256,
        "max_input_tokens": _contextual_safe_input_limit(job),
        "utf8_byte_limit": _document_context_utf8_byte_limit(job),
    }
    contextual_text = checkpoint.get("contextual_text")
    if factory.reserve(row):
        if not isinstance(contextual_text, str) or not contextual_text:
            raise ValueError("durable context checkpoint proof has no result")
        raw = checkpoint.get("raw_contextual_text")
        if raw is not None and not isinstance(raw, str):
            raise ValueError("durable raw context checkpoint is malformed")
        output = raw if raw is not None else contextual_text
        fitted, _ = fit_context_fields_to_embedding_budget(
            title_prefix="",
            content=row.text,
            metadata_suffix="",
            doc_summary=output,
            chunk_context="",
            tokenizer=embedding_tokenizer,
            embedding_token_limit=DOC_EMBEDDING_CONTEXT_SIZE,
        )
        if fitted != contextual_text:
            raise ValueError("durable fitted context no longer matches encoder budget")
        if raw is None:
            config["checkpoint_kind"] = "retained_fitted_context"
        config_sha = context_hash(config)
        calls.append(
            ContextGenerationCall(
                request_sha256=context_hash([request.request_hash, config_sha]),
                stage="durable_fitted_checkpoint" if raw is None else "durable_chunk",
                prompt_json=request.model_dump_json(),
                config_sha256=config_sha,
                output=output,
                source_text=request.prompt,
                token_budget=_contextual_safe_input_limit(job),
                tokenizer=f"{type(contextual_tokenizer).__module__}.{type(contextual_tokenizer).__qualname__}",
                generation_path="durable",
            )
        )
    elif contextual_text:
        raise ValueError("ineligible durable context has an unexpected result")
    else:
        contextual_text = ""
    if receipt.source_text_sha256 != context_hash(contextual_text + row.text):
        raise ValueError("durable embedding receipt differs from retained context")
    projection = FrozenContextProjection(
        canonical_chunk_id=row.id,
        canonical_dependency_ids=canonical_dependency_ids(row),
        source_snapshot_sha256=source.sha256,
        generation_path="durable",
        request_hashes=[call.request_sha256 for call in calls],
        embedding_input_sha256=context_hash(receipt.texts),
        embedding_config_sha256=context_hash(receipt.configuration),
        embedding_texts=receipt.texts,
        embedding_config=receipt.configuration,
        canonical_text_sha256=context_hash(row.text),
        metadata_sha256=context_hash(
            [
                row.chunk_metadata,
                row.heading_path,
                row.position,
                row.validity_start_date,
                row.validity_end_date,
            ]
        ),
        doc_summary=contextual_text,
        contextual_config=config,
        validity_start=frozen.reference_date if frozen else row.validity_start_date,
        validity_end=row.validity_end_date,
    )
    return PreparedContextView(
        projections=[projection], snapshots=[source], calls=calls
    )
