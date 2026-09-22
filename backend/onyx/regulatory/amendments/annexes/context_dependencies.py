"""Exact contextual preparation and embedding impact, separate from legal identity."""

import hashlib
import json
import threading
from collections.abc import Callable, Sequence
from datetime import date
from typing import TYPE_CHECKING, Literal

from pydantic import TypeAdapter

from onyx.llm.interfaces import LLM
from onyx.llm.models import LanguageModelInput
from onyx.regulatory.amendments.annexes.models import (
    AnnexContextImpact,
    ContextGenerationCall,
    ContextSourceSnapshot,
    ExistingIndexEmbeddingEvidence,
    FrozenContextProjection,
    PreparedContextView,
)
from onyx.tracing.flows import LLMFlow

if TYPE_CHECKING:
    from onyx.db.models import RegulatoryChunk
    from onyx.indexing.models import DocAwareChunk
    from onyx.natural_language_processing.search_nlp_models import EmbeddingModel


def context_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        ).encode()
    ).hexdigest()


def contextual_model_fingerprint(
    llm: LLM,
) -> dict[str, str | int | float | bool | None]:
    config = llm.config
    return {
        "model_provider": config.model_provider,
        "model_name": config.model_name,
        "temperature": config.temperature,
        "max_input_tokens": config.max_input_tokens,
        "api_version": config.api_version,
        "deployment_name": config.deployment_name,
        "endpoint_sha256": context_hash(config.api_base),
        "custom_config_sha256": context_hash(config.custom_config),
    }


class ContextGenerationRecorder:
    """Capture actual normal-path prompts at invocation; share summaries by hash.

    Cache validity requires the complete prompt and generation config. Consumers
    refer to shared calls; no pairwise document-consumer dependency graph is built.
    """

    def __init__(
        self, *, cached_calls: list[ContextGenerationCall] | None = None
    ) -> None:
        self.calls: dict[str, ContextGenerationCall] = {}
        self.consumer_requests: dict[str, list[str]] = {}
        self._outputs = {
            call.generation_input_sha256: call.output
            for call in cached_calls or []
            if call.generation_input_sha256
        }
        self._request_locks: dict[str, threading.Lock] = {}
        self._lock = threading.RLock()

    def invoke(
        self,
        *,
        llm: LLM,
        flow: LLMFlow,
        prompt: LanguageModelInput,
        generate: Callable[[], str],
        consumer_id: str,
        stage: Literal["summary", "chunk", "fallback_summary"],
        source_text: str,
        token_budget: int,
        tokenizer: str = "unspecified",
    ) -> str:
        from onyx.llm.models import ReasoningEffort
        from onyx.llm.utils import MAX_CONTEXT_TOKENS

        prompt_json = TypeAdapter(LanguageModelInput).dump_json(prompt).decode()
        config_hash = context_hash(
            {
                **contextual_model_fingerprint(llm),
                "max_output_tokens": MAX_CONTEXT_TOKENS,
                "reasoning_effort": ReasoningEffort.OFF,
                "use_streaming": False,
                "flow": flow.value,
            }
        )
        generation_input_hash = context_hash([prompt_json, config_hash])
        request_hash = context_hash(
            [generation_input_hash, stage, tokenizer, token_budget]
        )
        with self._lock:
            request_lock = self._request_locks.setdefault(
                generation_input_hash, threading.Lock()
            )
        # Serialize only identical inputs; unrelated chunks retain existing
        # bounded parallelism, and each shared summary is generated once.
        with request_lock:
            with self._lock:
                output = self._outputs.get(generation_input_hash)
            if output is None:
                output = generate()
                if not output.strip():
                    raise ValueError("context_generation_incomplete")
                with self._lock:
                    self._outputs[generation_input_hash] = output
        call = ContextGenerationCall(
            generation_input_sha256=generation_input_hash,
            request_sha256=request_hash,
            stage=stage,
            prompt_json=prompt_json,
            config_sha256=config_hash,
            output=output,
            source_text=source_text,
            token_budget=token_budget,
            tokenizer=tokenizer,
        )
        with self._lock:
            self.calls[request_hash] = call
            requests = self.consumer_requests.setdefault(consumer_id, [])
            if request_hash not in requests:
                requests.append(request_hash)
        return output

    def share(self, source_consumer: str, consumers: list[str]) -> None:
        with self._lock:
            requests = list(self.consumer_requests.get(source_consumer, []))
            for consumer in consumers:
                current = self.consumer_requests.setdefault(consumer, [])
                current.extend(
                    request for request in requests if request not in current
                )


def compare_context_views(
    *,
    old: PreparedContextView,
    new: PreparedContextView,
    direct_canonical_changes: list[str],
    metadata_only: list[str] | None = None,
    canonical_predecessors: dict[str, str] | None = None,
) -> AnnexContextImpact:
    """Compare every potential consumer after replaying old/new source selection.

    The caller must prepare the complete file views with the actual generation
    path; old dependency IDs or bounded annex neighbors are not sufficient input.
    Missing legacy provenance means evaluate/re-embed every potential consumer,
    never claiming an existing vector came from a reconstructed or guessed input.
    """
    old_by_id = {item.canonical_chunk_id: item for item in old.projections}
    new_by_id = {item.canonical_chunk_id: item for item in new.projections}
    if len(old_by_id) != len(old.projections) or len(new_by_id) != len(new.projections):
        raise ValueError("duplicate_context_consumer")
    predecessors = canonical_predecessors or {}
    if (
        not set(predecessors).issubset(new_by_id)
        or not set(predecessors.values()).issubset(old_by_id)
        or len(set(predecessors.values())) != len(predecessors)
        or set(predecessors).intersection(old_by_id)
    ):
        raise ValueError("invalid one-to-one canonical predecessor mapping")
    contextual: list[str] = []
    embeddings: list[str] = []
    unchanged: list[str] = []
    context_only: list[str] = []
    metadata = set(metadata_only or [])
    reasons: dict[str, list[str]] = {}
    for identifier, after in new_by_id.items():
        before = old_by_id.get(
            (canonical_predecessors or {}).get(identifier, identifier)
        )
        chunk_reasons: list[str] = []
        if before is None or not before.vector_reuse_verified:
            contextual.append(identifier)
            embeddings.append(identifier)
            chunk_reasons.append(
                "legacy_provenance_unavailable"
                if before is not None or identifier not in direct_canonical_changes
                else "new_canonical_content"
            )
        else:
            if (
                before.request_hashes != after.request_hashes
                or before.generation_path != after.generation_path
            ):
                contextual.append(identifier)
                chunk_reasons.append("context_input_changed")
            if (
                before.embedding_config_sha256 != after.embedding_config_sha256
                and identifier not in contextual
            ):
                contextual.append(identifier)
            if (
                before.embedding_input_sha256 != after.embedding_input_sha256
                or before.embedding_config_sha256 != after.embedding_config_sha256
            ):
                embeddings.append(identifier)
                chunk_reasons.append(
                    "embedding_input_changed"
                    if before.embedding_config_sha256 == after.embedding_config_sha256
                    else "embedding_configuration_changed"
                )
            else:
                chunk_reasons.append("embedding_input_unchanged")
            if (
                before.metadata_sha256 != after.metadata_sha256
                or before.source_snapshot_sha256 != after.source_snapshot_sha256
            ):
                metadata.add(identifier)
            if (
                identifier not in contextual
                and identifier not in embeddings
                and identifier not in metadata
                and identifier not in direct_canonical_changes
            ):
                unchanged.append(identifier)
        if identifier in contextual and identifier not in direct_canonical_changes:
            context_only.append(identifier)
        reasons[identifier] = chunk_reasons
    return AnnexContextImpact(
        direct_canonical_changes=sorted(set(direct_canonical_changes)),
        contextual_candidates=sorted(contextual),
        embedding_changes=sorted(embeddings),
        context_only=sorted(context_only),
        metadata_only=sorted(metadata - set(embeddings) - set(contextual)),
        retire_history=sorted(set(old_by_id) - set(new_by_id)),
        unchanged=sorted(unchanged),
        reasons=reasons,
        prepared=new,
        ready=not old.issues and not new.issues,
    )


def freeze_embedding_inputs(
    chunk: "DocAwareChunk", model: "EmbeddingModel", *, model_dim: int
) -> tuple[list[str], dict[str, str | int | float | bool | None]]:
    """Mirror the actual formatter and encode normalization, including title/minis."""
    from onyx.document_index.chunk_content_enrichment import (
        generate_enriched_content_for_chunk_embedding,
    )

    if chunk.large_chunk_reference_ids:
        raise ValueError("regulatory_context_large_chunk_requires_batch_scope")
    title = chunk.source_document.get_title_for_document_index()
    main = generate_enriched_content_for_chunk_embedding(chunk) or title
    if not main:
        raise ValueError("empty_embedding_input")
    texts = [main, *(chunk.mini_chunk_texts or []), *([title] if title else [])]
    return freeze_encoder_inputs(
        texts, model, model_dim=model_dim, formatter="normal-v1"
    )


def freeze_encoder_inputs(
    texts: list[str], model: "EmbeddingModel", *, model_dim: int, formatter: str
) -> tuple[list[str], dict[str, str | int | float | bool | None]]:
    """Freeze the text and configuration reaching the actual encoder transport."""
    from onyx.natural_language_processing.utils import tokenizer_trim_content
    from onyx.utils.text_processing import remove_invalid_unicode_chars
    from shared_configs.configs import DOC_EMBEDDING_CONTEXT_SIZE

    if not texts or not all(texts):
        raise ValueError("empty_embedding_input")
    if model.retrim_content:
        texts = [
            tokenizer_trim_content(
                content=text,
                desired_length=DOC_EMBEDDING_CONTEXT_SIZE,
                tokenizer=model.tokenizer,
            )
            for text in texts
        ]
    texts = [remove_invalid_unicode_chars(text) or "<>" for text in texts]
    return texts, encoder_model_fingerprint(
        model, model_dim=model_dim, formatter=formatter
    )


def encoder_model_fingerprint(
    model: "EmbeddingModel", *, model_dim: int, formatter: str
) -> dict[str, str | int | float | bool | None]:
    from shared_configs.configs import DOC_EMBEDDING_CONTEXT_SIZE

    config: dict[str, str | int | float | bool | None] = {
        "provider": str(model.provider_type) if model.provider_type else None,
        "model": model.model_name,
        "dimension": model_dim,
        "reduced_dimension": model.reduced_dimension,
        "normalize": model.normalize,
        "passage_prefix": model.passage_prefix,
        "retrim_content": model.retrim_content,
        "endpoint_sha256": context_hash(model.api_url),
        "api_version": model.api_version,
        "deployment_name": model.deployment_name,
        "tokenizer": f"{type(model.tokenizer).__module__}.{type(model.tokenizer).__qualname__}",
        "max_sequence_length": DOC_EMBEDDING_CONTEXT_SIZE,
        "formatter": formatter,
        "text_type": "passage",
    }
    return config


def canonical_dependency_closure(
    rows: list["RegulatoryChunk"], changed_ids: list[str]
) -> list[str]:
    """Traverse actual aggregate/companion consumers, without distance limits."""
    consumers: dict[str, set[str]] = {}
    for row in rows:
        for source in canonical_dependency_ids(row):
            consumers.setdefault(source, set()).add(row.id)
    visited = set(changed_ids)
    pending = list(changed_ids)
    while pending:
        source = pending.pop()
        for consumer in consumers.get(source, set()) - visited:
            visited.add(consumer)
            pending.append(consumer)
    return sorted(visited)


def canonical_dependency_ids(row: "RegulatoryChunk") -> list[str]:
    sources = row.chunk_metadata.get("source_regulatory_chunk_ids", [])
    if not isinstance(sources, list) or not all(
        isinstance(item, str) for item in sources
    ):
        raise ValueError("invalid_canonical_dependencies")
    binding = row.chunk_metadata.get("bound_to_regulatory_chunk_id")
    if binding is not None and not isinstance(binding, str):
        raise ValueError("invalid_canonical_image_binding")
    return list(dict.fromkeys([*sources, *([binding] if binding else [])]))


def verify_existing_index_evidence(
    projection: "FrozenContextProjection", evidence: "ExistingIndexEmbeddingEvidence"
) -> "FrozenContextProjection":
    """Accept exact existing-index proof without fabricating active DB history.

    Publication must revalidate the index/document/ordinal identity under its
    gate. Callers reconstruct this evidence from actual stored contexts and a
    proven embedding config; freshly generated old summaries are not evidence.
    """
    if (
        projection.canonical_chunk_id != evidence.canonical_chunk_id
        or projection.embedding_input_sha256 != evidence.embedding_input_sha256
        or projection.embedding_config_sha256 != evidence.embedding_config_sha256
        or projection.canonical_text_sha256 != evidence.canonical_text_sha256
        or evidence.vector_dimension != evidence.expected_dimension
    ):
        raise ValueError("existing index embedding evidence mismatch")
    return projection.model_copy(
        update={
            "vector_reuse_verified": True,
            "existing_index_evidence": evidence.model_dump(),
        }
    )


def rebuild_context_aggregates(
    rows: list["RegulatoryChunk"], *, changed_ids: list[str]
) -> list["RegulatoryChunk"]:
    """Regenerate the complete derived chain without mutating stored legal rows.

    Candidate source references/heading metadata must already describe the new
    snapshot. Missing sources, unknown roots and cycles are explicit blockers.
    """
    from onyx.db.models import RegulatoryChunk
    from onyx.regulatory.chunker import (
        hierarchical_aggregate_root_label,
        hierarchical_aggregate_text,
    )

    by_id = {row.id: row for row in rows}
    affected = set(canonical_dependency_closure(rows, changed_ids))
    pending = {
        row.id
        for row in rows
        if row.id in affected
        and row.chunk_metadata.get("chunk_variant") == "hierarchical_aggregate"
    }
    while pending:
        progress = False
        for identifier in list(pending):
            row = by_id[identifier]
            sources = canonical_dependency_ids(row)
            if any(source not in by_id for source in sources):
                raise ValueError("aggregate_source_unavailable")
            if any(source in pending for source in sources):
                continue
            root_path = row.chunk_metadata.get("hierarchy_root_path")
            if (
                not sources
                or not isinstance(root_path, list)
                or not root_path
                or not isinstance(root_path[-1], str)
            ):
                raise ValueError("aggregate_provenance_unavailable")
            text = hierarchical_aggregate_text(
                hierarchical_aggregate_root_label(row.chunk_metadata, row.text),
                [by_id[source].text for source in sources],
            )
            by_id[identifier] = RegulatoryChunk(
                id=row.id,
                user_file_id=row.user_file_id,
                text=text,
                position=row.position,
                projection_ordinal=row.projection_ordinal,
                heading_path=list(row.heading_path),
                chunk_metadata=dict(row.chunk_metadata),
                chunk_type=row.chunk_type,
                source=row.source,
                status=row.status,
                validity_start_date=row.validity_start_date,
                validity_end_date=row.validity_end_date,
            )
            pending.remove(identifier)
            progress = True
        if not progress:
            raise ValueError("cyclic_aggregate_dependencies")
    return [by_id[row.id] for row in rows]


def effective_context_rows(
    rows: list["RegulatoryChunk"], as_of_date: "date | None"
) -> list["RegulatoryChunk"]:
    from onyx.regulatory.contextual import validity_window_contains

    ordered = sorted(rows, key=lambda row: (row.position, row.id))
    if len({row.id for row in ordered}) != len(ordered):
        raise ValueError("duplicate_context_consumer")
    return [
        row
        for row in ordered
        if as_of_date is None
        or validity_window_contains(
            row.validity_start_date, row.validity_end_date, as_of_date
        )
    ]


def freeze_context_source_snapshot(
    *,
    rows: Sequence["RegulatoryChunk"],
    target: "RegulatoryChunk",
    reference_date: "date | None",
    generation_path: Literal["normal", "durable"],
    row_text: Callable[["RegulatoryChunk"], str],
) -> "ContextSourceSnapshot":
    from onyx.regulatory.amendments.annexes.models import (
        ContextSourceRange,
        ContextSourceSnapshot,
    )
    from onyx.regulatory.contextual import (
        context_reference_date,
        visible_regulatory_snapshot_for_target,
    )

    visible = visible_regulatory_snapshot_for_target(
        rows, target, reference_date=reference_date
    )
    reference = reference_date or context_reference_date(
        target.validity_start_date, target.validity_end_date
    )
    ranges: list[ContextSourceRange] = []
    parts: list[str] = []
    offset = 0
    for source in visible:
        text = row_text(source)
        ranges.append(
            ContextSourceRange(
                canonical_chunk_id=source.id, start=offset, end=offset + len(text)
            )
        )
        parts.append(text)
        offset += len(text) + 2
    text = "\n\n".join(parts)
    return ContextSourceSnapshot(
        sha256=context_hash(
            [
                str(target.user_file_id),
                text,
                [span.model_dump() for span in ranges],
                reference,
            ]
        ),
        selector=f"visible_regulatory_snapshot_for_target:{generation_path}-v1",
        reference_date=reference,
        text=text,
        ordered_ranges=ranges,
    )


def validate_complete_context_view(
    *,
    rows: list["RegulatoryChunk"],
    view: PreparedContextView,
    as_of_date: "date",
) -> None:
    """Require the exact effective consumers and shared source-selection output."""
    from onyx.regulatory.indexing_jobs.contextual import _row_block
    from onyx.regulatory.projection import _row_context_text

    expected = effective_context_rows(rows, as_of_date)
    if view.issues or [item.canonical_chunk_id for item in view.projections] != [
        row.id for row in expected
    ]:
        raise ValueError("prepared context consumer coverage is incomplete")
    snapshots = {snapshot.sha256: snapshot for snapshot in view.snapshots}
    calls = {call.request_sha256: call for call in view.calls}
    if len(snapshots) != len(view.snapshots) or len(calls) != len(view.calls):
        raise ValueError("duplicate prepared context dependency")
    used_snapshots: set[str] = set()
    used_calls: set[str] = set()
    for row, projection in zip(expected, view.projections, strict=True):
        snapshot = freeze_context_source_snapshot(
            rows=rows,
            target=row,
            reference_date=as_of_date,
            generation_path=projection.generation_path,
            row_text=_row_context_text
            if projection.generation_path == "normal"
            else _row_block,
        )
        if snapshots.get(projection.source_snapshot_sha256) != snapshot:
            raise ValueError("prepared context source range coverage changed")
        used_snapshots.add(snapshot.sha256)
        if not set(projection.request_hashes).issubset(calls):
            raise ValueError("prepared context generation dependencies missing")
        used_calls.update(projection.request_hashes)
        if (
            projection.canonical_text_sha256 != context_hash(row.text)
            or projection.metadata_sha256
            != context_hash(
                [
                    row.chunk_metadata,
                    row.heading_path,
                    row.position,
                    row.validity_start_date,
                    row.validity_end_date,
                ]
            )
            or projection.canonical_dependency_ids != canonical_dependency_ids(row)
        ):
            raise ValueError("prepared context canonical input changed")
        if (
            not projection.embedding_texts
            or not all(projection.embedding_texts)
            or not projection.embedding_config
            or projection.embedding_input_sha256
            != context_hash(projection.embedding_texts)
            or projection.embedding_config_sha256
            != context_hash(projection.embedding_config)
        ):
            raise ValueError("prepared context embedding identity changed")
        if (
            projection.validity_start != as_of_date
            or projection.validity_end != row.validity_end_date
        ):
            raise ValueError("prepared context effective interval changed")
        if projection.vector_reuse_verified:
            evidence = ExistingIndexEmbeddingEvidence.model_validate(
                projection.existing_index_evidence
            )
            verify_existing_index_evidence(projection, evidence)
    if used_snapshots != set(snapshots) or used_calls != set(calls):
        raise ValueError("prepared context has unrelated dependencies")
