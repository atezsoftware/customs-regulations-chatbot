"""Date-qualified canonical/source representations and exact ES serialization."""

import json
from datetime import date, datetime, timezone
from typing import TYPE_CHECKING

from onyx.regulatory.amendments.annexes.context_dependencies import (
    context_hash,
    effective_context_rows,
    rebuild_context_aggregates,
)
from onyx.regulatory.amendments.annexes.models import (
    AnnexCanonicalSnapshot,
    AnnexChangeDraft,
    AnnexLegalPublicationTimeline,
    AnnexProjectionAccess,
    AnnexTemporalProjection,
    FrozenContextProjection,
)
from onyx.regulatory.amendments.annexes.staging import (
    canonical_snapshot_rows,
    remap_new_chunk_evidence,
)

if TYPE_CHECKING:
    from onyx.db.models import RegulatoryChunk


def _as_date(value: object) -> date | None:
    return (
        datetime.fromtimestamp(value, tz=timezone.utc).date()
        if isinstance(value, int)
        else None
    )


def _epoch(value: date | None) -> int | None:
    return (
        int(datetime.combine(value, datetime.min.time(), timezone.utc).timestamp())
        if value is not None
        else None
    )


def _snapshot(row: "RegulatoryChunk") -> AnnexCanonicalSnapshot:
    return AnnexCanonicalSnapshot(
        id=row.id,
        user_file_id=str(row.user_file_id),
        chunk_type=row.chunk_type,
        status=row.status,
        projection_ordinal=row.projection_ordinal,
        supersedes_chunk_id=row.supersedes_chunk_id,
        superseded_by_chunk_id=row.superseded_by_chunk_id,
        position=row.position,
        text=row.text,
        heading_path=row.heading_path,
        metadata=json.loads(json.dumps(row.chunk_metadata)),
        source=row.source,
        validity_start_date=row.validity_start_date,
        validity_end_date=row.validity_end_date,
    )


def temporal_candidate_rows(
    draft: AnnexChangeDraft,
    legal: AnnexLegalPublicationTimeline,
    when: date,
    bindings: list[AnnexTemporalProjection] | None = None,
    positions: dict[str, int] | None = None,
) -> list["RegulatoryChunk"]:
    """Derive dated representations without rewriting baseline legal metadata."""
    rows = canonical_snapshot_rows(legal.canonical_rows)
    by_id = {row.id: row for row in rows}
    for row in rows:
        predecessor = legal.restoration_predecessors.get(row.id, row.id)
        matches = [
            binding
            for binding in bindings or []
            if json.loads(binding.projection.source_json)["regulatory_chunk_id"]
            == predecessor
            and (binding.effective_start is None or binding.effective_start <= when)
            and (binding.effective_end is None or when < binding.effective_end)
        ]
        if (
            len(
                {
                    context_hash(
                        [
                            binding.canonical_base_sha256,
                            binding.derived_role,
                            binding.representation_text,
                            binding.representation_metadata,
                            binding.semantic_position,
                        ]
                    )
                    for binding in matches
                }
            )
            > 1
        ):
            raise ValueError("dated canonical source representation is ambiguous")
        if matches:
            binding = matches[0]
            base = by_id[predecessor]
            if context_hash(base.text) != binding.canonical_base_sha256:
                raise ValueError("dated representation canonical base changed")
            if (
                binding.derived_role == "canonical"
                and row.text != binding.representation_text
            ):
                raise ValueError("direct legal representation mismatch")
            row.text = binding.representation_text
            row.chunk_metadata = dict(binding.representation_metadata)
            row.position = binding.semantic_position
    if draft.impact_strategy == "source_dependencies_v1":
        from onyx.regulatory.amendments.annexes.selective_impact import (
            recover_source_membership,
        )

        recovered = recover_source_membership(draft.baseline_scope)
        for row in rows:
            if row.id in recovered and not row.chunk_metadata.get(
                "source_regulatory_chunk_ids"
            ):
                row.chunk_metadata = {
                    **row.chunk_metadata,
                    "source_regulatory_chunk_ids": recovered[row.id],
                }
    for row in rows:
        if positions and row.id in positions:
            row.position = positions[row.id]
    for item in draft.items:
        if item.old_chunk_ids:
            anchor = by_id[item.old_chunk_ids[0]]
            for offset, chunk in enumerate(item.new_chunks):
                by_id[chunk.id].position = anchor.position + offset
    effective = effective_context_rows(rows, when)
    effective_ids = {row.id for row in effective}
    for item in reversed(draft.items):
        inserted = [chunk for chunk in item.new_chunks if chunk.id in effective_ids]
        if item.old_chunk_ids or not inserted:
            continue
        anchor = by_id.get(item.insertion_after_chunk_id or "")
        if anchor is None:
            raise ValueError("temporal insertion anchor missing")
        for row in effective:
            if row.position > anchor.position:
                row.position += len(inserted)
        for offset, chunk in enumerate(inserted, 1):
            by_id[chunk.id].position = anchor.position + offset
    for item in reversed(draft.items):
        active_new = [chunk for chunk in item.new_chunks if chunk.id in effective_ids]
        if not item.old_chunk_ids or not active_new:
            continue
        delta = len(active_new) - len(item.old_chunk_ids)
        bound = max(by_id[identifier].position for identifier in item.old_chunk_ids)
        staged_ids = {chunk.id for chunk in active_new}
        for row in effective:
            if row.id not in staged_ids and row.position > bound:
                row.position += delta
    mapping: dict[str, list[str]] = {}
    for item in draft.items:
        for old_id in item.old_chunk_ids:
            successors = [
                row.id
                for row in effective
                if row.supersedes_chunk_id == old_id
                or legal.restoration_predecessors.get(row.id) == old_id
            ]
            staged = [row.id for row in item.new_chunks if row.id in effective_ids]
            mapping[old_id] = (
                staged or successors or ([old_id] if old_id in effective_ids else [])
            )
    during = when >= legal.effective_start and (
        legal.effective_end is None or when < legal.effective_end
    )
    for row in effective:
        if (
            during
            and row.id in draft.source_only_canonical_ids
            and draft.new_evidence_remapping
            and draft.baseline
        ):
            element_positions = [
                position
                for position, element in enumerate(draft.baseline.elements)
                if element.canonical_chunk_id == row.id
            ]
            if draft.comparison and draft.comparison.schema_version == 2:
                if draft.new_extraction is None:
                    raise ValueError("source-only extraction missing")
                resolved = []
                for position in element_positions:
                    old_element = draft.baseline.elements[position]
                    matches = [
                        index
                        for index in draft.comparison.coverage.new_positions
                        if (
                            draft.new_extraction.elements[index].kind,
                            draft.new_extraction.elements[index].text,
                        )
                        == (old_element.kind, old_element.text)
                    ]
                    if len(matches) != 1:
                        raise ValueError("source-only correspondence is ambiguous")
                    resolved.append(matches[0])
                element_positions = resolved
            row.chunk_metadata = remap_new_chunk_evidence(
                row.chunk_metadata, element_positions, draft.new_evidence_remapping
            )
    insertions: dict[str, list[str]] = {}
    if draft.impact_strategy == "source_dependencies_v1" and during:
        for item in draft.items:
            if not item.old_chunk_ids and item.insertion_after_chunk_id:
                insertions.setdefault(item.insertion_after_chunk_id, []).extend(
                    chunk.id for chunk in item.new_chunks if chunk.id in effective_ids
                )
    for row in effective:
        metadata = dict(row.chunk_metadata)
        sources = metadata.get("source_regulatory_chunk_ids")
        if isinstance(sources, list):
            expanded = []
            for source in sources:
                if not isinstance(source, str):
                    continue
                expanded.append(source)
                for identifier in insertions.get(source, []):
                    root = metadata.get("hierarchy_root_path")
                    target = by_id[identifier]
                    replaces_anchor = any(
                        source in item.old_chunk_ids for item in draft.items
                    )
                    if not replaces_anchor and (
                        not isinstance(root, list)
                        or not root
                        or target.heading_path[: len(root)] != root
                    ):
                        raise ValueError(
                            "inserted aggregate boundary requires explicit source membership"
                        )
                    expanded.append(identifier)
            sources = expanded
            metadata["source_regulatory_chunk_ids"] = list(
                dict.fromkeys(
                    target
                    for source in sources
                    if isinstance(source, str)
                    for target in mapping.get(source, [source])
                )
            )
        binding = metadata.get("bound_to_regulatory_chunk_id")
        if isinstance(binding, str):
            targets = mapping.get(
                binding, [binding] if binding in effective_ids else []
            )
            if not targets:
                row.chunk_metadata = {**metadata, "annex_temporal_retired": True}
                continue
            if len(targets) != 1:
                raise ValueError("temporal companion authority is ambiguous")
            metadata["bound_to_regulatory_chunk_id"] = targets[0]
            target = by_id[targets[0]]
            for key in (
                "image_file_id",
                "image_file_ids",
                "source_asset_ids",
                "annex_element_ids",
            ):
                metadata.pop(key, None)
                if key in target.chunk_metadata:
                    metadata[key] = target.chunk_metadata[key]
        row.chunk_metadata = metadata
    retired = {
        row.id
        for row in effective
        if row.chunk_metadata.get("annex_temporal_retired") is True
    }
    while True:
        before = set(retired)
        for row in effective:
            if row.chunk_metadata.get("chunk_variant") != "hierarchical_aggregate":
                continue
            dependencies = row.chunk_metadata.get("source_regulatory_chunk_ids", [])
            if isinstance(dependencies, list):
                retained = [
                    identifier
                    for identifier in dependencies
                    if identifier not in retired
                ]
                row.chunk_metadata = {
                    **row.chunk_metadata,
                    "source_regulatory_chunk_ids": retained,
                }
                if not retained:
                    retired.add(row.id)
        if before == retired:
            break
    effective = [row for row in effective if row.id not in retired]
    return rebuild_context_aggregates(
        effective, changed_ids=[row.id for row in effective]
    )


def _source_template(
    row: "RegulatoryChunk",
    context: FrozenContextProjection,
    *,
    file_name: str,
    access: AnnexProjectionAccess,
    tenant_id: str,
    dimension: int,
) -> str:
    from onyx.configs.constants import DEFAULT_BOOST, DocumentSource
    from onyx.connectors.models import Document, TextSection
    from onyx.document_index.elasticsearch.elasticsearch_document_index import (
        serialize_publication_chunk,
    )
    from onyx.indexing.models import (
        ChunkEmbedding,
        DocMetadataAwareIndexChunk,
        IndexChunk,
    )
    from onyx.regulatory.chunk_evidence import chunk_evidence
    from onyx.regulatory.heading_path import normalize_regulatory_heading_path

    evidence = chunk_evidence(row.chunk_metadata)
    chunk = IndexChunk(
        source_document=Document(
            id=str(row.user_file_id),
            source=DocumentSource.USER_FILE,
            semantic_identifier=file_name,
            title=context.title or "",
            sections=[TextSection(text="", link=None)],
            metadata={},
        ),
        chunk_id=max(row.projection_ordinal, 0),
        blurb=row.text,
        content=row.text,
        source_links=evidence.source_links,
        image_file_id=evidence.image_file_id,
        section_continuation=False,
        title_prefix="",
        metadata_suffix_semantic="",
        metadata_suffix_keyword="",
        mini_chunk_texts=None,
        large_chunk_id=None,
        doc_summary=context.doc_summary,
        chunk_context=context.chunk_context,
        contextual_rag_reserved_tokens=0,
        regulatory_chunk_id=row.id,
        heading_path=normalize_regulatory_heading_path(
            row.heading_path,
            article_no=str(row.chunk_metadata["article_no"])
            if row.chunk_metadata.get("article_no") is not None
            else None,
            chunk_type=row.chunk_type,
        ),
        validity_start_date=row.validity_start_date,
        validity_end_date=row.validity_end_date,
        embeddings=ChunkEmbedding(
            full_embedding=[0.0] * dimension, mini_chunk_embeddings=[]
        ),
        title_embedding=[0.0] * dimension if context.title else None,
    )
    enriched = DocMetadataAwareIndexChunk.from_index_chunk(
        index_chunk=chunk,
        access=access.access,
        document_sets=set(access.document_sets),
        user_project=access.project_ids,
        personas=access.persona_ids,
        boost=DEFAULT_BOOST,
        tenant_id=tenant_id,
        aggregated_chunk_boost_factor=1.0,
    )
    source = json.loads(serialize_publication_chunk(enriched))
    source.pop("content_vector")
    source.pop("title_vector", None)
    return json.dumps(source)
