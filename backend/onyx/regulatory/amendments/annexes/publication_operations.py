"""Turn completed frozen embedding receipts into one immutable ES operation inventory."""

import json

from onyx.document_index.publication_models import (
    FrozenPublicationProjection,
    RetainedPublicationProjection,
)
from onyx.regulatory.amendments.annexes.models import (
    AnnexPublicationPreparation,
    AnnexTemporalProjection,
)
from onyx.regulatory.amendments.annexes.publication_execution_models import (
    AnnexPublicationDelivery,
    AnnexPublicationOperation,
    AnnexPublicationOperations,
)


def build_operations(
    delivery: AnnexPublicationDelivery, prepared: AnnexPublicationPreparation
) -> AnnexPublicationOperations:
    from onyx.db.regulatory_annex_execution import embedding_checkpoint

    operations = []
    from onyx.regulatory.amendments.annexes.publication_representations import _epoch

    for retained in prepared.retained:
        source = json.loads(retained.evidence.source_json)
        if retained.close_interval:
            source["validity_end_date"] = _epoch(retained.validity_end)
        operations.append(
            AnnexPublicationOperation(
                index_uuid=retained.evidence.index.index_uuid,
                ordinal=source["chunk_index"],
                kind="retain",
                retained=RetainedPublicationProjection(
                    evidence=retained.evidence, source_json=json.dumps(source)
                ),
            )
        )
    for plan in prepared.projections:
        source = json.loads(plan.source_template_json)
        if plan.reuse_from is not None:
            actual = json.loads(plan.reuse_from.source_json)
            source["content_vector"] = actual["content_vector"]
            if "title_vector" in actual:
                source["title_vector"] = actual["title_vector"]
        else:
            checkpoint = embedding_checkpoint(delivery, plan.id)
            if checkpoint.status != "complete" or checkpoint.vectors is None:
                raise ValueError("frozen embedding result is incomplete")
            vectors = checkpoint.vectors
            # Frozen normal input order is content, optional minis, then title.
            if len(vectors) != 1 + len(plan.context.mini_chunk_texts) + bool(
                source.get("title")
            ):
                raise ValueError("frozen projection encoder input roles are unresolved")
            source["content_vector"] = vectors[0]
            if source.get("title"):
                source["title_vector"] = vectors[-1]
        projection = FrozenPublicationProjection(
            ordinal=plan.ordinal,
            context_projection_id=str(plan.id),
            source_json=json.dumps(source),
            embedding_inputs=tuple(plan.context.embedding_texts),
            embedding_config_json=plan.reuse_from.embedding_config_json
            if plan.reuse_from
            else json.dumps(plan.context.embedding_config),
        )
        role = (
            "hierarchical_aggregate"
            if plan.row.metadata.get("chunk_variant") == "hierarchical_aggregate"
            else "image_companion"
            if plan.row.metadata.get("bound_to_regulatory_chunk_id")
            else "canonical"
        )
        binding = AnnexTemporalProjection(
            id=plan.id,
            index=plan.index,
            projection=projection,
            canonical_base_sha256=plan.canonical_base_sha256,
            derived_role=role,
            dependency_ids=plan.context.canonical_dependency_ids,
            representation_text=plan.row.text,
            representation_metadata=plan.row.metadata,
            context=plan.context,
            reference_date=plan.reference_date,
            effective_start=plan.effective_start,
            effective_end=plan.effective_end,
            semantic_position=plan.row.position,
        )
        # Full frozen upserts keep the same operation valid after a new fencing token.
        # Metadata-base CAS receipts cannot be replayed against a previously updated base.
        operations.append(
            AnnexPublicationOperation(
                index_uuid=plan.index.index_uuid,
                ordinal=plan.ordinal,
                kind="upsert",
                binding=binding,
            )
        )
    for index in prepared.indexes:
        operations.extend(
            AnnexPublicationOperation(
                index_uuid=index.index_uuid, ordinal=ordinal, kind="tombstone"
            )
            for ordinal in prepared.retired_ordinals[index.index_uuid]
        )
    return AnnexPublicationOperations(operations=operations)
