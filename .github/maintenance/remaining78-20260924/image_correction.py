"""The two explicitly approved image corrections, proven against original assets."""

import json
from datetime import date
from uuid import uuid4

from elasticsearch import Elasticsearch

from onyx.db.regulatory_publication import PublicationStore
from onyx.db.regulatory_writer_publication import OwnedWriterInputs
from onyx.document_index.elasticsearch.publication import FencedPublicationIndex
from onyx.document_index.publication_models import (
    FileOwnership,
    FrozenPublicationProjection,
    PublicationIndexSnapshot,
    publication_digest,
)
from onyx.indexing.embedder import DefaultIndexingEmbedder
from onyx.regulatory.amendments.annexes.analysis import resolve_review_context_llm
from onyx.regulatory.amendments.annexes.context_dependencies import (
    context_hash,
    freeze_encoder_inputs,
)
from onyx.regulatory.amendments.annexes.models import (
    AnnexCanonicalSnapshot,
    AnnexTemporalProjection,
)
from onyx.regulatory.amendments.annexes.publication_evidence import _encoder_receipt
from onyx.regulatory.amendments.annexes.publication_preparation import (
    _target_configuration,
)
from onyx.regulatory.amendments.annexes.publication_representations import (
    _source_template,
)
from onyx.regulatory.amendments.annexes.staging import canonical_snapshot_rows
from onyx.regulatory.indexing_jobs.embedding import _validate_response_vectors
from onyx.regulatory.projection import prepare_normal_context_view
from onyx.regulatory.publication_baseline import (
    observed_baseline_binding,
    observed_index_snapshot,
)
from onyx.regulatory.source_metadata_repair import (
    recover_markdown_image_parents,
    repair_canonical_source_metadata,
)
from onyx.regulatory.writer_publication_models import WriterPublicationManifest
from shared_configs.configs import MULTI_TENANT
from shared_configs.enums import EmbedTextType

FILE_ID = "bf99fba0-0e69-4e1f-8d2d-d4cef8c6d35b"
PARENTS = {
    "rc_b0b621007e8c13b95a62091af1f0dd40a463fa0d": "rc_1b82047eaf47617029176f0c4040ab65f05cc9d9",
    "rc_2d0d8c5d19de6de446f756016cbabdd12427e053": "rc_fd29ad6882bb6617cd9cb4f28b1c70ba95e1ea42",
}


def corrected_rows(
    rows: list[AnnexCanonicalSnapshot],
    markdown: bytes,
    assets: dict[str, str],
    headings: dict[str, list[str]],
) -> list[AnnexCanonicalSnapshot]:
    by_id = {row.id: row for row in rows}
    if not set(PARENTS) <= by_id.keys() or any(
        row.user_file_id != FILE_ID for row in rows
    ):
        raise ValueError("approved image correction scope changed")
    after = []
    for row in rows:
        if row.id not in PARENTS:
            after.append(row)
            continue
        parent = by_id[PARENTS[row.id]]
        if row.validity_start_date is not None or row.validity_end_date is not None:
            raise ValueError("approved image temporal scope changed")
        caption = row.metadata.get("image_alt")
        text = parent.text + (f"\n\n[Görsel: {caption}]" if caption else "")
        after.append(
            row.model_copy(
                update={
                    "text": text,
                    "heading_path": parent.heading_path,
                    "metadata": {
                        **row.metadata,
                        "heading_path": parent.heading_path,
                        "bound_to_regulatory_chunk_id": parent.id,
                        "source_regulatory_chunk_ids": [],
                    },
                }
            )
        )
    proofs = recover_markdown_image_parents(after, markdown, assets)
    if any(
        identifier not in proofs or proofs[identifier].parent_id != parent
        for identifier, parent in PARENTS.items()
    ):
        raise ValueError(
            "original image bytes and source position do not prove both corrections"
        )
    # Old ES headings belong to the wrong body for precisely these two rows.
    return repair_canonical_source_metadata(
        after,
        markdown=markdown,
        asset_sha256=assets,
        indexed_headings={
            key: value for key, value in headings.items() if key not in PARENTS
        },
    )


def prepare_image_correction(
    owner: FileOwnership,
    client: Elasticsearch,
    inputs: OwnedWriterInputs,
    after: list[AnnexCanonicalSnapshot],
) -> WriterPublicationManifest:
    if (
        str(owner.user_file_id) != FILE_ID
        or inputs.bindings
        or len(inputs.settings) != 1
    ):
        raise ValueError(
            "two-image adoption requires the exact unqualified legacy file"
        )
    settings = inputs.settings[0]
    authority = PublicationStore(owner.scope)
    physical = client.indices.get(index=settings.index_name)
    index_uuid = physical[settings.index_name]["settings"]["index"]["uuid"]
    observed = observed_index_snapshot(settings, index_uuid)
    adapter = FencedPublicationIndex(client, observed)
    authority.reserve_existing_ordinals(
        owner, adapter.existing_ordinals(authority.reservations(owner))
    )
    evidence = list(adapter.inventory_evidence(authority.reservations(owner)))
    actual_ids = [
        json.loads(item.source_json)["regulatory_chunk_id"] for item in evidence
    ]
    if len(actual_ids) != len(after) or set(actual_ids) != {row.id for row in after}:
        raise ValueError("two-image correction requires complete existing inventory")
    if any(
        item.frozen_projection is not None or item.observed_projection is not None
        for item in evidence
    ):
        raise ValueError("two-image correction refuses already-qualified history")
    unchanged = [
        observed_baseline_binding(item, after)
        for item in evidence
        if json.loads(item.source_json)["regulatory_chunk_id"] not in PARENTS
    ]
    embedder = DefaultIndexingEmbedder.from_db_search_settings(search_settings=settings)
    configuration = _target_configuration(settings, None, embedder)
    receipt = _encoder_receipt(configuration, resolution=context_hash(configuration))
    index = PublicationIndexSnapshot(
        index_name=settings.index_name,
        index_uuid=index_uuid,
        search_settings_id=settings.id,
        model_provider=receipt.authority.provider or "",
        model_name=receipt.authority.model,
        vector_dimension=settings.final_embedding_dim,
        embedding_config_sha256=context_hash(configuration),
        multitenant=MULTI_TENANT,
        encoder_authority=receipt.authority,
        encoder_receipts=(receipt,),
    )
    rows = canonical_snapshot_rows(after)
    by_id = {row.id: row for row in rows}
    view = prepare_normal_context_view(
        rows=rows,
        user_file=inputs.file,
        search_settings=settings,
        embedder=embedder,
        llm=resolve_review_context_llm(settings, None),
        cached=inputs.cached,
        as_of_date=date.today(),
        target_ids=set(PARENTS),
    )
    if {context.canonical_chunk_id for context in view.projections} != set(PARENTS):
        raise ValueError(
            "normal projection did not prepare exactly two approved images"
        )
    bindings = list(unchanged)
    for context in view.projections:
        row = by_id[context.canonical_chunk_id]
        texts, actual = freeze_encoder_inputs(
            context.embedding_texts,
            embedder.embedding_model,
            model_dim=settings.model_dim,
            formatter=str(context.embedding_config["formatter"]),
        )
        if texts != context.embedding_texts or actual != context.embedding_config:
            raise ValueError("image encoder differs from frozen inputs")
        vectors = _validate_response_vectors(
            embedder.embedding_model.encode(
                texts=texts,
                text_type=EmbedTextType.PASSAGE,
                tenant_id=owner.scope.tenant_id,
            ),
            expected_count=len(texts),
            expected_dimension=index.vector_dimension,
        )
        source = json.loads(
            _source_template(
                row,
                context,
                file_name=inputs.file.name,
                access=inputs.access,
                tenant_id=owner.scope.tenant_id,
                dimension=index.vector_dimension,
            )
        )
        source["content_vector"] = vectors[0]
        if source.get("title"):
            source["title_vector"] = vectors[-1]
        identifier = uuid4()
        projection = FrozenPublicationProjection(
            ordinal=row.projection_ordinal,
            context_projection_id=str(identifier),
            source_json=json.dumps(source),
            embedding_inputs=tuple(context.embedding_texts),
            embedding_config_json=json.dumps(context.embedding_config),
        )
        bindings.append(
            AnnexTemporalProjection(
                id=identifier,
                index=index,
                projection=projection,
                canonical_base_sha256=context_hash(row.text),
                derived_role="image_companion",
                dependency_ids=context.canonical_dependency_ids,
                representation_text=row.text,
                representation_metadata=row.chunk_metadata,
                context=context,
                reference_date=date.today(),
                effective_start=None,
                effective_end=None,
                semantic_position=row.position,
            )
        )
    return WriterPublicationManifest(
        id=uuid4(),
        scope=owner.scope,
        user_file_id=owner.user_file_id,
        kind="baseline",
        index_state_sha256=inputs.index_state_sha256,
        canonical_before_sha256=publication_digest(
            [row.model_dump(mode="json") for row in inputs.canonical]
        ),
        canonical_after=after,
        indexes=[index],
        previous_binding_ids=[],
        bindings=bindings,
        views=[view],
    )
