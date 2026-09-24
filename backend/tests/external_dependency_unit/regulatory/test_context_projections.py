from datetime import date, datetime, time, timezone
from uuid import UUID, uuid4

import pytest
from sqlalchemy.orm import Session

from onyx.db.models import DocumentSet
from onyx.db.regulatory_public_reads import load_public_temporal_bindings
from tests.external_dependency_unit.regulatory.test_amendment_sources import (
    source_session as source_session,
)
from tests.external_dependency_unit.regulatory.test_annex_baseline import _chunk, _file


@pytest.mark.parametrize("case", ["own_asset", "changed_asset", "changed_parent"])
def test_frozen_image_keeps_its_own_asset_for_unchanged_text_parent(
    source_session: Session, case: str
) -> None:
    import json

    from onyx.db.regulatory_context_projections import activate_temporal_projection
    from onyx.document_index.publication_models import (
        PublicationIndexSnapshot,
        publication_digest,
    )
    from onyx.regulatory.amendments.annexes.context_dependencies import context_hash
    from onyx.regulatory.amendments.annexes.models import AnnexTemporalProjection
    from tests.external_dependency_unit.regulatory.test_publication_primitives import (
        frozen_projection,
    )

    group = DocumentSet(name=str(uuid4()), description="", is_up_to_date=True)
    source_session.add(group)
    source_session.flush()
    file = _file(source_session, group)
    parent = _chunk(source_session, file, 0, "Text without embedded asset metadata")
    companion = _chunk(
        source_session,
        file,
        1,
        "Image caption",
        bound_to_regulatory_chunk_id=parent.id,
        image_file_id="own-image",
    )
    target = parent
    if case == "changed_parent":
        target = _chunk(source_session, file, 2, "New text")
        target.supersedes_chunk_id = parent.id
        source_session.flush()
    metadata = {**companion.chunk_metadata, "bound_to_regulatory_chunk_id": target.id}
    if case == "changed_asset":
        metadata["image_file_id"] = "invented-image"
    identity = uuid4()
    frozen = frozen_projection(file.id, 1, companion.text)
    source = json.loads(frozen.source_json)
    source.update(
        regulatory_chunk_id=companion.id,
        doc_summary="",
        chunk_context="",
        image_file_id=metadata["image_file_id"],
        source_links=json.dumps({0: ""}),
        validity_start_date=None,
        validity_end_date=None,
    )
    binding = AnnexTemporalProjection(
        id=identity,
        index=PublicationIndexSnapshot(
            index_name="fixture",
            index_uuid=str(uuid4()),
            search_settings_id=1,
            model_provider="fixture",
            model_name="fixture",
            vector_dimension=3,
            embedding_config_sha256=publication_digest({"model": "fixture"}),
            multitenant=False,
        ),
        projection=frozen.model_copy(
            update={
                "context_projection_id": str(identity),
                "source_json": json.dumps(source),
            }
        ),
        canonical_base_sha256=context_hash(companion.text),
        derived_role="image_companion",
        dependency_ids=[target.id],
        representation_text=companion.text,
        representation_metadata=metadata,
        reference_date=None,
        effective_start=None,
        effective_end=None,
        semantic_position=1,
    )
    if case != "own_asset":
        with pytest.raises(
            ValueError, match="image companion source evidence mismatch"
        ):
            activate_temporal_projection(
                source_session, user_file_id=file.id, binding=binding
            )
    else:
        activate_temporal_projection(
            source_session, user_file_id=file.id, binding=binding
        )


def test_context_only_history_retains_canonical_identity_and_validity(
    source_session: Session,
) -> None:
    from onyx.db.regulatory_context_projections import (
        activate_context_projection,
        get_effective_context_projection,
        persist_context_view,
        retire_context_projection,
    )
    from onyx.regulatory.amendments.annexes.context_dependencies import context_hash
    from onyx.regulatory.amendments.annexes.models import (
        ContextSourceRange,
        ContextSourceSnapshot,
        FrozenContextProjection,
        PreparedContextView,
    )

    document_set = DocumentSet(name=str(uuid4()), description="", is_up_to_date=True)
    source_session.add(document_set)
    source_session.flush()
    file = _file(source_session, document_set)
    chunk = _chunk(source_session, file, 0, "Unchanged legal text")
    chunk.validity_start_date = date(2026, 1, 1)
    chunk.validity_end_date = date(2027, 1, 1)

    def prepare(context: str) -> list[str]:
        text = context + chunk.text
        config: dict[str, str | int | float | bool | None] = {
            "model": "embedding",
            "dimension": 3,
        }
        snapshot = ContextSourceSnapshot(
            sha256=context_hash(context),
            selector="fixture",
            reference_date=date(2026, 1, 1),
            text=context,
            ordered_ranges=[
                ContextSourceRange(
                    canonical_chunk_id=chunk.id, start=0, end=len(context)
                )
            ],
        )
        projection = FrozenContextProjection(
            canonical_chunk_id=chunk.id,
            source_snapshot_sha256=snapshot.sha256,
            generation_path="normal",
            request_hashes=[],
            embedding_input_sha256=context_hash([text]),
            embedding_config_sha256=context_hash(config),
            embedding_config=config,
            embedding_texts=[text],
            canonical_text_sha256=context_hash(chunk.text),
            metadata_sha256="meta",
            doc_summary=context,
        )
        return persist_context_view(
            source_session,
            user_file_id=file.id,
            view=PreparedContextView(projections=[projection], snapshots=[snapshot]),
        )

    first = prepare("old context")
    assert first == prepare("old context")
    second = prepare("new context")
    assert first != second
    activate_context_projection(
        source_session,
        UUID(first[0]),
        effective_start=date(2026, 1, 1),
        effective_end=None,
    )
    retire_context_projection(
        source_session, UUID(first[0]), effective_end=date(2026, 9, 10)
    )
    activate_context_projection(
        source_session,
        UUID(second[0]),
        effective_start=date(2026, 9, 10),
        effective_end=None,
    )
    before = get_effective_context_projection(
        source_session, chunk.id, as_of_date=date(2026, 9, 9)
    )
    after = get_effective_context_projection(
        source_session, chunk.id, as_of_date=date(2026, 9, 10)
    )
    assert before is not None and before.id == UUID(first[0])
    assert after is not None and after.id == UUID(second[0])
    assert before.canonical_chunk_id == after.canonical_chunk_id == chunk.id
    assert (
        get_effective_context_projection(
            source_session, chunk.id, as_of_date=date(2027, 1, 1)
        )
        is None
    )
    assert chunk.text == "Unchanged legal text"


def test_context_snapshot_cannot_cross_file_scope(source_session: Session) -> None:
    import pytest

    from onyx.db.regulatory_context_projections import persist_context_view
    from onyx.regulatory.amendments.annexes.models import (
        ContextSourceRange,
        ContextSourceSnapshot,
        PreparedContextView,
    )

    document_set = DocumentSet(name=str(uuid4()), description="", is_up_to_date=True)
    source_session.add(document_set)
    source_session.flush()
    file = _file(source_session, document_set)
    other = _file(source_session, document_set)
    foreign = _chunk(source_session, other, 0, "private")
    snapshot = ContextSourceSnapshot(
        sha256="a" * 64,
        selector="fixture",
        reference_date=date.today(),
        text="private",
        ordered_ranges=[
            ContextSourceRange(canonical_chunk_id=foreign.id, start=0, end=7)
        ],
    )
    with pytest.raises(ValueError, match="scope"):
        persist_context_view(
            source_session,
            user_file_id=file.id,
            view=PreparedContextView(snapshots=[snapshot]),
        )


@pytest.mark.parametrize("open_start", [False, True])
@pytest.mark.parametrize("gap_at_anchor", [False, True])
def test_index_qualified_temporal_bindings_keep_two_configurations_and_source_history(
    source_session: Session,
    open_start: bool,
    gap_at_anchor: bool,
) -> None:
    from importlib import import_module

    import pytest

    repository = import_module("onyx.db.regulatory_context_projections")
    assert hasattr(repository, "activate_temporal_projection"), (
        "qualified activation is missing"
    )
    import json

    from onyx.document_index.publication_models import (
        FrozenPublicationProjection,
        PublicationIndexSnapshot,
        publication_digest,
    )
    from onyx.regulatory.amendments.annexes.context_dependencies import context_hash
    from onyx.regulatory.amendments.annexes.models import AnnexTemporalProjection
    from tests.external_dependency_unit.regulatory.test_publication_primitives import (
        frozen_projection,
    )

    group = DocumentSet(name=str(uuid4()), description="", is_up_to_date=True)
    source_session.add(group)
    source_session.flush()
    file = _file(source_session, group)
    canonical = _chunk(source_session, file, 0, "Legal text")
    canonical.validity_start_date = None if open_start else date(2020, 1, 1)
    source_session.flush()
    indices = [
        PublicationIndexSnapshot(
            index_name=f"index-{i}",
            index_uuid=str(uuid4()),
            search_settings_id=i,
            model_provider="fixture",
            model_name="fixture",
            vector_dimension=3,
            embedding_config_sha256=publication_digest({"model": "fixture"}),
            multitenant=False,
        )
        for i in (1, 2)
    ]

    def binding(
        index: PublicationIndexSnapshot,
        ordinal: int,
        image: str,
        start: date | None,
        end: date | None,
    ) -> AnnexTemporalProjection:
        frozen = frozen_projection(file.id, ordinal, "Legal text")
        source = json.loads(frozen.source_json)
        source.update(
            regulatory_chunk_id=canonical.id,
            image_file_id=image,
            source_links=json.dumps({0: ""}),
            validity_start_date=int(
                datetime.combine(start, time.min, timezone.utc).timestamp()
            )
            if start
            else None,
            validity_end_date=int(
                datetime.combine(end, time.min, timezone.utc).timestamp()
            )
            if end
            else None,
            doc_summary="",
            chunk_context="",
        )
        frozen = FrozenPublicationProjection(
            ordinal=ordinal,
            context_projection_id=str(uuid4()),
            source_json=json.dumps(source),
            embedding_inputs=frozen.embedding_inputs,
            embedding_config_json=frozen.embedding_config_json,
        )
        return AnnexTemporalProjection(
            id=UUID(frozen.context_projection_id),
            index=index,
            projection=frozen,
            canonical_base_sha256=context_hash(canonical.text),
            derived_role="canonical",
            dependency_ids=[],
            representation_text=canonical.text,
            representation_metadata={"image_file_id": image},
            reference_date=start,
            effective_start=start,
            effective_end=end,
            semantic_position=canonical.position,
        )

    first = binding(
        indices[0], 0, "old-image", canonical.validity_start_date, date(2026, 1, 1)
    )
    second = binding(indices[0], 2, "new-image", date(2026, 1, 1), None)
    future = binding(indices[1], 3, "future-image", date(2020, 1, 1), None)
    for item in (first, second, future):
        repository.activate_temporal_projection(
            source_session, user_file_id=file.id, binding=item
        )
    from sqlalchemy import update

    from onyx.db.models import RegulatoryTemporalProjection

    # An unrelated projection must never be hydrated by a bounded hit read;
    # selected evidence must still undergo its full integrity validation.
    with source_session.begin_nested() as probe:
        source_session.execute(
            update(RegulatoryTemporalProjection)
            .where(RegulatoryTemporalProjection.id == second.id)
            .values(payload_sha256="corrupt-fixture")
        )
        assert load_public_temporal_bindings(
            source_session,
            file.id,
            index=indices[0],
            as_of_date=date(2025, 1, 1),
            projection_ordinals=(first.projection.ordinal,),
        ) == [first]
        with pytest.raises(ValueError, match="payload changed"):
            load_public_temporal_bindings(
                source_session,
                file.id,
                index=indices[0],
                as_of_date=date(2026, 2, 1),
                projection_ordinals=(second.projection.ordinal,),
            )
        probe.rollback()
    for index, when, expected in (
        (indices[0], date(2025, 1, 1), first),
        (indices[0], date(2026, 2, 1), second),
        (indices[1], date(2026, 2, 1), future),
    ):
        result = repository.get_indexed_temporal_projection(
            source_session, canonical.id, index=index, as_of_date=when
        )
        assert result == expected

        inventory = load_public_temporal_bindings(
            source_session,
            file.id,
            index=index,
            as_of_date=when,
        )
        assert inventory == [expected]
        assert (
            load_public_temporal_bindings(
                source_session,
                file.id,
                index=index,
                as_of_date=when,
                projection_ordinals=(expected.projection.ordinal,),
            )
            == inventory
        )
        assert (
            load_public_temporal_bindings(
                source_session,
                file.id,
                index=index,
                as_of_date=when,
                projection_ordinals=(),
            )
            == []
        )
        from onyx.db.regulatory_chunks import get_bounded_same_provision_siblings
        from onyx.document_index.elasticsearch.elasticsearch_document_index import (
            convert_retrieved_elasticsearch_chunk_to_inference_chunk_uncleaned,
        )
        from onyx.document_index.elasticsearch.schema import DocumentChunkWithoutVectors
        from onyx.regulatory.provision_retrieval import _chunk_from_projection

        own_source = DocumentChunkWithoutVectors.model_validate_json(
            expected.projection.source_json
        )
        seed = convert_retrieved_elasticsearch_chunk_to_inference_chunk_uncleaned(
            own_source, 1, {}
        )
        seed.publication_index = index
        seed.image_file_id = "wrong-seed-image"
        seed.doc_summary = "wrong-seed-summary"
        seed.chunk_context = "wrong-seed-context"
        selected = get_bounded_same_provision_siblings(
            source_session,
            [canonical.id],
            query="Legal",
            as_of_date=when,
            query_indexes={file.id: index},
        )
        assert len(selected) == 1
        hydrated = _chunk_from_projection(selected[0], seed)
        assert hydrated.image_file_id == own_source.image_file_id
        assert hydrated.doc_summary == own_source.doc_summary
        assert hydrated.chunk_context == own_source.chunk_context
        assert hydrated.chunk_id == expected.projection.ordinal
        assert hydrated.structural_position == expected.semantic_position
        assert hydrated.publication_index == index
        assert (
            load_public_temporal_bindings(
                source_session,
                file.id,
                index=index.model_copy(update={"index_uuid": "recreated-index"}),
                as_of_date=when,
            )
            == []
        )
    from onyx.document_index.publication_models import (
        PublicationEncoderAuthority,
        PublicationEncoderReceipt,
    )

    authority = PublicationEncoderAuthority(
        provider="fixture",
        model="fixture",
        effective_dimension=3,
        endpoint_sha256="endpoint",
        deployment_name=None,
        api_version=None,
        normalize=True,
        passage_prefix=None,
    )
    from pydantic import JsonValue

    resolved: dict[str, JsonValue] = {
        "provider": "fixture",
        "dimension": 3,
        "endpoint_sha256": "endpoint",
        "deployment_name": None,
        "api_version": None,
        "normalize": True,
        "passage_prefix": None,
    }
    receipts = tuple(
        PublicationEncoderReceipt(
            configuration_json=json.dumps(config),
            authority=authority,
            resolved_fields=resolved,
            resolution_sha256="a" * 64,
        )
        for config in (
            {"model": "fixture"},
            {"model": "fixture", "formatter": "additional"},
        )
    )
    expanded = indices[0].model_copy(
        update={"encoder_authority": authority, "encoder_receipts": receipts}
    )
    assert (
        repository.get_indexed_temporal_projection(
            source_session, canonical.id, index=expanded, as_of_date=date(2025, 1, 1)
        )
        == first
    )
    assert load_public_temporal_bindings(
        source_session, file.id, index=expanded, as_of_date=date(2025, 1, 1)
    ) == [first]
    incompatible = expanded.model_copy(
        update={
            "encoder_receipts": (receipts[1],),
            "embedding_config_sha256": publication_digest(
                json.loads(receipts[1].configuration_json)
            ),
        }
    )
    with pytest.raises(ValueError, match="receipt"):
        repository.get_indexed_temporal_projection(
            source_session,
            canonical.id,
            index=incompatible,
            as_of_date=date(2025, 1, 1),
        )
    with pytest.raises(ValueError, match="receipt"):
        load_public_temporal_bindings(
            source_session, file.id, index=incompatible, as_of_date=date(2025, 1, 1)
        )
    companion = _chunk(source_session, file, 4, "Image caption")
    companion.validity_start_date = canonical.validity_start_date
    companion.chunk_metadata = {"bound_to_regulatory_chunk_id": canonical.id}
    source_session.flush()
    companions = []
    for ordinal, target in ((4, first), (5, second)):
        identity = uuid4()
        source = json.loads(target.projection.source_json)
        source.update(
            chunk_index=ordinal,
            regulatory_chunk_id=companion.id,
            content=companion.text,
        )
        projection = target.projection.model_copy(
            update={
                "ordinal": ordinal,
                "context_projection_id": str(identity),
                "source_json": json.dumps(source),
            }
        )
        item = target.model_copy(
            update={
                "id": identity,
                "projection": projection,
                "canonical_base_sha256": context_hash(companion.text),
                "derived_role": "image_companion",
                "dependency_ids": [canonical.id],
                "representation_text": companion.text,
                "representation_metadata": {
                    "bound_to_regulatory_chunk_id": canonical.id,
                    "image_file_id": target.representation_metadata["image_file_id"],
                },
                "semantic_position": companion.position,
            }
        )
        if ordinal == 4:
            # A lookup of OLD at the reference date cannot prove the whole window.
            invalid_source = {**source, "validity_end_date": None}
            spanning = item.model_copy(
                update={
                    "effective_end": None,
                    "projection": projection.model_copy(
                        update={"source_json": json.dumps(invalid_source)}
                    ),
                }
            )
            if not gap_at_anchor:
                with pytest.raises(ValueError, match="qualified dependency window"):
                    repository.activate_temporal_projection(
                        source_session, user_file_id=file.id, binding=spanning
                    )
            # No binding at the anchor must not hide known later qualified history.
            gap_index = indices[0].model_copy(update={"index_uuid": str(uuid4())})
            later = second.model_copy(update={"id": uuid4(), "index": gap_index})
            later = later.model_copy(
                update={
                    "projection": later.projection.model_copy(
                        update={"context_projection_id": str(later.id)}
                    )
                }
            )
            repository.activate_temporal_projection(
                source_session, user_file_id=file.id, binding=later
            )
            invalid_source["image_file_id"] = None
            gap = spanning.model_copy(
                update={
                    "index": gap_index,
                    "representation_metadata": {
                        "bound_to_regulatory_chunk_id": canonical.id
                    },
                    "projection": projection.model_copy(
                        update={"source_json": json.dumps(invalid_source)}
                    ),
                }
            )
            if gap_at_anchor:
                with pytest.raises(ValueError, match="qualified dependency window"):
                    repository.activate_temporal_projection(
                        source_session, user_file_id=file.id, binding=gap
                    )
        repository.activate_temporal_projection(
            source_session, user_file_id=file.id, binding=item
        )
        companions.append(item)
    assert (
        repository.get_indexed_temporal_projection(
            source_session, companion.id, index=indices[0], as_of_date=date(2025, 1, 1)
        )
        == companions[0]
    )
    assert (
        repository.get_indexed_temporal_projection(
            source_session, companion.id, index=indices[0], as_of_date=date(2026, 2, 1)
        )
        == companions[1]
    )
    assert companion.chunk_metadata == {"bound_to_regulatory_chunk_id": canonical.id}
    assert canonical.text == "Legal text"
    assert canonical.chunk_metadata.get("image_file_id") is None
    wrong = second.model_copy(update={"id": uuid4(), "canonical_base_sha256": "wrong"})
    with pytest.raises(ValueError, match="canonical"):
        repository.activate_temporal_projection(
            source_session, user_file_id=file.id, binding=wrong
        )


def test_derived_binding_requires_actual_aggregate_dependencies_and_keeps_old_canonical(
    source_session: Session,
) -> None:
    import json
    from datetime import datetime, timezone

    import pytest

    from onyx.db.regulatory_context_projections import activate_temporal_projection
    from onyx.document_index.publication_models import (
        PublicationIndexSnapshot,
        publication_digest,
    )
    from onyx.regulatory.amendments.annexes.context_dependencies import context_hash
    from onyx.regulatory.amendments.annexes.models import AnnexTemporalProjection
    from onyx.regulatory.chunker import hierarchical_aggregate_text
    from tests.external_dependency_unit.regulatory.test_publication_primitives import (
        frozen_projection,
    )

    group = DocumentSet(name=str(uuid4()), description="", is_up_to_date=True)
    source_session.add(group)
    source_session.flush()
    file = _file(source_session, group)
    child = _chunk(source_session, file, 0, "New leaf")
    child.validity_start_date = date(2026, 1, 1)
    old_child = _chunk(source_session, file, 3, "Old leaf")
    old_child.position = 0
    old_child.validity_start_date, old_child.validity_end_date = (
        date(2020, 1, 1),
        date(2026, 1, 1),
    )
    aggregate = _chunk(
        source_session, file, 1, hierarchical_aggregate_text("Root", [old_child.text])
    )
    aggregate.chunk_metadata = {
        "chunk_variant": "hierarchical_aggregate",
        "source_regulatory_chunk_ids": [old_child.id],
        "hierarchy_root_path": ["Root"],
    }
    source_session.flush()
    text = hierarchical_aggregate_text("Root", [child.text])
    frozen = frozen_projection(file.id, 2, text)
    source = json.loads(frozen.source_json)
    source.update(
        regulatory_chunk_id=aggregate.id,
        image_file_id=None,
        source_links=json.dumps({0: ""}),
        doc_summary="",
        chunk_context="",
        validity_start_date=int(datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp()),
        validity_end_date=None,
    )
    identity = uuid4()
    binding = AnnexTemporalProjection(
        id=identity,
        index=PublicationIndexSnapshot(
            index_name="fixture",
            index_uuid=str(uuid4()),
            search_settings_id=1,
            model_provider="fixture",
            model_name="fixture",
            vector_dimension=3,
            embedding_config_sha256=publication_digest({"model": "fixture"}),
            multitenant=False,
        ),
        projection=frozen.model_copy(
            update={
                "context_projection_id": str(identity),
                "source_json": json.dumps(source),
            }
        ),
        canonical_base_sha256=context_hash(aggregate.text),
        derived_role="hierarchical_aggregate",
        dependency_ids=[child.id],
        representation_text=text,
        representation_metadata={
            **aggregate.chunk_metadata,
            "source_regulatory_chunk_ids": [child.id],
        },
        reference_date=date(2026, 1, 1),
        effective_start=date(2026, 1, 1),
        effective_end=None,
        semantic_position=1,
    )
    with pytest.raises(ValueError, match="direct canonical"):
        activate_temporal_projection(
            source_session,
            user_file_id=file.id,
            binding=binding.model_copy(update={"derived_role": "canonical"}),
        )
    with pytest.raises(ValueError, match="aggregate content"):
        activate_temporal_projection(
            source_session,
            user_file_id=file.id,
            binding=binding.model_copy(
                update={"representation_text": "fabricated content"}
            ),
        )
    old_id = uuid4()
    old_source = {
        **source,
        "chunk_index": 3,
        "content": aggregate.text,
        "validity_start_date": int(
            datetime(2020, 1, 1, tzinfo=timezone.utc).timestamp()
        ),
        "validity_end_date": int(datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp()),
    }
    old_binding = binding.model_copy(
        update={
            "id": old_id,
            "projection": binding.projection.model_copy(
                update={
                    "ordinal": 3,
                    "context_projection_id": str(old_id),
                    "source_json": json.dumps(old_source),
                }
            ),
            "representation_text": aggregate.text,
            "representation_metadata": aggregate.chunk_metadata,
            "dependency_ids": [old_child.id],
            "effective_start": date(2020, 1, 1),
            "effective_end": date(2026, 1, 1),
            "reference_date": date(2020, 1, 1),
        }
    )
    activate_temporal_projection(
        source_session, user_file_id=file.id, binding=old_binding
    )
    activate_temporal_projection(source_session, user_file_id=file.id, binding=binding)
    from onyx.db.regulatory_context_projections import get_indexed_temporal_projection

    assert (
        get_indexed_temporal_projection(
            source_session,
            aggregate.id,
            index=binding.index,
            as_of_date=date(2025, 1, 1),
        )
        == old_binding
    )
    assert (
        get_indexed_temporal_projection(
            source_session,
            aggregate.id,
            index=binding.index,
            as_of_date=date(2026, 2, 1),
        )
        == binding
    )

    assert aggregate.text == hierarchical_aggregate_text("Root", [old_child.text])
    assert binding.canonical_base_sha256 != context_hash(binding.representation_text)
