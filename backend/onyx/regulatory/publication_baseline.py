"""Adopt verified existing search representations without generating embeddings."""

import json
from typing import TYPE_CHECKING, Literal
from uuid import uuid4

from elasticsearch import Elasticsearch
from pydantic import BaseModel, Field

from onyx.db.regulatory_publication import PublicationStore
from onyx.document_index.elasticsearch.publication import FencedPublicationIndex
from onyx.document_index.publication_models import (
    FileOwnership,
    IndexedProjectionEvidence,
    ObservedPublicationProjection,
    PublicationIndexSnapshot,
    matches_indexed_evidence,
    publication_digest,
    publication_list_digest,
    publication_source,
)
from onyx.regulatory.amendments.annexes.context_dependencies import context_hash
from onyx.regulatory.amendments.annexes.models import (
    AnnexCanonicalSnapshot,
    AnnexTemporalProjection,
)
from onyx.regulatory.amendments.annexes.publication_representations import (
    _epoch,
    validate_temporal_interval,
)
from onyx.regulatory.amendments.annexes.selective_impact import (
    aggregate_membership_is_valid,
    image_membership_is_valid,
    recover_source_membership,
    source_ids,
    source_window_covers,
)
from onyx.regulatory.writer_publication_models import WriterPublicationManifest
from onyx.utils.text_processing import remove_invalid_unicode_chars
from shared_configs.configs import MULTI_TENANT

if TYPE_CHECKING:
    from onyx.db.models import SearchSettings
    from onyx.db.regulatory_writer_publication import OwnedWriterInputs


def observed_runtime_digest(settings: "SearchSettings") -> str:
    """Current observation guard, explicitly not a historical encoder receipt."""
    return publication_digest(
        {
            "observation_schema": 1,
            "provider": str(settings.provider_type) if settings.provider_type else "",
            "model": settings.model_name,
            "dimension": settings.final_embedding_dim,
            "endpoint_sha256": context_hash(settings.api_url),
            "deployment": settings.deployment_name,
            "api_version": settings.api_version,
            "normalize": settings.normalize,
            "passage_prefix": settings.passage_prefix,
            "query_prefix": settings.query_prefix,
        }
    )


def observed_index_snapshot(
    settings: "SearchSettings", index_uuid: str
) -> PublicationIndexSnapshot:
    return PublicationIndexSnapshot(
        index_name=settings.index_name,
        index_uuid=index_uuid,
        search_settings_id=settings.id,
        model_provider=str(settings.provider_type) if settings.provider_type else "",
        model_name=settings.model_name,
        vector_dimension=settings.final_embedding_dim,
        embedding_config_sha256=observed_runtime_digest(settings),
        multitenant=MULTI_TENANT,
    )


def observed_baseline_binding(
    evidence: IndexedProjectionEvidence,
    canonical: list[AnnexCanonicalSnapshot],
) -> AnnexTemporalProjection:
    source = publication_source(evidence.source_json)
    rows = {row.id: row for row in canonical}
    row = rows.get(str(source.get("regulatory_chunk_id")))
    if row is None or (
        source.get("document_id") != row.user_file_id
        or source.get("chunk_index") != row.projection_ordinal
        or (
            source.get("heading_path") != row.heading_path
            and not (source.get("heading_path") is None and row.heading_path == [])
        )
        or source.get("validity_start_date") != _epoch(row.validity_start_date)
        or source.get("validity_end_date") != _epoch(row.validity_end_date)
    ):
        raise ValueError("baseline source identity, heading or legal interval mismatch")
    expected = remove_invalid_unicode_chars(
        str(source.get("doc_summary") or "")
        + row.text
        + str(source.get("chunk_context") or "")
        + str(source.get("metadata_suffix") or "")
    )
    if source.get("content") != expected:
        raise ValueError("baseline indexed content differs from canonical source")
    from onyx.regulatory.chunk_evidence import chunk_evidence

    citation = chunk_evidence(row.metadata)
    if source.get("image_file_id") != citation.image_file_id or source.get(
        "source_links"
    ) != (json.dumps(citation.source_links) if citation.source_links else None):
        raise ValueError("baseline source/image evidence mismatch")
    metadata = dict(row.metadata)
    aggregate = metadata.get("chunk_variant") == "hierarchical_aggregate"
    image = metadata.get("bound_to_regulatory_chunk_id") is not None
    members = source_ids(row)
    if aggregate:
        from onyx.regulatory.chunker import (
            hierarchical_aggregate_root_label,
            hierarchical_aggregate_text,
        )

        root = metadata.get("hierarchy_root_path")
        if not isinstance(root, list) or not root or not isinstance(root[-1], str):
            raise ValueError("baseline aggregate root unavailable: " + row.id)
        if not aggregate_membership_is_valid(row, rows):
            members = recover_source_membership(canonical).get(row.id, [])
        if (
            not members
            or hierarchical_aggregate_text(
                hierarchical_aggregate_root_label(metadata, row.text),
                [rows[identifier].text for identifier in members],
            )
            != row.text
        ):
            raise ValueError("baseline aggregate membership unresolved: " + row.id)
        metadata["source_regulatory_chunk_ids"] = members
    if image and any(identifier not in rows for identifier in members):
        members = recover_source_membership(canonical).get(row.id, [])
        if len(members) == 1:
            metadata["bound_to_regulatory_chunk_id"] = members[0]
            metadata["source_regulatory_chunk_ids"] = []
    if image and (
        len(members) != 1
        or members[0] not in rows
        or not image_membership_is_valid(row, rows[members[0]])
    ):
        raise ValueError("baseline image membership unresolved: " + row.id)
    for identifier in members:
        dependency = rows.get(identifier)
        if (
            dependency is None
            or dependency.user_file_id != row.user_file_id
            or not source_window_covers(dependency, row)
        ):
            raise ValueError(
                "baseline dependency identity or interval mismatch: " + row.id
            )
    identifier = uuid4()
    projection = ObservedPublicationProjection.observe(
        context_projection_id=str(identifier),
        source_json=json.dumps(source),
        observed_index=evidence.index,
    )
    binding = AnnexTemporalProjection(
        id=identifier,
        index=evidence.index,
        projection=projection,
        canonical_base_sha256=context_hash(row.text),
        derived_role="hierarchical_aggregate"
        if aggregate
        else "image_companion"
        if image
        else "canonical",
        dependency_ids=members,
        representation_text=row.text,
        representation_metadata=metadata,
        context=None,
        reference_date=None,
        effective_start=row.validity_start_date,
        effective_end=row.validity_end_date,
        semantic_position=row.position,
    )
    validate_temporal_interval(
        binding,
        canonical_status=row.status,
        canonical_start=row.validity_start_date,
        canonical_end=row.validity_end_date,
    )
    return binding


def validate_baseline_metadata_repair(
    inputs: "OwnedWriterInputs", after: list[AnnexCanonicalSnapshot]
) -> set[str]:
    """Only unqualified canonical metadata may change during baseline adoption."""
    before = {row.id: row for row in inputs.canonical}
    if len(after) != len(before) or {row.id for row in after} != set(before):
        raise ValueError("baseline metadata repair changed canonical inventory")
    changed: set[str] = set()
    for row in after:
        original = before[row.id]
        if row.model_dump(exclude={"metadata", "heading_path"}) != original.model_dump(
            exclude={"metadata", "heading_path"}
        ):
            raise ValueError(
                "baseline metadata repair changed canonical content or identity"
            )
        if row != original:
            changed.add(row.id)
    qualified = {
        str(json.loads(binding.projection.source_json)["regulatory_chunk_id"])
        for binding in inputs.bindings
    }
    if changed & qualified:
        raise ValueError("baseline metadata repair cannot rewrite qualified history")
    return changed


def prepare_owned_baseline(
    owner: FileOwnership,
    client: Elasticsearch,
    inputs: "OwnedWriterInputs",
    *,
    origin_proposal_id: int | None = None,
    canonical_after: list[AnnexCanonicalSnapshot] | None = None,
) -> WriterPublicationManifest | None:
    """Prepare the whole active index inventory under the ordinary writer lease."""
    changed = (
        validate_baseline_metadata_repair(inputs, canonical_after)
        if canonical_after is not None
        else set()
    )
    canonical = canonical_after if canonical_after is not None else inputs.canonical
    authority = PublicationStore(owner.scope)
    indexes: list[PublicationIndexSnapshot] = []
    bindings: list[AnnexTemporalProjection] = []
    revisions = dict(inputs.revisions)
    added = False
    for settings in inputs.settings:
        if not client.indices.exists(index=settings.index_name):
            if settings.status.is_current():
                raise ValueError("baseline current physical index is unavailable")
            continue
        physical = client.indices.get(index=settings.index_name)
        if set(physical) != {settings.index_name}:
            raise ValueError("baseline requires a concrete physical index")
        index = observed_index_snapshot(
            settings, physical[settings.index_name]["settings"]["index"]["uuid"]
        )
        previous = [
            binding
            for binding in inputs.bindings
            if binding.index.index_name == settings.index_name
        ]
        if any(not index.matches_temporal_index(binding.index) for binding in previous):
            raise ValueError("baseline physical index or model changed")
        adapter = FencedPublicationIndex(client, index)
        existing = adapter.existing_ordinals(authority.reservations(owner))
        authority.reserve_existing_ordinals(owner, existing)
        evidence = adapter.inventory_evidence(authority.reservations(owner))
        preflight = audit_baseline_inventory(
            canonical,
            list(evidence),
            previous,
            require_complete=settings.status.is_current(),
        )
        if preflight.issues:
            raise ValueError("baseline preflight failed: " + preflight.issues[0].detail)
        by_ordinal = {
            json.loads(item.source_json)["chunk_index"]: item for item in evidence
        }
        if len(by_ordinal) != len(evidence):
            raise ValueError("baseline duplicate indexed ordinal")
        if not evidence and not previous and settings.status.is_future():
            indexes.append(index)
            continue
        expected = {binding.projection.ordinal: binding for binding in previous}
        for ordinal, binding in expected.items():
            actual = by_ordinal.get(ordinal)
            if actual is None or not matches_indexed_evidence(
                binding.projection, actual
            ):
                raise ValueError("baseline qualified source differs from actual index")
        observed_ids = {
            str(json.loads(item.source_json)["regulatory_chunk_id"])
            for item in evidence
        }
        required_ids = {
            row.id
            for row in canonical
            if row.source == "indexed" or row.status == "active"
        }
        if settings.status.is_current() and not required_ids <= observed_ids:
            raise ValueError(
                f"baseline indexed source missing for {len(required_ids - observed_ids)} canonical chunks"
            )
        bindings.extend(previous)
        for ordinal, actual in by_ordinal.items():
            if ordinal in expected:
                continue
            if (
                actual.frozen_projection is not None
                or actual.observed_projection is not None
            ):
                raise ValueError(
                    "baseline protected source has lost its temporal binding"
                )
            binding = observed_baseline_binding(actual, canonical)
            bindings.append(binding)
            canonical_id = str(json.loads(actual.source_json)["regulatory_chunk_id"])
            if canonical_id not in changed:
                revisions[binding.id] = inputs.canonical_revisions[canonical_id]
            added = True
        # Keep actual accepted encoder authority for mixed files.
        qualified = next(
            (
                binding.index
                for binding in previous
                if binding.index.encoder_authority is not None
            ),
            None,
        )
        indexes.append(qualified or index)
    if not added:
        return None
    return WriterPublicationManifest(
        id=uuid4(),
        scope=owner.scope,
        user_file_id=owner.user_file_id,
        kind="baseline",
        baseline_origin_proposal_id=origin_proposal_id,
        index_state_sha256=inputs.index_state_sha256,
        canonical_before_sha256=publication_digest(
            [row.model_dump(mode="json") for row in inputs.canonical]
        ),
        canonical_after=canonical_after,
        indexes=indexes,
        bindings=bindings,
        previous_binding_ids=[
            binding.id
            for binding in inputs.bindings
            if binding.index.index_uuid in {index.index_uuid for index in indexes}
        ],
        canonical_revisions=revisions,
    )


def ensure_owned_baseline(
    owner: FileOwnership,
    client: Elasticsearch,
    *,
    origin_proposal_id: int | None = None,
) -> FileOwnership:
    from onyx.db.regulatory_writer_publication import load_owned_writer_inputs
    from onyx.regulatory.amendments.annexes.publication_execution import (
        publication_heartbeat,
    )
    from onyx.regulatory.writer_publication import execute_writer_publication

    with publication_heartbeat(owner) as lost:
        manifest = prepare_owned_baseline(
            owner,
            client,
            load_owned_writer_inputs(owner),
            origin_proposal_id=origin_proposal_id,
        )
        if lost.is_set():
            raise ValueError("baseline publication ownership heartbeat lost")
    if manifest is not None:
        execute_writer_publication(owner, client, manifest)
        return PublicationStore(owner.scope).advance_after_publication(owner)
    return owner


class BaselineIssue(BaseModel):
    code: str
    chunk_id: str | None = None
    detail: str


class BaselineInventoryAudit(BaseModel):
    state: Literal["ready", "legacy", "partial", "unresolved", "unindexed"]
    canonical_count: int
    indexed_count: int
    retained_count: int
    binding_count: int
    source_sha256: str
    vectors_sha256: str
    issues: list[BaselineIssue] = Field(default_factory=list)


def audit_baseline_inventory(
    canonical: list[AnnexCanonicalSnapshot],
    evidence: list[IndexedProjectionEvidence],
    bindings: list[AnnexTemporalProjection],
    *,
    require_complete: bool = True,
) -> BaselineInventoryAudit:
    """Pure, read-only preflight shared by operators and the normal writer."""
    issues: list[BaselineIssue] = []
    indexed_ids: set[str] = set()
    source_order: list[tuple[str, int]] = []
    required = {
        row.id for row in canonical if row.source == "indexed" or row.status == "active"
    }
    expected = {binding.projection.ordinal: binding for binding in bindings}
    if len(expected) != len(bindings):
        issues.append(
            BaselineIssue(
                code="duplicate_binding",
                detail="Multiple bindings have the same ordinal",
            )
        )
    seen: set[int] = set()
    retained = 0
    pending_dependencies: dict[str, set[str]] = {}
    for position, item in enumerate(evidence):
        source = publication_source(item.source_json)
        ordinal = source.get("chunk_index")
        identifier = str(source.get("regulatory_chunk_id"))
        indexed_ids.add(identifier)
        source_order.append((str(ordinal), position))
        if type(ordinal) is not int or ordinal in seen:
            issues.append(
                BaselineIssue(
                    code="invalid_ordinal",
                    chunk_id=identifier,
                    detail="Indexed ordinal is missing or duplicated",
                )
            )
            continue
        seen.add(ordinal)
        binding = expected.get(ordinal)
        try:
            if binding is not None:
                if not matches_indexed_evidence(binding.projection, item):
                    raise ValueError("Qualified source differs from actual index")
            elif (
                item.frozen_projection is not None
                or item.observed_projection is not None
            ):
                raise ValueError("Protected indexed source has lost its binding")
            else:
                adopted = observed_baseline_binding(item, canonical)
                pending_dependencies[identifier] = set(adopted.dependency_ids)
            retained += 1
        except ValueError as error:
            issues.append(
                BaselineIssue(
                    code="source_integrity", chunk_id=identifier, detail=str(error)
                )
            )
    while pending_dependencies:
        ready = {
            identifier
            for identifier, dependencies in pending_dependencies.items()
            if not dependencies.intersection(pending_dependencies)
        }
        if not ready:
            issues.append(
                BaselineIssue(
                    code="dependency_cycle",
                    detail="Derived source dependencies form a cycle",
                    chunk_id=sorted(pending_dependencies)[0],
                )
            )
            break
        for identifier in ready:
            del pending_dependencies[identifier]
    for ordinal in sorted(expected.keys() - seen):
        issues.append(
            BaselineIssue(
                code="missing_bound_projection",
                detail=f"Binding ordinal {ordinal} has no indexed representation",
            )
        )
    if require_complete:
        issues = [
            BaselineIssue(
                code="missing_indexed_source",
                chunk_id=missing,
                detail="Canonical source has no indexed representation",
            )
            for missing in sorted(required - indexed_ids)
        ] + issues
    # Preserve the historical lexicographic ordinal order, including equal-key order.
    ordered = sorted(source_order)
    return BaselineInventoryAudit(
        state="unresolved"
        if issues
        else "unindexed"
        if not evidence
        else "ready"
        if len(bindings) == len(evidence)
        else "partial"
        if bindings
        else "legacy",
        canonical_count=len(canonical),
        indexed_count=len(evidence),
        retained_count=retained,
        binding_count=len(bindings),
        source_sha256=publication_list_digest(
            publication_source(evidence[position].source_json)
            for _, position in ordered
        ),
        vectors_sha256=publication_list_digest(
            {
                "ordinal": source.get("chunk_index"),
                "id": source.get("regulatory_chunk_id"),
                "content": source.get("content_vector"),
                "title": source.get("title_vector"),
            }
            for source in (
                publication_source(evidence[position].source_json)
                for _, position in ordered
            )
        ),
        issues=issues,
    )
