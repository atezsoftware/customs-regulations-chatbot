"""Immutable preapproval artifacts and actual historical encoder evidence."""

import json
from datetime import date
from hashlib import sha256
from typing import TYPE_CHECKING
from uuid import NAMESPACE_URL, UUID, uuid5

from pydantic import JsonValue

from onyx.document_index.publication_models import (
    PublicationEncoderAuthority,
    PublicationEncoderReceipt,
    PublicationIndexSnapshot,
)
from onyx.file_store.file_store import get_default_file_store
from onyx.regulatory.amendments.annexes.context_dependencies import (
    context_hash,
)
from onyx.regulatory.amendments.annexes.models import (
    AnnexCanonicalSnapshot,
    AnnexIndexedBaseline,
    AnnexPublicationPreparation,
    AnnexPublicationProjectionPlan,
    AnnexPublicationReview,
    PreparedContextView,
)
from onyx.regulatory.amendments.annexes.publication import (
    exact_reusable_projection,
)
from onyx.regulatory.amendments.annexes.publication_representations import (
    _as_date,
    _epoch,
)

if TYPE_CHECKING:
    from onyx.natural_language_processing.search_nlp_models import EmbeddingModel


def _encoder_receipt(
    configuration: dict[str, str | int | float | bool | None], *, resolution: str
) -> PublicationEncoderReceipt:
    """Resolve only known synchronous facts; batch native defaults are explicit."""
    resolved: dict[str, JsonValue] = {}
    if configuration.get("transport") == "openrouter_batch":
        # This transport sends raw provider vectors with no local normalization,
        # prefix, deployment or API-version arguments (see OpenRouterBatchClient).
        resolved = {
            "normalize": None,
            "passage_prefix": None,
            "deployment_name": None,
            "api_version": None,
        }
    facts = {**configuration, **resolved}
    required = (
        "provider",
        "model",
        "dimension",
        "endpoint_sha256",
        "normalize",
        "passage_prefix",
        "deployment_name",
        "api_version",
    )
    if any(key not in facts for key in required):
        raise ValueError("encoder authority is unresolved")
    authority = PublicationEncoderAuthority.model_validate(
        {key: facts[key] for key in required if key != "dimension"}
        | {"effective_dimension": facts.get("reduced_dimension") or facts["dimension"]}
    )
    return PublicationEncoderReceipt(
        configuration_json=json.dumps(configuration),
        authority=authority,
        resolution_sha256=resolution,
        resolved_fields=resolved,
    )


def _validate_physical_encoder_authority(
    actual: AnnexIndexedBaseline,
    index: PublicationIndexSnapshot,
) -> None:
    """Known model-space authority cannot change within one physical index."""
    previous = actual.evidence.index
    verified = actual.evidence.frozen_projection
    if verified is None or previous.index_uuid != index.index_uuid:
        return
    configuration = json.loads(verified.embedding_config_json)
    known_facts = {
        key: value
        for key, value in configuration.items()
        if key in PublicationEncoderAuthority.model_fields
    }
    if "dimension" in configuration:
        known_facts["effective_dimension"] = (
            configuration.get("reduced_dimension") or configuration["dimension"]
        )
    previous_authority = previous.effective_authority()
    if previous_authority is None:
        try:
            previous_authority = _encoder_receipt(
                configuration,
                resolution="0" * 64,
            ).effective_authority()
        except ValueError:
            # Incomplete legacy receipts remain unverified for missing facts.
            pass
    if (
        (previous.model_name, previous.vector_dimension)
        != (index.model_name, index.vector_dimension)
        or previous_authority is None
        and previous.model_provider != index.model_provider
        or previous_authority is not None
        and previous_authority != index.effective_authority()
        or previous_authority is None
        and index.encoder_authority is not None
        and any(
            value != index.encoder_authority.model_dump()[key]
            for key, value in known_facts.items()
        )
    ):
        raise ValueError(
            "incompatible encoder authority requires a distinct physical index"
        )


def read_publication_preparation(
    review: AnnexPublicationReview,
) -> AnnexPublicationPreparation:
    with get_default_file_store().read_file(
        review.artifact_file_id, mode="b"
    ) as handle:
        payload = handle.read()
    if (
        not isinstance(payload, bytes)
        or len(payload) != review.artifact_byte_count
        or sha256(payload).hexdigest() != review.artifact_sha256
    ):
        raise ValueError("frozen publication artifact changed")
    result = AnnexPublicationPreparation.model_validate_json(payload)
    if (
        result.scope != review.scope
        or result.indexes != review.indexes
        or [item.id for item in result.projections] != review.projection_ids
        or result.counts != review.counts
    ):
        raise ValueError("publication artifact scope differs from review")
    index_ids = {index.index_uuid for index in result.indexes}
    if len(index_ids) != len(result.indexes) or result.reserved_ordinals != sorted(
        set(result.reserved_ordinals)
    ):
        raise ValueError("invalid frozen index/reservation inventory")
    if len(
        {(plan.index.index_uuid, plan.ordinal) for plan in result.projections}
    ) != len(result.projections):
        raise ValueError("duplicate frozen projection ordinal")
    for index in result.indexes:
        live = {plan.ordinal for plan in result.projections if plan.index == index}
        retained = [
            json.loads(item.evidence.source_json)["chunk_index"]
            for item in result.retained
            if item.evidence.index.matches_temporal_index(index)
        ]
        if (
            len(retained) != len(set(retained))
            or set(retained) & live
            or not set(retained).issubset(result.reserved_ordinals)
        ):
            raise ValueError("invalid frozen retained inventory")
        live.update(retained)
        if (
            set(result.retired_ordinals.get(index.index_uuid, []))
            != set(result.reserved_ordinals) - live
        ):
            raise ValueError("incomplete frozen tombstone inventory")
    if any(
        key != context_hash(view.model_dump(mode="json"))
        for key, view in result.views.items()
    ):
        raise ValueError("frozen context dependency artifact changed")
    for plan in result.projections:
        if (
            plan.view_sha256 not in result.views
            or plan.context not in result.views[plan.view_sha256].projections
        ):
            raise ValueError("frozen context view dependency missing")
        source = json.loads(plan.source_template_json)
        if (
            plan.index not in result.indexes
            or plan.ordinal not in result.reserved_ordinals
            or source.get("document_id") != str(result.user_file_id)
            or source.get("chunk_index") != plan.ordinal
            or source.get("regulatory_chunk_id") != plan.row.id
        ):
            raise ValueError("frozen projection scope/identity mismatch")
        if source.get("validity_start_date") != _epoch(
            plan.effective_start
        ) or source.get("validity_end_date") != _epoch(plan.effective_end):
            raise ValueError("frozen source interval mismatch")
        if (
            plan.context.embedding_input_sha256
            != context_hash(plan.context.embedding_texts)
            or plan.context.embedding_config_sha256
            != context_hash(plan.context.embedding_config)
            or not plan.index.accepts_encoder_configuration(
                plan.context.embedding_config
            )
        ):
            raise ValueError("frozen encoder input/config mismatch")
        if plan.reuse_from is not None and (
            list(plan.reuse_from.embedding_inputs) != plan.context.embedding_texts
            or json.loads(plan.reuse_from.embedding_config_json)
            != plan.context.embedding_config
            or not any(
                item.evidence.frozen_projection == plan.reuse_from
                for item in result.indexed_baseline
            )
        ):
            raise ValueError("frozen vector reuse has no exact actual evidence")
    return result


def _historical_plan(
    actual: AnnexIndexedBaseline,
    row: AnnexCanonicalSnapshot,
    index: PublicationIndexSnapshot,
    configuration: dict[str, str | int | float | bool | None],
    cutoff: date,
    embedding_model: "EmbeddingModel",
) -> AnnexPublicationProjectionPlan | None:
    _validate_physical_encoder_authority(actual, index)
    raw = json.loads(actual.evidence.source_json)
    source = {
        key: value for key, value in raw.items() if not key.startswith("publication_")
    }
    from onyx.utils.text_processing import remove_invalid_unicode_chars

    representation = actual.binding.representation_text if actual.binding else row.text
    expected_source = remove_invalid_unicode_chars(
        source.get("doc_summary", "")
        + representation
        + source.get("chunk_context", "")
        + (source.get("metadata_suffix") or "")
    )
    if source["content"] != expected_source:
        raise ValueError(
            "historical indexed content does not match its canonical/derived source authority"
        )
    canonical_base_sha256 = context_hash(row.text)
    if actual.binding is not None:
        if actual.binding.canonical_base_sha256 != canonical_base_sha256:
            raise ValueError("historical canonical base changed")
        bound = actual.binding.projection
        if (
            context_hash(json.loads(bound.source_json)) != context_hash(source)
            or actual.evidence.frozen_projection is not None
            and (
                bound.embedding_inputs
                != actual.evidence.frozen_projection.embedding_inputs
                or json.loads(bound.embedding_config_json)
                != json.loads(actual.evidence.frozen_projection.embedding_config_json)
            )
        ):
            raise ValueError("dated binding no longer matches actual indexed proof")
    start, end = (
        _as_date(source.get("validity_start_date")),
        _as_date(source.get("validity_end_date")),
    )
    if (
        actual.reference_date is None
        or actual.binding is None
        or actual.binding.context is None
    ):
        return None
    start = (
        max(value for value in (start, row.validity_start_date) if value is not None)
        if start or row.validity_start_date
        else None
    )
    end = (
        min(value for value in (end, row.validity_end_date) if value is not None)
        if end or row.validity_end_date
        else None
    )
    if start is not None and start >= cutoff:
        return None
    end = min(end, cutoff) if end is not None else cutoff
    if start is not None and start >= end:
        return None
    verified = actual.evidence.frozen_projection
    if verified:
        inputs = list(verified.embedding_inputs)
        encoder_config = json.loads(verified.embedding_config_json)
        # Different encoders cannot reuse vectors, but retain the exact old input.
        try:
            compatible = (
                _encoder_receipt(
                    encoder_config, resolution="0" * 64
                ).effective_authority()
                == index.effective_authority()
            )
        except ValueError:
            compatible = False
        if not compatible:
            encoder_config = {
                **configuration,
                "formatter": "historical-frozen-inputs-v1",
            }
    else:
        if source.get("metadata_suffix"):
            raise ValueError(
                "historical encoder input cannot be reconstructed from keyword-only metadata"
            )
        inputs = [
            source["content"],
            *([source["title"]] if source.get("title") else []),
        ]
        encoder_config = {**configuration, "formatter": "historical-stored-source-v1"}
    from onyx.regulatory.amendments.annexes.context_dependencies import (
        freeze_encoder_inputs,
    )

    if not verified and configuration.get("formatter") == "normal-v1":
        # Reconstruct the normal formatter from actual stored context and title.
        # This is a NEW conservative encoding plan, never old-vector proof.
        encoder_config = dict(configuration)
    if encoder_config != (
        json.loads(verified.embedding_config_json) if verified else None
    ):
        if configuration.get("transport") != "openrouter_batch":
            inputs, _ = freeze_encoder_inputs(
                inputs,
                embedding_model,
                model_dim=index.vector_dimension,
                formatter=str(encoder_config["formatter"]),
            )
    context = actual.binding.context.model_copy(
        update={
            "embedding_texts": inputs,
            "embedding_input_sha256": context_hash(inputs),
            "embedding_config": encoder_config,
            "embedding_config_sha256": context_hash(encoder_config),
        }
    )
    source.pop("content_vector")
    source.pop("title_vector", None)
    source["validity_start_date"], source["validity_end_date"] = (
        _epoch(start),
        _epoch(end),
    )
    same_index = actual.evidence.index.index_uuid == index.index_uuid
    identity = (
        UUID(verified.context_projection_id)
        if same_index and verified and _is_uuid(verified.context_projection_id)
        else uuid5(
            NAMESPACE_URL,
            f"annex-historical:{index.index_uuid}:{row.user_file_id}:{source['chunk_index']}",
        )
    )
    return AnnexPublicationProjectionPlan(
        id=identity,
        index=index,
        ordinal=source["chunk_index"] if same_index else -1,
        row=row.model_copy(
            update={
                "text": actual.binding.representation_text,
                "metadata": actual.binding.representation_metadata,
                "position": actual.binding.semantic_position,
                "heading_path": source.get("heading_path", row.heading_path),
            }
        ),
        canonical_base_sha256=canonical_base_sha256,
        context=context,
        view_sha256=context_hash(
            PreparedContextView(projections=[context]).model_dump(mode="json")
        ),
        effective_start=start,
        effective_end=end,
        reference_date=actual.reference_date,
        source_template_json=json.dumps(source),
        reuse_from=exact_reusable_projection(
            context, actual.evidence, predecessor_id=None
        ),
        reason="verified_historical_input"
        if verified
        else "legacy_encoder_provenance_unavailable",
    )


def _is_uuid(value: str) -> bool:
    try:
        UUID(value)
    except ValueError:
        return False
    return True


def _uncovered_history(
    plan: AnnexPublicationProjectionPlan, retained: list[AnnexPublicationProjectionPlan]
) -> list[AnnexPublicationProjectionPlan]:
    """Fill only absent target-index intervals from known source-index history."""
    intervals = [(plan.effective_start, plan.effective_end)]
    for existing in retained:
        if existing.row.id != plan.row.id:
            continue
        remaining = []
        for start, end in intervals:
            if (end or date.max) <= (existing.effective_start or date.min) or (
                existing.effective_end or date.max
            ) <= (start or date.min):
                remaining.append((start, end))
                continue
            if (start or date.min) < (existing.effective_start or date.min):
                remaining.append((start, existing.effective_start))
            if (existing.effective_end or date.max) < (end or date.max):
                remaining.append((existing.effective_end, end))
        intervals = remaining
    result = []
    for start, end in intervals:
        if (start, end) == (plan.effective_start, plan.effective_end):
            result.append(plan)
            continue
        source = json.loads(plan.source_template_json)
        source["validity_start_date"], source["validity_end_date"] = (
            _epoch(start),
            _epoch(end),
        )
        result.append(
            plan.model_copy(
                update={
                    "id": uuid5(plan.id, f"{start}:{end}"),
                    "ordinal": -1,
                    "effective_start": start,
                    "effective_end": end,
                    "source_template_json": json.dumps(source),
                }
            )
        )
    return result
