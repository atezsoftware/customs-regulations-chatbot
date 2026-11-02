"""One source-preserving reconsideration of a verified intake abstention."""

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from copy import deepcopy
from hashlib import sha256
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from onyx.regulatory.amendments.annexes.context_dependencies import context_hash
from onyx.regulatory.indexing_jobs.vertex_batch import VertexBatchRequest
from onyx.regulatory.labeling.provider import (
    MAX_REQUEST_BYTES,
    MAX_RESPONSE_BYTES,
    LabelingOutcome,
    TaxonomyDefinition,
    build_labeling_request,
    parse_labeling_batch_output,
    validate_labeling_response,
)

ABSTENTION_REPAIR_KEY = "intake_abstention_repair"
_REPAIR_VERSION = 1
_SHA_PATTERN = r"^[0-9a-f]{64}$"
_INSTRUCTION = """Reconsider this one earlier abstention using exactly the frozen target, taxonomy and interpretation context. The earlier abstention may remain correct; do not force a label or a successful result.
The target may be a partial provision. Use the existing context only to resolve the target's references and governing scope. Return only facets supported by the target in that scope. An unsupported effect, sector or other facet does not require abstaining from a separately supported topic. Never inherit another chunk's labels or assign a topic from context alone.
An effect label requires an unambiguous governing predicate and proof that it governs this target; a nearby rule, neighboring label or shared topic is insufficient. Preserve conditions, exceptions and opposite legal effects.
Every returned evidence_quote must remain an exact nonempty substring of target.text and must support its classification in context; never quote interpretation_context. Do not invent categories or change taxonomy definitions.
A formatting-only separator, empty grid or empty form field may have no applicable label: return labels=[] and abstained=false only when no supplied label applies. If the target's meaning or scope is still insufficient or contradictory, return labels=[] and abstained=true. Return the unchanged normal JSON schema. This is the sole semantic reconsideration; do not request another repair."""


def _sha(value: str) -> str:
    return sha256(value.encode()).hexdigest()


def _json_sha(value: object) -> str:
    return sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


class AbstentionRepair(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    request: VertexBatchRequest
    base_request: VertexBatchRequest
    chunk_id: str
    canonical_text_sha256: str = Field(pattern=_SHA_PATTERN)
    context_text_sha256: str = Field(pattern=_SHA_PATTERN)
    taxonomy_sha256: str = Field(pattern=_SHA_PATTERN)
    provider_binding: dict[str, JsonValue]
    provider_binding_sha256: str = Field(pattern=_SHA_PATTERN)
    model: str
    rejected_response: str
    rejected_response_sha256: str = Field(pattern=_SHA_PATTERN)


class AbstentionRepairOutput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    request_hash: str = Field(pattern=_SHA_PATTERN)
    response: str
    response_sha256: str = Field(pattern=_SHA_PATTERN)
    output_row_sha256: str = Field(pattern=_SHA_PATTERN)


class AbstentionRepairAudit(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    version: Literal[1] = 1
    method: Literal["intake_abstention_reconsideration"] = (
        "intake_abstention_reconsideration"
    )
    attempt: Literal[1] = 1
    chunk_id: str
    base_request_hash: str = Field(pattern=_SHA_PATTERN)
    rejected_response: str
    rejected_response_sha256: str = Field(pattern=_SHA_PATTERN)
    repair_request_hash: str = Field(pattern=_SHA_PATTERN)
    output_uri: str = Field(min_length=1, max_length=2048)
    output_row_sha256: str = Field(pattern=_SHA_PATTERN)
    output_response: str
    output_response_sha256: str = Field(pattern=_SHA_PATTERN)
    canonical_text_sha256: str = Field(pattern=_SHA_PATTERN)
    context_text_sha256: str = Field(pattern=_SHA_PATTERN)
    taxonomy_sha256: str = Field(pattern=_SHA_PATTERN)
    provider_binding_sha256: str = Field(pattern=_SHA_PATTERN)
    model: str
    outcome: LabelingOutcome
    audit_sha256: str = Field(pattern=_SHA_PATTERN)


def build_abstention_repair(
    *,
    base_request: VertexBatchRequest,
    rejected_response: str,
    text: str,
    chunk_id: str,
    source_hash: str,
    taxonomy: TaxonomyDefinition,
    provider_binding: Mapping[str, object],
) -> AbstentionRepair:
    """The caller supplies the body of an already correlated, complete STOP ACK."""
    original = validate_labeling_response(
        rejected_response, text=text, taxonomy=taxonomy
    )
    if not original.abstained or original.labels:
        raise ValueError("Semantic repair requires an explicit empty-label abstention")
    if context_hash(text) != source_hash:
        raise ValueError("Semantic repair requires the unchanged canonical target")
    try:
        prompt = json.loads(base_request.prompt)
        context = prompt["interpretation_context"]
        if not isinstance(context, str):
            raise ValueError("Invalid context")
        rebuilt = build_labeling_request(
            chunk_id=chunk_id,
            text=text,
            context=context,
            taxonomy=taxonomy,
            source_hash=source_hash,
        )
        binding = json.loads(
            json.dumps(dict(provider_binding), ensure_ascii=False, allow_nan=False)
        )
    except (KeyError, TypeError, ValueError):
        raise ValueError(
            "Semantic repair does not match its frozen base request"
        ) from None
    if base_request != rebuilt:
        raise ValueError("Semantic repair does not match its frozen base request")
    if (
        not isinstance(binding, dict)
        or binding.get("transport") != "vertex_direct_v1"
        or not isinstance(binding.get("model"), str)
        or not binding["model"].strip()
    ):
        raise ValueError("Semantic repair requires the frozen direct intake binding")
    prompt[ABSTENTION_REPAIR_KEY] = {
        "version": _REPAIR_VERSION,
        "attempt": 1,
        "base_request_hash": base_request.request_hash,
        "rejected_response": rejected_response,
        "rejected_response_sha256": _sha(rejected_response),
        "instruction": _INSTRUCTION,
    }
    request = VertexBatchRequest(
        prompt=json.dumps(
            prompt, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ),
        system_instruction=(base_request.system_instruction or "")
        + "\n\n"
        + _INSTRUCTION,
        generation_config=deepcopy(base_request.generation_config),
    )
    if (
        len(
            json.dumps(
                request.to_generate_content_request(), ensure_ascii=False
            ).encode()
        )
        > MAX_REQUEST_BYTES
    ):
        raise ValueError(
            "Semantic repair exceeds its request budget; source was not truncated"
        )
    return AbstentionRepair(
        request=request,
        base_request=base_request,
        chunk_id=chunk_id,
        canonical_text_sha256=source_hash,
        context_text_sha256=_sha(context),
        taxonomy_sha256=taxonomy.version_hash,
        provider_binding=cast(dict[str, JsonValue], binding),
        provider_binding_sha256=_json_sha(binding),
        model=binding["model"],
        rejected_response=rejected_response,
        rejected_response_sha256=_sha(rejected_response),
    )


def _current_repair(
    repair: AbstentionRepair, *, text: str, taxonomy: TaxonomyDefinition
) -> None:
    current = build_abstention_repair(
        base_request=repair.base_request,
        rejected_response=repair.rejected_response,
        text=text,
        chunk_id=repair.chunk_id,
        source_hash=repair.canonical_text_sha256,
        taxonomy=taxonomy,
        provider_binding=repair.provider_binding,
    )
    if current != repair:
        raise ValueError("The frozen semantic repair bindings changed")


def parse_abstention_repair_output(
    lines: Iterable[str],
    *,
    repair: AbstentionRepair,
    on_progress: Callable[[], None],
) -> AbstentionRepairOutput:
    """Accept one complete STOP candidate echoing this exact repair request."""
    accepted: AbstentionRepairOutput | None = None
    used_bytes = 0
    for line in lines:
        on_progress()
        if not isinstance(line, str):
            raise ValueError("Semantic repair output is not text")
        used_bytes += len(line.encode())
        if used_bytes > MAX_RESPONSE_BYTES:
            raise ValueError("Semantic repair output exceeds its size limit")
        if not line.strip():
            continue
        if accepted is not None:
            raise ValueError("Semantic repair requires exactly one response")
        payload = json.loads(line)
        if (
            not isinstance(payload, dict)
            or payload.get("key") != repair.request.request_hash
            or payload.get("request") != repair.request.to_generate_content_request()
        ):
            raise ValueError("Semantic repair output differs from its frozen request")
        response = payload.get("response")
        candidates = response.get("candidates") if isinstance(response, dict) else None
        if not isinstance(candidates, list) or len(candidates) != 1:
            raise ValueError("Semantic repair requires one complete candidate")
        parsed = parse_labeling_batch_output(
            iter([line]), {repair.request.request_hash}
        )[repair.request.request_hash]
        if parsed.error is not None or parsed.context is None:
            raise ValueError("Semantic repair provider response was rejected")
        accepted = AbstentionRepairOutput(
            request_hash=repair.request.request_hash,
            response=parsed.context,
            response_sha256=_sha(parsed.context),
            output_row_sha256=_sha(line),
        )
    if accepted is None:
        raise ValueError("Semantic repair output is missing")
    return accepted


def validate_abstention_repair(
    response: str,
    *,
    repair: AbstentionRepair,
    text: str,
    taxonomy: TaxonomyDefinition,
) -> LabelingOutcome:
    _current_repair(repair, text=text, taxonomy=taxonomy)
    # An ongoing abstention remains explicit; the caller must retain its failure.
    return validate_labeling_response(response, text=text, taxonomy=taxonomy)


def build_abstention_repair_audit(
    *,
    output: AbstentionRepairOutput,
    repair: AbstentionRepair,
    output_uri: str,
    text: str,
    taxonomy: TaxonomyDefinition,
) -> AbstentionRepairAudit:
    outcome = validate_abstention_repair(
        output.response, repair=repair, text=text, taxonomy=taxonomy
    )
    if (
        output.request_hash != repair.request.request_hash
        or output.response_sha256 != _sha(output.response)
    ):
        raise ValueError("Semantic repair output bindings changed")
    values = {
        "version": 1,
        "method": "intake_abstention_reconsideration",
        "attempt": 1,
        "chunk_id": repair.chunk_id,
        "base_request_hash": repair.base_request.request_hash,
        "rejected_response": repair.rejected_response,
        "rejected_response_sha256": repair.rejected_response_sha256,
        "repair_request_hash": repair.request.request_hash,
        "output_uri": output_uri,
        "output_row_sha256": output.output_row_sha256,
        "output_response": output.response,
        "output_response_sha256": output.response_sha256,
        "canonical_text_sha256": repair.canonical_text_sha256,
        "context_text_sha256": repair.context_text_sha256,
        "taxonomy_sha256": repair.taxonomy_sha256,
        "provider_binding_sha256": repair.provider_binding_sha256,
        "model": repair.model,
        "outcome": outcome.model_dump(mode="json"),
    }
    return AbstentionRepairAudit.model_validate(
        {**values, "audit_sha256": _json_sha(values)}
    )


def validate_stored_abstention_repair_audit(
    audit: AbstentionRepairAudit | Mapping[str, object],
    *,
    base_request: VertexBatchRequest,
    text: str,
    chunk_id: str,
    source_hash: str,
    taxonomy: TaxonomyDefinition,
    provider_binding: Mapping[str, object],
    labels: Sequence[str],
    assignments: Sequence[Mapping[str, object]],
) -> AbstentionRepairAudit:
    stored = AbstentionRepairAudit.model_validate(audit)
    values = stored.model_dump(mode="json", exclude={"audit_sha256"})
    if stored.audit_sha256 != _json_sha(values):
        raise ValueError("The stored semantic repair audit changed")
    repair = build_abstention_repair(
        base_request=base_request,
        rejected_response=stored.rejected_response,
        text=text,
        chunk_id=chunk_id,
        source_hash=source_hash,
        taxonomy=taxonomy,
        provider_binding=provider_binding,
    )
    output = AbstentionRepairOutput(
        request_hash=stored.repair_request_hash,
        response=stored.output_response,
        response_sha256=stored.output_response_sha256,
        output_row_sha256=stored.output_row_sha256,
    )
    rebuilt = build_abstention_repair_audit(
        output=output,
        repair=repair,
        output_uri=stored.output_uri,
        text=text,
        taxonomy=taxonomy,
    )
    if rebuilt != stored:
        raise ValueError(
            "The stored semantic repair does not match current frozen data"
        )
    if (
        stored.outcome.abstained
        or [item.label_id for item in stored.outcome.labels] != list(labels)
        or [item.model_dump(mode="json") for item in stored.outcome.labels]
        != [dict(item) for item in assignments]
    ):
        raise ValueError("Semantic repair does not prove these accepted assignments")
    return stored
