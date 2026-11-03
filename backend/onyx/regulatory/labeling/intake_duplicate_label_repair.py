"""One model selection among already grounded duplicate-label quote candidates."""

import json
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from hashlib import sha256
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from onyx.regulatory.amendments.annexes.context_dependencies import context_hash
from onyx.regulatory.indexing_jobs.vertex_batch import VertexBatchRequest
from onyx.regulatory.labeling.duplicate_ack_evidence import (
    SHA,
    DuplicateAckEvidence,
    complete_ack_response,
)
from onyx.regulatory.labeling.intake_quote_resolution import (
    resolve_intake_labeling_response,
)
from onyx.regulatory.labeling.provider import (
    MAX_LABELS,
    MAX_REQUEST_BYTES,
    MAX_RESPONSE_BYTES,
    LabelAssignment,
    LabelingOutcome,
    TaxonomyDefinition,
    build_labeling_request,
    validate_labeling_response,
)

DUPLICATE_REPAIR_KEY = "intake_duplicate_label_repair"
_INSTRUCTION = (
    "This is the sole duplicate-label selection attempt. For each duplicated label ID, "
    "select exactly one supplied candidate_id. Do not return fixed unique label IDs, "
    "new IDs, quotes or evidence from outside the supplied candidates. Return "
    "abstained=false. Missing, duplicate or invalid selections fail the entire repair."
)


def _sha(text: str) -> str:
    return sha256(text.encode()).hexdigest()


def _digest(value: object) -> str:
    return _sha(
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
    )


class DuplicateQuoteCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    candidate_id: str
    original_index: int = Field(ge=0, lt=MAX_LABELS)
    original_quote_sha256: str = Field(pattern=SHA)
    canonical_quote: str = Field(min_length=1, max_length=1024)
    whitespace_audits: list[dict[str, JsonValue]]


class DuplicateLabelRepair(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    request: VertexBatchRequest
    base_request: VertexBatchRequest
    original: LabelingOutcome
    rejected_response: str
    rejected_response_sha256: str = Field(pattern=SHA)
    chunk_id: str
    canonical_text_sha256: str = Field(pattern=SHA)
    context_text_sha256: str = Field(pattern=SHA)
    taxonomy_sha256: str = Field(pattern=SHA)
    provider_binding_sha256: str = Field(pattern=SHA)
    provider_binding: dict[str, JsonValue]
    model: str
    candidates: dict[str, list[DuplicateQuoteCandidate]]
    retained_quotes: dict[str, str]


class DuplicateSelection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    label_id: str
    candidate_id: str


class DuplicateSelections(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    labels: list[DuplicateSelection] = Field(min_length=1, max_length=MAX_LABELS)
    abstained: bool


class DuplicateSourceBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    document_set_id: int
    user_file_id: str = Field(pattern=r"^[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12}$")
    source_asset_id: str = Field(
        pattern=r"^[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12}$"
    )
    intake_id: str = Field(pattern=r"^[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12}$")
    package_id: str = Field(pattern=r"^[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12}$")
    owner_id: str = Field(pattern=r"^[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12}$")
    source_file_id: str
    original_file_id: str
    raw_sha256: str = Field(pattern=SHA)
    frozen_context_sources_sha256: str = Field(pattern=SHA)
    byte_count: int = Field(gt=0)


class DuplicateLabelRepairAudit(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    version: Literal[1]
    method: Literal["intake_duplicate_label_selection"]
    attempt: Literal[1]
    chunk_id: str
    base_request_hash: str = Field(pattern=SHA)
    repair_request_hash: str = Field(pattern=SHA)
    rejected_response: str = Field(max_length=MAX_RESPONSE_BYTES)
    rejected_response_sha256: str = Field(pattern=SHA)
    selection_response: str = Field(max_length=MAX_RESPONSE_BYTES)
    selection_response_sha256: str = Field(pattern=SHA)
    canonical_text_sha256: str = Field(pattern=SHA)
    context_text_sha256: str = Field(pattern=SHA)
    taxonomy_sha256: str = Field(pattern=SHA)
    provider_binding_sha256: str = Field(pattern=SHA)
    model: str
    source_binding: DuplicateSourceBinding
    base_ack: DuplicateAckEvidence
    repair_ack: DuplicateAckEvidence
    outcome: LabelingOutcome
    audit_sha256: str = Field(pattern=SHA)

    @model_validator(mode="before")
    @classmethod
    def exact_version_and_attempt(cls, value: object) -> object:
        if isinstance(value, Mapping):
            values = cast(Mapping[str, object], value)
            if any(
                type(values.get(field)) is not int or values.get(field) != 1
                for field in ("version", "attempt")
            ):
                raise ValueError(
                    "Duplicate-label audit requires explicit version1 and attempt1"
                )
        return value


def known_duplicate_shape(response: str, taxonomy: TaxonomyDefinition) -> bool:
    if len(response.encode()) > MAX_RESPONSE_BYTES:
        return False
    try:
        original = LabelingOutcome.model_validate_json(response)
    except ValueError:
        return False
    ids = [entry.label_id for entry in original.labels]
    return (
        not original.abstained
        and bool(ids)
        and len(ids) > len(set(ids))
        and not set(ids) - {label.id for label in taxonomy.labels}
    )


def build_duplicate_label_repair(
    *,
    base_request: VertexBatchRequest,
    rejected_response: str,
    text: str,
    chunk_id: str,
    source_hash: str,
    taxonomy: TaxonomyDefinition,
    provider_binding: Mapping[str, object],
) -> DuplicateLabelRepair:
    if not known_duplicate_shape(rejected_response, taxonomy):
        raise ValueError("Only nonempty known duplicate labels permit selection")
    original = LabelingOutcome.model_validate_json(rejected_response)
    prompt = json.loads(base_request.prompt)
    context = prompt.get("interpretation_context") if isinstance(prompt, dict) else None
    if (
        not isinstance(context, str)
        or source_hash != context_hash(text)
        or base_request
        != build_labeling_request(
            chunk_id=chunk_id,
            text=text,
            context=context,
            taxonomy=taxonomy,
            source_hash=source_hash,
        )
        or provider_binding.get("transport") != "vertex_direct_v1"
        or not isinstance(provider_binding.get("model"), str)
        or not str(provider_binding["model"]).strip()
    ):
        raise ValueError("Duplicate-label selection differs from the frozen base")
    counts = Counter(entry.label_id for entry in original.labels)
    candidates: dict[str, list[DuplicateQuoteCandidate]] = {}
    retained: dict[str, str] = {}
    rejected_sha = _sha(rejected_response)
    for index, entry in enumerate(original.labels):
        resolved, audits = resolve_intake_labeling_response(
            LabelingOutcome(labels=[entry], abstained=False).model_dump_json(),
            text=text,
            taxonomy=taxonomy,
        )
        quote = resolved.labels[0].evidence_quote
        if counts[entry.label_id] == 1:
            retained[entry.label_id] = quote
            continue
        candidate = DuplicateQuoteCandidate(
            candidate_id="c_"
            + _digest(
                [
                    base_request.request_hash,
                    rejected_sha,
                    source_hash,
                    entry.label_id,
                    index,
                    quote,
                ]
            ),
            original_index=index,
            original_quote_sha256=_sha(entry.evidence_quote),
            canonical_quote=quote,
            whitespace_audits=cast(list[dict[str, JsonValue]], audits),
        )
        candidates.setdefault(entry.label_id, []).append(candidate)
    prompt[DUPLICATE_REPAIR_KEY] = {
        "version": 1,
        "attempt": 1,
        "base_request_hash": base_request.request_hash,
        "rejected_response": rejected_response,
        "rejected_response_sha256": rejected_sha,
        "fixed_label_ids": sorted(retained),
        "duplicate_groups": {
            label: [entry.model_dump() for entry in values]
            for label, values in candidates.items()
        },
        "instruction": _INSTRUCTION,
    }
    schema: dict[str, JsonValue] = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "labels": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "label_id": {"type": "string", "enum": sorted(candidates)},
                        "candidate_id": {
                            "type": "string",
                            "enum": [
                                entry.candidate_id
                                for values in candidates.values()
                                for entry in values
                            ],
                        },
                    },
                    "required": ["label_id", "candidate_id"],
                },
            },
            "abstained": {"type": "boolean"},
        },
        "required": ["labels", "abstained"],
    }
    request = VertexBatchRequest(
        prompt=json.dumps(
            prompt, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ),
        system_instruction=(base_request.system_instruction or "")
        + "\n"
        + _INSTRUCTION,
        generation_config={
            **(base_request.generation_config or {}),
            "responseJsonSchema": schema,
        },
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
            "Duplicate-label selection exceeds its untruncated request budget"
        )
    return DuplicateLabelRepair(
        request=request,
        base_request=base_request,
        original=original,
        rejected_response=rejected_response,
        rejected_response_sha256=rejected_sha,
        chunk_id=chunk_id,
        canonical_text_sha256=source_hash,
        context_text_sha256=_sha(context),
        taxonomy_sha256=taxonomy.version_hash,
        provider_binding_sha256=_digest(dict(provider_binding)),
        provider_binding=cast(
            dict[str, JsonValue],
            json.loads(json.dumps(dict(provider_binding), allow_nan=False)),
        ),
        model=str(provider_binding["model"]),
        candidates=candidates,
        retained_quotes=retained,
    )


def validate_duplicate_selection(
    response: str,
    *,
    repair: DuplicateLabelRepair,
    text: str,
    taxonomy: TaxonomyDefinition,
) -> LabelingOutcome:
    if (
        len(response.encode()) > MAX_RESPONSE_BYTES
        or context_hash(text) != repair.canonical_text_sha256
    ):
        raise ValueError("Duplicate-label selection source or response budget changed")
    rebuilt = build_duplicate_label_repair(
        base_request=repair.base_request,
        rejected_response=repair.rejected_response,
        text=text,
        chunk_id=repair.chunk_id,
        source_hash=repair.canonical_text_sha256,
        taxonomy=taxonomy,
        provider_binding=repair.provider_binding,
    )
    if rebuilt != repair:
        raise ValueError(
            "The duplicate-label candidate or fixed quote authority changed"
        )
    selected = DuplicateSelections.model_validate_json(response)
    ids = [entry.label_id for entry in selected.labels]
    if (
        selected.abstained
        or len(ids) != len(set(ids))
        or set(ids) != set(repair.candidates)
    ):
        raise ValueError("Duplicate-label selection changed its exact group set")
    quotes = dict(repair.retained_quotes)
    for entry in selected.labels:
        matches = [
            candidate
            for candidate in repair.candidates[entry.label_id]
            if candidate.candidate_id == entry.candidate_id
        ]
        if len(matches) != 1:
            raise ValueError("Duplicate-label selection chose a foreign candidate")
        quotes[entry.label_id] = matches[0].canonical_quote
    ordered = list(dict.fromkeys(entry.label_id for entry in repair.original.labels))
    outcome = LabelingOutcome(
        labels=[
            LabelAssignment(label_id=label, evidence_quote=quotes[label])
            for label in ordered
        ],
        abstained=False,
    )
    return validate_labeling_response(
        outcome.model_dump_json(), text=text, taxonomy=taxonomy
    )


def parse_duplicate_output(
    lines: Iterable[str],
    *,
    request: VertexBatchRequest,
    on_progress: Callable[[], None],
) -> tuple[str, str]:
    accepted: tuple[str, str] | None = None
    used = 0
    for line in lines:
        on_progress()
        if not isinstance(line, str):
            raise ValueError("Duplicate-label output is not text")
        used += len(line.encode())
        if used > MAX_RESPONSE_BYTES:
            raise ValueError("Duplicate-label output exceeds its byte budget")
        if not line.strip():
            continue
        if accepted is not None:
            raise ValueError("Duplicate-label output contains multiple responses")
        accepted = complete_ack_response(line, request), line
    if accepted is None:
        raise ValueError("Duplicate-label output is absent")
    return accepted


def build_duplicate_repair_audit(
    *,
    repair: DuplicateLabelRepair,
    selection_response: str,
    base_ack: DuplicateAckEvidence,
    repair_ack: DuplicateAckEvidence,
    source_binding: DuplicateSourceBinding,
    text: str,
    taxonomy: TaxonomyDefinition,
) -> DuplicateLabelRepairAudit:
    if (
        base_ack.file_id != base_ack.expected_file_id()
        or repair_ack.file_id != repair_ack.expected_file_id()
        or base_ack.request_hash != repair.base_request.request_hash
        or repair_ack.request_hash != repair.request.request_hash
        or base_ack.binding_fingerprint != repair_ack.binding_fingerprint
        or base_ack.response_sha256 != repair.rejected_response_sha256
        or repair_ack.response_sha256 != _sha(selection_response)
    ):
        raise ValueError(
            "The duplicate-label ACK pointers differ from actual responses"
        )
    outcome = validate_duplicate_selection(
        selection_response, repair=repair, text=text, taxonomy=taxonomy
    )
    values = {
        "version": 1,
        "method": "intake_duplicate_label_selection",
        "attempt": 1,
        "chunk_id": repair.chunk_id,
        "base_request_hash": repair.base_request.request_hash,
        "repair_request_hash": repair.request.request_hash,
        "rejected_response": repair.rejected_response,
        "rejected_response_sha256": repair.rejected_response_sha256,
        "selection_response": selection_response,
        "selection_response_sha256": _sha(selection_response),
        "canonical_text_sha256": repair.canonical_text_sha256,
        "context_text_sha256": repair.context_text_sha256,
        "taxonomy_sha256": repair.taxonomy_sha256,
        "provider_binding_sha256": repair.provider_binding_sha256,
        "model": repair.model,
        "source_binding": source_binding.model_dump(mode="json"),
        "base_ack": base_ack.model_dump(mode="json"),
        "repair_ack": repair_ack.model_dump(mode="json"),
        "outcome": outcome.model_dump(mode="json"),
    }
    return DuplicateLabelRepairAudit.model_validate(
        {**values, "audit_sha256": _digest(values)}
    )
