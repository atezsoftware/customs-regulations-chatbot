"""Repair only failed intake quotes by selecting immutable target-source spans."""

import json
import re
from collections.abc import Callable, Iterable
from hashlib import sha256

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from onyx.regulatory.amendments.annexes.context_dependencies import context_hash
from onyx.regulatory.indexing_jobs.vertex_batch import VertexBatchRequest
from onyx.regulatory.labeling.intake_quote_resolution import (
    resolve_intake_labeling_response,
)
from onyx.regulatory.labeling.provider import (
    MAX_LABELS,
    MAX_REQUEST_BYTES,
    MAX_RESPONSE_BYTES,
    LabelingOutcome,
    TaxonomyDefinition,
    parse_labeling_batch_output,
    validate_labeling_response,
)

REPAIR_VERSION = 1
MAX_CATALOG_ENTRIES = 4096
QUOTE_ERRORS = frozenset(
    {
        "Label evidence must be an exact quote from the canonical target",
        "Intake label quote has no unique whitespace-only source match",
    }
)


class EvidenceSpan(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    evidence_id: str
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    text: str = Field(min_length=1, max_length=1024)


class QuoteRepair(BaseModel):
    model_config = ConfigDict(frozen=True)

    request: VertexBatchRequest
    original: LabelingOutcome
    retained_quotes: dict[str, str]
    repair_label_ids: frozenset[str]
    catalog: list[EvidenceSpan]
    source_sha256: str
    rejected_response_sha256: str
    whitespace_audits: list[dict[str, object]]


class EvidenceSelection(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    label_id: str
    evidence_id: str


class RepairOutcome(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    labels: list[EvidenceSelection] = Field(max_length=MAX_LABELS)
    abstained: bool


def evidence_catalog(text: str) -> list[EvidenceSpan]:
    """Include exact whole/sentence/line spans and cover every source character."""
    if not text.strip():
        raise ValueError("Quote repair requires the unchanged canonical target")
    source_hash = sha256(text.encode()).hexdigest()
    spans: set[tuple[int, int]] = set()

    def add(start: int, end: int) -> None:
        if 0 < end - start <= 1024:
            spans.add((start, end))
            if len(spans) > MAX_CATALOG_ENTRIES:
                raise ValueError("Intake quote repair evidence catalog is too large")

    # Nonoverlapping slices retain complete coverage even for an oversized line.
    start = 0
    while start < len(text):
        end = min(start + 1024, len(text))
        if end < len(text):
            boundaries = list(re.finditer(r"\s+", text[start:end]))
            if boundaries and boundaries[-1].end() > 0:
                end = start + boundaries[-1].end()
        add(start, end)
        start = end
    for pattern in (r"[^\n]*(?:\n|$)", r".+?(?:[.!?;](?=\s)|$)"):
        for match in re.finditer(
            pattern, text, flags=re.DOTALL if ".+" in pattern else 0
        ):
            add(*match.span())
    return [
        EvidenceSpan(
            evidence_id="e_"
            + sha256(
                f"intake-label-quote-v{REPAIR_VERSION}:{source_hash}:{start}:{end}:".encode()
                + text[start:end].encode()
            ).hexdigest(),
            start=start,
            end=end,
            text=text[start:end],
        )
        for start, end in sorted(spans)
    ]


def build_quote_repair(
    *,
    base_request: VertexBatchRequest,
    rejected_response: str,
    text: str,
    chunk_id: str,
    source_hash: str,
    taxonomy: TaxonomyDefinition,
) -> QuoteRepair:
    if len(rejected_response.encode()) > MAX_RESPONSE_BYTES:
        raise ValueError("Labeling response exceeds its size limit")
    original = LabelingOutcome.model_validate_json(rejected_response)
    identifiers = [assignment.label_id for assignment in original.labels]
    allowed = {label.id for label in taxonomy.labels}
    if (
        original.abstained
        or not identifiers
        or len(set(identifiers)) != len(identifiers)
        or set(identifiers) - allowed
    ):
        raise ValueError("Only known, unique, non-abstained labels permit quote repair")
    retained: dict[str, str] = {}
    failed: set[str] = set()
    whitespace_audits: list[dict[str, object]] = []
    for assignment in original.labels:
        if not assignment.evidence_quote.strip():
            raise ValueError("Blank label evidence does not permit quote repair")
        if assignment.evidence_quote in text:
            retained[assignment.label_id] = assignment.evidence_quote
            continue
        try:
            resolved, audits = resolve_intake_labeling_response(
                LabelingOutcome(labels=[assignment], abstained=False).model_dump_json(),
                text=text,
                taxonomy=taxonomy,
            )
        except ValueError as error:
            if str(error) not in QUOTE_ERRORS:
                raise
            failed.add(assignment.label_id)
        else:
            retained[assignment.label_id] = resolved.labels[0].evidence_quote
            whitespace_audits.extend(audits)
    if not failed:
        raise ValueError("No failed canonical quote requires repair")
    prompt = json.loads(base_request.prompt)
    target = prompt.get("target") if isinstance(prompt, dict) else None
    if (
        not isinstance(prompt, dict)
        or not isinstance(target, dict)
        or target.get("chunk_id") != chunk_id
        or target.get("text") != text
        or re.fullmatch(r"[0-9a-f]{64}", source_hash) is None
        or target.get("source_hash") != source_hash
        or context_hash(text) != source_hash
        or prompt.get("taxonomy_hash") != taxonomy.version_hash
        or prompt.get("taxonomy") != taxonomy.model_dump()
        or not isinstance(prompt.get("interpretation_context"), str)
        or base_request.generation_config is None
    ):
        raise ValueError("Quote repair does not match its frozen base request")
    catalog = evidence_catalog(text)
    source_hash = sha256(text.encode()).hexdigest()
    rejected_hash = sha256(rejected_response.encode()).hexdigest()
    prompt["quote_repair"] = {
        "version": REPAIR_VERSION,
        "attempt": 1,
        "original_request_hash": base_request.request_hash,
        "rejected_response_sha256": rejected_hash,
        "source_text_sha256": source_hash,
        "error": "Evidence was not an exact canonical quote. Do not paraphrase or normalize source words.",
        "rejected_response": rejected_response,
        "repair_label_ids": sorted(failed),
        "instruction": "Select exact source evidence IDs only for repair_label_ids. Preserve their exact set. Return abstained=false. Existing valid quotes and all original label IDs remain unchanged on the server.",
        "evidence_catalog": [entry.model_dump() for entry in catalog],
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
                        "label_id": {"type": "string", "enum": sorted(failed)},
                        "evidence_id": {
                            "type": "string",
                            "enum": [entry.evidence_id for entry in catalog],
                        },
                    },
                    "required": ["label_id", "evidence_id"],
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
        + "\nThis is a single quote-only repair. Respond with evidence_id selections using the repair schema; never rewrite source text or change label IDs or abstention.",
        generation_config={
            **base_request.generation_config,
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
            "Intake quote repair exceeds its request byte budget; source was not truncated"
        )
    return QuoteRepair(
        request=request,
        original=original,
        retained_quotes=retained,
        repair_label_ids=frozenset(failed),
        catalog=catalog,
        source_sha256=source_hash,
        rejected_response_sha256=rejected_hash,
        whitespace_audits=whitespace_audits,
    )


def validate_quote_repair(
    response: str, *, repair: QuoteRepair, text: str, taxonomy: TaxonomyDefinition
) -> LabelingOutcome:
    if sha256(text.encode()).hexdigest() != repair.source_sha256:
        raise ValueError("The canonical quote repair source changed")
    if len(response.encode()) > MAX_RESPONSE_BYTES:
        raise ValueError("Labeling response exceeds its size limit")
    selected = RepairOutcome.model_validate_json(response)
    identifiers = [assignment.label_id for assignment in selected.labels]
    if (
        selected.abstained
        or len(set(identifiers)) != len(identifiers)
        or set(identifiers) != repair.repair_label_ids
    ):
        raise ValueError("Quote repair changed the original label set or abstention")
    catalog = {entry.evidence_id: entry for entry in evidence_catalog(text)}
    if list(catalog.values()) != repair.catalog:
        raise ValueError("The canonical evidence catalog changed")
    quotes = dict(repair.retained_quotes)
    for assignment in selected.labels:
        entry = catalog.get(assignment.evidence_id)
        if entry is None:
            raise ValueError("Quote repair selected unknown canonical evidence")
        quotes[assignment.label_id] = text[entry.start : entry.end]
    canonical = repair.original.model_copy(
        update={
            "labels": [
                assignment.model_copy(
                    update={"evidence_quote": quotes[assignment.label_id]}
                )
                for assignment in repair.original.labels
            ]
        }
    )
    return validate_labeling_response(
        canonical.model_dump_json(), text=text, taxonomy=taxonomy
    )


def parse_quote_repair_output(
    lines: Iterable[str], *, repair: QuoteRepair, on_progress: Callable[[], None]
) -> str:
    """Require a sole complete response echoing the exact repair request."""
    accepted: str | None = None
    used_bytes = 0
    for line in lines:
        on_progress()
        if not isinstance(line, str):
            raise ValueError("Quote repair output is not text")
        used_bytes += len(line.encode())
        if used_bytes > MAX_RESPONSE_BYTES:
            raise ValueError("Quote repair output exceeds its size limit")
        if not line.strip():
            continue
        if accepted is not None:
            raise ValueError("Quote repair output must contain exactly one response")
        payload = json.loads(line)
        if (
            not isinstance(payload, dict)
            or payload.get("key") != repair.request.request_hash
            or payload.get("request") != repair.request.to_generate_content_request()
        ):
            raise ValueError("Quote repair output does not echo its frozen request")
        response = payload.get("response")
        candidates = response.get("candidates") if isinstance(response, dict) else None
        if not isinstance(candidates, list) or len(candidates) != 1:
            raise ValueError("Quote repair requires exactly one complete candidate")
        result = parse_labeling_batch_output(
            iter([line]), {repair.request.request_hash}
        )[repair.request.request_hash]
        if result.error is not None or result.context is None:
            raise ValueError("Quote repair provider response was rejected")
        accepted = result.context
    if accepted is None:
        raise ValueError("Quote repair output is missing its response")
    return accepted


def quote_repair_audits(
    response: str, *, repair: QuoteRepair
) -> list[dict[str, object]]:
    """Describe validated selections without persisting source or changing assignments."""
    selected = RepairOutcome.model_validate_json(response)
    catalog = {entry.evidence_id: entry for entry in repair.catalog}
    original = {
        entry.label_id: entry.evidence_quote for entry in repair.original.labels
    }
    selections: list[dict[str, object]] = [
        {
            "label_id": assignment.label_id,
            "version": REPAIR_VERSION,
            "method": "canonical_evidence_selector",
            "original_quote_sha256": sha256(
                original[assignment.label_id].encode()
            ).hexdigest(),
            "resolved_quote_sha256": sha256(
                catalog[assignment.evidence_id].text.encode()
            ).hexdigest(),
            "source_text_sha256": repair.source_sha256,
            "source_start_char": catalog[assignment.evidence_id].start,
            "source_end_char": catalog[assignment.evidence_id].end,
            "evidence_id": assignment.evidence_id,
        }
        for assignment in selected.labels
    ]
    return [*repair.whitespace_audits, *selections]
