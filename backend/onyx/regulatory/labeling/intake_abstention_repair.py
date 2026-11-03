"""One source-preserving reconsideration of a verified intake abstention."""

import json
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from decimal import Decimal
from hashlib import sha256
from typing import Any, Literal, cast

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
_NUMBERED_CAPTION = re.compile(
    r"\((?P<number>[1-9]\d{0,2})\) Yedek parçaların bulunabilirliği:"
)
_DIMENSION_CAPTION = re.compile(
    r"(?P<size>[1-9]\d{0,3}(?:[.,]\d{1,2})?) mm\n(?P<title>Etiket):"
)
_DIMENSION = re.compile(r"([1-9]\d{0,3}(?:[.,]\d{1,2})?) mm")
_DESIGN_SECTION = re.compile(
    r"(?P<number>[1-9]\d{0,2})\. (?:(?:Yoğuşturuculu|Yoğuşturucusuz) çamaşır "
    r"kurutma makineleri|Çamaşır kurutma makineleri) için etiket tasarımı"
)
_DIAGRAM_TOKEN = (
    r"(?:[0-9]{1,4}(?:[.,][0-9]{1,2})?|[0-9]{4}/[0-9]{1,5}|mm|kg|kWh|/|"
    r"[A-G]|ENERG|ENERGY|TRADEMARK|MODEL|IDENTIFIER|XYZ|XYdB|XY%|XY,Z|X:YZ)"
)
_DIAGRAM_LINE = re.compile(rf"{_DIAGRAM_TOKEN}(?: {_DIAGRAM_TOKEN}){{0,15}}")
_CAPTION_SNAPSHOT_FIELDS = (
    "position",
    "heading_path",
    "chunk_type",
    "chunk_metadata",
    "validity_start_date",
    "validity_end_date",
)


@dataclass(frozen=True)
class CaptionRepairSource:
    chunk_id: str
    file_id: str
    text: str
    canonical_text_sha256: str
    snapshot: dict[str, Any]
    current_text: str
    current_snapshot: dict[str, Any]
    active: bool


def legacy_annex_caption_shape(text: str) -> bool:
    """A routing hint only; current source and neighboring witnesses are required."""
    return len(text) <= 180 and bool(
        _NUMBERED_CAPTION.fullmatch(text) or _DIMENSION_CAPTION.fullmatch(text)
    )


def _caption_metadata(source: CaptionRepairSource) -> Mapping[str, Any]:
    metadata = source.snapshot.get("chunk_metadata")
    return cast(Mapping[str, Any], metadata) if isinstance(metadata, Mapping) else {}


def _caption_fact(
    source: CaptionRepairSource, field: str, quote: str, scope: str
) -> bool:
    audit = _caption_metadata(source).get("intake_legal_metadata")
    if not isinstance(audit, Mapping) or audit.get("input_stale") is not False:
        return False
    fields = audit.get("fields")
    fact = fields.get(field) if isinstance(fields, Mapping) else None
    return (
        isinstance(fact, Mapping)
        and fact.get("stale") is False
        and all(
            fact.get(key) == value
            for key, value in {
                "source": "llm",
                "basis": "explicit",
                "evidence_quote": quote,
                "evidence_scope": scope,
                "source_chunk_id": source.chunk_id,
            }.items()
        )
    )


def legacy_annex_caption_repair_candidate(
    *, chunk_id: str, sources: Sequence[CaptionRepairSource]
) -> bool:
    """Recognize only witnessed annex captions; never classify their content.

    Callers also verify the owned raw ingestion receipt and the complete current
    labeling context before provider submission and durable result acceptance.
    """
    by_id = {source.chunk_id: source for source in sources}
    target = by_id.get(chunk_id)
    if (
        target is None
        or len(sources) != len(by_id)
        or not 1 <= len(sources) <= 40
        or not legacy_annex_caption_shape(target.text)
    ):
        return False
    members = target.snapshot.get("context_member_ids")
    if (
        not isinstance(members, list)
        or not all(isinstance(value, str) for value in members)
        or len(members) != len(set(members))
        or set(members) != set(by_id)
    ):
        return False
    if any(
        not source.active
        or source.file_id != target.file_id
        or source.text != source.current_text
        or context_hash(source.current_text) != source.canonical_text_sha256
        or any(
            source.snapshot.get(key) != source.current_snapshot.get(key)
            for key in _CAPTION_SNAPSHOT_FIELDS
        )
        for source in sources
    ):
        return False
    metadata = _caption_metadata(target)
    headings = target.snapshot.get("heading_path")
    position = target.snapshot.get("position")
    appendix, section = metadata.get("appendix_label"), metadata.get("section_title")
    if (
        target.snapshot.get("chunk_type") != "paragraph"
        or metadata.get("intake_chunking_version") not in ("2", "3")
        or metadata.get("chunk_variant") != "atomic"
        or type(position) is not int
        or position < 0
        or not isinstance(headings, list)
        or not all(isinstance(value, str) for value in headings)
        or len(headings) < 3
        or metadata.get("heading_path") != headings
        or not isinstance(appendix, str)
        or re.fullmatch(r"EK\s+(?:[ivxlcdm]+|[1-9]\d*)", appendix, re.I) is None
        or headings.count(appendix) != 1
        or headings[-2] != appendix
        or headings[-1] != section
        or any(
            metadata.get(key) is not None
            for key in (
                "article_no",
                "clause_label",
                "subclause_label",
                "table_index",
                "table_row_index",
            )
        )
    ):
        return False
    if any(type(source.snapshot.get("position")) is not int for source in sources):
        return False
    adjacent = {source.snapshot.get("position"): source for source in sources}
    if len(adjacent) != len(sources):
        return False
    following = adjacent.get(position + 1)
    if following is None:
        return False
    following_meta = _caption_metadata(following)
    following_headings = following.snapshot.get("heading_path")
    if (
        following.snapshot.get("chunk_type") != "clause"
        or following_meta.get("clause_label") != "a"
        or following_meta.get("appendix_label") != appendix
        or following_meta.get("paragraph_no") != metadata.get("paragraph_no")
        or following_meta.get("section_title") != section
        or following_meta.get("article_no") is not None
        or following_meta.get("chunk_variant") != "atomic"
        or following_meta.get("intake_chunking_version") not in ("2", "3")
        or not isinstance(following_headings, list)
        or len(following_headings) != len(headings) + 1
        or following_headings[: len(headings)] != headings
        or following_meta.get("heading_path") != following_headings
        or not following.text.startswith("a) ")
    ):
        return False
    child_heading = following_headings[-1]
    if not isinstance(child_heading, str):
        return False
    child_prefix = (
        child_heading[:-3]
        if child_heading.endswith("...")
        else child_heading.removesuffix("…")
    )
    if (
        len(child_prefix) <= 3
        or not child_prefix.startswith("a) ")
        or not following.text.startswith(child_prefix)
    ):
        return False
    clause_audit = following_meta.get("intake_legal_metadata")
    fields = clause_audit.get("fields") if isinstance(clause_audit, Mapping) else None
    clause_fact = fields.get("clause_label") if isinstance(fields, Mapping) else None
    quote = (
        clause_fact.get("evidence_quote") if isinstance(clause_fact, Mapping) else None
    )
    if (
        not isinstance(quote, str)
        or not quote.startswith("a) ")
        or quote not in following.text
        or not _caption_fact(following, "clause_label", quote, "body")
    ):
        return False
    if match := _NUMBERED_CAPTION.fullmatch(target.text):
        title = target.text.removesuffix(":")
        return (
            str(metadata.get("paragraph_no"))
            in (match["number"], f"({match['number']})")
            and section == title
            and _caption_fact(target, "paragraph_no", target.text, "body")
            and _caption_fact(target, "section_title", title, "body")
        )
    match = _DIMENSION_CAPTION.fullmatch(target.text)
    previous = adjacent.get(position - 1)
    if match is None or previous is None or not isinstance(section, str):
        return False
    previous_meta = _caption_metadata(previous)
    section_match = _DESIGN_SECTION.fullmatch(section)
    if section_match is None or str(metadata.get("paragraph_no")) not in (
        section_match["number"],
        f"({section_match['number']})",
    ):
        return False
    lines = previous.text.splitlines()
    if (
        previous.snapshot.get("heading_path") != headings
        or previous_meta.get("appendix_label") != appendix
        or previous_meta.get("paragraph_no") != metadata.get("paragraph_no")
        or previous_meta.get("section_title") != section
        or len(lines) < 5
        or lines[0] != section
        or re.fullmatch(r"Şekil [1-9]\d{0,2}", lines[1]) is None
        or any(_DIAGRAM_LINE.fullmatch(line) is None for line in lines[2:])
        or previous_meta.get("article_no") is not None
        or re.search(rf"\b{re.escape(match['title'])}\b", section, re.I) is None
        or not _caption_fact(target, "paragraph_no", section, "heading")
        or not _caption_fact(target, "section_title", section, "heading")
        or not _caption_fact(previous, "paragraph_no", section, "body")
        or not _caption_fact(previous, "section_title", section, "body")
    ):
        return False
    tick_matches = [_DIMENSION.fullmatch(line) for line in lines[-3:]]
    if any(tick is None for tick in tick_matches):
        return False
    ticks = [
        Decimal(tick[1].replace(",", ".")) for tick in tick_matches if tick is not None
    ]
    size = Decimal(match["size"].replace(",", "."))
    return (
        ticks[1] - ticks[0] == ticks[2] - ticks[1] == size - ticks[2] > 0
        and following.text.startswith(f"a) {match['title']} ")
        and re.search(rf"\b{re.escape(lines[1])}\b", following.text) is not None
        and isinstance(quote, str)
        and quote.startswith(f"a) {match['title']} ")
        and quote in following.text
        and len(_DIMENSION.findall(quote)) >= 2
        and _caption_fact(following, "clause_label", quote, "body")
    )


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


def validate_stored_abstention_evidence(
    audit: AbstentionRepairAudit | Mapping[str, object],
    *,
    base_request: VertexBatchRequest,
    text: str,
    chunk_id: str,
    source_hash: str,
    taxonomy: TaxonomyDefinition,
    provider_binding: Mapping[str, object],
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
    return stored


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
    stored = validate_stored_abstention_evidence(
        audit,
        base_request=base_request,
        text=text,
        chunk_id=chunk_id,
        source_hash=source_hash,
        taxonomy=taxonomy,
        provider_binding=provider_binding,
    )
    if (
        stored.outcome.abstained
        or [item.label_id for item in stored.outcome.labels] != list(labels)
        or [item.model_dump(mode="json") for item in stored.outcome.labels]
        != [dict(item) for item in assignments]
    ):
        raise ValueError("Semantic repair does not prove these accepted assignments")
    return stored
