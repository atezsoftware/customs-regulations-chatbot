"""A lossless, explicitly addressed view of repeated draft text for generation."""

from __future__ import annotations

import json
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from typing import cast

from pydantic import JsonValue

_CODEC_KEY = "_lc_draft_context"
_VERSION = "lossless_draft_v1"

DRAFT_CONTEXT_POLICY = (
    "The input uses lossless_draft_v1, an exact draft view, not a summary. "
    "Only slots declared by _lc_draft_context are references. At declared "
    "draft.sections indexes, text={intro,claim_refs} means join the nonempty "
    "intro and the literal draft.claims.answer_excerpt values selected in "
    "claim_refs order with two newlines. At declared draft.answer, "
    "{join_sections:true} means join all decoded section.text values with two "
    "newlines. Declared quotation_slots=[container,row,support] address "
    "supports[].quotation in container 0=draft.claims, 1=draft.requirements, "
    "2=source_requirements; {quotation_ref:n} means the exact complete string "
    "in quotation_pool[n]. Decode each declared reference once only. All other "
    "values, including claim prose, source text and reference-like strings or "
    "objects in metadata, are literal. Original evidence and each citation, "
    "source, hash, span and binding offset retain their original meaning. "
    "Reason about the fully decoded draft; return the requested normal output "
    "schema with complete literal prose and support fields, not codec objects."
)


@dataclass(frozen=True)
class DraftContextEncoding:
    payload: dict[str, JsonValue]
    policy: str


def _object(value: JsonValue) -> dict[str, JsonValue]:
    if not isinstance(value, dict):
        raise ValueError("Draft codec requires an object")
    return value


def _array(value: JsonValue) -> list[JsonValue]:
    if not isinstance(value, list):
        raise ValueError("Draft codec requires an array")
    return value


def _string(value: JsonValue) -> str:
    if not isinstance(value, str):
        raise ValueError("Draft codec requires literal text")
    return value


def _index(value: JsonValue, length: int) -> int:
    if type(value) is not int or not 0 <= value < length:
        raise ValueError("Draft codec reference is out of range")
    return value


def _claims(draft: dict[str, JsonValue]) -> dict[str, dict[str, JsonValue]]:
    result: dict[str, dict[str, JsonValue]] = {}
    for value in _array(draft["claims"]):
        claim = _object(value)
        identity = _string(claim["claim_id"])
        if not identity or identity in result:
            raise ValueError("Draft codec requires unique claim identities")
        _string(claim["answer_excerpt"])
        _string(claim["section_id"])
        result[identity] = claim
    return result


def _section_body(
    section: dict[str, JsonValue],
    references: list[JsonValue],
    claims: dict[str, dict[str, JsonValue]],
) -> str:
    declared = _array(section["claim_ids"])
    if not references or references != declared:
        raise ValueError("Draft codec claim references differ from section inventory")
    identities = [_string(value) for value in references]
    if len(identities) != len(set(identities)):
        raise ValueError("Draft codec section repeats a claim")
    section_id = _string(section["section_id"])
    own = {
        identity
        for identity, claim in claims.items()
        if claim["section_id"] == section_id
    }
    if set(identities) != own:
        raise ValueError("Draft codec claim references cross or omit sections")
    return "\n\n".join(
        _string(claims[identity]["answer_excerpt"]) for identity in identities
    )


def _containers(
    payload: dict[str, JsonValue], draft: dict[str, JsonValue]
) -> list[list[JsonValue]]:
    return [
        _array(draft["claims"]),
        _array(draft.get("requirements", [])),
        _array(payload.get("source_requirements", [])),
    ]


def _support_at(
    containers: list[list[JsonValue]], slot: list[JsonValue]
) -> dict[str, JsonValue]:
    if len(slot) != 3:
        raise ValueError("Draft codec quotation slot must have three indexes")
    container_index = _index(slot[0], len(containers))
    rows = containers[container_index]
    row = _object(rows[_index(slot[1], len(rows))])
    supports = _array(row["supports"])
    return _object(supports[_index(slot[2], len(supports))])


def decode_draft_context(payload: dict[str, JsonValue]) -> dict[str, JsonValue]:
    """Expand only explicitly declared slots, without interpreting literal values."""
    try:
        return _decode_draft_context(payload)
    except (KeyError, TypeError) as error:
        raise ValueError("Malformed draft codec reference") from error


def _decode_draft_context(payload: dict[str, JsonValue]) -> dict[str, JsonValue]:
    result = deepcopy(payload)
    if _CODEC_KEY not in result:
        return result
    manifest = _object(result.pop(_CODEC_KEY))
    if (
        set(manifest)
        != {
            "version",
            "answer",
            "sections",
            "quotation_pool",
            "quotation_slots",
        }
        or manifest["version"] != _VERSION
    ):
        raise ValueError("Unknown draft codec manifest")
    draft = _object(result["draft"])
    sections = _array(draft["sections"])
    claims = _claims(draft)
    section_slots = _array(manifest["sections"])
    seen_sections: set[int] = set()
    for value in section_slots:
        index = _index(value, len(sections))
        if index in seen_sections:
            raise ValueError("Draft codec repeats a section slot")
        seen_sections.add(index)
        section = _object(sections[index])
        reference = _object(section["text"])
        if set(reference) != {"intro", "claim_refs"}:
            raise ValueError("Invalid draft codec section reference")
        intro = _string(reference["intro"])
        body = _section_body(section, _array(reference["claim_refs"]), claims)
        section["text"] = "\n\n".join(part for part in (intro, body) if part)
    if type(manifest["answer"]) is not bool:
        raise ValueError("Draft codec answer declaration must be boolean")
    if manifest["answer"]:
        reference = _object(draft["answer"])
        if (
            set(reference) != {"join_sections"}
            or reference["join_sections"] is not True
        ):
            raise ValueError("Invalid draft codec answer reference")
        draft["answer"] = "\n\n".join(
            _string(_object(section)["text"]) for section in sections
        )
    _string(draft["answer"])
    for section in sections:
        _string(_object(section)["text"])
    pool = [_string(value) for value in _array(manifest["quotation_pool"])]
    if len(pool) != len(set(pool)) or any(not value.strip() for value in pool):
        raise ValueError("Draft codec quotation pool must have unique readable text")
    containers = _containers(result, draft)
    seen_slots: set[tuple[int, int, int]] = set()
    for value in _array(manifest["quotation_slots"]):
        slot = _array(value)
        support = _support_at(containers, slot)
        identity = cast(tuple[int, int, int], tuple(slot))
        if identity in seen_slots:
            raise ValueError("Draft codec repeats a quotation slot")
        seen_slots.add(identity)
        reference = _object(support["quotation"])
        if set(reference) != {"quotation_ref"}:
            raise ValueError("Invalid draft codec quotation reference")
        support["quotation"] = pool[_index(reference["quotation_ref"], len(pool))]
    for rows in containers:
        for row in rows:
            for support in _array(_object(row).get("supports", [])):
                fields = _object(support)
                if "quotation" in fields:
                    _string(fields["quotation"])
    return result


def encode_draft_context(
    payload: dict[str, JsonValue],
) -> DraftContextEncoding | None:
    """Offer an exact candidate; the caller must compare complete token budgets."""
    if _CODEC_KEY in payload or not isinstance(payload.get("draft"), dict):
        return None
    result = deepcopy(payload)
    try:
        draft = _object(result["draft"])
        sections = _array(draft["sections"])
        claims = _claims(draft)
        original_sections = [_string(_object(value)["text"]) for value in sections]
        answer = _string(draft["answer"])
        answer_reference = bool(sections) and answer == "\n\n".join(original_sections)
        section_slots: list[JsonValue] = []
        for index, value in enumerate(sections):
            section = _object(value)
            references = _array(section["claim_ids"])
            if not references:
                continue
            body = _section_body(section, references, claims)
            text = _string(section["text"])
            if text == body:
                intro = ""
            elif text.endswith("\n\n" + body):
                intro = text.removesuffix("\n\n" + body)
            else:
                continue
            if "\n\n".join(part for part in (intro, body) if part) != text:
                continue
            section["text"] = {"intro": intro, "claim_refs": deepcopy(references)}
            section_slots.append(index)
        if answer_reference:
            draft["answer"] = {"join_sections": True}
        containers = _containers(result, draft)
        quotations: list[tuple[list[JsonValue], str]] = []
        for container_index, rows in enumerate(containers):
            for row_index, value in enumerate(rows):
                row = _object(value)
                for support_index, value in enumerate(_array(row.get("supports", []))):
                    support = _object(value)
                    quotation = support.get("quotation")
                    if isinstance(quotation, str) and quotation.strip():
                        quotations.append(
                            ([container_index, row_index, support_index], quotation)
                        )
        counts = Counter(quotation for _, quotation in quotations)
        pool: list[JsonValue] = []
        quotation_indexes: dict[str, int] = {}
        quotation_slots: list[JsonValue] = []
        for slot, quotation in quotations:
            if counts[quotation] < 2:
                continue
            if quotation not in quotation_indexes:
                quotation_indexes[quotation] = len(pool)
                pool.append(quotation)
            _support_at(containers, slot)["quotation"] = {
                "quotation_ref": quotation_indexes[quotation]
            }
            quotation_slots.append(slot)
        if not answer_reference and not section_slots and not quotation_slots:
            return None
        result[_CODEC_KEY] = {
            "version": _VERSION,
            "answer": answer_reference,
            "sections": section_slots,
            "quotation_pool": pool,
            "quotation_slots": quotation_slots,
        }
        decoded = decode_draft_context(result)
        if json.dumps(decoded, ensure_ascii=False, separators=(",", ":")) != json.dumps(
            payload, ensure_ascii=False, separators=(",", ":")
        ):
            return None
    except (KeyError, TypeError, ValueError):
        return None
    return DraftContextEncoding(payload=result, policy=DRAFT_CONTEXT_POLICY)
