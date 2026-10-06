"""Explain rejected source references without approving or changing their claims."""

from __future__ import annotations

import re

from pydantic import JsonValue

from onyx.asv3.assertions import assertion_inventory
from onyx.asv3.authority import (
    _canonical_law_aliases,
    _canonical_source,
    _formal_law_name,
    _named_native_references,
    _native_original_rows,
    _reference_name,
    folded,
    statute_references,
)
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import EvidenceItem, model_evidence_metadata
from onyx.asv3.witnesses import original_witness_spans


def _trusted_original(item: EvidenceItem) -> bool:
    return _canonical_source(item) and not model_evidence_metadata(item.metadata).get(
        "untrusted"
    )


def source_contained_reference_catalogue(
    ledger: EvidenceLedger,
    complete_originals: list[dict[str, JsonValue]],
) -> dict[str, JsonValue] | None:
    """Project foreign references only from exact full originals in a physical payload."""
    complete: dict[int, EvidenceItem] = {}
    for record in complete_originals:
        citation = record.get("citation")
        if type(citation) is not int:
            continue
        item = ledger.get(citation)
        if (
            item is not None
            and _trusted_original(item)
            and record.get("source_id") == item.source_id
            and record.get("chunk_id") == item.chunk_id
            and record.get("text_hash") == item.text_hash
            and record.get("text") == item.text
            and type(record.get("start_char", 0)) is int
            and record.get("start_char", 0) == 0
            and type(record.get("end_char", len(item.text))) is int
            and record.get("end_char", len(item.text)) == len(item.text)
            and record.get("truncated", False) is False
            and record.get("citable", True) is True
        ):
            complete[citation] = item
    if not complete:
        return None
    rows = _native_original_rows(ledger)
    aliases = _canonical_law_aliases(rows)
    own_identities: dict[int, dict[str, JsonValue]] = {}
    for row in rows:
        citation = row.get("citation")
        if type(citation) is int and citation in complete:
            own_identities[citation] = row
    references: list[JsonValue] = []
    seen: set[tuple[int, str, str | None, str | None, str]] = set()
    for citation, item in sorted(complete.items()):
        metadata = model_evidence_metadata(item.metadata)
        own = own_identities.get(citation)
        for span in original_witness_spans(citation, item.text):
            passage = item.text[span["start_char"] : span["end_char"]]
            for reference, name in _named_native_references(
                passage,
                aliases,
                strict_reference_boundaries=True,
                syntactic_reference_binding=True,
            ):
                own_numbers = own.get("instrument_numbers") if own else None
                own_names = own.get("formal_names") if own else None
                if (
                    isinstance(own_numbers, list) and reference.number in own_numbers
                ) or (
                    name is not None
                    and isinstance(own_names, list)
                    and name in own_names
                    and (not reference.number or not own_numbers)
                ):
                    continue
                key = (
                    citation,
                    reference.number,
                    name,
                    reference.article,
                    span["witness_id"],
                )
                if key in seen:
                    continue
                seen.add(key)
                references.append(
                    {
                        **span,
                        "citation": citation,
                        "source_id": item.source_id,
                        "chunk_id": item.chunk_id,
                        "text_hash": item.text_hash,
                        "source_title": metadata.get("title"),
                        "reference_text": reference.reference_text,
                        "instrument_number": reference.number or None,
                        "formal_name": name,
                        "article": reference.article,
                        "role": "source_contained_reference_navigation",
                    }
                )
    if not references:
        return None
    return {
        "instruction": (
            "These exact source passages contain references to another instrument; "
            "they are navigation, not required research needs or that instrument's "
            "governing support. When describing the source's referral, use a short "
            "verbatim quotation with that source's citation. Any independent result "
            "under the named target norm requires its own actual governing original. "
            "No missing law, legal assessment or citation choice is inferred here."
        ),
        "references": references,
    }


def _reference_identity(
    entry: dict[str, JsonValue],
) -> tuple[str | None, str | None] | None:
    reference = entry.get("reference_text")
    if not isinstance(reference, str) or not reference.strip():
        return None
    number = entry.get("instrument_number")
    if number is not None and not isinstance(number, str):
        return None
    name = entry.get("formal_name")
    if not isinstance(name, str):
        parsed = statute_references(
            reference,
            strict_reference_boundaries=True,
            syntactic_reference_binding=True,
        )
        name = next(
            (
                _reference_name(item)
                for item in parsed
                if number is None or item.number == number
            ),
            None,
        )
    if name is None:
        match = re.match(r"(.+?\bkanun(?:u|un|unun)?\b)", folded(reference))
        name = _formal_law_name(match[0]) if match is not None else None
    return (number, name) if number or name else None


def authority_reference_diagnostics(
    answer: str,
    gap: dict[str, JsonValue],
    ledger: EvidenceLedger,
    completely_delivered: set[int],
) -> dict[str, JsonValue]:
    """Return exact current blocks and their own source-contained reference witnesses."""
    units = {unit["unit_id"]: unit for unit in assertion_inventory(answer)}
    diagnostics: list[JsonValue] = []
    seen: set[tuple[str, str, str | None, str | None]] = set()
    for field, selector in (
        ("named_authority_gaps", "unit_id"),
        ("retained_authority_requirements", "origin_unit_id"),
    ):
        entries = gap.get(field)
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            unit_id = entry.get(selector)
            reference = entry.get("reference_text")
            identity = _reference_identity(entry)
            if (
                not isinstance(unit_id, str)
                or unit_id not in units
                or not isinstance(reference, str)
                or identity is None
            ):
                continue
            number, name = identity
            key = (field, unit_id, number, reference)
            if key in seen:
                continue
            seen.add(key)
            unit = units[unit_id]
            article = entry.get("article")
            aliases = {name: {number or ""}} if name else {}
            qualifier = entry.get("qualifier")
            if "qualifier" not in entry and isinstance(article, str):
                qualifier = next(
                    (
                        ref.qualifier
                        for ref, _ in _named_native_references(
                            reference,
                            aliases,
                            strict_reference_boundaries=True,
                            syntactic_reference_binding=True,
                        )
                        if ref.article == article
                    ),
                    None,
                )
            witnesses: list[JsonValue] = []
            for citation in unit["evidence_numbers"]:
                if citation not in completely_delivered:
                    continue
                item = ledger.get(citation)
                if item is None:
                    continue
                if not _trusted_original(item):
                    continue
                metadata = model_evidence_metadata(item.metadata)
                for span in original_witness_spans(citation, item.text):
                    passage = item.text[span["start_char"] : span["end_char"]]
                    references = _named_native_references(
                        passage,
                        aliases,
                        strict_reference_boundaries=True,
                        syntactic_reference_binding=True,
                    )
                    if not any(
                        (ref.number == number if number else ref_name == name)
                        and (
                            article is None
                            or (ref.article == article and ref.qualifier == qualifier)
                        )
                        for ref, ref_name in references
                    ):
                        continue
                    witnesses.append(
                        {
                            **span,
                            "citation": citation,
                            "source_id": item.source_id,
                            "chunk_id": item.chunk_id,
                            "text_hash": item.text_hash,
                            "source_title": metadata.get("title"),
                            "source_quote": passage,
                            "role": "source_contained_reference_navigation",
                        }
                    )
            diagnostics.append(
                {
                    "kind": field,
                    "unit_id": unit_id,
                    "unit_text": unit["text"],
                    "reference_text": reference,
                    "instrument_number": number,
                    "article": entry.get("article"),
                    **(
                        {"requirement_id": entry["requirement_id"]}
                        if field == "retained_authority_requirements"
                        and isinstance(entry.get("requirement_id"), str)
                        else {}
                    ),
                    "source_contained_reference_witnesses": witnesses,
                }
            )
    if not diagnostics:
        return gap
    return {
        **gap,
        "authority_reference_diagnostics": {
            "instruction": (
                "These are exact current answer blocks and reference passages from their "
                "own inline, fully delivered canonical originals. A source-contained "
                "reference is navigation, not the named instrument's governing support. "
                "If the block describes that source's referral, quote only its exact "
                "relevant contiguous phrase and retain the source citation. Any independent "
                "rule outside that quotation still needs its actual governing original. "
                "Existing retained requirements remain open until their unchanged guards "
                "are satisfied; a quotation edit does not discharge them. Do not append "
                "an identity candidate as support without examining its operative passage. "
                "An origin block is shown only when its exact stored unit ID occurs in "
                "this answer; no missing historical origin has been reconstructed."
            ),
            "units": diagnostics,
        },
    }
