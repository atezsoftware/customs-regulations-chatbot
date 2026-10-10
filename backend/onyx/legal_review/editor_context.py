"""A bounded editorial task with complete relevant originals and stable defect IDs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from typing import cast

from pydantic import JsonValue


def answer_revision(answer: str) -> str:
    return hashlib.sha256(answer.encode("utf-8")).hexdigest()


def _rows(value: JsonValue) -> Iterable[dict[str, JsonValue]]:
    return (
        (row for row in value if isinstance(row, dict))
        if isinstance(value, list)
        else ()
    )


def _citations(value: JsonValue) -> set[int]:
    """Select explicit evidence bindings, never numbers inside generated prose."""
    if isinstance(value, dict):
        selected = value.get("citation")
        return ({selected} if type(selected) is int else set()) | set().union(
            *(_citations(child) for child in value.values())
        )
    if isinstance(value, list):
        return set().union(*(_citations(child) for child in value))
    return set()


def editorial_context(state: dict[str, JsonValue]) -> dict[str, JsonValue]:
    """Keep whole articles, claim bases and material alternatives; omit research machinery."""
    base = state.get("repair_base")
    if not isinstance(base, dict):
        raise ValueError("Editorial context requires the current block inventory")
    selected = _citations(base)
    for key in (
        "editor_findings",
        "requirements",
        "finding_sources",
        "research_resolutions",
        "repair_resolutions",
        "material_source_leads",
    ):
        selected.update(_citations(state.get(key)))
    originals = list(_rows(state.get("original_evidence")))
    accounted: set[int] = set()
    for assessment in _rows(state.get("source_assessments")):
        citations = assessment.get("citations")
        if isinstance(citations, list):
            accounted.update(
                citation for citation in citations if type(citation) is int
            )
        if assessment.get("disposition") != "irrelevant":
            selected.update(_citations(assessment.get("supports")))
    # A phase can end after retrieval but before interpretation. Those originals
    # remain open evidence, rather than becoming implicitly rejected sources.
    selected.update(
        citation
        for row in originals
        if type(citation := row.get("citation")) is int and citation not in accounted
    )
    # Passage selectors may identify one paragraph. Keep every available paragraph
    # of its article so a nearby condition or contrary branch cannot disappear.
    article_keys: set[tuple[str, str]] = set()
    material_sources = {
        row.get("source_id") for row in _rows(state.get("material_source_leads"))
    }
    for row in originals:
        metadata = row.get("metadata")
        article = metadata.get("article_no") if isinstance(metadata, dict) else None
        source = row.get("source_id")
        if row.get("citation") in selected and isinstance(source, str) and article:
            article_keys.add((source, str(article)))
    included: list[JsonValue] = []
    for row in originals:
        metadata = row.get("metadata")
        article = metadata.get("article_no") if isinstance(metadata, dict) else None
        if (
            row.get("citation") in selected
            or row.get("source_id") in material_sources
            or (str(row.get("source_id")), str(article)) in article_keys
        ):
            included.append(row)
    known = {row.get("citation") for row in originals}
    if selected - known:
        raise ValueError("Editorial task selects unavailable canonical originals")
    result = {
        key: state[key]
        for key in (
            "request",
            "history",
            "plan",
            "requested_outcomes",
            "corpus_currency",
            "repair_base",
            "editor_findings",
            "finding_sources",
            "findings_answer_revision",
            "answer_revision",
        )
        if key in state
    }
    result["original_evidence"] = included
    result["editor_context_scope"] = {
        "included_citations": [row["citation"] for row in cast(list[dict], included)],
        "available_original_count": len(originals),
        "selection": "All claim and finding bases, relevant assessed passages, unassessed originals, material alternatives and their complete available articles",
    }
    return result


def group_editor_findings(
    findings: list[JsonValue],
) -> tuple[list[dict[str, JsonValue]], dict[str, list[str]], dict[str, JsonValue]]:
    """Merge only exact concrete equivalences and retain every originating check."""
    groups: dict[str, dict[str, JsonValue]] = {}
    passages: dict[str, JsonValue] = {}
    passage_ids: dict[str, str] = {}
    for row in findings:
        if not isinstance(row, dict) or not isinstance(row.get("check_id"), str):
            raise ValueError("Editor finding requires a known check identity")
        quotes = row.get("answer_quotes", [])
        ids: list[JsonValue] = []
        for quote in quotes if isinstance(quotes, list) else []:
            if not isinstance(quote, str):
                raise ValueError("Editor answer passages must be strings")
            if quote not in passage_ids:
                identity = f"a{len(passages) + 1:04d}"
                passage_ids[quote] = identity
                passages[identity] = quote
            ids.append(passage_ids[quote])
        semantic = {
            "target": row.get("target"),
            "required_change": row.get("required_change"),
            "repair_kind": row.get("repair_kind"),
            "supports": sorted(
                list(_rows(row.get("supports"))),
                key=lambda value: json.dumps(value, sort_keys=True),
            ),
            "answer_passage_ids": sorted(set(cast(list[str], ids))),
        }
        key = (
            json.dumps(semantic, sort_keys=True, ensure_ascii=False)
            if semantic["required_change"]
            else cast(str, row["check_id"])
        )
        if key not in groups:
            groups[key] = {
                **semantic,
                "reason": row.get("reason"),
                "check_ids": [],
            }
        cast(list[str], groups[key]["check_ids"]).append(cast(str, row["check_id"]))
    slots: dict[str, list[str]] = {}
    rows: list[dict[str, JsonValue]] = []
    for index, group in enumerate(groups.values(), 1):
        slot = f"q{index:04d}"
        slots[slot] = cast(list[str], group["check_ids"])
        rows.append({"slot": slot, **group})
    identities = [identity for members in slots.values() for identity in members]
    if len(identities) != len(set(identities)):
        raise ValueError("Editor finding identities must be unique")
    return rows, slots, passages
