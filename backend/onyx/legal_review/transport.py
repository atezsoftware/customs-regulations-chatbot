"""Lossless evidence normalization for Legal Review model requests."""

from __future__ import annotations

import copy
import json
from collections import defaultdict
from collections.abc import Sequence
from typing import cast

from pydantic import JsonValue

from onyx.tracing.flows import LLMFlow

_DIAGNOSTIC_KEYS = frozenset(
    {
        "source_count",
        "retrieved_result_count",
        "mapped_result_count",
        "hydrated_center_count",
        "retained_evidence_count",
        "unmapped_chunk_count",
        "unmapped_count",
        "evidence_truncated",
        "scan_truncated",
        "outline_truncated",
        "has_more",
        "truncated",
        "degraded",
        "error",
        "reason",
        "missing",
        "access_denied",
        "continuation",
        "next_cursor",
        "next_position",
        "next_offset",
        "evidence_next_position",
        "total_hits",
        "matched_count",
        "unmapped_result_count",
        "incomplete_closure_count",
        "unhydrated_centers",
        "context_policy",
    }
)


def _same_value(left: JsonValue, right: JsonValue) -> bool:
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(
        right, sort_keys=True, allow_nan=False
    )


def _heading_prefix(metadata: Sequence[dict[str, JsonValue]]) -> list[str]:
    paths = [item.get("heading_path") for item in metadata]
    if any(
        not isinstance(path, list) or any(not isinstance(part, str) for part in path)
        for path in paths
    ):
        return []
    headings = cast(list[list[str]], paths)
    if not headings:
        return []
    prefix: list[str] = []
    for parts in zip(*headings):
        if len(set(parts)) != 1:
            break
        prefix.append(parts[0])
    return prefix


def _normalize_originals(state: dict[str, JsonValue]) -> None:
    originals = state.get("original_evidence")
    if not isinstance(originals, list):
        raise ValueError("Legal Review original evidence must be an array")
    grouped: dict[str, list[dict[str, JsonValue]]] = defaultdict(list)
    for item in originals:
        if not isinstance(item, dict):
            raise ValueError("Legal Review original evidence must contain objects")
        source_id = item.get("source_id")
        if not isinstance(source_id, str) or not source_id:
            raise ValueError("Legal Review original source identity is missing")
        if not isinstance(item.get("metadata"), dict):
            raise ValueError("Legal Review original metadata is missing")
        grouped[source_id].append(item)

    registry: dict[str, JsonValue] = {}
    for index, (source_id, source_originals) in enumerate(grouped.items(), 1):
        source_ref = f"d{index:04d}"
        metadata = [
            cast(dict[str, JsonValue], original["metadata"])
            for original in source_originals
        ]
        shared = {
            key: value
            for key, value in metadata[0].items()
            if key != "heading_path"
            and all(
                key in other and _same_value(value, other[key])
                for other in metadata[1:]
            )
        }
        heading_prefix = (
            _heading_prefix(metadata)
            if all("heading_suffix" not in item for item in metadata)
            else []
        )
        entry: dict[str, JsonValue] = {
            "source_id": source_id,
            "metadata": shared,
        }
        if heading_prefix:
            entry["heading_prefix"] = cast(list[JsonValue], heading_prefix)
        registry[source_ref] = entry
        for original, original_metadata in zip(source_originals, metadata):
            original.pop("source_id")
            original.pop("text_hash", None)
            original["source_ref"] = source_ref
            local = {
                key: value
                for key, value in original_metadata.items()
                if key not in shared
            }
            if heading_prefix:
                path = cast(list[JsonValue], local.pop("heading_path"))
                local["heading_suffix"] = path[len(heading_prefix) :]
            original["metadata"] = local
    state["source_registry"] = registry


def _prepare_writer_evidence(state: dict[str, JsonValue]) -> None:
    """Use research as a source index, not an authoritative legal paraphrase."""
    requirements = state.get("requirements")
    if isinstance(requirements, list):
        state["requirements"] = [
            {key: value for key, value in row.items() if key != "rule"}
            for row in requirements
            if isinstance(row, dict)
        ]
    assessments = state.get("dimension_assessments")
    if isinstance(assessments, list):
        state["dimension_assessments"] = [
            {
                key: value
                for key, value in row.items()
                if key != "reason" or row.get("status") == "unresolved"
            }
            for row in assessments
            if isinstance(row, dict)
        ]
    draft = state.pop("draft", None)
    if isinstance(draft, dict):
        state["previous_unresolved_issue_ids"] = draft.get("unresolved_issue_ids")
        claims = draft.get("claims")
        if isinstance(claims, list):
            state["previous_claim_sources"] = [
                {key: value for key, value in row.items() if key != "answer_excerpt"}
                for row in claims
                if isinstance(row, dict)
            ]


def model_state(state: dict[str, JsonValue], flow: LLMFlow) -> dict[str, JsonValue]:
    """Keep canonical passages intact while sharing repeated provenance fields."""
    result = copy.deepcopy(state)
    if "original_evidence" in result:
        _normalize_originals(result)
    receipts = result.pop("source_operations", None)
    if isinstance(receipts, list):
        record: list[JsonValue] = []
        for receipt in receipts:
            if not isinstance(receipt, dict):
                raise ValueError("Legal Review source receipts must contain objects")
            data = receipt.get("data")
            record.append(
                {
                    **{key: value for key, value in receipt.items() if key != "data"},
                    "result_limitations": {
                        key: value
                        for key, value in (
                            data.items() if isinstance(data, dict) else []
                        )
                        if key in _DIAGNOSTIC_KEYS
                    },
                    **(
                        {"navigation": data}
                        if flow
                        in {
                            LLMFlow.LEGAL_REVIEW_READING,
                            LLMFlow.LEGAL_REVIEW_SOURCE_ACCOUNTING,
                        }
                        and receipt.get("tool") != "search_corpus"
                        and isinstance(data, dict)
                        else {}
                    ),
                }
            )
        result["research_record"] = record
    if flow in {LLMFlow.LEGAL_REVIEW_DRAFT, LLMFlow.LEGAL_REVIEW_REPAIR}:
        _prepare_writer_evidence(result)
        for key in (
            "tools",
            "reading_contract",
            "repair_contract",
            "final_adjudication",
            "source_assessments",
        ):
            result.pop(key, None)
        if isinstance(result.get("review_diagnoses"), dict):
            result.pop("early_review", None)
            result.pop("final_review", None)
        adjudication = result.pop("draft_adjudication", None)
        if isinstance(adjudication, dict) and isinstance(
            adjudication.get("findings"), list
        ):
            result["publication_corrections"] = [
                row
                for row in adjudication["findings"]
                if isinstance(row, dict) and row.get("disposition") == "defect"
            ]
    if isinstance(result.get("final_review"), dict):
        result.pop("early_review", None)
    return result
