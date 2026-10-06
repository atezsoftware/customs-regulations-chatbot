"""Keep selected originals in their first retained native acquisition message."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass

from pydantic import JsonValue

from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import ResearchTurn, model_evidence_metadata
from onyx.llm.models import ToolMessage

OriginalRange = tuple[int, str, str | None, str, int, int]


@dataclass(frozen=True)
class NativeOriginalProjection:
    turns: list[ResearchTurn]
    fallback_originals: list[dict[str, JsonValue]]
    metadata_catalogue: list[dict[str, JsonValue]]


def _canonical_range(
    record: dict[str, JsonValue], ledger: EvidenceLedger
) -> OriginalRange:
    citation, text, start = (
        record.get("citation"),
        record.get("text"),
        record.get("start_char", 0),
    )
    if type(citation) is not int:
        raise ValueError("Native original range does not match its canonical identity")
    item = ledger.get(citation)
    if (
        item is None
        or not isinstance(text, str)
        or type(start) is not int
        or start < 0
        or record.get("source_id") != item.source_id
        or record.get("chunk_id") != item.chunk_id
        or record.get("text_hash") != item.text_hash
        or start + len(text) > len(item.text)
        or item.text[start : start + len(text)] != text
        or ("end_char" in record and type(record["end_char"]) is not int)
        or record.get("end_char", start + len(text)) != start + len(text)
    ):
        raise ValueError("Native original range does not match its canonical identity")
    return (
        citation,
        item.source_id,
        item.chunk_id,
        item.text_hash,
        start,
        start + len(text),
    )


def _metadata_ref(identity: OriginalRange) -> dict[str, JsonValue]:
    return {"citation": identity[0], "text_hash": identity[3]}


def _project_record(
    record: dict[str, JsonValue],
    identity: OriginalRange,
    *,
    keep_text: bool,
    current_metadata: bool,
) -> dict[str, JsonValue]:
    result = {key: value for key, value in record.items() if keep_text or key != "text"}
    result["start_char"], result["end_char"] = identity[4], identity[5]
    if current_metadata:
        address = _metadata_ref(identity)
        if "metadata_ref" in result and result["metadata_ref"] != address:
            raise ValueError("Native metadata reference addresses another original")
        result.pop("metadata", None)
        result["metadata_ref"] = address
    return result


def project_native_originals(
    turns: list[ResearchTurn],
    selected_originals: Sequence[dict[str, JsonValue]],
    ledger: EvidenceLedger,
) -> NativeOriginalProjection:
    selected: dict[OriginalRange, dict[str, JsonValue]] = {}
    catalogue: dict[tuple[int, str], dict[str, JsonValue]] = {}
    for record in selected_originals:
        identity = _canonical_range(record, ledger)
        item = ledger.get(identity[0])
        assert item is not None
        metadata = model_evidence_metadata(item.metadata)
        if record.get("metadata") != metadata:
            raise ValueError(
                "Selected original metadata is not current canonical metadata"
            )
        selected.setdefault(identity, record)
        catalogue.setdefault(
            (identity[0], identity[3]),
            {
                "citation": identity[0],
                "source_id": identity[1],
                "chunk_id": identity[2],
                "text_hash": identity[3],
                "metadata": metadata,
            },
        )

    delivered: set[OriginalRange] = set()
    projected: list[ResearchTurn] = []
    for turn in turns:
        results: list[ToolMessage] = []
        for result in turn.results:
            try:
                payload = json.loads(result.content)
            except (ValueError, TypeError):
                results.append(result)
                continue
            if not isinstance(payload, dict):
                results.append(result)
                continue
            originals = payload.get("original_evidence")
            references = payload.get("original_evidence_refs")
            if not isinstance(originals, list) and not isinstance(references, list):
                results.append(result)
                continue
            kept: list[JsonValue] = []
            refs: list[JsonValue] = []
            for record in originals if isinstance(originals, list) else []:
                if not isinstance(record, dict) or "text" not in record:
                    kept.append(record)
                    continue
                identity = _canonical_range(record, ledger)
                current_metadata = (identity[0], identity[3]) in catalogue
                keep = identity in selected and identity not in delivered
                updated = _project_record(
                    record, identity, keep_text=keep, current_metadata=current_metadata
                )
                if keep:
                    kept.append(updated)
                    delivered.add(identity)
                else:
                    refs.append(updated)
            for reference in references if isinstance(references, list) else []:
                if not isinstance(reference, dict):
                    refs.append(reference)
                    continue
                citation, digest = reference.get("citation"), reference.get("text_hash")
                current = (
                    catalogue.get((citation, digest))
                    if type(citation) is int and isinstance(digest, str)
                    else None
                )
                if (
                    current is None
                    or reference.get("source_id") != current["source_id"]
                    or reference.get("chunk_id") != current["chunk_id"]
                ):
                    refs.append(reference)
                    continue
                address = {"citation": citation, "text_hash": digest}
                if "metadata_ref" in reference and reference["metadata_ref"] != address:
                    raise ValueError(
                        "Native metadata reference addresses another original"
                    )
                refs.append(
                    {
                        **{
                            key: value
                            for key, value in reference.items()
                            if key != "metadata"
                        },
                        "metadata_ref": address,
                    }
                )
            if isinstance(originals, list):
                if kept:
                    payload["original_evidence"] = kept
                else:
                    payload.pop("original_evidence", None)
            if refs:
                payload["original_evidence_refs"] = refs
            else:
                payload.pop("original_evidence_refs", None)
            results.append(
                result.model_copy(
                    update={"content": json.dumps(payload, ensure_ascii=False)}
                )
            )
        projected.append(turn.model_copy(update={"results": results}))

    fallback = [
        _project_record(record, identity, keep_text=True, current_metadata=True)
        for identity, record in selected.items()
        if identity not in delivered
    ]
    return NativeOriginalProjection(projected, fallback, list(catalogue.values()))
