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
_IDENTITY_FIELDS = frozenset(
    {"source_id", "chunk_id", "text_hash", "metadata", "metadata_ref"}
)
_CATALOGUE_FIELDS = frozenset(
    {"citation", "source_id", "chunk_id", "text_hash", "metadata"}
)


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
    compact_identities: bool = False,
) -> dict[str, JsonValue]:
    result = {key: value for key, value in record.items() if keep_text or key != "text"}
    result["start_char"], result["end_char"] = identity[4], identity[5]
    if current_metadata:
        address = _metadata_ref(identity)
        if "metadata_ref" in result and result["metadata_ref"] != address:
            raise ValueError("Native metadata reference addresses another original")
        result.pop("metadata", None)
        result["metadata_ref"] = address
        if compact_identities:
            return _compact_identity(result, identity[0])
    return result


def _compact_identity(
    record: dict[str, JsonValue], citation: int
) -> dict[str, JsonValue]:
    if "identity_ref" in record and (
        type(record["identity_ref"]) is not int or record["identity_ref"] != citation
    ):
        raise ValueError("Native identity reference addresses another original")
    return {
        **{key: value for key, value in record.items() if key not in _IDENTITY_FIELDS},
        "identity_ref": citation,
    }


def _validate_reference_range(record: dict[str, JsonValue], total_chars: int) -> None:
    start, end = record.get("start_char", 0), record.get("end_char", total_chars)
    if (
        type(start) is not int
        or type(end) is not int
        or start < 0
        or end < start
        or end > total_chars
    ):
        raise ValueError("Native identity reference has an invalid canonical range")


def project_native_originals(
    turns: list[ResearchTurn],
    selected_originals: Sequence[dict[str, JsonValue]],
    ledger: EvidenceLedger,
    *,
    compact_identities: bool = False,
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
                    record,
                    identity,
                    keep_text=keep,
                    current_metadata=current_metadata,
                    compact_identities=compact_identities,
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
                updated_reference = {
                    **{
                        key: value
                        for key, value in reference.items()
                        if key != "metadata"
                    },
                    "metadata_ref": address,
                }
                if compact_identities:
                    item = ledger.get(citation)
                    assert item is not None
                    if "text" in reference:
                        raise ValueError(
                            "Native identity reference cannot contain text"
                        )
                    _validate_reference_range(reference, len(item.text))
                    updated_reference = _compact_identity(updated_reference, citation)
                refs.append(updated_reference)
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
        _project_record(
            record,
            identity,
            keep_text=True,
            current_metadata=True,
            compact_identities=compact_identities,
        )
        for identity, record in selected.items()
        if identity not in delivered
    ]
    return NativeOriginalProjection(projected, fallback, list(catalogue.values()))


def decode_compact_originals(
    payloads: Sequence[dict[str, JsonValue]],
    final_host_catalogue: Sequence[dict[str, JsonValue]] | None,
    ledger: EvidenceLedger,
) -> list[dict[str, JsonValue]]:
    """Rehydrate only literal text rows against the caller's invocation-bound catalogue."""
    catalogue: dict[int, dict[str, JsonValue]] = {}
    for identity in final_host_catalogue or []:
        if not isinstance(identity, dict):
            raise ValueError("Native identity catalogue has an invalid record")
        citation = identity.get("citation")
        if type(citation) is not int or citation <= 0 or citation in catalogue:
            raise ValueError(
                "Native identity catalogue has an invalid or duplicate citation"
            )
        item = ledger.get(citation)
        if (
            set(identity) != _CATALOGUE_FIELDS
            or item is None
            or identity.get("source_id") != item.source_id
            or identity.get("chunk_id") != item.chunk_id
            or identity.get("text_hash") != item.text_hash
            or json.dumps(identity.get("metadata"), sort_keys=True)
            != json.dumps(model_evidence_metadata(item.metadata), sort_keys=True)
        ):
            raise ValueError(
                "Native identity catalogue is not current canonical evidence"
            )
        catalogue[citation] = identity

    decoded: list[dict[str, JsonValue]] = []
    for payload in payloads:
        for field in ("original_evidence", "evidence", "original_evidence_refs"):
            records = payload.get(field)
            if not isinstance(records, list):
                continue
            for record in records:
                if not isinstance(record, dict) or "identity_ref" not in record:
                    continue
                citation = record.get("citation")
                if (
                    type(citation) is not int
                    or type(record["identity_ref"]) is not int
                    or record["identity_ref"] != citation
                    or citation not in catalogue
                    or _IDENTITY_FIELDS.intersection(record)
                ):
                    raise ValueError(
                        "Native compact original has no matching canonical identity"
                    )
                item = ledger.get(citation)
                assert item is not None
                _validate_reference_range(record, len(item.text))
                if field == "original_evidence_refs":
                    if "text" in record:
                        raise ValueError(
                            "Native identity reference cannot contain text"
                        )
                    continue
                if "text" not in record:
                    continue
                text, start, end = (
                    record["text"],
                    record.get("start_char"),
                    record.get("end_char"),
                )
                if (
                    not isinstance(text, str)
                    or not text
                    or type(start) is not int
                    or type(end) is not int
                    or end - start != len(text)
                    or item.text[start:end] != text
                ):
                    raise ValueError(
                        "Native compact original text is not its canonical range"
                    )
                decoded.append(
                    {
                        **{
                            key: value
                            for key, value in record.items()
                            if key != "identity_ref"
                        },
                        **{
                            key: value
                            for key, value in catalogue[citation].items()
                            if key != "citation"
                        },
                    }
                )
    return decoded
