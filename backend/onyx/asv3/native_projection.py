"""Reference identical historical metadata only while its full original is present."""

from __future__ import annotations

import json
from collections.abc import Sequence

from pydantic import JsonValue

from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import ResearchTurn, model_evidence_metadata
from onyx.llm.models import ToolMessage


def reference_duplicate_metadata(
    turns: list[ResearchTurn],
    complete_originals: Sequence[dict[str, JsonValue]],
    ledger: EvidenceLedger,
) -> list[ResearchTurn]:
    catalogue: dict[tuple[int, str], dict[str, JsonValue]] = {}
    for record in complete_originals:
        citation = record.get("citation")
        if type(citation) is not int:
            continue
        item = ledger.get(citation)
        if (
            item is not None
            and record.get("source_id") == item.source_id
            and record.get("chunk_id") == item.chunk_id
            and record.get("text_hash") == item.text_hash
            and record.get("start_char", 0) == 0
            and record.get("text") == item.text
            and record.get("metadata") == model_evidence_metadata(item.metadata)
        ):
            catalogue[(citation, item.text_hash)] = record
    if not catalogue:
        return turns

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
            references = payload.get("original_evidence_refs")
            if not isinstance(references, list):
                results.append(result)
                continue
            changed = False
            updated: list[JsonValue] = []
            for reference in references:
                if not isinstance(reference, dict):
                    updated.append(reference)
                    continue
                citation, digest = reference.get("citation"), reference.get("text_hash")
                current = (
                    catalogue.get((citation, digest))
                    if type(citation) is int and isinstance(digest, str)
                    else None
                )
                if (
                    current is None
                    or "metadata_ref" in reference
                    or reference.get("source_id") != current.get("source_id")
                    or reference.get("chunk_id") != current.get("chunk_id")
                    or not isinstance(reference.get("metadata"), dict)
                    or reference["metadata"] != current.get("metadata")
                ):
                    updated.append(reference)
                    continue
                updated.append(
                    {
                        **{
                            key: value
                            for key, value in reference.items()
                            if key != "metadata"
                        },
                        "metadata_ref": {"citation": citation, "text_hash": digest},
                    }
                )
                changed = True
            if changed:
                payload["original_evidence_refs"] = updated
                results.append(
                    result.model_copy(
                        update={"content": json.dumps(payload, ensure_ascii=False)}
                    )
                )
            else:
                results.append(result)
        projected.append(turn.model_copy(update={"results": results}))
    return projected
