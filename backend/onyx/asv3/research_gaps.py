"""Expose recorded research gaps without inferring law or acquiring more sources."""

from collections.abc import Sequence

from pydantic import JsonValue

from onyx.asv3.models import HarnessView


def research_gap_signals(
    view: HarnessView, complete_originals: Sequence[dict[str, JsonValue]]
) -> list[dict[str, JsonValue]]:
    delivered_chunks: set[tuple[str, str]] = set()
    for record in complete_originals:
        source_id, chunk_id = record.get("source_id"), record.get("chunk_id")
        if isinstance(source_id, str) and isinstance(chunk_id, str):
            delivered_chunks.add((source_id, chunk_id))
    signals: list[dict[str, JsonValue]] = []
    seen_centers: set[tuple[str, str | int]] = set()
    for receipt in view.receipts:
        if receipt.call.name != "search_corpus":
            continue
        centers = receipt.outcome.data.get("unhydrated_centers")
        if not isinstance(centers, list):
            continue
        for center in centers:
            if not isinstance(center, dict):
                continue
            source_id = center.get("source_id")
            chunk_id = center.get("canonical_chunk_id")
            ordinal = center.get("projection_ordinal")
            if not isinstance(source_id, str) or not source_id:
                continue
            identity: tuple[str, str | int]
            if isinstance(chunk_id, str) and chunk_id:
                identity = (source_id, chunk_id)
                if identity in delivered_chunks:
                    continue
            elif type(ordinal) is int:
                identity = (source_id, ordinal)
            else:
                continue
            if identity in seen_centers:
                continue
            seen_centers.add(identity)
            signal: dict[str, JsonValue] = {
                "kind": "retrieved_original_not_delivered",
                "source_id": source_id,
                "call_id": receipt.call.call_id,
            }
            for key in ("canonical_chunk_id", "projection_ordinal"):
                if isinstance(center.get(key), (str, int)):
                    signal[key] = center[key]
            for key in ("coverage_item", "evidence_target"):
                value = receipt.call.arguments.get(key)
                if isinstance(value, str):
                    signal[key] = value
            signals.append(signal)
    needs = view.research_state.get("needs")
    if isinstance(needs, list):
        for need in needs:
            if not isinstance(need, dict):
                continue
            gap = need.get("gap")
            if (
                need.get("material") is False
                or need.get("status") not in ("open", "researching", "blocked")
                or not isinstance(gap, str)
                or not gap.strip()
            ):
                continue
            signal = {"kind": "recorded_research_gap", "gap": gap}
            for key in ("need_id", "question_ids", "completion_test"):
                if key in need:
                    signal[key] = need[key]
            signals.append(signal)
    return signals
