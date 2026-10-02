from __future__ import annotations

import json
import threading
from typing import Iterable

from pydantic import JsonValue

from onyx.asv3.models import EvidenceItem, RunContext
from onyx.context.search.models import SearchDoc


class EvidenceLedger:
    def __init__(self, max_records: int = 2000) -> None:
        self.max_records = max_records
        self._lock = threading.RLock()
        self._items: dict[int, EvidenceItem] = {}
        self._identities: dict[tuple[str, str | None, str], int] = {}
        self._included: set[int] = set()

    def add(self, items: Iterable[EvidenceItem], context: RunContext) -> list[int]:
        result: list[int] = []
        with self._lock:
            context.check_active()
            for item in items:
                existing = self._identities.get(item.identity)
                if existing is not None:
                    recorded = self._items[existing]
                    recorded.question_ids = list(
                        dict.fromkeys(recorded.question_ids + item.question_ids)
                    )
                    if recorded.search_doc is None and item.search_doc is not None:
                        recorded.search_doc = item.search_doc
                    result.append(existing)
                    continue
                if len(self._items) >= self.max_records:
                    raise ValueError("Evidence record capacity exceeded")
                context.budget.consume("evidence_bytes", len(item.text.encode("utf-8")))
                number = len(self._items) + 1
                self._identities[item.identity] = number
                self._items[number] = item.model_copy(deep=True)
                result.append(number)
        return result

    def get(self, number: int) -> EvidenceItem | None:
        with self._lock:
            item = self._items.get(number)
            return item.model_copy(deep=True) if item else None

    def citation_mapping(self) -> dict[int, SearchDoc]:
        with self._lock:
            return {
                number: item.search_doc
                for number, item in self._items.items()
                if item.search_doc is not None
            }

    def summaries(self, *, max_chars: int = 32000) -> list[dict[str, JsonValue]]:
        with self._lock:
            # A global top-relevance prefix can erase a later question. Round-robin
            # question/source groups, then allocate text only after source identities.
            groups: dict[str, list[tuple[int, EvidenceItem]]] = {}
            for number, item in self._items.items():
                group = item.question_ids[0] if item.question_ids else item.source_id
                groups.setdefault(group, []).append((number, item))
            ordered: list[tuple[int, EvidenceItem]] = []
            while any(groups.values()):
                for items in groups.values():
                    if items:
                        ordered.append(items.pop(0))
            output: list[dict[str, JsonValue]] = []
            selected: list[EvidenceItem] = []
            for number, item in ordered:
                entry: dict[str, JsonValue] = {
                    "citation": number,
                    "source_id": item.source_id,
                    "chunk_id": item.chunk_id,
                    "text_hash": item.text_hash,
                    "question_ids": list(item.question_ids),
                    "text": "",
                    "truncated": bool(item.text),
                }
                trial = output + [entry]
                if len(json.dumps(trial, ensure_ascii=False)) > max_chars:
                    break
                output.append(entry)
                selected.append(item)
            remaining = max(
                0, max_chars - len(json.dumps(output, ensure_ascii=False)) - len(output)
            )
            per_item = min(3000, max(0, remaining // max(1, len(output)) // 6))
            for entry, item in zip(output, selected):
                excerpt = item.text[:per_item]
                entry["text"] = excerpt
                entry["truncated"] = len(excerpt) < len(item.text)
            return output

    def export(self) -> dict[str, JsonValue]:
        with self._lock:
            return {
                "version": 1,
                "records": [
                    {"citation": number, "item": item.model_dump(mode="json")}
                    for number, item in self._items.items()
                ],
                "included": sorted(self._included),
            }

    def restore(self, payload: dict[str, JsonValue], context: RunContext) -> None:
        if payload.get("version") != 1 or not isinstance(payload.get("records"), list):
            raise ValueError("Invalid evidence checkpoint")
        records = payload["records"]
        assert isinstance(records, list)
        if len(records) > self.max_records:
            raise ValueError("Evidence record capacity exceeded")
        items: dict[int, EvidenceItem] = {}
        identities: dict[tuple[str, str | None, str], int] = {}
        for record in records:
            if not isinstance(record, dict) or record.get("citation") != len(items) + 1:
                raise ValueError("Invalid evidence citation sequence")
            item = EvidenceItem.model_validate(record.get("item"))
            if item.identity in identities:
                raise ValueError("Duplicate evidence identity")
            number = len(items) + 1
            items[number] = item
            identities[item.identity] = number
        included_raw = payload.get("included", [])
        if not isinstance(included_raw, list) or not all(
            isinstance(n, int) for n in included_raw
        ):
            raise ValueError("Invalid included citation list")
        included = {int(n) for n in included_raw if isinstance(n, int)}
        if not included.issubset(items):
            raise ValueError("Unknown included citation")
        if (
            sum(len(item.text.encode("utf-8")) for item in items.values())
            > context.budget.limits["evidence_bytes"]
        ):
            raise ValueError("Restored evidence exceeds the run budget")
        with self._lock:
            context.check_active()
            self._items, self._identities, self._included = items, identities, included

    def include(self, numbers: Iterable[int]) -> None:
        with self._lock:
            numbers = set(numbers)
            if not numbers.issubset(self._items):
                raise ValueError("Cannot include unknown evidence")
            self._included.update(numbers)

    def inspect(self, number: int) -> dict[str, JsonValue]:
        with self._lock:
            item = self._items.get(number)
            return {
                "recorded": item is not None,
                "included": number in self._included,
                "citation": number,
                "source_id": item.source_id if item else None,
            }
