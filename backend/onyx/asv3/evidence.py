from __future__ import annotations

import hashlib
import json
import re
import threading
from typing import Iterable

from pydantic import JsonValue

from onyx.asv3.models import (
    EvidenceItem,
    RunContext,
    RunStopped,
    model_evidence_metadata,
)
from onyx.asv3.witnesses import original_witness_spans
from onyx.context.search.models import SearchDoc


class EvidenceLedger:
    def __init__(self, max_records: int = 2000) -> None:
        self.max_records = max_records
        self._lock = threading.RLock()
        self._items: dict[int, EvidenceItem] = {}
        self._identities: dict[tuple[str, str | None, str], int] = {}
        self._included: set[int] = set()
        self._deliveries: list[dict[str, JsonValue]] = []

    def add(self, items: Iterable[EvidenceItem], context: RunContext) -> list[int]:
        result: list[int] = []
        with self._lock:
            context.check_active()
            for item in items:
                item = EvidenceItem.model_validate(item.model_dump(mode="python"))
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

    def citation_numbers(self) -> tuple[int, ...]:
        """Enumerate retained originals without copying passage text."""
        with self._lock:
            return tuple(self._items)

    def delivery_flow(self, call_id: str) -> str | None:
        with self._lock:
            return next(
                (
                    str(item.get("flow"))
                    for item in self._deliveries
                    if item.get("call_id") == call_id
                ),
                None,
            )

    def serialize_records(
        self,
        numbers: Iterable[int],
        *,
        required: Iterable[int] = (),
        max_chars: int = 180000,
        include_witness_spans: bool = False,
    ) -> str:
        """Fit full serialized originals, including provenance, without clipping required rules."""
        required_numbers = list(dict.fromkeys(required))
        required_set = set(required_numbers)
        ordered = list(dict.fromkeys([*required_numbers, *numbers]))
        records: list[dict[str, JsonValue]] = []
        used = 2
        with self._lock:
            for number in ordered:
                item = self._items.get(number)
                if item is None:
                    if number in required_set:
                        raise ValueError("Required evidence is not recorded")
                    continue
                record: dict[str, JsonValue] = {
                    "citation": number,
                    "source_id": item.source_id,
                    "chunk_id": item.chunk_id,
                    "text_hash": item.text_hash,
                    "text": item.text,
                    "truncated": False,
                    "citable": item.search_doc is not None,
                    "metadata": model_evidence_metadata(item.metadata),
                }
                if include_witness_spans:
                    record["witness_spans"] = [
                        dict(span) for span in original_witness_spans(number, item.text)
                    ]
                cost = len(json.dumps(record, ensure_ascii=False)) + (
                    2 if records else 0
                )
                if used + cost > max_chars:
                    if number in required_set:
                        raise RunStopped(
                            "Complete cited evidence and provenance exceed the serialized evidence limit"
                        )
                    continue
                records.append(record)
                used += cost
        return json.dumps(records, ensure_ascii=False)

    def authority_metadata(self) -> list[dict[str, JsonValue]]:
        """Inspect retained identities without copying original text or serialized receipts."""
        from onyx.regulatory.heading_path import parse_regulatory_article_heading

        with self._lock:
            records: list[dict[str, JsonValue]] = []
            for number, item in self._items.items():
                metadata = model_evidence_metadata(item.metadata)
                headings = metadata.get("heading_path")
                article = metadata.get("article_no")
                qualifier = None
                if isinstance(headings, list):
                    for heading in reversed(headings):
                        parsed = parse_regulatory_article_heading(str(heading))
                        if parsed:
                            article, qualifier = parsed.article_no, parsed.qualifier
                            break
                paragraph = metadata.get("paragraph_no")
                if (
                    paragraph is None
                    and metadata.get("clause_label")
                    and isinstance(headings, list)
                ):
                    # Atomic clauses can carry their enclosing paragraph only in the path.
                    for heading in reversed(headings[:-1]):
                        match = re.match(
                            r"^\s*(?:\((\d{1,4})\)|(\d{1,4})\.)\s", str(heading)
                        )
                        if match:
                            paragraph = match[1] or match[2]
                            break
                records.append(
                    {
                        "citation": number,
                        "citable": item.search_doc is not None,
                        "document_type": metadata.get("document_type"),
                        "title": metadata.get("title"),
                        "article_no": article,
                        "article_qualifier": qualifier,
                        "paragraph_no": paragraph,
                        "clause_label": metadata.get("clause_label"),
                        "heading_path": headings[:1]
                        if isinstance(headings, list)
                        else [],
                    }
                )
            return records

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
                metadata = model_evidence_metadata(item.metadata)
                entry: dict[str, JsonValue] = {
                    "citation": number,
                    "source_id": item.source_id,
                    "chunk_id": item.chunk_id,
                    "text_hash": item.text_hash,
                    "question_ids": list(item.question_ids),
                    "text": "",
                    "truncated": bool(item.text),
                }
                for key in ("document_type", "title", "article_no", "paragraph_no"):
                    value = metadata.get(key)
                    if isinstance(value, str):
                        entry[key] = value[:240]
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
                "deliveries": list(self._deliveries),
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
            raw_deliveries = payload.get("deliveries", [])
            if not isinstance(raw_deliveries, list) or len(raw_deliveries) > 200:
                raise ValueError("Invalid evidence delivery checkpoint")
            self._deliveries = []
            for delivery in raw_deliveries:
                if not isinstance(delivery, dict) or not isinstance(
                    delivery.get("records"), list
                ):
                    raise ValueError("Invalid evidence delivery record")
                for record in delivery["records"]:
                    if (
                        not isinstance(record, dict)
                        or record.get("citation") not in self._items
                    ):
                        raise ValueError("Unknown delivered evidence")
                    number = record["citation"]
                    assert isinstance(number, int)
                    if record.get("text_hash") != self._items[number].text_hash:
                        raise ValueError("Changed delivered evidence identity")
                self._deliveries.append(delivery)

    def record_delivery(
        self, call_id: str, flow: str, records: Iterable[dict[str, JsonValue]]
    ) -> None:
        """Record actual serialized source passages, not assumed model understanding."""
        with self._lock:
            delivered: list[JsonValue] = []
            for record in records:
                number, text = record.get("citation"), record.get("text")
                if not isinstance(number, int) or not isinstance(text, str):
                    continue
                item = self._items.get(number)
                if item is None or not text:
                    continue
                offset = record.get("start_char", 0)
                if not isinstance(offset, int) or offset < 0:
                    continue
                if item.text[offset : offset + len(text)] != text:
                    continue
                delivered.append(
                    {
                        "citation": number,
                        "source_id": item.source_id,
                        "chunk_id": item.chunk_id,
                        "text_hash": item.text_hash,
                        "start_char": offset,
                        "end_char": offset + len(text),
                        "passage_hash": hashlib.sha256(
                            text.encode("utf-8")
                        ).hexdigest(),
                        "complete": offset == 0 and text == item.text,
                    }
                )
            if delivered:
                self._deliveries.append(
                    {"call_id": call_id, "flow": flow, "records": delivered}
                )
                self._deliveries = self._deliveries[-200:]

    def include(self, numbers: Iterable[int]) -> None:
        with self._lock:
            numbers = set(numbers)
            if not numbers.issubset(self._items):
                raise ValueError("Cannot include unknown evidence")
            self._included.update(numbers)

    def completely_delivered(self, call_id: str) -> set[int]:
        with self._lock:
            numbers: set[int] = set()
            for delivery in self._deliveries:
                records = delivery.get("records")
                if delivery.get("call_id") != call_id or not isinstance(records, list):
                    continue
                for record in records:
                    number = (
                        record.get("citation") if isinstance(record, dict) else None
                    )
                    if (
                        isinstance(record, dict)
                        and record.get("complete") is True
                        and isinstance(number, int)
                    ):
                        numbers.add(number)
            return numbers

    def inspect(self, number: int) -> dict[str, JsonValue]:
        with self._lock:
            item = self._items.get(number)
            deliveries: list[JsonValue] = []
            for delivery in self._deliveries:
                records = delivery.get("records")
                if not isinstance(records, list):
                    continue
                matching = [
                    record
                    for record in records
                    if isinstance(record, dict) and record.get("citation") == number
                ]
                if matching:
                    deliveries.append({**delivery, "records": matching})
            return {
                "recorded": item is not None,
                "included": number in self._included,
                "citation": number,
                "source_id": item.source_id if item else None,
                "deliveries": deliveries,
            }
