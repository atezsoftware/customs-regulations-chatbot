"""Durable navigation leads, separate from original evidence and tool audit."""

from __future__ import annotations

import copy
import hashlib
import json
import threading
from collections import OrderedDict
from collections.abc import Mapping

from pydantic import JsonValue

from onyx.asv3.models import ToolReceipt


def encoded(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class WorkingMemory:
    _FIELDS = frozenset(
        "locator_id kind status receipt_id tool source_id name chunk_id citation text_hash information_need origin_chunk_id article qualifier position heading_path validity_start validity_end projection_ordinal source_name has_more next_offset next_position evidence_next_position scan_truncated evidence_truncated subunit_verified absence_proven paragraph clause operation query query_hash pattern pattern_hash mode read_as_of_date article_closure_complete context".split()
    )

    def __init__(self, scope: Mapping[str, JsonValue]) -> None:
        self.scope_hash = hashlib.sha256(encoded(dict(scope)).encode()).hexdigest()
        self._lock = threading.RLock()
        self._locators: OrderedDict[str, dict[str, JsonValue]] = OrderedDict()
        self._strategies: set[str] = set()
        self.revision = 0
        self.unchanged_streak = 0
        self.max_records = 2000
        self.archived_locators = 0

    def _put(
        self, kind: str, fields: dict[str, JsonValue], receipt: ToolReceipt
    ) -> bool:
        identity = {
            key: value
            for key, value in fields.items()
            if key
            in {
                "source_id",
                "chunk_id",
                "citation",
                "name",
                "article",
                "qualifier",
                "information_need",
                "origin_chunk_id",
                "query",
                "query_hash",
                "pattern",
                "pattern_hash",
                "operation",
                "mode",
                "paragraph",
                "clause",
            }
        }
        locator_id = hashlib.sha256(encoded([kind, identity]).encode()).hexdigest()[:24]
        locator = {
            "locator_id": locator_id,
            "kind": kind,
            "status": receipt.outcome.status.value,
            **fields,
        }
        previous = self._locators.get(locator_id)
        changed = (
            previous is None
            or {
                key: value
                for key, value in previous.items()
                if key not in {"receipt_id", "tool"}
            }
            != locator
        )
        locator.update(receipt_id=receipt.call.call_id, tool=receipt.call.name)
        self._locators[locator_id] = locator
        self._locators.move_to_end(locator_id)
        if len(self._locators) > self.max_records:
            removable = next(
                (
                    key
                    for key, value in self._locators.items()
                    if value["kind"] in {"chunk", "evidence"}
                ),
                next(iter(self._locators)),
            )
            self._locators.pop(removable)
            self.archived_locators += 1
        return changed

    def observe(self, receipt: ToolReceipt) -> None:
        with self._lock:
            data, args = receipt.outcome.data, receipt.call.arguments
            need = args.get("coverage_item", args.get("question_id"))
            purpose: dict[str, JsonValue] = (
                {"information_need": need} if isinstance(need, str) else {}
            )
            if isinstance(need, str) and len(need) > 512:
                purpose = {
                    "information_need": "sha256:"
                    + hashlib.sha256(need.encode()).hexdigest()
                }
            changed = False
            for field, kind in (
                ("sources", "source"),
                ("headings", "chunk"),
                ("references", "reference"),
                ("unhydrated_centers", "chunk"),
            ):
                rows = data.get(field)
                if not isinstance(rows, list):
                    continue
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    fields = {
                        key: value
                        for key, value in row.items()
                        if key
                        in {
                            "source_id",
                            "name",
                            "chunk_id",
                            "canonical_chunk_id",
                            "position",
                            "article",
                            "qualifier",
                            "heading_path",
                            "validity_start",
                            "validity_end",
                            "projection_ordinal",
                        }
                    }
                    if "canonical_chunk_id" in fields:
                        fields["chunk_id"] = fields.pop("canonical_chunk_id")
                    if "source_id" not in fields and isinstance(
                        args.get("source_id"), str
                    ):
                        fields["source_id"] = args["source_id"]
                    if kind == "reference":
                        fields["origin_chunk_id"] = args.get("chunk_id")
                    changed = self._put(kind, {**purpose, **fields}, receipt) or changed
            source_id = args.get("source_id")
            source_name = data.get("source_name")
            if isinstance(source_id, str) and isinstance(source_name, str):
                changed = (
                    self._put(
                        "source",
                        {**purpose, "source_id": source_id, "name": source_name},
                        receipt,
                    )
                    or changed
                )
            tools = data.get("tools")
            if isinstance(tools, list):
                for tool in tools:
                    function = tool.get("function") if isinstance(tool, dict) else None
                    if isinstance(function, dict) and isinstance(
                        function.get("name"), str
                    ):
                        changed = (
                            self._put("capability", {"name": function["name"]}, receipt)
                            or changed
                        )
            cursors = {
                key: value
                for key, value in data.items()
                if key
                in {
                    "has_more",
                    "next_offset",
                    "next_position",
                    "evidence_next_position",
                    "scan_truncated",
                    "evidence_truncated",
                    "subunit_verified",
                    "article",
                    "absence_proven",
                }
            }
            if cursors:
                locator_args = {
                    key: value
                    for key, value in args.items()
                    if key
                    in {
                        "source_id",
                        "article",
                        "paragraph",
                        "clause",
                        "operation",
                        "query",
                        "pattern",
                        "mode",
                    }
                }
                # Arguments remain exact in the receipt; a digest addresses long queries.
                for key, value in list(locator_args.items()):
                    if isinstance(value, str) and len(value) > 768:
                        locator_args.pop(key)
                        locator_args[key + "_hash"] = hashlib.sha256(
                            value.encode()
                        ).hexdigest()
                changed = (
                    self._put("cursor", {**purpose, **locator_args, **cursors}, receipt)
                    or changed
                )
            for number, item in zip(receipt.evidence_ids, receipt.outcome.evidence):
                fields: dict[str, JsonValue] = {
                    **purpose,
                    "source_id": item.source_id,
                    "chunk_id": item.chunk_id,
                    "citation": number,
                    "text_hash": item.text_hash,
                }
                for key in ("position", "read_as_of_date", "article_closure_complete"):
                    if key in item.metadata:
                        fields[key] = item.metadata[key]
                changed = self._put("evidence", fields, receipt) or changed
                continuation = item.metadata.get("article_closure_continuation")
                if isinstance(continuation, list):
                    for chunk_id in continuation:
                        if isinstance(chunk_id, str):
                            changed = (
                                self._put(
                                    "chunk",
                                    {
                                        **purpose,
                                        "source_id": item.source_id,
                                        "chunk_id": chunk_id,
                                        "context": "undelivered_continuation",
                                    },
                                    receipt,
                                )
                                or changed
                            )
            strategy = hashlib.sha256(
                encoded([receipt.call.name, args]).encode()
            ).hexdigest()
            substantive = receipt.call.name not in {
                "record_scenario",
                "report_progress",
                "read_research_state",
                "wait_researcher",
                "inspect_evidence_path",
            }
            new_strategy = substantive and strategy not in self._strategies
            self._strategies.add(strategy)
            scenario_changed = data.get("research_changed") is True
            if changed or new_strategy or scenario_changed:
                self.revision += 1
                self.unchanged_streak = 0
            else:
                self.unchanged_streak += 1

    def export(self) -> dict[str, JsonValue]:
        with self._lock:
            return copy.deepcopy(
                {
                    "version": 1,
                    "scope_hash": self.scope_hash,
                    "revision": self.revision,
                    "unchanged_streak": self.unchanged_streak,
                    "locators": list(self._locators.values()),
                    "strategies": sorted(self._strategies),
                    "archived_locators": self.archived_locators,
                }
            )

    def restore(self, state: dict[str, JsonValue]) -> None:
        if state.get("version") != 1 or state.get("scope_hash") != self.scope_hash:
            raise ValueError("Working locator scope changed")
        locators, strategies = state.get("locators"), state.get("strategies", [])
        if (
            not isinstance(locators, list)
            or len(locators) > self.max_records
            or not isinstance(strategies, list)
            or len(strategies) > self.max_records
        ):
            raise ValueError("Invalid working locator checkpoint")
        with self._lock:
            self._locators.clear()
            for locator in locators:
                if (
                    not isinstance(locator, dict)
                    or not isinstance(locator.get("locator_id"), str)
                    or set(locator) - self._FIELDS
                    or locator.get("kind")
                    not in {
                        "source",
                        "chunk",
                        "reference",
                        "capability",
                        "cursor",
                        "evidence",
                    }
                ):
                    raise ValueError("Invalid working locator")
                self._locators[locator["locator_id"]] = dict(locator)
            self._strategies = {value for value in strategies if isinstance(value, str)}
            for key in ("revision", "unchanged_streak", "archived_locators"):
                value = state.get(key, 0)
                if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                    raise ValueError("Invalid research progress")
                setattr(self, key, value)

    def view(
        self,
        *,
        offset: int = 0,
        locator_ids: list[str] | None = None,
        max_bytes: int = 24576,
    ) -> dict[str, JsonValue]:
        with self._lock:
            all_items = list(reversed(self._locators.values()))
            if locator_ids is not None:
                all_items = [
                    item for item in all_items if item["locator_id"] in locator_ids
                ]
            # First retain a lead for every independent information need/kind.
            groups: OrderedDict[str, list[dict[str, JsonValue]]] = OrderedDict()
            for item in all_items:
                key = str(item.get("information_need", "")) + ":" + str(item["kind"])
                groups.setdefault(key, []).append(item)
            ordered = []
            while any(groups.values()):
                for group in groups.values():
                    if group:
                        ordered.append(group.pop(0))
            result: dict[str, JsonValue] = {
                "locators": [],
                "offset": offset,
                "next_offset": offset,
                "total_locators": len(ordered),
                "omitted_locators": len(ordered),
                "revision": self.revision,
                "unchanged_streak": self.unchanged_streak,
                "archived_locators": self.archived_locators,
                "reopen": "read_research_state with locator_offset or locator_ids; locators are leads, not evidence",
            }
            if self.archived_locators:
                result["archive_reopen"] = (
                    "Older locators remain in original receipts: read_research_state with receipt_offset or receipt_ids."
                )
            if self.unchanged_streak:
                result["progress_hint"] = (
                    "Recent actions added no new locator, evidence or strategy. Use retained leads, reopen a needed original, choose a different useful action or report the exact gap; method and parallelism remain your choice."
                )
            selected: list[JsonValue] = []
            for index, item in enumerate(ordered[offset:], start=offset):
                trial = {
                    **result,
                    "locators": [*selected, item],
                    "next_offset": index + 1,
                    "omitted_locators": len(ordered) - len(selected) - 1,
                }
                if len(json.dumps(trial, ensure_ascii=False).encode()) > max_bytes:
                    break
                selected.append(item)
                result = trial
            return copy.deepcopy(result)
