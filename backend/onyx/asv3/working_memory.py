"""Durable navigation leads, separate from original evidence and tool audit."""

from __future__ import annotations

import copy
import hashlib
import json
import threading
from collections import OrderedDict
from collections.abc import Mapping
from uuid import UUID

from pydantic import JsonValue

from onyx.asv3.authority import statute_references
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.legal_source_reviews import serial_session_diagnostics_enabled
from onyx.asv3.models import CapabilityCall, OutcomeStatus, RunContext, ToolReceipt


def encoded(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _read_argument(value: JsonValue, results: dict[str, JsonValue]) -> JsonValue:
    if not isinstance(value, dict) or set(value) != {"$ref"}:
        return value
    reference = value["$ref"]
    if not isinstance(reference, str):
        return None
    path = reference.split(".")
    current = results.get(path[0])
    for part in path[1:]:
        if isinstance(current, dict):
            current = current.get(part)
        elif isinstance(current, list) and part.isdigit() and int(part) < len(current):
            current = current[int(part)]
        else:
            return None
    return current


class WorkingMemory:
    _FIELDS = frozenset(
        "locator_id kind status receipt_id tool source_id name chunk_id citation text_hash information_need origin_chunk_id origin_source_id instrument_number reference_text article qualifier position heading_path validity_start validity_end projection_ordinal source_name has_more next_offset next_position evidence_next_position scan_truncated evidence_truncated subunit_verified absence_proven paragraph clause operation query query_hash pattern pattern_hash mode read_as_of_date article_closure_complete context".split()
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
        self._compound_owner: tuple[str, str] | None = None

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
                "origin_source_id",
                "instrument_number",
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

    def _compound_read_cursors(
        self, receipt: ToolReceipt, context: RunContext | None
    ) -> bool:
        if (
            receipt.call.name != "compose_tool_calls"
            or context is None
            or not (
                context.services.get("research_profile") == "experimental"
                and (
                    context.services.get("experimental_parallel") is True
                    or (
                        serial_session_diagnostics_enabled(context)
                        and isinstance(owner := context.services.get("task_id"), str)
                        and bool(owner.strip())
                    )
                )
            )
        ):
            return False
        if (
            hashlib.sha256(encoded(context.scope).encode()).hexdigest()
            != self.scope_hash
        ):
            raise ValueError("Working locator scope changed")
        task = context.services.get("task_id")
        owner = task if isinstance(task, str) and task.strip() else "coordinator"
        binding = context.run_id, owner
        if self._compound_owner is not None and self._compound_owner != binding:
            raise ValueError("Compound read locator owner changed")
        ledger = context.services.get("evidence")
        steps = receipt.call.arguments.get("steps")
        results = receipt.outcome.data.get("steps")
        if (
            receipt.call.name != "compose_tool_calls"
            or receipt.outcome.status
            not in {OutcomeStatus.FOUND, OutcomeStatus.PARTIAL}
            or not isinstance(ledger, EvidenceLedger)
            or not isinstance(steps, list)
            or not isinstance(results, dict)
        ):
            return False
        declared = [row.get("id") for row in steps if isinstance(row, dict)]
        if (
            len(declared) != len(steps)
            or any(not isinstance(key, str) or not key for key in declared)
            or len(set(declared)) != len(declared)
            or set(results) - set(declared)
        ):
            return False
        originals = [
            item
            for number in receipt.evidence_ids
            if (item := ledger.get(number)) is not None
            and item.chunk_id
            and item.search_doc is not None
            and item.search_doc.document_id == item.source_id
            and not item.metadata.get("derived")
            and not item.metadata.get("external")
        ]
        if receipt.outcome.evidence:
            originals = [
                item
                for item in originals
                if any(
                    actual.identity == item.identity and actual.text == item.text
                    for actual in receipt.outcome.evidence
                )
            ]
        changed = False
        for step in steps:
            assert isinstance(step, dict)
            name, key, raw = step.get("tool"), step.get("id"), step.get("arguments", {})
            if (
                not isinstance(name, str)
                or name
                not in {
                    "read_provision",
                    "read_source_range",
                    "read_chunk_context",
                    "read_chunk",
                }
                or not isinstance(key, str)
                or not isinstance(raw, dict)
            ):
                continue
            result = results.get(key)
            if (
                not isinstance(result, dict)
                or not isinstance(result.get("status"), str)
                or result.get("status")
                not in {"found", "partial", "truncated", "version_unknown"}
            ):
                continue
            data = result.get("data")
            if not isinstance(data, dict):
                continue
            args = {
                field: _read_argument(value, results)
                for field, value in raw.items()
                if field
                in {
                    "source_id",
                    "chunk_id",
                    "article",
                    "paragraph",
                    "clause",
                    "start",
                    "offset",
                    "limit",
                }
            }
            if any(
                not isinstance(value, str)
                if field in {"source_id", "chunk_id", "article", "paragraph", "clause"}
                else type(value) is not int or value < (1 if field == "limit" else 0)
                for field, value in args.items()
            ):
                continue
            source = args.get("source_id")
            if not isinstance(source, str):
                continue
            try:
                source = str(UUID(source))
            except ValueError:
                continue
            candidates = [item for item in originals if item.source_id == source]
            if not candidates:
                continue
            if name == "read_provision":
                from onyx.asv3.corpus_tools import article_references
                from onyx.regulatory.heading_path import (
                    parse_regulatory_article_heading,
                )

                article = args.get("article")
                if not isinstance(article, str) or data.get("article") != article:
                    continue
                target = article_references(article)
                identities: set[tuple[str, str | None]] = set()
                for item in candidates:
                    headings = item.metadata.get("heading_path")
                    if not isinstance(headings, list):
                        continue
                    for heading in headings:
                        if not isinstance(heading, str):
                            break
                        parsed = parse_regulatory_article_heading(heading)
                        if parsed is not None:
                            identities.add((parsed.article_no, parsed.qualifier))
                            break
                if len(target) != 1 or target[0] not in identities:
                    continue
            elif name == "read_chunk":
                if not any(
                    item.chunk_id == args.get("chunk_id") for item in candidates
                ):
                    continue
            elif name == "read_chunk_context":
                emitted = data.get("delivered_chunk_ids")
                if (
                    data.get("seed_chunk_id") != args.get("chunk_id")
                    or not isinstance(emitted, list)
                    or not emitted
                    or any(
                        chunk not in {item.chunk_id for item in candidates}
                        for chunk in emitted
                    )
                ):
                    continue
            elif name == "read_source_range":
                start = args.get("start", 0)
                if not any(
                    type(position := item.metadata.get("position")) is int
                    and type(start) is int
                    and position >= start
                    for item in candidates
                ):
                    continue
            booleans = {
                "has_more",
                "scan_truncated",
                "evidence_truncated",
                "subunit_verified",
                "article_closure_complete",
                "absence_proven",
            }
            positions = {"next_offset", "next_position", "evidence_next_position"}
            if (
                any(
                    type(data[field]) is not bool for field in booleans if field in data
                )
                or any(
                    data[field] is not None
                    and (type(data[field]) is not int or data[field] < 0)
                    for field in positions
                    if field in data
                )
                or data.get("article_closure_complete") is True
                or data.get("absence_proven") is True
            ):
                continue
            fields: dict[str, JsonValue] = {
                "source_id": source,
                "operation": name,
                "query_hash": hashlib.sha256(encoded(args).encode()).hexdigest(),
                "context": {
                    "kind": "compound_read_navigation",
                    "run_id": context.run_id,
                    "owner": owner,
                    "compound_step_id": key,
                    "arguments": args,
                    "notice": "Returned read navigation; not current model delivery or whole-article proof.",
                },
                **{
                    field: value
                    for field, value in args.items()
                    if field in {"article", "paragraph", "clause", "chunk_id"}
                },
                **{
                    field: value
                    for field, value in data.items()
                    if field in booleans | positions
                },
            }
            nested = receipt.model_copy(
                update={
                    "call": CapabilityCall(
                        name=str(name), call_id=receipt.call.call_id, arguments=args
                    ),
                    "outcome": receipt.outcome.model_copy(
                        update={"status": OutcomeStatus(str(result["status"]))}
                    ),
                }
            )
            self._compound_owner = binding
            changed = self._put("cursor", fields, nested) or changed
        return changed

    def observe(
        self, receipt: ToolReceipt, *, context: RunContext | None = None
    ) -> None:
        with self._lock:
            data, args = receipt.outcome.data, receipt.call.arguments
            need = args.get(
                "_need_id", args.get("coverage_item", args.get("question_id"))
            )
            purpose: dict[str, JsonValue] = (
                {"information_need": need} if isinstance(need, str) else {}
            )
            if isinstance(need, str) and len(need) > 512:
                purpose = {
                    "information_need": "sha256:"
                    + hashlib.sha256(need.encode()).hexdigest()
                }
            changed = self._compound_read_cursors(receipt, context)
            reference_origins: list[tuple[str, str | None, str | None]] = [
                (item.text, item.source_id, item.chunk_id)
                for item in receipt.outcome.evidence
            ]
            if isinstance(data.get("text"), str):
                source_id, chunk_id = data.get("source_id"), data.get("chunk_id")
                reference_origins.append(
                    (
                        str(data["text"]),
                        source_id if isinstance(source_id, str) else None,
                        chunk_id if isinstance(chunk_id, str) else None,
                    )
                )
            for text, origin_source, origin_chunk in reference_origins:
                for reference in statute_references(text):
                    changed = (
                        self._put(
                            "reference",
                            {
                                **purpose,
                                "origin_source_id": origin_source,
                                "origin_chunk_id": origin_chunk,
                                "instrument_number": reference.number,
                                "article": reference.article,
                                "paragraph": reference.paragraph,
                                "clause": reference.clause,
                                "qualifier": reference.qualifier,
                                "reference_text": reference.reference_text,
                                "context": "reference_lead_not_original_governing_evidence",
                            },
                            receipt,
                        )
                        or changed
                    )
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
                encoded(
                    [
                        receipt.call.name,
                        {
                            key: value
                            for key, value in args.items()
                            if key not in {"_public_update", "_need_id"}
                        },
                    ]
                ).encode()
            ).hexdigest()
            substantive = receipt.call.name not in {
                "record_scenario",
                "report_progress",
                "read_research_state",
                "wait_researcher",
                "inspect_evidence_path",
                "read_evidence",
                "read_chunk",
                "read_provision",
                "read_source_range",
                "update_research",
                "inspect_research",
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
            self._compound_owner = None
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
                details = locator.get("context")
                if (
                    isinstance(details, dict)
                    and details.get("kind") == "compound_read_navigation"
                ):
                    run, owner = details.get("run_id"), details.get("owner")
                    if (
                        not isinstance(run, str)
                        or not run
                        or not isinstance(owner, str)
                        or not owner
                    ):
                        raise ValueError("Invalid compound read locator owner")
                    if self._compound_owner is not None and self._compound_owner != (
                        run,
                        owner,
                    ):
                        raise ValueError("Compound read locator owner changed")
                    self._compound_owner = run, owner
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
                # Reopen by locator_id; opaque provider signatures belong in the audit.
                projected = dict(item)
                receipt_id = projected.get("receipt_id")
                if isinstance(receipt_id, str) and len(receipt_id) > 256:
                    projected.pop("receipt_id")
                trial = {
                    **result,
                    "locators": [*selected, projected],
                    "next_offset": index + 1,
                    "omitted_locators": len(ordered) - len(selected) - 1,
                }
                if len(json.dumps(trial, ensure_ascii=False).encode()) > max_bytes:
                    break
                selected.append(projected)
                result = trial
            return copy.deepcopy(result)
