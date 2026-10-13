"""Topic ownership from actual composed dispatches and committed parent receipts."""

from __future__ import annotations

import copy
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from hashlib import sha256
from threading import RLock
from typing import TYPE_CHECKING

from pydantic import JsonValue

from onyx.asv3.models import (
    CapabilityCall,
    OutcomeStatus,
    RunContext,
    ToolOutcome,
    ToolReceipt,
)

if TYPE_CHECKING:
    from onyx.asv3.cropped_task_packets import CroppedTaskPacketStore
    from onyx.asv3.evidence import EvidenceLedger


@dataclass
class CapturedTopicStep:
    step_id: str
    call: CapabilityCall
    outcome: ToolOutcome


@dataclass
class _Parent:
    store: CroppedTaskPacketStore
    owner: str
    depth: int
    call: CapabilityCall
    children: dict[str, CapturedTopicStep] = field(default_factory=dict)
    outcome: ToolOutcome | None = None
    invalid: bool = False


_OWNER: ContextVar[_Parent | None] = ContextVar(
    "cropped_composed_topic_owner", default=None
)


def _has_ref(value: object) -> bool:
    if isinstance(value, dict):
        return "$ref" in value or any(_has_ref(v) for v in value.values())
    return isinstance(value, list) and any(_has_ref(v) for v in value)


def source_free_invalid_data(data: dict[str, JsonValue]) -> bool:
    """A rejected operation may be retained without claiming a source delivery."""
    allowed = set(topic_step_continuation(data)) | {
        "reason",
        "message",
        "error",
        "error_code",
    }
    return data.keys() <= allowed and all(
        not isinstance(value, (dict, list)) for value in data.values()
    )


def topic_step_continuation(data: dict[str, JsonValue]) -> dict[str, JsonValue]:
    return {
        key: copy.deepcopy(value)
        for key, value in data.items()
        if key
        in {
            "has_more",
            "not_read",
            "scan_truncated",
            "evidence_truncated",
            "truncated",
            "evidence_next_position",
            "next_position",
            "next_offset",
            "cursor",
            "next_cursor",
            "article_closure_complete",
            "absence_proven",
        }
    }


def topic_step_partitionable(
    status: str, continuation: dict[str, JsonValue], tool_name: str
) -> bool:
    terminal_position = (
        tool_name
        in {
            "read_provision",
            "read_named_provision",
            "search_source_text",
            "follow_reference",
        }
        and continuation.get("scan_truncated") is False
        and continuation.get("evidence_truncated") is False
    ) or (
        tool_name in {"read_source_range", "query_corpus"}
        and continuation.get("has_more") is False
    )
    terminal_offset = (
        tool_name in {"read_chunk_context", "resolve_source", "query_corpus"}
        and continuation.get("has_more") is False
    )
    return status == OutcomeStatus.FOUND.value and not (
        any(
            continuation.get(key) is True
            for key in (
                "has_more",
                "not_read",
                "scan_truncated",
                "evidence_truncated",
                "truncated",
            )
        )
        or any(
            continuation.get(key) is not None
            for key in (
                "evidence_next_position",
                "cursor",
                "next_cursor",
            )
        )
        or (continuation.get("next_position") is not None and not terminal_position)
        or (continuation.get("next_offset") is not None and not terminal_offset)
    )


class ComposedTopicCapture:
    def __init__(self, store: CroppedTaskPacketStore) -> None:
        self.store = store
        self._pending: dict[tuple[str, int, str], _Parent] = {}
        self._lock = RLock()

    @staticmethod
    def _key(call: CapabilityCall, context: RunContext) -> tuple[str, int, str]:
        return (
            str(context.services.get("task_id") or "root"),
            context.depth,
            call.call_id,
        )

    def clear(self) -> None:
        with self._lock:
            self._pending.clear()

    @contextmanager
    def parent(self, call: CapabilityCall, context: RunContext) -> Iterator[None]:
        self.store.check_context(context, self.store.request)
        outer = _OWNER.get()
        if outer is not None:
            outer.invalid = True
            yield
            return
        key = self._key(call, context)
        parent = _Parent(self.store, key[0], key[1], call.model_copy(deep=True))
        with self._lock:
            if key in self._pending:
                self._pending.pop(key)
                raise ValueError(
                    "Composed topic parent was dispatched twice before commit"
                )
            self._pending[key] = parent
        token = _OWNER.set(parent)
        try:
            yield
        except BaseException:
            with self._lock:
                self._pending.pop(key, None)
            raise
        finally:
            _OWNER.reset(token)
            if parent.invalid or parent.outcome is None:
                with self._lock:
                    self._pending.pop(key, None)

    def capture(
        self, call: CapabilityCall, outcome: ToolOutcome, context: RunContext
    ) -> None:
        parent = _OWNER.get()
        if parent is None:
            return
        self.store.check_context(context, self.store.request)
        if (
            parent.store is not self.store
            or (parent.owner, parent.depth) != self._key(call, context)[:2]
        ):
            parent.invalid = True
            return
        if call.name == "compose_tool_calls" and call.call_id == parent.call.call_id:
            if call != parent.call or outcome.status not in {
                OutcomeStatus.FOUND,
                OutcomeStatus.PARTIAL,
            }:
                parent.invalid = True
            else:
                parent.outcome = outcome.model_copy(deep=True)
            return
        source_free_invalid = (
            outcome.status == OutcomeStatus.INVALID
            and not outcome.evidence
            and not outcome.original_reads
            and source_free_invalid_data(outcome.data)
        )
        if (
            not source_free_invalid
            and outcome.status
            not in {
                OutcomeStatus.FOUND,
                OutcomeStatus.PARTIAL,
                OutcomeStatus.NOT_FOUND,
                OutcomeStatus.AMBIGUOUS,
            }
        ) or call.argument_error is not None:
            parent.invalid = True
            return
        steps = parent.call.arguments.get("steps")
        matches = (
            [
                row
                for row in steps
                if isinstance(row, dict)
                and row.get("tool") == call.name
                and row.get("arguments", {}) == call.arguments
                and not _has_ref(row)
            ]
            if isinstance(steps, list)
            else []
        )
        if len(matches) != 1 or not isinstance(matches[0].get("id"), str):
            parent.invalid = True
            return
        step_id = str(matches[0]["id"])
        with self._lock:
            if step_id in parent.children:
                parent.invalid = True
                return
            for item in outcome.evidence:
                if sha256(item.text.encode()).hexdigest() != item.text_hash:
                    parent.invalid = True
                    return
            parent.children[step_id] = CapturedTopicStep(
                step_id, call.model_copy(deep=True), outcome.model_copy(deep=True)
            )

    def take(
        self, receipt: ToolReceipt, context: RunContext
    ) -> list[CapturedTopicStep]:
        self.store.check_context(context, self.store.request)
        with self._lock:
            parent = self._pending.pop(self._key(receipt.call, context), None)
        if (
            parent is None
            or parent.invalid
            or parent.outcome is None
            or parent.call != receipt.call
        ):
            return []
        from onyx.asv3.artifacts import compact_json
        from onyx.asv3.cropped_article_bindings import without_ordered_source_read_data

        if receipt.outcome.data != compact_json(
            without_ordered_source_read_data(parent.outcome.data)
        ):
            return []
        if receipt.outcome.status != parent.outcome.status:
            return []
        steps = parent.call.arguments.get("steps")
        outcomes = parent.outcome.data.get("steps")
        if not isinstance(steps, list) or not isinstance(outcomes, dict):
            return []
        ids = [row.get("id") for row in steps if isinstance(row, dict)]
        if (
            len(ids) != len(steps)
            or len(set(ids)) != len(ids)
            or set(ids) != parent.children.keys()
            or set(ids) != outcomes.keys()
        ):
            return []
        for step_id, child in parent.children.items():
            if outcomes[step_id] != child.outcome.model_dump(
                mode="json", exclude={"evidence", "artifacts"}
            ):
                return []
        return [copy.deepcopy(parent.children[str(step_id)]) for step_id in ids]


def _store(context: RunContext) -> CroppedTaskPacketStore | None:
    from onyx.asv3.cropped_task_packets import CroppedTaskPacketStore

    store = context.services.get("cropped_task_packet_store")
    return (
        store
        if context.services.get("asv3_exp_cropped") is True
        and context.services.get("asv3_exp_cheap") is not True
        and isinstance(store, CroppedTaskPacketStore)
        else None
    )


@contextmanager
def cropped_composed_topic_capture(
    call: CapabilityCall, context: RunContext
) -> Iterator[None]:
    store = _store(context)
    if store is None or call.name != "compose_tool_calls":
        yield
        return
    with store.composed_capture.parent(call, context):
        yield


def capture_cropped_dispatched_topic(
    call: CapabilityCall, outcome: ToolOutcome, context: RunContext
) -> None:
    store = _store(context)
    if store is not None:
        store.composed_capture.capture(call, outcome, context)


def commit_cropped_composed_topics(
    receipt: ToolReceipt, context: RunContext, ledger: EvidenceLedger
) -> None:
    store = _store(context)
    if store is not None and receipt.call.name == "compose_tool_calls":
        store.commit_composed_topics(receipt, context, ledger)
