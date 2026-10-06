"""Coalesce identical canonical reads within one captured research run."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from concurrent.futures import Future, wait
from hashlib import sha256
from uuid import UUID

from pydantic import JsonValue

from onyx.asv3.models import OutcomeStatus, RunContext, ToolOutcome

SharedReadFence = Callable[[str, RunContext], str | ToolOutcome]
SharedReadContext = Callable[[RunContext], RunContext]
SharedReadExecution = Callable[[RunContext], ToolOutcome]

READ_TOOLS = frozenset(
    {"read_chunk", "read_source_range", "read_provision", "read_chunk_context"}
)


class SharedReads:
    def __init__(
        self,
        *,
        fence: SharedReadFence,
        producer_context: SharedReadContext,
    ) -> None:
        self._fence = fence
        self._producer_context = producer_context
        self._lock = threading.Lock()
        self._reads: dict[str, Future[ToolOutcome]] = {}

    @staticmethod
    def _arguments(name: str, arguments: dict[str, JsonValue]) -> dict[str, JsonValue]:
        result = {
            key: value for key, value in arguments.items() if not key.startswith("_")
        }
        if name == "read_source_range":
            result.setdefault("start", 0)
            result.setdefault("limit", 30)
        elif name == "read_chunk_context":
            result.setdefault("offset", 0)
            result.setdefault("limit", 100)
        # An absent provision start uses a structured locator; explicit zero does not.
        return result

    @staticmethod
    def _clone(outcome: ToolOutcome, reuse: str | None = None) -> ToolOutcome:
        result = outcome.model_copy(deep=True)
        if reuse is not None:
            result.data["shared_read_reuse"] = reuse
        return result

    @staticmethod
    def _validate_producer(caller: RunContext, producer: RunContext) -> None:
        if (
            producer.run_id != caller.run_id
            or producer.scope != caller.scope
            or producer.corpus_only != caller.corpus_only
            or producer.budget.tool_slots is not caller.budget.tool_slots
            or producer.budget.source_slots is not caller.budget.source_slots
            or producer.budget.model_slots is not caller.budget.model_slots
        ):
            raise ValueError("Shared read producer must retain the caller's authority")

    @staticmethod
    def _retain(outcome: ToolOutcome, source_id: str) -> bool:
        return (
            outcome.status
            in {OutcomeStatus.FOUND, OutcomeStatus.PARTIAL, OutcomeStatus.TRUNCATED}
            and bool(outcome.evidence)
            and all(
                item.source_id == source_id
                and bool(item.chunk_id)
                and item.search_doc is not None
                and item.search_doc.document_id == source_id
                for item in outcome.evidence
            )
        )

    def run(
        self,
        name: str,
        arguments: dict[str, JsonValue],
        caller: RunContext,
        execute: SharedReadExecution,
    ) -> ToolOutcome:
        source_id = arguments.get("source_id")
        if name not in READ_TOOLS or not isinstance(source_id, str):
            return execute(caller)
        caller.check_active()
        try:
            source_id = str(UUID(source_id))
        except ValueError:
            return execute(caller)
        fence = self._fence(source_id, caller)
        if isinstance(fence, ToolOutcome):
            caller.check_active()
            return self._clone(fence)
        if not fence:
            raise ValueError("Shared read requires a captured source fence")
        material = self._arguments(name, arguments)
        material["source_id"] = source_id
        key = sha256(
            json.dumps(
                [caller.run_id, caller.scope, fence, name, material],
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        caller.check_active()
        with self._lock:
            future = self._reads.get(key)
            leader = future is None
            if future is None:
                future = Future()
                self._reads[key] = future
            reuse = "completed" if future.done() else "inflight"

        if leader:
            try:
                producer = self._producer_context(caller)
                self._validate_producer(caller, producer)
                producer.check_active()
                outcome = execute(producer)
                producer.check_active()
                # Validate original hashes before retaining an immutable acquisition copy.
                snapshot = ToolOutcome.model_validate(outcome.model_dump(mode="python"))
                with self._lock:
                    future.set_result(snapshot)
                    if not self._retain(snapshot, source_id):
                        del self._reads[key]
            except BaseException as error:
                with self._lock:
                    future.set_exception(error)
                    del self._reads[key]
                raise
            caller.check_active()
            return self._clone(snapshot)

        while not future.done():
            caller.check_active()
            wait((future,), timeout=0.05)
        caller.check_active()
        snapshot = future.result()
        caller.check_active()
        return self._clone(snapshot, reuse)
