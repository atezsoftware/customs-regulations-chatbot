"""Keep canonical I/O independent of long searches in owned parallel sessions."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from threading import BoundedSemaphore

from onyx.asv3.legal_source_reviews import serial_session_diagnostics_enabled
from onyx.asv3.models import RunContext
from onyx.tracing.answer_graph import graph_step

CANONICAL_IO_TOOLS = frozenset(
    {
        "resolve_source",
        "read_provision",
        "read_named_provision",
        "read_chunk",
        "read_chunk_context",
        "read_source_range",
        "search_source_text",
        "follow_reference",
        "query_corpus",
        "diagnose_source",
    }
)


def parallel_execution_enabled(context: RunContext) -> bool:
    return context.services.get("research_profile") == "experimental" and (
        context.services.get("experimental_parallel") is True
        or (
            serial_session_diagnostics_enabled(context)
            and isinstance(owner := context.services.get("task_id"), str)
            and bool(owner.strip())
        )
    )


class ParallelExecutionSlots:
    def __init__(self) -> None:
        self.canonical_io = BoundedSemaphore(2)


@contextmanager
def capability_slot(name: str, context: RunContext) -> Iterator[None]:
    slots = context.budget.tool_slots
    lane = "general"
    shared = context.services.get("parallel_execution_slots")
    enabled = parallel_execution_enabled(context)
    if (
        enabled
        and isinstance(shared, ParallelExecutionSlots)
        and name in CANONICAL_IO_TOOLS
    ):
        slots, lane = shared.canonical_io, "canonical_io"
    acquired = False
    try:
        if enabled:
            with graph_step("asv3.tool_admission", {"tool": name, "lane": lane}):
                while not acquired:
                    context.check_active()
                    acquired = slots.acquire(timeout=0.05)
                context.check_active()
        else:
            while not acquired:
                context.check_active()
                acquired = slots.acquire(timeout=0.05)
        yield
    finally:
        if acquired:
            slots.release()
