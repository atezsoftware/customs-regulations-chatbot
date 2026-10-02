"""Source traversal keeps locators and selected evidence rather than whole files."""

import threading
import time
import tracemalloc
import weakref
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from uuid import UUID, uuid4

import pytest
from pydantic import JsonValue

from onyx.asv3.corpus_tools import MAX_SCAN_BYTES, CorpusBroker, build_corpus_specs
from onyx.asv3.models import (
    CapabilityCall,
    OutcomeStatus,
    RunContext,
    RunStopped,
    SharedBudget,
    ToolOutcome,
)
from onyx.asv3.registry import CapabilityRegistry
from onyx.asv3.source_tools import build_source_specs, source_slot
from onyx.context.search.models import IndexFilters
from onyx.db.asv3_corpus import CorpusChunk, CorpusSource


class LazyBroker(CorpusBroker):
    """Allocate each page on demand; no test fixture retains the input chunks."""

    def __init__(self, *, count: int = 500, chunk_size: int = 3000) -> None:
        self.items = {
            str(identifier): CorpusSource(
                identifier, f"Source {number}", str(identifier)
            )
            for number, identifier in enumerate(uuid4() for _ in range(5))
        }
        self.count = count
        self.chunk_size = chunk_size
        self.filters = IndexFilters(access_control_list=[])
        self.search_adapter = None
        self.refs: list[weakref.ReferenceType[CorpusChunk]] = []
        self.max_live_chunks = 0
        self.pages: list[tuple[str, int, date | None]] = []

    def provision_start(
        self, source_id: str, article: str, qualifier: str | None, context: RunContext
    ) -> int | None:
        del article, qualifier
        self.source(source_id, context)
        return None

    def source(self, source_id: str, context: RunContext) -> CorpusSource:
        context.check_active()
        if source_id not in self.items:
            raise PermissionError("Scope denied")
        return self.items[source_id]

    def page(
        self,
        source_id: str,
        context: RunContext,
        *,
        start: int = 0,
        limit: int = 50,
        as_of: date | None = None,
        historical_inventory: bool = False,
    ) -> tuple[CorpusSource, list[CorpusChunk], bool]:
        del historical_inventory
        source = self.source(source_id, context)
        self.pages.append((source_id, start, as_of))
        chunks = []
        for position in range(start, min(start + limit, self.count)):
            header = f"{position} {as_of or 'current'} "
            if 470 <= position < 480:
                header += "royalti "
            text = header + "x" * (self.chunk_size - len(header))
            chunk = CorpusChunk(
                str(position),
                UUID(source_id),
                text,
                position,
                position,
                (f"MADDE {position // 10 + 1}",) if position % 10 == 0 else (),
                {},
                date(2000, 1, 1),
                None,
                "active",
            )
            chunks.append(chunk)
            self.refs.append(weakref.ref(chunk))
        self.max_live_chunks = max(
            self.max_live_chunks, sum(ref() is not None for ref in self.refs)
        )
        return source, chunks, start + limit < self.count


def invoke(
    broker: LazyBroker, name: str, arguments: dict[str, JsonValue]
) -> ToolOutcome:
    return CapabilityRegistry(build_corpus_specs(broker)).dispatch(
        CapabilityCall(name=name, arguments=arguments), RunContext()
    )


def test_five_long_sources_keep_only_selected_chunks_and_one_live_page() -> None:
    broker = LazyBroker()
    tracemalloc.start()
    try:
        results = [
            invoke(broker, "read_provision", {"source_id": source_id, "article": "48"})
            for source_id in broker.items
        ]
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert all(result.status == OutcomeStatus.FOUND for result in results)
    assert all(
        [item.chunk_id for item in result.evidence]
        == [str(position) for position in range(470, 480)]
        for result in results
    )
    assert (
        len(broker.pages) == 25
    )  # All relevant and irrelevant chunks were still scanned.
    assert broker.max_live_chunks <= 101
    assert not any(ref() is not None for ref in broker.refs)
    assert (
        sum(len(item.text) for result in results for item in result.evidence) == 150_000
    )
    assert peak < 1_600_000


def test_heading_inventory_retains_only_locators_without_losing_any_heading() -> None:
    broker = LazyBroker()
    result = invoke(
        broker,
        "query_corpus",
        {"operation": "headings", "source_id": next(iter(broker.items))},
    )
    headings = result.data["headings"]
    assert isinstance(headings, list) and len(headings) == 500
    assert all(
        isinstance(locator, dict) and "text" not in locator for locator in headings
    )
    assert result.data["has_more"] is False
    assert result.data["next_position"] == 500
    assert broker.max_live_chunks <= 101


def test_byte_budget_is_partial_and_can_resume_without_false_not_found() -> None:
    broker = LazyBroker(count=500, chunk_size=10_000)
    source_id = next(iter(broker.items))
    result = invoke(
        broker, "search_source_text", {"source_id": source_id, "pattern": "royalti"}
    )
    assert result.status == OutcomeStatus.PARTIAL
    assert not result.evidence
    assert result.data["absence_proven"] is False
    assert result.data["next_position"] == MAX_SCAN_BYTES // 10_000
    resumed = invoke(
        broker,
        "search_source_text",
        {"source_id": source_id, "pattern": "royalti", "start": 420},
    )
    assert (
        resumed.status == OutcomeStatus.PARTIAL
    )  # Response text is clipped, not silently lost.
    assert [item.chunk_id for item in resumed.evidence] == [
        str(position) for position in range(470, 476)
    ]
    assert resumed.data["evidence_next_position"] == 476
    assert broker.max_live_chunks <= 101


def test_scoped_comparison_retains_only_both_requested_article_versions() -> None:
    broker = LazyBroker()
    result = invoke(
        broker,
        "compare_versions",
        {
            "source_id": next(iter(broker.items)),
            "old_date": "2024-01-01",
            "new_date": "2025-01-01",
            "article": "48",
        },
    )
    assert result.status == OutcomeStatus.FOUND
    assert len(result.evidence) == 20
    assert all(
        item.chunk_id is not None and 470 <= int(item.chunk_id) < 480
        for item in result.evidence
    )
    assert "2024-01-01" in str(result.data["diff"])
    assert "2025-01-01" in str(result.data["diff"])
    assert broker.max_live_chunks <= 101
    assert len(broker.pages) == 10


def test_original_operations_share_two_slots_for_the_complete_bytes_lifetime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from onyx.asv3 import source_tools

    broker = LazyBroker()
    context = RunContext()
    active = peak = 0
    lock = threading.Lock()

    def original(
        _broker: CorpusBroker, _source_id: str, run: RunContext
    ) -> tuple[bytes, str, str]:
        nonlocal active, peak
        # Nested file access must be reentrant and preserve the outer permit.
        with source_slot(run):
            with lock:
                active += 1
                peak = max(peak, active)
        return b"payload", "text/plain", "source"

    def extracted(*_args: object) -> object:
        nonlocal active
        time.sleep(0.03)
        with lock:
            assert active <= 2
            active -= 1
        return source_tools.EvidenceItem(
            source_id=next(iter(broker.items)),
            text="payload",
            metadata={"source_sha256": "a" * 64},
        )

    monkeypatch.setattr(source_tools, "read_verified_original", original)
    monkeypatch.setattr(source_tools, "original_evidence", extracted)
    spec = next(
        spec for spec in build_source_specs(broker) if spec.name == "open_source_file"
    )
    with ThreadPoolExecutor(max_workers=5) as pool:
        results = list(
            pool.map(
                lambda source_id: spec.handler(
                    {"source_id": source_id}, context.child()
                ),
                broker.items,
            )
        )
    assert all(result.status == OutcomeStatus.FOUND for result in results)
    assert peak == 2 and active == 0


def test_waiting_source_operation_cancels_without_starting_and_returns_permits() -> (
    None
):
    budget = SharedBudget(max_inflight_sources=1)
    context = RunContext(budget=budget)
    budget.source_slots.acquire()
    started = threading.Event()

    def wait() -> None:
        started.set()
        with source_slot(context):
            pytest.fail("Cancelled operation acquired a slot")

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(wait)
        assert started.wait(1)
        context.cancel()
        with pytest.raises(RunStopped, match="cancelled"):
            future.result(timeout=1)
    budget.source_slots.release()
    fresh = RunContext(budget=budget)
    with source_slot(fresh):
        with source_slot(fresh.child()):
            fresh.check_active()


def test_source_slot_honors_research_reserve_but_final_revalidation_can_continue() -> (
    None
):
    context = RunContext(
        deadline=time.monotonic() + 2, research_deadline=time.monotonic() - 1
    )
    with pytest.raises(RunStopped, match="finalization"):
        with source_slot(context):
            pytest.fail("Research used reserved finalization time")
    with source_slot(context, research=False):
        context.check_active()
