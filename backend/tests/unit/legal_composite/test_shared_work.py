from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Lock

import pytest

from onyx.asv3.models import EvidenceItem, RunContext
from onyx.context.search.models import SearchDoc
from onyx.legal_composite.shared_work import HydratedCenters, SharedCanonicalCenters
from tests.unit.onyx.legal_composite.test_review_assessment import original


def sample() -> tuple[SearchDoc, EvidenceItem]:
    item = original("One immutable canonical original.", "shared")
    assert item.search_doc is not None
    return item.search_doc, item


def test_twelve_lanes_share_one_read_and_receive_independent_copies() -> None:
    doc, item = sample()
    cache = SharedCanonicalCenters()
    start, lock = Barrier(12), Lock()
    calls = 0

    def execute(docs: list[SearchDoc], _context: RunContext) -> HydratedCenters:
        nonlocal calls
        with lock:
            calls += 1
        assert docs == [doc]
        return {(doc.document_id, doc.chunk_ind): [item]}

    def read() -> HydratedCenters:
        start.wait(timeout=3)
        return cache.read([doc], RunContext(), execute, {doc.document_id: 1})

    with ThreadPoolExecutor(max_workers=12) as pool:
        results = list(pool.map(lambda _: read(), range(12)))
    assert calls == 1
    pair = (doc.document_id, doc.chunk_ind)
    results[0][pair][0].question_ids.append("changed")
    assert not results[1][pair][0].question_ids
    assert not item.question_ids


def test_changed_prepared_revision_never_reuses_old_original() -> None:
    doc, item = sample()
    cache = SharedCanonicalCenters()
    calls = 0

    def execute(_docs: list[SearchDoc], _context: RunContext) -> HydratedCenters:
        nonlocal calls
        calls += 1
        return {(doc.document_id, doc.chunk_ind): [item]}

    for revision in (1, 1, 2):
        cache.read([doc], RunContext(), execute, {doc.document_id: revision})
    assert calls == 2


def test_failed_reader_does_not_poison_later_requests_or_spin_on_timeout() -> None:
    doc, item = sample()
    cache = SharedCanonicalCenters()

    def fail(_docs: list[SearchDoc], _context: RunContext) -> HydratedCenters:
        raise TimeoutError("upstream completed with timeout")

    with pytest.raises(TimeoutError, match="upstream"):
        cache.read([doc], RunContext(), fail, {doc.document_id: 1})
    pair = (doc.document_id, doc.chunk_ind)
    result = cache.read(
        [doc], RunContext(), lambda *_: {pair: [item]}, {doc.document_id: 1}
    )
    assert result[pair][0].identity == item.identity


def test_independent_original_sources_are_read_concurrently() -> None:
    doc, item = sample()
    other_item = original("Another immutable original.", "other-source")
    assert other_item.search_doc is not None
    other_doc = other_item.search_doc.model_copy(
        update={"document_id": "another-authorized-source"}
    )
    other_item = other_item.model_copy(
        update={"source_id": other_doc.document_id, "search_doc": other_doc}
    )
    barrier = Barrier(2)

    def execute(docs: list[SearchDoc], _context: RunContext) -> HydratedCenters:
        assert len(docs) == 1
        barrier.wait(timeout=3)
        selected = item if docs[0].document_id == doc.document_id else other_item
        return {(docs[0].document_id, docs[0].chunk_ind): [selected]}

    result = SharedCanonicalCenters().read(
        [doc, other_doc],
        RunContext(),
        execute,
        {doc.document_id: 1, other_doc.document_id: 1},
    )
    assert len(result) == 2
