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
