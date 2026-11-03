"""Normal ASv3 releases canonical leases before independent publication reads."""

from collections.abc import Generator
from contextlib import contextmanager
from threading import Barrier, Lock
from typing import Any, cast
from uuid import UUID, uuid4

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session
from sqlalchemy.pool import QueuePool

from onyx.asv3 import corpus_tools
from onyx.asv3.corpus_tools import CorpusBroker, evidence_for_chunk
from onyx.asv3.models import RunContext
from onyx.configs.constants import DocumentSource
from onyx.context.search.models import IndexFilters
from onyx.db import asv3_corpus
from onyx.db.asv3_corpus import CorpusChunk, CorpusSource
from onyx.db.models import User
from tests.unit.onyx.asv3.test_cropped_revalidation_packet import setup_packet


@pytest.mark.parametrize("revoked_at_exit", [False, True])
def test_standard_final_revalidation_does_not_self_exhaust_a_four_connection_pool(
    monkeypatch: pytest.MonkeyPatch, revoked_at_exit: bool
) -> None:
    engine = create_engine(
        "sqlite://",
        poolclass=QueuePool,
        pool_size=4,
        max_overflow=0,
        pool_timeout=0.05,
        connect_args={"check_same_thread": False},
    )
    gate = Barrier(4)
    lock = Lock()
    active_reads = 0
    peak_checkouts = 0
    acl_reads = 0
    publication_reads = 0
    identifiers = [uuid4() for _ in range(4)]
    sources = {key: CorpusSource(key, str(key), str(key)) for key in identifiers}
    chunks = {
        key: CorpusChunk(
            str(key),
            key,
            "Exact original condition.",
            0,
            7,
            ("MADDE 1",),
            {},
            None,
            None,
            "active",
        )
        for key in identifiers
    }

    @contextmanager
    def session() -> Generator[Session, None, None]:
        nonlocal peak_checkouts
        with Session(engine) as current:
            current.execute(text("SELECT 1"))
            with lock:
                peak_checkouts = max(
                    peak_checkouts, cast(QueuePool, engine.pool).checkedout()
                )
            yield current

    def publication(*_args: Any, **_kwargs: Any) -> object:
        nonlocal publication_reads
        with lock:
            assert active_reads == 0, (
                "Publication check retained a canonical read lease"
            )
            publication_reads += 1
        with session():
            return object()

    def exact_sources(
        _session: Session,
        *,
        source_ids: tuple[UUID, ...],
        filters: IndexFilters,
        **_kwargs: Any,
    ) -> dict[UUID, CorpusSource]:
        nonlocal acl_reads
        assert filters.asv3_document_set_id == 15
        assert filters.forced_document_set == ["PC Külliyatı"]
        acl_reads += 1
        # Real source discovery also makes an independent publication read.
        publication()
        if revoked_at_exit and acl_reads == 2:
            raise PermissionError("Source access revoked at final fence")
        return {key: sources[key] for key in source_ids}

    seen: set[UUID] = set()

    def legacy_source(
        _session: Session, *, source_id: UUID, **_kwargs: Any
    ) -> CorpusSource:
        # The previous per-source path holds all four leases before requesting
        # independent publication connections and consequently cannot finish.
        with lock:
            first = source_id not in seen
            seen.add(source_id)
        if first:
            gate.wait(timeout=2)
        return sources[source_id]

    def originals(
        _session: Session,
        *,
        source_id: UUID,
        chunk_ids: tuple[str, ...],
        check_active: Any,
        **_kwargs: Any,
    ) -> Generator[CorpusChunk, None, None]:
        nonlocal active_reads
        assert chunk_ids == (chunks[source_id].id,)
        with lock:
            active_reads += 1
        try:
            gate.wait(timeout=2)
            check_active()
            yield chunks[source_id]
        finally:
            with lock:
                active_reads -= 1

    monkeypatch.setattr(corpus_tools, "get_session_with_current_tenant", session)
    monkeypatch.setattr(asv3_corpus, "get_session_with_current_tenant", session)
    monkeypatch.setattr(corpus_tools, "require_source", legacy_source)
    monkeypatch.setattr(asv3_corpus, "require_source", legacy_source)
    monkeypatch.setattr(corpus_tools, "resolve_source_query_index", lambda *_: None)
    monkeypatch.setattr(asv3_corpus, "resolve_source_query_index", lambda *_: None)
    monkeypatch.setattr(asv3_corpus, "_require_exact_sources", exact_sources)
    monkeypatch.setattr(asv3_corpus, "_iter_exact_source_chunks", originals)
    monkeypatch.setattr(asv3_corpus, "observe_publication_read", publication)
    monkeypatch.setattr(asv3_corpus, "require_publication_files", publication)
    filters = IndexFilters(
        access_control_list=[],
        source_type=[DocumentSource.USER_FILE],
        forced_document_set=["PC Külliyatı"],
        asv3_document_set_id=15,
    )
    broker = CorpusBroker(cast(User, object()), filters)
    retained = [evidence_for_chunk(sources[key], chunks[key]) for key in identifiers]
    before = [item.model_dump(mode="json") for item in retained]
    try:
        if revoked_at_exit:
            with pytest.raises(PermissionError, match="final fence"):
                broker.revalidate_evidence(
                    retained, RunContext(services={"asv3_standard_normal": True})
                )
        else:
            broker.revalidate_evidence(
                retained, RunContext(services={"asv3_standard_normal": True})
            )
        assert peak_checkouts == 4
        assert cast(QueuePool, engine.pool).checkedout() == 0
        assert acl_reads == 2
        assert publication_reads >= 4
        assert before == [item.model_dump(mode="json") for item in retained]
    finally:
        engine.dispose()


def test_standard_packet_keeps_dated_duplicates_and_immutable_witnesses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = setup_packet(monkeypatch)
    context = RunContext(services={"asv3_standard_normal": True})
    state["broker"].revalidate_evidence(state["items"], context)
    assert [day for day, _ in state["calls"]] == state["days"]
    state["items"][-1].text_hash = "0" * 64
    with pytest.raises(asv3_corpus.CorpusScopeUnavailable):
        state["broker"].revalidate_evidence(state["items"], context)
