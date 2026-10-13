"""Exact search packet batching preserves live source and original-read fences."""

from collections.abc import Generator
from contextlib import nullcontext
from dataclasses import replace
from typing import Any, cast
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest
from sqlalchemy.orm import Session

from onyx.asv3 import corpus_tools
from onyx.asv3.models import RunContext
from onyx.context.search.models import IndexFilters
from onyx.db import asv3_corpus as corpus
from onyx.db.asv3_corpus import CorpusChunk, CorpusScopeUnavailable, CorpusSource
from onyx.db.models import User


def fixture(monkeypatch: pytest.MonkeyPatch, count: int = 2) -> dict[str, Any]:
    sources = {
        identifier: CorpusSource(identifier, str(n), str(n))
        for n in range(count)
        if (identifier := uuid4())
    }
    originals = {
        key: CorpusChunk(
            str(key),
            key,
            f"exact {n}",
            n,
            n,
            ("MADDE 1",),
            {"own": n},
            None,
            None,
            "active",
        )
        for n, key in enumerate(sources)
    }
    calls: list[tuple[Session, tuple[UUID, ...]]] = []
    sessions: list[Session] = []
    reads: list[UUID] = []

    def session() -> Any:
        value = MagicMock(spec=Session)
        sessions.append(value)
        return nullcontext(value)

    def find(current: Session, **kwargs: Any) -> tuple[list[CorpusSource], bool]:
        ids = kwargs["source_ids"]
        calls.append((current, ids))
        assert kwargs["limit"] == len(ids) <= 100
        return [sources[key] for key in ids], False

    def read(_session: Session, **kwargs: Any) -> Generator[CorpusChunk, None, None]:
        key = kwargs["source_id"]
        reads.append(key)
        assert kwargs["chunk_ids"] == (str(key),)
        kwargs["check_active"]()
        yield originals[key]

    monkeypatch.setattr(corpus, "get_session_with_current_tenant", session)
    monkeypatch.setattr(corpus, "find_sources", find)
    monkeypatch.setattr(corpus, "_iter_exact_source_chunks", read)
    monkeypatch.setattr(corpus, "resolve_source_query_index", lambda *_: None)
    monkeypatch.setattr(corpus, "observe_publication_read", lambda: "fixed observation")
    publication = MagicMock()
    monkeypatch.setattr(corpus, "require_publication_files", publication)
    return {
        "kwargs": dict(
            user=cast(User, object()),
            filters=IndexFilters(access_control_list=[]),
            source_chunk_ids={key: (str(key), str(key)) for key in sources},
            query_indexes={},
            check_active=lambda: None,
        ),
        "calls": calls,
        "sessions": sessions,
        "reads": reads,
        "sources": sources,
        "originals": originals,
        "publication": publication,
    }


def test_exact_source_batch_preserves_all_values_and_fresh_final_sessions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = fixture(monkeypatch)
    result = corpus.read_search_centers_batch(**state["kwargs"])
    assert list(result) == list(state["sources"])
    assert {key: value.source for key, value in result.items()} == state["sources"]
    assert {key: value.chunks for key, value in result.items()} == {
        key: (value,) for key, value in state["originals"].items()
    }
    assert len(state["calls"]) == 2
    assert state["calls"][0][0] is not state["calls"][1][0]
    assert state["publication"].call_count == 2
    assert all(
        call.args == ("fixed observation", tuple(state["sources"]))
        for call in state["publication"].call_args_list
    )


def test_more_than_100_requested_sources_are_all_authorized_at_both_fences(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = fixture(monkeypatch, 207)
    result = corpus.read_search_centers_batch(**state["kwargs"])
    assert len(result) == 207
    pages = [ids for _, ids in state["calls"]]
    assert [len(page) for page in pages] == [100, 100, 7, 100, 100, 7]
    assert pages[:3] == pages[3:]
    assert {key for page in pages[:3] for key in page} == set(state["sources"])


@pytest.mark.parametrize(
    "defect",
    [
        "entry_acl",
        "exit_acl",
        "source_identity",
        "entry_publication",
        "exit_publication",
        "cancel",
        "read",
        "duplicate",
        "wrong_source",
        "extra_id",
    ],
)
def test_failed_fence_never_returns_partial_search_packet(
    monkeypatch: pytest.MonkeyPatch, defect: str
) -> None:
    state = fixture(monkeypatch)
    find = corpus.find_sources
    calls = 0

    def checked(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        rows, more = find(*args, **kwargs)
        if (defect == "entry_acl" and calls == 1) or (
            defect == "exit_acl" and calls == 2
        ):
            return rows[:-1], False
        if defect == "source_identity" and calls == 2:
            rows[0] = replace(rows[0], file_id="replacement")
        return rows, more

    monkeypatch.setattr(corpus, "find_sources", checked)
    if defect.endswith("publication"):
        state["publication"].side_effect = (
            [PermissionError("publication changed")]
            if defect.startswith("entry")
            else [None, PermissionError("publication changed")]
        )
    if defect == "cancel":

        def active() -> None:
            if calls == 2:
                raise RuntimeError("cancelled before return")

        state["kwargs"]["check_active"] = active
    if defect in {"read", "duplicate", "wrong_source", "extra_id"}:
        read = corpus._iter_exact_source_chunks

        def changed(*args: Any, **kwargs: Any) -> Generator[CorpusChunk, None, None]:
            for item in read(*args, **kwargs):
                if defect == "wrong_source":
                    yield replace(item, source_id=uuid4())
                elif defect == "extra_id":
                    yield replace(item, id="unrequested")
                else:
                    yield item
                    if defect == "read":
                        raise RuntimeError("stream interrupted")
                    yield item

        monkeypatch.setattr(corpus, "_iter_exact_source_chunks", changed)
    returned: dict[UUID, Any] = {}
    with pytest.raises((PermissionError, CorpusScopeUnavailable, RuntimeError)):
        returned.update(corpus.read_search_centers_batch(**state["kwargs"]))
    assert not returned
    if defect.startswith("entry"):
        assert not state["reads"]


@pytest.mark.parametrize("cropped", [None, False, True, "true"])
def test_only_explicit_cropped_opt_in_dispatches_batch(
    monkeypatch: pytest.MonkeyPatch, cropped: object
) -> None:
    broker = corpus_tools.CorpusBroker(
        cast(User, object()), IndexFilters(access_control_list=[])
    )
    batch = MagicMock(return_value={})
    monkeypatch.setattr(broker, "_hydrate_search_centers_batch", batch)
    broker.hydrate_search_centers(
        [], RunContext(services={"asv3_exp_cropped": cropped})
    )
    assert batch.call_count == (1 if cropped is True else 0)


def test_broker_preserves_exact_evidence_metadata_and_missing_centers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from onyx.configs.constants import DocumentSource
    from onyx.context.search.models import SearchDoc
    from onyx.db.asv3_corpus import CorpusSearchCenterRead

    state = fixture(monkeypatch, 1)
    key = next(iter(state["sources"]))
    source = state["sources"][key]
    chunk = state["originals"][key]
    docs = [
        SearchDoc(
            document_id=str(key),
            chunk_ind=ordinal,
            semantic_identifier="Source",
            link=None,
            blurb="projection only",
            source_type=DocumentSource.USER_FILE,
            boost=0,
            hidden=False,
            metadata={"regulatory_chunk_id": identifier},
            match_highlights=[],
        )
        for identifier, ordinal in [
            (chunk.id, chunk.projection_ordinal),
            ("missing", 2),
            (chunk.id, 99),
        ]
    ]
    monkeypatch.setattr(
        corpus_tools,
        "get_session_with_current_tenant",
        lambda: nullcontext(MagicMock(spec=Session)),
    )
    monkeypatch.setattr(
        corpus_tools, "require_source", lambda *_args, **_kwargs: source
    )
    monkeypatch.setattr(corpus_tools, "resolve_source_query_index", lambda *_args: None)

    def stream(*_args: Any, **_kwargs: Any) -> Generator[CorpusChunk, None, None]:
        yield chunk

    monkeypatch.setattr(corpus_tools, "iter_source_chunks_by_ids", stream)
    monkeypatch.setattr(
        corpus_tools,
        "read_search_centers_batch",
        lambda **_kwargs: {key: CorpusSearchCenterRead(source, (chunk,), None)},
    )
    broker = corpus_tools.CorpusBroker(
        state["kwargs"]["user"], state["kwargs"]["filters"]
    )
    old = broker.hydrate_search_centers(docs, RunContext())
    new = broker.hydrate_search_centers(
        docs, RunContext(services={"asv3_exp_cropped": True})
    )
    assert old == new
    assert [len(value) for value in new.values()] == [1, 0, 0]
    assert new[(str(key), 0)][0].metadata["retrieved_projection_ordinal"] == 0


@pytest.mark.parametrize("when", ["captured_conflict", "concurrent_change"])
def test_broker_never_delivers_on_physical_index_conflict(
    monkeypatch: pytest.MonkeyPatch, when: str
) -> None:
    from threading import RLock
    from types import SimpleNamespace

    from onyx.configs.constants import DocumentSource
    from onyx.context.search.models import SearchDoc
    from onyx.db.asv3_corpus import CorpusSearchCenterRead
    from onyx.document_index.publication_models import PublicationIndexSnapshot

    state = fixture(monkeypatch, 1)
    key = next(iter(state["sources"]))
    doc = SearchDoc(
        document_id=str(key),
        chunk_ind=0,
        semantic_identifier="Source",
        link=None,
        blurb="",
        source_type=DocumentSource.USER_FILE,
        boost=0,
        hidden=False,
        metadata={"regulatory_chunk_id": str(key)},
        match_highlights=[],
    )
    broker = corpus_tools.CorpusBroker(
        state["kwargs"]["user"], state["kwargs"]["filters"]
    )
    previous = MagicMock(spec=PublicationIndexSnapshot)
    previous.matches_temporal_index.return_value = False
    captured = MagicMock(spec=PublicationIndexSnapshot)
    scope = SimpleNamespace(lock=RLock(), query_indexes={key: captured})
    monkeypatch.setattr(
        corpus_tools, "current_asv3_source_inventory_scope", lambda: scope
    )
    if when == "captured_conflict":
        broker.query_indexes[key] = previous

    def read(**_kwargs: Any) -> Any:
        broker.query_indexes[key] = previous
        return {
            key: CorpusSearchCenterRead(
                state["sources"][key], (state["originals"][key],), captured
            )
        }

    batch = MagicMock(side_effect=read)
    monkeypatch.setattr(corpus_tools, "read_search_centers_batch", batch)
    with pytest.raises(CorpusScopeUnavailable, match="captured authority"):
        broker.hydrate_search_centers(
            [doc], RunContext(services={"asv3_exp_cropped": True})
        )
    assert batch.call_count == (0 if when == "captured_conflict" else 1)
