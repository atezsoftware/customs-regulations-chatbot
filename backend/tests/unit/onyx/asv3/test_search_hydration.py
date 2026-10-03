"""Production closure selection with DB/ACL boundaries scripted, not live performance proof."""

import json
import time
import tracemalloc
from collections.abc import Callable, Iterator
from contextlib import nullcontext
from contextvars import ContextVar
from datetime import date
from threading import Barrier
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest
from sqlalchemy.orm import Session

from onyx.asv3 import corpus_tools
from onyx.asv3.corpus_tools import CorpusBroker, evidence_for_chunk
from onyx.asv3.models import RunContext, RunStopped
from onyx.configs.constants import DocumentSource
from onyx.context.search.models import IndexFilters, SearchDoc
from onyx.db import asv3_corpus
from onyx.db.asv3_corpus import (
    CorpusChunk,
    CorpusClosureRead,
    CorpusScopeUnavailable,
    CorpusSource,
)
from onyx.db.models import RegulatoryChunk, User
from onyx.document_index.publication_models import PublicationIndexSnapshot
from onyx.regulatory.amendments.annexes.models import AnnexTemporalProjection


class Rows:
    def __init__(self, values: list[Any]) -> None:
        self.values = values
        self.closed = False

    def __iter__(self) -> Iterator[Any]:
        yield from self.values

    def close(self) -> None:
        self.closed = True


def row(
    source: UUID, position: int, heading: list[str], text: str = "original legal text"
) -> RegulatoryChunk:
    return RegulatoryChunk(
        id=f"rc-{position}-{uuid4().hex[:6]}",
        user_file_id=source,
        position=position,
        projection_ordinal=position,
        text=text,
        status="active",
        heading_path=heading,
        chunk_metadata={},
        chunk_type="paragraph",
        validity_start_date=None,
        validity_end_date=None,
    )


def boundary(
    monkeypatch: pytest.MonkeyPatch,
    rows: list[RegulatoryChunk],
    *,
    as_of: date | None = None,
) -> tuple[MagicMock, dict[str, Any]]:
    source = CorpusSource(rows[0].user_file_id, "Test legal source", "original-file")
    session = MagicMock(spec=Session)
    streams: list[Rows] = []
    monkeypatch.setattr(asv3_corpus, "require_source", lambda *_args, **_kwargs: source)
    monkeypatch.setattr(asv3_corpus, "qualified_file_ids", lambda *_args: frozenset())
    monkeypatch.setattr(asv3_corpus, "observe_publication_read", lambda: object())
    monkeypatch.setattr(
        asv3_corpus,
        "filter_publication_read",
        lambda _observation, values, _key: values,
    )
    monkeypatch.setattr(asv3_corpus, "require_publication_files", lambda *_args: None)

    def visible(item: RegulatoryChunk) -> bool:
        if as_of is None:
            return item.status == "active"
        return (
            item.validity_start_date is None or item.validity_start_date <= as_of
        ) and (item.validity_end_date is None or item.validity_end_date > as_of)

    def execute(statement: Any) -> Rows:
        assert "text" not in [column.key for column in statement.selected_columns]
        result = Rows(
            [
                SimpleNamespace(
                    **{
                        name: getattr(item, name)
                        for name in (
                            "id",
                            "position",
                            "heading_path",
                            "chunk_metadata",
                            "chunk_type",
                            "status",
                            "validity_start_date",
                            "validity_end_date",
                            "projection_ordinal",
                        )
                    }
                )
                for item in rows
                if visible(item)
            ]
        )
        streams.append(result)
        return result

    def scalars(statement: Any) -> Rows:
        parameters = statement.compile().params
        identifiers = next(
            value for value in parameters.values() if isinstance(value, (list, tuple))
        )
        result = Rows(
            [item for item in rows if item.id in identifiers and visible(item)]
        )
        streams.append(result)
        return result

    session.execute.side_effect = execute
    session.scalars.side_effect = scalars
    return session, dict(
        session=session,
        user=cast(User, object()),
        filters=IndexFilters(
            access_control_list=[],
            source_type=[DocumentSource.USER_FILE],
            as_of_date=as_of,
        ),
        source_id=source.id,
        index=None,
        check_active=lambda: None,
        _streams=streams,
    )


def read(
    arguments: dict[str, Any], centers: list[RegulatoryChunk], **budgets: int
) -> CorpusClosureRead:
    return asv3_corpus.read_search_source_closures(
        **{key: value for key, value in arguments.items() if not key.startswith("_")},
        center_ids=tuple(item.id for item in centers),
        **budgets,
    )


def test_many_centers_share_one_metadata_outline_and_only_selected_text_reads(
    monkeypatch: pytest.MonkeyPatch, record_property: Callable[[str, object], None]
) -> None:
    source = uuid4()
    items = [
        row(
            source,
            n,
            ["Law", f"MADDE {n // 3 + 1}", f"({n % 3 + 1})"],
            "original " * 100,
        )
        for n in range(5000)
    ]
    centers = [items[n] for n in (0, 2004, 2499, 3003, 3600, 4200, 4800)]
    session, arguments = boundary(monkeypatch, items)
    tracemalloc.start()
    started = time.monotonic()
    try:
        result = read(arguments, centers)
        elapsed = time.monotonic() - started
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    expected = {
        item.id
        for center in centers
        for item in items
        if item.position // 3 == center.position // 3
    }
    assert result.outline_rows == 5000 and not result.outline_truncated
    assert {item.id for item in result.chunks} == expected
    assert all(result.complete.values())
    assert session.execute.call_count == 1
    assert session.scalars.call_count == 2
    assert all(stream.closed for stream in arguments["_streams"])
    assert sum(len(item.text) for item in result.chunks) == len(expected) * 900
    record_property("scripted_5000_row_seconds", elapsed)
    record_property("python_peak_bytes_without_fixture_allocation", peak)
    record_property("metadata_queries", session.execute.call_count)
    record_property("selected_text_queries", session.scalars.call_count)


def test_same_number_different_annex_and_qualifier_never_join(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = uuid4()
    items = [
        row(source, 0, ["Law", "Ek 1", "MADDE 3", "(1)"]),
        row(source, 1, ["Law", "Ek 1", "MADDE 3", "(2)"]),
        row(source, 2, ["Law", "Ek 2", "MADDE 3"]),
        row(source, 3, ["Law", "Ek 2", "GEÇİCİ MADDE 3"]),
        row(source, 4, ["Law", "Ek 2", "MÜKERRER MADDE 3"]),
    ]
    _, arguments = boundary(monkeypatch, items)
    result = read(arguments, [items[0], items[3], items[4]])
    assert result.members[items[0].id] == (items[0].id, items[1].id)
    assert result.members[items[3].id] == (items[3].id,)
    assert result.members[items[4].id] == (items[4].id,)
    assert items[2].id not in {chunk.id for chunk in result.chunks}
    assert all(result.complete.values())


def test_identityless_bridge_is_read_but_unknown_outer_tail_is_partial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = uuid4()
    items = [
        row(source, 0, ["Law", "MADDE 3", "(1)"]),
        row(source, 1, []),
        row(source, 2, ["Law", "MADDE 3", "(2)"]),
        row(source, 3, []),
    ]
    _, arguments = boundary(monkeypatch, items)
    result = read(arguments, [items[0]])
    assert result.members[items[0].id] == tuple(item.id for item in items[:3])
    assert not result.complete[items[0].id]


def test_visible_version_overlap_is_not_published_as_one_provision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = uuid4()
    items = [row(source, 0, ["Law", "MADDE 3"]), row(source, 0, ["Law", "MADDE 3"])]
    session, arguments = boundary(monkeypatch, items)
    with pytest.raises(CorpusScopeUnavailable, match="overlapping"):
        read(arguments, [items[0]])
    session.scalars.assert_not_called()


def test_text_budget_keeps_original_center_and_explicit_continuation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = uuid4()
    items = [
        row(source, n, ["Law", "MADDE 3", f"({n + 1})"], "rule" * 10) for n in range(3)
    ]
    _, arguments = boundary(monkeypatch, items)
    result = read(arguments, [items[2]], max_chars=80)
    assert result.chunks[0].id == items[2].id
    assert all(chunk.text == "rule" * 10 for chunk in result.chunks)
    assert result.continuation[items[2].id] == (items[1].id,)
    assert not result.complete[items[2].id]


def test_outline_budget_falls_back_to_exact_far_center_without_false_completeness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = uuid4()
    items = [row(source, n, ["Law", f"MADDE {n + 1}"]) for n in range(10)]
    _, arguments = boundary(monkeypatch, items)
    result = read(arguments, [items[-1]], max_outline_rows=2)
    assert result.outline_rows == 2 and result.outline_truncated
    assert [chunk.id for chunk in result.chunks] == [items[-1].id]
    assert not result.complete[items[-1].id]
    assert all(stream.closed for stream in arguments["_streams"])


@pytest.mark.parametrize(
    "as_of,expected", [(date(2025, 12, 31), "old"), (date(2026, 1, 1), "new")]
)
def test_qualified_closure_uses_same_snapshot_date_and_frozen_original_reader(
    monkeypatch: pytest.MonkeyPatch,
    as_of: date,
    expected: str,
) -> None:
    source = uuid4()
    old = row(source, 0, ["Law", "MADDE 3"])
    _, arguments = boundary(monkeypatch, [old], as_of=as_of)
    index = cast(PublicationIndexSnapshot, object())
    arguments["index"] = index
    monkeypatch.setattr(
        asv3_corpus, "qualified_file_ids", lambda *_args: frozenset([source])
    )
    calls: list[tuple[str, ...] | None] = []

    def bindings(
        _session: Session, identifier: UUID, **kwargs: Any
    ) -> Iterator[AnnexTemporalProjection]:
        assert identifier == source and kwargs["index"] is index
        assert kwargs["as_of_date"] == as_of
        calls.append(kwargs.get("canonical_chunk_ids"))
        if kwargs.get("canonical_chunk_ids") not in (None, (old.id,)):
            return
        yield cast(
            AnnexTemporalProjection,
            SimpleNamespace(
                derived_role="canonical",
                semantic_position=0,
                projection=SimpleNamespace(
                    ordinal=77,
                    source_json=json.dumps(
                        {
                            "regulatory_chunk_id": old.id,
                            "heading_path": ["Law", "MADDE 3"],
                        }
                    ),
                ),
                representation_text=expected + " frozen original",
                representation_metadata={},
                effective_start=date(2025, 1, 1)
                if expected == "old"
                else date(2026, 1, 1),
                effective_end=date(2026, 1, 1) if expected == "old" else None,
            ),
        )

    monkeypatch.setattr(asv3_corpus, "iter_public_temporal_bindings", bindings)
    result = read(arguments, [old])
    assert calls == [None, (old.id,)]
    assert result.chunks[0].text == expected + " frozen original"
    assert result.chunks[0].projection_ordinal == 77
    assert result.chunks[0].metadata["read_as_of_date"] == as_of.isoformat()
    assert result.complete[old.id]


def test_binding_validation_failure_or_cancel_never_becomes_citable_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = uuid4()
    item = row(source, 0, ["MADDE 3"])
    _, arguments = boundary(monkeypatch, [item])

    def cancelled() -> None:
        raise RunStopped("cancelled")

    arguments["check_active"] = cancelled
    with pytest.raises(RunStopped):
        read(arguments, [item])
    arguments["check_active"] = lambda: None
    arguments["index"] = cast(PublicationIndexSnapshot, object())
    monkeypatch.setattr(
        asv3_corpus, "qualified_file_ids", lambda *_args: frozenset([source])
    )

    def bad_binding(*_args: Any, **_kwargs: Any) -> Iterator[AnnexTemporalProjection]:
        raise ValueError("binding digest changed")
        yield

    monkeypatch.setattr(asv3_corpus, "iter_public_temporal_bindings", bad_binding)
    with pytest.raises(ValueError, match="digest changed"):
        read(arguments, [item])


def test_broker_batches_centers_by_source_and_graph_contains_only_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = uuid4()
    items = [row(source, n, ["Law", "MADDE 3"]) for n in range(2)]
    session, arguments = boundary(monkeypatch, items)
    broker = CorpusBroker(arguments["user"], arguments["filters"])
    monkeypatch.setattr(
        corpus_tools, "get_session_with_current_tenant", lambda: nullcontext(session)
    )
    monkeypatch.setattr(corpus_tools, "require_source", asv3_corpus.require_source)
    monkeypatch.setattr(corpus_tools, "resolve_source_query_index", lambda *_args: None)
    spans = []

    def span(_operation: str, _input: dict[str, Any], **_kwargs: Any) -> Any:
        step = SimpleNamespace(output_value=None)
        spans.append(step)
        return nullcontext(step)

    monkeypatch.setattr(corpus_tools, "graph_step", span)
    docs = [
        SearchDoc(
            document_id=str(source),
            chunk_ind=n,
            semantic_identifier="Source",
            link=None,
            blurb="",
            source_type=DocumentSource.USER_FILE,
            boost=0,
            hidden=False,
            metadata={"regulatory_chunk_id": item.id},
            match_highlights=[],
        )
        for n, item in enumerate(items)
    ]
    result = broker.hydrate_search_results(docs, RunContext())
    assert session.execute.call_count == 1 and len(spans) == 1
    assert all(len(group) == 2 for group in result.values())
    assert spans[0].output_value["hydrated_chunk_count"] == 2
    assert "original legal text" not in repr(spans[0].output_value)
    assert all(
        any(e.metadata["retrieved_center"] for e in group) for group in result.values()
    )


def retained(item: RegulatoryChunk, as_of: date | None = None) -> Any:
    source = CorpusSource(item.user_file_id, "Retained source", "original-file")
    return evidence_for_chunk(
        source,
        CorpusChunk(
            item.id,
            item.user_file_id,
            item.text,
            item.position,
            item.projection_ordinal,
            tuple(item.heading_path),
            {"read_as_of_date": as_of.isoformat() if as_of else None},
            item.validity_start_date,
            item.validity_end_date,
            item.status,
        ),
    )


def validation_broker(
    monkeypatch: pytest.MonkeyPatch, session: MagicMock, arguments: dict[str, Any]
) -> CorpusBroker:
    monkeypatch.setattr(
        corpus_tools, "get_session_with_current_tenant", lambda: nullcontext(session)
    )
    monkeypatch.setattr(corpus_tools, "require_source", asv3_corpus.require_source)
    monkeypatch.setattr(corpus_tools, "resolve_source_query_index", lambda *_args: None)
    return CorpusBroker(arguments["user"], arguments["filters"])


@pytest.mark.parametrize("operation", ["hydrate", "revalidate"])
def test_independent_original_reads_overlap_with_separate_sessions_and_captured_scope(
    monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    items = [row(uuid4(), 0, ["Law", "MADDE 3"], text) for text in ("first", "second")]
    user = cast(User, object())
    filters = IndexFilters(access_control_list=["user:authorized"])
    broker = CorpusBroker(user, filters)
    scope: ContextVar[str] = ContextVar("original_read_test_scope", default="missing")
    token = scope.set("captured-tenant-and-publication")
    barrier = Barrier(2)
    sessions: dict[UUID, Session] = {}
    sources = {
        item.user_file_id: CorpusSource(item.user_file_id, "Legal source", "file")
        for item in items
    }
    monkeypatch.setattr(
        corpus_tools,
        "get_session_with_current_tenant",
        lambda: nullcontext(MagicMock(spec=Session)),
    )

    def source_for(_session: Session, **kwargs: Any) -> CorpusSource:
        assert scope.get() == "captured-tenant-and-publication"
        assert kwargs["user"] is user
        assert kwargs["filters"].access_control_list == ["user:authorized"]
        return sources[kwargs["source_id"]]

    def originals(session: Session, **kwargs: Any) -> Iterator[CorpusChunk]:
        assert scope.get() == "captured-tenant-and-publication"
        assert kwargs["user"] is user and kwargs["index"] is None
        source_id = kwargs["source_id"]
        sessions[source_id] = session
        barrier.wait(timeout=3)  # A serialized read cannot satisfy this barrier.
        kwargs["check_active"]()
        item = next(item for item in items if item.user_file_id == source_id)
        assert kwargs["chunk_ids"] == (item.id,)
        yield CorpusChunk(
            item.id,
            source_id,
            item.text,
            0,
            0,
            tuple(item.heading_path),
            {},
            None,
            None,
            "active",
        )

    monkeypatch.setattr(corpus_tools, "require_source", source_for)
    monkeypatch.setattr(corpus_tools, "resolve_source_query_index", lambda *_: None)
    monkeypatch.setattr(corpus_tools, "iter_source_chunks_by_ids", originals)
    try:
        if operation == "hydrate":
            docs = [
                SearchDoc(
                    document_id=str(item.user_file_id),
                    chunk_ind=0,
                    semantic_identifier="Legal source",
                    link=None,
                    blurb="projection",
                    source_type=DocumentSource.USER_FILE,
                    boost=0,
                    hidden=False,
                    metadata={"regulatory_chunk_id": item.id},
                    match_highlights=[],
                )
                for item in items
            ]
            result = broker.hydrate_search_centers(docs, RunContext())
            assert list(result) == [(str(item.user_file_id), 0) for item in items]
            assert [group[0].text for group in result.values()] == ["first", "second"]
            assert all(
                group[0].text_hash == retained(item).text_hash
                for group, item in zip(result.values(), items, strict=True)
            )
        else:
            broker.revalidate_evidence([retained(item) for item in items], RunContext())
        assert (
            len(sessions) == 2
            and len({id(session) for session in sessions.values()}) == 2
        )
    finally:
        scope.reset(token)


def test_revalidation_reads_fifty_exact_ids_once_without_pages_or_outline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = uuid4()
    items = [row(source, n, [f"MADDE {n + 1}"]) for n in range(100)]
    session, arguments = boundary(monkeypatch, items)
    broker = validation_broker(monkeypatch, session, arguments)
    selected = items[::2]
    page = MagicMock(side_effect=AssertionError("whole page is forbidden"))
    monkeypatch.setattr(broker, "page", page)
    broker.revalidate_evidence([retained(item) for item in selected], RunContext())
    session.execute.assert_not_called()
    assert session.scalars.call_count == 1
    parameters = session.scalars.call_args.args[0].compile().params
    identifiers = next(
        value for value in parameters.values() if isinstance(value, (tuple, list))
    )
    assert set(identifiers) == {item.id for item in selected}
    assert all(stream.closed for stream in arguments["_streams"])


@pytest.mark.parametrize("defect", ["text", "hash", "ordinal", "missing"])
def test_batch_revalidation_rejects_changed_or_missing_originals(
    monkeypatch: pytest.MonkeyPatch,
    defect: str,
) -> None:
    source = uuid4()
    item = row(source, 0, ["MADDE 3"])
    session, arguments = boundary(monkeypatch, [item])
    evidence = retained(item)
    broker = validation_broker(monkeypatch, session, arguments)
    if defect == "text":
        item.text += " changed"
    elif defect == "hash":
        evidence.text_hash = "0" * 64
    elif defect == "ordinal":
        item.projection_ordinal = 999
    else:
        item.status = "superseded"
    with pytest.raises(CorpusScopeUnavailable):
        broker.revalidate_evidence([evidence], RunContext())
    assert all(stream.closed for stream in arguments["_streams"])


def test_batch_revalidation_keeps_captured_dates_separate_and_rejects_pinned_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = uuid4()
    items = [row(source, n, [f"MADDE {n + 1}"]) for n in range(2)]
    session, arguments = boundary(monkeypatch, items)
    broker = validation_broker(monkeypatch, session, arguments)
    broker.revalidate_evidence(
        [retained(items[0], date(2025, 1, 1)), retained(items[1], date(2026, 1, 1))],
        RunContext(),
    )
    assert session.scalars.call_count == 2
    observed = [
        call.args[0].compile().params for call in session.scalars.call_args_list
    ]
    assert any(date(2025, 1, 1) in parameters.values() for parameters in observed)
    assert any(date(2026, 1, 1) in parameters.values() for parameters in observed)
    broker.filters.as_of_date = date(2026, 1, 1)
    with pytest.raises(PermissionError, match="retained evidence date"):
        broker.revalidate_evidence([retained(items[0], date(2025, 1, 1))], RunContext())


def test_qualified_revalidation_requests_only_exact_ids_from_validated_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = uuid4()
    item = row(source, 0, ["MADDE 3"])
    session, arguments = boundary(monkeypatch, [item], as_of=date(2025, 12, 31))
    broker = validation_broker(monkeypatch, session, arguments)
    index = cast(PublicationIndexSnapshot, object())
    broker.query_indexes[source] = index
    monkeypatch.setattr(
        asv3_corpus, "qualified_file_ids", lambda *_args: frozenset([source])
    )
    calls = []

    def bindings(
        _session: Session, identifier: UUID, **kwargs: Any
    ) -> Iterator[AnnexTemporalProjection]:
        calls.append(kwargs)
        assert identifier == source and kwargs["index"] is index
        assert kwargs["as_of_date"] == date(2025, 12, 31)
        assert kwargs["canonical_chunk_ids"] == (item.id,)
        yield cast(
            AnnexTemporalProjection,
            SimpleNamespace(
                derived_role="canonical",
                semantic_position=0,
                projection=SimpleNamespace(
                    ordinal=0,
                    source_json=json.dumps(
                        {"regulatory_chunk_id": item.id, "heading_path": ["MADDE 3"]}
                    ),
                ),
                representation_text=item.text,
                representation_metadata={},
                effective_start=None,
                effective_end=None,
            ),
        )

    monkeypatch.setattr(asv3_corpus, "iter_public_temporal_bindings", bindings)
    broker.revalidate_evidence([retained(item, date(2025, 12, 31))], RunContext())
    assert len(calls) == 1
    session.execute.assert_not_called()
    session.scalars.assert_not_called()
