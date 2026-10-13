"""Numbered title batching retains canonical pages and fresh source fences."""

import json
from contextlib import nullcontext
from dataclasses import replace
from datetime import date
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock
from uuid import UUID

import pytest
from sqlalchemy.orm import Session

from onyx.asv3 import corpus_tools
from onyx.asv3.models import RunContext
from onyx.context.search.models import IndexFilters
from onyx.db import asv3_corpus as corpus
from onyx.db.asv3_corpus import CorpusChunk
from onyx.db.models import User
from tests.unit.onyx.asv3.test_numbered_source_resolution import chunk, filters, source


def setup_batch(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    sources = [source() for _ in range(3)]
    originals = {item.id: [chunk(item)] for item in sources}
    originals[sources[-1].id] = []
    sessions: list[Session] = []
    reads: list[UUID] = []

    def session() -> Any:
        current = MagicMock(spec=Session)
        sessions.append(current)
        return nullcontext(current)

    def page(_session: Session, **kwargs: Any) -> tuple[list[CorpusChunk], bool]:
        assert kwargs["start"] == 0 and kwargs["limit"] == 2
        reads.append(kwargs["source_id"])
        return originals[kwargs["source_id"]], False

    auth = MagicMock(return_value={item.id: item for item in sources})
    publication = MagicMock()
    monkeypatch.setattr(corpus, "get_session_with_current_tenant", session)
    monkeypatch.setattr(corpus, "_require_exact_sources", auth)
    monkeypatch.setattr(corpus, "_read_source_chunk_page", page)
    monkeypatch.setattr(corpus, "resolve_source_query_index", lambda *_: None)
    monkeypatch.setattr(corpus, "observe_publication_read", lambda: "fixed")
    monkeypatch.setattr(corpus, "require_publication_files", publication)
    return dict(
        sources=sources,
        originals=originals,
        sessions=sessions,
        reads=reads,
        auth=auth,
        publication=publication,
        kwargs=dict(
            user=cast(User, object()),
            filters=filters(),
            sources=sources,
            check_active=lambda: None,
        ),
    )


def test_batch_preserves_pages_empty_sources_order_and_fresh_fences(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = setup_batch(monkeypatch)
    result = corpus._read_identity_heads_batch(**state["kwargs"])
    assert result == state["originals"]
    assert list(result) == [item.id for item in state["sources"]]
    assert state["auth"].call_count == 2
    assert (
        state["auth"].call_args_list[0].args[0]
        is not state["auth"].call_args_list[1].args[0]
    )
    assert (
        state["auth"].call_args_list[0].kwargs["filters"] is state["kwargs"]["filters"]
    )
    assert (
        state["publication"].call_args_list[0].args
        == state["publication"].call_args_list[1].args
    )
    assert state["publication"].call_args.args == ("fixed", tuple(result))


@pytest.mark.parametrize(
    "defect",
    [
        "entry_acl",
        "exit_acl",
        "entry_identity",
        "exit_identity",
        "entry_publication",
        "exit_publication",
        "read",
        "index",
        "cancel",
        "duplicate",
    ],
)
def test_no_partial_identity_page_escapes_failed_fences(
    monkeypatch: pytest.MonkeyPatch,
    defect: str,
) -> None:
    state = setup_batch(monkeypatch)
    if defect.endswith("acl"):
        state["auth"].side_effect = (
            [PermissionError("revoked")]
            if defect.startswith("entry")
            else [state["auth"].return_value, PermissionError("revoked")]
        )
    elif defect.endswith("identity"):
        changed = dict(state["auth"].return_value)
        key = next(iter(changed))
        changed[key] = replace(changed[key], file_id="replaced")
        state["auth"].side_effect = (
            [changed]
            if defect.startswith("entry")
            else [state["auth"].return_value, changed]
        )
    elif defect.endswith("publication"):
        state["publication"].side_effect = (
            [PermissionError("publishing")]
            if defect.startswith("entry")
            else [None, PermissionError("publishing")]
        )
    elif defect in ("read", "index"):
        monkeypatch.setattr(
            corpus,
            "_read_source_chunk_page"
            if defect == "read"
            else "resolve_source_query_index",
            MagicMock(side_effect=ValueError("invalid binding")),
        )
    elif defect == "cancel":

        def active() -> None:
            if state["auth"].call_count == 2:
                raise RuntimeError("cancelled")

        state["kwargs"]["check_active"] = active
    else:
        state["sources"].append(state["sources"][0])
    delivered: dict[UUID, list[CorpusChunk]] = {}
    with pytest.raises((PermissionError, ValueError, RuntimeError)):
        delivered.update(corpus._read_identity_heads_batch(**state["kwargs"]))
    assert not delivered
    if defect.startswith("entry"):
        assert not state["reads"]


@pytest.mark.parametrize("qualified", [False, True])
def test_shared_page_reader_is_identical_and_keeps_third_binding_validation(
    monkeypatch: pytest.MonkeyPatch,
    qualified: bool,
) -> None:
    item = source()
    scope = filters()
    session = MagicMock(spec=Session)
    rows = [
        SimpleNamespace(
            id=f"chunk-{n}",
            user_file_id=item.id,
            text=f"exact-{n}",
            position=n,
            projection_ordinal=n + 10,
            heading_path=["Own law", f"MADDE {n}"],
            chunk_metadata={"version": n},
            validity_start_date=date(2020, 1, 1),
            validity_end_date=None,
            status="active",
        )
        for n in range(3)
    ]
    session.scalars.return_value = rows
    bindings = [
        SimpleNamespace(
            semantic_position=row.position,
            projection=SimpleNamespace(
                source_json=json.dumps(
                    {"regulatory_chunk_id": row.id, "heading_path": row.heading_path}
                ),
                ordinal=row.projection_ordinal,
            ),
            representation_text=row.text,
            representation_metadata=row.chunk_metadata,
            effective_start=row.validity_start_date,
            effective_end=None,
        )
        for row in rows
    ]
    bounded = MagicMock(return_value=bindings)
    monkeypatch.setattr(corpus, "_bounded_temporal_bindings", bounded)
    monkeypatch.setattr(
        corpus, "qualified_file_ids", lambda *_: {item.id} if qualified else set()
    )
    monkeypatch.setattr(corpus, "require_source", lambda *_, **__: item)
    monkeypatch.setattr(corpus, "observe_publication_read", lambda: "fixed")
    monkeypatch.setattr(corpus, "filter_publication_read", lambda _, values, __: values)
    monkeypatch.setattr(corpus, "require_publication_files", MagicMock())
    index = MagicMock()
    kwargs: dict[str, Any] = dict(
        filters=scope,
        source_id=item.id,
        start=0,
        limit=2,
        query_indexes={item.id: index} if qualified else None,
    )
    _, old, more = corpus.read_source_chunks(
        session, user=cast(User, object()), **kwargs
    )
    new, new_more = corpus._read_source_chunk_page(session, **kwargs)
    assert old == new and more is new_more is True
    assert [row.id for row in new] == ["chunk-0", "chunk-1"]
    assert scope.as_of_date is not None
    assert new[0].metadata["read_as_of_date"] == scope.as_of_date.isoformat()
    assert new[1].text == "exact-1" and new[1].projection_ordinal == 11
    if qualified:
        assert all(call.args[-1] == 3 for call in bounded.call_args_list)
        bounded.side_effect = corpus.CorpusScopeUnavailable("third binding invalid")
        with pytest.raises(corpus.CorpusScopeUnavailable, match="third binding"):
            corpus._read_source_chunk_page(session, **kwargs)
    else:
        assert session.scalars.call_args.args[0].compile().params["param_1"] == 3


@pytest.mark.parametrize("cropped", [None, False, True, "true"])
def test_broker_only_literal_cropped_threads_cancellation(
    monkeypatch: pytest.MonkeyPatch,
    cropped: object,
) -> None:
    monkeypatch.setattr(
        corpus_tools,
        "get_session_with_current_tenant",
        lambda: nullcontext(MagicMock()),
    )
    find = MagicMock(return_value=([], False))
    monkeypatch.setattr(corpus_tools, "find_sources", find)
    context = RunContext(services={"asv3_exp_cropped": cropped})
    broker = corpus_tools.CorpusBroker(
        cast(User, object()), IndexFilters(access_control_list=[])
    )
    broker.sources("4458 sayılı Gümrük Kanunu", context)
    assert find.call_args.kwargs["allow_canonical_filename_fallback"] is (
        cropped is True
    )
    assert (find.call_args.kwargs["check_active"] is not None) is (cropped is True)


@pytest.mark.parametrize("cropped", [False, True])
def test_numbered_fallback_preserves_candidates_and_pagination(
    monkeypatch: pytest.MonkeyPatch,
    cropped: bool,
) -> None:
    candidates = [source(), source()]
    monkeypatch.setattr(
        corpus,
        "_find_sources_with_title_queries",
        MagicMock(side_effect=[([], False), (candidates, True)]),
    )
    heads = {item.id: [chunk(item)] for item in candidates}
    batch = MagicMock(return_value=heads)
    monkeypatch.setattr(corpus, "_read_identity_heads_batch", batch)
    legacy = MagicMock(
        side_effect=lambda _, **kwargs: (
            candidates[0],
            heads[kwargs["source_id"]],
            False,
        )
    )
    monkeypatch.setattr(corpus, "read_source_chunks", legacy)
    monkeypatch.setattr(corpus, "resolve_source_query_index", lambda *_: None)
    result = corpus.find_sources(
        MagicMock(),
        user=cast(User, object()),
        filters=filters(),
        query="8237 sayılı Faaliyet Kanunu",
        limit=2,
        allow_numbered_title_fallback=True,
        allow_canonical_filename_fallback=cropped,
    )
    assert result == (candidates, True)
    assert batch.call_count == int(cropped)
    assert legacy.call_count == (0 if cropped else 2)
