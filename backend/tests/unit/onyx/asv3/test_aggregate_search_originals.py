"""Search aggregates resolve to direct originals, never to derived evidence."""

import json
from contextlib import nullcontext
from datetime import date
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock
from uuid import UUID

import pytest

from onyx.asv3 import corpus_tools
from onyx.asv3.corpus_tools import CorpusBroker
from onyx.asv3.models import EvidenceItem, RunContext
from onyx.asv3.workflow_variant import ASV3_TUNED_VARIANT
from onyx.configs.constants import DocumentSource
from onyx.context.search.models import IndexFilters, SearchDoc
from onyx.db import asv3_corpus, supersearch
from onyx.db.asv3_corpus import CorpusChunk, CorpusScopeUnavailable, CorpusSource
from onyx.db.models import User
from tests.unit.onyx.asv3.test_supersearch_scope import (
    aggregate_session,
    temporal_center,
)


@pytest.fixture
def captured() -> dict[str, Any]:
    return json.loads(
        (Path(__file__).parent / "fixtures/aggregate_centers_5238.json").read_text()
    )


def originals(captured: dict[str, Any]) -> list[CorpusChunk]:
    return [
        CorpusChunk(
            row["id"],
            UUID(captured["source_id"]),
            row["text"],
            row["position"],
            row["projection_ordinal"],
            tuple(row["heading_path"]),
            {},
            None,
            None,
            "active",
        )
        for row in captured["originals"]
    ]


def documents(captured: dict[str, Any]) -> list[SearchDoc]:
    return [
        SearchDoc(
            document_id=captured["source_id"],
            chunk_ind=row["ordinal"],
            semantic_identifier="Ortak Transit",
            link=None,
            blurb="derived navigation",
            source_type=DocumentSource.USER_FILE,
            boost=0,
            hidden=False,
            metadata={
                "regulatory_chunk_id": row["id"],
                "source_regulatory_chunk_ids": ["forged"],
            },
            match_highlights=[],
        )
        for row in captured["centers"]
    ]


def setup_read(
    monkeypatch: pytest.MonkeyPatch, captured: dict[str, Any]
) -> tuple[MagicMock, dict[str, Any], list[CorpusChunk]]:
    session = aggregate_session(monkeypatch)
    source_id = UUID(captured["source_id"])
    source = CorpusSource(source_id, "Ortak Transit", "file")
    snapshot = MagicMock()
    snapshot.matches_temporal_index.side_effect = lambda other: other is snapshot
    monkeypatch.setattr(supersearch, "qualified_file_ids", lambda *_: {source_id})
    monkeypatch.setattr(supersearch, "resolve_source_query_index", lambda *_: snapshot)
    parents = []
    for row in captured["centers"]:
        binding = temporal_center(
            row["id"],
            chunk_type="hierarchical_aggregate",
            role="hierarchical_aggregate",
            members=row["members"],
        )
        binding.projection.source_json = json.dumps({"regulatory_chunk_id": row["id"]})
        binding.representation_metadata = {
            "chunk_variant": "hierarchical_aggregate",
            "source_regulatory_chunk_ids": row["members"],
        }
        binding.projection.ordinal = row["ordinal"]
        parents.append(binding)
    children = [temporal_center(row["id"]) for row in captured["originals"]]
    monkeypatch.setattr(
        supersearch,
        "iter_public_temporal_bindings",
        MagicMock(side_effect=[(row for row in parents), (row for row in children)]),
    )
    monkeypatch.setattr(asv3_corpus, "require_source", MagicMock(return_value=source))
    monkeypatch.setattr(asv3_corpus, "resolve_source_query_index", lambda *_: snapshot)
    monkeypatch.setattr(asv3_corpus, "observe_publication_read", MagicMock)
    monkeypatch.setattr(asv3_corpus, "require_publication_files", MagicMock())
    monkeypatch.setattr(
        asv3_corpus, "get_session_with_current_tenant", lambda: nullcontext(MagicMock())
    )
    rows = originals(captured)
    monkeypatch.setattr(
        asv3_corpus,
        "iter_source_chunks_by_ids",
        MagicMock(side_effect=lambda *_args, **_kwargs: (row for row in rows)),
    )
    return (
        session,
        {
            "user": cast(User, object()),
            "filters": IndexFilters(
                access_control_list=["allowed"],
                as_of_date=date.fromisoformat(captured["as_of_date"]),
            ),
            "source_id": source_id,
            "center_ordinals": {r["id"]: r["ordinal"] for r in captured["centers"]},
            "index": snapshot,
            "check_active": lambda: None,
        },
        rows,
    )


def test_three_captured_aggregate_centers_deliver_all_nine_exact_originals(
    monkeypatch: pytest.MonkeyPatch, captured: dict[str, Any]
) -> None:
    session, args, rows = setup_read(monkeypatch, captured)
    source, resolved = asv3_corpus.read_aggregate_search_originals(session, **args)
    by_id = {row.id: row for row in rows}
    assert source.id == args["source_id"]
    assert resolved == {
        center["id"]: tuple(by_id[member] for member in center["members"])
        for center in captured["centers"]
    }
    assert len({row.id for group in resolved.values() for row in group}) == 9
    reader = cast(MagicMock, asv3_corpus.iter_source_chunks_by_ids)
    assert set(reader.call_args.kwargs["chunk_ids"]) == set(by_id)
    assert reader.call_args.kwargs["index"] is args["index"]
    assert reader.call_args.kwargs["filters"] is args["filters"]
    assert reader.call_args.kwargs["batch_publication_checks"] is True
    assert not set(args["center_ordinals"]) & set(by_id)


@pytest.mark.parametrize(
    "defect",
    [
        "missing",
        "duplicate",
        "outside",
        "revoked",
        "changed_source",
        "changed_index",
        "publication",
        "cancelled",
    ],
)
def test_no_partial_originals_escape_final_fences(
    monkeypatch: pytest.MonkeyPatch, captured: dict[str, Any], defect: str
) -> None:
    session, args, rows = setup_read(monkeypatch, captured)
    if defect == "missing":
        rows.pop()
    elif defect == "duplicate":
        rows.append(rows[0])
    elif defect == "outside":
        rows[0] = CorpusChunk(
            rows[0].id, UUID(int=7), rows[0].text, 0, 0, (), {}, None, None, "active"
        )
    elif defect == "revoked":
        cast(MagicMock, asv3_corpus.require_source).side_effect = [
            CorpusSource(args["source_id"], "Ortak Transit", "file"),
            PermissionError("revoked"),
        ]
    elif defect == "changed_source":
        cast(MagicMock, asv3_corpus.require_source).side_effect = [
            CorpusSource(args["source_id"], "Ortak Transit", "file"),
            CorpusSource(args["source_id"], "changed", "other"),
        ]
    elif defect == "changed_index":
        monkeypatch.setattr(asv3_corpus, "resolve_source_query_index", lambda *_: None)
    elif defect == "publication":
        cast(
            MagicMock, asv3_corpus.require_publication_files
        ).side_effect = PermissionError("publication")
    else:

        def cancelled() -> None:
            raise RuntimeError("cancelled")

        args["check_active"] = cancelled
    with pytest.raises((CorpusScopeUnavailable, PermissionError, RuntimeError)):
        asv3_corpus.read_aggregate_search_originals(session, **args)


@pytest.mark.parametrize("defect", ["index", "ordinal"])
def test_search_parent_must_match_captured_locator(
    monkeypatch: pytest.MonkeyPatch, captured: dict[str, Any], defect: str
) -> None:
    session, args, _ = setup_read(monkeypatch, captured)
    if defect == "index":
        args["index"].matches_temporal_index.return_value = False
        args["index"].matches_temporal_index.side_effect = None
    else:
        args["center_ordinals"][captured["centers"][0]["id"]] += 1
    with pytest.raises(CorpusScopeUnavailable, match="changed before resolution"):
        asv3_corpus.read_aggregate_search_originals(session, **args)
    cast(MagicMock, asv3_corpus.iter_source_chunks_by_ids).assert_not_called()


@pytest.mark.parametrize(
    "services,enabled",
    [
        ({}, False),
        ({"asv3_workflow_variant": "asv3_standard"}, False),
        ({"asv3_workflow_variant": ASV3_TUNED_VARIANT}, True),
        ({"asv3_exp_cropped": True}, True),
        ({"asv3_workflow_variant": ASV3_TUNED_VARIANT, "asv3_exp_cheap": True}, False),
    ],
)
def test_broker_scope_and_exact_atomic_path_stay_unchanged(
    monkeypatch: pytest.MonkeyPatch,
    captured: dict[str, Any],
    services: dict[str, Any],
    enabled: bool,
) -> None:
    broker = CorpusBroker(cast(User, object()), IndexFilters(access_control_list=[]))
    context = RunContext(services=services)
    docs = documents(captured)
    resolved = {
        r["id"]: tuple(row for row in originals(captured) if row.id in r["members"])
        for r in captured["centers"]
    }
    reader = MagicMock(
        return_value=(
            CorpusSource(UUID(captured["source_id"]), "Ortak Transit", "file"),
            resolved,
        )
    )
    monkeypatch.setattr(corpus_tools, "read_aggregate_search_originals", reader)
    monkeypatch.setattr(
        corpus_tools,
        "get_session_with_current_tenant",
        lambda: nullcontext(MagicMock()),
    )
    empty = {(doc.document_id, doc.chunk_ind): [] for doc in docs}
    monkeypatch.setattr(
        broker, "_hydrate_exact_search_centers", lambda *_: empty.copy()
    )
    actual = broker.hydrate_search_centers(docs, context)
    assert reader.call_count == int(enabled)
    if enabled:
        assert sum(map(len, actual.values())) == 9
        assert all(
            item.chunk_id != item.metadata["retrieved_aggregate_id"]
            for items in actual.values()
            for item in items
        )
        assert all(
            item.metadata["article_closure_complete"] is False
            for items in actual.values()
            for item in items
        )
    else:
        assert actual == empty
    reader.reset_mock()
    atomic = EvidenceItem(
        source_id=captured["source_id"], chunk_id="atomic", text="original"
    )
    known = {(doc.document_id, doc.chunk_ind): [atomic] for doc in docs}
    monkeypatch.setattr(broker, "_hydrate_exact_search_centers", lambda *_: known)
    assert broker.hydrate_search_centers(docs, context) is known
    reader.assert_not_called()


def test_conflicting_duplicate_center_ordinals_fail_before_resolution(
    monkeypatch: pytest.MonkeyPatch, captured: dict[str, Any]
) -> None:
    broker = CorpusBroker(cast(User, object()), IndexFilters(access_control_list=[]))
    docs = documents(captured)
    docs.append(docs[0].model_copy(update={"chunk_ind": 99999}))
    monkeypatch.setattr(broker, "_hydrate_exact_search_centers", lambda *_: {})
    resolver = MagicMock()
    monkeypatch.setattr(corpus_tools, "read_aggregate_search_originals", resolver)
    with pytest.raises(CorpusScopeUnavailable, match="conflicting projection"):
        broker.hydrate_search_centers(
            docs, RunContext(services={"asv3_workflow_variant": ASV3_TUNED_VARIANT})
        )
    resolver.assert_not_called()


def test_existing_supersearch_default_keeps_derived_role_only_behavior(
    monkeypatch: pytest.MonkeyPatch, captured: dict[str, Any]
) -> None:
    session, args, _ = setup_read(monkeypatch, captured)
    result = supersearch.resolve_supersearch_center_ids(
        session,
        user=args["user"],
        filters=args["filters"],
        source_id=args["source_id"],
        center_ids=tuple(args["center_ordinals"]),
        check_active=args["check_active"],
    )
    assert result == dict.fromkeys(args["center_ordinals"], ())
