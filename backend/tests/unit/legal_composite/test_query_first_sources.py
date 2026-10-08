"""Query candidates retain scoped original proof without a corpus startup crawl."""

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from contextvars import copy_context
from datetime import date
from threading import Barrier, Event, Lock
from typing import cast
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest
from pydantic import JsonValue
from sqlalchemy.orm import Session

from onyx.asv3.corpus_tools import CorpusBroker, evidence_for_chunk
from onyx.asv3.models import OutcomeStatus, RunContext, ToolOutcome
from onyx.asv3.registry import CapabilityRegistry
from onyx.db import legal_composite_sources as dao
from onyx.db.asv3_corpus import CorpusScopeUnavailable, CorpusSource
from onyx.db.legal_composite_sources import (
    RoutingOpening,
    SourceClassification,
    SourceKind,
)
from onyx.legal_composite import source_lanes
from onyx.legal_composite.acquisition import InvalidSourceAction
from onyx.legal_composite.models import SourceAction
from onyx.legal_composite.routing import SourceLaneRouter
from onyx.legal_composite.source_lanes import (
    CandidateSourceClassifier,
    SourceLaneBroker,
)
from shared_configs.contextvars import CURRENT_TENANT_ID_CONTEXTVAR
from tests.unit.legal_composite.test_parallel_selection import plan
from tests.unit.legal_composite.test_source_lanes import fixture_lane
from tests.unit.onyx.legal_composite.test_authority_dependencies import (
    chunk,
    setup_expander,
)


def candidate(
    source: CorpusSource, heading: str = "GÜMRÜK KANUNU"
) -> SourceClassification:
    return dao.classify_source(source, (), opening_texts=(heading,)).model_copy(
        update={"routing_only": True}
    )


def test_candidate_loader_authorizes_only_exact_hits_before_and_after_original_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base, _catalogue, source = fixture_lane()
    denied = uuid4()
    inventory = MagicMock(return_value=([source], False))
    openings = MagicMock(
        return_value={
            source.id: RoutingOpening(
                texts=("CUMHURBAŞKANLIĞI KARARNAMESİ",), identity_available=True
            )
        }
    )
    monkeypatch.setattr(dao, "find_source_inventory_page", inventory)
    monkeypatch.setattr(dao, "_routing_opening_batch", openings)
    monkeypatch.setattr(
        dao,
        "_document_types",
        MagicMock(side_effect=AssertionError("Metadata cannot be a gate")),
    )
    records = dao.classify_candidate_sources(
        cast(Session, MagicMock()),
        user=base.user,
        filters=base.filters,
        source_ids=(source.id, denied),
        check_active=lambda: None,
    )
    assert set(records) == {source.id}
    assert records[source.id].kind == SourceKind.PRESIDENTIAL_DECREE
    assert records[source.id].routing_only and records[source.id].admits(
        SourceKind.UNKNOWN
    )
    assert openings.call_args.args[1] == (source.id,)
    assert inventory.call_count == 2
    for call in inventory.call_args_list:
        assert call.kwargs["source_ids"] == (source.id, denied)
        assert call.kwargs["filters"] is base.filters
        assert call.kwargs["filters"].as_of_date == date(2022, 1, 1)
        assert call.kwargs["filters"].asv3_document_set_id == 15


def test_withdrawn_candidate_does_not_discard_authorized_neighbor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base, _catalogue, source = fixture_lane()
    withdrawn = CorpusSource(uuid4(), "withdrawn.md", "file")
    monkeypatch.setattr(
        dao,
        "find_source_inventory_page",
        MagicMock(side_effect=[([source, withdrawn], False), ([source], False)]),
    )
    monkeypatch.setattr(
        dao,
        "_routing_opening_batch",
        lambda *_args: {
            source.id: RoutingOpening(texts=("GÜMRÜK KANUNU",), identity_available=True)
        },
    )
    records = dao.classify_candidate_sources(
        cast(Session, MagicMock()),
        user=base.user,
        filters=base.filters,
        source_ids=(source.id, withdrawn.id),
        check_active=lambda: None,
    )
    assert set(records) == {source.id}
    assert records[source.id].kind == SourceKind.STATUTE


def test_unavailable_opening_remains_authorized_unknown_and_no_full_proof_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base, _catalogue, source = fixture_lane()
    monkeypatch.setattr(
        dao, "find_source_inventory_page", lambda *_args, **_kwargs: ([source], False)
    )
    monkeypatch.setattr(
        dao,
        "_routing_opening_batch",
        MagicMock(side_effect=CorpusScopeUnavailable("Index unavailable")),
    )
    record = dao.classify_candidate_sources(
        cast(Session, MagicMock()),
        user=base.user,
        filters=base.filters,
        source_ids=(source.id,),
        check_active=lambda: None,
    )[source.id]
    assert record.kind == SourceKind.UNKNOWN and record.opening_witnesses == ()
    assert record.original_kind is None and record.uncertain and record.routing_only
    assert record.admits(SourceKind.STATUTE)


def test_candidate_singleflight_shares_batch_with_tenant_context_and_own_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base, _catalogue, source = fixture_lane()
    classifier = CandidateSourceClassifier(base, RunContext())
    entered, release = Event(), Event()
    calls: list[tuple[UUID, ...]] = []
    sessions: list[Session] = []

    @contextmanager
    def session() -> Iterator[Session]:
        assert CURRENT_TENANT_ID_CONTEXTVAR.get() == "candidate-tenant"
        value = cast(Session, MagicMock())
        sessions.append(value)
        yield value

    def load(
        _session: Session, *, source_ids: tuple[UUID, ...], **_kwargs: object
    ) -> dict[UUID, SourceClassification]:
        assert CURRENT_TENANT_ID_CONTEXTVAR.get() == "candidate-tenant"
        calls.append(source_ids)
        entered.set()
        assert release.wait(timeout=2)
        return {source.id: candidate(source)}

    monkeypatch.setattr(source_lanes, "get_session_with_current_tenant", session)
    monkeypatch.setattr(source_lanes, "classify_candidate_sources", load)
    token = CURRENT_TENANT_ID_CONTEXTVAR.set("candidate-tenant")
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(
                copy_context().run, classifier.classify_sources, (str(source.id),)
            )
            assert entered.wait(timeout=2)
            second = pool.submit(
                copy_context().run, classifier.classify_sources, (str(source.id),)
            )
            release.set()
            assert first.result(timeout=2) == second.result(timeout=2)
    finally:
        CURRENT_TENANT_ID_CONTEXTVAR.reset(token)
    assert calls == [(source.id,)] and len(sessions) == 1
    assert classifier.source_kinds == {str(source.id): SourceKind.STATUTE}
    assert classifier.provenance()["inventory_complete"] is False


def test_candidate_classification_batch_size_and_parallel_limit_are_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base, _catalogue, _source = fixture_lane()
    classifier = CandidateSourceClassifier(base, RunContext())
    calls: list[tuple[UUID, ...]] = []
    monkeypatch.setattr(
        source_lanes,
        "get_session_with_current_tenant",
        lambda: nullcontext(cast(Session, MagicMock())),
    )

    def load(
        _session: Session, *, source_ids: tuple[UUID, ...], **_kwargs: object
    ) -> dict[UUID, SourceClassification]:
        calls.append(source_ids)
        return {
            identifier: candidate(CorpusSource(identifier, "source", "file"))
            for identifier in source_ids
        }

    monkeypatch.setattr(source_lanes, "classify_candidate_sources", load)
    ids = tuple(str(uuid4()) for _ in range(201))
    assert len(classifier.classify_sources(ids)) == 201
    assert [len(batch) for batch in calls] == [100, 100, 1]
    assert classifier.classify_sources(ids) and len(calls) == 3
    classifier.filters.as_of_date = date(2024, 1, 1)
    with pytest.raises(PermissionError, match="scope changed"):
        classifier.classify_sources(ids)


def test_all_twelve_empty_lazy_lanes_exist_and_direct_ids_are_lazily_authorized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base, _catalogue, source = fixture_lane()
    classifier = CandidateSourceClassifier(base, RunContext())
    load = MagicMock(return_value={source.id: candidate(source)})
    monkeypatch.setattr(source_lanes, "classify_candidate_sources", load)
    monkeypatch.setattr(
        source_lanes,
        "get_session_with_current_tenant",
        lambda: nullcontext(cast(Session, MagicMock())),
    )
    router = SourceLaneRouter(classifier, lambda _kind: CapabilityRegistry([]))
    assert router.inventory()["inventory_complete"] is False
    discovery = router.expand(
        [
            __import__(
                "onyx.legal_composite.models", fromlist=["SourceAction"]
            ).SourceAction(need_ids=["n1"], tool="search_corpus", arguments={})
        ],
        plan(),
    )
    assert [action.source_kind for action in discovery] == list(SourceKind)
    for action in discovery:
        router.registry(action)
    load.assert_not_called()
    for kind in SourceKind:
        lane = SourceLaneBroker(base, classifier, kind)
        assert lane.filters == base.filters

    direct = router.expand(
        [
            SourceAction(
                need_ids=["n1"],
                tool="read_source_range",
                arguments={"source_id": str(source.id)},
            )
        ],
        plan(),
    )
    assert len(direct) == 1 and direct[0].source_kind == SourceKind.STATUTE
    load.assert_called_once()
    load.return_value = {}
    with pytest.raises(InvalidSourceAction, match="authorized candidate scope"):
        router.expand(
            [
                SourceAction(
                    need_ids=["n1"],
                    tool="read_source_range",
                    arguments={"source_id": str(uuid4())},
                )
            ],
            plan(),
        )


def test_lazy_read_revalidates_full_current_original_proof_and_rejects_wrong_kind(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base, _catalogue, source = fixture_lane()
    classifier = CandidateSourceClassifier(base, RunContext())
    monkeypatch.setattr(
        source_lanes,
        "get_session_with_current_tenant",
        lambda: nullcontext(cast(Session, MagicMock())),
    )
    monkeypatch.setattr(
        source_lanes,
        "classify_candidate_sources",
        lambda *_args, **_kwargs: {
            source.id: candidate(source, "CUMHURBAŞKANLIĞI KARARNAMESİ")
        },
    )
    verify = MagicMock(return_value=source)
    monkeypatch.setattr(source_lanes, "revalidate_source_classification", verify)
    lane = SourceLaneBroker(base, classifier, SourceKind.PRESIDENTIAL_DECREE)
    assert lane.source(str(source.id), RunContext()) == source
    assert lane.source(str(source.id), RunContext()) == source
    assert verify.call_count == 2
    assert verify.call_args.kwargs["filters"] == base.filters
    lane._check_original_kind(str(source.id), {"document_type": "yonetmelik"})
    verify.side_effect = CorpusScopeUnavailable("Encoder authority changed")
    with pytest.raises(CorpusScopeUnavailable, match="Encoder authority"):
        lane.source(str(source.id), RunContext())
    with pytest.raises(CorpusScopeUnavailable, match="Encoder authority"):
        SourceLaneBroker(base, classifier, SourceKind.STATUTE).source(
            str(source.id), RunContext()
        )


def test_concurrent_search_receipts_stay_with_their_call_and_partial_zero_hits() -> (
    None
):
    base, catalogue, _source = fixture_lane()
    lane = SourceLaneBroker(base, catalogue, SourceKind.STATUTE)
    barrier = Barrier(2)

    def adapter(args: dict[str, JsonValue], _context: RunContext) -> ToolOutcome:
        lane.record_search(
            {
                "query": args["query"],
                "incomplete": True,
                "candidate_window_saturated": True,
            }
        )
        barrier.wait(timeout=2)
        return ToolOutcome(status=OutcomeStatus.NOT_FOUND, summary="No classified hit")

    guarded = lane.guard_search_adapter(adapter)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(
                guarded, cast(dict[str, JsonValue], {"query": query}), RunContext()
            )
            for query in ("one", "two")
        ]
        outputs = [future.result(timeout=2) for future in futures]
    assert all(output.status == OutcomeStatus.PARTIAL for output in outputs)
    assert [output.data["candidate_search"] for output in outputs] == [
        [{"query": query, "incomplete": True, "candidate_window_saturated": True}]
        for query in ("one", "two")
    ]


def test_one_bad_full_original_does_not_discard_other_source_in_same_lane(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base, _catalogue, good = fixture_lane()
    bad = CorpusSource(uuid4(), "bad", "file")
    classifier = CandidateSourceClassifier(base, RunContext())
    monkeypatch.setattr(
        source_lanes,
        "classify_candidate_sources",
        lambda *_args, **_kwargs: {
            source.id: candidate(source) for source in (good, bad)
        },
    )
    monkeypatch.setattr(
        source_lanes,
        "get_session_with_current_tenant",
        lambda: nullcontext(cast(Session, MagicMock())),
    )
    lane = SourceLaneBroker(base, classifier, SourceKind.STATUTE)
    item = evidence_for_chunk(
        good, chunk(good, "Whole authorized original", 0, title="GÜMRÜK KANUNU")
    )
    assert item.search_doc is not None
    good_doc = item.search_doc
    bad_doc = good_doc.model_copy(deep=True, update={"document_id": str(bad.id)})

    def verify(_session: Session, **kwargs: object) -> CorpusSource:
        record = cast(SourceClassification, kwargs["recorded"])
        if record.source_id == bad.id:
            raise CorpusScopeUnavailable("Full binding is stale")
        return good

    monkeypatch.setattr(source_lanes, "revalidate_source_classification", verify)
    monkeypatch.setattr(
        CorpusBroker,
        "hydrate_search_centers",
        lambda _self, docs, _context: {
            (docs[0].document_id, docs[0].chunk_ind): [item]
        },
    )

    def adapter(_args: dict[str, JsonValue], context: RunContext) -> ToolOutcome:
        result = lane.hydrate_search_centers([good_doc, bad_doc], context)
        return ToolOutcome(
            status=OutcomeStatus.FOUND,
            summary="Originals",
            evidence=[
                original for originals in result.values() for original in originals
            ],
        )

    outcome = lane.guard_search_adapter(adapter)({}, RunContext())
    assert outcome.status == OutcomeStatus.PARTIAL
    assert (
        len(outcome.evidence) == 1
        and outcome.evidence[0].text == "Whole authorized original"
    )
    assert outcome.data["candidate_search"] == [
        {
            "incomplete": True,
            "unavailable_source_ids": [str(bad.id)],
            "full_original_proof_unavailable": True,
            "corpus_absence_verified": False,
        }
    ]


def test_dependency_anchor_sees_source_kind_discovered_after_construction() -> None:
    expander, _ledger, _calls, broker, _court, law, _expand = setup_expander()
    _source, law_chunk = broker.chunk.return_value
    expander.verified_kinds.clear()
    expander.source_kinds.clear()
    item = evidence_for_chunk(law, law_chunk)
    assert expander._anchor(item) is None
    expander.source_kinds[str(law.id)] = SourceKind.STATUTE
    assert expander._anchor(item) is not None
    expander.verified_kinds[str(law.id)] = SourceKind.JUDICIAL_DECISION
    assert expander._anchor(item) is None


def test_parallel_candidate_batches_never_share_sessions_or_exceed_four(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base, _catalogue, _source = fixture_lane()
    classifier = CandidateSourceClassifier(base, RunContext())
    four_entered, release = Event(), Event()
    lock = Lock()
    active = maximum = 0
    sessions: list[Session] = []

    @contextmanager
    def session() -> Iterator[Session]:
        value = cast(Session, MagicMock())
        with lock:
            sessions.append(value)
        yield value

    def load(
        _session: Session, *, source_ids: tuple[UUID, ...], **_kwargs: object
    ) -> dict[UUID, SourceClassification]:
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
            if active == 4:
                four_entered.set()
        assert release.wait(timeout=3)
        try:
            return {
                identifier: candidate(CorpusSource(identifier, "source", "file"))
                for identifier in source_ids
            }
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(source_lanes, "get_session_with_current_tenant", session)
    monkeypatch.setattr(source_lanes, "classify_candidate_sources", load)
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [
            pool.submit(classifier.classify_sources, (str(uuid4()),)) for _ in range(8)
        ]
        assert four_entered.wait(timeout=2)
        assert len(sessions) == maximum == 4
        release.set()
        assert all(future.result(timeout=3) for future in futures)
    assert (
        maximum == 4
        and len(sessions) == 8
        and len({id(value) for value in sessions}) == 8
    )


def test_lazy_inventory_counts_use_one_snapshot_during_parallel_updates() -> None:
    base, _catalogue, _source = fixture_lane()
    classifier = CandidateSourceClassifier(base, RunContext())
    other = candidate(
        CorpusSource(uuid4(), "court", "file"), "ANAYASA MAHKEMESİ KARARI"
    ).model_copy(update={"uncertain": True})
    classifier.records[other.source_id] = other
    lane = SourceLaneBroker(base, classifier, SourceKind.STATUTE)
    router = SourceLaneRouter(classifier, lambda _kind: CapabilityRegistry([]))
    start = Barrier(2)

    def update() -> None:
        start.wait(timeout=2)
        for _ in range(500):
            source = CorpusSource(uuid4(), "source", "file")
            with classifier._lock:
                classifier.records[source.id] = candidate(source)

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(update)
        start.wait(timeout=2)
        for _ in range(500):
            provenance = lane.lane_provenance()
            lane_count = provenance["lane_source_count"]
            assert isinstance(lane_count, int)
            assert lane_count == provenance["source_count"]
            assert provenance["lane_uncertain_source_count"] == 1
            inventory = router.inventory()
            kinds = inventory["kinds"]
            assert isinstance(kinds, dict)
            statute_count = kinds["statute"]
            assert isinstance(statute_count, int)
            assert statute_count == inventory["source_count"]
        future.result(timeout=2)
    assert lane.lane_provenance()["source_count"] == 501


@pytest.mark.parametrize("status", [OutcomeStatus.PARTIAL, OutcomeStatus.NOT_FOUND])
@pytest.mark.parametrize("original_unavailable", [False, True])
def test_search_window_limit_is_distinct_from_missing_original_evidence(
    status: OutcomeStatus,
    original_unavailable: bool,
) -> None:
    from onyx.asv3.models import ToolSpec
    from onyx.legal_composite.dependencies import assess_dependencies
    from onyx.legal_composite.models import DependencyWitness, DraftAnswer, SourceAction
    from tests.unit.onyx.legal_composite.test_authority_dependencies import (
        plan as authority_plan,
    )
    from tests.unit.onyx.legal_composite.test_authority_dependencies import review

    expander, ledger, _calls, broker, _court, _law, _expand = setup_expander()
    broker.related_catalog_sources.return_value = ([], False)
    registry = expander.acquirer.registry_for_action(
        SourceAction(need_ids=["permit"], tool="read_provision", arguments={})
    )
    governing = registry.get("read_provision")
    ranges = registry.get("read_source_range")
    assert governing is not None and ranges is not None
    limited = CapabilityRegistry(
        [
            governing,
            ranges,
            ToolSpec(
                name="search_corpus",
                description="bounded",
                parameters={"type": "object"},
                handler=lambda _args, _context: ToolOutcome(
                    status=status,
                    summary="Bounded window has no further classified result",
                    data={
                        "candidate_search": [
                            {
                                "incomplete": True,
                                "candidate_window_saturated": True,
                                "full_original_proof_unavailable": original_unavailable,
                            }
                        ],
                        "corpus_absence_verified": False,
                    },
                ),
            ),
        ]
    )
    limited.register(expander.acquirer.host_registry.get("resolve_source"))
    limited.register(expander.acquirer.host_registry.get("dependency_related_sources"))
    expander.acquirer.host_registry = limited
    expander.acquirer.registry_for_action = lambda _action: limited
    frozen_plan = authority_plan()
    edge = expander.expand(frozen_plan)[0]
    assert edge.governing_citations and not edge.candidate_citations
    assert bool(edge.discovery_gaps) is original_unavailable
    assert bool(edge.discovery_limits) is not original_unavailable
    citation = edge.governing_citations[0]
    original = ledger.get(citation)
    assert original is not None
    critic = review(
        edge,
        witnesses=[
            DependencyWitness(
                citation=citation, quotation=original.text, role="governing"
            )
        ],
    )
    draft = DraftAnswer(answer="The governing rule applies.", unresolved_need_ids=[])
    assert assess_dependencies(
        [edge], frozen_plan, draft, critic, ledger, set(ledger.citation_numbers())
    )[:2] == (not original_unavailable, not original_unavailable)
