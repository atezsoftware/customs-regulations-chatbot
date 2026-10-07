"""Protect canonical scoped reading and exact evidence while reducing input overhead."""

import copy
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from threading import Event
from typing import cast
from uuid import uuid4

import pytest
from pydantic import JsonValue

from onyx.asv3.corpus_tools import build_corpus_specs
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.focused_source_target import focused_source_target
from onyx.asv3.judicial_sections import nonoperative_judicial_witness_role
from onyx.asv3.llm_adapter import ResearchModel
from onyx.asv3.models import (
    CapabilityCall,
    EvidenceItem,
    OutcomeStatus,
    RunContext,
    SharedBudget,
    ToolOutcome,
)
from onyx.asv3.outcome_map import OutcomeMap
from onyx.asv3.registry import CapabilityRegistry
from onyx.asv3.shared_reads import SharedReads
from onyx.asv3.workflow_variant import ASV3_TUNED_VARIANT
from onyx.db.asv3_corpus import CorpusChunk, CorpusSource
from onyx.llm.models import ToolMessage
from tests.unit.onyx.asv3.test_corpus_source_sandbox import MemoryBroker
from tests.unit.onyx.asv3.test_experimental_workflow import terminal_registry
from tests.unit.onyx.asv3.test_legal_source_reviews import deliver, review, seen
from tests.unit.onyx.asv3.test_native_authority import original
from tests.unit.onyx.asv3.test_native_metadata_projection import setup_original
from tests.unit.onyx.asv3.test_native_model_adapter import (
    model,
    turn,
    view,
)
from tests.unit.onyx.asv3.test_runtime import delivered_originals
from tests.unit.onyx.asv3.test_tuned_source_followthrough import tuned_context


class NamedBroker(MemoryBroker):
    def __init__(self) -> None:
        source_id = uuid4()
        rows = [
            CorpusChunk(
                f"clause-{index}",
                source_id,
                text,
                index,
                index,
                (f"MADDE {article}",),
                {"article_no": article},
                None,
                None,
                "active",
            )
            for index, (article, text) in enumerate(
                [
                    ("7", "MADDE 7: Authorization AND proof are required."),
                    (
                        "7",
                        "Except for the identified special status; then use the alternative.",
                    ),
                    ("8", "MADDE 8: A separate provision."),
                ]
            )
        ]
        super().__init__(rows)
        self.candidates = [self.item]
        self.more_sources = False
        self.denied = False
        self.acquisitions = 0

    def sources(
        self, query: str, context: RunContext, *, offset: int = 0, limit: int = 50
    ) -> tuple[list[CorpusSource], bool]:
        del query, offset, limit
        context.check_active()
        if self.denied:
            raise PermissionError("Denied")
        return self.candidates, self.more_sources

    def page(
        self,
        source_id: str,
        context: RunContext,
        *,
        start: int = 0,
        limit: int = 30,
        as_of: date | None = None,
        historical_inventory: bool = False,
    ) -> tuple[CorpusSource, list[CorpusChunk], bool]:
        self.acquisitions += 1
        return super().page(
            source_id,
            context,
            start=start,
            limit=limit,
            as_of=as_of,
            historical_inventory=historical_inventory,
        )

    def related_sources_for_provision(
        self,
        source: CorpusSource,
        evidence: list[EvidenceItem],
        target: tuple[str, str | None],
        context: RunContext,
        *,
        offset: int = 0,
    ) -> dict[str, JsonValue] | None:
        del source, evidence, target, context, offset
        return {"navigation": "A separately assessed related authority"}


def read_named(broker: NamedBroker, context: RunContext) -> ToolOutcome:
    registry = CapabilityRegistry(
        build_corpus_specs(broker, named_provision_reads=True)
    )
    return registry.dispatch(
        CapabilityCall(
            name="read_named_provision",
            arguments={
                "source_name": "Example Law",
                "article": "7",
            },
        ),
        context,
    )


def test_named_read_delivers_all_target_clauses_and_related_navigation() -> None:
    broker = NamedBroker()
    outcome = read_named(broker, RunContext())
    assert outcome.status == OutcomeStatus.FOUND
    assert [item.text for item in outcome.evidence] == [
        row.text for row in broker.items[:2]
    ]
    assert outcome.data["source_id"] == str(broker.item.id)
    assert outcome.data["related_source_candidates"] == {
        "navigation": "A separately assessed related authority"
    }


@pytest.mark.parametrize("failure", ["ambiguous", "paged", "missing", "denied"])
def test_named_read_never_guesses_a_source_or_scans_unrelated_articles(
    failure: str,
) -> None:
    broker = NamedBroker()
    if failure == "ambiguous":
        broker.candidates.append(CorpusSource(uuid4(), "Another instrument", "other"))
    elif failure == "paged":
        broker.more_sources = True
    elif failure == "missing":
        broker.candidates = []
    else:
        broker.denied = True
    outcome = read_named(broker, RunContext())
    assert (
        outcome.status
        == {
            "ambiguous": OutcomeStatus.AMBIGUOUS,
            "paged": OutcomeStatus.AMBIGUOUS,
            "missing": OutcomeStatus.NOT_FOUND,
            "denied": OutcomeStatus.DENIED,
        }[failure]
    )
    assert not outcome.evidence
    assert broker.acquisitions == 0


def test_title_and_id_reads_share_the_same_canonical_acquisition() -> None:
    broker, context = NamedBroker(), RunContext()
    context.services.update(asv3_workflow_variant=ASV3_TUNED_VARIANT)
    context.services["shared_reads"] = SharedReads(
        fence=lambda _source_id, _caller: "captured-revision",
        producer_context=lambda caller: caller,
    )
    first = read_named(broker, context)
    registry = CapabilityRegistry(
        build_corpus_specs(broker, named_provision_reads=True)
    )
    second = registry.dispatch(
        CapabilityCall(
            name="read_provision",
            arguments={
                "source_id": str(broker.item.id),
                "article": "7",
            },
        ),
        context,
    )
    assert second.data["shared_read_reuse"] == "completed"
    assert [item.identity for item in second.evidence] == [
        item.identity for item in first.evidence
    ]
    assert broker.acquisitions == 1


def test_named_read_reuses_verified_delivered_identity_before_catalog_search() -> None:
    broker, context = NamedBroker(), RunContext()
    ledger = EvidenceLedger()
    item = original("8765 sayılı Faaliyet Kanunu", "5")
    assert item.search_doc is not None
    item = item.model_copy(
        update={
            "source_id": str(broker.item.id),
            "search_doc": item.search_doc.model_copy(
                update={"document_id": str(broker.item.id)}
            ),
        }
    )
    ledger.add([item], context)
    context.services["evidence"] = ledger
    broker.denied = (
        True  # A title lookup is unnecessary; the canonical read still authorizes.
    )
    registry = CapabilityRegistry(
        build_corpus_specs(broker, named_provision_reads=True)
    )
    outcome = registry.dispatch(
        CapabilityCall(
            name="read_named_provision",
            arguments={"source_name": "8765 sayılı Faaliyet Kanunu", "article": "7"},
        ),
        context,
    )
    assert outcome.status == OutcomeStatus.FOUND
    assert outcome.data["source_id"] == str(broker.item.id)


def test_named_cache_subscriber_releases_admission_before_waiting_for_id_read() -> None:
    entered, release, producer_ready, stopped = Event(), Event(), Event(), Event()

    class PausedBroker(NamedBroker):
        def sources(
            self, query: str, context: RunContext, *, offset: int = 0, limit: int = 50
        ) -> tuple[list[CorpusSource], bool]:
            entered.set()
            assert release.wait(5)
            return super().sources(query, context, offset=offset, limit=limit)

    broker = PausedBroker()
    context = RunContext(
        budget=SharedBudget(max_inflight_tools=1), cancelled=stopped.is_set
    )
    context.services["asv3_workflow_variant"] = ASV3_TUNED_VARIANT

    def producer(caller: RunContext) -> RunContext:
        producer_ready.set()
        return caller

    context.services["shared_reads"] = SharedReads(
        fence=lambda _source_id, _caller: "captured-revision", producer_context=producer
    )
    registry = CapabilityRegistry(
        build_corpus_specs(broker, named_provision_reads=True)
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        named = pool.submit(
            registry.dispatch,
            CapabilityCall(
                name="read_named_provision",
                arguments={"source_name": "Example Law", "article": "7"},
            ),
            context,
        )
        try:
            assert entered.wait(5)
            by_id = pool.submit(
                registry.dispatch,
                CapabilityCall(
                    name="read_provision",
                    arguments={"source_id": str(broker.item.id), "article": "7"},
                ),
                context,
            )
            assert producer_ready.wait(5)
            release.set()
            assert named.result(timeout=5).status == OutcomeStatus.FOUND
            assert by_id.result(timeout=5).status == OutcomeStatus.FOUND
            assert broker.acquisitions == 1
        finally:
            release.set()
            stopped.set()


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        ("Ürün Yönetmeliği madde 7 işlem şartları", ("Ürün Yönetmeliği", "7")),
        (
            "8765 sayılı Faaliyet Kanunu madde 8 istisna",
            ("8765 sayılı Faaliyet Kanunu", "8"),
        ),
        (
            "Example Product Regulation article 9 conditions",
            ("Example Product Regulation", "9"),
        ),
        ("Faaliyet Kanunu geçici madde 2 kapsam", ("Faaliyet Kanunu", "gecici 2")),
        ("Faaliyet Kanunu madde 7 ve Uygulama Yönetmeliği şartları", None),
        ("Faaliyet Kanunu madde 7 ve madde 9", None),
        ("İthalatta vergi matrahı ve itiraz yolları", None),
        ("Bu kanun madde 7 nasıl uygulanır", None),
    ],
)
def test_explicit_target_is_single_instrument_navigation(
    target: str,
    expected: tuple[str, str] | None,
) -> None:
    assert focused_source_target({"evidence_target": target}) == expected
    assert (
        focused_source_target(
            {"evidence_target": target, "discover_related_sources": True}
        )
        is None
    )


@pytest.mark.parametrize("tuned", [False, True])
@pytest.mark.parametrize("related", [False, True])
def test_known_article_target_avoids_wrong_corpus_sources_without_closing_discovery(
    tuned: bool, related: bool
) -> None:
    broker, context = NamedBroker(), RunContext()
    indexed_calls: list[dict[str, JsonValue]] = []

    def indexed(args: dict[str, JsonValue], _context: RunContext) -> ToolOutcome:
        indexed_calls.append(args)
        return ToolOutcome(
            status=OutcomeStatus.FOUND, summary="Independent corpus discovery"
        )

    broker.search_adapter = indexed
    registry = CapabilityRegistry(
        build_corpus_specs(broker, named_provision_reads=tuned)
    )
    arguments: dict[str, JsonValue] = {
        "query": "Example Law article 7 authorization",
        "mode": "hybrid",
        "coverage_item": "Authorization conditions",
        "evidence_target": "Example Law article 7 authorization conditions",
    }
    if related:
        arguments["discover_related_sources"] = True
    outcome = registry.dispatch(
        CapabilityCall(name="search_corpus", arguments=arguments), context
    )
    if tuned and not related:
        assert not indexed_calls
        assert outcome.data["search_performed"] is False
        assert [item.text for item in outcome.evidence] == [
            row.text for row in broker.items[:2]
        ]
        assert outcome.data["related_source_candidates"]
    elif not tuned and related:
        assert outcome.status == OutcomeStatus.INVALID
    else:
        assert indexed_calls


def test_partial_named_read_keeps_continuation_and_canonical_source_identity() -> None:
    broker = NamedBroker()
    broker.items = broker.items[:2]
    broker.partial = True
    outcome = read_named(broker, RunContext())
    assert outcome.status == OutcomeStatus.PARTIAL
    assert outcome.data["scan_truncated"] is True
    assert outcome.data["source_id"] == str(broker.item.id)
    assert len(outcome.evidence) == 2


@pytest.mark.parametrize("tuned", [False, True])
def test_provider_projection_preserves_original_text_range_and_current_validity(
    tuned: bool,
) -> None:
    ledger, context, record = setup_original()
    item = ledger.get(1)
    assert item is not None
    item.metadata["canonical_metadata"] = {
        "title": "Verified instrument",
        "labels": ["Index navigation only"] * 200,
        "validity_start": "2026-01-01",
        "version_unknown": False,
    }
    ledger = EvidenceLedger()
    context.services["evidence"] = ledger
    ledger.add([item], context)
    record = cast(dict[str, JsonValue], json.loads(ledger.serialize_records([1]))[0])
    context.services.update(
        research_profile="normal",
        asv3_workflow_variant=ASV3_TUNED_VARIANT if tuned else "standard",
    )
    selected = model(limit=1000000)
    adapter = ResearchModel(selected, context, lean_native_mode=True)
    originals = [turn("read-one", [record])]
    saved = copy.deepcopy(originals)
    adapter.decide(view(turns=originals, original_evidence=[record]))
    actual = delivered_originals(selected.invoke.call_args.kwargs)[0]
    assert actual["text"] == record["text"]
    assert actual["text_hash"] == record["text_hash"]
    assert actual.get("start_char", 0) == record.get("start_char", 0)
    assert actual["metadata"]["validity_start"] == "2026-01-01"
    assert actual["metadata"]["version_unknown"] is False
    assert ledger.completely_delivered(adapter.last_call_id or "") == {1}
    assert originals == saved
    payloads = [
        json.loads(message.content)
        for message in selected.invoke.call_args.kwargs["prompt"]
        if isinstance(message, ToolMessage)
    ]
    if tuned:
        ref = payloads[0]["original_evidence"][0]
        assert "metadata" not in ref
        assert ref["identity_ref"] == 1
        assert ref["text"] == record["text"]
        assert "canonical_metadata" not in actual["metadata"]
        assert actual["metadata"]["title"] == "Verified instrument"
    else:
        ref = payloads[0]["original_evidence_refs"][0]
        assert ref["metadata"] == record["metadata"]
        assert actual["metadata"] == record["metadata"]


def test_optional_coverage_remains_available_without_repetition_on_every_tool() -> None:
    broker, context = NamedBroker(), RunContext()
    context.services.update(
        lean_native_mode=True,
        research_profile="normal",
        outcome_map=OutcomeMap(["Question"], context),
    )
    registry = CapabilityRegistry(build_corpus_specs(broker))
    baseline = registry.definitions(context)
    context.services["asv3_workflow_variant"] = ASV3_TUNED_VARIANT
    tuned = registry.definitions(context)
    by_name: dict[str, dict[str, JsonValue]] = {}
    for tool in tuned:
        function = tool["function"]
        assert isinstance(function, dict)
        name, parameters = function["name"], function["parameters"]
        assert isinstance(name, str) and isinstance(parameters, dict)
        properties = parameters["properties"]
        assert isinstance(properties, dict)
        by_name[name] = properties
    assert {"_outcomes", "_coverage"} <= set(by_name["read_provision"])
    assert "_coverage" not in by_name["read_source_range"]
    assert len(json.dumps(tuned)) < len(json.dumps(baseline)) * 0.7
    context.services.pop("asv3_workflow_variant")
    assert registry.definitions(context) == baseline


@pytest.mark.parametrize("status", ["examined", "not_material"])
@pytest.mark.parametrize("layout", ["application", "challenged_norm_continuation"])
def test_preliminary_judicial_original_cannot_close_an_operative_assessment(
    status: str,
    layout: str,
) -> None:
    context, ledger, reviews = tuned_context()
    seen(context, ledger, reviews)
    text = "**Karar Tarihi: 01/01/2026**\nİTİRAZIN KONUSU: The applicant requests annulment."
    items = [EvidenceItem(source_id="decision", text=text, chunk_id="application")]
    if layout == "challenged_norm_continuation":
        items = [
            judicial_chunk("I. CHALLENGED PROVISIONS\nThe challenged rule states:", 0),
            judicial_chunk('"An authorization is required."\nII. MERITS', 1),
        ]
        text = items[-1].text
    numbers = ledger.add(items, context)
    number = numbers[-1]
    deliver(ledger, "answer-call", [1, *numbers])
    context.services["last_model_call_id"] = "answer-call"
    outcome = terminal_registry([]).dispatch(
        CapabilityCall(
            name="submit_answer",
            arguments={
                "answer": f"The rule certainly still applies [{number}].",
                "basis": "originals",
                "_related_source_reviews": [
                    review(
                        status=status,
                        witnesses=[
                            {
                                "citation": number,
                                "start_char": 0,
                                "end_char": len(text),
                            }
                        ],
                    )
                ],
            },
        ),
        context,
    )
    assert outcome.status == OutcomeStatus.INVALID
    diagnostic = outcome.data["related_source_review_error"]
    assert isinstance(diagnostic, dict)
    assert diagnostic["code"] == "nonoperative_judicial_witnesses"
    assert reviews.view(context, ledger, {1, number})["pending_lead_ids"]


def judicial_chunk(text: str, position: int) -> EvidenceItem:
    item = original("decision", str(position), kind="court_decision", text=text)
    return item.model_copy(
        update={
            "metadata": {
                "position": position,
                "read_as_of_date": "2026-01-02",
                "heading_path": ["Court decision", "Case information"],
            }
        }
    )


@pytest.mark.parametrize(
    "break_context", [None, "gap", "source", "date", "duplicate", "derived"]
)
def test_judicial_section_continuation_requires_contiguous_canonical_originals(
    break_context: str | None,
) -> None:
    previous = judicial_chunk("I. İPTALİ İSTENEN KANUN HÜKMÜ\nThe challenged rule:", 0)
    current = judicial_chunk('"A fee is required."', 1)
    context = [previous]
    if break_context == "gap":
        current.metadata["position"] = 2
    elif break_context == "source":
        previous.source_id = "another-decision"
    elif break_context == "date":
        previous.metadata["read_as_of_date"] = "2025-01-02"
    elif break_context == "duplicate":
        context.append(judicial_chunk("I. MERITS\nThe request is rejected.", 0))
    elif break_context == "derived":
        previous.metadata["derived"] = True
    assert nonoperative_judicial_witness_role(
        current, 0, len(current.text), source_context=context
    ) == ("preliminary" if break_context is None else "unknown")


def test_judicial_metadata_and_section_transitions_do_not_reject_disposition_text() -> (
    None
):
    previous = judicial_chunk("I. CHALLENGED PROVISIONS\nThe challenged rule:", 0)
    current = judicial_chunk("II. MERITS\nThe request is granted.", 1)
    assert (
        nonoperative_judicial_witness_role(
            current, 0, len(current.text), source_context=[previous]
        )
        == "unknown"
    )
    current = judicial_chunk("The request is granted.", 1)
    current.metadata["heading_path"] = ["Court decision", "Disposition"]
    assert (
        nonoperative_judicial_witness_role(
            current, 0, len(current.text), source_context=[previous]
        )
        == "unknown"
    )


def test_structural_signal_does_not_reject_a_following_disposition_or_approve_unknown_text() -> (
    None
):
    text = "İTİRAZIN KONUSU: The requested relief.\n\n## HÜKÜM\nThe request is granted."
    item = EvidenceItem(source_id="court", text=text)
    assert (
        nonoperative_judicial_witness_role(item, 0, text.index("##")) == "preliminary"
    )
    assert nonoperative_judicial_witness_role(item, 0, len(text)) == "unknown"
    assert (
        nonoperative_judicial_witness_role(item, text.index("The request"), len(text))
        == "unknown"
    )
