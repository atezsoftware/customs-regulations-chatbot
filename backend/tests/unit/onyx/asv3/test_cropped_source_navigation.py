"""Actual registry navigation leaves full provision evidence and other modes intact."""

from __future__ import annotations

from dataclasses import replace
from datetime import date
from uuid import uuid4

import pytest

from onyx.asv3.corpus_tools import build_corpus_specs, evidence_for_chunk
from onyx.asv3.cropped_research_limit import CroppedResearchLimit
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import CapabilityCall, OutcomeStatus, RunContext
from onyx.asv3.registry import CapabilityRegistry
from onyx.db.asv3_corpus import CorpusChunk
from tests.unit.onyx.asv3.test_corpus_source_sandbox import MemoryBroker


def source(*, kind: str = "yonetmelik", count: int = 501) -> MemoryBroker:
    identifier = uuid4()
    chunks = [
        CorpusChunk(
            id="intro",
            source_id=identifier,
            text="Introductory navigation, not an operative article.",
            position=0,
            projection_ordinal=0,
            heading_path=("SOURCE", "Introduction"),
            metadata={"document_type": kind},
            validity_start=None,
            validity_end=None,
            status="active",
        )
    ]
    for position in range(1, count):
        article = str((position - 1) // 2 + 1)
        paragraph = "1" if position % 2 else "2"
        chunks.append(
            CorpusChunk(
                id=f"chunk-{position}",
                source_id=identifier,
                text=(
                    f"MADDE {article} - (1) All material eligibility conditions apply."
                    if paragraph == "1"
                    else f"(2) Exception to article {article}: qualifying event reverses the result."
                ),
                position=position,
                projection_ordinal=position,
                heading_path=("SOURCE", f"MADDE {article} - Conditions", paragraph),
                metadata={
                    "document_type": kind,
                    "article_no": article,
                    "paragraph_no": paragraph,
                },
                validity_start=date(2020, 1, 1),
                validity_end=date(2025, 1, 1) if paragraph == "1" else None,
                status="active",
            )
        )
    return MemoryBroker(chunks)


def child(
    *,
    cropped: bool = True,
    depth: int = 1,
    searches: int = 3,
    independent: bool = False,
    delegated: bool = True,
) -> tuple[RunContext, CroppedResearchLimit, EvidenceLedger]:
    context = RunContext(depth=depth, scope={"document_set": 15})
    ledger = EvidenceLedger()
    context.services.update(
        asv3_exp_cropped=cropped,
        independent_question=independent,
        task_id="actual-worker",
        evidence=ledger,
    )
    controller = CroppedResearchLimit(context, "Original requested outcomes")
    for index in range(searches):
        controller.record_search(f"actual-search-{index}")
    context.services["cropped_research_controller"] = controller
    if cropped and delegated and not independent:
        worker = context if context.depth else context.child()
        controller.reserve_model("actual-worker", context=worker)
    return context, controller, ledger


def read_page(broker: MemoryBroker, context: RunContext, *, start: int = 0):
    return CapabilityRegistry(build_corpus_specs(broker)).dispatch(
        CapabilityCall(
            name="read_source_range",
            arguments={"source_id": str(broker.item.id), "start": start, "limit": 100},
        ),
        context,
    )


def test_actual_registered_navigation_has_no_delivery_and_provision_keeps_exception() -> (
    None
):
    broker = source()
    context, controller, ledger = child()
    assert controller.long_tail is False  # Prospective gating precedes call eight.
    ledger.add([evidence_for_chunk(broker.item, broker.items[490])], context)
    before = ledger.export()
    navigation = read_page(broker, context)
    assert navigation.status == OutcomeStatus.PARTIAL
    assert navigation.evidence == [] and navigation.original_reads == []
    assert ledger.add(navigation.evidence, context) == []
    assert ledger.export() == before
    data = navigation.data
    assert data["requested_read_executed"] is False
    assert data["not_read"] is True
    assert data["absence_proven"] is False
    assert data["legal_approval"] is False
    assert data["article_closure_complete"] is False
    assert data["has_more"] is False
    assert data["next_position"] == 501
    assert data["navigation_scanned_chunks"] == 501
    assert len(data["headings"]) == 250
    assert data["unowned_headings"][0]["anchor_chunk_id"] == "intro"
    first = data["headings"][0]
    assert first["heading_path"] == ["SOURCE", "MADDE 1 - Conditions"]
    assert first["observed_position_ranges"] == [[1, 3]]
    assert first["chunk_count"] == 2
    assert first["observed_validity"] == [
        {
            "validity_start": "2020-01-01",
            "validity_end": "2025-01-01",
            "status": "active",
        },
        {"validity_start": "2020-01-01", "validity_end": None, "status": "active"},
    ]
    assert all("text" not in row for row in data["headings"])

    provision = CapabilityRegistry(build_corpus_specs(broker)).dispatch(
        CapabilityCall(
            name="read_provision",
            arguments={"source_id": str(broker.item.id), "article": "1"},
        ),
        context,
    )
    assert provision.status == OutcomeStatus.FOUND
    assert [item.text for item in provision.evidence] == [
        row.text for row in broker.items[1:3]
    ]
    assert "Exception" in provision.evidence[1].text
    assert provision.data["evidence_truncated"] is False
    assert provision.data["scan_truncated"] is False
    assert ledger.add(provision.evidence, context) == [2, 3]
    retained = ledger.get(1)
    assert retained is not None and retained.text == broker.items[490].text


@pytest.mark.parametrize(
    "kind", ["kanun", "Kanun", "yonetmelik", "Yönetmelik", "teblig", "Tebliğ"]
)
def test_only_observed_normative_metadata_variants_enable_navigation(kind: str) -> None:
    context, _, _ = child()
    assert read_page(source(kind=kind), context).data["navigation_only"] is True


@pytest.mark.parametrize(
    "settings",
    [
        {"cropped": False},
        {"depth": 0, "delegated": False},
        {"delegated": False},
        {"independent": True},
    ],
)
def test_protected_undelegated_and_independent_reads_have_same_originals(
    settings,
) -> None:
    broker = source()
    baseline, _, _ = child(cropped=False)
    context, _, _ = child(**settings)
    assert (
        read_page(broker, context).model_dump()
        == read_page(broker, baseline).model_dump()
    )


@pytest.mark.parametrize("depth", [0, 1])
@pytest.mark.parametrize("searches", [0, 1, 2, 3])
def test_actual_delegation_enables_root_and_worker_navigation_without_search_gate(
    depth: int, searches: int
) -> None:
    context, controller, _ = child(depth=depth, searches=searches)
    assert controller.has_delegated_research and not controller.long_tail
    navigation = read_page(source(), context)
    assert navigation.status == OutcomeStatus.PARTIAL
    assert navigation.data["not_read"] is True
    assert navigation.evidence == navigation.original_reads == []


@pytest.mark.parametrize("limit", [60, 70])
def test_delegated_root_incomplete_page_uses_navigation_and_keeps_prior_originals(
    limit: int,
) -> None:
    broker = source()
    context, controller, ledger = child(depth=0, searches=2)
    # The real root-page envelope had eight root and two delegated decisions.
    for _ in range(8):
        controller.reserve_model("root", context=context)
    worker = context.child()
    controller.reserve_model("actual-worker", context=worker)
    ledger.add([evidence_for_chunk(broker.item, broker.items[490])], context)
    before = ledger.export()
    registry = CapabilityRegistry(build_corpus_specs(broker))
    navigation = registry.dispatch(
        CapabilityCall(
            name="read_source_range",
            arguments={"source_id": str(broker.item.id), "start": 0, "limit": limit},
        ),
        context,
    )
    assert controller.progress()["research_model_calls"] == 10
    assert not controller.long_tail
    assert navigation.status == OutcomeStatus.PARTIAL
    assert navigation.data["requested_read_executed"] is False
    assert navigation.data["absence_proven"] is False
    assert navigation.data["source_text_delivered"] is False
    assert navigation.data["navigation_scanned_chunks"] == 501
    assert navigation.evidence == navigation.original_reads == []
    assert ledger.export() == before


def test_resumed_same_scope_delegation_keeps_navigation_without_new_worker_call() -> (
    None
):
    context, controller, _ = child(depth=0, searches=2)
    saved = controller.export()
    resumed = CroppedResearchLimit(
        context, controller.request, policy=controller.policy
    )
    context.services["cropped_research_controller"] = resumed
    resumed.restore(saved)
    before = resumed.progress()
    navigation = read_page(source(), context)
    assert navigation.data["navigation_only"] is True
    assert resumed.progress() == before


def test_root_only_large_page_stays_byte_equal_to_original_read() -> None:
    broker = source()
    context, controller, _ = child(depth=0, searches=2, delegated=False)
    for _ in range(7):
        controller.reserve_model("root", context=context)
    controller.record_sources(
        [evidence_for_chunk(broker.item, row) for row in broker.items[:243]]
    )
    assert not controller.long_tail
    assert not controller.has_delegated_research
    baseline, _, _ = child(cropped=False)
    actual = read_page(broker, context)
    assert actual.model_dump() == read_page(broker, baseline).model_dump()
    assert len(actual.evidence) == 100


def test_cheap_flag_does_not_activate_the_cropped_navigation() -> None:
    broker = source()
    baseline, _, _ = child(cropped=False)
    context, _, _ = child()
    context.services["asv3_exp_cheap"] = True
    assert (
        read_page(broker, context).model_dump()
        == read_page(broker, baseline).model_dump()
    )


@pytest.mark.parametrize(
    "kind", ["mahkeme", "court_decision", "yazi", "genelge", "", "kanun yorumları"]
)
def test_judgment_letter_unknown_or_unsupported_types_preserve_body(kind: str) -> None:
    broker = source(kind=kind)
    baseline, _, _ = child(cropped=False)
    context, _, _ = child()
    assert (
        read_page(broker, context).model_dump()
        == read_page(broker, baseline).model_dump()
    )


def test_small_complete_source_preserves_actual_full_read() -> None:
    broker = source(count=11)
    context, _, _ = child()
    outcome = read_page(broker, context)
    assert outcome.status == OutcomeStatus.FOUND
    assert len(outcome.evidence) == 11 and "navigation_only" not in outcome.data


def test_no_structural_heading_and_conflicting_metadata_preserve_actual_read() -> None:
    context, _, _ = child()
    broker = source()
    broker.items = [
        replace(row, heading_path=("SOURCE", "Continuous body")) for row in broker.items
    ]
    assert len(read_page(broker, context).evidence) == 100
    broker = source()
    broker.items[12] = replace(broker.items[12], metadata={"document_type": "mahkeme"})
    assert len(read_page(broker, context).evidence) == 100
    broker = source()
    broker.items[120] = replace(
        broker.items[120], metadata={"document_type": "unknown"}
    )
    assert len(read_page(broker, context).evidence) == 100


def test_scan_truncation_unowned_rows_and_position_gaps_are_honest() -> None:
    broker = source(count=2601)
    context, _, _ = child()
    broker.items = [
        replace(row, position=row.position + 1) if row.position >= 2 else row
        for row in broker.items
    ]
    broker.items[15] = replace(
        broker.items[15], heading_path=("SOURCE", "Annex identity unresolved")
    )
    outcome = read_page(broker, context)
    assert outcome.status == OutcomeStatus.PARTIAL and outcome.evidence == []
    assert outcome.data["has_more"] is True
    assert outcome.data["navigation_scanned_chunks"] == 2000
    assert outcome.data["next_position"] == 2001
    assert any(
        row["anchor_chunk_id"] == broker.items[15].id
        for row in outcome.data["unowned_headings"]
    )
    assert outcome.data["headings"][0]["observed_position_ranges"] == [[1, 2], [3, 4]]


def test_actual_truncated_navigation_continues_from_requested_position() -> None:
    broker = source(count=2701)
    context, _, ledger = child()
    first = read_page(broker, context)
    assert first.status == OutcomeStatus.PARTIAL
    assert first.data["navigation_scan_start"] == 0
    assert first.data["navigation_scanned_chunks"] == 2000
    assert first.data["next_position"] == 2000
    assert first.data["has_more"] is True

    second = read_page(broker, context, start=2000)
    assert second.status == OutcomeStatus.PARTIAL
    assert second.data["requested_start"] == 2000
    assert second.data["navigation_scan_start"] == 2000
    assert second.data["navigation_scanned_chunks"] == 701
    assert second.data["next_position"] == 2701
    assert second.data["has_more"] is False
    assert second.data["requested_read_executed"] is False
    assert second.data["article_closure_complete"] is False
    assert first.evidence == second.evidence == []
    assert first.original_reads == second.original_reads == []
    assert ledger.citation_numbers() == ()
    first_headings = first.data["headings"]
    second_headings = second.data["headings"]
    assert second_headings[0]["article"] == "1000"
    assert second_headings[0]["anchor_chunk_id"] == "chunk-2000"
    assert second_headings[0]["observed_position_ranges"] == [[2000, 2001]]
    assert second_headings[1]["article"] == "1001"
    assert second_headings[1]["observed_position_ranges"] == [[2001, 2003]]
    assert first_headings[-1]["observed_position_ranges"] == [[1999, 2000]]
    first_positions = {
        position
        for row in [*first_headings, *first.data["unowned_headings"]]
        for start, end in row["observed_position_ranges"]
        for position in range(start, end)
    }
    second_positions = {
        position
        for row in [*second_headings, *second.data["unowned_headings"]]
        for start, end in row["observed_position_ranges"]
        for position in range(start, end)
    }
    assert first_positions.isdisjoint(second_positions)
    assert first_positions | second_positions == set(range(2701))


def test_duplicate_canonical_positions_preserve_the_ordinary_original_read() -> None:
    broker = source()
    broker.items[2] = replace(broker.items[2], position=1)
    context, _, _ = child()
    baseline, _, _ = child(cropped=False)
    actual = read_page(broker, context)
    assert actual.model_dump() == read_page(broker, baseline).model_dump()
    assert len(actual.evidence) == 100
    assert actual.evidence[1].text == broker.items[1].text
    assert actual.evidence[2].text == broker.items[2].text


def test_foreign_controller_and_foreign_scan_fail_closed(monkeypatch) -> None:
    broker = source()
    context, _, _ = child()
    foreign, _, _ = child()
    context.services["cropped_research_controller"] = foreign.services[
        "cropped_research_controller"
    ]
    assert read_page(broker, context).status == OutcomeStatus.ERROR
    context, _, _ = child()
    original_scan = broker.iter_chunks

    def wrong_source(*args, **kwargs):
        scan = original_scan(*args, **kwargs)
        scan.source = source().item
        return scan

    monkeypatch.setattr(broker, "iter_chunks", wrong_source)
    assert read_page(broker, context).status == OutcomeStatus.DENIED
