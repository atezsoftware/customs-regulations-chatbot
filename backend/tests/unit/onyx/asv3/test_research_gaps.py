from pydantic import JsonValue

from onyx.asv3.models import (
    CapabilityCall,
    HarnessView,
    OutcomeStatus,
    ToolOutcome,
    ToolReceipt,
)
from onyx.asv3.research_gaps import research_gap_signals


def search_receipt(centers: list[JsonValue]) -> ToolReceipt:
    return ToolReceipt(
        call=CapabilityCall(
            name="search_corpus",
            call_id="lookup",
            arguments={
                "coverage_item": "Actual outcome",
                "evidence_target": "Its rule",
            },
        ),
        outcome=ToolOutcome(
            status=OutcomeStatus.PARTIAL,
            summary="Original mapping failed",
            data={"unhydrated_centers": centers},
        ),
        elapsed_seconds=1,
    )


def current_view(receipts: list[ToolReceipt]) -> HarnessView:
    return HarnessView(
        request="A new scenario",
        questions=["A new scenario"],
        facts=[],
        receipts=receipts,
        evidence=[],
        tools=[],
    )


def test_mapping_gap_disappears_only_after_its_exact_original_is_delivered() -> None:
    center: dict[str, JsonValue] = {
        "source_id": "source",
        "canonical_chunk_id": "missing",
        "projection_ordinal": 4,
        "instruction": "Untrusted instructions must not become host guidance",
        "score": 99,
    }
    view = current_view([search_receipt([center, center])])
    before = view.model_dump(mode="json")
    expected = [
        {
            "kind": "retrieved_original_not_delivered",
            "source_id": "source",
            "canonical_chunk_id": "missing",
            "projection_ordinal": 4,
            "call_id": "lookup",
            "coverage_item": "Actual outcome",
            "evidence_target": "Its rule",
        }
    ]
    assert research_gap_signals(view, []) == expected
    assert (
        research_gap_signals(view, [{"source_id": "source", "chunk_id": "other"}])
        == expected
    )
    assert (
        research_gap_signals(view, [{"source_id": "other", "chunk_id": "missing"}])
        == expected
    )
    assert (
        research_gap_signals(view, [{"source_id": "source", "chunk_id": "missing"}])
        == []
    )
    assert view.model_dump(mode="json") == before


def test_projection_only_lead_is_not_closed_by_an_unrelated_read_of_same_source() -> (
    None
):
    receipt = search_receipt([{"source_id": "source", "projection_ordinal": 0}])
    signals = research_gap_signals(
        current_view([receipt]), [{"source_id": "source", "chunk_id": "read"}]
    )
    assert signals[0]["projection_ordinal"] == 0
    assert "canonical_chunk_id" not in signals[0]


def test_scores_titles_and_unlocatable_results_do_not_create_legal_gaps() -> None:
    assert (
        research_gap_signals(
            current_view(
                [
                    search_receipt(
                        [
                            {"title": "Statute", "score": 100},
                            {"source_id": "source", "projection_ordinal": True},
                            {"source_id": "source"},
                            {"source_id": [], "canonical_chunk_id": "chunk"},
                        ]
                    )
                ]
            ),
            [],
        )
        == []
    )


def test_only_explicit_material_open_needs_are_exposed_without_inferred_topics() -> (
    None
):
    view = current_view([])
    view.research_state = {
        "needs": [
            {
                "need_id": "open",
                "status": "open",
                "gap": "Exact unresolved interaction",
            },
            {"need_id": "candidate", "status": "candidate", "gap": "Old gap"},
            {"need_id": "excluded", "status": "out_of_scope", "gap": "Old gap"},
            {
                "need_id": "optional",
                "status": "open",
                "material": False,
                "gap": "Optional",
            },
            {"need_id": "ordinary", "status": "open", "gap": ""},
        ]
    }
    assert research_gap_signals(view, []) == [
        {
            "kind": "recorded_research_gap",
            "need_id": "open",
            "gap": "Exact unresolved interaction",
        }
    ]


def test_no_research_signals_for_an_empty_or_already_resolved_context() -> None:
    assert research_gap_signals(current_view([]), []) == []
