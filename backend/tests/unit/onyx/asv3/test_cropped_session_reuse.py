"""Prior facts survive while old source bodies require actual new-question reads."""

import copy
import json
from typing import cast

import pytest
import tiktoken
from pydantic import JsonValue

from onyx.asv3.cropped_session_reuse import (
    active_cropped_session_numbers,
    install_cropped_session_reuse,
    record_cropped_session_receipt,
    session_numbers_to_remember,
)
from onyx.asv3.cropped_transport import prepare_cropped_writer_view
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.harness import Harness
from onyx.asv3.llm_adapter import ResearchModel, _NativeOriginalCatalogue
from onyx.asv3.models import (
    CapabilityCall,
    Decision,
    EvidenceItem,
    OutcomeStatus,
    RunContext,
    ToolOutcome,
    ToolSpec,
)
from onyx.asv3.outcome_map import OutcomeMap
from onyx.asv3.registry import CapabilityRegistry, build_core_specs
from onyx.asv3.research_state import ResearchState
from onyx.asv3.session_research import (
    retain_session_research,
    session_research_checkpoint,
)
from onyx.asv3.shared_originals import related_provision_originals
from onyx.asv3.workflow_variant import ASV3_TUNED_VARIANT
from onyx.tracing.flows import LLMFlow
from tests.unit.onyx.asv3.test_native_model_adapter import model
from tests.unit.onyx.asv3.test_shared_originals import full_record, original

CASE = (
    "İzinli gönderici yetkisine ve 3. düzey kapsamlı teminata sahip firma, "
    "kendi tesisinde mühürlediği araçla İtalya'ya tekstil gönderiyor. "
    "X-Ray taramasında koliler arasına gizlenmiş 200 kg dökme tütün bulunuyor."
)
QUESTION = (
    "Peki tespit edilen tütün için Gümrük Kanunu 235/5 mi, yoksa 5607 sayılı "
    "Kaçakçılıkla Mücadele Kanunu hükümleri mi uygulanır? İkisi arasında öncelik var mıdır?"
)
HISTORY = (
    f"user: {CASE}\nassistant: Önceki yetki/teminat değerlendirmesi.\nuser: {QUESTION}"
)


def fixture(
    count: int = 4, *, cropped: bool = True
) -> tuple[Harness, ResearchModel, dict[str, JsonValue]]:
    items = [
        original(
            "old-guarantee",
            "Earlier guarantee condition remains archived.",
            source="teminat",
            headings=["Transit Tebliği", "MADDE 11/A"],
            metadata={"title": "Transit Tebliği"},
        ),
        original(
            "operative",
            "Current operative paragraph includes its precise condition.",
            source="gk",
            headings=["Gümrük Kanunu", "MADDE 235"],
            metadata={"title": "Gümrük Kanunu"},
        ),
        original(
            "exception",
            "The same article's full exception and subsequent step.",
            source="gk",
            headings=["Gümrük Kanunu", "MADDE 235", "(5) Exception"],
            metadata={"title": "Gümrük Kanunu", "article_closure_complete": False},
        ),
        original(
            "other",
            "Another archived article has a separate effect.",
            source="gk",
            headings=["Gümrük Kanunu", "MADDE 85"],
            metadata={"title": "Gümrük Kanunu"},
        ),
        *(
            original(
                f"old-{n}",
                f"Archived literal original {n} condition and exception.",
                source=f"old-source-{n % 7}",
                headings=["Prior instrument", f"MADDE {n}"],
                metadata={"title": "Prior instrument"},
            )
            for n in range(4, count)
        ),
    ][:count]
    previous: dict[str, JsonValue] = {
        "scope": {"document_set": 15},
        "request": CASE,
        "session_research": {"requests": [CASE]},
        "evidence": {
            "records": [
                {"citation": n, "item": item.model_dump(mode="json")}
                for n, item in enumerate(items, 1)
            ]
        },
    }
    context = RunContext(
        scope={"document_set": 15},
        services={
            "asv3_exp_cropped": cropped,
            "cropped_user_context": QUESTION,
            "user_id": "actual-owner",
            "lean_native_mode": True,
            "research_profile": "normal",
            "asv3_workflow_variant": ASV3_TUNED_VARIANT,
        },
    )
    ledger = EvidenceLedger()
    retain_session_research(previous, context, ledger, lambda _items, _context: None)
    install_cropped_session_reuse(context, ledger, QUESTION, previous=previous)
    context.services["evidence"] = ledger
    context.services["research_state"] = ResearchState([QUESTION], context)
    context.services["outcome_map"] = OutcomeMap(
        [QUESTION], context, factual_context=QUESTION + "\n" + HISTORY
    )
    adapter = ResearchModel(
        model(limit=1_000_000),
        context,
        history=HISTORY,
        lean_native_mode=True,
        token_counter=lambda text: len(
            tiktoken.get_encoding("o200k_base").encode(text)
        ),
    )
    registry = CapabilityRegistry()
    harness = Harness(
        request=QUESTION,
        context=context,
        registry=registry,
        evidence=ledger,
        decide=lambda _view: Decision(answer="Unused"),
        on_receipt=lambda receipt: record_cropped_session_receipt(
            receipt, context, ledger
        ),
    )
    for spec in build_core_specs(registry, ledger, harness.snapshot):
        registry.register(spec)
    numbers = context.services["session_research"]["reused_evidence_numbers"]
    assert isinstance(numbers, list)
    for number in session_numbers_to_remember(numbers, context, ledger):
        item = ledger.get(cast(int, number))
        assert item is not None
        harness.evidence_working_set.remember(cast(int, number), 0, len(item.text))
    return harness, adapter, previous


def test_5172_like_first_packet_has_no285_old_source_catalogue_or_delivery() -> None:
    harness, adapter, previous = fixture(285)
    before = copy.deepcopy(previous)
    view = harness.view()
    assert view.original_evidence == []
    assert len(harness.evidence.citation_numbers()) == 285
    prompt, _, _ = adapter._fit_native_decision(view)
    bound = _NativeOriginalCatalogue.bind(prompt, cheap_metadata=True)
    assert bound.records == ()
    serialized = "\n".join(message.model_dump_json() for message in prompt)
    assert "original_metadata_catalogue" not in serialized
    raw = prompt[-1].model_dump(mode="json")["content"]
    text = raw if isinstance(raw, str) else "\n".join(row["text"] for row in raw)
    assert "original_evidence_ranges" not in json.loads(text)
    assert CASE in adapter.history and adapter.history == HISTORY
    assert "prior assistant assertions" in serialized
    assert "Prior instrument" in serialized
    memory = harness.context.services["session_research"]
    assert memory["requests"] == [CASE]
    assert len(memory["prior_source_navigation"]) == 9
    assert harness.evidence.completely_delivered("new-call") == set()
    assert previous == before


def test_exact_archived_read_promotes_only_source_and_preserves_article_siblings() -> (
    None
):
    harness, adapter, _ = fixture()
    receipts = harness.dispatch_repair_reads(
        [
            CapabilityCall(
                name="read_evidence", call_id="actual-read", arguments={"citation": 2}
            )
        ]
    )
    assert receipts[0].outcome.original_reads[0].citation == 2
    assert active_cropped_session_numbers(harness.evidence, harness.context) == [2]
    view = harness.view()
    prompt, tools, _ = adapter._fit_native_decision(view)
    bound = _NativeOriginalCatalogue.bind(prompt, cheap_metadata=True)
    assert {row["citation"] for row in bound.records} == {2, 3}
    assert cast(str, full_record(harness.evidence, 3)["text"]) in "\n".join(
        message.model_dump_json() for message in prompt
    )
    sibling = harness.evidence.get(3)
    assert sibling is not None and sibling.metadata["article_closure_complete"] is False
    assert harness.evidence.completely_delivered("actual-writer") == set()
    adapter._invoke_once(
        prompt,
        tools,
        LLMFlow.ASV3_COORDINATOR,
        max_tokens=100,
        research=True,
        structured=False,
        native_catalogue=bound,
    )
    assert harness.evidence.completely_delivered(adapter.last_call_id or "") == {2, 3}
    full = prepare_cropped_writer_view(
        view, harness.context, adapter._native_original_records(view.turns)
    )
    assert {row["citation"] for row in full.original_evidence} == {2, 3}
    assert harness.evidence.citation_numbers() == (1, 2, 3, 4)


def test_direct_same_source_read_and_every_new_original_stay_active_for_final() -> None:
    harness, adapter, _ = fixture()
    item = harness.evidence.get(2)
    assert item is not None
    harness.registry.register(
        ToolSpec(
            name="read_provision",
            description="Actual direct read",
            parameters={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
            handler=lambda _args, _context: ToolOutcome(
                status=OutcomeStatus.FOUND,
                summary="Actual direct original",
                evidence=[item],
            ),
        )
    )
    harness.dispatch_repair_reads(
        [CapabilityCall(name="read_provision", call_id="direct", arguments={})]
    )
    new_id = harness.evidence.add(
        [EvidenceItem(source_id="new-statute", text="New current-question original.")],
        harness.context,
    )[0]
    assert active_cropped_session_numbers(harness.evidence, harness.context) == [
        2,
        new_id,
    ]
    full = prepare_cropped_writer_view(
        harness.view(), harness.context, adapter._native_original_records(harness.turns)
    )
    assert {row["citation"] for row in full.original_evidence} == {2, new_id}


def test_partial_peek_is_not_full_delivery_and_unknown_required_old_source_survives() -> (
    None
):
    harness, _, _ = fixture()
    harness.dispatch_repair_reads(
        [
            CapabilityCall(
                name="read_evidence",
                call_id="peek",
                arguments={"citation": 2, "num_chars": 4},
            )
        ]
    )
    assert harness.evidence.completely_delivered("peek") == set()
    view = harness.view().model_copy(update={"required_evidence_numbers": [1]})
    full = prepare_cropped_writer_view(view, harness.context, [])
    assert {row["citation"] for row in full.original_evidence} == {1, 2}
    requested = harness.evidence.get(2)
    assert requested is not None and full.original_evidence[1]["text"] == requested.text


def test_question_fence_preserves_full_prior_user_scenario_context() -> None:
    harness, _, _ = fixture()
    context = harness.context
    context.services["cropped_user_context"] = CASE + "\n\n" + QUESTION
    memory = context.services["session_research"]
    memory.pop("cropped_source_archive")
    install_cropped_session_reuse(context, harness.evidence, QUESTION)
    assert active_cropped_session_numbers(harness.evidence, context) == []
    assert memory["requests"] == [CASE]


def test_archive_resume_preserves_activation_and_source_version_fence() -> None:
    harness, _, _ = fixture()
    harness.dispatch_repair_reads(
        [
            CapabilityCall(
                name="read_evidence", call_id="read", arguments={"citation": 2}
            )
        ]
    )
    memory = session_research_checkpoint(harness.context, QUESTION)
    saved_ledger = harness.evidence.export()
    restored = RunContext(
        run_id=harness.context.run_id,
        scope=harness.context.scope,
        services={
            "asv3_exp_cropped": True,
            "cropped_user_context": QUESTION,
            "user_id": "actual-owner",
            "session_research": memory,
        },
    )
    ledger = EvidenceLedger()
    ledger.restore(saved_ledger, restored)
    install_cropped_session_reuse(restored, ledger, QUESTION)
    assert active_cropped_session_numbers(ledger, restored) == [2]
    assert ledger.export() == saved_ledger
    ledger._items[2].metadata["version"] = "changed"
    with pytest.raises(ValueError, match="version"):
        active_cropped_session_numbers(ledger, restored)


@pytest.mark.parametrize("change", ["scope", "user", "request", "hash", "checkpoint"])
def test_unread_source_fences_fail_closed(change: str) -> None:
    harness, _, _ = fixture()
    context, ledger = harness.context, harness.evidence
    if change == "scope":
        context.scope["document_set"] = 16
    elif change == "user":
        context.services["user_id"] = "other-owner"
    elif change == "request":
        context.services["cropped_user_context"] = "Changed question"
    elif change == "hash":
        ledger._items[1].text = "Changed original"
    else:
        memory = context.services["session_research"]
        memory["cropped_source_archive"]["promoted_citations"] = [1]
        with pytest.raises(ValueError, match="checkpoint"):
            install_cropped_session_reuse(context, ledger, QUESTION)
        return
    with pytest.raises(ValueError):
        active_cropped_session_numbers(ledger, context)


def test_non_cropped_session_reuse_and_final_union_are_unchanged() -> None:
    harness, adapter, _ = fixture(cropped=False)
    assert len(harness.view().original_evidence) == 4
    assert "cropped_source_archive" not in harness.context.services["session_research"]
    assert "cropped_session_reuse" not in harness.context.services
    full = prepare_cropped_writer_view(
        harness.view(), harness.context, adapter._native_original_records(harness.turns)
    )
    assert {row["citation"] for row in full.original_evidence} == {1, 2, 3, 4}


def test_article_closure_does_not_cross_version_or_different_article() -> None:
    harness, _, _ = fixture()
    record = full_record(harness.evidence, 2)
    related = related_provision_originals(harness.evidence, [record])
    assert [row["citation"] for row in related] == [3]


def test_prior_citation_map_tracks_immutable_identity_after_renumbering() -> None:
    harness, _, previous = fixture(12)
    memory = harness.context.services["session_research"]
    previous_evidence = previous["evidence"]
    assert isinstance(previous_evidence, dict)
    old_rows = previous_evidence["records"]
    assert isinstance(old_rows, list)
    old = {
        row["citation"]: EvidenceItem.model_validate(row["item"]).identity
        for row in old_rows
        if isinstance(row, dict)
    }
    for prior, current in memory["prior_citation_map"]:
        item = harness.evidence.get(current)
        assert item is not None and item.identity == old[prior]
    assert len(memory["prior_citation_map"]) == 12
    assert any(prior != current for prior, current in memory["prior_citation_map"])
