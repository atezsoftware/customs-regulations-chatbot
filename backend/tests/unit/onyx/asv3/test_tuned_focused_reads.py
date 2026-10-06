"""Protect canonical scoped reading and exact evidence while reducing input overhead."""

import copy
import json
from datetime import date
from typing import cast
from uuid import uuid4

import pytest
from pydantic import JsonValue

from onyx.asv3.corpus_tools import build_corpus_specs
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.judicial_sections import nonoperative_judicial_witness_role
from onyx.asv3.llm_adapter import ResearchModel
from onyx.asv3.models import (
    CapabilityCall,
    EvidenceItem,
    OutcomeStatus,
    RunContext,
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
from tests.unit.onyx.asv3.test_native_metadata_projection import setup_original
from tests.unit.onyx.asv3.test_native_model_adapter import (
    last_payload,
    model,
    turn,
    view,
)
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
    actual = last_payload(selected)["original_evidence"][0]
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
    ref = payloads[0]["original_evidence_refs"][0]
    if tuned:
        assert "metadata" not in ref
        assert ref["metadata_ref"] == {"citation": 1, "text_hash": item.text_hash}
        assert "canonical_metadata" not in actual["metadata"]
        assert actual["metadata"]["title"] == "Verified instrument"
    else:
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
def test_preliminary_judicial_original_cannot_close_an_operative_assessment(
    status: str,
) -> None:
    context, ledger, reviews = tuned_context()
    seen(context, ledger, reviews)
    text = "**Karar Tarihi: 01/01/2026**\nİTİRAZIN KONUSU: The applicant requests annulment."
    number = ledger.add(
        [EvidenceItem(source_id="decision", text=text, chunk_id="application")], context
    )[0]
    deliver(ledger, "answer-call", [1, number])
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
