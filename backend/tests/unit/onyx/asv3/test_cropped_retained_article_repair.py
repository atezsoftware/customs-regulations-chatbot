"""A retained article fragment cannot suppress its missing operative passage."""

import json
from typing import Any, cast
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from onyx.asv3 import cropped_focused_repair
from onyx.asv3.corpus_tools import build_corpus_specs, evidence_for_chunk
from onyx.asv3.cropped_focused_repair import CroppedFocusedRepair
from onyx.asv3.cropped_repair_state import CroppedRepairState, CroppedRepairTarget
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.harness import Harness
from onyx.asv3.llm_adapter import ResearchModel
from onyx.asv3.models import Decision, RunContext
from onyx.asv3.registry import CapabilityRegistry
from onyx.db.asv3_corpus import CorpusChunk, CorpusSource
from tests.unit.onyx.asv3.test_corpus_source_sandbox import MemoryBroker


def _setup(
    law: str, article: str, retained: str, operative: str, *, complete: bool = False
) -> tuple[Harness, CroppedFocusedRepair, CroppedRepairTarget, MemoryBroker]:
    source_id = uuid4()
    name = f"{law} sayılı Kanun"
    chunks = [
        CorpusChunk(
            id=f"chunk-{index}",
            source_id=source_id,
            position=index,
            projection_ordinal=index,
            text=text,
            heading_path=(name, f"MADDE {number}"),
            metadata={"article_no": number, "title": name, "document_type": "kanun"},
            validity_start=None,
            validity_end=None,
            status="active",
        )
        for index, (text, number) in enumerate(
            [(retained, article), (operative, article), ("Sonraki madde.", "999")]
        )
    ]
    broker = MemoryBroker(chunks)
    broker.item = CorpusSource(source_id, name, "\n\n".join(c.text for c in chunks))
    context = RunContext(
        run_id="retained-article-repair",
        scope={"corpus": 15},
        services={"asv3_exp_cropped": True},
    )
    ledger = EvidenceLedger()
    ledger.add(
        (evidence_for_chunk(broker.item, c) for c in chunks[: 2 if complete else 1]),
        context,
    )
    harness = Harness(
        registry=CapabilityRegistry(build_corpus_specs(broker)),
        context=context,
        request="Atıf yapılan hükmün özgün koşullarını kontrol et.",
        decide=lambda _view: Decision(),
        evidence=ledger,
    )
    target = CroppedRepairTarget(
        check_kind="governing_original",
        instrument_number=law,
        article=article,
        evidence_numbers=[1],
        details={
            "gap_entries": [
                {"field": "named_authority_gaps", "entry": {"article": article}}
            ]
        },
    )
    return (
        harness,
        CroppedFocusedRepair(CroppedRepairState(context, harness.request)),
        target,
        broker,
    )


class _Checker:
    def __init__(self, ledger: EvidenceLedger, operative: str) -> None:
        self.ledger = ledger
        self.operative = operative
        self.last_call_id = ""
        self.payloads: list[dict[str, Any]] = []

    def invoke_text(self, _system: str, user: str, _flow: object, **kwargs: Any) -> str:
        packet = json.loads(user)
        self.payloads.append(packet)
        self.last_call_id = f"retained-check-{len(self.payloads)}"
        self.ledger.record_delivery(
            self.last_call_id, "asv3_targeted_evidence_check", packet["evidence"]
        )
        operative = next(
            (r for r in packet["evidence"] if r["text"] == self.operative), None
        )
        result = json.dumps(
            {
                "status": "found" if operative else "incomplete",
                "finding": "Requested operative paragraph retained."
                if operative
                else "Only another paragraph of the same article is retained.",
                "witnesses": [
                    {
                        "citation": operative["citation"],
                        "source_quote": operative["text"],
                    }
                ]
                if operative
                else [],
                "limitations": [],
            }
        )
        kwargs["response_validator"](result)
        return result


@pytest.mark.parametrize(
    "law,article,retained,operative",
    [
        (
            "5326",
            "17",
            "MADDE 17 — (1) Para cezasının türü belirlenir.",
            "(6) Ödeme süresinde ödenen cezaya indirim uygulanır.",
        ),
        (
            "4458",
            "241",
            "MADDE 241 — 6. e) Geçici ithalat taşıtına ilişkin usulsüzlük.",
            "1. Usulsüzlük yaptırımının kapsamı ayrıca belirlenir.",
        ),
    ],
)
def test_missing_operative_paragraph_gets_one_read_within_reserved_attempt(
    monkeypatch: pytest.MonkeyPatch,
    law: str,
    article: str,
    retained: str,
    operative: str,
) -> None:
    harness, repair, target, broker = _setup(law, article, retained, operative)
    model = cast(ResearchModel, MagicMock())
    checker = _Checker(harness.evidence, operative)
    monkeypatch.setattr(cropped_focused_repair, "fresh_model", lambda _model: checker)
    assert repair._governing_gap(target, harness, [1]) is None
    assert repair.state.reserve(target)
    numbers, observations = repair._acquire(target, harness, model)
    assert observations == []

    repair._check(target, harness, model, numbers, observations)

    assert len(checker.payloads) == 2
    assert len(harness.receipts) == 1
    assert harness.receipts[0].call.name == "read_provision"
    assert harness.receipts[0].call.arguments == {
        "source_id": str(broker.item.id),
        "article": article,
    }
    record = repair.state.records_for([target])[0]
    assert record["status"] == "found"
    assert record["evidence_numbers"] == [2]
    assert harness.evidence.completely_delivered(checker.last_call_id) == {1, 2}
    assert not repair.state.reserve(target)


def test_complete_retained_originals_are_verified_without_another_source_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operative = "(6) Ödeme süresinde ödenen cezaya indirim uygulanır."
    harness, repair, target, _broker = _setup(
        "5326",
        "17",
        "MADDE 17 — (1) Para cezasının türü belirlenir.",
        operative,
        complete=True,
    )
    model = cast(ResearchModel, MagicMock())
    checker = _Checker(harness.evidence, operative)
    monkeypatch.setattr(cropped_focused_repair, "fresh_model", lambda _model: checker)
    assert repair.state.reserve(target)
    numbers, observations = repair._acquire(target, harness, model)

    repair._check(target, harness, model, numbers, observations)

    assert len(checker.payloads) == 1
    assert harness.receipts == []
    assert repair.state.records_for([target])[0]["status"] == "found"


def test_denied_operative_read_stays_incomplete_without_repeated_acquisition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operative = "(6) Ödeme süresinde ödenen cezaya indirim uygulanır."
    harness, repair, target, broker = _setup(
        "5326", "17", "MADDE 17 — (1) Para cezasının türü belirlenir.", operative
    )
    model = cast(ResearchModel, MagicMock())
    checker = _Checker(harness.evidence, operative)
    monkeypatch.setattr(cropped_focused_repair, "fresh_model", lambda _model: checker)

    def denied(_source_id: str, context: RunContext) -> CorpusSource:
        context.check_active()
        raise PermissionError("source is outside the current scope")

    monkeypatch.setattr(broker, "source", denied)
    assert repair.state.reserve(target)
    numbers, observations = repair._acquire(target, harness, model)

    repair._check(target, harness, model, numbers, observations)

    assert len(checker.payloads) == 2
    assert len(harness.receipts) == 1
    assert harness.evidence.citation_numbers() == (1,)
    assert repair.state.records_for([target])[0]["status"] == "incomplete"
    assert not repair.state.reserve(target)
