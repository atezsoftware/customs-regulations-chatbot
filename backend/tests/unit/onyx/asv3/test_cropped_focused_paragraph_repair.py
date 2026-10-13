"""Focused repair retains the precise paragraph and its actual original receipt."""

import json
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock
from uuid import UUID

import pytest

from onyx.asv3 import cropped_focused_repair
from onyx.asv3.corpus_tools import build_corpus_specs, evidence_for_chunk
from onyx.asv3.cropped_article_bindings import CroppedArticleBindings
from onyx.asv3.cropped_focused_repair import CroppedFocusedRepair
from onyx.asv3.cropped_repair_state import CroppedRepairState, CroppedRepairTarget
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.harness import Harness
from onyx.asv3.llm_adapter import ResearchModel
from onyx.asv3.models import Decision, OutcomeStatus, RunContext
from onyx.asv3.registry import CapabilityRegistry
from onyx.db.asv3_corpus import CorpusChunk, CorpusSource
from tests.unit.onyx.asv3.test_corpus_source_sandbox import MemoryBroker

_CAPTURE = json.loads(
    (
        Path(__file__).parent / "fixtures" / "cropped_focused_5172_paragraphs.json"
    ).read_text()
)
_REQUEST = "5326 sayılı Kabahatler Kanunu m.15/3 önceliği nedir?"


def _setup(
    *, holey: bool = False
) -> tuple[Harness, CroppedFocusedRepair, CroppedRepairTarget]:
    rows = [_CAPTURE["rows"][0], _CAPTURE["rows"][2]] if holey else _CAPTURE["rows"]
    chunks = [
        CorpusChunk(
            id=row["chunk_id"],
            source_id=UUID(row["source_id"]),
            position=row["position"],
            text=row["text"],
            projection_ordinal=row["position"],
            heading_path=tuple(row["metadata"]["heading_path"]),
            metadata=row["metadata"],
            validity_start=None,
            validity_end=None,
            status="active",
        )
        for row in rows
    ]
    broker = MemoryBroker(chunks)
    broker.item = CorpusSource(
        chunks[0].source_id,
        rows[0]["metadata"]["title"],
        "\n\n".join(chunk.text for chunk in chunks),
    )
    context = RunContext(
        run_id="focused-paragraph-run",
        scope={"corpus": 15},
        services={"asv3_exp_cropped": True},
    )
    context.services["cropped_article_bindings"] = CroppedArticleBindings(
        context, _REQUEST
    )
    ledger = EvidenceLedger()
    ledger.add((evidence_for_chunk(broker.item, chunk) for chunk in chunks), context)
    registry = CapabilityRegistry(build_corpus_specs(broker))
    harness = Harness(
        registry=registry,
        context=context,
        request=_REQUEST,
        decide=lambda _view: Decision(),
        evidence=ledger,
    )
    paragraph_number = 2 if holey else 3
    target = CroppedRepairTarget(
        check_kind="governing_original",
        instrument_number="5326",
        article="15",
        evidence_numbers=[1],
        reference_text="5326 sayılı Kabahatler Kanunu m.15/3",
        details={
            "gap_entries": [
                {
                    "field": "named_authority_gaps",
                    "entry": {
                        "article": "15",
                        "paragraph": "3",
                        "inline_evidence": [paragraph_number],
                        "matching_original_evidence": [],
                    },
                }
            ]
        },
    )
    state = CroppedRepairState(context, _REQUEST)
    return harness, CroppedFocusedRepair(state), target


class _Checker:
    def __init__(self, ledger: EvidenceLedger, witness: int) -> None:
        self.ledger = ledger
        self.witness = witness
        self.last_call_id = "actual-scripted-checker"
        self.payload: dict[str, Any] = {}

    def invoke_text(self, _system: str, user: str, _flow: object, **kwargs: Any) -> str:
        self.payload = json.loads(user)
        # This stands in for the actual physical packet receipt, including literal text.
        self.ledger.record_delivery(
            self.last_call_id, "asv3_targeted_evidence_check", self.payload["evidence"]
        )
        item = self.ledger.get(self.witness)
        assert item is not None
        output = json.dumps(
            {
                "status": "found",
                "finding": item.text,
                "witnesses": [{"citation": self.witness, "source_quote": item.text}],
                "limitations": ["No whole-article or event-date validity approval."],
            }
        )
        kwargs["response_validator"](output)
        return output


def test_nonempty_article_candidate_does_not_skip_missing_paragraph_context() -> None:
    harness, repair, target = _setup()
    assert repair._governing_gap(target, harness, [1, 2, 3]) is not None
    numbers, observations = repair._acquire(target, harness, MagicMock())
    assert numbers == [1, 2, 3]
    assert len(observations) == 1
    assert observations[0]["name"] == "read_chunk_context"
    assert observations[0]["arguments"] == {
        "source_id": _CAPTURE["rows"][2]["source_id"],
        "chunk_id": _CAPTURE["rows"][2]["chunk_id"],
    }
    assert repair._governing_gap(target, harness, [3]) is None
    assert (
        len(harness.context.services["cropped_article_bindings"].export()["reads"]) == 1
    )


def test_existing_actual_binding_reuses_originals_without_another_source_read() -> None:
    harness, repair, target = _setup()
    repair._acquire(target, harness, MagicMock())
    numbers, observations = repair._acquire(target, harness, MagicMock())
    assert numbers == [1, 2, 3]
    assert observations == []
    assert len(harness.receipts) == 1


def test_checker_receives_original_paragraph_and_validates_its_actual_delivery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness, repair, target = _setup()
    numbers, observations = repair._acquire(target, harness, MagicMock())
    checker = _Checker(harness.evidence, 3)
    monkeypatch.setattr(cropped_focused_repair, "fresh_model", lambda _model: checker)
    assert repair.state.reserve(target)
    repair._check(
        target, harness, cast(ResearchModel, MagicMock()), numbers, observations
    )
    assert [row["citation"] for row in checker.payload["evidence"]] == [1, 2, 3]
    assert checker.payload["evidence"][2]["text"] == _CAPTURE["rows"][2]["text"]
    assert "answer" not in checker.payload and "conversation" not in checker.payload
    assert repair.state.records_for([target])[0]["status"] == "found"
    assert harness.evidence.completely_delivered(checker.last_call_id) == {1, 2, 3}
    assert not repair.state.reserve(target)


def test_article_first_paragraph_cannot_verify_requested_third_paragraph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness, repair, target = _setup()
    numbers, observations = repair._acquire(target, harness, MagicMock())
    checker = _Checker(harness.evidence, 1)
    monkeypatch.setattr(cropped_focused_repair, "fresh_model", lambda _model: checker)
    assert repair.state.reserve(target)
    with pytest.raises(ValueError, match="own matching original"):
        repair._check(
            target, harness, cast(ResearchModel, MagicMock()), numbers, observations
        )
    assert repair.state.records_for([target])[0]["status"] == "in_progress"


def test_holey_context_does_not_promote_unbound_paragraph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness, repair, target = _setup(holey=True)
    numbers, observations = repair._acquire(target, harness, MagicMock())
    assert len(observations) == 1
    assert repair._governing_gap(target, harness, [2]) is not None
    checker = _Checker(harness.evidence, 2)
    monkeypatch.setattr(cropped_focused_repair, "fresh_model", lambda _model: checker)
    assert repair.state.reserve(target)
    with pytest.raises(ValueError, match="own matching original"):
        repair._check(
            target, harness, cast(ResearchModel, MagicMock()), [1, 2], observations
        )


def test_bare_law_target_does_not_invent_none_article() -> None:
    harness, repair, _target = _setup()
    target = CroppedRepairTarget(
        check_kind="governing_original", instrument_number="5326", evidence_numbers=[1]
    )
    assert repair._governing_gap(target, harness, [1]) is None
    assert repair._governing_gap(target, harness, []) is not None


@pytest.mark.parametrize(
    "identity", [{"instrument_number": "4458"}, {"qualifier": "GEÇİCİ"}]
)
def test_bound_paragraph_cannot_satisfy_another_law_or_article_qualifier(
    identity: dict[str, str],
) -> None:
    harness, repair, target = _setup()
    repair._acquire(target, harness, MagicMock())
    different = target.model_copy(update=identity)
    assert repair._governing_numbers(different, harness) == []
    assert repair._governing_gap(different, harness, [3]) is not None


def test_bound_identity_without_actual_checker_delivery_is_not_verified(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness, repair, target = _setup()
    numbers, observations = repair._acquire(target, harness, MagicMock())
    checker = _Checker(harness.evidence, 3)
    monkeypatch.setattr(cropped_focused_repair, "fresh_model", lambda _model: checker)
    monkeypatch.setattr(harness.evidence, "record_delivery", lambda *_args: None)
    assert repair.state.reserve(target)
    with pytest.raises(ValueError, match="actual delivered original witness"):
        repair._check(
            target, harness, cast(ResearchModel, MagicMock()), numbers, observations
        )
    assert repair.state.records_for([target])[0]["status"] == "in_progress"


def test_missing_optional_gap_lists_and_inline_rows_are_safe() -> None:
    harness, repair, _target = _setup()
    target = CroppedRepairTarget(
        check_kind="governing_original",
        instrument_number="5326",
        article="15",
        details={"gap_entries": None},
    )
    assert repair._paragraphs(target) == []
    assert repair._paragraph_context_call(target, harness) is None
    target = target.model_copy(
        update={
            "details": {
                "gap_entries": [{"entry": {"paragraph": "3", "inline_evidence": None}}]
            }
        }
    )
    assert repair._paragraph_context_call(target, harness) is None


def test_missing_literal_paragraph_keeps_precise_canonical_read_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness, repair, target = _setup()
    target = target.model_copy(
        update={"details": {"gap_entries": [{"entry": {"paragraph": "4"}}]}}
    )
    numbers, observations = repair._acquire(target, harness, MagicMock())
    assert numbers == [1, 2, 3]
    assert observations[0]["name"] == "read_provision"
    arguments = observations[0]["arguments"]
    assert isinstance(arguments, dict) and arguments["paragraph"] == "4"
    assert harness.receipts[-1].outcome.status == OutcomeStatus.PARTIAL
    assert harness.receipts[-1].outcome.data["subunit_verified"] is False
    assert repair._governing_gap(target, harness, [1, 2, 3]) is not None
    checker = _Checker(harness.evidence, 3)
    monkeypatch.setattr(cropped_focused_repair, "fresh_model", lambda _model: checker)
    assert repair.state.reserve(target)
    with pytest.raises(ValueError, match="own matching original"):
        repair._check(
            target, harness, cast(ResearchModel, MagicMock()), numbers, observations
        )
    assert repair.state.records_for([target])[0]["status"] == "in_progress"
