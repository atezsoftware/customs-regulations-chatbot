"""An amending law retains its own identity without taking the affected article."""

import json
import re
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from pydantic import JsonValue

from onyx.asv3 import cropped_focused_repair
from onyx.asv3.assertions import assertion_inventory
from onyx.asv3.authority import (
    _named_native_references,
    explicit_reference_leads,
    native_named_statute_references,
    statute_references,
)
from onyx.asv3.cheap_authority import canonical_named_authority_gap
from onyx.asv3.cropped_focused_repair import CroppedFocusedRepair, source_gap_sentence
from onyx.asv3.cropped_repair_state import CroppedRepairState, extract_repair_targets
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.harness import Harness
from onyx.asv3.llm_adapter import ResearchModel
from onyx.asv3.models import (
    Decision,
    EvidenceItem,
    OutcomeStatus,
    RunContext,
    ToolOutcome,
    ToolSpec,
)
from onyx.asv3.registry import CapabilityRegistry
from tests.unit.onyx.asv3.test_native_authority import ledger_with, original

_CAPTURE = json.loads(
    (
        Path(__file__).parent / "fixtures" / "cropped_amendment_identity_7448.json"
    ).read_text()
)


def captured() -> tuple[EvidenceLedger, RunContext, dict[int, int], str]:
    ledger = EvidenceLedger()
    context = RunContext(services={"asv3_exp_cropped": True})
    numbers = ledger.add(
        [EvidenceItem.model_validate(row["item"]) for row in _CAPTURE["records"]],
        context,
    )
    mapping = {
        row["citation"]: number
        for row, number in zip(_CAPTURE["records"], numbers, strict=True)
    }
    draft = re.sub(
        r"\[(\d+)\]", lambda match: f"[{mapping[int(match[1])]}]", _CAPTURE["unit"]
    )
    return ledger, context, mapping, draft


@pytest.mark.parametrize("syntactic", [False, True])
@pytest.mark.parametrize(
    "instrument",
    ["Kanunla", "Kanunuyla", "Kanun ile", "Kanun'la", "Kanun’la"],
)
def test_instrumental_law_does_not_own_the_following_affected_article(
    instrument: str, syntactic: bool
) -> None:
    text = f"6455 sayılı {instrument} m.234 değişikliği yapılmıştır."
    references = statute_references(
        text,
        strict_reference_boundaries=True,
        syntactic_reference_binding=syntactic,
    )
    assert [(reference.number, reference.article) for reference in references] == [
        ("6455", None)
    ]


@pytest.mark.parametrize(
    "text",
    [
        "6455 sayılı Kanun m.11 uygulanır.",
        "6455 sayılı Kanunun 11 inci maddesiyle 4458 sayılı Kanunun 234 üncü maddesi değişmiştir.",
        "6455 sayılı Kanun’un 11. maddesi ile 4458 sayılı Kanun m.234 değiştirilmiştir.",
    ],
)
@pytest.mark.parametrize("syntactic", [False, True])
def test_explicit_own_amending_provision_remains_its_own_reference(
    text: str, syntactic: bool
) -> None:
    references = statute_references(
        text,
        strict_reference_boundaries=True,
        syntactic_reference_binding=syntactic,
        unicode_ordinals=True,
    )
    identities = {(reference.number, reference.article) for reference in references}
    assert ("6455", "11") in identities
    assert ("6455", "234") not in identities
    if "4458" in text:
        assert ("4458", "234") in identities


@pytest.mark.parametrize("syntactic", [False, True])
@pytest.mark.parametrize("named_alias", [False, True])
@pytest.mark.parametrize(
    ("locator", "article", "qualifier"),
    [
        ("11/A", "11/A", None),
        ("ek 11/A", "11/A", "ek"),
        ("geçici 11/A", "11/A", "gecici"),
        ("geçici 11", "11", "gecici"),
        ("mükerrer 11/A", "11/A", "mukerrer"),
    ],
)
def test_amending_own_inserted_article_and_qualifier_survive_both_parse_paths(
    syntactic: bool,
    named_alias: bool,
    locator: str,
    article: str,
    qualifier: str | None,
) -> None:
    identity = "Faaliyet Kanununun" if named_alias else "6455 sayılı Kanunun"
    text = f"{identity} {locator} maddesiyle 4458 sayılı Kanunun 234 üncü maddesi değişmiştir."
    options = {
        "strict_reference_boundaries": True,
        "syntactic_reference_binding": syntactic,
        "unicode_ordinals": True,
    }
    references = (
        [
            reference
            for reference, _name in _named_native_references(
                text, {"faaliyet kanun": {"6455"}}, **options
            )
        ]
        if named_alias
        else statute_references(text, **options)
    )
    assert {
        (reference.article, reference.qualifier)
        for reference in references
        if reference.number == "6455"
    } == {(article, qualifier)}
    assert ("4458", "234", None) in {
        (reference.number, reference.article, reference.qualifier)
        for reference in references
    }


def test_named_instrumental_alias_also_keeps_the_article_owner_open() -> None:
    ledger = ledger_with(original("7777 SAYILI FAALİYET KANUNU", "11"))
    references = native_named_statute_references(
        "Faaliyet Kanunuyla m.234 değiştirilmiştir.", ledger
    )
    assert {
        (reference["instrument_number"], reference["article"])
        for reference in references
    } == {("7777", None)}


def test_captured_amendment_table_keeps_the_instrument_need_without_inventing_an_article() -> (
    None
):
    ledger, _context, mapping, draft = captured()
    before = ledger.export()
    gap = canonical_named_authority_gap(
        draft,
        ledger,
        strict_reference_boundaries=True,
        syntactic_reference_binding=True,
    )
    assert gap is not None
    assert gap["named_authority_gaps"] == [
        {
            "unit_id": assertion_inventory(draft)[0]["unit_id"],
            "instrument_number": "6455",
            "reference_text": "6455 sayılı Kanunla",
            "article": None,
            "inline_evidence": sorted([mapping[310], mapping[344]]),
            "matching_original_evidence": [],
        }
    ]
    targets = extract_repair_targets(gap, draft, ledger)
    assert len(targets) == 1
    target = targets[0]
    assert (target.instrument_number, target.article) == ("6455", None)
    assert target.check_kind == "governing_original"
    assert "234" not in source_gap_sentence(target, "not_found")
    assert "11" not in source_gap_sentence(target, "not_found")
    assert ledger.export() == before


@pytest.mark.parametrize("old_citation", [336, 338])
def test_captured_explicit_article_eleven_remains_navigation_not_governing_proof(
    old_citation: int,
) -> None:
    ledger, _context, mapping, _draft = captured()
    number = mapping[old_citation]
    item = ledger.get(number)
    assert item is not None
    record: dict[str, JsonValue] = {
        "citation": number,
        "source_id": item.source_id,
        "chunk_id": item.chunk_id,
        "text_hash": item.text_hash,
        "text": item.text,
    }
    leads = explicit_reference_leads(ledger, [record], syntactic_reference_binding=True)
    assert any(
        lead["instrument_number"] == "6455" and lead["article"] == "11"
        for lead in leads
    )
    assert not any(
        lead["instrument_number"] == "6455" and lead["article"] == "234"
        for lead in leads
    )
    assert canonical_named_authority_gap(
        f"6455 sayılı Kanun m.11 uygulanır [{number}].",
        ledger,
        strict_reference_boundaries=True,
        syntactic_reference_binding=True,
    )


def test_article_unknown_repair_can_resolve_the_instrument_without_a_false_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger, context, _mapping, draft = captured()
    gap = canonical_named_authority_gap(
        draft,
        ledger,
        strict_reference_boundaries=True,
        syntactic_reference_binding=True,
    )
    assert gap is not None
    target = extract_repair_targets(gap, draft, ledger)[0]
    lookups: list[dict[str, JsonValue]] = []

    def lookup(arguments: dict[str, JsonValue], run: RunContext) -> ToolOutcome:
        run.check_active()
        lookups.append(arguments)
        return ToolOutcome(
            status=OutcomeStatus.NOT_FOUND,
            summary="No canonical original source identity was resolved.",
            data={"sources": [], "absence_proven": False},
        )

    registry = CapabilityRegistry(
        [
            ToolSpec(
                name="resolve_source",
                description="Resolve the exact canonical source identity.",
                parameters={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                },
                handler=lookup,
            )
        ]
    )
    harness = Harness(
        registry=registry,
        context=context,
        request="Değiştirici kanunun özgün metnini kontrol et.",
        decide=lambda _view: Decision(),
        evidence=ledger,
    )
    repair = CroppedFocusedRepair(CroppedRepairState(context, harness.request))
    planner = MagicMock()

    def plan(_system: str, payload: str, _flow: object, **_kwargs: Any) -> str:
        deficiency = json.loads(payload)["deficiency"]
        assert deficiency["instrument_number"] == "6455"
        assert deficiency["article"] is None
        return json.dumps(
            {
                "actions": [
                    {
                        "name": "resolve_source",
                        "arguments": {"query": "6455 sayılı Kanun"},
                    }
                ],
                "reason": "Resolve the amending instrument before selecting its own provision.",
            }
        )

    planner.invoke_text.side_effect = plan
    monkeypatch.setattr(cropped_focused_repair, "fresh_model", lambda _model: planner)
    assert repair.state.reserve(target)
    numbers, observations = repair._acquire(
        target, harness, cast(ResearchModel, MagicMock())
    )
    assert numbers == []
    assert lookups == [{"query": "6455 sayılı Kanun"}]
    assert [observation["name"] for observation in observations] == ["resolve_source"]
    assert all("article" not in receipt.call.arguments for receipt in harness.receipts)
