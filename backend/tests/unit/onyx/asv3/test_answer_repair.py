"""Actual model-adapter patches cannot change unrelated blocks or invent sources."""

import json

import pytest

from onyx.asv3.answer_repair import repair_publication_candidate
from onyx.asv3.assertions import assertion_inventory
from onyx.asv3.llm_adapter import ResearchModel
from onyx.asv3.models import OutcomeStatus, RunStopped, SharedBudget, ToolOutcome
from onyx.asv3.runtime import _evidence_record
from tests.unit.onyx.asv3.test_citation_contract import original_ledger
from tests.unit.onyx.asv3.test_model_adapter import scripted_model, text_response


@pytest.mark.parametrize(
    "bad_patch", ["other_unit", "repeat_attribution", "unknown_original", "valid"]
)
def test_localized_patch_preserves_untouched_detail_and_rejects_bad_replacements(
    bad_patch: str,
) -> None:
    ledger, context = original_ledger()
    rejected = "8917 sayılı Kanun uyarınca ilk şart gerekir [1]."
    tail = "İkinci sonuç: şart VE istisna; sonraki işlem de korunur [2]."
    answer = rejected + "\n\n" + tail
    units = assertion_inventory(answer)
    replacement = "İlk şart gerekir [1]."
    unit_id = units[0]["unit_id"]
    if bad_patch == "other_unit":
        unit_id = units[1]["unit_id"]
    elif bad_patch == "repeat_attribution":
        replacement = rejected
    elif bad_patch == "unknown_original":
        replacement = "İlk şart gerekir [999]."
    llm = scripted_model()
    llm.invoke.return_value = text_response(
        {"replacements": [{"unit_id": unit_id, "replacement": replacement}]}
    )
    candidate = repair_publication_candidate(
        ResearchModel(llm, context),
        ledger,
        answer=answer,
        scenario="facts",
        gap=ToolOutcome(
            status=OutcomeStatus.PARTIAL,
            summary="Missing operative original",
            data={"missing": ["8917"]},
        ),
        evidence=_evidence_record(ledger, answer),
        research_state={},
    )
    if bad_patch == "valid":
        assert candidate == replacement + "\n\n" + tail
        assert llm.invoke.call_count == 1
        content = llm.invoke.call_args.kwargs["prompt"][-1].content
        assert isinstance(content, str)
        payload = json.loads(content)
        assert payload["target_unit_ids"] == [units[0]["unit_id"]]
    else:
        assert candidate == answer and llm.invoke.call_count == 2


def test_targeted_research_cannot_spend_full_publication_capacity() -> None:
    budget = SharedBudget()
    for _ in range(33):
        budget.consume_repair_decision()
    with pytest.raises(RunStopped, match="publication reserve"):
        budget.consume_repair_decision()
    for _ in range(8):
        budget.consume("decisions")
    assert budget.snapshot()["decisions"] == 41


@pytest.mark.parametrize(
    "bad_patch", [None, "omit_requirement", "foreign_unit", "missing_citation"]
)
def test_late_witnessed_omission_is_inserted_without_regenerating_supported_answer(
    bad_patch: str | None,
) -> None:
    ledger, context = original_ledger()
    answer = "İlk destekli sonuç [1].\n\nİkinci destekli sonuç ve sonraki aşama [2]."
    units = assertion_inventory(answer)
    original = ledger.get(1)
    assert original is not None
    addition = "Eksik şart ve istisnası da korunur [1]."
    insertions = [
        {
            "after_unit_id": units[0]["unit_id"],
            "omission_ids": ["om0"],
            "text": addition,
        }
    ]
    if bad_patch == "omit_requirement":
        insertions = []
    elif bad_patch == "foreign_unit":
        insertions[0]["after_unit_id"] = "foreign"
    elif bad_patch == "missing_citation":
        insertions[0]["text"] = "Eksik şartı uygular."
    llm = scripted_model()
    llm.invoke.return_value = text_response(
        {"replacements": [], "insertions": insertions}
    )
    candidate = repair_publication_candidate(
        ResearchModel(llm, context),
        ledger,
        answer=answer,
        scenario="facts",
        gap=ToolOutcome(
            status=OutcomeStatus.PARTIAL,
            summary="Late original condition omitted",
            data={
                "omitted_material_source_details": [
                    {
                        "witness": {"citation": 1, "source_quote": original.text},
                        "determination_ids": ["q0:d0"],
                        "detail": "Eksik şart ve istisna",
                        "applicability": "İlk sonuç için",
                    }
                ]
            },
        ),
        evidence=_evidence_record(ledger, answer),
        research_state={},
    )
    if bad_patch is None:
        assert (
            candidate
            == units[0]["text"] + "\n\n" + addition + "\n\n" + units[1]["text"]
        )
        assert llm.invoke.call_count == 1
    else:
        assert candidate == answer
        assert llm.invoke.call_count == 2
