"""Exercise source-first delivery, immutable retention and candidate-independent reuse."""

import pytest

from onyx.asv3.assertions import AssertionWitness
from onyx.asv3.condition_memory import SourceConditionMemory
from onyx.asv3.llm_adapter import (
    MaterialSourceOmission,
    ResearchModel,
    StructuredOutputError,
)
from onyx.asv3.publication import publication_gap
from onyx.asv3.runtime import _evidence_record
from onyx.asv3.source_conditions import (
    complete_condition_review,
    condition_payload,
    retain_source_inventory,
)
from onyx.tracing.flows import LLMFlow
from tests.unit.onyx.asv3.test_citation_contract import original_ledger, supported
from tests.unit.onyx.asv3.test_model_adapter import scripted_model, text_response


def test_blind_inventory_retains_uncited_prerequisite_and_reuses_same_originals() -> (
    None
):
    ledger, context = original_ledger()
    question = "What procedure applies?"
    answer = "CANDIDATE MUST NOT REACH SOURCE INVENTORY [1]."
    evidence = _evidence_record(ledger, answer, include_supplemental_originals=True)
    payload, _ = condition_payload(answer, "facts", [question], evidence, "en")
    item = ledger.get(3)
    assert item is not None
    requirement = MaterialSourceOmission(
        witness=AssertionWitness(citation=3, source_quote=item.text),
        determination_ids=["q0:d0"],
        detail="Separate implementing prerequisite and its restricted scope.",
        applicability="The original question requires this procedure.",
    )
    llm = scripted_model()
    llm.invoke.return_value = text_response(
        {
            "examined_citations": [1, 2, 3],
            "requirements": [requirement.model_dump(mode="json")],
        }
    )
    model = ResearchModel(llm, context)
    memory = SourceConditionMemory(context.run_id, "scope", [question])
    receipt = retain_source_inventory(
        model, ledger, payload, [question], memory, consume_budget=True
    )
    serialized = str(llm.invoke.call_args.kwargs["prompt"])
    assert answer not in serialized and "answer_units" not in serialized
    assert ledger.delivery_flow(receipt) == LLMFlow.ASV3_SOURCE_INVENTORY.value
    assert ledger.completely_delivered(receipt) == {1, 2, 3}
    assert memory.citations() == [3] and len(memory.required_conditions()) == 1
    payload["answer_units"] = [{"unit_id": "changed-draft", "text": "Different answer"}]
    assert (
        retain_source_inventory(
            model, ledger, payload, [question], memory, consume_budget=True
        )
        == receipt
    )
    assert llm.invoke.call_count == 1
    payload["scenario"] = "A materially different scenario"
    retain_source_inventory(
        model, ledger, payload, [question], memory, consume_budget=True
    )
    assert llm.invoke.call_count == 2


@pytest.mark.parametrize("foreign_witness", [False, True])
def test_invalid_source_inventory_cannot_reach_answer_comparison(
    foreign_witness: bool,
) -> None:
    ledger, context = original_ledger()
    evidence = _evidence_record(
        ledger, "Rule [1].", include_supplemental_originals=True
    )
    payload, _ = condition_payload("Rule [1].", "facts", ["Question?"], evidence, "en")
    requirement = MaterialSourceOmission(
        witness=AssertionWitness(citation=3, source_quote="not in the source"),
        determination_ids=["q0:d0"],
        detail="Required condition",
        applicability="Relevant",
    )
    llm = scripted_model()
    llm.invoke.return_value = text_response(
        {
            "examined_citations": [1, 2, 3] if foreign_witness else [1],
            "requirements": [requirement.model_dump(mode="json")]
            if foreign_witness
            else [],
        }
    )
    memory = SourceConditionMemory(context.run_id, "scope", ["Question?"])
    with pytest.raises(StructuredOutputError):
        retain_source_inventory(
            ResearchModel(llm, context),
            ledger,
            payload,
            ["Question?"],
            memory,
            consume_budget=True,
        )
    assert llm.invoke.call_count == 2 and memory.required_conditions() == []


def test_answer_comparison_cannot_silently_drop_source_first_requirement() -> None:
    ledger, context = original_ledger()
    answer, question = "The operation is permitted [1].", "What procedure applies?"
    item = ledger.get(3)
    assert item is not None
    requirement = MaterialSourceOmission(
        witness=AssertionWitness(citation=3, source_quote=item.text),
        determination_ids=["q0:d0"],
        detail="The permission has a material prerequisite.",
        applicability="This requested operation is conditional.",
    )
    llm = scripted_model()
    llm.invoke.side_effect = [
        text_response(
            {
                "examined_citations": [1, 2, 3],
                "requirements": [requirement.model_dump(mode="json")],
            }
        ),
        text_response(
            {"examined_citations": [1, 2, 3], "conditions": [], "resolutions": []}
        ),
        text_response(
            {"examined_citations": [1, 2, 3], "conditions": [], "resolutions": []}
        ),
    ]
    model = ResearchModel(llm, context)
    model.last_call_id = "main-assertions"
    memory = SourceConditionMemory(context.run_id, "scope", [question])
    review = complete_condition_review(
        supported([1]),
        model,
        ledger,
        answer=answer,
        scenario="facts",
        questions=[question],
        evidence=_evidence_record(ledger, answer, include_supplemental_originals=True),
        language="en",
        consume_budget=True,
        condition_memory=memory,
    )
    assert not review.safe_to_publish and review.format_error
    assert review.source_inventory_call_id and len(memory.required_conditions()) == 1
    assert model.last_call_id == "main-assertions"
    prompts = [str(call.kwargs["prompt"]) for call in llm.invoke.call_args_list]
    assert answer not in prompts[0] and answer in prompts[1]
    assert "sc-" in prompts[1] and "Retained condition" in str(review.format_error)


def test_publication_requires_the_distinct_source_first_delivery_receipt() -> None:
    ledger, context = original_ledger()
    llm = scripted_model()
    llm.invoke.side_effect = [
        text_response({"examined_citations": [1, 2, 3], "requirements": []}),
        text_response({"examined_citations": [1, 2, 3], "conditions": []}),
    ]
    answer, question = "Definition [1].", "What is the definition?"
    result = complete_condition_review(
        supported([1]),
        ResearchModel(llm, context),
        ledger,
        answer=answer,
        scenario="facts",
        questions=[question],
        evidence=_evidence_record(ledger, answer, include_supplemental_originals=True),
        language="en",
        consume_budget=True,
    )
    assert (
        publication_gap(
            answer,
            result,
            [question],
            ledger,
            scenario="facts",
            require_condition_review=True,
            require_source_inventory=True,
        )
        is None
    )
    result.source_inventory_call_id = result.condition_review_call_id
    gap = publication_gap(
        answer,
        result,
        [question],
        ledger,
        scenario="facts",
        require_condition_review=True,
        require_source_inventory=True,
    )
    assert gap is not None and "source-first inventory" in str(gap.data)
