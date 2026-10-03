"""Provider and assessment regressions observed in multimodal transit research."""

import json

import pytest

from onyx.asv3.assertions import assertion_inventory, presentation_block
from onyx.asv3.llm_adapter import ResearchModel
from onyx.asv3.models import HarnessView, RunContext, RunStopped
from onyx.asv3.publication import publication_gap
from onyx.asv3.runtime import _evidence_record  # pyright: ignore[reportPrivateUsage]
from onyx.llm.model_response import Choice, Message, ModelResponse
from tests.unit.onyx.asv3.test_model_adapter import scripted_model, text_response
from tests.unit.onyx.asv3.test_safe_partial_conditions import partial_case


@pytest.mark.parametrize("provider", ["vertex_ai", "anthropic", "openai"])
def test_empty_coordinator_response_recovers_without_changing_selected_model(
    provider: str,
) -> None:
    llm = scripted_model()
    llm.config.model_provider = provider
    llm.invoke.side_effect = [
        ModelResponse(
            id="empty", created="0", choice=Choice(message=Message(content=""))
        ),
        ModelResponse(
            id="recovered",
            created="0",
            choice=Choice(message=Message(content="A retained candidate [1].")),
        ),
    ]
    context = RunContext()
    context.consume_research_decision()
    result = ResearchModel(llm, context).decide(
        HarnessView(
            request="Question",
            questions=["Question"],
            facts=[],
            receipts=[],
            evidence=[],
            tools=[],
        )
    )
    assert result.answer == "A retained candidate [1]."
    assert llm.invoke.call_count == 2
    assert context.budget.snapshot()["decisions"] == 2
    assert llm.config.model_provider == provider
    assert llm.config.model_name == "selected-model"


def test_repeated_empty_decision_stops_research_for_normal_finalization() -> None:
    llm = scripted_model()
    llm.invoke.return_value = ModelResponse(
        id="empty", created="0", choice=Choice(message=Message(content=""))
    )
    with pytest.raises(RunStopped, match="empty"):
        ResearchModel(llm, RunContext()).decide(
            HarnessView(
                request="Question",
                questions=["Question"],
                facts=[],
                receipts=[],
                evidence=[],
                tools=[],
            )
        )
    assert llm.invoke.call_count == 2


@pytest.mark.parametrize(
    "label",
    [
        "- **Tahsilat ve Kanıtlama Süreci:**",
        "* **Collection procedure:**",
        "1. **Documents:**",
    ],
)
def test_markdown_list_labels_are_presentation_and_not_merged_with_claims(
    label: str,
) -> None:
    assert presentation_block(label)
    units = assertion_inventory(label + "\n\n- A condition requires proof [1].")
    assert len(units) == 2
    assert units[0]["presentation_only"]
    assert not units[1]["presentation_only"]


@pytest.mark.parametrize(
    "text",
    [
        "- **Condition:** Release is unconditional.",
        "- **Condition:** [1]",
        "- Release is unconditional:",
    ],
)
def test_substantive_list_prose_still_requires_its_own_original(text: str) -> None:
    assert not presentation_block(text)


@pytest.mark.parametrize(
    "defect",
    ["phantom_witness", "supported_gap", "missing_gap", "supported_missing_question"],
)
def test_partial_review_bookkeeping_is_repaired_without_discarding_supported_answers(
    defect: str,
) -> None:
    ledger, context, state, answer, review, _condition = partial_case()
    entry = review.assertion_results[1].model_copy(deep=True)
    patched_question = []
    if defect == "phantom_witness":
        review.assertion_results[1].witnesses = [
            review.assertion_results[0].witnesses[0]
        ]
    elif defect == "supported_gap":
        review.assertion_results[1].status = "supported"
    elif defect == "missing_gap":
        review.assertion_results[1].missing_conditions = []
    else:
        review.question_results[1].status = "supported"
        review.question_results[1].determinations[0].status = "supported"
        fixed = review.question_results[1].model_copy(deep=True)
        fixed.status = "uncertain"
        fixed.determinations[0].status = "uncertain"
        patched_question = [fixed.model_dump(mode="json")]
    payload = json.dumps(
        {
            "claim": answer,
            "scenario": "Question",
            "assertion_units": assertion_inventory(answer),
            "evidence": _evidence_record(ledger, answer),
        }
    )
    patch = {
        "assertion_results": []
        if patched_question
        else [entry.model_dump(mode="json")],
        "question_results": patched_question,
        "need_results": [],
    }
    llm = scripted_model()
    llm.invoke.side_effect = [
        text_response(review.model_dump(mode="json")),
        text_response(patch),
    ]
    model = ResearchModel(llm, context)
    result = model.invoke_verification("Assess original support", payload)
    assert llm.invoke.call_count == 2
    assert result.format_error is None
    assert result.safe_to_publish and result.status == "incomplete"
    assert result.assertion_results[0] == review.assertion_results[0]
    assert result.assertion_results[1].status == "uncertain"
    assert not result.assertion_results[1].witnesses
    assert result.assertion_results[1].missing_conditions
    assert (
        publication_gap(
            answer,
            result,
            list(state.questions),
            ledger,
            allow_explicit_gaps=True,
            require_assertion_checks=True,
            require_determination_checks=True,
        )
        is None
    )


@pytest.mark.parametrize(
    "attempt", ["upgrade_gap", "erase_condition", "invent_witness"]
)
def test_gap_assessment_patch_cannot_turn_missing_law_into_approval(
    attempt: str,
) -> None:
    ledger, context, _state, answer, review, _condition = partial_case()
    review.assertion_results[1].witnesses = [review.assertion_results[0].witnesses[0]]
    entry = review.assertion_results[1].model_copy(deep=True)
    entry.witnesses = []
    if attempt == "upgrade_gap":
        entry.status = "supported"
        entry.basis = "original"
    elif attempt == "erase_condition":
        entry.missing_conditions = []
    else:
        entry.witnesses = [review.assertion_results[0].witnesses[0]]
    payload = json.dumps(
        {
            "claim": answer,
            "assertion_units": assertion_inventory(answer),
            "evidence": _evidence_record(ledger, answer),
        }
    )
    patch = {
        "assertion_results": [entry.model_dump(mode="json")],
        "question_results": [],
        "need_results": [],
    }
    llm = scripted_model()
    llm.invoke.side_effect = [
        text_response(review.model_dump(mode="json")),
        text_response(patch),
    ]
    result = ResearchModel(llm, context).invoke_verification("Assess", payload)
    assert result.format_error is not None and not result.safe_to_publish
    assert result.assertion_results == review.assertion_results
