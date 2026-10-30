"""Fast response priorities preserve operative originals and canonical tools."""

import pytest

from onyx.asv3.llm_adapter import ResearchModel
from onyx.asv3.models import RunContext, SharedBudget
from onyx.asv3.workflow_variant import (
    ASV3_FAST_CROPPED_VARIANT,
    ASV3_FAST_VARIANT,
    validate_asv3_variant_resume,
)
from onyx.prompts.asv3.fast import FAST_MATERIAL_OUTCOMES, FAST_TEK_MATERIAL_CONDITIONS
from tests.unit.onyx.asv3.test_native_model_adapter import adaptive_tool_view, model
from tests.unit.onyx.asv3.test_shared_originals import full_record, original, recorded


@pytest.mark.parametrize(
    "draft", [None, "Both conditions apply subject to exception [1]."]
)
def test_response_policy_preserves_originals_tools_and_output_capacity(
    draft: str | None,
) -> None:
    ledger = recorded(
        [
            original(
                "clause", "Rule requires A AND B; exception C and later procedure D."
            )
        ]
    )
    context = RunContext(
        budget=SharedBudget(unlimited_execution=True),
        services={"research_profile": "normal", "evidence": ledger},
    )
    current = adaptive_tool_view(
        original_evidence=[full_record(ledger, 1)], draft_to_repair=draft
    )
    selected = model()
    original_adapter = ResearchModel(selected, context, lean_native_mode=True)
    before = original_adapter._fit_native_decision(current)
    context.services.update(asv3_fast=True, asv3_fast_retrieval=True)
    fast_adapter = ResearchModel(selected, context, lean_native_mode=True)
    after = fast_adapter._fit_native_decision(current)
    prompt, tools, output = after
    restored_prompt = [
        message.model_copy(
            update={
                "content": message.content.replace(
                    "\n\n" + FAST_MATERIAL_OUTCOMES, ""
                ).replace("\n\n" + FAST_TEK_MATERIAL_CONDITIONS, "")
            }
        )
        if isinstance(message.content, str)
        else message
        for message in prompt
    ]
    assert (restored_prompt, tools, output) == before
    assert ledger.get(1) is not None
    assert "finish_research" not in str(after[1])


@pytest.mark.parametrize(
    "variant,policy",
    [
        (ASV3_FAST_VARIANT, "standard-fast-handoff-v1"),
        (ASV3_FAST_CROPPED_VARIANT, "cropped-fast-handoff-v1"),
        (ASV3_FAST_VARIANT, "standard-fast-batched-io-v2"),
        (ASV3_FAST_CROPPED_VARIANT, "cropped-fast-batched-io-v2"),
        (ASV3_FAST_VARIANT, "standard-fast-primary-material-outcomes-v3"),
    ],
)
def test_previous_fast_protocol_cannot_resume_as_material_outcomes_policy(
    variant: str, policy: str
) -> None:
    with pytest.raises(ValueError, match="own checkpoint policy"):
        validate_asv3_variant_resume(
            variant,
            {
                "asv3_workflow_variant": variant,
                "asv3_workflow_policy": policy,
                "research_profile": "normal",
                "parallel_research": False,
            },
        )
