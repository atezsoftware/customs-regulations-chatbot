"""Fence the separately selectable copy before entering the tuned engine."""

from typing import TypedDict

import pytest
from pydantic import JsonValue, ValidationError

from onyx.asv3.workflow_variant import (
    ASV3_EXP_CHEAP_VARIANT,
    ASV3_EXP_CROPPED_POLICY,
    ASV3_EXP_CROPPED_VARIANT,
    ASV3_TUNED_VARIANT,
    checkpoint_variant_fields,
    resolve_asv3_workflow,
    validate_asv3_variant_resume,
)
from onyx.db.asv3_runs import decode_asv3_checkpoint, encode_asv3_checkpoint
from onyx.server.query_and_chat.models import SendMessageRequest


class _ResolverOptions(TypedDict, total=False):
    research_profile: str
    parallel_research: bool
    exp_cropped: bool
    exp_cheap: bool
    guarded_experimental: bool
    guardrails_v2: bool
    guardrails_v3: bool


def test_cropped_selection_has_its_own_identity_without_cheap() -> None:
    request = SendMessageRequest.model_validate(
        {
            "message": "Question",
            "atez_search_v3": True,
            "asv3_research_profile": "normal",
            "asv3_exp_cropped": True,
        }
    )
    assert request.asv3_exp_cropped is True
    assert "asv3_exp_cheap" not in request.model_dump()
    selection = resolve_asv3_workflow("normal", False, exp_cropped=True)
    assert selection.workflow_variant == ASV3_EXP_CROPPED_VARIANT
    assert selection.selected_model_only is True
    assert selection.parallel_research is False
    assert selection.research_profile == "normal"
    assert (
        resolve_asv3_workflow("experimental", True).workflow_variant
        == ASV3_TUNED_VARIANT
    )
    assert (
        SendMessageRequest.model_validate({"message": "Question"}).asv3_exp_cropped
        is False
    )


@pytest.mark.parametrize(
    "change",
    [
        {"atez_search_v3": False},
        {"asv3_research_profile": "deep"},
        {"asv3_research_profile": "experimental"},
        {"asv3_parallel_research": True},
        {"asv3_guarded_experimental": True},
        {"asv3_guardrails_v2": True},
        {"asv3_guardrails_v3": True},
        {"legal_review": True},
        {"legal_composite": True},
        {"supersearch": True},
        {"deep_research": True},
        {"atez_search": True},
        {"atez_search_v2": True},
        {
            "llm_overrides": [
                {"model_provider_type": "openai", "model_version": "selected"},
                {"model_provider_type": "openai", "model_version": "another"},
            ]
        },
    ],
)
def test_cropped_request_rejects_other_workflows(change: dict[str, JsonValue]) -> None:
    with pytest.raises(ValidationError):
        SendMessageRequest.model_validate(
            {
                "message": "Question",
                "atez_search_v3": True,
                "asv3_research_profile": "normal",
                "asv3_exp_cropped": True,
                **change,
            }
        )


@pytest.mark.parametrize(
    "options",
    [
        {"research_profile": "deep"},
        {"research_profile": "experimental"},
        {"parallel_research": True},
        {"exp_cheap": True},
        {"guarded_experimental": True},
        {"guardrails_v2": True},
        {"guardrails_v3": True},
    ],
)
def test_cropped_resolver_fails_closed(options: _ResolverOptions) -> None:
    selection: _ResolverOptions = {
        "research_profile": "normal",
        "parallel_research": False,
        "exp_cropped": True,
        **options,
    }
    with pytest.raises(ValueError):
        resolve_asv3_workflow(**selection)


def test_cropped_checkpoint_roundtrip_retains_own_policy_and_resume_fence() -> None:
    snapshot: dict[str, JsonValue] = {
        "run_id": "test-cropped-run",
        "sequence": 1,
        "research_profile": "normal",
        "parallel_research": False,
        **checkpoint_variant_fields(ASV3_EXP_CROPPED_VARIANT),
    }
    restored = decode_asv3_checkpoint(encode_asv3_checkpoint(snapshot))
    assert restored == {**snapshot, "version": 1}
    assert restored["asv3_workflow_policy"] == ASV3_EXP_CROPPED_POLICY
    validate_asv3_variant_resume(ASV3_EXP_CROPPED_VARIANT, restored)
    for other in ("standard", ASV3_TUNED_VARIANT, ASV3_EXP_CHEAP_VARIANT):
        with pytest.raises(ValueError, match="same workflow variant"):
            validate_asv3_variant_resume(other, restored)
    with pytest.raises(ValueError, match="own checkpoint policy"):
        validate_asv3_variant_resume(
            ASV3_EXP_CROPPED_VARIANT, {**restored, "asv3_workflow_policy": "other"}
        )
