"""Resolve the legacy selection slot without changing established ASv3 profiles."""

from __future__ import annotations

from pydantic import JsonValue

from onyx.asv3.models import ASv3WorkflowSelection

ASV3_STANDARD_VARIANT = "standard"
ASV3_TUNED_VARIANT = "asv3_tuned"
ASV3_TUNED_POLICY = "normal-copy-v1"
ASV3_EXP_CHEAP_VARIANT = "asv3_exp_cheap"
ASV3_EXP_CHEAP_POLICY = "tuned-copy-metadata-source-repair-v4"
ASV3_EXP_CROPPED_VARIANT = "asv3_exp_cropped"
ASV3_EXP_CROPPED_POLICY = "tuned-open-topic-packets-v18"
ASV3_FAST_VARIANT = "asv3_fast"
ASV3_FAST_POLICY = "standard-fast-action-primary-sol-material-review-v6"
ASV3_FAST_CROPPED_VARIANT = "asv3_fast_cropped"
ASV3_FAST_CROPPED_POLICY = "cropped-fast-material-outcomes-v3"
ASV3_GUARDED_EXPERIMENTAL_VARIANT = "asv3_guarded_experimental"
ASV3_GUARDED_EXPERIMENTAL_POLICY = "guarded-normal-v1"
ASV3_GUARDRAILS_V2_VARIANT = "asv3_guardrails_v2"
ASV3_GUARDRAILS_V2_POLICY = "jev-gemini-review-v1"
ASV3_GUARDRAILS_V3_VARIANT = "asv3_guardrails_v3"
ASV3_GUARDRAILS_V3_POLICY = "issue-legal-review-v1"


def resolve_asv3_workflow(
    research_profile: str,
    parallel_research: bool,
    guarded_experimental: bool = False,
    guardrails_v2: bool = False,
    guardrails_v3: bool = False,
    exp_cheap: bool = False,
    exp_cropped: bool = False,
    fast: bool = False,
    fast_cropped: bool = False,
) -> ASv3WorkflowSelection:
    if fast or fast_cropped:
        if (
            sum(
                (
                    fast,
                    fast_cropped,
                    exp_cropped,
                    exp_cheap,
                    guarded_experimental,
                    guardrails_v2,
                    guardrails_v3,
                )
            )
            > 1
        ):
            raise ValueError("Fast ASv3 cannot be combined with other ASv3 variants")
        if research_profile != "normal" or parallel_research:
            raise ValueError("Fast ASv3 requires the normal non-parallel profile")
        return ASv3WorkflowSelection(
            research_profile="normal",
            parallel_research=False,
            workflow_variant=ASV3_FAST_CROPPED_VARIANT
            if fast_cropped
            else ASV3_FAST_VARIANT,
        )
    if exp_cropped:
        if exp_cheap or guarded_experimental or guardrails_v2 or guardrails_v3:
            raise ValueError("exp cropped cannot be combined with other ASv3 variants")
        if research_profile != "normal" or parallel_research:
            raise ValueError("exp cropped requires the normal non-parallel profile")
        return ASv3WorkflowSelection(
            research_profile="normal",
            parallel_research=False,
            workflow_variant=ASV3_EXP_CROPPED_VARIANT,
        )
    if exp_cheap:
        if guarded_experimental or guardrails_v2 or guardrails_v3:
            raise ValueError(
                "ASv3Exp Cheap cannot be combined with guardrails variants"
            )
        if research_profile != "normal" or parallel_research:
            raise ValueError("ASv3Exp Cheap requires the normal non-parallel profile")
        return ASv3WorkflowSelection(
            research_profile="normal",
            parallel_research=False,
            workflow_variant=ASV3_EXP_CHEAP_VARIANT,
        )
    if sum((guarded_experimental, guardrails_v2, guardrails_v3)) > 1:
        raise ValueError("ASv3 guardrails variants are mutually exclusive")
    if guardrails_v3:
        if research_profile != "normal":
            raise ValueError("Experimental Guardrails v3 requires the normal profile")
        if parallel_research:
            raise ValueError(
                "Experimental Guardrails v3 requires execution without parallel research"
            )
        return ASv3WorkflowSelection(
            research_profile="normal",
            parallel_research=False,
            workflow_variant=ASV3_GUARDRAILS_V3_VARIANT,
        )
    if guardrails_v2:
        if research_profile != "normal":
            raise ValueError("Experimental Guardrails v2 requires the normal profile")
        if parallel_research:
            raise ValueError(
                "Experimental Guardrails v2 requires execution without parallel research"
            )
        return ASv3WorkflowSelection(
            research_profile="normal",
            parallel_research=False,
            workflow_variant=ASV3_GUARDRAILS_V2_VARIANT,
        )
    if guarded_experimental:
        if research_profile != "normal":
            raise ValueError("Experimental Guardrails requires the normal profile")
        if parallel_research:
            raise ValueError(
                "Experimental Guardrails requires execution without parallel research"
            )
        return ASv3WorkflowSelection(
            research_profile="normal",
            parallel_research=False,
            workflow_variant=ASV3_GUARDED_EXPERIMENTAL_VARIANT,
        )
    if research_profile == "experimental" and parallel_research:
        return ASv3WorkflowSelection(
            research_profile="normal",
            parallel_research=False,
            workflow_variant=ASV3_TUNED_VARIANT,
        )
    return ASv3WorkflowSelection.model_validate(
        {"research_profile": research_profile, "parallel_research": parallel_research}
    )


def validate_asv3_variant_resume(
    workflow_variant: str, previous: dict[str, JsonValue] | None
) -> None:
    """Fence variants before a saved checkpoint can restore its execution profile."""
    if workflow_variant not in {
        ASV3_STANDARD_VARIANT,
        ASV3_TUNED_VARIANT,
        ASV3_EXP_CHEAP_VARIANT,
        ASV3_EXP_CROPPED_VARIANT,
        ASV3_FAST_VARIANT,
        ASV3_FAST_CROPPED_VARIANT,
        ASV3_GUARDED_EXPERIMENTAL_VARIANT,
        ASV3_GUARDRAILS_V2_VARIANT,
        ASV3_GUARDRAILS_V3_VARIANT,
    }:
        raise ValueError("Unknown ASv3 workflow variant")
    if previous is None:
        return
    if previous.get("parallel_research") is True:
        raise ValueError("The retired parallel workflow cannot be resumed")
    saved_variant = previous.get("asv3_workflow_variant", ASV3_STANDARD_VARIANT)
    if saved_variant != workflow_variant:
        raise ValueError("ASv3 resume requires the same workflow variant")
    fast_policy = {
        ASV3_FAST_VARIANT: ASV3_FAST_POLICY,
        ASV3_FAST_CROPPED_VARIANT: ASV3_FAST_CROPPED_POLICY,
    }.get(workflow_variant)
    if fast_policy is not None and (
        previous.get("asv3_workflow_policy") != fast_policy
        or previous.get("research_profile") != "normal"
        or previous.get("parallel_research") is not False
    ):
        raise ValueError("Fast ASv3 resume requires its own checkpoint policy")
    if workflow_variant == ASV3_EXP_CHEAP_VARIANT and (
        previous.get("asv3_workflow_policy") != ASV3_EXP_CHEAP_POLICY
        or previous.get("research_profile") != "normal"
        or previous.get("parallel_research") is not False
    ):
        raise ValueError("ASv3Exp Cheap resume requires its own checkpoint policy")
    if workflow_variant == ASV3_EXP_CROPPED_VARIANT and (
        previous.get("asv3_workflow_policy") != ASV3_EXP_CROPPED_POLICY
        or previous.get("research_profile") != "normal"
        or previous.get("parallel_research") is not False
    ):
        raise ValueError("exp cropped resume requires its own checkpoint policy")
    if workflow_variant == ASV3_TUNED_VARIANT and (
        previous.get("asv3_workflow_policy") != ASV3_TUNED_POLICY
        or previous.get("research_profile") != "normal"
        or previous.get("parallel_research") is not False
    ):
        raise ValueError(
            "Tuned ASv3 resume requires its normal-profile checkpoint policy"
        )
    if workflow_variant == ASV3_GUARDED_EXPERIMENTAL_VARIANT and (
        previous.get("asv3_workflow_policy") != ASV3_GUARDED_EXPERIMENTAL_POLICY
        or previous.get("research_profile") != "normal"
        or previous.get("parallel_research") is not False
    ):
        raise ValueError(
            "Experimental Guardrails resume requires its normal-profile checkpoint policy"
        )
    if workflow_variant == ASV3_GUARDRAILS_V2_VARIANT and (
        previous.get("asv3_workflow_policy") != ASV3_GUARDRAILS_V2_POLICY
        or previous.get("research_profile") != "normal"
        or previous.get("parallel_research") is not False
    ):
        raise ValueError(
            "Experimental Guardrails v2 resume requires its normal-profile checkpoint policy"
        )
    if workflow_variant == ASV3_GUARDRAILS_V3_VARIANT and (
        previous.get("asv3_workflow_policy") != ASV3_GUARDRAILS_V3_POLICY
        or previous.get("research_profile") != "normal"
        or previous.get("parallel_research") is not False
    ):
        raise ValueError(
            "Experimental Guardrails v3 resume requires its normal-profile checkpoint policy"
        )


def checkpoint_variant_fields(workflow_variant: str) -> dict[str, JsonValue]:
    fast_policy = {
        ASV3_FAST_VARIANT: ASV3_FAST_POLICY,
        ASV3_FAST_CROPPED_VARIANT: ASV3_FAST_CROPPED_POLICY,
    }.get(workflow_variant)
    if fast_policy is not None:
        return {
            "asv3_workflow_variant": workflow_variant,
            "asv3_workflow_policy": fast_policy,
        }
    if workflow_variant == ASV3_EXP_CROPPED_VARIANT:
        return {
            "asv3_workflow_variant": ASV3_EXP_CROPPED_VARIANT,
            "asv3_workflow_policy": ASV3_EXP_CROPPED_POLICY,
        }
    if workflow_variant == ASV3_EXP_CHEAP_VARIANT:
        return {
            "asv3_workflow_variant": ASV3_EXP_CHEAP_VARIANT,
            "asv3_workflow_policy": ASV3_EXP_CHEAP_POLICY,
        }
    if workflow_variant == ASV3_TUNED_VARIANT:
        return {
            "asv3_workflow_variant": ASV3_TUNED_VARIANT,
            "asv3_workflow_policy": ASV3_TUNED_POLICY,
        }
    if workflow_variant == ASV3_GUARDED_EXPERIMENTAL_VARIANT:
        return {
            "asv3_workflow_variant": ASV3_GUARDED_EXPERIMENTAL_VARIANT,
            "asv3_workflow_policy": ASV3_GUARDED_EXPERIMENTAL_POLICY,
        }
    if workflow_variant == ASV3_GUARDRAILS_V2_VARIANT:
        return {
            "asv3_workflow_variant": ASV3_GUARDRAILS_V2_VARIANT,
            "asv3_workflow_policy": ASV3_GUARDRAILS_V2_POLICY,
        }
    if workflow_variant == ASV3_GUARDRAILS_V3_VARIANT:
        return {
            "asv3_workflow_variant": ASV3_GUARDRAILS_V3_VARIANT,
            "asv3_workflow_policy": ASV3_GUARDRAILS_V3_POLICY,
        }
    return {}
