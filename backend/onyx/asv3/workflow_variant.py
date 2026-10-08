"""Resolve the legacy selection slot without changing established ASv3 profiles."""

from __future__ import annotations

from pydantic import JsonValue

from onyx.asv3.models import ASv3WorkflowSelection

ASV3_STANDARD_VARIANT = "standard"
ASV3_TUNED_VARIANT = "asv3_tuned"
ASV3_TUNED_POLICY = "normal-copy-v1"
ASV3_GUARDED_EXPERIMENTAL_VARIANT = "asv3_guarded_experimental"
ASV3_GUARDED_EXPERIMENTAL_POLICY = "guarded-normal-v1"


def resolve_asv3_workflow(
    research_profile: str,
    parallel_research: bool,
    guarded_experimental: bool = False,
) -> ASv3WorkflowSelection:
    if guarded_experimental:
        if research_profile != "normal":
            raise ValueError("Experimental Guardrails requires the normal profile")
        if parallel_research:
            raise ValueError("Experimental Guardrails requires execution without parallel research")
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
        ASV3_GUARDED_EXPERIMENTAL_VARIANT,
    }:
        raise ValueError("Unknown ASv3 workflow variant")
    if previous is None:
        return
    if previous.get("parallel_research") is True:
        raise ValueError("The retired parallel workflow cannot be resumed")
    saved_variant = previous.get("asv3_workflow_variant", ASV3_STANDARD_VARIANT)
    if saved_variant != workflow_variant:
        raise ValueError("ASv3 resume requires the same workflow variant")
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


def checkpoint_variant_fields(workflow_variant: str) -> dict[str, JsonValue]:
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
    return {}
