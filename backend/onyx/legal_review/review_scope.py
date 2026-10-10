"""Separate research admission from review of the text actually being published."""

from typing import Literal

from pydantic import JsonValue

ReviewPurpose = Literal["research_readiness", "publication"]


def scope_review_state(state: dict[str, JsonValue]) -> dict[str, JsonValue]:
    if state.get("draft") is None:
        return {**state, "review_target": "research"}

    findings: list[JsonValue] = []
    assessments = state.get("dimension_assessments")
    requirements = state.get("requirements")
    for row in requirements if isinstance(requirements, list) else []:
        if not isinstance(row, dict):
            continue
        issue_ids: list[JsonValue] = []
        for assessment in assessments if isinstance(assessments, list) else []:
            if not isinstance(assessment, dict):
                continue
            linked = assessment.get("requirement_ids")
            identity = assessment.get("issue_id")
            if (
                isinstance(linked, list)
                and row.get("requirement_id") in linked
                and isinstance(identity, str)
                and identity not in issue_ids
            ):
                issue_ids.append(identity)
        findings.append(
            {
                "requirement_id": row.get("requirement_id"),
                "issue_ids": issue_ids,
                "supports": row.get("supports"),
            }
        )
    # Private interpretations have their own research review. Carry source bindings,
    # not old conclusions that can be mistaken for assertions in a revised answer.
    result = {
        key: value
        for key, value in state.items()
        if key
        not in {
            "dimension_assessments",
            "requirements",
            "reading_contract",
            "source_assessments",
            "material_source_leads",
            "review_diagnoses",
            "research_resolutions",
            "repair_resolutions",
            "early_review",
            "final_review",
            "final_adjudication",
            "draft_adjudication",
        }
    }
    result["review_target"] = "literal_answer"
    result["finding_sources"] = findings
    return result


def publication_state(state: dict[str, JsonValue]) -> dict[str, JsonValue]:
    """Review all originals and the whole answer without replaying research machinery."""
    scoped = scope_review_state(state)
    result = {
        key: value
        for key, value in scoped.items()
        if key
        in {
            "request",
            "history",
            "plan",
            "requested_outcomes",
            "corpus_currency",
            "original_evidence",
            "draft",
            "finding_sources",
            "review_target",
            "issue_closures",
            "source_issue_triggers",
        }
    }
    result["review_purpose"] = "publication"
    return result


def research_examination_state(
    state: dict[str, JsonValue],
) -> dict[str, JsonValue]:
    """Keep actual investigations visible before deciding whether research is needed."""
    result = publication_state(state)
    for key in ("research_needs", "source_operations", "research_record"):
        if key in state:
            result[key] = state[key]
    result["review_purpose"] = "research_readiness"
    return result
