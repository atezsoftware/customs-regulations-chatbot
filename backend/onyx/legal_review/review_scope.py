"""Separate research admission from review of the text actually being published."""

from pydantic import JsonValue


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
