"""Preserve explicit user requests independently of model-chosen issue granularity."""

from __future__ import annotations

import re
from typing import Any

from pydantic import Field, JsonValue, create_model, model_validator

from onyx.legal_review.models import InitialDiscoveryPlan, RequestedOutcome, StrictModel

_NUMBERED_REQUEST = re.compile(r"(?m)^[ \t]{0,3}\d+[.)]\s+(?=\S)")


def request_units(request: str) -> dict[str, str]:
    """Keep authored list boundaries; do not infer legal issues or hidden requirements."""
    matches = list(_NUMBERED_REQUEST.finditer(request))
    values = (
        [
            request[
                match.end() : matches[index + 1].start()
                if index + 1 < len(matches)
                else len(request)
            ].strip()
            for index, match in enumerate(matches)
        ]
        if matches
        else [request.strip()]
    )
    return {f"q{index:04d}": value for index, value in enumerate(values, 1) if value}


class RequestCoverage(StrictModel):
    requested_result: str = Field(
        min_length=1,
        description="The requested result(s) in this exact user-authored unit. Preserve alternatives without assuming their legal answer.",
    )
    issue_ids: list[str] = Field(min_length=1)


class CoveredPlan(InitialDiscoveryPlan):
    @model_validator(mode="after")
    def known_coverage_issues(self) -> CoveredPlan:
        known = {issue.issue_id for issue in self.issues}
        for row in self.model_dump()["request_coverage"].values():
            if set(row["issue_ids"]) - known:
                raise ValueError("Explicit user requests must map to known issues")
        return self


class InitialPlanContract:
    def __init__(self, request: str) -> None:
        self.units = request_units(request)
        fields: dict[str, Any] = {
            slot: (RequestCoverage, Field()) for slot in self.units
        }
        coverage = create_model(
            "ExplicitRequestCoverage", __base__=StrictModel, **fields
        )
        self.response_model = create_model(
            "CoveredInitialDiscoveryPlan",
            __base__=CoveredPlan,
            request_coverage=(coverage, Field()),
        )

    def state(self) -> dict[str, JsonValue]:
        return {"explicit_request_units": dict(self.units)}

    def compile(self, response: InitialDiscoveryPlan) -> InitialDiscoveryPlan:
        data = response.model_dump(mode="python")
        coverage = data.pop("request_coverage")
        known = {issue.issue_id for issue in response.issues}
        outcomes: list[RequestedOutcome] = []
        for slot, text in self.units.items():
            row = RequestCoverage.model_validate(coverage[slot])
            if set(row.issue_ids) - known:
                raise ValueError("Explicit user requests must map to known issues")
            outcomes.append(RequestedOutcome(request=text, issue_ids=row.issue_ids))
        # Retain the model's finer decomposition as well as every literal user unit.
        for outcome in response.requested_outcomes:
            if outcome not in outcomes:
                outcomes.append(outcome)
        data["requested_outcomes"] = [row.model_dump() for row in outcomes]
        return InitialDiscoveryPlan.model_validate(data)
