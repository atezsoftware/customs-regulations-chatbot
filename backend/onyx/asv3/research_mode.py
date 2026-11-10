"""Map the workflow recorded when an assistant message is reserved."""

from typing import Literal


def legislation_research_mode(
    workflow: str | None,
) -> Literal["single", "parallel"] | None:
    if workflow == "asv3":
        return "single"
    if workflow == "asv3_exp_cropped":
        return "parallel"
    return None
