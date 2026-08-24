from dataclasses import dataclass
from typing import Literal

RegulatoryWorkflowMode = Literal["standard", "fast"]


@dataclass(frozen=True, slots=True)
class RegulatoryWorkflowProfile:
    mode: RegulatoryWorkflowMode
    max_parallel_search_calls: int | None
    max_concurrent_search_tools: int
    search_chunks_per_call: int
    max_candidate_reviews: int
    post_review_cycles: int
    include_auxiliary_searches: bool
    include_lexical_fallbacks: bool
    use_navigation_recovery: bool
    use_evidence_matrix: bool
    direct_synthesis_after_plan_search: bool


STANDARD_REGULATORY_WORKFLOW = RegulatoryWorkflowProfile(
    mode="standard",
    max_parallel_search_calls=32,
    max_concurrent_search_tools=8,
    search_chunks_per_call=10,
    max_candidate_reviews=2,
    post_review_cycles=3,
    include_auxiliary_searches=True,
    include_lexical_fallbacks=True,
    use_navigation_recovery=True,
    use_evidence_matrix=True,
    direct_synthesis_after_plan_search=False,
)

FAST_REGULATORY_WORKFLOW = RegulatoryWorkflowProfile(
    mode="fast",
    # The validated plan determines total fan-out. Concurrency remains bounded
    # independently, so a complex scenario is not semantically truncated.
    max_parallel_search_calls=None,
    max_concurrent_search_tools=8,
    search_chunks_per_call=10,
    max_candidate_reviews=0,
    post_review_cycles=0,
    include_auxiliary_searches=False,
    include_lexical_fallbacks=False,
    use_navigation_recovery=False,
    use_evidence_matrix=False,
    direct_synthesis_after_plan_search=True,
)


def get_regulatory_workflow_profile(
    mode: RegulatoryWorkflowMode,
) -> RegulatoryWorkflowProfile:
    return FAST_REGULATORY_WORKFLOW if mode == "fast" else STANDARD_REGULATORY_WORKFLOW
