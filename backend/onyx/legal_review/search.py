"""Preserve broad retrieval on owned forks without source-category routing."""

from __future__ import annotations

from typing import Any

from onyx.chat.emitter import Emitter
from onyx.server.query_and_chat.placement import Placement
from onyx.tools.models import (
    SearchToolOverrideKwargs,
    SearchToolRetrievalOverrideKwargs,
    ToolResponse,
)
from onyx.tools.tool_implementations.search.search_tool import SearchTool


class DiscoverySearchTool(SearchTool):
    NORMALIZED_RERANK_THRESHOLD = 0.82

    @classmethod
    def from_fork(cls, fork: SearchTool) -> DiscoverySearchTool:
        instance = cls.__new__(cls)
        instance.__dict__.update(fork.__dict__)
        if fork.user_selected_filters is not None:
            instance.user_selected_filters = fork.user_selected_filters.model_copy(
                deep=True
            )
        instance.auto_detect_filters = False
        instance.enable_slack_search = False
        instance.bypass_acl = False
        return instance

    def fork_for_independent_context(
        self, *, emitter: Emitter | None = None
    ) -> DiscoverySearchTool:
        return self.from_fork(super().fork_for_independent_context(emitter=emitter))

    def fork_for_parallel_calls(self, call_count: int) -> list[SearchTool]:
        return [
            self.from_fork(fork) for fork in super().fork_for_parallel_calls(call_count)
        ]

    def run(
        self,
        placement: Placement,
        override_kwargs: SearchToolOverrideKwargs,
        **llm_kwargs: Any,
    ) -> ToolResponse:
        # The parent request explains the case; each source operation has its own
        # evidence question, which must remain the relevance target during reranking.
        target = llm_kwargs.get("evidence_target")
        if not isinstance(target, str) or not target.strip():
            queries = llm_kwargs.get("queries", [])
            target = (
                "\n\n".join(query for query in queries if isinstance(query, str))
                if isinstance(queries, list)
                else ""
            )
        bounded = SearchToolRetrievalOverrideKwargs.model_validate(
            {
                **override_kwargs.model_dump(mode="python"),
                "rerank_context": target or None,
                "skip_query_expansion": True,
                "per_lane_num_hits": 256,
                "rerank_candidate_limit": 384,
                "regulatory_rerank_candidate_limit": 384,
                "max_llm_chunks": 50,
                "preserve_source_diversity": True,
                "reuse_diversity_comparisons": True,
                "guarded_decisions_advisory": False,
            }
        )
        return super().run(placement, bounded, **llm_kwargs)
