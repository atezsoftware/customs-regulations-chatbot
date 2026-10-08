"""Wider candidate delivery confined to Legal Composite search instances."""

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


class CompositeSearchTool(SearchTool):
    @classmethod
    def from_fork(cls, fork: SearchTool) -> CompositeSearchTool:
        # Only owned independent forks are wrapped; the caller's tool is untouched.
        instance = cls.__new__(cls)
        instance.__dict__.update(fork.__dict__)
        return instance

    def fork_for_independent_context(
        self, *, emitter: Emitter | None = None
    ) -> SearchTool:
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
        wider = SearchToolRetrievalOverrideKwargs.model_validate(
            {
                **override_kwargs.model_dump(),
                "per_lane_num_hits": 100,
                "rerank_candidate_limit": 160,
                "regulatory_rerank_candidate_limit": 160,
                "max_llm_chunks": 40,
                "preserve_source_diversity": True,
                "reuse_diversity_comparisons": True,
            }
        )
        return super().run(placement, wider, **llm_kwargs)
