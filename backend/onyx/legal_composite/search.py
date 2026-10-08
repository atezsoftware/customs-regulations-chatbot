"""Independent searches confined before retrieval by prepared canonical source types."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import JsonValue

from onyx.chat.emitter import Emitter
from onyx.context.search.models import (
    BaseFilters,
    ChunkSearchRequest,
    IndexFilters,
    InferenceChunk,
)
from onyx.context.search.pipeline import search_pipeline
from onyx.context.search.retrieval.parallel_retrieval_scope import (
    experimental_parallel_retrieval,
)
from onyx.db.legal_composite_sources import SourceKind
from onyx.federated_connectors.federated_retrieval import FederatedRetrievalInfo
from onyx.natural_language_processing.search_nlp_models import EmbeddingModel
from onyx.regulatory.heading_path import RegulatoryProvisionReference
from onyx.server.query_and_chat.placement import Placement
from onyx.tools.models import (
    SearchToolOverrideKwargs,
    SearchToolRetrievalOverrideKwargs,
    ToolResponse,
)
from onyx.tools.tool_implementations.search.search_tool import SearchTool
from onyx.tracing.answer_graph import graph_step


class CompositeSearchTool(SearchTool):
    _prepared_source_kind: SourceKind | None = None
    _record_prepared_search: Callable[[dict[str, JsonValue]], None] | None = None
    _check_prepared_search: Callable[[], None] | None = None

    @classmethod
    def from_fork(cls, fork: SearchTool) -> CompositeSearchTool:
        # Only owned independent forks are wrapped; the caller's tool is untouched.
        instance = cls.__new__(cls)
        instance.__dict__.update(fork.__dict__)
        if fork.user_selected_filters is not None:
            instance.user_selected_filters = fork.user_selected_filters.model_copy(
                deep=True
            )
        return instance

    def configure_prepared_source_lane(
        self,
        kind: SourceKind,
        record_search: Callable[[dict[str, JsonValue]], None] | None = None,
        check_active: Callable[[], None] | None = None,
    ) -> None:
        self._prepared_source_kind = kind
        self._record_prepared_search = record_search
        self._check_prepared_search = check_active

    def _configured_fork(self, fork: SearchTool) -> CompositeSearchTool:
        instance = self.from_fork(fork)
        instance._prepared_source_kind = self._prepared_source_kind
        instance._record_prepared_search = self._record_prepared_search
        instance._check_prepared_search = self._check_prepared_search
        return instance

    def fork_for_independent_context(
        self, *, emitter: Emitter | None = None
    ) -> CompositeSearchTool:
        return self._configured_fork(
            super().fork_for_independent_context(emitter=emitter)
        )

    def fork_for_parallel_calls(self, call_count: int) -> list[SearchTool]:
        return [
            self._configured_fork(fork)
            for fork in super().fork_for_parallel_calls(call_count)
        ]

    def _run_search_for_query(
        self,
        query: str,
        hybrid_alpha: float | None,
        high_term_coverage: bool,
        num_hits: int,
        acl_filters: list[str] | None,
        embedding_model: EmbeddingModel,
        federated_retrieval_infos: list[FederatedRetrievalInfo],
        effective_filters: BaseFilters | None,
        provision_reference: RegulatoryProvisionReference | None = None,
    ) -> list[InferenceChunk]:
        # An article mentioned in a query may belong to a different source.
        del provision_reference
        kind = self._prepared_source_kind
        if kind is None:
            return super()._run_search_for_query(
                query,
                hybrid_alpha,
                high_term_coverage,
                num_hits,
                acl_filters,
                embedding_model,
                federated_retrieval_infos,
                effective_filters,
                None,
            )
        if (
            self.bypass_acl
            or not isinstance(effective_filters, IndexFilters)
            or not effective_filters.regulatory_chunks_only
            or effective_filters.asv3_document_set_id is None
            or not effective_filters.attached_document_ids
            or acl_filters is None
        ):
            raise PermissionError(
                "Prepared source searches require captured IDs and PC ACLs."
            )
        if self._check_prepared_search is not None:
            self._check_prepared_search()
        captured = frozenset(effective_filters.attached_document_ids)
        with graph_step(
            "legal_composite.prepared_source_search",
            {"source_kind": kind.value, "source_count": len(captured)},
        ) as step:
            hits = search_pipeline(
                chunk_search_request=ChunkSearchRequest(
                    query=query,
                    hybrid_alpha=hybrid_alpha,
                    high_term_coverage=high_term_coverage,
                    user_selected_filters=effective_filters,
                    bypass_acl=False,
                    limit=num_hits,
                ),
                project_id_filter=self.project_id_filter,
                persona_id_filter=self.persona_id_filter,
                document_index=self.document_index,
                user=self.user,
                persona_search_info=self.persona_search_info,
                acl_filters=acl_filters,
                embedding_model=embedding_model,
                prefetched_federated_retrieval_infos=federated_retrieval_infos,
            )
            if self._check_prepared_search is not None:
                self._check_prepared_search()
            retained = [hit for hit in hits if hit.document_id in captured]
            outside = sorted(
                {hit.document_id for hit in hits if hit.document_id not in captured}
            )
            saturated = len(hits) >= num_hits
            receipt: dict[str, JsonValue] = {
                "source_kind": kind.value,
                "classification_basis": "prepared_canonical_source_kind",
                "source_kind_filter_before_retrieval": True,
                "runtime_opening_reads": 0,
                "runtime_classifications": 0,
                "captured_source_count": len(captured),
                "raw_hit_count": len(hits),
                "requested_hits": num_hits,
                "delivered_hits": min(num_hits, len(retained)),
                "unavailable_source_ids": outside,
                "candidate_window_saturated": saturated,
                "incomplete": saturated or bool(outside),
                "corpus_absence_verified": False,
                "borrowed_provision_heading_gate_applied": False,
            }
            if self._record_prepared_search is not None:
                self._record_prepared_search(receipt)
            step.output_value = receipt
            # Borrowed article references cannot gate another source's own heading.
            return retained[:num_hits]

    def run(
        self,
        placement: Placement,
        override_kwargs: SearchToolOverrideKwargs,
        **llm_kwargs: Any,
    ) -> ToolResponse:
        if self._prepared_source_kind is not None:
            self.enable_slack_search = False
        filters = self.user_selected_filters
        if (
            isinstance(filters, IndexFilters)
            and filters.attached_document_ids is not None
        ):
            if not filters.attached_document_ids or self.bypass_acl:
                raise PermissionError(
                    "Source lane search requires captured IDs and ACLs."
                )
            self.user_selected_filters = filters.model_copy(
                deep=True,
                update={
                    "document_set": None,
                    "hierarchy_node_ids": None,
                    "project_id_filter": None,
                    "persona_id_filter": None,
                },
            )
            self.persona_search_info = self.persona_search_info.model_copy(
                deep=True,
                update={
                    "document_set_names": [],
                    "hierarchy_node_ids": [],
                    "attached_document_ids": list(filters.attached_document_ids),
                },
            )
            self.project_id_filter = None
            self.persona_id_filter = None
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
        with experimental_parallel_retrieval(
            check_active=self._check_prepared_search or (lambda: None)
        ):
            return super().run(placement, wider, **llm_kwargs)
