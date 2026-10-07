"""Exercise variant settings through the real dispatcher and search selection."""

from datetime import date
from typing import Any
from unittest.mock import patch
from uuid import uuid4

import pytest

from onyx.asv3.models import EvidenceItem, OutcomeStatus, RunContext
from onyx.asv3.search_adapter import build_search_adapter
from onyx.asv3.workflow_variant import ASV3_TUNED_VARIANT
from onyx.configs.constants import DocumentSource
from onyx.context.search.models import InferenceChunk, SearchDoc, SearchDocsResponse
from onyx.db.reranking import RerankerRuntimeConfig
from onyx.reranking.models import RerankOutcome, RerankResult
from onyx.reranking.staged import STAGED_BATCH_SIZE
from onyx.tools.models import ParallelToolCallResponse, SearchToolRetrievalOverrides
from onyx.tools.tool_implementations.search.search_tool import SearchTool
from onyx.tools.tool_runner import run_tool_calls
from tests.unit.onyx.asv3.test_search_adapter import (
    MODULE,
    search_boundaries,
    tool_and_broker,
    user_message,
)
from tests.unit.onyx.tools.test_tool_runner import _make_tool_call


def candidate_chunks() -> list[InferenceChunk]:
    return [
        InferenceChunk(
            document_id=(source_id := str(uuid4())),
            chunk_id=index,
            content=f"Index projection {index}",
            source_type=DocumentSource.USER_FILE,
            semantic_identifier=f"Source {index}",
            title=f"Source {index}",
            boost=1,
            score=1 - index / 320,
            hidden=False,
            metadata={},
            match_highlights=[],
            doc_summary="",
            chunk_context="",
            updated_at=None,
            image_file_id=None,
            source_links=None,
            section_continuation=False,
            blurb=f"Index blurb {index}",
            file_id=source_id,
            regulatory_chunk_id=f"rc-{index}",
            heading_path=[f"MADDE {index + 1}"],
        )
        for index in range(320)
    ]


@pytest.mark.parametrize(
    "variant,profile,qualifying,candidate_count,delivered_count",
    [
        (ASV3_TUNED_VARIANT, "normal", 20, 256, 50),
        (ASV3_TUNED_VARIANT, "normal", 60, 256, 60),
        (ASV3_TUNED_VARIANT, "normal", 150, 256, 150),
        (None, "normal", 20, 96, 25),
        (None, "deep", 20, 96, 25),
        (None, "experimental", 20, 96, 25),
    ],
)
def test_real_adapter_retains_scoped_candidate_pool_and_canonical_delivery(
    variant: str | None,
    profile: str,
    qualifying: int,
    candidate_count: int,
    delivered_count: int,
) -> None:
    tool, broker, selected_model = tool_and_broker()
    broker.filters.as_of_date = date(2025, 1, 1)
    chunks = candidate_chunks()
    context = RunContext(services={"research_profile": profile})
    if variant is not None:
        context.services["asv3_workflow_variant"] = variant
    scenario = "Actual fixed scenario and requested independent outcome"
    adapter = build_search_adapter(
        tool,
        scenario,
        broker,
        message_history=lambda _: [user_message(scenario)],
    )
    responses: list[ParallelToolCallResponse] = []

    def execute(**kwargs: Any) -> ParallelToolCallResponse:
        response = run_tool_calls(**kwargs)
        responses.append(response)
        return response

    def score(**kwargs: Any) -> RerankResult:
        batch: list[InferenceChunk] = kwargs["chunks"]
        return RerankResult(
            ordered_chunks=batch,
            scores_by_chunk={
                (chunk.document_id, chunk.chunk_id): (
                    1.0 if chunk.chunk_id < qualifying else 0.0
                )
                for chunk in batch
            },
            submitted_count=len(batch),
            result_count=len(batch),
            outcome=RerankOutcome.SUCCESS,
            fallback_used=False,
        )

    def hydrate(
        docs: list[SearchDoc], captured: RunContext
    ) -> dict[tuple[str, int], list[EvidenceItem]]:
        assert captured is context
        return {
            (doc.document_id, doc.chunk_ind): [
                EvidenceItem(
                    source_id=doc.document_id,
                    chunk_id=str(doc.metadata["regulatory_chunk_id"]),
                    text=f"Canonical original {doc.chunk_ind}",
                    metadata={"version_unknown": True},
                )
            ]
            for doc in docs
        }

    with (
        search_boundaries() as (pipeline, _, _),
        patch(
            f"{MODULE}.get_reranker_configuration",
            return_value=RerankerRuntimeConfig(
                enabled=True,
                provider_type=None,
                model_name="BAAI/bge-reranker-v2-m3",
                api_key=None,
                configuration_generation="fixture",
            ),
        ),
        patch(f"{MODULE}.rerank_chunks", side_effect=score) as rerank,
        patch.object(
            broker, "hydrate_search_centers", side_effect=hydrate
        ) as hydration,
        patch(
            "onyx.asv3.search_adapter.run_tool_calls", side_effect=execute
        ) as dispatch,
    ):
        pipeline.side_effect = lambda **kwargs: chunks[
            : kwargs["chunk_search_request"].limit
        ]
        outcome = adapter(
            {
                "query": "Focused source conditions",
                "mode": "hybrid",
                "coverage_item": "Requested outcome",
                "evidence_target": "Applicable original conditions",
                "expand_query": True,
            },
            context,
        )

    assert outcome.status == OutcomeStatus.FOUND
    assert len(outcome.evidence) == delivered_count
    assert all(item.text.startswith("Canonical original") for item in outcome.evidence)
    assert all(item.metadata["version_unknown"] is True for item in outcome.evidence)
    hydration.assert_called_once()
    assert len(hydration.call_args.args[0]) == delivered_count
    assert selected_model.invoke.call_count == 2
    assert pipeline.call_count >= 2
    for call in pipeline.call_args_list:
        request = call.kwargs["chunk_search_request"]
        assert request.limit == max(candidate_count, 128)
        assert request.user_selected_filters == broker.filters
        assert request.user_selected_filters.as_of_date == date(2025, 1, 1)
    batches = [call.kwargs["chunks"] for call in rerank.call_args_list]
    assert [len(batch) for batch in batches] == [STAGED_BATCH_SIZE] * (
        candidate_count // STAGED_BATCH_SIZE
    )
    assert (
        len({chunk.unique_id for batch in batches for chunk in batch})
        == candidate_count
    )
    override = dispatch.call_args.kwargs["search_retrieval_overrides"]
    assert (override is not None) is (variant == ASV3_TUNED_VARIANT)
    if override is not None:
        assert override.preserve_source_diversity is True
    rich = responses[0].tool_responses[0].rich_response
    assert isinstance(rich, SearchDocsResponse)
    assert len(rich.search_docs) == delivered_count
    assert rich.displayed_docs is not None
    assert len(rich.displayed_docs) == min(delivered_count, 50)
    assert len(rich.citation_mapping) == delivered_count


@pytest.mark.parametrize(
    "tuned,diversity,target_score",
    [(True, True, 1.0), (True, True, 0.0), (True, False, 1.0), (False, False, 1.0)],
)
def test_late_source_is_scored_and_delivered_only_with_source_diversity(
    tuned: bool, diversity: bool, target_score: float
) -> None:
    tool, broker, _ = tool_and_broker()
    source_id = str(uuid4())
    dominant = [
        chunk.model_copy(update={"document_id": source_id, "file_id": source_id})
        for chunk in candidate_chunks()[:192]
    ]
    independent = [
        chunk.model_copy(update={"regulatory_chunk_id": f"rc-independent-{index}"})
        for index, chunk in enumerate(candidate_chunks()[:42])
    ]
    target = independent[25]
    context = RunContext(services={"research_profile": "normal"})
    if tuned:
        context.services["asv3_workflow_variant"] = ASV3_TUNED_VARIANT
    adapter = build_search_adapter(
        tool,
        "Fixed factual scenario",
        broker,
        message_history=lambda _: [user_message("Fixed factual scenario")],
    )
    submitted: list[InferenceChunk] = []

    def search(**kwargs: Any) -> list[InferenceChunk]:
        request = kwargs["chunk_search_request"]
        lane = (
            dominant[:95] + independent + dominant[95:150]
            if request.query == "lexical variant"
            else dominant
        )
        return lane[: request.limit]

    def score(**kwargs: Any) -> RerankResult:
        batch: list[InferenceChunk] = kwargs["chunks"]
        submitted.extend(batch)
        return RerankResult(
            ordered_chunks=batch,
            scores_by_chunk={
                (chunk.document_id, chunk.chunk_id): (
                    target_score if chunk.unique_id == target.unique_id else 0.5
                )
                for chunk in batch
            },
            submitted_count=len(batch),
            result_count=len(batch),
            outcome=RerankOutcome.SUCCESS,
            fallback_used=False,
        )

    def hydrate(
        docs: list[SearchDoc], _: RunContext
    ) -> dict[tuple[str, int], list[EvidenceItem]]:
        return {
            (doc.document_id, doc.chunk_ind): [
                EvidenceItem(
                    source_id=doc.document_id,
                    chunk_id=str(doc.metadata["regulatory_chunk_id"]),
                    text=f"Canonical original {doc.document_id}/{doc.chunk_ind}",
                )
            ]
            for doc in docs
        }

    with (
        search_boundaries() as (pipeline, _, _),
        patch(
            "onyx.asv3.search_adapter._TUNED_RETRIEVAL_OVERRIDES",
            SearchToolRetrievalOverrides(
                per_lane_num_hits=192,
                rerank_candidate_limit=192,
                regulatory_rerank_candidate_limit=192,
                max_llm_chunks=50,
                preserve_source_diversity=diversity,
            ),
        ),
        patch(
            f"{MODULE}.get_reranker_configuration",
            return_value=RerankerRuntimeConfig(
                enabled=True,
                provider_type=None,
                model_name="BAAI/bge-reranker-v2-m3",
                api_key=None,
                configuration_generation="fixture",
            ),
        ),
        patch(f"{MODULE}.rerank_chunks", side_effect=score),
        patch.object(broker, "hydrate_search_centers", side_effect=hydrate),
    ):
        pipeline.side_effect = search
        outcome = adapter(
            {
                "query": "Applicable conditions",
                "mode": "hybrid",
                "coverage_item": "Requested legal effect",
                "evidence_target": "Governing qualifications",
                "expand_query": True,
            },
            context,
        )

    assert len(submitted) == (192 if tuned else 96)
    assert (target.unique_id in {chunk.unique_id for chunk in submitted}) is diversity
    assert (target.document_id in {item.source_id for item in outcome.evidence}) is (
        diversity and target_score > 0
    )
    assert all(item.text.startswith("Canonical original") for item in outcome.evidence)


@pytest.mark.parametrize("tuned", [True, False])
def test_runner_keeps_independent_citation_bands_and_original_tool(tuned: bool) -> None:
    tool, _, _ = tool_and_broker()
    overrides = SearchToolRetrievalOverrides(
        per_lane_num_hits=192,
        rerank_candidate_limit=192,
        regulatory_rerank_candidate_limit=192,
        max_llm_chunks=50,
    )
    calls = [
        _make_tool_call(SearchTool.NAME, {"queries": [query]}, query)
        for query in ("first", "second")
    ]
    with patch(
        "onyx.tools.tool_runner.run_functions_tuples_in_parallel", return_value=[]
    ) as execute:
        run_tool_calls(
            tool_calls=calls,
            tools=[tool],
            message_history=[user_message("Fixed scenario")],
            user_memory_context=None,
            user_info=None,
            citation_mapping={},
            next_citation_num=1,
            search_retrieval_overrides=overrides if tuned else None,
        )
    invocations = execute.call_args.args[0]
    instances = [args[0] for _, args in invocations]
    settings = [args[2] for _, args in invocations]
    assert [item.starting_citation_num for item in settings] == [
        1,
        193 if tuned else 101,
    ]
    assert all(item.max_llm_chunks == (50 if tuned else 25) for item in settings)
    assert all(item.per_lane_num_hits == (192 if tuned else 50) for item in settings)
    assert all(
        ("regulatory_rerank_candidate_limit" in item.model_dump()) is tuned
        for item in settings
    )
    assert len({id(instance) for instance in instances}) == 2
    assert all(instance is not tool for instance in instances)
