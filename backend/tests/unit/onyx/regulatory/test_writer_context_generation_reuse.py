"""Owned publication reuses exact generations within one preparation only."""

from dataclasses import replace
from datetime import date
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from onyx.db.enums import IndexModelStatus
from onyx.db.models import RegulatoryChunk, SearchSettings, UserFile
from onyx.db.regulatory_writer_publication import OwnedWriterInputs
from onyx.indexing import indexing_pipeline
from onyx.indexing.embedder import DefaultIndexingEmbedder
from onyx.llm.interfaces import LLM, LLMConfig
from onyx.llm.models import LanguageModelInput, UserMessage
from onyx.regulatory import projection, writer_projection
from onyx.regulatory.amendments.annexes.context_dependencies import (
    ContextGenerationRecorder,
)
from onyx.regulatory.amendments.annexes.models import PreparedContextView
from onyx.regulatory.writer_publication_models import WriterPublicationManifest
from onyx.tracing.flows import LLMFlow
from tests.unit.onyx.regulatory.annexes.test_context_dependencies import (
    _durable_embedding_model,
)
from tests.unit.onyx.regulatory.indexing_jobs.owned_publication_test_helpers import (
    OwnedAuthority,
    canonical_row,
    writer_inputs,
)


class WriterContextHarness:
    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        file_id = uuid4()
        self.authority = OwnedAuthority(file_id)
        rows = [
            canonical_row(file_id, position, text)
            for position, text in enumerate(
                ["First legal rule", "Second legal rule", "Third legal rule"]
            )
        ]
        successor = canonical_row(file_id, 3, rows[2].text)
        successor.heading_path = list(rows[2].heading_path)
        successor.chunk_metadata = dict(rows[2].chunk_metadata)
        rows[2].validity_end_date = date(2026, 7, 1)
        successor.validity_start_date = date(2026, 7, 1)
        self.inputs = replace(
            writer_inputs(file_id, [*rows, successor]),
            settings=[
                SearchSettings(
                    id=identifier,
                    status=status,
                    index_name=f"context-index-{identifier}",
                    model_name="embedding",
                    model_dim=3,
                    normalize=True,
                    enable_contextual_rag=True,
                )
                for identifier, status in [
                    (1, IndexModelStatus.PRESENT),
                    (2, IndexModelStatus.FUTURE),
                ]
            ],
        )
        self.llm = MagicMock()
        self.llm.config = LLMConfig(
            model_provider="configured",
            model_name="context",
            temperature=0,
            max_input_tokens=10000,
        )
        seed = ContextGenerationRecorder()
        seed.invoke(
            llm=self.llm,
            flow=LLMFlow.CONTEXTUAL_RAG_DOC_SUMMARY,
            prompt=UserMessage(content="Unrelated retained generation"),
            generate=lambda: "Unrelated retained output",
            consumer_id="unrelated",
            stage="summary",
            source_text="Unrelated retained source",
            token_budget=100,
        )
        self.inputs = replace(
            self.inputs, cached=PreparedContextView(calls=list(seed.calls.values()))
        )
        self.initial_cache = self.inputs.cached.model_dump_json()
        self.completed: list[tuple[PreparedContextView, str]] = []
        self.supplied_caches: list[PreparedContextView] = []
        self.generate = MagicMock(return_value="Generated contextual note.")
        model = _durable_embedding_model()
        embedder = MagicMock(embedding_model=model)
        self.client = MagicMock()
        self.client.indices.get.side_effect = lambda *, index: {
            index: {"settings": {"index": {"uuid": f"physical-{index}"}}}
        }
        monkeypatch.setattr(
            writer_projection, "PublicationStore", self.authority.for_scope
        )
        monkeypatch.setattr(
            writer_projection.DefaultIndexingEmbedder,
            "from_db_search_settings",
            lambda **_: embedder,
        )
        monkeypatch.setattr(
            writer_projection, "resolve_review_context_llm", lambda *_: self.llm
        )
        monkeypatch.setattr(
            model,
            "encode",
            lambda texts, **_: [[0.1, 0.2, 0.3] for _ in texts],
        )
        monkeypatch.setattr(
            projection, "_get_contextual_tokenizer", lambda _: model.tokenizer
        )
        monkeypatch.setattr(indexing_pipeline, "MAX_TOKENS_FOR_FULL_INCLUSION", 0)
        monkeypatch.setattr(indexing_pipeline, "USE_DOCUMENT_SUMMARY", True)
        monkeypatch.setattr(indexing_pipeline, "USE_CHUNK_SUMMARY", True)
        monkeypatch.setattr(projection, "USE_CHUNK_SUMMARY", True)
        monkeypatch.setattr(
            indexing_pipeline, "_invoke_contextual_llm_with_retry", self.generate
        )
        monkeypatch.setattr(
            writer_projection, "prepare_normal_context_view", self.capture_view
        )

    def capture_view(
        self,
        *,
        rows: list[RegulatoryChunk],
        user_file: UserFile,
        search_settings: SearchSettings,
        embedder: DefaultIndexingEmbedder,
        llm: LLM | None,
        cached: PreparedContextView | None = None,
        as_of_date: date | None = None,
        target_ids: set[str] | None = None,
    ) -> PreparedContextView:
        assert cached is not None
        self.supplied_caches.append(cached)
        view = projection.prepare_normal_context_view(
            rows=rows,
            user_file=user_file,
            search_settings=search_settings,
            embedder=embedder,
            llm=llm,
            cached=cached,
            as_of_date=as_of_date,
            target_ids=target_ids,
        )
        self.completed.append((view, view.model_dump_json()))
        return view

    def prepare(
        self, inputs: OwnedWriterInputs | None = None
    ) -> WriterPublicationManifest:
        current = inputs if inputs is not None else self.inputs
        return writer_projection.prepare_owned_correction(
            self.authority.owner,
            self.client,
            current,
            current.canonical,
            changed_id=None,
        )


@pytest.fixture
def writer(monkeypatch: pytest.MonkeyPatch) -> WriterContextHarness:
    return WriterContextHarness(monkeypatch)


def test_writer_reuses_actual_generations_across_windows_and_indexes_without_mutation(
    writer: WriterContextHarness,
) -> None:
    prepared = writer.prepare()

    assert len(prepared.views) == 4
    assert len(prepared.bindings) == 12
    assert [len(view.calls) for view in prepared.views] == [4, 4, 4, 4]
    unique_generations = {
        (call.prompt_json, call.config_sha256)
        for view in prepared.views
        for call in view.calls
    }
    assert len(unique_generations) == writer.generate.call_count == 4
    assert len({view.snapshots[0].sha256 for view in prepared.views}) == 2
    assert writer.inputs.cached.model_dump_json() == writer.initial_cache
    assert all(view.model_dump_json() == frozen for view, frozen in writer.completed)
    assert len({id(cached.calls) for cached in writer.supplied_caches}) == 4
    assert all(cached is not writer.inputs.cached for cached in writer.supplied_caches)
    assert [len(cached.calls) for cached in writer.supplied_caches] == [1, 5, 5, 5]
    assert all(
        view.calls is not cached.calls
        for view in prepared.views
        for cached in writer.supplied_caches
    )
    assert {
        request
        for view in prepared.views
        for item in view.projections
        for request in item.request_hashes
    } == {call.request_sha256 for view in prepared.views for call in view.calls}


def test_writer_local_generations_do_not_leak_into_a_separate_preparation(
    writer: WriterContextHarness,
) -> None:
    first = writer.prepare()
    assert writer.generate.call_count == 4
    second = writer.prepare()
    assert writer.generate.call_count == 8
    for before, after in zip(first.views, second.views, strict=True):
        assert before.model_copy(update={"calls": []}) == after.model_copy(
            update={"calls": []}
        )
        assert {call.request_sha256: call for call in before.calls} == {
            call.request_sha256: call for call in after.calls
        }
    assert writer.inputs.cached.model_dump_json() == writer.initial_cache


def test_writer_different_index_model_configuration_cannot_reuse_local_generations(
    writer: WriterContextHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    future_llm = MagicMock(
        config=writer.llm.config.model_copy(update={"temperature": 0.7})
    )
    monkeypatch.setattr(
        writer_projection,
        "resolve_review_context_llm",
        lambda settings, _: writer.llm if settings.id == 1 else future_llm,
    )

    prepared = writer.prepare()

    assert len(prepared.views) == 4
    assert writer.generate.call_count == 8
    current_configs = {
        call.config_sha256 for view in prepared.views[:2] for call in view.calls
    }
    future_configs = {
        call.config_sha256 for view in prepared.views[2:] for call in view.calls
    }
    assert len(current_configs) == len(future_configs) == 2
    assert current_configs.isdisjoint(future_configs)
    assert {call.prompt_json for view in prepared.views[:2] for call in view.calls} == {
        call.prompt_json for view in prepared.views[2:] for call in view.calls
    }


@pytest.mark.parametrize(
    "change", ["prompt", "model_name", "api_base", "custom_config"]
)
def test_writer_revalidates_exact_prompt_and_configuration_before_reusing_retained_calls(
    writer: WriterContextHarness, change: str
) -> None:
    before = writer.prepare()
    retained = PreparedContextView(calls=list(before.views[0].calls))
    current = replace(writer.inputs, cached=retained)
    if change == "prompt":
        current = replace(
            current,
            canonical=[
                row.model_copy(update={"text": "Changed first legal rule"})
                if row.id == "row-0"
                else row
                for row in current.canonical
            ],
        )
    else:
        values: dict[str, object] = {
            "model_name": "other-context-model",
            "api_base": "https://other-context.example/v1",
            "custom_config": {"configured_option": "changed"},
        }
        writer.llm.config = writer.llm.config.model_copy(
            update={change: values[change]}
        )
    frozen_retained = retained.model_dump_json()
    after = writer.prepare(current)

    assert writer.generate.call_count == (6 if change == "prompt" else 8)
    assert retained.model_dump_json() == frozen_retained
    assert all(len(view.calls) == 4 for view in after.views)
    old_requests = {call.request_sha256 for call in before.views[0].calls}
    new_requests = {call.request_sha256 for call in after.views[0].calls}
    assert len(old_requests & new_requests) == (2 if change == "prompt" else 0)
    assert writer.inputs.cached.model_dump_json() == writer.initial_cache


def test_writer_failed_view_does_not_retain_partial_or_empty_provider_output(
    writer: WriterContextHarness,
) -> None:
    def incomplete(*, llm: LLM, flow: LLMFlow, prompt: LanguageModelInput) -> str:
        assert llm is writer.llm
        assert prompt
        return (
            "Generated contextual note."
            if flow == LLMFlow.CONTEXTUAL_RAG_DOC_SUMMARY
            else "   "
        )

    writer.generate.side_effect = incomplete
    with pytest.raises(indexing_pipeline.ContextualEnrichmentError) as raised:
        writer.prepare()
    assert isinstance(raised.value.__cause__, ValueError)
    assert str(raised.value.__cause__) == "context_generation_incomplete"
    failed_prompts = [
        call.kwargs["prompt"]
        for call in writer.generate.call_args_list
        if call.kwargs["flow"] == LLMFlow.CONTEXTUAL_RAG_CHUNK_CONTEXT
    ]
    assert failed_prompts
    assert writer.completed == []
    assert writer.inputs.cached.model_dump_json() == writer.initial_cache

    writer.generate.side_effect = None
    writer.generate.reset_mock()
    recovered = writer.prepare()

    assert len(recovered.views) == 4
    assert writer.generate.call_count == 4
    assert all(
        any(call.kwargs["prompt"] == prompt for call in writer.generate.call_args_list)
        for prompt in failed_prompts
    )
    assert all(
        call.output == "Generated contextual note."
        for view in recovered.views
        for call in view.calls
    )
    assert writer.inputs.cached.model_dump_json() == writer.initial_cache
