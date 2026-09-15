from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from onyx.llm.interfaces import LLMConfig
from onyx.llm.models import UserMessage
from onyx.tracing.flows import LLMFlow

if TYPE_CHECKING:
    from onyx.db.models import RegulatoryChunk, RegulatoryIndexingJob
    from onyx.natural_language_processing.search_nlp_models import EmbeddingModel
    from onyx.natural_language_processing.utils import BaseTokenizer
    from onyx.regulatory.amendments.annexes.models import ContextSourceSnapshot
    from onyx.regulatory.indexing_jobs.contextual import ContextualRequestFactory


def test_recorder_reuses_exact_prompt_with_same_config_and_records_changed_input() -> (
    None
):
    from onyx.regulatory.amendments.annexes.context_dependencies import (
        ContextGenerationRecorder,
    )

    llm = MagicMock()
    llm.config = LLMConfig(
        model_provider="configured",
        model_name="context",
        temperature=0,
        max_input_tokens=10000,
    )
    generate = MagicMock(return_value="same output")
    first = ContextGenerationRecorder()
    first.invoke(
        llm=llm,
        flow=LLMFlow.CONTEXTUAL_RAG_DOC_SUMMARY,
        prompt=UserMessage(content="old document"),
        generate=generate,
        consumer_id="a",
        stage="summary",
        source_text="old document",
        token_budget=100,
    )
    second = ContextGenerationRecorder(cached_calls=list(first.calls.values()))
    second.invoke(
        llm=llm,
        flow=LLMFlow.CONTEXTUAL_RAG_DOC_SUMMARY,
        prompt=UserMessage(content="old document"),
        generate=generate,
        consumer_id="b",
        stage="summary",
        source_text="old document",
        token_budget=100,
    )
    assert generate.call_count == 1
    second.invoke(
        llm=llm,
        flow=LLMFlow.CONTEXTUAL_RAG_DOC_SUMMARY,
        prompt=UserMessage(content="new document"),
        generate=generate,
        consumer_id="b",
        stage="summary",
        source_text="new document",
        token_budget=100,
    )
    assert generate.call_count == 2
    assert len(second.calls) == 2


def test_context_candidates_and_embedding_changes_are_distinct() -> None:
    from onyx.regulatory.amendments.annexes.context_dependencies import (
        compare_context_views,
    )
    from onyx.regulatory.amendments.annexes.models import (
        FrozenContextProjection,
        PreparedContextView,
    )

    def projection(
        identifier: str, request: str, embedding: str
    ) -> FrozenContextProjection:
        return FrozenContextProjection(
            vector_reuse_verified=True,
            canonical_chunk_id=identifier,
            source_snapshot_sha256="snapshot",
            generation_path="normal",
            request_hashes=[request],
            embedding_input_sha256=embedding,
            embedding_config_sha256="model",
            embedding_texts=["exact input"],
            canonical_text_sha256="text",
            metadata_sha256="meta",
        )

    old = PreparedContextView(
        projections=[
            projection("annex", "old", "old"),
            projection("outside", "old", "same"),
            projection("far", "old", "old"),
        ]
    )
    new = PreparedContextView(
        projections=[
            projection("annex", "new", "new"),
            projection("outside", "new", "same"),
            projection("far", "new", "new"),
        ]
    )
    result = compare_context_views(old=old, new=new, direct_canonical_changes=["annex"])
    assert result.contextual_candidates == ["annex", "far", "outside"]
    assert result.embedding_changes == ["annex", "far"]
    assert result.context_only == ["far", "outside"]
    assert result.reasons["outside"] == [
        "context_input_changed",
        "embedding_input_unchanged",
    ]


def test_unknown_legacy_embedding_provenance_never_claims_vector_reuse() -> None:
    from onyx.regulatory.amendments.annexes.context_dependencies import (
        compare_context_views,
    )
    from onyx.regulatory.amendments.annexes.models import (
        FrozenContextProjection,
        PreparedContextView,
    )

    projection = FrozenContextProjection(
        canonical_chunk_id="a",
        source_snapshot_sha256="snapshot",
        generation_path="normal",
        request_hashes=[],
        embedding_input_sha256="same",
        embedding_config_sha256="model",
        embedding_texts=["text"],
        canonical_text_sha256="text",
        metadata_sha256="meta",
    )
    result = compare_context_views(
        old=PreparedContextView(),
        new=PreparedContextView(projections=[projection]),
        direct_canonical_changes=[],
    )
    assert result.contextual_candidates == ["a"] and result.embedding_changes == ["a"]
    assert "legacy_provenance_unavailable" in result.reasons["a"]
    assert result.context_only == ["a"]


@pytest.mark.parametrize("full_inclusion,expected_calls", [(True, 8), (False, 6)])
def test_actual_normal_path_replays_all_consumers_and_reuses_equal_embedding_outputs(
    full_inclusion: bool, expected_calls: int
) -> None:
    from datetime import date
    from unittest.mock import patch
    from uuid import uuid4

    from onyx.db.models import RegulatoryChunk, SearchSettings, UserFile
    from onyx.natural_language_processing.utils import BaseTokenizer
    from onyx.regulatory.amendments.annexes.context_dependencies import (
        compare_context_views,
    )
    from onyx.regulatory.projection import prepare_normal_context_view

    class Tokenizer(BaseTokenizer):
        def encode(self, string: str) -> list[int]:
            return list(string.encode())

        def decode(self, tokens: list[int]) -> str:
            return bytes(tokens).decode(errors="ignore")

        def tokenize(self, string: str) -> list[str]:
            return list(string)

    file = UserFile(id=uuid4(), name="Regulation")

    def rows(rate: str) -> list[RegulatoryChunk]:
        return [
            RegulatoryChunk(
                id=identifier,
                user_file_id=file.id,
                text=text,
                position=position,
                projection_ordinal=position,
                heading_path=[heading],
                chunk_type="text",
                chunk_metadata={},
                source="indexed",
                status="active",
                validity_start_date=date(2026, 1, 1),
            )
            for position, (identifier, heading, text) in enumerate(
                [
                    ("a", "EK-1", rate),
                    ("b", "Outside annex", "Unchanged distant rule"),
                    ("c", "Outside annex", "Another rule"),
                ]
            )
        ]

    settings = SearchSettings(
        id=1,
        model_name="embedding",
        model_dim=3,
        normalize=True,
        enable_contextual_rag=True,
    )
    embedder = MagicMock()
    embedder.embedding_model.tokenizer = Tokenizer()
    embedder.embedding_model.provider_type = None
    embedder.embedding_model.model_name = "embedding"
    embedder.embedding_model.normalize = True
    embedder.embedding_model.passage_prefix = None
    embedder.embedding_model.retrim_content = True
    embedder.embedding_model.reduced_dimension = None
    embedder.embedding_model.api_url = None
    embedder.embedding_model.api_version = None
    embedder.embedding_model.deployment_name = None
    llm = MagicMock()
    llm.config = LLMConfig(
        model_provider="configured",
        model_name="context",
        temperature=0,
        max_input_tokens=10000,
    )
    with (
        patch(
            "onyx.regulatory.projection._get_contextual_tokenizer",
            return_value=Tokenizer(),
        ),
        patch(
            "onyx.indexing.indexing_pipeline.MAX_TOKENS_FOR_FULL_INCLUSION",
            10000 if full_inclusion else 0,
        ),
        patch("onyx.indexing.indexing_pipeline.USE_DOCUMENT_SUMMARY", True),
        patch("onyx.indexing.indexing_pipeline.USE_CHUNK_SUMMARY", True),
        patch("onyx.regulatory.projection.USE_CHUNK_SUMMARY", True),
        patch(
            "onyx.indexing.indexing_pipeline._invoke_contextual_llm_with_retry",
            return_value="Same generated context",
        ) as generate,
    ):
        before = prepare_normal_context_view(
            rows=rows("5%"),
            user_file=file,
            search_settings=settings,
            embedder=embedder,
            llm=llm,
        )
        after = prepare_normal_context_view(
            rows=rows("7%"),
            user_file=file,
            search_settings=settings,
            embedder=embedder,
            llm=llm,
            cached=before,
        )
    # This fixture represents the frozen inputs of an already verified index publication.
    before = before.model_copy(
        update={
            "projections": [
                projection.model_copy(update={"vector_reuse_verified": True})
                for projection in before.projections
            ]
        }
    )
    impact = compare_context_views(
        old=before, new=after, direct_canonical_changes=["a"]
    )
    assert impact.contextual_candidates == ["a", "b", "c"]
    assert impact.embedding_changes == ["a"]
    assert len(before.snapshots) == 1
    assert len(before.calls) == 4 and len(after.calls) == 4
    assert len(before.snapshots[0].ordered_ranges) == 3
    assert generate.call_count == expected_calls
    assert before.projections[1].embedding_texts == after.projections[1].embedding_texts


def test_actual_durable_path_captures_boundary_and_exact_no_delimiter_embedding() -> (
    None
):
    from unittest.mock import patch

    from onyx.regulatory.indexing_jobs.contextual import prepare_durable_context_view
    from tests.unit.onyx.regulatory.indexing_jobs.test_contextual import (
        _CharacterTokenizer,
        _job,
        _row,
    )

    job = _job()
    job.config_snapshot.update(
        {
            "embedding_provider": "openrouter",
            "embedding_model_name": "embedding",
            "effective_dimension": 3,
        }
    )
    rows = [
        _row(job, row_id="a", position=1, text="5%", heading_path=["EK-1"]),
        _row(
            job, row_id="b", position=2, text="Outside", heading_path=["Last heading"]
        ),
    ]
    for row in rows:
        row.chunk_metadata = {}
    generate = MagicMock(return_value="context")
    with patch(
        "onyx.regulatory.indexing_jobs.contextual._contextual_safe_input_limit",
        return_value=1200,
    ):
        before = prepare_durable_context_view(
            job=job,
            rows=rows,
            embedding_tokenizer=_CharacterTokenizer(),
            contextual_tokenizer=_CharacterTokenizer(),
            generate=generate,
            embedding_model=_durable_embedding_model(),
        )
        after = prepare_durable_context_view(
            job=job,
            rows=rows,
            embedding_tokenizer=_CharacterTokenizer(),
            contextual_tokenizer=_CharacterTokenizer(),
            generate=generate,
            embedding_model=_durable_embedding_model(),
            cached=before,
        )
    assert generate.call_count == 2
    assert before.projections[0].embedding_texts == [
        before.projections[0].doc_summary + "5%"
    ]
    assert before.projections[0].doc_summary.endswith("\n\r\n")
    assert before.projections[1].embedding_texts == [
        before.projections[1].doc_summary + "Outside"
    ]
    assert "Canonical position: 2" in before.calls[0].prompt_json
    assert before.projections == after.projections


def test_unverified_old_context_is_not_current_index_evidence() -> None:
    from onyx.regulatory.amendments.annexes.models import FrozenContextProjection

    projection = FrozenContextProjection(
        canonical_chunk_id="a",
        source_snapshot_sha256="snapshot",
        generation_path="normal",
        request_hashes=[],
        embedding_input_sha256="input",
        embedding_config_sha256="config",
        embedding_texts=["text"],
        canonical_text_sha256="text",
        metadata_sha256="meta",
    )
    assert not projection.vector_reuse_verified


def test_transitive_dependencies_include_aggregates_and_supporting_images() -> None:
    from onyx.db.models import RegulatoryChunk
    from onyx.regulatory.amendments.annexes.context_dependencies import (
        canonical_dependency_closure,
    )

    rows = [
        RegulatoryChunk(id="cell", chunk_metadata={}),
        RegulatoryChunk(
            id="row", chunk_metadata={"source_regulatory_chunk_ids": ["cell"]}
        ),
        RegulatoryChunk(
            id="section", chunk_metadata={"source_regulatory_chunk_ids": ["row"]}
        ),
        RegulatoryChunk(
            id="image", chunk_metadata={"bound_to_regulatory_chunk_id": "cell"}
        ),
        RegulatoryChunk(id="outside", chunk_metadata={}),
    ]
    assert canonical_dependency_closure(rows, ["cell"]) == [
        "cell",
        "image",
        "row",
        "section",
    ]


def test_existing_index_proof_allows_reuse_without_inventing_active_history() -> None:
    import pytest

    from onyx.regulatory.amendments.annexes.context_dependencies import (
        verify_existing_index_evidence,
    )
    from onyx.regulatory.amendments.annexes.models import (
        ExistingIndexEmbeddingEvidence,
        FrozenContextProjection,
    )

    projection = FrozenContextProjection(
        canonical_chunk_id="a",
        source_snapshot_sha256="snapshot",
        generation_path="normal",
        request_hashes=[],
        embedding_input_sha256="input",
        embedding_config_sha256="config",
        embedding_texts=["text"],
        canonical_text_sha256="text",
        metadata_sha256="meta",
    )
    proof = ExistingIndexEmbeddingEvidence(
        index_name="present",
        document_id="file",
        canonical_chunk_id="a",
        projection_ordinal=42,
        embedding_input_sha256="input",
        embedding_config_sha256="config",
        canonical_text_sha256="text",
        vector_dimension=3,
        expected_dimension=3,
    )
    verified = verify_existing_index_evidence(projection, proof)
    assert verified.vector_reuse_verified and verified.projection_id is None
    assert verified.existing_index_evidence["projection_ordinal"] == 42
    with pytest.raises(ValueError, match="existing index"):
        verify_existing_index_evidence(
            projection, proof.model_copy(update={"canonical_chunk_id": "wrong"})
        )


def test_durable_preparation_records_actual_request_and_shared_snapshot() -> None:
    from unittest.mock import patch

    from onyx.regulatory.indexing_jobs.contextual import ContextualRequestFactory
    from tests.unit.onyx.regulatory.indexing_jobs.test_contextual import (
        _CharacterTokenizer,
        _job,
        _row,
    )

    job = _job()
    rows = [
        _row(job, row_id="a", position=1, text="5%", heading_path=["EK-1"]),
        _row(job, row_id="b", position=2, text="Outside", heading_path=["Tail"]),
    ]
    factory = ContextualRequestFactory(
        job=job,
        rows=rows,
        contextual_tokenizer=_CharacterTokenizer(),
        embedding_tokenizer=_CharacterTokenizer(),
    )
    with patch(
        "onyx.regulatory.indexing_jobs.contextual._contextual_safe_input_limit",
        return_value=1200,
    ):
        request = factory.request(rows[0])
        provenance = factory.request_provenance(rows[0], request)
        first = factory.source_snapshot(rows[0])
        second = factory.source_snapshot(rows[1])
    assert provenance["request_hash"] == request.request_hash
    assert provenance["prompt"] == request.prompt
    assert provenance["source_snapshot_sha256"] == first.sha256
    assert first is second
    assert [span.canonical_chunk_id for span in first.ordered_ranges] == ["a", "b"]


def test_equal_actual_prompt_reuses_context_when_only_unused_budget_changes() -> None:
    from onyx.regulatory.amendments.annexes.context_dependencies import (
        ContextGenerationRecorder,
    )

    llm = MagicMock()
    llm.config = LLMConfig(
        model_provider="configured",
        model_name="context",
        temperature=0,
        max_input_tokens=10000,
    )
    generate = MagicMock(side_effect=["cached output", "different output"])
    recorder = ContextGenerationRecorder()
    outputs = [
        recorder.invoke(
            llm=llm,
            flow=LLMFlow.CONTEXTUAL_RAG_DOC_SUMMARY,
            prompt=UserMessage(content="same actual prompt"),
            generate=generate,
            consumer_id="a",
            stage="summary",
            source_text="same actual prompt",
            token_budget=budget,
        )
        for budget in (100, 200)
    ]
    assert outputs == ["cached output", "cached output"]
    assert generate.call_count == 1
    assert {call.token_budget for call in recorder.calls.values()} == {100, 200}


def test_transitive_aggregate_preparation_preserves_canonical_objects() -> None:
    from onyx.db.models import RegulatoryChunk
    from onyx.regulatory.amendments.annexes.context_dependencies import (
        rebuild_context_aggregates,
    )

    leaf = RegulatoryChunk(
        id="cell", text="A | 7%", chunk_metadata={}, heading_path=["Table"], position=0
    )
    row = RegulatoryChunk(
        id="row",
        text="A | 5%",
        chunk_metadata={
            "chunk_variant": "hierarchical_aggregate",
            "hierarchy_root_path": ["Table"],
            "source_regulatory_chunk_ids": ["cell"],
        },
        heading_path=["Table"],
        position=1,
    )
    section = RegulatoryChunk(
        id="section",
        text="Old section",
        chunk_metadata={
            "chunk_variant": "hierarchical_aggregate",
            "hierarchy_root_path": ["Annex"],
            "source_regulatory_chunk_ids": ["row"],
        },
        heading_path=["Annex"],
        position=2,
    )
    prepared = rebuild_context_aggregates([leaf, row, section], changed_ids=["cell"])
    assert prepared[1].text == "Table\n\nA | 7%"
    assert prepared[2].text == "Annex\n\nTable\n\nA | 7%"
    assert row.text == "A | 5%" and section.text == "Old section"


def test_durable_snapshot_identity_includes_context_reference_date() -> None:
    from datetime import date

    from onyx.regulatory.indexing_jobs.contextual import ContextualRequestFactory
    from tests.unit.onyx.regulatory.indexing_jobs.test_contextual import (
        _CharacterTokenizer,
        _job,
        _row,
    )

    job = _job()
    rows = [_row(job, row_id="a", position=1, text="Body", heading_path=["EK-1"])]
    factory = ContextualRequestFactory(
        job=job,
        rows=rows,
        contextual_tokenizer=_CharacterTokenizer(),
        reference_date_override=date(2026, 9, 9),
    )
    before = factory.source_snapshot(rows[0])
    factory.reference_date_override = date(2026, 9, 10)
    after = factory.source_snapshot(rows[0])
    assert before.text == after.text
    assert before.sha256 != after.sha256
    assert after.reference_date == date(2026, 9, 10)


def test_expired_normal_view_prepares_retirement_without_context_generation() -> None:
    from datetime import date
    from unittest.mock import patch
    from uuid import uuid4

    from onyx.db.models import RegulatoryChunk, SearchSettings, UserFile
    from onyx.regulatory.amendments.annexes.context_dependencies import (
        compare_context_views,
    )
    from onyx.regulatory.amendments.annexes.models import (
        FrozenContextProjection,
        PreparedContextView,
    )
    from onyx.regulatory.projection import prepare_normal_context_view

    file = UserFile(id=uuid4(), name="Expiry fixture")
    row = RegulatoryChunk(
        id="expired",
        user_file_id=file.id,
        text="Old rule",
        position=1,
        projection_ordinal=1,
        heading_path=[],
        chunk_metadata={},
        validity_start_date=date(2026, 1, 1),
        validity_end_date=date(2026, 9, 10),
    )
    before = PreparedContextView(
        projections=[
            FrozenContextProjection(
                canonical_chunk_id=row.id,
                source_snapshot_sha256="source",
                generation_path="normal",
                request_hashes=[],
                embedding_input_sha256="input",
                embedding_config_sha256="config",
                embedding_texts=[row.text],
                canonical_text_sha256="text",
                metadata_sha256="metadata",
            )
        ]
    )
    with patch(
        "onyx.regulatory.projection.effective_contextual_rag_enabled", return_value=True
    ):
        after = prepare_normal_context_view(
            rows=[row],
            user_file=file,
            search_settings=SearchSettings(),
            embedder=MagicMock(),
            llm=MagicMock(),
            as_of_date=date(2026, 9, 10),
        )
    assert after == PreparedContextView()
    impact = compare_context_views(old=before, new=after, direct_canonical_changes=[])
    assert impact.ready and impact.retire_history == [row.id]
    assert not impact.embedding_changes and not impact.contextual_candidates


def _durable_embedding_model() -> EmbeddingModel:
    from unittest.mock import patch

    from onyx.natural_language_processing.search_nlp_models import EmbeddingModel
    from shared_configs.enums import EmbeddingProvider
    from tests.unit.onyx.regulatory.indexing_jobs.test_contextual import (
        _CharacterTokenizer,
    )

    with patch(
        "onyx.natural_language_processing.search_nlp_models.get_tokenizer",
        return_value=_CharacterTokenizer(),
    ):
        return EmbeddingModel(
            server_host="localhost",
            server_port=9000,
            model_name="embedding",
            normalize=True,
            query_prefix=None,
            passage_prefix=None,
            api_key=None,
            api_url="https://encoder.example/v1",
            provider_type=EmbeddingProvider.OPENROUTER,
            retrim_content=True,
            reduced_dimension=3,
            api_version="v1",
            deployment_name="deployment",
        )


def _durable_fixture() -> tuple[RegulatoryIndexingJob, RegulatoryChunk, BaseTokenizer]:
    from shared_configs.enums import EmbeddingProvider
    from tests.unit.onyx.regulatory.indexing_jobs.test_contextual import (
        _CharacterTokenizer,
        _job,
        _row,
    )

    job = _job()
    job.config_snapshot.update(
        {
            "embedding_provider": EmbeddingProvider.OPENROUTER.value,
            "embedding_model_name": "embedding",
            "effective_dimension": 3,
        }
    )
    row = _row(job, row_id="a", position=1, text="5%", heading_path=["EK-1"])
    row.chunk_metadata = {}
    return job, row, _CharacterTokenizer()


@pytest.mark.parametrize(
    "attribute,value",
    [
        ("normalize", False),
        ("passage_prefix", "passage: "),
        ("api_url", "https://other.example/v1"),
        ("api_version", "v2"),
        ("deployment_name", "other-deployment"),
    ],
)
def test_durable_actual_encoder_configuration_change_cannot_reuse_vector(
    attribute: str, value: object
) -> None:
    from unittest.mock import patch

    from onyx.regulatory.amendments.annexes.context_dependencies import (
        compare_context_views,
    )
    from onyx.regulatory.indexing_jobs.contextual import prepare_durable_context_view

    job, row, tokenizer = _durable_fixture()
    model = _durable_embedding_model()
    generate = MagicMock(return_value="context")
    with patch(
        "onyx.regulatory.indexing_jobs.contextual._contextual_safe_input_limit",
        return_value=1200,
    ):
        before = prepare_durable_context_view(
            job=job,
            rows=[row],
            embedding_tokenizer=tokenizer,
            contextual_tokenizer=tokenizer,
            generate=generate,
            embedding_model=model,
        )
        setattr(model, attribute, value)
        after = prepare_durable_context_view(
            job=job,
            rows=[row],
            embedding_tokenizer=tokenizer,
            contextual_tokenizer=tokenizer,
            generate=generate,
            embedding_model=model,
            cached=before,
        )
    # A lone canonical row has no external context to generate.
    assert generate.call_count == 0
    before = before.model_copy(
        update={
            "projections": [
                before.projections[0].model_copy(update={"vector_reuse_verified": True})
            ]
        }
    )
    impact = compare_context_views(old=before, new=after, direct_canonical_changes=[])
    assert impact.embedding_changes == [row.id]
    assert impact.contextual_candidates == [row.id]
    assert "embedding_configuration_changed" in impact.reasons[row.id]
    assert (
        before.projections[0].embedding_input_sha256
        == after.projections[0].embedding_input_sha256
    )


def test_durable_missing_actual_encoder_blocks_before_generation() -> None:
    from onyx.regulatory.indexing_jobs.contextual import prepare_durable_context_view

    job, row, tokenizer = _durable_fixture()
    generate = MagicMock(return_value="context")
    with pytest.raises(ValueError, match="embedding_configuration_unavailable"):
        prepare_durable_context_view(
            job=job,
            rows=[row],
            embedding_tokenizer=tokenizer,
            contextual_tokenizer=tokenizer,
            generate=generate,
        )
    generate.assert_not_called()


def test_durable_freezes_actual_encoder_text_and_reuses_factory_snapshots() -> None:
    from unittest.mock import patch

    from onyx.regulatory.indexing_jobs.contextual import (
        ContextualRequestFactory,
        prepare_durable_context_view,
    )
    from shared_configs.enums import EmbedTextType

    job, row, tokenizer = _durable_fixture()
    row.text = "Long legal text exceeds limit"
    model = _durable_embedding_model()
    observed: list[ContextSourceSnapshot] = []
    source_snapshot = ContextualRequestFactory.source_snapshot

    def capture(
        factory: ContextualRequestFactory, target: RegulatoryChunk
    ) -> ContextSourceSnapshot:
        result = source_snapshot(factory, target)
        observed.append(result)
        return result

    with (
        patch(
            "onyx.regulatory.indexing_jobs.contextual._contextual_safe_input_limit",
            return_value=1200,
        ),
        patch.object(ContextualRequestFactory, "reserve", return_value=0),
        patch.object(ContextualRequestFactory, "source_snapshot", capture),
        patch("shared_configs.configs.DOC_EMBEDDING_CONTEXT_SIZE", 8),
    ):
        prepared = prepare_durable_context_view(
            job=job,
            rows=[row],
            embedding_tokenizer=tokenizer,
            contextual_tokenizer=tokenizer,
            generate=MagicMock(),
            embedding_model=model,
        )
    with patch.object(
        model, "_batch_encode_texts", return_value=[[1.0, 0.0, 0.0]]
    ) as encode:
        model.encode([row.text], text_type=EmbedTextType.PASSAGE, max_seq_length=8)
    assert prepared.projections[0].embedding_texts == encode.call_args.kwargs["texts"]
    assert prepared.projections[0].embedding_texts != [row.text]
    assert len(observed) == 1 and prepared.snapshots[0] is observed[0]


def test_durable_batch_freezes_raw_payload_and_ignores_unused_synchronous_settings() -> (
    None
):
    from unittest.mock import patch

    from onyx.regulatory.amendments.annexes.context_dependencies import (
        compare_context_views,
    )
    from onyx.regulatory.indexing_jobs.contextual import prepare_durable_context_view
    from onyx.regulatory.indexing_jobs.models import OpenRouterBatchConfig
    from onyx.regulatory.indexing_jobs.openrouter_batch import (
        OpenRouterEmbeddingBatchRequest,
        openrouter_embedding_payload,
    )

    job, row, tokenizer = _durable_fixture()
    row.text = "Long unchanged legal text outside the encoder trim"
    config = OpenRouterBatchConfig(
        api_url="https://batch.example/api/beta/batches",
        model_name="embedding",
        effective_dimension=3,
    )
    job.config_snapshot["openrouter_batch"] = config.model_dump(mode="json")
    model = _durable_embedding_model()
    model.normalize = False
    model.passage_prefix = "UNUSED: "
    with (
        patch(
            "onyx.regulatory.indexing_jobs.contextual._contextual_safe_input_limit",
            return_value=1200,
        ),
        patch("shared_configs.configs.DOC_EMBEDDING_CONTEXT_SIZE", 8),
    ):
        before = prepare_durable_context_view(
            job=job,
            rows=[row],
            embedding_tokenizer=tokenizer,
            contextual_tokenizer=tokenizer,
            generate=MagicMock(),
        )
        same = prepare_durable_context_view(
            job=job,
            rows=[row],
            embedding_tokenizer=tokenizer,
            contextual_tokenizer=tokenizer,
            generate=MagicMock(),
            embedding_model=model,
        )
        job.config_snapshot["openrouter_batch"] = config.model_copy(
            update={"api_url": "https://other.example/api/beta/batches"}
        ).model_dump(mode="json")
        after = prepare_durable_context_view(
            job=job,
            rows=[row],
            embedding_tokenizer=tokenizer,
            contextual_tokenizer=tokenizer,
            generate=MagicMock(),
        )
    assert same == before
    assert before.projections[0].embedding_texts == [row.text]
    request = OpenRouterEmbeddingBatchRequest(custom_id="fixture", inputs=[row.text])
    assert openrouter_embedding_payload([request], config=config) == {
        "endpoint": "/v1/embeddings",
        "model": "embedding",
        "requests": [
            {
                "custom_id": "fixture",
                "body": {
                    "model": "embedding",
                    "input": before.projections[0].embedding_texts,
                    "dimensions": 3,
                },
            }
        ],
    }
    verified = before.model_copy(
        update={
            "projections": [
                before.projections[0].model_copy(update={"vector_reuse_verified": True})
            ]
        }
    )
    impact = compare_context_views(old=verified, new=after, direct_canonical_changes=[])
    assert impact.contextual_candidates == [row.id] == impact.embedding_changes


@pytest.mark.parametrize(
    "attribute,value",
    [("model_name", "wrong"), ("reduced_dimension", 8), ("provider_type", None)],
)
def test_durable_model_snapshot_mismatch_blocks_generation(
    attribute: str, value: object
) -> None:
    from onyx.regulatory.indexing_jobs.contextual import prepare_durable_context_view

    job, row, tokenizer = _durable_fixture()
    model = _durable_embedding_model()
    setattr(model, attribute, value)
    generate = MagicMock()
    with pytest.raises(ValueError, match="embedding_configuration_mismatch"):
        prepare_durable_context_view(
            job=job,
            rows=[row],
            embedding_tokenizer=tokenizer,
            contextual_tokenizer=tokenizer,
            generate=generate,
            embedding_model=model,
        )
    generate.assert_not_called()


def test_expired_durable_view_prepares_retirement_without_encoder_or_generation() -> (
    None
):
    from datetime import date

    from onyx.regulatory.amendments.annexes.context_dependencies import (
        compare_context_views,
    )
    from onyx.regulatory.amendments.annexes.models import (
        FrozenContextProjection,
        PreparedContextView,
    )
    from onyx.regulatory.indexing_jobs.contextual import prepare_durable_context_view

    job, row, tokenizer = _durable_fixture()
    row.validity_start_date = date(2026, 1, 1)
    row.validity_end_date = date(2026, 9, 10)
    before = PreparedContextView(
        projections=[
            FrozenContextProjection(
                canonical_chunk_id=row.id,
                source_snapshot_sha256="source",
                generation_path="durable",
                request_hashes=[],
                embedding_input_sha256="input",
                embedding_config_sha256="config",
                embedding_texts=[row.text],
                canonical_text_sha256="text",
                metadata_sha256="metadata",
            )
        ]
    )
    generate = MagicMock()
    after = prepare_durable_context_view(
        job=job,
        rows=[row],
        embedding_tokenizer=tokenizer,
        contextual_tokenizer=tokenizer,
        generate=generate,
        as_of_date=date(2026, 9, 10),
    )
    assert after == PreparedContextView()
    generate.assert_not_called()
    impact = compare_context_views(old=before, new=after, direct_canonical_changes=[])
    assert impact.ready and impact.retire_history == [row.id]
    assert not impact.embedding_changes and not impact.contextual_candidates


def test_durable_review_context_has_bounded_parallelism_and_exact_cache_reuse() -> None:
    from contextvars import ContextVar
    from threading import Barrier
    from unittest.mock import patch

    from onyx.regulatory.indexing_jobs.contextual import (
        ContextualRequestFactory,
        prepare_durable_context_view,
    )
    from tests.unit.onyx.regulatory.indexing_jobs.test_contextual import _row

    job, _, tokenizer = _durable_fixture()
    rows = [
        _row(
            job,
            row_id=str(index),
            position=index,
            text=f"Legal clause {index}",
            heading_path=["EK-1"],
        )
        for index in range(8)
    ]
    for row in rows:
        row.chunk_metadata = {}
    barrier = Barrier(4, timeout=5)
    tenant = ContextVar("test_review_context_tenant", default="wrong")
    token = tenant.set("scoped-tenant")

    def generate(_request: object) -> str:
        assert tenant.get() == "scoped-tenant"
        barrier.wait()
        return "Verified context"

    progress: list[tuple[int, int]] = []
    try:
        with (
            patch.object(ContextualRequestFactory, "reserve", return_value=256),
            patch(
                "onyx.regulatory.indexing_jobs.contextual._contextual_safe_input_limit",
                return_value=1200,
            ),
        ):
            prepared = prepare_durable_context_view(
                job=job,
                rows=rows,
                embedding_tokenizer=tokenizer,
                contextual_tokenizer=tokenizer,
                generate=generate,
                embedding_model=_durable_embedding_model(),
                max_workers=4,
                progress=lambda completed, total: progress.append((completed, total)),
            )
            unused = MagicMock(
                side_effect=AssertionError("completed prompts must not be regenerated")
            )
            replay = prepare_durable_context_view(
                job=job,
                rows=rows,
                embedding_tokenizer=tokenizer,
                contextual_tokenizer=tokenizer,
                generate=unused,
                embedding_model=_durable_embedding_model(),
                max_workers=4,
                cached=prepared,
            )
            assert prepared == replay
            unused.assert_not_called()
    finally:
        tenant.reset(token)
    assert progress == [(index, 8) for index in range(1, 9)]
