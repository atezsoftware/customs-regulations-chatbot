import json
from dataclasses import replace
from datetime import date
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from onyx.db.enums import IndexModelStatus
from onyx.db.models import SearchSettings
from onyx.document_index.publication_models import (
    IndexedProjectionEvidence,
    ObservedPublicationProjection,
)
from onyx.regulatory import writer_projection
from tests.unit.onyx.regulatory.annexes.test_context_dependencies import (
    _durable_embedding_model,
)
from tests.unit.onyx.regulatory.indexing_jobs.owned_publication_test_helpers import (
    OwnedAuthority,
    canonical_row,
    writer_inputs,
)


class ProjectionAuthority(OwnedAuthority):
    def reservations(self, _owner: object) -> object:
        return object()


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize(
    "changed_count,scheduled", [(1, False), (4, False), (11, False), (1, True)]
)
def test_amendment_retains_unrelated_bindings_and_only_embeds_successors(
    changed_count: int, scheduled: bool, legacy: bool
) -> None:
    file_id = uuid4()
    authority = ProjectionAuthority(file_id)
    inputs = writer_inputs(
        file_id,
        [
            canonical_row(file_id, i, f"Article {i}: unchanged provision")
            for i in range(30)
        ],
    )
    if scheduled:
        inputs.canonical[29] = inputs.canonical[29].model_copy(
            update={"validity_end_date": date(2028, 1, 1)}
        )
    settings = SearchSettings(
        id=1,
        status=IndexModelStatus.PRESENT,
        index_name="selective-test",
        model_name="embedding",
        model_dim=3,
        normalize=True,
        enable_contextual_rag=scheduled,
    )
    inputs = replace(inputs, settings=[settings])
    model = _durable_embedding_model()
    embedder = MagicMock(embedding_model=model)
    client = MagicMock()
    client.indices.get.return_value = {
        settings.index_name: {"settings": {"index": {"uuid": "physical-test"}}}
    }
    from onyx.llm.interfaces import LLMConfig
    from onyx.regulatory.amendment_projection_impact import (
        ContextImpactDecision,
        ContextImpactResult,
    )

    llm = MagicMock()
    llm.config = LLMConfig(
        model_provider="configured",
        model_name="context",
        temperature=0,
        max_input_tokens=10000,
    )

    def unchanged_audit(*_args: object, **kwargs: object):
        return ContextImpactResult(
            decisions=[
                ContextImpactDecision(
                    key=key,
                    affected=False,
                    quote="",
                    reason="Generic context remains true.",
                )
                for key in json.loads(str(kwargs["user_prompt"]))["contexts"]
            ]
        )

    with (
        patch(
            "onyx.regulatory.amendment_projection_impact.generate_structured",
            side_effect=unchanged_audit,
        ),
        patch(
            "onyx.regulatory.projection._get_contextual_tokenizer",
            return_value=model.tokenizer,
        ),
        patch("onyx.indexing.indexing_pipeline.MAX_TOKENS_FOR_FULL_INCLUSION", 0),
        patch("onyx.indexing.indexing_pipeline.USE_DOCUMENT_SUMMARY", True),
        patch("onyx.indexing.indexing_pipeline.USE_CHUNK_SUMMARY", True),
        patch("onyx.regulatory.projection.USE_CHUNK_SUMMARY", True),
        patch(
            "onyx.indexing.indexing_pipeline._invoke_contextual_llm_with_retry",
            return_value="Generic import regulation context.",
        ),
        patch(
            "onyx.regulatory.projection.effective_contextual_rag_enabled",
            return_value=scheduled,
        ),
        patch.object(
            writer_projection, "PublicationStore", side_effect=authority.for_scope
        ),
        patch.object(
            writer_projection.DefaultIndexingEmbedder,
            "from_db_search_settings",
            return_value=embedder,
        ),
        patch.object(
            writer_projection,
            "resolve_review_context_llm",
            return_value=llm if scheduled else None,
        ),
        patch.object(
            model,
            "encode",
            side_effect=lambda texts, **_: [[0.1, 0.2, 0.3] for _ in texts],
        ) as encode,
    ):
        original = writer_projection.prepare_owned_correction(
            authority.owner, client, inputs, inputs.canonical, changed_id=None
        )
        if legacy:
            from onyx.regulatory.publication_baseline import observed_index_snapshot

            observation = observed_index_snapshot(settings, "physical-test")
            original = original.model_copy(
                update={
                    "bindings": [
                        b.model_copy(
                            update={
                                "index": observation,
                                "projection": ObservedPublicationProjection.observe(
                                    context_projection_id=str(b.id),
                                    source_json=b.projection.source_json,
                                    observed_index=observation,
                                ),
                                "context": None,
                            }
                        )
                        for b in original.bindings
                    ]
                }
            )
        inputs = replace(
            inputs,
            bindings=original.bindings,
            revisions={b.id: uuid4() for b in original.bindings},
        )
        changed = inputs.canonical[10 : 10 + changed_count]
        old = changed[0]
        changed_ids = {row.id for row in changed}
        boundary = date(2027, 1, 1)
        after = [
            r
            if r.id not in changed_ids
            else r.model_copy(
                update={"validity_end_date": boundary, "status": "superseded"}
            )
            for r in inputs.canonical
        ]
        after.extend(
            row.model_copy(
                update={
                    "id": f"successor-{i}",
                    "text": f"Article {10 + i}: approved new provision",
                    "validity_start_date": boundary,
                    "projection_ordinal": 40 + i,
                    "supersedes_chunk_id": row.id,
                }
            )
            for i, row in enumerate(changed)
        )
        evidence = [
            IndexedProjectionEvidence(
                index=b.index,
                source_json=b.projection.source_json,
                frozen_projection=None if legacy else b.projection,
                observed_projection=b.projection if legacy else None,
                payload_sha256=None,
            )
            for b in original.bindings
        ]
        encode.reset_mock()
        with patch(
            "onyx.document_index.elasticsearch.publication.FencedPublicationIndex.inventory_evidence",
            return_value=tuple(evidence),
        ):
            result = writer_projection.prepare_owned_correction(
                authority.owner,
                client,
                inputs,
                after,
                changed_id=None,
                selective_amendment=True,
            )
        retained = {
            b.id
            for b in original.bindings
            if json.loads(b.projection.source_json)["regulatory_chunk_id"]
            not in changed_ids
        }
        assert retained <= {b.id for b in result.bindings}
        assert len(result.bindings) == (59 if scheduled else 30) + changed_count
        assert len([b for b in result.bindings if b.id not in retained]) == (
            3 if scheduled else 2 * changed_count
        )
        assert encode.call_count == changed_count
        assert all(
            "approved new provision" in call.kwargs["texts"][0]
            for call in encode.call_args_list
        )
        assert {
            p.canonical_chunk_id for view in result.views for p in view.projections
        } == {f"successor-{i}" for i in range(changed_count)}
        if scheduled:
            successor_bindings = [
                b
                for b in result.bindings
                if json.loads(b.projection.source_json)["regulatory_chunk_id"]
                == "successor-0"
            ]
            assert [
                (b.effective_start, b.effective_end) for b in successor_bindings
            ] == [(boundary, date(2028, 1, 1)), (date(2028, 1, 1), None)]
        historical = next(
            b
            for b in result.bindings
            if json.loads(b.projection.source_json)["regulatory_chunk_id"] == old.id
        )
        assert historical.effective_end == boundary
        if legacy:
            assert isinstance(historical.projection, ObservedPublicationProjection)
            assert (
                json.loads(historical.projection.source_json)["content_vector"]
                == json.loads(original.bindings[10].projection.source_json)[
                    "content_vector"
                ]
            )
        else:
            assert (
                historical.projection.embedding_inputs
                == original.bindings[10].projection.embedding_inputs
            )

        from onyx.db import regulatory_writer_publication as repository
        from onyx.db.models import RegulatoryTemporalProjection

        active = [
            RegulatoryTemporalProjection(
                canonical_chunk_id=json.loads(b.projection.source_json)[
                    "regulatory_chunk_id"
                ],
                index_uuid=b.index.index_uuid,
                effective_start=b.effective_start,
                effective_end=b.effective_end,
            )
            for b in original.bindings
        ]
        with patch.object(repository.json, "loads", wraps=json.loads) as decode:
            repository._validate_history_coverage(
                active,
                result.model_copy(update={"kind": "amendment"}),
                inputs.canonical,
            )
        assert decode.call_count <= len(result.bindings)
