"""Native encoder receipts remain distinct from physical output dimensions."""

import json
from dataclasses import replace
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from onyx.db.enums import IndexModelStatus
from onyx.db.models import SearchSettings
from onyx.natural_language_processing.search_nlp_models import EmbeddingModel
from onyx.regulatory import writer_projection
from tests.unit.onyx.regulatory.annexes.test_context_dependencies import (
    _durable_embedding_model,
)
from tests.unit.onyx.regulatory.indexing_jobs.owned_publication_test_helpers import (
    OwnedAuthority,
    canonical_row,
    writer_inputs,
)


@pytest.mark.parametrize("native,output", [(3072, 1024), (3, 3)])
@pytest.mark.parametrize("failure", [None, "configuration", "input", "vector"])
def test_normal_writer_freezes_native_dimension_and_publishes_output_dimension(
    native: int,
    output: int,
    failure: str | None,
) -> None:
    file_id = uuid4()
    authority = OwnedAuthority(file_id)
    inputs = writer_inputs(file_id, [canonical_row(file_id, 0, "EK-1 Wheat duty 5%")])
    settings = SearchSettings(
        id=1,
        status=IndexModelStatus.PRESENT,
        index_name="dimension-test",
        model_name="embedding",
        model_dim=native,
        reduced_dimension=output if native != output else None,
        normalize=True,
        enable_contextual_rag=False,
    )
    inputs = replace(inputs, settings=[settings])
    model = _durable_embedding_model()
    model.reduced_dimension = settings.reduced_dimension
    embedder = MagicMock()
    embedder.embedding_model = model
    client = MagicMock()
    client.indices.get.return_value = {
        settings.index_name: {"settings": {"index": {"uuid": "physical-test"}}}
    }
    original_freeze = writer_projection.freeze_encoder_inputs

    def freeze(
        texts: list[str], model: EmbeddingModel, *, model_dim: int, formatter: str
    ) -> tuple[list[str], dict[str, str | int | float | bool | None]]:
        texts, config = original_freeze(
            texts, model, model_dim=model_dim, formatter=formatter
        )
        if failure == "configuration":
            config["normalize"] = not config["normalize"]
        if failure == "input":
            texts = [*texts, "unapproved input"]
        return texts, config

    with (
        patch.object(writer_projection, "freeze_encoder_inputs", side_effect=freeze),
        patch(
            "onyx.regulatory.projection.effective_contextual_rag_enabled",
            return_value=False,
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
            writer_projection, "resolve_review_context_llm", return_value=None
        ),
        patch.object(
            model,
            "encode",
            side_effect=lambda texts, **_: [
                [0.1] * (output + (1 if failure == "vector" else 0)) for _ in texts
            ],
        ) as encode,
    ):
        if failure:
            reason = (
                "writer encoder differs"
                if failure in {"configuration", "input"}
                else "vector dimension"
            )
            with pytest.raises(ValueError, match=reason):
                writer_projection.prepare_owned_correction(
                    authority.owner, client, inputs, inputs.canonical, changed_id=None
                )
            if failure in {"configuration", "input"}:
                encode.assert_not_called()
            return
        prepared = writer_projection.prepare_owned_correction(
            authority.owner, client, inputs, inputs.canonical, changed_id=None
        )
    assert len(prepared.bindings) == 1
    binding = prepared.bindings[0]
    configuration = json.loads(binding.projection.embedding_config_json)
    assert configuration["dimension"] == native
    assert configuration["reduced_dimension"] == settings.reduced_dimension
    assert binding.index.vector_dimension == output
    assert len(json.loads(binding.projection.source_json)["content_vector"]) == output
    assert binding.index.accepts_encoder_configuration(configuration)
    assert (
        tuple(encode.call_args.kwargs["texts"]) == binding.projection.embedding_inputs
    )


def test_durable_target_receipt_keeps_effective_dimension_contract() -> None:
    from onyx.regulatory.amendments.annexes.context_dependencies import (
        encoder_model_fingerprint,
    )
    from onyx.regulatory.amendments.annexes.publication_preparation import (
        _target_configuration,
    )
    from onyx.regulatory.indexing_jobs.models import RegulatoryIndexingConfigSnapshot

    settings = SearchSettings(model_dim=3072, reduced_dimension=1024)
    model = _durable_embedding_model()
    model.reduced_dimension = 1024
    embedder = MagicMock()
    embedder.embedding_model = model
    snapshot = MagicMock(spec=RegulatoryIndexingConfigSnapshot)
    snapshot.openrouter_batch = None
    expected = encoder_model_fingerprint(
        model, model_dim=1024, formatter="durable-context-before-text-v1"
    )
    expected["transport"] = "synchronous_encoder"
    assert _target_configuration(settings, snapshot, embedder) == expected
