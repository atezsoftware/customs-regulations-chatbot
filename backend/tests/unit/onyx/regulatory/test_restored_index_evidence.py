import json

import pytest

from onyx.document_index.publication_models import (
    FrozenPublicationProjection,
    publication_digest,
)
from onyx.regulatory.publication_baseline import observed_baseline_binding
from tests.unit.onyx.regulatory.test_publication_baseline import baseline_case


def restored_case():
    inputs, evidence = baseline_case()
    binding = observed_baseline_binding(evidence, inputs.canonical)
    configuration = {"model": "fixture"}
    index = binding.index.model_copy(
        update={"embedding_config_sha256": publication_digest(configuration)}
    )
    projection = FrozenPublicationProjection(
        ordinal=5,
        context_projection_id=str(binding.id),
        source_json=binding.projection.source_json,
        embedding_inputs=("Legal text",),
        embedding_config_json=json.dumps(configuration),
    )
    binding = binding.model_copy(update={"projection": projection, "index": index})
    source = json.loads(projection.source_json)
    source.update(
        publication_scope="scope",
        publication_floor=7,
        publication_token=7,
        publication_evidence={
            "index": index.model_copy(
                update={"index_uuid": "old-physical-uuid"}
            ).model_dump(mode="json"),
            "context_projection_id": str(binding.id),
            "embedding_inputs": ["Legal text"],
            "embedding_config": configuration,
        },
    )
    source["publication_payload"] = publication_digest(source)
    return binding, source


def test_restored_index_identity_requires_exact_active_binding() -> None:
    from onyx.regulatory.restored_index_evidence import verified_restored_index_source

    binding, source = restored_case()
    result = verified_restored_index_source(
        source, binding, old_index_uuid="old-physical-uuid"
    )
    assert (
        result["publication_evidence"]["index"]["index_uuid"]
        == binding.index.index_uuid
    )
    assert {k: v for k, v in source.items() if not k.startswith("publication_")} == {
        k: v for k, v in result.items() if not k.startswith("publication_")
    }
    assert source["publication_evidence"]["index"]["index_uuid"] == "old-physical-uuid"


@pytest.mark.parametrize(
    "field,value", [("content_vector", [0.9, 0.9]), ("content", "wrong source")]
)
def test_restored_index_rejects_payload_that_disagrees_with_active_binding(
    field, value
) -> None:
    from onyx.regulatory.restored_index_evidence import verified_restored_index_source

    binding, source = restored_case()
    source[field] = value
    source.pop("publication_payload")
    source["publication_payload"] = publication_digest(source)
    with pytest.raises(ValueError, match="active binding"):
        verified_restored_index_source(
            source, binding, old_index_uuid="old-physical-uuid"
        )
