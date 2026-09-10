"""Frozen publication accepts serialized search documents, not coercible substitutes."""

import json
from datetime import datetime, timezone

import pytest
from pydantic import JsonValue

from onyx.document_index.elasticsearch.schema import DocumentChunk
from onyx.document_index.publication_models import FrozenPublicationProjection


def serialized_source() -> dict[str, JsonValue]:
    timestamp = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return DocumentChunk(
        document_id="file-id",
        chunk_index=5,
        title="Title",
        title_vector=[0.1, 0.2],
        content="Legal text",
        content_vector=[0.3, 0.4],
        source_type="file",
        metadata_list=["kind:regulation"],
        last_updated=timestamp,
        created_at=timestamp,
        public=False,
        access_control_list=["user:1"],
        hidden=False,
        written_by_port=True,
        global_boost=1,
        semantic_identifier="Regulation",
        image_file_id="image",
        source_links='{"0":"source"}',
        blurb="Legal text",
        doc_summary="Summary",
        chunk_context="Context",
        metadata_suffix="metadata",
        document_sets=["1"],
        user_projects=[1],
        personas=[2],
        primary_owners=["owner"],
        secondary_owners=["other"],
        ancestor_hierarchy_node_ids=[3],
        regulatory_chunk_id="canonical-1",
        heading_path=["Annex"],
        provision_identifiers=["Article1"],
        decision_numbers=["1"],
        legal_dates=["2026-01-01"],
        validity_start_date=timestamp,
        validity_end_date=datetime(2027, 1, 1, tzinfo=timezone.utc),
    ).model_dump(mode="json")


def freeze(source: dict[str, JsonValue]) -> FrozenPublicationProjection:
    return FrozenPublicationProjection(
        ordinal=5,
        context_projection_id="context-1",
        source_json=json.dumps(source),
        embedding_inputs=("actual encoder input",),
        embedding_config_json='{"model":"fixture"}',
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("access_control_list", "user:1"),
        ("access_control_list", [123]),
        ("public", "false"),
        ("hidden", 0),
        ("hidden", None),
        ("validity_start_date", "2026-01-01"),
        ("validity_end_date", 1780000000.5),
        ("validity_start_date", True),
        ("last_updated", "2026-01-01T00:00:00Z"),
        ("created_at", False),
        ("document_sets", "1"),
        ("user_projects", ["1"]),
        ("source_links", {"0": "source"}),
        ("content_vector", [True, 0.4]),
        ("content_vector", "0.3,0.4"),
        ("title_vector", ["0.1", "0.2"]),
        ("heading_path", "Annex"),
        ("tenant_id", {"tenant_id": "other"}),
    ],
)
def test_freeze_rejects_malformed_serialized_fields(
    field: str, value: JsonValue
) -> None:
    source = serialized_source()
    source[field] = value
    with pytest.raises(ValueError):
        freeze(source)


def test_freeze_accepts_existing_full_serialization_without_rewriting() -> None:
    source = serialized_source()
    assert source["validity_start_date"] == 1767225600
    assert source["access_control_list"] == ["user:1"]
    assert json.loads(freeze(source).source_json) == source
    # Explicit tenant and unbounded temporal windows use serialized shapes directly,
    # independently of process-global TenantState deserialization.
    source["tenant_id"] = "tenant_fixture"
    source["validity_start_date"] = None
    source.pop("validity_end_date")
    assert json.loads(freeze(source).source_json) == source
