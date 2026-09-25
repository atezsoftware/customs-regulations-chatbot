"""Metadata repair retains immutable canonical history in PostgreSQL."""

import json
from dataclasses import replace
from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy.orm import Session

from onyx.db.models import DocumentSet, RegulatoryTemporalProjection
from onyx.db.regulatory_annex_publication import load_file_temporal_bindings
from onyx.db.regulatory_canonical_revisions import validate_temporal_canonical_revision
from onyx.db.regulatory_context_projections import activate_temporal_projection
from onyx.regulatory.publication_baseline import observed_baseline_binding
from onyx.regulatory.structure_metadata_repair import (
    plan_structure_repair,
    prepare_structure_metadata_manifest,
)
from tests.external_dependency_unit.regulatory.test_amendment_sources import (
    source_session as source_session,
)
from tests.external_dependency_unit.regulatory.test_annex_baseline import _chunk, _file
from tests.unit.onyx.regulatory.indexing_jobs.owned_publication_test_helpers import (
    OwnedAuthority,
    writer_inputs,
)
from tests.unit.onyx.regulatory.test_publication_baseline import baseline_case


def test_repaired_heading_activates_a_new_binding_without_rewriting_history(
    source_session: Session,
) -> None:
    session = source_session
    group = DocumentSet(name=str(uuid4()), description="", is_up_to_date=True)
    session.add(group)
    session.flush()
    file = _file(session, group)
    canonical = _chunk(session, file, 0, "ğ) Tanım.")
    canonical.chunk_type = "clause"
    canonical.heading_path = ["KANUN", "MADDE 8", "g) Tanım"]
    canonical.chunk_metadata = {"article_no": "8", "clause_label": "g"}
    session.flush()
    inputs = writer_inputs(file.id, [canonical])
    _, evidence = baseline_case()
    source = json.loads(evidence.source_json)
    source.update(
        document_id=str(file.id),
        regulatory_chunk_id=canonical.id,
        chunk_index=0,
        content=canonical.text,
        blurb=canonical.text,
        heading_path=canonical.heading_path,
    )
    evidence = evidence.model_copy(update={"source_json": json.dumps(source)})
    previous = observed_baseline_binding(evidence, inputs.canonical)
    activate_temporal_projection(session, user_file_id=file.id, binding=previous)
    retained = session.get(RegulatoryTemporalProjection, previous.id)
    assert retained is not None and retained.canonical_revision_id is not None
    inputs = replace(
        inputs,
        bindings=[previous],
        revisions={previous.id: retained.canonical_revision_id},
    )
    markdown = "KANUN\n\nMADDE 8 - (1) Tanımlar:\n\nğ) Tanım."
    plan = plan_structure_repair(
        inputs.canonical, markdown=markdown, source_file=file.name
    )
    manifest = prepare_structure_metadata_manifest(
        OwnedAuthority(file.id).owner,
        inputs,
        plan=plan,
        markdown=markdown,
    )
    retained.retired_at = datetime.now(timezone.utc)
    after = plan.changes[0].after
    canonical.chunk_metadata = after.metadata
    canonical.heading_path = after.heading_path
    canonical.chunk_type = after.chunk_type
    session.flush()
    activate_temporal_projection(
        session, user_file_id=file.id, binding=manifest.bindings[0]
    )
    validate_temporal_canonical_revision(session, retained)
    current = session.get(RegulatoryTemporalProjection, manifest.bindings[0].id)
    assert current is not None
    validate_temporal_canonical_revision(session, current)
    assert current.canonical_revision_id != retained.canonical_revision_id
    assert retained.payload == previous.model_dump(mode="json")
    assert current.projection_ordinal == retained.projection_ordinal == 0
    assert current.canonical_chunk_id == retained.canonical_chunk_id == canonical.id
    assert load_file_temporal_bindings(session, file.id) == manifest.bindings
    rewritten = json.loads(manifest.bindings[0].projection.source_json)
    assert {k: v for k, v in rewritten.items() if k != "heading_path"} == {
        k: v for k, v in source.items() if k != "heading_path"
    }
