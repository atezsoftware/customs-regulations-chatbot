from datetime import date

from onyx.document_index.elasticsearch.search import DocumentQuery
from onyx.document_index.interfaces_new import TenantState


def test_label_candidate_ids_are_intersected_with_existing_security_and_date_filters() -> (
    None
):
    def filters(identifiers: list[str] | None) -> list[dict[str, object]]:
        return DocumentQuery._get_search_filters(
            tenant_state=TenantState(tenant_id="tenant-a", multitenant=True),
            include_hidden=False,
            access_control_list=["user-a"],
            source_types=[],
            tags=[],
            document_sets=["allowed"],
            project_id_filter=None,
            persona_id_filter=None,
            created_at_range=None,
            updated_at_range=None,
            min_chunk_index=None,
            max_chunk_index=None,
            as_of_date=date(2023, 1, 1),
            regulatory_chunks_only=True,
            regulatory_candidate_ids=identifiers,
        )

    baseline = filters(None)
    scoped = filters(["c1", "c2"])
    assert all(clause in scoped for clause in baseline)
    assert {"terms": {"regulatory_chunk_id": ["c1", "c2"]}} in scoped
    empty = filters([])
    assert {"match_none": {}} in empty
