"""Immutable source names and actual annex links survive legal title changes."""

from sqlalchemy import delete
from sqlalchemy.orm import Session

from onyx.db.models import DocumentSet__UserFile, RegulatorySourceAsset, UserFile
from onyx.db.regulatory_catalog import list_catalog_files
from tests.external_dependency_unit.regulatory.test_amendment_source_family_resolution import (
    family,
)
from tests.external_dependency_unit.regulatory.test_labeling_jobs import (
    LabelingData,
)
from tests.external_dependency_unit.regulatory.test_labeling_jobs import (
    labeling_data as labeling_data,
)
from tests.external_dependency_unit.regulatory.test_labeling_jobs import (
    labeling_database as labeling_database,
)


def test_original_filename_remains_searchable_after_title_assignment(
    labeling_data: LabelingData,
) -> None:
    with Session(labeling_data.database.engine) as session:
        session.get_one(
            UserFile, labeling_data.file_id
        ).name = "resmigazete_20250105-2.htm.pdf"
        primary, child, _package, _batch, parent, _child, _pub = family(
            session, labeling_data
        )
        source = session.get_one(RegulatorySourceAsset, parent.source_asset_id)
        primary.name = "Assigned legal document title"
        session.flush()
        files, total = list_catalog_files(
            session,
            q="resmigazete_20250105-2",
            folder=None,
            recursive=True,
            offset=0,
            limit=25,
        )
        assert total == 1
        assert [file.id for file in files] == [primary.id]
        from onyx.db.document_intake_source_lineage import load_intake_source_lineages

        lineage = load_intake_source_lineages(session, [primary.id, child.id])
        assert lineage[primary.id].source.original_name == source.display_name
        assert lineage[child.id].parents[0].user_file_id == primary.id
        assert lineage[child.id].parents[0].title == primary.name
        assert lineage[primary.id].attachments[0].user_file_id == child.id


def test_source_name_search_treats_percent_and_underscore_literally(
    labeling_data: LabelingData,
) -> None:
    with Session(labeling_data.database.engine) as session:
        session.get_one(UserFile, labeling_data.file_id).name = "100%_original.pdf"
        primary, _child, _package, _batch, parent, _member, _pub = family(
            session, labeling_data
        )
        primary.name = "Assigned legal title"
        session.flush()
        for query, expected in [("100%_", [primary.id]), ("100x_", [])]:
            files, total = list_catalog_files(
                session, q=query, folder=None, recursive=True, offset=0, limit=25
            )
            assert [file.id for file in files] == expected
            assert total == len(expected)


def test_missing_source_edge_does_not_create_parent_metadata(
    labeling_data: LabelingData,
) -> None:
    from onyx.db.document_intake_source_lineage import load_intake_source_lineages

    with Session(labeling_data.database.engine) as session:
        primary, child, _package, _batch, parent, _member, _pub = family(
            session, labeling_data
        )
        parent.relationships = []
        session.flush()
        lineage = load_intake_source_lineages(session, [primary.id, child.id])
        assert lineage[child.id].parents == []
        assert lineage[primary.id].attachments == []


def test_removed_parent_membership_hides_parent_metadata(
    labeling_data: LabelingData,
) -> None:
    from onyx.db.document_intake_source_lineage import load_intake_source_lineages

    with Session(labeling_data.database.engine) as session:
        primary, child, _package, _batch, _parent, _member, _pub = family(
            session, labeling_data
        )
        session.execute(
            delete(DocumentSet__UserFile).where(
                DocumentSet__UserFile.user_file_id == primary.id
            )
        )
        session.flush()
        lineage = load_intake_source_lineages(session, [primary.id, child.id])
        assert primary.id not in lineage
        assert lineage[child.id].parents == []


def test_mismatched_owner_cannot_supply_search_alias_or_parent_metadata(
    labeling_data: LabelingData,
) -> None:
    from onyx.db.document_intake_source_lineage import load_intake_source_lineages

    with Session(labeling_data.database.engine) as session:
        session.get_one(UserFile, labeling_data.file_id).name = "original-owned.pdf"
        primary, child, *_ = family(session, labeling_data)
        primary.name = "New legal title"
        primary.user_id = None
        session.flush()
        lineage = load_intake_source_lineages(session, [primary.id, child.id])
        assert primary.id not in lineage
        assert lineage[child.id].parents == []
        files, total = list_catalog_files(
            session, q="original-owned", folder=None, recursive=True, offset=0, limit=25
        )
        assert not files and total == 0
