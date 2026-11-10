from sqlalchemy import select
from sqlalchemy.dialects import postgresql

from onyx.db.models import UserFile
from onyx.db.regulatory_catalog import (
    catalog_display_name_expression,
    catalog_display_path,
)


def test_display_leaf_keeps_archive_folders_and_official_slashes() -> None:
    assert (
        catalog_display_path(
            "Gümrük/Kanunlar/internal_name.md", "4458 Sayılı Gümrük Kanunu"
        )
        == "Gümrük/Kanunlar/4458 Sayılı Gümrük Kanunu"
    )
    assert (
        catalog_display_path("folder/internal.md", "2024/9366 Sayılı Karar")
        == "folder/2024/9366 Sayılı Karar"
    )


def test_filename_search_uses_original_metadata_without_source_url_or_chunk_body() -> (
    None
):
    query = select(UserFile.id).where(
        catalog_display_name_expression().icontains("Gümrük", autoescape=True)
    )
    sql = str(
        query.compile(
            dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}
        )
    )
    assert "official_title" in sql
    assert "file_record" in sql
    assert "original_url" not in sql
    assert "regulatory_chunk.text" not in sql
