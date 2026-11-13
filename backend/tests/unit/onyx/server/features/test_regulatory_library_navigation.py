import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from onyx.server.features.regulatory import library_api as api
from onyx.server.features.regulatory import library_repeals
from onyx.server.features.regulatory.library_models import LibraryRepeal


@pytest.mark.parametrize("source_changed", [False, True])
def test_content_cleans_navigation_and_binds_repeals_to_retained_revision(
    source_changed: bool,
) -> None:
    identifier = uuid4()
    source = (
        Path(__file__).parents[2]
        / "regulatory"
        / "fixtures"
        / "customs_law_introduction.md"
    ).read_text()
    source_hash = hashlib.sha256(source.encode()).hexdigest()
    evidence = LibraryRepeal(
        heading="MADDE 42-",
        source_label="Verified source",
        source_url="https://example.com",
    )
    retained = source + ("\n" if source_changed else "")
    navigation_start = source.index("**I.Kısım - I. Bölüm**")
    expected = (
        source[:navigation_start]
        + source[source.index("#######\n", navigation_start) :]
        + ("\n" if source_changed else "")
    )
    with (
        patch.object(
            api, "library_scope", return_value=SimpleNamespace(asv3_document_set_id=15)
        ),
        patch.object(api, "scoped_library_file_ids", return_value={identifier}),
        patch.object(
            api, "get_catalog_file", return_value=SimpleNamespace(id=identifier)
        ),
        patch.object(
            api, "catalog_display_names", return_value={identifier: "Gümrük Kanunu"}
        ),
        patch.object(api, "read_document_markdown", return_value=retained),
        patch.object(api, "observe_publication_read"),
        patch.object(api, "require_publication_files"),
        patch.dict(
            library_repeals._VERIFIED_REPEALS, {(identifier, source_hash): (evidence,)}
        ),
    ):
        result = api.get_content(identifier, user=MagicMock(), db_session=MagicMock())
    assert result.markdown == expected
    assert result.content_sha256 == hashlib.sha256(expected.encode()).hexdigest()
    assert result.content_sha256 != source_hash
    assert result.verified_repeals == ([] if source_changed else [evidence])
