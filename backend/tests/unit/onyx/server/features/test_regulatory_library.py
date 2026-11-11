from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError
from sqlalchemy.dialects import postgresql

from onyx.db.regulatory_catalog import catalog_file_query
from onyx.error_handling.exceptions import OnyxError
from onyx.server.features.regulatory import catalog_api
from onyx.server.features.regulatory import library_api as api
from onyx.server.features.regulatory.library_models import LibraryLanguage
from onyx.server.features.regulatory.library_repeals import verified_repeals
from onyx.server.features.regulatory.library_search_memory import (
    contextual_library_query,
)


def test_scoped_file_queries_do_not_leak_pending_or_other_document_sets() -> None:
    query = catalog_file_query(
        q="Gümrük", folder=None, recursive=True, document_set_id=15
    )
    sql = str(
        query.compile(
            dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}
        )
    )
    assert "document_set_id = 15" in sql
    assert "regulatory_document_intake.status !=" in sql
    assert "official_title" in sql


def test_content_outside_corpus_is_denied_before_opening_storage() -> None:
    identifier = uuid4()
    with (
        patch.object(
            api, "library_scope", return_value=SimpleNamespace(asv3_document_set_id=15)
        ),
        patch.object(api, "scoped_library_file_ids", return_value=set()),
        patch.object(api, "read_document_markdown") as read,
    ):
        with pytest.raises(OnyxError):
            api.get_content(identifier, user=MagicMock(), db_session=MagicMock())
        read.assert_not_called()


def test_full_content_keeps_markdown_and_original_title() -> None:
    identifier = uuid4()
    file = SimpleNamespace(id=identifier, name="laws/internal_name.md")
    with (
        patch.object(
            api, "library_scope", return_value=SimpleNamespace(asv3_document_set_id=15)
        ),
        patch.object(api, "scoped_library_file_ids", return_value={identifier}),
        patch.object(api, "get_catalog_file", return_value=file),
        patch.object(
            api, "catalog_display_names", return_value={identifier: "Gümrük Kanunu"}
        ),
        patch.object(
            api,
            "read_document_markdown",
            return_value="# Başlık\n\n| A | B |\n| - | - |\n| 1 | 2 |",
        ),
        patch.object(api, "observe_publication_read"),
        patch.object(api, "require_publication_files"),
    ):
        result = api.get_content(identifier, user=MagicMock(), db_session=MagicMock())
    assert result.name == "Gümrük Kanunu"
    assert result.markdown.startswith("# Başlık")
    assert "| 1 | 2 |" in result.markdown
    assert len(result.content_sha256) == 64
    assert result.verified_repeals == []


def test_repeal_annotations_require_the_verified_file_and_revision() -> None:
    file_id = UUID("12328001-2f21-407e-b230-7d50a8dbf28f")
    revision = "1db892a36effeeb8bf6e56871448b6148a935354fd788f7a8b2e555fe4deff64"
    assert {
        item.heading
        for item in verified_repeals(file_id, revision)
        if item.kind == "article"
    } == {
        "MADDE 42-",
        "MADDE 43-",
        "MADDE 44-",
        "Madde 222. -",
        "MADDE 240-",
    }
    assert [
        item.heading
        for item in verified_repeals(file_id, revision)
        if item.kind == "clause"
    ] == ["ı)"]
    assert not verified_repeals(file_id, "0" * 64)
    assert not verified_repeals(uuid4(), revision)


def test_previous_question_is_answered_without_retrieval() -> None:
    request = api.LibrarySearchRequest(
        query="Ben az önce ne sordum?",
        context=[
            {"question": "Önceki sorgu"},
            {
                "question": "4458 sayılı Kanunun 248. maddesi nedir?",
                "overview": "Özet",
                "legislations": ["4458 sayılı Gümrük Kanunu"],
            },
        ],
    )
    with (
        patch.object(api, "library_scope"),
        patch.object(api, "run_simple_search") as retrieve,
    ):
        result = api.search(request, user=MagicMock(), db_session=MagicMock())
    assert result.kind == "memory"
    assert request.context[-1].question in (result.summary or "")
    assert not result.search.items
    retrieve.assert_not_called()


def test_memory_rejects_chunk_data_and_unbounded_context() -> None:
    with pytest.raises(ValidationError):
        api.LibrarySearchRequest(
            query="Follow up",
            context=[{"question": "Earlier", "chunks": ["raw text"]}],
        )
    with pytest.raises(ValidationError):
        api.LibrarySearchRequest(
            query="Follow up", context=[{"question": "Earlier"}] * 4
        )


def test_contextual_article_follow_up_preserves_explicit_legal_identifiers() -> None:
    context = [
        {"question": "4458 sayılı Kanun", "legislations": ["4458 Gümrük Kanunu"]}
    ]
    follow_up = api.LibrarySearchRequest(query="Peki 42. maddesi?", context=context)
    assert (
        contextual_library_query(follow_up) == "4458 Gümrük Kanunu — Peki 42. maddesi?"
    )
    explicit = api.LibrarySearchRequest(
        query="Peki 5607 sayılı Kanun?", context=context
    )
    assert contextual_library_query(explicit) == explicit.query


@pytest.mark.parametrize(
    ("query", "language", "answer"),
    [
        ("NASILSIN?!", "tr", "İyiyim"),
        ("Selam, naber?", "tr", "İyiyim"),
        ("Merhaba!", "tr", "Merhaba"),
        ("How are you?", "en", "I'm doing well"),
        ("Hallo, wie geht's?", "de", "Mir geht es gut"),
    ],
)
def test_small_talk_returns_plain_text_without_running_search(
    query: str, language: LibraryLanguage, answer: str
) -> None:
    with (
        patch.object(api, "library_scope"),
        patch.object(api, "require_vector_db") as require_index,
        patch.object(api, "run_simple_search") as retrieve,
    ):
        result = api.search(
            api.LibrarySearchRequest(query=query, language=language),
            user=MagicMock(),
            db_session=MagicMock(),
        )
    assert result.kind == "memory"
    assert answer in (result.summary or "")
    assert not result.search.items
    require_index.assert_not_called()
    retrieve.assert_not_called()


def test_greeting_with_legal_question_still_runs_search() -> None:
    query = "Selam, 4458 sayılı Gümrük Kanunu Madde 248"
    with (
        patch.object(api, "library_scope"),
        patch.object(api, "require_vector_db") as require_index,
        patch.object(
            api,
            "run_simple_search",
            return_value=SimpleNamespace(
                search=SimpleNamespace(items=[]), summary=None, summary_error=None
            ),
        ) as retrieve,
    ):
        result = api.search(
            api.LibrarySearchRequest(query=query),
            user=MagicMock(),
            db_session=MagicMock(),
        )
    assert result.kind == "search"
    assert retrieve.call_args.args[0].query == query
    require_index.assert_called_once_with()


def test_library_search_keeps_existing_file_metadata_lookups() -> None:
    file = MagicMock(
        id=uuid4(),
        created_at=datetime.now(UTC),
        file_id="file",
        file_type="md",
        user_id=None,
        status=SimpleNamespace(value="completed"),
    )
    file.name = "Kanunlar/gumruk_kanunu"
    with (
        patch.object(
            catalog_api,
            "catalog_display_names",
            return_value={file.id: "Gümrük Kanunu"},
        ),
        patch.object(catalog_api, "catalog_chunk_counts", return_value={file.id: 52}),
        patch.object(
            catalog_api, "catalog_index_requests", return_value={}
        ) as index_requests,
        patch.object(
            catalog_api, "load_intake_source_lineages", return_value={}
        ) as source_lineages,
        patch.object(
            catalog_api,
            "fetch_latest_regulatory_indexing_progress_for_user_files",
            return_value={},
        ),
        patch.object(catalog_api, "pending_intake_file_ids", return_value=set()),
        patch.object(catalog_api, "load_reviewed_file_relations", return_value={}),
    ):
        file.document_sets = []
        result = catalog_api._file_snapshots(MagicMock(), [file])[file.id]
    assert result.display_name == "Gümrük Kanunu"
    assert result.chunk_count == 52
    assert not result.document_sets
    index_requests.assert_called_once()
    source_lineages.assert_called_once()
