from typing import cast
from unittest.mock import ANY, MagicMock, patch
from uuid import uuid4

import pytest

from onyx.db.models import RegulatoryChunk
from onyx.document_index.interfaces_new import IndexingMetadata
from onyx.indexing.models import DocAwareChunk
from onyx.regulatory.projection import (
    _build_document_shell,
    _contextualize_chunks,
    _project_rows_to_search_settings,
    _row_context_text,
    _rows_to_doc_aware_chunks,
    project_user_file_to_index,
)


def _settings(*, current: bool, future: bool = False) -> MagicMock:
    settings = MagicMock()
    settings.id = 1 if current else 2
    settings.status.is_current.return_value = current
    settings.status.is_future.return_value = future
    settings.enable_contextual_rag = False
    return settings


def test_projection_writes_present_before_future_and_keeps_canonical_rows() -> None:
    user_file = MagicMock()
    user_file.id = uuid4()
    user_file.chunk_count = 1
    user_file.secondary_reconcile_pending = True
    rows = [MagicMock(), MagicMock()]
    present = _settings(current=True)
    future = _settings(current=False, future=True)

    with (
        patch(
            "onyx.regulatory.projection.lock_completed_user_file_for_projection",
            return_value=user_file,
        ) as lock_user_file,
        patch("onyx.regulatory.projection.get_chunks_for_file", return_value=rows),
        patch(
            "onyx.regulatory.projection.get_active_search_settings_list",
            return_value=[present, future],
        ),
        patch(
            "onyx.regulatory.projection.fetch_user_project_ids_for_user_files",
            return_value={str(user_file.id): [5]},
        ),
        patch(
            "onyx.regulatory.projection.fetch_persona_ids_for_user_files",
            return_value={str(user_file.id): [7]},
        ),
        patch(
            "onyx.regulatory.projection._project_rows_to_search_settings",
            return_value=2,
        ) as project_one,
    ):
        count = project_user_file_to_index(MagicMock(), user_file, "tenant")

    assert count == 2
    lock_user_file.assert_called_once_with(ANY, user_file.id, include_chunked=False)
    assert [item.kwargs["search_settings"] for item in project_one.call_args_list] == [
        present,
        future,
    ]
    assert all(item.kwargs["rows"] is rows for item in project_one.call_args_list)
    assert user_file.secondary_reconcile_pending is False


def test_future_projection_failure_is_deferred_without_rolling_back_present() -> None:
    user_file = MagicMock()
    user_file.id = uuid4()
    user_file.chunk_count = 1
    user_file.secondary_reconcile_pending = False
    rows = [MagicMock()]
    present = _settings(current=True)
    future = _settings(current=False, future=True)

    with (
        patch(
            "onyx.regulatory.projection.lock_completed_user_file_for_projection",
            return_value=user_file,
        ),
        patch("onyx.regulatory.projection.get_chunks_for_file", return_value=rows),
        patch(
            "onyx.regulatory.projection.get_active_search_settings_list",
            return_value=[present, future],
        ),
        patch(
            "onyx.regulatory.projection.fetch_user_project_ids_for_user_files",
            return_value={},
        ),
        patch(
            "onyx.regulatory.projection.fetch_persona_ids_for_user_files",
            return_value={},
        ),
        patch(
            "onyx.regulatory.projection._project_rows_to_search_settings",
            side_effect=[1, RuntimeError("future unavailable")],
        ) as project_one,
    ):
        assert project_user_file_to_index(MagicMock(), user_file, "tenant") == 1

    assert project_one.call_count == 2
    assert user_file.secondary_reconcile_pending is True


def test_present_projection_failure_prevents_future_write() -> None:
    user_file = MagicMock()
    user_file.id = uuid4()
    user_file.chunk_count = 1
    rows = [MagicMock()]
    present = _settings(current=True)
    future = _settings(current=False, future=True)

    with (
        patch(
            "onyx.regulatory.projection.lock_completed_user_file_for_projection",
            return_value=user_file,
        ),
        patch("onyx.regulatory.projection.get_chunks_for_file", return_value=rows),
        patch(
            "onyx.regulatory.projection.get_active_search_settings_list",
            return_value=[present, future],
        ),
        patch(
            "onyx.regulatory.projection.fetch_user_project_ids_for_user_files",
            return_value={},
        ),
        patch(
            "onyx.regulatory.projection.fetch_persona_ids_for_user_files",
            return_value={},
        ),
        patch(
            "onyx.regulatory.projection._project_rows_to_search_settings",
            side_effect=RuntimeError("present unavailable"),
        ) as project_one,
    ):
        with pytest.raises(RuntimeError, match="present unavailable"):
            project_user_file_to_index(MagicMock(), user_file, "tenant")

    project_one.assert_called_once()


def test_projection_skips_file_that_cannot_be_locked_as_completed() -> None:
    user_file = MagicMock()
    user_file.id = uuid4()

    with (
        patch(
            "onyx.regulatory.projection.lock_completed_user_file_for_projection",
            return_value=None,
        ) as lock_user_file,
        patch("onyx.regulatory.projection.get_chunks_for_file") as get_rows,
        patch(
            "onyx.regulatory.projection._project_rows_to_search_settings"
        ) as project_rows,
    ):
        count = project_user_file_to_index(MagicMock(), user_file, "tenant")

    assert count == 0
    lock_user_file.assert_called_once_with(ANY, user_file.id, include_chunked=False)
    get_rows.assert_not_called()
    project_rows.assert_not_called()


def test_target_projection_builds_setting_specific_index_and_embedder() -> None:
    user_file = MagicMock()
    user_file.id = uuid4()
    user_file.name = "Mevzuat"
    settings = _settings(current=False, future=True)
    settings.id = 42
    row = MagicMock()
    index_chunk = MagicMock()
    enriched_chunk = MagicMock()
    doc_chunk = MagicMock()
    embedder = MagicMock()
    embedder.embedding_model.tokenizer.encode.return_value = [1]
    embedder.embed_chunks.return_value = [index_chunk]
    document_index = MagicMock()
    metadata = IndexingMetadata(doc_id_to_chunk_cnt_diff={})

    with (
        patch(
            "onyx.regulatory.projection.DefaultIndexingEmbedder.from_db_search_settings",
            return_value=embedder,
        ) as build_embedder,
        patch(
            "onyx.regulatory.projection._rows_to_doc_aware_chunks",
            return_value=[doc_chunk],
        ),
        patch(
            "onyx.regulatory.projection._enrich_index_chunks",
            return_value=[enriched_chunk],
        ),
        patch(
            "onyx.regulatory.projection.get_all_document_indices",
            return_value=[document_index],
        ) as get_indices,
    ):
        count = _project_rows_to_search_settings(
            user_file=user_file,
            rows=[row],
            search_settings=settings,
            tenant_id="tenant",
            project_ids={str(user_file.id): [5]},
            persona_ids={str(user_file.id): [7]},
            document_set_names={str(user_file.id): ["Regulations"]},
            user_file_access={},
            indexing_metadata=metadata,
        )

    assert count == 1
    build_embedder.assert_called_once_with(search_settings=settings)
    assert get_indices.call_args.args[:2] == (settings, None)
    embedder.embed_chunks.assert_called_once_with([doc_chunk], tenant_id="tenant")
    document_index.index.assert_called_once_with(
        chunks=[enriched_chunk], indexing_metadata=metadata
    )


def test_contextual_projection_rejects_incomplete_eligible_chunks() -> None:
    canonical_document = MagicMock()
    chunks = [cast(DocAwareChunk, MagicMock()), cast(DocAwareChunk, MagicMock())]
    rows = [cast(RegulatoryChunk, MagicMock()), cast(RegulatoryChunk, MagicMock())]
    for position, (chunk, row) in enumerate(zip(chunks, rows)):
        chunk.chunk_id = position
        chunk.content = f"chunk-{position}"
        chunk.title_prefix = ""
        chunk.metadata_suffix_semantic = ""
        chunk.doc_summary = ""
        chunk.chunk_context = ""
        chunk.source_document = canonical_document
        row.id = f"row-{position}"
        row.position = position
        row.text = f"row text {position}"
        row.heading_path = []
        row.validity_start_date = None
        row.validity_end_date = None

    user_file = MagicMock()
    user_file.id = uuid4()
    user_file.name = "Regulation"
    settings = _settings(current=False, future=True)
    settings.enable_contextual_rag = True
    llm = MagicMock()
    llm.config.model_name = "contextual-model"
    llm.config.model_provider = "provider"
    embedder = MagicMock()
    embedder.embedding_model.tokenizer.encode.return_value = [1]

    def leave_one_chunk_incomplete(*, chunks: list[DocAwareChunk], **_: object) -> None:
        chunks[0].chunk_context = "complete context"

    with (
        patch(
            "onyx.regulatory.projection.get_contextual_rag_llm_for_search_settings",
            return_value=llm,
        ),
        patch("onyx.regulatory.projection.get_tokenizer", return_value=MagicMock()),
        patch(
            "onyx.indexing.indexing_pipeline.add_contextual_summaries",
            side_effect=leave_one_chunk_incomplete,
        ) as add_summaries,
        patch("onyx.regulatory.projection.USE_CHUNK_SUMMARY", True),
        pytest.raises(RuntimeError, match="1/2 eligible chunks"),
    ):
        _contextualize_chunks(
            chunks=chunks,
            rows=rows,
            user_file=user_file,
            embedder=embedder,
            search_settings=settings,
        )

    assert add_summaries.call_args.kwargs["raise_on_failure"] is True
    assert all(chunk.source_document is canonical_document for chunk in chunks)


def test_projection_and_context_use_reverse_article_anchor() -> None:
    user_file = MagicMock()
    user_file.id = uuid4()
    user_file.name = "Belge"
    document = _build_document_shell(user_file)
    row = MagicMock()
    row.id = "row-4a"
    row.text = "Eklenen hüküm."
    row.heading_path = ["Belge", "MADDE 4", "4A Maddesi:", "(1)"]
    row.chunk_metadata = {"article_no": "4"}
    row.validity_start_date = None
    row.validity_end_date = None

    with patch("onyx.regulatory.projection.extract_blurb", return_value=row.text):
        chunks = _rows_to_doc_aware_chunks(document, [row], MagicMock())

    assert chunks[0].heading_path == ["Belge", "4A Maddesi:", "(1)"]
    assert _row_context_text(row) == "Belge > 4A Maddesi: > (1)\nEklenen hüküm."


def test_projection_and_context_repair_legacy_article_metadata_lineage() -> None:
    user_file = MagicMock()
    user_file.id = uuid4()
    user_file.name = "Belge"
    document = _build_document_shell(user_file)
    row = MagicMock()
    row.id = "row-75-intro"
    row.text = "2. Kapsamlı teminatın tutarı:"
    row.heading_path = ["Belge", "Teminatlar", "2. Kapsamlı teminatın tutarı"]
    row.chunk_metadata = {
        "article_no": "75",
        "paragraph_no": None,
        "clause_label": None,
    }
    row.chunk_type = "numbered_section"
    row.validity_start_date = None
    row.validity_end_date = None

    with patch("onyx.regulatory.projection.extract_blurb", return_value=row.text):
        chunks = _rows_to_doc_aware_chunks(document, [row], MagicMock())

    expected_path = [
        "Belge",
        "Teminatlar",
        "MADDE 75",
        "2. Kapsamlı teminatın tutarı",
    ]
    assert chunks[0].heading_path == expected_path
    assert _row_context_text(row) == (
        "Belge > Teminatlar > MADDE 75 > 2. Kapsamlı teminatın tutarı\n"
        "2. Kapsamlı teminatın tutarı:"
    )
