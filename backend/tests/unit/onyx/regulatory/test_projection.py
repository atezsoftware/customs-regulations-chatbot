import datetime
from types import SimpleNamespace
from typing import cast
from unittest.mock import ANY, MagicMock, patch
from uuid import uuid4

import pytest

from onyx.db.models import RegulatoryChunk, UserFileStatus
from onyx.document_index.interfaces_new import IndexingMetadata
from onyx.indexing.models import DocAwareChunk
from onyx.regulatory.projection import (
    _affected_amendment_row_ids,
    _build_document_shell,
    _contextualize_chunks,
    _project_amendment_rows_to_search_settings,
    _project_rows_to_search_settings,
    _row_context_text,
    _rows_in_structural_order,
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


def test_projection_targets_the_exact_validated_current_setting() -> None:
    user_file = MagicMock()
    user_file.id = uuid4()
    user_file.chunk_count = 1
    rows = [MagicMock()]
    present = _settings(current=True)
    present.id = 11
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
            return_value=1,
        ) as project_one,
    ):
        count = project_user_file_to_index(
            MagicMock(),
            user_file,
            "tenant",
            current_search_settings_id=11,
        )

    assert count == 1
    project_one.assert_called_once()
    assert project_one.call_args.kwargs["search_settings"] is present


def test_projection_rejects_a_promoted_unvalidated_current_setting() -> None:
    user_file = MagicMock()
    user_file.id = uuid4()
    rows = [MagicMock()]
    promoted = _settings(current=True)
    promoted.id = 12

    with (
        patch(
            "onyx.regulatory.projection.lock_completed_user_file_for_projection",
            return_value=user_file,
        ),
        patch("onyx.regulatory.projection.get_chunks_for_file", return_value=rows),
        patch(
            "onyx.regulatory.projection.get_active_search_settings_list",
            return_value=[promoted],
        ),
        patch(
            "onyx.regulatory.projection._project_rows_to_search_settings"
        ) as project_one,
        pytest.raises(RuntimeError, match="changed after validation"),
    ):
        project_user_file_to_index(
            MagicMock(),
            user_file,
            "tenant",
            current_search_settings_id=11,
        )

    project_one.assert_not_called()


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


def test_strict_current_projection_recovers_a_legacy_failed_file() -> None:
    user_file = MagicMock()
    user_file.id = uuid4()
    user_file.status = UserFileStatus.FAILED
    user_file.chunk_count = 1
    rows = [MagicMock()]
    present = _settings(current=True)
    present.id = 11
    db_session = MagicMock()

    with (
        patch(
            "onyx.regulatory.projection.lock_completed_user_file_for_projection",
            return_value=user_file,
        ) as lock_user_file,
        patch("onyx.regulatory.projection.get_chunks_for_file", return_value=rows),
        patch(
            "onyx.regulatory.projection.get_active_search_settings_list",
            return_value=[present],
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
            return_value=1,
        ),
    ):
        count = project_user_file_to_index(
            db_session,
            user_file,
            "tenant",
            current_search_settings_id=11,
            include_failed=True,
        )

    assert count == 1
    lock_user_file.assert_called_once_with(
        db_session,
        user_file.id,
        include_chunked=False,
        include_failed=True,
    )
    assert user_file.status is UserFileStatus.COMPLETED


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
        patch(
            "onyx.regulatory.projection.effective_contextual_rag_enabled",
            return_value=False,
        ),
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


def test_structural_order_does_not_reassign_immutable_projection_ordinals() -> None:
    imported_later = SimpleNamespace(
        id="imported-20",
        source="indexed",
        position=20,
        projection_ordinal=20,
        created_at=datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc),
    )
    imported_earlier = SimpleNamespace(
        id="imported-10",
        source="indexed",
        position=10,
        projection_ordinal=10,
        created_at=datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc),
    )
    first_amendment = SimpleNamespace(
        id="amendment-a",
        source="amendment",
        position=10,
        projection_ordinal=1_000_000_064,
        created_at=datetime.datetime(2026, 2, 1, tzinfo=datetime.timezone.utc),
    )
    second_amendment = SimpleNamespace(
        id="amendment-b",
        source="amendment",
        position=20,
        projection_ordinal=1_000_000_066,
        created_at=datetime.datetime(2026, 3, 1, tzinfo=datetime.timezone.utc),
    )

    ordered = _rows_in_structural_order(
        cast(
            list[RegulatoryChunk],
            [second_amendment, imported_later, first_amendment, imported_earlier],
        )
    )

    assert [row.id for row in ordered] == [
        "amendment-a",
        "imported-10",
        "amendment-b",
        "imported-20",
    ]
    assert [row.projection_ordinal for row in ordered] == [
        1_000_000_064,
        10,
        1_000_000_066,
        20,
    ]


def test_affected_amendment_rows_are_bounded_to_structural_neighbors() -> None:
    old_chunk = MagicMock()
    old_chunk.id = "old"
    old_chunk.position = 90
    old_chunk.validity_start_date = None
    old_chunk.validity_end_date = datetime.date(2026, 7, 4)
    old_chunk.chunk_metadata = {}
    new_chunk = MagicMock()
    new_chunk.id = "new"
    new_chunk.position = 90
    new_chunk.text = "updated provision"
    new_chunk.validity_start_date = datetime.date(2026, 7, 4)
    new_chunk.validity_end_date = None
    new_chunk.chunk_metadata = {}
    same_position = MagicMock()
    same_position.id = "same-position-version"
    same_position.position = 90
    same_position.chunk_metadata = {}
    aggregate = MagicMock()
    aggregate.id = "aggregate"
    aggregate.position = 89
    aggregate.chunk_metadata = {"source_regulatory_chunk_ids": ["old"]}
    unrelated = MagicMock()
    unrelated.id = "unrelated"
    unrelated.position = 400
    unrelated.chunk_metadata = {}

    with (
        patch(
            "onyx.regulatory.projection.get_bounded_same_provision_siblings",
            side_effect=[
                [SimpleNamespace(regulatory_chunk_id="old-neighbor")],
                [SimpleNamespace(regulatory_chunk_id="new-neighbor")],
            ],
        ),
        patch(
            "onyx.regulatory.projection.get_bounded_adjacent_provisions",
            side_effect=[
                [SimpleNamespace(regulatory_chunk_id="old-adjacent")],
                [SimpleNamespace(regulatory_chunk_id="new-adjacent")],
            ],
        ),
    ):
        affected = _affected_amendment_row_ids(
            MagicMock(),
            all_rows=[old_chunk, new_chunk, same_position, aggregate, unrelated],
            old_chunk=old_chunk,
            new_chunk=new_chunk,
        )

    assert affected == {
        "old",
        "new",
        "same-position-version",
        "aggregate",
        "old-neighbor",
        "new-neighbor",
        "old-adjacent",
        "new-adjacent",
    }


def test_amendment_projection_embeds_only_selected_rows_and_partial_upserts() -> None:
    user_file = MagicMock()
    user_file.id = uuid4()
    user_file.name = "Mevzuat"
    settings = _settings(current=True)
    all_rows = cast(list[RegulatoryChunk], [MagicMock() for _ in range(462)])
    for ordinal, row in enumerate(all_rows):
        row.id = f"row-{ordinal}"
        row.source = "indexed"
        row.projection_ordinal = ordinal
    all_rows[461].source = "amendment"
    all_rows[90].projection_ordinal = 90
    all_rows[461].projection_ordinal = 1_000_000_066
    selected_rows = [all_rows[90], all_rows[461]]
    doc_chunks = [MagicMock(), MagicMock()]
    index_chunks = [MagicMock(), MagicMock()]
    enriched_chunks = [MagicMock(), MagicMock()]
    embedder = MagicMock()
    embedder.embedding_model.tokenizer.encode.return_value = [1]
    embedder.embed_chunks.return_value = index_chunks
    document_index = MagicMock()

    with (
        patch(
            "onyx.regulatory.projection.DefaultIndexingEmbedder.from_db_search_settings",
            return_value=embedder,
        ),
        patch(
            "onyx.regulatory.projection._rows_to_doc_aware_chunks",
            return_value=doc_chunks,
        ) as build_chunks,
        patch("onyx.regulatory.projection._contextualize_chunks") as contextualize,
        patch(
            "onyx.regulatory.projection._enrich_index_chunks",
            return_value=enriched_chunks,
        ),
        patch(
            "onyx.regulatory.projection.get_all_document_indices",
            return_value=[document_index],
        ),
        patch(
            "onyx.regulatory.projection.effective_contextual_rag_enabled",
            return_value=True,
        ),
    ):
        count = _project_amendment_rows_to_search_settings(
            user_file=user_file,
            all_rows=all_rows,
            projection_rows=selected_rows,
            search_settings=settings,
            tenant_id="tenant",
            project_ids={},
            persona_ids={},
            document_set_names={},
            user_file_access={},
        )

    assert count == 2
    assert build_chunks.call_args.args[1] == selected_rows
    assert contextualize.call_args.kwargs["rows"] == [all_rows[90], all_rows[461]]
    assert contextualize.call_args.kwargs["context_rows"] is all_rows
    embedder.embed_chunks.assert_called_once_with(doc_chunks, tenant_id="tenant")
    document_index.verify_chunk_identities.assert_called_once_with(
        document_id=str(user_file.id),
        expected_by_ordinal={
            **{ordinal: f"row-{ordinal}" for ordinal in range(461)},
            1_000_000_066: "row-461",
        },
        required_ordinals=set(range(461)),
    )
    document_index.upsert_chunks.assert_called_once_with(enriched_chunks)
    document_index.index.assert_not_called()


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
            "onyx.regulatory.projection.require_contextual_rag_llm",
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
    row.projection_ordinal = 1_000_000_064

    with patch("onyx.regulatory.projection.extract_blurb", return_value=row.text):
        chunks = _rows_to_doc_aware_chunks(document, [row], MagicMock())

    assert chunks[0].chunk_id == 1_000_000_064
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
    row.projection_ordinal = 75

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
