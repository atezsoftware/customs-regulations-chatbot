from copy import deepcopy
from datetime import date
from unittest.mock import MagicMock, patch
from uuid import UUID, uuid4

import pytest

from onyx.db.models import RegulatoryChunk, SearchSettings, UserFile
from onyx.regulatory.amendments.annexes.context_dependencies import (
    effective_context_rows,
    validate_complete_context_view,
)
from onyx.regulatory.amendments.annexes.publication_representations import _snapshot
from onyx.regulatory.document_intake.temporal_context import (
    intake_temporal_context_rows,
)

JANUARY = date(2025, 1, 5)
MAY = date(2025, 5, 9)
JUNE = date(2025, 6, 1)


def _row(
    file_id: UUID,
    identifier: str,
    position: int,
    *,
    start: date | None = None,
    end: date | None = None,
    sources: list[str] | None = None,
    companion: str | None = None,
    text: str | None = None,
) -> RegulatoryChunk:
    metadata: dict[str, object] = {"intake_chunking_version": "1"}
    if sources is not None:
        metadata.update(
            chunk_variant="hierarchical_aggregate",
            source_regulatory_chunk_ids=sources,
            hierarchy_root_path=["MADDE 6"],
        )
    if companion is not None:
        metadata["bound_to_regulatory_chunk_id"] = companion
    return RegulatoryChunk(
        id=identifier,
        user_file_id=file_id,
        position=position,
        projection_ordinal=position,
        text=text or identifier,
        heading_path=[identifier],
        chunk_metadata=metadata,
        chunk_type="article",
        source="indexed",
        status="active",
        validity_start_date=start,
        validity_end_date=end,
    )


def _dated_rows(file_id: UUID) -> list[RegulatoryChunk]:
    return [
        _row(file_id, "header", 0, text="Document header"),
        _row(file_id, "early", 1, start=JANUARY, text="Early provision"),
        _row(file_id, "future", 2, start=MAY, text="HIDDEN future provision"),
        _row(
            file_id,
            "aggregate",
            3,
            sources=["early", "future"],
            text="Early provision and HIDDEN future provision",
        ),
    ]


@pytest.mark.parametrize(
    ("when", "expected"),
    [
        (date(2025, 1, 4), ["header"]),
        (JANUARY, ["header", "early"]),
        (date(2025, 5, 8), ["header", "early"]),
        (MAY, ["header", "early", "future", "aggregate"]),
    ],
)
def test_complete_derived_text_requires_all_sources_without_changing_canonical(
    when: date, expected: list[str]
) -> None:
    rows = _dated_rows(uuid4())
    before = [_snapshot(row) for row in rows]
    qualified = intake_temporal_context_rows(rows)
    assert [row.id for row in effective_context_rows(qualified, when)] == expected
    assert qualified[:3] == rows[:3]
    assert qualified[3] is not rows[3]
    assert qualified[3].validity_start_date == MAY
    assert qualified[3].text == rows[3].text
    assert qualified[3].chunk_metadata == rows[3].chunk_metadata
    assert [_snapshot(row) for row in rows] == before


def test_source_end_and_parent_limits_propagate_through_nested_and_companion() -> None:
    identifier = uuid4()
    rows = [
        _row(identifier, "a", 0, start=JANUARY, end=JUNE),
        _row(identifier, "b", 1, start=MAY),
        _row(identifier, "aggregate", 2, sources=["a", "b"]),
        _row(identifier, "parent", 3, start=date(2025, 5, 10), sources=["aggregate"]),
        _row(identifier, "image", 4, companion="parent"),
    ]
    before = [_snapshot(row) for row in rows]
    qualified = {row.id: row for row in intake_temporal_context_rows(rows)}
    assert (
        qualified["aggregate"].validity_start_date,
        qualified["aggregate"].validity_end_date,
    ) == (MAY, JUNE)
    for key in ("parent", "image"):
        assert (
            qualified[key].validity_start_date,
            qualified[key].validity_end_date,
        ) == (date(2025, 5, 10), JUNE)
    assert [
        row.id for row in effective_context_rows(list(qualified.values()), JUNE)
    ] == ["b"]
    assert [_snapshot(row) for row in rows] == before


def test_disjoint_half_open_sources_omit_only_derived_chain() -> None:
    identifier = uuid4()
    rows = [
        _row(identifier, "a", 0, start=JANUARY, end=MAY),
        _row(identifier, "b", 1, start=MAY),
        _row(identifier, "aggregate", 2, sources=["a", "b"]),
        _row(identifier, "nested", 3, sources=["aggregate"]),
        _row(identifier, "image", 4, companion="nested"),
    ]
    assert intake_temporal_context_rows(rows) == rows[:2]
    assert rows[2].validity_start_date is None and rows[2].validity_end_date is None


@pytest.mark.parametrize(
    "failure", ["missing", "cycle", "empty", "foreign", "duplicate"]
)
def test_invalid_derived_graph_fails_even_before_its_date(failure: str) -> None:
    file_id = uuid4()
    rows = [
        _row(file_id, "a", 0, start=MAY),
        _row(file_id, "aggregate", 1, sources=["a"]),
    ]
    if failure == "missing":
        rows[1].chunk_metadata["source_regulatory_chunk_ids"] = ["not-in-file"]
    elif failure == "cycle":
        rows.append(_row(file_id, "cycle", 2, sources=["aggregate"]))
        rows[1].chunk_metadata["source_regulatory_chunk_ids"] = ["cycle"]
    elif failure == "empty":
        rows[1].chunk_metadata["source_regulatory_chunk_ids"] = []
    elif failure == "foreign":
        rows[0].user_file_id = uuid4()
    else:
        rows.append(rows[0])
    with pytest.raises(ValueError, match="Intake"):
        intake_temporal_context_rows(rows)


def test_ordinary_update_rows_keep_exact_identity_and_existing_semantics() -> None:
    rows = _dated_rows(uuid4())
    for row in rows:
        row.chunk_metadata.pop("intake_chunking_version")
    before = [_snapshot(row) for row in rows]
    assert intake_temporal_context_rows(rows) is rows
    assert [_snapshot(row) for row in rows] == before


@pytest.mark.parametrize("when", [JANUARY, MAY])
def test_normal_context_sources_and_consumers_follow_derived_source_dates(
    when: date,
) -> None:
    from onyx.regulatory.projection import prepare_normal_context_view
    from tests.unit.onyx.regulatory.annexes.test_context_dependencies import (
        _durable_embedding_model,
    )

    file = UserFile(id=uuid4(), name="Regulation")
    rows = _dated_rows(file.id)
    before = [_snapshot(row) for row in rows]
    settings = SearchSettings(
        id=1,
        model_name="embedding",
        model_dim=3,
        normalize=True,
        enable_contextual_rag=False,
    )
    embedder = MagicMock(embedding_model=_durable_embedding_model())
    with patch(
        "onyx.regulatory.projection.effective_contextual_rag_enabled",
        return_value=False,
    ):
        view = prepare_normal_context_view(
            rows=rows,
            user_file=file,
            search_settings=settings,
            embedder=embedder,
            llm=None,
            as_of_date=when,
        )
    ids = [projection.canonical_chunk_id for projection in view.projections]
    assert ids == (["header", "early"] if when == JANUARY else [row.id for row in rows])
    assert all(
        ("HIDDEN" in snapshot.text) == (when == MAY) for snapshot in view.snapshots
    )
    validate_complete_context_view(
        rows=intake_temporal_context_rows(rows), view=view, as_of_date=when
    )
    assert [_snapshot(row) for row in rows] == before


@pytest.mark.parametrize("when", [JANUARY, MAY])
def test_durable_context_prompts_snapshots_and_cache_use_the_same_dated_rows(
    when: date,
) -> None:
    from onyx.regulatory.indexing_jobs.contextual import (
        ContextualRequestFactory,
        prepare_durable_context_view,
    )
    from tests.unit.onyx.regulatory.annexes.test_context_dependencies import (
        _durable_embedding_model,
        _durable_fixture,
    )

    job, _source, tokenizer = _durable_fixture()
    rows = _dated_rows(job.user_file_id)
    before = deepcopy([_snapshot(row) for row in rows])
    generate = MagicMock(return_value="Short verified context")
    with patch(
        "onyx.regulatory.indexing_jobs.contextual._contextual_safe_input_limit",
        return_value=10000,
    ):
        factory = ContextualRequestFactory(
            job=job,
            rows=rows,
            contextual_tokenizer=tokenizer,
            embedding_tokenizer=tokenizer,
            reference_date_override=when,
        )
        assert ("HIDDEN" in factory.request(rows[1]).prompt) == (when == MAY)
        view = prepare_durable_context_view(
            job=job,
            rows=rows,
            contextual_tokenizer=tokenizer,
            embedding_tokenizer=tokenizer,
            embedding_model=_durable_embedding_model(),
            generate=generate,
            as_of_date=when,
        )
        calls = generate.call_count
        replay = prepare_durable_context_view(
            job=job,
            rows=rows,
            contextual_tokenizer=tokenizer,
            embedding_tokenizer=tokenizer,
            embedding_model=_durable_embedding_model(),
            generate=generate,
            as_of_date=when,
            cached=view,
        )
    assert replay == view and generate.call_count == calls
    ids = [projection.canonical_chunk_id for projection in view.projections]
    assert ids == (["header", "early"] if when == JANUARY else [row.id for row in rows])
    assert all(
        ("HIDDEN" in snapshot.text) == (when == MAY) for snapshot in view.snapshots
    )
    validate_complete_context_view(
        rows=intake_temporal_context_rows(rows), view=view, as_of_date=when
    )
    assert [_snapshot(row) for row in rows] == before
