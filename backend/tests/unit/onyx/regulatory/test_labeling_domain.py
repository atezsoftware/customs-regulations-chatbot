from __future__ import annotations

from dataclasses import dataclass

from onyx.regulatory.labeling.domain import (
    LabelingChunkView,
    bounded_document_context,
    resolve_derived_sources,
    shard_by_count_and_bytes,
)


@dataclass(frozen=True)
class _Request:
    size: int


def test_shards_respect_item_and_byte_limits() -> None:
    requests = [(str(index), _Request(size)) for index, size in enumerate([4, 4, 7])]

    shards = shard_by_count_and_bytes(
        requests,
        item_limit=2,
        byte_limit=8,
        size_of=lambda request: request.size,
    )

    assert [[item_id for item_id, _ in shard] for shard in shards] == [
        ["0", "1"],
        ["2"],
    ]


def test_shards_reject_a_single_oversized_request() -> None:
    requests = [("too-large", _Request(9))]

    try:
        shard_by_count_and_bytes(
            requests,
            item_limit=2,
            byte_limit=8,
            size_of=lambda request: request.size,
        )
    except ValueError as error:
        assert str(error) == "labeling request exceeds the shard byte limit"
    else:
        raise AssertionError("oversized request was accepted")


def test_bounded_context_keeps_target_and_document_boundaries() -> None:
    rows = [
        LabelingChunkView("a", 0, ("Title",), "A" * 80),
        LabelingChunkView("b", 1, ("Title", "Article 1"), "target"),
        LabelingChunkView("c", 2, ("Title", "Article 2"), "C" * 80),
    ]

    context = bounded_document_context(rows, target_id="b", max_utf8_bytes=150)

    assert "Document boundaries: Title | Article 2" in context
    assert "Canonical chunk: b" in context
    assert "target" in context
    assert "Context truncated" in context
    assert len(context.encode("utf-8")) <= 150


def test_derived_sources_prefer_explicit_lineage_and_reject_legacy_duplicates() -> None:
    atomics = [
        LabelingChunkView(
            "a", 0, (), "same duplicated canonical text that is long enough"
        ),
        LabelingChunkView(
            "b", 1, (), "same duplicated canonical text that is long enough"
        ),
        LabelingChunkView("c", 2, (), "unique canonical text that is also long enough"),
    ]

    assert resolve_derived_sources(
        derived_text="irrelevant",
        explicit_dependencies=("a", "c"),
        atomics=atomics,
    ).source_ids == ("a", "c")
    ambiguous = resolve_derived_sources(
        derived_text="same duplicated canonical text that is long enough\nunique canonical text that is also long enough",
        explicit_dependencies=(),
        atomics=atomics,
    )
    assert ambiguous.source_ids == ()
    assert ambiguous.resolution == "unresolved"
    assert ambiguous.unresolved_reason == "ambiguous_legacy_containment"


def test_derived_sources_reject_incomplete_explicit_lineage() -> None:
    atomics = [LabelingChunkView("a", 0, (), "A sufficiently long atomic text")]

    resolution = resolve_derived_sources(
        derived_text=atomics[0].text,
        explicit_dependencies=("a", "missing"),
        atomics=atomics,
    )

    assert resolution.source_ids == ()
    assert resolution.unresolved_reason == "missing_explicit_dependency"
