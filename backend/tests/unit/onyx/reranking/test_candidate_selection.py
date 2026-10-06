from onyx.reranking.candidate_selection import select_lane_candidates
from tests.unit.onyx.reranking.test_diversity import _chunk


def test_diversity_keeps_strong_prefix_and_late_independent_sources() -> None:
    dominant = [
        _chunk("dominant", index, f"Independent clause {index}") for index in range(30)
    ]
    alternatives = [
        _chunk(f"source-{index}", 0, f"Alternative {index}") for index in range(6)
    ]
    fused = dominant + alternatives
    selected = select_lane_candidates(
        fused[:12], [fused], limit=12, diversity_candidates=fused
    )

    assert selected == dominant[:6] + alternatives
    assert len({chunk.unique_id for chunk in selected}) == 12
    assert select_lane_candidates(fused[:12], [fused], limit=12) == dominant[:12]


def test_diversity_preserves_independent_lane_heads() -> None:
    dominant = [_chunk("dominant", index, f"Clause {index}") for index in range(20)]
    lane_head = _chunk("independent", 0, "Independent lane head")
    other = _chunk("other", 0, "Separate source")
    fused = dominant + [lane_head, other]
    lanes = [dominant, [lane_head]]
    baseline = select_lane_candidates(fused, lanes, limit=6)

    selected = select_lane_candidates(
        fused, lanes, limit=12, diversity_candidates=fused
    )

    assert selected[:6] == baseline
    assert other in selected
    assert len(selected) == 12


def test_diversity_does_not_cap_complementary_passages_per_source() -> None:
    clauses = [_chunk("one-source", index, f"Clause {index}") for index in range(20)]

    assert (
        select_lane_candidates(
            clauses[:12], [clauses], limit=12, diversity_candidates=clauses
        )
        == clauses[:12]
    )


def test_short_pool_and_duplicate_lanes_return_each_candidate_once() -> None:
    first = _chunk("one", 0, "First")
    second = _chunk("two", 0, "Second")
    fused = [first, first, second]

    assert select_lane_candidates(
        fused, [fused, fused], limit=12, diversity_candidates=fused
    ) == [first, second]


def test_empty_or_single_slot_preserves_fused_winner() -> None:
    first = _chunk("one", 0, "First")
    second = _chunk("two", 0, "Second")
    for limit, expected in [(0, []), (1, [first])]:
        assert (
            select_lane_candidates(
                [first, second],
                [[second]],
                limit=limit,
                diversity_candidates=[second, first],
            )
            == expected
        )
