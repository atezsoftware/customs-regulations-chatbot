from uuid import uuid4

import pytest

from onyx.configs.constants import DocumentSource
from onyx.context.search.models import InferenceChunk
from onyx.regulatory.labeling.defaults import load_default_taxonomy
from onyx.regulatory.labeling.search_models import (
    LabelSearchHint,
    LabelSearchSnapshot,
    SearchLabelEvidence,
)
from onyx.regulatory.labeling.search_ranking import (
    fuse_label_candidate_scores,
    merge_label_candidates,
    resolve_label_facets,
    validate_search_hint,
)


def chunk(identifier: str) -> InferenceChunk:
    return InferenceChunk(
        document_id="file",
        chunk_id=int(identifier),
        regulatory_chunk_id=identifier,
        content="Source text",
        blurb="",
        source_links={},
        section_continuation=False,
        semantic_identifier="source",
        source_type=DocumentSource.USER_FILE,
        boost=1,
        score=1,
        hidden=False,
        metadata={},
        match_highlights=[],
        doc_summary="",
        chunk_context="",
        image_file_id=None,
        title=None,
        updated_at=None,
    )


def snapshot() -> LabelSearchSnapshot:
    return LabelSearchSnapshot(
        tenant_id="test",
        run_ids=(uuid4(),),
        taxonomy=load_default_taxonomy(),
        mode="hybrid",
    )


def evidence(identifier: str, label: str) -> SearchLabelEvidence:
    return SearchLabelEvidence(
        chunk_id=identifier,
        source_chunk_id=identifier,
        label_id=label,
        evidence_quote="Source text",
        source_text_sha256="a" * 64,
        context_sha256="b" * 64,
        run_id=uuid4(),
    )


def test_hint_validation_fails_open_without_unknown_or_effect_only_queries() -> None:
    state = snapshot()
    assert validate_search_hint({"label_ids": ["made.up"]}, state).label_ids == ()
    assert validate_search_hint({"label_ids": "SUB.TAX.VAT"}, state).label_ids == ()
    hint = validate_search_hint({"label_ids": ["EFF.EXEMPTION"]}, state)
    assert hint.candidate_label_ids(state.taxonomy) == ()
    assert validate_search_hint({"label_ids": ["SUB.TAX.VAT"]}, state).label_ids == (
        "SUB.TAX.VAT",
    )


def test_custom_prefix_is_not_a_facet_contract() -> None:
    state = snapshot()
    custom = state.taxonomy.model_copy(
        update={
            "labels": [
                state.taxonomy.labels[0].model_copy(
                    update={"description": "Different meaning"}
                )
            ]
        }
    )
    assert resolve_label_facets(custom)["SUB.CUS"] == "untyped"


def test_verified_label_lane_contributes_a_full_rank_fusion_score() -> None:
    baseline = [chunk(str(i)) for i in range(1, 9)]
    extra = [chunk("9"), chunk("10")]
    ranked, scores = fuse_label_candidate_scores(
        baseline, extra, label_ranked_ids=["10", "9", "2"], limit=8
    )
    assert [row.regulatory_chunk_id for row in ranked[:4]] == [
        "2",
        "1",
        "10",
        "9",
    ]
    assert {row.regulatory_chunk_id for row in ranked} >= {"1", "2", "3", "4", "5", "6"}
    assert scores[0].combined_score == pytest.approx(
        scores[0].baseline_score + scores[0].label_score
    )
    assert scores[0].label_score > 0
    assert baseline == [chunk(str(i)) for i in range(1, 9)]
    assert fuse_label_candidate_scores(
        baseline, extra, label_ranked_ids=[], limit=8
    ) == (merge_label_candidates(baseline, extra, limit=8), ())


def test_merge_reserves_baseline_and_deduplicates_sources() -> None:
    baseline = [chunk(str(i)) for i in range(1, 9)]
    added = [chunk("1"), chunk("9"), chunk("10")]
    merged = merge_label_candidates(baseline, added, limit=8)
    assert len(merged) == 8
    assert {c.regulatory_chunk_id for c in baseline[:6]} <= {
        c.regulatory_chunk_id for c in merged
    }
    assert {c.regulatory_chunk_id for c in merged} == {
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "9",
        "10",
    }
    assert merge_label_candidates(baseline, [], limit=8) == baseline


def test_evidence_requires_source_hash_and_bounded_quote() -> None:
    with pytest.raises(ValueError):
        evidence("1", "SUB.TAX.VAT").model_copy().model_validate({"chunk_id": "1"})


def test_bad_optional_plan_hints_preserve_research_question() -> None:
    from onyx.regulatory.coverage_plan import RegulatoryCoverageItem

    item = RegulatoryCoverageItem.model_validate(
        dict(
            research_question="VAT deduction",
            completion_test="Find the deduction rule",
            retrieval_queries=["import VAT deduction"],
            label_hints="malformed",
        )
    )
    assert item.research_question == "VAT deduction"
    assert item.label_hints == []


def test_hint_is_bound_to_its_subquery() -> None:
    from onyx.regulatory.coverage_plan import RegulatoryCoverageItem
    from onyx.regulatory.labeling.search_hints import hint_for_query

    item = RegulatoryCoverageItem.model_validate(
        dict(
            research_question="VAT and duty",
            completion_test="Both rules",
            retrieval_queries=["VAT deduction", "customs value"],
            label_hints=[{"query": "VAT deduction", "label_ids": ["SUB.TAX.VAT"]}],
        )
    )
    assert hint_for_query([item], "VAT deduction") == {"label_ids": ["SUB.TAX.VAT"]}
    assert hint_for_query([item], "customs value") == {"label_ids": []}


def test_disabled_runtime_never_opens_database(monkeypatch: pytest.MonkeyPatch) -> None:
    from onyx.regulatory.labeling import search_runtime

    monkeypatch.setattr(
        search_runtime, "label_read_session", lambda: pytest.fail("DB opened")
    )
    assert search_runtime.search_snapshot_for_run_ids(()) is None


def test_runtime_preserves_baseline_on_overlay_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from datetime import date
    from unittest.mock import MagicMock

    from sqlalchemy.exc import OperationalError

    from onyx.regulatory.labeling import search_runtime

    baseline = [chunk("1"), chunk("2")]
    monkeypatch.setattr(search_runtime, "get_current_tenant_id", lambda: "test")
    monkeypatch.setattr(
        search_runtime,
        "label_read_session",
        MagicMock(side_effect=OperationalError("read", {}, Exception("offline"))),
    )
    result = search_runtime.search_with_labels(
        baseline,
        snapshot=snapshot(),
        raw_hint={"label_ids": ["SUB.TAX.VAT"]},
        as_of_date=date.today(),
        limit=8,
        retrieve=lambda _ids: pytest.fail("retrieved on DB failure"),
    )
    assert result.candidates == baseline
    assert result.evidence_by_chunk == {}


@pytest.mark.parametrize(
    "invalid_extra", ["wrong_id", "stale_text", "missing_evidence"]
)
def test_extra_candidates_require_matching_current_evidence(
    monkeypatch: pytest.MonkeyPatch, invalid_extra: str
) -> None:
    from datetime import date
    from unittest.mock import MagicMock

    from onyx.regulatory.labeling import search_runtime
    from onyx.regulatory.labeling.search_models import LabelSearchOverlay

    baseline = [chunk("1"), chunk("2")]
    extra = chunk("3")
    overlay = LabelSearchOverlay(
        candidate_ids=("4",) if invalid_extra == "wrong_id" else ("3",),
        source_texts={
            "3": "old text" if invalid_extra == "stale_text" else extra.content
        },
        evidence_by_chunk={}
        if invalid_extra == "missing_evidence"
        else {"3": (evidence("3", "SUB.TAX.VAT"),)},
    )
    monkeypatch.setattr(search_runtime, "get_current_tenant_id", lambda: "test")
    monkeypatch.setattr(search_runtime, "label_read_session", MagicMock())
    monkeypatch.setattr(
        search_runtime, "load_label_overlay", MagicMock(return_value=overlay)
    )
    result = search_runtime.search_with_labels(
        baseline,
        snapshot=snapshot(),
        raw_hint={"label_ids": ["SUB.TAX.VAT"]},
        as_of_date=date.today(),
        limit=8,
        retrieve=lambda _ids: [extra],
    )
    assert result.candidates == baseline


def test_empty_overlay_preserves_full_baseline_before_later_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from datetime import date
    from unittest.mock import MagicMock

    from onyx.regulatory.labeling import search_runtime
    from onyx.regulatory.labeling.search_models import LabelSearchOverlay

    baseline = [chunk(str(i)) for i in range(100)]
    monkeypatch.setattr(search_runtime, "get_current_tenant_id", lambda: "test")
    monkeypatch.setattr(search_runtime, "label_read_session", MagicMock())
    monkeypatch.setattr(
        search_runtime,
        "load_label_overlay",
        MagicMock(return_value=LabelSearchOverlay()),
    )
    result = search_runtime.search_with_labels(
        baseline,
        snapshot=snapshot(),
        raw_hint={"label_ids": ["SUB.TAX.VAT"]},
        as_of_date=date.today(),
        limit=48,
        retrieve=lambda _ids: [],
    )
    assert result.candidates == baseline


def test_explicit_subject_hint_survives_optional_planner_omission() -> None:
    from onyx.regulatory.labeling.search_hints import explicit_subject_hint

    state = snapshot()
    assert explicit_subject_hint(state, "İthalatta KDV'nin indirimi").label_ids == (
        "SUB.TAX.VAT",
    )
    assert explicit_subject_hint(state, "Katma değer vergisi indirimi").label_ids == (
        "SUB.TAX.VAT",
    )
    assert (
        explicit_subject_hint(state, "Hangi istisnalar ve şartlar uygulanır?").label_ids
        == ()
    )
    assert explicit_subject_hint(state, "KDVXYZ adlı dosyada arama").label_ids == ()


def test_subject_hint_uses_query_after_source_neutral_planning() -> None:
    from onyx.regulatory.labeling.search_hints import query_subject_hint

    state = snapshot()
    assert "SUB.CUS.GUAR" in query_subject_hint(state, "teminat çözümü").label_ids
    assert query_subject_hint(state, "hangi şartlar uygulanır").label_ids == ()


@pytest.mark.parametrize(
    "scores, expected",
    [
        ([0.95, 0.8, 0.79, 0.78], ["1", "4", "2", "3"]),
        ([0.95, 0.8, 0.6, 0.59], ["1", "2", "4", "3"]),
        ([0.95, 0.8, 0.79, None], ["1", "2", "3", "4"]),
        ([0.95, 0.8, 0.79, float("nan")], ["1", "2", "3", "4"]),
    ],
)
def test_post_rerank_labels_only_promote_near_tied_verified_sources(scores, expected):
    from onyx.regulatory.labeling import search_ranking

    candidates = [chunk(str(i)) for i in range(1, 5)]
    before = [c.model_dump() for c in candidates]
    ranked = search_ranking.rank_near_tied_label_candidates(
        candidates,
        scores={("file", i + 1): s for i, s in enumerate(scores) if s is not None},
        evidence={"4": (evidence("4", "SUB.TAX.VAT"),)},
        hint=LabelSearchHint(label_ids=("SUB.TAX.VAT",)),
    )
    assert [c.regulatory_chunk_id for c in ranked] == expected
    assert [c.model_dump() for c in candidates] == before


@pytest.mark.parametrize("proposals", [["3", "4"], []])
def test_relevant_discovery_intersects_labels_without_global_candidate_fallback(
    monkeypatch: pytest.MonkeyPatch, proposals: list[str]
):
    from datetime import date
    from unittest.mock import MagicMock

    from onyx.regulatory.labeling import search_runtime
    from onyx.regulatory.labeling.search_models import LabelSearchOverlay

    monkeypatch.setattr(search_runtime, "get_current_tenant_id", lambda: "test")
    monkeypatch.setattr(search_runtime, "label_read_session", MagicMock())
    overlay = LabelSearchOverlay(
        candidate_ids=("4",) if proposals else (),
        evidence_by_chunk={"4": (evidence("4", "SUB.TAX.VAT"),)} if proposals else {},
        source_texts={"4": "Source text"} if proposals else {},
    )

    def load(_session: object, **kwargs: object) -> LabelSearchOverlay:
        assert kwargs["candidate_chunk_ids"] == tuple(proposals)
        return overlay

    monkeypatch.setattr(search_runtime, "load_label_overlay", load)
    result = search_runtime.search_with_labels(
        [chunk("1"), chunk("2")],
        snapshot=snapshot(),
        raw_hint={"label_ids": ["SUB.TAX.VAT"]},
        as_of_date=date.today(),
        limit=8,
        retrieve=lambda _ids: pytest.fail(
            "Already authorized search results must be reused"
        ),
        discover=lambda _hint: [chunk("1"), *(chunk(p) for p in proposals)],
    )
    assert {c.regulatory_chunk_id for c in result.candidates} == (
        {"1", "2", "4"} if proposals else {"1", "2"}
    )
    assert (result.hint.label_ids == ("SUB.TAX.VAT",)) == bool(proposals)


def test_verified_extra_reaches_top_without_external_reranker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from datetime import date
    from unittest.mock import MagicMock

    from onyx.regulatory.labeling import search_runtime
    from onyx.regulatory.labeling.search_models import LabelSearchOverlay

    monkeypatch.setattr(search_runtime, "get_current_tenant_id", lambda: "test")
    monkeypatch.setattr(search_runtime, "label_read_session", MagicMock())
    monkeypatch.setattr(
        search_runtime,
        "load_label_overlay",
        lambda *_args, **_kwargs: LabelSearchOverlay(
            candidate_ids=("49",),
            evidence_by_chunk={"49": (evidence("49", "SUB.TAX.VAT"),)},
            source_texts={"49": "Source text"},
        ),
    )
    baseline = [chunk(str(i)) for i in range(1, 49)]
    result = search_runtime.search_with_labels(
        baseline,
        snapshot=snapshot(),
        raw_hint={"label_ids": ["SUB.TAX.VAT"]},
        as_of_date=date.today(),
        limit=48,
        retrieve=lambda _ids: pytest.fail("Discovery candidates should be reused"),
        discover=lambda _hint: [chunk("49")],
    )
    assert [candidate.regulatory_chunk_id for candidate in result.candidates[:3]] == [
        "1",
        "49",
        "2",
    ]
    assert len(result.candidates) == 48
    assert result.fusion_scores[1].label_score > 0
    assert [candidate.regulatory_chunk_id for candidate in baseline] == [
        str(i) for i in range(1, 49)
    ]


def test_extra_run_receives_validation_budget_before_large_baseline_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from unittest.mock import MagicMock

    from onyx.db import regulatory_label_search as search
    from onyx.db.models import RegulatoryLabelingItem, RegulatoryLabelingRun

    first_run, second_run = uuid4(), uuid4()
    clock = [0.0]
    monkeypatch.setattr(search, "monotonic", lambda: clock[0])

    def validate(
        _session: object,
        *,
        run: RegulatoryLabelingRun,
        items: list[RegulatoryLabelingItem],
        deadline: float,
    ) -> frozenset[str]:
        valid = set()
        assert all(item.run_id == run.id for item in items)
        for item in items:
            if clock[0] >= deadline:
                break
            clock[0] += 0.4
            valid.add(item.regulatory_chunk_id)
        return frozenset(valid)

    monkeypatch.setattr(search, "current_labeling_items_for_search", validate)
    baseline = [
        RegulatoryLabelingItem(run_id=first_run, regulatory_chunk_id=str(i))
        for i in range(10)
    ]
    extra = RegulatoryLabelingItem(run_id=second_run, regulatory_chunk_id="extra")
    result = search._validate_items(
        MagicMock(),
        runs={
            first_run: RegulatoryLabelingRun(id=first_run),
            second_run: RegulatoryLabelingRun(id=second_run),
        },
        items=[baseline[0], extra, *baseline[1:]],
        baseline_ids={str(i) for i in range(10)},
        deadline=1.5,
    )
    assert "extra" in result
    assert "0" in result
    assert len(result) < 11
