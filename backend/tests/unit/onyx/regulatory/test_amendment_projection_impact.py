from datetime import date
from uuid import uuid4

import pytest

from onyx.regulatory.amendment_projection_impact import (
    ContextImpactDecision,
    changed_windows,
    merge_windows,
    validate_context_decisions,
)
from tests.unit.onyx.regulatory.indexing_jobs.owned_publication_test_helpers import (
    canonical_row,
    writer_inputs,
)


def test_replacement_affects_only_successor_lifetime() -> None:
    file_id = uuid4()
    before = writer_inputs(file_id, [canonical_row(file_id, 0, "old")]).canonical
    boundary = date(2027, 1, 1)
    after = [
        before[0].model_copy(
            update={"validity_end_date": boundary, "status": "superseded"}
        ),
        before[0].model_copy(
            update={"id": "new", "text": "new", "validity_start_date": boundary}
        ),
    ]
    assert changed_windows(before, after) == {
        before[0].id: [(boundary, date.max)],
        "new": [(boundary, date.max)],
    }


def test_status_and_bookkeeping_do_not_invalidate_historical_embedding() -> None:
    file_id = uuid4()
    before = writer_inputs(file_id, [canonical_row(file_id, 0, "same")]).canonical
    after = [before[0].model_copy(update={"superseded_by_chunk_id": "new"})]
    assert changed_windows(before, after) == {}


def test_disjoint_effective_windows_are_not_widened() -> None:
    a, b, c, d = [date(year, 1, 1) for year in (2025, 2026, 2027, 2028)]
    assert merge_windows([(c, d), (a, b), (a, b)]) == [(a, b), (c, d)]


@pytest.mark.parametrize(
    "decisions",
    [
        [],
        [
            ContextImpactDecision(
                key="foreign", affected=False, quote="", reason="unchanged"
            )
        ],
        [
            ContextImpactDecision(
                key="a", affected=True, quote="invented", reason="change"
            )
        ],
    ],
)
def test_context_audit_rejects_incomplete_foreign_or_unsupported_decisions(
    decisions: list[ContextImpactDecision],
) -> None:
    with pytest.raises(ValueError):
        validate_context_decisions({"a": "The rate is 5%."}, decisions)


def test_context_audit_selects_only_grounded_stale_context() -> None:
    decisions = [
        ContextImpactDecision(
            key="a", affected=True, quote="rate is 5%", reason="The rate changed to 7%."
        ),
        ContextImpactDecision(
            key="b", affected=False, quote="", reason="Generic scope remains true."
        ),
    ]
    assert validate_context_decisions(
        {"a": "The rate is 5%.", "b": "Import regulation."}, decisions
    ) == {"a"}


def test_context_audit_checks_retained_intervals_and_qualifies_decisions_by_date() -> (
    None
):
    import json
    from unittest.mock import MagicMock, patch

    from onyx.regulatory import amendment_projection_impact as impact

    file_id = uuid4()
    before = writer_inputs(
        file_id, [canonical_row(file_id, i, f"rule {i}") for i in range(3)]
    ).canonical
    early, late = date(2027, 1, 1), date(2028, 1, 1)
    after = [
        r.model_copy(update={"validity_end_date": early if i == 1 else late})
        if i < 2
        else r
        for i, r in enumerate(before)
    ]
    from onyx.regulatory.amendments.annexes.models import AnnexTemporalProjection

    bindings: list[AnnexTemporalProjection] = [
        MagicMock(
            effective_start=None,
            effective_end=None,
            projection=MagicMock(
                source_json=json.dumps(
                    {
                        "regulatory_chunk_id": r.id,
                        "chunk_context": "Rule one still applies.",
                    }
                )
            ),
        )
        for r in (before[0], before[2])
    ]
    seen = []

    def audit(*_args: object, **kwargs: object):
        cases = json.loads(str(kwargs["user_prompt"]))["contexts"]
        seen.extend(cases.values())
        return impact.ContextImpactResult(
            decisions=[
                impact.ContextImpactDecision(
                    key=k,
                    affected=v["effective_start"] == early.isoformat(),
                    quote="Rule one still applies.",
                    reason="Rule one was removed in this interval.",
                )
                for k, v in cases.items()
            ]
        )

    with patch.object(impact, "generate_structured", side_effect=audit):
        result = impact.include_context_consumers(
            impact.structural_windows(before, after),
            before=before,
            after=after,
            bindings=bindings,
            llm=MagicMock(),
        )
    assert result[before[0].id] == [(early, date.max)]
    assert result[before[2].id] == [(early, late)]
    assert len(seen) == 2  # identical context shared by consumers, once per interval
    assert seen[0]["effective_end"] == late.isoformat()


def test_structural_dependencies_do_not_expand_to_unrelated_rows() -> None:
    from onyx.regulatory.amendment_projection_impact import structural_windows

    file_id = uuid4()
    before = writer_inputs(
        file_id, [canonical_row(file_id, i, f"rule {i}") for i in range(4)]
    ).canonical
    before[1] = before[1].model_copy(
        update={"metadata": {"source_regulatory_chunk_ids": [before[0].id]}}
    )
    before[2] = before[2].model_copy(
        update={"metadata": {"bound_to_regulatory_chunk_id": before[1].id}}
    )
    after = [
        r.model_copy(update={"text": "changed rule"}) if i == 0 else r
        for i, r in enumerate(before)
    ]
    assert structural_windows(before, after) == {
        r.id: [(date.min, date.max)] for r in before[:3]
    }
