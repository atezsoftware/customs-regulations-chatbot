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


def test_published_successor_membership_cannot_be_silently_omitted() -> None:
    import json
    from unittest.mock import MagicMock

    from onyx.regulatory.amendment_projection_impact import analyze_amendment_impact
    from tests.unit.onyx.regulatory.annexes.test_publication_timeline import snapshot

    old = snapshot("old", None, date(2025, 1, 1))
    current = snapshot("current", date(2025, 1, 1), None)
    aggregate = snapshot("aggregate", None, None).model_copy(
        update={
            "metadata": {
                "chunk_variant": "hierarchical_aggregate",
                "source_regulatory_chunk_ids": ["old"],
            }
        }
    )
    binding = MagicMock(
        derived_role="hierarchical_aggregate",
        dependency_ids=["current"],
        representation_metadata={"source_regulatory_chunk_ids": ["current"]},
        effective_start=date(2025, 1, 1),
        effective_end=None,
        projection=MagicMock(
            source_json=json.dumps({"regulatory_chunk_id": "aggregate"})
        ),
    )
    report = analyze_amendment_impact(
        before=[old, current, aggregate],
        after=[old, current.model_copy(update={"text": "changed"}), aggregate],
        bindings=[binding],
        llm=None,
    )
    assert report.unresolved
    assert "published source membership" in report.unresolved[0]
    assert report.unchanged_ids == []


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
                    source_id=before[1].id,
                    source_side="before",
                    source_quote="rule 1",
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


def test_context_audit_rejects_unproven_source_and_uncertainty() -> None:
    decisions = [
        ContextImpactDecision(
            key="consumer",
            affected=True,
            quote="rate is 5%",
            reason="changed rate",
        )
    ]
    with pytest.raises(ValueError, match="changed source"):
        validate_context_decisions(
            {"consumer": "The rate is 5%."},
            decisions,
            changes={
                "consumer": {
                    "before": {"old": "The rate is 5%."},
                    "after": {"new": "The rate is 7%."},
                }
            },
        )


def test_structural_impact_does_not_include_expired_consumer() -> None:
    from onyx.regulatory.amendment_projection_impact import structural_windows
    from tests.unit.onyx.regulatory.annexes.test_publication_timeline import snapshot

    old = snapshot("source", None, None)
    expired = snapshot("expired", None, date(2025, 1, 1)).model_copy(
        update={"metadata": {"source_regulatory_chunk_ids": [old.id]}}
    )
    after = [old.model_copy(update={"validity_end_date": date(2027, 1, 1)}), expired]
    assert structural_windows([old, expired], after) == {
        "source": [(date(2027, 1, 1), date.max)]
    }


def test_source_usage_traverses_shared_consumers_and_preserves_unrelated_text() -> None:
    from onyx.regulatory.amendment_projection_impact import find_source_consumers
    from tests.unit.onyx.regulatory.annexes.test_publication_timeline import snapshot

    source = snapshot("source", None, None)
    parent = snapshot("parent", None, None).model_copy(
        update={
            "metadata": {
                "source_regulatory_chunk_ids": ["source"],
                "chunk_variant": "hierarchical_aggregate",
            }
        }
    )
    ancestor = snapshot("ancestor", None, None).model_copy(
        update={
            "metadata": {
                "source_regulatory_chunk_ids": ["source", "parent"],
                "chunk_variant": "hierarchical_aggregate",
            }
        }
    )
    unrelated = snapshot("unrelated", None, None)
    found = find_source_consumers([source, parent, ancestor, unrelated], {source.id})
    assert {r.consumer_id for r in found} == {"parent", "ancestor"}
    assert len(found) == 2
    assert all(r.source_id == "source" and r.relation == "aggregate" for r in found)
    assert all(r.path[0] == "source" and r.path[-1] == r.consumer_id for r in found)


@pytest.mark.parametrize(
    "fault",
    [None, "uncertain", "wrong_source", "missing", "transient", "repair_source_quote"],
)
def test_context_evidence_selects_actual_consumer_and_keeps_generic_summary(
    monkeypatch: pytest.MonkeyPatch,
    fault: str | None,
) -> None:
    import json
    from unittest.mock import MagicMock

    from onyx.llm.interfaces import LLMConfig
    from onyx.regulatory import amendment_projection_impact as impact
    from tests.unit.onyx.regulatory.annexes.test_publication_timeline import snapshot

    old = snapshot("old", None, None).model_copy(update={"text": "Import fee is 5%."})
    consumer, generic = (
        snapshot("consumer", None, None),
        snapshot("generic", None, None),
    )
    boundary = date(2027, 1, 1)
    before = [old, consumer, generic]
    after = [
        old.model_copy(update={"validity_end_date": boundary}),
        consumer,
        generic,
        snapshot("new", boundary, None).model_copy(
            update={"text": "Import fee is 7%.", "supersedes_chunk_id": "old"}
        ),
    ]
    from onyx.regulatory.amendments.annexes.models import AnnexTemporalProjection

    bindings: list[AnnexTemporalProjection] = [
        MagicMock(
            effective_start=None,
            effective_end=None,
            projection=MagicMock(
                source_json=json.dumps(
                    {
                        "regulatory_chunk_id": identifier,
                        "chunk_context": context,
                    }
                )
            ),
        )
        for identifier, context in [
            ("consumer", "The fee is 5%."),
            ("generic", "Import rules."),
        ]
    ]
    llm = MagicMock()
    llm.config = LLMConfig(
        model_provider="configured",
        model_name="context",
        temperature=0,
        max_input_tokens=10000,
    )

    attempts = 0

    def audit(*_args: object, **kwargs: object) -> impact.ContextImpactResult:
        nonlocal attempts
        attempts += 1
        data = json.loads(str(kwargs["user_prompt"]))
        if fault == "repair_source_quote" and attempts == 2:
            feedback = data["validation_feedback"]
            assert "context 0" in feedback["error"]
            assert feedback["previous_decisions"][0]["source_quote"] == "fee is 7 percent"
        decisions = [
            ContextImpactDecision(
                key=key,
                affected=case["text"] == "The fee is 5%.",
                quote="fee is 5%",
                reason="The rate changes; the generic scope does not.",
                source_side="after",
                source_id="new" if fault != "wrong_source" else "foreign",
                source_quote="fee is 7 percent"
                if fault == "repair_source_quote" and attempts == 1
                else "fee is 7%",
                uncertain=fault == "uncertain",
            )
            for key, case in data["contexts"].items()
        ]
        if fault == "missing" or fault == "transient" and attempts == 1:
            decisions.pop()
        return impact.ContextImpactResult(decisions=decisions)

    monkeypatch.setattr(impact, "generate_structured", audit)
    from collections.abc import Callable

    cache: dict[str, impact.ContextImpactResult] = {}

    def resolve(
        key: str, generate: Callable[[], impact.ContextImpactResult]
    ) -> impact.ContextImpactResult:
        if key not in cache:
            cache[key] = generate()
        return cache[key]

    report = impact.analyze_amendment_impact(
        before=before, after=after, bindings=bindings, llm=llm, audit_cache=resolve
    )
    if fault not in {None, "transient", "repair_source_quote"}:
        assert report.unresolved
        assert report.unchanged_ids == []
    else:
        assert set(report.affected_windows) == {"old", "new", "consumer"}
        assert report.unchanged_ids == ["generic"]
        assert len(report.context_evidence) == 2
        proof = next(p for p in report.context_evidence if p.outcome == "affected")
        assert proof.consumer_ids == ["consumer"]
        assert proof.source_id == "new" and proof.source_quote == "fee is 7%"
        assert proof.effective_start == boundary
        assert proof.context_sha256 and proof.audit_input_sha256

    if fault is None:
        prior_attempts = attempts
        replay = impact.analyze_amendment_impact(
            before=before, after=after, bindings=bindings, llm=llm, audit_cache=resolve
        )
        assert replay.affected_windows == report.affected_windows
        assert attempts == prior_attempts
        revised = [
            r.model_copy(update={"heading_path": ["Different legal unit"]})
            if r.id == "new"
            else r
            for r in after
        ]
        fresh = impact.analyze_amendment_impact(
            before=before,
            after=revised,
            bindings=bindings,
            llm=llm,
            audit_cache=resolve,
        )
        assert not fresh.unresolved
        assert len(cache) == 2


def test_context_audit_handles_bounded_provider_output_and_resumes_completed_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import json
    from collections import Counter
    from collections.abc import Callable
    from contextvars import ContextVar
    from threading import Lock
    from unittest.mock import MagicMock

    from onyx.llm.interfaces import LLMConfig
    from onyx.regulatory import amendment_projection_impact as impact
    from onyx.regulatory.structured_llm import StructuredOutputValidationError
    from tests.unit.onyx.regulatory.annexes.test_publication_timeline import snapshot

    old = snapshot("old", None, None).model_copy(update={"text": "Fee is 5%."})
    consumers = [snapshot(f"consumer-{i}", None, None) for i in range(25)]
    boundary = date(2027, 1, 1)
    before = [old, *consumers]
    after = [
        old.model_copy(update={"validity_end_date": boundary}),
        *consumers,
        snapshot("new", boundary, None).model_copy(
            update={"text": "Fee is 7%.", "supersedes_chunk_id": "old"}
        ),
    ]
    contexts = {
        row.id: f"Independent administrative rule {i}."
        for i, row in enumerate(consumers)
    }
    contexts[consumers[-1].id] = "The fee is 5%."
    bindings = [
        MagicMock(
            effective_start=None,
            effective_end=None,
            projection=MagicMock(
                source_json=json.dumps(
                    {"regulatory_chunk_id": key, "chunk_context": value}
                )
            ),
        )
        for key, value in contexts.items()
    ]
    llm = MagicMock()
    llm.config = LLMConfig(
        model_provider="configured",
        model_name="context",
        temperature=0,
        max_input_tokens=10000,
    )
    tenant = ContextVar("audit_test_tenant", default="missing")
    token = tenant.set("owned-tenant")
    calls: Counter[str] = Counter()
    cache: dict[str, impact.ContextImpactResult] = {}
    guard = Lock()
    blocked = True

    def audit(*_args: object, **kwargs: object) -> impact.ContextImpactResult:
        assert tenant.get() == "owned-tenant"
        data = json.loads(str(kwargs["user_prompt"]))["contexts"]
        # A bounded provider response cannot contain the entire file's decisions.
        if len(data) > 8:
            raise StructuredOutputValidationError("json_invalid: EOF in decisions")
        with guard:
            calls.update(case["text"] for case in data.values())
        if blocked and any(case["text"] == "The fee is 5%." for case in data.values()):
            raise StructuredOutputValidationError("json_invalid: EOF in last group")
        return impact.ContextImpactResult(
            decisions=[
                ContextImpactDecision(
                    key=key,
                    affected=case["text"] == "The fee is 5%.",
                    quote="fee is 5%" if case["text"] == "The fee is 5%." else "",
                    reason="Fee changes; administrative rules remain valid.",
                    source_side="after",
                    source_id="new",
                    source_quote="Fee is 7%.",
                )
                for key, case in data.items()
            ]
        )

    def resolve(
        key: str, generate: Callable[[], impact.ContextImpactResult]
    ) -> impact.ContextImpactResult:
        with guard:
            stored = cache.get(key)
        if stored is not None:
            return stored
        value = generate()
        with guard:
            cache[key] = value
        return value

    monkeypatch.setattr(impact, "generate_structured", audit)
    try:
        failed = impact.analyze_amendment_impact(
            before=before, after=after, bindings=bindings, llm=llm, audit_cache=resolve
        )
        assert failed.unresolved and not failed.unchanged_ids
        assert cache  # Successful groups survive a different group's provider failure.
        successful_calls = calls[contexts[consumers[0].id]]
        blocked = False
        result = impact.analyze_amendment_impact(
            before=before, after=after, bindings=bindings, llm=llm, audit_cache=resolve
        )
        assert not result.unresolved
        assert set(result.affected_windows) == {"old", "new", consumers[-1].id}
        assert set(result.unchanged_ids) == {r.id for r in consumers[:-1]}
        assert len(result.context_evidence) == len(contexts)
        assert calls[contexts[consumers[0].id]] == successful_calls
        assert all(calls[value] >= 1 for value in contexts.values())
    finally:
        tenant.reset(token)
