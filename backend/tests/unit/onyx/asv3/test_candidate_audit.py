from __future__ import annotations

import pytest

from onyx.asv3.models import RunContext


def _audit() -> object:
    from onyx.asv3.candidate_audit import CandidateAudit

    return CandidateAudit(
        RunContext(run_id="run-1", scope={"corpus": "authorized"}),
        request="Muafiyet ve iade prosedürü nedir?",
        max_records=4,
        max_serialized_chars=4000,
    )


def _record(**updates: object) -> object:
    from onyx.asv3.candidate_audit import CandidateAuditRecord

    fields: dict[str, object] = {
        "search_run_id": "search-1",
        "candidate_id": "source-1:chunk-1",
        "source_id": "source-1",
        "chunk_id": "chunk-1",
        "lane": "regulatory",
        "mode": "hybrid",
        "status": "excluded",
        "reason": "rerank_below_selection",
        "raw_score": 0.42,
        "normalized_score": 0.37,
        "rerank_position": 12,
        "outcome_ids": ["outcome-refund"],
        "scope_version": "publication-1",
    }
    fields.update(updates)
    return CandidateAuditRecord.model_validate(fields)


def test_candidate_audit_tracks_rerank_lifecycle_without_passage_text() -> None:
    audit = _audit()
    audit.record(_record())
    audit.record(
        _record(
            status="selected",
            reason="selected_for_llm_delivery",
            rerank_position=2,
        )
    )
    audit.record(
        _record(
            status="delivered",
            reason="hydrated_and_delivered",
            rerank_position=2,
        )
    )
    audit.record(
        _record(
            candidate_id="source-2:chunk-4",
            source_id="source-2",
            chunk_id="chunk-4",
            status="hydration_failed",
            reason="authorized_hydration_empty",
        )
    )

    exported = audit.export()

    assert exported["version"] == 1
    assert [row["status"] for row in exported["records"]] == [
        "delivered",
        "hydration_failed",
    ]
    assert "text" not in str(exported)
    assert exported["records"][0]["candidate_id"] == "source-1:chunk-1"


def test_candidate_audit_rejects_capacity_and_scope_mismatch() -> None:
    audit = _audit()
    for index in range(4):
        audit.record(
            _record(
                candidate_id=f"source-{index}:chunk-{index}",
                source_id=f"source-{index}",
                chunk_id=f"chunk-{index}",
            )
        )
    with pytest.raises(ValueError, match="capacity"):
        audit.record(
            _record(
                candidate_id="source-5:chunk-5",
                source_id="source-5",
                chunk_id="chunk-5",
            )
        )

    exported = audit.export()
    from onyx.asv3.candidate_audit import CandidateAudit

    foreign = CandidateAudit(
        RunContext(run_id="run-1", scope={"corpus": "different"}),
        request="Muafiyet ve iade prosedürü nedir?",
    )
    with pytest.raises(ValueError, match="scope"):
        foreign.restore(exported)
