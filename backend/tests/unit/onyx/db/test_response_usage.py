from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock
from uuid import uuid4

from onyx.db.response_usage import get_response_usage
from onyx.llm.usage_cost import GenerationCost, TokenCostLine


def test_llm_total_excludes_non_token_services_and_uses_wall_duration() -> None:
    finished = datetime.now(timezone.utc)
    run_id = uuid4()
    cost = GenerationCost(
        model="selected-model",
        provider="provider",
        source="model_registry",
        priced_at=finished,
        lines=[
            TokenCostLine(
                category="input", tokens=1000, usd_per_million=2, cost_usd=0.002
            )
        ],
        complete=True,
        known_cost_usd=0.002,
    )
    session = MagicMock()
    session.execute.side_effect = [
        [
            (
                12,
                50,
                run_id,
                "COMPLETE",
                "COMPLETE",
                finished - timedelta(seconds=10),
                finished,
            )
        ],
        [
            (
                run_id,
                {"usage_cost": cost.model_dump(mode="json")},
                "COMPLETE",
                "asv3_researcher",
            ),
            (run_id, {"model": "reranker"}, "COMPLETE", "rerank"),
            (run_id, {"model": "embedding"}, "COMPLETE", "embed_query"),
        ],
    ]
    summary = get_response_usage(session, [12], admin=True)[12]
    assert summary.status == "complete" and summary.total_cost_usd == 0.002
    assert summary.calls == 1 and summary.excluded_service_calls == 2
    assert summary.unpriced_calls == 0 and summary.duration_seconds == 10
    assert len(summary.models) == 1


def test_unpriced_llm_is_retained_as_an_unknown_cost() -> None:
    finished = datetime.now(timezone.utc)
    run_id = uuid4()
    session = MagicMock()
    session.execute.side_effect = [
        [(12, 50, run_id, "COMPLETE", "COMPLETE", finished, finished)],
        [
            (
                run_id,
                {"model": "unknown-llm", "usage": {"input_tokens": 10}},
                "COMPLETE",
                "untagged_invoke",
            )
        ],
    ]
    summary = get_response_usage(session, [12], admin=True)[12]
    assert summary.status == "unavailable" and summary.total_cost_usd is None
    assert summary.calls == 1 and summary.unpriced_calls == 1
    assert summary.excluded_service_calls == 0
