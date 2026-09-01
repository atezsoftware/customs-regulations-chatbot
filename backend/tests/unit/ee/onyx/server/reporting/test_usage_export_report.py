from datetime import datetime, timezone
from types import SimpleNamespace
from typing import cast

import pytest

from ee.onyx.server.reporting import usage_export_api
from ee.onyx.server.reporting.usage_export_api import GenerateUsageReportParams
from ee.onyx.server.reporting.usage_export_generation import (
    render_per_user_usage_csv,
    render_usage_summary_csv,
)
from ee.onyx.server.reporting.usage_export_models import UsageSummary
from onyx.configs.app_configs import JOB_TIMEOUT
from onyx.configs.constants import OnyxCeleryPriority, OnyxCeleryQueues
from onyx.db.models import User
from onyx.db.user_usage import PerUserUsageSummary


def test_heavy_worker_registers_usage_report_task() -> None:
    from ee.onyx.background.celery.apps.heavy import celery_app

    celery_app.loader.import_default_modules()

    assert "generate_usage_report_task" in celery_app.tasks


def test_generate_report_dispatches_to_consumed_csv_queue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent: dict[str, object] = {}

    def _send_task(task_name: str, **kwargs: object) -> None:
        sent["task_name"] = task_name
        sent.update(kwargs)

    monkeypatch.setattr(usage_export_api.client_app, "send_task", _send_task)
    monkeypatch.setattr(usage_export_api, "get_current_tenant_id", lambda: "tenant-1")

    usage_export_api.generate_report(
        GenerateUsageReportParams(),
        cast(
            User,
            SimpleNamespace(id="00000000-0000-0000-0000-000000000001"),
        ),
    )

    assert sent["queue"] == OnyxCeleryQueues.CSV_GENERATION
    assert sent["priority"] == OnyxCeleryPriority.MEDIUM
    assert sent["expires"] == JOB_TIMEOUT


def test_usage_summary_csv_contains_counts_tokens_cost_and_rates() -> None:
    summary = UsageSummary(
        total_user_queries=4,
        total_user_sessions=2,
        total_tokens=1_200,
        total_cost_cents=250.0,
        average_tokens_per_query=30.0,
        average_tokens_per_session=60.0,
        average_cost_cents_per_query=62.5,
        average_cost_cents_per_session=125.0,
        average_queries_per_session=2.0,
    )

    rows = render_usage_summary_csv(
        summary,
        period=(
            datetime(2026, 8, 1, tzinfo=timezone.utc),
            datetime(2026, 8, 31, tzinfo=timezone.utc),
        ),
    ).splitlines()

    assert rows == [
        "period_from,period_to,total_user_queries,total_user_sessions,total_tokens,total_cost_cents,average_tokens_per_query,average_tokens_per_session,average_cost_cents_per_query,average_cost_cents_per_session,average_queries_per_session",
        "2026-08-01T00:00:00+00:00,2026-08-31T00:00:00+00:00,4,2,1200,250.0,30.0,60.0,62.5,125.0,2.0",
    ]


def test_per_user_usage_csv_contains_token_cost_and_activity_rates() -> None:
    rows = render_per_user_usage_csv(
        [
            PerUserUsageSummary(
                email="alice@example.com",
                input_tokens=700,
                output_tokens=500,
                cache_read_tokens=80,
                cost_cents=250.0,
                total_tokens=1_200,
                total_user_queries=4,
                total_user_sessions=2,
                average_tokens_per_query=300.0,
                average_tokens_per_session=600.0,
                average_cost_cents_per_query=62.5,
                average_cost_cents_per_session=125.0,
                average_queries_per_session=2.0,
            )
        ]
    ).splitlines()

    assert rows == [
        "email,input_tokens,output_tokens,cache_read_tokens,cost_cents,total_tokens,total_user_queries,total_user_sessions,average_tokens_per_query,average_tokens_per_session,average_cost_cents_per_query,average_cost_cents_per_session,average_queries_per_session",
        "alice@example.com,700,500,80,250.0,1200,4,2,300.0,600.0,62.5,125.0,2.0",
    ]
