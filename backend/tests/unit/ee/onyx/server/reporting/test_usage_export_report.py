from datetime import datetime, timezone
from types import SimpleNamespace
from typing import cast

import pytest

from ee.onyx.server.reporting import usage_export_api
from ee.onyx.server.reporting.usage_export_api import GenerateUsageReportParams
from ee.onyx.server.reporting.usage_export_generation import (
    render_usage_summary_csv,
)
from ee.onyx.server.reporting.usage_export_models import UsageSummary
from onyx.configs.app_configs import JOB_TIMEOUT
from onyx.configs.constants import OnyxCeleryPriority, OnyxCeleryQueues
from onyx.db.models import User


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


def test_usage_summary_csv_contains_counts_tokens_and_rates() -> None:
    summary = UsageSummary(
        total_user_queries=4,
        total_user_sessions=2,
        total_query_tokens=120,
        average_tokens_per_query=30.0,
        average_tokens_per_session=60.0,
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
        "period_from,period_to,total_user_queries,total_user_sessions,total_query_tokens,average_tokens_per_query,average_tokens_per_session,average_queries_per_session",
        "2026-08-01T00:00:00+00:00,2026-08-31T00:00:00+00:00,4,2,120,30.0,60.0,2.0",
    ]
