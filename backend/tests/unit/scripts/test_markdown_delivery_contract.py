from unittest.mock import Mock

import pytest
from scripts import regulatory_annex_dev_cutover as cutover


def test_delivery_identity_omits_credentials_and_counts_exact_queue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from onyx.background.celery import celery_redis
    from onyx.background.celery.versioned_apps.client import app

    connection = Mock(
        hostname="owned-redis",
        port=6379,
        virtual_host="15",
        transport_cls="redis",
        userid="private-user",
        password="private-password",
    )
    connection.__enter__ = Mock(return_value=connection)
    connection.__exit__ = Mock(return_value=None)
    monkeypatch.setattr(app, "connection_for_write", Mock(return_value=connection))
    broker = Mock()
    monkeypatch.setattr(
        celery_redis, "celery_get_broker_client", Mock(return_value=broker)
    )
    count = Mock(return_value=4)
    monkeypatch.setattr(celery_redis, "celery_get_queue_length", count)
    result = cutover.read_markdown_delivery()
    digest = result["broker_target_sha256"]
    assert isinstance(digest, str) and len(digest) == 64
    assert result["processing_queue_depth"] == 4
    count.assert_called_once_with("user_file_processing", broker)
    broker.close.assert_called_once()
    assert "private" not in str(result)
    connection.password = "rotated-private-password"
    assert (
        cutover.read_markdown_delivery()["broker_target_sha256"]
        == result["broker_target_sha256"]
    )
    connection.virtual_host = "16"
    assert (
        cutover.read_markdown_delivery()["broker_target_sha256"]
        != result["broker_target_sha256"]
    )


def test_delivery_report_accepts_bounded_identity_only() -> None:
    report = {
        "stage": "markdown_worker",
        "role": "api",
        "status": "read",
        "database_read_only": True,
        "broker_target_sha256": "a" * 64,
        "processing_queue_depth": 4,
        "current_vector_disabled": False,
        "current_defer_indexing": True,
        "current_batch_indexing": False,
    }
    cutover.validate_markdown_worker_report(report)
    with pytest.raises(cutover.CutoverRefusal):
        cutover.validate_markdown_worker_report(
            {**report, "broker_target_sha256": "private-password"}
        )
