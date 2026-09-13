import json
from unittest.mock import Mock

import pytest
from scripts import regulatory_annex_dev_cutover as cutover


def test_redis_inventory_exposes_only_database_allocation_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from redis import Redis

    client = Mock()
    client.info.side_effect = [
        {"redis_mode": "standalone", "private": "secret"},
        {"db0": {"keys": 2}, "db14": {"keys": 1}},
    ]
    client.client_list.return_value = [
        {"db": "15", "addr": "private", "name": "secret"},
        {"db": "2"},
    ]
    monkeypatch.setattr(Redis, "from_url", Mock(return_value=client))
    report = cutover.read_redis_allocation()
    assert report["redis_keyspace_databases"] == "0,14"
    assert report["redis_client_databases"] == "2,15"
    assert report["redis_supported_databases"] == ",".join(map(str, range(16)))
    assert "secret" not in json.dumps(report)
    assert "private" not in json.dumps(report)
    assert report["redis_cluster_mode"] is False
    cutover.validate_markdown_worker_report(
        {**report, "stage": "markdown_worker", "database_read_only": True}
    )


def test_inventory_database_fields_reject_non_numeric_payloads() -> None:
    with pytest.raises(cutover.CutoverRefusal):
        cutover.validate_markdown_worker_report(
            {
                "stage": "markdown_worker",
                "database_read_only": True,
                "redis_client_databases": "secret",
            }
        )
