"""Standalone inspection owns a bounded read-only engine and releases it on failure."""

from collections.abc import Iterator
from contextlib import contextmanager
from unittest.mock import MagicMock

import pytest

from onyx.db import legal_review_dev_preflight as preflight
from onyx.db.engine.sql_engine import SqlEngine


def test_non_dev_refusal_happens_before_engine_initialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("POSTGRES_DB", "production")
    monkeypatch.setenv("REGULATORY_ANNEX_ENVIRONMENT", "dev")
    scoped = MagicMock()
    monkeypatch.setattr(SqlEngine, "scoped_engine", scoped)
    with pytest.raises(
        preflight.PreflightRefusal, match="explicit_dev_environment_required"
    ):
        preflight.inspect_dev()
    scoped.assert_not_called()


def test_engine_is_active_for_inspection_and_disposed_after_secret_bearing_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options = "-c default_transaction_read_only=on -c statement_timeout=15000"
    monkeypatch.setenv("PGOPTIONS", options)
    active = [False]

    @contextmanager
    def engine(**kwargs: object) -> Iterator[None]:
        assert kwargs["pool_size"] == 2 and kwargs["max_overflow"] == 0
        assert kwargs["pool_timeout"] == 5
        assert kwargs["connect_args"] == {"connect_timeout": 5, "options": options}
        active[0] = True
        try:
            yield
        finally:
            active[0] = False

    monkeypatch.setattr(SqlEngine, "scoped_engine", engine)
    with pytest.raises(preflight.PreflightRefusal) as caught:
        with preflight._inspection_engine():
            assert active[0]
            raise RuntimeError("credential material that must not appear")
    assert not active[0]
    assert str(caught.value) == "database_inspection_failed_RuntimeError"


def test_existing_fixed_refusal_survives_engine_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PGOPTIONS", "-c default_transaction_read_only=on")

    @contextmanager
    def engine(**_kwargs: object) -> Iterator[None]:
        yield

    monkeypatch.setattr(SqlEngine, "scoped_engine", engine)
    with pytest.raises(
        preflight.PreflightRefusal, match="connected_database_is_not_dev"
    ):
        with preflight._inspection_engine():
            raise preflight.PreflightRefusal("connected_database_is_not_dev")
