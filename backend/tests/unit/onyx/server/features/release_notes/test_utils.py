from unittest.mock import Mock

import pytest

from onyx.server.features.release_notes import utils


def test_vendor_upstream_disable_skips_release_note_fetch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(utils, "DISABLE_ONYX_UPSTREAM_CONNECTIONS", True)
    cache_check = Mock(side_effect=AssertionError("cache must not be consulted"))
    monkeypatch.setattr(utils, "is_cache_stale", cache_check)

    utils.ensure_release_notes_fresh_and_notify(Mock())

    cache_check.assert_not_called()
