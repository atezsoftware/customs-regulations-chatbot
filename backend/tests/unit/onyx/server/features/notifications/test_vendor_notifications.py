from unittest.mock import Mock

import pytest

from onyx.server.features.notifications import api


def test_vendor_upstream_disable_skips_vendor_notifications(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hooks = [
        "ensure_build_mode_intro_notification",
        "ensure_permissions_migration_notification",
        "ensure_release_notes_fresh_and_notify",
    ]
    mocks = {hook: Mock() for hook in hooks}
    monkeypatch.setattr(api, "DISABLE_ONYX_UPSTREAM_CONNECTIONS", True)
    for hook, mock in mocks.items():
        monkeypatch.setattr(api, hook, mock)

    api._check_for_notifications_to_create(Mock(), Mock())

    for mock in mocks.values():
        mock.assert_not_called()
