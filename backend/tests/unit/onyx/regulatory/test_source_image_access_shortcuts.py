"""The source-image namespace cannot fall through into unrelated asset ACLs."""

from unittest.mock import Mock
from uuid import uuid4

import pytest

from onyx.access import access
from onyx.db import regulatory_public_reads as reads


@pytest.mark.parametrize(
    "linked,permitted", [(False, False), (True, False), (True, True)]
)
def test_source_image_uses_only_actual_parent_acl(
    monkeypatch: pytest.MonkeyPatch, linked: bool, permitted: bool
) -> None:
    parent = uuid4()
    owners = (parent,) if linked else ()
    monkeypatch.setattr(
        reads, "regulatory_source_image_owners", Mock(return_value=owners)
    )
    user_acl = {"user:reader@example.test"}
    document_access = Mock(
        to_acl=Mock(return_value=user_acl if permitted else {"user:other@example.test"})
    )
    load_access = Mock(return_value={str(parent): document_access})
    monkeypatch.setattr(access, "get_access_for_user_files", load_access)
    monkeypatch.setattr(access, "get_acl_for_user", Mock(return_value=user_acl))
    for name in (
        "_user_can_access_persona_attached_file",
        "_user_can_access_connector_file",
    ):
        monkeypatch.setattr(
            access, name, Mock(side_effect=AssertionError("unrelated ACL path"))
        )
    monkeypatch.setattr(
        reads,
        "public_file_source_owners",
        Mock(side_effect=AssertionError("legacy source scan")),
    )
    session = Mock()
    assert access.user_can_access_chat_file("source-image", Mock(), session) is (
        linked and permitted
    )
    session.query.assert_not_called()
    session.execute.assert_not_called()
    if linked:
        load_access.assert_called_once_with([str(parent)], session)
    else:
        load_access.assert_not_called()


def test_public_owner_mapping_short_circuits_empty_scoped_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(reads, "regulatory_source_image_owners", Mock(return_value=()))
    session = Mock()
    assert reads.public_file_source_owners(session, "unlinked") == ()
    session.scalars.assert_not_called()


def test_legacy_asset_keeps_existing_owned_file_shortcut(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        reads, "regulatory_source_image_owners", Mock(return_value=None)
    )
    session = Mock()
    session.query.return_value.scalar.return_value = True
    assert access.user_can_access_chat_file("legacy-owned", Mock(), session)
    session.query.assert_called_once()
