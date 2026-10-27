from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from sqlalchemy.orm import Session

from onyx.db import document_intake as intake
from onyx.db import regulatory_labeling as labeling
from onyx.db.models import User


@pytest.mark.parametrize("replay", [False, True])
def test_automatic_admission_reuses_original_connection_on_idempotent_replay(
    monkeypatch: pytest.MonkeyPatch, replay: bool
) -> None:
    owner = cast(User, SimpleNamespace(id=uuid4()))
    package = SimpleNamespace(id=uuid4(), created_by=owner.id)
    existing = SimpleNamespace(
        source_package_id=package.id,
        created_by=owner.id,
        labeling_model_configuration_id=7,
    )
    session = MagicMock(spec=Session)
    session.scalar.return_value = existing if replay else None
    monkeypatch.setattr(
        intake, "resolve_intake_target", lambda *_args: SimpleNamespace(id=8)
    )
    monkeypatch.setattr(
        intake, "require_ready_source_package", lambda *_args, **_kwargs: package
    )
    select = MagicMock(return_value=SimpleNamespace(model_configuration_id=99))
    monkeypatch.setattr(intake, "select_intake_labeling_provider_binding", select)
    batch, created = intake.create_intake(
        session,
        source_package_id=package.id,
        idempotency_key=uuid4(),
        user=owner,
        environment="test",
    )
    assert created is not replay
    assert batch.labeling_model_configuration_id == (7 if replay else 99)
    if replay:
        assert batch is existing
        select.assert_not_called()
        session.add.assert_not_called()
    else:
        select.assert_called_once_with(session, user=owner)


@pytest.mark.parametrize("existing_run", [False, True])
def test_start_freezes_intake_model_and_only_new_files_or_reuses_frozen_run(
    monkeypatch: pytest.MonkeyPatch, existing_run: bool
) -> None:
    session = MagicMock(spec=Session)
    old_run = SimpleNamespace(id=uuid4(), status="running", model="gemini-3.8-flash")
    batch = SimpleNamespace(
        id=uuid4(),
        created_by=uuid4(),
        document_set_id=8,
        labeling_model_configuration_id=7,
        labeling_run_id=old_run.id if existing_run else None,
        status="preparing",
        lease_token=None,
    )
    token = uuid4()
    batch.lease_token = token
    files = [
        SimpleNamespace(id=uuid4(), status="chunked", user_file_id=uuid4())
        for _ in range(2)
    ]
    old_run.file_ids = [str(member.user_file_id) for member in files]
    monkeypatch.setattr(intake, "load_intake_job", lambda *_args: (batch, None, None))
    monkeypatch.setattr(intake, "intake_files", lambda *_args: files)
    session.get.return_value = old_run if existing_run else None
    binding = SimpleNamespace(
        model="gemini-3.5-flash",
        model_dump=lambda **_kwargs: {
            "model": "gemini-3.5-flash",
            "transport": "vertex_direct_v1",
        },
    )
    resolve = MagicMock(return_value=binding)
    monkeypatch.setattr(intake, "resolve_intake_labeling_provider_binding", resolve)
    monkeypatch.setattr(
        labeling,
        "get_label_settings",
        lambda *_args: SimpleNamespace(taxonomy=object()),
    )
    new_run = SimpleNamespace(id=uuid4())
    create = MagicMock(return_value=(new_run, True))
    monkeypatch.setattr(labeling, "create_labeling_run", create)
    monkeypatch.setattr(labeling, "get_active_run_id", lambda *_args: None)
    result = intake.start_intake_labeling(session, batch.id, token)
    assert result == (old_run.id if existing_run else new_run.id)
    if existing_run:
        resolve.assert_not_called()
        create.assert_not_called()
        assert old_run.model == "gemini-3.8-flash"
    else:
        assert create.call_args.kwargs["model"] == "gemini-3.5-flash"
        assert create.call_args.kwargs["provider_binding"] == binding.model_dump()
        assert create.call_args.kwargs["target_file_ids"] == [
            item.user_file_id for item in files
        ]
