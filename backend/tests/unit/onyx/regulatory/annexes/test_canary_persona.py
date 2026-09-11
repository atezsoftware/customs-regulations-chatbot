from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from onyx.db import regulatory_annex_acceptance as ownership
from onyx.regulatory.amendments.annexes import acceptance_canary as canary


def owned_run() -> ownership.CanaryRun:
    return ownership.CanaryRun(
        release_sha="a" * 40, user_id=uuid4(), document_set_id=24
    )


def test_canary_creates_private_scope_and_uses_explicit_chat_persona(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from datetime import date

    from onyx.context.search.models import BaseFilters, IndexFilters
    from onyx.context.search.pipeline import _build_index_filters
    from onyx.db.models import User
    from onyx.server.features.persona.models import PersonaUpsertRequest
    from onyx.server.query_and_chat.models import ChatSessionCreationRequest

    run = owned_run()
    saved = []
    requests = []
    chat_id = uuid4()
    monkeypatch.setattr(
        canary,
        "save_canary",
        lambda current: saved.append(current.model_dump(mode="json")),
    )
    monkeypatch.setattr(canary, "require_canary_persona", MagicMock(return_value=False))

    def request(_client: object, method: str, path: str, **kwargs: object) -> object:
        requests.append((method, path, kwargs))
        if path == "/tool":
            return [{"display_name": "Internal Search", "id": 7}]
        if path == "/persona":
            payload = PersonaUpsertRequest.model_validate(kwargs["json"])
            assert payload.document_set_ids == [24] and payload.is_public is False
            assert (
                payload.tool_ids == [7]
                and payload.default_model_configuration_id is None
            )
            assert (
                payload.user_file_ids in (None, [])
                and not payload.document_ids
                and not payload.hierarchy_node_ids
            )
            assert saved[-1]["creation_intents"][0]["kind"] == "persona"
            return {"id": 123}
        assert path == "/chat/create-chat-session"
        payload = ChatSessionCreationRequest.model_validate(kwargs["json"])
        assert payload.persona_id == 123 and payload.project_id is None
        return {"chat_session_id": str(chat_id)}

    monkeypatch.setattr(canary, "request_json", request)
    assert (
        canary.create_canary_chat(MagicMock(), run, purpose="dated 2026-09-09")
        == chat_id
    )
    assert run.persona_id == 123
    assert [path for _, path, _ in requests].count("/persona") == 1
    user = User(id=run.user_id, email="fixture@example.com")
    monkeypatch.setattr(
        "onyx.context.search.pipeline.filter_document_set_names_by_user_access",
        lambda **_kwargs: [run.name],
    )
    filters = BaseFilters(document_set=[run.name], as_of_date=date(2026, 9, 9))

    def build(persona_sets: list[str]) -> IndexFilters:
        return _build_index_filters(
            persona_document_sets=persona_sets,
            user=user,
            user_provided_filters=filters,
            project_id_filter=None,
            persona_id_filter=None,
            persona_time_cutoff=None,
            db_session=MagicMock(),
            acl_filters=["owned_acl"],
        )

    actual = build([run.name])
    assert (
        actual.document_set == [run.name]
        and actual.as_of_date == filters.as_of_date
        and actual.access_control_list == ["owned_acl"]
    )
    with pytest.raises(Exception, match="outside this agent's knowledge scope"):
        build(["production_scope"])


def test_persona_lost_post_recovers_unique_owner_without_reposting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import httpx

    run = owned_run()
    saved = []
    monkeypatch.setattr(
        canary,
        "save_canary",
        lambda current: saved.append(current.model_dump(mode="json")),
    )
    monkeypatch.setattr(ownership, "save_canary", MagicMock())
    monkeypatch.setattr(canary, "require_canary_persona", MagicMock(return_value=False))
    request = MagicMock(
        side_effect=[
            [{"display_name": "Internal Search", "id": 7}],
            httpx.ReadTimeout("lost response"),
        ]
    )
    monkeypatch.setattr(canary, "request_json", request)
    with pytest.raises(httpx.ReadTimeout):
        canary.ensure_canary_persona(MagicMock(), run)
    assert saved[-1]["creation_intents"][0]["persona_id"] is None
    monkeypatch.setattr(ownership, "matching_canary_persona_ids", lambda *_args: [123])
    assert ownership.recover_creation_intents(run)
    assert run.persona_id == 123
    canary.ensure_canary_persona(MagicMock(), run)
    assert request.call_count == 2


def test_mapped_persona_ownership_and_soft_deleted_scope() -> None:
    from onyx.db.models import DocumentSet, Persona

    run = owned_run()
    scope = DocumentSet(id=24, name=run.name, user_id=run.user_id, is_public=False)
    persona = Persona(
        id=123,
        name=run.name + " / assistant",
        user_id=run.user_id,
        is_public=False,
        builtin_persona=False,
        deleted=False,
        document_sets=[scope],
        default_model_configuration_id=None,
        search_start_date=None,
    )
    session = MagicMock()
    session.get.return_value = persona
    assert (
        ownership._require_canary_persona(session, run, 123, allow_deleted=False)
        is False
    )
    for field, value in (
        ("user_id", uuid4()),
        ("name", "foreign"),
        ("is_public", True),
        ("default_model_configuration_id", 77),
        ("builtin_persona", True),
    ):
        previous = getattr(persona, field)
        setattr(persona, field, value)
        with pytest.raises(ValueError, match="ownership_mismatch"):
            ownership._require_canary_persona(session, run, 123, allow_deleted=True)
        setattr(persona, field, previous)
    for field, value in (
        ("user_id", uuid4()),
        ("name", "foreign"),
        ("is_public", True),
        ("id", 25),
    ):
        previous = getattr(scope, field)
        setattr(scope, field, value)
        with pytest.raises(ValueError, match="scope_mismatch"):
            ownership._require_canary_persona(session, run, 123, allow_deleted=True)
        setattr(scope, field, previous)
    persona.deleted = True
    assert (
        ownership._require_canary_persona(session, run, 123, allow_deleted=True) is True
    )
    with pytest.raises(ValueError, match="ownership_mismatch"):
        ownership._require_canary_persona(session, run, 123, allow_deleted=False)


@pytest.mark.parametrize("matches", [[], [123, 124]])
def test_unresolved_persona_never_reposts_or_deletes(
    matches: list[int], monkeypatch: pytest.MonkeyPatch
) -> None:
    run = owned_run()
    run.creation_intents.append(
        ownership.CreationIntent(kind="persona", marker=run.name + " / assistant")
    )
    monkeypatch.setattr(
        ownership, "matching_canary_persona_ids", lambda *_args: matches
    )
    monkeypatch.setattr(ownership, "save_canary", MagicMock())
    request = MagicMock()
    monkeypatch.setattr(canary, "request_json", request)
    with pytest.raises(ValueError, match="creation_unresolved"):
        canary.ensure_canary_persona(MagicMock(), run)
    with pytest.raises(ValueError, match="cleanup_unresolved"):
        canary.cleanup_canary_persona(MagicMock(), run)
    request.assert_not_called()


def test_cleanup_checks_saved_identity_and_recovers_lost_delete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import httpx

    run = owned_run()
    run.persona_id = 123
    guard = MagicMock(side_effect=ValueError("foreign saved identity"))
    request = MagicMock()
    monkeypatch.setattr(canary, "require_canary_persona", guard)
    monkeypatch.setattr(canary, "request_json", request)
    monkeypatch.setattr(canary, "save_canary", MagicMock())
    with pytest.raises(ValueError, match="foreign saved identity"):
        canary.cleanup_canary_persona(MagicMock(), run)
    request.assert_not_called()
    guard.side_effect = [False, True]
    request.side_effect = httpx.ReadTimeout("lost delete response")
    client = MagicMock()
    with pytest.raises(httpx.ReadTimeout):
        canary.cleanup_canary_persona(client, run)
    request.assert_called_once_with(client, "DELETE", "/persona/123")
    canary.cleanup_canary_persona(client, run)
    assert request.call_count == 1
    assert run.evidence["persona_cleanup_complete"] is True


@pytest.mark.parametrize(
    "payload",
    [
        {"kind": "persona", "artifact_id": str(uuid4())},
        {"kind": "chat", "persona_id": 123},
        {"kind": "persona", "persona_id": 0},
        {"kind": "persona", "persona_id": True},
        {"kind": "persona", "persona_id": "123"},
    ],
)
def test_intent_identity_kinds_are_strict(payload: dict[str, object]) -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ownership.CreationIntent.model_validate({"marker": "owned", **payload})
