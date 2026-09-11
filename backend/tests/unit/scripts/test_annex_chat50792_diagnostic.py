"""Fixed retained-chat scope evidence without services, providers or corpus output."""

import json
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock, patch
from uuid import UUID

import pytest
from scripts import regulatory_annex_dev_cutover as runner
from sqlalchemy.orm import Session

from onyx.db import regulatory_annex_acceptance_diagnostic as diagnostic
from onyx.db.models import AmendmentBatch, ChatSession, DocumentSet, KVStore, Persona
from onyx.db.regulatory_annex_acceptance import CanaryRun, CreationIntent


def fixture() -> tuple[Mock, dict[type, Any], list[Any]]:
    run = CanaryRun(
        release_sha=runner.CHAT50792_RUNTIME,
        user_id=UUID("7e0d56bc-6f9c-4cec-b29b-5a8c9ec2b844"),
        run_id=UUID("97819935-8cd6-4bcf-aeee-04797d17e864"),
        file_id=UUID("df0750d8-ddc3-4303-91f1-1aa9e3d43620"),
        document_set_id=24,
        batch_id=49,
        package_id=UUID("06c999a5-1e98-4950-b088-6add68ed778d"),
        phase="cleaned",
        chat_ids=[UUID("01668c44-1692-469c-be15-93d38fa8a858")],
    )
    marker = run.name + " / dated 2026-09-09"
    run.creation_intents = [
        CreationIntent(kind="chat", marker=marker, artifact_id=run.chat_ids[0])
    ]
    scope = SimpleNamespace(id=24, name=run.name, user_id=run.user_id, is_public=False)
    rows: dict[type, Any] = {
        KVStore: SimpleNamespace(value=run.model_dump(mode="json")),
        DocumentSet: scope,
        AmendmentBatch: SimpleNamespace(
            document_set_id=24,
            created_by=run.user_id,
            user_file_ids=[str(run.file_id)],
            source_package_id=run.package_id,
        ),
        ChatSession: SimpleNamespace(
            id=run.chat_ids[0],
            user_id=run.user_id,
            description=marker,
            persona_id=0,
            deleted=True,
        ),
        Persona: SimpleNamespace(
            document_sets=[SimpleNamespace(id=99, name="PRIVATE CORPUS NAME")]
        ),
    }
    tool = SimpleNamespace(
        id=432,
        tool_id=7,
        chat_session_id=run.chat_ids[0],
        tool_call_children=[],
        search_docs=[],
        tool_call_arguments={"query": "PRIVATE PROMPT"},
        tool_call_response=json.dumps(
            {
                "error": "Selected Document Sets are outside this agent's knowledge scope. PRIVATE-CREDENTIAL"
            }
        ),
    )
    messages = [
        SimpleNamespace(
            id=12,
            message_type=SimpleNamespace(value="assistant"),
            message="PRIVATE ANSWER",
            error=None,
            search_docs=[SimpleNamespace(document_id=str(run.file_id))],
            citations={"1": 1},
            publication_read={"finalized": True},
            tool_calls=[tool],
        )
    ]
    session = Mock()
    session.get.side_effect = lambda model, _key: rows[model]
    return session, rows, messages


def test_retained_real_shape_scope_error_and_counts_are_redacted() -> None:
    session, _, messages = fixture()
    with patch(
        "onyx.db.chat.get_chat_messages_by_session", return_value=messages
    ) as read:
        result = diagnostic.load_chat50792_diagnostic(cast(Session, session))
    assert result["status"] == "read" and result["assistant_id"] == 0
    assert result["default_scope_count"] == result["assistant_scope_count"] == 1
    assert result["assistant_owned_scope_included"] is False
    assert result["default_owned_scope_included"] is False
    assert result["tools"] == [
        {
            "tool_call_id": 432,
            "tool_id": 7,
            "result_count": 0,
            "owned_result_count": 0,
            "error_category": "agent_document_set_scope",
        }
    ]
    message = cast(list[dict[str, Any]], result["messages"])[0]
    assert (
        message["document_count"]
        == message["owned_document_count"]
        == message["citation_count"]
        == 1
    )
    assert message["publication_finalized"] is True
    assert "PRIVATE" not in json.dumps(result)
    assert (
        session.get.call_args_list[0].args[1]
        == "regulatory_annex_acceptance:" + runner.CHAT50792_RUNTIME
    )
    assert read.call_args.args[:2] == (
        UUID("01668c44-1692-469c-be15-93d38fa8a858"),
        UUID("7e0d56bc-6f9c-4cec-b29b-5a8c9ec2b844"),
    )
    assert not session.commit.called
    runner.validate_chat50792_report(
        {"stage": "chat50792", "database_read_only": True, **result}
    )


@pytest.mark.parametrize(
    "target,field,value",
    [
        ("run", "release_sha", "a" * 40),
        ("run", "user_id", str(UUID(int=1))),
        ("run", "phase", "approved"),
        ("run", "batch_id", 48),
        ("run", "file_id", str(UUID(int=2))),
        ("run", "chat_ids", []),
        ("run", "creation_intents", []),
        ("scope", "is_public", True),
        ("scope", "name", "FOREIGN"),
        ("chat", "description", "FOREIGN"),
        ("chat", "user_id", UUID(int=1)),
        ("batch", "document_set_id", 22),
    ],
)
def test_foreign_retained_scope_never_reads_message_bodies(
    target: str, field: str, value: Any
) -> None:
    session, rows, _ = fixture()
    if target == "run":
        rows[KVStore].value[field] = value
    else:
        model = {"scope": DocumentSet, "chat": ChatSession, "batch": AmendmentBatch}[
            target
        ]
        setattr(rows[model], field, value)
    with patch("onyx.db.chat.get_chat_messages_by_session") as read:
        result = diagnostic.load_chat50792_diagnostic(cast(Session, session))
    assert result["status"] == "scope_refused"
    read.assert_not_called()


def test_actual_scope_membership_and_exact_document_identity() -> None:
    session, rows, messages = fixture()
    rows[Persona].document_sets = [rows[DocumentSet]]
    messages[0].search_docs.append(
        SimpleNamespace(document_id="prefix" + messages[0].search_docs[0].document_id)
    )
    messages[0].tool_calls[0].tool_call_response = "No fictional result"
    with patch("onyx.db.chat.get_chat_messages_by_session", return_value=messages):
        result = diagnostic.load_chat50792_diagnostic(cast(Session, session))
    assert result["assistant_owned_scope_included"] is True
    assert result["default_owned_scope_included"] is True
    assert (
        cast(list[dict[str, Any]], result["messages"])[0]["owned_document_count"] == 1
    )
    assert (
        cast(list[dict[str, Any]], result["tools"])[0]["error_category"]
        == "no_known_error"
    )


@pytest.mark.parametrize(
    "body,category",
    [
        (None, "empty"),
        ("User does not have access to document sets: PRIVATE", "document_set_access"),
        (
            "A source changed during this response. Please try again.",
            "publication_changed",
        ),
        (
            "No search settings configured — cannot run internal search",
            "search_configuration_missing",
        ),
        ("PRIVATE SECRET", "no_known_error"),
    ],
)
def test_error_matching_never_exports_unknown_text(
    body: str | None, category: str
) -> None:
    assert diagnostic.chat50792_error_category(body) == category


def test_fixed_runner_early_readonly_branch(capsys: pytest.CaptureFixture[str]) -> None:
    driver = Mock(spec=runner.Driver)
    driver.sha = runner.CHAT50792_RUNTIME
    driver.get.return_value = {"data": {"sha": driver.sha, "phase": "released"}}
    driver.pods.return_value = [
        {
            "metadata": {"name": "background"},
            "spec": {
                "containers": [
                    {"name": "worker", "image": runner.REPOSITORY + ":" + driver.sha}
                ]
            },
        }
    ]
    result = {
        "stage": "chat50792",
        "status": "scope_refused",
        "scope_failure": "run_missing",
        "database_read_only": True,
    }
    driver.command.return_value = json.dumps(result)
    with (
        patch.object(runner, "require_release_runs"),
        patch.object(runner, "verify_frontend"),
        patch.object(runner, "diagnose_batch44_logs") as obsolete,
        patch.object(runner, "diagnose_source861") as source,
    ):
        runner.diagnose_release(driver, "b" * 40)
    obsolete.assert_not_called()
    source.assert_not_called()
    command = driver.command.call_args.args[0]
    assert command[-2] == "chat50792-diagnostic" and "read_only=on" in command[-3]
    program = command[-1]
    compile(program, "fixed-chat-diagnostic", "exec")
    assert program.index("            configured_indices()") < program.index(
        "            with SqlEngine.scoped_engine"
    )
    assert "session.commit(" not in program and "get_default_llm" not in program
    assert driver.command.call_args.kwargs["timeout"] == 80
    assert json.loads(capsys.readouterr().out.splitlines()[-1]) == result


@pytest.mark.parametrize(
    "field,value",
    [
        ("body", "PRIVATE"),
        ("status", "PRIVATE"),
        ("error_category", "PRIVATE"),
        ("tools", [{}] * 65),
        ("database_read_only", False),
    ],
)
def test_report_refuses_unbounded_or_unrecognized_values(
    field: str, value: Any
) -> None:
    result = {
        "stage": "chat50792",
        "status": "read",
        "database_read_only": True,
        field: value,
    }
    with pytest.raises(runner.CutoverRefusal):
        runner.validate_chat50792_report(result)
