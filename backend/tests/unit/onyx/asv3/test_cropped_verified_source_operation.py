"""Only one synchronous Cropped provision shares its fresh authorization proof."""

from contextlib import nullcontext
from contextvars import copy_context
from dataclasses import replace
from datetime import date
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from sqlalchemy.orm import Session

from onyx.asv3 import corpus_tools
from onyx.asv3.models import OutcomeStatus, RunContext
from onyx.context.search.models import IndexFilters
from onyx.db import asv3_corpus as corpus
from onyx.db.models import User
from onyx.document_index.publication_models import PublicationIndexSnapshot


def setup_operation(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    source_id = uuid4()
    source = corpus.CorpusSource(source_id, "own law", "physical-file")
    state: dict[str, Any] = {"tenant": "public", "source": source, "sessions": []}

    def session() -> Any:
        value = MagicMock(spec=Session)
        state["sessions"].append(value)
        return nullcontext(value)

    find = MagicMock(return_value=([source], False))
    publication = MagicMock()
    monkeypatch.setattr(corpus, "get_session_with_current_tenant", session)
    monkeypatch.setattr(corpus_tools, "get_session_with_current_tenant", session)
    monkeypatch.setattr(corpus, "find_sources", find)
    monkeypatch.setattr(corpus, "observe_publication_read", lambda: "fixed")
    monkeypatch.setattr(corpus, "require_publication_files", publication)
    monkeypatch.setattr(
        "shared_configs.contextvars.get_current_tenant_id", lambda: state["tenant"]
    )
    index = PublicationIndexSnapshot(
        index_name="corpus",
        index_uuid="physical-index",
        search_settings_id=1,
        model_provider="provider",
        model_name="embedding",
        vector_dimension=1,
        embedding_config_sha256="a" * 64,
        multitenant=False,
    )
    monkeypatch.setattr(corpus, "resolve_source_query_index", lambda *_: index)
    check = MagicMock()
    state.update(
        find=find,
        publication=publication,
        index=index,
        check=check,
        kwargs=dict(
            user=cast(User, SimpleNamespace(id=uuid4())),
            filters=IndexFilters(
                access_control_list=["owner"], as_of_date=date(2026, 10, 13)
            ),
            source_id=source_id,
            check_active=check,
        ),
    )
    return state


def test_same_operation_reuses_only_its_source_and_finishes_with_fresh_acl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = setup_operation(monkeypatch)
    with corpus.verified_source_operation(**state["kwargs"]) as proof:
        assert proof is not None
        for _ in range(10):
            assert (
                corpus.require_source(
                    MagicMock(),
                    **{
                        key: value
                        for key, value in state["kwargs"].items()
                        if key != "check_active"
                    },
                )
                == state["source"]
            )
        assert state["find"].call_count == 1
    assert state["find"].call_count == 2
    assert (
        state["find"].call_args_list[0].args[0]
        is not state["find"].call_args_list[1].args[0]
    )
    assert not proof.active
    assert state["check"].call_count >= 13


@pytest.mark.parametrize("mismatch", ["source", "user", "date", "scope", "tenant"])
def test_foreign_operation_never_reuses_authority(
    monkeypatch: pytest.MonkeyPatch, mismatch: str
) -> None:
    state = setup_operation(monkeypatch)
    with corpus.verified_source_operation(**state["kwargs"]):
        call = {
            key: value
            for key, value in state["kwargs"].items()
            if key != "check_active"
        }
        if mismatch == "source":
            call["source_id"] = uuid4()
        elif mismatch == "user":
            call["user"] = cast(User, SimpleNamespace(id=uuid4()))
        elif mismatch == "date":
            call["filters"] = call["filters"].model_copy(
                update={"as_of_date": date(2025, 1, 1)}
            )
        elif mismatch == "scope":
            call["filters"] = call["filters"].model_copy(
                update={"access_control_list": ["other"]}
            )
        else:
            state["tenant"] = "foreign"
        corpus.require_source(MagicMock(), **call)
        assert state["find"].call_count == 2
        state["tenant"] = "public"
    assert state["find"].call_count == 3


@pytest.mark.parametrize(
    "defect",
    [
        "acl",
        "identity",
        "publication",
        "cancel",
        "tenant",
        "filters",
        "user_id",
        "body",
    ],
)
def test_exit_failure_cannot_return_an_operation_result(
    monkeypatch: pytest.MonkeyPatch, defect: str
) -> None:
    state = setup_operation(monkeypatch)
    returned = []
    proof = None
    with pytest.raises((PermissionError, ValueError, RuntimeError)):
        with corpus.verified_source_operation(**state["kwargs"]) as proof:
            if defect == "acl":
                state["find"].return_value = ([], False)
            elif defect == "identity":
                state["find"].return_value = (
                    [replace(state["source"], file_id="changed")],
                    False,
                )
            elif defect == "publication":
                state["publication"].side_effect = PermissionError("publishing")
            elif defect == "cancel":
                state["check"].side_effect = RuntimeError("cancelled")
            elif defect == "tenant":
                state["tenant"] = "other"
            elif defect == "filters":
                state["kwargs"]["filters"].as_of_date = date(2025, 1, 1)
            elif defect == "user_id":
                state["kwargs"]["user"].id = uuid4()
            else:
                raise RuntimeError("reader failed")
        returned.append("would deliver")
    assert not returned
    assert proof is not None and not proof.active
    assert corpus._VERIFIED_SOURCE_OPERATION.get() is None


def test_nested_and_stale_copied_context_require_normal_fences(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = setup_operation(monkeypatch)
    call = {
        key: value for key, value in state["kwargs"].items() if key != "check_active"
    }
    with corpus.verified_source_operation(**state["kwargs"]) as proof:
        saved = copy_context()
        with corpus.verified_source_operation(**state["kwargs"]) as nested:
            assert nested is None
            corpus.require_source(MagicMock(), **call)
        assert corpus._VERIFIED_SOURCE_OPERATION.get() is proof
    assert state["find"].call_count == 3
    saved.run(lambda: corpus.require_source(MagicMock(), **call))
    assert state["find"].call_count == 4


@pytest.mark.parametrize("change", ["preexisting", "during", "removed"])
def test_broker_rejects_index_conflict_before_delivery(
    monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    state = setup_operation(monkeypatch)
    broker = corpus_tools.CorpusBroker(
        state["kwargs"]["user"], state["kwargs"]["filters"]
    )
    key = state["source"].id
    changed = state["index"].model_copy(update={"index_uuid": "other"})
    if change == "preexisting":
        broker.query_indexes[key] = changed
    with pytest.raises(corpus.CorpusScopeUnavailable):
        with broker.verified_provision_read(
            str(key), RunContext(services={"asv3_exp_cropped": True})
        ):
            if change == "during":
                broker.query_indexes[key] = changed
            elif change == "removed":
                broker.query_indexes.pop(key)


@pytest.mark.parametrize("cropped", [None, False, "true"])
def test_original_profiles_do_not_establish_operation(
    monkeypatch: pytest.MonkeyPatch, cropped: object
) -> None:
    state = setup_operation(monkeypatch)
    broker = corpus_tools.CorpusBroker(
        state["kwargs"]["user"], state["kwargs"]["filters"]
    )
    with broker.verified_provision_read(
        str(state["source"].id), RunContext(services={"asv3_exp_cropped": cropped})
    ):
        assert corpus._VERIFIED_SOURCE_OPERATION.get() is None
    state["find"].assert_not_called()


def test_final_revocation_returns_no_evidence_from_actual_provision_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = setup_operation(monkeypatch)
    broker = corpus_tools.CorpusBroker(
        state["kwargs"]["user"], state["kwargs"]["filters"]
    )
    source = state["source"]
    rows = [
        corpus.CorpusChunk(
            str(n),
            source.id,
            "canonical original",
            n,
            n,
            (f"MADDE {n}",),
            {},
            None,
            None,
            "active",
        )
        for n in (235, 236)
    ]
    monkeypatch.setattr(broker, "provision_start", lambda *_: 0)
    monkeypatch.setattr(broker, "related_sources_for_provision", lambda *_: None)

    def page(*_: Any, **__: Any) -> Any:
        state["find"].return_value = ([], False)
        return source, rows, False

    monkeypatch.setattr(broker, "page", page)
    tool = next(
        spec
        for spec in corpus_tools.build_corpus_specs(broker)
        if spec.name == "read_provision"
    )
    result = tool.handler(
        {"source_id": str(source.id), "article": "235"},
        RunContext(services={"asv3_exp_cropped": True}),
    )
    assert result.status is OutcomeStatus.DENIED
    assert not result.evidence
