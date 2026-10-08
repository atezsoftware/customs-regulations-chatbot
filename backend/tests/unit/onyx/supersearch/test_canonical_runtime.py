"""Strict aggregate originals, selected model and frontend citation publication."""

import json
from contextlib import nullcontext
from copy import deepcopy
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import (
    OutcomeStatus,
    RunContext,
    SharedBudget,
    ToolOutcome,
    ToolSpec,
)
from onyx.chat.chat_state import ChatStateContainer
from onyx.chat.emitter import BufferedEmitter
from onyx.chat.models import ChatMessageSimple
from onyx.configs.constants import MessageType
from onyx.context.search.models import IndexFilters
from onyx.db.models import User
from onyx.llm.interfaces import LLM, LLMConfig
from onyx.server.query_and_chat.streaming_models import ASv3Progress, CitationInfo
from onyx.supersearch import corpus, gateway, runtime
from onyx.supersearch.models import WriterDecision
from onyx.tracing.flows import LLMFlow
from tests.unit.onyx.supersearch.test_engine import (
    ANSWER,
    SOURCE_ID,
    FixtureGateway,
    original,
    plan,
    review,
)


def selected_llm() -> LLM:
    llm = MagicMock(spec=LLM)
    llm.config = LLMConfig(
        model_provider="openai",
        model_name="chosen-model",
        temperature=0,
        max_input_tokens=128_000,
    )
    return cast(LLM, llm)


def test_gateway_uses_only_selected_model_and_records_delivered_originals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm = selected_llm()
    response = SimpleNamespace(
        choice=SimpleNamespace(
            finish_reason="stop",
            message=SimpleNamespace(
                tool_calls=None,
                content=json.dumps(
                    {"answer": ANSWER, "unresolved_need_ids": [], "actions": []}
                ),
            ),
        )
    )
    cast(MagicMock, llm).invoke.return_value = response
    span = SimpleNamespace(span_data=SimpleNamespace(model_config={}))
    monkeypatch.setattr(
        gateway, "llm_generation_span", lambda **_kwargs: nullcontext(span)
    )
    monkeypatch.setattr(gateway, "record_llm_response", lambda *_args: None)
    context = RunContext(
        timeout_seconds=float("inf"),
        budget=SharedBudget(unlimited_execution=True),
        scope={"asv3_document_set_id": 442},
    )
    ledger = EvidenceLedger()
    ledger.add([original()], context)
    generation = gateway.SelectedModelGateway(llm=llm, ledger=ledger, context=context)
    decision = generation.complete(
        "PC originals only",
        {"original_evidence": json.loads(ledger.serialize_records([1]))},
        WriterDecision,
        LLMFlow.SUPERSEARCH_ANSWER,
        True,
    )
    assert decision.answer == ANSWER
    assert cast(MagicMock, llm).invoke.call_count == 1
    assert generation.last_delivered_citations == {1}
    assert span.span_data.model_config["supersearch_document_set_id"] == "442"
    assert "max_tokens" not in cast(MagicMock, llm).invoke.call_args.kwargs


def test_aggregate_projection_metadata_cannot_supply_original_membership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    item = original()
    assert item.search_doc is not None
    projection = item.search_doc.model_copy(
        deep=True,
        update={
            "metadata": {
                "regulatory_chunk_id": "aggregate",
                "source_regulatory_chunk_ids": ["untrusted-projected-child"],
            }
        },
    )
    broker = corpus.SupersearchCorpusBroker.__new__(corpus.SupersearchCorpusBroker)
    broker.user = MagicMock()
    broker.filters = IndexFilters(access_control_list=[])
    monkeypatch.setattr(
        corpus, "get_session_with_current_tenant", lambda: nullcontext(MagicMock())
    )
    seen: list[tuple[str, ...]] = []

    def resolve(_session: Any, **kwargs: Any) -> dict[str, tuple[str, ...]]:
        seen.append(kwargs["center_ids"])
        assert str(kwargs["source_id"]) == SOURCE_ID
        return {"aggregate": ("atomic-clock",)}

    def hydrate(_broker: Any, docs: list[Any], _context: RunContext) -> dict[Any, Any]:
        assert [doc.metadata["regulatory_chunk_id"] for doc in docs] == ["atomic-clock"]
        return {(SOURCE_ID, 0): [item]}

    monkeypatch.setattr(corpus, "resolve_supersearch_center_ids", resolve)
    monkeypatch.setattr(corpus.CorpusBroker, "hydrate_search_results", hydrate)
    hydrated = broker.hydrate_search_centers([projection], RunContext())
    assert seen == [("aggregate",)]
    assert hydrated[SOURCE_ID, 0][0].chunk_id == "atomic-clock"
    assert hydrated[SOURCE_ID, 0][0].metadata["supersearch_aggregate_resolved"] is True


def test_source_to_streaming_publication_preserves_pc_fence_and_selected_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = ChatStateContainer()
    emitter = BufferedEmitter()
    llm = selected_llm()
    selected_user = User(id=uuid4(), email="fixture@example.com")
    bound_scopes: list[IndexFilters] = []
    snapshots: list[dict[str, Any]] = []
    order: list[str] = []
    selected_models: list[LLM] = []

    def bind(**kwargs: Any) -> IndexFilters:
        assert kwargs["document_set_names_override"] == ["PC Külliyatı"]
        scope = kwargs["filters"].model_copy(
            deep=True,
            update={
                "asv3_document_set_id": 442,
                "forced_document_set": ["PC Külliyatı"],
                "access_control_list": ["authorized"],
            },
        )
        bound_scopes.append(scope)
        return scope

    class FixtureBroker:
        def __init__(self, user: User, scope: IndexFilters, *, vision_llm: LLM) -> None:
            assert user is selected_user and vision_llm is llm
            self.user, self.filters = user, scope
            self.shared_read_fence = lambda *_args: "snapshot"

        def revalidate_evidence(self, items: list[Any], context: RunContext) -> None:
            context.check_active()
            assert len(items) == 1 and items[0].chunk_id == "atomic-clock"
            order.append("revalidate")

    class FixtureDependencies:
        receipts: list[Any] = []

        def __init__(self, **_kwargs: Any) -> None:
            pass

        def expand(self, frozen: Any, *, frontier: set[int]) -> list[Any]:
            assert frontier == {1} and frozen.needs[0].need_id == "clock"
            return []

    def read(_arguments: Any, _context: RunContext) -> ToolOutcome:
        return ToolOutcome(
            status=OutcomeStatus.FOUND, summary="Original", evidence=[original()]
        )

    def fixture_gateway(**kwargs: Any) -> FixtureGateway:
        selected_models.append(kwargs["llm"])
        return FixtureGateway(
            kwargs["ledger"],
            [
                plan(),
                WriterDecision(answer=ANSWER, unresolved_need_ids=[], actions=[]),
                review(),
            ],
        )

    def checkpoint(**kwargs: Any) -> None:
        assert "revalidate" in order
        order.append("checkpoint")
        snapshots.append(deepcopy(kwargs["snapshot"]))

    monkeypatch.setattr(runtime, "bind_supersearch_pc_scope", bind)
    monkeypatch.setattr(runtime, "SupersearchCorpusBroker", FixtureBroker)
    monkeypatch.setattr(runtime, "SupersearchDependencyExpander", FixtureDependencies)
    monkeypatch.setattr(runtime, "build_search_adapter", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        runtime,
        "corpus_specs",
        lambda _broker: [
            ToolSpec(
                name="read_named_provision",
                description="Canonical fixture read",
                parameters={"type": "object"},
                handler=read,
            )
        ],
    )
    monkeypatch.setattr(runtime, "SelectedModelGateway", fixture_gateway)
    monkeypatch.setattr(runtime, "save_asv3_checkpoint", checkpoint)
    monkeypatch.setattr(runtime, "is_connected", lambda *_args: True)
    monkeypatch.setattr(
        runtime, "ensure_trace", lambda *_args, **_kwargs: nullcontext()
    )
    runtime.run_supersearch_loop(
        emitter=emitter,
        state_container=state,
        simple_chat_history=[
            ChatMessageSimple(
                message="Başvuru süresi nedir?",
                token_count=8,
                message_type=MessageType.USER,
            )
        ],
        tools=[],
        llm=llm,
        user=selected_user,
        chat_session_id=uuid4(),
        user_message_id=2,
        assistant_message_id=3,
        cache=MagicMock(),
        filters=IndexFilters(access_control_list=["caller-acl"]),
        document_set_names_override=["PC Külliyatı"],
    )
    assert selected_models == [llm]
    assert bound_scopes[0].asv3_document_set_id == 442
    assert order == ["revalidate", "checkpoint", "checkpoint"]
    assert snapshots[-1]["asv3_workflow_variant"] == "supersearch"
    assert snapshots[-1]["publication_status"] == "verified"
    packets = list(emitter.get_packets())
    progress = [
        packet.obj for packet in packets if isinstance(packet.obj, ASv3Progress)
    ]
    assert progress and all(packet.workflow == "supersearch" for packet in progress)
    citations = [
        packet.obj for packet in packets if isinstance(packet.obj, CitationInfo)
    ]
    assert len(citations) == 1 and citations[0].preview_url == "/api/asv3/citation/3/1"
    answer = state.get_answer_tokens()
    assert answer is not None and "bildirim tarihinden itibaren bir yıl" in answer
