"""Strict aggregate originals, selected model and frontend citation publication."""

import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from copy import deepcopy
from datetime import date
from threading import Barrier, Lock, get_ident
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import (
    EvidenceItem,
    OutcomeStatus,
    RunContext,
    RunStopped,
    SharedBudget,
    ToolOutcome,
    ToolSpec,
)
from onyx.chat.chat_state import ChatStateContainer
from onyx.chat.emitter import BufferedEmitter
from onyx.chat.models import ChatMessageSimple
from onyx.configs.constants import MessageType
from onyx.context.search.models import IndexFilters, SearchDoc
from onyx.db.asv3_runs import checkpoint_progress_packets, encode_asv3_checkpoint
from onyx.db.models import User
from onyx.error_handling.error_codes import OnyxErrorCode
from onyx.error_handling.exceptions import OnyxError
from onyx.llm.interfaces import LLM, LLMConfig
from onyx.server.query_and_chat.streaming_models import (
    AgentResponseDelta,
    AgentResponseStart,
    ASv3Progress,
    CitationInfo,
)
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


def test_concurrent_searches_keep_all_originals_in_four_acquisition_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    broker = corpus.SupersearchCorpusBroker.__new__(corpus.SupersearchCorpusBroker)
    broker.user = MagicMock()
    broker.filters = IndexFilters(
        access_control_list=["authorized"],
        asv3_document_set_id=442,
        forced_document_set=["PC Külliyatı"],
        as_of_date=date(2026, 10, 8),
    )
    monkeypatch.setattr(
        corpus, "get_session_with_current_tenant", lambda: nullcontext(MagicMock())
    )
    template = original()
    assert template.search_doc is not None
    source_ids = [str(uuid4()) for _ in range(4)]
    documents: list[SearchDoc] = []
    originals: dict[str, dict[str, EvidenceItem]] = {}
    expected: dict[tuple[str, int], list[EvidenceItem]] = {}
    for source_id in source_ids:
        owned: dict[str, EvidenceItem] = {}
        for chunk_id in ("atomic0", "atomic1", "closure"):
            owned[chunk_id] = EvidenceItem(
                source_id=source_id,
                chunk_id=chunk_id,
                text=f"{source_id}: özgün {chunk_id} hükmü",
                metadata={"article_closure_complete": True},
                search_doc=template.search_doc.model_copy(
                    deep=True,
                    update={
                        "document_id": source_id,
                        "metadata": {"regulatory_chunk_id": chunk_id},
                    },
                ),
            )
        originals[source_id] = owned
        for ordinal, (center, members) in enumerate(
            (
                ("aggregate", ("atomic0", "closure", "atomic1")),
                ("atomic0", ("atomic0", "closure")),
                ("stale", ()),
            )
        ):
            documents.append(
                template.search_doc.model_copy(
                    deep=True,
                    update={
                        "document_id": source_id,
                        "chunk_ind": ordinal,
                        "metadata": {"regulatory_chunk_id": center},
                    },
                )
            )
            expected[source_id, ordinal] = [
                owned[member].model_copy(
                    update={
                        "metadata": {
                            **owned[member].metadata,
                            "supersearch_retrieved_center_id": center,
                            "supersearch_aggregate_resolved": center == "aggregate",
                        }
                    }
                )
                for member in members
            ]

    lock, first_read = Lock(), Barrier(4)
    active = peak = 0
    owners: dict[int, int] = {}
    visited: dict[int, list[str]] = {index: [] for index in range(4)}

    def resolve(_session: Any, **kwargs: Any) -> dict[str, tuple[str, ...]]:
        assert kwargs["user"] is broker.user and kwargs["filters"] is broker.filters
        assert kwargs["center_ids"] == ("aggregate", "atomic0", "stale")
        kwargs["check_active"]()
        return {
            "aggregate": ("atomic0", "atomic1"),
            "atomic0": ("atomic0",),
            "stale": (),
        }

    def hydrate(
        _broker: Any, docs: list[SearchDoc], context: RunContext
    ) -> dict[tuple[str, int], list[EvidenceItem]]:
        nonlocal active, peak
        index = context.services["fixture_search"]
        assert type(index) is int and context.scope == broker.filters.model_dump(
            mode="json"
        )
        assert get_ident() == owners[index]
        source_id = docs[0].document_id
        assert [doc.metadata["regulatory_chunk_id"] for doc in docs] == [
            "atomic0",
            "atomic1",
        ]
        with lock:
            active += 1
            peak = max(peak, active)
            visited[index].append(source_id)
            initial = len(visited[index]) == 1
        try:
            if initial:
                first_read.wait(timeout=5)
            owned = originals[source_id]
            return {
                (source_id, ordinal): [
                    owned[str(doc.metadata["regulatory_chunk_id"])],
                    owned["closure"],
                ]
                for ordinal, doc in enumerate(docs)
            }
        finally:
            with lock:
                active -= 1

    def search(index: int) -> dict[tuple[str, int], list[EvidenceItem]]:
        owners[index] = get_ident()
        context = RunContext(scope=broker.filters.model_dump(mode="json"))
        context.services["fixture_search"] = index
        return broker.hydrate_search_centers(documents, context)

    monkeypatch.setattr(corpus, "resolve_supersearch_center_ids", resolve)
    monkeypatch.setattr(corpus.CorpusBroker, "hydrate_search_results", hydrate)
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(search, range(4)))
    assert peak == 4 and active == 0
    assert all(sources == source_ids for sources in visited.values())
    for result in results:
        assert list(result) == list(expected)
        assert {
            key: [item.model_dump(mode="json") for item in items]
            for key, items in result.items()
        } == {
            key: [item.model_dump(mode="json") for item in items]
            for key, items in expected.items()
        }


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
    completed_batch = next(
        packet
        for packet in progress
        if packet.title == "Supersearch: özgün kaynaklar" and packet.active_tasks == 0
    )
    assert completed_batch.status == "running"
    assert {
        packet.phase
        for packet in progress
        if packet.sequence > completed_batch.sequence
    } >= {"final", "verification", "completed"}
    # The frontend ignores every later packet after an unscoped terminal status.
    assert [
        packet
        for packet in progress
        if not packet.task_id and packet.status in {"completed", "failed", "cancelled"}
    ] == [progress[-1]]
    assert progress[-1].phase == "completed"
    citations = [
        packet.obj for packet in packets if isinstance(packet.obj, CitationInfo)
    ]
    assert len(citations) == 1 and citations[0].preview_url == "/api/asv3/citation/3/1"
    answer = state.get_answer_tokens()
    assert answer is not None and "bildirim tarihinden itibaren bir yıl" in answer


@pytest.mark.parametrize(
    "failure_kind,checkpoint_fails",
    [
        ("provider", False),
        ("provider", True),
        ("cancelled", False),
        ("unavailable", False),
    ],
)
def test_failure_preserves_native_history_without_publishing_draft(
    monkeypatch: pytest.MonkeyPatch,
    failure_kind: str,
    checkpoint_fails: bool,
) -> None:
    state, emitter = ChatStateContainer(), BufferedEmitter()
    llm = selected_llm()
    user = User(id=uuid4(), email="fixture@example.com")
    failure = (
        TimeoutError("fixture provider timeout")
        if failure_kind == "provider"
        else RunStopped(f"fixture {failure_kind}")
    )
    snapshots: list[dict[str, Any]] = []
    broker = MagicMock()
    broker.shared_read_fence.return_value = "authorized-pc-snapshot"
    dependencies = MagicMock()
    dependencies.receipts = []
    dependencies.expand.return_value = []

    def bind(**kwargs: Any) -> IndexFilters:
        return kwargs["filters"].model_copy(
            update={
                "asv3_document_set_id": 442,
                "forced_document_set": ["PC Külliyatı"],
                "access_control_list": ["authorized"],
            }
        )

    def fixture_gateway(**kwargs: Any) -> FixtureGateway:
        fixture = FixtureGateway(
            kwargs["ledger"],
            [plan(), WriterDecision(answer=ANSWER, unresolved_need_ids=[], actions=[])],
        )
        complete = fixture.complete

        def fail_review(*args: Any, **options: Any) -> Any:
            if args[3] == LLMFlow.SUPERSEARCH_REVIEW:
                assert kwargs["ledger"].citation_numbers() == (1,)
                raise failure
            return complete(*args, **options)

        monkeypatch.setattr(fixture, "complete", fail_review)
        return fixture

    def checkpoint(**kwargs: Any) -> None:
        assert kwargs["message_id"] == 3 and kwargs["user_id"] == user.id
        snapshots.append(deepcopy(kwargs["snapshot"]))
        if checkpoint_fails:
            raise RuntimeError("fixture checkpoint failure")

    monkeypatch.setattr(runtime, "bind_supersearch_pc_scope", bind)
    monkeypatch.setattr(
        runtime, "SupersearchCorpusBroker", lambda *_args, **_kwargs: broker
    )
    monkeypatch.setattr(
        runtime, "SupersearchDependencyExpander", lambda **_kwargs: dependencies
    )
    monkeypatch.setattr(runtime, "build_search_adapter", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        runtime,
        "corpus_specs",
        lambda _broker: [
            ToolSpec(
                name="read_named_provision",
                description="Canonical fixture read",
                parameters={"type": "object"},
                handler=lambda _arguments, _context: ToolOutcome(
                    status=OutcomeStatus.FOUND,
                    summary="Original",
                    evidence=[original()],
                ),
            )
        ],
    )
    monkeypatch.setattr(runtime, "SelectedModelGateway", fixture_gateway)
    monkeypatch.setattr(runtime, "save_asv3_checkpoint", checkpoint)
    monkeypatch.setattr(runtime, "is_connected", lambda *_args: True)
    monkeypatch.setattr(
        runtime, "ensure_trace", lambda *_args, **_kwargs: nullcontext()
    )
    expected_exception = TimeoutError if failure_kind == "provider" else OnyxError
    with pytest.raises(expected_exception) as raised:
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
            user=user,
            chat_session_id=uuid4(),
            user_message_id=2,
            assistant_message_id=3,
            cache=MagicMock(),
            filters=IndexFilters(access_control_list=[]),
            document_set_names_override=["PC Külliyatı"],
        )
    if failure_kind == "provider":
        assert raised.value is failure
    else:
        assert isinstance(raised.value, OnyxError)
        assert raised.value.error_code == OnyxErrorCode.LLM_PROVIDER_ERROR
    assert len(snapshots) == 1
    snapshot = snapshots[0]
    minimal_fields = {
        "run_id",
        "sequence",
        "request",
        "scope",
        "asv3_workflow_variant",
        "prompt_version",
        "publication_status",
        "processing_seconds",
        "progress",
    }
    if failure_kind == "provider":
        assert set(snapshot) == minimal_fields
    else:
        assert set(snapshot) == minimal_fields | {
            "evidence",
            "supersearch",
            "authority_dependencies",
            "source_receipts",
            "acquisition_counts",
        }
        assert snapshot["supersearch"]["answer"] is None
    assert snapshot["asv3_workflow_variant"] == "supersearch"
    assert snapshot["publication_status"] == (
        "cancelled" if failure_kind == "cancelled" else "unavailable"
    )
    assert snapshot["scope"]["asv3_document_set_id"] == 442
    assert snapshot["processing_seconds"] >= 0
    replay = checkpoint_progress_packets(encode_asv3_checkpoint(snapshot))
    assert replay[-1]["workflow"] == "supersearch"
    assert replay[-1]["status"] == "failed"
    assert replay[-1]["phase"] == (
        "cancelled" if failure_kind == "cancelled" else "failed"
    )
    assert ANSWER not in json.dumps(snapshot, ensure_ascii=False)
    packets = list(emitter.get_packets())
    assert not any(
        isinstance(packet.obj, (CitationInfo, AgentResponseStart, AgentResponseDelta))
        for packet in packets
    )
    assert state.get_answer_tokens() is None
    broker.revalidate_evidence.assert_not_called()
