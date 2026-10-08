"""Final chat publication is fenced by fresh original authorization."""

from collections.abc import Callable
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from datetime import date
from typing import Any, Iterator, NamedTuple, cast
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from pydantic import JsonValue

from onyx.asv3.corpus_tools import build_corpus_specs
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import EvidenceItem, RunContext, RunStopped, ToolSpec
from onyx.asv3.registry import CapabilityRegistry
from onyx.cache.interface import CacheBackend
from onyx.chat.chat_state import ChatStateContainer
from onyx.chat.emitter import BufferedEmitter
from onyx.chat.models import ChatMessageSimple
from onyx.configs.constants import MessageType
from onyx.context.search.models import BaseFilters, IndexFilters
from onyx.db.legal_composite_sources import (
    SourceClassification,
    SourceKind,
    SourceLaneCatalogue,
    source_scope_sha256,
)
from onyx.db.models import User
from onyx.legal_composite import runtime
from onyx.legal_composite.models import WorkflowResult
from onyx.legal_composite.routing import SourceLaneRouter
from onyx.llm.interfaces import LLM, LLMConfig
from onyx.server.query_and_chat.streaming_models import (
    AgentResponseDelta,
    AgentResponseStart,
    ASv3Progress,
)
from onyx.tools.interface import Tool
from onyx.tools.tool_implementations.search.search_tool import SearchTool
from tests.unit.onyx.legal_composite.test_review_assessment import original


class RuntimeHarness(NamedTuple):
    state: ChatStateContainer
    emitter: BufferedEmitter
    snapshots: list[dict[str, JsonValue]]
    events: list[str]
    broker: MagicMock
    search: MagicMock
    gateway: MagicMock
    binding_scopes: list[IndexFilters]
    run: Callable[[BaseFilters | None], None]


@pytest.fixture
def harness(monkeypatch: pytest.MonkeyPatch) -> RuntimeHarness:
    state = ChatStateContainer()
    emitter = BufferedEmitter()
    snapshots: list[dict[str, JsonValue]] = []
    events: list[str] = []
    broker = MagicMock()
    gateway = MagicMock()
    binding_scopes: list[IndexFilters] = []
    search = MagicMock(spec=SearchTool)
    search.auto_detect_filters = True
    search.bypass_acl = True
    search.enable_slack_search = True
    fork = MagicMock(spec=runtime.CompositeSearchTool)
    search.fork_for_independent_context.return_value = fork
    llm_mock = MagicMock(spec=LLM)
    llm_mock.config = LLMConfig(
        model_provider="openai",
        model_name="gpt-5-mini",
        temperature=1,
        max_input_tokens=32_000,
    )
    search.llm = llm_mock

    class FakeEngine:
        selection = None

        def __init__(
            self, ledger: EvidenceLedger, report: Callable[[str, str], None]
        ) -> None:
            self.ledger = ledger
            self.report = report

        def run(
            self, request: str, history: str, instructions: str | None
        ) -> WorkflowResult:
            assert request and history and instructions is None
            self.report("tools", "tr")
            uncited = original("Uncited original.", "uncited")
            assert uncited.search_doc is not None
            uncited.search_doc.chunk_ind = 2
            self.ledger.add(
                [original("Authorized original.", "cited"), uncited], RunContext()
            )
            return WorkflowResult(
                answer="Kaynakla desteklenen sonuç [1].", status="verified", gaps=[]
            )

    def engine_factory(**kwargs: Any) -> FakeEngine:
        assert kwargs["gateway"] is gateway
        assert gateway.budget.deadline == kwargs["acquirer"].context.deadline
        assert fork.llm is gateway.research_proxy.return_value
        return FakeEngine(kwargs["ledger"], kwargs["report"])

    def gateway_factory(**kwargs: Any) -> MagicMock:
        gateway.budget = kwargs["budget"]
        return gateway

    def bind_scope(*, user: User, filters: IndexFilters) -> IndexFilters:
        assert user.id
        binding_scopes.append(filters.model_copy(deep=True))
        filters.access_control_list = ["recomputed-user-acl"]
        return filters

    def revalidate(items: list[EvidenceItem], context: RunContext) -> None:
        context.check_active()
        assert len(items) == 1 and items[0].chunk_id == "cited"
        events.append("revalidate")

    def save_checkpoint(
        *, message_id: int, user_id: Any, snapshot: dict[str, JsonValue]
    ) -> None:
        assert message_id == 3 and user_id
        assert "revalidate" in events
        events.append("checkpoint")
        snapshots.append(deepcopy(snapshot))

    broker.revalidate_evidence.side_effect = revalidate

    def corpus_specs(
        _broker: Any,
        *,
        require_search_targets: bool,
        source_identity_guidance: bool,
    ) -> list[ToolSpec]:
        assert require_search_targets and source_identity_guidance
        specs = build_corpus_specs(
            _broker,
            require_search_targets=require_search_targets,
            source_identity_guidance=source_identity_guidance,
        )
        assert {spec.name for spec in specs} == {
            "resolve_source",
            "read_source_range",
            "read_chunk",
            "read_chunk_context",
            "read_provision",
            "search_source_text",
            "query_corpus",
            "follow_reference",
            "diagnose_source",
            "compare_versions",
            "search_corpus",
        }
        assert all(not spec.external and not spec.orchestrates for spec in specs)
        assert "own title/name" in specs[0].description
        return specs

    monkeypatch.setattr(runtime, "is_connected", lambda _session, _cache: True)
    monkeypatch.setattr(runtime, "bind_pc_corpus_scope", bind_scope)

    monkeypatch.setattr(runtime, "build_source_selector", lambda **_kwargs: None)
    monkeypatch.setattr(runtime.CompositeSearchTool, "from_fork", lambda _fork: _fork)

    def broker_factory(user: User, scope: IndexFilters) -> MagicMock:
        broker.user = user
        broker.filters = scope
        return broker

    monkeypatch.setattr(runtime, "CorpusBroker", broker_factory)
    monkeypatch.setattr(
        runtime, "build_search_adapter", lambda *_args, **_kwargs: object()
    )
    monkeypatch.setattr(runtime, "build_corpus_specs", corpus_specs)
    monkeypatch.setattr(runtime, "build_core_specs", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        runtime, "get_session_with_current_tenant", lambda: nullcontext(MagicMock())
    )
    monkeypatch.setattr(runtime, "BudgetedGateway", gateway_factory)
    monkeypatch.setattr(runtime, "LegalCompositeEngine", engine_factory)
    monkeypatch.setattr(
        runtime, "ensure_trace", lambda *_args, **_kwargs: nullcontext()
    )
    monkeypatch.setattr(runtime, "save_asv3_checkpoint", save_checkpoint)
    user = User(id=uuid4(), email="unit@example.com")

    def prepared_catalogue(_session, *, user, filters, check_active):
        check_active()
        return SourceLaneCatalogue(
            user_id=user.id,
            scope_sha256=source_scope_sha256(user, filters),
            records=(
                SourceClassification(
                    source_id=uuid4(),
                    name="prepared",
                    kind=SourceKind.STATUTE,
                    method="prepared_original_opening",
                    uncertain=False,
                    observed_document_types=(),
                    prepared=True,
                ),
            ),
            complete=True,
        )

    monkeypatch.setattr(
        runtime, "load_prepared_source_lane_catalogue", prepared_catalogue
    )

    def run(filters: BaseFilters | None) -> None:
        runtime.run_legal_composite_loop(
            emitter=emitter,
            state_container=state,
            simple_chat_history=[
                ChatMessageSimple(
                    message="Kaynaklı sonucu açıklayın.",
                    token_count=8,
                    message_type=MessageType.USER,
                )
            ],
            tools=[cast(Tool, search)],
            llm=cast(LLM, llm_mock),
            user=user,
            chat_session_id=uuid4(),
            user_message_id=2,
            assistant_message_id=3,
            cache=cast(CacheBackend, MagicMock()),
            filters=filters,
            document_set_names_override=["PC Külliyatı"],
        )

    return RuntimeHarness(
        state, emitter, snapshots, events, broker, search, gateway, binding_scopes, run
    )


def test_runtime_wraps_typed_search_with_captured_lane_guard(
    harness: RuntimeHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    lane = MagicMock()
    adapter = object()
    guarded = object()
    lane.guard_search_adapter.return_value = guarded
    monkeypatch.setattr(runtime, "build_lane_broker", lambda *_args: lane)
    monkeypatch.setattr(
        runtime, "build_search_adapter", lambda *_args, **_kwargs: adapter
    )

    def router_factory(
        catalogue: SourceLaneCatalogue,
        build_registry: Callable[[SourceKind], CapabilityRegistry],
    ) -> SourceLaneRouter:
        build_registry(SourceKind.STATUTE)
        return SourceLaneRouter(catalogue, build_registry)

    monkeypatch.setattr(runtime, "SourceLaneRouter", router_factory)
    harness.run(None)
    lane.guard_search_adapter.assert_called_once_with(adapter)
    assert lane.search_adapter is guarded


@pytest.mark.parametrize(
    "reason", ["Original permission revoked", "Cited original hash changed"]
)
def test_denied_original_emits_no_final_tokens_or_verified_checkpoint(
    harness: RuntimeHarness, reason: str
) -> None:
    harness.broker.revalidate_evidence.side_effect = RuntimeError(reason)
    with pytest.raises(RuntimeError, match=reason):
        harness.run(None)
    assert harness.snapshots == []
    assert harness.state.get_answer_tokens() is None
    assert harness.state.get_all_search_docs() == {}
    assert harness.state.get_citation_to_doc() == {}
    assert not any(
        isinstance(packet.obj, (AgentResponseStart, AgentResponseDelta))
        for packet in harness.emitter.get_packets()
    )


def test_revalidated_cited_documents_only_are_published(
    harness: RuntimeHarness,
) -> None:
    harness.run(None)
    assert harness.events[0] == "revalidate"
    assert harness.snapshots and all(
        snapshot["publication_status"] == "verified" for snapshot in harness.snapshots
    )
    starts = [
        packet.obj
        for packet in harness.emitter.get_packets()
        if isinstance(packet.obj, AgentResponseStart)
    ]
    assert len(starts) == 1
    assert len(starts[0].final_documents or []) == 1
    assert (starts[0].final_documents or [])[0].blurb == "Authorized original."
    assert len(harness.state.get_all_search_docs()) == 1
    assert "Uncited" not in str(harness.state.get_citation_to_doc())
    assert harness.state.get_answer_tokens()


@pytest.mark.parametrize("prebound", [False, True])
def test_independent_search_uses_shared_budget_and_preserves_filter_scope(
    harness: RuntimeHarness, prebound: bool
) -> None:
    label_run_id = uuid4()
    filters = (
        IndexFilters(
            access_control_list=["caller-supplied-admin"],
            as_of_date=date(2024, 4, 5),
            regulatory_workflow_mode="fast",
            regulatory_label_search_enabled=True,
            regulatory_label_run_ids=(label_run_id,),
        )
        if prebound
        else None
    )
    harness.run(filters)
    harness.search.fork_for_independent_context.assert_called_once()
    fork = harness.search.fork_for_independent_context.return_value
    assert fork.bypass_acl is False
    assert fork.auto_detect_filters is False
    assert fork.enable_slack_search is False
    harness.gateway.research_proxy.assert_called_once_with()
    assert fork.llm is harness.gateway.research_proxy.return_value
    assert fork.llm is not harness.search.llm
    assert len(harness.binding_scopes) == 1
    bound_input = harness.binding_scopes[0]
    assert bound_input.access_control_list == []
    assert bound_input.forced_document_set == ["PC Külliyatı"]
    if filters is not None:
        assert bound_input.as_of_date == filters.as_of_date
        assert bound_input.regulatory_workflow_mode == "fast"
        assert bound_input.regulatory_label_search_enabled is True
        assert bound_input.regulatory_label_run_ids == (label_run_id,)
        assert filters.access_control_list == ["caller-supplied-admin"]
    else:
        assert bound_input.as_of_date is None
        assert bound_input.regulatory_workflow_mode == "standard"
        assert bound_input.regulatory_label_search_enabled is False
        assert bound_input.regulatory_label_run_ids == ()
    assert (
        harness.search.bypass_acl is True and harness.search.auto_detect_filters is True
    )


def test_prepared_catalogue_starts_without_any_opening_read(
    harness: RuntimeHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    from onyx.db import legal_composite_sources
    from onyx.legal_composite.source_lanes import CandidateSourceClassifier

    order: list[str] = []
    enumerate_sources = MagicMock(
        side_effect=AssertionError("Full enumeration is forbidden")
    )
    classify_candidates = MagicMock(
        side_effect=AssertionError("No candidate exists yet")
    )
    monkeypatch.setattr(
        legal_composite_sources, "load_source_lane_catalogue", enumerate_sources
    )
    monkeypatch.setattr(
        CandidateSourceClassifier, "classify_sources", classify_candidates
    )
    original_gateway = runtime.BudgetedGateway

    def provider(**kwargs: Any) -> Any:
        order.append("provider")
        return original_gateway(**kwargs)

    @contextmanager
    def trace_start(*_args: Any, **kwargs: Any) -> Iterator[None]:
        assert kwargs["metadata"]["assistant_message_id"] == 3
        order.append("trace_start")
        try:
            yield
        finally:
            order.append("trace_end")

    monkeypatch.setattr(runtime, "ensure_trace", trace_start)
    monkeypatch.setattr(runtime, "BudgetedGateway", provider)
    harness.run(None)
    assert order == ["trace_start", "provider", "trace_end"]
    enumerate_sources.assert_not_called()
    classify_candidates.assert_not_called()
    progress = [
        packet.obj
        for packet in harness.emitter.get_packets()
        if isinstance(packet.obj, ASv3Progress)
    ]
    assert "tür" in progress[0].title.lower() or "kaynak" in progress[0].title.lower()
    assert "kayıt" in (progress[0].message or "").lower()
    assert [event.sequence for event in progress] == list(range(1, len(progress) + 1))
    assert harness.snapshots[-1]["progress"] == [
        event.model_dump(mode="json") for event in progress
    ]


def test_query_first_cancellation_never_creates_provider_or_final_answer(
    harness: RuntimeHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = MagicMock(wraps=runtime.BudgetedGateway)
    monkeypatch.setattr(runtime, "BudgetedGateway", provider)
    monkeypatch.setattr(runtime, "is_connected", lambda *_args: False)
    with pytest.raises(RunStopped):
        harness.run(None)
    provider.assert_not_called()
    assert harness.snapshots == [] and harness.state.get_answer_tokens() is None


def test_query_first_progress_failure_prevents_provider_setup(
    harness: RuntimeHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = MagicMock(wraps=runtime.BudgetedGateway)
    original_emit = harness.emitter.emit

    def failing_emit(packet: Any) -> None:
        if isinstance(packet.obj, ASv3Progress):
            raise RuntimeError("Progress transport disconnected")
        original_emit(packet)

    monkeypatch.setattr(harness.emitter, "emit", failing_emit)
    monkeypatch.setattr(runtime, "BudgetedGateway", provider)
    with pytest.raises(RuntimeError, match="Progress transport disconnected"):
        harness.run(None)
    provider.assert_not_called()
    assert harness.snapshots == [] and harness.state.get_answer_tokens() is None
