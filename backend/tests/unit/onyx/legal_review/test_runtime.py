"""Provider-free canonical acquisition, publication and persisted chat state."""

import copy
from contextlib import nullcontext
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from pydantic import JsonValue, SecretStr

from onyx.asv3.models import (
    EvidenceItem,
    OutcomeStatus,
    RunContext,
    RunStopped,
    ToolOutcome,
    ToolSpec,
)
from onyx.cache.interface import CacheBackend
from onyx.chat.chat_state import ChatStateContainer
from onyx.chat.emitter import Emitter
from onyx.chat.models import ChatMessageSimple
from onyx.configs import app_configs
from onyx.configs.constants import MessageType
from onyx.context.search.models import IndexFilters
from onyx.db.models import User
from onyx.error_handling.exceptions import OnyxError
from onyx.legal_review import runtime
from onyx.legal_review.models import (
    DecisionProviderConfig,
    PassageSupport,
    ReviewResult,
    WorkflowPolicy,
    WorkflowResult,
)
from onyx.legal_review.source_accounting import SourceAssessment, SourceInventory
from onyx.llm.interfaces import LLM, LLMConfig
from onyx.server.query_and_chat.streaming_models import (
    AgentResponseDelta,
    AgentResponseStart,
    ASv3Progress,
    CitationInfo,
    Packet,
    SectionEnd,
)
from onyx.tools.tool_implementations.search.search_tool import SearchTool
from tests.unit.onyx.legal_review.test_engine import (
    RULE,
    FakeGateway,
    FakeReviewer,
    draft,
    original,
    plan,
    reading,
)


class CanonicalBrokerFixture:
    def __init__(self) -> None:
        self.search_adapter: object | None = None
        self.calls: list[dict[str, JsonValue]] = []
        self.revalidated: list[EvidenceItem] = []
        self.reject_publication = False

    def search_originals(
        self, arguments: dict[str, JsonValue], context: RunContext
    ) -> ToolOutcome:
        context.check_active()
        self.calls.append(arguments)
        return ToolOutcome(
            status=OutcomeStatus.FOUND,
            summary="Authorized canonical full original",
            data={"unmapped_chunk_count": 0, "evidence_truncated": False},
            evidence=[original()],
        )

    def revalidate_evidence(
        self, evidence: list[EvidenceItem], context: RunContext
    ) -> None:
        context.check_active()
        if self.reject_publication:
            raise ValueError("Source publication was revoked")
        assert all(
            item.text == RULE and item.search_doc is not None for item in evidence
        )
        self.revalidated = list(evidence)


@pytest.fixture
def runtime_fixture(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    monkeypatch.setattr(
        runtime,
        "WorkflowPolicy",
        lambda: WorkflowPolicy(
            always_review_research=True, assume_current_corpus=False
        ),
    )
    broker = CanonicalBrokerFixture()
    inventory = SourceInventory(
        source_assessments=[
            SourceAssessment(
                slot="s0001",
                disposition="supports_existing_finding",
                reason=RULE,
                passage_roles=["operative_rule"],
                established_effect=RULE,
                supports=[PassageSupport(citation=1, span_number=1)],
                missing_effect=None,
                content_status="operative_effect_read",
                requested_read=None,
            )
        ]
    )
    gateway = FakeGateway([plan(), inventory, reading(), draft()])
    reviewer = FakeReviewer(["pass", "pass"])
    checkpoints: list[dict[str, JsonValue]] = []
    packets: list[Packet] = []
    emitter = MagicMock(spec=Emitter)
    emitter.emit.side_effect = packets.append
    llm = MagicMock(spec=LLM)
    llm.config = LLMConfig(
        model_provider="vertex_ai",
        model_name="gemini-3.8-flash",
        temperature=0,
        max_input_tokens=200000,
    )
    llm.with_stream_cancellation_check.return_value = llm
    search = MagicMock(spec=SearchTool)
    search.user_selected_filters = None
    search.fork_for_independent_context.return_value = search
    adapter = MagicMock()
    adapter.prepare_batch.return_value = {}
    monkeypatch.setattr(
        runtime,
        "resolve_legal_review_decision",
        lambda _user: DecisionProviderConfig(api_key=SecretStr("fixture-only-key")),
    )
    monkeypatch.setattr(
        runtime, "ensure_trace", lambda *_args, **_kwargs: nullcontext()
    )
    monkeypatch.setattr(runtime, "is_connected", lambda *_args: True)
    monkeypatch.setattr(
        runtime, "bind_pc_corpus_scope", lambda **kwargs: kwargs["filters"]
    )
    monkeypatch.setattr(runtime, "CorpusBroker", lambda *_args: broker)
    monkeypatch.setattr(
        runtime, "build_search_adapter", lambda *_args, **_kwargs: adapter
    )
    monkeypatch.setattr(runtime, "GeminiGateway", lambda **_kwargs: gateway)
    monkeypatch.setattr(runtime, "DecisionsReviewer", lambda **_kwargs: reviewer)
    monkeypatch.setattr(runtime, "build_core_specs", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        runtime,
        "build_corpus_specs",
        lambda *_args, **_kwargs: [
            ToolSpec(
                name="search_corpus",
                description="Discover and hydrate authorized originals",
                parameters={"type": "object", "additionalProperties": True},
                handler=broker.search_originals,
            )
        ],
    )

    def save(**kwargs: object) -> None:
        checkpoints.append(
            copy.deepcopy(cast(dict[str, JsonValue], kwargs["snapshot"]))
        )

    monkeypatch.setattr(runtime, "save_asv3_checkpoint", save)
    user = MagicMock(spec=User)
    user.id = uuid4()
    history = [
        ChatMessageSimple(
            message=f"Olayın verilen {index}. olgusu",
            token_count=8,
            message_type=MessageType.USER,
        )
        for index in range(10)
    ]
    history.append(
        ChatMessageSimple(
            message="Başvuru şartı nedir?", token_count=8, message_type=MessageType.USER
        )
    )
    state = ChatStateContainer()
    arguments = dict(
        emitter=emitter,
        state_container=state,
        simple_chat_history=history,
        tools=[search],
        llm=llm,
        user=user,
        chat_session_id=uuid4(),
        user_message_id=1,
        assistant_message_id=2,
        cache=MagicMock(spec=CacheBackend),
        filters=IndexFilters(access_control_list=[]),
        token_counter=len,
    )
    return SimpleNamespace(
        broker=broker,
        gateway=gateway,
        reviewer=reviewer,
        checkpoints=checkpoints,
        packets=packets,
        state=state,
        llm=llm,
        arguments=arguments,
    )


@pytest.mark.parametrize("include_citations", [False, True])
def test_published_answer_and_canonical_checkpoint_persist_for_both_citation_modes(
    runtime_fixture: SimpleNamespace, include_citations: bool
) -> None:
    fixture = runtime_fixture
    runtime.run_legal_review_loop(
        **fixture.arguments, include_citations=include_citations
    )
    rendered = "".join(
        packet.obj.content
        for packet in fixture.packets
        if isinstance(packet.obj, AgentResponseDelta)
    )
    assert fixture.state.get_answer_tokens() == rendered
    assert RULE in rendered and "Yürürlük sınırı" not in rendered
    assert fixture.state.get_citation_to_doc()[1].document_id == "source-1"
    assert fixture.broker.revalidated[0].text == RULE
    assert len(fixture.broker.calls) == 1
    assert [snapshot["sequence"] for snapshot in fixture.checkpoints] == [1, 2]
    checkpoint = fixture.checkpoints[-1]
    assert checkpoint["publication_status"] == "partial"
    assert checkpoint["asv3_workflow_variant"] == "legal_review"
    assert isinstance(checkpoint["evidence"], dict)
    assert "Olayın verilen 0. olgusu" in fixture.gateway.calls[0][1]["history"]
    assert "Olayın verilen 0. olgusu" in fixture.reviewer.states[0]["history"]
    research_record = fixture.reviewer.states[0]["research_record"]
    assert isinstance(research_record, list) and research_record[0]["status"] == "found"
    citations = [p.obj for p in fixture.packets if isinstance(p.obj, CitationInfo)]
    assert bool(citations) is include_citations
    if citations:
        assert citations[0].preview_url == "/api/asv3/citation/2/1"
    assert not fixture.llm.invoke.called and not fixture.llm.stream.called
    assert isinstance(fixture.packets[-1].obj, ASv3Progress)
    assert fixture.packets[-1].obj.phase == "completed"


def test_publication_revalidation_failure_does_not_emit_answer_or_completed(
    runtime_fixture: SimpleNamespace,
) -> None:
    fixture = runtime_fixture
    fixture.broker.reject_publication = True
    with pytest.raises(ValueError, match="revoked"):
        runtime.run_legal_review_loop(**fixture.arguments)
    assert fixture.state.get_answer_tokens() is None
    assert not any(isinstance(p.obj, AgentResponseStart) for p in fixture.packets)
    assert not any(
        isinstance(p.obj, ASv3Progress) and p.obj.phase == "completed"
        for p in fixture.packets
    )
    assert fixture.checkpoints[-1]["publication_status"] == "unavailable"
    assert fixture.checkpoints[-1]["legal_review"]["answer"] is None


def test_missing_decision_provider_fails_before_work_without_legacy_key_fallback(
    runtime_fixture: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = runtime_fixture
    monkeypatch.setattr(app_configs, "TYPESAFE_API_KEY", "legacy-key-fixture")
    monkeypatch.setattr(runtime, "resolve_legal_review_decision", lambda _user: None)
    with pytest.raises(OnyxError, match="OpenAI gpt-6-luna provider"):
        runtime.run_legal_review_loop(**fixture.arguments)
    assert not fixture.gateway.calls and not fixture.broker.calls
    assert fixture.checkpoints == []
    assert not fixture.llm.invoke.called and not fixture.llm.stream.called


def test_resolved_openai_decision_config_reaches_reviewer_before_any_work(
    runtime_fixture: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = runtime_fixture
    monkeypatch.setattr(
        runtime,
        "resolve_legal_review_decision",
        lambda _user: DecisionProviderConfig(
            api_key=SecretStr("openai-fixture-key"),
            provider_name="Existing OpenAI",
        ),
    )
    factory = MagicMock(return_value=fixture.reviewer)
    monkeypatch.setattr(runtime, "DecisionsReviewer", factory)
    runtime.run_legal_review_loop(**fixture.arguments)
    factory.assert_called_once()
    assert "route" not in factory.call_args.kwargs
    assert factory.call_args.kwargs["api_key"] == "openai-fixture-key"
    assert callable(factory.call_args.kwargs["before_request"])
    assert callable(factory.call_args.kwargs["check_active"])
    assert fixture.checkpoints[-1]["review_provider"] == {
        "route": "openai_decisions",
        "provider_name": "Existing OpenAI",
    }
    assert "jev_provider" not in fixture.checkpoints[-1]
    assert "openai-fixture-key" not in str(fixture.checkpoints)


def test_rejected_review_streams_only_notice_and_retains_unavailable_checkpoint(
    runtime_fixture: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = runtime_fixture
    result = WorkflowResult(
        status="unavailable",
        non_publication_reason="review_rejected",
        final_review=ReviewResult(completed=True),
        gaps=["Final legal review did not pass"],
    )
    monkeypatch.setattr(runtime.LegalReviewEngine, "run", lambda *_args: result)

    runtime.run_legal_review_loop(**fixture.arguments)

    rendered = "".join(
        p.obj.content for p in fixture.packets if isinstance(p.obj, AgentResponseDelta)
    )
    assert "taslak cevap yayımlanmadı" in rendered
    assert fixture.state.get_answer_tokens() == rendered
    assert RULE not in rendered
    assert fixture.state.get_citation_to_doc() == {}
    assert not any(isinstance(p.obj, CitationInfo) for p in fixture.packets)
    assert isinstance(fixture.packets[-1].obj, SectionEnd)
    checkpoint = fixture.checkpoints[-1]
    assert checkpoint["publication_status"] == "unavailable"
    assert checkpoint["legal_review"]["answer"] is None
    assert checkpoint["legal_review"]["non_publication_reason"] == "review_rejected"
    assert checkpoint["progress"][-1]["phase"] == "withheld"
    assert not any(
        isinstance(p.obj, ASv3Progress) and p.obj.status == "completed"
        for p in fixture.packets
    )


def test_provider_failure_is_not_reported_as_completed_review(
    runtime_fixture: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = runtime_fixture
    result = WorkflowResult(
        status="unavailable",
        final_review=ReviewResult(completed=False, failure_reason="provider_timeout"),
        gaps=["provider_timeout"],
    )
    monkeypatch.setattr(runtime.LegalReviewEngine, "run", lambda *_args: result)
    with pytest.raises(OnyxError, match="provider_timeout"):
        runtime.run_legal_review_loop(**fixture.arguments)
    assert fixture.state.get_answer_tokens() is None
    assert not any(isinstance(p.obj, AgentResponseStart) for p in fixture.packets)
    assert fixture.checkpoints[-1]["publication_status"] == "unavailable"


def test_time_exhaustion_after_deadline_streams_notice_without_unchecked_answer(
    runtime_fixture: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    def exhausted(engine: runtime.LegalReviewEngine, *_args: object) -> WorkflowResult:
        engine.context.deadline = 0
        return WorkflowResult(
            status="unavailable", non_publication_reason="time_exhausted"
        )

    monkeypatch.setattr(runtime.LegalReviewEngine, "run", exhausted)
    fixture = runtime_fixture
    runtime.run_legal_review_loop(**fixture.arguments)
    rendered = "".join(
        p.obj.content for p in fixture.packets if isinstance(p.obj, AgentResponseDelta)
    )
    assert "süresi" in rendered
    assert RULE not in rendered
    assert fixture.state.get_answer_tokens() == rendered
    assert isinstance(fixture.packets[-1].obj, SectionEnd)
    assert fixture.checkpoints[-1]["publication_status"] == "unavailable"


def test_rejected_review_notice_honors_stop_before_streaming(
    runtime_fixture: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = runtime_fixture
    result = WorkflowResult(
        status="unavailable",
        non_publication_reason="review_rejected",
        final_review=ReviewResult(completed=True),
    )
    monkeypatch.setattr(runtime.LegalReviewEngine, "run", lambda *_args: result)
    monkeypatch.setattr(runtime, "is_connected", lambda *_args: False)
    with pytest.raises(RunStopped):
        runtime.run_legal_review_loop(**fixture.arguments)
    assert fixture.state.get_answer_tokens() is None
    assert not any(isinstance(p.obj, AgentResponseStart) for p in fixture.packets)


@pytest.mark.parametrize(
    "changes",
    [
        {"status": "partial"},
        {"answer": "Unchecked draft"},
        {"final_review": None},
        {"final_review": {"completed": False}},
    ],
)
def test_review_rejection_cannot_publish_answer_or_hide_incomplete_review(
    changes: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="completed review and no answer"):
        WorkflowResult.model_validate(
            {
                "status": "unavailable",
                "non_publication_reason": "review_rejected",
                "final_review": {"completed": True},
                **changes,
            }
        )


@pytest.mark.parametrize("revoke", [False, True])
def test_expired_generation_delivers_partial_with_citations_after_authorization_check(
    runtime_fixture: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, revoke: bool
) -> None:
    fixture = runtime_fixture
    fixture.broker.reject_publication = revoke

    def exhausted(engine: runtime.LegalReviewEngine, *_args: object) -> WorkflowResult:
        engine.ledger.add([original()], engine.context)
        engine.context.deadline = 0
        return WorkflowResult(
            status="partial",
            publication_mode="limit_reached",
            answer=draft().answer + "\n\nSon kontrol tamamlanamadı.",
        )

    monkeypatch.setattr(runtime.LegalReviewEngine, "run", exhausted)
    if revoke:
        with pytest.raises(ValueError, match="revoked"):
            runtime.run_legal_review_loop(**fixture.arguments)
        assert not any(isinstance(p.obj, AgentResponseDelta) for p in fixture.packets)
        return
    runtime.run_legal_review_loop(**fixture.arguments)
    rendered = fixture.state.get_answer_tokens()
    assert rendered and RULE in rendered and "Kısmi yanıt" in rendered
    assert fixture.broker.revalidated
    assert any(isinstance(p.obj, CitationInfo) for p in fixture.packets)
    assert (
        fixture.checkpoints[-1]["legal_review"]["publication_mode"] == "limit_reached"
    )


def test_unfinished_editor_result_cannot_be_labelled_verified() -> None:
    with pytest.raises(ValueError, match="must remain partial"):
        WorkflowResult(
            status="verified", answer="Text", publication_mode="editor_adjusted"
        )
