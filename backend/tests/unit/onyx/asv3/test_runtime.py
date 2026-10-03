"""Exercise the production harness and citation path with only external boundaries faked."""

import hashlib
import json
import threading
from dataclasses import replace
from datetime import date
from queue import Queue
from typing import Any
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from onyx.asv3 import runtime
from onyx.asv3.assertions import AssertionWitness, assertion_inventory
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.llm_adapter import (
    NeedVerification,
    ResearchModel,
    SourceConditionAuditResult,
    VerificationResult,
)
from onyx.asv3.models import (
    EvidenceItem,
    OutcomeStatus,
    RunContext,
    RunStopped,
    SharedBudget,
    ToolOutcome,
    ToolSpec,
)
from onyx.asv3.source_conditions import answer_hash, complete_condition_review
from onyx.cache.interface import CacheBackend
from onyx.chat.chat_state import ChatStateContainer
from onyx.chat.emitter import Emitter
from onyx.chat.models import ChatMessageSimple
from onyx.configs.constants import DocumentSource, MessageType
from onyx.context.search.models import BaseFilters, IndexFilters
from onyx.db.asv3_corpus import CorpusChunk, CorpusSource
from onyx.db.models import User
from onyx.llm.interfaces import LLM, LLMConfig
from onyx.llm.model_response import (
    ChatCompletionMessageToolCall,
    Choice,
    FunctionCall,
    Message,
    ModelResponse,
)
from onyx.llm.models import ReasoningEffort, UserMessage
from onyx.server.query_and_chat.streaming_models import (
    AgentResponseDelta,
    AgentResponseStart,
    ASv3Progress,
    CitationInfo,
    Packet,
)
from onyx.tracing.flows import LLMFlow

pytestmark = pytest.mark.usefixtures("empty_source_inventory")


def response(
    text: str | None = None, calls: list[tuple[str, dict[str, Any]]] | None = None
) -> ModelResponse:
    return ModelResponse(
        id=str(uuid4()),
        created="0",
        choice=Choice(
            message=Message(
                content=text,
                tool_calls=[
                    ChatCompletionMessageToolCall(
                        id=str(uuid4()),
                        function=FunctionCall(name=name, arguments=json.dumps(args)),
                    )
                    for name, args in calls
                ]
                if calls
                else None,
            )
        ),
    )


def research_update(
    question_count: int = 1, need_ids: tuple[str, ...] = ("comparison",)
) -> tuple[str, dict[str, Any]]:
    return (
        "update_research",
        {
            "needs": [
                {
                    "need_id": need_id,
                    "question_ids": [f"q{i}" for i in range(question_count)],
                    "purpose": "Resolve the requested outcome and conditions",
                    "completion_test": "Operative originals establish the requested outcome and applicable conditions",
                }
                for need_id in need_ids
            ]
        },
    )


def supported_review(
    numbers: list[int],
    question_count: int = 1,
    *,
    draft: str = "Tamir [1], değiştirme [2].",
    quotes: dict[int, str] | None = None,
) -> ModelResponse:
    return response(
        json.dumps(
            {
                "status": "supported",
                "explanation": "Her sorunun koşulları özgün kaynaklarla doğrulandı.",
                "required_conditions": [],
                "missing_conditions": [],
                "evidence_numbers": numbers,
                "safe_to_publish": True,
                "unsupported_claims": [],
                "need_results": [
                    {
                        "need_id": "comparison",
                        "status": "supported",
                        "evidence_numbers": numbers,
                        "missing_conditions": [],
                    }
                ],
                "quotation_checks": [],
                "omitted_material_source_details": [],
                "assertion_results": [
                    {
                        "unit_id": unit["unit_id"],
                        "status": "supported",
                        "witnesses": [
                            {
                                "citation": number,
                                "source_quote": (quotes or {}).get(
                                    number, "Tamir şartları."
                                ),
                            }
                            for number in unit["evidence_numbers"]
                        ],
                        "missing_conditions": [],
                        "explanation": "Scripted source assessment for runtime contracts.",
                    }
                    for unit in assertion_inventory(draft)
                ],
                "question_results": [
                    {
                        "question_id": f"q{index}",
                        "status": "supported",
                        "evidence_numbers": numbers,
                        "missing_conditions": [],
                        "determinations": [
                            {
                                "determination_id": f"q{index}:d0",
                                "status": "supported",
                                "answer_unit_ids": [
                                    unit["unit_id"]
                                    for unit in assertion_inventory(draft)
                                ],
                                "evidence_numbers": numbers,
                                "missing_conditions": [],
                            }
                        ],
                    }
                    for index in range(question_count)
                ],
            }
        )
    )


def request_data(arguments: dict[str, Any]) -> dict[str, Any]:
    message = next(
        item for item in reversed(arguments["prompt"]) if isinstance(item, UserMessage)
    )
    content = message.content
    if isinstance(content, list):
        content = content[0].text
    assert isinstance(content, str)
    return json.loads(content) if content.startswith("{") else {}


def unsafe_review(claim: str, question_count: int = 1) -> ModelResponse:
    return response(
        json.dumps(
            {
                "status": "incomplete",
                "explanation": "Hukuki sonuç özgün kaynakla doğrulanmadı.",
                "required_conditions": [],
                "missing_conditions": ["özgün kaynak"],
                "evidence_numbers": [],
                "safe_to_publish": False,
                "unsupported_claims": [claim],
                "need_results": [
                    {
                        "need_id": "comparison",
                        "status": "incomplete",
                        "evidence_numbers": [],
                        "missing_conditions": ["özgün kaynak"],
                    }
                ],
                "quotation_checks": [],
                "assertion_results": [],
                "omitted_material_source_details": [],
                "question_results": [
                    {
                        "question_id": f"q{index}",
                        "status": "incomplete",
                        "evidence_numbers": [],
                        "missing_conditions": ["özgün kaynak"],
                    }
                    for index in range(question_count)
                ],
            }
        )
    )


class CorpusBoundary:
    def __init__(self) -> None:
        self.sources = [
            CorpusSource(uuid4(), f"Tamir mevzuatı {n}", f"file-{n}") for n in range(2)
        ]
        self.chunks = {
            str(source.id): CorpusChunk(
                id=f"chunk-{n}",
                source_id=source.id,
                text=("Tamir şartları. " * 900) + f"ORIGINAL_TAIL_{n}",
                position=0,
                projection_ordinal=0,
                heading_path=(f"Madde {142 + n}",),
                metadata={"source_links": {"0": f"https://example.test/law-{n}"}},
                validity_start=date(2020, 1, 1),
                validity_end=None,
                status="published",
            )
            for n, source in enumerate(self.sources)
        }
        self.barrier = threading.Barrier(2)
        self.entered = threading.Event()
        self.release = threading.Event()
        self.cancelled = threading.Event()
        self.block = False
        self.active = self.peak = 0
        self.lock = threading.Lock()
        self.revalidated: list[EvidenceItem] = []
        self.search_adapter: Any = None
        self.scope: IndexFilters | None = None

    def page(
        self, source_id: str, _context: RunContext, **_kwargs: Any
    ) -> tuple[CorpusSource, list[CorpusChunk], bool]:
        with self.lock:
            self.active += 1
            self.peak = max(self.active, self.peak)
        self.barrier.wait(3)
        self.entered.set()
        if self.block:
            assert self.release.wait(3)
        with self.lock:
            self.active -= 1
        source = next(source for source in self.sources if str(source.id) == source_id)
        return source, [self.chunks[source_id]], False

    def revalidate_evidence(
        self, evidence: list[EvidenceItem], context: RunContext
    ) -> None:
        context.check_active()
        for item in evidence:
            assert item.text == self.chunks[item.source_id].text
        self.revalidated.extend(evidence)


def setup_run(
    monkeypatch: pytest.MonkeyPatch,
    final: str = "Tamir sonucu [1]; değiştirme sonucu [2].",
) -> tuple[dict[str, Any], CorpusBoundary, MagicMock, list[dict[str, Any]], Queue[Any]]:
    broker = CorpusBoundary()

    def bind_scope(*, user: User, filters: IndexFilters) -> IndexFilters:
        assert user.id is not None
        assert filters.source_type == [DocumentSource.USER_FILE]
        return filters.model_copy(
            deep=True,
            update={
                "tenant_id": "test-tenant",
                "asv3_document_set_id": 91,
                "forced_document_set": ["PC Külliyatı"],
            },
        )

    def create_broker(
        _user: User, scope: IndexFilters, **_kwargs: Any
    ) -> CorpusBoundary:
        broker.scope = scope
        return broker

    # Only the DB binding boundary is faked; its real tenant/ACL behavior has
    # separate DB acceptance tests. The runtime receives the real request shape.
    monkeypatch.setattr(runtime, "bind_pc_corpus_scope", bind_scope)
    monkeypatch.setattr(runtime, "CorpusBroker", create_broker)
    checkpoints: list[dict[str, Any]] = []
    monkeypatch.setattr(
        runtime,
        "save_asv3_checkpoint",
        lambda **kwargs: checkpoints.append(kwargs["snapshot"]),
    )
    monkeypatch.setattr(runtime, "load_asv3_checkpoint", lambda **_kwargs: None)
    # The independent provider boundary has its own contract and runtime-repair tests.
    monkeypatch.setattr(runtime, "complete_condition_review", scripted_condition_review)
    llm = MagicMock(spec=LLM)
    llm.config = LLMConfig(
        model_provider="openai",
        model_name="scripted",
        temperature=0,
        max_input_tokens=100000,
    )
    notifications = {
        phase: [
            "Tamir koşullarını inceliyorum",
            "Tamir ve değişim için uygulanacak koşulları karşılaştırıyorum.",
        ]
        for phase in (
            "started",
            "tools",
            "worker",
            "final",
            "completed",
            "failed",
            "cancelled",
            "native_citation",
            "resume",
            "interrupted",
        )
    }
    notifications["failed"] = [
        "Kaynak doğrulaması tamamlanamadı",
        "Bu sorudaki hukuki sonuçları mevcut kaynaklarla doğrulayamadım.",
    ]
    llm.invoke.side_effect = [
        response(json.dumps({"language": "tr", "notifications": notifications})),
        response(
            calls=[
                (
                    "update_research",
                    {
                        "needs": [
                            {
                                "need_id": "comparison",
                                "question_ids": ["q0"],
                                "purpose": "Compare the requested outcomes",
                                "completion_test": "Applicable original rules and their conditions support both outcomes",
                            }
                        ]
                    },
                )
            ]
            + [
                (
                    "read_source_range",
                    {"source_id": str(source.id), "_need_id": "comparison"},
                )
                for source in broker.sources
            ]
            + [
                (
                    "report_progress",
                    {
                        "title": "Tamir ve değişim",
                        "message": "Aynı makinenin tamiri ile yeni makine gönderilmesine ilişkin hükümleri karşılaştırıyorum.",
                    },
                )
            ]
        ),
        response("Tamir [1], değiştirme [2]."),
        supported_review([1, 2]),
        response(final),
        supported_review([1, 2], draft=final),
    ]
    cache = MagicMock(spec=CacheBackend)
    cache.exists.side_effect = lambda _key: broker.cancelled.is_set()
    user = MagicMock(spec=User)
    user.id = uuid4()
    queue: Queue[Any] = Queue()
    state = ChatStateContainer()
    kwargs = dict(
        emitter=Emitter(queue),
        state_container=state,
        simple_chat_history=[
            ChatMessageSimple(
                message="Garanti kapsamındaki tamir ve yeni makine aynı sonucu verir mi?",
                token_count=30,
                message_type=MessageType.USER,
            )
        ],
        tools=[],
        llm=llm,
        token_counter=len,
        user=user,
        filters=BaseFilters(
            regulatory_chunks_only=True, source_type=[DocumentSource.USER_FILE]
        ),
        document_set_names_override=None,
        user_identity=None,
        chat_session_id=uuid4(),
        user_message_id=1,
        assistant_message_id=2,
        reasoning_effort=ReasoningEffort.AUTO,
        include_citations=True,
        cache=cache,
    )
    return kwargs, broker, llm, checkpoints, queue


def scripted_condition_review(
    review: VerificationResult,
    _model: ResearchModel,
    ledger: EvidenceLedger,
    *,
    answer: str,
    evidence: str,
    **_kwargs: Any,
) -> VerificationResult:
    if not review.safe_to_publish or review.format_error:
        return review
    records = json.loads(evidence)
    call_id = "condition-fixture-" + answer_hash(answer)
    ledger.record_delivery(call_id, LLMFlow.ASV3_CONDITION_REVIEW.value, records)
    inventory_id = "source-fixture-" + answer_hash(answer)
    ledger.record_delivery(inventory_id, LLMFlow.ASV3_SOURCE_INVENTORY.value, records)
    return review.model_copy(
        update={
            "condition_review": SourceConditionAuditResult(
                examined_citations=[row["citation"] for row in records], conditions=[]
            ),
            "condition_review_call_id": call_id,
            "source_inventory_call_id": inventory_id,
            "condition_review_answer_hash": answer_hash(answer),
        }
    )


@pytest.mark.parametrize("provider", ["vertex_ai", "openai", "anthropic"])
@pytest.mark.parametrize("forget_previous", [False, True])
def test_source_condition_audit_repair_preserves_sources_and_selected_provider(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    forget_previous: bool,
) -> None:
    kwargs, broker, llm, checkpoints, queue = setup_run(monkeypatch)
    monkeypatch.setattr(runtime, "complete_condition_review", complete_condition_review)
    llm.config.model_provider = provider
    # Parallel reads assign global citations on arrival, not source-list order.
    for source_id, chunk in broker.chunks.items():
        broker.chunks[source_id] = replace(
            chunk, text=chunk.text + " Onaylı belge sunulmalıdır."
        )
    script = list(llm.invoke.side_effect)
    draft = "Tamir [1], değiştirme [2]."
    final = "Tamir [1], değiştirme [2]. Onaylı belge sunulmalıdır [1]."

    def audit(answer: str, omitted: bool) -> ModelResponse:
        return response(
            json.dumps(
                {
                    "examined_citations": [1, 2],
                    "conditions": [
                        {
                            "witness": {
                                "citation": 1,
                                "source_quote": "Onaylı belge sunulmalıdır.",
                            },
                            "determination_ids": ["q0:d0"],
                            "detail": "Onaylı belge sunulmalıdır.",
                            "applicability": "Tamir koşulunun gerekli belgesi.",
                            "disposition": "omitted" if omitted else "covered",
                            "answer_unit_ids": []
                            if omitted
                            else [assertion_inventory(answer)[0]["unit_id"]],
                        }
                    ],
                }
            )
        )

    forgotten = "Tamir muafiyeti [1], değiştirme rejimi [2]."
    forgotten_review = (
        [
            response(forgotten),
            supported_review([1, 2], draft=forgotten),
            response(json.dumps({"examined_citations": [1, 2], "conditions": []})),
            response(json.dumps({"examined_citations": [1, 2], "conditions": []})),
        ]
        if forget_previous
        else []
    )
    llm.invoke.side_effect = [
        *script[:3],
        supported_review([1, 2], draft=draft),
        audit(draft, True),
        *forgotten_review,
        response(final),
        supported_review([1, 2], draft=final),
        audit(final, False),
    ]
    runtime.run_asv3_loop(**kwargs)
    assert checkpoints[-1]["last_draft"] == final
    assert llm.invoke.call_count == (12 if forget_previous else 8)
    repair = request_data(llm.invoke.call_args_list[5].kwargs)
    assert repair["draft_to_repair"] == draft
    assert (
        repair["publication_gap"]["omitted_material_source_details"][0]["detail"]
        == "Onaylı belge sunulmalıdır."
    )
    saved = checkpoints[-1]
    assert saved["last_draft"] == final and saved["final_publication_gap"] is None
    assert (
        sum(row["call"]["name"] == "read_source_range" for row in saved["receipts"])
        == 2
    )
    assert saved["publication_review"]["condition_review_answer_hash"] == answer_hash(
        final
    )
    emitted = packets(queue)
    assert any(isinstance(packet.obj, CitationInfo) for packet in emitted)


def packets(queue: Queue[Any]) -> list[Packet]:
    result = []
    while not queue.empty():
        _, packet = queue.get_nowait()
        result.append(packet)
    return result


def test_host_support_defect_skips_independent_audit_until_targeted_repair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs, _broker, llm, checkpoints, _queue = setup_run(monkeypatch)
    monkeypatch.setattr(runtime, "complete_condition_review", complete_condition_review)
    script = list(llm.invoke.side_effect)
    bad = "Relevant legislation establishes:\n\nTamir [1], değiştirme [2]."
    final = "Tamir [1], değiştirme [2]."
    bad_review = supported_review([1, 2], draft=bad)
    assert bad_review.choice.message.content is not None
    assessment = json.loads(bad_review.choice.message.content)
    assessment["assertion_results"][0].update(basis="presentation", witnesses=[])
    llm.invoke.side_effect = [
        *script[:2],
        response(bad),
        response(json.dumps(assessment)),
        response(final),
        supported_review([1, 2], draft=final),
        response(json.dumps({"examined_citations": [1, 2], "conditions": []})),
    ]
    runtime.run_asv3_loop(**kwargs)
    assert llm.invoke.call_count == 7
    repair = request_data(llm.invoke.call_args_list[4].kwargs)
    assert (
        repair["publication_gap"]["assertion_gaps"][0]["text"]
        == "Relevant legislation establishes:"
    )
    assert checkpoints[-1]["last_draft"] == final
    assert checkpoints[-1]["final_publication_gap"] is None
    assert checkpoints[-1]["publication_review"][
        "condition_review_answer_hash"
    ] == answer_hash(final)


def test_invalid_publication_review_can_reassess_unchanged_draft_without_source_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs, broker, llm, checkpoints, queue = setup_run(monkeypatch)
    script = list(llm.invoke.side_effect)
    draft = script[2].choice.message.content
    assert draft is not None
    contradictory = json.loads(script[3].choice.message.content)
    contradictory["unsupported_claims"] = ["An assertion has no original support"]
    llm.invoke.side_effect = script[:3] + [
        response(json.dumps({"assertion_results": contradictory["assertion_results"]})),
        response(json.dumps(contradictory)),
        response(draft),
        supported_review([1, 2], draft=draft),
    ]
    runtime.run_asv3_loop(**kwargs)
    assert llm.invoke.call_count == 7
    repair = request_data(llm.invoke.call_args_list[5].kwargs)
    assert repair["draft_to_repair"] == draft
    assert "verification_format_error" in repair["publication_gap"]
    failed = [
        item
        for item in checkpoints
        if (item.get("publication_review") or {}).get("format_error")
    ]
    assert failed and all(item["last_draft"] == draft for item in failed)
    assert all(item["draft_approval"] is None for item in failed)
    for index in (3, 6):
        payload = request_data(llm.invoke.call_args_list[index].kwargs)
        assert payload["claim"] == draft
        evidence = json.loads(payload["evidence"])
        assert {item["text"] for item in evidence} == {
            chunk.text for chunk in broker.chunks.values()
        }
        assert not any(item["truncated"] for item in evidence)
        assert "available_evidence" not in payload
    receipts = checkpoints[-1]["receipts"]
    assert sum(row["call"]["name"] == "read_source_range" for row in receipts) == 2
    assert len(checkpoints[-1]["evidence"]["records"]) == 2
    assert checkpoints[-1]["last_draft"] == draft
    assert checkpoints[-1]["publication_stop_reason"] == "verified_draft_published"
    assert (
        len(
            [
                packet
                for packet in packets(queue)
                if isinstance(packet.obj, CitationInfo)
            ]
        )
        == 2
    )


def test_runtime_parameter_review_keeps_publication_checks_without_regeneration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs, _broker, llm, checkpoints, queue = setup_run(monkeypatch)
    script = list(llm.invoke.side_effect)
    assessment = json.loads(script[3].choice.message.content)
    script[3] = response(json.dumps({"parameter": assessment}))
    llm.invoke.side_effect = script
    runtime.run_asv3_loop(**kwargs)
    assert llm.invoke.call_count == 4
    assert checkpoints[-1]["publication_stop_reason"] == "verified_draft_published"
    assert checkpoints[-1]["publication_review"]["format_error"] is None
    assert checkpoints[-1]["last_draft"] == script[2].choice.message.content
    assert (
        len(
            [
                packet
                for packet in packets(queue)
                if isinstance(packet.obj, CitationInfo)
            ]
        )
        == 2
    )


@pytest.mark.parametrize(
    "consent,intent", [(False, False), (False, True), (True, False), (True, True)]
)
def test_runtime_external_capabilities_require_both_permission_and_user_intent(
    monkeypatch: pytest.MonkeyPatch, consent: bool, intent: bool
) -> None:
    kwargs, _broker, llm, _checkpoints, _queue = setup_run(monkeypatch)
    script = list(llm.invoke.side_effect)
    profile = json.loads(script[0].choice.message.content)
    profile["external_requested"] = intent
    script[0] = response(json.dumps(profile))
    llm.invoke.side_effect = script
    monkeypatch.setattr(
        runtime,
        "build_external_specs",
        lambda *_args, **_kwargs: [
            ToolSpec(
                name="external_read_public_source",
                description="Authorized external read",
                parameters={"type": "object", "properties": {}},
                external=True,
                handler=lambda _args, _ctx: ToolOutcome(
                    status=OutcomeStatus.FOUND, summary="Read"
                ),
            )
        ],
    )
    runtime.run_asv3_loop(**kwargs, allow_external=consent)
    names = {
        tool["function"]["name"]
        for tool in llm.invoke.call_args_list[1].kwargs["tools"]
    }
    assert ("external_read_public_source" in names) is (consent and intent)


def test_resume_reuses_saved_question_language_without_reclassifying(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs, _broker, llm, checkpoints, queue = setup_run(monkeypatch)
    runtime.run_asv3_loop(**kwargs)
    previous = checkpoints[-1]
    packets(queue)
    monkeypatch.setattr(runtime, "load_asv3_checkpoint", lambda **_kwargs: previous)
    llm.reset_mock()
    llm.invoke.side_effect = [
        response("Tamir [1], değiştirme [2]."),
        supported_review([1, 2]),
        response("Tamir sonucu [1]; değiştirme sonucu [2]."),
        supported_review([1, 2], draft="Tamir sonucu [1]; değiştirme sonucu [2]."),
    ]
    runtime.run_asv3_loop(**kwargs, resume_message_id=2)
    assert llm.invoke.call_count == 2
    assert all(
        packet.obj.language == "tr"
        for packet in packets(queue)
        if isinstance(packet.obj, ASv3Progress)
    )


def test_explicit_continuation_renews_execution_but_keeps_original_memory_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs, _broker, llm, checkpoints, queue = setup_run(monkeypatch)
    runtime.run_asv3_loop(**kwargs)
    previous = checkpoints[-1]
    previous["budget"]["tools"] = 64
    previous["budget"]["decisions"] = 41
    retained_bytes = previous["budget"]["evidence_bytes"]
    packets(queue)
    monkeypatch.setattr(runtime, "load_asv3_checkpoint", lambda **_kwargs: previous)
    llm.reset_mock()
    llm.invoke.side_effect = [
        response("Tamir [1], değiştirme [2]."),
        supported_review([1, 2]),
    ]
    runtime.run_asv3_loop(**kwargs, resume_message_id=2)
    saved = checkpoints[-1]
    assert saved["publication_status"] == "found"
    assert saved["budget"]["decisions"] < 41
    assert saved["budget"]["evidence_bytes"] == retained_bytes
    assert saved["continuation_usage"]["message_id"] == 2
    assert saved["continuation_usage"]["budget"]["decisions"] == 41
    assert saved["continuation_usage"]["budget"]["tools"] == 64
    assert saved["evidence"]["records"] == previous["evidence"]["records"]


@pytest.mark.parametrize(
    "provider,model_name",
    [
        ("vertex_ai", "gemini-3.8-flash"),
        ("openai", "scripted"),
        ("anthropic", "scripted"),
    ],
)
def test_runtime_parallel_sources_full_original_review_and_final_citations(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    model_name: str,
) -> None:
    kwargs, broker, llm, checkpoints, queue = setup_run(monkeypatch)
    scripted = iter(list(llm.invoke.side_effect))
    invocation = 0

    def answer_from_supplied_evidence(**arguments: Any) -> ModelResponse:
        nonlocal invocation
        current = invocation
        invocation += 1
        scheduled = next(scripted)
        if current == 3:
            data = request_data(arguments)
            with queue.mutex:
                updates = [
                    packet.obj
                    for _, packet in queue.queue
                    if isinstance(packet.obj, ASv3Progress)
                ]
            assert updates[-1].phase == "final" and updates[-1].language == "tr"
            originals = {row["citation"]: row for row in json.loads(data["evidence"])}
            review = json.loads(
                supported_review([1, 2], draft=data["claim"]).choice.message.content
            )
            for item in review["assertion_results"]:
                item.pop("explanation")
                for witness in item["witnesses"]:
                    witness.pop("source_quote")
                    witness["witness_id"] = originals[witness["citation"]][
                        "witness_spans"
                    ][0]["witness_id"]
            return response(json.dumps(review))
        if current not in (2, 4):
            return scheduled
        evidence = request_data(arguments)["evidence"]
        if isinstance(evidence, str):
            evidence = json.loads(evidence)
        numbers = {item["chunk_id"]: item["citation"] for item in evidence}
        return response(
            f"Tamir sonucu [{numbers['chunk-0']}]; değiştirme sonucu [{numbers['chunk-1']}]."
        )

    llm.invoke.side_effect = answer_from_supplied_evidence
    llm.config = LLMConfig(
        model_provider=provider,
        model_name=model_name,
        temperature=0,
        max_input_tokens=100000,
    )
    kwargs["custom_agent_prompt"] = "Her soruya ayrı ve kısa uygulama sonucu ver."
    runtime.run_asv3_loop(**kwargs)
    assert llm.invoke.call_count == 4
    for call in llm.invoke.call_args_list[1:3]:
        assert (
            request_data(call.kwargs)["assistant_instructions"]
            == kwargs["custom_agent_prompt"]
        )
    from onyx.prompts.asv3.research import FINAL_PROMPT

    assert all(
        call.kwargs["prompt"][0].content != FINAL_PROMPT
        for call in llm.invoke.call_args_list
    )
    assert checkpoints[-1]["stop_reason"] == "verified_draft"
    assert checkpoints[-1]["publication_stop_reason"] == "verified_draft_published"
    assert checkpoints[-1]["final_publication_gap"] is None
    approval = checkpoints[-1]["draft_approval"]
    assert approval["verification_call_id"]
    approved_text = request_data(llm.invoke.call_args_list[-1].kwargs)["claim"]
    assert approval["text_hash"] == hashlib.sha256(approved_text.encode()).hexdigest()
    assert checkpoints[-1]["last_draft"] == approved_text
    assert broker.peak == 2
    assert {item.chunk_id for item in broker.revalidated} == {"chunk-0", "chunk-1"}
    for call in llm.invoke.call_args_list[-1:]:
        data = json.loads(call.kwargs["prompt"][1].content)
        assert "available_evidence" not in data
        evidence = json.loads(data["evidence"])
        assert {item["text"] for item in evidence} == {
            chunk.text for chunk in broker.chunks.values()
        }
        assert not any(item["truncated"] for item in evidence)
    output = packets(queue)
    for packet in output:
        if isinstance(packet.obj, CitationInfo):
            assert packet.obj.preview_url == (
                f"/api/asv3/citation/{kwargs['assistant_message_id']}/{packet.obj.citation_number}"
            )
    narration = [
        packet.obj for packet in output if isinstance(packet.obj, ASv3Progress)
    ]
    assert all(event.language == "tr" for event in narration)
    assert any("Aynı makinenin" in (event.message or "") for event in narration)
    assert not any(
        "read_source" in (event.message or "")
        or "report_progress" in (event.message or "")
        for event in narration
    )
    completed = [
        i
        for i, packet in enumerate(output)
        if isinstance(packet.obj, ASv3Progress)
        and packet.obj.status == "completed"
        and packet.obj.task_id is None
    ]
    assert len(completed) == 1
    assert completed[0] > max(
        i
        for i, packet in enumerate(output)
        if isinstance(packet.obj, (AgentResponseDelta, CitationInfo))
    )
    state = kwargs["state_container"]
    assert {doc.link for doc in state.citation_to_doc.values()} == {
        "https://example.test/law-0",
        "https://example.test/law-1",
    }
    for number, doc in state.citation_to_doc.items():
        assert f"[[{number}]]({doc.link})" in state.answer_tokens
        expected_claim = (
            "Tamir sonucu" if doc.link.endswith("law-0") else "değiştirme sonucu"
        )
        assert f"{expected_claim} [[{number}]]({doc.link})" in state.answer_tokens
    assert len(state.citation_to_doc) == 2
    assert checkpoints[-1]["evidence"]["included"] == [1, 2]
    turn = checkpoints[-1]["turns"][0]
    calls = turn["assistant"]["tool_calls"]
    assert [result["tool_call_id"] for result in turn["results"]] == [
        call["id"] for call in calls
    ]
    assert [call["function"]["name"] for call in calls] == [
        "update_research",
        "read_source_range",
        "read_source_range",
        "report_progress",
    ]
    assert llm.config.model_provider == provider and llm.config.model_name == model_name


def reserve_final_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise rejection when the run has no post-publication repair allocation."""

    class FinalOnlyContext(RunContext):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(
                budget=SharedBudget(max_decisions=11, final_decision_reserve=4),
                **kwargs,
            )

    monkeypatch.setattr(runtime, "RunContext", FinalOnlyContext)


def incomplete_script(llm: MagicMock) -> list[ModelResponse | Exception]:
    script = list(llm.invoke.side_effect)
    return [
        *script[:2],
        script[2],
        unsafe_review("Missing condition"),
        script[2],
        script[2],
        *script[4:],
    ]


def test_numbered_questions_are_verified_separately_and_approved_details_are_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from onyx.prompts.asv3.research import FINAL_PROMPT

    kwargs, _broker, llm, checkpoints, _queue = setup_run(monkeypatch)
    request = (
        "The goods are unused and permission is absent.\n\n"
        "1. What is the result and its prerequisites?\n"
        "2. Does replacement change the result?\n"
        "3. Which documents and subsequent steps are required?"
    )
    kwargs["simple_chat_history"][0].message = request
    draft = (
        "Prerequisites and the actual outcome [1].\n"
        "Replacement alternative [2].\n"
        "Application, documents and later settlement [1, 2]."
    )
    script = list(llm.invoke.side_effect)
    script[1] = response(
        calls=[research_update(3)]
        + [
            (call.function.name, json.loads(call.function.arguments))
            for call in script[1].choice.message.tool_calls or []
            if call.function.name != "update_research"
        ]
    )
    llm.invoke.side_effect = [
        *script[:2],
        response(draft),
        supported_review([1, 2], question_count=3, draft=draft),
    ]
    runtime.run_asv3_loop(**kwargs)
    assert llm.invoke.call_count == 4
    assert checkpoints[-1]["last_draft"] == draft
    assert checkpoints[-1]["publication_status"] == "found"
    review_data = request_data(llm.invoke.call_args.kwargs)
    assert review_data["scenario"] == request
    assert [item["question_id"] for item in review_data["questions"]] == [
        "q0",
        "q1",
        "q2",
    ]
    assert [item["question"] for item in review_data["questions"]] == [
        "What is the result and its prerequisites?",
        "Does replacement change the result?",
        "Which documents and subsequent steps are required?",
    ]
    assert all(
        call.kwargs["prompt"][0].content != FINAL_PROMPT
        for call in llm.invoke.call_args_list
    )


def test_partial_rewrite_verifier_receives_draft_details_and_their_originals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reserve_final_only(monkeypatch)
    from onyx.prompts.asv3.research import FINAL_PROMPT

    kwargs, _broker, llm, _checkpoints, _queue = setup_run(
        monkeypatch, final="Only the headline remains [1]."
    )
    script = incomplete_script(llm)
    draft = "Outcome [1]; application documents and later settlement [2]."
    for index in (2, 4, 5):
        script[index] = response(draft)
    script[-1] = unsafe_review("Missing the supported settlement detail")
    llm.invoke.side_effect = script
    runtime.run_asv3_loop(**kwargs)
    final_review = request_data(llm.invoke.call_args.kwargs)
    assert final_review["claim"] == "Only the headline remains [1]."
    reference = final_review["preservation_reference"]
    assert reference["draft"] == draft
    previous_review = script[3]
    assert isinstance(previous_review, ModelResponse)
    assert isinstance(previous_review.choice.message.content, str)
    assert reference[
        "previous_review"
    ] == runtime.VerificationResult.model_validate_json(
        previous_review.choice.message.content
    ).model_dump(mode="json")
    synthesis = next(
        request_data(call.kwargs)
        for call in llm.invoke.call_args_list
        if call.kwargs["prompt"][0].content == FINAL_PROMPT
    )
    assert synthesis["draft"] == draft
    assert synthesis["publication_gap"]["status"] == "partial"
    assert "Missing condition" in json.dumps(synthesis["publication_gap"])
    assert synthesis["authority_obligations"] == []
    evidence = json.loads(final_review["evidence"])
    assert {item["citation"] for item in evidence} == {1, 2}
    assert all(item["truncated"] is False for item in evidence)


def test_supported_finalization_after_research_stop_reports_answer_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reserve_final_only(monkeypatch)
    kwargs, _broker, llm, checkpoints, queue = setup_run(monkeypatch)
    llm.invoke.side_effect = incomplete_script(llm)
    runtime.run_asv3_loop(**kwargs)
    saved = checkpoints[-1]
    assert saved["publication_review"]["status"] == "supported"
    assert saved["final_publication_gap"] is None
    assert saved["publication_status"] == "found"
    assert saved["publication_stop_reason"] == "verified_draft_published"
    assert saved["stop_reason"] != "verified_draft"
    assert kwargs["state_container"].answer_tokens
    statuses = [
        packet.obj.status
        for packet in packets(queue)
        if isinstance(packet.obj, ASv3Progress)
    ]
    assert statuses[-1] == "completed"
    assert "failed" not in statuses


def test_explicit_unresolved_question_does_not_become_complete_after_finalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reserve_final_only(monkeypatch)
    kwargs, _broker, llm, checkpoints, _queue = setup_run(monkeypatch)
    script = incomplete_script(llm)
    review = script[-1]
    assert isinstance(review, ModelResponse)
    assert isinstance(review.choice.message.content, str)
    partial = json.loads(review.choice.message.content)
    partial["question_results"][0].update(
        status="incomplete", missing_conditions=["Missing the applicable exception"]
    )
    script[-1] = response(json.dumps(partial))
    llm.invoke.side_effect = script
    runtime.run_asv3_loop(**kwargs)
    assert checkpoints[-1]["publication_status"] == "partial"
    assert checkpoints[-1]["publication_stop_reason"] != "verified_draft_published"


def test_precise_source_condition_gap_does_not_replace_partial_answer_with_failure_notice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.unit.onyx.asv3.test_safe_partial_conditions import partial_case

    class FinalOnlyContext(RunContext):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(
                budget=SharedBudget(max_decisions=5, final_decision_reserve=3),
                **kwargs,
            )

    monkeypatch.setattr(runtime, "RunContext", FinalOnlyContext)
    kwargs, _broker, llm, checkpoints, queue = setup_run(monkeypatch)
    monkeypatch.setattr(runtime, "complete_condition_review", complete_condition_review)
    _ledger, _context, case_state, answer, review, condition = partial_case()
    request = "\n".join(
        f"{index + 1}. {question}"
        for index, question in enumerate(case_state.questions)
    )
    kwargs["simple_chat_history"] = [
        ChatMessageSimple(
            message=request, token_count=30, message_type=MessageType.USER
        )
    ]
    script = list(llm.invoke.side_effect)
    tool_response = script[1]
    assert isinstance(tool_response, ModelResponse)
    for call in tool_response.choice.message.tool_calls or []:
        if call.function.name == "update_research":
            arguments = json.loads(call.function.arguments)
            arguments["needs"][0]["question_ids"] = ["q0", "q1"]
            call.function.arguments = json.dumps(arguments)
    review.need_results = [
        NeedVerification(
            need_id="comparison",
            status="incomplete",
            evidence_numbers=[1],
            missing_conditions=list(review.missing_conditions),
        )
    ]
    review.assertion_results[0].witnesses = [
        AssertionWitness(citation=1, source_quote="Tamir şartları.")
    ]
    condition.witness = AssertionWitness(citation=2, source_quote="Tamir şartları.")
    llm.invoke.side_effect = [
        *script[:2],
        response(answer),
        response(review.model_dump_json()),
        response(
            SourceConditionAuditResult(
                examined_citations=[1, 2], conditions=[condition]
            ).model_dump_json()
        ),
    ]
    runtime.run_asv3_loop(**kwargs)
    saved = checkpoints[-1]
    assert llm.invoke.call_count == 5
    assert saved["publication_status"] == "partial"
    assert saved["publication_stop_reason"] != "publication_guard_rejected"
    assert saved["final_publication_gap"] is None
    assert "The operation is permitted" in kwargs["state_container"].answer_tokens
    assert review.missing_conditions[0] in kwargs["state_container"].answer_tokens
    assert any(isinstance(packet.obj, CitationInfo) for packet in packets(queue))
    conditions = saved["research_state"]["source_conditions"]["conditions"]
    assert len(conditions) == 1


@pytest.mark.parametrize("omitted_condition", [False, True])
def test_model_selected_partial_submission_keeps_exact_candidate_and_source_guards(
    monkeypatch: pytest.MonkeyPatch, omitted_condition: bool
) -> None:
    from onyx.prompts.asv3.research import FINAL_PROMPT
    from tests.unit.onyx.asv3.test_safe_partial_conditions import partial_case

    if omitted_condition:

        class NoRepairReserveContext(RunContext):
            def __init__(self, **kwargs: Any) -> None:
                super().__init__(
                    budget=SharedBudget(max_decisions=8, final_decision_reserve=5),
                    **kwargs,
                )

        monkeypatch.setattr(runtime, "RunContext", NoRepairReserveContext)
    kwargs, _broker, llm, checkpoints, queue = setup_run(monkeypatch)
    monkeypatch.setattr(runtime, "complete_condition_review", complete_condition_review)
    _ledger, _context, case_state, answer, review, condition = partial_case()
    kwargs["simple_chat_history"] = [
        ChatMessageSimple(
            message="\n".join(
                f"{index + 1}. {question}"
                for index, question in enumerate(case_state.questions)
            ),
            token_count=30,
            message_type=MessageType.USER,
        )
    ]
    script = list(llm.invoke.side_effect)
    for call in script[1].choice.message.tool_calls or []:
        if call.function.name == "update_research":
            arguments = json.loads(call.function.arguments)
            arguments["needs"][0]["question_ids"] = ["q0", "q1"]
            call.function.arguments = json.dumps(arguments)
    review.need_results = [
        NeedVerification(
            need_id="comparison",
            status="incomplete",
            evidence_numbers=[1],
            missing_conditions=list(review.missing_conditions),
        )
    ]
    review.assertion_results[0].witnesses = [
        AssertionWitness(citation=1, source_quote="Tamir şartları.")
    ]
    condition.witness = AssertionWitness(citation=2, source_quote="Tamir şartları.")
    if omitted_condition:
        condition.disposition = "omitted"
        condition.answer_unit_ids = []
    llm.invoke.side_effect = [
        *script[:2],
        response(
            calls=[
                (
                    "submit_partial_answer",
                    {
                        "answer": answer,
                        "reason": "The reference's exact effect cannot be established with the admitted evidence.",
                    },
                )
            ]
        ),
        response(review.model_dump_json()),
        response(
            SourceConditionAuditResult(
                examined_citations=[1, 2], conditions=[condition]
            ).model_dump_json()
        ),
    ]
    runtime.run_asv3_loop(**kwargs)
    saved = checkpoints[-1]
    assert llm.invoke.call_count == 5
    assert saved["last_draft"] == answer
    assert saved["stop_reason"] == "model_requested_partial_publication"
    assert saved["publication_status"] == "partial"
    assert all(
        call.kwargs["prompt"][0].content != FINAL_PROMPT
        for call in llm.invoke.call_args_list
    )
    verification = request_data(llm.invoke.call_args_list[3].kwargs)
    assert verification["publication_mode"] == "partial_allowed"
    assert verification["claim"] == answer
    emitted = packets(queue)
    if omitted_condition:
        assert saved["final_publication_gap"] is not None
        assert saved["publication_stop_reason"] == "publication_guard_rejected"
        assert (
            "The operation is permitted" not in kwargs["state_container"].answer_tokens
        )
    else:
        assert saved["final_publication_gap"] is None
        assert saved["publication_stop_reason"] == "verified_partial_published"
        assert "The operation is permitted" in kwargs["state_container"].answer_tokens
        assert review.missing_conditions[0] in kwargs["state_container"].answer_tokens
        assert any(isinstance(packet.obj, CitationInfo) for packet in emitted)
    statuses = [
        packet.obj.status for packet in emitted if isinstance(packet.obj, ASv3Progress)
    ]
    assert statuses[-1] == ("failed" if omitted_condition else "completed")


def test_failed_final_verification_preserves_successful_final_original_delivery_without_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from onyx.tracing.flows import LLMFlow

    kwargs, broker, llm, checkpoints, queue = setup_run(monkeypatch)
    script = incomplete_script(llm)
    script[-1] = RuntimeError("Final verifier failed after successful synthesis")
    llm.invoke.side_effect = script
    with pytest.raises(RuntimeError, match="Final verifier failed"):
        runtime.run_asv3_loop(**kwargs)
    saved = checkpoints[-1]
    assert saved["publication_status"] == "partial"
    final_deliveries = [
        delivery
        for delivery in saved["evidence"]["deliveries"]
        if delivery["flow"] == LLMFlow.ASV3_FINAL.value
    ]
    assert len(final_deliveries) == 1
    records = final_deliveries[0]["records"]
    assert {record["chunk_id"] for record in records} == {
        chunk.id for chunk in broker.chunks.values()
    }
    assert all(record["complete"] for record in records)
    originals = {
        record["citation"]: record["item"] for record in saved["evidence"]["records"]
    }
    assert all(
        record["text_hash"] == originals[record["citation"]]["text_hash"]
        for record in records
    )
    assert not kwargs["state_container"].answer_tokens
    assert not any(
        isinstance(packet.obj, (AgentResponseDelta, CitationInfo))
        for packet in packets(queue)
    )


def test_final_evidence_keeps_uncited_same_source_exception_without_unrelated_sources() -> (
    None
):
    from onyx.asv3.evidence import EvidenceLedger

    ledger = EvidenceLedger()
    ledger.add(
        [
            EvidenceItem(
                source_id="law",
                chunk_id="rule",
                text="Full rule and its first condition.",
            ),
            EvidenceItem(
                source_id="law",
                chunk_id="exception",
                text="Decisive exception and additional condition.",
            ),
            EvidenceItem(
                source_id="unrelated", chunk_id="other", text="Unrelated source."
            ),
        ],
        RunContext(),
    )
    supplied = json.loads(runtime._evidence_record(ledger, "Apply the rule [1]."))
    assert [record["citation"] for record in supplied] == [1, 2]
    assert supplied[1]["text"] == "Decisive exception and additional condition."
    assert all(record["truncated"] is False for record in supplied)


def test_runtime_recovers_uncited_governing_source_without_losing_special_procedure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs, broker, llm, checkpoints, _queue = setup_run(monkeypatch)
    kwargs["simple_chat_history"][
        0
    ].message = (
        "Bir işletmenin faaliyete başlamasının koşulları ve izin işlemleri nelerdir?"
    )
    for source, kind, text in zip(
        broker.sources,
        ("genelge", "kanun"),
        (
            "Yetkili özel usul: elektronik bildirim kabul edilir.",
            "Kanuni koşul: izin gerekir.",
        ),
    ):
        key = str(source.id)
        broker.chunks[key] = replace(
            broker.chunks[key],
            text=text + (" Özgün hüküm devamı." * 200),
            metadata={
                **broker.chunks[key].metadata,
                "document_type": kind,
                "title": f"Faaliyete başlama {kind}",
            },
        )
    scripts = list(llm.invoke.side_effect)
    citations: dict[str, int] = {}

    def first_draft(**arguments: Any) -> ModelResponse:
        data = request_data(arguments)
        citations.update(
            {item["document_type"]: item["citation"] for item in data["evidence"]}
        )
        return response(f"Elektronik bildirim kabul edilir [{citations['genelge']}].")

    def first_review(**arguments: Any) -> ModelResponse:
        data = request_data(arguments)
        originals = json.loads(data["evidence"])
        assert {item["citation"] for item in originals} == set(citations.values())
        assert all(item["truncated"] is False for item in originals)
        assert "available_evidence" not in data
        assert {item["metadata"]["document_type"] for item in originals} == {
            "kanun",
            "genelge",
        }
        assert any(item["citation"] == citations["kanun"] for item in originals)
        return response(
            json.dumps(
                {
                    "status": "incomplete",
                    "explanation": "Kanuni dayanağın özgün hükmü ve atfı eksik.",
                    "need_results": [
                        {
                            "need_id": "comparison",
                            "status": "incomplete",
                            "evidence_numbers": [],
                            "missing_conditions": ["kanuni dayanak"],
                        }
                    ],
                    "quotation_checks": [],
                    "assertion_results": [],
                    "omitted_material_source_details": [],
                    "required_conditions": [],
                    "missing_conditions": ["kanuni dayanağın özgün hükmü ve atfı"],
                    "evidence_numbers": [citations["genelge"]],
                    "safe_to_publish": False,
                    "unsupported_claims": [],
                    "question_results": [
                        {
                            "question_id": "q0",
                            "status": "incomplete",
                            "evidence_numbers": [citations["genelge"]],
                            "missing_conditions": ["kanuni dayanak"],
                        }
                    ],
                }
            )
        )

    def reopen(**arguments: Any) -> ModelResponse:
        assert "kanuni dayanak" in str(request_data(arguments))
        return response(
            calls=[
                (
                    "read_evidence",
                    {"citation": citations["kanun"], "_need_id": "comparison"},
                )
            ]
        )

    final = ""

    def corrected_draft(**_arguments: Any) -> ModelResponse:
        nonlocal final
        final = (
            f"İzin gerekir [{citations['kanun']}]. "
            f"Yetkili özel usulde elektronik bildirim kabul edilir [{citations['genelge']}]."
        )
        return response(final)

    def corrected_review(**arguments: Any) -> ModelResponse:
        data = request_data(arguments)
        originals = json.loads(data["evidence"])
        assert {item["citation"] for item in originals} == set(citations.values())
        for item in originals:
            stored = next(
                chunk
                for chunk in broker.chunks.values()
                if chunk.id == item["chunk_id"]
            )
            assert item["text"] == stored.text and item["truncated"] is False
        return supported_review(
            sorted(citations.values()),
            draft=data["claim"],
            quotes={item["citation"]: item["text"][:160] for item in originals},
        )

    stages = iter(
        [
            scripts[0],
            scripts[1],
            first_draft,
            first_review,
            reopen,
            corrected_draft,
            corrected_review,
        ]
    )

    def invoke(**arguments: Any) -> ModelResponse:
        stage = next(stages)
        return stage(**arguments) if callable(stage) else stage

    llm.invoke.side_effect = invoke
    runtime.run_asv3_loop(**kwargs)
    assert llm.invoke.call_count == 7
    assert checkpoints[-1]["publication_status"] == "found"
    assert checkpoints[-1]["publication_stop_reason"] == "verified_draft_published"
    assert checkpoints[-1]["last_draft"] == final
    answer = kwargs["state_container"].answer_tokens
    assert "İzin gerekir" in answer and "elektronik bildirim" in answer
    assert len(kwargs["state_container"].citation_to_doc) == 2


def test_runtime_follows_named_statutory_basis_before_accepting_a_supported_draft(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs, broker, llm, checkpoints, _queue = setup_run(monkeypatch)
    for source, kind, title, article, text in zip(
        broker.sources,
        ("genelge", "kanun"),
        ("Uygulama Genelgesi", "8917 sayılı Faaliyet Kanunu"),
        ("3", "27"),
        (
            "8917 sayılı Faaliyet Kanunu’nun 27 nci maddesi uyarınca izin gerekir.",
            "İzin gerekir. Elektronik bildirim kabul edilebilir.",
        ),
    ):
        key = str(source.id)
        broker.chunks[key] = replace(
            broker.chunks[key],
            text=text,
            metadata={
                "document_type": kind,
                "title": title,
                "article_no": article,
                "heading_path": [title],
            },
            heading_path=(title, f"MADDE {article}"),
        )
    scripts = list(llm.invoke.side_effect)
    numbers: dict[str, int] = {}

    def initial_draft(**arguments: Any) -> ModelResponse:
        data = request_data(arguments)
        numbers.update(
            {row["document_type"]: row["citation"] for row in data["evidence"]}
        )
        return response(
            f"8917 sayılı Faaliyet Kanunu’nun 27 nci maddesi uyarınca izin gerekir [{numbers['genelge']}]."
        )

    def follow_missing(**arguments: Any) -> ModelResponse:
        data = request_data(arguments)
        assert "unresolved_original" in json.dumps(data)
        assert "8917" in json.dumps(data)
        return response(
            calls=[
                (
                    "read_evidence",
                    {"citation": numbers["kanun"], "_need_id": "comparison"},
                )
            ]
        )

    final = ""

    def complete_draft(**_arguments: Any) -> ModelResponse:
        nonlocal final
        final = f"8917 sayılı Faaliyet Kanunu’nun 27 nci maddesi uyarınca izin gerekir [{numbers['kanun']}]."
        return response(final)

    def review(**arguments: Any) -> ModelResponse:
        data = request_data(arguments)
        assert data["claim"] == final
        assert data["authority_obligations"][0]["status"] == "original_cited"
        assert data["authority_obligations"][0]["cited_original_evidence"] == [
            numbers["kanun"]
        ]
        return supported_review(
            [numbers["kanun"]],
            draft=final,
            quotes={numbers["kanun"]: "İzin gerekir."},
        )

    stages = iter(
        [scripts[0], scripts[1], initial_draft, follow_missing, complete_draft, review]
    )

    def invoke(**arguments: Any) -> ModelResponse:
        stage = next(stages)
        return stage(**arguments) if callable(stage) else stage

    llm.invoke.side_effect = invoke
    runtime.run_asv3_loop(**kwargs)
    assert llm.invoke.call_count == 6
    assert checkpoints[-1]["publication_stop_reason"] == "verified_draft_published"
    assert checkpoints[-1]["last_draft"] == final
    assert "8917" in kwargs["state_container"].answer_tokens
    assert any(
        receipt["call"]["name"] == "finalization_status"
        for receipt in checkpoints[-1]["receipts"]
    )


def test_runtime_drops_unrecorded_citation_and_publishes_only_localized_gap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reserve_final_only(monkeypatch)
    kwargs, _broker, llm, checkpoints, queue = setup_run(monkeypatch, "Kaynak [999].")
    llm.invoke.side_effect = incomplete_script(llm)
    runtime.run_asv3_loop(**kwargs)
    output = packets(queue)
    assert not any(isinstance(packet.obj, CitationInfo) for packet in output)
    assert all(
        "999" not in packet.obj.content
        for packet in output
        if isinstance(packet.obj, AgentResponseDelta)
    )
    assert [
        packet.obj.status for packet in output if isinstance(packet.obj, ASv3Progress)
    ][-1] == "failed"
    assert (
        kwargs["state_container"].answer_tokens
        == "Bu sorudaki hukuki sonuçları mevcut kaynaklarla doğrulayamadım."
    )
    assert checkpoints[-1]["publication_status"] == "partial"


def test_runtime_supported_draft_does_not_authorize_unsafe_final_wording(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reserve_final_only(monkeypatch)
    unsafe_final = "Her bedelsiz yeni makine vergiden muaftır [1]."
    kwargs, _broker, llm, checkpoints, queue = setup_run(monkeypatch, unsafe_final)
    script = incomplete_script(llm)
    script[-1] = unsafe_review(unsafe_final)
    llm.invoke.side_effect = script
    runtime.run_asv3_loop(**kwargs)
    assert request_data(llm.invoke.call_args_list[-1].kwargs)["claim"] == unsafe_final
    assert checkpoints[-1]["publication_review"]["unsupported_claims"] == [unsafe_final]
    assert checkpoints[-1]["publication_status"] == "partial"
    assert kwargs["state_container"].answer_tokens == (
        "Bu sorudaki hukuki sonuçları mevcut kaynaklarla doğrulayamadım."
    )
    assert not any(
        isinstance(packet.obj, CitationInfo)
        or (
            isinstance(packet.obj, ASv3Progress)
            and packet.obj.status == "completed"
            and packet.obj.task_id is None
        )
        or (
            isinstance(packet.obj, AgentResponseDelta)
            and "vergiden muaftır" in packet.obj.content
        )
        for packet in packets(queue)
    )


def test_runtime_cancel_discards_late_source_results_and_durable_writes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs, broker, llm, checkpoints, queue = setup_run(monkeypatch)
    broker.block = True
    errors: list[Exception] = []

    def run() -> None:
        try:
            runtime.run_asv3_loop(**kwargs)
        except Exception as error:
            errors.append(error)

    thread = threading.Thread(target=run)
    thread.start()
    try:
        assert broker.entered.wait(3)
        writes_before_cancel = len(checkpoints)
        broker.cancelled.set()
        thread.join(3)

        assert not thread.is_alive()
        assert len(errors) == 1 and isinstance(errors[0], RunStopped)
        broker.release.set()
        assert len(checkpoints) == writes_before_cancel
        assert llm.invoke.call_count == 2
        assert kwargs["state_container"].answer_tokens is None
        output = packets(queue)
        assert not any(
            isinstance(packet.obj, (AgentResponseStart, AgentResponseDelta))
            for packet in output
        )
        assert [
            packet.obj.status
            for packet in output
            if isinstance(packet.obj, ASv3Progress)
        ][-1] == "cancelled"
    finally:
        broker.release.set()
        thread.join(3)


def test_runtime_incident_2888_zero_evidence_never_publishes_legal_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reserve_final_only(monkeypatch)
    kwargs, broker, llm, checkpoints, queue = setup_run(monkeypatch)
    language_response = list(llm.invoke.side_effect)[0]
    failures = {
        "resolve_source": ToolOutcome(
            status=OutcomeStatus.UNAVAILABLE,
            summary="This broker reads regulatory user files only.",
        ),
        "query_corpus": ToolOutcome(
            status=OutcomeStatus.UNAVAILABLE,
            summary="This broker reads regulatory user files only.",
        ),
        "search_corpus": ToolOutcome(
            status=OutcomeStatus.INVALID,
            summary="History cannot be empty for query rephrasing",
        ),
        "run_research_code": ToolOutcome(
            status=OutcomeStatus.UNAVAILABLE,
            summary="Isolated code execution is disabled by deployment policy.",
        ),
    }

    def replace_boundary(spec: ToolSpec) -> ToolSpec:
        failure = failures.get(spec.name)
        if failure is None:
            return spec
        return spec.model_copy(
            update={"handler": lambda _args, _ctx: failure.model_copy(deep=True)}
        )

    corpus_factory = runtime.build_corpus_specs
    sandbox_factory = runtime.build_sandbox_specs
    monkeypatch.setattr(
        runtime,
        "build_corpus_specs",
        lambda boundary: [replace_boundary(spec) for spec in corpus_factory(boundary)],
    )
    monkeypatch.setattr(
        runtime,
        "build_sandbox_specs",
        lambda boundary: [replace_boundary(spec) for spec in sandbox_factory(boundary)],
    )
    unsupported = "Yeni makine garanti kapsamında olsa da vergiye tabidir."
    llm.invoke.side_effect = [
        language_response,
        response(
            calls=[
                research_update(),
                (
                    "record_scenario",
                    {
                        "questions": [
                            "Ücretsiz tamir?",
                            "Yeni makine?",
                            "Bedelli tamir?",
                        ],
                        "facts": ["Standart değişim izni yok."],
                    },
                ),
                (
                    "resolve_source",
                    {"query": "Hariçte işleme", "_need_id": "comparison"},
                ),
                ("query_corpus", {"operation": "inventory", "_need_id": "comparison"}),
                (
                    "search_corpus",
                    {
                        "query": "garanti tamir",
                        "mode": "keyword",
                        "_need_id": "comparison",
                    },
                ),
                (
                    "run_research_code",
                    {"code": "print('source inventory')", "_need_id": "comparison"},
                ),
            ]
        ),
        response(unsupported),
        response(unsupported),
        response(unsupported),
        response(unsupported),
        unsafe_review(unsupported, question_count=4),
    ]
    runtime.run_asv3_loop(**kwargs)
    final_checkpoint = checkpoints[-1]
    receipts = final_checkpoint["receipts"]
    assert [item["call"]["name"] for item in receipts[:2]] == [
        "update_research",
        "record_scenario",
    ]
    assert [item["outcome"]["status"] for item in receipts[:2]] == [
        "found",
        "found",
    ]
    # Independent tool receipts arrive in completion order.
    assert len(receipts[2:6]) == 4
    assert {
        item["call"]["name"]: item["outcome"]["status"] for item in receipts[2:6]
    } == {name: outcome.status.value for name, outcome in failures.items()}
    assert sum(item["call"]["name"] == "finalization_status" for item in receipts) == 3
    assert final_checkpoint["evidence"]["records"] == []
    assert final_checkpoint["publication_status"] == "partial"
    assert final_checkpoint["publication_review"]["safe_to_publish"] is False
    assert llm.invoke.call_count == 7
    assert broker.scope is not None
    assert broker.scope.source_type == [DocumentSource.USER_FILE]
    assert broker.scope.asv3_document_set_id == 91
    assert (
        kwargs["state_container"].answer_tokens
        == "Bu sorudaki hukuki sonuçları mevcut kaynaklarla doğrulayamadım."
    )
    output = packets(queue)
    assert not any(isinstance(packet.obj, CitationInfo) for packet in output)
    assert not any(
        isinstance(packet.obj, ASv3Progress) and packet.obj.status == "completed"
        for packet in output
    )
    assert all(
        unsupported not in packet.obj.content
        for packet in output
        if isinstance(packet.obj, AgentResponseDelta)
    )


def test_runtime_review_gap_drives_new_source_before_supported_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from onyx.prompts.asv3.research import COORDINATOR_PROMPT

    kwargs, broker, llm, checkpoints, queue = setup_run(monkeypatch)
    broker.barrier = threading.Barrier(1)
    initial_profile = list(llm.invoke.side_effect)[0]
    missing = "Yeni makinenin ayrı serbest dolaşıma giriş koşulları"
    review_text = unsafe_review("Yeni makine de muaftır [1].").choice.message.content
    assert review_text is not None
    incomplete = json.loads(review_text)
    incomplete["evidence_numbers"] = [1]
    incomplete["missing_conditions"] = [missing]
    incomplete["question_results"][0]["missing_conditions"] = [missing]
    script = iter(
        [
            initial_profile,
            response(
                calls=[
                    research_update(),
                    (
                        "read_source_range",
                        {
                            "source_id": str(broker.sources[0].id),
                            "_need_id": "comparison",
                        },
                    ),
                ]
            ),
            response("Tamir ve yeni makine de muaftır [1]."),
            response(json.dumps(incomplete)),
            None,
            response("Tamir şartları [1]; yeni makinenin farklı şartları [2]."),
            supported_review(
                [1, 2], draft="Tamir şartları [1]; yeni makinenin farklı şartları [2]."
            ),
            response("Tamir sonucu [1]; yeni makine sonucu [2]."),
            supported_review([1, 2], draft="Tamir sonucu [1]; yeni makine sonucu [2]."),
        ]
    )
    recovery_observed = False

    def invoke(**arguments: Any) -> ModelResponse:
        nonlocal recovery_observed
        next_response = next(script)
        if next_response is not None:
            return next_response
        assert arguments["prompt"][0].content == COORDINATOR_PROMPT
        data = request_data(arguments)
        gap = data["publication_gap"]
        assert missing in gap["review"]["missing_conditions"]
        assert data["draft_to_repair"]
        assert [item["citation"] for item in data["evidence"]] == [1]
        recovery_observed = True
        return response(
            calls=[
                (
                    "read_source_range",
                    {"source_id": str(broker.sources[1].id), "_need_id": "comparison"},
                )
            ]
        )

    llm.invoke.side_effect = invoke
    runtime.run_asv3_loop(**kwargs)
    assert recovery_observed
    assert llm.invoke.call_count == 7
    assert checkpoints[-1]["publication_status"] == "found"
    assert checkpoints[-1]["evidence"]["included"] == [1, 2]
    assert len(kwargs["state_container"].citation_to_doc) == 2
    output = packets(queue)
    assert all(
        "de muaftır" not in packet.obj.content
        for packet in output
        if isinstance(packet.obj, AgentResponseDelta)
    )
    assert [
        packet.obj.status for packet in output if isinstance(packet.obj, ASv3Progress)
    ][-1] == "completed"


def test_finalization_waits_for_admitted_worker_originals_after_shared_decision_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from onyx.asv3.workers import WorkerPool
    from onyx.prompts.asv3.research import (
        COORDINATOR_PROMPT,
        FINAL_PROMPT,
        RESEARCHER_PROMPT,
        VERIFICATION_PROMPT,
    )

    kwargs, broker, llm, checkpoints, _queue = setup_run(monkeypatch)
    profile = (
        llm.invoke.side_effect[0] if isinstance(llm.invoke.side_effect, list) else None
    )
    # MagicMock stores iterable side effects as an iterator.
    if profile is None:
        profile = next(llm.invoke.side_effect)
    broker.barrier = threading.Barrier(1)
    broker.block = True
    final_observed = False
    selected = str(broker.sources[0].id)

    class TightRunContext(RunContext):
        def __init__(self, **arguments: Any) -> None:
            super().__init__(
                budget=SharedBudget(
                    max_decisions=15,
                    final_decision_reserve=12,
                    coordinator_decision_reserve=0,
                ),
                **arguments,
            )

    monkeypatch.setattr(runtime, "RunContext", TightRunContext)
    original_settle = WorkerPool.settle

    def settle(pool: WorkerPool) -> None:
        assert broker.entered.is_set()
        broker.release.set()
        original_settle(pool)

    monkeypatch.setattr(WorkerPool, "settle", settle)

    def save(**arguments: Any) -> None:
        snapshot = arguments["snapshot"]
        if any(
            item["call"]["name"] == "spawn_researcher" for item in snapshot["receipts"]
        ):
            assert broker.entered.wait(3)
        checkpoints.append(snapshot)

    monkeypatch.setattr(runtime, "save_asv3_checkpoint", save)

    def invoke(**arguments: Any) -> ModelResponse:
        nonlocal final_observed
        instruction = arguments["prompt"][0].content
        data = request_data(arguments)
        if instruction == COORDINATOR_PROMPT:
            return response(
                calls=[
                    research_update(),
                    (
                        "spawn_researcher",
                        {"task": "independent condition", "need_ids": ["comparison"]},
                    ),
                ]
            )
        if instruction == RESEARCHER_PROMPT:
            return response(
                calls=[
                    (
                        "read_source_range",
                        {"source_id": selected, "_need_id": "comparison"},
                    )
                ]
            )
        if instruction == FINAL_PROMPT:
            originals = json.loads(data["evidence"])
            assert len(originals) == 1
            assert originals[0]["text"].endswith("ORIGINAL_TAIL_0")
            final_observed = True
            return response("Tamir şartı özgün hükme göre uygulanır [1].")
        if isinstance(instruction, str) and instruction.startswith(VERIFICATION_PROMPT):
            assert json.loads(data["evidence"])[0]["text"].endswith("ORIGINAL_TAIL_0")
            return supported_review([1], draft=data["claim"])
        assert isinstance(profile, ModelResponse)
        return profile

    llm.invoke.side_effect = invoke
    runtime.run_asv3_loop(**kwargs)
    assert final_observed
    assert kwargs["state_container"].answer_tokens.startswith("Tamir şartı")
    assert checkpoints[-1]["evidence"]["included"] == [1]
    assert checkpoints[-1]["publication_stop_reason"] != "publication_guard_rejected"


def test_runtime_researchers_keep_scenario_facts_isolated_and_selected_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from onyx.prompts.asv3.research import (
        COORDINATOR_PROMPT,
        FINAL_PROMPT,
        RESEARCHER_PROMPT,
        VERIFICATION_PROMPT,
    )

    kwargs, _broker, llm, checkpoints, queue = setup_run(
        monkeypatch, "Araştırma tamamlandı."
    )
    lock = threading.Lock()
    coordinator_started = False
    child_steps: dict[str, int] = {}
    child_facts: dict[str, list[str]] = {}
    final_scenario: dict[str, Any] = {}
    notifications = {
        phase: ["Araştırma", "Tamir ve değişim koşullarını inceliyorum."]
        for phase in (
            "started",
            "tools",
            "worker",
            "final",
            "completed",
            "failed",
            "cancelled",
            "native_citation",
            "resume",
            "interrupted",
        )
    }

    def invoke(**arguments: Any) -> ModelResponse:
        nonlocal coordinator_started
        prompt = arguments["prompt"]
        instruction = prompt[0].content
        data = request_data(arguments)
        with lock:
            if instruction == COORDINATOR_PROMPT:
                if not coordinator_started:
                    coordinator_started = True
                    return response(
                        calls=[
                            research_update(need_ids=("branch_a", "branch_b")),
                            (
                                "record_scenario",
                                {"questions": ["parent"], "facts": ["parent fact"]},
                            ),
                            (
                                "spawn_researcher",
                                {"task": "branch-A", "need_ids": ["branch_a"]},
                            ),
                            (
                                "spawn_researcher",
                                {"task": "branch-B", "need_ids": ["branch_b"]},
                            ),
                        ]
                    )
                if any(
                    task["status"] in ("queued", "running")
                    for task in data["research_tasks"]
                ):
                    return response(
                        calls=[("wait_researcher", {"timeout_seconds": 0.05})]
                    )
                return response("Bağımsız çalışmalar değerlendirildi.")
            if instruction == RESEARCHER_PROMPT:
                task = data["request"]
                step = child_steps.get(task, 0)
                child_steps[task] = step + 1
                if step == 0:
                    assert data["facts"] == []
                    return response(
                        calls=[
                            (
                                "record_scenario",
                                {"questions": [task], "facts": [task + " fact"]},
                            )
                        ]
                    )
                child_facts[task] = data["facts"]
                return response(task + " tamamlandı.")
            if instruction.startswith(VERIFICATION_PROMPT):
                return unsafe_review(data["claim"], len(data["questions"]))
            if instruction == FINAL_PROMPT:
                final_scenario.update(data["scenario"])
                return response("Araştırma tamamlandı.")
            return response(
                json.dumps({"language": "tr", "notifications": notifications})
            )

    llm.invoke.side_effect = invoke
    selected_name = llm.config.model_name
    runtime.run_asv3_loop(**kwargs)
    assert child_facts == {"branch-A": ["branch-A fact"], "branch-B": ["branch-B fact"]}
    assert final_scenario["facts"] == ["parent fact"]
    assert kwargs["state_container"].answer_tokens == notifications["failed"][1]
    assert checkpoints[-1]["publication_status"] == "partial"
    assert not any(
        isinstance(packet.obj, ASv3Progress)
        and packet.obj.status == "completed"
        and packet.obj.task_id is None
        for packet in packets(queue)
    )
    assert llm.config.model_name == selected_name


def test_incomplete_supported_review_retains_exact_publication_rejection_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reserve_final_only(monkeypatch)
    kwargs, _broker, llm, checkpoints, queue = setup_run(monkeypatch)
    script = incomplete_script(llm)
    review = supported_review([1, 2])
    content = review.choice.message.content
    assert isinstance(content, str)
    parsed = json.loads(content)
    # Supported/safe flags alone cannot override missing question coverage.
    parsed["question_results"][0]["question_id"] = "unrelated-question"
    script[-1] = response(json.dumps(parsed))
    llm.invoke.side_effect = script
    runtime.run_asv3_loop(**kwargs)
    saved = checkpoints[-1]
    assert saved["stop_reason"] == "repeated_publication_gap"
    assert saved["publication_gap"]["data"]["review"]["missing_conditions"]
    assert saved["publication_review"]["status"] == "supported"
    assert saved["publication_review"]["safe_to_publish"] is True
    assert saved["publication_stop_reason"] == "publication_guard_rejected"
    assert saved["final_publication_gap"]["data"]["gaps"]
    assert saved["publication_status"] == "partial"
    assert saved["draft_approval"] is None
    assert not any(isinstance(packet.obj, CitationInfo) for packet in packets(queue))
    from onyx.prompts.asv3.research import FINAL_PROMPT

    assert any(
        call.kwargs["prompt"][0].content == FINAL_PROMPT
        for call in llm.invoke.call_args_list
    )


def test_verified_draft_publication_still_revalidates_acl_and_rejects_late_cancel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs, broker, llm, checkpoints, queue = setup_run(monkeypatch)
    writes_before_cancel = 0

    def revoke_during_revalidation(
        _items: list[EvidenceItem], context: RunContext
    ) -> None:
        nonlocal writes_before_cancel
        writes_before_cancel = len(checkpoints)
        broker.cancelled.set()
        context.check_active()

    monkeypatch.setattr(broker, "revalidate_evidence", revoke_during_revalidation)
    with pytest.raises(RunStopped):
        runtime.run_asv3_loop(**kwargs)
    assert llm.invoke.call_count == 4
    assert checkpoints[-1]["draft_approval"]["verification_call_id"]
    assert len(checkpoints) == writes_before_cancel
    assert kwargs["state_container"].answer_tokens is None
    assert not any(
        isinstance(packet.obj, (AgentResponseStart, AgentResponseDelta, CitationInfo))
        for packet in packets(queue)
    )


def test_rejected_final_candidate_can_reopen_research_and_publish_exact_repaired_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs, _broker, llm, checkpoints, queue = setup_run(
        monkeypatch, final="Only first rule [1]."
    )
    script = incomplete_script(llm)
    script[-1] = unsafe_review("The second rule and later procedure were omitted")
    corrected = (
        "First rule and its conditions [1]. Second rule and later procedure [2]."
    )
    llm.invoke.side_effect = [
        *script,
        response(calls=[("read_evidence", {"citation": 2, "_need_id": "comparison"})]),
        response(corrected),
        supported_review([1, 2], draft=corrected),
    ]
    runtime.run_asv3_loop(**kwargs)
    saved = checkpoints[-1]
    assert saved["publication_status"] == "found"
    assert saved["last_draft"] == corrected
    assert saved["publication_stop_reason"] == "verified_draft_published"
    assert saved["final_publication_gap"] is None
    assert kwargs["state_container"].answer_tokens.startswith("First rule")
    assert any(isinstance(packet.obj, CitationInfo) for packet in packets(queue))
    repair_call = next(
        call
        for call in llm.invoke.call_args_list
        if request_data(call.kwargs).get("draft_to_repair") == "Only first rule [1]."
    )
    assert request_data(repair_call.kwargs)["publication_gap"]
    assert llm.config.model_name == "scripted"


def test_local_assertion_failure_repairs_only_rejected_outcome_despite_broad_approval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs, broker, llm, checkpoints, _queue = setup_run(monkeypatch)
    originals = ["Koşullu tamire izin verilir.", "Başvuru sonucu idareye bildirilir."]
    for source, text in zip(broker.sources, originals):
        broker.chunks[str(source.id)] = replace(
            broker.chunks[str(source.id)], text=text
        )
    numbers: dict[str, int] = {}
    draft = corrected = ""
    repair_seen = False

    def initial_draft(**arguments: Any) -> ModelResponse:
        nonlocal draft, corrected
        numbers.update(
            {
                item["chunk_id"]: item["citation"]
                for item in request_data(arguments)["evidence"]
            }
        )
        draft = f"Koşullu tamire izin verilir [{numbers['chunk-0']}].\n\nBaşvuru sonucunun bildirilmesi tüm şartları kaldırır [{numbers['chunk-1']}]."
        corrected = f"Koşullu tamire izin verilir [{numbers['chunk-0']}].\n\nBaşvuru sonucu idareye bildirilir [{numbers['chunk-1']}]."
        return response(draft)

    def first_review(**_arguments: Any) -> ModelResponse:
        review = supported_review(
            list(numbers.values()),
            draft=draft,
            quotes={numbers["chunk-0"]: originals[0], numbers["chunk-1"]: originals[1]},
        )
        assert isinstance(review.choice.message.content, str)
        data = json.loads(review.choice.message.content)
        data["assertion_results"][1].update(
            status="unsupported",
            missing_conditions=["Bildirim tüm şartları kaldırmaz."],
            explanation="Özgün kaynak yalnız bildirim öngörür.",
        )
        return response(json.dumps(data))

    def repair(**arguments: Any) -> ModelResponse:
        nonlocal repair_seen
        payload = request_data(arguments)
        assert payload["draft_to_repair"] == draft
        gaps = payload["publication_gap"]["assertion_gaps"]
        assert len(gaps) == 1 and gaps[0]["text"].endswith(
            f"kaldırır [{numbers['chunk-1']}]."
        )
        repair_seen = True
        return response(corrected)

    def corrected_review(**_arguments: Any) -> ModelResponse:
        return supported_review(
            list(numbers.values()),
            draft=corrected,
            quotes={numbers["chunk-0"]: originals[0], numbers["chunk-1"]: originals[1]},
        )

    stages = iter(
        [
            *list(llm.invoke.side_effect)[:2],
            initial_draft,
            first_review,
            repair,
            corrected_review,
        ]
    )

    def invoke(**arguments: Any) -> ModelResponse:
        scheduled = next(stages)
        return scheduled(**arguments) if callable(scheduled) else scheduled

    llm.invoke.side_effect = invoke
    runtime.run_asv3_loop(**kwargs)
    assert repair_seen
    assert checkpoints[-1]["publication_status"] == "found"
    assert checkpoints[-1]["last_draft"] == corrected
    assert kwargs["state_container"].answer_tokens.startswith(
        "Koşullu tamire izin verilir"
    )
    assert "tüm şartları kaldırır" not in kwargs["state_container"].answer_tokens
