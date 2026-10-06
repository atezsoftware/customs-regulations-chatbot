"""Compare real serial coordinators with task-owned serial child sessions."""

import copy
import hashlib
import threading
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, Literal, cast
from unittest.mock import MagicMock

import pytest
from pydantic import JsonValue

from onyx.asv3 import runtime
from onyx.asv3.corpus_tools import CorpusBroker, build_corpus_specs, evidence_for_chunk
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.harness import Harness
from onyx.asv3.legal_source_reviews import LegalSourceReviews
from onyx.asv3.llm_adapter import COORDINATOR_SESSION_ACTIONS
from onyx.asv3.models import (
    EvidenceItem,
    OutcomeStatus,
    RunContext,
    SharedBudget,
    ToolOutcome,
    ToolSpec,
)
from onyx.asv3.parallel_answers import ParallelAnswerReceipts
from onyx.asv3.sandbox import build_sandbox_specs
from onyx.asv3.serial_experimental_session import (
    SERIAL_SESSION_POLICY,
    SerialExperimentalSession,
)
from onyx.asv3.session_research import retain_session_research
from onyx.asv3.source_tools import build_source_specs
from onyx.asv3.supplemental_tools import build_supplemental_specs
from onyx.chat.models import ChatMessageSimple
from onyx.configs.constants import MessageType
from onyx.llm.model_response import ModelResponse
from onyx.llm.models import ReasoningEffort
from onyx.prompts.asv3.experimental import (
    EXPERIMENTAL_COORDINATOR_PROMPT,
    EXPERIMENTAL_PROMPT_VERSION,
    EXPERIMENTAL_RESEARCHER_PROMPT,
)
from tests.unit.onyx.asv3.test_native_cache_projection import actual_originals
from tests.unit.onyx.asv3.test_runtime import (
    CorpusBoundary,
    response,
    setup_run,
    user_payload,
)
from tests.unit.onyx.asv3.test_session_research import previous

pytestmark = pytest.mark.usefixtures("empty_source_inventory")

SCENARIO = "Alıcı on bir parça için işlem yapıyor; işlem tarihi 6 Ekim, izin tarihi bilinmiyor. Ücretli ve ücretsiz alternatif farklıdır."
FOCUS = "Bu işlemin kaynakta belirtilen koşulu nedir?"
INSTRUCTION = EXPERIMENTAL_COORDINATOR_PROMPT + "\n\n" + COORDINATOR_SESSION_ACTIONS
SERIAL_INSTRUCTION_HASH = (
    "bf8b166e5daca156751835eac0f02d50b2e76602f50971f62916e20aea2fea83"
)
Kind = Literal[
    "conversation",
    "scenario",
    "clarification",
    "partial",
    "partial-originals",
    "originals",
]


def declaration() -> dict[str, JsonValue]:
    return {"outcome_id": "local-result", "question_ids": ["q0"], "detail": FOCUS}


def tool_definitions(call: dict[str, Any]) -> dict[str, Any]:
    return {item["function"]["name"]: item for item in call["tools"]}


def scoped_gate(
    outcome: ToolOutcome,
) -> tuple[OutcomeStatus, str, dict[str, JsonValue]]:
    data = copy.deepcopy(outcome.data)
    requirements = data.get("retained_authority_requirements")
    if isinstance(requirements, list):
        for row in requirements:
            assert isinstance(row, dict)
            assert row.get("owner") in {"coordinator", "owned-task"}
            row.pop("owner", None)
            row.pop("requirement_id", None)
    return outcome.status, outcome.summary, data


def terminal(kind: Kind, body: str) -> ModelResponse:
    metadata = {"_language": "tr", "_outcomes": [declaration()]}
    if kind == "clarification":
        return response(calls=[("ask_user", {"question": body, "_language": "tr"})])
    if kind in {"partial", "partial-originals"}:
        return response(calls=[("submit_partial_answer", {"answer": body, **metadata})])
    return response(
        calls=[("submit_answer", {"answer": body, "basis": kind, **metadata})]
    )


@dataclass
class SerialRun:
    broker: CorpusBoundary
    selected: MagicMock
    harness: Harness
    calls: list[dict[str, Any]]
    checkpoints: list[dict[str, Any]]
    history: str


def serial_run(monkeypatch: pytest.MonkeyPatch, kind: Kind, body: str) -> SerialRun:
    kwargs, broker, selected, checkpoints, _queue = setup_run(monkeypatch)
    kwargs.pop("test_language")
    kwargs["research_profile"] = "experimental"
    kwargs["custom_agent_prompt"] = (
        "Kullanıcının sağladığı olguları ve özgün koşulları koru."
    )
    kwargs["simple_chat_history"] = [
        ChatMessageSimple(
            message=SCENARIO, token_count=len(SCENARIO), message_type=MessageType.USER
        ),
        ChatMessageSimple(
            message=FOCUS, token_count=len(FOCUS), message_type=MessageType.USER
        ),
    ]
    selected.config = selected.config.model_copy(update={"model_name": "gpt-6-luna"})
    broker.barrier = threading.Barrier(1)
    broker.sources[0] = replace(broker.sources[0], name="Example Law")
    first_source = str(broker.sources[0].id)
    broker.chunks[first_source] = replace(
        broker.chunks[first_source], heading_path=("Example Law", "MADDE 142")
    )
    calls: list[dict[str, Any]] = []
    harnesses: list[Harness] = []
    real_harness = runtime.Harness

    def capture(**arguments: Any) -> Harness:
        harness = real_harness(**arguments)
        harnesses.append(harness)
        return harness

    monkeypatch.setattr(runtime, "Harness", capture)

    def invoke(**arguments: Any) -> ModelResponse:
        calls.append(arguments)
        source_call = kind in {"originals", "partial-originals"}
        if source_call and len(calls) == 1:
            return response(
                calls=[
                    (
                        "read_source_range",
                        {"source_id": str(broker.sources[0].id), "_language": "tr"},
                    )
                ]
            )
        assert len(calls) == (2 if source_call else 1), [
            receipt.outcome.model_dump(mode="json", exclude={"evidence"})
            for receipt in harnesses[-1].receipts
        ]
        return terminal(kind, body)

    selected.invoke.side_effect = invoke
    runtime.run_asv3_loop(**kwargs)
    assert len(harnesses) == 1
    history = "\n".join(
        f"{row.message_type.value}: {row.message}"
        for row in kwargs["simple_chat_history"]
    )
    return SerialRun(broker, selected, harnesses[0], calls, checkpoints, history)


def outer(
    run: SerialRun, *, owner: str = "owned-task", assignment: str = "independent-a"
) -> RunContext:
    return RunContext(
        run_id=run.harness.context.run_id,
        depth=1,
        scope=copy.deepcopy(run.harness.context.scope),
        timeout_seconds=float("inf"),
        budget=SharedBudget(unlimited_execution=True),
        services={
            "task_id": owner,
            "assignment_id": assignment,
            "research_profile": "experimental",
            "experimental_parallel": True,
            "independent_question": True,
            "assistant_instructions": "Kullanıcının sağladığı olguları ve özgün koşulları koru.",
        },
    )


def session(
    run: SerialRun,
    envelope: RunContext,
    ledger: EvidenceLedger,
    *,
    verify: Callable[[RunContext, dict[str, JsonValue]], ToolOutcome] | None = None,
) -> SerialExperimentalSession:
    def capabilities(context: RunContext) -> list[ToolSpec]:
        context.services.update(
            legal_source_navigation=run.broker.related_source_navigation,
            legal_source_navigation_acquire=run.broker.related_sources_for_evidence,
        )
        broker = cast(CorpusBroker, run.broker)
        return [
            *build_corpus_specs(
                broker, require_search_targets=True, source_identity_guidance=False
            ),
            *build_source_specs(broker),
            *build_sandbox_specs(broker),
            *build_supplemental_specs(),
        ]

    return SerialExperimentalSession(
        outer_context=envelope,
        request=FOCUS,
        scenario_request=SCENARIO,
        history=run.history,
        ledger=ledger,
        llm=run.selected,
        reasoning_effort=ReasoningEffort.AUTO,
        token_counter=len,
        capability_factory=capabilities,
        verify=verify
        or (
            lambda _context, _arguments: ToolOutcome(
                status=OutcomeStatus.FOUND, summary="Selected verification completed"
            )
        ),
    )


@pytest.mark.parametrize(
    "kind,body",
    [
        ("conversation", "Merhaba! Nasıl yardımcı olabilirim?"),
        ("scenario", "Kullanıcının verdiği miktar on bir parçadır."),
        ("clarification", "İzin hangi tarihte verilmişti?"),
        (
            "partial",
            "İstenen sonraki aşamayı belirleyen özgün metin bu araştırmada henüz doğrulanamadı.",
        ),
        (
            "partial-originals",
            "Okunan özgün koşul korunmuştur [1].\n\nİşlemin sonraki aşamasındaki tasfiye etkisini belirleyen özgün metin bu araştırmada henüz doğrulanamadı.",
        ),
        ("originals", "İşleme uygulanacak özgün koşul okunmuştur [1]."),
    ],
)
def test_real_serial_and_owned_session_use_same_prompt_schemas_and_terminal_handlers(
    monkeypatch: pytest.MonkeyPatch, kind: Kind, body: str
) -> None:
    run = serial_run(monkeypatch, kind, body)
    baseline_calls = list(run.calls)
    run.calls.clear()
    run.selected.reset_mock()
    child = session(run, outer(run), EvidenceLedger())
    result = child.run()

    assert hashlib.sha256(INSTRUCTION.encode()).hexdigest() == SERIAL_INSTRUCTION_HASH
    assert EXPERIMENTAL_PROMPT_VERSION == "asv3-experimental-2026-10-06.7"
    assert len(run.calls) == len(baseline_calls)
    for base, actual in zip(baseline_calls, run.calls, strict=True):
        assert actual["prompt"][0].content == base["prompt"][0].content == INSTRUCTION
        actual_definitions = copy.deepcopy(tool_definitions(actual))
        need = actual_definitions["update_research"]["function"]["parameters"]["$defs"][
            "ResearchNeed"
        ]["properties"]
        assert need["question_ids"]["items"].pop("enum") == ["q0"]
        assert need["determination_ids"]["items"].pop("enum") == ["q0:d0"]
        assert actual_definitions == tool_definitions(base)
        assert actual["tool_choice"] == base["tool_choice"]
        assert actual["timeout_override"] is base["timeout_override"] is None
        request = user_payload(actual["prompt"][1])
        assert request["request"] == FOCUS
        assert SCENARIO in request["conversation"]
        assert (
            request["assistant_instructions"]
            == "Kullanıcının sağladığı olguları ve özgün koşulları koru."
        )
        assert "gpt-6-luna" == run.selected.config.model_name
        assert run.selected.config.seed is None
    assert result.summary == body
    assert result.status == (
        OutcomeStatus.PARTIAL
        if kind in {"partial", "partial-originals", "clarification"}
        else OutcomeStatus.FOUND
    )
    assert child.outcomes.outcome_ids() == (
        [] if kind == "clarification" else ["local-result"]
    )
    assert child.context.depth == 0 and child.outer_context.depth == 1
    assert child.context.services["experimental_parallel"] is False
    assert child.context.budget is child.outer_context.budget
    assert child.context.budget.unlimited_execution is True
    assert child.harness.adaptive_tool_parallelism is False
    definitions = tool_definitions(run.calls[0])
    assert {
        "ask_user",
        "spawn_researcher",
        "verify_claim",
        "submit_partial_answer",
    } <= definitions.keys()
    submit = definitions["submit_answer"]["function"]["parameters"]["properties"]
    assert submit["basis"]["enum"] == ["conversation", "scenario", "originals"]
    assert "_outcomes" in submit
    if kind in {"originals", "partial-originals"}:
        actual = actual_originals(run.calls[-1]["prompt"])
        assert [(row["citation"], row["text"]) for row in actual] == [
            (1, run.broker.chunks[str(run.broker.sources[0].id)].text)
        ]
        assert child.ledger.completely_delivered(child.model.last_call_id or "") == {1}


@pytest.mark.parametrize(
    "kind", ["unknown", "undelivered", "governing", "related", "supported"]
)
def test_session_source_gate_matches_actual_serial_gate(
    monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    run = serial_run(monkeypatch, "originals", "Özgün koşul okunmuştur [1].")
    run.calls.clear()
    child = session(run, outer(run), EvidenceLedger())
    child.run()
    if kind == "undelivered":
        for ledger, context in [
            (run.harness.evidence, run.harness.context),
            (child.ledger, child.context),
        ]:
            source = run.broker.sources[1]
            assert ledger.add(
                [evidence_for_chunk(source, run.broker.chunks[str(source.id)])], context
            ) == [2]
        candidate = "Sonraki hüküm uygulanır [2]."
    elif kind == "unknown":
        candidate = "Sonraki hüküm uygulanır [999]."
    elif kind == "governing":
        candidate = "8917 sayılı Faaliyet Kanunu m.27 gereğince kesin izin vardır [1]."
    elif kind == "related":
        navigation: list[dict[str, JsonValue]] = [
            {
                "anchor_source_id": str(run.broker.sources[0].id),
                "article_no": "142",
                "qualifier": None,
                "candidates": [
                    {
                        "source_id": "counter-decision",
                        "name": "Related judicial candidate",
                        "candidate_role": "judicial_candidate",
                    }
                ],
            }
        ]
        reviews = run.harness.context.services["legal_source_reviews"]
        base_call = run.harness.context.services["last_model_call_id"]
        assert isinstance(reviews, LegalSourceReviews) and isinstance(base_call, str)
        assert child.model.last_call_id is not None
        reviews.record_delivery(
            base_call,
            run.harness.context,
            navigation,
            run.harness.evidence,
        )
        child.reviews.record_delivery(
            child.model.last_call_id, child.context, navigation, child.ledger
        )
        candidate = "Okunmuş hüküm uygulanır [1]."
    else:
        candidate = "Okunmuş hüküm uygulanır [1]."
    assert run.harness.draft_guard is not None
    expected = run.harness.draft_guard(candidate)
    actual = child.publication_gap(candidate, child.model.last_call_id)
    if kind == "supported":
        assert actual is expected is None
    else:
        assert actual is not None and expected is not None
        assert scoped_gate(actual) == scoped_gate(expected)
        if kind == "undelivered":
            assert actual.data["undelivered_citations"] == [2]
        if kind == "related":
            assert actual.data["pending_related_source_review"] is True


def test_fresh_local_state_reuses_authorized_session_originals_without_source_research(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = serial_run(monkeypatch, "conversation", "Merhaba!")
    root = outer(run)
    root.services["task_outcome_ids"] = ["root-only-outcome"]
    root.services["submitted_answer"] = "Sibling body must not become this answer"
    ledger = EvidenceLedger()
    source = run.broker.sources[0]
    saved = previous(evidence_for_chunk(source, run.broker.chunks[str(source.id)]))
    saved["scope"] = copy.deepcopy(root.scope)
    checked: list[EvidenceItem] = []
    retain_session_research(
        saved, root, ledger, lambda items, _context: checked.extend(items)
    )
    unchanged = copy.deepcopy(root.services["session_research"])
    first, sibling = (
        session(run, root, ledger),
        session(
            run, outer(run, owner="sibling-task", assignment="independent-b"), ledger
        ),
    )
    root_keys = set(root.services)
    assert (
        first.context.services["evidence"]
        is sibling.context.services["evidence"]
        is ledger
    )
    assert first.outcomes is not sibling.outcomes
    assert first.research is not sibling.research
    assert first.reviews is not sibling.reviews
    assert first.requirements is not sibling.requirements
    assert first.harness.working_memory is not sibling.harness.working_memory
    assert first.outcomes.outcome_ids() == sibling.outcomes.outcome_ids() == []
    assert first.requirements.syntactic_reference_binding is False
    assert "submitted_answer" not in first.context.services
    assert "task_outcome_ids" not in first.context.services
    assert checked and ledger.completely_delivered("historical-call") == set()
    run.calls.clear()

    def from_memory(**arguments: Any) -> ModelResponse:
        run.calls.append(arguments)
        assert len(run.calls) == 1, [
            receipt.outcome.status for receipt in first.harness.receipts
        ]
        originals = actual_originals(arguments["prompt"])
        assert len(originals) == 1 and originals[0]["text"] == checked[0].text
        assert SCENARIO in user_payload(arguments["prompt"][1])["conversation"]
        assert (
            user_payload(arguments["prompt"][-1])["session_research"]["status"]
            == "revalidated"
        )
        return terminal("originals", "Önceki özgün koşul bu senaryoya uygulanır [1].")

    run.selected.invoke.side_effect = from_memory
    try:
        result = first.run()
        assert result.status == OutcomeStatus.FOUND and len(run.calls) == 1
        assert [receipt.call.name for receipt in first.harness.receipts] == [
            "submit_answer"
        ]
        assert first.ledger.completely_delivered(first.model.last_call_id or "") == {1}
        assert sibling.outcomes.outcome_ids() == []
        assert sibling.harness.turns == [] and sibling.harness.receipts == []
        assert root.services["session_research"] == unchanged
        assert set(root.services) == root_keys
    finally:
        sibling.workers.close()


def test_exact_long_serial_body_seals_under_outer_owner_and_restores_without_reexecution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = (
        "Koşul ve sonraki aşama [1].\n\n"
        + ("İstisna, belge ve ihtilaf ayrıntısı; 東京 İĞŞçöü.\n" * 950)
        + "\nKaynaklı takip sorusu: Bu belgenin içeriğini ayrıntılandırayım mı?"
    )
    assert len(body.encode()) > 30000
    run = serial_run(monkeypatch, "originals", body)
    run.calls.clear()
    root = RunContext(
        run_id=run.harness.context.run_id,
        scope=copy.deepcopy(run.harness.context.scope),
        budget=SharedBudget(unlimited_execution=True),
    )
    envelope = outer(run)
    ledger = EvidenceLedger()
    child = session(run, envelope, ledger)
    result = child.run()
    call_id = child.model.last_call_id
    assert call_id and result.summary == body
    state = child.source_state()
    accepted = state["accepted"]
    assert isinstance(accepted, dict)
    assert accepted["model_call_id"] == call_id
    assert accepted["answer_hash"] == hashlib.sha256(body.encode()).hexdigest()
    envelope.services["last_model_call_id"] = call_id
    assignment: dict[str, JsonValue] = {
        "question_id": "independent-a",
        "question": FOCUS,
        "task_id": "owned-task",
        "parent_question_ids": [1],
    }
    receipts = ParallelAnswerReceipts(root, SCENARIO, user_id="owner")
    receipt_id = receipts.seal(
        envelope,
        assignment=assignment,
        answer=body,
        status=result.status,
        model_call_id=call_id,
        ledger=ledger,
        validate_body=lambda: child.validate_accepted(body, call_id, result.status),
        source_state=state,
    )
    snapshot = child.snapshot()
    assert ledger.export()["pinned_delivery_calls"] == [call_id]
    run.selected.invoke.side_effect = AssertionError(
        "Accepted restore must not invoke any model"
    )
    restored = session(run, envelope, ledger)
    try:
        restored.restore(snapshot)
        assert restored.snapshot()["last_draft"] == body
        replayed = restored.run()
        assert replayed.summary == body and replayed.status == result.status
        assert restored.source_state() == state
        assert restored.validate_accepted(body, call_id, result.status) is None
        receipts.verify(
            root,
            receipt_id=receipt_id,
            task_id="owned-task",
            assignment=assignment,
            answer=body,
            status=result.status,
            ledger=ledger,
            validate_body=lambda: restored.validate_accepted(
                body, call_id, result.status
            ),
            source_state=restored.source_state(),
        )
        with pytest.raises(ValueError, match="changed"):
            restored.validate_accepted(body[:-1], call_id, result.status)
        exported = receipts.export()["receipts"]
        assert isinstance(exported, list) and isinstance(exported[0], dict)
        assert exported[0]["answer"] == body
    finally:
        restored.workers.close()


@pytest.mark.parametrize(
    "tamper",
    ["missing-policy", "old-policy", "owner", "assignment_id", "scenario_hash"],
)
def test_new_serial_session_checkpoint_rejects_old_or_foreign_policy(
    monkeypatch: pytest.MonkeyPatch, tamper: str
) -> None:
    run = serial_run(monkeypatch, "conversation", "Merhaba!")
    run.calls.clear()
    child = session(run, outer(run), EvidenceLedger())
    child.run()
    snapshot = child.snapshot()
    state = snapshot["serial_experimental_session"]
    assert isinstance(state, dict) and state["policy"] == SERIAL_SESSION_POLICY
    if tamper == "missing-policy":
        snapshot.pop("serial_experimental_session")
    elif tamper == "old-policy":
        state["policy"] = "old-custom-parallel-session"
    else:
        state[tamper] = "different-owned-binding"
    restored = session(run, child.outer_context, child.ledger)
    try:
        with pytest.raises(ValueError):
            restored.restore(snapshot)
    finally:
        restored.workers.close()


def test_optional_worker_keeps_ordinary_serial_researcher_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = serial_run(monkeypatch, "conversation", "Merhaba!")
    run.calls.clear()
    child = session(run, outer(run), EvidenceLedger())
    worker_calls: list[str] = []
    coordinator_calls = 0

    def invoke(**arguments: Any) -> ModelResponse:
        nonlocal coordinator_calls
        instruction = arguments["prompt"][0].content
        worker_calls.append(instruction)
        assert user_payload(arguments["prompt"][1])["conversation"].count(SCENARIO) >= 1
        if instruction == EXPERIMENTAL_RESEARCHER_PROMPT:
            return response("Kullanıcı on bir parça vermiştir.")
        assert instruction == INSTRUCTION
        coordinator_calls += 1
        tools = tool_definitions(arguments)
        assert "spawn_researcher" in tools
        if coordinator_calls == 1:
            return response(
                calls=[
                    (
                        "spawn_researcher",
                        {"task": "Kullanıcının verdiği miktarı kontrol et."},
                    )
                ]
            )
        if coordinator_calls == 2:
            assert len(child.workers.list()) == 1
            return response(
                calls=[
                    ("wait_researcher", {"task_id": child.workers.list()[0].task_id})
                ]
            )
        assert coordinator_calls == 3
        tasks = child.workers.results(full=True)
        assert len(tasks) == 1 and tasks[0].outcome is not None
        return terminal("scenario", "Kullanıcının verdiği miktar on bir parçadır.")

    run.selected.invoke.side_effect = invoke
    result = child.run()
    assert result.status == OutcomeStatus.FOUND
    assert worker_calls.count(INSTRUCTION) == 3
    assert worker_calls.count(EXPERIMENTAL_RESEARCHER_PROMPT) == 1
    assert worker_calls[0] == worker_calls[-1] == INSTRUCTION
    workers = child.snapshot()["workers"]
    assert isinstance(workers, dict) and workers["tasks"]


@pytest.mark.parametrize("role", ["user", "USER"])
def test_full_scenario_already_last_in_history_is_not_serialized_twice(
    monkeypatch: pytest.MonkeyPatch, role: str
) -> None:
    run = serial_run(monkeypatch, "conversation", "Merhaba!")
    run.calls.clear()
    run.history = (
        f"user: Önceki oturum olgusu\nassistant: Önceki cevap\n{role}: {SCENARIO}"
    )
    child = session(run, outer(run), EvidenceLedger())
    child.run()
    assert child.history == run.history
    payload = user_payload(run.calls[0]["prompt"][1])
    assert payload["conversation"] == run.history
    assert payload["conversation"].count(SCENARIO) == 1


def test_selected_verification_dispatch_is_bound_to_local_serial_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = serial_run(monkeypatch, "conversation", "Merhaba!")
    run.calls.clear()
    callbacks: list[tuple[RunContext, dict[str, JsonValue]]] = []

    def verify(context: RunContext, arguments: dict[str, JsonValue]) -> ToolOutcome:
        callbacks.append((context, copy.deepcopy(arguments)))
        return ToolOutcome(
            status=OutcomeStatus.FOUND,
            summary="The selected original supports this condition.",
        )

    envelope = outer(run)
    child = session(run, envelope, EvidenceLedger(), verify=verify)

    def invoke(**arguments: Any) -> ModelResponse:
        run.calls.append(arguments)
        if len(run.calls) == 1:
            return response(
                calls=[
                    ("read_source_range", {"source_id": str(run.broker.sources[0].id)})
                ]
            )
        if len(run.calls) == 2:
            return response(
                calls=[
                    (
                        "verify_claim",
                        {
                            "claim": "The selected original supports this condition.",
                            "citations": [1],
                            "facts": ["Alıcı on bir parça için işlem yapıyor."],
                        },
                    )
                ]
            )
        assert len(run.calls) == 3
        return terminal("originals", "Özgün koşul uygulanır [1].")

    run.selected.invoke.side_effect = invoke
    result = child.run()
    assert result.status == OutcomeStatus.FOUND
    assert [receipt.call.name for receipt in child.harness.receipts] == [
        "read_source_range",
        "verify_claim",
        "submit_answer",
    ]
    assert len(callbacks) == 1
    context, arguments = callbacks[0]
    assert context is not envelope
    assert context.run_id == child.context.run_id
    assert context.services["task_id"] == child.context.services["task_id"]
    assert context.budget is child.context.budget
    assert context.depth == 0 and context.services["experimental_parallel"] is False
    assert (
        context.scope == envelope.scope and context.services["evidence"] is child.ledger
    )
    assert context.services["outcome_map"] is child.outcomes
    assert arguments["citations"] == [1]
    assert child.ledger.completely_delivered(child.model.last_call_id or "") == {1}
