import json
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from pydantic import JsonValue

from onyx.asv3 import llm_adapter
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.llm_adapter import ResearchModel
from onyx.asv3.models import (
    CapabilityCall,
    EvidenceItem,
    HarnessView,
    OutcomeStatus,
    ResearchTurn,
    RunContext,
    RunStopped,
    ToolOutcome,
    ToolReceipt,
)
from onyx.llm.interfaces import LLM, LLMConfig
from onyx.llm.model_response import Choice, Message, ModelResponse
from onyx.llm.models import (
    AssistantMessage,
    ChatCompletionMessage,
    FunctionCall,
    ToolCall,
    ToolMessage,
    UserMessage,
)


def model(limit: int = 100000) -> MagicMock:
    llm = MagicMock(spec=LLM)
    llm.config = LLMConfig(
        model_provider="openai",
        model_name="selected-model",
        temperature=0,
        max_input_tokens=limit,
    )
    llm.invoke.return_value = ModelResponse(
        id="answer", created="0", choice=Choice(message=Message(content="Rule [1]."))
    )
    return llm


def original(
    ledger: EvidenceLedger, context: RunContext, text: str
) -> dict[str, JsonValue]:
    number = ledger.add(
        [EvidenceItem(source_id="law", chunk_id=str(len(text)), text=text)], context
    )[0]
    return cast(dict[str, JsonValue], json.loads(ledger.serialize_records([number]))[0])


def turn(
    identity: str, records: list[dict[str, JsonValue]], *, extra: str = ""
) -> ResearchTurn:
    return ResearchTurn(
        assistant=AssistantMessage(
            tool_calls=[
                ToolCall(
                    id=identity,
                    function=FunctionCall(
                        name="read_provision", arguments='{"article":"168"}'
                    ),
                )
            ]
        ),
        results=[
            ToolMessage(
                tool_call_id=identity,
                content=json.dumps(
                    {
                        "outcome": {"status": "found", "data": {"provision": "168"}},
                        "evidence_ids": [record["citation"] for record in records],
                        "original_evidence": records,
                        "navigation": extra,
                    },
                    ensure_ascii=False,
                ),
            )
        ],
    )


def view(**kwargs: Any) -> HarnessView:
    return HarnessView(
        request="Explain both outcomes without dropping conditions.",
        questions=["Explain both outcomes without dropping conditions."],
        facts=[],
        receipts=[],
        evidence=[],
        tools=[],
        **kwargs,
    )


def last_payload(llm: MagicMock) -> dict[str, Any]:
    content = llm.invoke.call_args.kwargs["prompt"][-1].content
    return cast(dict[str, Any], json.loads(content[0].text))


def first_user_payload(message: ChatCompletionMessage) -> tuple[dict[str, Any], str]:
    assert isinstance(message, UserMessage)
    assert isinstance(message.content, str)
    question, offset = json.JSONDecoder().raw_decode(message.content)
    assert isinstance(question, dict)
    return cast(dict[str, Any], question), message.content[offset:]


def seed_forks(llm: MagicMock) -> list[MagicMock]:
    forks: list[MagicMock] = []

    def fork(seed: int) -> MagicMock:
        selected = model(llm.config.max_input_tokens)
        selected.config = llm.config.model_copy(update={"seed": seed})
        selected.invoke.side_effect = llm.invoke
        forks.append(selected)
        return selected

    llm.with_seed.side_effect = fork
    return forks


def test_native_coordinator_pins_first_and_adaptive_decisions_without_extra_calls() -> (
    None
):
    llm = model(500000)
    llm.config = llm.config.model_copy(
        update={
            "model_provider": "vertex_ai",
            "model_name": "gemini-3.8-flash",
            "temperature": 1,
        }
    )
    forks = seed_forks(llm)
    adapter = ResearchModel(llm, RunContext(), lean_native_mode=True)
    adapter._native_output_capacity = 65536
    adapter.decide(view())
    first_state = adapter.native_sampling_snapshot()
    adapter.decide(view())
    assert [fork.config.seed for fork in forks] == [31, 1424088823]
    assert [fork.invoke.call_count for fork in forks] == [1, 1]
    assert llm.invoke.call_count == 2
    assert llm.config.seed is None
    assert all(fork.config.temperature == 1 for fork in forks)
    assert all(fork.invoke.call_args.kwargs["max_tokens"] == 65536 for fork in forks)
    assert first_state["first_decision_started"] is True
    assert first_state["first_decision_completed"] is True
    assert first_state["settings"] == {
        "mode": "native_coordinator",
        "version": 1,
        "first_seed": 31,
        "continuation_seed": 1424088823,
    }


@pytest.mark.parametrize(
    ("provider", "name", "seed", "depth", "native"),
    [
        ("vertex_ai", "gemini-3.8-flash", 0, 0, True),
        ("vertex_ai", "gemini-3.8-flash", 42, 0, True),
        ("openai", "gemini-3.8-flash", None, 0, True),
        ("vertex_ai", "another-model", None, 0, True),
        ("vertex_ai", "gemini-3.8-flash", None, 1, True),
        ("vertex_ai", "gemini-3.8-flash", None, 0, False),
    ],
)
def test_native_default_seed_profile_preserves_explicit_seed_and_other_llms(
    provider: str,
    name: str,
    seed: int | None,
    depth: int,
    native: bool,
) -> None:
    llm = model()
    llm.config = llm.config.model_copy(
        update={"model_provider": provider, "model_name": name, "seed": seed}
    )
    adapter = ResearchModel(llm, RunContext(depth=depth), lean_native_mode=native)
    assert adapter._native_decision_llm(view()) is llm
    llm.with_seed.assert_not_called()
    assert llm.config.seed == seed


def test_native_provider_retries_keep_the_same_first_seed_fork(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm = model()
    llm.config = llm.config.model_copy(
        update={"model_provider": "vertex_ai", "model_name": "gemini-3.8-flash"}
    )
    successful = llm.invoke.return_value
    llm.invoke.side_effect = [
        RuntimeError("retryable provider failure"),
        successful,
        successful,
    ]
    monkeypatch.setattr(llm_adapter, "is_retryable_provider_error", lambda _error: True)
    monkeypatch.setattr(llm_adapter, "provider_retry_delay", lambda _error, _attempt: 0)
    forks = seed_forks(llm)
    adapter = ResearchModel(llm, RunContext(), lean_native_mode=True)
    adapter._native_output_capacity = 65536
    adapter.decide(view())
    adapter.decide(view())
    assert [fork.config.seed for fork in forks] == [31, 1424088823]
    assert [fork.invoke.call_count for fork in forks] == [2, 1]
    assert llm.config.seed is None


@pytest.mark.parametrize(
    "progress",
    [
        {"pending_calls": [{"name": "read_provision"}]},
        {"turns": [{}]},
        {"receipts": [{}]},
        {"evidence": {"records": [{}]}},
        {"last_draft": "Supported outcome [1]."},
    ],
)
def test_native_resume_uses_continuation_for_recorded_decisions(
    progress: dict[str, JsonValue],
) -> None:
    llm = model()
    llm.config = llm.config.model_copy(
        update={"model_provider": "vertex_ai", "model_name": "gemini-3.8-flash"}
    )
    forks = seed_forks(llm)
    adapter = ResearchModel(llm, RunContext(), lean_native_mode=True)
    interrupted = adapter.native_sampling_snapshot()
    interrupted["first_decision_started"] = True
    for saved in (progress, {**progress, "native_coordinator_sampling": interrupted}):
        adapter.restore_native_sampling(saved)
        adapter._native_decision_llm(view())
    assert [fork.config.seed for fork in forks] == [1424088823, 1424088823]


def test_native_interrupted_first_resume_replays_first_and_rejects_changed_profile() -> (
    None
):
    llm = model()
    llm.config = llm.config.model_copy(
        update={"model_provider": "vertex_ai", "model_name": "gemini-3.8-flash"}
    )
    forks = seed_forks(llm)
    adapter = ResearchModel(llm, RunContext(), lean_native_mode=True)
    interrupted = adapter.native_sampling_snapshot()
    interrupted["first_decision_started"] = True
    adapter.restore_native_sampling({"native_coordinator_sampling": interrupted})
    adapter._native_decision_llm(view())
    assert forks[0].config.seed == 31
    llm.config = llm.config.model_copy(update={"seed": 0})
    with pytest.raises(ValueError, match="same coordinator sampling profile"):
        adapter.restore_native_sampling({"native_coordinator_sampling": interrupted})
    llm.invoke.assert_not_called()


def test_native_history_preserves_each_complete_batch_and_delivers_originals_once() -> (
    None
):
    context, ledger = RunContext(), EvidenceLedger()
    context.services["evidence"] = ledger
    context.services["assistant_instructions"] = "Answer briefly in Turkish."
    first = original(ledger, context, "The applicant must request the relief.")
    second = original(
        ledger, context, "Release requires completion of institution checks."
    )
    llm = model()
    history = "User: Preserve the two requested alternatives."
    adapter = ResearchModel(llm, context, lean_native_mode=True, history=history)
    batch1, batch2 = turn("read-law", [first]), turn("read-condition", [second])
    original_turns = [batch1.model_dump(mode="json"), batch2.model_dump(mode="json")]
    current = view(
        turns=[batch1, batch2],
        original_evidence=[first, second],
        research_state={"needs": [{"description": "UNNEEDED_BOARD" * 10000}]},
    )
    current = current.model_copy(
        update={
            "questions": ["First outcome?", "Second outcome?"],
            "facts": ["The user requested both outcomes."],
            "draft_to_repair": "Explain the request and later control [1] [2].",
            "publication_gap": {"summary": "Communicate the later control."},
        }
    )
    canonical_request = current.request
    canonical_questions = list(current.questions)
    adapter.decide(current)
    prompt = llm.invoke.call_args.kwargs["prompt"]
    first_user, preference_paragraph = first_user_payload(prompt[1])
    assert first_user == {
        "request": canonical_request,
        "questions": canonical_questions,
        "conversation": history,
        "assistant_instructions": "Answer briefly in Turkish.",
    }
    assert preference_paragraph == "\n\n" + llm_adapter.DEFAULT_RESPONSE_PREFERENCES
    assert current.request == canonical_request
    assert current.questions == canonical_questions
    assert context.services["assistant_instructions"] == "Answer briefly in Turkish."
    tool_results = [message for message in prompt if isinstance(message, ToolMessage)]
    assert [message.tool_call_id for message in tool_results] == [
        "read-law",
        "read-condition",
    ]
    assert last_payload(llm)["original_evidence"] == [first, second]
    assert last_payload(llm)["request"] == current.request
    assert last_payload(llm)["questions"] == current.questions
    assert last_payload(llm)["recorded_facts"] == current.facts
    assert last_payload(llm)["draft_to_repair"] == current.draft_to_repair
    assert last_payload(llm)["publication_gap"] == current.publication_gap
    for result, record in zip(tool_results, [first, second], strict=True):
        payload = json.loads(result.content)
        assert "original_evidence" not in payload
        assert payload["original_evidence_refs"] == [
            {key: value for key, value in record.items() if key != "text"}
        ]
        assert payload["evidence_ids"] == [record["citation"]]
        assert payload["outcome"] == {"status": "found", "data": {"provision": "168"}}
    assert [
        batch1.model_dump(mode="json"),
        batch2.model_dump(mode="json"),
    ] == original_turns
    assert "UNNEEDED_BOARD" not in json.dumps(
        [message.model_dump(mode="json") for message in prompt]
    )
    assert adapter.last_call_id is not None
    assert ledger.completely_delivered(adapter.last_call_id) == {1, 2}
    assert llm.invoke.call_count == 1
    first_prefix = prompt[:2]
    adapter.decide(current)
    assert llm.invoke.call_args.kwargs["prompt"][:2] == first_prefix
    assert llm.invoke.call_count == 2


@pytest.mark.parametrize("researcher", [False, True], ids=["coordinator", "researcher"])
def test_core_role_instructions_survive_response_preference_eviction(
    researcher: bool,
) -> None:
    context, ledger = RunContext(depth=int(researcher)), EvidenceLedger()
    context.services["evidence"] = ledger
    context.services["assistant_instructions"] = "Keep the requested units in Turkish."
    first = original(ledger, context, "The first condition must hold. " * 16)
    second = original(ledger, context, "The second condition also applies. " * 16)
    llm = model()
    adapter = ResearchModel(llm, context, lean_native_mode=True, token_counter=len)
    current = view(original_evidence=[first, second]).model_copy(
        update={
            "request": "İki sonucu ve belge şartlarını aynen açıkla.\nSonra süreyi belirt."
        }
    )
    canonical_request = current.request
    complete_prompt, tools, _ = adapter._fit_native_decision(current)
    question_content = complete_prompt[1].content
    assert isinstance(question_content, str)
    question, preference_paragraph = first_user_payload(complete_prompt[1])
    assert preference_paragraph == "\n\n" + llm_adapter.DEFAULT_RESPONSE_PREFERENCES
    expected_question = {"request": canonical_request}
    if not researcher:
        expected_question["assistant_instructions"] = (
            "Keep the requested units in Turkish."
        )
    assert question == expected_question
    _, offset = json.JSONDecoder().raw_decode(question_content)
    json_only = question_content[:offset]
    prompt_without_preferences = [
        complete_prompt[0],
        UserMessage(content=json_only),
        *complete_prompt[2:],
    ]
    required_input = adapter._input_cost(prompt_without_preferences, tools)
    llm.config.max_input_tokens = (required_input * 4 + 2) // 3
    ceiling, _ = adapter._limits(adapter._native_output_limit())
    assert required_input <= ceiling
    assert adapter._input_cost(complete_prompt, tools) > ceiling

    adapter.decide(current)
    prompt = llm.invoke.call_args.kwargs["prompt"]
    assert prompt[0].content == (
        llm_adapter.RESEARCHER_PROMPT if researcher else llm_adapter.COORDINATOR_PROMPT
    )
    first_content = prompt[1].content
    assert isinstance(first_content, str)
    delivered_question, preference_paragraph = first_user_payload(prompt[1])
    assert first_content == json_only
    assert delivered_question == expected_question
    assert preference_paragraph == ""
    assert current.request == canonical_request
    assert (
        context.services["assistant_instructions"]
        == "Keep the requested units in Turkish."
    )
    assert last_payload(llm)["original_evidence"] == [first, second]
    assert "original_evidence_omitted" not in last_payload(llm)
    assert adapter.last_call_id is not None
    assert ledger.completely_delivered(adapter.last_call_id) == {1, 2}
    assert llm.invoke.call_count == 1


def test_context_fitting_drops_transcript_atomically_but_restores_required_clause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(llm_adapter, "COORDINATOR_PROMPT", "Use original evidence.")
    context, ledger = RunContext(), EvidenceLedger()
    context.services["evidence"] = ledger
    first = original(ledger, context, "Only if institutional checks are complete.")
    second = original(ledger, context, "The security must also be acceptable.")
    llm = model(4000)
    adapter = ResearchModel(llm, context, lean_native_mode=True, token_counter=len)
    adapter.decide(
        view(
            turns=[
                turn("old", [first], extra="OLD_NAVIGATION" * 1000),
                turn("latest", [second]),
            ],
            original_evidence=[first, second],
            required_evidence_numbers=[1, 2],
        )
    )
    prompt = llm.invoke.call_args.kwargs["prompt"]
    assert [
        message.tool_call_id for message in prompt if isinstance(message, ToolMessage)
    ] == ["latest"]
    restored = last_payload(llm)["original_evidence"]
    assert restored == [first, second]
    assert "OLD_NAVIGATION" not in json.dumps(
        [message.model_dump(mode="json") for message in prompt]
    )
    assert adapter.last_call_id is not None
    assert ledger.completely_delivered(adapter.last_call_id) == {1, 2}
    request = llm.invoke.call_args.kwargs
    assert (
        adapter._input_cost(prompt, []) + request["max_tokens"]
        <= llm.config.max_input_tokens
    )


@pytest.mark.parametrize("full_first", [False, True])
def test_final_originals_replace_only_literally_covered_partial_ranges(
    full_first: bool,
) -> None:
    context, ledger = RunContext(), EvidenceLedger()
    context.services["evidence"] = ledger
    full = original(
        ledger, context, "Permission requires the notice and subsequent control."
    )
    text = cast(str, full["text"])
    partial = {
        **full,
        "text": text[11:34],
        "start_char": 11,
        "end_char": 34,
        "total_chars": len(text),
        "truncated": True,
    }
    records = [full, partial] if full_first else [partial, full]
    llm = model()
    adapter = ResearchModel(llm, context, lean_native_mode=True)
    adapter.decide(
        view(turns=[turn("read-parts", records)], original_evidence=[partial, full])
    )
    assert last_payload(llm)["original_evidence"] == [full]
    assert ledger.completely_delivered(adapter.last_call_id or "") == {1}
    assert llm.invoke.call_count == 1


def test_final_originals_keep_overlapping_ranges_without_reading_the_full_ledger_text() -> (
    None
):
    context, ledger = RunContext(), EvidenceLedger()
    context.services["evidence"] = ledger
    full = original(
        ledger,
        context,
        "First permission. Later notice. Final control. Separate exception.",
    )
    text = cast(str, full["text"])
    parts = [
        {
            **full,
            "text": text[start:end],
            "start_char": start,
            "end_char": end,
            "total_chars": len(text),
            "truncated": True,
        }
        for start, end in [(0, 25), (15, 45), (18, 23)]
    ]
    llm = model()
    adapter = ResearchModel(llm, context, lean_native_mode=True)
    adapter.decide(view(turns=[turn("read-parts", parts)], original_evidence=parts))
    assert last_payload(llm)["original_evidence"] == parts[:2]
    assert ledger.completely_delivered(adapter.last_call_id or "") == set()
    assert llm.invoke.call_count == 1


def test_final_originals_clear_only_current_ledger_verified_omissions() -> None:
    context, ledger = RunContext(), EvidenceLedger()
    context.services["evidence"] = ledger
    full = original(
        ledger, context, "A request requires notice followed by authority control."
    )
    unknown = {"citation": 2, "reason": "serialized_evidence_limit"}
    changed = {"citation": 1, "text_hash": "another-version", "reason": "old_context"}
    beyond = {"citation": 1, "start_char": 0, "end_char": 1000, "reason": "old_context"}
    llm = model()
    adapter = ResearchModel(llm, context, lean_native_mode=True)
    adapter.decide(
        view(
            turns=[turn("read-full", [full])],
            original_evidence_omitted=[{"citation": 1}, unknown, changed, beyond],
        )
    )
    assert last_payload(llm)["original_evidence"] == [full]
    assert last_payload(llm)["original_evidence_omitted"] == [unknown, changed, beyond]
    assert llm.invoke.call_count == 1


def test_physical_capacity_omission_is_explicit_and_never_changes_ledger_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(llm_adapter, "COORDINATOR_PROMPT", "Use originals.")
    context, ledger = RunContext(), EvidenceLedger()
    context.services["evidence"] = ledger
    required = original(ledger, context, "Required qualifying clause.")
    optional = original(ledger, context, "Older supplementary source. " * 150)
    llm = model(2500)
    adapter = ResearchModel(llm, context, lean_native_mode=True, token_counter=len)
    adapter.decide(
        view(original_evidence=[required, optional], required_evidence_numbers=[1])
    )
    payload = last_payload(llm)
    assert payload["original_evidence"] == [required]
    assert payload["original_evidence_omitted"][0]["citation"] == 2
    assert payload["original_evidence_omitted"][0]["reason"] == "physical_model_context"
    stored = ledger.get(2)
    assert stored is not None and stored.text == optional["text"]
    assert adapter.last_call_id is not None
    assert ledger.completely_delivered(adapter.last_call_id) == {1}


def test_required_original_cannot_be_clipped_to_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(llm_adapter, "COORDINATOR_PROMPT", "Use originals.")
    context, ledger = RunContext(), EvidenceLedger()
    required = original(ledger, context, "Complete required condition. " * 150)
    llm = model(1500)
    adapter = ResearchModel(llm, context, lean_native_mode=True, token_counter=len)
    with pytest.raises(RunStopped, match="required originals"):
        adapter.decide(
            view(original_evidence=[required], required_evidence_numbers=[1])
        )
    assert llm.invoke.call_count == 0


@pytest.mark.parametrize("lean", [False, True])
def test_lean_unlimited_context_defers_provider_timeout(lean: bool) -> None:
    context = RunContext(timeout_seconds=float("inf"))
    llm = model()
    ResearchModel(llm, context, lean_native_mode=lean).decide(view())
    assert llm.invoke.call_args.kwargs["timeout_override"] == (None if lean else 120)


@pytest.mark.parametrize(
    "arguments", ['{"article":"168","article":"169"}', '{"article":NaN}']
)
def test_invalid_historical_json_is_not_replayed(arguments: str) -> None:
    historical = turn("invalid", [])
    assert historical.assistant.tool_calls is not None
    historical.assistant.tool_calls[0].function.arguments = arguments
    llm = model()
    ResearchModel(llm, RunContext(), lean_native_mode=True).decide(
        view(turns=[historical])
    )
    assert not any(
        isinstance(message, ToolMessage)
        for message in llm.invoke.call_args.kwargs["prompt"]
    )


def test_orphan_turn_is_not_sent_and_failed_arguments_remain_actionable() -> None:
    orphan = turn("broken", [])
    orphan.results = []
    receipt = ToolReceipt(
        call=CapabilityCall(name="read_provision", call_id="broken", arguments={}),
        outcome=ToolOutcome(
            status=OutcomeStatus.INVALID, summary="Specify the exact article."
        ),
        elapsed_seconds=0,
    )
    llm = model()
    current = view(turns=[turn("valid", [])])
    current.turns.insert(0, orphan)
    current.receipts = [receipt]
    ResearchModel(llm, RunContext(), lean_native_mode=True).decide(current)
    prompt = llm.invoke.call_args.kwargs["prompt"]
    assert [
        message.tool_call_id for message in prompt if isinstance(message, ToolMessage)
    ] == ["valid"]
    assert (
        last_payload(llm)["failed_calls"][0]["summary"] == "Specify the exact article."
    )
    assert llm.invoke.call_count == 1


def test_native_mode_inherits_into_worker_without_changing_targeted_structured_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        llm_adapter, "RESEARCHER_PROMPT", "Research the assigned original."
    )
    context = RunContext()
    context.services["lean_native_mode"] = True
    context.services["assistant_instructions"] = "Coordinator-only instruction."
    llm = model()
    adapter = ResearchModel(llm, context.child())
    assert adapter.lean_native_mode
    current = view()
    canonical_request = current.request
    adapter.decide(current)
    assert (
        llm.invoke.call_args.kwargs["prompt"][0].content
        == "Research the assigned original."
    )
    assert llm.invoke.call_args.kwargs["structured_response_format"] is None
    first_user, preference_paragraph = first_user_payload(
        llm.invoke.call_args.kwargs["prompt"][1]
    )
    assert first_user == {"request": canonical_request}
    assert preference_paragraph == "\n\n" + llm_adapter.DEFAULT_RESPONSE_PREFERENCES
    assert current.request == canonical_request
    assert context.services["assistant_instructions"] == "Coordinator-only instruction."
    llm.invoke.return_value = ModelResponse(
        id="review",
        created="0",
        choice=Choice(
            message=Message(
                content=json.dumps(
                    {
                        "status": "supported",
                        "explanation": "The exact claim is supported.",
                        "required_conditions": [],
                        "missing_conditions": [],
                        "evidence_numbers": [1],
                        "safe_to_publish": True,
                    }
                )
            )
        ),
    )
    result = adapter.invoke_verification(
        "Check this exact claim.", '{"claim":"Rule [1]."}'
    )
    assert result.status == "supported"
    assert llm.invoke.call_args.kwargs["structured_response_format"] is not None
    assert llm.invoke.call_count == 2
    assert adapter.last_call_id is not None
    assert context.services.get("evidence") is None


def test_native_completed_answer_uses_provider_output_capacity_without_an_extra_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(llm_adapter, "get_llm_max_output_tokens", lambda *_: 16000)
    llm = model()
    llm.invoke.return_value.choice.finish_reason = "stop"
    adapter = ResearchModel(llm, RunContext(), lean_native_mode=True)
    result = adapter.decide(view())
    assert result.answer == "Rule [1]."
    assert llm.invoke.call_count == 1
    assert llm.invoke.call_args.kwargs["max_tokens"] == 16000
    assert adapter.last_finish_reason == "stop" and not adapter.last_response_truncated


@pytest.mark.parametrize("finish_reason", ["length", "max_tokens", "MAX_OUTPUT_TOKENS"])
def test_native_truncated_text_continues_from_same_originals_and_preserves_exact_prefix(
    monkeypatch: pytest.MonkeyPatch, finish_reason: str
) -> None:
    monkeypatch.setattr(llm_adapter, "get_llm_max_output_tokens", lambda *_: 12000)
    context, ledger = RunContext(), EvidenceLedger()
    context.services["evidence"] = ledger
    record = original(ledger, context, "The quoted condition and later outcome.")
    llm = model()
    prefix = "## 1\nThe original states “The quoted condition"
    suffix = " and later outcome.” [1]\n\n## 2\nThe second requested outcome [1]."
    llm.invoke.side_effect = [
        ModelResponse(
            id="cut",
            created="0",
            choice=Choice(
                finish_reason=finish_reason,
                message=Message(content=prefix),
            ),
        ),
        ModelResponse(
            id="complete",
            created="0",
            choice=Choice(
                finish_reason="stop",
                message=Message(content=suffix),
            ),
        ),
    ]
    adapter = ResearchModel(llm, context, lean_native_mode=True)
    result = adapter.decide(view(original_evidence=[record]))
    assert result.answer == prefix + suffix
    assert llm.invoke.call_count == 2
    continuation = llm.invoke.call_args.kwargs
    assert continuation["tools"] is None
    assert any(
        isinstance(message, AssistantMessage) and message.content == prefix
        for message in continuation["prompt"]
    )
    assert ledger.completely_delivered(adapter.last_call_id or "") == {1}
    assert not adapter.last_response_truncated


def test_native_truncated_action_never_executes_partial_arguments_when_no_capacity_remains(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(llm_adapter, "get_llm_max_output_tokens", lambda *_: 12000)
    llm = model()
    llm.invoke.return_value = ModelResponse(
        id="cut",
        created="0",
        choice=Choice(
            finish_reason="length",
            message=Message.model_validate(
                {
                    "tool_calls": [
                        {
                            "id": "unfinished",
                            "function": {
                                "name": "read_provision",
                                "arguments": '{"source_id":',
                            },
                        }
                    ]
                }
            ),
        ),
    )
    adapter = ResearchModel(llm, RunContext(), lean_native_mode=True)
    with pytest.raises(RunStopped, match="truncated action arguments"):
        adapter.decide(view())
    assert llm.invoke.call_count == 1 and adapter.last_response_truncated


def test_native_failed_continuation_never_returns_a_complete_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(llm_adapter, "get_llm_max_output_tokens", lambda *_: 12000)
    llm = model()
    llm.invoke.side_effect = [
        ModelResponse(
            id="cut",
            created="0",
            choice=Choice(
                finish_reason="length", message=Message(content="Unfinished claim")
            ),
        ),
        ModelResponse(
            id="empty",
            created="0",
            choice=Choice(finish_reason="stop", message=Message(content="")),
        ),
    ]
    adapter = ResearchModel(llm, RunContext(), lean_native_mode=True)
    with pytest.raises(RunStopped, match="did not complete"):
        adapter.decide(view())
    assert llm.invoke.call_count == 2 and adapter.last_response_truncated
