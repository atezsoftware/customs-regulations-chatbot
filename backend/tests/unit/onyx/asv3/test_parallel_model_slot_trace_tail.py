"""Completed provider calls release scoped admission before trace persistence."""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock

import pytest

from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.llm_adapter import ResearchModel
from onyx.asv3.models import RunContext, SharedBudget
from onyx.llm.model_response import Choice, Message, ModelResponse
from onyx.llm.models import ChatCompletionMessage, UserMessage
from onyx.tracing.flows import LLMFlow
from onyx.tracing.framework.span_data import GenerationSpanData
from tests.unit.onyx.asv3.test_model_adapter import scripted_model
from tests.unit.onyx.asv3.test_shared_originals import full_record, original


def contexts(
    profile: str, parallel: bool, hosted: bool, *, slots: int = 1
) -> tuple[RunContext, RunContext, EvidenceLedger]:
    budget = SharedBudget(max_inflight_models=slots)
    ledger = EvidenceLedger()
    services: dict[str, object] = {
        "research_profile": profile,
        "experimental_parallel": parallel,
        "evidence": ledger,
    }
    if hosted:
        services.update(serial_session_diagnostics=True, lean_native_mode=True)
    first = RunContext(services={**services, "task_id": "first"}, budget=budget)
    second = RunContext(
        run_id=first.run_id,
        services={**services, "task_id": "second"},
        budget=budget,
    )
    ledger.add([original("own", "The complete operative original.")], first)
    return first, second, ledger


def response() -> ModelResponse:
    return ModelResponse(
        id="selected-response",
        created="0",
        choice=Choice(message=Message(content="Exact selected answer [1].")),
    )


@pytest.mark.parametrize(
    "profile,parallel,hosted,releases_early",
    [
        ("experimental", True, False, True),
        ("experimental", False, True, True),
        ("experimental", False, False, False),
        ("normal", True, False, False),
        ("deep", True, False, False),
    ],
)
def test_actual_sibling_provider_can_enter_while_first_trace_tail_is_blocked(
    monkeypatch: pytest.MonkeyPatch,
    profile: str,
    parallel: bool,
    hosted: bool,
    releases_early: bool,
) -> None:
    first, second, ledger = contexts(profile, parallel, hosted)
    selected = scripted_model()
    prompt: list[ChatCompletionMessage] = [
        UserMessage(content=json.dumps({"original_evidence": [full_record(ledger, 1)]}))
    ]
    tools: list[dict[str, Any]] = [
        {"type": "function", "function": {"name": "read", "parameters": {}}}
    ]
    first_tail = threading.Event()
    release_tail = threading.Event()
    second_provider = threading.Event()
    lock = threading.Lock()
    spans: list[MagicMock] = []
    provider_calls: list[dict[str, Any]] = []
    errors: list[BaseException] = []
    results: list[ModelResponse] = []

    @contextmanager
    def generation(*args: Any) -> Iterator[MagicMock]:
        with lock:
            index = len(spans)
            span = MagicMock()
            span.span_id = f"physical-call-{index}"
            span.span_data = GenerationSpanData(
                model=args[0].config.model_name,
                input=args[2],
                tools=args[3],
            )
            spans.append(span)
        yield span
        if index == 0:
            first_tail.set()
            assert release_tail.wait(3)

    def invoke(**kwargs: Any) -> ModelResponse:
        with lock:
            provider_calls.append(kwargs)
            if len(provider_calls) == 2:
                second_provider.set()
        return response()

    monkeypatch.setattr("onyx.asv3.llm_adapter.llm_generation_span", generation)
    selected.invoke.side_effect = invoke
    models = [
        ResearchModel(selected, context, lean_native_mode=True)
        for context in (first, second)
    ]

    def run(model: ResearchModel) -> None:
        try:
            results.append(
                model._invoke_once(
                    prompt,
                    tools,
                    LLMFlow.ASV3_RESEARCHER,
                    max_tokens=100,
                    research=False,
                )
            )
        except BaseException as error:
            errors.append(error)

    threads = [threading.Thread(target=run, args=(model,)) for model in models]
    threads[0].start()
    try:
        assert first_tail.wait(3)
        assert ledger.completely_delivered("physical-call-0") == {1}
        threads[1].start()
        assert second_provider.wait(0.2) is releases_early
        assert threads[0].is_alive()
    finally:
        release_tail.set()
        for thread in threads:
            if thread.ident is not None:
                thread.join(3)
    assert not any(thread.is_alive() for thread in threads)
    assert errors == [] and results == [response(), response()]
    assert selected.invoke.call_count == 2
    assert provider_calls[0] == provider_calls[1]
    assert provider_calls[0]["prompt"] == prompt
    assert provider_calls[0]["tools"] == tools
    for index, model in enumerate(models):
        call_id = model.last_call_id
        assert isinstance(call_id, str) and call_id == f"physical-call-{index}"
        assert model.context.services["last_model_call_id"] == call_id
        assert ledger.completely_delivered(call_id) == {1}
        assert spans[index].span_data.input == prompt
        assert spans[index].span_data.tools == tools
        assert spans[index].span_data.output == [
            {"role": "assistant", "content": "Exact selected answer [1]."}
        ]


@pytest.mark.parametrize("hosted", [False, True])
def test_scoped_provider_admission_still_respects_shared_concurrency_cap(
    hosted: bool,
) -> None:
    first, _, _ = contexts("experimental", not hosted, hosted, slots=2)
    selected = scripted_model()
    lock = threading.Lock()
    release = threading.Event()
    active = peak = 0
    errors: list[BaseException] = []

    def invoke(**_kwargs: Any) -> ModelResponse:
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
            if active == 2:
                release.set()
        assert release.wait(3)
        with lock:
            active -= 1
        return response()

    selected.invoke.side_effect = invoke

    def run(index: int) -> None:
        try:
            context = RunContext(
                run_id=first.run_id,
                services={**first.services, "task_id": f"owned-{index}"},
                budget=first.budget,
            )
            ResearchModel(selected, context, lean_native_mode=True)._invoke_once(
                [UserMessage(content="Exact request")],
                [],
                LLMFlow.ASV3_RESEARCHER,
                max_tokens=100,
                research=False,
            )
        except BaseException as error:
            errors.append(error)

    threads = [threading.Thread(target=run, args=(index,)) for index in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(3)
    assert errors == [] and not any(thread.is_alive() for thread in threads)
    assert peak == 2 and active == 0 and selected.invoke.call_count == 6


@pytest.mark.parametrize("failure", ["span_enter", "provider", "span_exit"])
def test_scoped_resource_errors_do_not_leak_or_double_release_slot(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    context, _, _ = contexts("experimental", True, False)
    selected = scripted_model()
    error = RuntimeError(failure)

    @contextmanager
    def generation(*_args: Any) -> Iterator[MagicMock]:
        if failure == "span_enter":
            raise error
        span = MagicMock()
        span.span_id = "physical-call"
        span.span_data = GenerationSpanData(model="selected-model")
        yield span
        if failure == "span_exit":
            raise error

    monkeypatch.setattr("onyx.asv3.llm_adapter.llm_generation_span", generation)
    selected.invoke.side_effect = error if failure == "provider" else None
    selected.invoke.return_value = response()
    with pytest.raises(RuntimeError) as caught:
        ResearchModel(selected, context, lean_native_mode=True)._invoke_once(
            [UserMessage(content="Exact request")],
            [],
            LLMFlow.ASV3_RESEARCHER,
            max_tokens=100,
            research=False,
        )
    assert caught.value is error
    assert context.budget.model_slots.acquire(blocking=False)
    assert not context.budget.model_slots.acquire(blocking=False)
    context.budget.model_slots.release()
