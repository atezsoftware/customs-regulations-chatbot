"""Native fitting reuses exact token counts without changing message contents."""

import pytest
from pydantic import JsonValue

from onyx.asv3.llm_adapter import ResearchModel
from onyx.asv3.models import RunContext
from onyx.llm.models import ChatCompletionMessage, SystemMessage, UserMessage
from tests.unit.onyx.asv3.test_native_model_adapter import model


@pytest.mark.parametrize(
    "profile,parallel,hosted,cached",
    [
        ("experimental", True, False, True),
        ("experimental", False, True, True),
        ("experimental", False, False, False),
        ("normal", True, False, False),
        ("deep", False, True, False),
    ],
)
def test_adapter_fitting_reuses_counts_only_in_parallel_or_hosted_serial_sessions(
    profile: str, parallel: bool, hosted: bool, cached: bool
) -> None:
    context = RunContext(
        services={
            "research_profile": profile,
            "experimental_parallel": parallel,
            "serial_session_diagnostics": hosted,
        }
    )
    counts: list[int] = []

    def counter(text: str) -> int:
        count = len(text.encode("utf-8")) + 7
        counts.append(count)
        return count

    adapter = ResearchModel(
        model(), context, token_counter=counter, lean_native_mode=True
    )
    prompt: list[ChatCompletionMessage] = [
        SystemMessage(content="Sabit yönerge ve özgün koşullar."),
        UserMessage(content="Mevcut koşula ilişkin farklı güncel veri."),
    ]
    tools: list[dict[str, JsonValue]] = [
        {
            "type": "function",
            "function": {
                "name": "read_original",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    first = adapter._input_cost(prompt, tools)
    initial_calls = len(counts)
    assert initial_calls == 3
    assert adapter._input_cost(prompt, tools) == first
    assert len(counts) == (initial_calls if cached else 2 * initial_calls)
    assert prompt[0].content == "Sabit yönerge ve özgün koşullar."
    assert prompt[1].content == "Mevcut koşula ilişkin farklı güncel veri."
    prompt[1] = UserMessage(content="Yeni koşula ilişkin güncel veri.")
    before_changed = len(counts)
    changed = adapter._input_cost(prompt, tools)
    assert len(counts) == before_changed + (1 if cached else initial_calls)
    context.services["research_profile"] = "normal"
    assert adapter._input_cost(prompt, tools) == changed
    assert (
        len(counts) == before_changed + (1 if cached else initial_calls) + initial_calls
    )
