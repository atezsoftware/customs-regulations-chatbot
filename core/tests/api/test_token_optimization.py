"""Deterministic accuracy/token benchmark for the agent loop.

Run directly to print the JSON benchmark report:
`uv run --package fs-explorer-api python tests/api/test_token_optimization.py`.
"""

from __future__ import annotations

import asyncio
import json
import math
from dataclasses import dataclass
from typing import Any, AsyncIterator

import pytest

from fs_explorer_api.agent import FsExplorerAgent, TOOLS
from fs_explorer_api.llm import ChatTurn, LLMUsage
from fs_explorer_api.models import (
    Action,
    ContextSummary,
    StopAction,
    ToolBatchAction,
    ToolCallAction,
    ToolCallArg,
)


EVIDENCE = {
    "DIRECT_RULE": "Transit süresi aşılırsa Kanun 241 uyarınca kademeli ceza uygulanabilir.",
    "EXCEPTION": "Belgelenmiş mücbir sebep varsa ceza uygulanmaz.",
    "CROSS_REFERENCE": "Gecikme anlaşılırsa en yakın gümrük idaresinden süre uzatımı istenir.",
}


def _estimated_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))


def _payload_tokens(history: list[ChatTurn], system_prompt: str) -> int:
    return _estimated_tokens(
        system_prompt + "\n" + "\n".join(turn.text for turn in history)
    )


def _tool_action(name: str, **kwargs: Any) -> Action:
    return Action(
        reason=f"Need {name}",
        action=ToolCallAction(
            tool_name=name,  # type: ignore[arg-type]
            tool_input=[
                ToolCallArg(parameter_name=key, parameter_value=value)
                for key, value in kwargs.items()
            ],
        ),
    )


def _batch_action(*actions: Action) -> Action:
    calls = [action.action for action in actions]
    assert all(isinstance(call, ToolCallAction) for call in calls)
    return Action(
        reason="Run independent retrievals together",
        action=ToolBatchAction(tool_calls=calls),  # type: ignore[arg-type]
    )


@dataclass
class Scenario:
    name: str
    question: str
    actions: list[Action]
    required_evidence: set[str]


class BenchmarkLLMClient:
    model = "benchmark-model"

    def __init__(self, actions: list[Action]) -> None:
        self.actions = list(actions)
        self._last_stream_usage: LLMUsage | None = None
        self.calls: list[dict[str, Any]] = []

    async def generate_structured(
        self,
        history: list[ChatTurn],
        system_prompt: str,
        schema,
        *,
        thinking_level: str | None = None,
    ):
        input_tokens = _payload_tokens(history, system_prompt)
        thinking_tokens = 100 if thinking_level in {"minimal", "low"} else 600
        self.calls.append(
            {
                "purpose": "summary" if schema is ContextSummary else "action",
                "input_tokens": input_tokens,
                "thinking_tokens": thinking_tokens,
                "thinking_level": thinking_level,
            }
        )
        if schema is ContextSummary:
            text = "\n".join(turn.text for turn in history)
            markers = [marker for marker in EVIDENCE if marker in text]
            summary = " ".join(f"{marker}: {EVIDENCE[marker]}" for marker in markers)
            return ContextSummary(
                summary=summary or "No material evidence yet."
            ), LLMUsage(
                input_tokens=input_tokens,
                output_tokens=_estimated_tokens(summary or "No material evidence yet."),
                thinking_tokens=thinking_tokens,
            )

        action = self.actions.pop(0)
        return action, LLMUsage(
            input_tokens=input_tokens,
            output_tokens=_estimated_tokens(action.model_dump_json()),
            thinking_tokens=thinking_tokens,
        )

    async def stream_text(
        self,
        history: list[ChatTurn],
        system_prompt: str,
        *,
        thinking_level: str | None = None,
    ) -> AsyncIterator[str]:
        input_tokens = _payload_tokens(history, system_prompt)
        thinking_tokens = 600 if thinking_level in {None, "high"} else 200
        text = "\n".join(turn.text for turn in history)
        present = [marker for marker in EVIDENCE if marker in text]
        answer = " ".join(EVIDENCE[marker] for marker in present)
        if present:
            answer += "\n\n## Sources\n- Gümrük Kanunu\n- Transit Rejimi Tebliği"
        self._last_stream_usage = LLMUsage(
            input_tokens=input_tokens,
            output_tokens=_estimated_tokens(answer),
            thinking_tokens=thinking_tokens,
        )
        self.calls.append(
            {
                "purpose": "final",
                "input_tokens": input_tokens,
                "thinking_tokens": thinking_tokens,
                "thinking_level": thinking_level,
            }
        )
        yield answer

    def last_stream_usage(self) -> LLMUsage | None:
        return self._last_stream_usage


SCENARIOS = [
    Scenario(
        name="direct_rule",
        question="Transit süresi aşılırsa ceza uygulanır mı?",
        actions=[
            _tool_action("semantic_search", query="transit süre aşımı ceza"),
            _tool_action("get_chunk_context", chunk_id="chunk_direct"),
            Action(reason="Enough evidence", action=StopAction(final_result="ready")),
        ],
        required_evidence={"DIRECT_RULE"},
    ),
    Scenario(
        name="exception",
        question="Mücbir sebepte transit cezası uygulanır mı?",
        actions=[
            _tool_action("semantic_search", query="transit mücbir sebep"),
            _tool_action("get_chunk_context", chunk_id="chunk_exception"),
            Action(reason="Enough evidence", action=StopAction(final_result="ready")),
        ],
        required_evidence={"EXCEPTION"},
    ),
    Scenario(
        name="cross_reference",
        question="Gecikme halinde ceza ve süre uzatımı birlikte nasıl işler?",
        actions=[
            _batch_action(
                _tool_action("semantic_search", query="transit gecikme ceza"),
                _tool_action("semantic_search", query="transit süre uzatımı başvuru"),
            ),
            _batch_action(
                _tool_action("get_chunk_context", chunk_id="chunk_direct"),
                _tool_action("get_chunk_context", chunk_id="chunk_cross"),
            ),
            Action(reason="Enough evidence", action=StopAction(final_result="ready")),
        ],
        required_evidence={"DIRECT_RULE", "CROSS_REFERENCE"},
    ),
]


TOOL_RESULTS = {
    "transit süre aşımı ceza": "chunk_id=chunk_direct doc_id=doc_direct DIRECT_RULE",
    "transit mücbir sebep": "chunk_id=chunk_exception doc_id=doc_exception EXCEPTION",
    "transit gecikme ceza": "chunk_id=chunk_direct doc_id=doc_direct DIRECT_RULE",
    "transit süre uzatımı başvuru": "chunk_id=chunk_cross doc_id=doc_cross CROSS_REFERENCE",
    "doc_direct": ("DIRECT_RULE " + EVIDENCE["DIRECT_RULE"] + " ") * 120,
    "doc_exception": ("EXCEPTION " + EVIDENCE["EXCEPTION"] + " ") * 120,
    "doc_cross": ("CROSS_REFERENCE " + EVIDENCE["CROSS_REFERENCE"] + " ") * 120,
    "chunk_direct": ("DIRECT_RULE " + EVIDENCE["DIRECT_RULE"] + " ") * 8,
    "chunk_exception": ("EXCEPTION " + EVIDENCE["EXCEPTION"] + " ") * 8,
    "chunk_cross": ("CROSS_REFERENCE " + EVIDENCE["CROSS_REFERENCE"] + " ") * 8,
}


async def _run_scenario(scenario: Scenario) -> dict[str, Any]:
    client = BenchmarkLLMClient(scenario.actions)
    agent = FsExplorerAgent(llm_client=client)
    agent.configure_task(scenario.question)
    answer = ""

    while True:
        result = await agent.take_action()
        assert result is not None
        action, action_type = result
        if action_type == "stop":
            async for chunk in agent.stream_final_answer(
                fallback_answer=action.action.final_result  # type: ignore[union-attr]
            ):
                answer += chunk
            break
        if isinstance(action.action, ToolBatchAction):
            await agent.call_tools(
                [
                    (call.tool_name, call.to_fn_args())
                    for call in action.action.tool_calls
                ]
            )
        else:
            assert isinstance(action.action, ToolCallAction)
            agent.call_tool(action.action.tool_name, action.action.to_fn_args())
        agent.configure_task("Use the tool result and continue the original task.")

    for marker in scenario.required_evidence:
        assert EVIDENCE[marker] in answer
    assert "## Sources" in answer

    return {
        "scenario": scenario.name,
        "accuracy": True,
        "api_calls": agent.token_usage.api_calls,
        "input_tokens": agent.token_usage.prompt_tokens,
        "output_tokens": agent.token_usage.completion_tokens,
        "thinking_tokens": agent.token_usage.thinking_tokens,
        "total_tokens": agent.token_usage.total_tokens,
        "context_summaries": agent.token_usage.context_summaries,
    }


async def run_benchmark() -> dict[str, Any]:
    originals = dict(TOOLS)
    TOOLS["semantic_search"] = lambda query, **_: TOOL_RESULTS[query]
    TOOLS["get_document"] = lambda doc_id, **_: TOOL_RESULTS[doc_id]
    TOOLS["get_chunk_context"] = lambda chunk_id, **_: TOOL_RESULTS[chunk_id]
    try:
        results = [await _run_scenario(scenario) for scenario in SCENARIOS]
    finally:
        TOOLS.clear()
        TOOLS.update(originals)

    totals = {
        key: sum(int(result[key]) for result in results)
        for key in (
            "api_calls",
            "input_tokens",
            "output_tokens",
            "thinking_tokens",
            "total_tokens",
            "context_summaries",
        )
    }
    return {
        "accuracy_passed": len(results),
        "accuracy_total": len(results),
        "totals": totals,
        "scenarios": results,
    }


@pytest.mark.asyncio
async def test_accuracy_and_token_benchmark() -> None:
    report = await run_benchmark()
    assert report["accuracy_passed"] == report["accuracy_total"] == 3


if __name__ == "__main__":
    print(json.dumps(asyncio.run(run_benchmark()), ensure_ascii=False, indent=2))
