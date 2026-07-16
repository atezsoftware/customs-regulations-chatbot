"""Tests for workflow.py's per-run agent construction and isolation."""

import os

import pytest
from unittest.mock import patch

from fs_explorer_api.agent import FsExplorerAgent
from fs_explorer_api.llm import LLMUsage
from fs_explorer_api.models import (
    Action,
    AskHumanAction,
    StopAction,
    ToolCallAction,
    ToolCallArg,
    ToolBatchAction,
)
from fs_explorer_api.workflow import (
    AskHumanEvent,
    ExplorationEndEvent,
    ToolCallEvent,
    ToolBatchEvent,
    get_run_agent,
    new_workflow,
    resume_agent_run,
)


@patch.dict(os.environ, {"GOOGLE_API_KEY": "test-api-key"})
class TestNewWorkflowAgentConfig:
    """`new_workflow()` must bind model/temperature/hook per call, not via
    module-level globals — see its docstring for the concurrent-request
    race that design avoids.
    """

    def test_agent_is_constructed_with_given_config(self) -> None:
        _workflow, resource_manager = new_workflow(model="model-a", temperature=0.3)

        agent = get_run_agent(resource_manager)

        assert agent._llm.model == "model-a"
        assert agent._llm.temperature == 0.3

    def test_two_calls_do_not_leak_config_into_each_other(self) -> None:
        # Simulates two requests whose new_workflow() calls interleave:
        # under the old set_agent_llm_config()-then-get_agent() design, the
        # second call here would have overwritten the module globals the
        # first request's agent hadn't been constructed from yet. Now each
        # call constructs and registers its own agent immediately, with no
        # shared mutable state in between.
        _workflow_a, resource_manager_a = new_workflow(model="model-a")
        _workflow_b, resource_manager_b = new_workflow(model="model-b")

        agent_a = get_run_agent(resource_manager_a)
        agent_b = get_run_agent(resource_manager_b)

        assert agent_a._llm.model == "model-a"
        assert agent_b._llm.model == "model-b"
        assert agent_a is not agent_b

    def test_default_config_when_omitted(self) -> None:
        from fs_explorer_api.llm.gemini import DEFAULT_GEMINI_MODEL

        _workflow, resource_manager = new_workflow()

        agent = get_run_agent(resource_manager)

        assert agent._llm.model == DEFAULT_GEMINI_MODEL


class _QueuedLLMClient:
    """LLMClient whose `generate_structured` responses come from a fixed queue.

    Used to drive `resume_agent_run` deterministically without a real model
    call, mirroring the interrupted-then-resumed agent it's designed for:
    the queue simulates "what the model decides next," one call at a time.
    """

    def __init__(self, actions: list[Action]) -> None:
        self.model = "queued"
        self.temperature = None
        self._actions = list(actions)
        self.calls = 0

    async def generate_structured(
        self, history, system_prompt, schema, *, thinking_level=None
    ):
        self.calls += 1
        action = self._actions.pop(0)
        return action, LLMUsage(input_tokens=100, output_tokens=10)

    async def stream_text(self, history, system_prompt, *, thinking_level=None):
        return
        yield ""  # pragma: no cover - makes this an async generator

    def last_stream_usage(self):
        return None


async def _drain(gen, *, human_responses: list[str] | None = None):
    """Drive an async generator with asend(), feeding queued human
    responses back in whenever an AskHumanEvent is yielded."""
    responses = list(human_responses or [])
    events = []
    send_value = None
    while True:
        try:
            event = await gen.asend(send_value)
        except StopAsyncIteration:
            break
        events.append(event)
        send_value = responses.pop(0) if isinstance(event, AskHumanEvent) else None
    return events


@patch.dict(os.environ, {"GOOGLE_API_KEY": "test-api-key"})
class TestResumeAgentRun:
    """`resume_agent_run` continues an already-in-progress agent directly,
    without going through `InputEvent`/`start_exploration` again — see its
    docstring in workflow.py for why re-entering via a fresh `InputEvent`
    would be wrong (it would prepend a second "what should you do first"
    prompt on top of history that already has an answer to that question).
    """

    @pytest.mark.asyncio
    async def test_continues_from_existing_history_to_a_tool_call_then_stop(
        self,
    ) -> None:
        llm_client = _QueuedLLMClient(
            [
                Action(
                    action=ToolCallAction(
                        tool_name="glob",
                        tool_input=[
                            ToolCallArg(
                                parameter_name="directory", parameter_value="."
                            ),
                            ToolCallArg(
                                parameter_name="pattern", parameter_value="*.md"
                            ),
                        ],
                    ),
                    reason="look around",
                ),
                Action(action=StopAction(final_result="all done"), reason="finished"),
            ]
        )
        agent = FsExplorerAgent(llm_client=llm_client)
        # Simulates history already accumulated by a previous, interrupted
        # connection — resume_agent_run must not touch or duplicate this.
        agent.configure_task("Given ... what should you do first?")

        gen = resume_agent_run(
            agent,
            use_index=False,
            current_directory=".",
            initial_task="find the readme",
        )
        events = await _drain(gen)

        assert len(events) == 2
        assert isinstance(events[0], ToolCallEvent)
        assert events[0].tool_name == "glob"
        assert isinstance(events[1], ExplorationEndEvent)
        assert events[1].final_result == "all done"
        assert llm_client.calls == 2
        # The pre-existing turn is untouched — no second "what should you
        # do first" prompt was prepended ahead of it.
        assert agent._chat_history[0].text == "Given ... what should you do first?"

    @pytest.mark.asyncio
    async def test_ask_human_round_trip_via_asend(self) -> None:
        llm_client = _QueuedLLMClient(
            [
                Action(
                    action=AskHumanAction(question="Which document?"),
                    reason="need clarification",
                ),
                Action(action=StopAction(final_result="answered"), reason="finished"),
            ]
        )
        agent = FsExplorerAgent(llm_client=llm_client)
        agent.configure_task("Given ... what should you do first?")

        gen = resume_agent_run(
            agent, use_index=False, current_directory=".", initial_task="find it"
        )
        events = await _drain(gen, human_responses=["the TIR document"])

        assert isinstance(events[0], AskHumanEvent)
        assert isinstance(events[1], ExplorationEndEvent)
        assert events[1].final_result == "answered"
        # The human's answer was threaded back into history before the
        # next take_action() call, exactly like receive_human_answer does.
        assert any(
            "the TIR document" in turn.text
            for turn in agent._chat_history
            if turn.role == "user"
        )

    @pytest.mark.asyncio
    async def test_batch_tool_calls_share_one_planning_round_trip(self) -> None:
        calls = [
            ToolCallAction(
                tool_name="glob",
                tool_input=[
                    ToolCallArg(
                        parameter_name="directory", parameter_value="tests/testfiles"
                    ),
                    ToolCallArg(parameter_name="pattern", parameter_value="file1.*"),
                ],
            ),
            ToolCallAction(
                tool_name="glob",
                tool_input=[
                    ToolCallArg(
                        parameter_name="directory", parameter_value="tests/testfiles"
                    ),
                    ToolCallArg(parameter_name="pattern", parameter_value="file2.*"),
                ],
            ),
        ]
        llm_client = _QueuedLLMClient(
            [
                Action(
                    action=ToolBatchAction(tool_calls=calls),
                    reason="independent searches",
                ),
                Action(action=StopAction(final_result="done"), reason="finished"),
            ]
        )
        agent = FsExplorerAgent(llm_client=llm_client)
        agent.configure_task("find both files")

        events = await _drain(
            resume_agent_run(
                agent,
                use_index=False,
                current_directory=".",
                initial_task="find both files",
            )
        )

        assert isinstance(events[0], ToolBatchEvent)
        assert len(events[0].tool_calls) == 2
        assert isinstance(events[1], ExplorationEndEvent)
        assert llm_client.calls == 2
        assert (
            sum("Batch tool results" in turn.text for turn in agent._chat_history) == 1
        )
