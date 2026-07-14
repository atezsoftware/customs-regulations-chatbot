"""
Workflow orchestration for the FsExplorer agent.

This module defines the event-driven workflow that coordinates the agent's
exploration of the filesystem, handling tool calls, directory navigation,
and human interaction.
"""

import contextvars
import os

from workflows import Workflow, Context, step
from workflows.events import (
    StartEvent,
    StopEvent,
    Event,
    InputRequiredEvent,
    HumanResponseEvent,
)
from workflows.resource import Resource, ResourceManager
from pydantic import BaseModel
from typing import Annotated, AsyncGenerator, cast, Any

from .agent import FsExplorerAgent, OnLLMCall, describe_indexed_context
from .models import (
    GoDeeperAction,
    ToolCallAction,
    StopAction,
    AskHumanAction,
    Action,
    Tools,
)
from fs_explorer_shared.fs import describe_dir_content

# Per-asyncio-task fallback agent storage, consulted only by get_agent()'s
# factory below — which itself is only reached if a workflow was built
# without going through new_workflow() (see get_agent()'s docstring).
_AGENT_VAR: contextvars.ContextVar[FsExplorerAgent | None] = contextvars.ContextVar(
    "_AGENT_VAR", default=None
)


def get_agent() -> FsExplorerAgent:
    """Fallback factory backing the `Resource(get_agent)` annotation on every step.

    Production callers never actually reach this: `new_workflow()`
    constructs the agent explicitly from its arguments and pre-registers it
    into the run's own `ResourceManager` before any step executes, so
    `ResourceManager.get()` returns that pre-registered agent without ever
    invoking this factory. It still has to exist (and keep this exact
    `__qualname__`, used as the resource cache key both here and in
    `new_workflow()`/`get_run_agent()`) so the `Resource(get_agent)`
    annotations below have something to construct as a last resort if a
    `FsExplorerWorkflow` is ever built directly instead.

    This used to also read per-request model/temperature/hook config from
    module-level globals set via `set_agent_llm_config()`/
    `set_llm_call_hook()`. That was racy under concurrent requests: those
    setters and this factory's read of them could be arbitrarily far apart
    in time (a step's resource resolution can run in a different asyncio
    task, scheduled whenever the workflow engine gets to it), so a second
    concurrent request's `set_...()` call could land in between and hand
    the first request's agent the wrong model/temperature/hook. Since nothing
    reaches this fallback in practice, it just builds a default agent now —
    real per-request config always goes through `new_workflow()`'s
    arguments instead, which has no such window.
    """
    agent = _AGENT_VAR.get()
    if agent is None:
        agent = FsExplorerAgent()
        _AGENT_VAR.set(agent)
    return agent


def reset_agent() -> None:
    """Reset get_agent()'s fallback-path agent. See get_agent()'s docstring."""
    _AGENT_VAR.set(None)


def get_run_agent(resource_manager: ResourceManager) -> FsExplorerAgent:
    """Return the agent that a specific run's steps actually used.

    Must be passed the `ResourceManager` returned alongside that run's
    workflow by `new_workflow()`. Calling bare `get_agent()` again after
    `await handler` does NOT reliably return that same agent: workflow
    steps execute inside internal worker `asyncio.Task`s spawned by the
    `workflows` engine, so the `_AGENT_VAR.set(...)` a step's resource
    resolution performs is invisible to whichever task called
    `workflow.run(...)` — a later bare `get_agent()` call there sees an
    untouched (still-reset) contextvar and silently constructs a brand
    new, empty agent instead. `ResourceManager.resources` is a plain dict
    on a shared object, not contextvar-scoped, so reading the agent back
    out of it here is reliable regardless of which task populated it.
    """
    agent = resource_manager.get_all().get(get_agent.__qualname__)
    if agent is None:
        # No step ever resolved the resource (e.g. the run failed before
        # start_exploration ran) — fall back to a fresh, empty agent so
        # callers can still read `.token_usage` etc. without crashing.
        agent = FsExplorerAgent()
    return cast(FsExplorerAgent, agent)


class WorkflowState(BaseModel):
    """State maintained throughout the workflow execution."""

    initial_task: str = ""
    root_directory: str = "."
    current_directory: str = "."
    use_index: bool = False
    enable_semantic: bool = False
    enable_metadata: bool = False


class InputEvent(StartEvent):
    """Initial event containing the user's task."""

    task: str
    folder: str = "."
    use_index: bool = False
    enable_semantic: bool = False
    enable_metadata: bool = False


class GoDeeperEvent(Event):
    """Event triggered when navigating into a subdirectory."""

    directory: str
    reason: str


class ToolCallEvent(Event):
    """Event triggered when executing a tool."""

    tool_name: str
    tool_input: dict[str, Any]
    reason: str


class AskHumanEvent(InputRequiredEvent):
    """Event triggered when human input is required."""

    question: str
    reason: str


class HumanAnswerEvent(HumanResponseEvent):
    """Event containing the human's response."""

    response: str


class ExplorationEndEvent(StopEvent):
    """Event signaling the end of exploration."""

    final_result: str | None = None
    error: str | None = None


# Type alias for the union of possible workflow events
WorkflowEvent = ExplorationEndEvent | GoDeeperEvent | ToolCallEvent | AskHumanEvent


def _handle_action_result(
    action: Action,
    action_type: str,
    ctx: Context[WorkflowState],
) -> WorkflowEvent:
    """
    Convert an action result into the appropriate workflow event.

    This helper extracts the common logic for handling agent action results,
    reducing code duplication across workflow steps.

    Args:
        action: The action returned by the agent
        action_type: The type of action ("godeeper", "toolcall", "askhuman", "stop")
        ctx: The workflow context for state updates and event streaming

    Returns:
        The appropriate workflow event based on the action type
    """
    if action_type == "godeeper":
        godeeper = cast(GoDeeperAction, action.action)
        event = GoDeeperEvent(directory=godeeper.directory, reason=action.reason)
        ctx.write_event_to_stream(event)
        return event

    elif action_type == "toolcall":
        toolcall = cast(ToolCallAction, action.action)
        event = ToolCallEvent(
            tool_name=toolcall.tool_name,
            tool_input=toolcall.to_fn_args(),
            reason=action.reason,
        )
        ctx.write_event_to_stream(event)
        return event

    elif action_type == "askhuman":
        askhuman = cast(AskHumanAction, action.action)
        # InputRequiredEvent is written to the stream by default
        return AskHumanEvent(question=askhuman.question, reason=action.reason)

    else:  # stop
        stopaction = cast(StopAction, action.action)
        return ExplorationEndEvent(final_result=stopaction.final_result)


async def _process_agent_action(
    agent: FsExplorerAgent,
    ctx: Context[WorkflowState],
    update_directory: bool = False,
) -> WorkflowEvent:
    """
    Process the agent's next action and return the appropriate event.

    Args:
        agent: The agent instance
        ctx: The workflow context
        update_directory: Whether to update the current directory on godeeper action

    Returns:
        The appropriate workflow event
    """
    result = await agent.take_action()

    if result is None:
        return ExplorationEndEvent(error="Could not produce action to take")

    action, action_type = result

    # Update directory state if needed for godeeper actions
    if update_directory and action_type == "godeeper":
        godeeper = cast(GoDeeperAction, action.action)
        async with ctx.store.edit_state() as state:
            state.current_directory = godeeper.directory

    return _handle_action_result(action, action_type, ctx)


class FsExplorerWorkflow(Workflow):
    """
    Event-driven workflow for filesystem exploration.

    Coordinates the agent's actions through a series of steps:
    - start_exploration: Initial task processing
    - go_deeper_action: Directory navigation
    - tool_call_action: Tool execution
    - receive_human_answer: Human interaction handling
    """

    @step
    async def start_exploration(
        self,
        ev: InputEvent,
        ctx: Context[WorkflowState],
        agent: Annotated[FsExplorerAgent, Resource(get_agent)],
    ) -> WorkflowEvent:
        """Initialize exploration with the user's task."""
        root_directory = ev.folder if ev.use_index else os.path.abspath(ev.folder)
        if not ev.use_index and (
            not os.path.exists(root_directory) or not os.path.isdir(root_directory)
        ):
            return ExplorationEndEvent(error=f"No such directory: {root_directory}")

        async with ctx.store.edit_state() as state:
            state.initial_task = ev.task
            state.root_directory = root_directory
            state.current_directory = root_directory
            state.use_index = ev.use_index
            state.enable_semantic = ev.enable_semantic
            state.enable_metadata = ev.enable_metadata

        dirdescription = (
            describe_indexed_context()
            if ev.use_index
            else describe_dir_content(root_directory)
        )
        if ev.enable_semantic and ev.enable_metadata:
            index_hint = (
                "An index is available. Start with `semantic_search` (with optional "
                "filters) for fast retrieval, then use chunk-backed tools for deep dives."
            )
        elif ev.enable_semantic:
            index_hint = (
                "An index is available. Use `semantic_search` (no filters) for "
                "similarity search, then use chunk-backed tools for details."
            )
        elif ev.enable_metadata:
            index_hint = (
                "An index is available. Use `semantic_search` with metadata "
                "filters, then use chunk-backed tools for details."
            )
        else:
            index_hint = (
                "Prefer absolute paths from the directory listing when calling tools."
            )
        agent.configure_task(
            f"Given that the current directory ('{root_directory}') looks like this:\n\n"
            f"```text\n{dirdescription}\n```\n\n"
            f"And that the user is giving you this task: '{ev.task}', "
            f"what action should you take first? {index_hint}"
        )

        return await _process_agent_action(agent, ctx, update_directory=True)

    @step
    async def go_deeper_action(
        self,
        ev: GoDeeperEvent,
        ctx: Context[WorkflowState],
        agent: Annotated[FsExplorerAgent, Resource(get_agent)],
    ) -> WorkflowEvent:
        """Handle navigation into a subdirectory."""
        state = await ctx.store.get_state()
        dirdescription = (
            describe_indexed_context()
            if state.use_index
            else describe_dir_content(state.current_directory)
        )

        agent.configure_task(
            f"Given that the current directory ('{state.current_directory}') "
            f"looks like this:\n\n```text\n{dirdescription}\n```\n\n"
            f"And that the user is giving you this task: '{state.initial_task}', "
            f"what action should you take next?"
        )

        return await _process_agent_action(agent, ctx, update_directory=True)

    @step
    async def receive_human_answer(
        self,
        ev: HumanAnswerEvent,
        ctx: Context[WorkflowState],
        agent: Annotated[FsExplorerAgent, Resource(get_agent)],
    ) -> WorkflowEvent:
        """Process the human's response to a question."""
        state = await ctx.store.get_state()

        agent.configure_task(
            f"Human response to your question: {ev.response}\n\n"
            f"Based on it, proceed with your exploration based on the "
            f"original task: {state.initial_task}"
        )

        return await _process_agent_action(agent, ctx, update_directory=True)

    @step
    async def tool_call_action(
        self,
        ev: ToolCallEvent,
        ctx: Context[WorkflowState],
        agent: Annotated[FsExplorerAgent, Resource(get_agent)],
    ) -> WorkflowEvent:
        """Process the result of a tool call."""
        agent.call_tool(
            tool_name=cast(Tools, ev.tool_name),
            tool_input=ev.tool_input,
        )
        agent.configure_task(
            "Given the result from the tool call you just performed, "
            "what action should you take next?"
        )

        return await _process_agent_action(agent, ctx, update_directory=True)


# Workflow timeout for complex multi-document analysis (10 minutes) — raised
# alongside agent.py's _MAX_STEPS default (10 -> 40): a run now allowed to
# take many more genuine research steps needs proportionally more wall-clock
# room to actually reach them before this (not the step count) cuts it off.
WORKFLOW_TIMEOUT_SECONDS = 600


def new_workflow(
    *,
    model: str | None = None,
    temperature: float | None = None,
    on_llm_call: OnLLMCall | None = None,
) -> tuple[FsExplorerWorkflow, ResourceManager]:
    """Build a fresh workflow instance with its own ResourceManager and agent.

    `Resource(get_agent)` caches on the *Workflow instance's* resource
    manager (see workflows.resource.ResourceManager), not per-run. A
    module-level singleton workflow would therefore cache the same
    `FsExplorerAgent` — and its `_chat_history` — for the lifetime of the
    process, silently leaking one chat's history (and eventually its
    context-limit errors) into every other chat. Callers must use a fresh
    instance per request instead of reusing a shared one.

    The agent for this run is constructed here, directly from `model`/
    `temperature`/`on_llm_call`, and pre-registered into the fresh
    `ResourceManager` — *not* built lazily by `get_agent()`'s factory from
    module-level config globals. Two concurrent requests both calling a
    `set_agent_llm_config()`-style setter then relying on a shared
    factory to read it back later have a race window between the setter
    call and whenever a step's resource resolution actually runs (which
    can be a different asyncio task, scheduled arbitrarily later by the
    workflow engine) — a second request's setter call landing in that
    window would hand the first request's agent the wrong config. Passing
    the config here and registering the agent before any step runs closes
    that window entirely: `get_agent()`'s factory is never even invoked
    for a workflow built this way.

    Returns the `ResourceManager` too — callers need it after the run
    completes to look up the agent the run actually used, via
    `get_run_agent()`. See that function's docstring for why a plain
    `get_agent()` call doesn't work for this.
    """
    resource_manager = ResourceManager()
    agent = FsExplorerAgent(
        model=model, temperature=temperature, on_llm_call=on_llm_call
    )
    resource_manager.resources[get_agent.__qualname__] = agent
    workflow = FsExplorerWorkflow(
        timeout=WORKFLOW_TIMEOUT_SECONDS, resource_manager=resource_manager
    )
    return workflow, resource_manager


async def resume_agent_run(
    agent: FsExplorerAgent,
    *,
    use_index: bool,
    current_directory: str,
    initial_task: str,
) -> AsyncGenerator[WorkflowEvent, str | None]:
    """Continue an already-in-progress agent's decision loop directly.

    Used only to resume a run whose original `/ws/explore` connection was
    lost or explicitly stopped mid-run (see `runs.py`): `agent` already
    carries its accumulated `_chat_history`/`_step_count` from before the
    interruption, so this calls `take_action()` again exactly as
    `tool_call_action`/`go_deeper_action`/`receive_human_answer` would have
    — it does not go through the `workflows` engine/`InputEvent` at all
    (re-entering via `InputEvent` would re-run `start_exploration` and
    prepend a second "what should you do first" prompt on top of history
    that already has an answer to that question). It yields the same
    `WorkflowEvent` subclasses the engine-driven fresh-run path yields, so
    the caller can share event-translation code between both paths.

    For an `AskHumanEvent`, advance this generator with `asend(response)`
    (the human's answer text) instead of a plain `__anext__()`/`asend(None)`
    — mirrors how the `workflows` engine threads `HumanResponseEvent` back
    into `receive_human_answer`, just without that engine in the loop.
    """
    directory = current_directory
    human_response: str | None = None

    while True:
        if human_response is not None:
            agent.configure_task(
                f"Human response to your question: {human_response}\n\n"
                f"Based on it, proceed with your exploration based on the "
                f"original task: {initial_task}"
            )
            human_response = None

        result = await agent.take_action()
        if result is None:
            yield ExplorationEndEvent(error="Could not produce action to take")
            return

        action, action_type = result

        if action_type == "godeeper":
            godeeper = cast(GoDeeperAction, action.action)
            directory = godeeper.directory
            yield GoDeeperEvent(directory=godeeper.directory, reason=action.reason)
            dirdescription = (
                describe_indexed_context()
                if use_index
                else describe_dir_content(directory)
            )
            agent.configure_task(
                f"Given that the current directory ('{directory}') "
                f"looks like this:\n\n```text\n{dirdescription}\n```\n\n"
                f"And that the user is giving you this task: '{initial_task}', "
                f"what action should you take next?"
            )

        elif action_type == "toolcall":
            toolcall = cast(ToolCallAction, action.action)
            tool_input = toolcall.to_fn_args()
            yield ToolCallEvent(
                tool_name=toolcall.tool_name,
                tool_input=tool_input,
                reason=action.reason,
            )
            agent.call_tool(tool_name=toolcall.tool_name, tool_input=tool_input)
            agent.configure_task(
                "Given the result from the tool call you just performed, "
                "what action should you take next?"
            )

        elif action_type == "askhuman":
            askhuman = cast(AskHumanAction, action.action)
            received = yield AskHumanEvent(
                question=askhuman.question, reason=action.reason
            )
            human_response = received if isinstance(received, str) else ""

        else:  # stop
            stopaction = cast(StopAction, action.action)
            yield ExplorationEndEvent(final_result=stopaction.final_result)
            return


# Kept for the CLI (single-shot process, one workflow run per invocation —
# no cross-request caching risk) and for tests that import `workflow`
# directly. Long-lived server processes must use `new_workflow()` instead.
workflow, _default_resource_manager = new_workflow()
