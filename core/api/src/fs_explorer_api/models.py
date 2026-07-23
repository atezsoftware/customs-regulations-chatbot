"""
Pydantic models for FsExplorer agent actions.

This module defines the structured data models used to represent
the actions the agent can take during filesystem exploration.
"""

from pydantic import BaseModel, Field
from typing import TypeAlias, Literal, Any


# =============================================================================
# Type Aliases
# =============================================================================

Tools: TypeAlias = Literal[
    "read",
    "grep",
    "glob",
    "scan_folder",
    "preview_file",
    "parse_file",
    "semantic_search",
    "get_chunk_context",
    "get_document",
    "list_indexed_documents",
]
"""Available tool names that the agent can invoke."""

ActionType: TypeAlias = Literal["stop", "godeeper", "toolcall", "toolbatch", "askhuman"]
"""Types of actions the agent can take."""


# =============================================================================
# Action Models
# =============================================================================


class StopAction(BaseModel):
    """
    Action indicating the task is complete.

    Used when the agent has gathered enough information to provide
    a final answer to the user's query.
    """

    final_result: str = Field(
        description="Final result of the operation with the answer to the user's query"
    )


class AskHumanAction(BaseModel):
    """
    Action requesting clarification from the user.

    Used when the agent needs additional information or context
    to proceed with the task.
    """

    question: str = Field(description="Clarification question to ask the user")


class GoDeeperAction(BaseModel):
    """
    Action to navigate into a subdirectory.

    Used when the agent needs to explore a subdirectory
    to find relevant files.
    """

    directory: str = Field(description="Path to the directory to navigate into")


class ToolCallArg(BaseModel):
    """
    A single argument for a tool call.

    Represents a parameter name-value pair to pass to a tool.
    """

    parameter_name: str = Field(description="Name of the parameter")
    parameter_value: Any = Field(description="Value for the parameter")


class ToolCallAction(BaseModel):
    """
    Action to invoke a filesystem tool.

    Used when the agent needs to read files, search for patterns,
    or parse documents to gather information.
    """

    tool_name: Tools = Field(description="Name of the tool to invoke")
    tool_input: list[ToolCallArg] = Field(description="Arguments to pass to the tool")

    def to_fn_args(self) -> dict[str, Any]:
        """
        Convert tool input to a dictionary for function calls.

        Returns:
            Dictionary mapping parameter names to values.
        """
        return {arg.parameter_name: arg.parameter_value for arg in self.tool_input}


class ToolBatchAction(BaseModel):
    """Two or three independent tool calls planned in one model turn."""

    tool_calls: list[ToolCallAction] = Field(min_length=2, max_length=3)


class ContextSummary(BaseModel):
    """
    Compacted summary of an earlier stretch of an exploration run's history.

    Produced mid-run when the accumulated chat history approaches the
    model's context window, so older tool-call/reasoning turns can be
    replaced with a compact standalone paragraph instead of being resent
    verbatim on every subsequent call. See `FsExplorerAgent._maybe_summarize_history`.
    """

    summary: str = Field(
        description=(
            "Compact paragraph summarizing the exploration steps and tool "
            "results being replaced: concrete facts, document names/paths, "
            "article/section numbers, and findings relevant to the task. "
            "Does not answer the user's task itself."
        )
    )


class Action(BaseModel):
    """
    Container for an agent action with reasoning.

    Wraps any of the specific action types (stop, go deeper,
    tool call, ask human) along with the agent's explanation
    for why this action was chosen.
    """

    action: (
        ToolBatchAction | ToolCallAction | GoDeeperAction | StopAction | AskHumanAction
    ) = Field(description="The specific action to take")
    reason: str = Field(description="Explanation for why this action was chosen")

    def to_action_type(self) -> ActionType:
        """
        Get the type of this action.

        Returns:
            The action type string: "toolcall", "godeeper", "askhuman", or "stop".
        """
        if isinstance(self.action, ToolCallAction):
            return "toolcall"
        elif isinstance(self.action, ToolBatchAction):
            return "toolbatch"
        elif isinstance(self.action, GoDeeperAction):
            return "godeeper"
        elif isinstance(self.action, AskHumanAction):
            return "askhuman"
        else:
            return "stop"


# =============================================================================
# Benchmark Judge Model
# =============================================================================


class JudgmentResult(BaseModel):
    """Structured LLM-judge score for one benchmark candidate answer.

    Scored against a fixed rubric (see `benchmark_runner.JUDGE_SYSTEM_PROMPT`)
    on four 1-5 dimensions; the weighted `overall_score` is computed by the
    caller from these, not requested from the judge model directly, so the
    weighting stays server-controlled regardless of which judge model runs.

    Deliberately no `ge=`/`le=` bounds here (unlike the DB columns storing
    these scores, which do enforce 1-5): a Pydantic numeric constraint shows
    up as JSON Schema `minimum`/`maximum`, which OpenRouter's `strict: true`
    structured-output mode rejects the whole request for — every other
    schema in this module avoids numeric bounds for the same reason. Out-of
    range values are clamped defensively in `benchmark_runner.judge_answer`
    instead.
    """

    correctness: int = Field(
        description="Factual match to the reference answer/expected facts, 1-5"
    )
    groundedness: int = Field(
        description="How well claims are backed by cited sources, 1-5"
    )
    completeness: int = Field(
        description="Coverage of the question, including relevant exceptions, 1-5",
    )
    clarity: int = Field(
        description="How clear, direct, and actionable the answer is, 1-5"
    )
    rationale: str = Field(
        description="Short, specific explanation for the scores given"
    )
