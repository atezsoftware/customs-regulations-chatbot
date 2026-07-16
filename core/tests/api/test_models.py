import pytest
from pydantic import ValidationError

from fs_explorer_api.models import (
    ToolCallAction,
    Action,
    ToolCallArg,
    GoDeeperAction,
    StopAction,
    ToolBatchAction,
)


def test_tool_call_action_to_tool_args() -> None:
    tool_call_action = ToolCallAction(
        tool_name="glob",
        tool_input=[
            ToolCallArg(parameter_name="directory", parameter_value="tests/testfiles"),
            ToolCallArg(parameter_name="pattern", parameter_value="file?.*"),
        ],
    )
    assert tool_call_action.to_fn_args() == {
        "directory": "tests/testfiles",
        "pattern": "file?.*",
    }


def test_action_to_action_type() -> None:
    action = Action(
        action=ToolCallAction(
            tool_name="glob",
            tool_input=[
                ToolCallArg(
                    parameter_name="directory", parameter_value="tests/testfiles"
                ),
                ToolCallArg(parameter_name="pattern", parameter_value="file?.*"),
            ],
        ),
        reason="",
    )
    assert action.to_action_type() == "toolcall"
    action = Action(action=GoDeeperAction(directory="tests/testfiles/last"), reason="")
    assert action.to_action_type() == "godeeper"
    action = Action(action=StopAction(final_result="hello"), reason="")
    assert action.to_action_type() == "stop"


def test_tool_batch_action_accepts_two_or_three_calls() -> None:
    call = ToolCallAction(tool_name="list_indexed_documents", tool_input=[])

    action = Action(
        action=ToolBatchAction(tool_calls=[call, call]),
        reason="independent reads",
    )

    assert action.to_action_type() == "toolbatch"
    assert len(action.action.tool_calls) == 2


def test_tool_batch_action_rejects_more_than_three_calls() -> None:
    call = ToolCallAction(tool_name="list_indexed_documents", tool_input=[])

    with pytest.raises(ValidationError):
        ToolBatchAction(tool_calls=[call, call, call, call])
