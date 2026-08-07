from unittest.mock import MagicMock, patch

from onyx.configs.constants import DEFAULT_PERSONA_ID
from onyx.db.persona import get_effective_persona_tools, merge_persona_tools


def _tool(tool_id: int, name: str) -> MagicMock:
    tool = MagicMock()
    tool.id = tool_id
    tool.name = name
    return tool


def test_merge_persona_tools_inherits_default_first_and_deduplicates_by_id() -> None:
    default_search = _tool(1, "internal_search")
    default_reader = _tool(2, "read_file")
    directly_assigned_reader = _tool(2, "read_file")
    custom_action = _tool(3, "custom_action")

    merged = merge_persona_tools(
        default_tools=[default_search, default_reader],
        persona_tools=[directly_assigned_reader, custom_action],
    )

    assert [tool.id for tool in merged] == [1, 2, 3]
    assert merged[1] is default_reader


def test_default_persona_keeps_only_its_direct_tools() -> None:
    default_search = _tool(1, "internal_search")
    persona = MagicMock(id=DEFAULT_PERSONA_ID, tools=[default_search])

    with patch("onyx.db.persona.get_default_behavior_persona") as get_default:
        effective = get_effective_persona_tools(persona, MagicMock())

    assert effective == [default_search]
    get_default.assert_not_called()


def test_custom_persona_reads_current_default_tools_at_runtime() -> None:
    default_search = _tool(1, "internal_search")
    custom_action = _tool(3, "custom_action")
    persona = MagicMock(id=7, tools=[custom_action])
    default_persona = MagicMock(tools=[default_search])
    session = MagicMock()

    with patch(
        "onyx.db.persona.get_default_behavior_persona",
        return_value=default_persona,
    ) as get_default:
        effective = get_effective_persona_tools(persona, session)

    assert [tool.id for tool in effective] == [1, 3]
    get_default.assert_called_once_with(
        db_session=session,
        eager_load_for_tools=True,
    )
