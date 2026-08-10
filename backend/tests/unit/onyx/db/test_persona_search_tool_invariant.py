from unittest.mock import MagicMock

from onyx.db.persona import _normalize_persona_tools_for_document_sets


def _tool(tool_id: int, in_code_tool_id: str | None) -> MagicMock:
    return MagicMock(id=tool_id, in_code_tool_id=in_code_tool_id)


def test_create_with_document_sets_adds_search_tool() -> None:
    session = MagicMock()
    search_tool = _tool(1, "SearchTool")
    session.scalar.return_value = search_tool

    result = _normalize_persona_tools_for_document_sets(
        db_session=session,
        explicit_tools=[],
        existing_tools=[],
        document_sets=[MagicMock()],
    )

    assert result == [search_tool]
    session.scalar.assert_called_once()


def test_update_with_omitted_tools_repairs_legacy_scoped_persona() -> None:
    session = MagicMock()
    unrelated_tool = _tool(2, "MemoryTool")
    search_tool = _tool(1, "SearchTool")
    session.scalar.return_value = search_tool

    result = _normalize_persona_tools_for_document_sets(
        db_session=session,
        explicit_tools=None,
        existing_tools=[unrelated_tool],
        document_sets=[MagicMock()],
    )

    assert result == [unrelated_tool, search_tool]


def test_update_cannot_remove_search_while_document_sets_remain() -> None:
    session = MagicMock()
    search_tool = _tool(1, "SearchTool")
    session.scalar.return_value = search_tool

    result = _normalize_persona_tools_for_document_sets(
        db_session=session,
        explicit_tools=[],
        existing_tools=[search_tool],
        document_sets=[MagicMock()],
    )

    assert result == [search_tool]


def test_unscoped_update_preserves_omitted_tools() -> None:
    session = MagicMock()

    result = _normalize_persona_tools_for_document_sets(
        db_session=session,
        explicit_tools=None,
        existing_tools=[_tool(2, "MemoryTool")],
        document_sets=[],
    )

    assert result is None
    session.scalar.assert_not_called()
