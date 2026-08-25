"""Tests for tool construction when DISABLE_VECTOR_DB is True.

Verifies that:
- SearchTool.is_available() returns False when vector DB is disabled
- OpenURLTool.is_available() returns True when vector DB is disabled (crawl-only)
- The force-add SearchTool block is suppressed when DISABLE_VECTOR_DB
- FileReaderTool.is_available() returns True when vector DB is disabled
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from onyx.configs.constants import DEFAULT_PERSONA_ID
from onyx.context.search.models import BaseFilters
from onyx.db.enums import UserFileProjectionRepairStatus, UserFileStatus
from onyx.error_handling.exceptions import OnyxError
from onyx.tools.models import SearchToolUsage
from onyx.tools.tool_constructor import (
    SearchToolConfig,
    _construct_tools_impl,
    _resolve_document_set_file_ids,
    construct_tools,
)
from onyx.tools.tool_implementations.file_reader.file_reader_tool import FileReaderTool
from onyx.tools.tool_implementations.search.search_tool import SearchTool

APP_CONFIGS_MODULE = "onyx.configs.app_configs"
FILE_READER_MODULE = "onyx.tools.tool_implementations.file_reader.file_reader_tool"
OPEN_URL_MODULE = "onyx.tools.tool_implementations.open_url.open_url_tool"


# ------------------------------------------------------------------
# SearchTool.is_available()
# ------------------------------------------------------------------


class TestSearchToolAvailability:
    @patch(f"{APP_CONFIGS_MODULE}.DISABLE_VECTOR_DB", True)
    def test_unavailable_when_vector_db_disabled(self) -> None:
        from onyx.tools.tool_implementations.search.search_tool import SearchTool

        assert SearchTool.is_available(MagicMock()) is False

    @patch("onyx.db.connector.check_user_files_exist", return_value=True)
    @patch(
        "onyx.tools.tool_implementations.search.search_tool.check_federated_connectors_exist",
        return_value=False,
    )
    @patch(
        "onyx.tools.tool_implementations.search.search_tool.check_connectors_exist",
        return_value=False,
    )
    @patch(f"{APP_CONFIGS_MODULE}.DISABLE_VECTOR_DB", False)
    def test_available_when_vector_db_enabled_and_files_exist(
        self,
        mock_connectors: MagicMock,  # noqa: ARG002
        mock_federated: MagicMock,  # noqa: ARG002
        mock_user_files: MagicMock,  # noqa: ARG002
    ) -> None:
        from onyx.tools.tool_implementations.search.search_tool import SearchTool

        assert SearchTool.is_available(MagicMock()) is True


# ------------------------------------------------------------------
# OpenURLTool.is_available()
# ------------------------------------------------------------------


class TestOpenURLToolAvailability:
    @patch(f"{OPEN_URL_MODULE}.DISABLE_VECTOR_DB", True)
    def test_available_when_vector_db_disabled(self) -> None:
        from onyx.tools.tool_implementations.open_url.open_url_tool import OpenURLTool

        assert OpenURLTool.is_available(MagicMock()) is True

    @patch(f"{OPEN_URL_MODULE}.DISABLE_VECTOR_DB", False)
    def test_available_when_vector_db_enabled(self) -> None:
        from onyx.tools.tool_implementations.open_url.open_url_tool import OpenURLTool

        assert OpenURLTool.is_available(MagicMock()) is True


# ------------------------------------------------------------------
# FileReaderTool.is_available()
# ------------------------------------------------------------------


class TestFileReaderToolAvailability:
    @patch(f"{FILE_READER_MODULE}.DISABLE_VECTOR_DB", True)
    def test_available_when_vector_db_disabled(self) -> None:
        assert FileReaderTool.is_available(MagicMock()) is True

    @patch(f"{FILE_READER_MODULE}.DISABLE_VECTOR_DB", False)
    def test_unavailable_when_vector_db_enabled(self) -> None:
        assert FileReaderTool.is_available(MagicMock()) is False


# ------------------------------------------------------------------
# Force-add SearchTool suppression
# ------------------------------------------------------------------


class TestForceAddSearchToolGuard:
    def test_force_add_block_checks_disable_vector_db(self) -> None:
        """The force-add SearchTool block in construct_tools should include
        `not DISABLE_VECTOR_DB` so that forced search is also suppressed
        without a vector DB."""
        import inspect

        from onyx.tools.tool_constructor import _construct_tools_impl

        source = inspect.getsource(_construct_tools_impl)
        assert "DISABLE_VECTOR_DB" in source, (
            "construct_tools should reference DISABLE_VECTOR_DB to suppress force-adding SearchTool"
        )


class TestEffectivePersonaTools:
    def test_document_set_file_scope_fails_closed_without_access(self) -> None:
        with patch(
            "onyx.tools.tool_constructor.filter_document_set_names_by_user_access",
            return_value=set(),
        ):
            with pytest.raises(OnyxError, match="scope is unavailable"):
                _resolve_document_set_file_ids(
                    db_session=MagicMock(),
                    user=MagicMock(),
                    document_set_names=["Benchmark Set"],
                )

    def test_document_set_file_scope_fails_closed_when_set_is_empty(self) -> None:
        with (
            patch(
                "onyx.tools.tool_constructor.filter_document_set_names_by_user_access",
                return_value={"Benchmark Set"},
            ),
            patch(
                "onyx.tools.tool_constructor.get_document_sets_by_name",
                return_value=[
                    SimpleNamespace(
                        id=99,
                        connector_credential_pairs=[],
                        federated_connectors=[],
                    )
                ],
            ),
            patch(
                "onyx.tools.tool_constructor.fetch_user_files_for_document_set",
                return_value=[],
            ),
        ):
            with pytest.raises(OnyxError, match="no searchable files"):
                _resolve_document_set_file_ids(
                    db_session=MagicMock(),
                    user=MagicMock(),
                    document_set_names=["Benchmark Set"],
                )

    def test_document_set_file_scope_rejects_connector_content(self) -> None:
        with (
            patch(
                "onyx.tools.tool_constructor.filter_document_set_names_by_user_access",
                return_value={"Benchmark Set"},
            ),
            patch(
                "onyx.tools.tool_constructor.get_document_sets_by_name",
                return_value=[
                    SimpleNamespace(
                        id=99,
                        connector_credential_pairs=[SimpleNamespace(id=7)],
                        federated_connectors=[],
                    )
                ],
            ),
        ):
            with pytest.raises(OnyxError, match="only uploaded files"):
                _resolve_document_set_file_ids(
                    db_session=MagicMock(),
                    user=MagicMock(),
                    document_set_names=["Benchmark Set"],
                )

    @pytest.mark.parametrize(
        "unready_status",
        [UserFileStatus.CHUNKED, UserFileStatus.INDEXING, UserFileStatus.FAILED],
    )
    def test_document_set_file_scope_rejects_mixed_file_statuses(
        self, unready_status: UserFileStatus
    ) -> None:
        with (
            patch(
                "onyx.tools.tool_constructor.filter_document_set_names_by_user_access",
                return_value={"Benchmark Set"},
            ),
            patch(
                "onyx.tools.tool_constructor.get_document_sets_by_name",
                return_value=[
                    SimpleNamespace(
                        id=99,
                        connector_credential_pairs=[],
                        federated_connectors=[],
                    )
                ],
            ),
            patch(
                "onyx.tools.tool_constructor.fetch_user_files_for_document_set",
                return_value=[
                    SimpleNamespace(
                        id="completed-file", status=UserFileStatus.COMPLETED
                    ),
                    SimpleNamespace(id="unready-file", status=unready_status),
                ],
            ),
        ):
            with pytest.raises(OnyxError, match="must be completed"):
                _resolve_document_set_file_ids(
                    db_session=MagicMock(),
                    user=MagicMock(),
                    document_set_names=["Benchmark Set"],
                )

    @pytest.mark.parametrize(
        "repair_status",
        [
            UserFileProjectionRepairStatus.PENDING,
            UserFileProjectionRepairStatus.RUNNING,
            UserFileProjectionRepairStatus.FAILED,
        ],
    )
    def test_document_id_scope_does_not_depend_on_projection_repair_state(
        self, repair_status: UserFileProjectionRepairStatus
    ) -> None:
        user_file = SimpleNamespace(id="repair-file", status=UserFileStatus.COMPLETED)
        with (
            patch(
                "onyx.tools.tool_constructor.filter_document_set_names_by_user_access",
                return_value={"Benchmark Set"},
            ),
            patch(
                "onyx.tools.tool_constructor.get_document_sets_by_name",
                return_value=[
                    SimpleNamespace(
                        id=99,
                        connector_credential_pairs=[],
                        federated_connectors=[],
                    )
                ],
            ),
            patch(
                "onyx.tools.tool_constructor.fetch_user_files_for_document_set",
                return_value=[user_file],
            ),
            patch(
                "onyx.tools.tool_constructor.get_chunk_counts_for_files",
                return_value={user_file.id: 1},
            ),
            patch(
                "onyx.tools.tool_constructor."
                "fetch_user_file_projection_repair_statuses",
                return_value={user_file.id: repair_status},
                create=True,
            ),
        ):
            file_ids = _resolve_document_set_file_ids(
                db_session=MagicMock(),
                user=MagicMock(),
                document_set_names=["Benchmark Set"],
            )

        assert file_ids == ["repair-file"]

    @pytest.mark.parametrize(
        ("user_files", "chunk_counts"),
        [
            (
                [SimpleNamespace(id="generic-file", status=UserFileStatus.COMPLETED)],
                {},
            ),
            (
                [
                    SimpleNamespace(
                        id="regulatory-file", status=UserFileStatus.COMPLETED
                    ),
                    SimpleNamespace(id="generic-file", status=UserFileStatus.COMPLETED),
                ],
                {"regulatory-file": 3},
            ),
        ],
    )
    def test_document_set_file_scope_rejects_missing_regulatory_projection(
        self,
        user_files: list[SimpleNamespace],
        chunk_counts: dict[str, int],
    ) -> None:
        with (
            patch(
                "onyx.tools.tool_constructor.filter_document_set_names_by_user_access",
                return_value={"Benchmark Set"},
            ),
            patch(
                "onyx.tools.tool_constructor.get_document_sets_by_name",
                return_value=[
                    SimpleNamespace(
                        id=99,
                        connector_credential_pairs=[],
                        federated_connectors=[],
                    )
                ],
            ),
            patch(
                "onyx.tools.tool_constructor.fetch_user_files_for_document_set",
                return_value=user_files,
            ),
            patch(
                "onyx.tools.tool_constructor.get_chunk_counts_for_files",
                return_value=chunk_counts,
            ),
        ):
            with pytest.raises(OnyxError, match="must have regulatory chunks"):
                _resolve_document_set_file_ids(
                    db_session=MagicMock(),
                    user=MagicMock(),
                    document_set_names=["Benchmark Set"],
                )

    def test_default_persona_preserves_explicit_standard_search_mode(self) -> None:
        configured_search_tool = MagicMock(
            id=1,
            name="internal_search",
            in_code_tool_id=SearchTool.__name__,
        )
        persona = MagicMock(
            id=DEFAULT_PERSONA_ID,
            name="default",
            tools=[configured_search_tool],
        )
        persona.document_sets = []
        persona.attached_documents = []
        persona.hierarchy_nodes = []
        persona.search_start_date = None
        user = MagicMock(oauth_accounts=[], enable_memory_tool=False)

        with (
            patch(
                "onyx.tools.tool_constructor.get_current_search_settings",
                return_value=MagicMock(),
            ),
            patch(
                "onyx.tools.tool_constructor.get_default_document_index",
                return_value=MagicMock(),
            ),
            patch(
                "onyx.tools.tool_constructor.get_built_in_tool_by_id",
                return_value=SearchTool,
            ),
            patch.object(SearchTool, "is_available", return_value=True),
        ):
            result = _construct_tools_impl(
                persona=persona,
                db_session=MagicMock(),
                emitter=MagicMock(),
                user=user,
                llm=MagicMock(),
                search_tool_config=SearchToolConfig(
                    user_selected_filters=BaseFilters(regulatory_chunks_only=False)
                ),
                search_usage_forcing_setting=SearchToolUsage.ENABLED,
            )

        search_tool = result[1][0]
        assert isinstance(search_tool, SearchTool)
        assert search_tool.user_selected_filters is not None
        assert search_tool.user_selected_filters.regulatory_chunks_only is False

    def test_construct_tools_passes_inherited_tools_to_runtime_constructor(
        self,
    ) -> None:
        persona = MagicMock()
        session = MagicMock()
        inherited_tool = MagicMock(id=1, name="internal_search")
        direct_tool = MagicMock(id=3, name="custom_action")

        with (
            patch(
                "onyx.tools.tool_constructor.get_effective_persona_tools",
                return_value=[inherited_tool, direct_tool],
            ) as get_effective,
            patch(
                "onyx.tools.tool_constructor._construct_tools_impl",
                return_value={},
            ) as construct_impl,
        ):
            construct_tools(
                persona=persona,
                emitter=MagicMock(),
                user=MagicMock(),
                llm=MagicMock(),
                db_session=session,
                allowed_tool_ids=[1],
            )

        get_effective.assert_called_once_with(persona, session)
        assert construct_impl.call_args.kwargs["persona_tools"] == [
            inherited_tool,
            direct_tool,
        ]
        assert construct_impl.call_args.kwargs["allowed_tool_ids"] == [1]

    def test_forced_search_respects_allowed_tool_ids(self) -> None:
        persona = MagicMock(id=7, name="custom", tools=[])
        persona.document_sets = []
        persona.attached_documents = []
        persona.hierarchy_nodes = []
        user = MagicMock(oauth_accounts=[], enable_memory_tool=False)
        search_db_tool = MagicMock(id=1)

        with (
            patch(
                "onyx.tools.tool_constructor.get_current_search_settings",
                return_value=MagicMock(),
            ),
            patch(
                "onyx.tools.tool_constructor.get_default_document_index",
                return_value=MagicMock(),
            ),
            patch(
                "onyx.tools.tool_constructor.get_builtin_tool",
                return_value=search_db_tool,
            ),
        ):
            result = _construct_tools_impl(
                persona=persona,
                db_session=MagicMock(),
                emitter=MagicMock(),
                user=user,
                llm=MagicMock(),
                allowed_tool_ids=[99],
                search_usage_forcing_setting=SearchToolUsage.ENABLED,
            )

        assert result == {}

    def test_document_set_persona_restores_missing_search_tool(self) -> None:
        document_set = SimpleNamespace(name="Agent Knowledge")
        persona = MagicMock(id=7, name="custom", tools=[])
        persona.document_sets = [document_set]
        persona.attached_documents = []
        persona.hierarchy_nodes = []
        persona.search_start_date = None
        user = MagicMock(oauth_accounts=[], enable_memory_tool=False)
        search_db_tool = MagicMock(id=1)

        with (
            patch(
                "onyx.tools.tool_constructor.get_current_search_settings",
                return_value=MagicMock(),
            ),
            patch(
                "onyx.tools.tool_constructor.get_default_document_index",
                return_value=MagicMock(),
            ),
            patch(
                "onyx.tools.tool_constructor.get_builtin_tool",
                return_value=search_db_tool,
            ),
        ):
            result = _construct_tools_impl(
                persona=persona,
                db_session=MagicMock(),
                emitter=MagicMock(),
                user=user,
                llm=MagicMock(),
                search_usage_forcing_setting=SearchToolUsage.AUTO,
            )

        search_tools = result[1]
        assert len(search_tools) == 1
        assert isinstance(search_tools[0], SearchTool)
        assert search_tools[0].persona_search_info.document_set_names == [
            "Agent Knowledge"
        ]

    def test_explicit_search_scope_overrides_persona_document_sets(self) -> None:
        configured_search_tool = MagicMock(
            id=1,
            name="internal_search",
            in_code_tool_id=SearchTool.__name__,
        )
        persona = MagicMock(
            id=DEFAULT_PERSONA_ID,
            name="default",
            tools=[configured_search_tool],
        )
        persona.document_sets = [SimpleNamespace(name="Unrelated Persona Set")]
        persona.attached_documents = [SimpleNamespace(id="attached-document")]
        persona.hierarchy_nodes = [SimpleNamespace(id=41)]
        persona.search_start_date = MagicMock()
        user = MagicMock(oauth_accounts=[], enable_memory_tool=False)

        with (
            patch(
                "onyx.tools.tool_constructor.get_current_search_settings",
                return_value=MagicMock(),
            ),
            patch(
                "onyx.tools.tool_constructor.get_default_document_index",
                return_value=MagicMock(),
            ),
            patch(
                "onyx.tools.tool_constructor.get_built_in_tool_by_id",
                return_value=SearchTool,
            ),
            patch.object(SearchTool, "is_available", return_value=True),
            patch(
                "onyx.tools.tool_constructor.filter_document_set_names_by_user_access",
                return_value={"Benchmark Set"},
            ),
            patch(
                "onyx.tools.tool_constructor.get_document_sets_by_name",
                return_value=[
                    SimpleNamespace(
                        id=99,
                        connector_credential_pairs=[],
                        federated_connectors=[],
                    )
                ],
            ),
            patch(
                "onyx.tools.tool_constructor.fetch_user_files_for_document_set",
                return_value=[
                    SimpleNamespace(
                        id="benchmark-file-a", status=UserFileStatus.COMPLETED
                    ),
                    SimpleNamespace(
                        id="benchmark-file-b", status=UserFileStatus.COMPLETED
                    ),
                ],
            ),
            patch(
                "onyx.tools.tool_constructor.get_chunk_counts_for_files",
                return_value={"benchmark-file-a": 4, "benchmark-file-b": 7},
            ),
        ):
            result = _construct_tools_impl(
                persona=persona,
                db_session=MagicMock(),
                emitter=MagicMock(),
                user=user,
                llm=MagicMock(),
                search_tool_config=SearchToolConfig(
                    user_selected_filters=BaseFilters(document_set=["Benchmark Set"]),
                    document_set_names_override=["Benchmark Set"],
                    project_id_filter=12,
                    persona_id_filter=13,
                ),
                search_usage_forcing_setting=SearchToolUsage.ENABLED,
            )

        search_tool = result[1][0]
        assert isinstance(search_tool, SearchTool)
        assert search_tool.persona_search_info.document_set_names == []
        assert search_tool.persona_search_info.attached_document_ids == [
            "benchmark-file-a",
            "benchmark-file-b",
        ]
        assert search_tool.persona_search_info.hierarchy_node_ids == []
        assert search_tool.persona_search_info.search_start_date is None
        assert search_tool.user_selected_filters is not None

        assert search_tool.user_selected_filters.document_set is None
        assert search_tool.project_id_filter is None
        assert search_tool.persona_id_filter is None

    def test_legacy_document_set_persona_respects_empty_allowlist(self) -> None:
        persona = MagicMock(id=7, name="legacy", tools=[])
        persona.document_sets = [SimpleNamespace(name="Agent Knowledge")]
        persona.attached_documents = []
        persona.hierarchy_nodes = []
        persona.search_start_date = None
        user = MagicMock(oauth_accounts=[], enable_memory_tool=False)
        search_db_tool = MagicMock(id=1)

        with (
            patch(
                "onyx.tools.tool_constructor.get_current_search_settings",
                return_value=MagicMock(),
            ),
            patch(
                "onyx.tools.tool_constructor.get_default_document_index",
                return_value=MagicMock(),
            ),
            patch(
                "onyx.tools.tool_constructor.get_builtin_tool",
                return_value=search_db_tool,
            ),
        ):
            result = _construct_tools_impl(
                persona=persona,
                db_session=MagicMock(),
                emitter=MagicMock(),
                user=user,
                llm=MagicMock(),
                allowed_tool_ids=[],
                search_usage_forcing_setting=SearchToolUsage.AUTO,
            )

        assert result == {}

    def test_document_set_persona_does_not_restore_filtered_search_tool(self) -> None:
        document_set = SimpleNamespace(name="Agent Knowledge")
        configured_search_tool = MagicMock(
            id=1,
            name="internal_search",
            in_code_tool_id=SearchTool.__name__,
        )
        persona = MagicMock(
            id=7,
            name="custom",
            tools=[configured_search_tool],
        )
        persona.document_sets = [document_set]
        persona.attached_documents = []
        persona.hierarchy_nodes = []
        persona.search_start_date = None
        user = MagicMock(oauth_accounts=[], enable_memory_tool=False)

        with (
            patch(
                "onyx.tools.tool_constructor.get_current_search_settings",
                return_value=MagicMock(),
            ),
            patch(
                "onyx.tools.tool_constructor.get_default_document_index",
                return_value=MagicMock(),
            ),
        ):
            result = _construct_tools_impl(
                persona=persona,
                db_session=MagicMock(),
                emitter=MagicMock(),
                user=user,
                llm=MagicMock(),
                allowed_tool_ids=[],
                search_usage_forcing_setting=SearchToolUsage.AUTO,
            )

        assert result == {}


# ------------------------------------------------------------------
# Persona API — _validate_vector_db_knowledge
# ------------------------------------------------------------------


class TestValidateVectorDbKnowledge:
    @patch(
        "onyx.server.features.persona.api.DISABLE_VECTOR_DB",
        True,
    )
    def test_rejects_document_set_ids(self) -> None:
        from fastapi import HTTPException

        from onyx.server.features.persona.api import _validate_vector_db_knowledge

        request = MagicMock()
        request.document_set_ids = [1]
        request.hierarchy_node_ids = []
        request.document_ids = []

        with __import__("pytest").raises(HTTPException) as exc_info:
            _validate_vector_db_knowledge(request)
        assert exc_info.value.status_code == 400
        assert "document sets" in exc_info.value.detail

    @patch(
        "onyx.server.features.persona.api.DISABLE_VECTOR_DB",
        True,
    )
    def test_rejects_hierarchy_node_ids(self) -> None:
        from fastapi import HTTPException

        from onyx.server.features.persona.api import _validate_vector_db_knowledge

        request = MagicMock()
        request.document_set_ids = []
        request.hierarchy_node_ids = [1]
        request.document_ids = []

        with __import__("pytest").raises(HTTPException) as exc_info:
            _validate_vector_db_knowledge(request)
        assert exc_info.value.status_code == 400
        assert "hierarchy nodes" in exc_info.value.detail

    @patch(
        "onyx.server.features.persona.api.DISABLE_VECTOR_DB",
        True,
    )
    def test_rejects_document_ids(self) -> None:
        from fastapi import HTTPException

        from onyx.server.features.persona.api import _validate_vector_db_knowledge

        request = MagicMock()
        request.document_set_ids = []
        request.hierarchy_node_ids = []
        request.document_ids = ["doc-abc"]

        with __import__("pytest").raises(HTTPException) as exc_info:
            _validate_vector_db_knowledge(request)
        assert exc_info.value.status_code == 400
        assert "documents" in exc_info.value.detail

    @patch(
        "onyx.server.features.persona.api.DISABLE_VECTOR_DB",
        True,
    )
    def test_allows_user_files_only(self) -> None:
        from onyx.server.features.persona.api import _validate_vector_db_knowledge

        request = MagicMock()
        request.document_set_ids = []
        request.hierarchy_node_ids = []
        request.document_ids = []

        _validate_vector_db_knowledge(request)

    @patch(
        "onyx.server.features.persona.api.DISABLE_VECTOR_DB",
        False,
    )
    def test_allows_everything_when_vector_db_enabled(self) -> None:
        from onyx.server.features.persona.api import _validate_vector_db_knowledge

        request = MagicMock()
        request.document_set_ids = [1, 2]
        request.hierarchy_node_ids = [3]
        request.document_ids = ["doc-x"]

        _validate_vector_db_knowledge(request)
