from typing import Type, Union

from onyx.tools.tool_implementations.coding_agent.coding_agent_tool import (
    CodingAgentTool,
)
from onyx.tools.tool_implementations.file_reader.file_reader_tool import FileReaderTool
from onyx.tools.tool_implementations.images.image_generation_tool import (
    ImageGenerationTool,
)
from onyx.tools.tool_implementations.knowledge_graph.knowledge_graph_tool import (
    KnowledgeGraphTool,
)
from onyx.tools.tool_implementations.memory.memory_tool import MemoryTool
from onyx.tools.tool_implementations.search.search_tool import SearchTool
from onyx.utils.logger import setup_logger

logger = setup_logger()

# Code execution and internet access (PythonTool, WebSearchTool, OpenURLTool)
# are intentionally not registered here: this deployment only ever answers
# from indexed regulatory chunks, so those tools are removed from the
# built-in registry rather than just hidden. See
# alembic/versions/*_remove_code_exec_and_web_search_tools.py for the
# migration that detaches any persona still referencing them.
BUILT_IN_TOOL_TYPES = Union[
    SearchTool,
    ImageGenerationTool,
    KnowledgeGraphTool,
    FileReaderTool,
    MemoryTool,
    CodingAgentTool,
]

BUILT_IN_TOOL_MAP: dict[str, Type[BUILT_IN_TOOL_TYPES]] = {
    SearchTool.__name__: SearchTool,
    ImageGenerationTool.__name__: ImageGenerationTool,
    KnowledgeGraphTool.__name__: KnowledgeGraphTool,
    FileReaderTool.__name__: FileReaderTool,
    MemoryTool.__name__: MemoryTool,
    CodingAgentTool.__name__: CodingAgentTool,
}

STOPPING_TOOLS_NAMES: list[str] = [ImageGenerationTool.NAME]
CITEABLE_TOOLS_NAMES: list[str] = [
    SearchTool.NAME,
]


def get_built_in_tool_ids() -> list[str]:
    return list(BUILT_IN_TOOL_MAP.keys())


def get_built_in_tool_by_id(in_code_tool_id: str) -> Type[BUILT_IN_TOOL_TYPES]:
    return BUILT_IN_TOOL_MAP[in_code_tool_id]


def _tool_llm_name(cls: Type[BUILT_IN_TOOL_TYPES]) -> str:
    """Extract the LLM-facing tool name from a tool class."""
    name_attr = cls.__dict__.get("name")
    if isinstance(name_attr, property) and name_attr.fget is not None:
        return name_attr.fget(cls)
    if isinstance(name_attr, str):
        return name_attr
    raise ValueError(
        f"Built-in tool {cls.__name__} must define a valid LLM-facing tool name"
    )


def _build_tool_name_to_class() -> dict[str, Type[BUILT_IN_TOOL_TYPES]]:
    """Build a mapping from LLM-facing tool name to tool class."""
    return {_tool_llm_name(cls): cls for cls in BUILT_IN_TOOL_MAP.values()}


TOOL_NAME_TO_CLASS: dict[str, Type[BUILT_IN_TOOL_TYPES]] = _build_tool_name_to_class()
