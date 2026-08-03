"""remove code exec and web search tools

Revision ID: 9ce718a30332
Revises: 2010a61d7d88
Create Date: 2026-07-31 16:21:17.926622

This deployment answers only from indexed regulatory chunks, so code
execution and internet access are removed rather than just hidden:
PythonTool, WebSearchTool, and OpenURLTool were dropped from
onyx/tools/built_in_tools.py's registry. Deleting the corresponding `tool`
rows here detaches them from every persona via persona__tool's ON DELETE
CASCADE, so no persona can reference a tool id the registry no longer knows
about.
"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "9ce718a30332"
down_revision = "2010a61d7d88"
branch_labels = None
depends_on = None

_REMOVED_TOOL_IDS = ("WebSearchTool", "OpenURLTool", "PythonTool")

_TOOL_TABLE = sa.table(
    "tool",
    sa.column("id", sa.Integer),
    sa.column("name", sa.String),
    sa.column("display_name", sa.String),
    sa.column("description", sa.Text),
    sa.column("in_code_tool_id", sa.String),
    sa.column("passthrough_auth", sa.Boolean),
    sa.column("enabled", sa.Boolean),
)

# Snapshot of the rows being removed, for downgrade().
_REMOVED_TOOL_ROWS = [
    {
        "name": "web_search",
        "display_name": "Web Search",
        "description": (
            "The Web Search Action allows the agent to perform internet "
            "searches for up-to-date information."
        ),
        "in_code_tool_id": "WebSearchTool",
    },
    {
        "name": "open_url",
        "display_name": "Open URL",
        "description": (
            "The Open URL Action allows the agent to fetch and read "
            "contents of web pages."
        ),
        "in_code_tool_id": "OpenURLTool",
    },
    {
        "name": "run_python",
        "display_name": "Code Interpreter",
        "description": (
            "The Code Interpreter Action allows the assistant to execute "
            "Python code in a secure, isolated environment for data "
            "analysis, computation, visualization, and file processing."
        ),
        "in_code_tool_id": "PythonTool",
    },
]


def upgrade() -> None:
    conn = op.get_bind()
    conn.execute(
        _TOOL_TABLE.delete().where(_TOOL_TABLE.c.in_code_tool_id.in_(_REMOVED_TOOL_IDS))
    )


def downgrade() -> None:
    conn = op.get_bind()
    for row in _REMOVED_TOOL_ROWS:
        conn.execute(
            _TOOL_TABLE.insert().values(
                **row,
                passthrough_auth=False,
                enabled=True,
            )
        )
