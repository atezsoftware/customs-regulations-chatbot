"""seed optional regulatory provision tool

Revision ID: 3ef6b03f58af
Revises: 25ed6e902979
Create Date: 2026-09-24 13:10:41.404056

"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "3ef6b03f58af"
down_revision = "25ed6e902979"
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()
    conn.execute(
        sa.text("""
        INSERT INTO tool (name, display_name, description, in_code_tool_id, enabled)
        SELECT 'get_regulatory_provision', 'Mevzuat Maddesi Bul',
               'Optional lookup of a named legal instrument and provision.',
               'RegulatoryProvisionTool', true
        WHERE NOT EXISTS (SELECT 1 FROM tool WHERE in_code_tool_id = 'RegulatoryProvisionTool')
    """)
    )
    conn.execute(
        sa.text("""
        INSERT INTO persona__tool (persona_id, tool_id)
        SELECT DISTINCT existing.persona_id, provision.id
        FROM persona__tool existing
        JOIN tool search ON search.id = existing.tool_id
        CROSS JOIN tool provision
        WHERE search.in_code_tool_id = 'SearchTool'
          AND search.enabled = true
          AND provision.in_code_tool_id = 'RegulatoryProvisionTool'
        ON CONFLICT DO NOTHING
    """)
    )


def downgrade() -> None:
    conn = op.get_bind()
    conn.execute(
        sa.text("""
        DELETE FROM persona__tool WHERE tool_id IN
        (SELECT id FROM tool WHERE in_code_tool_id = 'RegulatoryProvisionTool')
    """)
    )
    conn.execute(
        sa.text("DELETE FROM tool WHERE in_code_tool_id = 'RegulatoryProvisionTool'")
    )
