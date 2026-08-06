"""add user files to document sets

Revision ID: c6a8d9e4f2b1
Revises: f4c9b1a8e672

Legacy Files directories reused UserProject rows and were only manageable by
admins. We therefore migrate file-backed projects owned by an ADMIN, plus any
project explicitly referenced by regulatory amendment/benchmark data. Generic
users' chat projects stay projects. The migrated sets remain public to preserve
the former shared regulatory corpus; active and in-flight UserFiles are marked
dirty so files outside those sets lose the old forced-public regulatory ACL.
Downgrade-only compatibility projects use a negative project ID paired with an
epoch creation time so a later re-upgrade can restore mixed-source sets without
trusting user-editable project fields.
"""

import sqlalchemy as sa
from alembic import op

revision = "c6a8d9e4f2b1"
down_revision = "f4c9b1a8e672"
branch_labels = None
depends_on = None


def _migrate_referenced_projects_to_document_sets() -> None:
    op.execute(
        """
        DO $$
        DECLARE
            project_row RECORD;
            base_name TEXT;
            candidate_name TEXT;
            collision_number INTEGER;
        BEGIN
            FOR project_row IN
                SELECT
                    project.id,
                    project.name,
                    project.description,
                    project.user_id,
                    CASE
                        WHEN project.id < 0
                          AND project.created_at =
                              TIMESTAMPTZ '1970-01-01 00:00:00+00'
                        THEN -project.id::BIGINT
                    END AS compatibility_document_set_id
                FROM user_project AS project
                WHERE (
                    EXISTS (
                        SELECT 1
                        FROM project__user_file AS project_file
                        WHERE project_file.project_id = project.id
                    )
                    AND EXISTS (
                        SELECT 1
                        FROM "user" AS project_owner
                        WHERE project_owner.id = project.user_id
                          AND project_owner.role = 'ADMIN'
                    )
                ) OR EXISTS (
                    SELECT 1
                    FROM amendment_batch
                    WHERE amendment_batch.project_id = project.id
                ) OR EXISTS (
                    SELECT 1
                    FROM benchmark_question
                    WHERE benchmark_question.project_id = project.id
                ) OR (
                    project.id < 0
                    AND project.created_at =
                        TIMESTAMPTZ '1970-01-01 00:00:00+00'
                )
                ORDER BY project.id
            LOOP
                IF project_row.compatibility_document_set_id IS NULL OR NOT EXISTS (
                    SELECT 1
                    FROM document_set
                    WHERE document_set.id::BIGINT =
                        project_row.compatibility_document_set_id
                      AND document_set.migrated_from_project_id IS NULL
                ) THEN
                    base_name := COALESCE(
                        NULLIF(BTRIM(project_row.name), ''),
                        FORMAT('Project %s', project_row.id)
                    );
                    candidate_name := base_name;
                    IF EXISTS (
                        SELECT 1 FROM document_set WHERE name = candidate_name
                    ) THEN
                        candidate_name := FORMAT(
                            '%s (Project %s)', base_name, project_row.id
                        );
                        collision_number := 2;
                        WHILE EXISTS (
                            SELECT 1 FROM document_set WHERE name = candidate_name
                        ) LOOP
                            candidate_name := FORMAT(
                                '%s (Project %s, %s)',
                                base_name,
                                project_row.id,
                                collision_number
                            );
                            collision_number := collision_number + 1;
                        END LOOP;
                    END IF;

                    INSERT INTO document_set (
                        name,
                        description,
                        user_id,
                        migrated_from_project_id,
                        is_up_to_date,
                        is_public,
                        is_deleting
                    ) VALUES (
                        candidate_name,
                        project_row.description,
                        project_row.user_id,
                        project_row.id,
                        TRUE,
                        TRUE,
                        FALSE
                    );
                END IF;
            END LOOP;
        END $$
        """
    )
    op.execute(
        """
        WITH compatibility_targets AS (
            SELECT
                project.id AS project_id,
                target_document_set.id AS document_set_id
            FROM user_project AS project
            JOIN document_set AS target_document_set
              ON target_document_set.id::BIGINT = -project.id::BIGINT
             AND target_document_set.migrated_from_project_id IS NULL
            WHERE project.id < 0
              AND project.created_at = TIMESTAMPTZ '1970-01-01 00:00:00+00'
        ),
        project_targets AS (
            SELECT migrated_from_project_id AS project_id, id AS document_set_id
            FROM document_set
            WHERE migrated_from_project_id IS NOT NULL
            UNION ALL
            SELECT project_id, document_set_id
            FROM compatibility_targets
        )
        INSERT INTO document_set__user_file (
            document_set_id,
            user_file_id,
            created_at
        )
        SELECT
            project_targets.document_set_id,
            project_file.user_file_id,
            project_file.created_at
        FROM project_targets
        JOIN project__user_file AS project_file
          ON project_file.project_id = project_targets.project_id
        ON CONFLICT (document_set_id, user_file_id) DO NOTHING
        """
    )


def _migrate_regulatory_scope_to_document_sets() -> None:
    op.add_column(
        "amendment_batch",
        sa.Column("document_set_id", sa.Integer(), nullable=True),
    )
    op.add_column(
        "benchmark_question",
        sa.Column("document_set_id", sa.Integer(), nullable=True),
    )

    op.execute(
        """
        WITH compatibility_targets AS (
            SELECT
                project.id AS project_id,
                target_document_set.id AS document_set_id
            FROM user_project AS project
            JOIN document_set AS target_document_set
              ON target_document_set.id::BIGINT = -project.id::BIGINT
             AND target_document_set.migrated_from_project_id IS NULL
            WHERE project.id < 0
              AND project.created_at = TIMESTAMPTZ '1970-01-01 00:00:00+00'
        ),
        project_targets AS (
            SELECT migrated_from_project_id AS project_id, id AS document_set_id
            FROM document_set
            WHERE migrated_from_project_id IS NOT NULL
            UNION ALL
            SELECT project_id, document_set_id
            FROM compatibility_targets
        )
        UPDATE amendment_batch
        SET document_set_id = project_targets.document_set_id
        FROM project_targets
        WHERE project_targets.project_id = amendment_batch.project_id
        """
    )
    op.execute(
        """
        WITH compatibility_targets AS (
            SELECT
                project.id AS project_id,
                target_document_set.id AS document_set_id
            FROM user_project AS project
            JOIN document_set AS target_document_set
              ON target_document_set.id::BIGINT = -project.id::BIGINT
             AND target_document_set.migrated_from_project_id IS NULL
            WHERE project.id < 0
              AND project.created_at = TIMESTAMPTZ '1970-01-01 00:00:00+00'
        ),
        project_targets AS (
            SELECT migrated_from_project_id AS project_id, id AS document_set_id
            FROM document_set
            WHERE migrated_from_project_id IS NOT NULL
            UNION ALL
            SELECT project_id, document_set_id
            FROM compatibility_targets
        )
        UPDATE benchmark_question
        SET document_set_id = project_targets.document_set_id
        FROM project_targets
        WHERE project_targets.project_id = benchmark_question.project_id
        """
    )
    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1 FROM amendment_batch WHERE document_set_id IS NULL
            ) OR EXISTS (
                SELECT 1 FROM benchmark_question WHERE document_set_id IS NULL
            ) THEN
                RAISE EXCEPTION
                    'Could not map all regulatory rows to document sets';
            END IF;
        END $$
        """
    )

    op.alter_column("amendment_batch", "document_set_id", nullable=False)
    op.alter_column("benchmark_question", "document_set_id", nullable=False)
    op.create_foreign_key(
        "amendment_batch_document_set_id_fkey",
        "amendment_batch",
        "document_set",
        ["document_set_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_foreign_key(
        "benchmark_question_document_set_id_fkey",
        "benchmark_question",
        "document_set",
        ["document_set_id"],
        ["id"],
        ondelete="RESTRICT",
    )
    op.create_index(
        "ix_amendment_batch_document_set_id",
        "amendment_batch",
        ["document_set_id"],
    )
    op.create_index(
        "ix_benchmark_question_document_set_id",
        "benchmark_question",
        ["document_set_id"],
    )

    op.drop_index("ix_amendment_batch_project_id", table_name="amendment_batch")
    op.drop_index("ix_benchmark_question_project_id", table_name="benchmark_question")
    op.drop_constraint(
        "amendment_batch_project_id_fkey", "amendment_batch", type_="foreignkey"
    )
    op.drop_constraint(
        "benchmark_question_project_id_fkey",
        "benchmark_question",
        type_="foreignkey",
    )
    op.drop_column("amendment_batch", "project_id")
    op.drop_column("benchmark_question", "project_id")


def upgrade() -> None:
    op.add_column(
        "document_set",
        sa.Column("migrated_from_project_id", sa.Integer(), nullable=True),
    )
    op.create_index(
        "ux_document_set_migrated_from_project_id",
        "document_set",
        ["migrated_from_project_id"],
        unique=True,
    )
    op.add_column(
        "document_set",
        sa.Column(
            "is_deleting",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
    )
    op.add_column(
        "user_file",
        sa.Column(
            "needs_document_set_sync",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
    )
    op.create_table(
        "document_set__user_file",
        sa.Column("document_set_id", sa.Integer(), nullable=False),
        sa.Column("user_file_id", sa.UUID(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.ForeignKeyConstraint(
            ["document_set_id"], ["document_set.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(["user_file_id"], ["user_file.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("document_set_id", "user_file_id"),
    )
    op.create_index(
        "ix_document_set__user_file_document_set_id_created_at",
        "document_set__user_file",
        ["document_set_id", sa.text("created_at DESC")],
    )
    op.create_index(
        "ix_document_set__user_file_user_file_id",
        "document_set__user_file",
        ["user_file_id"],
    )

    _migrate_referenced_projects_to_document_sets()
    _migrate_regulatory_scope_to_document_sets()
    op.execute(
        """
        UPDATE user_file AS uf
        SET chunk_count = GREATEST(
            COALESCE(uf.chunk_count, 0),
            regulatory_counts.chunk_count
        )
        FROM (
            SELECT user_file_id, COUNT(*)::INTEGER AS chunk_count
            FROM regulatory_chunk
            GROUP BY user_file_id
        ) AS regulatory_counts
        WHERE uf.id = regulatory_counts.user_file_id
          AND uf.status = 'FAILED'
        """
    )
    op.execute(
        """
        UPDATE user_file
        SET needs_document_set_sync = TRUE
        WHERE status IN ('PROCESSING', 'INDEXING', 'COMPLETED', 'FAILED')
        """
    )


def _ensure_migration_generated_sets_are_downgrade_safe() -> None:
    op.execute(
        """
        DO $$
        DECLARE
            blocked_document_set_ids TEXT;
        BEGIN
            SELECT STRING_AGG(document_set.id::TEXT, ', ' ORDER BY document_set.id)
            INTO blocked_document_set_ids
            FROM document_set
            WHERE (
                NOT document_set.is_public
                AND (
                    EXISTS (
                        SELECT 1
                        FROM document_set__user_file
                        WHERE document_set_id = document_set.id
                    )
                    OR EXISTS (
                        SELECT 1
                        FROM amendment_batch
                        WHERE document_set_id = document_set.id
                    )
                    OR EXISTS (
                        SELECT 1
                        FROM benchmark_question
                        WHERE document_set_id = document_set.id
                    )
                )
            ) OR (
                document_set.migrated_from_project_id IS NOT NULL
                AND (
                  NOT document_set.is_public
                  OR document_set.is_deleting
                  OR EXISTS (
                      SELECT 1
                      FROM document_set__connector_credential_pair
                      WHERE document_set_id = document_set.id
                  )
                  OR EXISTS (
                      SELECT 1
                      FROM federated_connector__document_set
                      WHERE document_set_id = document_set.id
                  )
                  OR EXISTS (
                      SELECT 1
                      FROM persona__document_set
                      WHERE document_set_id = document_set.id
                  )
                  OR EXISTS (
                      SELECT 1
                      FROM document_set__user
                      WHERE document_set_id = document_set.id
                  )
                  OR EXISTS (
                      SELECT 1
                      FROM document_set__user_group
                      WHERE document_set_id = document_set.id
                  )
                  OR (
                      EXISTS (
                          SELECT 1
                          FROM user_project
                          WHERE id = document_set.migrated_from_project_id
                      )
                      AND (
                          EXISTS (
                              SELECT 1
                              FROM document_set__user_file AS document_set_file
                              WHERE document_set_file.document_set_id =
                                  document_set.id
                                AND NOT EXISTS (
                                    SELECT 1
                                    FROM project__user_file AS project_file
                                    WHERE project_file.project_id =
                                        document_set.migrated_from_project_id
                                      AND project_file.user_file_id =
                                          document_set_file.user_file_id
                                )
                          )
                          OR EXISTS (
                              SELECT 1
                              FROM project__user_file AS project_file
                              WHERE project_file.project_id =
                                  document_set.migrated_from_project_id
                                AND NOT EXISTS (
                                    SELECT 1
                                    FROM document_set__user_file AS document_set_file
                                    WHERE document_set_file.document_set_id =
                                        document_set.id
                                      AND document_set_file.user_file_id =
                                          project_file.user_file_id
                                )
                          )
                      )
                  )
                )
            );

            IF blocked_document_set_ids IS NOT NULL THEN
                RAISE EXCEPTION
                    'Cannot downgrade document sets [%]: publish private file/regulatory sets; for file-migrated sets also finish deletion, remove connector/persona/user/group relationships, and restore source-Project file membership',
                    blocked_document_set_ids;
            END IF;
        END $$
        """
    )


def _restore_regulatory_scope_to_compatibility_projects() -> None:
    op.add_column(
        "amendment_batch", sa.Column("project_id", sa.Integer(), nullable=True)
    )
    op.add_column(
        "benchmark_question", sa.Column("project_id", sa.Integer(), nullable=True)
    )
    op.execute(
        """
        UPDATE amendment_batch
        SET project_id = document_set.migrated_from_project_id
        FROM document_set
        WHERE document_set.id = amendment_batch.document_set_id
          AND document_set.migrated_from_project_id IS NOT NULL
          AND EXISTS (
              SELECT 1
              FROM user_project
              WHERE user_project.id = document_set.migrated_from_project_id
          )
        """
    )
    op.execute(
        """
        UPDATE benchmark_question
        SET project_id = document_set.migrated_from_project_id
        FROM document_set
        WHERE document_set.id = benchmark_question.document_set_id
          AND document_set.migrated_from_project_id IS NOT NULL
          AND EXISTS (
              SELECT 1
              FROM user_project
              WHERE user_project.id = document_set.migrated_from_project_id
          )
        """
    )
    op.execute(
        """
        INSERT INTO project__user_file (project_id, user_file_id, created_at)
        SELECT
            document_set.migrated_from_project_id,
            document_set_file.user_file_id,
            document_set_file.created_at
        FROM document_set
        JOIN document_set__user_file AS document_set_file
          ON document_set_file.document_set_id = document_set.id
        WHERE document_set.migrated_from_project_id IS NOT NULL
          AND EXISTS (
              SELECT 1
              FROM user_project
              WHERE user_project.id = document_set.migrated_from_project_id
          )
        ON CONFLICT (project_id, user_file_id) DO NOTHING
        """
    )
    op.execute(
        """
        DO $$
        DECLARE
            document_set_row RECORD;
            compatibility_project_id INTEGER;
            compatibility_owner_id UUID;
        BEGIN
            FOR document_set_row IN
                SELECT DISTINCT
                    document_set.id,
                    document_set.name,
                    document_set.description,
                    document_set.user_id,
                    document_set.migrated_from_project_id
                FROM document_set
                WHERE (
                    document_set.migrated_from_project_id IS NOT NULL
                    AND NOT EXISTS (
                        SELECT 1
                        FROM user_project
                        WHERE user_project.id = document_set.migrated_from_project_id
                    )
                ) OR (
                    document_set.migrated_from_project_id IS NULL
                    AND (
                      EXISTS (
                          SELECT 1
                          FROM amendment_batch
                          WHERE amendment_batch.document_set_id = document_set.id
                      ) OR EXISTS (
                          SELECT 1
                          FROM benchmark_question
                          WHERE benchmark_question.document_set_id = document_set.id
                      ) OR EXISTS (
                          SELECT 1
                          FROM document_set__user_file
                          WHERE document_set__user_file.document_set_id = document_set.id
                      )
                    )
                )
                ORDER BY document_set.id
            LOOP
                compatibility_owner_id := COALESCE(
                    (
                        SELECT document_set_owner.id
                        FROM "user" AS document_set_owner
                        WHERE document_set_owner.id = document_set_row.user_id
                          AND document_set_owner.role = 'ADMIN'
                    ),
                    (
                        SELECT user_file.user_id
                        FROM document_set__user_file AS document_set_file
                        JOIN user_file
                          ON user_file.id = document_set_file.user_file_id
                        JOIN "user" AS file_owner
                          ON file_owner.id = user_file.user_id
                        WHERE document_set_file.document_set_id = document_set_row.id
                          AND user_file.user_id IS NOT NULL
                          AND file_owner.role = 'ADMIN'
                        ORDER BY
                            user_file.created_at NULLS LAST,
                            user_file.id
                        LIMIT 1
                    ),
                    (
                        SELECT amendment_batch.created_by
                        FROM amendment_batch
                        JOIN "user" AS amendment_owner
                          ON amendment_owner.id = amendment_batch.created_by
                        WHERE amendment_batch.document_set_id = document_set_row.id
                          AND amendment_owner.role = 'ADMIN'
                        ORDER BY amendment_batch.id
                        LIMIT 1
                    ),
                    (
                        SELECT regulatory_owner.id
                        FROM benchmark_question
                        JOIN "user" AS regulatory_owner
                          ON regulatory_owner.id IN (
                              benchmark_question.created_by,
                              benchmark_question.updated_by
                          )
                        WHERE benchmark_question.document_set_id = document_set_row.id
                          AND regulatory_owner.role = 'ADMIN'
                        ORDER BY
                            benchmark_question.id,
                            CASE
                                WHEN regulatory_owner.id = benchmark_question.created_by
                                THEN 0
                                ELSE 1
                            END
                        LIMIT 1
                    ),
                    document_set_row.user_id,
                    (
                        SELECT user_file.user_id
                        FROM document_set__user_file AS document_set_file
                        JOIN user_file
                          ON user_file.id = document_set_file.user_file_id
                        WHERE document_set_file.document_set_id = document_set_row.id
                          AND user_file.user_id IS NOT NULL
                        ORDER BY
                            user_file.created_at NULLS LAST,
                            user_file.id
                        LIMIT 1
                    ),
                    (
                        SELECT amendment_batch.created_by
                        FROM amendment_batch
                        WHERE amendment_batch.document_set_id = document_set_row.id
                          AND amendment_batch.created_by IS NOT NULL
                        ORDER BY amendment_batch.id
                        LIMIT 1
                    ),
                    (
                        SELECT COALESCE(
                            benchmark_question.created_by,
                            benchmark_question.updated_by
                        )
                        FROM benchmark_question
                        WHERE benchmark_question.document_set_id = document_set_row.id
                          AND COALESCE(
                              benchmark_question.created_by,
                              benchmark_question.updated_by
                          ) IS NOT NULL
                        ORDER BY benchmark_question.id
                        LIMIT 1
                    )
                );

                IF compatibility_owner_id IS NULL THEN
                    RAISE EXCEPTION
                        'Cannot restore document set % to a compatibility project: no related owner exists',
                        document_set_row.id;
                END IF;

                IF document_set_row.id <= 0 THEN
                    RAISE EXCEPTION
                        'Cannot restore document set % to a compatibility project: expected a positive id',
                        document_set_row.id;
                END IF;

                compatibility_project_id :=
                    (-document_set_row.id::BIGINT)::INTEGER;
                IF EXISTS (
                    SELECT 1
                    FROM user_project
                    WHERE id = compatibility_project_id
                      AND created_at <>
                          TIMESTAMPTZ '1970-01-01 00:00:00+00'
                ) THEN
                    RAISE EXCEPTION
                        'Cannot reserve compatibility project id % for document set %',
                        compatibility_project_id,
                        document_set_row.id;
                END IF;

                IF NOT EXISTS (
                    SELECT 1
                    FROM user_project
                    WHERE id = compatibility_project_id
                      AND created_at =
                          TIMESTAMPTZ '1970-01-01 00:00:00+00'
                ) THEN
                    INSERT INTO user_project (
                        id,
                        user_id,
                        name,
                        description,
                        instructions,
                        created_at
                    ) VALUES (
                        compatibility_project_id,
                        compatibility_owner_id,
                        document_set_row.name,
                        document_set_row.description,
                        '',
                        TIMESTAMPTZ '1970-01-01 00:00:00+00'
                    );
                ELSE
                    UPDATE user_project
                    SET user_id = compatibility_owner_id,
                        name = document_set_row.name,
                        description = document_set_row.description
                    WHERE id = compatibility_project_id;
                END IF;

                IF document_set_row.migrated_from_project_id IS NOT NULL THEN
                    UPDATE document_set
                    SET migrated_from_project_id = compatibility_project_id
                    WHERE id = document_set_row.id;
                END IF;

                UPDATE amendment_batch
                SET project_id = compatibility_project_id
                WHERE document_set_id = document_set_row.id;
                UPDATE benchmark_question
                SET project_id = compatibility_project_id
                WHERE document_set_id = document_set_row.id;
                INSERT INTO project__user_file (
                    project_id,
                    user_file_id,
                    created_at
                )
                SELECT
                    compatibility_project_id,
                    user_file_id,
                    created_at
                FROM document_set__user_file
                WHERE document_set_id = document_set_row.id
                ON CONFLICT (project_id, user_file_id) DO NOTHING;
            END LOOP;

            IF EXISTS (
                SELECT 1 FROM amendment_batch WHERE project_id IS NULL
            ) OR EXISTS (
                SELECT 1 FROM benchmark_question WHERE project_id IS NULL
            ) THEN
                RAISE EXCEPTION
                    'Could not restore all regulatory rows to projects';
            END IF;
        END $$
        """
    )

    op.alter_column("amendment_batch", "project_id", nullable=False)
    op.alter_column("benchmark_question", "project_id", nullable=False)
    op.create_foreign_key(
        "amendment_batch_project_id_fkey",
        "amendment_batch",
        "user_project",
        ["project_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_foreign_key(
        "benchmark_question_project_id_fkey",
        "benchmark_question",
        "user_project",
        ["project_id"],
        ["id"],
        ondelete="RESTRICT",
    )
    op.create_index("ix_amendment_batch_project_id", "amendment_batch", ["project_id"])
    op.create_index(
        "ix_benchmark_question_project_id", "benchmark_question", ["project_id"]
    )

    op.drop_index("ix_amendment_batch_document_set_id", table_name="amendment_batch")
    op.drop_index(
        "ix_benchmark_question_document_set_id", table_name="benchmark_question"
    )
    op.drop_constraint(
        "amendment_batch_document_set_id_fkey",
        "amendment_batch",
        type_="foreignkey",
    )
    op.drop_constraint(
        "benchmark_question_document_set_id_fkey",
        "benchmark_question",
        type_="foreignkey",
    )
    op.drop_column("amendment_batch", "document_set_id")
    op.drop_column("benchmark_question", "document_set_id")


def _delete_migration_generated_document_sets() -> None:
    # These relationships do not all cascade in the legacy schema. They must be
    # cleared before removing only the document sets created by this migration.
    for association_table in (
        "persona__document_set",
        "document_set__user",
        "document_set__connector_credential_pair",
        "document_set__user_group",
        "federated_connector__document_set",
    ):
        op.execute(
            sa.text(
                f"""
                DELETE FROM {association_table}
                WHERE document_set_id IN (
                    SELECT id
                    FROM document_set
                    WHERE migrated_from_project_id IS NOT NULL
                )
                """
            )
        )
    op.execute("DELETE FROM document_set WHERE migrated_from_project_id IS NOT NULL")


def downgrade() -> None:
    _ensure_migration_generated_sets_are_downgrade_safe()
    _restore_regulatory_scope_to_compatibility_projects()
    _delete_migration_generated_document_sets()
    op.drop_index("ux_document_set_migrated_from_project_id", table_name="document_set")
    op.drop_column("document_set", "migrated_from_project_id")
    op.drop_index(
        "ix_document_set__user_file_user_file_id",
        table_name="document_set__user_file",
    )
    op.drop_index(
        "ix_document_set__user_file_document_set_id_created_at",
        table_name="document_set__user_file",
    )
    op.drop_table("document_set__user_file")
    op.drop_column("user_file", "needs_document_set_sync")
    op.drop_column("document_set", "is_deleting")
