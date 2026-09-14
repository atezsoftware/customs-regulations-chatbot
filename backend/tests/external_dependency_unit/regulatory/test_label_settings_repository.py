import importlib.util
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from types import ModuleType
from uuid import uuid4

import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import inspect, select, text, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from onyx.db import regulatory_labeling as repository
from onyx.db.labeling_configuration import resolve_labeling_provider_binding
from onyx.db.models import (
    RegulatoryChunk,
    RegulatoryDerivedLabelProjection,
    RegulatoryLabelingItem,
    RegulatoryLabelingRun,
    RegulatoryLabelingShard,
    RegulatoryLabelSettings,
    RegulatoryLabelTaxonomy,
    User,
    UserFile,
)
from onyx.regulatory.labeling.provider import LabelDefinition, TaxonomyDefinition
from tests.external_dependency_unit.regulatory.test_labeling_jobs import (
    LabelingData,
    _prepare,
    _start,
)
from tests.external_dependency_unit.regulatory.test_labeling_jobs import (
    labeling_data as labeling_data,
)
from tests.external_dependency_unit.regulatory.test_labeling_jobs import (
    labeling_database as labeling_database,
)


def test_settings_are_seeded_in_db_and_concurrent_edits_do_not_overwrite(
    labeling_data: LabelingData,
) -> None:
    with Session(labeling_data.database.engine) as session:
        settings = repository.get_label_settings(session)
        assert settings.revision == 2
        original_id = settings.taxonomy_id
        assert settings.taxonomy.label_count == 165
        original = TaxonomyDefinition.model_validate(settings.taxonomy.definition)
    barrier = Barrier(2)

    def save(description: str) -> str:
        with Session(labeling_data.database.engine) as session:
            barrier.wait(timeout=5)
            try:
                repository.update_label_settings(
                    session,
                    labels=[
                        LabelDefinition(
                            id="editable", name="Editable", description=description
                        )
                    ],
                    expected_revision=2,
                    updated_by_id=labeling_data.user_id,
                )
                session.commit()
                return description
            except repository.LabelingStateConflictError:
                session.rollback()
                return "conflict"

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(save, ["First edit", "Second edit"]))
    assert results.count("conflict") == 1
    winner = next(result for result in results if result != "conflict")
    with Session(labeling_data.database.engine) as session:
        current = repository.get_label_settings(session)
        assert current.revision == 3
        assert current.updated_by_id == labeling_data.user_id
        assert (
            TaxonomyDefinition.model_validate(current.taxonomy.definition)
            .labels[0]
            .description
            == winner
        )
        previous = repository.get_taxonomy(session, original_id)
        assert previous is not None
        assert previous.definition == original.model_dump()


def test_settings_save_and_default_replay_preserve_source_chunks_and_original_run(
    labeling_data: LabelingData,
) -> None:
    key = uuid4()
    with Session(labeling_data.database.engine) as session:
        source_query = (
            select(RegulatoryChunk.__table__)
            .where(RegulatoryChunk.user_file_id == labeling_data.file_id)
            .order_by(RegulatoryChunk.id)
        )
        source_before = [dict(row) for row in session.execute(source_query).mappings()]
        user = session.get(User, labeling_data.user_id)
        assert user is not None
        binding = resolve_labeling_provider_binding(
            session, labeling_data.model_id, user=user
        )
        original = repository.get_label_settings(session)
        run, created = repository.create_labeling_run(
            session,
            document_set_id=labeling_data.document_set_id,
            taxonomy=original.taxonomy,
            model_configuration_id=labeling_data.model_id,
            model="gemini-3.8-flash",
            provider_binding=binding.model_dump(mode="json"),
            requested_by_id=user.id,
            idempotency_key=key,
            uses_current_labels=True,
        )
        assert created
        original_run_id = run.id
        original_taxonomy_id = run.taxonomy_id
        session.commit()
        updated = repository.update_label_settings(
            session,
            labels=[
                LabelDefinition(
                    id="updated", name="Updated name", description="Updated description"
                )
            ],
            expected_revision=original.revision,
            updated_by_id=user.id,
        )
        session.commit()
        replay, created_again = repository.create_labeling_run(
            session,
            document_set_id=labeling_data.document_set_id,
            taxonomy=updated.taxonomy,
            model_configuration_id=labeling_data.model_id,
            model="gemini-3.8-flash",
            provider_binding=binding.model_dump(mode="json"),
            requested_by_id=user.id,
            idempotency_key=key,
            uses_current_labels=True,
        )
        assert not created_again
        assert replay.id == original_run_id
        assert replay.taxonomy_id == original_taxonomy_id
        assert replay.taxonomy_id != updated.taxonomy_id
        assert [
            dict(row) for row in session.execute(source_query).mappings()
        ] == source_before


def test_invalid_save_leaves_the_active_revision_unchanged(
    labeling_data: LabelingData,
) -> None:
    with Session(labeling_data.database.engine) as session:
        current = repository.get_label_settings(session)
        revision, taxonomy_id = current.revision, current.taxonomy_id
        duplicate = LabelDefinition(
            id="same", name="Same", description="Same definition"
        )
        with pytest.raises(ValueError, match="unique"):
            repository.update_label_settings(
                session,
                labels=[duplicate, duplicate],
                expected_revision=revision,
                updated_by_id=labeling_data.user_id,
            )
        session.rollback()
        current = repository.get_label_settings(session)
        assert (current.revision, current.taxonomy_id) == (revision, taxonomy_id)


def test_settings_migration_round_trip_preserves_existing_jobs_and_source_chunks(
    labeling_data: LabelingData,
) -> None:
    run_id, _ = _start(labeling_data)
    source_query = (
        select(RegulatoryChunk.__table__)
        .where(RegulatoryChunk.user_file_id == labeling_data.file_id)
        .order_by(RegulatoryChunk.id)
    )
    run_query = select(RegulatoryLabelingRun.__table__).where(
        RegulatoryLabelingRun.id == run_id
    )
    with Session(labeling_data.database.engine) as session:
        source_before = [dict(row) for row in session.execute(source_query).mappings()]
        run_before = dict(session.execute(run_query).mappings().one())
    migration = _migration("8d19d521d9fa_add_editable_regulatory_label_settings.py")
    with labeling_data.database.engine.begin() as connection:
        connection.execute(
            text(f'SET LOCAL search_path TO "{labeling_data.database.schema}"')
        )
        with Operations.context(MigrationContext.configure(connection)):
            migration.downgrade()
            assert not inspect(connection).has_table(
                "regulatory_label_settings", schema=labeling_data.database.schema
            )
            migration.upgrade()
    with Session(labeling_data.database.engine) as session:
        settings = repository.get_label_settings(session)
        assert settings.taxonomy.label_count == 255
        assert [
            dict(row) for row in session.execute(source_query).mappings()
        ] == source_before
        assert dict(session.execute(run_query).mappings().one()) == run_before


def _migration(filename: str) -> ModuleType:
    path = Path(__file__).resolve().parents[3] / "alembic/versions" / filename
    spec = importlib.util.spec_from_file_location(filename.removesuffix(".py"), path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_future_migrations_preserve_labels_seeded_before_full_deployment(
    labeling_data: LabelingData,
) -> None:
    jobs = _migration("c8b7a6d5e4f3_add_durable_regulatory_chunk_labeling.py")
    settings_migration = _migration(
        "8d19d521d9fa_add_editable_regulatory_label_settings.py"
    )
    edited = TaxonomyDefinition(
        name="Edited in Label Settings",
        labels=[
            LabelDefinition(
                id="kept", name="Saved name", description="Saved description"
            )
        ],
    )
    taxonomy_id = uuid4()
    with labeling_data.database.engine.begin() as connection:
        connection.execute(
            text(f'SET LOCAL search_path TO "{labeling_data.database.schema}"')
        )
        with Operations.context(MigrationContext.configure(connection)):
            settings_migration.downgrade()
            jobs.downgrade()
        taxonomy_table = RegulatoryLabelTaxonomy.metadata.tables[
            "regulatory_label_taxonomy"
        ]
        settings_table = RegulatoryLabelSettings.metadata.tables[
            "regulatory_label_settings"
        ]
        taxonomy_table.create(connection)
        settings_table.create(connection)
        connection.execute(
            taxonomy_table.insert().values(
                id=taxonomy_id,
                name=edited.name,
                definition=edited.model_dump(),
                version_hash=edited.version_hash,
                label_count=1,
            )
        )
        connection.execute(
            settings_table.insert().values(
                id=1,
                taxonomy_id=taxonomy_id,
                revision=2,
            )
        )
        with Operations.context(MigrationContext.configure(connection)):
            jobs.upgrade()
            settings_migration.upgrade()
    with Session(labeling_data.database.engine) as session:
        current = repository.get_label_settings(session)
        assert current.revision == 2
        assert current.taxonomy_id == taxonomy_id
        assert current.taxonomy.definition == edited.model_dump()
        assert session.get(RegulatoryChunk, labeling_data.first_id) is not None


def test_label_migration_rejects_incompatible_preexisting_table(
    labeling_data: LabelingData,
) -> None:
    migration = _migration("c8b7a6d5e4f3_add_durable_regulatory_chunk_labeling.py")
    with labeling_data.database.engine.connect() as connection:
        transaction = connection.begin()
        try:
            connection.execute(
                text(f'SET LOCAL search_path TO "{labeling_data.database.schema}"')
            )
            connection.execute(
                text(
                    "ALTER TABLE regulatory_label_taxonomy ALTER COLUMN name DROP NOT NULL"
                )
            )
            with (
                Operations.context(MigrationContext.configure(connection)),
                pytest.raises(RuntimeError, match="schema|incompatible|column"),
            ):
                migration.upgrade()
        finally:
            transaction.rollback()


_LEGACY_DEFAULT_HASH = (
    "5a89e4d393c2974a900e15bb57914814633e70ce65b0f5f39f02b7d24b7b50bd"
)
_CHUNK_DEFAULT_HASH = "6ff25f4865bcd1107dd7f7dc51120476f05339327498d5d9a2b422b130193193"
_SCOPE_MIGRATION = "b27e6a4c1d90_scope_default_chunk_labels.py"


def test_scope_migration_updates_only_untouched_default_and_preserves_history(
    labeling_data: LabelingData,
) -> None:
    with Session(labeling_data.database.engine) as session:
        legacy = session.scalar(
            select(RegulatoryLabelTaxonomy).where(
                RegulatoryLabelTaxonomy.version_hash == _LEGACY_DEFAULT_HASH
            )
        )
        assert legacy is not None
        legacy_id = legacy.id
    run_id, _ = _start(labeling_data._replace(taxonomy_id=legacy_id))
    _prepare(labeling_data, run_id)
    migration = _migration(_SCOPE_MIGRATION)
    with labeling_data.database.engine.connect() as connection:
        transaction = connection.begin()
        try:
            connection.execute(
                text(f'SET LOCAL search_path TO "{labeling_data.database.schema}"')
            )
            connection.execute(
                update(RegulatoryLabelSettings)
                .where(RegulatoryLabelSettings.id == 1)
                .values(
                    taxonomy_id=legacy_id,
                    revision=1,
                    updated_by_id=None,
                    updated_at=text("'2000-01-01T00:00:00Z'::timestamptz"),
                )
            )
            preserved_tables = (
                RegulatoryChunk.__table__,
                UserFile.__table__,
                RegulatoryLabelingRun.__table__,
                RegulatoryLabelingItem.__table__,
                RegulatoryLabelingShard.__table__,
                RegulatoryDerivedLabelProjection.__table__,
            )
            before = {
                table: [
                    dict(row)
                    for row in connection.execute(
                        select(table).order_by(table.c.id)
                    ).mappings()
                ]
                for table in preserved_tables
            }
            legacy_query = select(RegulatoryLabelTaxonomy.__table__).where(
                RegulatoryLabelTaxonomy.id == legacy_id
            )
            legacy_before = dict(connection.execute(legacy_query).mappings().one())
            with Operations.context(MigrationContext.configure(connection)):
                migration.upgrade()
            current = (
                connection.execute(select(RegulatoryLabelSettings.__table__))
                .mappings()
                .one()
            )
            assert current["revision"] == 2
            assert current["updated_by_id"] is None
            assert current["updated_at"].year > 2000
            scoped = (
                connection.execute(
                    select(RegulatoryLabelTaxonomy.__table__).where(
                        RegulatoryLabelTaxonomy.id == current["taxonomy_id"]
                    )
                )
                .mappings()
                .one()
            )
            assert scoped["version_hash"] == _CHUNK_DEFAULT_HASH
            assert scoped["label_count"] == 165
            assert len(scoped["definition"]["labels"]) == 165
            assert all(
                label["id"].startswith(("SUB.", "EFF.", "ANX.", "SEC."))
                for label in scoped["definition"]["labels"]
            )
            assert current["taxonomy_id"] != legacy_id
            with Operations.context(MigrationContext.configure(connection)):
                migration.upgrade()
                migration.downgrade()
            assert dict(
                connection.execute(select(RegulatoryLabelSettings.__table__))
                .mappings()
                .one()
            ) == dict(current)
            assert (
                dict(connection.execute(legacy_query).mappings().one()) == legacy_before
            )
            for table in preserved_tables:
                assert [
                    dict(row)
                    for row in connection.execute(
                        select(table).order_by(table.c.id)
                    ).mappings()
                ] == before[table]
        finally:
            transaction.rollback()


@pytest.mark.parametrize(
    "current_kind", ["custom", "edited_legacy", "attributed_legacy", "scoped"]
)
def test_scope_migration_preserves_current_or_edited_settings(
    labeling_data: LabelingData, current_kind: str
) -> None:
    migration = _migration(_SCOPE_MIGRATION)
    with labeling_data.database.engine.connect() as connection:
        transaction = connection.begin()
        try:
            connection.execute(
                text(f'SET LOCAL search_path TO "{labeling_data.database.schema}"')
            )
            if current_kind == "scoped":
                seed_path = (
                    Path(__file__).resolve().parents[3]
                    / "onyx/regulatory/labeling/data/tariff-regulatory-intelligence-chunk-labels-v1.json"
                )
                scoped = TaxonomyDefinition.model_validate_json(seed_path.read_text())
                connection.execute(
                    insert(RegulatoryLabelTaxonomy)
                    .values(
                        id=uuid4(),
                        name=scoped.name,
                        version_hash=scoped.version_hash,
                        definition=scoped.model_dump(),
                        label_count=len(scoped.labels),
                    )
                    .on_conflict_do_nothing(index_elements=["version_hash"])
                )
            taxonomy_id = (
                labeling_data.taxonomy_id
                if current_kind == "custom"
                else connection.execute(
                    select(RegulatoryLabelTaxonomy.id).where(
                        RegulatoryLabelTaxonomy.version_hash
                        == (
                            _CHUNK_DEFAULT_HASH
                            if current_kind == "scoped"
                            else _LEGACY_DEFAULT_HASH
                        )
                    )
                ).scalar_one()
            )
            connection.execute(
                update(RegulatoryLabelSettings)
                .where(RegulatoryLabelSettings.id == 1)
                .values(
                    taxonomy_id=taxonomy_id,
                    revision=2 if current_kind == "edited_legacy" else 1,
                    updated_by_id=(
                        labeling_data.user_id
                        if current_kind == "attributed_legacy"
                        else None
                    ),
                )
            )
            settings_query = select(RegulatoryLabelSettings.__table__)
            taxonomy_query = select(RegulatoryLabelTaxonomy.__table__).order_by(
                RegulatoryLabelTaxonomy.id
            )
            before = dict(connection.execute(settings_query).mappings().one())
            taxonomies_before = [
                dict(row) for row in connection.execute(taxonomy_query).mappings()
            ]
            with Operations.context(MigrationContext.configure(connection)):
                migration.upgrade()
                migration.downgrade()
                migration.upgrade()
            assert dict(connection.execute(settings_query).mappings().one()) == before
            assert [
                dict(row) for row in connection.execute(taxonomy_query).mappings()
            ] == taxonomies_before
        finally:
            transaction.rollback()
