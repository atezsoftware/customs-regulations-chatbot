"""Run catalogue lookups on an isolated SQL database and audit additive DDL."""

import importlib.util
import io
import json
import sqlite3
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock
from uuid import UUID

import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from pydantic import ValidationError
from sqlalchemy import (
    JSON,
    Column,
    Integer,
    MetaData,
    Table,
    Uuid,
    create_engine,
    event,
    select,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from onyx.context.search.models import IndexFilters, Tag
from onyx.db import regulatory_source_notices as notices
from onyx.db.asv3_corpus import CorpusScopeUnavailable
from onyx.db.models import RegulatorySourceNotice, User
from onyx.regulatory.source_notice_identity import notice_title_alias
from onyx.regulatory.source_notices import (
    SourceNotice,
    parse_source_notice_id,
    source_notice_id,
)

FIRST = UUID(int=101)
SECOND = UUID(int=102)
OTHER_SET = UUID(int=103)
OWNER = cast(User, SimpleNamespace(id=UUID(int=1)))


def notice_values(identifier: UUID = FIRST, **changes: Any) -> dict[str, Any]:
    return {
        "id": identifier,
        "document_set_id": 15,
        "identity_key": "circular:2012:7",
        "instrument_type": "circular",
        "instrument_number": "7",
        "instrument_year": 2012,
        "instrument_date": None,
        "catalog_title": "Gümrük İşlemleri (Mülga)",
        "aliases": ["2012/7 sayılı Genelge"],
        "legistation_metadata": {
            "validity_status": "Mülga",
            "repeal_note": "Kaynak liste kaydı.",
        },
        "provenance": {"sha256": "immutable-excel-hash", "rows": [1, 9]},
        "content_state": "metadata_only",
        "user_file_id": None,
        "revision": 1,
        "created_at": datetime(2026, 10, 22, tzinfo=timezone.utc),
        "updated_at": datetime(2026, 10, 22, tzinfo=timezone.utc),
        **changes,
    }


@pytest.fixture
def database(monkeypatch: pytest.MonkeyPatch) -> Iterator[Session]:
    metadata = MetaData()
    Table("document_set", metadata, Column("id", Integer, primary_key=True))
    Table("user_file", metadata, Column("id", Uuid, primary_key=True))
    registry = cast(Table, RegulatorySourceNotice.__table__).to_metadata(metadata)
    for column in registry.columns:
        if isinstance(column.type, JSONB):
            column.type = JSON()
        if isinstance(column.type, PGUUID):
            column.type = Uuid()
        column.server_default = None
    engine = create_engine("sqlite://")

    @event.listens_for(engine, "connect")
    def enable_jsonb_membership(connection: object, _record: object) -> None:
        cast(sqlite3.Connection, connection).create_function(
            "jsonb_exists", 2, lambda value, key: key in json.loads(value)
        )

    metadata.create_all(engine)
    with engine.begin() as connection:
        connection.execute(
            registry.insert(),
            [
                notice_values(),
                notice_values(
                    SECOND,
                    identity_key="law:2908",
                    instrument_type="law",
                    instrument_year=None,
                    instrument_number="2908",
                ),
                notice_values(OTHER_SET, document_set_id=16),
            ],
        )

    def bind_scope(
        _session: Session, *, user: User, filters: IndexFilters
    ) -> IndexFilters:
        assert user.id == OWNER.id
        return filters.model_copy(
            update={"asv3_document_set_id": 15, "forced_document_set": ["PC Külliyatı"]}
        )

    scope = MagicMock(side_effect=bind_scope)
    monkeypatch.setattr(notices, "resolve_pc_corpus_scope", scope)
    with Session(engine) as session:
        yield session
    engine.dispose()


def filters(**changes: Any) -> IndexFilters:
    return IndexFilters(access_control_list=None, **changes)


def test_exact_typed_reference_reads_only_requested_identity(database: Session) -> None:
    rows = notices.find_source_notices(
        database, OWNER, filters(), queries=["2012/7 sayılı Genelge"]
    )
    assert len(rows) == 1
    assert rows[0].id == str(FIRST)
    assert rows[0].provenance["rows"] == [1, 9]
    assert rows[0].legistation_metadata.validity_status == "Mülga"
    assert rows[0].user_file_id is None
    assert rows[0].content_state == "metadata_only"
    assert not hasattr(rows[0], "text")
    assert not database.new
    assert not database.dirty
    assert not database.deleted


@pytest.mark.parametrize(
    "query",
    [
        "7",
        "2012/7",
        "2013/7 sayılı Genelge",
        "2012/7 sayılı Karar",
        "Gümrük",
        "5253 sayılı Kanun",
        "2012/7 sayılı Genelge%%%",
    ],
)
def test_incomplete_wrong_type_year_title_or_injection_does_not_broaden_lookup(
    database: Session, query: str
) -> None:
    rows = notices.find_source_notices(database, OWNER, filters(), queries=[query])
    assert [row.id for row in rows] == ([str(FIRST)] if query.endswith("%%%") else [])


def test_ids_revalidate_set_and_ignore_malformed_uuid(database: Session) -> None:
    rows = notices.find_source_notices(
        database,
        OWNER,
        filters(),
        notice_ids=[str(FIRST), str(OTHER_SET), "not-a-uuid"],
    )
    assert [row.id for row in rows] == [str(FIRST)]
    assert notices.find_source_notices(database, OWNER, filters()) == []
    assert notices.find_source_notices(database, OWNER, filters(), queries=[]) == []


@pytest.mark.parametrize(
    "scope_error",
    [PermissionError("Private set"), CorpusScopeUnavailable("Missing PC set")],
)
def test_failed_scope_never_returns_registry_existence(
    database: Session, monkeypatch: pytest.MonkeyPatch, scope_error: Exception
) -> None:
    gate = MagicMock(side_effect=scope_error)
    monkeypatch.setattr(notices, "resolve_pc_corpus_scope", gate)
    lookup = MagicMock()
    monkeypatch.setattr(database, "scalars", lookup)
    with pytest.raises(type(scope_error)):
        notices.find_source_notices(
            database, OWNER, filters(tenant_id="wrong-tenant"), notice_ids=[str(FIRST)]
        )
    lookup.assert_not_called()


@pytest.mark.parametrize(
    "narrow",
    [
        {"tags": [Tag(tag_key="only", tag_value="private")]},
        {"regulatory_candidate_ids": []},
        {"regulatory_source_hint": "Some file"},
        {"hierarchy_node_ids": [7]},
    ],
)
def test_unsupported_narrow_scopes_fail_closed(
    database: Session, narrow: dict[str, Any]
) -> None:
    with pytest.raises(CorpusScopeUnavailable):
        notices.find_source_notices(
            database, OWNER, filters(**narrow), notice_ids=[str(FIRST)]
        )


@pytest.mark.parametrize(
    "narrow",
    [
        {"project_id_filter": 4},
        {"persona_id_filter": 8},
        {"attached_document_ids": [str(UUID(int=900))]},
        {"document_set": ["Other authorized set"]},
    ],
)
def test_metadata_only_does_not_invent_file_scope_membership(
    database: Session, narrow: dict[str, Any]
) -> None:
    assert (
        notices.find_source_notices(
            database, OWNER, filters(**narrow), notice_ids=[str(FIRST)]
        )
        == []
    )


def test_later_materialized_notice_revalidates_actual_file_publication_and_acl(
    database: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = database.get(RegulatorySourceNotice, FIRST)
    assert record is not None
    record.content_state = "materialized"
    record.user_file_id = UUID(int=999)
    record.revision = 2
    database.commit()
    require = MagicMock()
    monkeypatch.setattr(notices, "require_source", require)
    result = notices.find_source_notices(
        database, OWNER, filters(project_id_filter=4), notice_ids=[str(FIRST)]
    )
    assert result[0].content_state == "materialized"
    assert result[0].revision == 2
    assert result[0].user_file_id == str(UUID(int=999))
    assert require.call_args.kwargs["source_id"] == UUID(int=999)
    assert require.call_args.kwargs["filters"].project_id_filter == 4
    require.side_effect = PermissionError("File ACL revoked")
    assert (
        notices.find_source_notices(database, OWNER, filters(), notice_ids=[str(FIRST)])
        == []
    )


def test_registry_owns_identity_and_content_constraints(database: Session) -> None:
    with pytest.raises(IntegrityError):
        database.add(RegulatorySourceNotice(**notice_values(UUID(int=777))))
        database.flush()
    database.rollback()
    with pytest.raises(IntegrityError):
        database.add(
            RegulatorySourceNotice(
                **notice_values(
                    UUID(int=778), identity_key="law:999", content_state="materialized"
                )
            )
        )
        database.flush()
    database.rollback()
    assert len(list(database.scalars(select(RegulatorySourceNotice)))) == 3


def test_read_lookup_never_flushes_unrelated_pending_writes(database: Session) -> None:
    pending = RegulatorySourceNotice(
        **notice_values(UUID(int=778), identity_key="law:999")
    )
    database.add(pending)
    rows = notices.find_source_notices(
        database, OWNER, filters(), queries=["2012/7 sayılı Genelge"]
    )
    assert [row.id for row in rows] == [str(FIRST)]
    assert pending in database.new
    database.rollback()
    assert len(list(database.scalars(select(RegulatorySourceNotice)))) == 3


def add_catalogue_observation(
    database: Session,
    identifier: int,
    title: str,
    *,
    aliases: list[str] | None = None,
    document_set_id: int = 15,
) -> str:
    key = "catalog:" + f"{identifier:064x}"
    title_key = notice_title_alias(title)
    assert title_key is not None
    database.add(
        RegulatorySourceNotice(
            **notice_values(
                UUID(int=identifier),
                document_set_id=document_set_id,
                identity_key=key,
                instrument_type="catalogue_observation",
                instrument_number=None,
                instrument_year=None,
                instrument_date=None,
                catalog_title=title,
                aliases=[key, title_key, title, *(aliases or [])],
            )
        )
    )
    database.commit()
    return key


def test_name_only_catalogue_observation_resolves_by_exact_title_and_opaque_key(
    database: Session,
) -> None:
    key = add_catalogue_observation(database, 501, "(Mülga) Tarihsiz Belge Başlığı")
    for query in [
        key,
        "Tarihsiz Belge Başlığı",
        "TARİHSİZ BELGE BAŞLIĞI",
        notice_title_alias("Tarihsiz Belge Başlığı"),
    ]:
        assert query is not None
        rows = notices.find_source_notices(database, OWNER, filters(), queries=[query])
        assert [row.id for row in rows] == [str(UUID(int=501))]
        assert rows[0].instrument_date is None and rows[0].instrument_number is None
        assert rows[0].instrument_type == "catalogue_observation"
    assert (
        notices.find_source_notices(
            database, OWNER, filters(), queries=["Tarihsiz Belge"]
        )
        == []
    )


def test_same_number_different_topics_are_distinct_ambiguous_candidates(
    database: Session,
) -> None:
    add_catalogue_observation(
        database,
        511,
        "1995/5 sayılı Genelge — Konu A (Mülga)",
        aliases=["circular:1995:5"],
    )
    add_catalogue_observation(
        database,
        512,
        "1995/5 sayılı Genelge — Konu B (Mülga)",
        aliases=["circular:1995:5"],
    )
    generic = notices.find_source_notices(
        database, OWNER, filters(), queries=["1995/5 sayılı Genelge"]
    )
    assert [row.id for row in generic] == [str(UUID(int=511)), str(UUID(int=512))]
    for identifier, topic in [(511, "A"), (512, "B")]:
        exact = notices.find_source_notices(
            database,
            OWNER,
            filters(),
            queries=[f"1995/5 sayılı Genelge — Konu {topic}"],
        )
        assert [row.id for row in exact] == [str(UUID(int=identifier))]
    batch = notices.find_source_notices(
        database,
        OWNER,
        filters(),
        queries=["1995/5 sayılı Genelge — Konu A", "law:2908"],
    )
    assert {row.id for row in batch} == {str(SECOND), str(UUID(int=511))}


def test_exact_title_aliases_are_not_merged_and_remain_document_set_scoped(
    database: Session,
) -> None:
    add_catalogue_observation(database, 521, "Kamu Düzeni (Mülga)")
    add_catalogue_observation(database, 522, "Kamu Düzeni (Mülga)")
    add_catalogue_observation(database, 523, "Kamu Düzeni (Mülga)", document_set_id=16)
    rows = notices.find_source_notices(
        database, OWNER, filters(), queries=["Kamu Düzeni"]
    )
    assert [row.id for row in rows] == [str(UUID(int=521)), str(UUID(int=522))]
    assert len({row.identity_key for row in rows}) == 2


def test_legacy_typed_notice_full_title_resolves_without_rewriting_aliases(
    database: Session,
) -> None:
    other = database.get(RegulatorySourceNotice, SECOND)
    assert other is not None
    other.catalog_title = "Farklı Eski Kayıt"
    database.commit()
    record = database.get(RegulatorySourceNotice, FIRST)
    assert record is not None
    aliases_before = list(record.aliases)
    assert not any(value.startswith("title:") for value in aliases_before)
    rows = notices.find_source_notices(
        database, OWNER, filters(), queries=["GÜMRÜK İŞLEMLERİ"]
    )
    assert [row.id for row in rows] == [str(FIRST)]
    assert record.aliases == aliases_before
    assert not database.new and not database.dirty and not database.deleted


def test_typed_notice_and_explicit_target_are_separate_from_file_identity(
    database: Session,
) -> None:
    notice = notices.find_source_notices(
        database, OWNER, filters(), notice_ids=[str(FIRST)]
    )[0]
    assert source_notice_id(notice.id) == "notice:" + str(FIRST)
    assert parse_source_notice_id(source_notice_id(notice.id)) == str(FIRST)
    assert parse_source_notice_id(str(FIRST)) is None
    assert parse_source_notice_id("notice:invalid") is None
    with pytest.raises(ValidationError):
        SourceNotice.model_validate(
            {**notice.model_dump(), "content_state": "materialized"}
        )


def test_migration_sql_creates_only_new_registry_and_own_indexes() -> None:
    path = (
        Path(__file__).resolve().parents[4]
        / "alembic/versions/f3b39e5c99db_add_regulatory_source_notice_registry.py"
    )
    spec = importlib.util.spec_from_file_location("source_notice_migration", path)
    assert spec is not None and spec.loader is not None
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    buffer = io.StringIO()
    context = MigrationContext.configure(
        dialect_name="postgresql", opts={"as_sql": True, "output_buffer": buffer}
    )
    setattr(migration, "op", Operations(context))
    getattr(migration, "upgrade")()
    sql = buffer.getvalue()
    statements = [part.strip() for part in sql.split(";") if part.strip()]
    assert len(statements) == 2
    assert statements[0].startswith("CREATE TABLE regulatory_source_notice")
    assert statements[1].startswith(
        "CREATE INDEX ix_regulatory_source_notice_instrument ON regulatory_source_notice"
    )
    assert "UNIQUE (document_set_id, identity_key)" in sql
    assert "REFERENCES document_set (id) ON DELETE RESTRICT" in sql
    assert "REFERENCES user_file (id) ON DELETE RESTRICT" in sql
    assert all(
        value not in sql.upper()
        for value in [
            "ALTER TABLE",
            "DROP ",
            "DELETE FROM",
            "UPDATE ",
            "INSERT INTO",
            "ALEMBIC_VERSION",
        ]
    )
