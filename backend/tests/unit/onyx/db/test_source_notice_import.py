import hashlib
from pathlib import Path
from typing import cast

import pytest
from openpyxl import Workbook
from openpyxl.worksheet.worksheet import Worksheet

from onyx.db.source_notice_import import (
    ExistingNoticeSource,
    build_source_notice_manifest,
    read_notice_workbook,
)


def _workbook(tmp_path: Path, rows: list[list[str | int | None]]) -> tuple[Path, str]:
    workbook = Workbook()
    sheet = cast(Worksheet, workbook.active)
    for row in rows:
        sheet.append(row)
    path = tmp_path / "mülga.xlsx"
    workbook.save(path)
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_first_row_duplicates_and_existing_source_are_preserved(tmp_path: Path) -> None:
    path, digest = _workbook(
        tmp_path,
        [
            ["Genelgeler", 2012, 7, "Eski taşıma işlemi (Mülga)"],
            ["Genelgeler", 2012, 7, "Eski taşıma işlemi (Mülga)"],
            [
                "Kararname ve Mahkeme Kararları",
                2024,
                9366,
                "Muafiyet kararı 2024/9366 (Mülga)",
            ],
            ["Diğer Yönetmelikler", None, "Aynı adlı yönetmelik (Mülga)", None],
        ],
    )
    source = ExistingNoticeSource(id="existing", name="KARARLAR/karar_2024-9366.md")
    manifest = build_source_notice_manifest(
        workbook_path=path, workbook_sha256=digest, sources=[source], document_set_id=15
    )
    assert manifest.raw_row_count == 4
    assert len(manifest.targets) == 1
    target = manifest.targets[0]
    assert target.identity_key == "circular:2012:7"
    assert target.provenance["excel_rows"] == [1, 2]
    assert target.content_state == "metadata_only"
    assert target.user_file_id is None
    assert target.legistation_metadata == {
        "validity_status": "Mülga",
        "repeal_note": "Eski taşıma işlemi (Mülga)",
        "validity_source": path.name,
    }
    assert "official_title" not in target.legistation_metadata
    assert (
        next(
            decision
            for decision in manifest.decisions
            if decision.identity_key == "decision:2024:9366"
        ).classification
        == "existing"
    )
    assert manifest.decisions[-1].classification == "ambiguous"


def test_same_year_number_different_issuer_is_held(tmp_path: Path) -> None:
    path, digest = _workbook(
        tmp_path,
        [
            ["Genelgeler", 2012, 7, "Makam A taşıma işlemi (Mülga)"],
            ["Genelgeler", 2012, 7, "Makam B başka işlem (Mülga)"],
        ],
    )
    manifest = build_source_notice_manifest(
        workbook_path=path,
        workbook_sha256=digest,
        sources=[],
        document_set_id=15,
        identity_heads=[],
    )
    assert manifest.targets == []
    assert (
        manifest.decisions[0].reason
        == "multiple_catalog_records_share_unqualified_identity"
    )
    assert manifest.decisions[0].excel_rows == [1, 2]


def test_possible_number_or_title_is_held_without_claiming_same_identity(
    tmp_path: Path,
) -> None:
    path, digest = _workbook(
        tmp_path, [["Diğer Kanunlar", 2908, "Dernekler Kanunu (Mülga)", None]]
    )
    manifest = build_source_notice_manifest(
        workbook_path=path,
        workbook_sha256=digest,
        sources=[ExistingNoticeSource(id="successor", name="5253_Dernekler_Kanunu.md")],
        document_set_id=15,
        identity_heads=[],
    )
    # The law identity is retained independently; a shared two-word title alone
    # is not enough to assert that the existing successor is the older law.
    assert manifest.decisions[0].identity_key == "law:2908"
    assert manifest.decisions[0].classification != "existing"


def test_unrelated_number_inside_description_is_not_own_identity(
    tmp_path: Path,
) -> None:
    path, digest = _workbook(
        tmp_path,
        [
            [
                "Tebliğler",
                1,
                "6111 sayılı Kanun uygulanmasına ilişkin Tebliğ (Mülga)",
                None,
            ]
        ],
    )
    manifest = build_source_notice_manifest(
        workbook_path=path,
        workbook_sha256=digest,
        sources=[],
        document_set_id=15,
        identity_heads=[],
    )
    assert manifest.targets == []
    assert manifest.decisions[0].identity_key is None


def test_conflicting_catalog_year_is_held(tmp_path: Path) -> None:
    path, digest = _workbook(
        tmp_path,
        [["Kararname ve Mahkeme Kararları", 2023, 6622, "Karar 2022/6622 (Mülga)"]],
    )
    manifest = build_source_notice_manifest(
        workbook_path=path,
        workbook_sha256=digest,
        sources=[],
        document_set_id=15,
        identity_heads=[],
    )
    assert manifest.targets == []
    assert (
        manifest.decisions[0].reason
        == "structured_identity_conflicts_with_catalog_text"
    )


def test_changed_workbook_and_formulas_are_rejected(tmp_path: Path) -> None:
    path, digest = _workbook(tmp_path, [["Genelgeler", 2012, 7, "=1+1"]])
    with pytest.raises(ValueError, match="Formula"):
        read_notice_workbook(path, expected_sha256=digest)
    with pytest.raises(ValueError, match="SHA256"):
        read_notice_workbook(path, expected_sha256="0" * 64)


def test_apply_writes_only_notice_inserts_and_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from unittest.mock import MagicMock

    from onyx.db import source_notice_import
    from onyx.db.source_notice_import import (
        apply_source_notice_manifest,
        canonical_digest,
    )

    path, digest = _workbook(
        tmp_path, [["Diğer Kanunlar", 2908, "Dernekler Kanunu (Mülga)", None]]
    )
    manifest = build_source_notice_manifest(
        workbook_path=path,
        workbook_sha256=digest,
        sources=[],
        document_set_id=15,
        identity_heads=[],
    )
    monkeypatch.setattr(
        source_notice_import, "read_current_inventory", lambda _session, _scope: []
    )
    session = MagicMock()
    session.execute.return_value.mappings.return_value.first.return_value = None
    inserted = apply_source_notice_manifest(
        session,
        manifest,
        expected_manifest_sha256=canonical_digest(manifest.model_dump(mode="json")),
        expected_workbook_sha256=digest,
    )
    assert inserted == [manifest.targets[0].id]
    statements = [str(call.args[0]).strip() for call in session.execute.call_args_list]
    writes = [
        statement
        for statement in statements
        if statement.startswith(("INSERT", "UPDATE", "DELETE", "ALTER", "DROP"))
    ]
    assert len(writes) == 1
    assert writes[0].startswith("INSERT INTO regulatory_source_notice")
    session.commit.assert_not_called()
    session.execute.reset_mock()
    session.execute.return_value.mappings.return_value.all.return_value = [
        {
            "id": manifest.targets[0].id,
            "identity_key": manifest.targets[0].identity_key,
            "catalog_title": manifest.targets[0].catalog_title,
            "provenance": manifest.targets[0].provenance,
        }
    ]
    assert (
        apply_source_notice_manifest(
            session,
            manifest,
            expected_manifest_sha256=canonical_digest(manifest.model_dump(mode="json")),
            expected_workbook_sha256=digest,
        )
        == []
    )
    assert not any(
        str(call.args[0]).strip().startswith("INSERT")
        for call in session.execute.call_args_list
    )


def test_changed_catalog_or_manifest_stops_before_insert(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from unittest.mock import MagicMock

    from onyx.db import source_notice_import
    from onyx.db.source_notice_import import (
        apply_source_notice_manifest,
        canonical_digest,
    )

    path, digest = _workbook(
        tmp_path, [["Diğer Kanunlar", 2908, "Dernekler Kanunu (Mülga)", None]]
    )
    manifest = build_source_notice_manifest(
        workbook_path=path,
        workbook_sha256=digest,
        sources=[],
        document_set_id=15,
        identity_heads=[],
    )
    session = MagicMock()
    with pytest.raises(ValueError, match="Manifest digest"):
        apply_source_notice_manifest(
            session,
            manifest,
            expected_manifest_sha256="0" * 64,
            expected_workbook_sha256=digest,
        )
    session.execute.assert_not_called()
    monkeypatch.setattr(
        source_notice_import,
        "read_current_inventory",
        lambda _session, _scope: [
            ExistingNoticeSource(id="new-source", name="2908_kanun.md")
        ],
    )
    with pytest.raises(ValueError, match="source catalog changed"):
        apply_source_notice_manifest(
            session,
            manifest,
            expected_manifest_sha256=canonical_digest(manifest.model_dump(mode="json")),
            expected_workbook_sha256=digest,
        )
    session.execute.assert_not_called()


@pytest.mark.parametrize(
    "statement",
    [
        "UPDATE user_file SET name='x'",
        "INSERT INTO file_record (file_id) VALUES ('x')",
        "DROP INDEX existing_index",
        "ALTER TABLE regulatory_source_notice ADD COLUMN x text",
        "DELETE FROM regulatory_source_notice",
        "INSERT INTO regulatory_source_notice (id) VALUES ('x'); UPDATE user_file SET name='x'",
        "SELECT 1; DELETE FROM user_file",
    ],
)
def test_sql_guard_rejects_all_existing_table_writes(statement: str) -> None:
    from onyx.db.source_notice_import import guard_notice_import_statement

    with pytest.raises(ValueError):
        guard_notice_import_statement(statement)


@pytest.mark.parametrize(
    "statement",
    [
        "SELECT id FROM user_file",
        "SHOW transaction_isolation",
        "INSERT INTO regulatory_source_notice (id) VALUES (:id)",
    ],
)
def test_sql_guard_allows_only_the_new_table_insert(statement: str) -> None:
    from onyx.db.source_notice_import import guard_notice_import_statement

    guard_notice_import_statement(statement)


def test_head_identity_is_a_veto_not_existing_identity(tmp_path: Path) -> None:
    from onyx.db.source_notice_import import SourceIdentityHead

    path, digest = _workbook(
        tmp_path, [["Genelgeler", 2012, 7, "Eski taşıma işlemi (Mülga)"]]
    )
    manifest = build_source_notice_manifest(
        workbook_path=path,
        workbook_sha256=digest,
        sources=[ExistingNoticeSource(id="generic", name="document.md")],
        document_set_id=15,
        identity_heads=[
            SourceIdentityHead(
                source_id="generic",
                chunk_id="head",
                position=0,
                projection_ordinal=0,
                heading_path=["Genelge (2012/7)"],
                text_head="Genelge (2012/7)",
            )
        ],
    )
    assert manifest.targets == []
    assert manifest.decisions[0].classification == "ambiguous"
    assert (
        manifest.decisions[0].reason
        == "canonical_identity_head_candidate_requires_review"
    )
    assert manifest.decisions[0].existing_source_ids == ["generic"]
    assert manifest.identity_heads_fingerprint is not None
    assert manifest.identity_heads_count == 1


def test_body_references_and_bullet_sentences_are_not_own_header_identity() -> None:
    from onyx.db.source_notice_import import SourceIdentityHead, identity_head_veto_keys

    head = SourceIdentityHead(
        source_id="generic",
        chunk_id="head",
        position=0,
        projection_ordinal=0,
        heading_path=["- 765 sayılı Türk Ceza Kanunu kapsamında yapılan işlemler"],
        text_head="Açıklama\n765 sayılı Türk Ceza Kanunu kapsamında yapılan işlemler\n\nİlgi: 2012/7 sayılı Genelge",
    )
    assert identity_head_veto_keys(head) == set()


def test_document_header_number_is_a_potential_own_identity() -> None:
    from onyx.db.source_notice_import import SourceIdentityHead, identity_head_veto_keys

    head = SourceIdentityHead(
        source_id="generic",
        chunk_id="head",
        position=0,
        projection_ordinal=0,
        heading_path=[],
        text_head="TÜRK CEZA KANUNU\nKanun Numarası : 765\nKabul Tarihi: 01.01.1926",
    )
    assert identity_head_veto_keys(head) == {"law:765"}


@pytest.mark.parametrize(
    "statement",
    [
        "SELECT * INTO user_file FROM regulatory_source_notice",
        "SELECT setval('existing_seq',100)",
        "SELECT nextval('existing_seq')",
        "SELECT pg_logical_emit_message(true,'x','y')",
        "INSERT INTO regulatory_source_notice (id) VALUES (:id) ON CONFLICT (id) DO UPDATE SET catalog_title='changed'",
        "COPY regulatory_source_notice TO '/tmp/file'",
    ],
)
def test_sql_guard_rejects_select_side_effects_and_upserts(statement: str) -> None:
    from onyx.db.source_notice_import import guard_notice_import_statement

    with pytest.raises(ValueError):
        guard_notice_import_statement(statement)


def test_letter_split_header_and_metadata_are_vetoes_without_body_number_guessing() -> (
    None
):
    from onyx.db.source_notice_import import SourceIdentityHead, identity_head_veto_keys

    header = SourceIdentityHead(
        source_id="generic",
        chunk_id="head",
        position=0,
        projection_ordinal=0,
        heading_path=[],
        text_head="T.C. Bakanlık\nSayı: 34220\nTarih: 03.12.2007\nKonu: Uygulama",
    )
    assert identity_head_veto_keys(header) == {"administrative_letter:2007-12-03:34220"}
    metadata = header.model_copy(
        update={
            "document_type": "unknown",
            "document_number": "2007/34220",
            "document_date": "2007-12-03",
            "text_head": "Açıklama metnindeki 12345 sayısı kaynak kimliği değildir.",
        }
    )
    assert identity_head_veto_keys(metadata) == {
        "administrative_letter:2007-12-03:34220"
    }


def test_postcommit_verification_checks_every_business_field_and_skipped_rows(
    tmp_path: Path,
) -> None:
    from onyx.db.source_notice_import import validate_imported_notice_targets

    path, digest = _workbook(
        tmp_path,
        [
            ["Diğer Kanunlar", 2908, "Dernekler Kanunu (Mülga)", None],
            ["Diğer Yönetmelikler", None, "Belirsiz Yönetmelik (Mülga)", None],
        ],
    )
    manifest = build_source_notice_manifest(
        workbook_path=path,
        workbook_sha256=digest,
        sources=[],
        document_set_id=15,
        identity_heads=[],
    )
    result = validate_imported_notice_targets(manifest, manifest.targets)
    assert result["verified_notice_count"] == 1
    assert result["held_raw_rows_skipped"] == 1
    assert result["skipped_source_row_intersections"] == 0
    changed = manifest.targets[0].model_copy(update={"catalog_title": "changed"})
    with pytest.raises(ValueError, match="fields differ"):
        validate_imported_notice_targets(manifest, [changed])
    with pytest.raises(ValueError, match="identity set differs"):
        validate_imported_notice_targets(manifest, [])
