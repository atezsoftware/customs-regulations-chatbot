"""Actual missing header identifiers and unpublished duplicate source isolation."""

import copy
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from sqlalchemy.orm import Session

from onyx.db.enums import UserFileStatus
from onyx.db.models import (
    AmendmentSourcePackage,
    DocumentSet,
    LLMProvider,
    ModelConfiguration,
    RegulatoryChunk,
    RegulatoryDocumentIntake,
    RegulatoryDocumentIntakeFile,
    RegulatoryFilePublication,
    RegulatorySourceAsset,
    UserFile,
)
from onyx.db.regulatory_amendment_targets import load_amendment_source_identities
from onyx.regulatory.amendments.models import AmendmentInstruction
from onyx.regulatory.source_identity import source_file_identity_matches
from tests.external_dependency_unit.regulatory.test_amendment_sources import (
    source_session as source_session,
)
from tests.external_dependency_unit.regulatory.test_amendment_structural_source_identity import (
    _file,
    _retriever,
)
from tests.external_dependency_unit.regulatory.test_annex_baseline import _chunk


def _actual() -> dict[str, Any]:
    return json.loads(
        Path(__file__)
        .with_name("fixtures")
        .joinpath("main12_source_identity.json")
        .read_text()
    )


def _source(session: Session) -> tuple[DocumentSet, UserFile, RegulatoryChunk]:
    actual = _actual()
    group = DocumentSet(name=str(uuid4()), description="", is_up_to_date=True)
    session.add(group)
    session.flush()
    file = _file(session, group)
    file.name = actual["filename"]
    row = _chunk(session, file, 0, actual["text"])
    row.heading_path = actual["heading_path"]
    row.chunk_metadata = actual["metadata"]
    row.chunk_type = "free_text"
    session.flush()
    return group, file, row


def _intake(session: Session, group: DocumentSet, file: UserFile, state: str) -> None:
    provider = LLMProvider(name=f"identity-{uuid4()}", provider="vertex_ai")
    model = ModelConfiguration(llm_provider=provider, name="fixture", is_visible=True)
    package = AmendmentSourcePackage(
        document_set_id=group.id,
        created_by=file.user_id,
        environment="local-test",
        idempotency_key=str(uuid4()),
        request_hash="a" * 64,
        input_spec={},
        status="ready",
    )
    session.add_all([model, package])
    session.flush()
    asset = RegulatorySourceAsset(
        package_id=package.id,
        sha256="a" * 64,
        file_id=str(uuid4()),
        mime_type="application/pdf",
        display_name="source.pdf",
        byte_count=1,
    )
    session.add(asset)
    session.flush()
    batch = RegulatoryDocumentIntake(
        source_package_id=package.id,
        document_set_id=group.id,
        created_by=file.user_id,
        idempotency_key=uuid4(),
        labeling_model_configuration_id=model.id,
        status="approving" if state == "approving_intake" else "completed",
        stage="completed",
    )
    session.add(batch)
    session.flush()
    session.add(
        RegulatoryDocumentIntakeFile(
            intake_id=batch.id,
            source_asset_id=asset.id,
            user_file_id=file.id,
            role="main",
            status={
                "pending_intake": "ready",
                "source_evidence": "source_evidence",
            }.get(state, "completed"),
        )
    )
    session.flush()


def test_actual_345_header_restores_source_identity_without_promoting_stale_audit(
    source_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = source_session
    actual = _actual()
    group, file, header = _source(session)
    original = copy.deepcopy(header.chunk_metadata)
    assert not source_file_identity_matches(
        actual["target"], file.name, header.heading_path[0]
    )
    paragraph = _chunk(
        session, file, 1, "MADDE 9- (1) Tüzüğü esas alınarak hazırlanmıştır."
    )
    paragraph.heading_path = [header.heading_path[0], "MADDE 9"]
    paragraph.chunk_metadata = {"article_no": "9", "paragraph_no": "1"}
    paragraph.chunk_type = "paragraph"
    session.flush()
    identity = load_amendment_source_identities(session, [file.id])[0]
    assert source_file_identity_matches(
        actual["target"], identity.name, identity.root_heading
    )
    assert identity.root_heading.endswith("(2023/826/AB) (SGM-2024/5)")
    retriever, fallback, search = _retriever(session, monkeypatch, group, [file])
    candidates = retriever.search(
        AmendmentInstruction(
            instruction_text="Aynı Tebliğin 9 uncu maddesinin birinci fıkrasında yer alan ibare değiştirilmiştir.",
            target_source=actual["target"],
            article_reference="Madde 9 fıkra 1",
        )
    )
    assert candidates and candidates[0].chunk_id == paragraph.id
    assert candidates[0].source_verified
    fallback.assert_not_called()
    search.assert_not_called()
    assert (
        header.chunk_metadata == original
        and original["intake_legal_metadata"]["input_stale"] is True
    )


@pytest.mark.parametrize(
    "case",
    [
        "body_citation",
        "amendment",
        "amendment_next_line",
        "wrong_EU",
        "wrong_SGM",
        "wrong_metadata",
        "missing_metadata",
        "filename_conflict",
        "other_law",
        "not_first",
        "aggregate",
        "bound_companion",
        "superseded",
        "different_title",
        "no_masthead",
    ],
)
def test_header_reconstruction_cannot_borrow_or_override_identity(
    source_session: Session, case: str
) -> None:
    session = source_session
    _, file, row = _source(session)
    root = row.heading_path[0]
    if case == "body_citation":
        row.text = row.text.replace(
            "(2023/826/AB) (SGM-2024/5)",
            "MADDE 1- (1) Atıf: (2023/826/AB) (SGM-2024/5)",
        )
    elif case == "amendment":
        row.text = row.text.replace(
            "(SGM-2024/5)",
            "(SGM-2024/5)’DE DEĞİŞİKLİK YAPILMASINA DAİR TEBLİĞ (SGM-2026/8)",
        )
    elif case == "amendment_next_line":
        row.text = row.text.replace(
            "Amaç, kapsam ve dayanak", "DEĞİŞİKLİK YAPILMASINA DAİR TEBLİĞ"
        )
    elif case == "wrong_EU":
        row.text = row.text.replace("2023/826/AB", "2023/827/AB")
    elif case == "wrong_SGM":
        row.text = row.text.replace("SGM-2024/5", "SGM-2024/6")
    elif case in {"wrong_metadata", "missing_metadata"}:
        row.chunk_metadata = {
            **row.chunk_metadata,
            "document_number": "SGM-2024/6" if case == "wrong_metadata" else None,
        }
    elif case == "filename_conflict":
        file.name = "Başka Tebliğ (SGM-2024/6).pdf"
    elif case == "other_law":
        row.heading_path = ["4458 sayılı Gümrük Kanunu"]
    elif case == "not_first":
        row.position = 2
    elif case == "aggregate":
        row.chunk_type = "hierarchical_aggregate"
    elif case == "bound_companion":
        row.chunk_metadata = {
            **row.chunk_metadata,
            "bound_to_regulatory_chunk_id": str(uuid4()),
        }
    elif case == "superseded":
        successor = _chunk(session, file, 1, "Yeni kaynak")
        row.superseded_by_chunk_id = successor.id
    elif case == "different_title":
        row.text = row.text.replace("EV VE BÜRO TİPİ", "BAŞKA BİR BELGE")
    elif case == "no_masthead":
        row.text = row.text.replace("Resmî Gazete", "Atıf")
    session.flush()
    identity = load_amendment_source_identities(session, [file.id])[0]
    assert not source_file_identity_matches(
        _actual()["target"], identity.name, identity.root_heading
    )
    if case not in {"wrong_EU", "filename_conflict", "other_law"}:
        assert identity.root_heading == (
            row.heading_path[0] if case == "other_law" else root
        )


@pytest.mark.parametrize(
    "state",
    [
        "CHUNKED",
        "PROCESSING",
        "INDEXING",
        "FAILED",
        "DELETING",
        "epoch_zero",
        "gate_closed",
        "manifest",
        "manifest_sha",
        "no_publication",
        "completed_intake_no_pub",
        "pending_intake",
        "approving_intake",
        "source_evidence",
        "published_intake",
        "published",
    ],
)
def test_only_published_sources_participate_in_version_ambiguity(
    source_session: Session, monkeypatch: pytest.MonkeyPatch, state: str
) -> None:
    session = source_session
    group, first, header = _source(session)
    second = _file(session, group)
    second.name = first.name
    other = _chunk(session, second, 0, header.text)
    other.heading_path = list(header.heading_path)
    other.chunk_metadata = copy.deepcopy(header.chunk_metadata)
    other.chunk_type = "free_text"
    # Equal source titles cannot deduplicate materially different existing provisions.
    for file, text in [
        (first, "MADDE 2- (1) a) ... l) Mevcut hükümler."),
        (second, "MADDE 2- (1) a) ... ç) Hazırlanan hükümler."),
    ]:
        article = _chunk(session, file, 1, text)
        article.heading_path = [header.heading_path[0], "MADDE 2"]
        article.chunk_metadata = {"article_no": "2", "paragraph_no": "1"}
        article.chunk_type = "paragraph"
    publication = session.get_one(RegulatoryFilePublication, second.id)
    if state in UserFileStatus.__members__:
        second.status = UserFileStatus[state]
    elif state == "epoch_zero":
        publication.epoch = 0
    elif state == "gate_closed":
        publication.gate_closed = True
    elif state == "manifest":
        publication.writer_manifest = {"pending": True}
    elif state == "manifest_sha":
        publication.writer_manifest_sha256 = "a" * 64
    elif state == "no_publication":
        session.delete(publication)
    elif state == "completed_intake_no_pub":
        _intake(session, group, second, state)
        session.delete(publication)
    elif state in {
        "pending_intake",
        "approving_intake",
        "source_evidence",
        "published_intake",
    }:
        _intake(session, group, second, state)
    session.flush()
    eligible = state in {"published", "no_publication", "published_intake"}
    identities = load_amendment_source_identities(session, [first.id, second.id])
    assert {row.user_file_id for row in identities} == (
        {first.id, second.id} if eligible else {first.id}
    )
    retriever, _, search = _retriever(session, monkeypatch, group, [first, second])
    result = retriever.search(
        AmendmentInstruction(
            instruction_text="Aynı Tebliğin 2 nci maddesinin birinci fıkrasına aşağıdaki bent eklenmiştir: “m) Yeni kural.”",
            target_source=_actual()["target"],
            article_reference="Madde 2 fıkra 1",
        )
    )
    if eligible:
        assert result == [] and "Multiple target source files" in (
            retriever.last_attention or ""
        )
    else:
        assert result and all(row.user_file_id == str(first.id) for row in result)
    search.assert_not_called()


def test_published_source_must_remain_in_caller_current_document_set(
    source_session: Session,
) -> None:
    group, file, _ = _source(source_session)
    assert load_amendment_source_identities(
        source_session, [file.id], document_set_id=group.id
    )
    assert not load_amendment_source_identities(
        source_session, [file.id], document_set_id=group.id + 100000
    )
    assert not load_amendment_source_identities(
        source_session, [], document_set_id=group.id
    )
