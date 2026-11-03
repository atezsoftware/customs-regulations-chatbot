"""Legacy structural reconsideration preserves current source and model authority."""

import json
from copy import deepcopy
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock
from uuid import uuid4

import pytest
from sqlalchemy.orm import Session

from onyx.db import regulatory_labeling_semantic_repair as repository
from onyx.db.models import RegulatoryLabelingItem, RegulatoryLabelingRun
from onyx.regulatory.amendments.annexes.context_dependencies import context_hash
from onyx.regulatory.labeling.intake_structural_repair import (
    legacy_structural_repair_candidate,
)
from tests.unit.onyx.regulatory.labeling.abstention_repair_test_helpers import (
    BINDING,
    REJECTED,
    audit,
    build,
    response,
    taxonomy,
    validate_audit,
)

HEADING = "Çeşitli ve Son Hükümler"
FOLIO = "Yönetmelik Adı – 75"


def _snapshot(text: str = HEADING) -> dict[str, Any]:
    """Relevant frozen fields from the two actual DEV abstentions."""
    footer = text == FOLIO
    headings = (
        [
            "EK-1",
            "EK 6",
            "3. Tren makinist brövesi aşağıda verilen örnek biçiminde düzenlenecektir",
        ]
        if footer
        else [
            "BOZULABİLİR GIDA MADDELERİNİN TAŞIMACILIĞINDA KULLANILACAK ÖZEL EKİPMANLAR HAKKINDA YÖNETMELİK",
            "SEKİZİNCİ BÖLÜM",
        ]
    )
    return {
        "position": 244 if footer else 153,
        "chunk_type": "free_text",
        "heading_path": headings,
        "validity_start_date": None,
        "validity_end_date": None,
        "chunk_metadata": {
            "intake_chunking_version": "2" if footer else "3",
            "chunk_variant": "atomic",
            "heading_path": headings,
            "appendix_label": "EK 6" if footer else None,
            "section_title": None if footer else HEADING,
            "intake_legal_metadata": {
                "input_stale": False,
                "fields": {
                    "section_title": {
                        "source": "llm",
                        "basis": "explicit",
                        "stale": False,
                        "evidence_scope": "body",
                        "evidence_quote": HEADING,
                        "source_chunk_id": "target",
                    }
                },
            },
        },
    }


@pytest.mark.parametrize("text", [HEADING, FOLIO])
@pytest.mark.parametrize("version", ["2", "3"])
def test_exact_frozen_structural_source_is_eligible(text: str, version: str) -> None:
    snapshot = _snapshot(text)
    snapshot["chunk_metadata"]["intake_chunking_version"] = version
    before = deepcopy(snapshot)
    assert legacy_structural_repair_candidate(
        chunk_id="target", text=text, source_snapshot=snapshot
    )
    assert snapshot == before


@pytest.mark.parametrize(
    "field",
    [
        "article_no",
        "paragraph_no",
        "clause_label",
        "subclause_label",
        "table_index",
        "table_row_index",
    ],
)
@pytest.mark.parametrize("text", [HEADING, FOLIO])
def test_operative_or_table_locator_never_qualifies(field: str, text: str) -> None:
    snapshot = _snapshot(text)
    snapshot["chunk_metadata"][field] = "1"
    assert not legacy_structural_repair_candidate(
        chunk_id="target", text=text, source_snapshot=snapshot
    )


@pytest.mark.parametrize(
    "change",
    [
        "short-body",
        "extra-body",
        "table",
        "non-atomic",
        "version",
        "position",
        "chapter",
        "heading-drift",
        "title",
        "stale-input",
        "stale-field",
        "context-quote",
        "other-source",
        "partial-quote",
        "inferred",
        "no-audit",
    ],
)
def test_structure_requires_positive_whole_body_proof(change: str) -> None:
    snapshot, text = _snapshot(), HEADING
    metadata = snapshot["chunk_metadata"]
    field = metadata["intake_legal_metadata"]["fields"]["section_title"]
    if change == "short-body":
        text = "Vergi uygulanır."
    elif change == "extra-body":
        text += "\n(1) Bu Yönetmelik uygulanır."
    elif change == "table":
        snapshot["chunk_type"] = "table"
    elif change == "non-atomic":
        metadata["chunk_variant"] = "parent"
    elif change == "version":
        metadata["intake_chunking_version"] = "1"
    elif change == "position":
        snapshot["position"] = -1
    elif change == "chapter":
        snapshot["heading_path"][-1] = "Çeşitli Hükümler"
    elif change == "heading-drift":
        metadata["heading_path"] = ["BAŞKA BÖLÜM"]
    elif change == "title":
        metadata["section_title"] = "Başka başlık"
    elif change == "stale-input":
        metadata["intake_legal_metadata"]["input_stale"] = True
    elif change == "stale-field":
        field["stale"] = True
    elif change == "context-quote":
        field["evidence_scope"] = "heading"
    elif change == "other-source":
        field["source_chunk_id"] = "neighbor"
    elif change == "partial-quote":
        field["evidence_quote"] = "Son Hükümler"
    elif change == "inferred":
        field["basis"] = "inferred"
    else:
        del metadata["intake_legal_metadata"]
    assert not legacy_structural_repair_candidate(
        chunk_id="target", text=text, source_snapshot=snapshot
    )


@pytest.mark.parametrize(
    "text",
    [
        "Yönetmelik Adı – 0",
        "Yönetmelik Adı – 75\nTren makinisti yetkilidir.",
        "MADDE 1 – Vergi uygulanır.",
        "(1) Vergi uygulanır.",
        "a) Vergi uygulanır.",
        "EK 6",
        "| Yönetmelik Adı – 75 |",
    ],
)
def test_similar_or_operative_body_is_not_a_folio(text: str) -> None:
    assert not legacy_structural_repair_candidate(
        chunk_id="target", text=text, source_snapshot=_snapshot(FOLIO)
    )


@pytest.mark.parametrize("text", [HEADING, FOLIO])
@pytest.mark.parametrize("drift", [None, "text", "position", "metadata", "hash"])
def test_current_source_is_rechecked_before_legacy_repair(
    monkeypatch: pytest.MonkeyPatch, text: str, drift: str | None
) -> None:
    snapshot = _snapshot(text)
    file_id, run_id = uuid4(), uuid4()
    item = SimpleNamespace(
        run_id=run_id,
        user_file_id=file_id,
        regulatory_chunk_id="target",
        text_snapshot=text,
        canonical_text_sha256=context_hash(text),
        source_snapshot=deepcopy(snapshot),
    )
    row = SimpleNamespace(
        id="target",
        text=text,
        status="active",
        superseded_by_chunk_id=None,
        chunk_metadata=snapshot["chunk_metadata"],
        position=snapshot["position"],
        heading_path=snapshot["heading_path"],
        chunk_type="free_text",
        validity_start_date=None,
        validity_end_date=None,
    )
    run = SimpleNamespace(id=run_id, file_ids=[str(file_id)], document_set_id=15)
    if drift == "text":
        row.text += " changed"
    elif drift == "position":
        row.position += 1
    elif drift == "metadata":
        row.chunk_metadata["section_title"] = "changed"
    elif drift == "hash":
        item.canonical_text_sha256 = "f" * 64
    session = Mock(spec=Session)
    session.scalar.return_value = row
    owned = Mock()
    monkeypatch.setattr(repository, "require_owned_intake_source", owned)

    def check() -> None:
        repository.require_current_intake_repair_item(
            session,
            item=cast(RegulatoryLabelingItem, item),
            run=cast(RegulatoryLabelingRun, run),
        )

    if drift:
        with pytest.raises(ValueError, match="snapshot changed"):
            check()
        owned.assert_not_called()
    else:
        check()
        owned.assert_called_once_with(session, user_file_id=file_id, document_set_id=15)
        repair = build(text=text)
        item.request_payload = repair.base_request.model_dump(
            mode="json", exclude_computed_fields=True
        )
        item.request_hash = repair.base_request.request_hash
        item.context_snapshot = json.loads(repair.base_request.prompt)[
            "interpretation_context"
        ]
        item.context_sha256 = context_hash("frozen context fingerprint")
        item.labels, item.assignments = [], []
        run.taxonomy_id, run.provider_binding, run.model = (
            uuid4(),
            BINDING,
            BINDING["model"],
        )
        session.get_one.return_value = SimpleNamespace(
            definition=taxonomy().model_dump(mode="json"),
            version_hash=taxonomy().version_hash,
        )
        accepted = audit(repair, response(empty=True), text=text)
        repository.validate_item_semantic_repair_audit(
            session,
            item=cast(RegulatoryLabelingItem, item),
            run=cast(RegulatoryLabelingRun, run),
            audit=accepted,
            labels=[],
            assignments=[],
        )
        item.source_snapshot[repository.SEMANTIC_REPAIR_KEY] = accepted.model_dump(
            mode="json"
        )
        repository.validate_item_semantic_repair_audit(
            session,
            item=cast(RegulatoryLabelingItem, item),
            run=cast(RegulatoryLabelingRun, run),
        )


@pytest.mark.parametrize("text", [HEADING, FOLIO])
def test_eligible_structure_still_requires_non_abstained_model_decision(
    text: str,
) -> None:
    repair = build(text=text)
    continued = audit(repair, REJECTED, text=text)
    with pytest.raises(ValueError, match="accepted assignments"):
        validate_audit(
            continued,
            repair,
            text=text,
            source_hash=context_hash(text),
            labels=[],
            assignments=[],
        )
    accepted = audit(repair, response(empty=True), text=text)
    assert (
        validate_audit(
            accepted,
            repair,
            text=text,
            source_hash=context_hash(text),
            labels=[],
            assignments=[],
        ).outcome.abstained
        is False
    )
