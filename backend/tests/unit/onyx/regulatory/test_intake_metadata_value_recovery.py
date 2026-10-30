"""Known evidence/value contradictions receive bounded exact source feedback."""

import json

import pytest

from onyx.regulatory.document_intake import metadata

_TITLE = (
    "BOZULABİLİR GIDA MADDELERİNİN TAŞIMACILIĞINDA KULLANILACAK "
    "ÖZEL EKİPMANLAR HAKKINDA YÖNETMELİK"
)


def _sources() -> list[metadata.MetadataChunk]:
    return [
        metadata.MetadataChunk(
            chunk_id="section",
            text="Test, Muayene ve Belgelendirme Süreçleri",
            heading_path=[_TITLE, "BEŞİNCİ BÖLÜM"],
            chunk_type="free_text",
        ),
        metadata.MetadataChunk(
            chunk_id="rc_72d610712085cdd5c6dfff3412673f631505f1e7",
            text=(
                "23.10.2026 11:17 2 Temmuz 2021 CUMA\n"
                "2 Temmuz 2021 CUMA Resmî Gazete Sayı : 31529\n"
                "YÖNETMELİK\nUlaştırma ve Altyapı Bakanlığından:\n"
                "BOZULABİLİR GIDA MADDELERİNİN TAŞIMACILIĞINDA KULLANILACAK\n"
                "ÖZEL EKİPMANLAR HAKKINDA YÖNETMELİK\n"
                "Amaç, Kapsam, Dayanak ve Tanımlar"
            ),
            heading_path=[_TITLE],
            chunk_type="free_text",
        ),
    ]


def _entry(
    sources: list[metadata.MetadataChunk], text: str, scope: str
) -> metadata.MetadataEvidence:
    return next(
        entry
        for entry in metadata.build_metadata_evidence_catalog(sources)
        if entry.text == text and entry.scope == scope
    )


def _fact(field: str, value: str | int, evidence: str) -> dict[str, str | int]:
    return {
        "field": field,
        "value": value,
        "evidence_id": evidence,
        "basis": "explicit",
        "reference_evidence_id": "",
    }


def _response(fields: list[dict[str, str | int]], target: str = "section") -> str:
    return json.dumps(
        {"chunks": [{"chunk_id": target, "fields": fields}], "warnings": []},
        ensure_ascii=False,
    )


def test_live_section_title_and_typo_recovery_explain_actual_selected_evidence() -> (
    None
):
    sources = _sources()
    body = _entry(sources, sources[0].text, "body")
    heading = _entry(sources, "BEŞİNCİ BÖLÜM", "heading")
    title = _entry(sources, _TITLE, "heading")
    rejected = _response(
        [
            _fact("section_title", "BEŞİNCİ BÖLÜM", body.evidence_id),
            _fact(
                "document_title",
                _TITLE.replace("KULLANILACAK", "KULLANILACAL"),
                title.evidence_id,
            ),
        ]
    )
    base = metadata.build_metadata_request(sources, ["section"])
    repair = metadata.build_metadata_repair_request(
        base, rejected, validation_category="value_not_covered_by_evidence"
    )
    frozen_base, frozen_repair = base.model_dump_json(), repair.model_dump_json()
    recovery = metadata.build_metadata_catalog_recovery_request(repair, rejected)
    assert recovery is not None
    payload = json.loads(recovery.prompt)
    feedback = payload["validation_feedback"]["value_evidence_feedback"]
    assert feedback["invalid_value_total"] == 2
    assert feedback["errors"][0]["selected_evidence"] == body.model_dump(mode="json")
    assert (
        heading.model_dump(mode="json") in feedback["errors"][0]["available_evidence"]
    )
    assert feedback["errors"][1]["rejected_value"].find("KULLANILACAL") >= 0
    for key in (
        "complete_document",
        "evidence_catalog",
        "target_chunk_ids",
        "input_sha256",
        "version",
    ):
        assert payload[key] == json.loads(base.prompt)[key]
    assert (
        base.model_dump_json() == frozen_base
        and repair.model_dump_json() == frozen_repair
    )
    assert recovery == metadata.build_metadata_catalog_recovery_request(
        repair, rejected
    )
    assert metadata.build_metadata_catalog_recovery_request(recovery, rejected) is None
    with pytest.raises(ValueError, match="contradicts its quoted"):
        metadata.validate_metadata_transport_response(
            rejected, chunks=sources, target_ids=["section"]
        )


def test_recovery_feedback_is_bounded_and_never_changes_selected_catalog_evidence() -> (
    None
):
    sources = _sources()
    targets = [f"target-{index}" for index in range(8)]
    sources = [sources[0].model_copy(update={"chunk_id": target}) for target in targets]
    base = metadata.build_metadata_request(sources, targets)
    catalog = metadata.build_metadata_evidence_catalog(sources)
    rejected = json.dumps(
        {
            "chunks": [
                {
                    "chunk_id": target,
                    "fields": [
                        _fact(
                            "section_title",
                            "BEŞİNCİ BÖLÜM",
                            next(
                                entry.evidence_id
                                for entry in catalog
                                if entry.source_chunk_id == target
                                and entry.scope == "body"
                            ),
                        ),
                        _fact(
                            "document_title",
                            "Typo",
                            next(
                                entry.evidence_id
                                for entry in catalog
                                if entry.source_chunk_id == target
                                and entry.scope == "body"
                            ),
                        ),
                    ],
                }
                for target in targets
            ],
            "warnings": [],
        }
    )
    repair = metadata.build_metadata_repair_request(
        base, rejected, validation_category="value_not_covered_by_evidence"
    )
    recovery = metadata.build_metadata_catalog_recovery_request(repair, rejected)
    assert recovery is not None
    feedback = json.loads(recovery.prompt)["validation_feedback"][
        "value_evidence_feedback"
    ]
    assert feedback["invalid_value_total"] == 16
    assert len(feedback["errors"]) == 8
    entries = {entry.evidence_id: entry.model_dump(mode="json") for entry in catalog}
    for error in feedback["errors"]:
        assert (
            error["selected_evidence"]
            == entries[error["selected_evidence"]["evidence_id"]]
        )
        assert len(error["available_evidence"]) <= 8
