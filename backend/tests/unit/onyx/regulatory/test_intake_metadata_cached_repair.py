"""Cached BOZULABİLİR metadata repairs retain evidence and bounded recovery."""

import hashlib
import json
from collections.abc import Sequence
from threading import Event
from typing import cast

import pytest

from onyx.regulatory.document_intake import metadata, metadata_job
from onyx.regulatory.indexing_jobs.vertex_batch import (
    VertexBatchGateway,
    VertexBatchRequest,
)
from tests.unit.onyx.regulatory.test_intake_metadata_budget import _envelope
from tests.unit.onyx.regulatory.test_intake_metadata_scope_repair import _CachedGateway

_TITLE = (
    "BOZULABİLİR GIDA MADDELERİNİN TAŞIMACILIĞINDA KULLANILACAK "
    "ÖZEL EKİPMANLAR HAKKINDA YÖNETMELİK"
)


def _sources(case: str) -> list[metadata.MetadataChunk]:
    # Public source excerpts from the two failed production repair windows.
    if case == "generic":
        return [
            metadata.MetadataChunk(
                chunk_id="rc_0f0a7d4d7f4efa0467aae83371dccf5eb4de3b4a",
                text=(
                    "ü) Üretimin uygunluğunun kontrolü: İlgili ulusal ve uluslararası "
                    "düzenlemelerde öngörülmüş kapsam ve\n"
                    "belirlenmiş kriterlere göre üretim yerinde gerçekleştirilen kontrolü,\n"
                    "ifade eder."
                ),
                heading_path=[
                    _TITLE,
                    "BİRİNCİ BÖLÜM",
                    "MADDE 4 - Tanımlar",
                    "(1) Bu Yönetmelikte geçen;",
                ],
                chunk_type="clause",
            )
        ]
    return [
        metadata.MetadataChunk(
            chunk_id="rc_da370cdd2c36c0068bdd6adb09937d59075f0c2f",
            text=(
                "(3) Kamu kurum ve kuruluşları hariç olmak üzere bu Yönetmelik kapsamında "
                "bozulabilir gıda taşımacılığı\n"
                "faaliyetinde bulunacak gerçek ve tüzel kişiler, Karayolu Taşıma "
                "Yönetmeliğine göre C1, C2, K1, K2, L1, L2, M1, M2,\n"
                "M3, N1, N2, P1 ve P2 yetki belgelerinden herhangi birine sahip olmak zorundadırlar."
            ),
            heading_path=[
                _TITLE,
                "İKİNCİ BÖLÜM",
                "MADDE 5 - Taşımacılık faaliyetlerinde genel kurallar",
            ],
            chunk_type="paragraph",
        ),
        metadata.MetadataChunk(
            chunk_id="rc_86c1e3de119373ac1b7eb5ee8d9b0b858550c62d",
            text="MADDE 28 –\n(1) Bu Yönetmelik 1/1/2022 tarihinde yürürlüğe girer.",
            heading_path=[
                _TITLE,
                "SEKİZİNCİ BÖLÜM",
                "MADDE 28 - Yürürlük",
                "(1) Bu Yönetmelik 1/1/2022 tarihinde yürürlüğe girer.",
            ],
            chunk_type="paragraph",
        ),
    ]


def _entry(sources: Sequence[metadata.MetadataChunk], text: str, scope: str) -> str:
    return next(
        entry.evidence_id
        for entry in metadata.build_metadata_evidence_catalog(sources)
        if entry.text == text and entry.scope == scope
    )


def _fact(field: str, value: str, evidence_id: str) -> dict[str, str]:
    return {
        "field": field,
        "value": value,
        "evidence_id": evidence_id,
        "basis": "explicit",
        "reference_evidence_id": "",
    }


def _response(target: str, fields: list[dict[str, str]]) -> str:
    return json.dumps(
        {"chunks": [{"chunk_id": target, "fields": fields}], "warnings": []},
        ensure_ascii=False,
    )


def _case(
    case: str,
) -> tuple[
    list[metadata.MetadataChunk], VertexBatchRequest, str, VertexBatchRequest, str, str
]:
    sources = _sources(case)
    target = sources[0].chunk_id
    base = metadata.build_metadata_request(sources, [target])
    if case == "generic":
        title_id = _entry(sources, _TITLE, "heading")
        valid_article = _entry(sources, "MADDE 4 - Tanımlar", "heading")
        invalid_base = _response(
            target,
            [
                _fact(
                    "document_title",
                    _TITLE.replace("KULLANILACAK", "KULLANILACAL"),
                    title_id,
                )
            ],
        )
        repair = metadata.build_metadata_repair_request(
            base, invalid_base, validation_category="value_not_covered_by_evidence"
        )
        invalid_repair = _response(
            target,
            [
                _fact(
                    "article_no",
                    "4",
                    "3b7b870485b69035e51f44c85d72288c54b0891cd0e2b89e2f9a3d0a099588d",
                )
            ],
        )
        valid = _response(
            target,
            [
                _fact("document_title", _TITLE, title_id),
                _fact("article_no", "4", valid_article),
            ],
        )
    else:
        date_text = "(1) Bu Yönetmelik 1/1/2022 tarihinde yürürlüğe girer."
        date_heading = _entry(sources, date_text, "heading")
        date_body = _entry(sources, date_text, "body")
        invalid_base = _response(
            target, [_fact("validity_start_date", "2022-01-01", date_heading)]
        )
        scoped = metadata.build_metadata_body_scope_repair_request(base, invalid_base)
        assert scoped is not None
        repair = scoped
        invalid_article = (
            "52ed5007cc4179e59cd347caf7b7caf7c62c97caaf747be20d4b28165e64fa4"
        )
        invalid_repair = _response(
            target,
            [
                _fact("article_no", "5", invalid_article),
                _fact(
                    "article_title",
                    "Taşımacılık faaliyetlerinde genel kurallar",
                    invalid_article,
                ),
                _fact("validity_start_date", "2022-01-01", date_body),
            ],
        )
        article = _entry(
            sources, "MADDE 5 - Taşımacılık faaliyetlerinde genel kurallar", "heading"
        )
        valid = _response(
            target,
            [
                _fact("article_no", "5", article),
                _fact(
                    "article_title",
                    "Taşımacılık faaliyetlerinde genel kurallar",
                    article,
                ),
                _fact("validity_start_date", "2022-01-01", date_body),
            ],
        )
    return sources, base, invalid_base, repair, invalid_repair, valid


def _run(
    gateway: _CachedGateway, sources: list[metadata.MetadataChunk]
) -> metadata_job.MetadataResults:
    return metadata_job._extract_metadata_targets(
        gateway=cast(VertexBatchGateway, gateway),
        chunks=sources,
        targets=[sources[0].chunk_id],
        identity_prefix="sgm4-frozen",
        heartbeat=lambda: None,
        stopped=Event(),
    )


@pytest.mark.parametrize("case", ["generic", "scoped"])
@pytest.mark.parametrize("cached_valid", [False, True])
def test_cached_repairs_reuse_valid_ack_or_recover_unknown_id_once(
    case: str, cached_valid: bool
) -> None:
    sources, base, invalid_base, repair, invalid_repair, valid = _case(case)
    gateway = _CachedGateway(lambda request: _envelope(request, "STOP", valid))
    gateway.seed(base, invalid_base)
    gateway.seed(repair, valid if cached_valid else invalid_repair)
    before = dict(gateway.lines)
    with pytest.raises(ValueError, match="not in this document catalog"):
        metadata.validate_metadata_transport_response(
            invalid_repair, chunks=sources, target_ids=[sources[0].chunk_id]
        )

    result = _run(gateway, sources)

    assert all(gateway.lines[key] == value for key, value in before.items())
    assert len(gateway.new_requests) == (0 if cached_valid else 1)
    assert result[0][1] == metadata.validate_metadata_transport_response(
        valid, chunks=sources, target_ids=[sources[0].chunk_id]
    )
    if cached_valid:
        assert result[0][0] == repair.request_hash
    else:
        recovery = gateway.new_requests[0]
        payload, original = json.loads(recovery.prompt), json.loads(repair.prompt)
        feedback = payload["validation_feedback"]
        assert "catalog_feedback" in feedback
        assert repair.request_hash in json.dumps(feedback)
        assert hashlib.sha256(invalid_repair.encode()).hexdigest() in json.dumps(
            feedback
        )
        for key in (
            "complete_document",
            "evidence_catalog",
            "target_chunk_ids",
            "input_sha256",
            "version",
        ):
            assert payload[key] == original[key]
        if case == "scoped":
            assert (
                feedback["body_scope_feedback"]
                == original["validation_feedback"]["body_scope_feedback"]
            )
        assert result[0][0] == recovery.request_hash
    assert _run(gateway, sources) == result
    assert len(gateway.new_requests) == (0 if cached_valid else 1)


@pytest.mark.parametrize("case", ["generic", "scoped"])
@pytest.mark.parametrize("existing_repair", [False, True])
def test_invalid_feedback_is_terminal_and_retains_every_cached_response(
    case: str, existing_repair: bool
) -> None:
    sources, base, invalid_base, repair, invalid_repair, _valid = _case(case)
    gateway = _CachedGateway(lambda request: _envelope(request, "STOP", invalid_repair))
    gateway.seed(base, invalid_base)
    if existing_repair:
        gateway.seed(repair, invalid_repair)
    before = dict(gateway.lines)

    with pytest.raises(ValueError, match="after bounded repair"):
        _run(gateway, sources)

    expected_calls = 1 if existing_repair else 2
    assert len(gateway.new_requests) == expected_calls
    assert all(gateway.lines[key] == value for key, value in before.items())
    with pytest.raises(ValueError):
        metadata.validate_metadata_transport_response(
            invalid_repair, chunks=sources, target_ids=[sources[0].chunk_id]
        )
    with pytest.raises(ValueError, match="after bounded repair"):
        _run(gateway, sources)
    assert len(gateway.new_requests) == expected_calls


@pytest.mark.parametrize("case", ["generic", "scoped"])
def test_cached_repair_transport_error_never_authorizes_catalog_recovery(
    case: str,
) -> None:
    sources, base, invalid_base, repair, invalid_repair, _valid = _case(case)
    gateway = _CachedGateway(
        lambda _request: pytest.fail("No new request is permitted")
    )
    gateway.seed(base, invalid_base)
    gateway.seed(repair, invalid_repair, finish="SAFETY")
    before = dict(gateway.lines)

    with pytest.raises(ValueError, match="incomplete or invalid"):
        _run(gateway, sources)

    assert not gateway.new_requests
    assert gateway.lines == before


@pytest.mark.parametrize("case", ["generic", "scoped"])
def test_catalog_recovery_requires_unrecovered_repair_and_unknown_ids(
    case: str,
) -> None:
    sources, base, _invalid_base, repair, invalid_repair, valid = _case(case)
    assert (
        metadata.build_metadata_catalog_recovery_request(base, invalid_repair) is None
    )
    assert metadata.build_metadata_catalog_recovery_request(repair, valid) is None
    recovered = metadata.build_metadata_catalog_recovery_request(repair, invalid_repair)
    assert recovered is not None
    assert (
        metadata.build_metadata_catalog_recovery_request(recovered, invalid_repair)
        is None
    )
    invalid_scope = _response(
        sources[0].chunk_id,
        [
            _fact(
                "validity_start_date", "2022-01-01", _entry(sources, _TITLE, "heading")
            )
        ],
    )
    with pytest.raises(ValueError, match="requires body evidence"):
        metadata.validate_metadata_transport_response(
            invalid_scope, chunks=sources, target_ids=[sources[0].chunk_id]
        )
    assert (
        metadata.build_metadata_catalog_recovery_request(recovered, invalid_scope)
        is None
    )
