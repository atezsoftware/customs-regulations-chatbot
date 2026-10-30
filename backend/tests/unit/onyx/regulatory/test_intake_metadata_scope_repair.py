"""Exact SGM-2024/4 date-scope failures preserve strict evidence and old ACKs."""

import json
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from threading import Event
from types import SimpleNamespace
from typing import Any, TypedDict, cast
from unittest.mock import MagicMock

import pytest
from google.genai import types as genai_types

from onyx.regulatory.document_intake import metadata, metadata_job
from onyx.regulatory.indexing_jobs.vertex_batch import (
    VertexBatchGateway,
    VertexBatchJobStatus,
    VertexBatchRequest,
    VertexBatchState,
)
from onyx.regulatory.labeling import vertex_direct
from tests.unit.onyx.regulatory.labeling.test_vertex_direct import _gateway, _Store
from tests.unit.onyx.regulatory.test_intake_metadata_budget import _envelope


class _SourceRow(TypedDict):
    id: str
    text: str
    heading_path: list[str]
    chunk_type: str


# Frozen public source excerpts from the failed eight-target SGM-2024/4 window.
_SGM4_ROWS: list[_SourceRow] = [
    {
        "id": "rc_4e68923acb6867065486b2aacdf7404082d59556",
        "text": "23.10.2026 11:14\n"
        "resmigazete.gov.tr/eskiler/2025/01/20250105-2.htm\n"
        "5 Ocak 2025 PAZAR\n"
        "Resmî Gazete\n"
        "Sayı : 32773\n"
        "TEBLİĞ\n"
        "Sanayi ve Teknoloji Bakanlığından:\n"
        "EV TİPİ ÇAMAŞIR KURUTMA MAKİNELERİNİN\n"
        "ENERJİ ETİKETLEMESİNE DAİR TEBLİĞ\n"
        "(2023/2534/AB) (SGM-2024/4)\n"
        "Amaç, kapsam ve dayanak",
        "heading_path": [
            "EV TİPİ ÇAMAŞIR KURUTMA MAKİNELERİNİN ENERJİ ETİKETLEMESİNE DAİR TEBLİĞ"
        ],
        "chunk_type": "free_text",
    },
    {
        "id": "rc_40fa107ddf04c987e81a2030300a2124c3e6b7a6",
        "text": "MADDE 6-\n"
        "(1) Bakanlık bu Tebliğ kapsamındaki ürünler için Enerji Etiketlemesi Çerçeve "
        "Yönetmeliğinin 9\n"
        "uncu maddesinde belirtilen piyasa gözetimi ve denetimini gerçekleştirirken bu "
        "Tebliğin Ek IX’unda belirtilen\n"
        "doğrulama yöntemlerini uygular.\n"
        "Danışma kurulu işlemleri",
        "heading_path": [
            "EV TİPİ ÇAMAŞIR KURUTMA MAKİNELERİNİN ENERJİ ETİKETLEMESİNE DAİR TEBLİĞ",
            "MADDE 6",
            "(1) Bakanlık bu Tebliğ kapsamındaki ürünler için Enerji Etiketlemesi "
            "Çerçeve Yönetmeliğinin 9",
        ],
        "chunk_type": "paragraph",
    },
    {
        "id": "rc_e97c59b9d0d8463634bf94146753e6121f536f4f",
        "text": "MADDE 7-\n"
        "(1) Bakanlık; bu Tebliğ ile ilgili olarak Avrupa Komisyonu tarafından ev tipi "
        "çamaşır kurutma\n"
        "makinelerinin enerji tüketimi, işlevsel ve çevresel performansına ilişkin "
        "iyileştirme potansiyeli, son kullanıcıları daha\n"
        "enerji ve kaynak verimli cihazlar satın almaya ve daha enerji ve kaynak verimli "
        "programlar kullanmaya teşvik etmek\n"
        "ve döngüsel ekonomi hususlarında çalışmalar yapmak üzere kurulan danışma kurulu "
        "toplantılarına katılım sağlar.\n"
        "AB mevzuatına uyum",
        "heading_path": [
            "EV TİPİ ÇAMAŞIR KURUTMA MAKİNELERİNİN ENERJİ ETİKETLEMESİNE DAİR TEBLİĞ",
            "MADDE 7",
            "(1) Bakanlık",
        ],
        "chunk_type": "paragraph",
    },
    {
        "id": "rc_0634106ccdf8e56b13898d9ba40cb41fd5d44353",
        "text": "MADDE 8-\n"
        "(1) Bu Tebliğ, Avrupa Parlamentosu ve Konseyinin (AB) 2017/1369 sayılı Tüzüğünü ev "
        "tipi\n"
        "çamaşır kurutma makinelerinin enerji etiketlemesine dair tamamlayan ve (AB) "
        "392/2012 sayılı Yetki Devrine Dayanan\n"
        "Komisyon Tüzüğünü yürürlükten kaldıran 13/7/2023 tarihli ve (AB) 2023/2534 sayılı "
        "Yetki Devrine Dayanan\n"
        "Komisyon Tüzüğü esas alınarak AB mevzuatına uyum çerçevesinde hazırlanmıştır.\n"
        "Yürürlükten kaldırılan tebliğ",
        "heading_path": [
            "EV TİPİ ÇAMAŞIR KURUTMA MAKİNELERİNİN ENERJİ ETİKETLEMESİNE DAİR TEBLİĞ",
            "MADDE 8",
            "(1) Bu Tebliğ, Avrupa Parlamentosu ve Konseyinin (AB) 2017/1369 sayılı "
            "Tüzüğünü ev tipi",
        ],
        "chunk_type": "paragraph",
    },
    {
        "id": "rc_1dbca5596b6209e59ec6196432ba650b4d1fc6bf",
        "text": "MADDE 9-\n"
        "(1) 15/5/2013 tarihli ve 28648 sayılı Resmî Gazete’de yayımlanan Ev Tipi Çamaşır "
        "Kurutma\n"
        "Makinelerinin Enerji Etiketlemesine Dair Tebliğ (SGM-2013/6) yürürlükten "
        "kaldırılmıştır.\n"
        "Geçiş hükümleri",
        "heading_path": [
            "EV TİPİ ÇAMAŞIR KURUTMA MAKİNELERİNİN ENERJİ ETİKETLEMESİNE DAİR TEBLİĞ",
            "MADDE 9",
            "(1) 15/5/2013 tarihli ve 28648 sayılı Resmî Gazete’de yayımlanan Ev Tipi "
            "Çamaşır Kurutma",
        ],
        "chunk_type": "paragraph",
    },
    {
        "id": "rc_ee869704f7c71e4b31d792032dff9a1bc9bec71e",
        "text": "GEÇİCİ MADDE 1-\n"
        "(1) 30/6/2025 tarihine kadar, 9 uncu madde ile yürürlükten kaldırılan Ev Tipi "
        "Çamaşır\n"
        "Kurutma Makinelerinin Enerji Etiketlemesine Dair Tebliğ (SGM-2013/6)’in 5 inci "
        "maddesinin birinci fıkrasının (b)\n"
        "bendi uyarınca gerekli olan ürün bilgi formu, ürünle birlikte basılı olarak "
        "sunulmak yerine ürün veri tabanından veya\n"
        "tedarikçinin internet sitesi üzerinden sunulabilir. Ancak, satıcının talep etmesi "
        "halinde, tedarikçi ürün bilgi formunun\n"
        "basılı olarak sunulmasını sağlar.\n"
        "https://www.resmigazete.gov.tr/eskiler/2025/01/20250105-2.htm\n"
        "2/3\n"
        "23.10.2026 11:14\n"
        "resmigazete.gov.tr/eskiler/2025/01/20250105-2.htm\n"
        "Yürürlük",
        "heading_path": [
            "EV TİPİ ÇAMAŞIR KURUTMA MAKİNELERİNİN ENERJİ ETİKETLEMESİNE DAİR TEBLİĞ",
            "GEÇİCİ MADDE 1",
            "(1) 30/6/2025 tarihine kadar, 9 uncu madde ile yürürlükten kaldırılan Ev "
            "Tipi Çamaşır",
        ],
        "chunk_type": "paragraph",
    },
    {
        "id": "rc_a9037babb591412f73a5e6055fc6c6e998acb681",
        "text": "MADDE 10-\n"
        "(1) Bu Tebliğin;\n"
        "a) 1 inci, geçici 1 inci, 10 uncu ve 11 inci maddeleri yayımı tarihinde,",
        "heading_path": [
            "EV TİPİ ÇAMAŞIR KURUTMA MAKİNELERİNİN ENERJİ ETİKETLEMESİNE DAİR TEBLİĞ",
            "MADDE 10",
            "(1) Bu Tebliğin;",
            "a) 1 inci, geçici 1 inci, 10 uncu ve 11 inci maddeleri yayımı tarihinde,",
        ],
        "chunk_type": "clause",
    },
    {
        "id": "rc_9df54e1c649c2133e25e065ce8c4a1bfc7e9beb8",
        "text": "b) 3 üncü maddesinin birinci fıkrasının (a) ve (b) bentleri 1/3/2025 tarihinde,",
        "heading_path": [
            "EV TİPİ ÇAMAŞIR KURUTMA MAKİNELERİNİN ENERJİ ETİKETLEMESİNE DAİR TEBLİĞ",
            "MADDE 10",
            "(1) Bu Tebliğin;",
            "b) 3 üncü maddesinin birinci fıkrasının (a) ve (b) bentleri 1/3/2025 "
            "tarihinde,",
        ],
        "chunk_type": "clause",
    },
    {
        "id": "rc_3ad20a950984e16c37c0a6da05bc8b2697cfc492",
        "text": "c) Diğer hükümleri 1/7/2025 tarihinde,\nyürürlüğe girer.\nYürütme",
        "heading_path": [
            "EV TİPİ ÇAMAŞIR KURUTMA MAKİNELERİNİN ENERJİ ETİKETLEMESİNE DAİR TEBLİĞ",
            "MADDE 10",
            "(1) Bu Tebliğin;",
            "c) Diğer hükümleri 1/7/2025 tarihinde,",
        ],
        "chunk_type": "clause",
    },
]


def _sources() -> list[metadata.MetadataChunk]:
    return [
        metadata.MetadataChunk(
            chunk_id=row["id"],
            text=row["text"],
            heading_path=row["heading_path"],
            chunk_type=row["chunk_type"],
        )
        for row in _SGM4_ROWS
    ]


def _entry(
    sources: Sequence[metadata.MetadataChunk], part: str, scope: str
) -> metadata.MetadataEvidence:
    return next(
        entry
        for entry in metadata.build_metadata_evidence_catalog(sources)
        if entry.scope == scope
        and entry.text.startswith(part)
        and (part != "5 Ocak" or "2025" in entry.text)
    )


def _response(sources: list[metadata.MetadataChunk], *, valid: bool) -> str:
    a = _entry(sources, "a) 1 inci,", "body" if valid else "heading")
    b = _entry(sources, "b) 3 üncü", "heading")
    c = _entry(sources, "c) Diğer", "body" if valid else "heading")
    publication = _entry(sources, "5 Ocak", "body")
    targets = sources[1:]
    facts = []
    for index, target in enumerate(targets):
        evidence, value, basis, reference = c, "2025-07-01", "explicit", ""
        if index >= 4:
            evidence, value, basis, reference = (
                a,
                "2025-01-05",
                "relative_publication",
                publication.evidence_id,
            )
        if not valid and index == 6:
            evidence, value, basis, reference = b, "2025-03-01", "explicit", ""
        elif not valid and index == 7:
            evidence, value, basis, reference = c, "2025-07-01", "explicit", ""
        facts.append(
            {
                "chunk_id": target.chunk_id,
                "fields": [
                    {
                        "field": "validity_start_date",
                        "value": value,
                        "evidence_id": evidence.evidence_id,
                        "basis": basis,
                        "reference_evidence_id": reference,
                    }
                ],
            }
        )
    return json.dumps({"chunks": facts, "warnings": []}, ensure_ascii=False)


def _request(sources: list[metadata.MetadataChunk]) -> VertexBatchRequest:
    return metadata.build_metadata_request(
        sources, [source.chunk_id for source in sources[1:]]
    )


def _scoped(
    request: VertexBatchRequest, invalid: str, *, cached: str | None = None
) -> VertexBatchRequest:
    result = metadata.build_metadata_body_scope_repair_request(
        request, invalid, cached_repair_response=cached
    )
    assert result is not None
    return result


def test_actual_sgm4_three_heading_scopes_have_exact_body_candidates_and_strict_final_result() -> (
    None
):
    sources = _sources()
    request, invalid = _request(sources), _response(sources, valid=False)
    with pytest.raises(ValueError, match="requires body evidence"):
        metadata.validate_metadata_transport_response(
            invalid, chunks=sources, target_ids=[s.chunk_id for s in sources[1:]]
        )
    repair = _scoped(request, invalid, cached=invalid)
    payload, original = json.loads(repair.prompt), json.loads(request.prompt)
    feedback = payload["validation_feedback"]["body_scope_feedback"]
    assert feedback["invalid_scope_total"] == 8
    assert len(feedback["errors"]) == 8
    cited = set()
    for error in feedback["errors"]:
        assert error["required_scope"] == "body"
        assert error["target_chunk_id"] in original["target_chunk_ids"]
        candidates = error["available_evidence"]
        assert 1 <= len(candidates) <= 8
        assert all(item["scope"] == "body" for item in candidates)
        assert len({item["source_chunk_id"] for item in candidates}) == 1
        cited.update(item["text"] for item in candidates)
    assert any(text.startswith("a) 1 inci,") for text in cited)
    assert any(text.startswith("b) 3 üncü") for text in cited)
    assert any(text.startswith("c) Diğer") for text in cited)
    for key in (
        "complete_document",
        "evidence_catalog",
        "target_chunk_ids",
        "input_sha256",
        "version",
    ):
        assert payload[key] == original[key]
    assert repair == _scoped(request, invalid, cached=invalid)
    assert repair.request_hash != request.request_hash
    result = metadata.validate_metadata_transport_response(
        _response(sources, valid=True),
        chunks=sources,
        target_ids=original["target_chunk_ids"],
    )
    assert len(result.chunks) == 8
    assert [str(r.fields[0].value) for r in result.chunks[-3:]] == ["2025-01-05"] * 3
    # An explanation never makes the rejected heading IDs acceptable.
    with pytest.raises(ValueError, match="requires body evidence"):
        metadata.validate_metadata_transport_response(
            invalid, chunks=sources, target_ids=original["target_chunk_ids"]
        )


@pytest.mark.parametrize("case", ["unknown", "identity", "outside_window", "malformed"])
def test_scope_hints_do_not_resolve_ids_or_expand_target_scope(case: str) -> None:
    sources = _sources()
    request = _request(sources)
    raw = json.loads(_response(sources, valid=False))
    for target in raw["chunks"]:
        for fact in target["fields"]:
            if case == "unknown":
                fact["evidence_id"] = "f" * 64
            elif case == "identity":
                fact["field"], fact["value"] = "article_title", "Yürürlük"
    if case == "outside_window":
        raw["chunks"] = [{**raw["chunks"][0], "chunk_id": sources[0].chunk_id}]
    response = "{broken" if case == "malformed" else json.dumps(raw)
    assert metadata.build_metadata_body_scope_repair_request(request, response) is None
    if case == "unknown":
        with pytest.raises(ValueError, match="not in this document catalog"):
            metadata.validate_metadata_transport_response(
                response, chunks=sources, target_ids=[s.chunk_id for s in sources[1:]]
            )


def test_scope_feedback_caps_errors_and_preserves_entire_input() -> None:
    sources = _sources()
    request = _request(sources)
    raw = json.loads(_response(sources, valid=False))
    for target in raw["chunks"]:
        target["fields"].append({**target["fields"][0], "field": "publication_date"})
    response = json.dumps(raw)
    payload = json.loads(_scoped(request, response).prompt)
    feedback = payload["validation_feedback"]["body_scope_feedback"]
    assert feedback["invalid_scope_total"] == 16
    assert len(feedback["errors"]) == 8
    assert all(len(error["available_evidence"]) <= 8 for error in feedback["errors"])
    assert payload["evidence_catalog"] == json.loads(request.prompt)["evidence_catalog"]
    assert len(raw["chunks"]) == 8 and all(len(t["fields"]) == 2 for t in raw["chunks"])


def test_scope_repair_checks_frozen_catalog_and_final_request_byte_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources = _sources()
    request = _request(sources)
    raw = json.loads(request.prompt)
    raw["evidence_catalog"][0]["text"] = "Forged"
    forged = request.model_copy(update={"prompt": json.dumps(raw)})
    with pytest.raises(ValueError, match="catalog differs"):
        _scoped(forged, _response(sources, valid=False))
    generic = metadata.build_metadata_repair_request(
        request,
        _response(sources, valid=False),
        validation_category="invalid_coverage_or_evidence",
    )
    generic_bytes = len(
        json.dumps(generic.to_generate_content_request(), ensure_ascii=False).encode()
    )
    monkeypatch.setattr(metadata, "MAX_METADATA_REQUEST_BYTES", generic_bytes + 1025)
    with pytest.raises(ValueError, match="source was not truncated"):
        _scoped(request, _response(sources, valid=False))


class _CachedGateway:
    def __init__(self, respond: Callable[[VertexBatchRequest], dict[str, Any]]) -> None:
        self.respond = respond
        self.lines: dict[str, str] = {}
        self.new_requests: list[VertexBatchRequest] = []
        self.probes: list[str] = []

    def seed(
        self, request: VertexBatchRequest, text: str, finish: str = "STOP"
    ) -> None:
        key = metadata_job._metadata_submission_key("sgm4-frozen", request)
        self.lines[key] = json.dumps(_envelope(request, finish, text))

    def submit(
        self,
        requests: Sequence[VertexBatchRequest],
        *,
        submission_key: str,
        max_jsonl_bytes: int,
    ) -> VertexBatchState:
        assert len(requests) == 1 and max_jsonl_bytes > 0
        if submission_key not in self.lines:
            self.new_requests.append(requests[0])
            self.lines[submission_key] = json.dumps(self.respond(requests[0]))
        return VertexBatchState(
            remote_job_name=submission_key,
            status=VertexBatchJobStatus.SUCCEEDED,
            output_uri=submission_key,
        )

    def reconcile_submission(self, submission_key: str) -> VertexBatchState | None:
        self.probes.append(submission_key)
        if submission_key not in self.lines:
            return None
        return VertexBatchState(
            remote_job_name=submission_key,
            status=VertexBatchJobStatus.SUCCEEDED,
            output_uri=submission_key,
        )

    def read_results(self, output_uri: str) -> Iterator[str]:
        yield self.lines[output_uri]


def _run(
    gateway: object,
    sources: list[metadata.MetadataChunk],
    heartbeat: Callable[[], None] = lambda: None,
) -> metadata_job.MetadataResults:
    return metadata_job._extract_metadata_targets(
        gateway=cast(VertexBatchGateway, gateway),
        chunks=sources,
        targets=[s.chunk_id for s in sources[1:]],
        identity_prefix="sgm4-frozen",
        heartbeat=heartbeat,
        stopped=Event(),
    )


@pytest.mark.parametrize("cached", ["valid", "invalid", "absent"])
def test_legacy_valid_repair_reuses_exact_ack_invalid_repair_gets_only_one_new_call(
    cached: str,
) -> None:
    sources = _sources()
    base, invalid, valid = (
        _request(sources),
        _response(sources, valid=False),
        _response(sources, valid=True),
    )
    generic = metadata.build_metadata_repair_request(
        base, invalid, validation_category="invalid_coverage_or_evidence"
    )
    gateway = _CachedGateway(lambda request: _envelope(request, "STOP", valid))
    gateway.seed(base, invalid)
    if cached != "absent":
        gateway.seed(generic, valid if cached == "valid" else invalid)
    before = dict(gateway.lines)
    results = _run(gateway, sources)
    assert all(gateway.lines[key] == value for key, value in before.items())
    assert len(gateway.new_requests) == (0 if cached == "valid" else 1)
    if cached == "valid":
        assert results[0][0] == generic.request_hash
    else:
        payload = json.loads(gateway.new_requests[0].prompt)["validation_feedback"]
        assert "body_scope_feedback" in payload
        assert ("cached_repair_response_sha256" in payload) == (cached == "invalid")
        assert results[0][0] == gateway.new_requests[0].request_hash
    assert _run(gateway, sources) == results
    assert len(gateway.new_requests) == (0 if cached == "valid" else 1)


@pytest.mark.parametrize("cached", ["valid", "invalid_scope"])
def test_unknown_id_base_with_cached_heading_date_repair_gets_one_scoped_repair(
    cached: str,
) -> None:
    sources = _sources()
    base = _request(sources)
    invalid_scope = _response(sources, valid=False)
    unknown = json.loads(invalid_scope)
    unknown["chunks"][0]["fields"][0]["evidence_id"] = "f" * 64
    invalid_id = json.dumps(unknown, ensure_ascii=False)
    with pytest.raises(ValueError, match="not in this document catalog"):
        metadata.validate_metadata_transport_response(
            invalid_id, chunks=sources, target_ids=[s.chunk_id for s in sources[1:]]
        )
    generic = metadata.build_metadata_repair_request(
        base, invalid_id, validation_category="invalid_coverage_or_evidence"
    )
    valid = _response(sources, valid=True)
    gateway = _CachedGateway(lambda request: _envelope(request, "STOP", valid))
    gateway.seed(base, invalid_id)
    gateway.seed(generic, valid if cached == "valid" else invalid_scope)
    before = dict(gateway.lines)

    results = _run(gateway, sources)

    assert all(gateway.lines[key] == value for key, value in before.items())
    assert len(gateway.new_requests) == (0 if cached == "valid" else 1)
    if cached == "valid":
        assert results[0][0] == generic.request_hash
    else:
        repaired = gateway.new_requests[0]
        feedback = json.loads(repaired.prompt)["validation_feedback"]
        assert "body_scope_feedback" in feedback
        assert feedback["cached_repair_request_hash"] == generic.request_hash
        assert results[0][0] == repaired.request_hash
    assert results[0][1] == metadata.validate_metadata_transport_response(
        valid, chunks=sources, target_ids=[s.chunk_id for s in sources[1:]]
    )
    assert _run(gateway, sources) == results
    assert len(gateway.new_requests) == (0 if cached == "valid" else 1)


def test_invalid_targeted_repair_is_terminal_and_raw_caches_remain() -> None:
    sources = _sources()
    base, invalid = _request(sources), _response(sources, valid=False)
    generic = metadata.build_metadata_repair_request(
        base, invalid, validation_category="invalid_coverage_or_evidence"
    )
    gateway = _CachedGateway(lambda request: _envelope(request, "STOP", invalid))
    gateway.seed(base, invalid)
    gateway.seed(generic, invalid)
    before = dict(gateway.lines)
    with pytest.raises(ValueError, match="after its single repair"):
        _run(gateway, sources)
    assert len(gateway.new_requests) == 1
    assert all(gateway.lines[key] == value for key, value in before.items())
    with pytest.raises(ValueError, match="after its single repair"):
        _run(gateway, sources)
    assert len(gateway.new_requests) == 1


def test_cached_repair_envelope_error_does_not_authorize_new_repair() -> None:
    sources = _sources()
    base = _request(sources)
    invalid = _response(sources, valid=False)
    generic = metadata.build_metadata_repair_request(
        base, invalid, validation_category="invalid_coverage_or_evidence"
    )
    gateway = _CachedGateway(
        lambda _request: pytest.fail("No new request is permitted")
    )
    gateway.seed(base, invalid)
    gateway.seed(generic, invalid)
    key = metadata_job._metadata_submission_key("sgm4-frozen", generic)
    raw = json.loads(gateway.lines[key])
    raw["response"]["candidates"][0]["finishReason"] = "SAFETY"
    gateway.lines[key] = json.dumps(raw)
    with pytest.raises(ValueError, match="incomplete or invalid"):
        _run(gateway, sources)
    assert gateway.new_requests == []


def test_heartbeat_loss_before_cache_probe_or_submit_makes_no_new_request() -> None:
    sources = _sources()
    base = _request(sources)
    invalid = _response(sources, valid=False)
    gateway = _CachedGateway(
        lambda _request: pytest.fail("No new request is permitted")
    )
    gateway.seed(base, invalid)
    calls = 0

    def heartbeat() -> None:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("Lease lost")

    with pytest.raises(RuntimeError, match="Lease lost"):
        _run(gateway, sources, heartbeat)
    assert not gateway.new_requests and not gateway.probes


def test_valid_legacy_repair_is_reused_before_constructing_larger_feedback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources = _sources()
    base, invalid = _request(sources), _response(sources, valid=False)
    generic = metadata.build_metadata_repair_request(
        base, invalid, validation_category="invalid_coverage_or_evidence"
    )
    gateway = _CachedGateway(lambda _request: pytest.fail("No new call"))
    gateway.seed(base, invalid)
    gateway.seed(generic, _response(sources, valid=True))
    monkeypatch.setattr(
        metadata_job,
        "build_metadata_body_scope_repair_request",
        lambda *_args, **_kwargs: pytest.fail("Valid legacy ACK needs no new feedback"),
    )
    assert _run(gateway, sources)[0][0] == generic.request_hash
    assert not gateway.new_requests


def test_cached_eight_target_base_success_never_probes_or_submits_a_repair() -> None:
    sources = _sources()
    base = _request(sources)
    gateway = _CachedGateway(lambda _request: pytest.fail("No new call"))
    gateway.seed(base, _response(sources, valid=True))
    before = dict(gateway.lines)
    assert _run(gateway, sources)[0][0] == base.request_hash
    assert not gateway.new_requests and not gateway.probes
    assert gateway.lines == before


def test_cached_max_tokens_uses_existing_smaller_base_requests_without_scope_repair() -> (
    None
):
    sources = _sources()
    base, invalid = _request(sources), _response(sources, valid=False)
    generic = metadata.build_metadata_repair_request(
        base, invalid, validation_category="invalid_coverage_or_evidence"
    )

    def respond(request: VertexBatchRequest) -> dict[str, Any]:
        payload = json.loads(request.prompt)
        assert "validation_feedback" not in payload
        return _envelope(
            request,
            "STOP",
            json.dumps(
                {
                    "chunks": [
                        {"chunk_id": target, "fields": []}
                        for target in payload["target_chunk_ids"]
                    ],
                    "warnings": [],
                }
            ),
        )

    gateway = _CachedGateway(respond)
    gateway.seed(base, invalid)
    gateway.seed(generic, "", finish="MAX_TOKENS")
    result = _run(gateway, sources)
    assert len(result) == 2 and len(gateway.new_requests) == 2
    assert [
        len(json.loads(r.prompt)["target_chunk_ids"]) for r in gateway.new_requests
    ] == [4, 4]


def test_real_direct_gateway_keeps_base_and_old_repair_bytes_and_resumes_new_ack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources = _sources()
    base, invalid, valid = (
        _request(sources),
        _response(sources, valid=False),
        _response(sources, valid=True),
    )
    generic = metadata.build_metadata_repair_request(
        base, invalid, validation_category="invalid_coverage_or_evidence"
    )
    store, client = _Store(), MagicMock()

    def generate(**kwargs: Any) -> genai_types.GenerateContentResponse:
        payload = json.loads(kwargs["contents"][0].parts[0].text)
        feedback = payload.get("validation_feedback", {})
        response = valid if "body_scope_feedback" in feedback else invalid
        return genai_types.GenerateContentResponse.model_validate(
            {
                "candidates": [
                    {
                        "finishReason": "STOP",
                        "content": {"role": "model", "parts": [{"text": response}]},
                    }
                ]
            }
        )

    @contextmanager
    def traced(**_kwargs: Any) -> Iterator[Any]:
        yield SimpleNamespace(span_data=SimpleNamespace(usage=None))

    client.models.generate_content.side_effect = generate
    gateway, _factory = _gateway(store, client)
    monkeypatch.setattr(vertex_direct, "traced_llm_call", traced)
    for request in (base, generic):
        gateway.submit(
            [request],
            submission_key=metadata_job._metadata_submission_key(
                "sgm4-frozen", request
            ),
            max_jsonl_bytes=metadata.MAX_METADATA_REQUEST_BYTES,
        )
    assert client.models.generate_content.call_count == 2
    frozen_legacy = dict(store.files)
    result = _run(gateway, sources)
    assert client.models.generate_content.call_count == 3
    assert all(store.files[key] == value for key, value in frozen_legacy.items())
    acknowledged = dict(store.files)
    assert _run(gateway, sources) == result
    assert client.models.generate_content.call_count == 3
    assert store.files == acknowledged
