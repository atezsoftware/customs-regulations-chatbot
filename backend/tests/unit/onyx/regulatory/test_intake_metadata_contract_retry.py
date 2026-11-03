"""A contract repair declares new requests and cannot borrow inherited clause hints."""

import json
from threading import Event
from types import SimpleNamespace
from typing import cast
from uuid import uuid4

import pytest

from onyx.db.document_intake_metadata_contract_retry import _previous_lineage
from onyx.db.labeling_configuration import LabelingProviderBinding
from onyx.db.models import RegulatoryChunk
from onyx.regulatory.document_intake import metadata, metadata_job
from onyx.regulatory.document_intake.metadata_checkpoint import (
    MetadataCheckpoint,
    metadata_digest,
)
from onyx.regulatory.document_intake.metadata_contract_retry import (
    ContractInvalidClause,
    ContractRetryDocument,
    MetadataContractRetry,
    build_contract_retry_request,
    require_contract_accepted_request,
    require_contract_base_request,
    validate_contract_retry_response,
)
from onyx.regulatory.indexing_jobs.models import VertexAuthenticationMode
from onyx.regulatory.indexing_jobs.vertex_batch import VertexBatchGateway
from tests.unit.onyx.regulatory.test_intake_metadata_budget import _envelope, _Gateway


def _sources(
    body: str = "b) Bakanlığın gerekçeli talebi yerine getirilir.",
) -> list[metadata.MetadataChunk]:
    return [
        metadata.MetadataChunk(
            chunk_id="own",
            text=body,
            heading_path=["MADDE 13", "b) Eski başlık"],
            chunk_type="clause",
            structural_metadata={"clause_label": "b", "article_no": "13"},
        ),
        metadata.MetadataChunk(
            chunk_id="other",
            text="a) Belgeler korunur.",
            heading_path=["MADDE 13", "a) Belgeler korunur."],
            chunk_type="clause",
        ),
    ]


def _response(
    chunks: list[metadata.MetadataChunk], *, own: bool = True, omitted: bool = False
) -> str:
    evidence = next(
        e
        for e in metadata.build_metadata_evidence_catalog(chunks)
        if e.source_chunk_id == ("own" if own else "other")
        and e.scope == ("body" if own else "heading")
    )
    return json.dumps(
        {
            "chunks": [
                {
                    "chunk_id": "own",
                    "fields": []
                    if omitted
                    else [
                        {
                            "field": "clause_label",
                            "value": "b",
                            "basis": "explicit",
                            "evidence_id": evidence.evidence_id,
                            "reference_evidence_id": "",
                        }
                    ],
                }
            ],
            "warnings": [],
        }
    )


@pytest.mark.parametrize(
    "body", ["Kayma riski bulunmamalıdır.", "1.3.5 İlgili hüküm uygulanır."]
)
def test_unlettered_target_can_omit_but_cannot_resurrect_its_old_heading(
    body: str,
) -> None:
    chunks = _sources(body)
    result = validate_contract_retry_response(
        _response(chunks, omitted=True), chunks=chunks, target_id="own"
    )
    assert result.chunks[0].fields == []
    old_heading = next(
        e
        for e in metadata.build_metadata_evidence_catalog(chunks)
        if e.source_chunk_id == "own"
        and e.scope == "heading"
        and e.text.startswith("b)")
    )
    raw = json.loads(_response(chunks))
    raw["chunks"][0]["fields"][0]["evidence_id"] = old_heading.evidence_id
    with pytest.raises(ValueError, match="own BODY prefix"):
        validate_contract_retry_response(
            json.dumps(raw), chunks=chunks, target_id="own"
        )


def test_new_identity_freezes_policy_and_full_context_without_mutating_hints() -> None:
    chunks = _sources()
    before = [c.model_dump(mode="json") for c in chunks]
    request = build_contract_retry_request(uuid4(), chunks, "own")
    require_contract_base_request(request, chunks=chunks, target_id="own")
    assert (
        request.request_hash
        != metadata.build_metadata_request(chunks, ["own"]).request_hash
    )
    assert [c.model_dump(mode="json") for c in chunks] == before
    changed = json.loads(request.prompt)
    changed["metadata_contract_retry"]["policy"] = "trust_heading"
    with pytest.raises(ValueError, match="exact frozen"):
        require_contract_base_request(
            request.model_copy(update={"prompt": json.dumps(changed)}),
            chunks=chunks,
            target_id="own",
        )
    altered = [chunks[0], chunks[1].model_copy(update={"text": "a) Değişen kaynak."})]
    with pytest.raises(ValueError, match="exact frozen"):
        require_contract_base_request(request, chunks=altered, target_id="own")


def test_identical_heading_and_body_text_still_requires_actual_body_catalog_scope() -> (
    None
):
    chunks = _sources()
    chunks[0] = chunks[0].model_copy(
        update={"heading_path": ["MADDE 13", chunks[0].text]}
    )
    heading = next(
        e
        for e in metadata.build_metadata_evidence_catalog(chunks)
        if e.source_chunk_id == "own"
        and e.scope == "heading"
        and e.text == chunks[0].text
    )
    raw = json.loads(_response(chunks))
    raw["chunks"][0]["fields"][0]["evidence_id"] = heading.evidence_id
    # Hydrated quote text alone cannot prove which catalog scope the model selected.
    metadata.validate_metadata_transport_response(
        json.dumps(raw), chunks=chunks, target_ids=["own"]
    )
    with pytest.raises(ValueError, match="own BODY prefix"):
        validate_contract_retry_response(
            json.dumps(raw), chunks=chunks, target_id="own"
        )


@pytest.mark.parametrize("valid", [True, False])
def test_actual_strict_request_engine_keeps_one_declared_target_and_bounded_attempts(
    valid: bool,
) -> None:
    chunks = _sources()
    intent_id = uuid4()
    base = build_contract_retry_request(intent_id, chunks, "own")
    gateway = _Gateway(
        lambda request: _envelope(request, "STOP", _response(chunks, own=valid))
    )
    accepted = []

    def run() -> metadata_job.MetadataResults:
        return metadata_job._extract_metadata_targets(
            gateway=cast(VertexBatchGateway, gateway),
            chunks=chunks,
            targets=["own"],
            identity_prefix="contract-test",
            heartbeat=lambda: None,
            stopped=Event(),
            request_override=base,
            on_accepted=lambda request, result: accepted.append((request, result)),
        )

    if valid:
        result = run()
        assert len(result) == len(accepted) == len(gateway.requests) == 1
        require_contract_accepted_request(
            accepted[0][0], contract_id=intent_id, chunks=chunks, target_id="own"
        )
    else:
        with pytest.raises(ValueError, match="strict validation"):
            run()
        assert accepted == [] and len(gateway.requests) == 2
    for request in gateway.requests:
        payload = json.loads(request.prompt)
        assert payload["target_chunk_ids"] == ["own"]
        assert len(payload["complete_document"]) == len(chunks)


@pytest.mark.parametrize(
    "tokens", [("i", "ii"), ("I", "II"), ("xv", "xvi"), ("XV", "XVI")]
)
@pytest.mark.parametrize("parenthesized", [True, False])
def test_roman_sublist_ambiguity_is_local_to_selected_roman_quote(
    tokens: tuple[str, str],
    parenthesized: bool,
) -> None:
    first, second = tokens
    prefix = "(" if parenthesized else ""
    body = f"a) Asıl bent hükmü.\n{prefix}{first}) Alt liste.\nDevam eden satır.\n{prefix}{second}) İkinci alt liste."
    source = _sources(body)[0]
    assert not metadata.selected_clause_evidence_contradicts(
        value="a", evidence_quote=f"{prefix}{first}) Alt liste.", source=source
    )
    assert metadata.selected_clause_evidence_contradicts(
        value="b", evidence_quote="a) Asıl bent hükmü.", source=source
    )


@pytest.mark.parametrize(
    "value,body",
    [
        ("çç", "çç) İstasyon"),
        ("ıı", "II) İstasyon"),
        ("ii", "İİ) İstasyon"),
        ("ff", "(ff) İstasyon"),
        ("jj", "(jj)İstasyon"),
    ],
)
def test_genuine_turkish_double_markers_keep_their_identity(
    value: str, body: str
) -> None:
    assert metadata.clause_body_identity(body) == value


@pytest.mark.parametrize("marker", ["Iİ", "İI", "ſ", "ſſ"])
def test_mixed_turkish_i_pairs_do_not_become_formal_clause_identities(
    marker: str,
) -> None:
    assert metadata._clause_value(marker) is None
    assert metadata.clause_body_identity(f"{marker}) Örnek") is None
    assert not metadata.selected_clause_evidence_contradicts(
        value="b", evidence_quote=f"{marker}) Örnek"
    )


def _intent() -> MetadataContractRetry:
    chunks = _sources()
    intake_id, package_id, owner_id, member_id, file_id = (
        uuid4(),
        uuid4(),
        uuid4(),
        uuid4(),
        uuid4(),
    )
    identity = uuid4()
    source_hash = metadata.metadata_input_hash(chunks)
    old = MetadataCheckpoint.build(
        recipe_sha256="a" * 64,
        provider_binding={"provider_id": 7},
        source_snapshot={
            "intake_id": str(intake_id),
            "package_id": str(package_id),
            "created_by": str(owner_id),
            "documents": {
                str(member_id): {
                    "user_file_id": str(file_id),
                    "input_sha256": source_hash,
                    "canonical_source_sha256": "b" * 64,
                }
            },
        },
        request_hashes={str(member_id): ["c" * 64]},
    )
    target = ContractInvalidClause(
        reason="selected_clause_contradiction",
        previous_value="b",
        previous_fact={"source": "llm"},
        previous_request_hash="c" * 64,
        previous_base_request_hash="c" * 64,
        previous_checkpoint_request_hash="c" * 64,
        previous_repair_attempt=1,
        previous_raw_ack_sha256="d" * 64,
        previous_cache_file_id="ack",
        before_audit_sha256="e" * 64,
        pending_audit_sha256="f" * 64,
    )
    doc = ContractRetryDocument(
        user_file_id=file_id,
        input_sha256=source_hash,
        canonical_source_sha256="b" * 64,
        preserved_metadata_sha256="f" * 64,
        chunks=chunks,
        targets={"own": target},
        request_hashes={
            "own": build_contract_retry_request(identity, chunks, "own").request_hash
        },
    )
    return MetadataContractRetry(
        id=identity,
        intake_id=intake_id,
        package_id=package_id,
        owner_id=owner_id,
        previous_labeling_run_id=uuid4(),
        previous_checkpoint=old,
        recipe_sha256="f" * 64,
        scope_metadata_sha256={str(member_id): "f" * 64},
        documents={str(member_id): doc},
    )


@pytest.mark.parametrize(
    "change",
    [
        "owner",
        "package",
        "intake",
        "member",
        "file",
        "input",
        "canonical",
        "request",
        "scope",
        "checksum",
    ],
)
def test_frozen_intent_cross_binds_every_owner_member_source_and_request(
    change: str,
) -> None:
    intent = _intent()
    raw = intent.model_dump(mode="json")
    member = next(iter(raw["documents"]))
    doc = raw["documents"][member]
    if change in {"owner", "package", "intake"}:
        raw[change + "_id"] = str(uuid4())
    elif change == "member":
        raw["documents"][str(uuid4())] = raw["documents"].pop(member)
    elif change == "file":
        doc["user_file_id"] = str(uuid4())
    elif change == "input":
        doc["input_sha256"] = "0" * 64
    elif change == "canonical":
        doc["canonical_source_sha256"] = "0" * 64
    elif change == "request":
        doc["request_hashes"]["own"] = "0" * 64
    elif change == "scope":
        raw["scope_metadata_sha256"] = {}
    else:
        raw["previous_checkpoint"]["recipe_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        MetadataContractRetry.model_validate(raw)
    # Existing checkpoint serialization gains no absent optional contract fields.
    assert intent.previous_checkpoint.checkpoint_sha256 == metadata_digest(
        intent.previous_checkpoint.model_dump(
            mode="json", exclude={"checkpoint_sha256"}
        )
    )


def test_complete_intent_size_limit_is_enforced_without_truncation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from onyx.regulatory.document_intake import metadata_contract_retry as contract

    intent = _intent()
    raw = intent.model_dump(mode="json")
    monkeypatch.setattr(
        contract, "MAX_CONTRACT_BYTES", len(intent.model_dump_json().encode()) - 1
    )
    with pytest.raises(ValueError, match="bounded size"):
        MetadataContractRetry.model_validate(raw)


@pytest.mark.parametrize(
    "leaf", ["MADDE 13", "c) Başka bent", "b) Önceki paragraf", ""]
)
def test_target_opposition_needs_matching_direct_leaf_and_exact_body_prefix(
    leaf: str,
) -> None:
    target = _sources("(d) Normal çalışma koşullarında uygulanır.")[0]
    target = target.model_copy(update={"heading_path": ["EK-1", leaf]})
    assert not metadata.target_clause_evidence_contradicts(value="b", target=target)
    real = target.model_copy(
        update={"heading_path": ["EK-1", "(d) Normal çalışma koşullarında..."]}
    )
    assert metadata.target_clause_evidence_contradicts(value="b", target=real)


@pytest.mark.parametrize(
    "mode",
    ["child", "reserve-repair", "wrong-parent", "non-bisected-child", "bad-catalog"],
)
def test_prior_ack_keeps_literal_reserve_repair_and_frozen_parent_ancestry(
    mode: str,
) -> None:
    chunks = [
        metadata.MetadataChunk(
            chunk_id=f"chunk-{i}",
            text=f"a) Kaynak hüküm {i}.",
            heading_path=[f"a) Kaynak hüküm {i}."],
            chunk_type="clause",
        )
        for i in range(8)
    ]
    targets = (
        ["chunk-4", "chunk-6"]
        if mode == "non-bisected-child"
        else ["chunk-4", "chunk-5"]
    )
    root = metadata.build_metadata_request(chunks, [c.chunk_id for c in chunks])
    base = metadata.build_metadata_request(chunks, targets)
    request = base
    if mode == "reserve-repair":
        reserved = metadata.build_metadata_output_reserve_request(base)
        assert reserved is not None
        base = reserved
        request = metadata.build_metadata_repair_request(
            base, '{"chunks":[]}', validation_category="invalid_coverage_or_evidence"
        )
    evidence = next(
        e
        for e in metadata.build_metadata_evidence_catalog(chunks)
        if e.source_chunk_id == "chunk-0" and e.scope == "heading"
    )
    response = json.dumps(
        {
            "chunks": [
                {
                    "chunk_id": target,
                    "fields": [
                        {
                            "field": "clause_label",
                            "value": "b",
                            "basis": "explicit",
                            "evidence_id": evidence.evidence_id,
                            "reference_evidence_id": "",
                        }
                    ]
                    if target == "chunk-4"
                    else [],
                }
                for target in targets
            ],
            "warnings": [],
        }
    )
    if mode == "bad-catalog":
        payload = json.loads(request.prompt)
        payload["evidence_catalog"][0]["text"] = "Forged catalog"
        request = request.model_copy(update={"prompt": json.dumps(payload)})
    binding = LabelingProviderBinding(
        model_configuration_id=1,
        provider_id=2,
        project="fixture",
        location="global",
        authentication_mode=VertexAuthenticationMode.WORKLOAD_IDENTITY,
        credential_identity="workload_identity",
        model=metadata.METADATA_MODEL,
        transport="vertex_direct_v1",
    )
    intake_id, member_id = uuid4(), uuid4()
    prefix = f"metadata-v2:public:{intake_id}:{member_id}:{binding.fingerprint}"
    line = json.dumps(_envelope(request, "STOP", response))
    import hashlib

    from onyx.regulatory.document_intake.metadata_contract_retry import (
        metadata_ack_file_id,
    )

    raw = json.dumps(
        {
            "request_payload": request.model_dump(
                mode="json", exclude={"request_hash"}
            ),
            "request_hash": request.request_hash,
            "binding_sha256": binding.fingerprint,
            "submission_key": metadata_job._metadata_submission_key(prefix, request),
            "line": line,
            "line_sha256": hashlib.sha256(line.encode()).hexdigest(),
        }
    ).encode()
    cache = metadata_ack_file_id(
        tenant_id="public",
        intake_id=intake_id,
        document_id=member_id,
        binding_sha256=binding.fingerprint,
        request_hash=request.request_hash,
    )
    entry = {
        "request_hash": request.request_hash,
        "basis": "explicit",
        "evidence_quote": evidence.text,
        "source_chunk_id": evidence.source_chunk_id,
        "evidence_scope": "heading",
    }
    row = cast(
        RegulatoryChunk,
        SimpleNamespace(id="chunk-4", chunk_metadata={"clause_label": "b"}),
    )
    previous = ["f" * 64] if mode == "wrong-parent" else [root.request_hash]

    def verify() -> tuple[str, int, str]:
        return _previous_lineage(
            raw,
            binding=binding,
            cache_id=cache,
            row=row,
            entry=entry,
            chunks=chunks,
            previous_hashes=previous,
        )

    if mode in {"child", "reserve-repair"}:
        assert verify() == (
            base.request_hash,
            int(mode == "reserve-repair"),
            root.request_hash,
        )
    else:
        with pytest.raises(ValueError):
            verify()


@pytest.mark.parametrize("broker_failed", [False, True])
def test_contract_dispatch_uses_metadata_worker_and_keeps_lost_hint_visible(
    monkeypatch: pytest.MonkeyPatch, broker_failed: bool
) -> None:
    from unittest.mock import MagicMock

    from sqlalchemy.orm import Session

    from onyx.background.celery.tasks.regulatory_amendments import (
        document_intake as tasks,
    )
    from onyx.db.models import RegulatoryDocumentIntake
    from onyx.server.features.regulatory import document_intake_api as api

    intent = _intent()
    checkpoint = MetadataCheckpoint.build(
        recipe_sha256=intent.recipe_sha256,
        provider_binding=intent.previous_checkpoint.provider_binding,
        source_snapshot={
            **intent.previous_checkpoint.source_snapshot,
            "metadata_contract_retry": intent.model_dump(mode="json"),
        },
        request_hashes=intent.previous_checkpoint.request_hashes,
        predecessor=intent.previous_checkpoint,
    )
    batch = RegulatoryDocumentIntake(
        id=intent.intake_id,
        source_package_id=intent.package_id,
        created_by=intent.owner_id,
        status="preparing",
        stage="metadata",
        metadata_recovery_count=1,
        metadata_checkpoint=checkpoint.model_dump(mode="json"),
    )
    session = MagicMock(spec=Session)
    session.get.return_value = batch
    normal, recovery = (
        MagicMock(),
        MagicMock(side_effect=RuntimeError("Lost hint") if broker_failed else None),
    )
    monkeypatch.setattr(tasks, "enqueue_intake_batch", normal)
    monkeypatch.setattr(tasks, "enqueue_intake_metadata_recovery", recovery)
    api._enqueue_batch(session, batch, "public")
    recovery.assert_called_once_with(batch_id=batch.id, tenant_id="public")
    normal.assert_not_called()
    assert batch.metadata_checkpoint == checkpoint.model_dump(mode="json")
    assert batch.status == "preparing" and batch.metadata_recovery_count == 1
    if broker_failed:
        assert batch.error and "delivery is pending" in batch.error
        session.commit.assert_called_once()
