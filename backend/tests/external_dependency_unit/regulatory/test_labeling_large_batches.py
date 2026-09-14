"""Submission-time shard coalescing preserves frozen requests and context windows."""

import datetime
from dataclasses import replace
from uuid import UUID, uuid4

import pytest
from sqlalchemy import delete, event, inspect, select, update
from sqlalchemy.orm import Session

from onyx.db import regulatory_labeling as repository
from onyx.db.models import (
    RegulatoryChunk,
    RegulatoryLabelingItem,
    RegulatoryLabelingShard,
)
from onyx.regulatory.indexing_jobs.vertex_batch import (
    VertexBatchRequest,
    vertex_batch_submission_key,
    vertex_jsonl_line_size,
)
from onyx.regulatory.labeling.provider import TaxonomyDefinition, build_labeling_request
from tests.external_dependency_unit.regulatory.test_labeling_jobs import (
    LabelingData,
    _start,
)
from tests.external_dependency_unit.regulatory.test_labeling_jobs import (
    labeling_data as labeling_data,
)
from tests.external_dependency_unit.regulatory.test_labeling_jobs import (
    labeling_database as labeling_database,
)


def _prepared_pair(
    data: LabelingData, *, excluded_chunk_ids: list[str] | None = None
) -> tuple[repository.RunLease, UUID, UUID]:
    run_id, _ = _start(data)
    with Session(data.database.engine) as session:
        lease = repository.claim_run(
            session, run_id=run_id, expected_generation=0, lease_seconds=300
        )
        assert lease is not None
        if excluded_chunk_ids:
            session.execute(
                update(RegulatoryLabelingItem)
                .where(
                    RegulatoryLabelingItem.run_id == run_id,
                    RegulatoryLabelingItem.regulatory_chunk_id.in_(excluded_chunk_ids),
                )
                .values(status="failed", error="Excluded test target")
            )
        run = repository.load_claimed_run(session, lease)
        taxonomy = TaxonomyDefinition.model_validate(run.taxonomy.definition)
        shard_ids: list[UUID] = []
        for ordinal in range(2):
            items = repository.prepare_next_item_page(session, lease, limit=1)
            assert len(items) == 1
            item = items[0]
            assert item.context_snapshot is not None
            assert item.canonical_text_sha256 is not None
            request = build_labeling_request(
                chunk_id=item.regulatory_chunk_id,
                text=item.text_snapshot,
                context=item.context_snapshot,
                taxonomy=taxonomy,
                source_hash=item.canonical_text_sha256,
            )
            key = vertex_batch_submission_key(
                [request],
                tenant_id=data.database.schema,
                job_id=run_id,
                output_prefix=f"regulatory-labeling/{run_id}/{ordinal}",
                submission_attempt=1,
            ).replace("regulatory-context-", "regulatory-labeling-", 1)
            repository.store_prepared_shards(
                session,
                lease,
                requests=[
                    repository.PreparedRequest(
                        item_id=item.id,
                        request_hash=request.request_hash,
                        request_payload=request.model_dump(
                            mode="json", exclude_computed_fields=True
                        ),
                    )
                ],
                shards=[
                    repository.PreparedShard(
                        ordinal=ordinal, item_ids=(item.id,), submission_key=key
                    )
                ],
                failed_items={},
            )
            shard_id = session.scalar(
                select(RegulatoryLabelingShard.id).where(
                    RegulatoryLabelingShard.run_id == run_id,
                    RegulatoryLabelingShard.ordinal == ordinal,
                )
            )
            assert shard_id is not None
            shard_ids.append(shard_id)
        session.commit()
        return lease, shard_ids[0], shard_ids[1]


def _merge(
    session: Session,
    data: LabelingData,
    lease: repository.RunLease,
    anchor: UUID,
    *,
    max_items: int = 2048,
    max_jsonl_bytes: int = 128 * 1024 * 1024,
) -> RegulatoryLabelingShard:
    return repository.coalesce_prepared_shards(
        session,
        lease,
        tenant_id=data.database.schema,
        anchor_shard_id=anchor,
        max_items=max_items,
        max_jsonl_bytes=max_jsonl_bytes,
    )


def test_coalescing_preserves_frozen_requests_and_is_atomic_and_repeatable(
    labeling_data: LabelingData,
) -> None:
    lease, anchor_id, donor_id = _prepared_pair(labeling_data)
    with Session(labeling_data.database.engine) as session:
        items = list(
            session.scalars(
                select(RegulatoryLabelingItem)
                .where(RegulatoryLabelingItem.run_id == lease.run_id)
                .order_by(RegulatoryLabelingItem.id)
            )
        )
        frozen = {item.id: (item.request_hash, item.request_payload) for item in items}
        original_anchor = session.get(RegulatoryLabelingShard, anchor_id)
        assert original_anchor is not None
        original_key = original_anchor.submission_key
        requests = [
            VertexBatchRequest.model_validate(item.request_payload) for item in items
        ]
        expected_key = vertex_batch_submission_key(
            requests,
            tenant_id=labeling_data.database.schema,
            job_id=lease.run_id,
            output_prefix=f"regulatory-labeling/{lease.run_id}/0",
            submission_attempt=1,
        ).replace("regulatory-context-", "regulatory-labeling-", 1)
        merged = _merge(session, labeling_data, lease, anchor_id)
        session.flush()
        assert (merged.id, merged.ordinal, merged.submission_key) == (
            anchor_id,
            0,
            expected_key,
        )
        assert set(merged.item_ids) == {str(item.id) for item in items}
        assert session.get(RegulatoryLabelingShard, donor_id) is None
        session.expire_all()
        for item in items:
            assert item.shard_id == anchor_id
            assert (item.request_hash, item.request_payload) == frozen[item.id]
        assert (
            _merge(session, labeling_data, lease, anchor_id).submission_key
            == expected_key
        )
        session.rollback()
        assert session.get(RegulatoryLabelingShard, donor_id) is not None
        anchor = session.get(RegulatoryLabelingShard, anchor_id)
        assert anchor is not None and anchor.submission_key == original_key
        _merge(session, labeling_data, lease, anchor_id)
        session.commit()
    with Session(labeling_data.database.engine) as session:
        assert session.get(RegulatoryLabelingShard, donor_id) is None
        assert len(repository.load_shard_requests(session, lease, anchor_id)) == 2


@pytest.mark.parametrize("boundary", ["count", "bytes", "exact_bytes"])
def test_coalescing_honors_request_and_byte_caps(
    labeling_data: LabelingData, boundary: str
) -> None:
    lease, anchor_id, donor_id = _prepared_pair(labeling_data)
    with Session(labeling_data.database.engine) as session:
        payloads = session.scalars(
            select(RegulatoryLabelingItem.request_payload).where(
                RegulatoryLabelingItem.run_id == lease.run_id
            )
        )
        total_bytes = sum(
            vertex_jsonl_line_size(VertexBatchRequest.model_validate(payload))
            for payload in payloads
        )
        merged = _merge(
            session,
            labeling_data,
            lease,
            anchor_id,
            max_items=1 if boundary == "count" else 2,
            max_jsonl_bytes=total_bytes - (1 if boundary == "bytes" else 0),
        )
        session.flush()
        assert len(merged.item_ids) == (2 if boundary == "exact_bytes" else 1)
        assert (session.get(RegulatoryLabelingShard, donor_id) is None) == (
            boundary == "exact_bytes"
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("status", "submitting"),
        ("status", "reconcile_required"),
        ("status", "submitted"),
        ("attempt_count", 1),
        ("remote_job_name", "projects/test/jobs/1"),
        ("input_uri", "gs://test/input"),
        ("output_uri", "gs://test/output"),
        (
            "reconcile_until",
            datetime.datetime(2100, 1, 1, tzinfo=datetime.timezone.utc),
        ),
    ],
)
def test_coalescing_never_absorbs_attempted_or_uncertain_donors(
    labeling_data: LabelingData, field: str, value: object
) -> None:
    lease, anchor_id, donor_id = _prepared_pair(labeling_data)
    with Session(labeling_data.database.engine) as session:
        donor = session.get(RegulatoryLabelingShard, donor_id)
        assert donor is not None
        setattr(donor, field, value)
        session.flush()
        merged = _merge(session, labeling_data, lease, anchor_id)
        assert len(merged.item_ids) == 1
        assert session.get(RegulatoryLabelingShard, donor_id) is not None


def test_coalescing_preserves_attempted_anchor_and_rejects_lost_lease(
    labeling_data: LabelingData,
) -> None:
    lease, anchor_id, donor_id = _prepared_pair(labeling_data)
    with Session(labeling_data.database.engine) as session:
        anchor = session.get(RegulatoryLabelingShard, anchor_id)
        assert anchor is not None
        anchor.attempt_count = 1
        original_key = anchor.submission_key
        session.flush()
        merged = _merge(session, labeling_data, lease, anchor_id)
        assert merged.submission_key == original_key and len(merged.item_ids) == 1
        with pytest.raises(repository.LabelingStateConflictError):
            _merge(session, labeling_data, replace(lease, token=uuid4()), anchor_id)
        assert session.get(RegulatoryLabelingShard, donor_id) is not None


def test_result_item_pages_do_not_load_repeated_requests_or_context(
    labeling_data: LabelingData,
) -> None:
    lease, anchor_id, _ = _prepared_pair(labeling_data)
    with Session(labeling_data.database.engine) as session:
        _merge(session, labeling_data, lease, anchor_id)
        session.commit()
    with Session(labeling_data.database.engine) as session:
        first = repository.load_shard_result_items(session, lease, anchor_id, limit=1)
        assert len(first) == 1
        assert {"request_payload", "context_snapshot"} <= inspect(first[0]).unloaded
        second = repository.load_shard_result_items(
            session, lease, anchor_id, after_id=first[0].id, limit=1
        )
        assert len(second) == 1 and second[0].id > first[0].id
        assert {"request_payload", "context_snapshot"} <= inspect(second[0]).unloaded


@pytest.mark.parametrize("source_changed", [False, True])
def test_large_batch_revalidates_each_frozen_window_without_union_false_staleness(
    labeling_data: LabelingData, source_changed: bool
) -> None:
    noise_ids = [str(uuid4()) for _ in range(1025)]
    with Session(labeling_data.database.engine) as session:
        second = session.get(RegulatoryChunk, labeling_data.second_id)
        assert second is not None
        second.position = 5000
        for position, identifier in enumerate(noise_ids, start=100):
            session.add(
                RegulatoryChunk(
                    id=identifier,
                    user_file_id=labeling_data.file_id,
                    text=f"Unrelated middle provision {position}.",
                    position=position,
                    projection_ordinal=position,
                    heading_path=["Middle provisions"],
                    chunk_type="article",
                    chunk_metadata={"chunk_variant": "atomic"},
                    source="indexed",
                    status="active",
                )
            )
        session.commit()
    lease, anchor_id, donor_id = _prepared_pair(
        labeling_data, excluded_chunk_ids=noise_ids
    )
    with Session(labeling_data.database.engine) as session:
        anchor = repository.load_claimed_shard(session, lease, anchor_id)
        donor = repository.load_claimed_shard(session, lease, donor_id)
        anchor.item_ids = [*anchor.item_ids, *donor.item_ids]
        session.execute(
            update(RegulatoryLabelingItem)
            .where(RegulatoryLabelingItem.shard_id == donor_id)
            .values(shard_id=anchor_id)
        )
        session.execute(
            delete(RegulatoryLabelingShard).where(
                RegulatoryLabelingShard.id == donor_id
            )
        )
        if source_changed:
            target = session.get(RegulatoryChunk, labeling_data.first_id)
            assert target is not None
            target.text += " Changed after preparation."
        session.flush()
        items = repository.load_shard_requests(session, lease, anchor_id)
        outcomes: dict[str, tuple[list[str], list[dict[str, object]]]] = {
            item.request_hash: (
                ["origin"],
                [{"label_id": "origin", "evidence_quote": item.text_snapshot}],
            )
            for item in items
            if item.request_hash is not None
        }
        repository.apply_shard_results(
            session, lease, shard_id=anchor_id, outcomes=outcomes
        )
        statuses = {item.regulatory_chunk_id: item.status for item in items}
        if source_changed:
            assert statuses[labeling_data.first_id] == "stale"
        else:
            assert set(statuses.values()) == {"completed"}
        assert anchor.status == "succeeded"


def test_hash_only_submission_identity_matches_full_request_identity() -> None:
    from onyx.regulatory.labeling.domain import labeling_submission_key_from_hashes

    requests = [
        VertexBatchRequest(prompt="Birinci hüküm"),
        VertexBatchRequest(prompt="İkinci hüküm"),
    ]
    run_id = UUID("aafa826c-77e1-4d71-9d6c-e16789d29b7e")
    expected = vertex_batch_submission_key(
        requests,
        tenant_id="tenant_test",
        job_id=run_id,
        output_prefix=f"regulatory-labeling/{run_id}/7",
        submission_attempt=1,
    ).replace("regulatory-context-", "regulatory-labeling-", 1)
    assert (
        labeling_submission_key_from_hashes(
            [request.request_hash for request in reversed(requests)],
            tenant_id="tenant_test",
            run_id=run_id,
            ordinal=7,
        )
        == expected
    )


def test_coalescing_rejects_changed_frozen_payload_before_mutation(
    labeling_data: LabelingData,
) -> None:
    lease, anchor_id, donor_id = _prepared_pair(labeling_data)
    with Session(labeling_data.database.engine) as session:
        item = repository.load_shard_requests(session, lease, donor_id)[0]
        assert item.request_payload is not None
        item.request_payload = {
            **item.request_payload,
            "prompt": "Unexpected replacement",
        }
        session.flush()
        with pytest.raises(repository.LabelingStateConflictError, match="frozen"):
            _merge(session, labeling_data, lease, anchor_id)
        assert item.shard_id == donor_id
        assert session.get(RegulatoryLabelingShard, donor_id) is not None


@pytest.mark.parametrize("expired", [False, True])
def test_lease_renewal_extends_only_current_unexpired_owner(
    labeling_data: LabelingData, expired: bool
) -> None:
    lease, _, _ = _prepared_pair(labeling_data)
    with Session(labeling_data.database.engine) as session:
        run = repository.load_claimed_run(session, lease)
        assert run.lease_expires_at is not None
        original_expiry = run.lease_expires_at
        if expired:
            run.lease_expires_at = datetime.datetime.now(
                datetime.timezone.utc
            ) - datetime.timedelta(seconds=1)
            session.flush()
            with pytest.raises(repository.LabelingStateConflictError):
                repository.renew_run_lease(session, lease, lease_seconds=600)
        else:
            repository.renew_run_lease(session, lease, lease_seconds=600)
            session.refresh(run)
            assert run.lease_expires_at > original_expiry
            assert (
                run.lease_generation == lease.generation
                and run.lease_token == lease.token
            )
            with pytest.raises(repository.LabelingStateConflictError):
                repository.renew_run_lease(session, replace(lease, token=uuid4()))


def test_result_pages_can_commit_and_resume_without_reapplying_terminal_items(
    labeling_data: LabelingData,
) -> None:
    lease, anchor_id, _ = _prepared_pair(labeling_data)
    with Session(labeling_data.database.engine) as session:
        _merge(session, labeling_data, lease, anchor_id)
        session.commit()
    with Session(labeling_data.database.engine) as session:
        requests = repository.load_shard_request_page(
            session, lease, anchor_id, limit=1
        )
        assert len(requests) == 1
        assert {"context_snapshot", "text_snapshot", "source_snapshot"} <= inspect(
            requests[0]
        ).unloaded
        first = repository.load_shard_result_items(session, lease, anchor_id, limit=1)[
            0
        ]
        first_id, first_hash = first.id, first.request_hash
        assert first_hash is not None
        assert (
            repository.apply_shard_result_page(
                session,
                lease,
                shard_id=anchor_id,
                item_ids=[first_id],
                outcomes={
                    first_hash: (
                        ["origin"],
                        [{"label_id": "origin", "evidence_quote": first.text_snapshot}],
                    )
                },
            )
            == 1
        )
        assert repository.load_claimed_run(session, lease).completed_chunks == 1
        with pytest.raises(repository.LabelingStateConflictError, match="unfinished"):
            repository.finalize_shard_results(session, lease, anchor_id)
        session.commit()
    with Session(labeling_data.database.engine) as session:
        assert (
            repository.apply_shard_result_page(
                session,
                lease,
                shard_id=anchor_id,
                item_ids=[first_id],
                outcomes={first_hash: "Must not replace a committed result"},
            )
            == 0
        )
        assert repository.load_claimed_run(session, lease).completed_chunks == 1
        second = repository.load_shard_result_items(
            session, lease, anchor_id, after_id=first_id
        )[0]
        assert (
            repository.apply_shard_result_page(
                session,
                lease,
                shard_id=anchor_id,
                item_ids=[second.id],
                outcomes={},
            )
            == 1
        )
        assert repository.load_claimed_run(session, lease).failed_chunks == 1
        repository.finalize_shard_results(session, lease, anchor_id)
        session.commit()
        first = session.get(RegulatoryLabelingItem, first_id)
        assert (
            first is not None
            and first.status == "completed"
            and first.labels == ["origin"]
        )
        assert second.status == "failed" and "omitted" in (second.error or "")
        assert (
            repository.load_claimed_shard(session, lease, anchor_id).status
            == "succeeded"
        )


def test_mark_submitting_updates_status_without_loading_request_payloads(
    labeling_data: LabelingData,
) -> None:
    lease, anchor_id, _ = _prepared_pair(labeling_data)
    statements: list[str] = []

    def record_statement(
        _connection: object,
        _cursor: object,
        statement: str,
        _parameters: object,
        _context: object,
        _executemany: bool,
    ) -> None:
        statements.append(statement)

    with Session(labeling_data.database.engine) as session:
        _merge(session, labeling_data, lease, anchor_id)
        session.commit()
    event.listen(
        labeling_data.database.engine, "before_cursor_execute", record_statement
    )
    try:
        with Session(labeling_data.database.engine) as session:
            repository.mark_shard_submitting(
                session, lease, shard_id=anchor_id, reconcile_seconds=300
            )
            session.commit()
    finally:
        event.remove(
            labeling_data.database.engine, "before_cursor_execute", record_statement
        )
    assert not any(
        statement.lstrip().upper().startswith("SELECT")
        and "request_payload" in statement
        for statement in statements
    )
    with Session(labeling_data.database.engine) as session:
        statuses = session.scalars(
            select(RegulatoryLabelingItem.status).where(
                RegulatoryLabelingItem.run_id == lease.run_id,
                RegulatoryLabelingItem.shard_id == anchor_id,
            )
        ).all()
        assert statuses == ["submitted", "submitted"]


def test_coalescing_crosses_candidate_pages_with_metadata_only(
    labeling_data: LabelingData,
) -> None:
    lease, anchor_id, _ = _prepared_pair(labeling_data)
    with Session(labeling_data.database.engine) as session:
        for ordinal in range(2, 70):
            request = VertexBatchRequest(prompt=f"Frozen request {ordinal}")
            item_id = uuid4()
            shard = RegulatoryLabelingShard(
                run_id=lease.run_id,
                ordinal=ordinal,
                item_ids=[str(item_id)],
                submission_key="regulatory-labeling-" + uuid4().hex + uuid4().hex,
            )
            session.add(shard)
            session.flush()
            session.add(
                RegulatoryLabelingItem(
                    id=item_id,
                    run_id=lease.run_id,
                    shard_id=shard.id,
                    regulatory_chunk_id=f"frozen-test-source-{uuid4()}",
                    user_file_id=labeling_data.file_id,
                    text_snapshot=request.prompt,
                    source_snapshot={"position": ordinal},
                    request_hash=request.request_hash,
                    request_payload=request.model_dump(
                        mode="json", exclude_computed_fields=True
                    ),
                )
            )
        session.commit()
    with Session(labeling_data.database.engine) as session:
        merged = _merge(
            session,
            labeling_data,
            lease,
            anchor_id,
            max_items=200_000,
            max_jsonl_bytes=1_000_000_000,
        )
        assert len(merged.item_ids) == 70
        assert list(
            session.scalars(
                select(RegulatoryLabelingShard.id).where(
                    RegulatoryLabelingShard.run_id == lease.run_id
                )
            )
        ) == [anchor_id]


def test_coalescing_renews_lease_while_streaming_past_original_expiry(
    labeling_data: LabelingData, monkeypatch: pytest.MonkeyPatch
) -> None:
    lease, anchor_id, _ = _prepared_pair(labeling_data)
    clock = datetime.datetime.now(datetime.timezone.utc)
    start = clock
    original_size = repository.vertex_jsonl_line_size

    def slow_line_size(request: VertexBatchRequest) -> int:
        nonlocal clock
        clock += datetime.timedelta(seconds=181)
        return original_size(request)

    monkeypatch.setattr(repository, "_utcnow", lambda: clock)
    monkeypatch.setattr(
        repository, "monotonic", lambda: (clock - start).total_seconds(), raising=False
    )
    monkeypatch.setattr(repository, "vertex_jsonl_line_size", slow_line_size)
    with Session(labeling_data.database.engine) as session:
        run = repository.load_claimed_run(session, lease)
        original_expiry = run.lease_expires_at
        assert original_expiry is not None
        merged = _merge(session, labeling_data, lease, anchor_id)
        session.refresh(run)
        assert len(merged.item_ids) == 2
        assert clock > original_expiry
        assert run.lease_expires_at is not None
        assert run.lease_expires_at > clock + datetime.timedelta(seconds=250)
        assert run.lease_generation == lease.generation


def test_lease_renewal_rejects_recorded_cancellation(
    labeling_data: LabelingData,
) -> None:
    lease, _, _ = _prepared_pair(labeling_data)
    with Session(labeling_data.database.engine) as session:
        run = repository.load_claimed_run(session, lease)
        original_expiry = run.lease_expires_at
        run.cancel_requested = True
        session.flush()
        with pytest.raises(repository.LabelingCancellationRequested):
            repository.renew_run_lease(session, lease)
        session.refresh(run)
        assert run.lease_expires_at == original_expiry


@pytest.mark.parametrize("operation", ["coalesce", "apply", "finalize"])
def test_large_batch_mutations_reject_expired_lease(
    labeling_data: LabelingData, operation: str
) -> None:
    lease, anchor_id, _ = _prepared_pair(labeling_data)
    with Session(labeling_data.database.engine) as session:
        run = repository.load_claimed_run(session, lease)
        run.lease_expires_at = datetime.datetime.now(
            datetime.timezone.utc
        ) - datetime.timedelta(seconds=1)
        session.flush()
        with pytest.raises(repository.LabelingStateConflictError, match="expired"):
            if operation == "coalesce":
                _merge(session, labeling_data, lease, anchor_id)
            elif operation == "apply":
                shard = repository.load_claimed_shard(session, lease, anchor_id)
                repository.apply_shard_result_page(
                    session,
                    lease,
                    shard_id=anchor_id,
                    item_ids=[UUID(shard.item_ids[0])],
                    outcomes={},
                )
            else:
                repository.finalize_shard_results(session, lease, anchor_id)


def test_preparation_store_uses_bulk_updates_and_no_corpus_aggregates(
    labeling_data: LabelingData,
) -> None:
    run_id, _ = _start(labeling_data)
    statements: list[str] = []

    def record_statement(
        _connection: object,
        _cursor: object,
        statement: str,
        _parameters: object,
        _context: object,
        _executemany: bool,
    ) -> None:
        statements.append(statement.lower())

    with Session(labeling_data.database.engine) as session:
        lease = repository.claim_run(
            session, run_id=run_id, expected_generation=0, lease_seconds=300
        )
        assert lease is not None
        items = repository.prepare_next_item_page(session, lease, limit=128)
        assert len(items) == 2
        requests = [
            repository.PreparedRequest(
                item_id=item.id,
                request_hash=f"{ordinal:064x}",
                request_payload={"prompt": item.text_snapshot},
            )
            for ordinal, item in enumerate(items)
        ]
        event.listen(
            labeling_data.database.engine, "before_cursor_execute", record_statement
        )
        try:
            stage = repository.store_prepared_shards(
                session,
                lease,
                requests=requests,
                failed_items={},
                shards=[
                    repository.PreparedShard(
                        ordinal=0,
                        item_ids=tuple(item.id for item in items),
                        submission_key="regulatory-labeling-" + "a" * 64,
                    )
                ],
            )
        finally:
            event.remove(
                labeling_data.database.engine, "before_cursor_execute", record_statement
            )
        assert stage == "submitting"
        assert not any("count(" in sql or "group by" in sql for sql in statements)
        item_updates = [
            sql
            for sql in statements
            if sql.startswith("update") and "regulatory_labeling_item" in sql
        ]
        assert len(item_updates) == 1
        session.expire_all()
        shard_id = session.scalar(
            select(RegulatoryLabelingShard.id).where(
                RegulatoryLabelingShard.run_id == run_id
            )
        )
        assert shard_id is not None
        frozen = repository.load_shard_requests(session, lease, shard_id)
        assert {item.request_hash for item in frozen} == {
            request.request_hash for request in requests
        }


def test_preparation_failures_increment_once_and_reject_nonowned_items(
    labeling_data: LabelingData,
) -> None:
    run_id, _ = _start(labeling_data)
    with Session(labeling_data.database.engine) as session:
        lease = repository.claim_run(
            session, run_id=run_id, expected_generation=0, lease_seconds=300
        )
        assert lease is not None
        items = repository.prepare_next_item_page(session, lease, limit=128)
        stage = repository.store_prepared_shards(
            session,
            lease,
            requests=[],
            shards=[],
            failed_items={items[0].id: "failure"},
        )
        assert stage == "preparing"
        assert repository.load_claimed_run(session, lease).failed_chunks == 1
        session.commit()
        with pytest.raises(repository.LabelingStateConflictError, match="ownership"):
            repository.store_prepared_shards(
                session,
                lease,
                requests=[],
                shards=[],
                failed_items={items[0].id: "replay"},
            )
        session.rollback()
        with pytest.raises(repository.LabelingStateConflictError, match="ownership"):
            repository.store_prepared_shards(
                session,
                lease,
                requests=[],
                shards=[],
                failed_items={uuid4(): "foreign"},
            )
        session.rollback()
        assert repository.load_claimed_run(session, lease).failed_chunks == 1


def test_ambiguous_preparation_counts_all_failed_targets_without_full_scan(
    labeling_data: LabelingData, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_id, _ = _start(labeling_data)
    monkeypatch.setattr(repository, "_CONTEXT_CANDIDATE_LIMIT", 1)
    with Session(labeling_data.database.engine) as session:
        lease = repository.claim_run(
            session, run_id=run_id, expected_generation=0, lease_seconds=300
        )
        assert lease is not None
        assert repository.prepare_next_item_page(session, lease, limit=128) == []
        run = repository.load_claimed_run(session, lease)
        assert run.failed_chunks == 2
        assert (
            repository.store_prepared_shards(
                session, lease, requests=[], shards=[], failed_items={}
            )
            == "submitting"
        )
        session.commit()
        assert run.failed_chunks == 2
        assert set(
            session.scalars(
                select(RegulatoryLabelingItem.status).where(
                    RegulatoryLabelingItem.run_id == run_id
                )
            )
        ) == {"failed"}


@pytest.mark.parametrize("foreign", [False, True])
def test_preparation_bulk_update_rejects_already_frozen_or_foreign_request(
    labeling_data: LabelingData, foreign: bool
) -> None:
    run_id, _ = _start(labeling_data)
    with Session(labeling_data.database.engine) as session:
        lease = repository.claim_run(
            session, run_id=run_id, expected_generation=0, lease_seconds=300
        )
        assert lease is not None
        item = repository.prepare_next_item_page(session, lease, limit=1)[0]
        item_id = item.id
        original = repository.PreparedRequest(
            item_id=item_id,
            request_hash="a" * 64,
            request_payload={"prompt": "original"},
        )
        assert (
            repository.store_prepared_shards(
                session,
                lease,
                requests=[original],
                failed_items={},
                shards=[
                    repository.PreparedShard(
                        ordinal=0, item_ids=(item_id,), submission_key="a" * 64
                    )
                ],
            )
            == "preparing"
        )
        session.commit()
        target_id = uuid4() if foreign else item_id
        with pytest.raises(repository.LabelingStateConflictError, match="ownership"):
            repository.store_prepared_shards(
                session,
                lease,
                failed_items={},
                requests=[
                    repository.PreparedRequest(
                        item_id=target_id,
                        request_hash="b" * 64,
                        request_payload={"prompt": "replacement"},
                    )
                ],
                shards=[
                    repository.PreparedShard(
                        ordinal=1, item_ids=(target_id,), submission_key="b" * 64
                    )
                ],
            )
        session.rollback()
        session.refresh(item)
        assert item.request_hash == original.request_hash
        assert item.request_payload == original.request_payload
        assert (
            len(
                list(
                    session.scalars(
                        select(RegulatoryLabelingShard.id).where(
                            RegulatoryLabelingShard.run_id == run_id
                        )
                    )
                )
            )
            == 1
        )


def test_initial_submission_error_gets_full_reconciliation_window(
    labeling_data: LabelingData, monkeypatch: pytest.MonkeyPatch
) -> None:
    lease, anchor_id, _ = _prepared_pair(labeling_data)
    now = datetime.datetime.now(datetime.timezone.utc)
    with Session(labeling_data.database.engine) as session:
        with monkeypatch.context() as clock:
            clock.setattr(
                repository, "_utcnow", lambda: now - datetime.timedelta(minutes=10)
            )
            shard = repository.mark_shard_submitting(
                session, lease, shard_id=anchor_id, reconcile_seconds=300
            )
        assert shard.reconcile_until is not None and shard.reconcile_until < now
        key, attempt = shard.submission_key, shard.attempt_count
        monkeypatch.setattr(repository, "_utcnow", lambda: now)
        repository.record_shard_state(
            session,
            lease,
            shard_id=anchor_id,
            status="reconcile_required",
            reconcile_seconds=300,
            error="The long upload ended with an unknown response",
        )
        assert shard.reconcile_until == now + datetime.timedelta(seconds=300)
        assert (shard.submission_key, shard.attempt_count) == (key, attempt)
        assert shard.status == "reconcile_required"
        monkeypatch.setattr(
            repository, "_utcnow", lambda: now + datetime.timedelta(seconds=60)
        )
        repository.record_shard_state(
            session,
            lease,
            shard_id=anchor_id,
            status="reconcile_required",
            reconcile_seconds=300,
            error="The remote job is not visible yet",
        )
        assert shard.reconcile_until == now + datetime.timedelta(seconds=300)
        assert (shard.submission_key, shard.attempt_count) == (key, attempt)


@pytest.mark.parametrize(
    "status,seconds",
    [("submitted", 300), ("reconcile_required", 0), ("reconcile_required", -1)],
)
def test_reconciliation_window_rejects_invalid_configuration(
    labeling_data: LabelingData, status: str, seconds: int
) -> None:
    lease, anchor_id, _ = _prepared_pair(labeling_data)
    with Session(labeling_data.database.engine) as session:
        with pytest.raises(ValueError, match="reconciliation"):
            repository.record_shard_state(
                session,
                lease,
                shard_id=anchor_id,
                status=status,
                reconcile_seconds=seconds,
            )
        assert (
            repository.load_claimed_shard(session, lease, anchor_id).status
            == "prepared"
        )
