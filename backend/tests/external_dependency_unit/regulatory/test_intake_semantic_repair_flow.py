"""Normal result apply retains one bounded repair and its source-current receipt."""

import json
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from copy import deepcopy
from hashlib import sha256
from typing import cast
from uuid import uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from onyx.db import document_intake_edit as editing
from onyx.db import regulatory_labeling as labeling
from onyx.db.enums import UserFileStatus
from onyx.db.models import (
    RegulatoryChunk,
    RegulatoryDocumentIntake,
    RegulatoryDocumentIntakeFile,
    RegulatoryLabelingItem,
    RegulatoryLabelingRun,
    RegulatoryLabelingShard,
    RegulatoryLabelTaxonomy,
    UserFile,
)
from onyx.db.regulatory_annex_changes import capture_canonical_scope
from onyx.document_index.publication_models import publication_digest
from onyx.regulatory.indexing_jobs.vertex_batch import (
    VertexBatchGateway,
    VertexBatchJobStatus,
    VertexBatchRequest,
    VertexBatchState,
)
from onyx.regulatory.labeling import orchestrator
from onyx.regulatory.labeling.intake_abstention_repair import (
    ABSTENTION_REPAIR_KEY,
    AbstentionRepairAudit,
)
from onyx.regulatory.labeling.provider import TaxonomyDefinition, build_labeling_request
from tests.external_dependency_unit.regulatory.test_intake_semantic_repair_audit import (
    seed_v4_owned_intake_source,
)
from tests.external_dependency_unit.regulatory.test_labeling_jobs import (
    LabelingData,
)
from tests.external_dependency_unit.regulatory.test_labeling_jobs import (
    labeling_data as labeling_data,
)
from tests.external_dependency_unit.regulatory.test_labeling_jobs import (
    labeling_database as labeling_database,
)

REJECTED = '{"labels":[],"abstained":true}'
ACCEPTED = '{"labels":[{"label_id":"customs_value","evidence_quote":"customs value"}],"abstained":false}'
SECOND = '{"labels":[{"label_id":"origin","evidence_quote":"certificate of origin"}],"abstained":false}'


def _line(request: VertexBatchRequest, response: str) -> str:
    return json.dumps(
        {
            "key": request.request_hash,
            "request": request.to_generate_content_request(),
            "response": {
                "candidates": [
                    {"finishReason": "STOP", "content": {"parts": [{"text": response}]}}
                ]
            },
        }
    )


class _CachedGateway:
    def __init__(self, originals: list[str], repaired: str) -> None:
        self.originals = originals
        self.repaired = repaired
        self.repair_request: VertexBatchRequest | None = None
        self.submissions = 0

    def submit(
        self,
        requests: Sequence[VertexBatchRequest],
        *,
        submission_key: str,
        max_jsonl_bytes: int,
    ) -> VertexBatchState:
        assert len(requests) == 1 and submission_key and max_jsonl_bytes > 0
        self.submissions += 1
        self.repair_request = requests[0]
        return VertexBatchState(
            remote_job_name="cached-semantic-repair",
            status=VertexBatchJobStatus.SUCCEEDED,
            output_uri="fixture://repair",
        )

    def read_results(self, output_uri: str) -> Iterator[str]:
        if output_uri == "fixture://original":
            return iter(self.originals)
        assert output_uri == "fixture://repair" and self.repair_request is not None
        return iter([_line(self.repair_request, self.repaired)])


@pytest.mark.parametrize(
    ("mode", "version"),
    [
        ("accepted", "4"),
        ("continued", "4"),
        ("legacy", "3"),
        ("accepted", "5"),
        ("continued", "5"),
    ],
)
def test_one_semantic_repair_is_persisted_reused_and_source_checked(
    labeling_data: LabelingData,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    version: str,
) -> None:
    data = labeling_data
    with data.database.engine.connect() as connection:
        transaction = connection.begin()
        try:
            with Session(
                connection, join_transaction_mode="create_savepoint"
            ) as session:
                seed_v4_owned_intake_source(session, data)
                session.get_one(UserFile, data.file_id).status = UserFileStatus.CHUNKED
                if version == "5":
                    for identifier in (data.first_id, data.second_id):
                        row = session.get_one(RegulatoryChunk, identifier)
                        row.chunk_metadata = {
                            **row.chunk_metadata,
                            "intake_chunking_version": "5",
                            "intake_preparation_policy": {
                                **row.chunk_metadata["intake_preparation_policy"],
                                "chunking_version": "5",
                            },
                        }
                if mode == "legacy":
                    for identifier in (data.first_id, data.second_id):
                        row = session.get_one(RegulatoryChunk, identifier)
                        row.chunk_metadata = {
                            **row.chunk_metadata,
                            "intake_chunking_version": "3",
                        }
                taxonomy = session.get_one(RegulatoryLabelTaxonomy, data.taxonomy_id)
                binding: dict[str, object] = {
                    "transport": "vertex_direct_v1",
                    "model": "gemini-3.8-flash",
                    "model_configuration_id": data.model_id,
                }

                def create() -> RegulatoryLabelingRun:
                    return labeling.create_labeling_run(
                        session,
                        document_set_id=data.document_set_id,
                        taxonomy=taxonomy,
                        model_configuration_id=data.model_id,
                        model="gemini-3.8-flash",
                        provider_binding=binding,
                        requested_by_id=data.user_id,
                        idempotency_key=uuid4(),
                        target_file_ids=[data.file_id],
                    )[0]

                run = create()
                batch = session.scalar(
                    select(RegulatoryDocumentIntake).where(
                        RegulatoryDocumentIntake.created_by == data.user_id
                    )
                )
                assert batch is not None
                batch.status, batch.stage, batch.labeling_run_id = (
                    "preparing",
                    "labeling",
                    run.id,
                )
                member = session.scalar(
                    select(RegulatoryDocumentIntakeFile).where(
                        RegulatoryDocumentIntakeFile.intake_id == batch.id
                    )
                )
                assert member is not None
                member.status = "labeling"
                lease = labeling.claim_run(
                    session, run_id=run.id, expected_generation=0, lease_seconds=600
                )
                assert lease is not None
                items = labeling.prepare_next_item_page(session, lease, limit=128)
                vocabulary = TaxonomyDefinition.model_validate(taxonomy.definition)
                requests = {
                    item.regulatory_chunk_id: build_labeling_request(
                        chunk_id=item.regulatory_chunk_id,
                        text=item.text_snapshot,
                        context=item.context_snapshot or "",
                        taxonomy=vocabulary,
                        source_hash=item.canonical_text_sha256 or "",
                    )
                    for item in items
                }
                labeling.store_prepared_shards(
                    session,
                    lease,
                    requests=[
                        labeling.PreparedRequest(
                            item_id=item.id,
                            request_hash=requests[
                                item.regulatory_chunk_id
                            ].request_hash,
                            request_payload=requests[
                                item.regulatory_chunk_id
                            ].model_dump(mode="json", exclude_computed_fields=True),
                        )
                        for item in items
                    ],
                    shards=[
                        labeling.PreparedShard(
                            ordinal=0,
                            item_ids=tuple(item.id for item in items),
                            submission_key="regulatory-labeling-"
                            + sha256(str(run.id).encode()).hexdigest(),
                        )
                    ],
                    failed_items={},
                )
                shard = session.scalar(
                    select(RegulatoryLabelingShard).where(
                        RegulatoryLabelingShard.run_id == run.id
                    )
                )
                assert shard is not None
                labeling.mark_shard_submitting(
                    session, lease, shard_id=shard.id, reconcile_seconds=30
                )
                labeling.record_shard_state(
                    session, lease, shard_id=shard.id, status="submitted"
                )
                session.flush()
                originals = [
                    _line(requests[data.first_id], REJECTED),
                    _line(requests[data.second_id], SECOND),
                ]
                gateway = _CachedGateway(
                    originals, REJECTED if mode == "continued" else ACCEPTED
                )

                @contextmanager
                def current_session() -> Iterator[Session]:
                    yield session

                monkeypatch.setattr(
                    orchestrator, "get_session_with_current_tenant", current_session
                )
                monkeypatch.setattr(
                    orchestrator, "_lease_heartbeat", lambda _lease: lambda: None
                )
                before = publication_digest(
                    [
                        row.model_dump(mode="json")
                        for row in capture_canonical_scope(session, data.file_id)
                    ]
                )
                orchestrator._read_and_apply_results(
                    lease,
                    shard.id,
                    cast(VertexBatchGateway, gateway),
                    "fixture://original",
                )
                first = session.scalar(
                    select(RegulatoryLabelingItem).where(
                        RegulatoryLabelingItem.run_id == run.id,
                        RegulatoryLabelingItem.regulatory_chunk_id == data.first_id,
                    )
                )
                assert first is not None
                assert gateway.submissions == (0 if mode == "legacy" else 1)
                assert gateway.originals == originals
                assert first.request_hash == requests[data.first_id].request_hash
                saved = deepcopy(first.source_snapshot), first.status, first.error
                orchestrator._read_and_apply_results(
                    lease,
                    shard.id,
                    cast(VertexBatchGateway, gateway),
                    "fixture://original",
                )
                assert gateway.submissions == (0 if mode == "legacy" else 1)
                assert (first.source_snapshot, first.status, first.error) == saved
                if mode != "accepted":
                    assert first.status == "failed" and "abstained" in (
                        first.error or ""
                    )
                    assert ABSTENTION_REPAIR_KEY not in first.source_snapshot
                else:
                    assert first.status == "completed" and first.labels == [
                        "customs_value"
                    ]
                    audit = AbstentionRepairAudit.model_validate(
                        first.source_snapshot[ABSTENTION_REPAIR_KEY]
                    ).model_dump(mode="json")
                    assert (
                        audit["attempt"] == 1
                        and audit["base_request_hash"] == first.request_hash
                    )
                    completed = list(
                        session.scalars(
                            select(RegulatoryLabelingItem).where(
                                RegulatoryLabelingItem.run_id == run.id
                            )
                        )
                    )
                    assert labeling.current_labeling_items_for_search(
                        session, run=run, items=completed
                    ) == {data.first_id, data.second_id}
                    run.status, run.stage, run.lease_token, run.lease_expires_at = (
                        "completed",
                        "finished",
                        None,
                        None,
                    )
                    session.flush()
                    replacement = create()
                    assert (
                        editing.reuse_completed_intake_labels(
                            session, previous=run, replacement=replacement
                        )
                        == 2
                    )
                    copied = session.scalar(
                        select(RegulatoryLabelingItem).where(
                            RegulatoryLabelingItem.run_id == replacement.id,
                            RegulatoryLabelingItem.regulatory_chunk_id == data.first_id,
                        )
                    )
                    assert (
                        copied is not None
                        and copied.source_snapshot[ABSTENTION_REPAIR_KEY] == audit
                    )
                    assert (
                        copied.request_hash == first.request_hash
                        and copied.context_sha256 == first.context_sha256
                    )
                    corrupted = deepcopy(audit)
                    corrupted["output_response_sha256"] = "0" * 64
                    copied.source_snapshot = {
                        **copied.source_snapshot,
                        ABSTENTION_REPAIR_KEY: corrupted,
                    }
                    session.flush()
                    assert (
                        data.first_id
                        not in labeling.current_labeling_items_for_search(
                            session, run=replacement, items=[copied]
                        )
                    )
                assert (
                    publication_digest(
                        [
                            row.model_dump(mode="json")
                            for row in capture_canonical_scope(session, data.file_id)
                        ]
                    )
                    == before
                )
        finally:
            transaction.rollback()
