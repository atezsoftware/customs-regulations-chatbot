"""Historical no-label failures require exact retained ACKs and current source."""

import json
from hashlib import sha256
from io import BytesIO

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from onyx.configs.constants import FileOrigin
from onyx.db import regulatory_labeling as labels
from onyx.db.labeling_configuration import LabelingProviderBinding
from onyx.db.models import (
    RegulatoryChunk,
    RegulatoryLabelingItem,
    RegulatoryLabelingRun,
)
from onyx.file_store.postgres_file_store import PostgresBackedFileStore
from onyx.regulatory.indexing_jobs.vertex_batch import VertexBatchRequest
from onyx.regulatory.labeling.accepted_result import LabelingApplyOutcome
from tests.external_dependency_unit.regulatory.test_labeling_jobs import (
    LabelingData,
    _outcomes,
    _prepare,
    _start,
)
from tests.external_dependency_unit.regulatory.test_labeling_jobs import (
    labeling_data as labeling_data,
)
from tests.external_dependency_unit.regulatory.test_labeling_jobs import (
    labeling_database as labeling_database,
)


@pytest.mark.parametrize("tamper", ["none", "source", "ack"])
def test_retained_abstention_normalizes_without_provider_or_source_changes(
    labeling_data: LabelingData, tamper: str
) -> None:
    from onyx.db.regulatory_labeling_abstentions import complete_retained_abstentions

    run_id, _ = _start(labeling_data)
    lease, shard_id = _prepare(labeling_data, run_id)
    with Session(labeling_data.database.engine) as s:
        run = s.get_one(RegulatoryLabelingRun, run_id)
        outcomes: dict[str, LabelingApplyOutcome] = dict(_outcomes(s, lease, shard_id))
        item = s.scalar(
            select(RegulatoryLabelingItem).where(
                RegulatoryLabelingItem.run_id == run_id,
                RegulatoryLabelingItem.regulatory_chunk_id == labeling_data.first_id,
            )
        )
        assert item is not None
        assert item.request_hash is not None
        outcomes[item.request_hash] = (
            "Model abstained: insufficient evidence after bounded semantic reconsideration"
        )
        labels.apply_shard_results(s, lease, shard_id=shard_id, outcomes=outcomes)
        labels.project_derived_labels(s, lease)
        labels.release_run(
            s, lease, status="completed_with_errors", stage="finished", finished=True
        )
        run.provider_binding = {
            **run.provider_binding,
            "transport": "vertex_direct_v1",
            "staging_uri": None,
            "api_key_hash": None,
        }
        binding = LabelingProviderBinding.model_validate(run.provider_binding)
        request = VertexBatchRequest.model_validate(item.request_payload)
        shard = run.shards[0]
        response = '{"labels":[],"abstained":true}'
        line = json.dumps(
            {
                "key": request.request_hash,
                "request": request.to_generate_content_request(),
                "response": {
                    "candidates": [
                        {
                            "finishReason": "STOP",
                            "content": {"parts": [{"text": response}]},
                        }
                    ]
                },
            }
        )
        raw = json.dumps(
            {
                "binding_sha256": binding.fingerprint,
                "submission_key": shard.submission_key,
                "request_hash": request.request_hash,
                "request_payload": request.model_dump(
                    mode="json", exclude={"request_hash"}
                ),
                "line_sha256": sha256(line.encode()).hexdigest(),
                "line": line,
            }
        ).encode()
        file_id = f"regulatory-labeling-direct/{binding.fingerprint}/{shard.submission_key.removeprefix('regulatory-labeling-')}/{request.request_hash}.json"
        store = PostgresBackedFileStore()
        store.save_file(
            BytesIO(raw),
            display_name="retained-real-format-ack.json",
            file_origin=FileOrigin.OTHER,
            file_type="application/json",
            file_id=file_id,
            file_metadata={
                "sha256": sha256(raw).hexdigest() if tamper != "ack" else "f" * 64,
                "binding_sha256": binding.fingerprint,
            },
            db_session=s,
        )
        if tamper == "source":
            s.get_one(RegulatoryChunk, labeling_data.first_id).text += " Changed"
        s.flush()
        before = s.get_one(RegulatoryChunk, labeling_data.first_id).text
        if tamper != "none":
            with pytest.raises(ValueError):
                complete_retained_abstentions(s, run_id)
            s.refresh(item)
            assert item.status == "failed"
        else:
            assert complete_retained_abstentions(s, run_id) == 1
            assert (
                item.status == "completed" and item.labels == [] and item.error is None
            )
            assert run.failed_chunks == 0
            assert complete_retained_abstentions(s, run_id) == 0
            from onyx.regulatory.labeling.abstention import ProviderAbstentionReceipt

            receipt = ProviderAbstentionReceipt.model_validate(
                item.source_snapshot["provider_abstention"]
            )
            assert receipt.response_text == response
        assert s.get_one(RegulatoryChunk, labeling_data.first_id).text == before
        s.rollback()
