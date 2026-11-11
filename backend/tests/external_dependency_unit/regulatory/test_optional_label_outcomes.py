"""Optional classification completes without fabricating labels or hiding errors."""

import json
from collections.abc import Iterator

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from onyx.db.models import (
    RegulatoryChunk,
    RegulatoryLabelingItem,
    RegulatoryLabelingRun,
)
from tests.external_dependency_unit.regulatory.test_labeling_jobs import (
    LabelingData,
    RecordingBatch,
    _finish,
    _start,
)
from tests.external_dependency_unit.regulatory.test_labeling_jobs import (
    fake_batch as fake_batch,
)
from tests.external_dependency_unit.regulatory.test_labeling_jobs import (
    labeling_data as labeling_data,
)
from tests.external_dependency_unit.regulatory.test_labeling_jobs import (
    labeling_database as labeling_database,
)


@pytest.mark.parametrize(
    "response,valid",
    [
        ('{"labels":[],"abstained":true}', True),
        ('{"labels":[],"abstained":false}', True),
        ("not JSON", False),
        (
            '{"labels":[{"label_id":"unknown","evidence_quote":"bad"}],"abstained":false}',
            False,
        ),
    ],
)
def test_valid_empty_classification_is_successful_without_retry(
    labeling_data: LabelingData,
    fake_batch: RecordingBatch,
    monkeypatch: pytest.MonkeyPatch,
    response: str,
    valid: bool,
) -> None:
    with Session(labeling_data.database.engine) as session:
        session.delete(session.get_one(RegulatoryChunk, labeling_data.incomplete_id))
        session.commit()
    original = fake_batch.read_results

    def results(uri: str) -> Iterator[str]:
        for line in original(uri):
            row = json.loads(line)
            requests = fake_batch.requests[uri.removeprefix("files/output-")]
            request = next(r for r in requests if r.request_hash == row["key"])
            if (
                json.loads(request.prompt)["target"]["chunk_id"]
                == labeling_data.first_id
            ):
                row["response"]["candidates"][0]["content"]["parts"][0]["text"] = (
                    response
                )
            yield json.dumps(row)

    monkeypatch.setattr(fake_batch, "read_results", results)
    run_id, _ = _start(labeling_data)
    _finish(labeling_data, run_id)
    assert fake_batch.submit_calls == 1
    with Session(labeling_data.database.engine) as session:
        run = session.get_one(RegulatoryLabelingRun, run_id)
        if not valid:
            assert run.status == "completed_with_errors" and run.failed_chunks == 1
            failed = session.scalar(
                select(RegulatoryLabelingItem).where(
                    RegulatoryLabelingItem.run_id == run_id,
                    RegulatoryLabelingItem.regulatory_chunk_id
                    == labeling_data.first_id,
                )
            )
            assert failed is not None and failed.status == "failed" and failed.error
            return
        assert run.status == "completed"
        assert run.completed_chunks == 2 and run.failed_chunks == 0
        item = session.scalar(
            select(RegulatoryLabelingItem).where(
                RegulatoryLabelingItem.run_id == run_id,
                RegulatoryLabelingItem.regulatory_chunk_id == labeling_data.first_id,
            )
        )
        assert item is not None and item.status == "completed" and item.error is None
        assert item.labels == [] and item.assignments == []
        from onyx.db.regulatory_labeling import completed_classification_counts

        assert completed_classification_counts(session, run_id) == (1, 1)
        if json.loads(response)["abstained"]:
            from onyx.regulatory.labeling.abstention import ProviderAbstentionReceipt

            receipt = ProviderAbstentionReceipt.model_validate(
                item.source_snapshot["provider_abstention"]
            )
            assert (
                receipt.response_text == response
                and receipt.request_hash == item.request_hash
            )
