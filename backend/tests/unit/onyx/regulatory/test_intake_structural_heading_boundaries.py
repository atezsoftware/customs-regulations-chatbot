"""Reader-only structural projection must preserve ordinary classifier failures."""

import json
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from sqlalchemy.orm import Session

from onyx.db import document_intake_labeling
from onyx.db.models import RegulatoryLabelingItem
from onyx.db.regulatory_labeling import RunLease
from onyx.regulatory.indexing_jobs.vertex_batch import VertexBatchGateway
from onyx.regulatory.labeling import orchestrator
from onyx.regulatory.labeling.provider import (
    LabelDefinition,
    TaxonomyDefinition,
    build_labeling_request,
)


def test_ordinary_body_abstention_stays_a_failure_outcome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text = "İthalatçı menşe belgelerini gümrük idaresine ibraz eder."
    taxonomy = TaxonomyDefinition(
        name="Classification",
        labels=[LabelDefinition(id="origin", name="Menşe", description="Menşe ispatı")],
    )
    request = build_labeling_request(
        chunk_id="rc_ordinary_body",
        text=text,
        context="Surrounding source is advisory only.",
        taxonomy=taxonomy,
        source_hash="a" * 64,
    )
    lease = RunLease(run_id=uuid4(), generation=1, token=uuid4())
    shard_id = uuid4()
    item = RegulatoryLabelingItem(
        id=uuid4(),
        run_id=lease.run_id,
        regulatory_chunk_id="rc_ordinary_body",
        user_file_id=uuid4(),
        text_snapshot=text,
        source_snapshot={"chunk_type": "paragraph", "chunk_metadata": {}},
        status="submitted",
        request_hash=request.request_hash,
        request_payload=request.model_dump(mode="json"),
    )
    line = json.dumps(
        {
            "key": request.request_hash,
            "request": request.to_generate_content_request(),
            "response": {
                "candidates": [
                    {
                        "finishReason": "STOP",
                        "content": {
                            "parts": [{"text": '{"labels":[],"abstained":true}'}]
                        },
                    }
                ]
            },
        }
    )
    gateway = cast(
        VertexBatchGateway,
        SimpleNamespace(read_results=lambda _uri: iter([line])),
    )
    session = MagicMock(spec=Session)
    session.__enter__.return_value = session
    monkeypatch.setattr(
        orchestrator, "get_session_with_current_tenant", lambda: session
    )
    monkeypatch.setattr(orchestrator, "_lease_heartbeat", lambda _lease: lambda: None)
    monkeypatch.setattr(orchestrator, "_result_pages", lambda *_args: iter([[item]]))
    run = SimpleNamespace(
        taxonomy=SimpleNamespace(definition=taxonomy.model_dump()),
        provider_binding={"transport": "vertex_direct_v1"},
    )
    monkeypatch.setattr(orchestrator.repository, "load_claimed_run", lambda *_args: run)
    monkeypatch.setattr(
        orchestrator.repository, "set_claimed_run_stage", lambda *_args: None
    )
    monkeypatch.setattr(
        document_intake_labeling, "is_current_intake_labeling_run", lambda *_args: False
    )
    repair = MagicMock(
        side_effect=AssertionError("Abstention cannot become quote repair")
    )
    monkeypatch.setattr(orchestrator, "_repair_intake_quotes", repair)
    apply = MagicMock()
    monkeypatch.setattr(orchestrator.repository, "apply_shard_result_page", apply)

    orchestrator._read_and_apply_results(lease, shard_id, gateway, "retained-ack")

    assert apply.call_args.kwargs["outcomes"] == {
        request.request_hash: "Model abstained: insufficient evidence"
    }
    assert apply.call_args.kwargs["item_ids"] == [item.id]
    repair.assert_not_called()
