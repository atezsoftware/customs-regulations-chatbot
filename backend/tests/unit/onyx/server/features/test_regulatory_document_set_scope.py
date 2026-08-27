import datetime
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from onyx.db.models import BenchmarkQuestion, DocumentSet
from onyx.db.regulatory_benchmark import _question_snapshot
from onyx.error_handling.error_codes import OnyxErrorCode
from onyx.error_handling.exceptions import OnyxError
from onyx.server.features.regulatory.benchmark_api import (
    _expected_citation_snapshots,
)
from onyx.server.features.regulatory.benchmark_api import (
    router as benchmark_router,
)
from onyx.server.features.regulatory.benchmark_models import (
    BenchmarkQuestionSnapshot,
)
from onyx.server.features.regulatory.models import (
    AmendmentBatchSnapshot,
    AmendmentProposalSnapshot,
)


def test_benchmark_citation_must_belong_to_selected_document_set() -> None:
    selected_file = SimpleNamespace(id=uuid4(), name="Selected.pdf")
    outside_chunk = SimpleNamespace(
        id="outside-chunk",
        user_file_id=uuid4(),
        heading_path=["Article 1"],
        text="Outside the selected set",
    )
    document_set = cast(
        DocumentSet,
        SimpleNamespace(id=12, user_files=[selected_file]),
    )

    with (
        patch(
            "onyx.server.features.regulatory.benchmark_api.get_chunk_by_id",
            return_value=outside_chunk,
        ),
        pytest.raises(OnyxError) as exc_info,
    ):
        _expected_citation_snapshots(
            MagicMock(),
            document_set=document_set,
            citations=[{"chunk_id": outside_chunk.id}],
        )

    assert exc_info.value.error_code is OnyxErrorCode.INVALID_INPUT
    assert "selected document set" in exc_info.value.detail


def test_benchmark_citation_snapshot_uses_document_set_file() -> None:
    user_file = SimpleNamespace(id=uuid4(), name="Regulation.pdf")
    chunk = SimpleNamespace(
        id="chunk-1",
        user_file_id=user_file.id,
        heading_path=["Chapter I", "Article 3"],
        text="A" * 1200,
    )
    document_set = cast(
        DocumentSet,
        SimpleNamespace(id=15, user_files=[user_file]),
    )

    with patch(
        "onyx.server.features.regulatory.benchmark_api.get_chunk_by_id",
        return_value=chunk,
    ):
        snapshots = _expected_citation_snapshots(
            MagicMock(),
            document_set=document_set,
            citations=[
                {
                    "chunk_id": chunk.id,
                    "requirement": "supporting",
                    "notes": "Useful context",
                }
            ],
        )

    assert snapshots == [
        {
            "chunk_id": "chunk-1",
            "requirement": "supporting",
            "notes": "Useful context",
            "file_name": "Regulation.pdf",
            "heading_path": ["Chapter I", "Article 3"],
            "text_excerpt": "A" * 1000,
        }
    ]


def test_benchmark_scope_snapshots_use_canonical_document_set_fields() -> None:
    created_at = datetime.datetime.now(datetime.timezone.utc)
    question = cast(
        BenchmarkQuestion,
        SimpleNamespace(
            id=4,
            title="Scope test",
            prompt="What applies?",
            reference_answer=None,
            expected_facts=[],
            expected_citations=[],
            as_of_date=None,
            rubric_notes=None,
            tags=[],
            document_set_id=21,
            document_set=SimpleNamespace(name="Tax Regulations"),
            is_active=True,
            created_at=created_at,
            updated_at=created_at,
        ),
    )

    api_snapshot = BenchmarkQuestionSnapshot.from_model(question)
    run_snapshot = _question_snapshot(question)

    assert api_snapshot.document_set_id == 21
    assert api_snapshot.document_set_name == "Tax Regulations"
    assert "project_id" not in api_snapshot.model_dump()
    assert run_snapshot["document_set_id"] == 21
    assert run_snapshot["document_set_name"] == "Tax Regulations"
    assert "project_id" not in run_snapshot


def test_amendment_batch_snapshot_uses_document_set_id() -> None:
    created_at = datetime.datetime.now(datetime.timezone.utc)
    snapshot = AmendmentBatchSnapshot.from_model(
        SimpleNamespace(
            id=8,
            document_set_id=34,
            raw_text="Amending text",
            reference_date=None,
            status="analyzed",
            error_message=None,
            created_by=None,
            created_at=created_at,
            updated_at=created_at,
        )
    )

    assert snapshot.document_set_id == 34
    assert "project_id" not in snapshot.model_dump()


def test_amendment_proposal_snapshot_serializes_grouped_instructions_with_fallback() -> (
    None
):
    created_at = datetime.datetime.now(datetime.timezone.utc)
    proposal_fields = {
        "id": 9,
        "batch_id": 8,
        "instruction_index": 2,
        "instruction_text": "Replace Article 2.",
        "old_chunk_id": "chunk-2",
        "old_chunk_snapshot": {},
        "new_chunk_draft": {"text": "Updated Article 2."},
        "match_confidence": 0.92,
        "match_rationale": "Exact article reference",
        "date_rationale": None,
        "status": "pending",
        "applied_new_chunk_id": None,
        "decided_by": None,
        "decided_at": None,
        "created_at": created_at,
        "updated_at": created_at,
    }

    grouped = AmendmentProposalSnapshot.from_model(
        SimpleNamespace(
            **proposal_fields,
            instruction_indices=[2, 5],
            instruction_texts=["Replace Article 2.", "Add the new exception."],
        )
    )
    historical = AmendmentProposalSnapshot.from_model(
        SimpleNamespace(**proposal_fields)
    )

    assert grouped.instruction_indices == [2, 5]
    assert grouped.instruction_texts == [
        "Replace Article 2.",
        "Add the new exception.",
    ]
    assert historical.instruction_indices == [2]
    assert historical.instruction_texts == ["Replace Article 2."]


def test_benchmark_citation_route_is_document_set_canonical() -> None:
    paths = {
        str(path)
        for route in benchmark_router.routes
        if (path := getattr(route, "path", None)) is not None
    }

    assert (
        "/regulatory/benchmark/document-sets/{document_set_id}/citation-options"
        in paths
    )
    assert "/regulatory/benchmark/projects/{project_id}/citation-options" not in paths
