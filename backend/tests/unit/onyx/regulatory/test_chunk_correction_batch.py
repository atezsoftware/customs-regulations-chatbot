"""A same-file correction is preflighted and published once without version edits."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import date, datetime, timezone
from threading import Event
from typing import cast
from unittest.mock import MagicMock, Mock
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sqlalchemy.orm import Session

from onyx.auth.schemas import UserRole
from onyx.auth.users import current_user
from onyx.db.engine.sql_engine import get_session
from onyx.db.enums import Permission, UserFileStatus
from onyx.db.models import RegulatoryChunk, User
from onyx.db.regulatory_chunk_edits import ChunkEditConflict
from onyx.db.regulatory_publication import PublicationOwnershipLost
from onyx.error_handling.exceptions import register_onyx_exception_handlers
from onyx.regulatory import writer_projection
from onyx.regulatory import writer_publication as writer
from onyx.regulatory.amendments.annexes.models import AnnexCanonicalSnapshot
from onyx.regulatory.chunk_corrections import ChunkCorrectionBatch
from onyx.regulatory.document_intake.embedding import (
    FrozenEmbeddingGroup,
    encode_intake_embedding_groups,
)
from onyx.regulatory.writer_publication_models import WriterPublicationManifest
from onyx.server.features.regulatory import api
from tests.unit.onyx.regulatory.indexing_jobs.owned_publication_test_helpers import (
    OwnedAuthority,
    canonical_row,
    writer_inputs,
)
from tests.unit.onyx.regulatory.test_writer_context_generation_reuse import (
    WriterContextHarness,
)

NOW = datetime(2026, 11, 4, tzinfo=timezone.utc)


def body() -> dict[str, object]:
    return {
        "updates": [
            {
                "chunk_id": "row-0",
                "expected_updated_at": NOW.isoformat(),
                "text": "Source-faithful correction",
            }
        ]
    }


@pytest.mark.parametrize(
    "change",
    [
        "empty",
        "too_many",
        "duplicate",
        "timestamp",
        "naive",
        "date",
        "status",
        "source",
        "id",
        "blank",
        "null",
        "no_content",
    ],
)
def test_batch_rejects_ambiguous_or_version_changing_requests(change: str) -> None:
    payload = body()
    updates = cast(list[dict[str, object]], payload["updates"])
    if change == "empty":
        payload["updates"] = []
    elif change == "too_many":
        payload["updates"] = [
            {**updates[0], "chunk_id": str(index)} for index in range(33)
        ]
    elif change == "duplicate":
        updates.append(dict(updates[0]))
    elif change == "timestamp":
        del updates[0]["expected_updated_at"]
    elif change == "naive":
        updates[0]["expected_updated_at"] = "2026-11-04T00:00:00"
    elif change == "date":
        updates[0]["validity_start_date"] = "2027-01-01"
    elif change in {"status", "source", "id"}:
        updates[0][change] = "replacement"
    elif change == "blank":
        updates[0]["text"] = "  \n"
    elif change == "null":
        updates[0]["chunk_metadata"] = None
    elif change == "no_content":
        del updates[0]["text"]
    with pytest.raises(ValidationError):
        ChunkCorrectionBatch.model_validate(payload)


def test_maximum_distinct_batch_and_full_metadata_are_valid() -> None:
    assert (
        len(
            ChunkCorrectionBatch.model_validate(
                {
                    "updates": [
                        {
                            "chunk_id": str(index),
                            "expected_updated_at": NOW,
                            "chunk_metadata": {},
                        }
                        for index in range(32)
                    ]
                }
            ).updates
        )
        == 32
    )


def test_one_stale_target_stops_everything_before_recovery_or_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preflight = Mock(side_effect=ChunkEditConflict("changed"))
    recover, load, deferred, prepare, publish = [Mock() for _ in range(5)]
    monkeypatch.setattr(
        "onyx.db.regulatory_chunk_edits.require_owned_chunk_revisions", preflight
    )
    monkeypatch.setattr(writer, "recover_owned_writer_before_next", recover)
    monkeypatch.setattr(
        "onyx.db.regulatory_writer_publication.load_owned_writer_inputs", load
    )
    monkeypatch.setattr(
        "onyx.db.regulatory_writer_publication.apply_owned_deferred_edit", deferred
    )
    monkeypatch.setattr(
        "onyx.regulatory.writer_projection.prepare_owned_correction", prepare
    )
    monkeypatch.setattr(writer, "execute_writer_publication", publish)
    with pytest.raises(ChunkEditConflict):
        writer.correct_owned_chunks(
            OwnedAuthority(uuid4()).owner,
            MagicMock(),
            ChunkCorrectionBatch.model_validate(body()),
        )
    for operation in (recover, load, deferred, prepare, publish):
        operation.assert_not_called()


@pytest.mark.parametrize("failure", [None, "heartbeat", "reservation", "encoder"])
def test_multirow_writer_preserves_all_other_fields_and_guards_real_encoder(
    monkeypatch: pytest.MonkeyPatch, failure: str | None
) -> None:
    authority = OwnedAuthority(uuid4())
    rows = [
        canonical_row(authority.owner.user_file_id, index, f"Source {index}")
        for index in range(3)
    ]
    rows[1].validity_start_date, rows[1].validity_end_date = (
        date(2022, 1, 1),
        date(2026, 10, 17),
    )
    rows[1].status = "superseded"
    rows[1].superseded_by_chunk_id = "current-version"
    inputs = writer_inputs(authority.owner.user_file_id, rows)
    before = [row.model_dump(mode="json") for row in inputs.canonical]
    corrections = ChunkCorrectionBatch.model_validate(
        {
            "updates": [
                {
                    "chunk_id": "row-0",
                    "expected_updated_at": NOW,
                    "text": "Corrected source",
                    "chunk_metadata": {"image_file_id": "stored-page"},
                },
                {
                    "chunk_id": "row-1",
                    "expected_updated_at": NOW,
                    "heading_path": ["Original chapter heading"],
                },
            ]
        }
    )
    lost = Event()

    @contextmanager
    def heartbeat(_owner: object) -> Iterator[Event]:
        yield lost

    store = MagicMock()
    reservation_checks = 0

    def reservations(_owner: object) -> None:
        nonlocal reservation_checks
        reservation_checks += 1
        if failure == "reservation" and reservation_checks >= 3:
            raise PublicationOwnershipLost("reservation changed")

    store.reservations.side_effect = reservations
    encode = Mock(return_value=[[0.1, 0.2, 0.3]])
    if failure == "encoder":
        encode.side_effect = RuntimeError("encoder failed")
    prepared = Mock()

    def prepare(*args: object, **kwargs: object) -> Mock:
        after = cast(list[AnnexCanonicalSnapshot], args[3])
        assert kwargs["changed_id"] is None
        dumped = [row.model_dump(mode="json") for row in after]
        assert dumped[0] == {
            **before[0],
            "text": "Corrected source",
            "metadata": {"image_file_id": "stored-page"},
        }
        assert dumped[1] == {**before[1], "heading_path": ["Original chapter heading"]}
        assert dumped[2] == before[2]
        if failure == "heartbeat":
            lost.set()
        encode_intake_embedding_groups(
            [
                FrozenEmbeddingGroup(str(index), (f"chunk {index}",))
                for index in range(3)
            ],
            encode=encode,
            expected_dimension=3,
            check_owner=cast(Callable[[], None], kwargs["before_embedding_request"]),
        )
        return prepared

    preflight, load, deferred, publish = (
        Mock(),
        Mock(return_value=inputs),
        Mock(return_value=False),
        Mock(),
    )
    monkeypatch.setattr(
        "onyx.db.regulatory_chunk_edits.require_owned_chunk_revisions", preflight
    )
    monkeypatch.setattr(writer, "recover_owned_writer_before_next", lambda owner: owner)
    monkeypatch.setattr(writer, "PublicationStore", lambda _scope: store)
    monkeypatch.setattr(writer, "publication_heartbeat", heartbeat)
    monkeypatch.setattr(
        "onyx.db.regulatory_writer_publication.load_owned_writer_inputs", load
    )
    monkeypatch.setattr(
        "onyx.db.regulatory_writer_publication.apply_owned_deferred_edit", deferred
    )
    prepare_spy = Mock(side_effect=prepare)
    monkeypatch.setattr(
        "onyx.regulatory.writer_projection.prepare_owned_correction", prepare_spy
    )
    monkeypatch.setattr(writer, "execute_writer_publication", publish)
    if failure is None:
        assert (
            writer.correct_owned_chunks(authority.owner, MagicMock(), corrections)
            == authority.owner
        )
        assert publish.call_count == 1 and publish.call_args.args[2] == prepared
        assert encode.call_count == 3 and reservation_checks >= 3
    else:
        with pytest.raises((PublicationOwnershipLost, RuntimeError)):
            writer.correct_owned_chunks(authority.owner, MagicMock(), corrections)
        publish.assert_not_called()
    load.assert_called_once()
    prepare_spy.assert_called_once()
    assert [row.model_dump(mode="json") for row in inputs.canonical] == before


def test_actual_multiwindow_projection_uses_guarded_encoder_and_one_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = WriterContextHarness(monkeypatch)
    before = list(harness.inputs.canonical)
    request = ChunkCorrectionBatch.model_validate(
        {
            "updates": [
                {
                    "chunk_id": before[0].id,
                    "expected_updated_at": NOW,
                    "text": "Correct first source paragraph",
                },
                {
                    "chunk_id": before[1].id,
                    "expected_updated_at": NOW,
                    "heading_path": ["Correct second source heading"],
                },
            ]
        }
    )

    @contextmanager
    def heartbeat(_owner: object) -> Iterator[Event]:
        yield Event()

    authority = MagicMock()
    monkeypatch.setattr(
        "onyx.db.regulatory_chunk_edits.require_owned_chunk_revisions", Mock()
    )
    monkeypatch.setattr(writer, "recover_owned_writer_before_next", lambda owner: owner)
    monkeypatch.setattr(writer, "PublicationStore", lambda _scope: authority)
    monkeypatch.setattr(writer, "publication_heartbeat", heartbeat)
    monkeypatch.setattr(
        "onyx.db.regulatory_writer_publication.load_owned_writer_inputs",
        Mock(return_value=harness.inputs),
    )
    monkeypatch.setattr(
        "onyx.db.regulatory_writer_publication.apply_owned_deferred_edit",
        Mock(return_value=False),
    )
    vectors = Mock(wraps=writer_projection._intake_view_vectors)
    monkeypatch.setattr(writer_projection, "_intake_view_vectors", vectors)
    publish = Mock()
    monkeypatch.setattr(writer, "execute_writer_publication", publish)
    writer.correct_owned_chunks(harness.authority.owner, harness.client, request)
    publish.assert_called_once()
    manifest = cast(WriterPublicationManifest, publish.call_args.args[2])
    assert len(manifest.bindings) == 12 and len(manifest.views) == 4
    assert manifest.canonical_after == [
        before[0].model_copy(update={"text": "Correct first source paragraph"}),
        before[1].model_copy(
            update={"heading_path": ["Correct second source heading"]}
        ),
        *before[2:],
    ]
    assert vectors.call_count == 4
    assert authority.reservations.call_count > 2


@pytest.mark.parametrize(
    "boundary", ["ok", "permission", "processing", "intake", "conflict"]
)
def test_batch_http_boundary_and_single_lease(
    monkeypatch: pytest.MonkeyPatch, boundary: str
) -> None:
    authority = OwnedAuthority(uuid4())
    user = User(
        id=uuid4(),
        role=UserRole.ADMIN,
        effective_permissions=[]
        if boundary == "permission"
        else [Permission.FULL_ADMIN_PANEL_ACCESS.value],
    )
    row: RegulatoryChunk = canonical_row(
        authority.owner.user_file_id, 0, "Corrected source"
    )
    row.created_at = row.updated_at = NOW
    session = MagicMock(spec=Session)
    session.get.return_value = Mock(
        user_id=user.id,
        status=UserFileStatus.PROCESSING
        if boundary == "processing"
        else UserFileStatus.COMPLETED,
    )
    release, acquire = Mock(), Mock(return_value=authority.owner)
    store = Mock(acquire=acquire, release=release)
    correct = Mock(return_value=authority.owner)
    if boundary == "conflict":
        correct.side_effect = ChunkEditConflict("stale target")
    monkeypatch.setattr(api, "get_current_tenant_id", lambda: "tenant-a")
    monkeypatch.setattr(
        api,
        "pending_intake_file_ids",
        lambda *_args, **_kwargs: [row.user_file_id] if boundary == "intake" else [],
    )
    monkeypatch.setattr(api, "get_chunk_snapshot_by_id", lambda *_args: row)
    monkeypatch.setattr(
        "onyx.db.regulatory_publication.PublicationStore", lambda _scope: store
    )
    monkeypatch.setattr(writer, "correct_owned_chunks", correct)
    monkeypatch.setattr(
        "onyx.document_index.elasticsearch.client.ElasticsearchClient", MagicMock()
    )
    app = FastAPI()
    register_onyx_exception_handlers(app)
    app.include_router(api.router)
    app.dependency_overrides[current_user] = lambda: user
    app.dependency_overrides[get_session] = lambda: session
    with TestClient(app) as client:
        response = client.patch(
            f"/regulatory/files/{row.user_file_id}/chunks", json=body()
        )
    assert (
        response.status_code
        == {
            "ok": 200,
            "permission": 403,
            "processing": 409,
            "intake": 409,
            "conflict": 409,
        }[boundary]
    ), response.text
    if boundary in {"permission", "processing", "intake"}:
        acquire.assert_not_called()
        correct.assert_not_called()
    else:
        acquire.assert_called_once()
        release.assert_called_once_with(authority.owner)
        if boundary == "ok":
            assert response.json()[0]["id"] == row.id
