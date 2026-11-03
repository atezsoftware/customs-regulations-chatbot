"""An explicit Retry upgrades only a byte-identical frozen request authority."""

from copy import deepcopy
from typing import Any
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from sqlalchemy.orm import Session

from onyx.db import document_intake as intake
from onyx.db import document_intake_metadata_checkpoint as checkpoints
from onyx.db import labeling_configuration
from onyx.db.models import RegulatoryDocumentIntake
from onyx.regulatory.document_intake import metadata_job
from onyx.regulatory.document_intake.metadata_checkpoint import MetadataCheckpoint
from onyx.regulatory.indexing_jobs.models import VertexAuthenticationMode


@pytest.mark.parametrize(
    "change",
    [
        "none",
        "source",
        "request",
        "provider",
        "active_lease",
        "execution_owned",
        "bad_checksum",
        "invalid_receipt",
    ],
)
def test_explicit_recipe_retry_preserves_exact_source_request_and_owner(
    monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    binding = labeling_configuration.LabelingProviderBinding(
        model_configuration_id=1,
        provider_id=2,
        project="metadata-test",
        location="global",
        authentication_mode=VertexAuthenticationMode.WORKLOAD_IDENTITY,
        credential_identity="workload_identity",
        model="gemini-3.5-flash",
        transport="vertex_direct_v1",
    )
    source = {"documents": {"doc": {"input_sha256": "c" * 64}}}
    requests = {"doc": ["d" * 64]}
    saved = MetadataCheckpoint.build(
        recipe_sha256="a" * 64,
        provider_binding=binding.model_dump(mode="json"),
        source_snapshot=deepcopy(source),
        request_hashes=deepcopy(requests),
    )
    batch = RegulatoryDocumentIntake(
        id=uuid4(),
        created_by=uuid4(),
        document_set_id=15,
        labeling_model_configuration_id=1,
        status="failed",
        stage="metadata",
        metadata_checkpoint=saved.model_dump(mode="json"),
    )
    if change == "source":
        source["documents"]["doc"]["input_sha256"] = "e" * 64
    elif change == "request":
        requests["doc"] = ["e" * 64]
    elif change == "provider":
        binding = binding.model_copy(update={"model": "different-model"})
    elif change == "active_lease":
        batch.lease_token = uuid4()
    elif change == "bad_checksum":
        assert batch.metadata_checkpoint is not None
        batch.metadata_checkpoint["recipe_sha256"] = "e" * 64
    original = deepcopy(batch.metadata_checkpoint)
    session = MagicMock(spec=Session)

    def snapshot(
        *_args: object,
    ) -> tuple[dict[str, Any], dict[str, list[str]], set[str]]:
        if change == "invalid_receipt":
            raise ValueError(
                "Metadata paragraph locator contradicts its source evidence"
            )
        return source, requests, set()

    monkeypatch.setattr(checkpoints, "_source_snapshot", snapshot)
    monkeypatch.setattr(
        checkpoints,
        "try_reserve_metadata_execution",
        lambda *_args: change != "execution_owned",
    )
    monkeypatch.setattr(
        labeling_configuration,
        "resolve_intake_labeling_provider_binding",
        lambda *_args, **_kwargs: binding,
    )
    monkeypatch.setattr(metadata_job, "metadata_recipe_sha256", lambda: "b" * 64)
    if change != "none":
        with pytest.raises(intake.IntakeConflictError):
            checkpoints.rebase_metadata_checkpoint_for_retry(session, batch)
        assert batch.metadata_checkpoint == original
        return

    with pytest.raises(intake.IntakeConflictError, match="contract changed"):
        checkpoints._validated_checkpoint(
            session, batch, binding, "b" * 64, capture=True
        )
    assert batch.metadata_checkpoint == original
    checkpoints.rebase_metadata_checkpoint_for_retry(session, batch)
    updated = MetadataCheckpoint.model_validate(batch.metadata_checkpoint)
    updated.validate_checksum()
    assert updated.recipe_sha256 == "b" * 64
    assert updated.source_snapshot == saved.source_snapshot
    assert updated.provider_binding == saved.provider_binding
    assert updated.request_hashes == saved.request_hashes
    assert updated.predecessor is not None
    assert updated.predecessor.checkpoint_sha256 == saved.checkpoint_sha256
    assert updated.predecessor.version == saved.version
    assert updated.predecessor.recipe_sha256 == saved.recipe_sha256
    before_repeat = deepcopy(batch.metadata_checkpoint)
    checkpoints.rebase_metadata_checkpoint_for_retry(session, batch)
    assert batch.metadata_checkpoint == before_repeat
    assert batch.status == "failed" and batch.stage == "metadata"
    session.execute.assert_not_called()


def test_existing_checkpoint_checksum_remains_compatible_without_predecessor() -> None:
    saved = MetadataCheckpoint.build(
        recipe_sha256="a" * 64,
        provider_binding={},
        source_snapshot={},
        request_hashes={},
    )
    assert "predecessor" not in saved.model_dump(mode="json")
    MetadataCheckpoint.model_validate(saved.model_dump(mode="json")).validate_checksum()
