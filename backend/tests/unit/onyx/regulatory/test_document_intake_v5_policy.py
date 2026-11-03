"""New logical table rows retain frozen budgets and historical generation bytes."""

from types import SimpleNamespace
from typing import cast
from unittest.mock import patch
from uuid import uuid4

import pytest
from sqlalchemy.orm import Session

from onyx.db.models import RegulatoryChunk, RegulatoryIndexingJob
from onyx.regulatory.amendments.annexes.context_dependencies import (
    freeze_encoder_inputs,
)
from onyx.regulatory.document_intake.embedding_policy import (
    embedding_token_limit_for_metadata,
)
from onyx.regulatory.document_intake.encoder_budget import row_encoder_options
from onyx.regulatory.document_intake.models import (
    IntakePreparationPolicy,
    IntakeSourceContext,
)
from onyx.regulatory.indexing_jobs import configuration
from onyx.regulatory.indexing_jobs.contextual import (
    ContextualMappingError,
    ContextualRequestFactory,
)
from onyx.regulatory.indexing_jobs.embedding_receipts import (
    synchronous_embedding_receipts,
)
from onyx.regulatory.indexing_jobs.models import RegulatoryIndexingConfigSnapshot
from shared_configs.enums import EmbeddingProvider, EmbedTextType
from tests.unit.onyx.regulatory.indexing_jobs.test_configuration import (
    _install_admin_configuration,
)
from tests.unit.onyx.regulatory.whole_table_encoder_helpers import _model, _table
from tests.unit.onyx.regulatory.whole_table_policy_helpers import _context


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        (None, "d83d932ce22546ed811c6aab4146d8ee9b6c911a4ba37d5b44618fdacd1927c2"),
        ("1", "630ae2f841c57c08aff6b90480428a1a2313624e9c31da7e1bc35510d580a770"),
        ("2", "f312687a910fc9c028363f311b0a858d65e7c76f6d31399ff24945120b1cf3b6"),
        ("3", "588a23c1c67dcc0812976620e19465466d224b7b281a18d7b9ee6bdc2fa0158a"),
        ("4", "7a84a9687eeac7c04b2424657803767a50b8f9d297f499d6058038639709e36b"),
    ],
)
def test_existing_generation_hashes_are_byte_stable(
    version: str | None, expected: str
) -> None:
    assert (
        configuration.compute_regulatory_chunk_generation_hash(
            embedding_provider=EmbeddingProvider.GOOGLE,
            embedding_model_name="gemini-embedding-2",
            intake_chunking_version=version,
            intake_preparation_policy=IntakePreparationPolicy()
            if version == "4"
            else None,
        )
        == expected
    )
    assert IntakePreparationPolicy().model_dump(mode="json") == {
        "version": 1,
        "chunking_version": "4",
        "provider": "google",
        "model_name": "gemini-embedding-2",
        "body_token_limit": 512,
        "table_token_limit": 2048,
    }


@pytest.mark.parametrize("version", ["4", "5"])
def test_source_metadata_and_frozen_hash_require_the_same_policy_version(
    version: str,
) -> None:
    policy = IntakePreparationPolicy.model_validate({"chunking_version": version})
    context = {
        **_context().model_dump(mode="json"),
        "chunking_version": version,
        "preparation_policy": policy.model_dump(mode="json"),
    }
    assert IntakeSourceContext.model_validate(context).preparation_policy == policy
    metadata = {
        "intake_chunking_version": version,
        "intake_preparation_policy": policy.model_dump(mode="json"),
    }
    assert embedding_token_limit_for_metadata(metadata, "table") == 2048
    assert embedding_token_limit_for_metadata(metadata, "paragraph") == 512
    other = "4" if version == "5" else "5"
    with pytest.raises(ValueError, match="version"):
        IntakeSourceContext.model_validate({**context, "chunking_version": other})
    with pytest.raises(ValueError, match="version"):
        embedding_token_limit_for_metadata(
            {**metadata, "intake_chunking_version": other}, "table"
        )
    with pytest.raises(ValueError, match="version"):
        configuration.compute_regulatory_chunk_generation_hash(
            embedding_provider=EmbeddingProvider.GOOGLE,
            embedding_model_name="gemini-embedding-2",
            intake_chunking_version=other,
            intake_preparation_policy=policy,
        )


def test_explicit_frozen_profile_snapshots_roundtrip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_admin_configuration(monkeypatch)
    old = configuration.resolve_regulatory_indexing_snapshot(
        cast(Session, SimpleNamespace())
    )
    assert "intake_preparation_policy" not in old.model_dump(mode="json")
    for version in ("4", "5"):
        policy = IntakePreparationPolicy.model_validate({"chunking_version": version})
        payload = old.model_dump(mode="json") | {
            "embedding_provider": "google",
            "embedding_model_name": "gemini-embedding-2",
            "intake_chunking_version": version,
            "intake_preparation_policy": policy.model_dump(mode="json"),
            "openrouter_batch": None,
        }
        assert (
            RegulatoryIndexingConfigSnapshot.model_validate(payload).model_dump(
                mode="json"
            )
            == payload
        )
        with pytest.raises(ValueError, match="policy"):
            RegulatoryIndexingConfigSnapshot.model_validate(
                payload | {"intake_chunking_version": "4" if version == "5" else "5"}
            )


@pytest.mark.parametrize("version", ["4", "5"])
def test_scoped_encoder_keeps_full_tail_and_generic_durable_paths_reject(
    version: str,
) -> None:
    model = _model()
    policy = IntakePreparationPolicy.model_validate({"chunking_version": version})
    row = RegulatoryChunk(
        id="table",
        user_file_id=uuid4(),
        text=_table(),
        chunk_type="table",
        chunk_metadata={
            "intake_chunking_version": version,
            "intake_preparation_policy": policy.model_dump(mode="json"),
        },
    )
    options = row_encoder_options(row, model)
    assert options == {"max_seq_length": 2048, "strict_full_input": True}
    texts, _ = freeze_encoder_inputs(
        [row.text], model, model_dim=3, formatter="normal-v1", **options
    )
    assert texts == [row.text] and "SON_SATIR_KORUNDU" in texts[0]
    with patch.object(
        model, "_batch_encode_texts", return_value=[[1.0, 0.0, 0.0]]
    ) as send:
        model.encode(texts, text_type=EmbedTextType.PASSAGE, **options)
    assert (
        send.call_args.kwargs["texts"] == texts
        and send.call_args.kwargs["strict_full_input"] is True
    )
    job = RegulatoryIndexingJob(
        user_file_id=row.user_file_id, config_snapshot={"effective_dimension": 3}
    )
    with pytest.raises(ContextualMappingError, match="normal writer"):
        ContextualRequestFactory(job, [row], model.tokenizer)
    with pytest.raises(ValueError, match="normal writer"):
        synchronous_embedding_receipts(job=job, rows=[row], items=[], model=model)
