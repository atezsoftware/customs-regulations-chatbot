"""Larger source tables must preserve rows without changing prose or retries."""

import pytest

from onyx.regulatory.document_intake.embedding_policy import (
    embedding_token_limit_for_metadata,
    preparation_policy_for_model,
)
from onyx.regulatory.document_intake.models import (
    IntakePreparationPolicy,
    IntakeSourceContext,
)
from onyx.regulatory.document_intake.versions import reprepare_target_versions
from tests.unit.onyx.regulatory.whole_table_policy_helpers import _context


def test_legacy_source_serialization_and_retry_targets_remain_unchanged() -> None:
    legacy = _context().model_dump(mode="json")
    assert "preparation_policy" not in legacy
    assert legacy["chunking_version"] == "3"
    assert reprepare_target_versions({}) == ("2", 3)
    assert reprepare_target_versions(
        {"target_chunking_version": "3", "target_formatter_version": 4}
    ) == ("3", 4)
    with pytest.raises(ValueError, match="frozen preparation policy"):
        IntakeSourceContext.model_validate({**legacy, "chunking_version": "4"})
    with pytest.raises(ValueError):
        reprepare_target_versions(
            {"target_chunking_version": "4", "target_formatter_version": 4}
        )
    policy = IntakePreparationPolicy().model_dump(mode="json")
    assert reprepare_target_versions(
        {
            "target_chunking_version": "4",
            "target_formatter_version": 4,
            "target_preparation_policy": policy,
        }
    ) == ("4", 4)


def test_policy_budget_requires_frozen_identity_and_matching_provider() -> None:
    policy = IntakePreparationPolicy().model_dump(mode="json")
    metadata = {"intake_chunking_version": "4", "intake_preparation_policy": policy}
    assert (
        embedding_token_limit_for_metadata(
            metadata, "table", provider="google", model_name="gemini-embedding-2"
        )
        == 2048
    )
    assert embedding_token_limit_for_metadata(metadata, "paragraph") == 512
    assert embedding_token_limit_for_metadata({}, "table") == 512
    with pytest.raises(ValueError, match="differs"):
        embedding_token_limit_for_metadata(
            metadata, "table", provider="google", model_name="gemini-embedding-001"
        )
    with pytest.raises(ValueError):
        embedding_token_limit_for_metadata({"intake_chunking_version": "4"}, "table")
    assert preparation_policy_for_model(
        "google", "gemini-embedding-2"
    ) == IntakePreparationPolicy(chunking_version="5")


def test_v5_new_target_is_explicit_and_old_frozen_v4_target_stays_v4() -> None:
    new = preparation_policy_for_model("google", "gemini-embedding-2")
    assert new is not None and new.chunking_version == "5"
    assert reprepare_target_versions(
        {
            "target_chunking_version": "5",
            "target_formatter_version": 4,
            "target_preparation_policy": new.model_dump(mode="json"),
        }
    ) == ("5", 4)
    with pytest.raises(ValueError, match="differ"):
        reprepare_target_versions(
            {
                "target_chunking_version": "4",
                "target_formatter_version": 4,
                "target_preparation_policy": new.model_dump(mode="json"),
            }
        )
