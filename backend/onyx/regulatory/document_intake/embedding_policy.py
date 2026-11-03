"""Resolve frozen intake budgets without changing legacy embedding contracts."""

from collections.abc import Mapping
from typing import Any

from onyx.regulatory.document_intake.models import IntakePreparationPolicy

BODY_EMBEDDING_TOKEN_LIMIT = 512
TABLE_EMBEDDING_TOKEN_LIMIT = 2048
PREFERRED_CONTEXT_RESERVE = 200


def preparation_policy_for_model(
    provider: str | None, model_name: str | None
) -> IntakePreparationPolicy | None:
    if provider == "google" and model_name == "gemini-embedding-2":
        return IntakePreparationPolicy()
    return None


def embedding_token_limit_for_metadata(
    metadata: Mapping[str, Any],
    chunk_type: str | None,
    *,
    provider: str | None = None,
    model_name: str | None = None,
) -> int:
    if metadata.get("intake_chunking_version") not in ("4", "5"):
        return BODY_EMBEDDING_TOKEN_LIMIT
    policy = IntakePreparationPolicy.model_validate(
        metadata.get("intake_preparation_policy")
    )
    if policy.chunking_version != metadata["intake_chunking_version"]:
        raise ValueError("The frozen intake policy differs from its chunking version")
    if (provider is not None or model_name is not None) and (
        provider != policy.provider or model_name != policy.model_name
    ):
        raise ValueError("The embedding model differs from the frozen intake policy")
    return (
        policy.table_token_limit if chunk_type == "table" else policy.body_token_limit
    )
