"""Fast copies retain distinct routes, checkpoint ownership and source presentation."""

from typing import Any

import pytest
from pydantic import JsonValue, ValidationError

from onyx.asv3.citation_presentation import citation_presentation_enabled
from onyx.asv3.workflow_variant import (
    ASV3_EXP_CROPPED_VARIANT,
    ASV3_FAST_CROPPED_VARIANT,
    ASV3_FAST_VARIANT,
    ASV3_STANDARD_VARIANT,
    checkpoint_variant_fields,
    resolve_asv3_workflow,
    validate_asv3_variant_resume,
)
from onyx.chat.response_workflow import response_workflow
from onyx.db import asv3_runs
from onyx.server.query_and_chat.models import SendMessageRequest


@pytest.mark.parametrize(
    ("flag", "variant", "selected_model_only"),
    [
        ("fast", ASV3_FAST_VARIANT, False),
        ("fast_cropped", ASV3_FAST_CROPPED_VARIANT, True),
    ],
)
def test_fast_selection_preserves_base_model_policy_and_distinct_identity(
    flag: str, variant: str, selected_model_only: bool
) -> None:
    selection = resolve_asv3_workflow("normal", False, **{flag: True})
    assert selection.workflow_variant == variant
    assert selection.selected_model_only is selected_model_only
    assert citation_presentation_enabled(variant, "normal", False)
    assert not citation_presentation_enabled(variant, "deep", False)
    assert not citation_presentation_enabled(variant, "normal", True)
    assert (
        resolve_asv3_workflow("normal", False).workflow_variant == ASV3_STANDARD_VARIANT
    )
    assert (
        resolve_asv3_workflow("normal", False, exp_cropped=True).workflow_variant
        == ASV3_EXP_CROPPED_VARIANT
    )


@pytest.mark.parametrize("flag", ["asv3_fast", "asv3_fast_cropped"])
def test_api_records_the_exact_fast_copy(flag: str) -> None:
    request = SendMessageRequest.model_validate(
        {
            "message": "Soru",
            "atez_search_v3": True,
            "asv3_research_profile": "normal",
            flag: True,
        }
    )
    assert response_workflow(request) == flag


@pytest.mark.parametrize("flag", ["asv3_fast", "asv3_fast_cropped"])
@pytest.mark.parametrize(
    "conflict",
    [
        {"atez_search_v3": False},
        {"asv3_research_profile": "deep"},
        {"asv3_parallel_research": True},
        {"asv3_exp_cropped": True},
        {"asv3_guardrails_v3": True},
        {"legal_review": True},
        {"supersearch": True},
        {"legal_composite": True},
        {"deep_research": True},
    ],
)
def test_api_rejects_mixed_or_unowned_fast_modes(
    flag: str, conflict: dict[str, Any]
) -> None:
    with pytest.raises(ValidationError):
        SendMessageRequest.model_validate(
            {
                "message": "Soru",
                "atez_search_v3": True,
                "asv3_research_profile": "normal",
                flag: True,
                **conflict,
            }
        )


def test_fast_copies_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="combined"):
        resolve_asv3_workflow("normal", False, fast=True, fast_cropped=True)
    with pytest.raises(ValidationError):
        SendMessageRequest.model_validate(
            {
                "message": "Soru",
                "atez_search_v3": True,
                "asv3_research_profile": "normal",
                "asv3_fast": True,
                "asv3_fast_cropped": True,
            }
        )


@pytest.mark.parametrize("variant", [ASV3_FAST_VARIANT, ASV3_FAST_CROPPED_VARIANT])
def test_checkpoint_roundtrip_and_policy_fence(variant: str) -> None:
    snapshot: dict[str, JsonValue] = {
        **checkpoint_variant_fields(variant),
        "research_profile": "normal",
        "parallel_research": False,
        "run_id": "fast-run",
        "sequence": 7,
        "original_text": "Özgün kaynak, şart ve istisna.",
    }
    restored = asv3_runs.decode_asv3_checkpoint(
        asv3_runs.encode_asv3_checkpoint(snapshot)
    )
    assert restored == {**snapshot, "version": 1}
    validate_asv3_variant_resume(variant, restored)
    for other in (
        ASV3_STANDARD_VARIANT,
        ASV3_EXP_CROPPED_VARIANT,
        ASV3_FAST_VARIANT,
        ASV3_FAST_CROPPED_VARIANT,
    ):
        if other != variant:
            with pytest.raises(ValueError, match="same workflow"):
                validate_asv3_variant_resume(other, restored)
    with pytest.raises(ValueError, match="own checkpoint policy"):
        validate_asv3_variant_resume(
            variant, {**restored, "asv3_workflow_policy": "other"}
        )


def test_fast_checkpoint_capacity_matches_its_base() -> None:
    assert asv3_runs._checkpoint_limits(
        checkpoint_variant_fields(ASV3_FAST_VARIANT)
    ) == (
        asv3_runs.MAX_CHECKPOINT_BYTES,
        asv3_runs.MAX_DECODED_CHECKPOINT_BYTES,
    )
    assert asv3_runs._checkpoint_limits(
        checkpoint_variant_fields(ASV3_FAST_CROPPED_VARIANT)
    ) == (
        asv3_runs.MAX_CROPPED_CHECKPOINT_BYTES,
        asv3_runs.MAX_CROPPED_DECODED_CHECKPOINT_BYTES,
    )


@pytest.mark.parametrize("authorized", [True, False])
def test_resumed_fast_label_is_written_only_to_an_owned_response(
    monkeypatch: pytest.MonkeyPatch, authorized: bool
) -> None:
    from contextlib import nullcontext
    from types import SimpleNamespace
    from unittest.mock import MagicMock
    from uuid import uuid4

    message = SimpleNamespace(workflow="asv3_deep")
    session = MagicMock()
    session.scalar.return_value = message if authorized else None
    monkeypatch.setattr(
        asv3_runs, "get_session_with_current_tenant", lambda: nullcontext(session)
    )
    if not authorized:
        with pytest.raises(PermissionError, match="outside"):
            asv3_runs.retain_fast_response_workflow(
                message_id=42, user_id=uuid4(), workflow_variant=ASV3_FAST_VARIANT
            )
        session.commit.assert_not_called()
        assert message.workflow == "asv3_deep"
    else:
        asv3_runs.retain_fast_response_workflow(
            message_id=42, user_id=uuid4(), workflow_variant=ASV3_FAST_VARIANT
        )
        assert message.workflow == ASV3_FAST_VARIANT
        session.commit.assert_called_once()
