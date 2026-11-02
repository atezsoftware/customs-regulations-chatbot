"""Kompozit copies retain their own public identity and durable resume fences."""

from typing import Any
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from pydantic import JsonValue
from sqlalchemy.exc import OperationalError

from onyx.asv3.citation_presentation import citation_presentation_enabled
from onyx.asv3.workflow_variant import (
    ASV3_EXP_CROPPED_VARIANT,
    ASV3_FAST_CROPPED_VARIANT,
    ASV3_FAST_VARIANT,
    ASV3_KOMPOZIT_CROPPED_POLICY,
    ASV3_KOMPOZIT_CROPPED_VARIANT,
    ASV3_KOMPOZIT_POLICY,
    ASV3_KOMPOZIT_VARIANT,
    ASV3_STANDARD_VARIANT,
    checkpoint_variant_fields,
    resolve_asv3_workflow,
    validate_asv3_variant_resume,
)
from onyx.db import asv3_runs
from onyx.db.workflow_usage import empty_workflow_usage

_COPIES = (
    (ASV3_KOMPOZIT_VARIANT, ASV3_KOMPOZIT_POLICY, False),
    (ASV3_KOMPOZIT_CROPPED_VARIANT, ASV3_KOMPOZIT_CROPPED_POLICY, True),
)
_VARIANTS = (ASV3_KOMPOZIT_VARIANT, ASV3_KOMPOZIT_CROPPED_VARIANT)


@pytest.mark.parametrize("variant,policy,cropped", _COPIES)
def test_copy_selection_retains_distinct_identity_and_model_ownership(
    variant: str, policy: str, cropped: bool
) -> None:
    selected = resolve_asv3_workflow(
        "normal", False, kompozit=not cropped, kompozit_cropped=cropped
    )
    assert selected.workflow_variant == variant
    assert selected.research_profile == "normal"
    assert selected.parallel_research is False
    assert selected.selected_model_only
    assert not selected.uses_guardrails
    assert not selected.uses_guardrails_v2
    assert not selected.uses_guardrails_v3
    assert checkpoint_variant_fields(variant) == {
        "asv3_workflow_variant": variant,
        "asv3_workflow_policy": policy,
    }


@pytest.mark.parametrize("cropped", [False, True])
@pytest.mark.parametrize(
    "flags",
    [
        {"fast": True},
        {"fast_cropped": True},
        {"exp_cropped": True},
        {"exp_cheap": True},
        {"guarded_experimental": True},
        {"guardrails_v2": True},
        {"guardrails_v3": True},
    ],
)
def test_copies_cannot_select_another_existing_execution_policy(
    cropped: bool, flags: dict[str, bool]
) -> None:
    with pytest.raises(ValueError, match="combined with other"):
        resolve_asv3_workflow(
            "normal", False, kompozit=not cropped, kompozit_cropped=cropped, **flags
        )


def test_copies_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="combined with other"):
        resolve_asv3_workflow("normal", False, kompozit=True, kompozit_cropped=True)


@pytest.mark.parametrize("cropped", [False, True])
@pytest.mark.parametrize(
    "profile,parallel", [("deep", False), ("experimental", False), ("normal", True)]
)
def test_copy_profile_is_normal_without_parallel_execution(
    cropped: bool, profile: str, parallel: bool
) -> None:
    with pytest.raises(ValueError, match="normal non-parallel"):
        resolve_asv3_workflow(
            profile, parallel, kompozit=not cropped, kompozit_cropped=cropped
        )


@pytest.mark.parametrize("variant", _VARIANTS)
def test_copy_checkpoints_only_resume_with_the_same_variant_and_policy(
    variant: str,
) -> None:
    checkpoint: dict[str, JsonValue] = {
        **checkpoint_variant_fields(variant),
        "research_profile": "normal",
        "parallel_research": False,
    }
    validate_asv3_variant_resume(variant, None)
    validate_asv3_variant_resume(variant, checkpoint)
    other_copy = (
        ASV3_KOMPOZIT_VARIANT
        if variant == ASV3_KOMPOZIT_CROPPED_VARIANT
        else ASV3_KOMPOZIT_CROPPED_VARIANT
    )
    for destination in (
        ASV3_STANDARD_VARIANT,
        ASV3_EXP_CROPPED_VARIANT,
        ASV3_FAST_VARIANT,
        ASV3_FAST_CROPPED_VARIANT,
        other_copy,
    ):
        with pytest.raises(ValueError, match="same workflow variant"):
            validate_asv3_variant_resume(destination, checkpoint)
        with pytest.raises(ValueError, match="same workflow variant"):
            validate_asv3_variant_resume(
                variant,
                {**checkpoint, "asv3_workflow_variant": destination},
            )


@pytest.mark.parametrize("variant", _VARIANTS)
@pytest.mark.parametrize(
    "changed",
    [
        {"asv3_workflow_policy": None},
        {"asv3_workflow_policy": "other-policy"},
        {"research_profile": "experimental"},
        {"parallel_research": 0},
        {"parallel_research": None},
    ],
)
def test_resume_rejects_missing_or_changed_copy_policy(
    variant: str, changed: dict[str, Any]
) -> None:
    with pytest.raises(ValueError, match="own checkpoint policy"):
        validate_asv3_variant_resume(
            variant,
            {
                **checkpoint_variant_fields(variant),
                "research_profile": "normal",
                "parallel_research": False,
                **changed,
            },
        )


@pytest.mark.parametrize("variant", _VARIANTS)
def test_copies_preserve_source_title_enrichment_and_usage_identity(
    variant: str,
) -> None:
    assert citation_presentation_enabled(variant, "normal", False)
    assert not citation_presentation_enabled(variant, "deep", False)
    assert not citation_presentation_enabled(variant, "normal", True)
    usage = empty_workflow_usage()
    assert usage[variant].query_count == 0
    assert usage[variant].cost_status == "complete"
    assert usage[ASV3_KOMPOZIT_VARIANT] is not usage[ASV3_KOMPOZIT_CROPPED_VARIANT]


def test_cropped_copy_round_trips_with_the_cropped_checkpoint_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(asv3_runs, "MAX_DECODED_CHECKPOINT_BYTES", 200)
    monkeypatch.setattr(asv3_runs, "MAX_CROPPED_DECODED_CHECKPOINT_BYTES", 2_000)
    snapshot: dict[str, JsonValue] = {
        **checkpoint_variant_fields(ASV3_KOMPOZIT_CROPPED_VARIANT),
        "run_id": "copy-run",
        "request": "Original source " * 30,
    }
    payload = asv3_runs.encode_asv3_checkpoint(snapshot)
    assert asv3_runs.decode_asv3_checkpoint(payload) == {**snapshot, "version": 1}
    with pytest.raises(ValueError, match="decoded payload budget"):
        asv3_runs.encode_asv3_checkpoint(
            {**snapshot, **checkpoint_variant_fields(ASV3_KOMPOZIT_VARIANT)}
        )
    monkeypatch.setattr(asv3_runs, "MAX_CROPPED_CHECKPOINT_BYTES", 1)
    with pytest.raises(ValueError, match="durable payload budget"):
        asv3_runs.decode_asv3_checkpoint(payload)


def test_cropped_copy_retries_only_transient_checkpoint_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class TransientDatabaseError(Exception):
        pgcode = "40001"

    save = MagicMock(
        side_effect=[
            OperationalError("checkpoint", None, TransientDatabaseError()),
            None,
        ]
    )
    pause = MagicMock()
    monkeypatch.setattr(asv3_runs, "_save_checkpoint_payload", save)
    monkeypatch.setattr(asv3_runs.time, "sleep", pause)
    asv3_runs.save_asv3_checkpoint(
        message_id=10,
        user_id=uuid4(),
        snapshot={
            **checkpoint_variant_fields(ASV3_KOMPOZIT_CROPPED_VARIANT),
            "run_id": "copy-run",
            "sequence": 1,
        },
    )
    assert save.call_count == 2
    pause.assert_called_once_with(asv3_runs._CHECKPOINT_RETRY_DELAYS[0])


@pytest.mark.parametrize("variant", _VARIANTS)
def test_resumed_copy_workflow_is_written_only_to_the_owned_response(
    monkeypatch: pytest.MonkeyPatch, variant: str
) -> None:
    session, message = MagicMock(), MagicMock()
    owner = uuid4()
    owned = MagicMock(return_value=message)
    monkeypatch.setattr(
        asv3_runs, "get_session_with_current_tenant", MagicMock(return_value=session)
    )
    monkeypatch.setattr(asv3_runs, "_require_owned_message", owned)
    asv3_runs.retain_asv3_response_workflow(
        message_id=10, user_id=owner, workflow_variant=variant
    )
    owned.assert_called_once_with(session.__enter__.return_value, 10, owner)
    assert message.workflow == variant
    session.__enter__.return_value.commit.assert_called_once()
    with pytest.raises(ValueError, match="Fast ASv3"):
        asv3_runs.retain_fast_response_workflow(
            message_id=10, user_id=owner, workflow_variant=variant
        )
