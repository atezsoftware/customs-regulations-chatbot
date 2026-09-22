from onyx.regulatory.approval_execution_state import (
    ApprovalExecutionState,
    read_execution_state,
)


def test_failure_state_survives_serialization_and_hides_exception_details() -> None:
    running = ApprovalExecutionState.running("baseline", attempt=2)
    failed = running.failed(ValueError("secret database credential"))
    encoded = failed.model_dump_json()
    assert read_execution_state(encoded) == failed
    assert "secret" not in encoded
    assert failed.code == "baseline_integrity"
    assert failed.retryable is False
    assert failed.attempt == 2


def test_old_messages_are_not_misread_as_typed_execution_state() -> None:
    assert read_execution_state("Indexing failed") is None
    assert read_execution_state(None) is None
    assert read_execution_state('{"message":"arbitrary"}') is None


def test_provider_failure_keeps_resumable_stage() -> None:
    failed = ApprovalExecutionState.running("context", attempt=1).failed(TimeoutError())
    assert failed.stage == "context"
    assert failed.retryable
    assert "Review" not in failed.message
