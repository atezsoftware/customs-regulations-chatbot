"""Safe durable approval diagnostics, stored compatibly in the existing text column."""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError

ApprovalStage = Literal["baseline", "review", "context", "publication"]


class ApprovalExecutionState(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["approval-execution-v1"] = "approval-execution-v1"
    stage: ApprovalStage
    state: Literal["running", "failed"]
    code: str | None = None
    retryable: bool = True
    attempt: int = Field(ge=1)
    message: str
    manifest_sha256: str | None = None

    @classmethod
    def running(cls, stage: ApprovalStage, *, attempt: int) -> Self:
        return cls(
            stage=stage,
            state="running",
            attempt=attempt,
            message={
                "baseline": "Preparing existing index records. Embeddings are being preserved.",
                "review": "Checking the approved proposal against the current source version.",
                "context": "Checking affected context and preparing changed chunks.",
                "publication": "Publishing and verifying the updated index records.",
            }[stage],
        )

    def failed(self, error: Exception, *, manifest_sha256: str | None = None) -> Self:
        integrity = self.stage == "baseline" and isinstance(error, ValueError)
        conflict = self.stage == "review" and isinstance(error, ValueError)
        return self.model_copy(
            update={
                "state": "failed",
                "code": "baseline_integrity"
                if integrity
                else "proposal_conflict"
                if conflict
                else f"{self.stage}_interrupted",
                "retryable": manifest_sha256 is not None or not (integrity or conflict),
                "manifest_sha256": manifest_sha256,
                "message": (
                    "Publication was interrupted. The saved operation will resume."
                    if manifest_sha256 is not None
                    else "Existing index records could not be verified. Source and index data were preserved; the affected records need reconciliation."
                    if integrity
                    else "The proposal no longer matches the current source version. Review its target and changes."
                    if conflict
                    else f"Approval was interrupted during {self.stage}. Retry to continue; existing records are preserved."
                ),
            }
        )


def read_execution_state(value: str | None) -> ApprovalExecutionState | None:
    if not value:
        return None
    try:
        return ApprovalExecutionState.model_validate_json(value)
    except ValidationError:
        return None
