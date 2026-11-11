"""A valid empty provider classification is a successful optional-label result."""

from hashlib import sha256
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from onyx.regulatory.labeling.provider import MAX_RESPONSE_BYTES, LabelingOutcome

ABSTENTION_KEY = "provider_abstention"


class ProviderAbstentionReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    version: Literal[1] = 1
    request_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    response_text: str = Field(min_length=1, max_length=MAX_RESPONSE_BYTES)
    response_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def valid_abstention(self) -> "ProviderAbstentionReceipt":
        if (
            len(self.response_text.encode()) > MAX_RESPONSE_BYTES
            or sha256(self.response_text.encode()).hexdigest() != self.response_sha256
        ):
            raise ValueError("The provider abstention response changed")
        outcome = LabelingOutcome.model_validate_json(self.response_text)
        if not outcome.abstained or outcome.labels:
            raise ValueError("Only an explicit valid empty abstention can be accepted")
        return self


def accept_provider_abstention(
    request_hash: str, response_text: str
) -> ProviderAbstentionReceipt:
    return ProviderAbstentionReceipt(
        request_hash=request_hash,
        response_text=response_text,
        response_sha256=sha256(response_text.encode()).hexdigest(),
    )
