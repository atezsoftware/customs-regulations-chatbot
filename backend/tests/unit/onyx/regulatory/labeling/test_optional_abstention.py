"""Only a verified explicit empty response is a normal optional result."""

import pytest

from onyx.regulatory.labeling.abstention import (
    ProviderAbstentionReceipt,
    accept_provider_abstention,
)


@pytest.mark.parametrize(
    "response",
    [
        "not json",
        '{"labels":[],"abstained":"true"}',
        '{"labels":[],"abstained":false}',
        '{"labels":[],"abstained":true,"extra":1}',
        '{"labels":[{"label_id":"x","evidence_quote":"x"}],"abstained":true}',
    ],
)
def test_invalid_or_nonabstained_receipts_remain_errors(response: str) -> None:
    with pytest.raises(ValueError):
        accept_provider_abstention("a" * 64, response)


def test_actual_response_and_request_identity_are_retained_and_tampering_rejected() -> (
    None
):
    response = '{"labels": [], "abstained": true}'
    receipt = accept_provider_abstention("a" * 64, response)
    assert receipt.response_text == response and receipt.request_hash == "a" * 64
    raw = receipt.model_dump()
    raw["response_text"] = '{"abstained":true,"labels":[]}'
    with pytest.raises(ValueError, match="response changed"):
        ProviderAbstentionReceipt.model_validate(raw)
