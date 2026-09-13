from types import SimpleNamespace
from typing import cast

import pytest

from onyx.db.labeling_configuration import LabelingBatchGateway
from onyx.error_handling.error_codes import OnyxErrorCode
from onyx.error_handling.exceptions import OnyxError
from onyx.regulatory.indexing_jobs.models import IndexingGatewayHTTPError
from onyx.server.features.document_set.labeling_api import _probe_labeling_gateway


@pytest.mark.parametrize(
    ("status_code", "expected_code"),
    [
        (400, OnyxErrorCode.INVALID_INPUT),
        (403, OnyxErrorCode.INVALID_INPUT),
        (500, OnyxErrorCode.BAD_GATEWAY),
    ],
)
def test_native_probe_maps_http_errors_without_exposing_upstream_detail(
    status_code: int, expected_code: OnyxErrorCode
) -> None:
    secret = "secret-service-account-detail"

    def probe() -> None:
        error = IndexingGatewayHTTPError(status_code)
        error.add_note(secret)
        raise error

    gateway = cast(
        LabelingBatchGateway,
        SimpleNamespace(probe_gemini_read_access=probe),
    )

    with pytest.raises(OnyxError) as raised:
        _probe_labeling_gateway(gateway)

    assert raised.value.error_code is expected_code
    assert secret not in raised.value.detail
