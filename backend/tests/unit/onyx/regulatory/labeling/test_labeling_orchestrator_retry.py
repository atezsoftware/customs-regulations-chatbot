import pytest

from onyx.regulatory.indexing_jobs.models import IndexingGatewayHTTPError
from onyx.regulatory.labeling.orchestrator import (
    LABELING_MAX_PROVIDER_FAILURES,
    _provider_error_is_terminal,
)


@pytest.mark.parametrize("status_code", [400, 401, 403])
def test_permanent_provider_errors_are_terminal_on_first_failure(
    status_code: int,
) -> None:
    assert _provider_error_is_terminal(IndexingGatewayHTTPError(status_code), 0)


@pytest.mark.parametrize("status_code", [408, 409, 429, 500, 503])
def test_transient_provider_errors_retain_bounded_retry_budget(
    status_code: int,
) -> None:
    error = IndexingGatewayHTTPError(status_code)

    assert not _provider_error_is_terminal(error, 0)
    assert _provider_error_is_terminal(error, LABELING_MAX_PROVIDER_FAILURES - 1)
