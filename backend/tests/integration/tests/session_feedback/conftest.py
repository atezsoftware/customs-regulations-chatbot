"""Exercise session feedback through the running frontend API proxy."""

from collections.abc import Generator

import httpx
import pytest

from tests.integration.common_utils import http_client
from tests.integration.common_utils.constants import API_SERVER_URL


@pytest.fixture(scope="session", autouse=True)
def _test_client() -> Generator[httpx.Client, None, None]:
    with httpx.Client(base_url=API_SERVER_URL, timeout=30) as client:
        http_client.set_test_client(client)
        try:
            yield client
        finally:
            http_client.set_test_client(None)
