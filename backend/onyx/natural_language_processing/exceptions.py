class ModelServerRateLimitError(Exception):
    """
    Exception raised for rate limiting errors from the model server.
    """


class CohereBillingLimitError(Exception):
    """
    Raised when Cohere rejects requests because the billing cap is reached.
    """


class EmbeddingProviderHTTPError(Exception):
    """Secret-safe HTTP failure returned by a maintained embedding provider."""

    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        super().__init__(f"embedding provider returned HTTP {status_code}")


class EmbeddingProviderTimeoutError(Exception):
    """Secret-safe embedding provider timeout."""


class EmbeddingProviderConnectionError(Exception):
    """Secret-safe embedding provider network failure."""


class EmbeddingProviderResponseError(ValueError):
    """Secret-safe terminal embedding response contract failure."""
