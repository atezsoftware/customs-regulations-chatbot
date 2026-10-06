"""Bounded token counts for one model's parallel research lifetime."""

from collections import OrderedDict
from collections.abc import Callable, Mapping
from hashlib import sha256

DEFAULT_NATIVE_TOKEN_CACHE_ENTRIES = 1_024
TokenCounter = Callable[[str], int]


def native_token_cache_enabled(services: Mapping[str, object]) -> bool:
    return services.get("research_profile") == "experimental" and (
        services.get("experimental_parallel") is True
        or services.get("serial_session_diagnostics") is True
    )


def uncached_token_count(text: str, counter: TokenCounter | None) -> int:
    """Preserve the adapter's tokenizer validation and unsupported fallback."""
    if counter is None:
        return len(text.encode("utf-8"))
    try:
        counted = counter(text)
    except (NotImplementedError, LookupError):
        return len(text.encode("utf-8"))
    if isinstance(counted, bool) or not isinstance(counted, int) or counted < 0:
        raise ValueError("The selected model token counter returned an invalid count")
    return counted


class NativeTokenCountCache:
    """LRU metadata only; never retains prompt strings or encoded content.

    The tokenizer must be deterministic for its callable's lifetime. Replace
    the callable when its tokenizer/configuration changes. Unsupported results
    are not retained, so transient failures still get the uncached behavior.
    Each ResearchModel owns its cache; it is not a shared worker resource.
    """

    def __init__(self, max_entries: int = DEFAULT_NATIVE_TOKEN_CACHE_ENTRIES) -> None:
        if isinstance(max_entries, bool) or not isinstance(max_entries, int):
            raise ValueError("Token cache capacity must be a nonnegative integer")
        if max_entries < 0:
            raise ValueError("Token cache capacity must be a nonnegative integer")
        self.max_entries = max_entries
        self._counter: TokenCounter | None = None
        self._counts: OrderedDict[tuple[bytes, int], int] = OrderedDict()

    @property
    def entry_count(self) -> int:
        return len(self._counts)

    def count(self, text: str, counter: TokenCounter | None) -> int:
        if counter is not self._counter:
            self._counts.clear()
            self._counter = counter
        if counter is None or not self.max_entries:
            return uncached_token_count(text, counter)
        try:
            encoded = text.encode("utf-8")
        except UnicodeEncodeError:
            # A tokenizer may accept a string that the fallback cannot encode.
            return uncached_token_count(text, counter)
        key = (sha256(encoded).digest(), len(encoded))
        cached = self._counts.get(key)
        if cached is not None:
            self._counts.move_to_end(key)
            return cached
        try:
            counted = counter(text)
        except (NotImplementedError, LookupError):
            return len(encoded)
        if isinstance(counted, bool) or not isinstance(counted, int) or counted < 0:
            raise ValueError(
                "The selected model token counter returned an invalid count"
            )
        self._counts[key] = counted
        if len(self._counts) > self.max_entries:
            self._counts.popitem(last=False)
        return counted
