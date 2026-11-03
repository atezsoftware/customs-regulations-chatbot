import logging
import threading
import time
from uuid import UUID

from onyx.cache.interface import CACHE_TRANSIENT_ERRORS, CacheBackend

logger = logging.getLogger(__name__)

PREFIX = "chatsessionstop"
FENCE_PREFIX = f"{PREFIX}_fence"
FENCE_TTL = 10 * 60  # 10 minutes


class TurnCancellationProbe:
    """Keep an observed explicit stop bound to its original turn."""

    def __init__(self, chat_session_id: UUID, cache: CacheBackend) -> None:
        self._chat_session_id = chat_session_id
        self._cache = cache
        self._cancelled = threading.Event()
        self._warning_lock = threading.Lock()
        self._last_warning: float | None = None

    def __call__(self) -> bool:
        if self._cancelled.is_set():
            return False
        try:
            connected = is_connected(self._chat_session_id, self._cache)
        except CACHE_TRANSIENT_ERRORS:
            # An unavailable stop lookup is not an explicit cancellation.
            # The next invocation retries; healthy results are never cached.
            now = time.monotonic()
            with self._warning_lock:
                if self._last_warning is None or now - self._last_warning >= 60:
                    self._last_warning = now
                    logger.warning(
                        "Chat stop-signal cache lookup unavailable; retrying",
                        exc_info=True,
                    )
            return not self._cancelled.is_set()
        if not connected:
            self._cancelled.set()
        # Another consumer may have observed a stop during this lookup.
        return connected and not self._cancelled.is_set()


def _get_fence_key(chat_session_id: UUID) -> str:
    """Generate the cache key for a chat session stop signal fence.

    Args:
        chat_session_id: The UUID of the chat session

    Returns:
        The fence key string. Tenant isolation is handled automatically
        by the cache backend (Redis key-prefixing or Postgres schema routing).
    """
    return f"{FENCE_PREFIX}_{chat_session_id}"


def set_fence(chat_session_id: UUID, cache: CacheBackend, value: bool) -> None:
    """Set or clear the stop signal fence for a chat session.

    Args:
        chat_session_id: The UUID of the chat session
        cache: Tenant-aware cache backend
        value: True to set the fence (stop signal), False to clear it
    """
    fence_key = _get_fence_key(chat_session_id)
    if not value:
        cache.delete(fence_key)
        return
    cache.set(fence_key, 0, ex=FENCE_TTL)


def is_connected(chat_session_id: UUID, cache: CacheBackend) -> bool:
    """Check if the chat session should continue (not stopped).

    Args:
        chat_session_id: The UUID of the chat session to check
        cache: Tenant-aware cache backend

    Returns:
        True if the session should continue, False if it should stop
    """
    return not cache.exists(_get_fence_key(chat_session_id))


def reset_cancel_status(chat_session_id: UUID, cache: CacheBackend) -> None:
    """Clear the stop signal for a chat session.

    Args:
        chat_session_id: The UUID of the chat session
        cache: Tenant-aware cache backend
    """
    cache.delete(_get_fence_key(chat_session_id))
