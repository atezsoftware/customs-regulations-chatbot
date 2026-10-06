from collections.abc import Callable
from typing import cast
from unittest.mock import Mock

import pytest

from onyx.asv3.native_token_cache import (
    NativeTokenCountCache,
    native_token_cache_enabled,
    uncached_token_count,
)


def test_exact_counts_are_reused_without_replacing_unicode_with_byte_lengths() -> None:
    text = "Ücretsiz tamir: aynı makine, farklı şartlar."
    counter = Mock(return_value=13)
    cache = NativeTokenCountCache()
    assert cache.count(text, counter) == uncached_token_count(text, counter) == 13
    assert cache.count(text, counter) == 13
    assert counter.call_count == 2
    zero = Mock(return_value=0)
    assert cache.count("", zero) == cache.count("", zero) == 0
    zero.assert_called_once_with("")


def test_lru_eviction_recounts_only_the_evicted_text() -> None:
    counter = Mock(side_effect=len)
    cache = NativeTokenCountCache(max_entries=2)
    for text in ("a", "bb", "a", "ccc", "bb"):
        assert cache.count(text, counter) == len(text)
    assert [call.args[0] for call in counter.call_args_list] == [
        "a",
        "bb",
        "ccc",
        "bb",
    ]
    assert cache.entry_count == 2


@pytest.mark.parametrize("error", [NotImplementedError, LookupError])
def test_unsupported_fallback_is_not_cached(error: type[Exception]) -> None:
    text = "Şart ve istisna"
    counter = Mock(side_effect=[error("unsupported"), 7])
    cache = NativeTokenCountCache()
    assert cache.count(text, counter) == len(text.encode("utf-8"))
    assert cache.entry_count == 0
    assert cache.count(text, counter) == cache.count(text, counter) == 7
    assert counter.call_count == 2


@pytest.mark.parametrize("value", [-1, True, 1.5, "7", None])
def test_invalid_counts_are_rejected_and_not_cached(value: object) -> None:
    counter = Mock(side_effect=[value, 7])
    cache = NativeTokenCountCache()
    with pytest.raises(ValueError, match="invalid count"):
        cache.count("original", counter)
    assert cache.entry_count == 0
    assert cache.count("original", counter) == 7


def test_other_exceptions_propagate_without_retaining_a_count() -> None:
    counter = Mock(side_effect=[RuntimeError("counter failed"), 3])
    cache = NativeTokenCountCache()
    with pytest.raises(RuntimeError, match="counter failed"):
        cache.count("original", counter)
    assert cache.entry_count == 0
    assert cache.count("original", counter) == 3


def test_changing_tokenizer_identity_invalidates_existing_counts() -> None:
    first = Mock(return_value=5)
    second = Mock(return_value=8)
    cache = NativeTokenCountCache()
    assert cache.count("same", first) == 5
    assert cache.count("same", second) == 8
    assert cache.count("same", first) == 5
    assert first.call_count == 2
    assert cache.entry_count == 1


def test_cache_keys_retain_only_fixed_size_digests_and_lengths() -> None:
    text = "Çok büyük kaynak metni. " * 50_000
    cache = NativeTokenCountCache()
    assert cache.count(text, len) == len(text)
    for (digest, byte_length), count in cache._counts.items():
        assert isinstance(digest, bytes) and len(digest) == 32
        assert byte_length == len(text.encode("utf-8"))
        assert count == len(text)
    assert text not in repr(cache.__dict__)


def test_disabled_capacity_and_missing_counter_preserve_uncached_behavior() -> None:
    text = "İthalat"
    counter = Mock(return_value=3)
    cache = NativeTokenCountCache(max_entries=0)
    assert cache.count(text, counter) == cache.count(text, counter) == 3
    assert counter.call_count == 2
    assert cache.count(text, None) == len(text.encode("utf-8"))
    assert cache.entry_count == 0


def test_non_utf8_string_keeps_tokenizer_behavior_without_a_cache_entry() -> None:
    text = "\ud800"
    counter: Callable[[str], int] = Mock(return_value=1)
    cache = NativeTokenCountCache()
    assert cache.count(text, counter) == uncached_token_count(text, counter) == 1
    assert cache.entry_count == 0
    with pytest.raises(UnicodeEncodeError):
        cache.count(text, None)


@pytest.mark.parametrize("capacity", [-1, True, 1.5])
def test_invalid_capacity_is_rejected(capacity: object) -> None:
    with pytest.raises(ValueError, match="nonnegative integer"):
        NativeTokenCountCache(cast(int, capacity))


def test_enablement_is_limited_to_trusted_parallel_execution_markers() -> None:
    assert native_token_cache_enabled(
        {"research_profile": "experimental", "experimental_parallel": True}
    )
    assert native_token_cache_enabled(
        {"research_profile": "experimental", "serial_session_diagnostics": True}
    )
    assert not native_token_cache_enabled({"research_profile": "experimental"})
    assert not native_token_cache_enabled(
        {"research_profile": "experimental", "experimental_parallel": "true"}
    )
    assert not native_token_cache_enabled(
        {"research_profile": "normal", "experimental_parallel": True}
    )
