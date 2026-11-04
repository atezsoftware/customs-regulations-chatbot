"""Native reuse is bounded, input-bound, and discarded after each operation."""

from collections.abc import Callable
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
from unittest.mock import Mock

import pytest

from onyx.regulatory.labeling import chapter_caption as chapter
from onyx.regulatory.labeling import chapter_caption_cache as cache
from tests.unit.onyx.regulatory.labeling.test_chapter_caption import (
    ASSET,
    FILE,
    RAW,
    sources,
)


def identity() -> cache.NativeCaptionIdentity:
    return cache.NativeCaptionIdentity(
        ASSET, FILE, "retained-pdf", sha256(RAW).hexdigest(), len(RAW), "tenant-test"
    )


def _proof(read: Callable[[], bytes]) -> chapter.ChapterCaptionSourceProof:
    return cache.chapter_caption_native_proof(
        identity=identity(), chunk_id="caption", sources=sources(), read_raw=read
    )


def test_repeated_contexts_reuse_exact_proof_but_next_operation_reopens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    read = Mock(return_value=RAW)
    parse = Mock(wraps=cache.build_chapter_caption_source_proof)
    monkeypatch.setattr(cache, "build_chapter_caption_source_proof", parse)
    with cache.native_caption_proof_operation():
        proofs = [_proof(read) for _ in range(12)]
        assert all(proof == proofs[0] for proof in proofs)
    assert read.call_count == parse.call_count == 1
    with cache.native_caption_proof_operation():
        assert _proof(read) == proofs[0]
    assert read.call_count == parse.call_count == 2
    assert proofs[0].model_dump(
        mode="json"
    ) == chapter.build_chapter_caption_source_proof(
        raw_pdf=RAW,
        source_sha256=identity().source_sha256,
        source_asset_id=ASSET,
        user_file_id=FILE,
        chunk_id="caption",
        sources=sources(),
    ).model_dump(mode="json")


@pytest.mark.parametrize(
    "change",
    [
        "current_text",
        "current_metadata",
        "frozen_metadata",
        "active",
        "qualification",
        "source_hash",
        "source_size",
    ],
)
def test_warm_proof_cannot_hide_input_or_qualification_drift(
    change: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    read = Mock(return_value=RAW)
    with cache.native_caption_proof_operation():
        _proof(read)
        changed = deepcopy(sources())
        raw_identity = identity()
        if change == "current_text":
            changed[0] = replace(
                changed[0], current_text=changed[0].current_text + " changed"
            )
        elif change == "current_metadata":
            snapshot = deepcopy(changed[0].current_snapshot)
            snapshot["chunk_metadata"]["changed"] = True
            changed[0] = replace(changed[0], current_snapshot=snapshot)
        elif change == "frozen_metadata":
            snapshot = deepcopy(changed[0].snapshot)
            snapshot["chunk_metadata"]["changed"] = True
            changed[0] = replace(changed[0], snapshot=snapshot)
        elif change == "active":
            changed[0] = replace(changed[0], active=False)
        elif change == "qualification":
            monkeypatch.setattr(
                chapter,
                "_PROFILES",
                tuple(
                    replace(profile, chapter_lines=("Changed",))
                    for profile in chapter._PROFILES
                ),
            )
        elif change == "source_hash":
            raw_identity = replace(raw_identity, source_sha256="0" * 64)
        else:
            raw_identity = replace(raw_identity, byte_count=raw_identity.byte_count + 1)
        with pytest.raises(ValueError):
            cache.chapter_caption_native_proof(
                identity=raw_identity,
                chunk_id="caption",
                sources=changed,
                read_raw=read,
            )


@pytest.mark.parametrize("limit", ["_MAX_RAW_BYTES", "_MAX_RAW_ENTRIES"])
def test_memory_caps_fall_back_to_validation(
    limit: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cache, limit, 0)
    monkeypatch.setattr(cache, "_MAX_PROOF_ENTRIES", 0)
    read = Mock(return_value=RAW)
    with cache.native_caption_proof_operation():
        for _ in range(2):
            _proof(read)
    assert read.call_count == 2


def test_read_and_parse_failures_are_not_cached_and_scope_resets() -> None:
    read = Mock(side_effect=[OSError("unavailable"), RAW + b"bad", RAW, RAW])
    with pytest.raises(RuntimeError), cache.native_caption_proof_operation():
        for error in (OSError, ValueError):
            with pytest.raises(error):
                _proof(read)
        _proof(read)
        raise RuntimeError("caller failed")
    _proof(read)
    assert read.call_count == 4


def test_failed_pure_proof_does_not_retain_raw_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    read = Mock(return_value=RAW)
    actual = cache.build_chapter_caption_source_proof
    calls = 0

    def parse(**_kwargs: object) -> chapter.ChapterCaptionSourceProof:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError("native proof failed")
        return actual(
            raw_pdf=RAW,
            source_sha256=identity().source_sha256,
            source_asset_id=ASSET,
            user_file_id=FILE,
            chunk_id="caption",
            sources=sources(),
        )

    monkeypatch.setattr(cache, "build_chapter_caption_source_proof", parse)
    with cache.native_caption_proof_operation():
        with pytest.raises(ValueError):
            _proof(read)
        _proof(read)
    assert calls == read.call_count == 2


def test_default_off_and_file_tenant_bound_identity() -> None:
    read = Mock(return_value=RAW)
    for _ in range(2):
        _proof(read)
    assert read.call_count == 2
    with cache.native_caption_proof_operation():
        for tenant, file_id in (
            ("tenant-a", "file-a"),
            ("tenant-b", "file-a"),
            ("tenant-a", "file-b"),
        ):
            cache.chapter_caption_native_proof(
                identity=replace(identity(), tenant_id=tenant, file_id=file_id),
                chunk_id="caption",
                sources=sources(),
                read_raw=read,
            )
    assert read.call_count == 5
