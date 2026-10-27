"""Frozen provider requests retain their shape through bounded concurrency."""

import json
import math
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from dataclasses import replace
from hashlib import sha256
from threading import Barrier, Event, Lock
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from onyx.db.enums import IndexModelStatus
from onyx.db.models import SearchSettings
from onyx.document_index.publication_models import FrozenPublicationProjection
from onyx.regulatory import writer_projection
from onyx.regulatory.document_intake import embedding
from onyx.regulatory.document_intake.embedding import (
    FrozenEmbeddingGroup,
    encode_intake_embedding_groups,
)
from tests.unit.onyx.regulatory.annexes.test_context_dependencies import (
    _durable_embedding_model,
)
from tests.unit.onyx.regulatory.indexing_jobs.owned_publication_test_helpers import (
    OwnedAuthority,
    canonical_row,
    writer_inputs,
)


def _vector(text: str) -> list[float]:
    value = int(sha256(text.encode()).hexdigest()[:6], 16) / 16777216
    return [value, value / 2, value / 3]


def test_two_requests_run_concurrently_with_unchanged_groups_and_context() -> None:
    groups = [
        FrozenEmbeddingGroup(str(index), (f"content {index}", "exact duplicate title"))
        for index in range(6)
    ]
    tenant = ContextVar("embedding_test_tenant", default="wrong")
    tenant.set("owned-tenant")
    rendezvous = Barrier(2)
    lock = Lock()
    calls: list[list[str]] = []
    active = 0
    maximum = 0

    def encode(texts: list[str]) -> list[list[float]]:
        nonlocal active, maximum
        assert tenant.get() == "owned-tenant"
        tenant.set("worker-local-mutation")
        with lock:
            active += 1
            maximum = max(maximum, active)
            calls.append(texts)
        rendezvous.wait(timeout=3)
        with lock:
            active -= 1
        return [_vector(text) for text in texts]

    result = encode_intake_embedding_groups(
        groups, encode=encode, expected_dimension=3, check_owner=lambda: None
    )
    assert maximum == 2
    assert sorted(calls) == sorted([list(group.texts) for group in groups])
    assert list(result) == [group.key for group in groups]
    assert result == {
        group.key: [_vector(text) for text in group.texts] for group in groups
    }
    assert tenant.get() == "owned-tenant"


def test_whole_request_utf8_budget_is_checked_without_splitting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(embedding, "MAX_REQUEST_BYTES", 1100)
    groups = [FrozenEmbeddingGroup(str(index), ("ğ" * 20,)) for index in range(4)]
    calls = []

    def encode(texts: list[str]) -> list[list[float]]:
        calls.append(texts)
        assert len(json.dumps(texts, ensure_ascii=False).encode()) + 1024 <= 1100
        return [_vector(text) for text in texts]

    encode_intake_embedding_groups(
        groups, encode=encode, expected_dimension=3, check_owner=lambda: None
    )
    assert calls == [["ğ" * 20]] * 4


@pytest.mark.parametrize("failure", ["oversize", "empty", "duplicate"])
def test_all_input_prevalidation_finishes_before_any_provider_request(
    failure: str,
) -> None:
    encode = MagicMock()
    groups = [FrozenEmbeddingGroup("first", ("valid",))]
    groups.append(
        FrozenEmbeddingGroup(
            "first" if failure == "duplicate" else "bad",
            ("x" * (256 * 1024),)
            if failure == "oversize"
            else ("",)
            if failure == "empty"
            else ("valid",),
        )
    )
    with pytest.raises(ValueError):
        encode_intake_embedding_groups(
            groups, encode=encode, expected_dimension=3, check_owner=lambda: None
        )
    encode.assert_not_called()


@pytest.mark.parametrize("failure", ["count", "dimension", "bool", "nan", "infinity"])
def test_invalid_vectors_stop_dispatch_and_never_return_partial_results(
    failure: str,
) -> None:
    groups = [FrozenEmbeddingGroup(str(index), (str(index),)) for index in range(10)]

    def encode(texts: list[str]) -> list[object]:
        vectors: list[object] = [_vector(text) for text in texts]
        if failure == "count":
            return vectors[:-1]
        vectors[0] = (
            [0.1]
            if failure == "dimension"
            else [True, 0.2, 0.3]
            if failure == "bool"
            else [math.nan, 0.2, 0.3]
            if failure == "nan"
            else [math.inf, 0.2, 0.3]
        )
        return vectors

    provider = MagicMock(side_effect=encode)
    with pytest.raises(ValueError):
        encode_intake_embedding_groups(
            groups, encode=provider, expected_dimension=3, check_owner=lambda: None
        )
    assert 1 <= provider.call_count <= 2


@pytest.mark.parametrize("lost_check", [1, 2, 3, 4, 5])
def test_lost_ownership_stops_dispatch_and_discards_acknowledged_results(
    lost_check: int,
) -> None:
    checks = 0
    calls = 0

    def guard() -> None:
        nonlocal checks
        checks += 1
        if checks == lost_check:
            raise RuntimeError("lease lost")

    def encode(texts: list[str]) -> list[list[float]]:
        nonlocal calls
        calls += 1
        return [_vector(text) for text in texts]

    with pytest.raises(RuntimeError, match="lease lost"):
        encode_intake_embedding_groups(
            [FrozenEmbeddingGroup(str(index), (str(index),)) for index in range(10)],
            encode=encode,
            expected_dimension=3,
            check_owner=guard,
        )
    assert calls <= 2 and (calls == 0 if lost_check == 1 else True)


def test_first_failure_drains_in_flight_request_without_launching_more() -> None:
    both_started = Barrier(3)
    fail_first = Event()
    finish_second = Event()
    calls = []

    def encode(texts: list[str]) -> list[list[float]]:
        calls.append(texts)
        both_started.wait(timeout=3)
        if texts == ["first"]:
            assert fail_first.wait(timeout=3)
            raise RuntimeError("provider failed")
        assert finish_second.wait(timeout=3)
        return [_vector(text) for text in texts]

    with ThreadPoolExecutor(max_workers=1) as caller:
        result = caller.submit(
            encode_intake_embedding_groups,
            [
                FrozenEmbeddingGroup(text, (text,))
                for text in ["first", "second", "third"]
            ],
            encode=encode,
            expected_dimension=3,
            check_owner=lambda: None,
        )
        both_started.wait(timeout=3)
        fail_first.set()
        assert not result.done()
        finish_second.set()
        with pytest.raises(RuntimeError, match="provider failed"):
            result.result(timeout=3)
    assert sorted(calls) == [["first"], ["second"]]


@pytest.mark.parametrize("concurrent", [False, True])
@pytest.mark.parametrize("has_title", [False, True])
def test_actual_writer_preserves_encoder_configuration_vectors_and_legacy_request_order(
    concurrent: bool,
    has_title: bool,
) -> None:
    file_id = uuid4()
    authority = OwnedAuthority(file_id)
    inputs = writer_inputs(
        file_id,
        [
            canonical_row(file_id, index, f"Article {index}: exact frozen source")
            for index in range(18)
        ],
    )
    settings = SearchSettings(
        id=1,
        status=IndexModelStatus.PRESENT,
        index_name="intake-batch-test",
        model_name="embedding",
        model_dim=3,
        normalize=True,
        enable_contextual_rag=False,
    )
    inputs = replace(inputs, settings=[settings])
    if not has_title:
        inputs.file.name = ""
    model = _durable_embedding_model()
    embedder = MagicMock(embedding_model=model)
    client = MagicMock()
    client.indices.get.return_value = {
        settings.index_name: {"settings": {"index": {"uuid": "physical-test"}}}
    }
    checks = MagicMock()
    with (
        patch(
            "onyx.regulatory.projection.effective_contextual_rag_enabled",
            return_value=False,
        ),
        patch.object(
            writer_projection, "PublicationStore", side_effect=authority.for_scope
        ),
        patch.object(
            writer_projection.DefaultIndexingEmbedder,
            "from_db_search_settings",
            return_value=embedder,
        ),
        patch.object(
            writer_projection, "resolve_review_context_llm", return_value=None
        ),
        patch.object(
            model,
            "encode",
            side_effect=lambda texts, **_: [_vector(text) for text in texts],
        ) as encode,
    ):
        prepared = writer_projection.prepare_owned_correction(
            authority.owner,
            client,
            inputs,
            inputs.canonical,
            changed_id=None,
            before_embedding_request=checks if concurrent else None,
        )
    calls = encode.call_args_list
    expected_sizes = [2] * 18 if has_title else [1] * 18
    assert [len(call.kwargs["texts"]) for call in calls] == expected_sizes
    for call in calls:
        assert call.kwargs["tenant_id"] == authority.owner.scope.tenant_id
    assert checks.call_count >= (2 * len(expected_sizes) if concurrent else 0)
    if not concurrent:
        checks.assert_not_called()
    assert len(prepared.bindings) == 18
    frozen_inputs = []
    frozen_groups = []
    for binding in prepared.bindings:
        assert isinstance(binding.projection, FrozenPublicationProjection)
        source = json.loads(binding.projection.source_json)
        frozen = binding.projection.embedding_inputs
        frozen_groups.append(frozen)
        frozen_inputs.extend(frozen)
        assert source["content_vector"] == _vector(frozen[0])
        if has_title:
            assert source["title_vector"] == _vector(frozen[-1])
        else:
            assert "title_vector" not in source and len(frozen) == 1
        assert binding.index.accepts_encoder_configuration(
            json.loads(binding.projection.embedding_config_json)
        )
    if concurrent:
        assert sorted(tuple(call.kwargs["texts"]) for call in calls) == sorted(
            frozen_groups
        )
    else:
        assert [
            text for call in calls for text in call.kwargs["texts"]
        ] == frozen_inputs


def test_reused_and_duplicate_frozen_groups_do_not_add_provider_calls() -> None:
    file_id = uuid4()
    authority = OwnedAuthority(file_id)
    inputs = writer_inputs(
        file_id,
        [canonical_row(file_id, index, "Exact duplicate source") for index in range(3)],
    )
    settings = SearchSettings(
        id=1,
        status=IndexModelStatus.PRESENT,
        index_name="intake-reuse-test",
        model_name="embedding",
        model_dim=3,
        normalize=True,
        enable_contextual_rag=False,
    )
    inputs = replace(inputs, settings=[settings])
    model = _durable_embedding_model()
    embedder = MagicMock(embedding_model=model)
    client = MagicMock()
    client.indices.get.return_value = {
        settings.index_name: {"settings": {"index": {"uuid": "physical-test"}}}
    }
    with (
        patch(
            "onyx.regulatory.projection.effective_contextual_rag_enabled",
            return_value=False,
        ),
        patch.object(
            writer_projection, "PublicationStore", side_effect=authority.for_scope
        ),
        patch.object(
            writer_projection.DefaultIndexingEmbedder,
            "from_db_search_settings",
            return_value=embedder,
        ),
        patch.object(
            writer_projection, "resolve_review_context_llm", return_value=None
        ),
        patch.object(
            model,
            "encode",
            side_effect=lambda texts, **_: [_vector(text) for text in texts],
        ) as encode,
    ):
        prepared = writer_projection.prepare_owned_correction(
            authority.owner,
            client,
            inputs,
            inputs.canonical,
            changed_id=None,
            before_embedding_request=lambda: None,
        )
    assert encode.call_count == 1 and len(encode.call_args.kwargs["texts"]) == 2
    assert len(prepared.bindings) == 3
