"""Only a currently associated intake run gets smaller direct-labeling shards."""

from hashlib import sha256
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from sqlalchemy.orm import Session

from onyx.db import document_intake_labeling
from onyx.db.regulatory_labeling import RunLease
from onyx.regulatory.labeling import orchestrator
from onyx.regulatory.labeling.provider import LabelDefinition, TaxonomyDefinition


@pytest.mark.parametrize(
    ("transport", "current_intake", "concurrency", "sizes"),
    [
        ("vertex_direct_v1", True, 4, [8] * 8 + [1]),
        ("vertex_direct_v1", False, 1, [64, 1]),
        ("vertex_gcs_v1", True, 1, [64, 1]),
    ],
)
def test_preparation_and_gateway_keep_ordinary_behavior_and_scope_intake_fast_lane(
    monkeypatch: pytest.MonkeyPatch,
    transport: str,
    current_intake: bool,
    concurrency: int,
    sizes: list[int],
) -> None:
    session = MagicMock(spec=Session)
    session.__enter__.return_value = session
    monkeypatch.setattr(
        orchestrator, "get_session_with_current_tenant", lambda: session
    )
    lease = RunLease(run_id=uuid4(), generation=1, token=uuid4())
    taxonomy = TaxonomyDefinition(
        name="Labels",
        labels=[LabelDefinition(id="customs", name="Customs", description="Documents")],
    )
    run = SimpleNamespace(
        id=lease.run_id,
        cancel_requested=False,
        taxonomy=SimpleNamespace(definition=taxonomy.model_dump()),
        provider_binding={"transport": transport},
        model_configuration_id=7,
    )
    monkeypatch.setattr(orchestrator.repository, "load_claimed_run", lambda *_args: run)
    scope = MagicMock(return_value=current_intake)
    monkeypatch.setattr(
        document_intake_labeling, "is_current_intake_labeling_run", scope
    )
    items = [
        SimpleNamespace(
            id=uuid4(),
            regulatory_chunk_id=f"chunk-{i}",
            canonical_text_sha256=sha256(str(i).encode()).hexdigest(),
            context_snapshot="Complete context.",
            text_snapshot=f"Source {i}",
        )
        for i in range(65)
    ]
    monkeypatch.setattr(
        orchestrator.repository,
        "prepare_next_item_page",
        lambda *_args, **_kwargs: items,
    )
    monkeypatch.setattr(orchestrator.repository, "next_shard_ordinal", lambda *_args: 0)
    stored = MagicMock(return_value="submitting")
    monkeypatch.setattr(orchestrator.repository, "store_prepared_shards", stored)
    monkeypatch.setattr(orchestrator.repository, "release_run", MagicMock())
    orchestrator._prepare(lease, "public")
    assert [len(shard.item_ids) for shard in stored.call_args.kwargs["shards"]] == sizes
    assert len(stored.call_args.kwargs["requests"]) == 65
    user = object()
    monkeypatch.setattr(
        orchestrator.repository, "get_claimed_run_requester", lambda *_args: user
    )
    binding = SimpleNamespace(transport=transport)
    monkeypatch.setattr(
        orchestrator,
        "LabelingProviderBinding",
        SimpleNamespace(model_validate=lambda _value: binding),
    )
    resolver = MagicMock()
    monkeypatch.setattr(orchestrator, "resolve_labeling_gateway", resolver)
    assert orchestrator._gateway_for_claimed_run(lease) is resolver.return_value
    assert resolver.call_args.kwargs["direct_max_concurrency"] == concurrency
    assert "direct_request_timeout_seconds" not in resolver.call_args.kwargs
    assert resolver.call_args.kwargs["expected_binding"] is binding
    if transport == "vertex_direct_v1":
        assert scope.call_count == 2
    else:
        scope.assert_not_called()
