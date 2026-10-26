"""Intake-only labeling performance policy is fenced by durable current scope."""

import json
from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest
from google.auth.credentials import Credentials
from google.genai import types as genai_types
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from onyx.db import document_intake as intake
from onyx.db import document_intake_edit as editing
from onyx.db import document_intake_reprepare as reprepare
from onyx.db import regulatory_labeling as labeling
from onyx.db.document_intake_labeling import is_current_intake_labeling_run
from onyx.db.enums import UserFileStatus
from onyx.db.labeling_configuration import LabelingProviderBinding
from onyx.db.models import (
    DocumentSet,
    ModelConfiguration,
    RegulatoryChunk,
    RegulatoryDerivedLabelProjection,
    RegulatoryDocumentIntake,
    RegulatoryDocumentIntakeFile,
    RegulatoryLabelingItem,
    RegulatoryLabelingRun,
    RegulatoryLabelingShard,
    User,
    UserFile,
)
from onyx.file_store.file_store import FileStore
from onyx.regulatory.indexing_jobs.models import (
    VertexAuthenticationMode,
    VertexBatchConfig,
)
from onyx.regulatory.labeling import (
    intake_quote_resolution,
    orchestrator,
    vertex_direct,
)
from onyx.regulatory.labeling.provider import (
    LabelAssignment,
    LabelDefinition,
    TaxonomyDefinition,
    validate_labeling_response,
)
from onyx.tracing.flows import LLMFlow
from tests.external_dependency_unit.regulatory.test_document_intake import (
    _admitted,
    _completed_labels,
)
from tests.external_dependency_unit.regulatory.test_document_intake import (
    intake_owner as intake_owner,
)
from tests.external_dependency_unit.regulatory.test_document_intake import (
    intake_session as intake_session,
)
from tests.unit.onyx.regulatory.labeling.test_vertex_direct import _Store


def _current_run(
    session: Session, owner: tuple[User, DocumentSet, ModelConfiguration]
) -> tuple[
    RegulatoryDocumentIntake,
    list[RegulatoryDocumentIntakeFile],
    RegulatoryLabelingRun,
]:
    user, target, model = owner
    batch, members = _admitted(session, user, target, model)
    run = _completed_labels(session, batch, members)
    definition: dict[str, object] = {
        "name": "Intake labels",
        "labels": [{"id": "customs", "name": "Gümrük", "description": "Gümrük"}],
    }
    run.taxonomy.definition = definition
    binding = LabelingProviderBinding(
        model_configuration_id=model.id,
        provider_id=model.llm_provider.id,
        project="intake-policy-test",
        location="global",
        authentication_mode=VertexAuthenticationMode.WORKLOAD_IDENTITY,
        credential_identity="fixture",
        model="gemini-3.5-flash",
        transport="vertex_direct_v1",
    )
    run.provider_binding = binding.model_dump(mode="json")
    run.requested_by_id = user.id
    run.status, run.stage = "queued", "preparing"
    for member in members:
        member.status = "labeling"
    session.flush()
    return batch, members, run


@pytest.mark.parametrize("scope", ["current", "ordinary", "broadened", "published"])
def test_only_current_unpublished_intake_gets_direct_policy(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    monkeypatch: pytest.MonkeyPatch,
    scope: str,
) -> None:
    batch, members, run = _current_run(intake_session, intake_owner)
    if scope == "ordinary":
        batch.labeling_run_id = None
    elif scope == "broadened":
        run.file_ids = [*run.file_ids, str(uuid4())]
    elif scope == "published":
        intake_session.get_one(
            UserFile, members[0].user_file_id
        ).status = UserFileStatus.COMPLETED
    intake_session.flush()
    assert is_current_intake_labeling_run(intake_session, run) == (scope == "current")
    lease = labeling.claim_run(
        intake_session, run_id=run.id, expected_generation=0, lease_seconds=300
    )
    assert lease is not None

    @contextmanager
    def session_scope() -> Iterator[Session]:
        yield intake_session

    resolver = MagicMock()
    grouping = MagicMock(wraps=orchestrator.shard_by_count_and_bytes)
    monkeypatch.setattr(orchestrator, "get_session_with_current_tenant", session_scope)
    monkeypatch.setattr(orchestrator, "resolve_labeling_gateway", resolver)
    monkeypatch.setattr(orchestrator, "shard_by_count_and_bytes", grouping)
    orchestrator._gateway_for_claimed_run(lease)
    assert resolver.call_args.kwargs["direct_max_concurrency"] == (
        4 if scope == "current" else 1
    )
    orchestrator._prepare(lease, "public")
    assert grouping.call_args.kwargs["item_limit"] == (8 if scope == "current" else 64)


def test_completed_previous_run_loses_intake_policy_when_replaced(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
) -> None:
    batch, members, previous = _current_run(intake_session, intake_owner)
    previous.status = "completed"
    current = _completed_labels(intake_session, batch, members)
    current.status = "queued"
    current.requested_by_id = intake_owner[0].id
    intake_session.flush()
    assert not is_current_intake_labeling_run(intake_session, previous)
    assert is_current_intake_labeling_run(intake_session, current)


def test_association_check_reads_database_instead_of_cached_batch(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
) -> None:
    batch, _members, run = _current_run(intake_session, intake_owner)
    assert is_current_intake_labeling_run(intake_session, run)
    intake_session.execute(
        update(RegulatoryDocumentIntake)
        .where(RegulatoryDocumentIntake.id == batch.id)
        .values(labeling_run_id=None)
        .execution_options(synchronize_session=False)
    )
    assert batch.labeling_run_id == run.id
    assert not is_current_intake_labeling_run(intake_session, run)


def test_active_lease_prevents_edit_or_reprepare_reassociation(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
) -> None:
    batch, members, run = _current_run(intake_session, intake_owner)
    lease = labeling.claim_run(
        intake_session, run_id=run.id, expected_generation=0, lease_seconds=300
    )
    assert lease is not None
    batch.status = "partial"
    intake_session.flush()
    assert editing.edit_review_digest(intake_session, batch) is None
    with pytest.raises(intake.IntakeConflictError, match="active.*labeling"):
        reprepare.reprepare_snapshot(intake_session, batch)
    assert batch.labeling_run_id == run.id
    assert {str(member.user_file_id) for member in members} == set(run.file_ids)
    batch.status = "preparing"
    intake_session.flush()
    assert is_current_intake_labeling_run(intake_session, run)


@pytest.mark.parametrize(
    "case", ["current", "ordinary", "ambiguous", "substantive", "normal-retry"]
)
def test_cached_label_quotes_resolve_only_for_current_intake_and_preserve_receipts(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    user, target, model = intake_owner
    batch, members = _admitted(intake_session, user, target, model)
    chunks = list(
        intake_session.scalars(
            select(RegulatoryChunk).where(
                RegulatoryChunk.user_file_id.in_([m.user_file_id for m in members])
            )
        )
    )
    root = next(m for m in members if m.role == "root")
    main = next(c for c in chunks if c.user_file_id == root.user_file_id)
    annex = next(c for c in chunks if c.user_file_id != root.user_file_id)
    main.text = (
        "Teknik\nbelge saklanır. Teknik \tbelge saklanır."
        if case == "ambiguous"
        else "İthalatçı, ürünün teknik\nbelgesini saklar."
    )
    annex.text = "Ek belge açıkça sunulur."
    derived = RegulatoryChunk(
        id=f"rc_quote_aggregate_{uuid4().hex}",
        user_file_id=main.user_file_id,
        text=main.text,
        position=1,
        projection_ordinal=1,
        heading_path=list(main.heading_path),
        chunk_type="hierarchical_aggregate",
        chunk_metadata={
            "chunk_variant": "hierarchical_aggregate",
            "source_regulatory_chunk_ids": [main.id],
        },
        status="active",
        source="indexed",
    )
    intake_session.add(derived)
    chunks.append(derived)
    taxonomy = labeling.create_taxonomy(
        intake_session,
        created_by_id=user.id,
        taxonomy=TaxonomyDefinition(
            name="Intake quote labels",
            labels=[LabelDefinition(id="customs", name="Gümrük", description="Gümrük")],
        ),
    )
    binding = LabelingProviderBinding(
        model_configuration_id=model.id,
        provider_id=model.llm_provider.id,
        project="intake-quote-test",
        location="global",
        authentication_mode=VertexAuthenticationMode.WORKLOAD_IDENTITY,
        credential_identity="fixture",
        model="gemini-3.5-flash",
        transport="vertex_direct_v1",
    )
    monkeypatch.setattr(
        intake, "resolve_intake_labeling_provider_binding", lambda *_a, **_k: binding
    )
    monkeypatch.setattr(
        labeling, "get_label_settings", lambda *_a: SimpleNamespace(taxonomy=taxonomy)
    )
    token = intake.claim_intake(intake_session, batch.id)
    assert token is not None
    for member in members:
        intake.mark_intake_file_prepared(intake_session, batch.id, token, member.id)
    run_id = intake.start_intake_labeling(intake_session, batch.id, token)
    if case == "ordinary":
        batch.labeling_run_id = None
    intake_session.flush()
    canonical_before = {
        chunk.id: (chunk.text, deepcopy(chunk.chunk_metadata), list(chunk.heading_path))
        for chunk in chunks
    }
    store, sdk = _Store(), MagicMock()

    def generate(**kwargs: Any) -> genai_types.GenerateContentResponse:
        payload = json.loads(kwargs["contents"][0].parts[0].text)
        source_id = payload["target"]["chunk_id"]
        quote = (
            (
                "Teknik belge"
                if case == "ambiguous"
                else "ürünün mali belgesini"
                if case == "substantive"
                else "ürünün teknik belgesini"
            )
            if source_id == main.id
            else annex.text
        )
        return genai_types.GenerateContentResponse.model_validate(
            {
                "candidates": [
                    {
                        "finishReason": "STOP",
                        "content": {
                            "role": "model",
                            "parts": [
                                {
                                    "text": json.dumps(
                                        {
                                            "labels": [
                                                {
                                                    "label_id": "customs",
                                                    "evidence_quote": quote,
                                                }
                                            ],
                                            "abstained": False,
                                        },
                                        ensure_ascii=False,
                                    )
                                }
                            ],
                        },
                    }
                ]
            }
        )

    sdk.models.generate_content.side_effect = generate
    gateway = vertex_direct.LabelingVertexDirectGateway(
        config=VertexBatchConfig(
            model_configuration_id=model.id,
            model_name=binding.model,
            project=binding.project,
            location=binding.location,
            authentication_mode=binding.authentication_mode,
        ),
        binding_sha256=binding.fingerprint,
        credential_identity=binding.credential_identity,
        credential_json_provider=lambda: None,
        credentials_factory=lambda: cast(Credentials, object()),
        client_factory=lambda _credentials: sdk,
        store=cast(FileStore, store),
        flow=LLMFlow.REGULATORY_LABELING_BATCH,
    )

    @contextmanager
    def session_scope() -> Iterator[Session]:
        yield intake_session

    @contextmanager
    def traced(**_kwargs: Any) -> Iterator[Any]:
        yield SimpleNamespace(span_data=SimpleNamespace(usage=None))

    monkeypatch.setattr(orchestrator, "get_session_with_current_tenant", session_scope)
    monkeypatch.setattr(
        orchestrator, "resolve_labeling_gateway", lambda *_a, **_k: gateway
    )
    monkeypatch.setattr(vertex_direct, "traced_llm_call", traced)
    resolver = intake_quote_resolution.resolve_intake_labeling_response
    if case == "normal-retry":
        monkeypatch.setattr(
            intake_quote_resolution,
            "resolve_intake_labeling_response",
            lambda response, *, text, taxonomy: (
                validate_labeling_response(response, text=text, taxonomy=taxonomy),
                [],
            ),
        )

    def drive(identifier: UUID, *, stop_after_ack: bool = False) -> None:
        for _step in range(20):
            intake_session.expire_all()
            current = intake_session.get_one(RegulatoryLabelingRun, identifier)
            if current.status not in {"queued", "running"}:
                return
            shards = list(
                intake_session.scalars(
                    select(RegulatoryLabelingShard).where(
                        RegulatoryLabelingShard.run_id == identifier
                    )
                )
            )
            if (
                stop_after_ack
                and shards
                and all(
                    shard.status == "submitted" and shard.output_uri for shard in shards
                )
            ):
                return
            orchestrator.run_labeling_step(
                identifier, current.lease_generation, "public"
            )
        pytest.fail("The local fake-provider labeling run did not finish")

    drive(run_id, stop_after_ack=True)
    assert sdk.models.generate_content.call_count == 2
    raw_receipts = deepcopy(store.files)
    original_items = list(
        intake_session.scalars(
            select(RegulatoryLabelingItem).where(
                RegulatoryLabelingItem.run_id == run_id
            )
        )
    )
    requests_before = {
        item.id: (
            item.request_hash,
            deepcopy(item.request_payload),
            item.context_sha256,
        )
        for item in original_items
    }
    drive(run_id)
    assert store.files == raw_receipts
    assert sdk.models.generate_content.call_count == 2
    for item in original_items:
        assert (
            item.request_hash,
            item.request_payload,
            item.context_sha256,
        ) == requests_before[item.id]
    main_item = next(
        item for item in original_items if item.regulatory_chunk_id == main.id
    )
    annex_item = next(
        item for item in original_items if item.regulatory_chunk_id == annex.id
    )
    assert annex_item.status == "completed"
    assert main_item.status == ("completed" if case == "current" else "failed")
    if case == "normal-retry":
        good_before = deepcopy(annex_item.assignments)
        monkeypatch.setattr(
            intake_quote_resolution, "resolve_intake_labeling_response", resolver
        )
        intake.refresh_intake(intake_session, batch)
        assert batch.status == "partial"
        intake.retry_intake(intake_session, batch)
        retry_token = intake.claim_intake(intake_session, batch.id)
        assert retry_token is not None
        for member in members:
            intake.mark_intake_file_prepared(
                intake_session, batch.id, retry_token, member.id
            )
        replacement_id = intake.start_intake_labeling(
            intake_session, batch.id, retry_token
        )
        assert replacement_id != run_id
        drive(replacement_id)
        assert sdk.models.generate_content.call_count == 3
        replacement = list(
            intake_session.scalars(
                select(RegulatoryLabelingItem).where(
                    RegulatoryLabelingItem.run_id == replacement_id
                )
            )
        )
        assert all(item.status == "completed" for item in replacement)
        assert (
            next(
                item for item in replacement if item.regulatory_chunk_id == annex.id
            ).assignments
            == good_before
        )
        assert all(store.files[key] == value for key, value in raw_receipts.items())
        main_item = next(
            item for item in replacement if item.regulatory_chunk_id == main.id
        )
    if case in {"current", "normal-retry"}:
        assert main_item.assignments == [
            {"label_id": "customs", "evidence_quote": "ürünün teknik\nbelgesini"}
        ]
        LabelAssignment.model_validate(main_item.assignments[0])
        projection = intake_session.scalar(
            select(RegulatoryDerivedLabelProjection).where(
                RegulatoryDerivedLabelProjection.run_id == main_item.run_id,
                RegulatoryDerivedLabelProjection.regulatory_chunk_id == derived.id,
            )
        )
        assert projection is not None and projection.resolution == "lineage"
        assert projection.labels == ["customs"]
        provenance = cast(list[dict[str, Any]], projection.provenance["customs"])
        assert provenance[0]["canonical_chunk_id"] == main.id
        assert provenance[0]["assignments"] == main_item.assignments
    assert {
        chunk.id: (chunk.text, chunk.chunk_metadata, chunk.heading_path)
        for chunk in chunks
    } == canonical_before
