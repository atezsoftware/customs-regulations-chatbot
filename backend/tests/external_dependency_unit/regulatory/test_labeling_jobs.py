"""Committed labeling jobs use an owned PostgreSQL schema and fake provider data."""

import datetime
import json
import os
import subprocess
import sys
from collections.abc import Generator, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from threading import Barrier
from typing import NamedTuple, cast
from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import Engine, delete, select, text
from sqlalchemy.orm import Session

from onyx.auth.users import current_user
from onyx.db import labeling_configuration
from onyx.db import regulatory_labeling as repository
from onyx.db.engine.sql_engine import SqlEngine, get_session
from onyx.db.enums import Permission
from onyx.db.labeling_configuration import resolve_labeling_provider_binding
from onyx.db.models import (
    DocumentSet,
    DocumentSet__UserFile,
    LLMProvider,
    ModelConfiguration,
    RegulatoryChunk,
    RegulatoryDerivedLabelProjection,
    RegulatoryLabelingItem,
    RegulatoryLabelingRun,
    RegulatoryLabelingShard,
    RegulatoryLabelTaxonomy,
    User,
    UserFile,
    UserRole,
)
from onyx.error_handling.exceptions import register_onyx_exception_handlers
from onyx.llm.constants import LlmProviderNames
from onyx.llm.well_known_providers.constants import (
    VERTEX_AUTH_METHOD_KWARG,
    VERTEX_AUTH_METHOD_SERVICE_ACCOUNT,
    VERTEX_CREDENTIALS_FILE_KWARG,
)
from onyx.regulatory.indexing_jobs.models import (
    IndexingGatewayConnectionError,
    IndexingGatewayIndeterminateSubmissionError,
)
from onyx.regulatory.indexing_jobs.vertex_batch import (
    VertexBatchJobStatus,
    VertexBatchRequest,
    VertexBatchState,
)
from onyx.regulatory.labeling import orchestrator
from onyx.regulatory.labeling.provider import (
    TaxonomyDefinition,
    build_labeling_request,
)
from onyx.server.features.document_set import labeling_api
from shared_configs.contextvars import (
    CURRENT_TENANT_ID_CONTEXTVAR,
    get_current_tenant_id,
)

_FIRST_TEXT = "Importers must declare the customs value of all imported goods."
_SECOND_TEXT = "Exporters must retain the certificate of origin for five years."
_ORIGINAL_WAKE = labeling_api._wake


class LabelingDatabase(NamedTuple):
    schema: str
    engine: Engine


class LabelingData(NamedTuple):
    database: LabelingDatabase
    document_set_id: int
    other_document_set_id: int
    user_id: UUID
    file_id: UUID
    provider_id: int
    model_id: int
    taxonomy_id: UUID
    first_id: str
    second_id: str
    derived_id: str
    incomplete_id: str


@pytest.fixture(scope="module")
def labeling_database() -> Generator[LabelingDatabase, None, None]:
    schema = f"tenant_labeling_tests_{uuid4().hex}"
    backend = Path(__file__).resolve().parents[3]
    SqlEngine.init_engine(pool_size=10, max_overflow=5)
    base_engine = SqlEngine.get_engine()
    with base_engine.connect() as connection:
        extensions = set(connection.scalars(text("SELECT extname FROM pg_extension")))
    assert {"pgcrypto", "pg_trgm"} <= extensions, (
        "Labeling tests require a migrated test PostgreSQL database with pgcrypto "
        "and pg_trgm already installed; the fixture does not create shared extensions."
    )
    try:
        migration = subprocess.run(
            [
                sys.executable,
                "-m",
                "alembic",
                "-x",
                f"schemas={schema}",
                "upgrade",
                "head",
            ],
            cwd=backend,
            env={**os.environ, "MULTI_TENANT": "true"},
            text=True,
            capture_output=True,
            check=False,
        )
        assert migration.returncode == 0, migration.stderr
        engine = base_engine.execution_options(schema_translate_map={None: schema})
        yield LabelingDatabase(schema, engine)
    finally:
        with base_engine.begin() as connection:
            connection.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))


@pytest.fixture
def labeling_data(
    labeling_database: LabelingDatabase,
) -> Generator[LabelingData, None, None]:
    token = CURRENT_TENANT_ID_CONTEXTVAR.set(labeling_database.schema)
    with Session(labeling_database.engine) as session:
        user = User(
            id=uuid4(),
            email=f"labeling-{uuid4().hex}@test.local",
            hashed_password="unused",
            role=UserRole.ADMIN,
            effective_permissions=[Permission.FULL_ADMIN_PANEL_ACCESS.value],
            is_active=True,
            is_verified=True,
            is_superuser=False,
        )
        session.add(user)
        session.flush()
        document_set = DocumentSet(
            name=f"labeling-set-{uuid4().hex}", user_id=user.id, is_public=False
        )
        other_set = DocumentSet(name=f"labeling-other-{uuid4().hex}", is_public=False)
        session.add_all([document_set, other_set])
        session.flush()
        file = UserFile(
            id=uuid4(),
            user_id=user.id,
            file_id=str(uuid4()),
            name="Regulation",
            file_type="text/markdown",
        )
        session.add(file)
        session.flush()
        session.add(
            DocumentSet__UserFile(document_set_id=document_set.id, user_file_id=file.id)
        )
        first_id, second_id, derived_id, incomplete_id = [
            str(uuid4()) for _ in range(4)
        ]
        for position, identifier, content, metadata in (
            (0, first_id, _FIRST_TEXT, {"chunk_variant": "atomic"}),
            (1, second_id, _SECOND_TEXT, {"chunk_variant": "atomic"}),
            (
                2,
                derived_id,
                _FIRST_TEXT + "\n" + _SECOND_TEXT,
                {
                    "chunk_variant": "hierarchical_aggregate",
                    "source_regulatory_chunk_ids": [first_id, second_id],
                },
            ),
            (
                3,
                incomplete_id,
                "A derived chunk with unavailable canonical lineage.",
                {
                    "chunk_variant": "hierarchical_aggregate",
                    "source_regulatory_chunk_ids": [first_id, "missing-source-id"],
                },
            ),
        ):
            session.add(
                RegulatoryChunk(
                    id=identifier,
                    user_file_id=file.id,
                    text=content,
                    position=position,
                    projection_ordinal=position,
                    heading_path=["Customs regulation", f"Article {position}"],
                    chunk_type="article",
                    chunk_metadata=metadata,
                    source="indexed",
                    status="active",
                )
            )
        provider = LLMProvider(
            name=f"labeling-provider-{uuid4().hex}",
            provider=LlmProviderNames.VERTEX_AI,
            is_public=True,
            custom_config={
                VERTEX_AUTH_METHOD_KWARG: VERTEX_AUTH_METHOD_SERVICE_ACCOUNT,
                VERTEX_CREDENTIALS_FILE_KWARG: json.dumps(
                    {
                        "project_id": "labeling-test",
                        "client_email": "labeler@example.test",
                        "private_key": "fake-test-key-never-sent",
                    }
                ),
            },
        )
        model = ModelConfiguration(
            llm_provider=provider, name="gemini-3.8-flash", is_visible=True
        )
        session.add(model)
        taxonomy = repository.create_taxonomy(
            session,
            created_by_id=user.id,
            taxonomy=TaxonomyDefinition.model_validate(
                {
                    "name": f"Labeling taxonomy {uuid4().hex}",
                    "labels": [
                        {
                            "id": "customs_value",
                            "name": "Customs value",
                            "description": "A customs value declaration requirement.",
                        },
                        {
                            "id": "origin",
                            "name": "Origin",
                            "description": "A certificate of origin requirement.",
                        },
                    ],
                }
            ),
        )
        session.commit()
        fixture = LabelingData(
            labeling_database,
            document_set.id,
            other_set.id,
            user.id,
            file.id,
            provider.id,
            model.id,
            taxonomy.id,
            first_id,
            second_id,
            derived_id,
            incomplete_id,
        )
    try:
        yield fixture
    finally:
        with Session(labeling_database.engine) as session:
            session.execute(
                delete(DocumentSet).where(
                    DocumentSet.id.in_(
                        [fixture.document_set_id, fixture.other_document_set_id]
                    )
                )
            )
            session.execute(delete(UserFile).where(UserFile.id == fixture.file_id))
            session.execute(
                delete(RegulatoryLabelTaxonomy).where(
                    RegulatoryLabelTaxonomy.id == fixture.taxonomy_id
                )
            )
            session.execute(
                delete(LLMProvider).where(LLMProvider.id == fixture.provider_id)
            )
            user = session.get(User, fixture.user_id)
            assert user is not None
            session.delete(user)
            session.commit()
        CURRENT_TENANT_ID_CONTEXTVAR.reset(token)


def _start(data: LabelingData, *, key: UUID | None = None) -> tuple[UUID, bool]:
    with Session(data.database.engine) as session:
        user = session.get(User, data.user_id)
        taxonomy = session.get(RegulatoryLabelTaxonomy, data.taxonomy_id)
        assert user is not None and taxonomy is not None
        binding = resolve_labeling_provider_binding(session, data.model_id, user=user)
        run, created = repository.create_labeling_run(
            session,
            document_set_id=data.document_set_id,
            taxonomy=taxonomy,
            model_configuration_id=data.model_id,
            model="gemini-3.8-flash",
            provider_binding=binding.model_dump(mode="json"),
            requested_by_id=user.id,
            idempotency_key=key or uuid4(),
        )
        session.commit()
        return run.id, created


def _prepare(data: LabelingData, run_id: UUID) -> tuple[repository.RunLease, UUID]:
    with Session(data.database.engine) as session:
        lease = repository.claim_run(
            session, run_id=run_id, expected_generation=0, lease_seconds=300
        )
        assert lease is not None
        run = repository.load_claimed_run(session, lease)
        taxonomy = TaxonomyDefinition.model_validate(run.taxonomy.definition)
        items = repository.prepare_next_item_page(session, lease, limit=10)
        requests = []
        for item in items:
            assert item.context_snapshot is not None
            assert item.canonical_text_sha256 is not None
            request = build_labeling_request(
                chunk_id=item.regulatory_chunk_id,
                text=item.text_snapshot,
                context=item.context_snapshot,
                taxonomy=taxonomy,
                source_hash=item.canonical_text_sha256,
            )
            requests.append(
                repository.PreparedRequest(
                    item_id=item.id,
                    request_hash=request.request_hash,
                    request_payload=request.model_dump(
                        mode="json", exclude={"request_hash"}
                    ),
                )
            )
        repository.store_prepared_shards(
            session,
            lease,
            requests=requests,
            failed_items={},
            shards=[
                repository.PreparedShard(
                    ordinal=0,
                    item_ids=tuple(item.id for item in items),
                    submission_key="regulatory-labeling-" + uuid4().hex + uuid4().hex,
                )
            ],
        )
        shard = repository.next_due_shard(session, lease, max_in_flight=2)
        assert shard is not None
        repository.mark_shard_submitting(
            session, lease, shard_id=shard.id, reconcile_seconds=300
        )
        repository.record_shard_state(
            session,
            lease,
            shard_id=shard.id,
            status="submitted",
            remote_job_name="batches/test",
        )
        session.commit()
        return lease, shard.id


def _outcomes(
    session: Session, lease: repository.RunLease, shard_id: UUID
) -> dict[str, tuple[list[str], list[dict[str, object]]]]:
    result: dict[str, tuple[list[str], list[dict[str, object]]]] = {}
    for item in repository.load_shard_requests(session, lease, shard_id):
        assert item.request_hash is not None
        label = "customs_value" if item.text_snapshot == _FIRST_TEXT else "origin"
        result[item.request_hash] = (
            [label],
            [{"label_id": label, "evidence_quote": item.text_snapshot}],
        )
    return result


def test_start_is_idempotent_and_reads_are_document_set_scoped(
    labeling_data: LabelingData,
) -> None:
    key = uuid4()
    run_id, created = _start(labeling_data, key=key)
    assert created
    assert _start(labeling_data, key=key) == (run_id, False)
    with pytest.raises(repository.LabelingStateConflictError, match="already active"):
        _start(labeling_data)
    with Session(labeling_data.database.engine) as session:
        counts, _ = repository.get_labeling_counts(
            session, labeling_data.document_set_id
        )
        assert (counts.files, counts.canonical_chunks, counts.derived_chunks) == (
            1,
            2,
            2,
        )
        assert (
            repository.get_run(
                session,
                document_set_id=labeling_data.other_document_set_id,
                run_id=run_id,
            )
            is None
        )
        assert (
            repository.list_items(
                session,
                document_set_id=labeling_data.other_document_set_id,
                run_id=run_id,
                offset=0,
                limit=10,
            )
            is None
        )
        assert (
            repository.request_cancellation(
                session,
                document_set_id=labeling_data.other_document_set_id,
                run_id=run_id,
            )
            is None
        )
        items = list(
            session.scalars(
                select(RegulatoryLabelingItem).where(
                    RegulatoryLabelingItem.run_id == run_id
                )
            )
        )
        assert {item.regulatory_chunk_id for item in items} == {
            labeling_data.first_id,
            labeling_data.second_id,
        }
        assert {item.text_snapshot for item in items} == {_FIRST_TEXT, _SECOND_TEXT}


@pytest.mark.parametrize("same_key", [True, False])
def test_concurrent_start_creates_one_run_and_one_atomic_snapshot(
    labeling_data: LabelingData, same_key: bool
) -> None:
    shared_key = uuid4()
    barrier = Barrier(2)

    def start() -> tuple[UUID, bool] | str:
        barrier.wait(timeout=5)
        try:
            return _start(labeling_data, key=shared_key if same_key else uuid4())
        except repository.LabelingStateConflictError:
            return "active"

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: start(), range(2)))
    successful = [result for result in results if isinstance(result, tuple)]
    assert sum(created for _, created in successful) == 1
    if same_key:
        assert len(successful) == 2
        assert successful[0][0] == successful[1][0]
    else:
        assert len(successful) == 1
        assert results.count("active") == 1
    with Session(labeling_data.database.engine) as session:
        runs = repository.list_runs(session, labeling_data.document_set_id)
        assert len(runs) == 1
        items = list(
            session.scalars(
                select(RegulatoryLabelingItem).where(
                    RegulatoryLabelingItem.run_id == runs[0].id
                )
            )
        )
        assert len(items) == 2


def test_malformed_legacy_lineage_does_not_become_a_canonical_target(
    labeling_data: LabelingData,
) -> None:
    with Session(labeling_data.database.engine) as session:
        derived = session.get(RegulatoryChunk, labeling_data.incomplete_id)
        assert derived is not None
        derived.chunk_metadata = {"source_regulatory_chunk_ids": "invalid-old-lineage"}
        session.commit()
    run_id, _ = _start(labeling_data)
    with Session(labeling_data.database.engine) as session:
        items = list(
            session.scalars(
                select(RegulatoryLabelingItem).where(
                    RegulatoryLabelingItem.run_id == run_id
                )
            )
        )
        assert {item.regulatory_chunk_id for item in items} == {
            labeling_data.first_id,
            labeling_data.second_id,
        }


def test_partial_provider_output_marks_missing_item_and_incomplete_union_explicitly(
    labeling_data: LabelingData,
) -> None:
    run_id, _ = _start(labeling_data)
    lease, shard_id = _prepare(labeling_data, run_id)
    with Session(labeling_data.database.engine) as session:
        outcomes = _outcomes(session, lease, shard_id)
        first = session.scalar(
            select(RegulatoryLabelingItem).where(
                RegulatoryLabelingItem.run_id == run_id,
                RegulatoryLabelingItem.regulatory_chunk_id == labeling_data.first_id,
            )
        )
        assert first is not None and first.request_hash is not None
        repository.apply_shard_results(
            session,
            lease,
            shard_id=shard_id,
            outcomes={first.request_hash: outcomes[first.request_hash]},
        )
        repository.project_derived_labels(session, lease)
        assert repository.final_status(session, lease) == "completed_with_errors"
        session.commit()
    with Session(labeling_data.database.engine) as session:
        run = session.get(RegulatoryLabelingRun, run_id)
        assert run is not None
        assert (run.completed_chunks, run.failed_chunks, run.stale_chunks) == (1, 1, 0)
        projections = list(
            session.scalars(
                select(RegulatoryDerivedLabelProjection).where(
                    RegulatoryDerivedLabelProjection.run_id == run_id
                )
            )
        )
        assert len(projections) == 2
        assert all(
            row.labels == [] and row.resolution == "unresolved" for row in projections
        )
        assert {row.unresolved_reason for row in projections} == {
            "source_label_result_unavailable",
            "missing_explicit_dependency",
        }


def test_completed_canonical_labels_form_derived_union_with_evidence_provenance(
    labeling_data: LabelingData,
) -> None:
    run_id, _ = _start(labeling_data)
    lease, shard_id = _prepare(labeling_data, run_id)
    with Session(labeling_data.database.engine) as session:
        repository.apply_shard_results(
            session,
            lease,
            shard_id=shard_id,
            outcomes=_outcomes(session, lease, shard_id),
        )
        assert repository.project_derived_labels(session, lease) == 2
        session.commit()
    with Session(labeling_data.database.engine) as session:
        projection = session.scalar(
            select(RegulatoryDerivedLabelProjection).where(
                RegulatoryDerivedLabelProjection.run_id == run_id,
                RegulatoryDerivedLabelProjection.regulatory_chunk_id
                == labeling_data.derived_id,
            )
        )
        assert projection is not None
        assert projection.labels == ["customs_value", "origin"]
        assert projection.resolution == "lineage"
        customs_provenance = cast(
            list[dict[str, object]], projection.provenance["customs_value"]
        )
        origin_provenance = cast(
            list[dict[str, object]], projection.provenance["origin"]
        )
        assert customs_provenance[0]["canonical_chunk_id"] == labeling_data.first_id
        assert origin_provenance[0]["canonical_chunk_id"] == labeling_data.second_id
        source = session.get(RegulatoryChunk, labeling_data.first_id)
        assert source is not None and source.text == _FIRST_TEXT


@pytest.mark.parametrize("drift", ["target", "neighbor", "detach"])
def test_frozen_source_and_context_drift_rejects_result_application(
    labeling_data: LabelingData, drift: str
) -> None:
    run_id, _ = _start(labeling_data)
    lease, shard_id = _prepare(labeling_data, run_id)
    with Session(labeling_data.database.engine) as session:
        if drift == "detach":
            session.execute(
                delete(DocumentSet__UserFile).where(
                    DocumentSet__UserFile.document_set_id
                    == labeling_data.document_set_id
                )
            )
        else:
            changed_id = (
                labeling_data.first_id if drift == "target" else labeling_data.second_id
            )
            chunk = session.get(RegulatoryChunk, changed_id)
            assert chunk is not None
            chunk.text = "Changed canonical legal requirement."
        session.commit()
    with Session(labeling_data.database.engine) as session:
        repository.apply_shard_results(
            session,
            lease,
            shard_id=shard_id,
            outcomes=_outcomes(session, lease, shard_id),
        )
        session.commit()
    with Session(labeling_data.database.engine) as session:
        items = list(
            session.scalars(
                select(RegulatoryLabelingItem).where(
                    RegulatoryLabelingItem.run_id == run_id
                )
            )
        )
        assert {item.status for item in items} == {"stale"}
        assert all(item.labels == [] for item in items)
        assert {item.text_snapshot for item in items} == {_FIRST_TEXT, _SECOND_TEXT}


def test_crashed_worker_lease_is_recovered_once_and_stale_result_is_fenced(
    labeling_data: LabelingData,
) -> None:
    run_id, _ = _start(labeling_data)
    lease, shard_id = _prepare(labeling_data, run_id)
    now = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=301)
    with Session(labeling_data.database.engine) as session:
        recoverable = repository.recoverable_runs(session, now=now)
        assert run_id in {row.run_id for row in recoverable}

    def claim() -> repository.RunLease | None:
        with Session(labeling_data.database.engine) as session:
            claimed = repository.claim_run(
                session,
                run_id=run_id,
                expected_generation=lease.generation,
                lease_seconds=300,
                now=now,
            )
            session.commit()
            return claimed

    with ThreadPoolExecutor(max_workers=2) as executor:
        claims = list(executor.map(lambda _: claim(), range(2)))
    assert len([claim for claim in claims if claim is not None]) == 1
    with Session(labeling_data.database.engine) as session:
        with pytest.raises(repository.LabelingStateConflictError, match="lease"):
            repository.apply_shard_results(
                session, lease, shard_id=shard_id, outcomes={}
            )
        session.rollback()
        items = list(
            session.scalars(
                select(RegulatoryLabelingItem).where(
                    RegulatoryLabelingItem.run_id == run_id
                )
            )
        )
        assert {item.status for item in items} == {"submitted"}


def test_cancel_request_prevents_late_provider_results_becoming_completed(
    labeling_data: LabelingData,
) -> None:
    run_id, _ = _start(labeling_data)
    lease, shard_id = _prepare(labeling_data, run_id)
    with Session(labeling_data.database.engine) as session:
        repository.request_cancellation(
            session, document_set_id=labeling_data.document_set_id, run_id=run_id
        )
        session.commit()
    with Session(labeling_data.database.engine) as session:
        outcomes = _outcomes(session, lease, shard_id)
        try:
            repository.apply_shard_results(
                session, lease, shard_id=shard_id, outcomes=outcomes
            )
        except repository.LabelingStateConflictError:
            session.rollback()
        else:
            session.commit()
        items = list(
            session.scalars(
                select(RegulatoryLabelingItem).where(
                    RegulatoryLabelingItem.run_id == run_id
                )
            )
        )
        assert all(item.status != "completed" and item.labels == [] for item in items)


def test_derived_projection_does_not_adopt_results_after_source_changes(
    labeling_data: LabelingData,
) -> None:
    run_id, _ = _start(labeling_data)
    lease, shard_id = _prepare(labeling_data, run_id)
    with Session(labeling_data.database.engine) as session:
        repository.apply_shard_results(
            session,
            lease,
            shard_id=shard_id,
            outcomes=_outcomes(session, lease, shard_id),
        )
        session.commit()
    with Session(labeling_data.database.engine) as session:
        chunk = session.get(RegulatoryChunk, labeling_data.first_id)
        assert chunk is not None
        chunk.text = "The former declaration requirement has been repealed."
        session.commit()
    with Session(labeling_data.database.engine) as session:
        repository.project_derived_labels(session, lease)
        session.commit()
        projection = session.scalar(
            select(RegulatoryDerivedLabelProjection).where(
                RegulatoryDerivedLabelProjection.run_id == run_id,
                RegulatoryDerivedLabelProjection.regulatory_chunk_id
                == labeling_data.derived_id,
            )
        )
        assert projection is not None
        assert projection.resolution == "unresolved"
        assert projection.labels == []


@pytest.fixture
def labeling_client(
    labeling_data: LabelingData, monkeypatch: pytest.MonkeyPatch
) -> Generator[TestClient, None, None]:
    def session_dependency() -> Generator[Session, None, None]:
        with Session(labeling_data.database.engine) as session:
            yield session

    def user_dependency() -> User:
        with Session(labeling_data.database.engine) as session:
            user = session.get(User, labeling_data.user_id)
            assert user is not None
            session.expunge(user)
            return user

    app = FastAPI()
    register_onyx_exception_handlers(app)
    app.include_router(labeling_api.router)
    app.dependency_overrides[get_session] = session_dependency
    app.dependency_overrides[current_user] = user_dependency
    app.dependency_overrides[get_current_tenant_id] = lambda: (
        labeling_data.database.schema
    )
    monkeypatch.setattr(labeling_api, "_wake", lambda *_args: None)
    with TestClient(app) as client:
        yield client


def _api_path(
    data: LabelingData, suffix: str, *, document_set_id: int | None = None
) -> str:
    set_id = document_set_id if document_set_id is not None else data.document_set_id
    return f"/manage/admin/document-set/{set_id}/labeling/{suffix}"


def _start_body(data: LabelingData) -> dict[str, str | int]:
    return {
        "taxonomy_id": str(data.taxonomy_id),
        "model_configuration_id": data.model_id,
        "idempotency_key": str(uuid4()),
    }


@pytest.fixture
def restore_label_settings(labeling_data: LabelingData) -> Generator[None, None, None]:
    with Session(labeling_data.database.engine) as session:
        original = repository.get_label_settings(session)
        original_labels = TaxonomyDefinition.model_validate(
            original.taxonomy.definition
        ).labels
        original_taxonomy_id = original.taxonomy_id
    try:
        yield
    finally:
        with Session(labeling_data.database.engine) as session:
            current = repository.get_label_settings(session)
            if current.taxonomy_id != original_taxonomy_id:
                repository.update_label_settings(
                    session,
                    labels=original_labels,
                    expected_revision=current.revision,
                    updated_by_id=labeling_data.user_id,
                )
                session.commit()


def test_api_setup_and_start_expose_only_safe_persisted_state(
    labeling_data: LabelingData, labeling_client: TestClient
) -> None:
    response = labeling_client.get(_api_path(labeling_data, "setup"))
    assert response.status_code == 200, response.text
    setup = response.json()
    assert setup["model"] == "gemini-3.8-flash"
    assert setup["counts"] == {"files": 1, "canonical_chunks": 2, "derived_chunks": 2}
    assert setup["providers"] == [
        {"id": labeling_data.model_id, "name": setup["providers"][0]["name"]}
    ]
    assert setup["taxonomies"][0]["id"] == str(labeling_data.taxonomy_id)
    assert setup["active_run_id"] is None
    assert "fake-test-key" not in response.text
    body = _start_body(labeling_data)
    first = labeling_client.post(_api_path(labeling_data, "runs"), json=body)
    assert first.status_code == 200, first.text
    second = labeling_client.post(_api_path(labeling_data, "runs"), json=body)
    assert second.status_code == 200, second.text
    assert first.json()["id"] == second.json()["id"]
    assert first.json()["total_chunks"] == 2
    assert first.json()["status"] == "queued"
    assert "provider_binding" not in first.json()
    assert "fake-test-key" not in first.text
    page = labeling_client.get(
        _api_path(labeling_data, f"runs/{first.json()['id']}/items")
    )
    assert page.status_code == 200, page.text
    assert page.json()["total"] == 2


@pytest.mark.usefixtures("restore_label_settings")
def test_api_label_settings_save_drive_setup_and_new_batch_prompts(
    labeling_data: LabelingData,
    labeling_client: TestClient,
) -> None:
    path = _api_path(labeling_data, "label-settings")
    original = labeling_client.get(path)
    assert original.status_code == 200, original.text
    original_body = original.json()
    labels = [
        {
            "id": "editable-customs-value",
            "name": "Düzenlenebilir gümrük kıymeti",
            "description": "Gümrük kıymetini düzenleyen hükümler.",
        },
        {
            "id": "editable-origin",
            "name": "Düzenlenebilir menşe",
            "description": "Menşe ispatını düzenleyen hükümler.",
        },
    ]
    saved = labeling_client.put(
        path,
        json={"expected_revision": original_body["revision"], "labels": labels},
    )
    assert saved.status_code == 200, saved.text
    assert saved.json()["revision"] == original_body["revision"] + 1
    assert saved.json()["taxonomy_id"] != original_body["taxonomy_id"]
    assert saved.json()["labels"] == labels
    assert labeling_client.get(path).json() == saved.json()
    assert (
        labeling_client.get(_api_path(labeling_data, "setup")).json()[
            "default_label_count"
        ]
        == 2
    )

    body = _start_body(labeling_data)
    del body["taxonomy_id"]
    started = labeling_client.post(_api_path(labeling_data, "runs"), json=body)
    assert started.status_code == 200, started.text
    lease, shard_id = _prepare(labeling_data, UUID(started.json()["id"]))
    with Session(labeling_data.database.engine) as session:
        items = repository.load_shard_requests(session, lease, shard_id)
        assert items
        for item in items:
            assert item.request_payload is not None
            request = VertexBatchRequest.model_validate(item.request_payload)
            assert json.loads(request.prompt)["taxonomy"]["labels"] == labels


@pytest.mark.usefixtures("restore_label_settings")
def test_api_label_settings_reject_stale_revision_without_losing_saved_edit(
    labeling_data: LabelingData,
    labeling_client: TestClient,
) -> None:
    path = _api_path(labeling_data, "label-settings")
    original = labeling_client.get(path).json()
    first_labels = [{"id": "winner", "name": "Kazanan", "description": "İlk kayıt."}]
    saved = labeling_client.put(
        path,
        json={"expected_revision": original["revision"], "labels": first_labels},
    )
    assert saved.status_code == 200, saved.text
    stale = labeling_client.put(
        path,
        json={
            "expected_revision": original["revision"],
            "labels": [
                {"id": "loser", "name": "Kaybeden", "description": "Eski kayıt."}
            ],
        },
    )
    assert stale.status_code == 409, stale.text
    assert labeling_client.get(path).json() == saved.json()


@pytest.mark.usefixtures("restore_label_settings")
def test_api_implicit_replay_keeps_the_original_labels_after_settings_change(
    labeling_data: LabelingData,
    labeling_client: TestClient,
) -> None:
    settings_path = _api_path(labeling_data, "label-settings")
    original_settings = labeling_client.get(settings_path).json()
    body = _start_body(labeling_data)
    del body["taxonomy_id"]
    runs_path = _api_path(labeling_data, "runs")
    started = labeling_client.post(runs_path, json=body)
    assert started.status_code == 200, started.text
    original_taxonomy_id = started.json()["taxonomy_id"]

    saved = labeling_client.put(
        settings_path,
        json={
            "expected_revision": original_settings["revision"],
            "labels": [
                {
                    "id": "future-only",
                    "name": "Yalnız gelecek işler",
                    "description": "Bu değişiklik yalnız yeni işleri etkiler.",
                }
            ],
        },
    )
    assert saved.status_code == 200, saved.text
    assert saved.json()["taxonomy_id"] != original_taxonomy_id
    replayed = labeling_client.post(runs_path, json=body)
    assert replayed.status_code == 200, replayed.text
    assert replayed.json()["id"] == started.json()["id"]
    assert replayed.json()["taxonomy_id"] == original_taxonomy_id
    lease, shard_id = _prepare(labeling_data, UUID(started.json()["id"]))
    with Session(labeling_data.database.engine) as session:
        items = repository.load_shard_requests(session, lease, shard_id)
        assert items
        for item in items:
            assert item.request_payload is not None
            request = VertexBatchRequest.model_validate(item.request_payload)
            assert (
                json.loads(request.prompt)["taxonomy"]["labels"]
                == (original_settings["labels"])
            )


def test_api_label_settings_validate_bounds_and_unknown_fields(
    labeling_data: LabelingData,
    labeling_client: TestClient,
) -> None:
    path = _api_path(labeling_data, "label-settings")
    revision = labeling_client.get(path).json()["revision"]
    label = {"id": "one", "name": "Bir", "description": "Bir etiket."}
    for body in (
        {"expected_revision": 0, "labels": [label]},
        {"expected_revision": revision, "labels": []},
        {"expected_revision": revision, "labels": [label], "extra": True},
        {
            "expected_revision": revision,
            "labels": [
                {
                    "id": f"label-{index}",
                    "name": f"Etiket {index}",
                    "description": "Tanım.",
                }
                for index in range(1025)
            ],
        },
    ):
        response = labeling_client.put(path, json=body)
        assert response.status_code == 422, response.text


@pytest.mark.parametrize("explicit_null", [False, True])
def test_api_default_labels_need_no_upload_and_reach_the_batch_prompt(
    labeling_data: LabelingData, labeling_client: TestClient, explicit_null: bool
) -> None:
    with Session(labeling_data.database.engine) as session:
        settings = repository.get_label_settings(session)
        default_taxonomy_id = settings.taxonomy_id
        definition = settings.taxonomy.definition
    setup = labeling_client.get(_api_path(labeling_data, "setup"))
    assert setup.status_code == 200, setup.text
    assert setup.json()["default_label_count"] == 255
    body: dict[str, str | int | None] = {**_start_body(labeling_data)}
    if explicit_null:
        body["taxonomy_id"] = None
    else:
        del body["taxonomy_id"]
    path = _api_path(labeling_data, "runs")
    first = labeling_client.post(path, json=body)
    assert first.status_code == 200, first.text
    repeated = labeling_client.post(path, json=body)
    assert repeated.status_code == 200, repeated.text
    assert repeated.json()["id"] == first.json()["id"]
    explicit = labeling_client.post(
        path, json={**body, "taxonomy_id": first.json()["taxonomy_id"]}
    )
    assert explicit.status_code == 200, explicit.text
    assert explicit.json()["id"] == first.json()["id"]
    changed = labeling_client.post(
        path, json={**body, "taxonomy_id": str(labeling_data.taxonomy_id)}
    )
    assert changed.status_code == 409, changed.text
    with Session(labeling_data.database.engine) as session:
        settings = repository.get_label_settings(session)
        assert settings.taxonomy_id == default_taxonomy_id
        assert settings.taxonomy.created_by_id is None
        assert settings.taxonomy.definition == definition
    lease, shard_id = _prepare(labeling_data, UUID(first.json()["id"]))
    with Session(labeling_data.database.engine) as session:
        items = repository.load_shard_requests(session, lease, shard_id)
        assert len(items) == 2
        for item in items:
            assert item.request_payload is not None
            request = VertexBatchRequest.model_validate(item.request_payload)
            assert json.loads(request.prompt)["taxonomy"] == definition


def test_api_concurrent_default_starts_share_one_definition_and_run(
    labeling_data: LabelingData, labeling_client: TestClient
) -> None:
    with Session(labeling_data.database.engine) as session:
        taxonomy_ids_before = {row.id for row in repository.list_taxonomies(session)}
    body = _start_body(labeling_data)
    del body["taxonomy_id"]
    barrier = Barrier(2)

    def start() -> str:
        barrier.wait(timeout=5)
        response = labeling_client.post(_api_path(labeling_data, "runs"), json=body)
        assert response.status_code == 200, response.text
        return str(response.json()["id"])

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: start(), range(2)))
    assert results[0] == results[1]
    with Session(labeling_data.database.engine) as session:
        assert {row.id for row in repository.list_taxonomies(session)} == (
            taxonomy_ids_before
        )
        assert len(repository.list_runs(session, labeling_data.document_set_id)) == 1


@pytest.mark.parametrize("use_default_labels", [False, True])
def test_api_start_survives_broker_failure_and_replays_after_provider_removal(
    labeling_data: LabelingData,
    labeling_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    use_default_labels: bool,
) -> None:
    from onyx.background.celery.tasks.regulatory_labeling import tasks

    deliveries: list[UUID] = []

    def failed_delivery(*, run_id: UUID, tenant_id: str) -> None:
        assert tenant_id == labeling_data.database.schema
        with Session(labeling_data.database.engine) as session:
            assert session.get(RegulatoryLabelingRun, run_id) is not None
        deliveries.append(run_id)
        raise ConnectionError("Test broker unavailable")

    monkeypatch.setattr(tasks, "enqueue_labeling_run", failed_delivery)
    monkeypatch.setattr(labeling_api, "_wake", _ORIGINAL_WAKE)
    body = _start_body(labeling_data)
    if use_default_labels:
        del body["taxonomy_id"]
    path = _api_path(labeling_data, "runs")
    created = labeling_client.post(path, json=body)
    assert created.status_code == 200, created.text
    assert deliveries == [UUID(created.json()["id"])]
    with Session(labeling_data.database.engine) as session:
        model = session.get(ModelConfiguration, labeling_data.model_id)
        assert model is not None
        session.delete(model)
        session.commit()
    replay = labeling_client.post(path, json=body)
    assert replay.status_code == 200, replay.text
    assert replay.json()["id"] == created.json()["id"]
    assert len(deliveries) == 1
    changed = labeling_client.post(
        path, json={**body, "model_configuration_id": labeling_data.model_id + 1000}
    )
    assert changed.status_code == 409, changed.text
    with Session(labeling_data.database.engine) as session:
        assert len(repository.list_runs(session, labeling_data.document_set_id)) == 1


def test_api_enforces_admin_permission_and_editable_document_set(
    labeling_data: LabelingData, labeling_client: TestClient
) -> None:
    missing = labeling_client.get(
        _api_path(labeling_data, "setup", document_set_id=2147483647)
    )
    assert missing.status_code == 404
    with Session(labeling_data.database.engine) as session:
        user = session.get(User, labeling_data.user_id)
        assert user is not None
        user.role = UserRole.BASIC
        user.effective_permissions = [Permission.READ_DOCUMENT_SETS.value]
        session.commit()
    for method, suffix, body in [
        ("GET", "setup", None),
        ("GET", "label-settings", None),
        (
            "PUT",
            "label-settings",
            {
                "expected_revision": 1,
                "labels": [{"id": "one", "name": "One", "description": "One label."}],
            },
        ),
        ("POST", "runs", _start_body(labeling_data)),
        ("GET", "runs", None),
    ]:
        response = labeling_client.request(
            method,
            _api_path(labeling_data, suffix),
            json=body,
        )
        assert response.status_code == 403, response.text
    with Session(labeling_data.database.engine) as session:
        user = session.get(User, labeling_data.user_id)
        assert user is not None
        user.effective_permissions = [Permission.FULL_ADMIN_PANEL_ACCESS.value]
        session.commit()
    forbidden = labeling_client.get(
        _api_path(
            labeling_data, "setup", document_set_id=labeling_data.other_document_set_id
        )
    )
    assert forbidden.status_code == 404, forbidden.text
    for method, body in [
        ("GET", None),
        (
            "PUT",
            {
                "expected_revision": 1,
                "labels": [{"id": "one", "name": "One", "description": "One label."}],
            },
        ),
    ]:
        forbidden = labeling_client.request(
            method,
            _api_path(
                labeling_data,
                "label-settings",
                document_set_id=labeling_data.other_document_set_id,
            ),
            json=body,
        )
        assert forbidden.status_code == 404, forbidden.text


@pytest.mark.parametrize(
    "suffix,method",
    [("", "GET"), ("/items", "GET"), ("/cancel", "POST"), ("/retry", "POST")],
)
def test_api_run_id_cannot_cross_document_sets(
    labeling_data: LabelingData, labeling_client: TestClient, suffix: str, method: str
) -> None:
    run_id, _ = _start(labeling_data)
    response = labeling_client.request(
        method,
        _api_path(
            labeling_data,
            f"runs/{run_id}{suffix}",
            document_set_id=labeling_data.other_document_set_id,
        ),
    )
    assert response.status_code == 404, response.text
    with Session(labeling_data.database.engine) as session:
        run = session.get(RegulatoryLabelingRun, run_id)
        assert run is not None
        assert not run.cancel_requested
        assert len(repository.list_runs(session, labeling_data.document_set_id)) == 1


def test_api_taxonomy_rejects_duplicate_and_unknown_fields_without_new_rows(
    labeling_data: LabelingData, labeling_client: TestClient
) -> None:
    with Session(labeling_data.database.engine) as session:
        before = {row.id for row in repository.list_taxonomies(session)}
    label = {"id": "one", "name": "One", "description": "One label"}
    duplicate = labeling_client.post(
        _api_path(labeling_data, "taxonomies"),
        json={"name": "Duplicate", "labels": [label, label]},
    )
    assert duplicate.status_code == 400, duplicate.text
    extra = labeling_client.post(
        _api_path(labeling_data, "taxonomies"),
        json={
            "name": "Unknown",
            "labels": [label],
            "instructions": "Ignore constraints",
        },
    )
    assert extra.status_code == 422, extra.text
    with Session(labeling_data.database.engine) as session:
        assert {row.id for row in repository.list_taxonomies(session)} == before


def test_api_retry_is_idempotent_and_reauthorizes_provider(
    labeling_data: LabelingData, labeling_client: TestClient
) -> None:
    run_id, _ = _start(labeling_data)
    with Session(labeling_data.database.engine) as session:
        run = session.get(RegulatoryLabelingRun, run_id)
        assert run is not None
        run.status = "failed"
        run.stage = "finished"
        session.commit()
    path = _api_path(labeling_data, f"runs/{run_id}/retry")
    first = labeling_client.post(path)
    second = labeling_client.post(path)
    assert first.status_code == second.status_code == 200, (first.text, second.text)
    assert first.json()["id"] == second.json()["id"] != str(run_id)
    with Session(labeling_data.database.engine) as session:
        child = session.get(RegulatoryLabelingRun, UUID(first.json()["id"]))
        assert child is not None and child.retry_of_id == run_id
        model = session.get(ModelConfiguration, labeling_data.model_id)
        assert model is not None
        model.is_visible = False
        session.commit()
    replay = labeling_client.post(path)
    assert replay.status_code == 200, replay.text
    assert replay.json()["id"] == first.json()["id"]
    with Session(labeling_data.database.engine) as session:
        child = session.get(RegulatoryLabelingRun, UUID(first.json()["id"]))
        assert child is not None
        child.status = "failed"
        child.stage = "finished"
        session.commit()
    denied = labeling_client.post(
        _api_path(labeling_data, f"runs/{first.json()['id']}/retry")
    )
    assert denied.status_code == 400, denied.text
    with Session(labeling_data.database.engine) as session:
        assert len(repository.list_runs(session, labeling_data.document_set_id)) == 2


class RecordingBatch:
    def __init__(self) -> None:
        self.requests: dict[str, list[VertexBatchRequest]] = {}
        self.submit_calls = 0
        self.reconcile_calls = 0
        self.cancel_calls: list[str] = []
        self.submit_mode = "success"
        self.partial_output = False
        self.complete_jobs = True
        self.reconcile_visible = True
        self.get_calls = 0

    def submit(
        self,
        requests: Sequence[VertexBatchRequest],
        *,
        submission_key: str,
        max_jsonl_bytes: int,
    ) -> VertexBatchState:
        assert max_jsonl_bytes > 0
        self.submit_calls += 1
        self.requests[submission_key] = list(requests)
        if self.submit_mode == "indeterminate":
            raise IndexingGatewayIndeterminateSubmissionError(
                "Connection lost after remote acceptance"
            )
        if self.submit_mode == "crash":
            raise SystemExit("Worker exited after remote acceptance")
        return self._state(submission_key)

    def _state(self, submission_key: str) -> VertexBatchState:
        return VertexBatchState(
            remote_job_name=f"batches/{submission_key}",
            status=VertexBatchJobStatus.SUCCEEDED,
            input_uri=f"files/input-{submission_key}",
            output_uri=f"files/output-{submission_key}",
        )

    def get(self, remote_job_name: str) -> VertexBatchState:
        self.get_calls += 1
        if not self.complete_jobs:
            return VertexBatchState(
                remote_job_name=remote_job_name, status=VertexBatchJobStatus.RUNNING
            )
        return self._state(remote_job_name.removeprefix("batches/"))

    def reconcile_submission(self, submission_key: str) -> VertexBatchState | None:
        self.reconcile_calls += 1
        if not self.reconcile_visible:
            return None
        return self._state(submission_key) if submission_key in self.requests else None

    def read_results(self, output_uri: str) -> Iterator[str]:
        requests = self.requests[output_uri.removeprefix("files/output-")]
        for request in requests[:1] if self.partial_output else requests:
            target = json.loads(request.prompt)["target"]
            label = "customs_value" if target["text"] == _FIRST_TEXT else "origin"
            outcome = {
                "labels": [{"label_id": label, "evidence_quote": target["text"]}],
                "abstained": False,
            }
            yield json.dumps(
                {
                    "key": request.request_hash,
                    "response": {
                        "candidates": [
                            {
                                "finishReason": "STOP",
                                "content": {"parts": [{"text": json.dumps(outcome)}]},
                            }
                        ]
                    },
                }
            )

    def cancel(self, remote_job_name: str) -> None:
        self.cancel_calls.append(remote_job_name)

    def delete(self, remote_job_name: str) -> None:
        pass

    def cleanup(self, prefix: str) -> None:
        pass


@pytest.fixture
def fake_batch(
    labeling_data: LabelingData, monkeypatch: pytest.MonkeyPatch
) -> RecordingBatch:
    gateway = RecordingBatch()

    @contextmanager
    def session_factory() -> Iterator[Session]:
        with Session(labeling_data.database.engine) as session:
            yield session

    def gateway_factory(*_args: object, **_kwargs: object) -> RecordingBatch:
        return gateway

    monkeypatch.setattr(
        orchestrator, "get_session_with_current_tenant", session_factory
    )
    monkeypatch.setattr(
        labeling_configuration, "GoogleGeminiFilesBatchGateway", gateway_factory
    )
    return gateway


def _step(
    data: LabelingData, run_id: UUID, *, expire_lease: bool = False
) -> orchestrator.LabelingStepResult:
    with Session(data.database.engine) as session:
        run = session.get(RegulatoryLabelingRun, run_id)
        assert run is not None
        run.next_retry_at = None
        if expire_lease:
            run.lease_expires_at = datetime.datetime.now(
                datetime.timezone.utc
            ) - datetime.timedelta(seconds=1)
        for shard in session.scalars(
            select(RegulatoryLabelingShard).where(
                RegulatoryLabelingShard.run_id == run_id
            )
        ):
            shard.next_retry_at = None
        generation = run.lease_generation
        session.commit()
    return orchestrator.run_labeling_step(run_id, generation, data.database.schema)


def _finish(data: LabelingData, run_id: UUID) -> None:
    for _ in range(12):
        result = _step(data, run_id)
        if result.outcome is orchestrator.LabelingStepOutcome.TERMINAL:
            return
        assert result.outcome is not orchestrator.LabelingStepOutcome.SKIPPED
    pytest.fail("Labeling worker did not reach a terminal state in twelve steps")


@pytest.mark.parametrize("submit_mode", ["success", "indeterminate", "crash"])
def test_worker_recovers_accepted_submission_without_duplicate_batch(
    labeling_data: LabelingData, fake_batch: RecordingBatch, submit_mode: str
) -> None:
    run_id, _ = _start(labeling_data)
    fake_batch.submit_mode = submit_mode
    prepared = _step(labeling_data, run_id)
    assert prepared.outcome is orchestrator.LabelingStepOutcome.NEXT
    if submit_mode == "crash":
        with pytest.raises(SystemExit):
            _step(labeling_data, run_id)
        with Session(labeling_data.database.engine) as session:
            shard = session.scalar(
                select(RegulatoryLabelingShard).where(
                    RegulatoryLabelingShard.run_id == run_id
                )
            )
            assert shard is not None and shard.status == "submitting"
            assert shard.remote_job_name is None
        _step(labeling_data, run_id, expire_lease=True)
    else:
        _step(labeling_data, run_id)
    _finish(labeling_data, run_id)
    assert fake_batch.submit_calls == 1
    assert fake_batch.reconcile_calls == (0 if submit_mode == "success" else 1)
    with Session(labeling_data.database.engine) as session:
        run = session.get(RegulatoryLabelingRun, run_id)
        assert run is not None
        assert run.status == "completed_with_errors", run.error
        assert (
            run.completed_chunks,
            run.failed_chunks,
            run.unresolved_derived_chunks,
        ) == (2, 0, 1)
        projection = session.scalar(
            select(RegulatoryDerivedLabelProjection).where(
                RegulatoryDerivedLabelProjection.run_id == run_id,
                RegulatoryDerivedLabelProjection.regulatory_chunk_id
                == labeling_data.derived_id,
            )
        )
        assert projection is not None and projection.labels == [
            "customs_value",
            "origin",
        ]


def test_worker_partial_batch_output_explicitly_fails_missing_item(
    labeling_data: LabelingData, fake_batch: RecordingBatch
) -> None:
    run_id, _ = _start(labeling_data)
    fake_batch.partial_output = True
    _finish(labeling_data, run_id)
    assert fake_batch.submit_calls == 1
    with Session(labeling_data.database.engine) as session:
        run = session.get(RegulatoryLabelingRun, run_id)
        assert run is not None
        assert run.status == "completed_with_errors", run.error
        assert (run.completed_chunks, run.failed_chunks) == (1, 1)
        items = list(
            session.scalars(
                select(RegulatoryLabelingItem).where(
                    RegulatoryLabelingItem.run_id == run_id
                )
            )
        )
        missing = next(item for item in items if item.status == "failed")
        assert missing.labels == [] and missing.error


@pytest.mark.parametrize("submit_mode", ["success", "indeterminate"])
def test_worker_cancellation_cancels_remote_batch_and_never_applies_output(
    labeling_data: LabelingData, fake_batch: RecordingBatch, submit_mode: str
) -> None:
    run_id, _ = _start(labeling_data)
    fake_batch.submit_mode = submit_mode
    _step(labeling_data, run_id)
    _step(labeling_data, run_id)
    with Session(labeling_data.database.engine) as session:
        repository.request_cancellation(
            session, document_set_id=labeling_data.document_set_id, run_id=run_id
        )
        session.commit()
    _finish(labeling_data, run_id)
    assert fake_batch.submit_calls == 1
    assert len(fake_batch.cancel_calls) == 1
    with Session(labeling_data.database.engine) as session:
        run = session.get(RegulatoryLabelingRun, run_id)
        assert run is not None and run.status == "cancelled"
        items = list(
            session.scalars(
                select(RegulatoryLabelingItem).where(
                    RegulatoryLabelingItem.run_id == run_id
                )
            )
        )
        assert all(item.labels == [] and item.status == "cancelled" for item in items)


def test_worker_reauthorizes_before_submitting_frozen_run(
    labeling_data: LabelingData, fake_batch: RecordingBatch
) -> None:
    run_id, _ = _start(labeling_data)
    _step(labeling_data, run_id)
    with Session(labeling_data.database.engine) as session:
        model = session.get(ModelConfiguration, labeling_data.model_id)
        assert model is not None
        model.is_visible = False
        session.commit()
    _finish(labeling_data, run_id)
    assert fake_batch.submit_calls == 0
    with Session(labeling_data.database.engine) as session:
        run = session.get(RegulatoryLabelingRun, run_id)
        assert run is not None and run.status == "failed"
        assert run.completed_chunks == 0


def test_worker_preparation_pages_and_in_flight_limit_are_durable(
    labeling_data: LabelingData,
    fake_batch: RecordingBatch,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(orchestrator, "LABELING_PREPARATION_PAGE", 1)
    monkeypatch.setattr(orchestrator, "LABELING_SHARD_ITEMS", 1)
    monkeypatch.setattr(orchestrator, "LABELING_MAX_IN_FLIGHT", 1)
    fake_batch.complete_jobs = False
    run_id, _ = _start(labeling_data)
    _step(labeling_data, run_id)
    with Session(labeling_data.database.engine) as session:
        run = session.get(RegulatoryLabelingRun, run_id)
        assert run is not None and run.stage == "preparing"
        items = list(
            session.scalars(
                select(RegulatoryLabelingItem).where(
                    RegulatoryLabelingItem.run_id == run_id
                )
            )
        )
        assert sum(item.request_hash is not None for item in items) == 1
    _step(labeling_data, run_id)
    assert fake_batch.submit_calls == 0
    _step(labeling_data, run_id)
    assert fake_batch.submit_calls == 1
    _step(labeling_data, run_id)
    assert fake_batch.submit_calls == 1 and fake_batch.get_calls == 1
    fake_batch.complete_jobs = True
    _finish(labeling_data, run_id)
    assert fake_batch.submit_calls == 2
    with Session(labeling_data.database.engine) as session:
        run = session.get(RegulatoryLabelingRun, run_id)
        assert run is not None and run.completed_chunks == 2
        assert len(run.shards) == 2


def test_worker_indeterminate_visibility_expires_without_resubmission(
    labeling_data: LabelingData, fake_batch: RecordingBatch
) -> None:
    fake_batch.submit_mode = "indeterminate"
    fake_batch.reconcile_visible = False
    run_id, _ = _start(labeling_data)
    _step(labeling_data, run_id)
    _step(labeling_data, run_id)
    _step(labeling_data, run_id)
    assert fake_batch.submit_calls == fake_batch.reconcile_calls == 1
    with Session(labeling_data.database.engine) as session:
        shard = session.scalar(
            select(RegulatoryLabelingShard).where(
                RegulatoryLabelingShard.run_id == run_id
            )
        )
        assert shard is not None and shard.status == "reconcile_required"
        shard.reconcile_until = datetime.datetime.now(
            datetime.timezone.utc
        ) - datetime.timedelta(seconds=1)
        session.commit()
    _finish(labeling_data, run_id)
    assert fake_batch.submit_calls == 1
    with Session(labeling_data.database.engine) as session:
        run = session.get(RegulatoryLabelingRun, run_id)
        assert run is not None and run.status == "failed"
        assert run.completed_chunks == 0 and run.failed_chunks == 2
        assert "indeterminate" in (run.shards[0].error or "")


def test_preparation_uses_the_neighbor_version_visible_during_target_validity(
    labeling_data: LabelingData,
) -> None:
    successor_id = str(uuid4())
    successor_text = "A successor rule introduced only after the target expired."
    with Session(labeling_data.database.engine) as session:
        target = session.get(RegulatoryChunk, labeling_data.first_id)
        predecessor = session.get(RegulatoryChunk, labeling_data.second_id)
        assert target is not None and predecessor is not None
        target.validity_start_date = datetime.date(2020, 1, 1)
        target.validity_end_date = datetime.date(2021, 1, 1)
        predecessor.validity_start_date = datetime.date(2019, 1, 1)
        predecessor.validity_end_date = datetime.date(2022, 1, 1)
        session.add(
            RegulatoryChunk(
                id=successor_id,
                user_file_id=labeling_data.file_id,
                text=successor_text,
                position=1,
                projection_ordinal=4,
                heading_path=["Customs regulation", "Article 1"],
                chunk_metadata={"chunk_variant": "atomic"},
                source="amendment",
                status="active",
                validity_start_date=datetime.date(2022, 1, 1),
            )
        )
        session.commit()
    run_id, _ = _start(labeling_data)
    lease, shard_id = _prepare(labeling_data, run_id)
    with Session(labeling_data.database.engine) as session:
        target_item = session.scalar(
            select(RegulatoryLabelingItem).where(
                RegulatoryLabelingItem.run_id == run_id,
                RegulatoryLabelingItem.regulatory_chunk_id == labeling_data.first_id,
            )
        )
        assert target_item is not None and target_item.context_snapshot is not None
        assert _SECOND_TEXT in target_item.context_snapshot
        assert successor_text not in target_item.context_snapshot
        assert target_item.source_snapshot["context_member_ids"] == [
            labeling_data.first_id,
            labeling_data.second_id,
        ]
        repository.apply_shard_results(
            session,
            lease,
            shard_id=shard_id,
            outcomes=_outcomes(session, lease, shard_id),
        )
        session.commit()
        assert all(
            item.status == "completed"
            for item in repository.load_shard_requests(session, lease, shard_id)
        )


@pytest.mark.parametrize("after_application", [False, True])
def test_new_context_member_invalidates_frozen_labels_before_application_or_projection(
    labeling_data: LabelingData, after_application: bool
) -> None:
    run_id, _ = _start(labeling_data)
    lease, shard_id = _prepare(labeling_data, run_id)
    if after_application:
        with Session(labeling_data.database.engine) as session:
            repository.apply_shard_results(
                session,
                lease,
                shard_id=shard_id,
                outcomes=_outcomes(session, lease, shard_id),
            )
            session.commit()
    with Session(labeling_data.database.engine) as session:
        session.add(
            RegulatoryChunk(
                id=str(uuid4()),
                user_file_id=labeling_data.file_id,
                text="An additional rule changes the interpretation of nearby declarations.",
                position=4,
                projection_ordinal=4,
                heading_path=["Customs regulation", "Article 4"],
                chunk_metadata={"chunk_variant": "atomic"},
                source="amendment",
                status="active",
            )
        )
        session.commit()
    with Session(labeling_data.database.engine) as session:
        if after_application:
            repository.project_derived_labels(session, lease)
        else:
            repository.apply_shard_results(
                session,
                lease,
                shard_id=shard_id,
                outcomes=_outcomes(session, lease, shard_id),
            )
        session.commit()
        items = repository.load_shard_requests(session, lease, shard_id)
        assert {item.status for item in items} == {"stale"}
        assert all(item.labels == [] for item in items)


@pytest.mark.parametrize("past_reconciliation_deadline", [False, True])
def test_worker_reconciles_crashed_submission_after_full_lease_expiry(
    labeling_data: LabelingData,
    fake_batch: RecordingBatch,
    past_reconciliation_deadline: bool,
) -> None:
    fake_batch.submit_mode = "crash"
    run_id, _ = _start(labeling_data)
    _step(labeling_data, run_id)
    with pytest.raises(SystemExit):
        _step(labeling_data, run_id)
    elapsed = datetime.timedelta(
        seconds=(
            orchestrator.LABELING_RECONCILE_SECONDS
            if past_reconciliation_deadline
            else orchestrator.LABELING_LEASE_SECONDS
        )
        + 1
    )
    with Session(labeling_data.database.engine) as session:
        run = session.get(RegulatoryLabelingRun, run_id)
        assert run is not None and run.lease_expires_at is not None
        run.lease_expires_at -= elapsed
        shard = session.scalar(
            select(RegulatoryLabelingShard).where(
                RegulatoryLabelingShard.run_id == run_id
            )
        )
        assert shard is not None and shard.reconcile_until is not None
        shard.reconcile_until -= elapsed
        session.commit()
    _finish(labeling_data, run_id)
    assert fake_batch.submit_calls == fake_batch.reconcile_calls == 1
    with Session(labeling_data.database.engine) as session:
        run = session.get(RegulatoryLabelingRun, run_id)
        assert run is not None and run.completed_chunks == 2, run.error if run else None


def test_worker_retries_cancelling_a_newly_reconciled_remote_job(
    labeling_data: LabelingData,
    fake_batch: RecordingBatch,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_batch.submit_mode = "indeterminate"

    def transient_cancel(remote_job_name: str) -> None:
        fake_batch.cancel_calls.append(remote_job_name)
        if len(fake_batch.cancel_calls) == 1:
            raise IndexingGatewayConnectionError()

    monkeypatch.setattr(fake_batch, "cancel", transient_cancel)
    run_id, _ = _start(labeling_data)
    _step(labeling_data, run_id)
    _step(labeling_data, run_id)
    with Session(labeling_data.database.engine) as session:
        shard = session.scalar(
            select(RegulatoryLabelingShard).where(
                RegulatoryLabelingShard.run_id == run_id
            )
        )
        assert shard is not None
        shard.reconcile_until = datetime.datetime.now(
            datetime.timezone.utc
        ) - datetime.timedelta(seconds=1)
        repository.request_cancellation(
            session, document_set_id=labeling_data.document_set_id, run_id=run_id
        )
        session.commit()
    _finish(labeling_data, run_id)
    assert fake_batch.submit_calls == 1
    assert len(fake_batch.cancel_calls) == 2
    assert fake_batch.cancel_calls[0] == fake_batch.cancel_calls[1]
    with Session(labeling_data.database.engine) as session:
        run = session.get(RegulatoryLabelingRun, run_id)
        assert run is not None and run.status == "cancelled"


def test_derived_lineage_metadata_drift_prevents_union_of_completed_labels(
    labeling_data: LabelingData,
) -> None:
    run_id, _ = _start(labeling_data)
    lease, shard_id = _prepare(labeling_data, run_id)
    with Session(labeling_data.database.engine) as session:
        repository.apply_shard_results(
            session,
            lease,
            shard_id=shard_id,
            outcomes=_outcomes(session, lease, shard_id),
        )
        session.commit()
    with Session(labeling_data.database.engine) as session:
        derived = session.get(RegulatoryChunk, labeling_data.derived_id)
        assert derived is not None
        derived.chunk_metadata = {
            "chunk_variant": "hierarchical_aggregate",
            "source_regulatory_chunk_ids": [labeling_data.first_id],
        }
        session.commit()
    with Session(labeling_data.database.engine) as session:
        repository.project_derived_labels(session, lease)
        session.commit()
        projection = session.scalar(
            select(RegulatoryDerivedLabelProjection).where(
                RegulatoryDerivedLabelProjection.run_id == run_id,
                RegulatoryDerivedLabelProjection.regulatory_chunk_id
                == labeling_data.derived_id,
            )
        )
        assert projection is not None
        assert projection.labels == []
        assert projection.resolution == "unresolved"
        assert projection.unresolved_reason == "derived_target_changed"


def _start_projecting_run(data: LabelingData) -> UUID:
    run_id, _ = _start(data)
    for _ in range(8):
        with Session(data.database.engine) as session:
            run = session.get(RegulatoryLabelingRun, run_id)
            assert run is not None
            if run.stage == "projecting":
                assert run.completed_chunks == 2
                return run_id
            assert run.status in {"queued", "running"}, run.error
        _step(data, run_id)
    pytest.fail("Labeling worker did not reach projection after applying both atomics")


@pytest.mark.parametrize("cancel_between_pages", [False, True])
def test_worker_projection_pages_resume_or_cancel_after_all_atomics_complete(
    labeling_data: LabelingData,
    fake_batch: RecordingBatch,
    monkeypatch: pytest.MonkeyPatch,
    cancel_between_pages: bool,
) -> None:
    monkeypatch.setattr(orchestrator, "LABELING_PROJECTION_PAGE", 1)
    run_id = _start_projecting_run(labeling_data)
    first_page = _step(labeling_data, run_id)
    assert first_page.outcome is orchestrator.LabelingStepOutcome.NEXT
    with Session(labeling_data.database.engine) as session:
        run = session.get(RegulatoryLabelingRun, run_id)
        assert run is not None and run.status == "running" and run.stage == "projecting"
        projections = list(
            session.scalars(
                select(RegulatoryDerivedLabelProjection).where(
                    RegulatoryDerivedLabelProjection.run_id == run_id
                )
            )
        )
        completed = [row for row in projections if row.resolution != "pending"]
        assert len(completed) == 1
        first = completed[0]
        checkpoint = (first.id, first.resolution, first.labels, first.provenance)
        if cancel_between_pages:
            repository.request_cancellation(
                session, document_set_id=labeling_data.document_set_id, run_id=run_id
            )
            session.commit()
    submissions, polls = fake_batch.submit_calls, fake_batch.get_calls
    _finish(labeling_data, run_id)
    assert (fake_batch.submit_calls, fake_batch.get_calls) == (submissions, polls)
    with Session(labeling_data.database.engine) as session:
        run = session.get(RegulatoryLabelingRun, run_id)
        assert run is not None and run.completed_chunks == 2
        first = session.get(RegulatoryDerivedLabelProjection, checkpoint[0])
        assert first is not None
        assert (
            first.id,
            first.resolution,
            first.labels,
            first.provenance,
        ) == checkpoint
        projections = list(
            session.scalars(
                select(RegulatoryDerivedLabelProjection).where(
                    RegulatoryDerivedLabelProjection.run_id == run_id
                )
            )
        )
        if cancel_between_pages:
            assert run.status == "cancelled"
            remaining = next(row for row in projections if row.id != checkpoint[0])
            assert remaining.labels == []
            assert remaining.resolution not in {"lineage", "legacy_containment"}
        else:
            assert run.status == "completed_with_errors"
            assert all(row.resolution != "pending" for row in projections)
            derived = next(
                row
                for row in projections
                if row.regulatory_chunk_id == labeling_data.derived_id
            )
            assert derived.labels == ["customs_value", "origin"]
            assert run.unresolved_derived_chunks == 1


def test_projection_page_checkpoint_survives_crash_and_fences_previous_worker(
    labeling_data: LabelingData, fake_batch: RecordingBatch
) -> None:
    run_id = _start_projecting_run(labeling_data)
    with Session(labeling_data.database.engine) as session:
        run = session.get(RegulatoryLabelingRun, run_id)
        assert run is not None
        old_lease = repository.claim_run(
            session,
            run_id=run_id,
            expected_generation=run.lease_generation,
            lease_seconds=300,
        )
        assert old_lease is not None
        assert repository.project_derived_labels(session, old_lease, limit=1) == 1
        session.commit()
    with Session(labeling_data.database.engine) as session:
        new_lease = repository.claim_run(
            session,
            run_id=run_id,
            expected_generation=old_lease.generation,
            lease_seconds=300,
            now=datetime.datetime.now(datetime.timezone.utc)
            + datetime.timedelta(seconds=301),
        )
        assert new_lease is not None
        session.commit()
    with Session(labeling_data.database.engine) as session:
        with pytest.raises(repository.LabelingStateConflictError, match="lease"):
            repository.project_derived_labels(session, old_lease, limit=1)
        session.rollback()
    with Session(labeling_data.database.engine) as session:
        assert repository.project_derived_labels(session, new_lease, limit=1) == 1
        repository.release_run(session, new_lease, stage="projecting")
        session.commit()
    _finish(labeling_data, run_id)
    assert fake_batch.submit_calls == 1
    with Session(labeling_data.database.engine) as session:
        run = session.get(RegulatoryLabelingRun, run_id)
        assert run is not None and run.status == "completed_with_errors"
        assert run.completed_chunks == 2 and run.unresolved_derived_chunks == 1
