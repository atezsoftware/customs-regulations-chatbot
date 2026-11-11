"""Prepare independent original documents, then label their exact frozen scope."""

import hashlib
import json
from collections.abc import Callable
from functools import partial
from uuid import UUID

from onyx.db import document_intake as repository
from onyx.db.document_intake_preparation_lease import (
    maintain_preparation_lease,
    preparation_execution_lock,
)
from onyx.db.engine.sql_engine import get_session_with_current_tenant
from onyx.file_store.file_store import FileStore, get_default_file_store
from onyx.regulatory.amendments.annexes.evidence import read_source_graph
from onyx.regulatory.document_intake.metadata_checkpoint import MetadataLeaseLost
from onyx.regulatory.document_intake.models import (
    IntakePreparationPolicy,
    IntakeSourceAsset,
)
from onyx.regulatory.document_intake.preparation import (
    clone_intake_original,
    prepare_intake_file,
)
from onyx.regulatory.document_intake.source_graph import parse_intake_graph
from onyx.utils.logger import setup_logger

logger = setup_logger()


def run_document_intake(*, intake_id: UUID, tenant_id: str) -> UUID | None:
    with preparation_execution_lock(intake_id) as acquired:
        if not acquired:
            return None
        return _run_document_intake(intake_id=intake_id, tenant_id=tenant_id)


def _run_document_intake(*, intake_id: UUID, tenant_id: str) -> UUID | None:
    with get_session_with_current_tenant() as session:
        token = repository.claim_intake(session, intake_id)
        session.commit()
    if token is None:
        return None
    try:
        with maintain_preparation_lease(intake_id, token) as check_owner:
            _prepare_intake_sources(
                intake_id=intake_id,
                tenant_id=tenant_id,
                token=token,
                check_owner=check_owner,
            )
        from onyx.regulatory.document_intake.metadata_job import run_intake_metadata

        run_intake_metadata(intake_id=intake_id, lease_token=token, tenant_id=tenant_id)
        with get_session_with_current_tenant() as session:
            run_id = repository.start_intake_labeling(session, intake_id, token)
            session.commit()
        return run_id
    except MetadataLeaseLost:
        return None
    except Exception as error:
        logger.exception("New document intake failed: %s", intake_id)
        with get_session_with_current_tenant() as session:
            repository.fail_intake(session, intake_id, token, str(error))
            session.commit()
        raise


def _clone_owned_original(
    *,
    check_owner: Callable[[], None],
    intake_id: UUID,
    asset: IntakeSourceAsset,
    store: "FileStore",
) -> str:
    check_owner()
    return clone_intake_original(intake_id=intake_id, asset=asset, store=store)


def _prepare_intake_sources(
    *, intake_id: UUID, tenant_id: str, token: UUID, check_owner: Callable[[], None]
) -> None:
    check_owner()
    with get_session_with_current_tenant() as session:
        batch, package, assets = repository.load_intake_job(session, intake_id, token)
        if (
            package.status != "ready"
            or package.issues
            or not package.manifest_file_id
            or not package.manifest_sha256
        ):
            raise ValueError("Source package is not complete and verified")
        store = get_default_file_store()
        links = read_source_graph(
            store,
            manifest_file_id=package.manifest_file_id,
            manifest_sha256=package.manifest_sha256,
            assets=assets,
        )
        with store.read_file(package.manifest_file_id) as stream:
            manifest_bytes = stream.read(150 * 1024 * 1024 + 1)
        if hashlib.sha256(manifest_bytes).hexdigest() != package.manifest_sha256:
            raise ValueError("Source manifest changed")
        manifest = json.loads(manifest_bytes)
        root_hash = manifest["assets"][0]["sha256"]
        graph = parse_intake_graph(
            manifest,
            input_kind=package.input_spec.get("kind"),
            asset_hashes=[asset.sha256 for asset in assets],
            links=links,
        )
        asset_views = [IntakeSourceAsset.model_validate(asset) for asset in assets]
        owned_file_ids = (
            {
                asset.id: _clone_owned_original(
                    check_owner=check_owner,
                    intake_id=batch.id,
                    asset=asset,
                    store=store,
                )
                for asset in asset_views
            }
            if not repository.intake_files(session, batch)
            else None
        )
        admit = repository.admit_intake_assets
        if graph is not None:
            admit = partial(
                admit,
                intake_graph=graph,
                source_manifest_bytes=manifest_bytes,
            )
        check_owner()
        admit(
            session,
            batch,
            assets,
            links,
            root_hash=root_hash,
            owned_file_ids=owned_file_ids,
        )
        package_id = package.id
        policy = (
            IntakePreparationPolicy.model_validate(batch.preparation_policy)
            if batch.preparation_policy is not None
            else None
        )
        by_id = {asset.id: asset for asset in asset_views}
        work = repository.intake_preparation_work(session, batch)
        session.commit()
    for document_id, file_id, asset_id, status, original_file_id in work:
        check_owner()
        if status not in {"chunked", "labeling", "ready"}:
            prepare = prepare_intake_file
            if policy is not None:
                prepare = partial(
                    prepare,
                    chunking_version=policy.chunking_version,
                    preparation_policy=policy,
                )
            if graph is not None:
                prepare = partial(
                    prepare,
                    parent_asset_hashes=graph.parents_for(by_id[asset_id].sha256),
                )
            prepare(
                package_id=package_id,
                asset=by_id[asset_id],
                user_file_id=file_id,
                links=links,
                package_assets=asset_views,
                tenant_id=tenant_id,
                store=store,
                original_file_id=original_file_id,
            )
        check_owner()
        with get_session_with_current_tenant() as session:
            repository.mark_intake_file_prepared(session, intake_id, token, document_id)
            session.commit()
