"""Prepare independent original documents, then label their exact frozen scope."""

import hashlib
import json
from functools import partial
from uuid import UUID

from onyx.db import document_intake as repository
from onyx.db.engine.sql_engine import get_session_with_current_tenant
from onyx.file_store.file_store import get_default_file_store
from onyx.regulatory.amendments.annexes.evidence import read_source_graph
from onyx.regulatory.document_intake.models import IntakeSourceAsset
from onyx.regulatory.document_intake.preparation import (
    clone_intake_original,
    prepare_intake_file,
)
from onyx.regulatory.document_intake.source_graph import parse_intake_graph
from onyx.utils.logger import setup_logger

logger = setup_logger()


def run_document_intake(*, intake_id: UUID, tenant_id: str) -> UUID | None:
    with get_session_with_current_tenant() as session:
        token = repository.claim_intake(session, intake_id)
        session.commit()
    if token is None:
        return None
    try:
        with get_session_with_current_tenant() as session:
            batch, package, assets = repository.load_intake_job(
                session, intake_id, token
            )
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
                    asset.id: clone_intake_original(
                        intake_id=batch.id, asset=asset, store=store
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
            admit(
                session,
                batch,
                assets,
                links,
                root_hash=root_hash,
                owned_file_ids=owned_file_ids,
            )
            package_id = package.id
            by_id = {asset.id: asset for asset in asset_views}
            work = repository.intake_preparation_work(session, batch)
            session.commit()
        for document_id, file_id, asset_id, status, original_file_id in work:
            if status not in {"chunked", "labeling", "ready"}:
                prepare = prepare_intake_file
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
            with get_session_with_current_tenant() as session:
                repository.mark_intake_file_prepared(
                    session, intake_id, token, document_id
                )
                session.commit()
        from onyx.regulatory.document_intake.metadata_job import run_intake_metadata

        run_intake_metadata(intake_id=intake_id, lease_token=token, tenant_id=tenant_id)
        with get_session_with_current_tenant() as session:
            run_id = repository.start_intake_labeling(session, intake_id, token)
            session.commit()
        return run_id
    except Exception as error:
        logger.exception("New document intake failed: %s", intake_id)
        with get_session_with_current_tenant() as session:
            repository.fail_intake(session, intake_id, token, str(error))
            session.commit()
        raise
