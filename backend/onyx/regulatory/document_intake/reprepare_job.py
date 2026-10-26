"""Replay the same frozen originals only for an explicit unpublished reprepare."""

from functools import partial
from uuid import UUID

from onyx.db import document_intake as intake
from onyx.db import document_intake_reprepare as repository
from onyx.db.engine.sql_engine import get_session_with_current_tenant
from onyx.file_store.file_store import get_default_file_store
from onyx.regulatory.amendments.annexes.evidence import read_source_graph
from onyx.regulatory.document_intake.metadata_job import run_intake_metadata
from onyx.regulatory.document_intake.models import IntakeSourceAsset
from onyx.regulatory.document_intake.preparation import prepare_intake_file


def run_document_intake_reprepare(
    *, intake_id: UUID, request_id: UUID, tenant_id: str
) -> UUID | None:
    from onyx.regulatory.writer_publication import reprepare_owned_intake_file

    with get_session_with_current_tenant() as session:
        token = repository.claim_reprepare(session, intake_id, request_id)
        session.commit()
    if token is None:
        return None
    prepared = False
    try:
        with get_session_with_current_tenant() as session:
            batch, package, assets = intake.load_intake_job(session, intake_id, token)
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
            views = [IntakeSourceAsset.model_validate(asset) for asset in assets]
            by_id = {asset.id: asset for asset in views}
            package_id = package.id
            work = intake.intake_preparation_work(session, batch)
            request = repository.latest_reprepare(session, intake_id)
            if request is None or request.id != request_id:
                raise ValueError("The active reprepare request changed")
            done = {
                key
                for key, entry in request.documents.items()
                if entry["status"] == "prepared"
            }
            session.commit()
        for document_id, file_id, asset_id, _status, original_file_id in work:
            if str(document_id) in done:
                continue
            prepare_intake_file(
                package_id=package_id,
                asset=by_id[asset_id],
                user_file_id=file_id,
                links=links,
                package_assets=views,
                tenant_id=tenant_id,
                store=store,
                original_file_id=original_file_id,
                chunk_file=partial(
                    reprepare_owned_intake_file,
                    intake_id=intake_id,
                    request_id=request_id,
                    lease_token=token,
                ),
            )
        with get_session_with_current_tenant() as session:
            repository.finish_reprepare(session, intake_id, request_id, token)
            session.commit()
        prepared = True
        run_intake_metadata(intake_id=intake_id, lease_token=token, tenant_id=tenant_id)
        with get_session_with_current_tenant() as session:
            run_id = intake.start_intake_labeling(session, intake_id, token)
            session.commit()
        return run_id
    except Exception as error:
        with get_session_with_current_tenant() as session:
            if prepared:
                intake.fail_intake(session, intake_id, token, str(error))
            else:
                repository.fail_reprepare(
                    session, intake_id, request_id, token, str(error)
                )
            session.commit()
        raise
