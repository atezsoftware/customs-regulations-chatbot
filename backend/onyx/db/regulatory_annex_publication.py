"""Short scoped database reads for final annex publication preparation."""

import json
from datetime import date
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from onyx.db.models import RegulatoryTemporalProjection, SearchSettings, UserFile
from onyx.regulatory.amendments.annexes.models import (
    AnnexChangeDraft,
    AnnexPositionView,
    AnnexProjectionAccess,
    AnnexTemporalProjection,
    PreparedContextView,
)


def load_annex_context_settings(session: Session) -> SearchSettings:
    """Materialize provider fields before the caller closes its read session."""
    from onyx.db.search_settings import get_current_search_settings

    settings = get_current_search_settings(session)
    _ = settings.cloud_provider
    return settings


def load_annex_publication_inputs(
    session: Session, draft: AnnexChangeDraft
) -> tuple[UserFile, list[SearchSettings], AnnexProjectionAccess]:
    from onyx.access.access import get_access_for_user_files
    from onyx.db.regulatory_amendments import get_batch
    from onyx.db.regulatory_annex_changes import validate_annex_review_scope
    from onyx.db.regulatory_annexes import require_annex_file_scope
    from onyx.db.search_settings import get_active_search_settings_list
    from onyx.db.user_file import (
        fetch_document_set_names_for_user_files,
        fetch_persona_ids_for_user_files,
        fetch_user_project_ids_for_user_files,
    )
    from onyx.regulatory.amendments.annexes.config import REGULATORY_ANNEX_ENVIRONMENT

    if draft.user_file_id is None or draft.batch_id is None:
        raise ValueError("publication preparation scope missing")
    batch = get_batch(session, draft.batch_id)
    if batch is None:
        raise ValueError("publication batch missing")
    validate_annex_review_scope(
        session, batch=batch, draft=draft, environment=REGULATORY_ANNEX_ENVIRONMENT
    )
    file = require_annex_file_scope(session, batch.document_set_id, draft.user_file_id)
    settings = get_active_search_settings_list(session)
    for setting in settings:
        _ = setting.cloud_provider
    if sum(item.status.is_current() for item in settings) != 1:
        raise ValueError("publication requires one current index")
    identifier = str(file.id)
    access = get_access_for_user_files([identifier], session)
    return (
        file,
        settings,
        AnnexProjectionAccess(
            access=access[identifier],
            project_ids=fetch_user_project_ids_for_user_files(
                [identifier], session
            ).get(identifier, []),
            persona_ids=fetch_persona_ids_for_user_files([identifier], session).get(
                identifier, []
            ),
            document_sets=fetch_document_set_names_for_user_files(
                [identifier], session
            ).get(identifier, []),
        ),
    )


def load_file_temporal_bindings(
    session: Session,
    user_file_id: UUID,
    *,
    refresh: bool = False,
    projection_ordinals: tuple[int, ...] | None = None,
    index_uuid: str | None = None,
) -> list[AnnexTemporalProjection]:
    from onyx.db.regulatory_canonical_revisions import (
        validate_temporal_canonical_revisions,
    )
    from onyx.document_index.publication_models import publication_digest

    if projection_ordinals == ():
        return []
    query = (
        select(RegulatoryTemporalProjection)
        .execution_options(populate_existing=refresh)
        .where(
            RegulatoryTemporalProjection.user_file_id == user_file_id,
            RegulatoryTemporalProjection.retired_at.is_(None),
        )
    )
    if projection_ordinals is not None:
        query = query.where(
            RegulatoryTemporalProjection.projection_ordinal.in_(projection_ordinals)
        )
    if index_uuid is not None:
        query = query.where(RegulatoryTemporalProjection.index_uuid == index_uuid)
    rows = list(session.scalars(query))
    for row in rows:
        if publication_digest(row.payload) != row.payload_sha256:
            raise ValueError("temporal binding payload changed")
    validate_temporal_canonical_revisions(session, rows)
    bindings = [AnnexTemporalProjection.model_validate(row.payload) for row in rows]
    for row, binding in zip(rows, bindings):
        if (
            row.projection_ordinal != binding.projection.ordinal
            or row.index_uuid != binding.index.index_uuid
            or str(row.user_file_id)
            != json.loads(binding.projection.source_json)["document_id"]
        ):
            raise ValueError("temporal binding lookup identity mismatch")
    return bindings


def load_binding_context_sources(
    session: Session, user_file_id: UUID, bindings: list[AnnexTemporalProjection]
) -> "PreparedContextView":
    from onyx.db.models import RegulatoryContextSnapshot
    from onyx.regulatory.amendments.annexes.models import (
        ContextSourceSnapshot,
        PreparedContextView,
    )

    projections = [
        binding.context for binding in bindings if binding.context is not None
    ]
    hashes = {projection.source_snapshot_sha256 for projection in projections}
    snapshots = [
        ContextSourceSnapshot.model_validate(row.payload)
        for row in session.scalars(
            select(RegulatoryContextSnapshot).where(
                RegulatoryContextSnapshot.user_file_id == user_file_id,
                RegulatoryContextSnapshot.sha256.in_(hashes),
            )
        )
    ]
    return PreparedContextView(projections=projections, snapshots=snapshots)


def load_file_position_views(
    session: Session, user_file_id: UUID
) -> list[AnnexPositionView]:
    from onyx.db.models import AnnexChangeSet, AnnexPublicationManifest

    payloads = session.scalars(
        select(AnnexPublicationManifest.payload)
        .join(
            AnnexChangeSet, AnnexChangeSet.id == AnnexPublicationManifest.change_set_id
        )
        .where(
            AnnexChangeSet.user_file_id == user_file_id,
            AnnexPublicationManifest.approved_at.is_not(None),
        )
        .order_by(
            AnnexPublicationManifest.approved_at, AnnexPublicationManifest.change_set_id
        )
    )
    return [
        AnnexPositionView.model_validate(view)
        for payload in payloads
        for view in payload.get("position_views", [])
    ]


def effective_positions(views: list[AnnexPositionView], when: date) -> dict[str, int]:
    positions: dict[str, int] = {}
    for view in views:
        if (view.effective_start is None or view.effective_start <= when) and (
            view.effective_end is None or when < view.effective_end
        ):
            positions.update(view.positions)
    return positions


def publication_input_scope_hash(
    file: UserFile, settings: list[SearchSettings], access: AnnexProjectionAccess
) -> str:
    """Bind every persisted index/file input and current ACL without exposing secrets."""
    from onyx.db.regulatory_configuration_fingerprint import configuration_fingerprint

    return configuration_fingerprint(
        [
            file,
            sorted(settings, key=lambda item: item.id),
            access.model_dump(mode="json"),
        ]
    )
