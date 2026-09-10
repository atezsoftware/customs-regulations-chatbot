"""Public source associations and qualified temporal inventory.

These reads never grant access: callers retain the existing file/document ACLs.
"""

import json
from datetime import date
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from onyx.db.models import (
    RegulatoryChunk,
    RegulatoryFilePublication,
    RegulatoryTemporalProjection,
    UserFile,
)
from onyx.db.regulatory_annex_publication import load_file_temporal_bindings
from onyx.document_index.publication_models import PublicationIndexSnapshot
from onyx.regulatory.amendments.annexes.models import AnnexTemporalProjection


def protected_file_ids(session: Session, file_ids: tuple[UUID, ...]) -> frozenset[UUID]:
    return frozenset(
        session.scalars(
            select(RegulatoryFilePublication.user_file_id).where(
                RegulatoryFilePublication.user_file_id.in_(file_ids),
                RegulatoryFilePublication.epoch > 0,
            )
        )
    )


def qualified_file_ids(session: Session, file_ids: tuple[UUID, ...]) -> frozenset[UUID]:
    # Retired rows still establish that timeless fallback is no longer authority.
    return frozenset(
        session.scalars(
            select(RegulatoryTemporalProjection.user_file_id)
            .where(
                RegulatoryTemporalProjection.user_file_id.in_(file_ids),
            )
            .distinct()
        )
    )


def load_public_temporal_bindings(
    session: Session,
    user_file_id: UUID,
    *,
    index: PublicationIndexSnapshot,
    as_of_date: date,
) -> list[AnnexTemporalProjection]:
    """Use each immutable activated binding's positive receipt and actual physical index."""
    from onyx.regulatory.contextual import validity_window_contains

    canonical = {
        row.id: row
        for row in session.scalars(
            select(RegulatoryChunk)
            .execution_options(populate_existing=True)
            .where(
                RegulatoryChunk.user_file_id == user_file_id,
            )
        )
    }
    selected = []
    for binding in load_file_temporal_bindings(session, user_file_id, refresh=True):
        if binding.index.temporal_lookup_identity() != index.temporal_lookup_identity():
            continue
        source = json.loads(binding.projection.source_json)
        row = canonical.get(source["regulatory_chunk_id"])
        if (
            row is None
            or not validity_window_contains(
                row.validity_start_date, row.validity_end_date, as_of_date
            )
            or not validity_window_contains(
                binding.effective_start, binding.effective_end, as_of_date
            )
        ):
            continue
        if not index.accepts_encoder_configuration(
            json.loads(binding.projection.embedding_config_json)
        ):
            raise ValueError("temporal binding encoder receipt is not accepted")
        selected.append(binding)
    return sorted(
        selected, key=lambda item: (item.semantic_position, item.projection.ordinal)
    )


def public_file_source_owners(session: Session, file_id: str) -> tuple[UUID, ...]:
    """Map immutable raw/source/image identity to its protected parent files."""
    from sqlalchemy import cast, or_
    from sqlalchemy.dialects.postgresql import JSONB

    from onyx.db.models import (
        RegulatoryAnnex,
        RegulatoryAnnexRevision,
        RegulatorySourceAsset,
    )

    owners = set(
        session.scalars(select(UserFile.id).where(UserFile.file_id == file_id))
    )
    owners.update(
        session.scalars(
            select(RegulatoryChunk.user_file_id).where(
                RegulatoryChunk.chunk_metadata["image_file_id"].astext == file_id,
            )
        )
    )
    source = cast(
        RegulatoryTemporalProjection.payload["projection"]["source_json"].astext, JSONB
    )
    owners.update(
        session.scalars(
            select(RegulatoryTemporalProjection.user_file_id).where(
                source["image_file_id"].astext == file_id,
            )
        )
    )
    owners.update(
        session.scalars(
            select(RegulatoryAnnex.user_file_id)
            .join(
                RegulatoryAnnexRevision,
                RegulatoryAnnexRevision.annex_id == RegulatoryAnnex.id,
            )
            .join(
                RegulatorySourceAsset,
                RegulatorySourceAsset.id == RegulatoryAnnexRevision.source_asset_id,
            )
            .where(
                or_(
                    RegulatorySourceAsset.file_id == file_id,
                    RegulatorySourceAsset.text_file_id == file_id,
                ),
                RegulatoryAnnexRevision.approved_at.is_not(None),
            )
        )
    )
    return tuple(sorted(owners))


def current_qualified_file_ids(file_ids: tuple[UUID, ...]) -> frozenset[UUID]:
    from onyx.db.engine.sql_engine import get_session_with_current_tenant

    with get_session_with_current_tenant() as session:
        return qualified_file_ids(session, file_ids)


def query_temporal_bindings(
    file_ids: tuple[UUID, ...],
    *,
    index: PublicationIndexSnapshot,
    as_of_date: date,
) -> dict[UUID, list[AnnexTemporalProjection]]:
    from onyx.db.engine.sql_engine import get_session_with_current_tenant

    with get_session_with_current_tenant() as session:
        return {
            file_id: load_public_temporal_bindings(
                session,
                file_id,
                index=index,
                as_of_date=as_of_date,
            )
            for file_id in file_ids
        }


def current_protected_file_ids(file_ids: tuple[UUID, ...]) -> frozenset[UUID]:
    from onyx.db.engine.sql_engine import get_session_with_current_tenant

    with get_session_with_current_tenant() as session:
        return protected_file_ids(session, file_ids)


def resolve_public_query_index(
    index_name: str, index_uuid: str
) -> PublicationIndexSnapshot:
    """Freeze activated positive receipts against actual runtime encoder facts."""
    from onyx.db.engine.sql_engine import get_session_with_current_tenant
    from onyx.db.models import SearchSettings
    from onyx.document_index.publication_models import (
        PublicationEncoderAuthority,
        publication_digest,
    )
    from onyx.regulatory.amendments.annexes.context_dependencies import context_hash
    from shared_configs.configs import MULTI_TENANT

    with get_session_with_current_tenant() as session:
        settings = session.scalars(
            select(SearchSettings).where(SearchSettings.index_name == index_name)
        ).one_or_none()
        if settings is None:
            raise ValueError(
                "qualified query requires explicit configured index authority"
            )
        actual = PublicationEncoderAuthority(
            provider=str(settings.provider_type) if settings.provider_type else None,
            model=settings.model_name,
            effective_dimension=settings.final_embedding_dim,
            endpoint_sha256=context_hash(settings.api_url),
            deployment_name=settings.deployment_name,
            api_version=settings.api_version,
            normalize=settings.normalize,
            passage_prefix=settings.passage_prefix,
        )
        rows = session.scalars(
            select(RegulatoryTemporalProjection).where(
                RegulatoryTemporalProjection.index_uuid == index_uuid,
                RegulatoryTemporalProjection.retired_at.is_(None),
            )
        )
        accepted: PublicationIndexSnapshot | None = None
        receipts = {}
        for row in rows:
            if publication_digest(row.payload) != row.payload_sha256:
                raise ValueError("temporal binding payload changed")
            binding = AnnexTemporalProjection.model_validate(row.payload)
            index = binding.index
            if (
                index.index_name != index_name
                or index.search_settings_id != settings.id
                or index.multitenant != MULTI_TENANT
            ):
                continue
            if index.encoder_authority != actual:
                raise ValueError(
                    "qualified query runtime encoder differs from activated authority"
                )
            if (
                accepted is not None
                and accepted.temporal_lookup_identity()
                != index.temporal_lookup_identity()
            ):
                raise ValueError("ambiguous activated query index authority")
            accepted = index
            receipts.update(
                {
                    receipt.configuration_json: receipt
                    for receipt in index.encoder_receipts
                }
            )
        if accepted is None:
            raise ValueError(
                "qualified query has no activated compatible index authority"
            )
        return PublicationIndexSnapshot.model_validate(
            accepted.model_copy(
                update={"encoder_receipts": tuple(receipts.values())}
            ).model_dump(mode="json")
        )


def current_file_read_owners(file_id: str) -> tuple[tuple[UUID, ...], frozenset[UUID]]:
    """Return source parents and the protected subset whose bytes are raw originals."""
    from onyx.db.engine.sql_engine import get_session_with_current_tenant

    with get_session_with_current_tenant() as session:
        parents = public_file_source_owners(session, file_id)
        originals = tuple(
            session.scalars(select(UserFile.id).where(UserFile.file_id == file_id))
        )
        return parents, protected_file_ids(session, originals)


def current_canonical_positions(chunk_ids: tuple[str, ...]) -> dict[str, int]:
    from onyx.db.engine.sql_engine import get_session_with_current_tenant

    with get_session_with_current_tenant() as session:
        return dict(
            session.execute(
                select(RegulatoryChunk.id, RegulatoryChunk.position).where(
                    RegulatoryChunk.id.in_(chunk_ids)
                )
            )
            .tuples()
            .all()
        )
