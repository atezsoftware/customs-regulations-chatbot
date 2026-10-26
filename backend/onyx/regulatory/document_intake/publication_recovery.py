"""Rebuild one approved intake publication while its original gate stays closed."""

import json
from uuid import UUID, uuid4

from onyx.db.document_intake_publication_recovery import (
    finish_recovery,
    load_recovery_inputs,
    stage_recovery_replacement,
)
from onyx.db.regulatory_publication import PublicationStore
from onyx.db.user_file import validate_user_file_index_request
from onyx.document_index.elasticsearch.client import ElasticsearchClient
from onyx.document_index.publication_models import (
    FileOwnership,
    FrozenPublicationProjection,
    PublicationScope,
)
from onyx.regulatory.amendments.annexes import config
from onyx.regulatory.amendments.annexes.publication_execution import (
    LEASE_TTL,
    publication_heartbeat,
)
from onyx.regulatory.writer_projection import prepare_owned_correction
from onyx.regulatory.writer_publication import (
    complete_writer_index_inventory,
    execute_writer_publication,
)
from onyx.regulatory.writer_publication_models import WriterPublicationManifest


def _fresh_projection_ordinals(
    owner: FileOwnership, recovery_id: UUID, manifest: WriterPublicationManifest
) -> WriterPublicationManifest:
    bindings = []
    for binding in manifest.bindings:
        if not isinstance(binding.projection, FrozenPublicationProjection):
            raise ValueError("An unpublished recovery requires newly prepared vectors")
        ordinal = PublicationStore(owner.scope).allocate(
            owner,
            f"intake-recovery:{recovery_id}:{binding.index.index_uuid}:{binding.id}",
        )
        source = json.loads(binding.projection.source_json)
        source["chunk_index"] = ordinal
        projection = FrozenPublicationProjection.model_validate(
            {
                **binding.projection.model_dump(mode="json"),
                "ordinal": ordinal,
                "source_json": json.dumps(source, ensure_ascii=False),
            }
        )
        bindings.append(binding.model_copy(update={"projection": projection}))
    return WriterPublicationManifest.model_validate(
        manifest.model_copy(update={"bindings": bindings}).model_dump(mode="json")
    )


def recover_intake_publication(
    *, user_file_id: UUID, tenant_id: str, attempt_id: UUID, recovery_id: UUID
) -> None:
    authority = PublicationStore(
        PublicationScope(
            tenant_id=tenant_id,
            environment=config.REGULATORY_ANNEX_ENVIRONMENT,
            database_identity=config.ANNEX_DATABASE_IDENTITY,
        )
    )
    owner = authority.acquire(user_file_id, owner_id=uuid4(), ttl=LEASE_TTL)
    try:
        validate_user_file_index_request(owner, attempt_id)
        inputs = load_recovery_inputs(owner, recovery_id, attempt_id)
        if inputs == "completed":
            return
        if inputs is not None:
            with publication_heartbeat(owner) as lost:
                with ElasticsearchClient() as transport:
                    manifest = prepare_owned_correction(
                        owner,
                        transport.publication_client(),
                        inputs,
                        inputs.canonical,
                        changed_id=None,
                    )
                    manifest = manifest.model_copy(
                        update={
                            "complete_file": True,
                            "chunk_count_after": len(inputs.canonical),
                            "secondary_reconcile_pending": False,
                        }
                    )
                    manifest = _fresh_projection_ordinals(owner, recovery_id, manifest)
                    manifest = complete_writer_index_inventory(
                        owner, transport.publication_client(), manifest
                    )
                if lost.is_set():
                    raise ValueError("Publication recovery ownership heartbeat lost")
            stage_recovery_replacement(owner, recovery_id, attempt_id, manifest)
        # A prepared replacement is replayed verbatim, without repeating model work.
        with ElasticsearchClient() as transport:
            execute_writer_publication(owner, transport.publication_client())
        finish_recovery(owner, recovery_id)
    except Exception as error:
        try:
            finish_recovery(owner, recovery_id, error=type(error).__name__)
        except ValueError:
            pass
        raise
    finally:
        try:
            authority.release(owner)
        except ValueError:
            pass
