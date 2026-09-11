"""Historical citation preview through the real route and qualified PG/ES readers."""

import json
from datetime import date, datetime, time, timedelta, timezone
from uuid import UUID, uuid4

import pytest
from elasticsearch import Elasticsearch
from fastapi import HTTPException
from sqlalchemy import select

from onyx.db.engine.sql_engine import get_session_with_tenant
from tests.external_dependency_unit.regulatory.test_publication_primitives import (
    es as es,
)
from tests.external_dependency_unit.regulatory.test_publication_primitives import (
    owned_file as owned_file,
)


def test_historical_citation_preview_retains_qualified_read_guards(
    owned_file: UUID, es: tuple[Elasticsearch, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from onyx.access.utils import prefix_user_email
    from onyx.context.search.models import IndexFilters
    from onyx.db.enums import IndexModelStatus
    from onyx.db.models import (
        RegulatoryChunk,
        RegulatoryTemporalProjection,
        SearchSettings,
        User,
        UserFile,
    )
    from onyx.db.regulatory_context_projections import activate_temporal_projection
    from onyx.document_index.elasticsearch.publication import FencedPublicationIndex
    from onyx.document_index.factory import build_elasticsearch_document_index
    from onyx.document_index.interfaces_new import DocumentSectionRequest
    from onyx.document_index.publication_models import (
        FrozenPublicationProjection,
        PublicationEncoderAuthority,
        PublicationEncoderReceipt,
        PublicationIndexSnapshot,
        publication_digest,
    )
    from onyx.regulatory.amendments.annexes.context_dependencies import context_hash
    from onyx.regulatory.amendments.annexes.models import AnnexTemporalProjection
    from onyx.regulatory.publication_reads import public_read_store
    from onyx.server.documents import document as route
    from tests.external_dependency_unit.regulatory.test_publication_primitives import (
        frozen_projection,
    )

    setting_id = None
    authority = public_read_store()
    owner = authority.acquire(owned_file, owner_id=uuid4(), ttl=timedelta(minutes=2))
    try:
        with get_session_with_tenant(tenant_id="public") as session:
            setting = SearchSettings(
                index_name=es[1],
                model_name="fixture",
                model_dim=3,
                normalize=True,
                status=IndexModelStatus.PAST,
                query_prefix="",
                passage_prefix="",
                enable_contextual_rag=False,
            )
            session.add(setting)
            session.flush()
            setting_id = setting.id
            file = session.get_one(UserFile, owned_file)
            user = session.get_one(User, file.user_id)
            acl = [prefix_user_email(user.email)]
            rows = list(
                session.scalars(
                    select(RegulatoryChunk)
                    .where(RegulatoryChunk.user_file_id == owned_file)
                    .order_by(RegulatoryChunk.position)
                )
            )
            rows[0].text = "The rate is 5%."
            rows[0].validity_start_date = date(2020, 1, 1)
            rows[0].validity_end_date = date(2026, 9, 10)
            rows[1].text = "The rate is 7%."
            rows[1].validity_start_date = date(2026, 9, 10)
            rows[1].validity_end_date = None
            session.commit()
            encoder = PublicationEncoderAuthority(
                provider=None,
                model=setting.model_name,
                effective_dimension=3,
                endpoint_sha256=context_hash(None),
                deployment_name=None,
                api_version=None,
                normalize=True,
                passage_prefix=setting.passage_prefix,
            )
            config = encoder.model_dump(mode="json")
            config["dimension"] = config.pop("effective_dimension")
            receipt = PublicationEncoderReceipt(
                configuration_json=json.dumps(config),
                authority=encoder,
                resolution_sha256=publication_digest(config),
            )
            index = PublicationIndexSnapshot(
                index_name=es[1],
                index_uuid=es[0].indices.get(index=es[1])[es[1]]["settings"]["index"][
                    "uuid"
                ],
                search_settings_id=setting.id,
                model_provider="",
                model_name="fixture",
                vector_dimension=3,
                embedding_config_sha256=publication_digest(config),
                multitenant=False,
                encoder_authority=encoder,
                encoder_receipts=(receipt,),
            )
            adapter = FencedPublicationIndex(es[0], index)
            authority.close_gate(owner)
            inventory = authority.reservations(owner)
            adapter.seal(inventory)
            bindings = []
            for row in rows:
                base = frozen_projection(owned_file, row.projection_ordinal, row.text)
                source = json.loads(base.source_json)

                def epoch(when: date | None) -> int | None:
                    return (
                        int(datetime.combine(when, time(), timezone.utc).timestamp())
                        if when
                        else None
                    )

                source.update(
                    regulatory_chunk_id=row.id,
                    access_control_list=acl,
                    doc_summary="",
                    chunk_context="",
                    max_chunk_size=DocumentSectionRequest(
                        document_id=str(owned_file)
                    ).max_chunk_size,
                    validity_start_date=epoch(row.validity_start_date),
                    validity_end_date=epoch(row.validity_end_date),
                    source_links=json.dumps({0: ""}),
                    image_file_id=None,
                )
                identity = uuid4()
                projection = FrozenPublicationProjection(
                    ordinal=row.projection_ordinal,
                    context_projection_id=str(identity),
                    source_json=json.dumps(source),
                    embedding_inputs=(row.text,),
                    embedding_config_json=json.dumps(config),
                )
                binding = AnnexTemporalProjection(
                    id=identity,
                    index=index,
                    projection=projection,
                    canonical_base_sha256=context_hash(row.text),
                    derived_role="canonical",
                    dependency_ids=[],
                    representation_text=row.text,
                    reference_date=row.validity_start_date,
                    effective_start=row.validity_start_date,
                    effective_end=row.validity_end_date,
                    semantic_position=row.position,
                )
                bindings.append(binding)
                adapter.upsert(inventory, projection)
            proof = adapter.verify(
                inventory, tuple(item.projection for item in bindings)
            )
            for binding in bindings:
                activate_temporal_projection(
                    session, user_file_id=owned_file, binding=binding
                )
            authority.finalize(session, owner, proof)
            session.commit()
            es[0].indices.refresh(index=es[1])
            reader = build_elasticsearch_document_index(setting)
            monkeypatch.setattr(route, "get_current_search_settings", lambda _: setting)
            monkeypatch.setattr(route, "get_default_document_index", lambda *_: reader)
            from unittest.mock import Mock

            monkeypatch.setattr(route, "get_tokenizer", lambda **_: Mock(encode=list))

            def preview(
                ordinal: int, caller: User = user, document_id: str = str(owned_file)
            ) -> str:
                return route.get_chunk_info(
                    document_id=document_id,
                    chunk_id=ordinal,
                    user=caller,
                    db_session=session,
                ).content

            old, current = (item.projection.ordinal for item in bindings)
            assert (
                reader.id_based_retrieval(
                    [
                        DocumentSectionRequest(
                            document_id=str(owned_file),
                            min_chunk_ind=old,
                            max_chunk_ind=old,
                        )
                    ],
                    IndexFilters(access_control_list=acl),
                )
                == []
            ), "ordinary current-date retrieval must continue to hide the old rate"
            assert "7%" in preview(current)
            assert "5%" in preview(old)
            for ordinal, caller, document_id in (
                (
                    old,
                    User(id=uuid4(), email="other@test.local", prior_emails=[]),
                    str(owned_file),
                ),
                (old, user, str(uuid4())),
                (999, user, str(owned_file)),
            ):
                with pytest.raises(HTTPException) as missing:
                    preview(ordinal, caller, document_id)
                assert missing.value.status_code == 404
            authority.close_gate(owner)
            with pytest.raises(HTTPException) as gated:
                preview(old)
            assert gated.value.status_code == 404
            authority.finalize(
                session,
                owner,
                adapter.verify(
                    authority.reservations(owner),
                    tuple(item.projection for item in bindings),
                ),
            )
            session.commit()
            retained = session.get_one(RegulatoryTemporalProjection, bindings[0].id)
            frozen_payload = dict(retained.payload)
            retained.retired_at = datetime.now(timezone.utc)
            session.commit()
            with pytest.raises(HTTPException) as retired:
                preview(old)
            assert retired.value.status_code == 404
            assert retained.payload == frozen_payload
    finally:
        authority.release(owner)
        if setting_id is not None:
            with get_session_with_tenant(tenant_id="public") as session:
                setting = session.get_one(SearchSettings, setting_id)
                session.delete(setting)
                session.commit()
