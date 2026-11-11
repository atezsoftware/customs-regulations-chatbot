"""Parent-scoped source-image associations and ACLs use real PostgreSQL rows."""

import json
from datetime import datetime, timezone
from uuid import uuid4

import pytest
from sqlalchemy import event
from sqlalchemy.orm import Session

from onyx.access.access import user_can_access_chat_file
from onyx.auth.schemas import UserRole
from onyx.configs.constants import FileOrigin
from onyx.db.enums import Permission
from onyx.db.models import (
    DocumentSet,
    FileRecord,
    RegulatoryChunk,
    RegulatoryTemporalProjection,
    User,
    UserFile,
)
from onyx.db.regulatory_public_reads import (
    public_file_source_owners,
    regulatory_source_image_owners,
)
from onyx.document_index.publication_models import publication_digest
from tests.external_dependency_unit.regulatory.test_labeling_jobs import (
    LabelingData,
)
from tests.external_dependency_unit.regulatory.test_labeling_jobs import (
    labeling_data as labeling_data,
)
from tests.external_dependency_unit.regulatory.test_labeling_jobs import (
    labeling_database as labeling_database,
)


def image_record(
    parent: str, *, purpose: str = "regulatory_source_image"
) -> FileRecord:
    return FileRecord(
        file_id=str(uuid4()),
        file_origin=FileOrigin.OTHER,
        file_type="image/png",
        file_metadata={"purpose": purpose, "regulatory_user_file_id": parent},
        bucket_name="postgres",
        object_key=str(uuid4()),
    )


def temporal(
    parent: UserFile,
    image_id: str,
    *,
    source_parent: str | None = None,
    malformed: bool = False,
) -> RegulatoryTemporalProjection:
    payload = {
        "projection": {
            "source_json": "not-json"
            if malformed
            else json.dumps(
                {
                    "document_id": source_parent or str(parent.id),
                    "image_file_id": image_id,
                }
            )
        }
    }
    return RegulatoryTemporalProjection(
        id=uuid4(),
        user_file_id=parent.id,
        canonical_chunk_id="historical-canonical-source",
        index_uuid=str(uuid4()),
        index_identity_sha256="a" * 64,
        projection_ordinal=7,
        payload=payload,
        payload_sha256=publication_digest(payload),
        retired_at=datetime.now(timezone.utc),
    )


@pytest.mark.parametrize(
    "link",
    ["none", "canonical", "canonical_secondary", "canonical_plural", "historical"],
)
@pytest.mark.parametrize("public", [False, True])
def test_actual_link_required_before_private_or_public_parent_acl(
    labeling_data: LabelingData, link: str, public: bool
) -> None:
    with Session(labeling_data.database.engine) as session:
        owner = session.get_one(User, labeling_data.user_id)
        parent = session.get_one(UserFile, labeling_data.file_id)
        group = session.get_one(DocumentSet, labeling_data.document_set_id)
        group.is_public = public
        outsider = User(
            id=uuid4(),
            email=f"source-image-{uuid4().hex}@test.local",
            hashed_password="unused",
            role=UserRole.BASIC,
            effective_permissions=[Permission.BASIC_ACCESS.value],
            is_active=True,
            is_verified=True,
            is_superuser=False,
        )
        session.add(outsider)
        image = image_record(str(parent.id))
        session.add(image)
        if link in {"canonical", "canonical_secondary", "canonical_plural"}:
            chunk = session.get_one(RegulatoryChunk, labeling_data.first_id)
            chunk.chunk_metadata = {
                **chunk.chunk_metadata,
                "image_file_id": image.file_id,
            }
            if link == "canonical_secondary":
                primary = image_record(str(parent.id))
                session.add(primary)
                chunk.chunk_metadata = {
                    **chunk.chunk_metadata,
                    "image_file_id": primary.file_id,
                    "image_file_ids": [primary.file_id, image.file_id],
                }
            elif link == "canonical_plural":
                chunk.chunk_metadata = {"image_file_ids": [image.file_id]}
        elif link == "historical":
            session.add(temporal(parent, image.file_id))
        session.flush()
        assert regulatory_source_image_owners(session, image.file_id) == (
            (parent.id,) if link != "none" else ()
        )
        assert public_file_source_owners(session, image.file_id) == (
            (parent.id,) if link != "none" else ()
        )
        assert user_can_access_chat_file(image.file_id, owner, session) is (
            link != "none"
        )
        assert user_can_access_chat_file(image.file_id, outsider, session) is (
            link != "none" and public
        )
        session.rollback()


@pytest.mark.parametrize(
    "malformation",
    [
        "missing_parent",
        "invalid_parent",
        "nonstring_parent",
        "foreign_parent",
        "wrong_origin",
        "wrong_mime",
        "foreign_temporal_source",
        "foreign_canonical",
        "foreign_secondary_canonical",
        "secondary_scalar",
        "secondary_object",
        "secondary_nested_array",
    ],
)
def test_metadata_and_foreign_links_never_authorize(
    labeling_data: LabelingData, malformation: str
) -> None:
    with Session(labeling_data.database.engine) as session:
        owner = session.get_one(User, labeling_data.user_id)
        parent = session.get_one(UserFile, labeling_data.file_id)
        image = image_record(str(parent.id))
        session.add(image)
        chunk = session.get_one(RegulatoryChunk, labeling_data.first_id)
        if malformation == "missing_parent":
            image.file_metadata = {"purpose": "regulatory_source_image"}
        elif malformation == "invalid_parent":
            image.file_metadata = {
                "purpose": "regulatory_source_image",
                "regulatory_user_file_id": "invalid",
            }
        elif malformation == "nonstring_parent":
            image.file_metadata = {
                "purpose": "regulatory_source_image",
                "regulatory_user_file_id": 123,
            }
        elif malformation == "foreign_parent":
            image.file_metadata = {
                "purpose": "regulatory_source_image",
                "regulatory_user_file_id": str(uuid4()),
            }
        elif malformation == "wrong_origin":
            image.file_origin = FileOrigin.CHAT_IMAGE_GEN
        elif malformation == "wrong_mime":
            image.file_type = "application/pdf"
        elif malformation == "foreign_temporal_source":
            session.add(temporal(parent, image.file_id, source_parent=str(uuid4())))
        elif malformation in {"foreign_canonical", "foreign_secondary_canonical"}:
            foreign = UserFile(
                id=uuid4(),
                user_id=owner.id,
                file_id=str(uuid4()),
                name="Other source",
                file_type="application/pdf",
            )
            session.add(foreign)
            session.flush()
            chunk.user_file_id = foreign.id
        if malformation == "foreign_secondary_canonical":
            primary = image_record(str(chunk.user_file_id))
            session.add(primary)
            chunk.chunk_metadata = {
                "image_file_id": primary.file_id,
                "image_file_ids": [primary.file_id, image.file_id],
            }
        elif malformation.startswith("secondary_"):
            primary = image_record(str(parent.id))
            session.add(primary)
            chunk.chunk_metadata = {
                "image_file_id": primary.file_id,
                "image_file_ids": image.file_id
                if malformation == "secondary_scalar"
                else {image.file_id: True}
                if malformation == "secondary_object"
                else [[image.file_id]],
            }
        if malformation not in {
            "foreign_temporal_source",
            "foreign_secondary_canonical",
            "secondary_scalar",
            "secondary_object",
            "secondary_nested_array",
        }:
            chunk.chunk_metadata = {
                **chunk.chunk_metadata,
                "image_file_id": image.file_id,
            }
        session.flush()
        assert regulatory_source_image_owners(session, image.file_id) == ()
        assert public_file_source_owners(session, image.file_id) == ()
        assert not user_can_access_chat_file(image.file_id, owner, session)
        session.rollback()


def test_unlinked_lookup_never_parses_unrelated_temporal_payloads(
    labeling_data: LabelingData,
) -> None:
    with Session(labeling_data.database.engine) as session:
        parent = session.get_one(UserFile, labeling_data.file_id)
        foreign = UserFile(
            id=uuid4(),
            user_id=parent.user_id,
            file_id=str(uuid4()),
            name="Unrelated corpus",
            file_type="application/pdf",
        )
        session.add(foreign)
        session.flush()
        image = image_record(str(parent.id))
        session.add(image)
        session.add(temporal(foreign, image.file_id, malformed=True))
        session.flush()
        session.expunge(image)
        statements: list[str] = []

        def capture(
            _connection: object,
            _cursor: object,
            statement: str,
            _parameters: object,
            _context: object,
            _executemany: object,
        ) -> None:
            statements.append(statement)

        event.listen(session.connection(), "before_cursor_execute", capture)
        try:
            assert public_file_source_owners(session, image.file_id) == ()
        finally:
            event.remove(session.connection(), "before_cursor_execute", capture)
        temporal_queries = [
            statement
            for statement in statements
            if "regulatory_temporal_projection" in statement
        ]
        assert len(temporal_queries) == 1
        assert "CASE WHEN" in temporal_queries[0]
        assert (
            "user_file_id =" in temporal_queries[0] and "LIMIT" in temporal_queries[0]
        )
        assert len(statements) == 3
        assert not any(
            "regulatory_source_asset" in statement or "connector" in statement
            for statement in statements
        )
        session.rollback()


def test_legacy_image_association_and_generated_origin_keep_existing_behavior(
    labeling_data: LabelingData,
) -> None:
    with Session(labeling_data.database.engine) as session:
        owner = session.get_one(User, labeling_data.user_id)
        image = image_record(str(labeling_data.file_id), purpose="legacy_other")
        session.add(image)
        chunk = session.get_one(RegulatoryChunk, labeling_data.first_id)
        chunk.chunk_metadata = {**chunk.chunk_metadata, "image_file_id": image.file_id}
        generated = image_record(str(labeling_data.file_id), purpose="legacy_generated")
        generated.file_origin = FileOrigin.CHAT_IMAGE_GEN
        session.add(generated)
        session.flush()
        assert regulatory_source_image_owners(session, image.file_id) is None
        assert public_file_source_owners(session, image.file_id) == (
            labeling_data.file_id,
        )
        assert user_can_access_chat_file(image.file_id, owner, session)
        assert regulatory_source_image_owners(session, generated.file_id) is None
        assert user_can_access_chat_file(generated.file_id, owner, session)
        session.rollback()
