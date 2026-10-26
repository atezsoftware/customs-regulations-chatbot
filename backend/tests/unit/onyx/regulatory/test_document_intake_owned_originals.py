from hashlib import sha256
from io import BytesIO
from typing import IO, cast
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest

from onyx.configs.constants import FileOrigin
from onyx.connectors.models import Document
from onyx.file_processing.original_ingestion import LoadedUserFileDocuments
from onyx.file_store.file_store import FileStore
from onyx.regulatory.document_intake.models import IntakeSourceAsset
from onyx.regulatory.document_intake.preparation import (
    clone_intake_original,
    prepare_intake_file,
)


def test_deleting_owned_original_retains_frozen_source_and_preparation_proof() -> None:
    raw, text = b"%PDF original", b"MADDE 1 - Original rule."
    asset = IntakeSourceAsset(
        id=uuid4(),
        package_id=uuid4(),
        sha256=sha256(raw).hexdigest(),
        file_id="frozen-source",
        text_file_id="frozen-text",
        text_sha256=sha256(text).hexdigest(),
        mime_type="application/pdf",
        display_name="EK-1.pdf",
        byte_count=len(raw),
    )
    contents = {asset.file_id: raw, "frozen-text": text}
    store = MagicMock(spec=FileStore)
    store.read_file.side_effect = lambda identifier: BytesIO(contents[identifier])

    def save(content: IO[bytes], **kwargs: object) -> str:
        assert kwargs["file_origin"] == FileOrigin.USER_FILE
        identifier = str(kwargs["file_id"])
        contents[identifier] = content.read()
        return identifier

    store.save_file.side_effect = save
    store.delete_file.side_effect = lambda identifier: contents.pop(identifier)
    intake_id = uuid4()
    owned_id = clone_intake_original(
        intake_id=intake_id, asset=asset, store=cast(FileStore, store)
    )
    assert owned_id != asset.file_id
    assert contents[owned_id] == raw
    assert (
        clone_intake_original(
            intake_id=intake_id, asset=asset, store=cast(FileStore, store)
        )
        == owned_id
    )
    assert set(contents) == {asset.file_id, "frozen-text", owned_id}
    prepared: list[Document] = []

    def chunk(_identifier: UUID, _tenant: str, documents: list[Document]) -> bool:
        assert isinstance(documents, LoadedUserFileDocuments)
        assert documents.original_extraction.file_id == owned_id
        assert documents.original_extraction.raw_sha256 == asset.sha256
        prepared.extend(documents)
        return True

    prepare_intake_file(
        package_id=asset.package_id,
        asset=asset,
        user_file_id=uuid4(),
        tenant_id="tenant",
        links=[],
        store=cast(FileStore, store),
        original_file_id=owned_id,
        chunk_file=chunk,
        store_plaintext=lambda *_args: True,
    )
    assert prepared[0].file_id == owned_id
    store.delete_file(owned_id)
    assert contents == {asset.file_id: raw, "frozen-text": text}


def test_changed_owned_copy_cannot_reach_canonical_chunking() -> None:
    raw, text = b"original", b"MADDE 1 - Original rule."
    asset = IntakeSourceAsset(
        id=uuid4(),
        package_id=uuid4(),
        sha256=sha256(raw).hexdigest(),
        file_id="frozen-source",
        text_file_id="frozen-text",
        text_sha256=sha256(text).hexdigest(),
        mime_type="application/pdf",
        display_name="law.pdf",
        byte_count=len(raw),
    )
    contents = {"frozen-source": raw, "frozen-text": text, "owned-original": b"changed"}
    store = MagicMock(spec=FileStore)
    store.read_file.side_effect = lambda identifier: BytesIO(contents[identifier])
    with pytest.raises(ValueError, match="integrity"):
        prepare_intake_file(
            package_id=asset.package_id,
            asset=asset,
            user_file_id=uuid4(),
            tenant_id="tenant",
            links=[],
            store=cast(FileStore, store),
            original_file_id="owned-original",
            chunk_file=lambda *_args: pytest.fail("Changed original was chunked"),
        )
