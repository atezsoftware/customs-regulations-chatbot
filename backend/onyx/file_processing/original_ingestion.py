"""Proof linking the actual extracted original stream to initial canonical input."""

from collections.abc import Sequence
from hashlib import sha256
from typing import Literal

from pydantic import BaseModel, ConfigDict

from onyx.connectors.models import Document
from onyx.document_index.publication_models import publication_digest


class OriginalExtractionReceipt(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    version: Literal[1] = 1
    file_id: str
    raw_sha256: str
    documents_sha256: str
    plaintext_sha256: str


class OriginalIngestionReceipt(OriginalExtractionReceipt):
    canonical_sha256: str
    generation_hash: str


def documents_sha256(documents: Sequence[Document]) -> str:
    return publication_digest(
        [
            document.model_dump(mode="json", exclude={"doc_updated_at"})
            for document in documents
        ]
    )


def extracted_plaintext(documents: Sequence[Document]) -> str:
    return " ".join(
        text
        for document in documents
        for section in document.sections
        if (text := section.materialize_text())
    )


class LoadedUserFileDocuments(list[Document]):
    """List-compatible transport; proof never enters provider or document metadata."""

    def __init__(
        self, documents: list[Document], *, file_id: str, raw_sha256: str
    ) -> None:
        super().__init__(documents)
        self.original_extraction = OriginalExtractionReceipt(
            file_id=file_id,
            raw_sha256=raw_sha256,
            documents_sha256=documents_sha256(documents),
            plaintext_sha256=sha256(
                extracted_plaintext(documents).encode()
            ).hexdigest(),
        )
