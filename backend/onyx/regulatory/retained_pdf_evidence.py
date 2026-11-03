"""Exact original PDF receipts without conversion or publication side effects."""

import datetime
import hashlib
import io
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

MAX_RETAINED_PDF_BYTES = 25 * 1024 * 1024
PdfRenditionRelation = Literal["equivalent_rendition", "historical_rendition"]


class RetainedPdfDescriptor(BaseModel):
    model_config = ConfigDict(extra="forbid")
    document_set_id: int = Field(gt=0)
    expected_canonical_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    idempotency_key: UUID
    original_filename: str = Field(min_length=1, max_length=500)
    relation: PdfRenditionRelation
    reason: str = Field(min_length=1, max_length=2000)
    source_url: str | None = Field(default=None, max_length=8192)

    @field_validator("original_filename")
    @classmethod
    def safe_filename(cls, value: str) -> str:
        if any(character in value for character in "/\\\r\n\0"):
            raise ValueError("Supply the original PDF filename without a path")
        return value

    @field_validator("reason")
    @classmethod
    def review_reason(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Supply a review reason")
        return value.strip()

    @field_validator("source_url")
    @classmethod
    def public_source_url(cls, value: str | None) -> str | None:
        from pydantic import HttpUrl, TypeAdapter

        if value is not None:
            TypeAdapter(HttpUrl).validate_python(value)
        return value


class RetainedPdfReceipt(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    document_set_id: int
    user_file_id: UUID
    sha256: str
    byte_count: int
    mime_type: str
    original_filename: str
    relation: PdfRenditionRelation
    reason: str
    source_url: str | None
    canonical_sha256: str
    canonical_generation_hash: str | None
    idempotency_key: UUID
    created_by: UUID | None
    created_at: datetime.datetime


class RetainedPdfList(BaseModel):
    document_set_id: int
    user_file_id: UUID
    canonical_sha256: str
    canonical_generation_hash: str | None
    items: list[RetainedPdfReceipt]


def validate_retained_pdf(content: bytes, mime_type: str | None) -> str:
    from pypdf import PdfReader
    from pypdf.errors import PyPdfError

    if not 0 < len(content) <= MAX_RETAINED_PDF_BYTES or mime_type != "application/pdf":
        raise ValueError("Supply an application/pdf original between 1 byte and 25 MiB")
    try:
        reader = PdfReader(io.BytesIO(content), strict=True)
        if reader.is_encrypted or not reader.pages:
            raise ValueError("An unencrypted PDF with at least one page is required")
        for page in reader.pages:
            _ = page.mediabox
    except (PyPdfError, OSError) as error:
        raise ValueError("The original PDF cannot be read") from error
    return hashlib.sha256(content).hexdigest()
