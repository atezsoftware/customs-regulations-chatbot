"""Validate exact source-page images without conversion or document ingestion."""

import hashlib
import io
from typing import Literal
from uuid import UUID

from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict

MAX_SOURCE_IMAGE_BYTES = 8 * 1024 * 1024
MAX_SOURCE_IMAGE_PIXELS = 16_000_000
MAX_SOURCE_IMAGE_DIMENSION = 8192
SourceImageMimeType = Literal["image/png", "image/jpeg"]


class SourceImageDescriptor(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    sha256: str
    byte_count: int
    mime_type: SourceImageMimeType
    width: int
    height: int


class SourceImageUploadReceipt(SourceImageDescriptor):
    file_id: str
    user_file_id: UUID


def validate_source_image(
    content: bytes, mime_type: str | None
) -> SourceImageDescriptor:
    if not 0 < len(content) <= MAX_SOURCE_IMAGE_BYTES:
        raise ValueError("Supply an image between 1 byte and 8 MiB")
    if mime_type == "image/png":
        validated_mime: SourceImageMimeType = "image/png"
        expected_format = "PNG"
        if not content.startswith(b"\x89PNG\r\n\x1a\n") or not content.endswith(
            b"\x00\x00\x00\x00IEND\xaeB`\x82"
        ):
            raise ValueError("Supply a complete PNG image")
    elif mime_type == "image/jpeg":
        validated_mime = "image/jpeg"
        expected_format = "JPEG"
        if not content.startswith(b"\xff\xd8\xff") or not content.endswith(b"\xff\xd9"):
            raise ValueError("Supply a complete JPEG image")
    else:
        raise ValueError("Only image/png and image/jpeg source images are supported")

    try:
        with Image.open(io.BytesIO(content), formats=[expected_format]) as image:
            width, height = image.size
            if (
                image.format != expected_format
                or not 0 < width <= MAX_SOURCE_IMAGE_DIMENSION
                or not 0 < height <= MAX_SOURCE_IMAGE_DIMENSION
                or width * height > MAX_SOURCE_IMAGE_PIXELS
                or getattr(image, "n_frames", 1) != 1
            ):
                raise ValueError(
                    "Supply one image of at most 16 megapixels and 8192 pixels per side"
                )
            image.verify()
        # JPEG verify() checks its container; decoding also rejects truncated pixels.
        with Image.open(io.BytesIO(content), formats=[expected_format]) as image:
            image.load()
    except (
        UnidentifiedImageError,
        OSError,
        SyntaxError,
        Image.DecompressionBombError,
    ) as error:
        raise ValueError("The source image cannot be decoded") from error

    return SourceImageDescriptor(
        sha256=hashlib.sha256(content).hexdigest(),
        byte_count=len(content),
        mime_type=validated_mime,
        width=width,
        height=height,
    )
