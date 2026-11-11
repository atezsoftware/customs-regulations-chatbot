"""Freeze embedded image originals and their transcription under the Markdown owner."""

import hashlib
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass
from io import BytesIO
from typing import Literal, cast
from uuid import UUID, uuid5

from pydantic import BaseModel, ConfigDict, Field

from onyx.configs.constants import FileOrigin
from onyx.connectors.models import Document
from onyx.db.file_record import FileRecordNotFoundError
from onyx.db.models import FileRecord
from onyx.file_store.file_store import FileStore
from onyx.regulatory.amendments.annexes.models import SourceLink
from onyx.regulatory.document_intake.markdown_images import (
    EmbeddedMarkdownImage,
    embedded_markdown_images,
)
from onyx.regulatory.document_intake.models import IntakeSourceAsset

_RECEIPT_MIME = "application/json"
_RECEIPT_PURPOSE = "markdown_image_transcription"
_MAX_RECEIPT_BYTES = 1024 * 1024
MARKDOWN_IMAGES_KEY = "regulatory_markdown_images"


class MarkdownImageReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version: Literal[1] = 1
    user_file_id: UUID
    source_asset_id: UUID
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    image_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    image_file_id: str
    transcript_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    transcript: str = Field(min_length=1, max_length=200_000)
    model_provider: str
    model_name: str


class MarkdownImageSpan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    start: int = Field(ge=0)
    end: int = Field(gt=0)
    image_file_id: str = Field(min_length=1, max_length=256)


class MarkdownImageEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version: Literal[1] = 1
    user_file_id: UUID
    source_asset_id: UUID
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    plaintext_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    spans: tuple[MarkdownImageSpan, ...] = Field(min_length=1, max_length=32)


@dataclass(frozen=True)
class PreparedMarkdownImages:
    text: str
    spans: tuple[MarkdownImageSpan, ...]


def _sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _read(store: FileStore, file_id: str, *, limit: int) -> bytes:
    try:
        with store.read_file(file_id) as stream:
            content = stream.read(limit + 1)
    except (FileNotFoundError, FileRecordNotFoundError, KeyError, ValueError) as error:
        raise ValueError(
            "The frozen Markdown image source or receipt is missing"
        ) from error
    if len(content) > limit:
        raise ValueError("The frozen Markdown image receipt exceeds its budget")
    return content


def _record(store: FileStore, file_id: str) -> FileRecord | None:
    try:
        return store.read_file_record(file_id)
    except FileRecordNotFoundError:
        return None


def _verify_image(
    store: FileStore,
    record: FileRecord,
    image: EmbeddedMarkdownImage,
    *,
    asset: IntakeSourceAsset,
    user_file_id: UUID,
) -> None:
    from onyx.regulatory.source_image_evidence import MAX_SOURCE_IMAGE_BYTES

    raw_metadata: object = record.file_metadata
    metadata = cast(dict[str, object], raw_metadata)
    if (
        record.file_origin != FileOrigin.OTHER
        or record.file_type != image.descriptor.mime_type
        or not isinstance(raw_metadata, dict)
        or metadata.get("regulatory_user_file_id") != str(user_file_id)
        or metadata.get("purpose") != "regulatory_source_image"
        or metadata.get("source_asset_id") != str(asset.id)
        or any(
            metadata.get(key) != value
            for key, value in image.descriptor.model_dump().items()
        )
        or _read(store, record.file_id, limit=MAX_SOURCE_IMAGE_BYTES) != image.content
    ):
        raise ValueError(
            "The frozen Markdown image source integrity or metadata changed"
        )


def _receipt(
    image: EmbeddedMarkdownImage,
    *,
    asset: IntakeSourceAsset,
    user_file_id: UUID,
    store: FileStore,
    allow_transcription: bool,
    deadline: float,
) -> MarkdownImageReceipt:
    identity = f"markdown-image-v1:{asset.sha256}:{image.descriptor.sha256}"
    image_id = str(uuid5(user_file_id, identity))
    receipt_id = str(uuid5(user_file_id, identity + ":transcript"))
    image_record = _record(store, image_id)
    if image_record is not None:
        _verify_image(
            store, image_record, image, asset=asset, user_file_id=user_file_id
        )
    record = _record(store, receipt_id)
    if record is not None:
        raw = _read(store, receipt_id, limit=_MAX_RECEIPT_BYTES)
        raw_metadata: object = record.file_metadata
        metadata = cast(dict[str, object], raw_metadata)
        if (
            not isinstance(raw_metadata, dict)
            or record.file_origin != FileOrigin.OTHER
            or record.file_type != _RECEIPT_MIME
            or metadata.get("purpose") != _RECEIPT_PURPOSE
            or metadata.get("regulatory_user_file_id") != str(user_file_id)
            or metadata.get("sha256") != _sha(raw)
        ):
            raise ValueError("The frozen Markdown image receipt integrity changed")
        receipt = MarkdownImageReceipt.model_validate_json(raw)
    else:
        if not allow_transcription:
            raise ValueError(
                "The frozen Markdown image transcription receipt is missing"
            )
        from onyx.regulatory.amendments.analysis_llm import get_amendment_analysis_llm
        from onyx.regulatory.amendments.annexes.models import AcquiredAsset
        from onyx.regulatory.amendments.annexes.source_images import (
            prepare_image_source,
        )

        source = AcquiredAsset(
            sha256=image.descriptor.sha256,
            content=image.content,
            mime_type=image.descriptor.mime_type,
            display_name=f"{asset.display_name} — embedded source",
            page_count=1,
        )
        model = get_amendment_analysis_llm(temperature=0)
        prepared = prepare_image_source(
            source,
            llm=model,
            deadline=deadline,
        )
        transcript = prepared.text.strip()
        if not transcript or embedded_markdown_images(
            transcript, path=asset.display_name
        ):
            raise ValueError("The Markdown image transcription is incomplete")
        receipt = MarkdownImageReceipt(
            user_file_id=user_file_id,
            source_asset_id=asset.id,
            source_sha256=asset.sha256,
            image_sha256=image.descriptor.sha256,
            image_file_id=image_id,
            transcript_sha256=_sha(transcript.encode()),
            transcript=transcript,
            model_provider=model.config.model_provider,
            model_name=model.config.model_name,
        )
        if len(receipt.model_dump_json().encode()) > _MAX_RECEIPT_BYTES:
            raise ValueError("The Markdown image transcription exceeds its budget")
        if image_record is None:
            store.save_file(
                BytesIO(image.content),
                source.display_name,
                FileOrigin.OTHER,
                image.descriptor.mime_type,
                file_id=image_id,
                file_metadata={
                    "purpose": "regulatory_source_image",
                    "regulatory_user_file_id": str(user_file_id),
                    "source_asset_id": str(asset.id),
                    **image.descriptor.model_dump(),
                },
            )
        raw = receipt.model_dump_json().encode()
        store.save_file(
            BytesIO(raw),
            "embedded-image-transcription.json",
            FileOrigin.OTHER,
            _RECEIPT_MIME,
            file_id=receipt_id,
            file_metadata={
                "purpose": _RECEIPT_PURPOSE,
                "regulatory_user_file_id": str(user_file_id),
                "sha256": _sha(raw),
            },
        )
    if (
        receipt.user_file_id != user_file_id
        or receipt.source_asset_id != asset.id
        or receipt.source_sha256 != asset.sha256
        or receipt.image_sha256 != image.descriptor.sha256
        or receipt.image_file_id != image_id
        or receipt.transcript_sha256 != _sha(receipt.transcript.encode())
    ):
        raise ValueError(
            "The frozen Markdown image source or transcript integrity changed"
        )
    image_record = store.read_file_record(image_id)
    _verify_image(store, image_record, image, asset=asset, user_file_id=user_file_id)
    return receipt


def prepare_markdown_images(
    text: str,
    *,
    asset: IntakeSourceAsset,
    user_file_id: UUID,
    store: FileStore,
    allow_transcription: bool,
    links: Sequence[SourceLink] = (),
    package_assets: Sequence[IntakeSourceAsset] = (),
) -> PreparedMarkdownImages:
    images = embedded_markdown_images(text, path=asset.display_name)
    if not images:
        return PreparedMarkdownImages(text, ())
    from onyx.regulatory.document_intake.text import (
        normalize_attachment_navigation,
        require_resolved_attachment_references,
    )

    require_resolved_attachment_references(text, asset_sha256=asset.sha256, links=links)
    normalized = normalize_attachment_navigation(
        text, asset_sha256=asset.sha256, links=links, package_assets=package_assets
    )
    if normalized != text:
        text = normalized
        images = embedded_markdown_images(text, path=asset.display_name)
    from onyx.regulatory.amendments.annexes.source_limits import (
        source_preparation_seconds,
    )

    deadline = time.monotonic() + source_preparation_seconds(len(images))
    replacements: list[tuple[int, int, str, str | None, int, int]] = []
    for image in images:
        receipt = _receipt(
            image,
            asset=asset,
            user_file_id=user_file_id,
            store=store,
            allow_transcription=allow_transcription,
            deadline=deadline,
        )
        captions = list(
            dict.fromkeys(
                value.strip()
                for value in (image.alt, image.title or "")
                if value.strip()
            )
        )
        rendered = "\n\n".join([*captions, receipt.transcript])
        line_start = text.rfind("\n", 0, image.start) + 1
        line_end = text.find("\n", image.end)
        line_end = len(text) if line_end < 0 else line_end
        in_table = bool(
            re.match(r"\s*(?:>\s*)*\|", text[line_start : image.start])
        ) and text[image.end : line_end].rstrip().endswith("|")
        inline = bool(
            text[line_start : image.start].strip() or text[image.end : line_end].strip()
        )
        if in_table:
            rendered = rendered.replace("\\", "\\\\").replace("|", "\\|")
        if inline:
            rendered = rendered.replace("\n", "<br/>")
        padding = "" if inline else "\n\n"
        replacements.append(
            (
                image.start,
                image.end,
                padding + rendered + padding,
                receipt.image_file_id,
                len(padding),
                len(rendered),
            )
        )
    resources = {span for image in images for span in image.resource_spans}
    replacements.extend((start, end, "\n", None, 0, 0) for start, end in resources)
    output: list[str] = []
    spans: list[MarkdownImageSpan] = []
    cursor = offset = 0
    for start, end, replacement, image_id, padding, rendered_length in sorted(
        replacements
    ):
        if not cursor <= start < end <= len(text):
            raise ValueError("Embedded Markdown image source spans overlap")
        prefix = text[cursor:start]
        output.extend((prefix, replacement))
        offset += len(prefix)
        if image_id is not None:
            spans.append(
                MarkdownImageSpan(
                    start=offset + padding,
                    end=offset + padding + rendered_length,
                    image_file_id=image_id,
                )
            )
        offset += len(replacement)
        cursor = end
    output.append(text[cursor:])
    return PreparedMarkdownImages("".join(output), tuple(spans))


def canonical_image_spans(
    before: str, after: str, spans: Sequence[MarkdownImageSpan]
) -> tuple[MarkdownImageSpan, ...]:
    """Map source spans across the whitespace-only canonical formatter."""
    if not spans:
        return ()
    left = [index for index, character in enumerate(before) if not character.isspace()]
    right = [index for index, character in enumerate(after) if not character.isspace()]
    if "".join(before[index] for index in left) != "".join(
        after[index] for index in right
    ):
        raise ValueError("Markdown image preparation changed the source wording")
    from bisect import bisect_left

    mapped: list[MarkdownImageSpan] = []
    cursor = 0
    for span in spans:
        lower = bisect_left(left, span.start)
        upper = bisect_left(left, span.end)
        if lower == upper:
            raise ValueError("A Markdown image has no reviewable transcription")
        start, end = right[lower], right[upper - 1] + 1
        if start < cursor:
            raise ValueError("Canonical Markdown image spans overlap")
        mapped.append(
            MarkdownImageSpan(start=start, end=end, image_file_id=span.image_file_id)
        )
        cursor = end
    return tuple(mapped)


def bound_markdown_images(
    document: Document, text: str
) -> tuple[MarkdownImageSpan, ...]:
    raw = (document.doc_metadata or {}).get(MARKDOWN_IMAGES_KEY)
    if raw is None:
        return ()
    evidence = MarkdownImageEvidence.model_validate(raw)
    context = (document.doc_metadata or {}).get("regulatory_document_intake")
    if (
        not isinstance(context, dict)
        or str(evidence.user_file_id) != document.id
        or context.get("asset_id") != str(evidence.source_asset_id)
        or context.get("asset_sha256") != evidence.source_sha256
        or evidence.plaintext_sha256 != _sha(text.encode())
    ):
        raise ValueError("The Markdown source image context changed")
    cursor = 0
    for span in evidence.spans:
        if not cursor <= span.start < span.end <= len(text):
            raise ValueError("The Markdown source image ranges changed")
        cursor = span.end
    return evidence.spans
