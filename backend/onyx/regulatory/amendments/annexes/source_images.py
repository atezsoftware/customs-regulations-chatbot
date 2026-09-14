"""Source image transcription without scene descriptions or inferred amendments."""

import hashlib
import time

from onyx.file_store.file_store import FileStore
from onyx.llm.interfaces import LLM
from onyx.regulatory.amendments.annexes.extraction import extract_annex_structure
from onyx.regulatory.amendments.annexes.models import AcquiredAsset
from onyx.regulatory.amendments.annexes.sources import MAX_TEXT_CHARS
from onyx.regulatory.amendments.pdf_vision import pdf_transcript


def prepare_image_source(
    asset: AcquiredAsset, *, llm: LLM | None, deadline: float
) -> AcquiredAsset:
    if llm is None:
        raise ValueError("image_vision_model_required")
    extraction = extract_annex_structure(
        asset.content,
        asset.mime_type,
        vision_llm=llm,
        vision_deadline=deadline,
        source_text_only=True,
    )
    if any(
        item.extraction_method == "vision" and item.kind == "image_region"
        for item in extraction.elements
    ):
        raise ValueError("image_source_has_no_readable_document_text")
    text = pdf_transcript(extraction)
    if not text.strip():
        raise ValueError("image_source_has_no_readable_document_text")
    if time.monotonic() >= deadline:
        raise TimeoutError("image_source_preparation_deadline")
    return asset.model_copy(update={"text": text})


def reuse_image_source(
    asset: AcquiredAsset, *, store: FileStore, text_file_id: str, text_sha256: str
) -> AcquiredAsset:
    with store.read_file(text_file_id) as stream:
        text_bytes = stream.read(4 * MAX_TEXT_CHARS + 1)
    if (
        len(text_bytes) > 4 * MAX_TEXT_CHARS
        or hashlib.sha256(text_bytes).hexdigest() != text_sha256
    ):
        raise ValueError("image_source_text_integrity_mismatch")
    text = text_bytes.decode("utf-8")
    if len(text) > MAX_TEXT_CHARS:
        raise ValueError("image_source_text_limit")
    if not text.strip():
        raise ValueError("image_source_has_no_readable_document_text")
    return asset.model_copy(update={"text": text})
