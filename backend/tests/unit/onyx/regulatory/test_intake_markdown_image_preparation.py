"""Embedded source images must become grounded text with immutable owner links."""

import hashlib
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any, TypedDict, cast
from uuid import UUID, uuid4

import pytest

from onyx.configs.constants import FileOrigin
from onyx.db.file_record import FileRecordNotFoundError
from onyx.db.models import FileRecord
from onyx.file_store.file_store import FileStore
from onyx.llm.interfaces import LLM
from onyx.regulatory.amendments import analysis_llm
from onyx.regulatory.amendments.annexes import source_images
from onyx.regulatory.amendments.annexes.models import AcquiredAsset, SourceLink
from onyx.regulatory.document_intake.markdown_image_preparation import (
    bound_markdown_images,
)
from onyx.regulatory.document_intake.models import IntakeSourceAsset
from onyx.regulatory.document_intake.preparation import load_verified_intake_documents

FIXTURE = Path(__file__).parent / "fixtures/docling_737039bf/20230928-1-1.md"
TRANSCRIPT = "| Tarih | İşlem |\n| --- | --- |\n| 15 Nisan | Tren çizgisi için son başvuru tarihi |\n| 1 Kasım | Orerin yayımlanması |\n| 15 Aralık | Yeni şebeke bildiriminin yayımlanması |"


class PreparationArguments(TypedDict):
    package_id: UUID
    asset: IntakeSourceAsset
    user_file_id: UUID
    links: list[SourceLink]
    store: FileStore
    allow_image_transcription: bool


class MemorySources:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.records: dict[str, FileRecord] = {}

    def read_file(self, key: str, **_kwargs: Any) -> BytesIO:
        return BytesIO(self.objects[key])

    def save_file(
        self,
        content: Any,
        display_name: str | None,
        file_origin: FileOrigin,
        file_type: str,
        file_metadata: dict[str, Any] | None = None,
        file_id: str | None = None,
    ) -> str:
        identifier = file_id or str(uuid4())
        self.objects[identifier] = content.read()
        self.records[identifier] = FileRecord(
            file_id=identifier,
            display_name=display_name,
            file_origin=file_origin,
            file_type=file_type,
            file_metadata=file_metadata,
            bucket_name="local",
            object_key=identifier,
        )
        return identifier

    def has_file(self, key: str, file_origin: FileOrigin, file_type: str) -> bool:
        r = self.records.get(key)
        return (
            r is not None
            and r.file_origin == file_origin
            and (r.file_type == file_type)
        )

    def read_file_record(self, key: str) -> FileRecord:
        if key not in self.records:
            raise FileRecordNotFoundError(key)
        return self.records[key]


def setup_image(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[MemorySources, IntakeSourceAsset, list[str]]:
    raw = FIXTURE.read_bytes()
    store = MemorySources()
    source = store.save_file(
        BytesIO(raw), FIXTURE.name, FileOrigin.OTHER, "text/markdown"
    )
    text = store.save_file(BytesIO(raw), "prepared", FileOrigin.OTHER, "text/plain")
    asset = IntakeSourceAsset(
        id=uuid4(),
        package_id=uuid4(),
        sha256=hashlib.sha256(raw).hexdigest(),
        file_id=source,
        text_file_id=text,
        text_sha256=hashlib.sha256(raw).hexdigest(),
        mime_type="text/markdown",
        display_name=FIXTURE.name,
        byte_count=len(raw),
    )
    calls = []

    def transcribe(
        image: AcquiredAsset, *, llm: LLM | None, deadline: float
    ) -> AcquiredAsset:
        assert llm is not None and deadline > 0
        assert image.mime_type == "image/png" and image.content.startswith(b"\x89PNG")
        calls.append(image.sha256)
        return image.model_copy(update={"text": TRANSCRIPT})

    monkeypatch.setattr(
        analysis_llm,
        "get_amendment_analysis_llm",
        lambda **_kwargs: cast(
            LLM,
            SimpleNamespace(
                config=SimpleNamespace(
                    model_provider="local-test", model_name="image-transcriber"
                )
            ),
        ),
    )
    monkeypatch.setattr(source_images, "prepare_image_source", transcribe)
    return (store, asset, calls)


def test_real_embedded_diagram_enters_text_and_keeps_original_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (store, asset, calls) = setup_image(monkeypatch)
    owner = uuid4()
    documents = load_verified_intake_documents(
        package_id=asset.package_id,
        asset=asset,
        user_file_id=owner,
        links=[],
        store=cast(FileStore, store),
        allow_image_transcription=True,
    )
    document = documents[0]
    text = document.get_text_content()
    assert "data:image/png;base64" not in text
    assert TRANSCRIPT in text
    images = bound_markdown_images(document, text)
    assert len(images) == 1 and "15 Nisan" in text[images[0].start : images[0].end]
    record = store.read_file_record(images[0].image_file_id)
    metadata = cast(dict[str, object], record.file_metadata)
    assert metadata["purpose"] == "regulatory_source_image"
    assert metadata["regulatory_user_file_id"] == str(owner)
    assert (
        hashlib.sha256(store.objects[record.file_id]).hexdigest() == metadata["sha256"]
    )
    assert store.objects[asset.file_id] == FIXTURE.read_bytes()
    assert len(calls) == 1
    assert documents.original_extraction.raw_sha256 == asset.sha256


def test_replay_uses_frozen_transcription_even_when_model_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (store, asset, calls) = setup_image(monkeypatch)
    owner = uuid4()
    kwargs: PreparationArguments = dict(
        package_id=asset.package_id,
        asset=asset,
        user_file_id=owner,
        links=[],
        store=cast(FileStore, store),
        allow_image_transcription=True,
    )
    first = load_verified_intake_documents(**kwargs)
    monkeypatch.setattr(
        analysis_llm,
        "get_amendment_analysis_llm",
        lambda **_kwargs: pytest.fail("Replay requested a model"),
    )
    kwargs["allow_image_transcription"] = False
    second = load_verified_intake_documents(**kwargs)
    assert first.original_extraction == second.original_extraction
    assert len(calls) == 1
    assert first[0].sections == second[0].sections


def test_missing_frozen_receipt_cannot_trigger_conversion_during_source_verification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, asset, calls = setup_image(monkeypatch)
    with pytest.raises(ValueError, match="transcription receipt is missing"):
        load_verified_intake_documents(
            package_id=asset.package_id,
            asset=asset,
            user_file_id=uuid4(),
            links=[],
            store=cast(FileStore, store),
        )
    assert not calls


@pytest.mark.parametrize(
    "corrupt", ["image", "receipt", "image_metadata", "receipt_type", "receipt_origin"]
)
def test_tampered_cached_source_never_reaches_canonical_input(
    monkeypatch: pytest.MonkeyPatch, corrupt: str
) -> None:
    (store, asset, _calls) = setup_image(monkeypatch)
    owner = uuid4()
    kwargs: PreparationArguments = dict(
        package_id=asset.package_id,
        asset=asset,
        user_file_id=owner,
        links=[],
        store=cast(FileStore, store),
        allow_image_transcription=True,
    )
    first = load_verified_intake_documents(**kwargs)
    if corrupt == "image":
        key = bound_markdown_images(first[0], first[0].get_text_content())[
            0
        ].image_file_id
        store.objects[key] = b"corrupt-image"
    elif corrupt == "receipt":
        key = next(
            (k for (k, r) in store.records.items() if r.file_type == "application/json")
        )
        store.objects[key] = store.objects[key].replace(b"15 Nisan", b"16 Nisan")
    elif corrupt == "image_metadata":
        key = bound_markdown_images(first[0], first[0].get_text_content())[
            0
        ].image_file_id
        metadata = dict(cast(dict[str, object], store.records[key].file_metadata))
        metadata["source_asset_id"] = str(uuid4())
        store.records[key].file_metadata = metadata
    else:
        key = next(
            k
            for k, record in store.records.items()
            if record.file_type == "application/json"
        )
        if corrupt == "receipt_type":
            store.records[key].file_type = "text/plain"
        else:
            store.records[key].file_origin = FileOrigin.CHAT_UPLOAD
    with pytest.raises(ValueError, match="integrity|changed|receipt|transcript"):
        load_verified_intake_documents(**kwargs)


@pytest.mark.parametrize("reference", [False, True])
def test_image_title_and_reference_resource_are_consumed_without_losing_text(
    monkeypatch: pytest.MonkeyPatch, reference: bool
) -> None:
    from onyx.regulatory.document_intake.markdown_images import embedded_markdown_images

    store, asset, calls = setup_image(monkeypatch)
    image = embedded_markdown_images(FIXTURE.read_text(), path=FIXTURE.name)[0]
    import base64

    uri = "data:image/png;base64," + base64.b64encode(image.content).decode()
    if reference:
        text = f'# EK-1\n\nÖnce\n\n![Takvim][şema]\n\nSonra\n\n[şema]: {uri} "Başvuru süreleri"\n'
    else:
        text = f'# EK-1\n\nÖnce\n\n![Takvim]({uri} "Başvuru süreleri")\n\nSonra\n'
    raw = text.encode()
    store.objects[asset.file_id] = raw
    store.objects[asset.text_file_id or ""] = raw
    asset = asset.model_copy(
        update={
            "sha256": hashlib.sha256(raw).hexdigest(),
            "text_sha256": hashlib.sha256(raw).hexdigest(),
            "byte_count": len(raw),
        }
    )
    document = load_verified_intake_documents(
        package_id=asset.package_id,
        asset=asset,
        user_file_id=uuid4(),
        links=[],
        store=cast(FileStore, store),
        allow_image_transcription=True,
    )[0]
    prepared = document.get_text_content()
    assert "data:image/" not in prepared and "[şema]:" not in prepared
    assert "Önce" in prepared and "Sonra" in prepared
    assert "Takvim" in prepared and "Başvuru süreleri" in prepared
    assert TRANSCRIPT in prepared and len(calls) == 1
    assert bound_markdown_images(document, prepared)


def test_embedded_diagram_inside_a_table_does_not_break_its_parent_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import re

    (store, asset, _calls) = setup_image(monkeypatch)
    match = re.search(
        "!\\[Image\\]\\(data:image/png;base64,[A-Za-z0-9+/=]+\\)", FIXTURE.read_text()
    )
    assert match is not None
    image = match.group(0)
    raw = (
        "# EK-1\n\n| Başlık | İçerik |\n| --- | --- |\n| Takvim | " + image + " |\n"
    ).encode()
    store.objects[asset.file_id] = raw
    store.objects[asset.text_file_id or ""] = raw
    asset = asset.model_copy(
        update={
            "sha256": hashlib.sha256(raw).hexdigest(),
            "text_sha256": hashlib.sha256(raw).hexdigest(),
            "byte_count": len(raw),
        }
    )
    doc = load_verified_intake_documents(
        package_id=asset.package_id,
        asset=asset,
        user_file_id=uuid4(),
        links=[],
        store=cast(FileStore, store),
        allow_image_transcription=True,
    )[0]
    table_row = next(
        (
            line
            for line in doc.get_text_content().splitlines()
            if line.startswith("| Takvim |")
        )
    )
    assert "15 Nisan" in table_row and "<br/>" in table_row
    assert "data:image/" not in table_row


def test_binding_rejects_changed_plaintext_or_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (store, asset, _calls) = setup_image(monkeypatch)
    doc = load_verified_intake_documents(
        package_id=asset.package_id,
        asset=asset,
        user_file_id=uuid4(),
        links=[],
        store=cast(FileStore, store),
        allow_image_transcription=True,
    )[0]
    with pytest.raises(ValueError, match="context changed"):
        bound_markdown_images(doc, doc.get_text_content() + " altered")
    doc.id = str(uuid4())
    with pytest.raises(ValueError, match="context changed"):
        bound_markdown_images(doc, doc.get_text_content())
