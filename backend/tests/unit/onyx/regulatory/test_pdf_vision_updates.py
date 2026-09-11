"""Mixed and image-only PDF source evidence survives ordinary targeted drafting."""

import base64
import hashlib
import time
from io import BytesIO
from typing import cast
from unittest.mock import MagicMock

import pytest
from PIL import Image, ImageDraw
from pypdf import PdfReader, PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from onyx.regulatory.amendments.annexes.models import AcquiredAsset, AnnexVisionResult


def pdf_fixture(*, native_text: bool) -> bytes:
    image = Image.new("RGB", (480, 320), "white")
    draw = ImageDraw.Draw(image)
    draw.text((20, 20), "MADDE 3 - Oran tablosu degistirilmistir.", fill="black")
    draw.text((20, 100), "Urun       Oran", fill="black")
    draw.text((20, 140), "Bugday     17%", fill="black")
    draw.text((20, 220), "Not: yalniz test", fill="black")
    stream = BytesIO()
    image.save(stream, "PDF")
    writer = PdfWriter()
    writer.add_page(PdfReader(stream).pages[0])
    if native_text:
        page = writer.pages[0]
        fonts = cast(DictionaryObject, page["/Resources"]).setdefault(
            NameObject("/Font"), DictionaryObject()
        )
        fonts[NameObject("/FNative")] = DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
        text = DecodedStreamObject()
        text.set_data(
            b"BT /FNative 10 Tf 20 280 Td (MADDE 3 - Oran tablosu degistirilmistir.) Tj ET"
        )
        page[NameObject("/Contents")] = writer._add_object(text)
        # Keep the image's original content stream as well.
        original = PdfReader(stream).pages[0]
        page.merge_page(original)
    result = BytesIO()
    writer.write(result)
    return result.getvalue()


def vision_result() -> AnnexVisionResult:
    def cell(
        text: str, box: tuple[float, float, float, float], role: str
    ) -> dict[str, object]:
        return {
            "kind": "table_cell",
            "text": text,
            "box": box,
            "table_role": role,
            "status": "readable",
            "issues": [],
        }

    return AnnexVisionResult.model_validate(
        {
            "elements": [
                {
                    "kind": "text",
                    "text": "MADDE 3 - Oran tablosu degistirilmistir.",
                    "box": [0.04, 0.04, 0.95, 0.2],
                    "status": "readable",
                    "issues": [],
                },
                cell("Urun", (0.04, 0.3, 0.4, 0.4), "column_header"),
                cell("Oran", (0.5, 0.3, 0.9, 0.4), "column_header"),
                cell("Bugday", (0.04, 0.43, 0.4, 0.53), "data"),
                cell("17%", (0.5, 0.43, 0.9, 0.53), "data"),
                {
                    "kind": "footnote",
                    "text": "Not: yalniz test",
                    "box": [0.04, 0.7, 0.9, 0.8],
                    "status": "readable",
                    "issues": [],
                },
            ]
        }
    )


@pytest.mark.parametrize("native_text", [True, False])
def test_pdf_source_uses_real_rendered_pixels_and_preserves_table(
    monkeypatch: pytest.MonkeyPatch, native_text: bool
) -> None:
    from onyx.regulatory.amendments import pdf_vision
    from onyx.regulatory.amendments.annexes import extraction

    content = pdf_fixture(native_text=native_text)
    assert "17%" not in (PdfReader(BytesIO(content)).pages[0].extract_text() or "")
    model = MagicMock()
    model.config.model_provider = "fixture"
    model.config.model_name = "vision"
    generate = MagicMock(return_value=vision_result())
    monkeypatch.setattr(extraction, "generate_structured", generate)
    blobs: dict[str, bytes] = {}
    store = MagicMock()

    def save(stream: BytesIO, **_kwargs: object) -> str:
        key = str(len(blobs))
        blobs[key] = stream.read()
        return key

    store.save_file.side_effect = save
    asset = AcquiredAsset(
        sha256=hashlib.sha256(content).hexdigest(),
        content=content,
        mime_type="application/pdf",
        display_name="update.pdf",
        text="native paragraph" if native_text else "",
    )
    result = pdf_vision.prepare_pdf_source(
        asset, store=store, llm=model, deadline=time.monotonic() + 180
    )
    assert result.content == content and result.sha256 == asset.sha256
    assert result.native_text == asset.text
    assert result.text.count("MADDE 3") == 1
    assert "Bugday | 17%" in result.text and "Not: yalniz test" in result.text
    assert result.pdf_vision is not None
    assert (
        hashlib.sha256(blobs[result.pdf_vision.file_id]).hexdigest()
        == result.pdf_vision.sha256
    )
    call = generate.call_args.kwargs
    image_url = call["image_parts"][0].image_url.url
    assert base64.b64decode(image_url.split(",", 1)[1]).startswith(b"\x89PNG")
    assert call["max_attempts"] == 1 and call["provider_max_attempts"] == 1
    assert call["timeout_override"] <= 45


def test_source_pdf_refuses_missing_model_or_expired_budget() -> None:
    from onyx.regulatory.amendments import pdf_vision

    content = pdf_fixture(native_text=False)
    asset = AcquiredAsset(
        sha256=hashlib.sha256(content).hexdigest(),
        content=content,
        mime_type="application/pdf",
        display_name="update.pdf",
    )
    with pytest.raises(ValueError, match="vision_model_required"):
        pdf_vision.prepare_pdf_source(
            asset, store=MagicMock(), llm=None, deadline=time.monotonic() + 180
        )
    with pytest.raises(TimeoutError):
        pdf_vision.prepare_pdf_source(
            asset, store=MagicMock(), llm=MagicMock(), deadline=time.monotonic() - 1
        )


def frozen_case(monkeypatch: pytest.MonkeyPatch):
    from uuid import uuid4

    from onyx.regulatory.amendments import pdf_vision
    from onyx.regulatory.amendments.annexes.models import (
        AcquisitionResult,
        AnnexExtraction,
        AnnexLocator,
        AnnexModelSnapshot,
        AnnexRenderedPage,
        ExtractedAnnexElement,
    )

    original = pdf_fixture(native_text=True)
    sha = hashlib.sha256(original).hexdigest()
    elements = [
        ExtractedAnnexElement(
            kind=item.kind,
            text=item.text,
            table_role=item.table_role,
            extraction_method="vision",
            locator=AnnexLocator(page=1, normalized_box=item.box),
        )
        for item in vision_result().elements
    ]
    extraction = AnnexExtraction(
        source_sha256=sha,
        mime_type="application/pdf",
        page_count=1,
        elements=elements,
        model_snapshot=AnnexModelSnapshot(
            model_provider="fixture", model_name="vision"
        ),
    )
    monkeypatch.setattr(
        pdf_vision, "extract_annex_structure", MagicMock(return_value=extraction)
    )
    blobs: dict[str, bytes] = {"original": original}
    store = MagicMock()

    def save(stream: BytesIO, **_kwargs: object) -> str:
        key = f"blob-{len(blobs)}"
        blobs[key] = stream.read()
        return key

    store.save_file.side_effect = save
    store.read_file.side_effect = lambda key: BytesIO(blobs[key])
    asset = pdf_vision.prepare_pdf_source(
        AcquiredAsset(
            content=original,
            sha256=sha,
            mime_type="application/pdf",
            display_name="update.pdf",
            text="native",
        ),
        store=store,
        llm=MagicMock(),
        deadline=time.monotonic() + 180,
    )
    manifest = (
        AcquisitionResult(status="ready", assets=[asset]).model_dump_json().encode()
    )
    blobs["manifest"] = manifest
    file_id = uuid4()
    source = pdf_vision.PdfBatchSource(
        batch_id=44,
        package_id=uuid4(),
        source_text_sha256=pdf_vision.digest(asset.text),
        manifest_file_id="manifest",
        manifest_sha256=hashlib.sha256(manifest).hexdigest(),
        user_file_ids=[file_id],
        originals=[
            pdf_vision.PdfOriginal(asset_id=uuid4(), file_id="original", sha256=sha)
        ],
    )
    png = BytesIO()
    Image.new("RGB", (10, 10), "white").save(png, "PNG")
    # The real renderer is separately covered above; these tests isolate binding.
    render = MagicMock(
        return_value=[
            AnnexRenderedPage(page=1, width=480, height=320, png=png.getvalue())
        ]
    )
    monkeypatch.setattr(pdf_vision, "run_in_isolated_process", render)
    return source, asset, store, blobs, render


@pytest.mark.parametrize(
    "mutation", [None, "original", "manifest", "extraction", "anchor"]
)
def test_pdf_draft_evidence_refuses_changed_bytes_and_missing_anchor(
    monkeypatch: pytest.MonkeyPatch, mutation: str | None
) -> None:
    from onyx.regulatory.amendments import pdf_vision
    from onyx.regulatory.amendments.models import AmendmentInstruction

    source, asset, store, blobs, render = frozen_case(monkeypatch)
    instruction = AmendmentInstruction(
        instruction_text="MADDE 3 - Oran tablosu degistirilmistir.",
        article_reference="MADDE 3",
    )
    if mutation in {"original", "manifest"}:
        blobs[mutation] += b"changed"
    if mutation == "extraction":
        assert asset.pdf_vision is not None
        blobs[asset.pdf_vision.file_id] += b"changed"
    if mutation == "anchor":
        instruction = instruction.model_copy(
            update={
                "instruction_text": "other instruction",
                "article_reference": "MADDE 30",
            }
        )
    if mutation:
        with pytest.raises(ValueError):
            pdf_vision.prepare_pdf_draft_evidence(source, [instruction], store)
    else:
        evidence = pdf_vision.prepare_pdf_draft_evidence(source, [instruction], store)
        assert evidence is not None
        assert evidence.pages[0].source_sha256 == asset.sha256
        assert "17%" in evidence.transcription
        assert render.call_args.args[1] == blobs["original"]
        assert evidence.image_parts[0].image_url.url.startswith(
            "data:image/png;base64,"
        )


@pytest.mark.parametrize(
    "supported,ambiguous", [(True, False), (False, False), (True, True)]
)
def test_non_annex_pipeline_receives_images_and_requires_grounding(
    monkeypatch: pytest.MonkeyPatch, supported: bool, ambiguous: bool
) -> None:
    from onyx.file_store import file_store
    from onyx.llm import factory
    from onyx.regulatory.amendments import drafter, pdf_vision, pipeline
    from onyx.regulatory.amendments.models import (
        AmendmentInstruction,
        ChunkFieldsDraft,
        DateResolution,
        DraftResult,
        MatchResult,
    )

    source, asset, store, blobs, render = frozen_case(monkeypatch)
    instruction = AmendmentInstruction(
        instruction_text="MADDE 3 - Oran tablosu degistirilmistir.",
        article_reference="MADDE 3",
    )
    match = MatchResult(old_chunk_id="old", confidence=0.99, rationale="exact article")
    context = pipeline.InstructionDraftContext(
        match=match,
        old_chunk_snapshot={
            "id": "old",
            "user_file_id": str(source.user_file_ids[0]),
            "text": "MADDE 3\nBugday | 5%",
            "chunk_type": "article",
            "heading_path": ["MADDE 3"],
            "metadata": {"article_no": "3"},
        },
        target_user_file_id=source.user_file_ids[0],
        target_position=1,
        sibling_reference=None,
        base_metadata={"article_no": "3"},
        base_heading_path=["MADDE 3"],
    )
    vision = MagicMock()
    monkeypatch.setattr(factory, "get_default_llm_with_vision", lambda: vision)
    monkeypatch.setattr(file_store, "get_default_file_store", lambda: store)
    draft = DraftResult(
        new_chunk=ChunkFieldsDraft(text="MADDE 3\nBugday | 17%", chunk_type="article"),
        dates=DateResolution(
            effective_start_date="2026-09-11",
            effective_end_date=None,
            rationale="explicit",
        ),
    )
    drafting = MagicMock(return_value=draft)
    grounding = MagicMock(
        return_value=pdf_vision.PdfGroundingVerdict(
            supported=supported, ambiguous=ambiguous, rationale="fixture"
        )
    )
    monkeypatch.setattr(drafter, "generate_structured", drafting)
    monkeypatch.setattr(pdf_vision, "generate_structured", grounding)
    from onyx.regulatory.amendments.models import ProposalDraft

    def create_proposal() -> ProposalDraft:
        return pipeline.draft_instruction_group_proposal(
            MagicMock(),
            instruction_indices=[0],
            instructions=[instruction],
            matches=[match],
            reference_date="2026-09-11",
            context=context,
            pdf_source=source,
        )

    if not supported or ambiguous:
        with pytest.raises(ValueError, match="unambiguously"):
            create_proposal()
    else:
        proposal = create_proposal()
        receipt = pdf_vision.PdfProposalEvidence.model_validate(
            proposal.old_chunk_snapshot[pdf_vision.PDF_EVIDENCE_KEY]
        )
        assert (
            receipt.old_chunk_id == "old"
            and receipt.draft_text_sha256 == pdf_vision.digest(draft.new_chunk.text)
        )
        assert pdf_vision.PDF_EVIDENCE_KEY not in proposal.new_chunk_draft["metadata"]
    assert (
        drafting.call_args.args[0] is vision and grounding.call_args.args[0] is vision
    )
    assert (
        drafting.call_args.kwargs["image_parts"]
        == grounding.call_args.kwargs["image_parts"]
    )
    assert (
        "5%" in drafting.call_args.kwargs["user_prompt"]
        and "17%" in drafting.call_args.kwargs["user_prompt"]
    )
    assert "MADDE 3" in grounding.call_args.kwargs["user_prompt"]


def test_source_worker_freezes_vision_text_for_existing_readback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace
    from uuid import uuid4

    from onyx.llm import factory
    from onyx.regulatory.amendments.annexes import evidence, job
    from onyx.regulatory.amendments.annexes.models import AcquisitionResult

    source, asset, store, blobs, _ = frozen_case(monkeypatch)
    package = SimpleNamespace(
        input_spec={"mime_type": "application/pdf", "display_name": "update.pdf"},
        input_file_id="original",
        manifest_file_id=None,
    )
    monkeypatch.setattr(
        job, "claim_source_package", MagicMock(return_value=(package, uuid4()))
    )
    monkeypatch.setattr(job, "list_source_assets", MagicMock(return_value=[]))
    monkeypatch.setattr(job, "get_session_with_current_tenant", MagicMock())
    monkeypatch.setattr(job, "get_default_file_store", lambda: store)
    monkeypatch.setattr(factory, "get_default_llm_with_vision", lambda: MagicMock())
    native = asset.model_copy(
        update={
            "text": "native-only paragraph",
            "native_text": None,
            "pdf_vision": None,
        }
    )
    monkeypatch.setattr(
        job,
        "acquire_source_package",
        MagicMock(return_value=AcquisitionResult(status="ready", assets=[native])),
    )
    finish = MagicMock()
    monkeypatch.setattr(job, "finish_source_package", finish)
    failed = MagicMock()
    monkeypatch.setattr(job, "mark_source_package_failed", failed)
    job.run_source_package(package_id=source.package_id, environment="local-test")
    failed.assert_not_called()
    persisted = finish.call_args.kwargs["assets"]
    text, _ = evidence.read_original_source_text(store, persisted)
    assert "Bugday | 17%" in text and "native-only paragraph" not in text
    frozen = finish.call_args.kwargs["result"].assets[0]
    assert (
        frozen.native_text == "native-only paragraph"
        and frozen.content == blobs["original"]
    )
    assert frozen.pdf_vision is not None
    manifest = blobs[finish.call_args.kwargs["manifest_file_id"]]
    assert (
        hashlib.sha256(manifest).hexdigest()
        == finish.call_args.kwargs["manifest_sha256"]
    )
    assert b"pdf_vision" in manifest


def test_source_derivative_reuse_does_not_invoke_vision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from onyx.regulatory.amendments import pdf_vision

    _, asset, store, _, _ = frozen_case(monkeypatch)
    extract = MagicMock(side_effect=AssertionError("must reuse frozen derivative"))
    monkeypatch.setattr(pdf_vision, "extract_annex_structure", extract)
    reused = pdf_vision.reuse_pdf_source(asset, asset.model_dump(mode="json"), store)
    assert reused.pdf_vision == asset.pdf_vision and reused.text == asset.text
    extract.assert_not_called()


def test_source_refuses_uncertain_or_overlapping_table_cells(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from onyx.regulatory.amendments import pdf_vision
    from onyx.regulatory.amendments.annexes.models import AnnexExtraction

    _, asset, _, blobs, _ = frozen_case(monkeypatch)
    assert asset.pdf_vision is not None
    original = AnnexExtraction.model_validate_json(blobs[asset.pdf_vision.file_id])
    uncertain = original.model_copy(deep=True)
    uncertain.elements[4].status = "uncertain"
    with pytest.raises(ValueError, match="uncertain"):
        pdf_vision.pdf_transcript(uncertain)
    original.elements[4].locator.normalized_box = original.elements[
        3
    ].locator.normalized_box
    with pytest.raises(ValueError, match="overlap"):
        pdf_vision.pdf_transcript(original)


def test_multiple_source_pages_with_same_instruction_are_unresolved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import json

    from onyx.regulatory.amendments import pdf_vision
    from onyx.regulatory.amendments.annexes.models import AnnexExtraction
    from onyx.regulatory.amendments.models import AmendmentInstruction

    source, asset, store, blobs, _ = frozen_case(monkeypatch)
    assert asset.pdf_vision is not None
    extraction = AnnexExtraction.model_validate_json(blobs[asset.pdf_vision.file_id])
    second = [element.model_copy(deep=True) for element in extraction.elements]
    for element in second:
        element.locator.page = 2
    extraction.elements.extend(second)
    extraction.page_count = 2
    data = extraction.model_dump_json().encode()
    blobs[asset.pdf_vision.file_id] = data
    manifest = json.loads(blobs["manifest"])
    entry = manifest["assets"][0]
    entry["text"] = pdf_vision.pdf_transcript(extraction)
    entry["pdf_vision"]["sha256"] = hashlib.sha256(data).hexdigest()
    entry["pdf_vision"]["transcript_sha256"] = pdf_vision.digest(entry["text"])
    blobs["manifest"] = json.dumps(manifest).encode()
    source = source.model_copy(
        update={"manifest_sha256": hashlib.sha256(blobs["manifest"]).hexdigest()}
    )
    with pytest.raises(ValueError, match="ambiguous"):
        pdf_vision.prepare_pdf_draft_evidence(
            source,
            [
                AmendmentInstruction(
                    instruction_text="MADDE 3 - Oran tablosu degistirilmistir."
                )
            ],
            store,
        )


@pytest.mark.parametrize("path", ["asset", "analysis"])
@pytest.mark.parametrize("mode", ["frozen", "tampered", "legacy"])
def test_annex_new_pdf_reuses_verified_derivative(
    monkeypatch: pytest.MonkeyPatch, path: str, mode: str
) -> None:
    import json
    from types import SimpleNamespace

    from onyx.db import amendment_sources
    from onyx.regulatory.amendments.annexes import analysis, extraction
    from onyx.regulatory.amendments.annexes.models import (
        AnnexExtraction,
        AnnexOriginalEvidence,
    )

    source, asset, store, blobs, _ = frozen_case(monkeypatch)
    assert asset.pdf_vision is not None
    expected = AnnexExtraction.model_validate_json(blobs[asset.pdf_vision.file_id])
    if mode == "tampered":
        blobs[asset.pdf_vision.file_id] += b" "
    if mode == "legacy":
        manifest = json.loads(blobs["manifest"])
        manifest["assets"][0].pop("pdf_vision")
        blobs["manifest"] = json.dumps(manifest).encode()
        source = source.model_copy(
            update={"manifest_sha256": hashlib.sha256(blobs["manifest"]).hexdigest()}
        )
    invoke = MagicMock(return_value=expected)
    monkeypatch.setattr(extraction, "extract_annex_structure", invoke)
    monkeypatch.setattr(analysis, "extract_annex_structure", invoke)
    original = source.originals[0]
    monkeypatch.setattr(
        amendment_sources,
        "require_ready_source_package",
        lambda *_args, **_kwargs: SimpleNamespace(
            manifest_file_id=source.manifest_file_id,
            manifest_sha256=source.manifest_sha256,
        ),
    )
    monkeypatch.setattr(
        amendment_sources,
        "get_source_asset",
        lambda *_args, **_kwargs: SimpleNamespace(
            id=original.asset_id,
            file_id=original.file_id,
            byte_count=len(blobs["original"]),
            sha256=original.sha256,
            mime_type="application/pdf",
        ),
    )

    def consume() -> AnnexExtraction:
        if path == "asset":
            return extraction.extract_source_asset(
                MagicMock(),
                store,
                package_id=source.package_id,
                asset_id=original.asset_id,
                document_set_id=18,
                environment="DEV",
                vision_llm=MagicMock(),
            )
        return analysis._extract_prepared_source(
            store,
            AnnexOriginalEvidence(
                file_id=original.file_id,
                sha256=original.sha256,
                mime_type="application/pdf",
                available=True,
            ),
            {},
            MagicMock(),
            manifest_file_id=source.manifest_file_id,
            manifest_sha256=source.manifest_sha256,
        )

    if mode == "tampered":
        with pytest.raises(ValueError, match="integrity"):
            consume()
    else:
        result = consume()
        assert result.elements[4].text == "17%"
        if path == "asset":
            assert all(
                item.source_asset_id == str(original.asset_id)
                for item in result.elements
            )
    assert invoke.call_count == (1 if mode == "legacy" else 0)


@pytest.mark.parametrize("repeated", [False, True])
def test_unrelated_table_does_not_require_images_for_plain_text_update(
    monkeypatch: pytest.MonkeyPatch, repeated: bool
) -> None:
    import json

    from onyx.regulatory.amendments import pdf_vision
    from onyx.regulatory.amendments.annexes.models import AnnexExtraction
    from onyx.regulatory.amendments.models import AmendmentInstruction

    source, asset, store, blobs, render = frozen_case(monkeypatch)
    assert asset.pdf_vision is not None
    extraction = AnnexExtraction.model_validate_json(blobs[asset.pdf_vision.file_id])
    instruction = "MADDE 9 - Basvuru suresi on gundur."
    for page in range(2, 4 if repeated else 3):
        text = extraction.elements[0].model_copy(deep=True)
        text.text = instruction
        text.locator.page = page
        extraction.elements.append(text)
        extraction.page_count = page
    data = extraction.model_dump_json().encode()
    blobs[asset.pdf_vision.file_id] = data
    manifest = json.loads(blobs["manifest"])
    entry = manifest["assets"][0]
    entry["text"] = pdf_vision.pdf_transcript(extraction)
    entry["pdf_vision"]["sha256"] = hashlib.sha256(data).hexdigest()
    entry["pdf_vision"]["transcript_sha256"] = pdf_vision.digest(entry["text"])
    blobs["manifest"] = json.dumps(manifest).encode()
    source = source.model_copy(
        update={"manifest_sha256": hashlib.sha256(blobs["manifest"]).hexdigest()}
    )
    assert (
        pdf_vision.prepare_pdf_draft_evidence(
            source,
            [AmendmentInstruction(instruction_text=instruction)],
            store,
        )
        is None
    )
    render.assert_not_called()
