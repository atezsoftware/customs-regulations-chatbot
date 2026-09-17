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

from onyx.regulatory.amendments.annexes.models import (
    AcquiredAsset,
    AnnexVisionWireResult,
)


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


def vision_result() -> AnnexVisionWireResult:
    def cell(
        text: str, box: tuple[float, float, float, float], role: str
    ) -> dict[str, object]:
        return {
            "kind": "table_cell",
            "text": text,
            "box": {"left": box[0], "top": box[1], "right": box[2], "bottom": box[3]},
            "table_role": role,
            "status": "readable",
            "issues": [],
        }

    return AnnexVisionWireResult.model_validate(
        {
            "elements": [
                {
                    "kind": "text",
                    "text": "MADDE 3 - Oran tablosu degistirilmistir.",
                    "box": {"left": 0.04, "top": 0.04, "right": 0.95, "bottom": 0.2},
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
                    "box": {"left": 0.04, "top": 0.7, "right": 0.9, "bottom": 0.8},
                    "status": "readable",
                    "issues": [],
                },
            ]
        }
    )


@pytest.mark.parametrize("native_text", [True, False])
def test_pdf_source_sends_original_pdf_once_and_preserves_table(
    monkeypatch: pytest.MonkeyPatch, native_text: bool
) -> None:
    from onyx.regulatory.amendments import pdf_vision
    from onyx.regulatory.amendments.annexes import pdf_document

    content = pdf_fixture(native_text=native_text)
    assert "17%" not in (PdfReader(BytesIO(content)).pages[0].extract_text() or "")
    model = MagicMock()
    model.config.model_provider = "fixture"
    model.config.model_name = "vision"
    generate = MagicMock(
        return_value=pdf_document.PdfVisionDocument(
            pages=[
                pdf_document.PdfVisionPage(
                    page=1, complete=True, elements=vision_result().elements
                )
            ]
        )
    )
    monkeypatch.setattr(pdf_document, "generate_structured", generate)
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
    deadline = time.monotonic() + 180
    result = pdf_vision.prepare_pdf_source(
        asset, store=store, llm=model, deadline=deadline
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
    file_data = call["file_parts"][0].file.file_data
    assert base64.b64decode(file_data.split(",", 1)[1]) == content
    assert call["max_attempts"] == call["provider_max_attempts"] == 1
    assert call["deadline"] <= deadline
    assert call["timeout_override"] <= 68
    assert generate.call_count == 1


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
            locator=AnnexLocator(page=1, normalized_box=item.box.as_tuple()),
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
        # An instruction with no connection to any page of the attached PDF
        # (e.g. it amends the tebliğ's body text, not this table) is not
        # ambiguous or tampered evidence — it simply isn't about this PDF at
        # all, so evidence generation quietly yields nothing rather than
        # refusing the whole group.
        instruction = instruction.model_copy(
            update={
                "instruction_text": "other instruction",
                "article_reference": "MADDE 30",
            }
        )
        assert (
            pdf_vision.prepare_pdf_draft_evidence(source, [instruction], store) is None
        )
    elif mutation:
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
    from onyx.regulatory.amendments import analysis_llm, drafter, pdf_vision, pipeline
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
    monkeypatch.setattr(
        analysis_llm, "get_amendment_analysis_llm", lambda **_kwargs: vision
    )
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

    from onyx.regulatory.amendments import analysis_llm
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
    monkeypatch.setattr(
        analysis_llm, "get_amendment_analysis_llm", lambda **_kwargs: MagicMock()
    )
    native = asset.model_copy(
        update={
            "text": "native-only paragraph",
            "native_text": None,
            "pdf_vision": None,
            "page_count": 1,
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


@pytest.mark.parametrize("page_seconds", [30, 170])
def test_source_worker_prepares_twelve_visual_pages_with_a_total_deadline(
    monkeypatch: pytest.MonkeyPatch, page_seconds: int
) -> None:
    from types import SimpleNamespace
    from uuid import uuid4

    from onyx.llm.model_response import Choice, Message, ModelResponse
    from onyx.regulatory.amendments import analysis_llm
    from onyx.regulatory.amendments.annexes import job, pdf_document
    from onyx.regulatory.amendments.annexes.models import (
        AcquisitionResult,
        AnnexExtraction,
    )

    now = [100.0]
    monkeypatch.setattr(job.time, "monotonic", lambda: now[0])
    package = SimpleNamespace(
        input_spec={"url": "https://example.gov/update.htm"},
        input_file_id=None,
        manifest_file_id=None,
    )
    monkeypatch.setattr(
        job, "claim_source_package", MagicMock(return_value=(package, uuid4()))
    )
    monkeypatch.setattr(job, "list_source_assets", MagicMock(return_value=[]))
    monkeypatch.setattr(job, "get_session_with_current_tenant", MagicMock())
    content = b"twelve image-only PDF pages"
    original = AcquiredAsset(
        sha256=hashlib.sha256(content).hexdigest(),
        content=content,
        mime_type="application/pdf",
        display_name="linked.pdf",
        text="Resmi Gazete",
        page_count=12,
    )
    monkeypatch.setattr(
        job,
        "acquire_source_package",
        lambda **_kwargs: AcquisitionResult(status="ready", assets=[original]),
    )

    monkeypatch.setattr(
        pdf_document,
        "run_in_isolated_process",
        lambda *_args, **_kwargs: [(200, 300)] * 12,
    )
    model = MagicMock()
    model.config.model_provider = "fixture"
    model.config.model_name = "vision"

    # Pages are transcribed in contiguous groups, so the model is asked for one
    # bounded range at a time and answers with exactly that range.
    next_page = [1]

    def invoke(*_args: object, **_kwargs: object) -> ModelResponse:
        now[0] += page_seconds
        first = next_page[0]
        last = min(first + pdf_document._PAGE_GROUP_SIZE - 1, 12)
        next_page[0] = last + 1
        return ModelResponse(
            id="fixture",
            created="2026-09-14",
            choice=Choice(
                message=Message(
                    content=pdf_document.PdfVisionDocument(
                        pages=[
                            pdf_document.PdfVisionPage(
                                page=page,
                                complete=True,
                                elements=vision_result().elements,
                            )
                            for page in range(first, last + 1)
                        ]
                    ).model_dump_json()
                )
            ),
        )

    model.invoke.side_effect = invoke
    vision_factory = MagicMock(return_value=model)
    monkeypatch.setattr(
        analysis_llm,
        "get_amendment_analysis_llm",
        lambda **kwargs: vision_factory(**kwargs),
    )
    blobs: dict[str, bytes] = {}
    store = MagicMock()

    def save(stream: BytesIO, **_kwargs: object) -> str:
        key = str(len(blobs))
        blobs[key] = stream.read()
        return key

    store.save_file.side_effect = save
    monkeypatch.setattr(job, "get_default_file_store", lambda: store)
    finish, failed = MagicMock(), MagicMock()
    monkeypatch.setattr(job, "finish_source_package", finish)
    monkeypatch.setattr(job, "mark_source_package_failed", failed)
    monkeypatch.setattr(
        job, "extend_source_package_lease", MagicMock(return_value=True), raising=False
    )
    if page_seconds == 170:
        with pytest.raises(TimeoutError, match="deadline"):
            job.run_source_package(package_id=uuid4(), environment="local-test")
        assert model.invoke.call_count == 1
        assert not blobs
        finish.assert_not_called()
        assert isinstance(failed.call_args.kwargs["failure"], TimeoutError)
        return

    job.run_source_package(package_id=uuid4(), environment="local-test")
    frozen = finish.call_args.kwargs["result"].assets[0]
    vision_factory.assert_called_once_with(temperature=0)
    assert frozen.native_text == "Resmi Gazete"
    assert frozen.pdf_vision is not None
    prepared = AnnexExtraction.model_validate_json(blobs[frozen.pdf_vision.file_id])
    assert prepared.page_count == 12
    assert {element.locator.page for element in prepared.elements} == set(range(1, 13))
    assert frozen.text.count("Bugday | 17%") == 12
    assert model.invoke.call_count == 3
    assert all(
        call.kwargs["use_streaming"] is False for call in model.invoke.call_args_list
    )
    failed.assert_not_called()


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
@pytest.mark.parametrize(
    "instruction",
    [
        "MADDE 9 - Basvuru suresi on gundur.",
        "MADDE 9 - Replace the following sentence with the sentence below.",
        "MADDE 9 - Replace corporate growth with annual growth.",
        "MADDE 9 - Replace the rate of 5% with 7%.",
        "MADDE 9 - Yuzde 5 orani yuzde 7 olarak degistirilmistir.",
    ],
)
def test_unrelated_table_does_not_require_images_for_plain_text_update(
    monkeypatch: pytest.MonkeyPatch, repeated: bool, instruction: str
) -> None:
    import json

    from onyx.regulatory.amendments import pdf_vision
    from onyx.regulatory.amendments.annexes.models import AnnexExtraction
    from onyx.regulatory.amendments.models import AmendmentInstruction

    source, asset, store, blobs, render = frozen_case(monkeypatch)
    assert asset.pdf_vision is not None
    extraction = AnnexExtraction.model_validate_json(blobs[asset.pdf_vision.file_id])
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


@pytest.mark.parametrize("mode", ["associated", "multiple", "boundary", "missing"])
@pytest.mark.parametrize(
    "table_term", ["tablosu", "çizelge", "cizelge", "tablodaki", "çizelgedeki"]
)
def test_cross_page_ordinary_table_update_requires_original_image(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    table_term: str,
) -> None:
    import json

    from onyx.regulatory.amendments import pdf_vision
    from onyx.regulatory.amendments.annexes.models import AnnexExtraction
    from onyx.regulatory.amendments.models import AmendmentInstruction

    source, asset, store, blobs, render = frozen_case(monkeypatch)
    assert asset.pdf_vision is not None
    extraction = AnnexExtraction.model_validate_json(blobs[asset.pdf_vision.file_id])
    instruction = f"MADDE 3 - Ekli {table_term} degistirilmistir."
    extraction.elements[0].text = instruction
    for element in extraction.elements[1:]:
        element.locator.page = 2
    extraction.page_count = 2
    if mode == "multiple":
        another = [item.model_copy(deep=True) for item in extraction.elements[1:]]
        for item in another:
            item.locator.page = 3
        extraction.elements.extend(another)
        extraction.page_count = 3
    if mode == "boundary":
        boundary = extraction.elements[0].model_copy(deep=True)
        boundary.text = "MADDE 4 - Farkli degisiklik."
        boundary.locator.page = 2
        extraction.elements.insert(1, boundary)
    if mode == "missing":
        extraction.elements[0].text = "MADDE 9 - Baska konu."
    pages = []
    for page in range(1, (extraction.page_count or 0) + 1):
        image = Image.new("RGB", (480, 320), "white")
        draw = ImageDraw.Draw(image)
        for _, element in pdf_vision.page_elements(extraction, page):
            box = element.locator.normalized_box
            assert box is not None
            draw.text(
                (int(box[0] * 480), int(box[1] * 320)), element.text, fill="black"
            )
        pages.append(image)
    original_pdf = BytesIO()
    pages[0].save(original_pdf, "PDF", save_all=True, append_images=pages[1:])
    blobs["original"] = original_pdf.getvalue()
    original_sha = hashlib.sha256(blobs["original"]).hexdigest()
    extraction.source_sha256 = original_sha
    source = source.model_copy(
        update={
            "originals": [
                source.originals[0].model_copy(update={"sha256": original_sha})
            ]
        }
    )
    data = extraction.model_dump_json().encode()
    blobs[asset.pdf_vision.file_id] = data
    manifest = json.loads(blobs["manifest"])
    entry = manifest["assets"][0]
    entry["sha256"] = original_sha
    entry["text"] = pdf_vision.pdf_transcript(extraction)
    entry["pdf_vision"]["sha256"] = hashlib.sha256(data).hexdigest()
    entry["pdf_vision"]["transcript_sha256"] = pdf_vision.digest(entry["text"])
    blobs["manifest"] = json.dumps(manifest).encode()
    source = source.model_copy(
        update={"manifest_sha256": hashlib.sha256(blobs["manifest"]).hexdigest()}
    )
    if mode == "associated":
        from onyx.utils.process_isolation import run_in_isolated_process

        render = MagicMock(wraps=run_in_isolated_process)
        monkeypatch.setattr(pdf_vision, "run_in_isolated_process", render)
    if mode != "associated":
        with pytest.raises(ValueError, match="ambiguous"):
            pdf_vision.prepare_pdf_draft_evidence(
                source, [AmendmentInstruction(instruction_text=instruction)], store
            )
        render.assert_not_called()
    else:
        evidence = pdf_vision.prepare_pdf_draft_evidence(
            source, [AmendmentInstruction(instruction_text=instruction)], store
        )
        assert evidence is not None
        assert [page.page for page in evidence.pages] == [1, 2]
        assert len(evidence.image_parts) == 2
        assert evidence.image_parts[0] != evidence.image_parts[1]
        assert evidence.pages[1].positions == list(range(1, len(extraction.elements)))
        assert render.call_args.args[1] == blobs["original"]
        grounding = MagicMock(
            return_value=pdf_vision.PdfGroundingVerdict(
                supported=True, ambiguous=False, rationale="fixture"
            )
        )
        monkeypatch.setattr(pdf_vision, "generate_structured", grounding)
        receipt = pdf_vision.verify_pdf_draft(
            MagicMock(),
            evidence=evidence,
            instructions=[AmendmentInstruction(instruction_text=instruction)],
            old_chunk={"id": "old", "text": "Bugday | 5%"},
            draft_text="Bugday | 17%",
        )
        assert receipt.pages == evidence.pages
        assert grounding.call_args.kwargs["image_parts"] == evidence.image_parts


def test_legacy_partial_pdf_retry_keeps_entire_package_text_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import json
    from types import SimpleNamespace
    from uuid import uuid4

    from onyx.regulatory.amendments import analysis_llm
    from onyx.regulatory.amendments.annexes import job
    from onyx.regulatory.amendments.annexes.models import AcquisitionResult

    source, asset, store, blobs, _ = frozen_case(monkeypatch)
    old = asset.model_copy(
        update={"pdf_vision": None, "native_text": None, "text": "frozen legacy text"}
    )
    previous = (
        AcquisitionResult(status="partial", assets=[old]).model_dump_json().encode()
    )
    blobs["manifest"] = previous
    cached = SimpleNamespace(
        id=source.originals[0].asset_id,
        sha256=asset.sha256,
        file_id="original",
        byte_count=len(blobs["original"]),
        mime_type="application/pdf",
        original_url="https://fixture.invalid/root.pdf",
        final_url="https://fixture.invalid/root.pdf",
    )
    package = SimpleNamespace(
        input_spec={"url": cached.final_url},
        input_file_id=None,
        manifest_file_id="manifest",
        manifest_sha256=hashlib.sha256(previous).hexdigest(),
    )
    monkeypatch.setattr(
        job, "claim_source_package", MagicMock(return_value=(package, uuid4()))
    )
    monkeypatch.setattr(job, "list_source_assets", MagicMock(return_value=[cached]))
    monkeypatch.setattr(job, "get_session_with_current_tenant", MagicMock())
    monkeypatch.setattr(job, "get_default_file_store", lambda: store)
    vision = MagicMock(side_effect=AssertionError("legacy retry must not use vision"))
    monkeypatch.setattr(
        analysis_llm, "get_amendment_analysis_llm", lambda **kwargs: vision(**kwargs)
    )
    download = MagicMock(side_effect=AssertionError("cached PDF must not download"))
    monkeypatch.setattr(job, "download_source", download)
    new_bytes = b"new linked PDF fixture"
    new_asset = AcquiredAsset(
        content=new_bytes,
        sha256=hashlib.sha256(new_bytes).hexdigest(),
        mime_type="application/pdf",
        display_name="linked.pdf",
        text="linked native text",
    )

    def acquire(**kwargs):
        fetched = kwargs["fetch"](cached.final_url)
        assert fetched.content == blobs["original"]
        return AcquisitionResult(
            status="ready",
            assets=[old.model_copy(update={"text": "parser changed"}), new_asset],
        )

    monkeypatch.setattr(job, "acquire_source_package", acquire)
    finish, failed = MagicMock(), MagicMock()
    monkeypatch.setattr(job, "finish_source_package", finish)
    monkeypatch.setattr(job, "mark_source_package_failed", failed)
    job.run_source_package(package_id=source.package_id, environment="local-test")
    failed.assert_not_called()
    result = finish.call_args.kwargs["result"]
    assert [item.text for item in result.assets] == [
        "frozen legacy text",
        "linked native text",
    ]
    assert all(
        item.pdf_vision is None and item.native_text is None for item in result.assets
    )
    assert len(finish.call_args.kwargs["assets"]) == 1
    assert blobs["manifest"] == previous
    assert not any(
        item.get("pdf_vision")
        for item in json.loads(blobs[finish.call_args.kwargs["manifest_file_id"]])[
            "assets"
        ]
    )
    vision.assert_not_called()
    download.assert_not_called()


def test_one_incomplete_page_group_is_retried_without_losing_the_others(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dropped page must cost one retry of its group, not the whole document."""

    from onyx.regulatory.amendments.annexes import pdf_document

    monkeypatch.setattr(
        pdf_document,
        "run_in_isolated_process",
        lambda *_args, **_kwargs: [(200, 300)] * 6,
    )
    requested: list[tuple[int, int]] = []

    def generate(*_args: object, **kwargs: object) -> pdf_document.PdfVisionDocument:
        prompt = cast(str, kwargs["user_prompt"])
        first, last = (1, 4) if "pages 1 through 4" in prompt else (5, 6)
        requested.append((first, last))
        # The first attempt at the opening group silently drops its last page.
        pages = range(
            first,
            last
            if (first, last) == (1, 4) and requested.count((1, 4)) == 1
            else last + 1,
        )
        return pdf_document.PdfVisionDocument(
            pages=[
                pdf_document.PdfVisionPage(
                    page=page, complete=True, elements=vision_result().elements
                )
                for page in pages
            ]
        )

    monkeypatch.setattr(pdf_document, "generate_structured", generate)
    model = MagicMock()
    model.config.model_provider = "fixture"
    model.config.model_name = "vision"

    extraction = pdf_document.extract_pdf_document(
        b"six page pdf", llm=model, deadline=time.monotonic() + 600
    )

    assert extraction.page_count == 6
    assert {element.locator.page for element in extraction.elements} == set(range(1, 7))
    # Two groups, with exactly one extra attempt spent on the group that failed.
    assert requested == [(1, 4), (1, 4), (5, 6)]
