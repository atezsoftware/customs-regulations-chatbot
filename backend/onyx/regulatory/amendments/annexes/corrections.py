"""Typed human corrections reconciled against unchanged frozen source evidence."""

import base64
import hashlib
from uuid import UUID

from onyx.file_store.file_store import FileStore, get_default_file_store
from onyx.llm.interfaces import LLM
from onyx.llm.models import ImageContentPart, ImageUrlDetail
from onyx.prompts.regulatory_annex_review import ANNEX_CORRECTION_RECONCILIATION_PROMPT
from onyx.regulatory.amendments.annexes.evidence import evidence_view_hash
from onyx.regulatory.amendments.annexes.models import (
    AnnexChangeDraft,
    AnnexCorrectionReconciliation,
    AnnexElementCorrection,
    AnnexExtraction,
    AnnexRenderedPage,
    AnnexReviewEvidence,
)
from onyx.regulatory.structured_llm import generate_structured
from onyx.tracing.flows import LLMFlow


def apply_bound_corrections(
    raw: AnnexExtraction, corrections: list[AnnexElementCorrection]
) -> AnnexExtraction:
    corrected = raw.model_copy(deep=True)
    if len({item.position for item in corrections}) != len(corrections):
        raise ValueError("duplicate correction position")
    for item in corrections:
        if (
            item.position >= len(raw.elements)
            or raw.elements[item.position].text != item.before_text
        ):
            raise ValueError("correction before value or position changed")
        corrected.elements[item.position].text = item.corrected_text
    if corrected.evidence_view is not None:
        corrected.evidence_view = corrected.evidence_view.model_copy(
            update={"sha256": evidence_view_hash(corrected)}
        )
    return corrected


def read_frozen_evidence(store: FileStore, evidence: AnnexReviewEvidence) -> bytes:
    with store.read_file(evidence.file_id) as stream:
        content = stream.read(evidence.byte_count + 1)
    if (
        len(content) != evidence.byte_count
        or hashlib.sha256(content).hexdigest() != evidence.sha256
    ):
        raise ValueError("frozen evidence integrity changed")
    return content


def frozen_comparison_pages(
    store: FileStore, draft: AnnexChangeDraft, side: str
) -> list[AnnexRenderedPage]:
    extraction = draft.old_extraction if side == "old" else draft.new_extraction
    if extraction is None:
        raise ValueError("review extraction missing")
    pages: list[AnnexRenderedPage] = []
    for evidence in draft.evidence:
        if evidence.side != side or evidence.kind != "comparison_page":
            continue
        page = evidence.locator.page
        if extraction.evidence_view is not None:
            view = extraction.evidence_view
            parent_indices = [
                index
                for index, parent in enumerate(view.parents)
                if parent.file_id == evidence.parent_file_id
            ]
            mapped = [
                item.view_page
                for item in view.pages
                if item.parent_index in parent_indices and item.original_page == page
            ]
            if len(mapped) != 1:
                raise ValueError("frozen comparison page mapping changed")
            page = mapped[0]
        if (
            page is None
            or evidence.locator.original_width is None
            or evidence.locator.original_height is None
        ):
            raise ValueError("frozen comparison page dimensions missing")
        pages.append(
            AnnexRenderedPage(
                page=page,
                png=read_frozen_evidence(store, evidence),
                width=evidence.locator.original_width,
                height=evidence.locator.original_height,
            )
        )
    return sorted(pages, key=lambda item: item.page)


def correction_source_inputs(
    draft: AnnexChangeDraft,
    corrections: list[AnnexElementCorrection],
) -> tuple[list[ImageContentPart], list[str]]:
    """Re-read independent originals and require evidence at every corrected locator."""
    from onyx.regulatory.amendments.annexes.evidence import validate_evidence_view
    from onyx.regulatory.amendments.annexes.extraction import extract_annex_structure

    raw = draft.raw_new_extraction
    if raw is None or validate_evidence_view(raw):
        raise ValueError("correction source scope missing or invalid")
    store = get_default_file_store()
    native: list[str] = []
    images: list[ImageContentPart] = []
    parsed: dict[str, AnnexExtraction] = {}
    supplied_images: set[UUID] = set()
    for correction in corrections:
        position = correction.position
        locator = raw.elements[position].locator
        original_position = position
        if raw.evidence_view is not None:
            mapping = raw.evidence_view.element_mappings[position]
            parent = raw.evidence_view.parents[mapping.parent_index]
            parent_id, parent_hash, mime = (
                parent.file_id,
                parent.sha256,
                parent.mime_type,
            )
            original_position, locator = (
                mapping.original_position,
                mapping.original_locator,
            )
        else:
            matches = [
                evidence
                for evidence in draft.evidence
                if evidence.side == "new"
                and evidence.kind == "original"
                and evidence.parent_sha256 == raw.source_sha256
                and evidence.mime_type == raw.mime_type
            ]
            if len(matches) != 1:
                raise ValueError("correction source original missing or ambiguous")
            parent_id, parent_hash, mime = (
                matches[0].parent_file_id,
                raw.source_sha256,
                raw.mime_type,
            )
        originals = [
            evidence
            for evidence in draft.evidence
            if evidence.side == "new"
            and evidence.kind == "original"
            and evidence.parent_file_id == parent_id
            and evidence.parent_sha256 == parent_hash
            and evidence.sha256 == parent_hash
            and evidence.mime_type == mime
        ]
        if len(originals) != 1:
            raise ValueError("correction source original missing or ambiguous")
        original = originals[0]
        content = read_frozen_evidence(store, original)
        if mime.startswith("text/") or mime.endswith(
            ("wordprocessingml.document", "spreadsheetml.sheet")
        ):
            if original.file_id not in parsed:
                parsed[original.file_id] = extract_annex_structure(content, mime)
            source = parsed[original.file_id]
            if original_position >= len(source.elements):
                raise ValueError("correction source position missing")
            element = source.elements[original_position]
            if (
                element.locator != locator
                or element.extraction_method != "native"
                or element.status != "readable"
                or element.issues
                or not element.text.strip()
            ):
                raise ValueError("correction source native locator unavailable")
            native.append(
                f"parent_file_id={parent_id}; parent_sha256={parent_hash}; original_position={original_position}; original_locator="
                + locator.model_dump_json()
                + "\n"
                + element.model_dump_json()
            )
        else:
            pages = [
                evidence
                for evidence in draft.evidence
                if evidence.side == "new"
                and evidence.kind == "comparison_page"
                and evidence.parent_file_id == parent_id
                and evidence.parent_sha256 == parent_hash
                and locator.page is not None
                and evidence.locator.page == locator.page
            ]
            if not pages:
                raise ValueError("correction source visual locator unavailable")
            for evidence in pages:
                if evidence.id not in supplied_images:
                    image = read_frozen_evidence(store, evidence)
                    images.append(
                        ImageContentPart(
                            image_url=ImageUrlDetail(
                                url="data:image/png;base64,"
                                + base64.b64encode(image).decode()
                            )
                        )
                    )
                    supplied_images.add(evidence.id)
    return images, native


def reconcile_corrections(
    *, draft: AnnexChangeDraft, corrections: list[AnnexElementCorrection], llm: LLM
) -> AnnexCorrectionReconciliation:
    if draft.raw_new_extraction is None:
        raise ValueError("immutable raw extraction missing")
    apply_bound_corrections(draft.raw_new_extraction, corrections)
    images, native = correction_source_inputs(draft, corrections)
    verdict = generate_structured(
        llm,
        flow=LLMFlow.REGULATORY_ANNEX_CORRECTION,
        system_prompt=ANNEX_CORRECTION_RECONCILIATION_PROMPT,
        user_prompt="Derived transcription under correction (not original source authority):\n"
        + draft.raw_new_extraction.model_dump_json()
        + "\nOriginal native source:\n"
        + "\n".join(native)
        + "\nProposed corrections:\n"
        + "\n".join(item.model_dump_json() for item in corrections),
        image_parts=images,
        response_model=AnnexCorrectionReconciliation,
        timeout_override=60,
    )

    from onyx.regulatory.amendments.annexes.models import AnnexModelSnapshot

    return verdict.model_copy(
        update={
            "input_sha256": correction_input_hash(draft, corrections),
            "model_snapshot": AnnexModelSnapshot(
                model_provider=llm.config.model_provider,
                model_name=llm.config.model_name,
            ),
        }
    )


def correction_input_hash(
    draft: AnnexChangeDraft, corrections: list[AnnexElementCorrection]
) -> str:
    from onyx.regulatory.amendments.annexes.context_dependencies import context_hash

    return context_hash(
        [
            draft.raw_new_extraction.model_dump(mode="json")
            if draft.raw_new_extraction
            else None,
            [item.model_dump(mode="json") for item in corrections],
            [
                item.model_dump(mode="json")
                for item in draft.evidence
                if item.side == "new"
            ],
        ]
    )


def revalidate_annex_review(
    *,
    draft: AnnexChangeDraft,
    corrections: list[AnnexElementCorrection],
    corrected_by: UUID,
    llm: LLM,
    vision_llm: LLM | None,
) -> AnnexChangeDraft:
    from onyx.db.amendment_sources import list_source_assets
    from onyx.db.engine.sql_engine import get_session_with_current_tenant
    from onyx.db.regulatory_annex_changes import capture_canonical_scope
    from onyx.regulatory.amendments.annexes.analysis import prepare_review_context
    from onyx.regulatory.amendments.annexes.comparison import compare_annexes
    from onyx.regulatory.amendments.annexes.evidence import build_new_evidence_remapping
    from onyx.regulatory.amendments.annexes.patch_plan import prepare_annex_patch
    from onyx.regulatory.amendments.annexes.staging import stage_canonical_items

    if (
        draft.user_file_id is None
        or draft.source_package_id is None
        or draft.raw_new_extraction is None
        or draft.old_extraction is None
        or draft.baseline is None
    ):
        if corrections or draft.batch_id is None:
            raise ValueError("raw source scope missing for corrections")
        from onyx.regulatory.amendments.annexes.analysis import prepare_annex_group
        from onyx.regulatory.amendments.annexes.models import AnnexInstructionGroup

        prepared = prepare_annex_group(
            batch_id=draft.batch_id,
            group=AnnexInstructionGroup(
                annex_label=draft.annex_label,
                instruction_indices=draft.instruction_indices,
                instruction_texts=draft.instruction_texts,
                target_sources=draft.target_sources,
            ),
            effective_date=draft.effective_date,
            llm=llm,
            vision_llm=vision_llm,
        )
        return prepared.model_copy(update={"date_resolution": draft.date_resolution})
    user_file_id, source_package_id, baseline, old_extraction = (
        draft.user_file_id,
        draft.source_package_id,
        draft.baseline,
        draft.old_extraction,
    )
    from onyx.regulatory.amendments.annexes.context_dependencies import context_hash

    configuration = dict(draft.preparation_configuration)
    configuration["analysis_model"] = context_hash(llm.config.model_dump(mode="json"))
    configuration["vision_model"] = (
        context_hash(vision_llm.config.model_dump(mode="json"))
        if vision_llm
        else "none"
    )
    corrected = apply_bound_corrections(draft.raw_new_extraction, corrections)
    draft = draft.model_copy(update={"preparation_configuration": configuration})
    reconciliation = (
        reconcile_corrections(
            draft=draft, corrections=corrections, llm=vision_llm or llm
        )
        if corrections
        else AnnexCorrectionReconciliation(
            supported=True, rationale="No human source corrections"
        )
    )
    draft = draft.model_copy(
        update={
            "corrections": corrections,
            "corrected_by": corrected_by,
            "correction_reconciliation": reconciliation,
            "new_extraction": corrected,
            "items": [],
            "impact": None,
            "issues": [],
        }
    )
    if not reconciliation.supported:
        return draft.model_copy(
            update={
                "issues": ["human_correction_unsupported:" + reconciliation.rationale]
            }
        )
    store = get_default_file_store()
    comparison = compare_annexes(
        old=old_extraction,
        new=corrected,
        old_pages=frozen_comparison_pages(store, draft, "old"),
        new_pages=frozen_comparison_pages(store, draft, "new"),
        llm=vision_llm or llm,
        instruction="\n\n".join(draft.instruction_texts),
    )
    plan = prepare_annex_patch(
        baseline=baseline,
        old=old_extraction,
        new=corrected,
        comparison=comparison,
        effective_date=draft.effective_date,
        package_complete=True,
    )
    with get_session_with_current_tenant() as session:
        baseline_scope = capture_canonical_scope(session, user_file_id)
        assets = list_source_assets(session, source_package_id)
    draft = draft.model_copy(
        update={
            "baseline_scope": baseline_scope,
            "comparison": comparison,
            "patch_plan": plan,
            "source_only_canonical_ids": [
                element.canonical_chunk_id
                for element in baseline.elements
                if element.canonical_chunk_id
            ]
            if not comparison.changes
            else [],
        }
    )
    if not plan.ready:
        return draft.model_copy(update={"issues": plan.issues})
    mapping = build_new_evidence_remapping(
        new=corrected,
        evidence=draft.evidence,
        asset_ids={asset.file_id: asset.id for asset in assets},
    )
    items = stage_canonical_items(
        plan=plan,
        baseline_scope=baseline_scope,
        comparison=comparison,
        insertion_after_chunk_id=draft.insertion_after_chunk_id,
        evidence_remapping=mapping,
    )
    prepared = prepare_review_context(
        draft.model_copy(update={"new_evidence_remapping": mapping, "items": items})
    )
    if (
        prepared.date_resolution is not None
        and prepared.date_resolution.effective_end_date is not None
    ):
        prepared = prepared.model_copy(
            update={"issues": ["temporary_annex_publication_contract_required"]}
        )
    return prepared
