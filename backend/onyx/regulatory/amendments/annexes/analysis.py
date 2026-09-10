"""Prepare complete immutable annex reviews from the live Updates batch scope."""

import hashlib
import re
from datetime import date

from onyx.db.engine.sql_engine import get_session_with_current_tenant
from onyx.db.models import SearchSettings
from onyx.db.regulatory_amendments import get_batch
from onyx.db.regulatory_annex_changes import (
    capture_canonical_scope,
    capture_preparation_configuration,
    legacy_text_annex_is_complete,
    load_review_evidence_scope,
    persist_annex_checkpoint,
    resolve_annex_instruction_file,
)
from onyx.db.regulatory_annexes import normalize_annex_label, require_annex_file_scope
from onyx.file_store.file_store import FileStore, get_default_file_store
from onyx.llm.interfaces import LLM
from onyx.regulatory.amendments.annexes import config
from onyx.regulatory.amendments.annexes.baseline import prepare_legacy_baseline
from onyx.regulatory.amendments.annexes.comparison import compare_annexes
from onyx.regulatory.amendments.annexes.context_dependencies import (
    compare_context_views,
    context_hash,
)
from onyx.regulatory.amendments.annexes.evidence import (
    build_new_evidence_remapping,
    choose_original_evidence,
    combine_annex_evidence_views,
    freeze_review_original,
    freeze_review_pages,
    read_original_source_text,
    read_source_graph,
    select_annex_evidence_view,
    select_new_annex_sources,
    validate_baseline_evidence_view,
)
from onyx.regulatory.amendments.annexes.extraction import extract_annex_structure
from onyx.regulatory.amendments.annexes.models import (
    AnnexChangeDraft,
    AnnexExtraction,
    AnnexInstructionGroup,
    AnnexOriginalEvidence,
    AnnexRenderedPage,
    AnnexReviewEvidence,
    AnnexReviewEvidenceScope,
)
from onyx.regulatory.amendments.annexes.patch_plan import prepare_annex_patch
from onyx.regulatory.amendments.annexes.staging import (
    canonical_snapshot_rows,
    prepare_staged_candidate_rows,
    stage_canonical_items,
    staged_canonical_predecessors,
)
from onyx.regulatory.amendments.models import AmendmentInstruction, DateResolution
from onyx.regulatory.indexing_jobs.models import RegulatoryIndexingConfigSnapshot

_ANNEX_REFERENCE = re.compile(
    r"(?<![\w])(?:ek|annex|appendix)[\s:–—-]*(?:[0-9]+[a-z]?|[ivxlcdm]+[a-z]?|[a-z])(?:[/.-][0-9a-z]+)*(?![\w/])",
    re.IGNORECASE,
)


def group_annex_instructions(
    instructions: list[AmendmentInstruction],
) -> list[AnnexInstructionGroup]:
    groups: dict[tuple[str, str], AnnexInstructionGroup] = {}
    for index, instruction in enumerate(instructions):
        from onyx.regulatory.heading_path import (
            extract_single_regulatory_provision_reference,
        )

        reference = instruction.article_reference or ""
        reference_labels = list(_ANNEX_REFERENCE.finditer(reference))
        if (
            not reference_labels
            and extract_single_regulatory_provision_reference(reference) is not None
        ):
            continue
        if re.search(
            r"(?:maddesin(?:de|in)|fıkrasın(?:da|ın)|bendin(?:de|in)|article\s+\d+)",
            instruction.instruction_text,
            re.IGNORECASE,
        ):
            continue
        labels = list(
            dict.fromkeys(
                normalize_annex_label(match.group())
                for match in (
                    reference_labels
                    or list(_ANNEX_REFERENCE.finditer(instruction.instruction_text))
                )
            )
        )
        if not labels:
            continue
        # Ambiguous multi-annex instructions stay grouped and blocked; never truncate a label.
        label = labels[0] if len(labels) == 1 else "ambiguous:" + ",".join(labels)
        group = groups.setdefault(
            (label, (instruction.target_source or "").casefold().strip()),
            AnnexInstructionGroup(
                annex_label=label,
                instruction_indices=[],
                instruction_texts=[],
                target_sources=[],
            ),
        )
        group.instruction_indices.append(index)
        group.instruction_texts.append(instruction.instruction_text)
        if (
            instruction.target_source
            and instruction.target_source not in group.target_sources
        ):
            group.target_sources.append(instruction.target_source)
    return list(groups.values())


def _verified_bytes(store: FileStore, original: AnnexOriginalEvidence) -> bytes:
    with store.read_file(original.file_id) as stream:
        content = stream.read(25 * 1024 * 1024 + 1)
    if (
        len(content) > 25 * 1024 * 1024
        or hashlib.sha256(content).hexdigest() != original.sha256
    ):
        raise ValueError("original_integrity_mismatch")
    return content


def _extract_cached(
    store: FileStore,
    original: AnnexOriginalEvidence,
    cache: dict[str, AnnexExtraction],
    vision_llm: LLM | None,
) -> AnnexExtraction:
    key = context_hash([original.sha256, original.mime_type])
    if key not in cache:
        cache[key] = extract_annex_structure(
            _verified_bytes(store, original),
            original.mime_type or "",
            vision_llm=vision_llm,
        )
    return cache[key].model_copy(deep=True)


def _freeze_side(
    store: FileStore,
    scope: AnnexReviewEvidenceScope,
    side: str,
    originals: list[AnnexOriginalEvidence],
    extraction: AnnexExtraction,
) -> tuple[list[AnnexRenderedPage], list[AnnexReviewEvidence]]:
    pages: list[AnnexRenderedPage] = []
    evidence: list[AnnexReviewEvidence] = []
    for original in originals:
        frozen = freeze_review_original(
            store, scope=scope, side=side, original=original
        )
        rendered, derived = freeze_review_pages(
            store, scope=scope, original=frozen, extraction=extraction
        )
        pages.extend(rendered)
        evidence.extend([frozen, *derived])
    return pages, evidence


def prepare_annex_group(
    *,
    batch_id: int,
    group: AnnexInstructionGroup,
    effective_date: date | None,
    llm: LLM,
    vision_llm: LLM | None,
    cache: dict[str, AnnexExtraction] | None = None,
) -> AnnexChangeDraft:
    from onyx.db.amendment_sources import (
        list_source_assets,
        require_ready_source_package,
    )

    store = get_default_file_store()
    cache = cache if cache is not None else {}
    with get_session_with_current_tenant() as session:
        batch = get_batch(session, batch_id)
        if batch is None:
            raise ValueError("batch_missing")
        draft = AnnexChangeDraft(
            batch_id=batch_id,
            target_sources=group.target_sources,
            instruction_indices=group.instruction_indices,
            instruction_texts=group.instruction_texts,
            annex_label=group.annex_label,
            effective_date=effective_date,
            source_package_id=batch.source_package_id,
            source_text_sha256=hashlib.sha256(batch.raw_text.encode()).hexdigest(),
            submitted_source_text=batch.raw_text,
        )
        if effective_date is None:
            return draft.model_copy(update={"issues": ["effective_date_unresolved"]})
        try:
            if group.annex_label.startswith("ambiguous:"):
                raise ValueError("instruction_annex_scope_ambiguous")
            if batch.source_package_id is None:
                raise ValueError("source_package_missing")
            package = require_ready_source_package(
                session,
                package_id=batch.source_package_id,
                document_set_id=batch.document_set_id,
                environment=config.REGULATORY_ANNEX_ENVIRONMENT,
            )
            if package.created_by != batch.created_by:
                raise ValueError("source_owner_mismatch")
            user_file_id = resolve_annex_instruction_file(
                session,
                batch=batch,
                annex_label=group.annex_label,
                target_sources=group.target_sources,
                effective_date=effective_date,
            )
            baseline_scope = capture_canonical_scope(session, user_file_id)
            baseline = prepare_legacy_baseline(
                session,
                store,
                document_set_id=batch.document_set_id,
                user_file_id=user_file_id,
                annex_label=group.annex_label,
                as_of_date=effective_date,
            )
            scope = load_review_evidence_scope(
                session,
                batch_id=batch_id,
                user_file_id=user_file_id,
                environment=config.REGULATORY_ANNEX_ENVIRONMENT,
                created_by=batch.created_by,
            )
            assets = list_source_assets(session, package.id)
            preparation_configuration = capture_preparation_configuration(
                session, user_file_id=user_file_id
            )
            preparation_configuration["analysis_model"] = context_hash(
                llm.config.model_dump(mode="json")
            )
            preparation_configuration["vision_model"] = (
                context_hash(vision_llm.config.model_dump(mode="json"))
                if vision_llm
                else "none"
            )
            manifest_id, manifest_sha256 = (
                package.manifest_file_id,
                package.manifest_sha256,
            )
            draft = draft.model_copy(
                update={
                    "user_file_id": user_file_id,
                    "baseline_scope": baseline_scope,
                    "baseline": baseline,
                    "source_manifest_sha256": manifest_sha256,
                    "preparation_configuration": preparation_configuration,
                }
            )
            # All acquisition authority is resolved before model or source preparation.
            session.commit()
        except ValueError as exc:
            return draft.model_copy(update={"issues": [str(exc)]})
    try:
        if manifest_id is None:
            raise ValueError("source_manifest_missing")
        if manifest_sha256 is None:
            raise ValueError("source_manifest_missing")
        links = read_source_graph(
            store,
            manifest_file_id=manifest_id,
            manifest_sha256=manifest_sha256,
            assets=assets,
        )
        known_urls = {
            url
            for asset in assets
            for url in (asset.original_url, asset.final_url)
            if url
        }
        known_urls.update(
            url for link in links for url in (link.requested_url, link.final_url) if url
        )
        submitted_urls = set(
            re.findall(r"https?://[^\s<>\"']+", draft.submitted_source_text or "")
        )
        if submitted_urls - known_urls:
            raise ValueError("edited_source_requires_complete_acquisition")
        draft = draft.model_copy(
            update={
                "source_graph": links,
                "source_graph_sha256": context_hash(
                    [link.model_dump(mode="json") for link in links]
                ),
                "original_source_text_sha256": read_original_source_text(store, assets)[
                    1
                ],
            }
        )
        canonical_ids = [
            element.canonical_chunk_id
            for element in baseline.elements
            if element.canonical_chunk_id
        ]
        old_originals = choose_original_evidence(
            baseline.originals, canonical_chunk_ids=canonical_ids
        )
        old_views = [
            select_annex_evidence_view(
                extraction=_extract_cached(store, original, cache, vision_llm),
                original=original,
                annex_label=group.annex_label,
                canonical_labels=[group.annex_label],
                canonical_chunk_ids=original.canonical_chunk_ids,
            )
            for original in old_originals
        ]
        old = (
            old_views[0]
            if len(old_views) == 1
            else combine_annex_evidence_views(
                old_views, canonical_chunk_ids=canonical_ids
            )
        )
        baseline_issues = validate_baseline_evidence_view(baseline, old)
        if baseline_issues:
            raise ValueError(",".join(baseline_issues))
        selected: list[tuple[AnnexOriginalEvidence, AnnexExtraction]] = []
        selection_errors: list[str] = []
        for asset in assets:
            original = AnnexOriginalEvidence(
                file_id=asset.file_id,
                sha256=asset.sha256,
                mime_type=asset.mime_type,
                available=True,
            )
            extraction = _extract_cached(store, original, cache, vision_llm)
            labels = [
                link.label for link in links if link.target_asset_hash == asset.sha256
            ]
            try:
                view = select_annex_evidence_view(
                    extraction=extraction,
                    original=original,
                    annex_label=group.annex_label,
                    canonical_labels=[group.annex_label],
                    canonical_chunk_ids=canonical_ids,
                    source_labels=labels,
                )
                selected.append((original, view))
            except ValueError as exc:
                selection_errors.append(str(exc))
        if not selected:
            raise ValueError("new_annex_evidence_missing:" + ";".join(selection_errors))
        new_originals, new = select_new_annex_sources(selected, links)
        old_pages, old_evidence = _freeze_side(store, scope, "old", old_originals, old)
        new_pages, new_evidence = _freeze_side(store, scope, "new", new_originals, new)
        evidence = [*old_evidence, *new_evidence]
        comparison = compare_annexes(
            old=old,
            new=new,
            old_pages=old_pages,
            new_pages=new_pages,
            llm=vision_llm or llm,
            instruction="\n\n".join(group.instruction_texts),
        )
        plan = prepare_annex_patch(
            baseline=baseline,
            old=old,
            new=new,
            comparison=comparison,
            effective_date=effective_date,
            package_complete=True,
        )
        draft = draft.model_copy(
            update={
                "old_extraction": old,
                "new_extraction": new,
                "raw_new_extraction": new,
                "comparison": comparison,
                "patch_plan": plan,
                "evidence": evidence,
            }
        )
        if not plan.ready:
            return draft.model_copy(update={"issues": plan.issues})
        mapping = build_new_evidence_remapping(
            new=new,
            evidence=evidence,
            asset_ids={asset.file_id: asset.id for asset in assets},
        )
        # Explicit group anchor is the final canonical annex row, not a copied numeric position.
        anchor = canonical_ids[-1]
        items = stage_canonical_items(
            plan=plan,
            baseline_scope=baseline_scope,
            comparison=comparison,
            insertion_after_chunk_id=anchor,
            evidence_remapping=mapping,
        )
        draft = draft.model_copy(
            update={
                "new_evidence_remapping": mapping,
                "insertion_after_chunk_id": anchor,
                "items": items,
                "source_only_canonical_ids": canonical_ids
                if not comparison.changes
                else [],
            }
        )
        return prepare_review_context(draft)
    except ValueError as exc:
        return draft.model_copy(update={"issues": [str(exc)]})


def run_annex_groups(
    *,
    batch_id: int,
    lease_generation: int,
    instructions: list[AmendmentInstruction],
    processed_indices: set[int],
    reference_date: str | None,
    llm: LLM,
) -> set[int]:
    from onyx.llm.factory import get_default_llm_with_vision

    if not config.REGULATORY_ANNEX_UPDATES_ENABLED:
        return set()
    groups = group_annex_instructions(instructions)
    if not groups:
        return set()
    vision_llm = get_default_llm_with_vision()
    cache: dict[str, AnnexExtraction] = {}
    covered: set[int] = set()
    for group in groups:
        indices = set(group.instruction_indices)
        if indices.issubset(processed_indices):
            covered.update(indices)
            continue
        if indices.intersection(processed_indices):
            raise ValueError("partial_annex_checkpoint_coverage")
        with get_session_with_current_tenant() as session:
            batch = get_batch(session, batch_id)
            if batch is None:
                raise ValueError("batch missing")
            if legacy_text_annex_is_complete(
                session,
                batch=batch,
                group=group,
                reference_date=date.fromisoformat(reference_date)
                if reference_date
                else date.today(),
            ):
                continue
            covered.update(indices)
            source_text = batch.raw_text
        resolved = resolve_group_effective_date(
            llm=llm, group=group, source_text=source_text, reference_date=reference_date
        )
        draft = prepare_annex_group(
            batch_id=batch_id,
            group=group,
            effective_date=date.fromisoformat(resolved.effective_start_date)
            if resolved.effective_start_date
            else None,
            llm=llm,
            vision_llm=vision_llm,
            cache=cache,
        )
        draft = draft.model_copy(update={"date_resolution": resolved})
        if resolved.effective_end_date is not None:
            draft = draft.model_copy(
                update={
                    "issues": [
                        *draft.issues,
                        "temporary_annex_publication_contract_required",
                    ]
                }
            )
        with get_session_with_current_tenant() as session:
            if (
                persist_annex_checkpoint(
                    session,
                    batch_id=batch_id,
                    lease_generation=lease_generation,
                    draft=draft,
                    environment=config.REGULATORY_ANNEX_ENVIRONMENT,
                )
                is None
            ):
                raise RuntimeError("annex batch lost its analysis lease")
    return covered


def prepare_review_context(draft: AnnexChangeDraft) -> AnnexChangeDraft:
    from onyx.configs.app_configs import REGULATORY_BATCH_INDEXING_ENABLED
    from onyx.db.models import RegulatoryIndexingJob
    from onyx.db.search_settings import get_current_search_settings
    from onyx.indexing.embedder import DefaultIndexingEmbedder
    from onyx.regulatory.indexing_jobs.configuration import (
        resolve_regulatory_indexing_snapshot,
    )
    from onyx.regulatory.projection import prepare_normal_context_view

    if (
        draft.user_file_id is None
        or draft.effective_date is None
        or draft.patch_plan is None
    ):
        raise ValueError("context preparation scope missing")
    with get_session_with_current_tenant() as session:
        batch = get_batch(session, draft_batch_id(draft))
        if batch is None:
            raise ValueError("batch missing")
        file = require_annex_file_scope(
            session, batch.document_set_id, draft.user_file_id
        )
        settings = get_current_search_settings(session)
        snapshot = (
            resolve_regulatory_indexing_snapshot(session)
            if REGULATORY_BATCH_INDEXING_ENABLED
            else None
        )
        configuration = capture_preparation_configuration(
            session, user_file_id=draft.user_file_id
        )
    embedder = DefaultIndexingEmbedder.from_db_search_settings(search_settings=settings)
    context_llm = resolve_review_context_llm(settings, snapshot)
    configuration["context_model"] = (
        context_hash(context_llm.config.model_dump(mode="json"))
        if context_llm
        else "none"
    )
    configuration["transport"] = (
        context_hash(snapshot.model_dump(mode="json")) if snapshot else "normal"
    )
    old_rows = canonical_snapshot_rows(draft.baseline_scope)
    candidate = prepare_staged_candidate_rows(
        baseline_scope=draft.baseline_scope,
        items=draft.items,
        effective_date=draft.effective_date,
        evidence_remapping=draft.new_evidence_remapping,
    )
    if snapshot is None:
        before = prepare_normal_context_view(
            rows=old_rows,
            user_file=file,
            search_settings=settings,
            embedder=embedder,
            llm=context_llm,
            as_of_date=draft.effective_date,
            cached=draft.baseline_context,
        )
        after = prepare_normal_context_view(
            rows=candidate,
            user_file=file,
            search_settings=settings,
            embedder=embedder,
            llm=context_llm,
            cached=before,
            as_of_date=draft.effective_date,
        )
    else:
        from onyx.llm.constants import LlmProviderNames
        from onyx.llm.models import ChatCompletionMessage, TextContentPart, UserMessage
        from onyx.llm.utils import llm_response_to_string
        from onyx.regulatory.indexing_jobs.contextual import (
            get_contextual_token_budget_tokenizer,
            prepare_durable_context_view,
        )
        from onyx.regulatory.indexing_jobs.vertex_batch import VertexBatchRequest
        from onyx.tracing.flows import LLMFlow
        from onyx.tracing.llm_utils import llm_generation_span

        if (
            context_llm is None
            or context_llm.config.model_provider != LlmProviderNames.VERTEX_AI
            or context_llm.config.model_name != snapshot.vertex.model_name
        ):
            raise ValueError(
                "durable contextual model differs from frozen configuration"
            )
        job = RegulatoryIndexingJob(
            user_file_id=file.id, config_snapshot=snapshot.model_dump(mode="json")
        )
        tokenizer = get_contextual_token_budget_tokenizer(
            model_provider=LlmProviderNames.VERTEX_AI,
            model_name=snapshot.vertex.model_name,
        )

        def generate(request: VertexBatchRequest) -> str:
            messages: list[ChatCompletionMessage] = [
                UserMessage(content=[TextContentPart(text=request.prompt)])
            ]
            with llm_generation_span(
                llm=context_llm,
                flow=LLMFlow.REGULATORY_CONTEXTUAL_BATCH,
                input_messages=messages,
            ):
                return llm_response_to_string(
                    context_llm.invoke(messages, timeout_override=60, max_tokens=256)
                )

        before = prepare_durable_context_view(
            job=job,
            rows=old_rows,
            embedding_tokenizer=embedder.embedding_model.tokenizer,
            contextual_tokenizer=tokenizer,
            embedding_model=embedder.embedding_model,
            generate=generate,
            cached=draft.baseline_context,
            as_of_date=draft.effective_date,
        )
        after = prepare_durable_context_view(
            job=job,
            rows=candidate,
            embedding_tokenizer=embedder.embedding_model.tokenizer,
            contextual_tokenizer=tokenizer,
            embedding_model=embedder.embedding_model,
            generate=generate,
            cached=before,
            as_of_date=draft.effective_date,
        )
    impact = compare_context_views(
        old=before,
        new=after,
        direct_canonical_changes=[
            chunk.id for item in draft.items for chunk in item.new_chunks
        ],
        metadata_only=draft.patch_plan.metadata_only,
        canonical_predecessors=staged_canonical_predecessors(draft.items),
    )
    return draft.model_copy(
        update={
            "baseline_context": before,
            "impact": impact,
            "preparation_configuration": {
                **draft.preparation_configuration,
                **configuration,
            },
            "indexing_configuration": snapshot.model_dump(mode="json")
            if snapshot
            else None,
        }
    )


def draft_batch_id(draft: AnnexChangeDraft) -> int:
    if draft.batch_id is None:
        raise ValueError("live draft batch identity missing")
    return draft.batch_id


def validate_live_review_configuration(draft: AnnexChangeDraft) -> None:
    from onyx.configs.app_configs import REGULATORY_BATCH_INDEXING_ENABLED
    from onyx.db.amendment_sources import (
        list_source_assets,
        require_ready_source_package,
    )
    from onyx.db.search_settings import get_current_search_settings
    from onyx.llm.factory import get_default_llm, get_default_llm_with_vision
    from onyx.regulatory.indexing_jobs.configuration import (
        resolve_regulatory_indexing_snapshot,
    )

    if draft.user_file_id is None or draft.source_package_id is None:
        raise ValueError("review preparation scope missing")
    with get_session_with_current_tenant() as session:
        actual = capture_preparation_configuration(
            session, user_file_id=draft.user_file_id
        )
        settings = get_current_search_settings(session)
        snapshot = (
            resolve_regulatory_indexing_snapshot(session)
            if REGULATORY_BATCH_INDEXING_ENABLED
            else None
        )
        assets = list_source_assets(session, draft.source_package_id)
        batch = get_batch(session, draft_batch_id(draft))
        if batch is None:
            raise ValueError("batch missing")
        package = require_ready_source_package(
            session,
            package_id=draft.source_package_id,
            document_set_id=batch.document_set_id,
            environment=config.REGULATORY_ANNEX_ENVIRONMENT,
        )
        manifest_id, manifest_hash = package.manifest_file_id, package.manifest_sha256
    if (
        manifest_id is None
        or manifest_hash != draft.source_manifest_sha256
        or manifest_hash is None
    ):
        raise ValueError("source manifest changed")
    graph = read_source_graph(
        get_default_file_store(),
        manifest_file_id=manifest_id,
        manifest_sha256=manifest_hash,
        assets=assets,
    )
    if (
        graph != draft.source_graph
        or context_hash([item.model_dump(mode="json") for item in graph])
        != draft.source_graph_sha256
    ):
        raise ValueError("source graph changed")
    context_llm = resolve_review_context_llm(settings, snapshot)
    vision_llm = get_default_llm_with_vision()
    actual["analysis_model"] = context_hash(
        get_default_llm().config.model_dump(mode="json")
    )
    actual["vision_model"] = (
        context_hash(vision_llm.config.model_dump(mode="json"))
        if vision_llm
        else "none"
    )
    actual["context_model"] = (
        context_hash(context_llm.config.model_dump(mode="json"))
        if context_llm
        else "none"
    )
    actual["transport"] = (
        context_hash(snapshot.model_dump(mode="json")) if snapshot else "normal"
    )
    if actual != draft.preparation_configuration:
        raise ValueError("prepared runtime/model/index configuration changed")
    _, original_text_hash = read_original_source_text(get_default_file_store(), assets)
    if original_text_hash != draft.original_source_text_sha256:
        raise ValueError("original extracted source text changed")


def resolve_group_effective_date(
    *,
    llm: LLM,
    group: AnnexInstructionGroup,
    source_text: str,
    reference_date: str | None,
) -> "DateResolution":
    from onyx.prompts.regulatory_annex_review import ANNEX_EFFECTIVE_DATE_PROMPT
    from onyx.regulatory.amendments.models import DateResolution
    from onyx.regulatory.structured_llm import generate_structured
    from onyx.tracing.flows import LLMFlow

    return generate_structured(
        llm,
        flow=LLMFlow.REGULATORY_ANNEX_EFFECTIVE_DATE,
        system_prompt=ANNEX_EFFECTIVE_DATE_PROMPT,
        user_prompt=f"Reference/publication date: {reference_date}\nFull submitted source:\n{source_text}\nGrouped target instructions:\n"
        + "\n".join(group.instruction_texts),
        response_model=DateResolution,
        timeout_override=60,
    )


def resolve_review_context_llm(
    settings: "SearchSettings", snapshot: "RegulatoryIndexingConfigSnapshot | None"
) -> LLM | None:
    from onyx.indexing.contextual_settings import require_contextual_rag_llm

    if snapshot is None:
        return require_contextual_rag_llm(settings)
    from onyx.db.llm import fetch_model_configuration_by_id
    from onyx.llm.factory import llm_from_provider
    from onyx.server.manage.llm.models import LLMProviderView

    with get_session_with_current_tenant() as session:
        model = fetch_model_configuration_by_id(
            session, snapshot.vertex.model_configuration_id
        )
        if model is None or model.name != snapshot.vertex.model_name:
            raise ValueError("durable contextual model configuration changed")
        provider = LLMProviderView.from_model(model.llm_provider)
    return llm_from_provider(
        model_name=snapshot.vertex.model_name,
        llm_provider=provider,
        timeout=60,
        temperature=0,
    )
