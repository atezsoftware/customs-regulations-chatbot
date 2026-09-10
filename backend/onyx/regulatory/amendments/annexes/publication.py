"""Preapproval legal timelines and actual indexed-vector proof comparison."""

import re
from datetime import date
from hashlib import sha256
from typing import TYPE_CHECKING
from uuid import NAMESPACE_URL, uuid5

from onyx.regulatory.amendments.annexes.context_dependencies import context_hash
from onyx.regulatory.amendments.annexes.models import (
    AnnexChangeDraft,
    AnnexLegalPublicationTimeline,
)

if TYPE_CHECKING:
    from onyx.document_index.publication_models import (
        FrozenPublicationProjection,
        IndexedProjectionEvidence,
    )
    from onyx.regulatory.amendments.annexes.models import FrozenContextProjection


def prepare_legal_publication_timeline(
    draft: AnnexChangeDraft,
) -> AnnexLegalPublicationTimeline:
    """Retain scheduled law and stage explicit temporary restoration identities.

    Only the legal rows named by reviewed items are modified. Context aggregates
    and companion representations are published through dated bindings instead.
    """
    start = draft.effective_date
    if start is None:
        raise ValueError("publication effective date missing")
    end = (
        date.fromisoformat(draft.date_resolution.effective_end_date)
        if draft.date_resolution and draft.date_resolution.effective_end_date
        else None
    )
    if end is not None and end <= start:
        raise ValueError("invalid temporary publication window")
    validate_after_window_authority(draft)
    rows = {row.id: row for row in draft.baseline_scope}
    restoration: dict[str, str] = {}
    # The exact immutable item/source identity makes restoration IDs retry stable.
    identity = context_hash(
        [
            draft.user_file_id,
            draft.source_manifest_sha256,
            start,
            end,
            [item.model_dump(mode="json") for item in draft.items],
        ]
    )
    for item in draft.items:
        old_rows = [rows[identifier] for identifier in item.old_chunk_ids]
        for old in old_rows:
            if (
                old.validity_start_date is not None and old.validity_start_date >= start
            ) or (old.validity_end_date is not None and old.validity_end_date <= start):
                raise ValueError(
                    "reviewed predecessor is not effective before publication"
                )
            if (
                end is not None
                and old.validity_end_date is not None
                and end > old.validity_end_date
                and draft.after_window_authority
                and draft.after_window_authority.kind == "restore_predecessor"
            ):
                raise ValueError(
                    "restoration contradicts intervening scheduled canonical authority"
                )
            successors = [
                row for row in draft.baseline_scope if row.supersedes_chunk_id == old.id
            ]
            if old.superseded_by_chunk_id is not None:
                successor = rows.get(old.superseded_by_chunk_id)
                if successor is None or successor not in successors:
                    raise ValueError("scheduled successor lineage is ambiguous")
            if any(
                row.validity_start_date != old.validity_end_date for row in successors
            ):
                raise ValueError("scheduled successor overlaps reviewed predecessor")
            rows[old.id] = old.model_copy(
                update={
                    "validity_end_date": start,
                    "status": "superseded",
                    "superseded_by_chunk_id": item.new_chunks[0].id
                    if len(item.new_chunks) == 1
                    else None,
                }
            )
            if (
                draft.after_window_authority is not None
                and draft.after_window_authority.kind == "restore_predecessor"
                and end is not None
            ) and (old.validity_end_date is None or end < old.validity_end_date):
                identifier = str(
                    uuid5(NAMESPACE_URL, f"annex-restoration:{identity}:{old.id}")
                )
                restoration[identifier] = old.id
                rows[identifier] = old.model_copy(
                    update={
                        "id": identifier,
                        "validity_start_date": end,
                        "status": "active",
                        "source": "amendment",
                        "projection_ordinal": -1,
                        "supersedes_chunk_id": item.new_chunks[0].id
                        if len(item.new_chunks) == 1
                        else None,
                    }
                )
        successor_bounds = {row.validity_end_date for row in old_rows}
        if len(successor_bounds) > 1:
            raise ValueError("scheduled split/merge successors have ambiguous windows")
        for new in item.new_chunks:
            bounds = [
                value
                for value in (end, new.validity_end_date, *successor_bounds)
                if value is not None
            ]
            new_end = min(bounds) if bounds else None
            if new_end is not None and new_end <= start:
                raise ValueError("scheduled successor leaves no publication interval")
            rows[new.id] = new.model_copy(update={"validity_end_date": new_end})
    for identifier, predecessor in restoration.items():
        item = next(item for item in draft.items if predecessor in item.old_chunk_ids)
        if len(item.old_chunk_ids) == len(item.new_chunks) == 1:
            successor = item.new_chunks[0].id
            rows[successor] = rows[successor].model_copy(
                update={"superseded_by_chunk_id": identifier}
            )
    if (
        end is not None
        and draft.after_window_authority
        and draft.after_window_authority.kind == "cessation"
    ):
        for identifier in draft.source_only_canonical_ids:
            old = rows[identifier]
            rows[identifier] = old.model_copy(
                update={
                    "validity_end_date": min(old.validity_end_date, end)
                    if old.validity_end_date
                    else end
                }
            )
    return AnnexLegalPublicationTimeline(
        canonical_rows=sorted(rows.values(), key=lambda row: (row.position, row.id)),
        restoration_predecessors=restoration,
        effective_start=start,
        effective_end=end,
    )


def validate_after_window_authority(draft: AnnexChangeDraft) -> None:
    end = draft.date_resolution.effective_end_date if draft.date_resolution else None
    if end is None:
        if draft.after_window_authority is not None:
            raise ValueError("after-window authority without an effective end")
        return
    authority = draft.after_window_authority
    if authority is None or authority.kind == "unresolved":
        raise ValueError("temporary after-window authority is unresolved")
    if authority.effective_date != date.fromisoformat(end):
        raise ValueError("after-window authority date mismatch")
    rows = {row.id: row for row in draft.baseline_scope}
    affected = {
        identifier for item in draft.items for identifier in item.old_chunk_ids
    } | set(draft.source_only_canonical_ids)
    if set(authority.predecessor_ids) != affected:
        raise ValueError("after-window predecessor scope mismatch")
    if authority.kind == "scheduled_successor":
        successors = [
            row for row in rows.values() if row.supersedes_chunk_id in affected
        ]
        if (
            not successors
            or len(successors) != len(affected)
            or {row.id for row in successors} != set(authority.successor_ids)
            or any(
                row.validity_start_date != authority.effective_date
                for row in successors
            )
            or {row.supersedes_chunk_id for row in successors} != affected
        ):
            raise ValueError("after-window scheduled authority is ambiguous")
        return
    source = draft.submitted_source_text
    if (
        source is None
        or sha256(source.encode()).hexdigest() != authority.source_text_sha256
        or not authority.source_quote
        or source[authority.source_start : authority.source_end]
        != authority.source_quote
    ):
        raise ValueError("after-window source authority mismatch")
    from onyx.db.regulatory_annexes import normalize_annex_label

    labels = {
        normalize_annex_label(match.group())
        for match in re.finditer(
            r"(?:ek|annex)\s*[-:]?\s*\d+(?:/\d+)?",
            authority.source_quote,
            re.IGNORECASE,
        )
    }
    if labels != {normalize_annex_label(draft.annex_label)}:
        raise ValueError("after-window source does not identify the exact annex scope")
    when = authority.effective_date
    date_forms = (
        when.isoformat(),
        when.strftime("%d.%m.%Y"),
        when.strftime("%d/%m/%Y"),
    )
    months = (
        "ocak",
        "şubat",
        "mart",
        "nisan",
        "mayıs",
        "haziran",
        "temmuz",
        "ağustos",
        "eylül",
        "ekim",
        "kasım",
        "aralık",
    )
    normalized_quote = authority.source_quote.casefold()
    if not any(value in normalized_quote for value in date_forms) and not re.search(
        rf"\b0?{when.day}\s+{months[when.month - 1]}\s+{when.year}\b", normalized_quote
    ):
        raise ValueError(
            "after-window source does not support the exact transition date"
        )
    quote = authority.source_quote.lower().replace("ı", "i")
    if re.search(
        r"(?:uygulanmay|yürürlüğe girmez|shall not|will not|not restore|not resume)",
        quote,
    ):
        raise ValueError("after-window source contradicts restoration")
    restoration = bool(
        re.search(r"(?:önceki|eski|previous|prior)", quote)
        and re.search(
            r"(?:yeniden uygulan|uygulanmaya devam|yeniden yürür|restore|resume)", quote
        )
    )
    cessation = bool(
        re.search(
            r"(?:yürürlükten kalk|yürürlükten kaldir|uygulanmaz|sona er|cease|repeal)",
            quote,
        )
    )
    if (
        authority.successor_ids
        or restoration == cessation
        or (authority.kind == "restore_predecessor") != restoration
    ):
        raise ValueError("after-window source does not explicitly support the decision")


def exact_reusable_projection(
    wanted: "FrozenContextProjection",
    actual: "IndexedProjectionEvidence",
    *,
    predecessor_id: str | None,
) -> "FrozenPublicationProjection | None":
    """Only actual encoder evidence and validated identity/lineage permit reuse.

    The caller derives an injective predecessor mapping from reviewed staged
    items. Legacy source without encoder provenance is retained as evidence but
    cannot certify a vector by pairing it with newly generated OLD context.
    """
    import json

    projection = actual.frozen_projection
    if projection is None or actual.payload_sha256 is None:
        return None
    source = json.loads(projection.source_json)
    if source["regulatory_chunk_id"] not in (wanted.canonical_chunk_id, predecessor_id):
        return None
    if (
        list(projection.embedding_inputs) != wanted.embedding_texts
        or json.loads(projection.embedding_config_json) != wanted.embedding_config
        or wanted.embedding_input_sha256 != context_hash(wanted.embedding_texts)
        or wanted.embedding_config_sha256 != context_hash(wanted.embedding_config)
        or len(source["content_vector"]) != actual.index.vector_dimension
    ):
        return None
    return projection


def resolve_after_window_authority(draft: AnnexChangeDraft) -> AnnexChangeDraft:
    """Select exact retained evidence, never an arbitrary model-proposed identity."""
    from onyx.regulatory.amendments.annexes.models import AnnexAfterWindowAuthority

    end = draft.date_resolution.effective_end_date if draft.date_resolution else None
    if end is None:
        return draft.model_copy(update={"after_window_authority": None})
    affected = sorted(
        {identifier for item in draft.items for identifier in item.old_chunk_ids}
        | set(draft.source_only_canonical_ids)
    )
    successors = [
        row
        for row in draft.baseline_scope
        if row.supersedes_chunk_id in affected
        and row.validity_start_date == date.fromisoformat(end)
    ]
    source = draft.submitted_source_text or ""
    candidates: list[AnnexAfterWindowAuthority] = []
    if successors and {row.supersedes_chunk_id for row in successors} == set(affected):
        candidates.append(
            AnnexAfterWindowAuthority(
                effective_date=date.fromisoformat(end),
                source_text_sha256=sha256(source.encode()).hexdigest(),
                predecessor_ids=affected,
                kind="scheduled_successor",
                source_start=0,
                source_end=0,
                source_quote="",
                successor_ids=[row.id for row in successors],
            )
        )
    for match in re.finditer(r"[^\n;]+", source):
        for kind in ("restore_predecessor", "cessation"):
            authority = AnnexAfterWindowAuthority(
                effective_date=date.fromisoformat(end),
                source_text_sha256=sha256(source.encode()).hexdigest(),
                predecessor_ids=affected,
                kind=kind,
                source_start=match.start(),
                source_end=match.end(),
                source_quote=match.group(),
                successor_ids=[],
            )
            candidate = draft.model_copy(update={"after_window_authority": authority})
            try:
                validate_after_window_authority(candidate)
            except ValueError:
                continue
            candidates.append(authority)
    kinds = {item.kind for item in candidates}
    if len(kinds) != 1:
        raise ValueError("temporary after-window authority is missing or contradictory")
    result = draft.model_copy(update={"after_window_authority": candidates[0]})
    validate_after_window_authority(result)
    return result
