"""Keep a heading replacement and a subordinate addition in one reviewed change."""

import re
from typing import Any

from onyx.regulatory.amendments.draft_integrity import (
    DraftIntegrityError,
    explicit_replacement_body,
    validate_complete_scope_replacement,
)
from onyx.regulatory.amendments.models import ProposalChunkChange, ProposalDraft
from onyx.regulatory.amendments.structural_target import article_identity

_HEADING_CHANGE = re.compile(
    r'başlığı\s+["“](?P<title>[^"”]+)["”]\s+şeklinde\s+değiştirilmiş(?:tir)?\s+ve\s+aynı\s+maddeye',
    re.IGNORECASE,
)
_ARTICLE_HEADING_CHANGE = re.compile(
    r'başlığı\s+["“](?P<title>[^"”]+)["”]\s+şeklinde\s+değiştirilmiş(?:tir)?(?=[\s.;:]|$)',
    re.IGNORECASE,
)
_HEADING = re.compile(
    r"^(?P<marker>(?:(?:EK|GEÇİCİ|MÜKERRER)\s+)?MADDE\s+\d+[A-Z]?)(?:\s*[-–—:]\s*(?P<title>.+))?$",
    re.IGNORECASE,
)


def compound_heading_title(instruction_text: str) -> str | None:
    match = _HEADING_CHANGE.search(instruction_text)
    return match.group("title").strip() if match else None


def article_heading_title(instruction_text: str) -> str | None:
    match = _ARTICLE_HEADING_CHANGE.search(instruction_text)
    return match.group("title").strip() if match else None


def is_article_heading_only(instruction_text: str) -> bool:
    match = _ARTICLE_HEADING_CHANGE.search(instruction_text)
    return match is not None and not instruction_text[match.end() :].strip(" .;:\n\t")


def reject_compound_descendant_replacement(
    instruction_texts: list[str],
    *,
    has_descendants: bool,
) -> None:
    if has_descendants and any(
        explicit_replacement_body(text) for text in instruction_texts
    ):
        raise DraftIntegrityError(
            "A complete replacement with a heading change requires one descendant-consumption review; "
            "old descendants cannot be copied into the new parent."
        )


def apply_article_heading_change(
    draft: dict[str, Any], article_no: str, title: str
) -> dict[str, Any]:
    path = list(draft.get("heading_path") or [])
    old_title: str | None = None
    for index, heading in enumerate(path):
        match = _HEADING.fullmatch(heading)
        if match and article_identity(heading) == article_no:
            old_title = match.group("title")
            path[index] = f"{match.group('marker')} - {title}"
            break
    else:
        raise DraftIntegrityError("Heading change has no verified article heading")
    text = draft["text"]
    if old_title:
        # Only a standalone heading line; never rewrite a citation in prose.
        text = re.sub(
            rf"(?m)^(\s*(?:\*\*|__|#{{1,6}}\s*)?){re.escape(old_title)}((?:\*\*|__)?\s*)$",
            lambda match: f"{match.group(1)}{title}{match.group(2)}",
            text,
        )
    return {
        **draft,
        "text": text,
        "heading_path": path,
        "metadata": {
            **draft.get("metadata", {}),
            "article_title": title,
            "heading_path": path,
        },
    }


def attach_heading_changes(
    proposal: ProposalDraft,
    *,
    snapshots: list[dict[str, Any]],
    article_no: str,
    title: str,
) -> ProposalDraft:
    if not snapshots:
        raise DraftIntegrityError(
            "Heading change requires the complete verified parent scope"
        )
    originals = proposal.chunk_changes or [
        ProposalChunkChange(
            old_chunk_id=proposal.old_chunk_id,
            old_chunk_snapshot=proposal.old_chunk_snapshot,
            new_chunk_draft=proposal.new_chunk_draft,
            instruction_indices=proposal.instruction_indices,
            instruction_texts=proposal.instruction_texts,
            match_confidence=proposal.match_confidence,
            match_rationale=proposal.match_rationale,
            date_rationale=proposal.date_rationale,
        )
    ]
    snapshot_by_id = {snapshot["id"]: snapshot for snapshot in snapshots}
    target_ids = [
        change.old_chunk_id for change in originals if change.old_chunk_id is not None
    ]
    if len(set(target_ids)) != len(target_ids) or not set(target_ids).issubset(
        snapshot_by_id
    ):
        raise DraftIntegrityError("Heading group has duplicate or out-of-scope targets")
    heading_operations = [
        (index, text)
        for index, text in zip(
            proposal.instruction_indices, proposal.instruction_texts, strict=True
        )
        if article_heading_title(text) == title
    ]
    if not heading_operations:
        raise DraftIntegrityError("Heading change has no explicit source operation")
    scope = {"article_no": article_no, "chunk_ids": list(snapshot_by_id)}
    heading_change = {"article_no": article_no, "title": title}
    primary = proposal.new_chunk_draft
    dates = (primary.get("effective_start_date"), primary.get("effective_end_date"))
    consumed_ids: set[str] = set()
    for change in originals:
        descendants = change.old_chunk_snapshot.get("descendant_snapshots") or []
        if not descendants or not any(
            explicit_replacement_body(text) for text in change.instruction_texts
        ):
            continue
        validate_complete_scope_replacement(change.instruction_texts)
        for descendant in descendants:
            identifier = descendant.get("id")
            if (
                identifier not in snapshot_by_id
                or identifier in consumed_ids
                or identifier in target_ids
                or any(
                    descendant.get(key) != value
                    for key, value in snapshot_by_id[identifier].items()
                )
            ):
                raise DraftIntegrityError(
                    "Replacement descendants conflict with the reviewed heading scope"
                )
            consumed_ids.add(identifier)
    changes = []
    for change in originals:
        if (
            change.new_chunk_draft.get("effective_start_date"),
            change.new_chunk_draft.get("effective_end_date"),
        ) != dates:
            raise DraftIntegrityError(
                "Dependent heading operations have different effective intervals"
            )
        if str(change.new_chunk_draft["user_file_id"]) != str(primary["user_file_id"]):
            raise DraftIntegrityError("Heading change escaped the verified source")
        for instruction_text in change.instruction_texts:
            body = re.search(
                r'eklenmiştir\s*[:.]?\s*["“](.+)["”]\s*$',
                instruction_text,
                re.IGNORECASE | re.DOTALL,
            )
            if body and " ".join(body.group(1).split()) not in " ".join(
                change.new_chunk_draft["text"].split()
            ):
                raise DraftIntegrityError(
                    "Combined amendment draft omitted supplied addition text"
                )
        snapshot = dict(change.old_chunk_snapshot)
        if change.old_chunk_id is not None:
            if any(
                snapshot.get(key) != value
                for key, value in snapshot_by_id[change.old_chunk_id].items()
            ):
                raise DraftIntegrityError("Heading group canonical snapshot changed")
            snapshot["heading_change"] = heading_change
        changes.append(
            change.model_copy(
                update={
                    "old_chunk_snapshot": snapshot,
                    "new_chunk_draft": apply_article_heading_change(
                        change.new_chunk_draft, article_no, title
                    ),
                }
            )
        )
    for snapshot in snapshots:
        if str(snapshot["user_file_id"]) != str(primary["user_file_id"]):
            raise DraftIntegrityError("Heading change escaped the verified source")
        if snapshot["id"] in target_ids or snapshot["id"] in consumed_ids:
            continue
        fields = {
            key: snapshot.get(key)
            for key in (
                "user_file_id",
                "position",
                "text",
                "chunk_type",
                "heading_path",
                "metadata",
            )
        }
        fields.update(
            effective_start_date=dates[0],
            effective_end_date=dates[1],
        )
        changed = apply_article_heading_change(fields, article_no, title)
        if changed != fields:
            changes.append(
                ProposalChunkChange(
                    old_chunk_id=snapshot["id"],
                    old_chunk_snapshot={
                        **snapshot,
                        "heading_change": heading_change,
                    },
                    new_chunk_draft=changed,
                    instruction_indices=[index for index, _ in heading_operations],
                    instruction_texts=[text for _, text in heading_operations],
                    match_confidence=proposal.match_confidence,
                    match_rationale=proposal.match_rationale,
                    date_rationale=proposal.date_rationale,
                )
            )
    old_snapshot = {**changes[0].old_chunk_snapshot, "heading_change_scope": scope}
    changes[0] = changes[0].model_copy(update={"old_chunk_snapshot": old_snapshot})
    return proposal.model_copy(
        update={
            "old_chunk_snapshot": old_snapshot,
            "new_chunk_draft": changes[0].new_chunk_draft,
            "chunk_changes": changes,
        }
    )
