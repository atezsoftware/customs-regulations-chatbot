"""Keep a heading replacement and a subordinate addition in one reviewed change."""

import re
from typing import Any

from onyx.regulatory.amendments.draft_integrity import DraftIntegrityError
from onyx.regulatory.amendments.models import ProposalChunkChange, ProposalDraft
from onyx.regulatory.amendments.structural_target import article_identity

_HEADING_CHANGE = re.compile(
    r'başlığı\s+["“](?P<title>[^"”]+)["”]\s+şeklinde\s+değiştirilmiş(?:tir)?\s+ve\s+aynı\s+maddeye',
    re.IGNORECASE,
)
_HEADING = re.compile(
    r"^(?P<marker>(?:(?:EK|GEÇİCİ|MÜKERRER)\s+)?MADDE\s+\d+[A-Z]?)(?:\s*[-–—:]\s*(?P<title>.+))?$",
    re.IGNORECASE,
)


def compound_heading_title(instruction_text: str) -> str | None:
    match = _HEADING_CHANGE.search(instruction_text)
    return match.group("title").strip() if match else None


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
    if not snapshots or proposal.old_chunk_id is not None:
        raise DraftIntegrityError(
            "Combined heading/addition requires the complete verified parent scope"
        )
    for instruction_text in proposal.instruction_texts:
        body = re.search(
            r'eklenmiştir\s*[:.]?\s*["“](.+)["”]\s*$',
            instruction_text,
            re.IGNORECASE | re.DOTALL,
        )
        if body and " ".join(body.group(1).split()) not in " ".join(
            proposal.new_chunk_draft["text"].split()
        ):
            raise DraftIntegrityError(
                "Combined amendment draft omitted supplied addition text"
            )
    new_draft = apply_article_heading_change(
        proposal.new_chunk_draft, article_no, title
    )
    old_snapshot = {
        **proposal.old_chunk_snapshot,
        "heading_change_scope": {
            "article_no": article_no,
            "chunk_ids": [snapshot["id"] for snapshot in snapshots],
        },
    }
    changes = [
        ProposalChunkChange(
            old_chunk_id=None,
            old_chunk_snapshot=old_snapshot,
            new_chunk_draft=new_draft,
            instruction_indices=proposal.instruction_indices,
            instruction_texts=proposal.instruction_texts,
            match_confidence=proposal.match_confidence,
            match_rationale=proposal.match_rationale,
            date_rationale=proposal.date_rationale,
        )
    ]
    for snapshot in snapshots:
        if str(snapshot["user_file_id"]) != str(new_draft["user_file_id"]):
            raise DraftIntegrityError("Heading change escaped the verified source")
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
            effective_start_date=new_draft.get("effective_start_date"),
            effective_end_date=new_draft.get("effective_end_date"),
        )
        changed = apply_article_heading_change(fields, article_no, title)
        if changed != fields:
            changes.append(
                ProposalChunkChange(
                    old_chunk_id=snapshot["id"],
                    old_chunk_snapshot={
                        **snapshot,
                        "heading_change": {"article_no": article_no, "title": title},
                    },
                    new_chunk_draft=changed,
                    instruction_indices=proposal.instruction_indices,
                    instruction_texts=proposal.instruction_texts,
                    match_confidence=proposal.match_confidence,
                    match_rationale=proposal.match_rationale,
                    date_rationale=proposal.date_rationale,
                )
            )
    return proposal.model_copy(
        update={
            "old_chunk_snapshot": old_snapshot,
            "new_chunk_draft": new_draft,
            "chunk_changes": changes,
        }
    )
