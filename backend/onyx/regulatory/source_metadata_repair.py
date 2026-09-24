"""Conservative source proofs for legacy image metadata; no text/vector rewriting."""

import base64
import binascii
import hashlib
import re
from collections import Counter

from pydantic import BaseModel, ConfigDict

from onyx.regulatory.amendments.annexes.models import AnnexCanonicalSnapshot
from onyx.regulatory.amendments.annexes.selective_impact import (
    image_membership_is_valid,
    recover_source_membership,
    recovered_source_row,
    validate_canonical_source_integrity,
)

_EMBEDDED_IMAGE = re.compile(r"!\[[^\]]*\]\(data:image/[^;]+;base64,([^)]*)\)")


def _normalized_text(value: str) -> str:
    return " ".join(value.split())


class MarkdownImageParentProof(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    parent_id: str
    source_sha256: str
    image_sha256: str
    image_order: int
    image_offset: int
    normalized_source_start: int
    normalized_source_end: int
    preceding_anchor_id: str | None
    following_anchor_id: str | None


def recover_markdown_image_parents(
    rows: list[AnnexCanonicalSnapshot],
    markdown: bytes,
    asset_sha256: dict[str, str],
) -> dict[str, MarkdownImageParentProof]:
    """Require exact asset bytes and a unique, ordered source occurrence of its body."""
    raw = markdown.decode("utf-8")
    images = list(_EMBEDDED_IMAGE.finditer(raw))
    source = _normalized_text(_EMBEDDED_IMAGE.sub("", raw))
    atomic = sorted(
        (
            row
            for row in rows
            if row.metadata.get("chunk_variant") != "hierarchical_aggregate"
            and not row.metadata.get("bound_to_regulatory_chunk_id")
            and row.chunk_type != "image"
        ),
        key=lambda row: (row.position, row.id),
    )
    bodies = {row.id: _normalized_text(row.text) for row in atomic}
    body_counts = Counter(bodies.values())
    occurrences = {
        row.id: [
            (match.start(), match.end())
            for match in re.finditer(re.escape(bodies[row.id]), source)
        ]
        if bodies[row.id]
        else []
        for row in atomic
    }
    anchors = [
        row
        for row in atomic
        if len(occurrences[row.id]) == 1 and body_counts[bodies[row.id]] == 1
    ]
    result: dict[str, MarkdownImageParentProof] = {}
    for image in rows:
        if not image.metadata.get("bound_to_regulatory_chunk_id"):
            continue
        order = image.metadata.get("image_order")
        asset = image.metadata.get("image_file_id")
        if (
            type(order) is not int
            or not 1 <= order <= len(images)
            or not isinstance(asset, str)
        ):
            continue
        match = images[order - 1]
        try:
            image_bytes = base64.b64decode(
                "".join(match.group(1).split()), validate=True
            )
        except (ValueError, binascii.Error):
            continue
        digest = hashlib.sha256(image_bytes).hexdigest()
        if asset_sha256.get(asset) != digest:
            continue
        offset = len(_normalized_text(_EMBEDDED_IMAGE.sub("", raw[: match.start()])))
        candidates: list[MarkdownImageParentProof] = []
        for parent in atomic:
            if not image_membership_is_valid(image, parent) or image.text not in (
                parent.text,
                parent.text
                + "\n\n[Görsel: "
                + str(image.metadata.get("image_alt") or "")
                + "]",
            ):
                continue
            before = next(
                (row for row in reversed(anchors) if row.position < parent.position),
                None,
            )
            after = next(
                (row for row in anchors if row.position > parent.position), None
            )
            lower = occurrences[before.id][0][1] if before else 0
            upper = occurrences[after.id][0][0] if after else len(source)
            spans = [
                (start, end)
                for start, end in occurrences[parent.id]
                if lower <= start and end <= upper
            ]
            if len(spans) != 1:
                continue
            start, end = spans[0]
            if start - 1 <= offset <= end:
                candidates.append(
                    MarkdownImageParentProof(
                        parent_id=parent.id,
                        source_sha256=hashlib.sha256(markdown).hexdigest(),
                        image_sha256=digest,
                        image_order=order,
                        image_offset=offset,
                        normalized_source_start=start,
                        normalized_source_end=end,
                        preceding_anchor_id=before.id if before else None,
                        following_anchor_id=after.id if after else None,
                    )
                )
        if len(candidates) == 1:
            result[image.id] = candidates[0]
    return result


def repair_canonical_source_metadata(
    rows: list[AnnexCanonicalSnapshot],
    *,
    markdown: bytes | None = None,
    asset_sha256: dict[str, str] | None = None,
    indexed_headings: dict[str, list[str]] | None = None,
) -> list[AnnexCanonicalSnapshot]:
    """Reconstruct proven metadata while preserving every source byte and identity."""
    from onyx.regulatory.chunker import RegulatoryChunker

    recovered = recover_source_membership(rows)
    after = [
        recovered_source_row(row, recovered[row.id]) if row.id in recovered else row
        for row in rows
    ]
    headings = indexed_headings or {}
    mismatched = {
        row.id
        for row in after
        if row.id in headings and row.heading_path != headings[row.id]
    }
    if mismatched:
        if markdown is None:
            raise ValueError("canonical heading repair requires original source")
        parsed = RegulatoryChunker().chunk_text(markdown.decode("utf-8")).chunks
        for index, row in enumerate(after):
            if row.id not in mismatched:
                continue
            matches = [chunk for chunk in parsed if chunk.text == row.text]
            if (
                len(matches) != 1
                or list(matches[0].metadata.heading_path) != headings[row.id]
            ):
                raise ValueError(
                    "original source does not uniquely prove indexed heading: " + row.id
                )
            after[index] = row.model_copy(
                update={
                    "heading_path": headings[row.id],
                    "metadata": {**row.metadata, "heading_path": headings[row.id]},
                }
            )
    if markdown is not None:
        proofs = recover_markdown_image_parents(after, markdown, asset_sha256 or {})
        after = [
            recovered_source_row(row, [proofs[row.id].parent_id])
            if row.id in proofs
            else row
            for row in after
        ]
    validate_canonical_source_integrity(after)
    return after
