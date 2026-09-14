from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from hashlib import sha256
from typing import TypeVar
from uuid import UUID


@dataclass(frozen=True, slots=True)
class LabelingChunkView:
    id: str
    position: int
    heading_path: tuple[str, ...]
    text: str


@dataclass(frozen=True, slots=True)
class DerivedSourceResolution:
    source_ids: tuple[str, ...]
    resolution: str
    unresolved_reason: str | None = None


_RequestT = TypeVar("_RequestT")


def shard_by_count_and_bytes(
    requests: Sequence[tuple[str, _RequestT]],
    *,
    item_limit: int,
    byte_limit: int,
    size_of: Callable[[_RequestT], int],
) -> list[list[tuple[str, _RequestT]]]:
    if item_limit < 1 or byte_limit < 1:
        raise ValueError("labeling shard limits must be positive")
    shards: list[list[tuple[str, _RequestT]]] = []
    current: list[tuple[str, _RequestT]] = []
    current_bytes = 0
    for item in requests:
        request_bytes = size_of(item[1])
        if request_bytes > byte_limit:
            raise ValueError("labeling request exceeds the shard byte limit")
        if current and (
            len(current) >= item_limit or current_bytes + request_bytes > byte_limit
        ):
            shards.append(current)
            current = []
            current_bytes = 0
        current.append(item)
        current_bytes += request_bytes
    if current:
        shards.append(current)
    return shards


def _chunk_block(row: LabelingChunkView) -> str:
    heading = " > ".join(row.heading_path)
    return (
        f"[Canonical chunk: {row.id}; position: {row.position}]\n"
        f"Heading path: {heading}\n{row.text}"
    )


def _utf8_prefix(value: str, byte_limit: int) -> str:
    return value.encode("utf-8")[:byte_limit].decode("utf-8", errors="ignore")


def bounded_document_context(
    rows: Sequence[LabelingChunkView], *, target_id: str, max_utf8_bytes: int
) -> str:
    if max_utf8_bytes < 1:
        raise ValueError("labeling context byte limit must be positive")
    ordered = sorted(rows, key=lambda row: (row.position, row.id))
    target_index = next(
        (index for index, row in enumerate(ordered) if row.id == target_id), None
    )
    if target_index is None:
        raise ValueError("labeling target is not present in its document snapshot")
    first_heading = ordered[0].heading_path[-1] if ordered[0].heading_path else ""
    last_heading = ordered[-1].heading_path[-1] if ordered[-1].heading_path else ""
    prefix = f"[Document boundaries: {first_heading} | {last_heading}]\n"
    target_block = _chunk_block(ordered[target_index])
    minimum = f"{prefix}{target_block}"
    truncation_marker = (
        "\n[Context truncated; target.text is supplied separately in full.]"
    )
    if len(minimum.encode("utf-8")) >= max_utf8_bytes:
        available = max(0, max_utf8_bytes - len(truncation_marker.encode("utf-8")))
        return _utf8_prefix(minimum, available) + truncation_marker

    chosen = {target_index}
    left = target_index - 1
    right = target_index + 1
    while left >= 0 or right < len(ordered):
        candidate_indexes: list[int] = []
        if left >= 0:
            candidate_indexes.append(left)
        if right < len(ordered):
            candidate_indexes.append(right)
        added = False
        for candidate in candidate_indexes:
            trial = sorted([*chosen, candidate])
            value = prefix + "\n\n".join(
                _chunk_block(ordered[index]) for index in trial
            )
            if len(value.encode("utf-8")) <= max_utf8_bytes:
                chosen.add(candidate)
                added = True
        if not added:
            break
        left -= 1
        right += 1
    context = prefix + "\n\n".join(
        _chunk_block(ordered[index]) for index in sorted(chosen)
    )
    if len(chosen) != len(ordered):
        available = max(0, max_utf8_bytes - len(truncation_marker.encode("utf-8")))
        context = _utf8_prefix(context, available) + truncation_marker
    return context


def resolve_derived_sources(
    *,
    derived_text: str,
    explicit_dependencies: Sequence[str],
    atomics: Sequence[LabelingChunkView],
) -> DerivedSourceResolution:
    atomic_ids = {row.id for row in atomics}
    if explicit_dependencies:
        dependencies = tuple(dict.fromkeys(explicit_dependencies))
        if any(dependency not in atomic_ids for dependency in dependencies):
            return DerivedSourceResolution(
                (), "unresolved", "missing_explicit_dependency"
            )
        return DerivedSourceResolution(
            dependencies,
            "lineage",
        )

    ids_by_text: dict[str, list[str]] = {}
    for row in atomics:
        if row.text:
            ids_by_text.setdefault(row.text, []).append(row.id)
    contained = {
        text: identifiers
        for text, identifiers in ids_by_text.items()
        if len(text.strip()) >= 32 and text in derived_text
    }
    if any(len(identifiers) != 1 for identifiers in contained.values()):
        return DerivedSourceResolution((), "unresolved", "ambiguous_legacy_containment")
    resolved = tuple(sorted(identifiers[0] for identifiers in contained.values()))
    if not resolved:
        return DerivedSourceResolution((), "unresolved", "no_legacy_containment")
    return DerivedSourceResolution(resolved, "legacy_containment")


def labeling_submission_key_from_hashes(
    request_hashes: Sequence[str], *, tenant_id: str, run_id: UUID, ordinal: int
) -> str:
    """Retain the provider submission identity without retaining full prompts."""
    if (
        not request_hashes
        or not tenant_id.strip()
        or ordinal < 0
        or len(set(request_hashes)) != len(request_hashes)
        or any(
            len(value) != 64 or any(c not in "0123456789abcdef" for c in value)
            for value in request_hashes
        )
    ):
        raise ValueError("The labeling submission identity is invalid")
    identity = json.dumps(
        {
            "job_id": str(run_id),
            "output_prefix": f"regulatory-labeling/{run_id}/{ordinal}",
            "request_hashes": sorted(request_hashes),
            "submission_attempt": 1,
            "tenant_id": tenant_id,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return f"regulatory-labeling-{sha256(identity.encode()).hexdigest()}"
