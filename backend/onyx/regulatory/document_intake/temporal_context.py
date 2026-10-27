"""Date-qualified derived reading rows without changing canonical legal records."""

from datetime import date

from onyx.db.models import RegulatoryChunk
from onyx.regulatory.amendments.annexes.context_dependencies import (
    canonical_dependency_ids,
)
from onyx.regulatory.document_intake.chunking_profile import (
    SUPPORTED_INTAKE_CHUNKING_VERSIONS,
)


def intake_temporal_context_rows(rows: list[RegulatoryChunk]) -> list[RegulatoryChunk]:
    """A complete derived text is visible only while every legal source is visible."""
    versions: list[object] = []
    for row in rows:
        metadata = getattr(row, "chunk_metadata", None)
        if isinstance(metadata, dict):
            version = metadata.get("intake_chunking_version")
            if version is not None:
                versions.append(version)
    if not versions:
        return rows
    if (
        len({row.id for row in rows}) != len(rows)
        or len({row.user_file_id for row in rows}) != 1
        or any(
            not isinstance(version, str)
            or version not in SUPPORTED_INTAKE_CHUNKING_VERSIONS
            for version in versions
        )
        or len({version for version in versions if isinstance(version, str)}) != 1
    ):
        raise ValueError("Intake temporal context has inconsistent canonical scope")
    by_id = {row.id: row for row in rows}
    derived = {
        row.id
        for row in rows
        if row.chunk_metadata.get("chunk_variant") == "hierarchical_aggregate"
        or row.chunk_metadata.get("bound_to_regulatory_chunk_id") is not None
    }
    dependencies = {
        identifier: canonical_dependency_ids(by_id[identifier])
        for identifier in derived
    }
    if any(
        not sources or any(source not in by_id for source in sources)
        for sources in dependencies.values()
    ):
        raise ValueError("Intake derived context has unavailable legal sources")
    windows: dict[str, tuple[date | None, date | None] | None] = {
        row.id: (row.validity_start_date, row.validity_end_date)
        for row in rows
        if row.id not in derived
    }
    pending = set(derived)
    replacements: dict[str, RegulatoryChunk] = {}
    while pending:
        ready = sorted(
            identifier
            for identifier in pending
            if not pending.intersection(dependencies[identifier])
        )
        if not ready:
            raise ValueError("Intake derived context has cyclic legal sources")
        for identifier in ready:
            row = by_id[identifier]
            source_windows = [windows[source] for source in dependencies[identifier]]
            if any(window is None for window in source_windows):
                windows[identifier] = None
                pending.remove(identifier)
                continue
            intervals = [
                (row.validity_start_date, row.validity_end_date),
                *(window for window in source_windows if window is not None),
            ]
            starts = [start for start, _end in intervals if start is not None]
            ends = [end for _start, end in intervals if end is not None]
            start, end = max(starts) if starts else None, min(ends) if ends else None
            if start is not None and end is not None and start >= end:
                windows[identifier] = None
                pending.remove(identifier)
                continue
            windows[identifier] = (start, end)
            replacements[identifier] = RegulatoryChunk(
                id=row.id,
                user_file_id=row.user_file_id,
                text=row.text,
                position=row.position,
                projection_ordinal=row.projection_ordinal,
                chunk_type=row.chunk_type,
                heading_path=list(row.heading_path),
                chunk_metadata=dict(row.chunk_metadata),
                source=row.source,
                status=row.status,
                validity_start_date=start,
                validity_end_date=end,
                supersedes_chunk_id=row.supersedes_chunk_id,
                superseded_by_chunk_id=row.superseded_by_chunk_id,
            )
            pending.remove(identifier)
    return [
        replacements[row.id] if row.id in derived else row
        for row in rows
        if row.id not in derived or windows[row.id] is not None
    ]
