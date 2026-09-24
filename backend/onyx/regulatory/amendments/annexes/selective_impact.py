"""Deterministic dependency traversal over frozen canonical and context sources."""

from collections import defaultdict, deque
from uuid import UUID

from onyx.regulatory.amendments.annexes.models import (
    AnnexCanonicalSnapshot,
    AnnexChangeItemDraft,
    AnnexDependencyImpact,
    PreparedContextView,
)


def source_ids(row: AnnexCanonicalSnapshot) -> list[str]:
    values = row.metadata.get("source_regulatory_chunk_ids", [])
    if not isinstance(values, list) or any(not isinstance(v, str) for v in values):
        raise ValueError("invalid canonical source membership")
    result = [v for v in values if isinstance(v, str)]
    bound = row.metadata.get("bound_to_regulatory_chunk_id")
    if bound is not None:
        if not isinstance(bound, str):
            raise ValueError("invalid image source membership")
        result.append(bound)
    return list(dict.fromkeys(result))


def dependency_impact(
    *,
    before: list[AnnexCanonicalSnapshot],
    after: list[AnnexCanonicalSnapshot],
    items: list[AnnexChangeItemDraft],
    contexts: PreparedContextView,
    contextual_ids: set[str],
) -> AnnexDependencyImpact:
    """No provider calls. Missing or global context lineage is never guessed local."""
    old = {row.id: row for row in before}
    new = {row.id: row for row in after}
    if len(old) != len(before) or len(new) != len(after):
        raise ValueError("duplicate canonical identity")
    if len({row.user_file_id for row in [*before, *after]}) > 1:
        raise ValueError("dependency graph crosses file scope")
    old_ids = {identifier for item in items for identifier in item.old_chunk_ids}
    new_ids = {row.id for item in items for row in item.new_chunks}
    if not old_ids <= old.keys() or not new_ids <= new.keys():
        raise ValueError("changed canonical identity outside source scope")
    reverse: dict[str, set[str]] = defaultdict(set)
    unresolved: dict[str, list[str]] = defaultdict(list)
    reasons: dict[str, list[str]] = defaultdict(list)
    for view in (old, new):
        for row in view.values():
            sources = source_ids(row)
            if (
                row.metadata.get("chunk_variant") == "hierarchical_aggregate"
                and not sources
            ):
                unresolved[row.id].append("aggregate source membership missing")
            for source in sources:
                if source not in view:
                    unresolved[row.id].append(f"source missing:{source}")
                elif source != row.id:
                    reverse[source].add(row.id)
    snapshots = {snapshot.sha256: snapshot for snapshot in contexts.snapshots}
    for identifier in contextual_ids:
        projections = [
            p for p in contexts.projections if p.canonical_chunk_id == identifier
        ]
        if not projections:
            unresolved[identifier].append("context source ranges unavailable")
        for projection in projections:
            snapshot = snapshots.get(projection.source_snapshot_sha256)
            if snapshot is None:
                unresolved[identifier].append("context source ranges unavailable")
                continue
            for span in snapshot.ordered_ranges:
                if span.canonical_chunk_id not in old:
                    unresolved[identifier].append(
                        "context source outside canonical snapshot"
                    )
                else:
                    reverse[span.canonical_chunk_id].add(identifier)
    seeds = old_ids | new_ids
    # Structural changes and window membership changes are observable without an LLM.
    for identifier in old.keys() & new.keys():
        left, right = old[identifier], new[identifier]
        if (left.text, left.heading_path, left.metadata) != (
            right.text,
            right.heading_path,
            right.metadata,
        ):
            seeds.add(identifier)
            reasons[identifier].append("source representation changed")
    affected = set(seeds)
    queue = deque(sorted(seeds))
    while queue:
        source = queue.popleft()
        for consumer in sorted(reverse[source]):
            reasons[consumer].append(f"source:{source}")
            if consumer not in affected:
                affected.add(consumer)
                queue.append(consumer)
    for identifier in old_ids | new_ids:
        reasons[identifier].append("reviewed canonical change")
    return AnnexDependencyImpact(
        changed_old_ids=sorted(old_ids),
        changed_new_ids=sorted(new_ids),
        affected_ids=sorted(affected),
        unchanged_ids=sorted((new.keys() & old.keys()) - affected - unresolved.keys()),
        unresolved={
            key: sorted(set(value)) for key, value in sorted(unresolved.items())
        },
        reasons={key: sorted(set(value)) for key, value in sorted(reasons.items())},
    )


def local_source_rows(
    rows: list[AnnexCanonicalSnapshot], target_id: str
) -> list[AnnexCanonicalSnapshot]:
    """The same explicit source membership drives generation and recorded ranges."""
    by_id = {row.id: row for row in rows}
    selected: set[str] = set()
    pending = [target_id]
    while pending:
        identifier = pending.pop()
        if identifier in selected:
            continue
        row = by_id.get(identifier)
        if row is None:
            raise ValueError("local context source missing")
        selected.add(identifier)
        pending.extend(source_ids(row))
    return [row for row in rows if row.id in selected]


def review_units(items: list[AnnexChangeItemDraft]) -> list[list[int]]:
    """Keep shared source elements and insertions anchored to removals atomic."""
    parents = list(range(len(items)))

    def root(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    owners: dict[tuple[str, str | int], int] = {}
    for index, item in enumerate(items):
        keys: list[tuple[str, str | int]] = [
            *(("old", identifier) for identifier in item.old_chunk_ids),
            *(("old-element", position) for position in item.old_positions),
            *(("new-element", position) for position in item.new_positions),
        ]
        if item.insertion_after_chunk_id:
            keys.append(("old", item.insertion_after_chunk_id))
        for key in keys:
            if key in owners:
                parents[root(index)] = root(owners[key])
            else:
                owners[key] = index
    groups: dict[int, list[int]] = {}
    for index in range(len(items)):
        groups.setdefault(root(index), []).append(index)
    return sorted(groups.values(), key=lambda group: group[0])


def source_window_covers(
    source: AnnexCanonicalSnapshot, dependent: AnnexCanonicalSnapshot
) -> bool:
    """Unchanged sources can support a narrower historical derived window."""
    start, end = dependent.validity_start_date, dependent.validity_end_date
    if start is not None and end is not None and start >= end:
        # Empty archived versions retain their original exact relationship.
        return (
            start == end
            and source.validity_start_date == start
            and source.validity_end_date == end
        )
    return (
        source.validity_start_date is None
        or start is not None
        and source.validity_start_date <= start
    ) and (
        source.validity_end_date is None
        or end is not None
        and end <= source.validity_end_date
    )


def aggregate_membership_is_valid(
    aggregate: AnnexCanonicalSnapshot, rows: dict[str, AnnexCanonicalSnapshot]
) -> bool:
    from onyx.regulatory.chunker import (
        hierarchical_aggregate_root_label,
        hierarchical_aggregate_text,
    )

    members = source_ids(aggregate)
    root = aggregate.metadata.get("hierarchy_root_path")
    if (
        not members
        or aggregate.id in members
        or not isinstance(root, list)
        or not root
        or not isinstance(root[-1], str)
    ):
        return False
    selected = [rows.get(identifier) for identifier in members]
    if any(
        row is None
        or row.user_file_id != aggregate.user_file_id
        or not source_window_covers(row, aggregate)
        for row in selected
    ):
        return False
    return (
        hierarchical_aggregate_text(
            hierarchical_aggregate_root_label(aggregate.metadata, aggregate.text),
            [row.text for row in selected if row is not None],
        )
        == aggregate.text
    )


def recover_source_membership(
    rows: list[AnnexCanonicalSnapshot],
) -> dict[str, list[str]]:
    """Recover only exact, unique membership within the same file and legal version."""
    from onyx.regulatory.chunker import (
        hierarchical_aggregate_root_label,
        hierarchical_aggregate_text,
    )

    recovered: dict[str, list[str]] = {}
    by_id = {row.id: row for row in rows}
    for image in rows:
        parent = image.metadata.get("bound_to_regulatory_chunk_id")
        if not isinstance(parent, str) or parent in by_id:
            continue
        candidates = [
            row
            for row in rows
            if row.text.strip()
            and row.chunk_type not in {"image", "hierarchical_aggregate"}
            and image_membership_is_valid(image, row)
            and image.text
            in (
                row.text,
                row.text
                + "\n\n[Görsel: "
                + str(image.metadata.get("image_alt") or "")
                + "]",
            )
        ]
        recorded = [
            row
            for row in candidates
            if isinstance(split := row.metadata.get("oversized_split"), dict)
            and split.get("source_chunk_id") == parent
        ]
        candidates = recorded or [
            row for row in candidates if row.metadata.get("oversized_split") is None
        ]
        if len(candidates) == 1:
            recovered[image.id] = [candidates[0].id]
    for aggregate in rows:
        if aggregate.metadata.get(
            "chunk_variant"
        ) != "hierarchical_aggregate" or aggregate_membership_is_valid(
            aggregate, by_id
        ):
            continue
        root = aggregate.metadata.get("hierarchy_root_path")
        if (
            not isinstance(root, list)
            or not root
            or not all(isinstance(part, str) for part in root)
        ):
            continue
        root_label = hierarchical_aggregate_root_label(
            aggregate.metadata, aggregate.text
        )
        root_paths = [root, [*root[:-1], root_label]]
        candidates = sorted(
            (
                row
                for row in rows
                if row.id != aggregate.id
                and row.user_file_id == aggregate.user_file_id
                and source_window_covers(row, aggregate)
                and row.metadata.get("chunk_variant") != "hierarchical_aggregate"
                and not row.metadata.get("bound_to_regulatory_chunk_id")
                and row.heading_path[: len(root)] in root_paths
            ),
            key=lambda row: (row.position, row.id),
        )
        orders = aggregate.metadata.get("source_chunk_orders")
        if isinstance(orders, list) and orders:
            from onyx.db.regulatory_chunks import make_regulatory_chunk_id

            recorded_ids = source_ids(aggregate)
            if recorded_ids and len(recorded_ids) != len(orders):
                continue
            identity_proven = len(recorded_ids) == len(orders)
            selected = []
            for offset, order in enumerate(orders):
                if not isinstance(order, int) or isinstance(order, bool) or order < 0:
                    break
                matches = [
                    row
                    for row in candidates
                    if (
                        (
                            row.id == recorded_ids[offset]
                            or make_regulatory_chunk_id(
                                UUID(row.user_file_id), order, row.text
                            )
                            == recorded_ids[offset]
                        )
                        if identity_proven
                        else row.metadata.get("chunk_order") == order
                    )
                ]
                if len(matches) != 1:
                    break
                selected.append(matches[0])
            else:
                if (
                    len({row.id for row in selected}) == len(selected)
                    and hierarchical_aggregate_text(
                        root_label, [row.text for row in selected]
                    )
                    == aggregate.text
                ):
                    recovered[aggregate.id] = [row.id for row in selected]
                    continue
            # Repeated heading bodies need the entire recorded contiguous window.
            if (
                identity_proven
                and all(
                    isinstance(order, int)
                    and not isinstance(order, bool)
                    and order >= 0
                    for order in orders
                )
                and orders == list(range(orders[0], orders[0] + len(orders)))
            ):
                windows = [
                    candidates[start : start + len(orders)]
                    for start in range(len(candidates) - len(orders) + 1)
                    if all(
                        row.id == identifier
                        or make_regulatory_chunk_id(
                            UUID(row.user_file_id), order, row.text
                        )
                        == identifier
                        for row, identifier, order in zip(
                            candidates[start : start + len(orders)],
                            recorded_ids,
                            orders,
                        )
                    )
                    and hierarchical_aggregate_text(
                        root_label,
                        [row.text for row in candidates[start : start + len(orders)]],
                    )
                    == aggregate.text
                ]
                if len(windows) == 1:
                    recovered[aggregate.id] = [row.id for row in windows[0]]
                    continue
            if recorded_ids:
                # Exact text cannot overrule contradictory persisted identity.
                continue
        matches: list[list[str]] = []
        for start in range(len(candidates)):
            texts: list[str] = []
            for end in range(start, len(candidates)):
                texts.append(candidates[end].text)
                rendered = hierarchical_aggregate_text(root_label, texts)
                if len(rendered) > len(aggregate.text):
                    break
                if rendered == aggregate.text:
                    matches.append([row.id for row in candidates[start : end + 1]])
                    break
            if len(matches) > 1:
                break
        if len(matches) == 1:
            recovered[aggregate.id] = matches[0]
    return recovered


def recovered_source_row(
    row: AnnexCanonicalSnapshot, members: list[str]
) -> AnnexCanonicalSnapshot:
    metadata = dict(row.metadata)
    if metadata.get("bound_to_regulatory_chunk_id") is not None:
        if len(members) != 1:
            raise ValueError("image source recovery must be unique")
        metadata["bound_to_regulatory_chunk_id"] = members[0]
        metadata["source_regulatory_chunk_ids"] = []
    else:
        metadata["source_regulatory_chunk_ids"] = members
    return row.model_copy(update={"metadata": metadata})


def validate_canonical_source_integrity(rows: list[AnnexCanonicalSnapshot]) -> None:
    """Check the final ingest graph after all transformations and ID allocation."""
    by_id = {row.id: row for row in rows}
    if len(by_id) != len(rows) or len({row.user_file_id for row in rows}) > 1:
        raise ValueError("canonical source membership crosses identity scope")
    pending: dict[str, set[str]] = {}
    for row in rows:
        members = source_ids(row)
        if row.metadata.get(
            "chunk_variant"
        ) == "hierarchical_aggregate" and not aggregate_membership_is_valid(row, by_id):
            raise ValueError(
                "aggregate source membership does not reconstruct final text: " + row.id
            )
        for member in members:
            if member == row.id or member not in by_id:
                raise ValueError(
                    "canonical source membership is dangling or self-referential: "
                    + row.id
                )
        pending[row.id] = set(members)
    while pending:
        ready = {
            identifier
            for identifier, members in pending.items()
            if not members.intersection(pending)
        }
        if not ready:
            raise ValueError("canonical source membership is cyclic")
        for identifier in ready:
            del pending[identifier]


def image_membership_is_valid(
    image: AnnexCanonicalSnapshot, parent: AnnexCanonicalSnapshot
) -> bool:
    if (
        image.id == parent.id
        or image.user_file_id != parent.user_file_id
        or not source_window_covers(parent, image)
        or parent.metadata.get("chunk_variant")
        in {"hierarchical_aggregate", "image_companion"}
        or parent.metadata.get("bound_to_regulatory_chunk_id") is not None
        or image.heading_path != parent.heading_path
    ):
        return False
    if image.text in (
        parent.text,
        parent.text
        + "\n\n[Görsel: "
        + str(image.metadata.get("image_alt") or "")
        + "]",
    ):
        return True
    asset = image.metadata.get("image_file_id")
    parent_assets = parent.metadata.get("image_file_ids")
    return (
        isinstance(asset, str)
        and bool(asset)
        and (
            parent.metadata.get("image_file_id") == asset
            or isinstance(parent_assets, list)
            and asset in parent_assets
        )
    )
