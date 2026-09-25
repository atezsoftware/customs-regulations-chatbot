"""Rebuild dated derived consumers of an already reviewed canonical transition."""

import hashlib
from datetime import date

from onyx.regulatory.amendment_projection_impact import structural_windows
from onyx.regulatory.amendments.annexes.models import AnnexCanonicalSnapshot
from onyx.regulatory.amendments.annexes.selective_impact import (
    recover_source_membership,
    recovered_source_row,
    source_ids,
)
from onyx.regulatory.chunker import (
    hierarchical_aggregate_root_label,
    hierarchical_aggregate_text,
)


def rebuild_amendment_dependents(
    before: list[AnnexCanonicalSnapshot],
    after: list[AnnexCanonicalSnapshot],
) -> list[AnnexCanonicalSnapshot]:
    """Return prospective canonical rows; new ordinals must be reserved by the writer.

    Membership is explicit or recovered by exact reconstruction. Version changes
    propagate through consumers, never through unrelated siblings in a file.
    """
    old = {r.id: r for r in before}
    desired = {r.id: r for r in after}
    if len(old) != len(before) or len(desired) != len(after):
        raise ValueError("duplicate canonical identity")
    if (
        len({r.user_file_id for r in before + after}) != 1
        or not old.keys() <= desired.keys()
    ):
        raise ValueError("amendment dependency scope or history changed")
    recovered = recover_source_membership(before)

    def with_membership(
        rows: list[AnnexCanonicalSnapshot],
    ) -> list[AnnexCanonicalSnapshot]:
        return [
            recovered_source_row(r, recovered[r.id]) if r.id in recovered else r
            for r in rows
        ]

    impact_windows = structural_windows(with_membership(before), with_membership(after))
    dependencies: dict[str, list[str]] = {}
    for row in before:
        members = recovered.get(row.id, source_ids(row))
        if (
            row.metadata.get("chunk_variant") == "hierarchical_aggregate"
            and not members
        ):
            raise ValueError(f"aggregate source membership unavailable: {row.id}")
        if members:
            dependencies[row.id] = members
    from onyx.regulatory.position_rebase import position_only_changes

    position_only = position_only_changes(before, after)
    changed = {
        i
        for i in old
        if i not in dependencies and old[i] != desired[i] and i not in position_only
    }
    affected = set(changed)
    while True:
        consumers = {i for i, ids in dependencies.items() if affected.intersection(ids)}
        if consumers <= affected:
            break
        affected.update(consumers)
    pending = affected.intersection(dependencies)

    def versions(identifier: str) -> list[AnnexCanonicalSnapshot]:
        found = {identifier}
        while True:
            children = {
                r.id for r in desired.values() if r.supersedes_chunk_id in found
            }
            # A complete-unit replacement may map multiple removed descendants
            # to its single reviewed successor.
            children.update(
                successor
                for i in found
                if (successor := desired[i].superseded_by_chunk_id) is not None
            )
            if not children <= desired.keys():
                raise ValueError("derived successor missing")
            if children <= found:
                break
            found.update(children)
        return [desired[i] for i in sorted(found)]

    while pending:
        ready = sorted(i for i in pending if not pending.intersection(dependencies[i]))
        if not ready:
            raise ValueError("cyclic aggregate dependencies")
        for identifier in ready:
            # Keep exact recovery evidence when dates stop permitting reconstruction.
            original = with_membership([old[identifier]])[0]
            original = original.model_copy(
                update={"position": desired[identifier].position}
            )
            members = dependencies[identifier]
            if any(i not in old for i in members):
                raise ValueError("derived source missing: " + identifier)
            alternatives = {i: versions(i) for i in members}
            lower = original.validity_start_date or date.min
            upper = original.validity_end_date or date.max
            boundaries = sorted(
                {lower, upper}
                | {
                    d
                    for w in impact_windows.get(identifier, [])
                    for d in w
                    if lower < d < upper
                }
                | {
                    d
                    for rs in alternatives.values()
                    for r in rs
                    for d in (r.validity_start_date, r.validity_end_date)
                    if d is not None and lower < d < upper
                }
            )
            successors: list[AnnexCanonicalSnapshot] = []
            first_change: date | None = None
            for start, end in zip(boundaries, boundaries[1:]):
                if first_change is None and not any(
                    a <= start and end <= b
                    for a, b in impact_windows.get(identifier, [])
                ):
                    continue
                selected: dict[str, AnnexCanonicalSnapshot] = {}
                for source, candidates in alternatives.items():
                    visible = [
                        r
                        for r in candidates
                        if (r.validity_start_date or date.min)
                        <= start
                        < (r.validity_end_date or date.max)
                    ]
                    if len(visible) > 1:
                        raise ValueError("ambiguous derived source version")
                    if visible:
                        selected[source] = visible[0]
                if (
                    first_change is None
                    and len(selected) == len(members)
                    and all(
                        selected[i].id == i
                        and selected[i].text == old[i].text
                        and selected[i].metadata == old[i].metadata
                        for i in members
                    )
                ):
                    continue
                first_change = min(first_change or start, start)
                sources = list({r.id: r for r in selected.values()}.values())
                if not sources:
                    continue
                metadata = dict(original.metadata)
                if metadata.get("chunk_variant") == "hierarchical_aggregate":
                    root = metadata.get("hierarchy_root_path")
                    if (
                        not isinstance(root, list)
                        or not root
                        or not isinstance(root[-1], str)
                    ):
                        raise ValueError("aggregate root unavailable")
                    text = hierarchical_aggregate_text(
                        hierarchical_aggregate_root_label(metadata, original.text),
                        [r.text for r in sources],
                    )
                    metadata["source_regulatory_chunk_ids"] = [r.id for r in sources]
                    metadata["source_chunk_orders"] = [
                        r.metadata.get("chunk_order", r.position) for r in sources
                    ]
                else:
                    if len(sources) != 1 or len(members) != 1:
                        raise ValueError("ambiguous image companion source")
                    parent = sources[0]
                    for key in ("image_file_id", "image_file_ids", "source_asset_ids"):
                        if parent.metadata.get(key) != old[members[0]].metadata.get(
                            key
                        ):
                            raise ValueError(
                                "changed image asset requires a reviewed companion"
                            )
                    text = original.text.replace(old[members[0]].text, parent.text)
                    metadata["bound_to_regulatory_chunk_id"] = parent.id
                digest = hashlib.sha256(
                    f"{identifier}:{start}:{end}:{','.join(r.id for r in sources)}:{text}".encode()
                ).hexdigest()
                successors.append(
                    original.model_copy(
                        update={
                            "id": "rc_" + digest,
                            "text": text,
                            "metadata": metadata,
                            "projection_ordinal": -1,
                            "status": "active",
                            "source": "amendment",
                            "validity_start_date": None if start == date.min else start,
                            "validity_end_date": None if end == date.max else end,
                            "supersedes_chunk_id": original.id,
                            "superseded_by_chunk_id": None,
                        }
                    )
                )
            if first_change is not None:
                desired[identifier] = original.model_copy(
                    update={
                        "validity_end_date": first_change,
                        "status": "superseded",
                        "superseded_by_chunk_id": successors[0].id
                        if successors
                        else None,
                    }
                )
                desired.update({r.id: r for r in successors})
            pending.remove(identifier)
    return list(desired.values())
