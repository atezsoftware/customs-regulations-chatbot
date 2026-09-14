"""Geometry checks that keep merged form cells distinct from inferred table rows."""

from collections.abc import Sequence

NormalizedBox = tuple[float, float, float, float]


def require_disjoint_table_cells(boxes: Sequence[NormalizedBox]) -> None:
    ordered = sorted(boxes, key=lambda box: (box[1], box[0]))
    for index, box in enumerate(ordered):
        for other in ordered[index + 1 :]:
            if other[1] >= box[3]:
                break
            if min(box[2], other[2]) > max(box[0], other[0]) and min(
                box[3], other[3]
            ) > max(box[1], other[1]):
                raise ValueError(
                    "pdf_table_cells_overlap: distinct table cells must not "
                    "overlap in both dimensions"
                )


def has_ambiguous_table_rows(boxes: Sequence[NormalizedBox]) -> bool:
    row: list[NormalizedBox] = []
    bottom = 0.0
    for box in sorted(boxes, key=lambda box: (box[1], box[0])):
        if not row or box[1] >= bottom:
            row = [box]
            bottom = box[3]
            continue
        for previous in row:
            overlap = min(box[3], previous[3]) - max(box[1], previous[1])
            if overlap < 0.5 * min(box[3] - box[1], previous[3] - previous[1]):
                return True
        row.append(box)
        bottom = max(bottom, box[3])
    return False
