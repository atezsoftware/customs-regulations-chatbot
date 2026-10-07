"""Choose connected source structure without interpreting legal effect."""

import re
from collections.abc import Sequence

_PARAGRAPH = re.compile(r"^(?:\(\d+\)|paragraph\s+\d+\b)", re.I)
_CLAUSE = re.compile(r"^(?:[a-zçğıöşü]+\)|\([a-zçğıöşü]+\))", re.I)


def source_group_indices(
    paths: Sequence[tuple[str, ...]], seed_path: tuple[str, ...]
) -> set[int]:
    """A clause retains its paragraph; a decision retains its own primary structure."""
    seed = tuple(" ".join(part.split()) for part in seed_path)
    if len(seed) < 2:
        return set()
    prefix: tuple[str, ...] | None = None
    title = " ".join(seed[:2]).casefold()
    if ("mahkem" in title and "karar" in title) or any(
        term in title for term in ("judgment", "court decision", "tribunal decision")
    ):
        # A different second heading starts another document, e.g. an appended submission.
        prefix = seed[:2]
    else:
        for index, heading in enumerate(seed):
            if _PARAGRAPH.match(heading):
                prefix = seed[: index + 1]
                break
        if prefix is None and _CLAUSE.match(seed[-1]):
            prefix = seed[:-1]
    if prefix is None:
        return set()
    return {
        index
        for index, path in enumerate(paths)
        if tuple(" ".join(part.split()) for part in path)[: len(prefix)] == prefix
    }
