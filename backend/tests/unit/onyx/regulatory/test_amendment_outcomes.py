from copy import deepcopy
from typing import Any

import pytest


@pytest.mark.parametrize(
    "difference",
    [None, "text", "metadata", "heading", "date", "unknown_date", "source", "identity"],
)
def test_already_applied_requires_exact_version_text_and_structure(
    difference: str | None,
) -> None:
    from onyx.regulatory.amendments.outcomes import identical_applied_result

    old: dict[str, Any] = {
        "id": "current",
        "source": "amendment",
        "status": "active",
        "supersedes_chunk_id": "previous",
        "user_file_id": "file",
        "position": 3,
        "text": "ğ) Current rule.",
        "chunk_type": "clause",
        "heading_path": ["MADDE 8", "(1)", "ğ) Current rule"],
        "metadata": {"article_no": "8", "paragraph_no": "1", "clause_label": "ğ"},
        "validity_start_date": "2026-08-09",
        "validity_end_date": None,
    }
    new: dict[str, Any] = {
        **deepcopy(old),
        "effective_start_date": "2026-08-09",
        "effective_end_date": None,
    }
    new["metadata"]["heading_path"] = old["heading_path"]
    if difference == "text":
        new["text"] = "ğ) Changed rule."
    elif difference == "metadata":
        old["metadata"]["clause_label"] = "g"
    elif difference == "heading":
        old["heading_path"] = ["MADDE 7"]
    elif difference == "date":
        new["effective_start_date"] = "2027-08-09"
    elif difference == "unknown_date":
        new["effective_start_date"] = old["validity_start_date"] = None
    elif difference == "source":
        old["source"] = "file"
    elif difference == "identity":
        new["user_file_id"] = "different-version"
    assert identical_applied_result(old, new) is (difference is None)
