"""A family is a source-backed ownership component, never a title deduplication."""

from uuid import uuid4

import pytest

from onyx.db.models import RegulatoryDocumentIntakeFile
from onyx.db.regulatory_amendment_source_families import _component, _own_appendix
from onyx.regulatory.amendments.structural_target import normalize_appendix_label


@pytest.mark.parametrize(
    "change",
    [
        "valid",
        "primary_only",
        "child_only",
        "cycle",
        "cross_batch",
        "missing_parent",
        "second_root",
        "wrong_role",
        "duplicate_file",
        "unknown_file",
    ],
)
def test_only_one_actual_primary_component_can_collapse(change: str) -> None:
    batch = uuid4()
    primary = RegulatoryDocumentIntakeFile(
        id=uuid4(),
        intake_id=batch,
        user_file_id=uuid4(),
        source_asset_id=uuid4(),
        role="root",
    )
    child = RegulatoryDocumentIntakeFile(
        id=uuid4(),
        intake_id=batch,
        user_file_id=uuid4(),
        source_asset_id=uuid4(),
        role="annex",
        parent_document_id=primary.id,
    )
    matched = {primary.user_file_id, child.user_file_id}
    members = [primary, child]
    if change == "primary_only":
        matched = {primary.user_file_id}
    elif change == "child_only":
        matched = {child.user_file_id}
    elif change == "cycle":
        primary.parent_document_id = child.id
    elif change == "cross_batch":
        child.intake_id = uuid4()
    elif change == "missing_parent":
        child.parent_document_id = uuid4()
    elif change == "second_root":
        child.parent_document_id, child.role = None, "root"
    elif change == "wrong_role":
        child.role = "root"
    elif change == "duplicate_file":
        child.user_file_id = primary.user_file_id
    elif change == "unknown_file":
        matched.add(uuid4())
    result = _component(members, matched)
    if change in {"valid", "primary_only"}:
        assert result is not None and result[0] is primary and result[1] == members
    else:
        assert result is None


@pytest.mark.parametrize("label", ["EK I", "EK II", "EK IV", "EK-1"])
def test_exact_own_appendix_before_provisions(label: str) -> None:
    assert _own_appendix(
        ["Own law", label, "MADDE 2"],
        {"appendix_label": label},
        normalize_appendix_label(label),
    )


@pytest.mark.parametrize(
    "path,metadata",
    [
        (["Own law", "MADDE 2", "EK I"], {"appendix_label": "EK I"}),
        (["Own law", "(1)", "EK I"], {"appendix_label": "EK I"}),
        (["Own law", "a)", "EK I"], {"appendix_label": "EK I"}),
        (["Own law", "EK I uyarınca"], {"appendix_label": "EK I"}),
        (["Own law"], {"appendix_label": "EK I"}),
        (["Own law", "EK I"], {"appendix_label": "EK II"}),
    ],
)
def test_body_citation_or_metadata_alone_cannot_create_appendix_scope(
    path: list[str], metadata: dict[str, object]
) -> None:
    assert not _own_appendix(path, metadata, normalize_appendix_label("EK I"))
