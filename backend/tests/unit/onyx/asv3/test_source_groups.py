"""Connected members retain conditions without crossing document/paragraph boundaries."""

from onyx.asv3.source_groups import source_group_indices


def test_clause_includes_its_parent_intro_and_all_alternatives_only() -> None:
    paths = [
        ("Instrument", "MADDE 7", "(1) Introduction"),
        ("Instrument", "MADDE 7", "(1) Introduction", "a) First condition"),
        ("Instrument", "MADDE 7", "(1) Introduction", "b) Alternative"),
        ("Instrument", "MADDE 7", "(2) Separate rule"),
        ("Instrument", "Annex", "MADDE 7", "(1) Introduction"),
    ]
    assert source_group_indices(paths, paths[1]) == {0, 1, 2}
    assert source_group_indices(paths, paths[0]) == {0, 1, 2}


def test_primary_decision_keeps_reasons_and_disposition_without_appended_submission() -> (
    None
):
    paths = [
        ("COURT JUDGMENT", "Decision identity"),
        ("COURT JUDGMENT", "Decision identity", "Reasons"),
        ("COURT JUDGMENT", "Decision identity", "Disposition"),
        ("COURT JUDGMENT", "Appended referral", "Submissions"),
    ]
    assert source_group_indices(paths, paths[1]) == {0, 1, 2}
    assert source_group_indices(paths, paths[2]) == {0, 1, 2}


def test_missing_structure_does_not_expand_a_source() -> None:
    assert source_group_indices([("Document",)], ("Document",)) == set()
    assert source_group_indices([("Law", "MADDE 8")], ("Law", "MADDE 8")) == set()
