"""Unresolved evidence states never approve unsupported legal conclusions."""

import pytest

from onyx.asv3.authority import native_named_authority_gap
from onyx.asv3.cropped_focused_repair import (
    _check_description,
    authority_answer_with_cropped_disclosures,
    source_gap_sentence,
)
from onyx.asv3.cropped_repair_state import (
    CroppedRepairState,
    CroppedRepairTarget,
    RepairStatus,
)
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import RunContext


@pytest.mark.parametrize("status", ["not_found", "incomplete", "failed"])
def test_only_exact_host_unresolved_notice_is_exempt(status: RepairStatus) -> None:
    state = CroppedRepairState(RunContext(), "İtiraz koşulları nelerdir?")
    target = CroppedRepairTarget(
        check_kind="governing_original",
        instrument_number="6455",
        article="11",
    )
    notice = source_gap_sentence(target, status)
    assert (
        native_named_authority_gap(
            authority_answer_with_cropped_disclosures(notice, state),
            EvidenceLedger(),
            strict_reference_boundaries=True,
            syntactic_reference_binding=True,
        )
        is not None
        or status == "not_found"
    )
    state.reserve(target)
    state.finish(
        target.key, status, [], "This focused attempt has no verified finding."
    )
    assert authority_answer_with_cropped_disclosures(notice, state) == ""
    unsupported = "6455 sayılı Kanun 11. maddesi uyarınca ceza uygulanmaz."
    for text in (notice + " " + unsupported, notice + "\n\n" + unsupported):
        probe = authority_answer_with_cropped_disclosures(text, state)
        assert (
            native_named_authority_gap(
                probe,
                EvidenceLedger(),
                strict_reference_boundaries=True,
                syntactic_reference_binding=True,
            )
            is not None
        )
    if status != "not_found":
        assert "özgün metni elde edilemedi" not in notice


def test_requested_condition_survives_but_answer_text_does_not() -> None:
    requested = _check_description(
        {
            "gap_entries": [
                {
                    "field": "generic",
                    "entry": {
                        "condition_id": "condition_1",
                        "source_id": "law-1",
                        "omitted_condition": "İzin başvurusu otuz gün içinde yapılmalıdır.",
                        "answer_units": [
                            {"unit_id": "unit_1", "text": "REJECTED ANSWER"}
                        ],
                        "draft_to_repair": "FULL DRAFT",
                    },
                }
            ],
            "unit_texts": {"unit_1": "REJECTED ANSWER"},
        }
    )
    assert isinstance(requested, dict)
    rows = requested["gap_entries"]
    assert isinstance(rows, list) and isinstance(rows[0], dict)
    assert rows[0]["entry"] == {
        "condition_id": "condition_1",
        "source_id": "law-1",
        "omitted_condition": "İzin başvurusu otuz gün içinde yapılmalıdır.",
    }
    assert "REJECTED ANSWER" not in str(requested)
    assert "FULL DRAFT" not in str(requested)
