from typing import Any
from unittest.mock import Mock
from uuid import uuid4

import pytest

from onyx.db.regulatory_annex_acceptance import CanaryRun
from onyx.regulatory.amendments.annexes import acceptance_canary as canary


@pytest.mark.parametrize(
    "answer",
    [
        "Oran %7 olarak belirlenmiştir.",
        "Oran 7%.",
        "% 7",
        "7 %",
        "%7,0",
        "7.00 %",
        "+7%",
    ],
)
def test_equivalent_percentage_spellings(answer: str) -> None:
    assert canary.contains_percentage(answer, "7%")


@pytest.mark.parametrize(
    "answer",
    [
        "17%",
        "%70",
        "7.5%",
        "%7,5",
        "-7%",
        "−7 %",
        "- 7%",
        "-%7",
        "- %7",
        "% -7",
        "7",
        "abc7%",
        "1.7%",
        "1,7%",
        "7.0001%",
    ],
)
def test_other_values_and_partial_numbers_refuse(answer: str) -> None:
    assert not canary.contains_percentage(answer, "7%")


@pytest.mark.parametrize(
    "override,expected_error",
    [
        ({}, None),
        ({"answer": "Oran %17."}, "canary_dated_chat_evidence_failed"),
        ({"error_msg": "tool failed"}, "canary_dated_chat_evidence_failed"),
        ({"top_documents": []}, "canary_dated_chat_evidence_failed"),
        ({"citation_info": []}, "canary_dated_chat_evidence_failed"),
        (
            {"top_documents": [{"document_id": "foreign"}]},
            "canary_chat_did_not_retrieve_owned_file",
        ),
    ],
)
def test_real_canary_checks_prefix_answer_and_preserves_evidence_guards(
    override: dict[str, Any],
    expected_error: str | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = CanaryRun(release_sha="a" * 40, user_id=uuid4(), document_set_id=24)
    result = {
        "answer": "1001.10 için oran %7 olarak belirlenmiştir [3].",
        "top_documents": [{"document_id": str(run.file_id), "chunk_ind": 5}],
        "citation_info": [
            {"citation_num": 3, "document_id": str(run.file_id), "chunk_ind": 5}
        ],
        **override,
    }
    request = Mock(side_effect=[[{"id": 1, "display_name": "Internal Search"}], result])
    monkeypatch.setattr(canary, "request_json", request)
    monkeypatch.setattr(canary, "create_canary_chat", Mock(return_value=uuid4()))
    monkeypatch.setattr(canary, "save_canary", Mock())
    if expected_error:
        with pytest.raises(ValueError, match=expected_error):
            canary.chat_canary(Mock(), run, as_of="2026-09-10", rate="7%")
        assert "chat_2026-09-10" not in run.evidence
    else:
        canary.chat_canary(Mock(), run, as_of="2026-09-10", rate="7%")
        assert run.evidence["chat_2026-09-10"] is True
    payload = request.call_args.kwargs["json"]
    assert "7%" not in payload["message"] and "%7" not in payload["message"]
    assert payload["internal_search_filters"] == {
        "document_set": [run.name],
        "as_of_date": "2026-09-10",
    }
    assert payload["forced_tool_id"] == 1
