"""The fixed preflight uses production PDF evidence with controlled transport only."""

import json
from unittest.mock import MagicMock

import pytest


def test_pdf_fixture_is_mixed_and_pinned() -> None:
    from io import BytesIO

    from pypdf import PdfReader

    from onyx.regulatory.amendments.annexes import acceptance_pdf_vision as probe

    content = probe.load_fixture()
    pages = PdfReader(BytesIO(content)).pages
    text = pages[0].extract_text()
    assert len(pages) == 1 and "MADDE 3" in text
    assert "17%" not in text and "11%" not in text


@pytest.mark.parametrize(
    "outcome",
    [
        "correct",
        "wrong_value",
        "unsupported",
        "transport_failure",
        "invalid_json",
        "rate_limit",
        "rate_limit_exhausted",
        "rate_limit_budget",
        "duplicate_http",
        "auxiliary_http",
    ],
)
def test_real_pdf_helpers_count_attempts_and_preserve_receipt(
    monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    import httpx

    from onyx.llm.factory import get_llm
    from onyx.regulatory import structured_llm
    from onyx.regulatory.amendments.annexes import acceptance_pdf_vision as probe

    monkeypatch.setattr(structured_llm.time, "sleep", lambda _seconds: None)
    stage_calls: dict[str, int] = {}
    calls: list[dict[str, object]] = []

    def send(
        _client: httpx.Client, request: httpx.Request, **_kwargs: object
    ) -> httpx.Response:
        body = json.loads(request.content)
        calls.append(body)
        assert body.get("stream", False) is False
        assert "image_url" in str(body["messages"])
        stage_calls[report.probe_stage] = stage_calls.get(report.probe_stage, 0) + 1
        if outcome == "duplicate_http":
            return httpx.Client.send(_client, request)
        if outcome in {"rate_limit", "rate_limit_exhausted", "rate_limit_budget"} and (
            outcome != "rate_limit" or stage_calls[report.probe_stage] == 1
        ):
            return httpx.Response(
                429,
                request=request,
                headers={
                    "Retry-After": "90" if outcome == "rate_limit_budget" else "0"
                },
                json={"error": {"message": "fixture capacity"}},
            )
        if outcome == "transport_failure":
            return httpx.Response(
                500, request=request, json={"error": {"message": "DO_NOT_LOG_secret"}}
            )
        if report.probe_stage == "source":
            result = {} if outcome == "invalid_json" else vision_response()
        elif report.probe_stage == "draft":
            text = probe.OLD_TEXT.replace("5%", "17%").replace(
                "11%", "13%" if outcome == "wrong_value" else "11%"
            )
            result = {
                "new_chunk": {
                    "text": text,
                    "chunk_type": "article",
                    "heading_path": ["MADDE 3"],
                    "metadata_changes": {},
                },
                "dates": {
                    "effective_start_date": None,
                    "effective_end_date": None,
                    "rationale": "No dates",
                },
            }
        else:
            result = {
                "supported": outcome != "unsupported",
                "ambiguous": False,
                "rationale": "Fixed fictional source",
            }
        return httpx.Response(
            200,
            request=request,
            json={
                "id": "fixture",
                "object": "chat.completion",
                "created": 1,
                "model": "openai/fictional-model",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": json.dumps(result)},
                    }
                ],
            },
        )

    monkeypatch.setattr(httpx.Client, "send", send)
    llm = get_llm(
        provider="openrouter",
        model="openai/fictional-model",
        max_input_tokens=32000,
        deployment_name=None,
        api_key="fixture-only",
        temperature=0,
        timeout=60,
    )
    if outcome == "auxiliary_http":
        from onyx.regulatory.amendments import pdf_vision

        def unexpected_http(*_args: object, **_kwargs: object) -> None:
            with httpx.Client() as client:
                client.post("https://fixture.invalid/auxiliary")

        monkeypatch.setattr(pdf_vision, "prepare_pdf_source", unexpected_http)
    report = probe.PdfVisionProbeReport()
    retained: list[probe.PdfVisionProbeReport] = []
    probe.run_probe(
        llm, report, lambda value: retained.append(value.model_copy(deep=True))
    )
    assert report.status == (
        "passed" if outcome in {"correct", "rate_limit"} else "failed"
    ), report.model_dump_json()
    expected = (
        1
        if outcome
        in {"invalid_json", "rate_limit_budget", "duplicate_http", "transport_failure"}
        else 2
        if outcome == "wrong_value"
        else 3
    )
    if outcome == "rate_limit":
        expected = 6
    elif outcome == "auxiliary_http":
        expected = 0
    assert report.attempt_count == report.http_request_count == len(calls) == expected
    assert "DO_NOT_LOG_secret" not in report.model_dump_json()
    if outcome == "invalid_json":
        assert "ValidationError" in (report.failure or "")
    if outcome in {"correct", "rate_limit"}:
        assert (
            report.native_value_absent
            and report.image_evidence
            and report.grounding_verified
        )
        assert (
            report.draft_sha256 and report.receipt_sha256 and report.transcript_sha256
        )
        assert report.page_count == 1 and report.fixture_verified
        assert (
            probe.report_from_output((report.model_dump_json() + "\n").encode())[
                "status"
            ]
            == "passed"
        )
    assert retained[-1].status == report.status


def vision_response() -> dict[str, object]:
    elements: list[dict[str, object]] = [
        {
            "kind": "text",
            "text": "MADDE 3 - Bugday orani asagidaki tabloda gosterilen deger olarak degistirilmistir. Diger oranlar degismemistir.",
            "box": [0.08, 0.08, 0.92, 0.28],
            "status": "readable",
            "issues": [],
        }
    ]
    for row, values in enumerate(
        [("Urun", "Oran"), ("Bugday", "17%"), ("Pirinc", "11%")]
    ):
        for column, text in enumerate(values):
            elements.append(
                {
                    "kind": "table_cell",
                    "text": text,
                    "box": [
                        0.1 + column * 0.42,
                        0.40 + row * 0.12,
                        0.46 + column * 0.42,
                        0.49 + row * 0.12,
                    ],
                    "table_role": "column_header" if row == 0 else "data",
                    "status": "readable",
                    "issues": [],
                }
            )
    return {"elements": elements}


def test_probe_scope_refuses_before_start(monkeypatch: pytest.MonkeyPatch) -> None:
    from onyx.regulatory.amendments.annexes import acceptance_pdf_vision as probe

    monkeypatch.setenv("POSTGRES_DB", "customs-regulations-test")
    process = MagicMock()
    monkeypatch.setattr(probe.subprocess, "Popen", process)
    assert probe.run_pdf_vision_probe()["status"] == "failed"
    process.assert_not_called()


def test_partial_probe_report_survives_timeout_without_false_success() -> None:
    from onyx.regulatory.amendments.annexes import acceptance_pdf_vision as probe

    report = probe.PdfVisionProbeReport(
        probe_stage="draft", attempt_count=2, http_request_count=2
    )
    final = probe.report_from_output(
        (report.model_dump_json() + "\n").encode(), "process_timeout"
    )
    assert final["status"] == "failed" and final["attempt_count"] == 2
    assert final["failure"] == "process_timeout" and not final["attempt_count_complete"]
    invalid = probe.report_from_output(
        (probe.PdfVisionProbeReport(status="passed").model_dump_json() + "\n").encode()
    )
    assert invalid["status"] == "failed"


def test_pdf_probe_failed_evidence_survives_runner_sanitizer(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from scripts import regulatory_annex_dev_cutover as cutover

    report = {
        "phase": "preflight",
        "status": "failed",
        "release_sha_metadata": "a" * 40,
        "calibration": {"status": "passed"},
        "failure_stage": "pdf_vision",
        "exception_type": "ValidationError",
        "pdf_vision_probe": {
            "status": "failed",
            "probe_stage": "source",
            "attempt_count": 1,
            "http_request_count": 1,
            "failure": '{"stage":"pdf_vision","exceptions":[]}',
        },
    }
    with pytest.raises(cutover.CutoverRefusal):
        cutover.emit_acceptance_report(json.dumps(report), "preflight", "a" * 40)
    retained = json.loads(capsys.readouterr().out)
    assert retained == report
