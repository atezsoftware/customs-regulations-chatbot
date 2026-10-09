from contextlib import contextmanager
from types import SimpleNamespace
from typing import Iterator
from unittest.mock import Mock

import pytest

from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import RunStopped
from onyx.legal_composite.acquisition import InvalidSourceAction
from onyx.legal_composite.engine import LegalCompositeEngine, _admission_reason
from onyx.legal_composite.models import WorkflowPolicy


@pytest.mark.parametrize(
    "error,reason",
    [
        (
            RunStopped(
                "Provider usage exceeded the estimate; no further spend authorized"
            ),
            "usage_estimate_overrun",
        ),
        (RunStopped("The model response failed the workflow schema"), "output_schema"),
        (InvalidSourceAction("private original context"), "source_action"),
        (RunStopped("private original context"), "other_stop"),
    ],
)
def test_terminal_reason_has_no_source_or_error_text(
    monkeypatch: pytest.MonkeyPatch,
    error: RunStopped | InvalidSourceAction,
    reason: str,
) -> None:
    recorded: list[object] = []

    @contextmanager
    def capture(
        operation: str, attributes: object, *, summary: str
    ) -> Iterator[SimpleNamespace]:
        step = SimpleNamespace(output_value=None)
        yield step
        recorded.append((operation, attributes, summary, step.output_value))

    monkeypatch.setattr("onyx.legal_composite.engine.graph_step", capture)
    value = LegalCompositeEngine(
        gateway=Mock(),
        acquirer=Mock(),
        ledger=EvidenceLedger(),
        policy=WorkflowPolicy(),
        reviewer=Mock(),
        check_active=lambda: None,
        research_available=lambda: True,
    )
    monkeypatch.setattr(value, "_run", Mock(side_effect=error))
    result = value.run("private user question")
    assert result.status == "unavailable"
    assert result.answer is None
    assert recorded == [
        ("legal_composite.run_stop", {}, f"reason={reason}", {"reason": reason})
    ]
    assert "private" not in repr(recorded)


def test_legacy_consumer_does_not_emit_new_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture = Mock()
    monkeypatch.setattr("onyx.legal_composite.engine.graph_step", capture)
    value = LegalCompositeEngine(
        gateway=Mock(),
        acquirer=Mock(),
        ledger=EvidenceLedger(),
        policy=WorkflowPolicy(),
        check_active=lambda: None,
        research_available=lambda: True,
    )
    monkeypatch.setattr(value, "_run", Mock(side_effect=RunStopped("private error")))
    assert value.run("private question").status == "unavailable"
    capture.assert_not_called()


@pytest.mark.parametrize(
    "message,reason",
    [
        ("Support span ID is not in this exact original", "span_identity"),
        ("Support quotation conflicts with its original span", "span_quotation"),
        ("private original identity and quotation", "source_action"),
    ],
)
def test_admission_reason_does_not_export_dynamic_source_errors(
    message: str, reason: str
) -> None:
    assert _admission_reason(InvalidSourceAction(message)) == reason
