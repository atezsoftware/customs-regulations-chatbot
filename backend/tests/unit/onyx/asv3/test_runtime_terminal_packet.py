"""ASv3 successful publication ends the public packet stream exactly once."""

import json
from typing import Any

import pytest

from onyx.asv3 import runtime
from onyx.asv3.models import EvidenceItem, RunContext, RunStopped
from onyx.asv3.workers import WorkerPool
from onyx.asv3.workflow_variant import (
    ASV3_EXP_CROPPED_VARIANT,
    ASV3_FAST_CROPPED_VARIANT,
    ASV3_FAST_VARIANT,
    ASV3_STANDARD_VARIANT,
)
from onyx.chat.emitter import Emitter
from onyx.regulatory.publication_reads import PublicationReadChanged
from onyx.server.query_and_chat.streaming_models import (
    AgentResponseDelta,
    CitationInfo,
    OverallStop,
    Packet,
    SectionEnd,
)
from tests.unit.onyx.asv3.test_cropped_focused_repair_runtime import (
    priced_scripted_provider,  # noqa: F401
)
from tests.unit.onyx.asv3.test_native_model_adapter import model
from tests.unit.onyx.asv3.test_runtime import packets, response, setup_run, user_payload

pytestmark = pytest.mark.usefixtures(
    "empty_source_inventory", "priced_scripted_provider"
)


@pytest.mark.parametrize(
    "variant",
    [
        ASV3_STANDARD_VARIANT,
        ASV3_EXP_CROPPED_VARIANT,
        ASV3_FAST_CROPPED_VARIANT,
    ],
)
@pytest.mark.parametrize("terminal", ["answer", "partial", "clarification"])
def test_publication_emits_one_terminal_stop_after_its_durable_result(
    monkeypatch: pytest.MonkeyPatch, variant: str, terminal: str
) -> None:
    kwargs, _broker, selected, checkpoints, queue = setup_run(monkeypatch)
    kwargs.pop("test_language")
    kwargs.update(research_profile="normal", workflow_variant=variant)
    kwargs["emitter"] = Emitter(queue, model_idx=2)
    body = "Merhaba, nasıl yardımcı olabilirim?"
    if terminal == "answer":
        kwargs["simple_chat_history"][0].message = "Merhaba"
        name, arguments = "submit_answer", {"answer": body, "basis": "conversation"}
        expected_reason = "native_answer_published"
    elif terminal == "partial":
        body = "İstenen özgün hüküm bu denemede doğrulanamadı."
        name, arguments = "submit_partial_answer", {"answer": body}
        expected_reason = "native_partial_published"
    else:
        body = "Makine garanti kapsamında mı gönderiliyor?"
        name, arguments = "ask_user", {"question": body}
        expected_reason = "clarification_requested"
    selected.invoke.side_effect = [response(calls=[(name, arguments)])]
    closed = False
    original_close = WorkerPool.close

    def close(workers: WorkerPool) -> None:
        nonlocal closed
        original_close(workers)
        closed = True

    monkeypatch.setattr(WorkerPool, "close", close)
    final_checkpoint_before_stop: list[tuple[str, str, bool]] = []
    original_emit = kwargs["emitter"].emit

    def record(packet: Packet) -> None:
        if isinstance(packet.obj, OverallStop):
            final_checkpoint_before_stop.append(
                (
                    checkpoints[-1]["publication_stop_reason"],
                    kwargs["state_container"].answer_tokens,
                    closed,
                )
            )
        original_emit(packet)

    monkeypatch.setattr(kwargs["emitter"], "emit", record)
    runtime.run_asv3_loop(**kwargs)

    output = packets(queue)
    stops = [packet for packet in output if isinstance(packet.obj, OverallStop)]
    assert len(stops) == 1 and stops[0] is output[-1]
    assert isinstance(output[-2].obj, SectionEnd)
    assert stops[0].placement.model_index == 2
    assert isinstance(stops[0].obj, OverallStop)
    assert stops[0].obj.stop_reason is None
    assert final_checkpoint_before_stop == [(expected_reason, body, True)]
    assert any(isinstance(packet.obj, AgentResponseDelta) for packet in output[:-2])
    assert selected.invoke.call_count == 1


@pytest.mark.parametrize("variant", [ASV3_STANDARD_VARIANT, ASV3_FAST_VARIANT])
def test_original_citations_finish_before_terminal_stop(
    monkeypatch: pytest.MonkeyPatch, variant: str
) -> None:
    kwargs, broker, _selected, checkpoints, queue = setup_run(monkeypatch)
    kwargs.pop("test_language")
    kwargs.update(research_profile="normal", workflow_variant=variant)
    reviewer = model()

    def review(**arguments: Any) -> Any:
        payload = user_payload(arguments["prompt"][-1])
        return response(
            json.dumps(
                {
                    "status": "complete",
                    "candidate_sha256": payload["candidate_sha256"],
                    "source_set_sha256": payload["source_set_sha256"],
                    "examined_citations": payload["required_evidence_numbers"],
                    "findings": [],
                }
            )
        )

    reviewer.invoke.side_effect = review
    monkeypatch.setattr(runtime, "fast_material_review_model", lambda _user: reviewer)
    runtime.run_asv3_loop(**kwargs)
    output = packets(queue)
    assert isinstance(output[-1].obj, OverallStop)
    assert sum(isinstance(packet.obj, OverallStop) for packet in output) == 1
    assert (
        len([packet for packet in output[:-1] if isinstance(packet.obj, CitationInfo)])
        == 2
    )
    assert {item.chunk_id for item in broker.revalidated} == {"chunk-0", "chunk-1"}
    assert checkpoints[-1]["publication_status"] == "found"
    if variant == ASV3_FAST_VARIANT:
        assert reviewer.invoke.call_count == 1


@pytest.mark.parametrize(
    "failure", ["provider", "cancel", "source", "checkpoint", "teardown"]
)
def test_failed_or_cancelled_publication_never_emits_success_stop(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    kwargs, broker, selected, _checkpoints, queue = setup_run(monkeypatch)
    kwargs.pop("test_language")
    kwargs["research_profile"] = "normal"
    expected: type[Exception]
    if failure == "provider":
        expected = ValueError
        selected.invoke.side_effect = ValueError("Provider rejected the request")
    elif failure == "cancel":
        expected = RunStopped
        broker.cancelled.set()
    elif failure == "teardown":
        expected = ValueError
        original_close = WorkerPool.close

        def fail_close(workers: WorkerPool) -> None:
            original_close(workers)
            raise ValueError("Worker teardown failed")

        monkeypatch.setattr(WorkerPool, "close", fail_close)
    else:
        expected = PublicationReadChanged if failure == "source" else ValueError

        def reject(_items: list[EvidenceItem], _context: RunContext) -> None:
            raise PublicationReadChanged()

        if failure == "source":
            monkeypatch.setattr(broker, "revalidate_evidence", reject)
        else:
            original_checkpoint = runtime.save_asv3_checkpoint

            def fail_final_checkpoint(**arguments: Any) -> None:
                if (
                    arguments["snapshot"].get("publication_stop_reason")
                    == "native_answer_published"
                ):
                    raise ValueError("Final durable checkpoint failed")
                original_checkpoint(**arguments)

            monkeypatch.setattr(runtime, "save_asv3_checkpoint", fail_final_checkpoint)
    with pytest.raises(expected):
        runtime.run_asv3_loop(**kwargs)
    assert not any(isinstance(packet.obj, OverallStop) for packet in packets(queue))
