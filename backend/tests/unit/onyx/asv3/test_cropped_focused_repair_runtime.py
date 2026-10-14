"""Native delegated repair preserves source guards and stops one repeated gap."""

import copy
import json
import threading
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, cast
from unittest.mock import MagicMock

import pytest

from onyx.asv3 import runtime
from onyx.asv3.corpus_tools import ChunkScan, CorpusBroker
from onyx.asv3.cropped_focused_repair import source_gap_sentence
from onyx.asv3.cropped_handoff_transport import expand_cropped_handoff_transport
from onyx.asv3.cropped_repair_state import CroppedRepairState, CroppedRepairTarget
from onyx.asv3.harness import Harness
from onyx.asv3.models import OutcomeStatus, RunContext
from onyx.asv3.workflow_variant import (
    ASV3_EXP_CROPPED_VARIANT,
    ASV3_TUNED_VARIANT,
)
from onyx.db.asv3_corpus import CorpusSource
from onyx.llm.model_response import ModelResponse, Usage
from onyx.llm.models import ToolMessage, UserMessage
from onyx.tracing.flows import LLMFlow
from tests.unit.onyx.asv3.test_runtime import (
    CorpusBoundary,
    configure_governing_basis_sources,
    response,
    setup_run,
)
from tests.unit.onyx.asv3.test_runtime import (
    delivered_originals as raw_delivered_originals,
)
from tests.unit.onyx.asv3.test_runtime import (
    user_payload as raw_user_payload,
)
from tests.unit.onyx.asv3.test_spawned_researcher_terminal import (
    capture_harnesses,
    scripted_coordinator,
)

pytestmark = pytest.mark.usefixtures(
    "empty_source_inventory", "priced_scripted_provider"
)


def user_payload(message: Any) -> dict[str, Any]:
    return cast(
        dict[str, Any], expand_cropped_handoff_transport(raw_user_payload(message))
    )


def delivered_originals(arguments: dict[str, Any]) -> list[dict[str, Any]]:
    decoded = {
        **arguments,
        "prompt": [
            message.model_copy(
                update={
                    "content": json.dumps(user_payload(message), ensure_ascii=False)
                }
            )
            if isinstance(message, UserMessage)
            else message
            for message in arguments["prompt"]
        ],
    }
    return raw_delivered_originals(decoded)


def cropped_coordinator_with_needs(
    tasks: list[str], child: Callable[..., ModelResponse]
) -> Callable[..., ModelResponse]:
    ordinary = scripted_coordinator(tasks, child)

    def invoke(**arguments: Any) -> ModelResponse:
        result = ordinary(**arguments)
        calls = result.choice.message.tool_calls or []
        if not calls or any(call.function.name != "spawn_researcher" for call in calls):
            return result
        needs: list[dict[str, Any]] = []
        spawned: list[tuple[str, dict[str, Any]]] = []
        for index, call in enumerate(calls):
            args = json.loads(call.function.arguments or "{}")
            need_id = f"fixture_source_need_{index}"
            needs.append(
                {
                    "need_id": need_id,
                    "question_ids": ["q0"],
                    "purpose": args["task"],
                    "completion_test": "Full operative original and relevant exceptions, or a precise unverified source gap.",
                    "material": True,
                }
            )
            spawned.append(("spawn_researcher", {**args, "need_ids": [need_id]}))
        return response(calls=[("update_research", {"needs": needs}), *spawned])

    return invoke


@pytest.fixture
def priced_scripted_provider(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    import litellm

    from tests.unit.onyx.asv3 import test_runtime, test_spawned_researcher_terminal

    monkeypatch.setitem(
        litellm.model_cost,
        "scripted",
        {
            "litellm_provider": "openai",
            "mode": "chat",
            "input_cost_per_token": 0.000001,
            "output_cost_per_token": 0.000002,
            "cache_read_input_token_cost": 0.0000001,
            "cache_creation_input_token_cost": 0.00000125,
            **{
                f"{field}_above_{threshold}k_tokens": value
                for threshold in (128, 200, 272)
                for field, value in (
                    ("input_cost_per_token", 0.000001),
                    ("output_cost_per_token", 0.000002),
                    ("cache_read_input_token_cost", 0.0000001),
                    ("cache_creation_input_token_cost", 0.00000125),
                )
            },
        },
    )

    def with_usage(
        original: Callable[..., ModelResponse],
    ) -> Callable[..., ModelResponse]:
        def measured(
            text: str | None = None,
            calls: list[tuple[str, dict[str, Any]]] | None = None,
        ) -> ModelResponse:
            result = original(text=text, calls=calls)
            result.usage = Usage(
                prompt_tokens=1,
                completion_tokens=1,
                total_tokens=2,
                cache_read_input_tokens=0,
                cache_creation_input_tokens=0,
            )
            return result

        return measured

    # Every scripted root/child/check response has an explicit synthetic receipt.
    for module in (test_runtime, test_spawned_researcher_terminal, request.module):
        monkeypatch.setattr(module, "response", with_usage(module.response))


_TASK = "6455 sayılı Kanunun 11. maddesinin özgün metnini araştır."
_LOWER_BODY = "Başvuruda belge sunulur [1]."
_DISCLOSURE = "6455 sayılı Kanun 11. maddesinin özgün metni elde edilemedi."
_FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "cropped5094.json").read_text()
)


class _SourceCandidates(list[CorpusSource]):
    def __init__(self, sources: list[CorpusSource], *, found: bool) -> None:
        super().__init__(sources)
        self.found = found
        self.lookups: list[str] = []

    def __call__(
        self, query: str, context: RunContext, **_kwargs: Any
    ) -> tuple[list[CorpusSource], bool]:
        context.check_active()
        self.lookups.append(query)
        assert query == "6455 sayılı Kanun"
        return ([self[1]] if self.found else []), False


def _install_source_boundary(
    monkeypatch: pytest.MonkeyPatch, broker: CorpusBoundary, *, found: bool
) -> tuple[str, str, _SourceCandidates]:
    lower, governing = configure_governing_basis_sources(broker)
    if found:
        broker.sources[1] = replace(broker.sources[1], name="6455 SAYILI KANUN")
        chunk = broker.chunks[governing]
        broker.chunks[governing] = replace(
            chunk,
            text="MADDE 11 — Faaliyet için izin alınır.",
            heading_path=("6455 SAYILI KANUN", "MADDE 11"),
            metadata={
                **chunk.metadata,
                "title": "6455 SAYILI KANUN",
                "article_no": "11",
            },
        )
    sources = _SourceCandidates(broker.sources, found=found)
    monkeypatch.setattr(broker, "sources", sources)

    def source(source_id: str, context: RunContext) -> CorpusSource:
        context.check_active()
        return next(row for row in sources if str(row.id) == source_id)

    def iter_chunks(source_id: str, context: RunContext, **kwargs: Any) -> ChunkScan:
        return ChunkScan(cast(CorpusBroker, broker), source_id, context, **kwargs)

    monkeypatch.setattr(broker, "source", source, raising=False)
    monkeypatch.setattr(broker, "iter_chunks", iter_chunks, raising=False)
    return lower, governing, sources


@pytest.mark.parametrize("accept_disclosure", [True, False])
def test_cropped_native_child_missing_original_has_one_attempt_and_integration(
    monkeypatch: pytest.MonkeyPatch, accept_disclosure: bool
) -> None:
    kwargs, broker, llm, checkpoints, _queue = setup_run(monkeypatch)
    kwargs.pop("test_language")
    kwargs.update(research_profile="normal", workflow_variant=ASV3_EXP_CROPPED_VARIANT)
    lower, _governing, sources = _install_source_boundary(
        monkeypatch, broker, found=False
    )
    harnesses = capture_harnesses(monkeypatch)
    rejected = _LOWER_BODY + "\n\n" + _FIXTURE["text"]
    steps = 0
    integration_packets: list[dict[str, Any]] = []

    def child(**arguments: Any) -> ModelResponse:
        nonlocal steps
        steps += 1
        if steps == 1:
            return response(
                calls=[
                    (
                        "read_source_range",
                        {"source_id": lower, "_need_id": "fixture_source_need_0"},
                    )
                ]
            )
        if steps == 2:
            return response(calls=[("submit_partial_answer", {"answer": rejected})])
        assert steps == 3
        payload = user_payload(arguments["prompt"][-1])
        checks = payload["publication_gap"]["cropped_evidence_checks"]
        assert len(checks) == 1
        assert checks[0]["status"] == "not_found"
        assert checks[0]["absence_proven"] is False
        assert checks[0]["evidence_numbers"] == []
        assert checks[0]["source_gap_sentence"] == _DISCLOSURE
        assert not any(isinstance(row, ToolMessage) for row in arguments["prompt"])
        assert {row["citation"] for row in delivered_originals(arguments)} == {1}
        assert {tool["function"]["name"] for tool in arguments["tools"]} == {
            "submit_retained_answer",
            "submit_retained_partial_answer",
        }
        units = payload["draft_to_repair"]["units"]
        assert units[0]["text"] == _LOWER_BODY + "\n\n"
        target = next(row for row in units if "6455" in row["text"])
        assert target["text"] == _FIXTURE["text"]
        integration_packets.append(payload)
        return response(
            calls=[
                (
                    "submit_retained_partial_answer",
                    {
                        "retained_answer_edits": [
                            {
                                "unit_id": target["unit_id"],
                                "replacement": _DISCLOSURE
                                if accept_disclosure
                                else target["text"],
                            }
                        ]
                    },
                )
            ]
        )

    coordinator = cropped_coordinator_with_needs([_TASK], child)

    def invoke(**arguments: Any) -> ModelResponse:
        assert (arguments.get("structured_response_format") or {}).get(
            "json_schema", {}
        ).get("name") != "_EvidenceCheck"
        payload = user_payload(arguments["prompt"][-1])
        if not accept_disclosure and payload["request"] != _TASK:
            tasks = payload.get("research_tasks", [])
            if tasks and tasks[0]["status"] == "completed":
                assert tasks[0]["outcome"]["status"] == "truncated"
                assert tasks[0]["outcome"]["summary"] == "Research incomplete"
                return response(
                    calls=[("submit_partial_answer", {"answer": _DISCLOSURE})]
                )
        return coordinator(**arguments)

    llm.invoke.side_effect = invoke
    runtime.run_asv3_loop(**kwargs)
    child_harness = next(row for row in harnesses if row.context.depth)
    assert steps == 3 and len(integration_packets) == 1
    assert sources.lookups == ["6455 sayılı Kanun"]
    reads = [
        receipt
        for receipt in child_harness.receipts
        if receipt.call.name == "read_named_provision"
    ]
    assert len(reads) == 1
    assert reads[0].call.arguments == {
        "article": "11",
        "source_name": "6455 sayılı Kanun",
    }
    assert reads[0].outcome.status == OutcomeStatus.NOT_FOUND
    assert reads[0].outcome.data["absence_proven"] is False
    assert child_harness.context.services.get("asv3_exp_cheap") is not True
    assert checkpoints[-1]["asv3_workflow_variant"] == ASV3_EXP_CROPPED_VARIANT
    state = checkpoints[-1]["cropped_repair_state"]
    assert len(state["records"]) == 1
    assert state["records"][0]["status"] == "not_found"
    if accept_disclosure:
        assert child_harness.partial_submission is not None
        assert child_harness.partial_submission() == _LOWER_BODY + "\n\n" + _DISCLOSURE
        assert child_harness.publication_gap is None
        assert child_harness.stop_reason == "model_requested_partial_publication"
    else:
        assert child_harness.partial_submission is not None
        assert child_harness.partial_submission() is None
        assert child_harness.publication_gap is not None
        assert "already received" in (child_harness.stop_reason or "")


def test_cropped_native_child_found_original_checker_is_blind_writer_delivers_union(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs, broker, llm, checkpoints, _queue = setup_run(monkeypatch)
    kwargs.pop("test_language")
    kwargs.update(research_profile="normal", workflow_variant=ASV3_EXP_CROPPED_VARIANT)
    lower, governing, sources = _install_source_boundary(
        monkeypatch, broker, found=True
    )
    harnesses = capture_harnesses(monkeypatch)
    owner_thread = threading.get_ident()
    root_captures: list[tuple[str, int]] = []
    original_snapshot = Harness.snapshot
    original_control = Harness.snapshot_control

    def snapshot(active: Harness) -> dict[str, Any]:
        if active.context.depth == 0:
            current_thread = threading.get_ident()
            root_captures.append(("snapshot", current_thread))
            assert current_thread == owner_thread
        return original_snapshot(active)

    def snapshot_control(active: Harness) -> dict[str, Any]:
        if active.context.depth == 0:
            current_thread = threading.get_ident()
            root_captures.append(("snapshot_control", current_thread))
            assert current_thread == owner_thread
        return original_control(active)

    monkeypatch.setattr(Harness, "snapshot", snapshot)
    monkeypatch.setattr(Harness, "snapshot_control", snapshot_control)
    rejected = (
        _LOWER_BODY + "\n\n6455 sayılı Kanun 11. maddesi uyarınca izin gerekir [1]."
    )
    accepted = (
        _LOWER_BODY + "\n\n6455 sayılı Kanun 11. maddesi uyarınca izin gerekir [2]."
    )
    steps = 0
    checks: list[dict[str, Any]] = []

    def child(**arguments: Any) -> ModelResponse:
        nonlocal steps
        steps += 1
        if steps == 1:
            return response(
                calls=[
                    (
                        "read_source_range",
                        {"source_id": lower, "_need_id": "fixture_source_need_0"},
                    )
                ]
            )
        if steps == 2:
            return response(
                calls=[("submit_answer", {"answer": rejected, "basis": "originals"})]
            )
        assert steps == 3
        payload = user_payload(arguments["prompt"][-1])
        records = payload["publication_gap"]["cropped_evidence_checks"]
        assert records[0]["status"] == "found" and records[0]["evidence_numbers"] == [2]
        originals = delivered_originals(arguments)
        assert {row["citation"] for row in originals} == {1, 2}
        assert {row["source_id"] for row in originals} == {lower, governing}
        assert all(
            row["text"] == broker.chunks[row["source_id"]].text for row in originals
        )
        assert not any(isinstance(row, ToolMessage) for row in arguments["prompt"])
        units = payload["draft_to_repair"]["units"]
        assert units[0]["text"] == _LOWER_BODY + "\n\n"
        target = next(row for row in units if "6455" in row["text"])
        return response(
            calls=[
                (
                    "submit_retained_answer",
                    {
                        "retained_answer_edits": [
                            {
                                "unit_id": target["unit_id"],
                                "replacement": accepted.split("\n\n")[1],
                            }
                        ]
                    },
                )
            ]
        )

    coordinator = cropped_coordinator_with_needs([_TASK], child)

    def invoke(**arguments: Any) -> ModelResponse:
        schema = (arguments.get("structured_response_format") or {}).get(
            "json_schema", {}
        )
        if schema.get("name") != "_EvidenceCheck":
            return coordinator(**arguments)
        payload = user_payload(arguments["prompt"][-1])
        assert set(payload) == {
            "request",
            "deficiency",
            "requested_check",
            "evidence",
            "research_results",
        }
        assert not {"unit_ids", "unit_texts", "details", "answer", "history"} & set(
            payload["deficiency"]
        )
        assert "draft_to_repair" not in payload
        requested = json.dumps(payload["requested_check"], ensure_ascii=False)
        assert rejected not in requested and _LOWER_BODY not in requested
        assert not any(
            key in requested
            for key in ("unit_texts", "draft_to_repair", '"answer"', '"history"')
        )
        originals = payload["evidence"]
        assert len(originals) == 1
        assert originals[0]["citation"] == 2
        assert originals[0]["source_id"] == governing
        assert originals[0]["text"] == broker.chunks[governing].text
        assert _LOWER_BODY not in json.dumps(payload, ensure_ascii=False)
        assert rejected not in json.dumps(payload, ensure_ascii=False)
        checks.append(payload)
        return response(
            json.dumps(
                {
                    "status": "found",
                    "finding": "Özgün hükümde izin koşulu okunmuştur.",
                    "witnesses": [
                        {"citation": 2, "source_quote": "Faaliyet için izin alınır."}
                    ],
                    "limitations": [],
                },
                ensure_ascii=False,
            )
        )

    llm.invoke.side_effect = invoke
    runtime.run_asv3_loop(**kwargs)
    child_harness = next(row for row in harnesses if row.context.depth)
    assert steps == 3 and len(checks) == 1
    assert sources.lookups == ["6455 sayılı Kanun"]
    assert child_harness.complete_submission is not None
    assert child_harness.complete_submission() == accepted
    assert child_harness.publication_gap is None
    delivery = child_harness.evidence.export()["deliveries"]
    checker_calls = [
        row
        for row in delivery
        if row["flow"] == LLMFlow.ASV3_TARGETED_EVIDENCE_CHECK.value
    ]
    writer_calls = [
        row for row in delivery if row["flow"] == LLMFlow.ASV3_ANSWER_REPAIR.value
    ]
    assert len(checker_calls) == len(writer_calls) == 1
    assert child_harness.evidence.completely_delivered(checker_calls[0]["call_id"]) == {
        2
    }
    assert child_harness.evidence.completely_delivered(writer_calls[0]["call_id"]) == {
        1,
        2,
    }
    assert root_captures
    assert any(name == "snapshot_control" for name, _thread in root_captures)
    assert all(thread == owner_thread for _name, thread in root_captures)
    saved_found: list[dict[str, Any]] = []
    for saved in checkpoints:
        for record in saved["cropped_repair_state"]["records"]:
            if record["status"] != "found":
                continue
            saved_found.append(saved)
            assert record["evidence_numbers"] == [2]
            checker_call = json.loads(record["detail"])["check_call_id"]
            canonical = next(
                row["item"]
                for row in saved["evidence"]["records"]
                if row["citation"] == 2
            )
            assert canonical["source_id"] == governing
            assert canonical["chunk_id"] == broker.chunks[governing].id
            assert canonical["text"] == broker.chunks[governing].text
            receipt = next(
                row
                for row in saved["evidence"]["deliveries"]
                if row["call_id"] == checker_call
            )
            assert receipt["flow"] == LLMFlow.ASV3_TARGETED_EVIDENCE_CHECK.value
            witness = next(row for row in receipt["records"] if row["citation"] == 2)
            assert witness["complete"] is True
            assert witness["source_id"] == canonical["source_id"]
            assert witness["chunk_id"] == canonical["chunk_id"]
            assert witness["text_hash"] == canonical["text_hash"]
    assert saved_found


def test_original_experimental_child_keeps_history_and_ordinary_repair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs, broker, llm, checkpoints, _queue = setup_run(monkeypatch)
    kwargs.pop("test_language")
    kwargs.update(research_profile="normal", workflow_variant=ASV3_TUNED_VARIANT)
    lower, _governing, sources = _install_source_boundary(
        monkeypatch, broker, found=False
    )
    harnesses = capture_harnesses(monkeypatch)
    rejected = _LOWER_BODY + "\n\n" + _FIXTURE["text"]
    steps = 0

    def child(**arguments: Any) -> ModelResponse:
        nonlocal steps
        steps += 1
        if steps == 1:
            return response(calls=[("read_source_range", {"source_id": lower})])
        if steps == 2:
            return response(calls=[("submit_partial_answer", {"answer": rejected})])
        assert steps == 3
        payload = user_payload(arguments["prompt"][-1])
        assert "cropped_evidence_checks" not in payload["publication_gap"]
        assert any(isinstance(row, ToolMessage) for row in arguments["prompt"])
        target = next(
            row for row in payload["draft_to_repair"]["units"] if "6455" in row["text"]
        )
        return response(
            calls=[
                (
                    "submit_retained_partial_answer",
                    {
                        "retained_answer_edits": [
                            {"unit_id": target["unit_id"], "replacement": _DISCLOSURE}
                        ]
                    },
                )
            ]
        )

    llm.invoke.side_effect = scripted_coordinator([_TASK], child)
    runtime.run_asv3_loop(**kwargs)
    child_harness = next(row for row in harnesses if row.context.depth)
    assert steps == 3 and sources.lookups == ["6455 sayılı Kanun"]
    assert (
        child_harness.context.services.get("tuned_governing_acquisitions") is not None
    )
    assert child_harness.repair_decision is None
    assert child_harness.context.services.get("asv3_exp_cropped") is not True
    assert "cropped_repair_state" not in checkpoints[-1]
    assert child_harness.partial_submission is not None
    assert child_harness.partial_submission() == _LOWER_BODY + "\n\n" + _DISCLOSURE


@pytest.mark.parametrize("with_prior_notice", [False, True])
def test_cropped_root_resume_preserves_exhausted_repair_without_model_or_read(
    monkeypatch: pytest.MonkeyPatch,
    with_prior_notice: bool,
) -> None:
    kwargs, broker, llm, checkpoints, _queue = setup_run(monkeypatch)
    kwargs.pop("test_language")
    kwargs.update(research_profile="normal", workflow_variant=ASV3_EXP_CROPPED_VARIANT)
    lower, _governing, sources = _install_source_boundary(
        monkeypatch, broker, found=False
    )
    page = MagicMock(wraps=broker.page)
    monkeypatch.setattr(broker, "page", page)
    harnesses = capture_harnesses(monkeypatch)
    prior = CroppedRepairTarget(
        check_kind="governing_original", instrument_number="6455", article="10"
    )
    previous_notice = (
        source_gap_sentence(prior, "not_found") + "\n\n" if with_prior_notice else ""
    )
    rejected = _LOWER_BODY + "\n\n" + previous_notice + _FIXTURE["text"]
    invocations = 0

    def invoke(**arguments: Any) -> ModelResponse:
        nonlocal invocations
        invocations += 1
        if invocations == 1:
            return response(
                calls=[
                    (
                        "update_research",
                        {
                            "needs": [
                                {
                                    "need_id": "fixture_source_need_0",
                                    "question_ids": ["q0"],
                                    "purpose": "Obtain the governing original for the requested result.",
                                    "completion_test": "Full governing provision or an honest unresolved original gap.",
                                    "material": True,
                                }
                            ]
                        },
                    ),
                    (
                        "read_source_range",
                        {
                            "source_id": lower,
                            "_language": "tr",
                            "_need_id": "fixture_source_need_0",
                        },
                    ),
                ]
            )
        if invocations == 2:
            if with_prior_notice:
                state = harnesses[-1].context.services["cropped_repair_state"]
                assert isinstance(state, CroppedRepairState)
                state.reserve(prior)
                state.finish(
                    prior.key, "not_found", [], "Earlier focused source attempt."
                )
                state.reserve_integration([prior], "coordinator")
            return response(calls=[("submit_partial_answer", {"answer": rejected})])
        assert invocations == 3
        payload = user_payload(arguments["prompt"][-1])
        checks = payload["publication_gap"]["cropped_evidence_checks"]
        assert checks[0]["status"] == "not_found"
        target = next(
            unit
            for unit in payload["draft_to_repair"]["units"]
            if _FIXTURE["text"] in unit["text"]
        )
        return response(
            calls=[
                (
                    "submit_retained_partial_answer",
                    {
                        "retained_answer_edits": [
                            {
                                "unit_id": target["unit_id"],
                                "replacement": target["text"],
                            }
                        ]
                    },
                )
            ]
        )

    llm.invoke.side_effect = invoke
    runtime.run_asv3_loop(**kwargs)
    assert invocations == 3
    assert sources.lookups == ["6455 sayılı Kanun"]
    assert page.call_count == 1
    assert checkpoints[-1]["publication_status"] == "partial"
    assert checkpoints[-1]["publication_stop_reason"] == "native_partial_published"
    previous = copy.deepcopy(
        next(saved for saved in reversed(checkpoints) if saved.get("publication_gap"))
    )
    assert previous["last_draft"] == rejected
    assert previous["publication_gap"]["data"]["named_authority_gaps"]
    saved_repair = previous["cropped_repair_state"]
    assert len(saved_repair["records"]) == (2 if with_prior_notice else 1)
    assert all(record["status"] == "not_found" for record in saved_repair["records"])
    assert len(saved_repair["integrations"]["coordinator"]) == (
        2 if with_prior_notice else 1
    )
    first = harnesses[-1]
    assert first.complete_submission is None
    assert first.partial_submission is not None
    assert (
        first.partial_submission()
        == _LOWER_BODY + "\n\n" + previous_notice + _DISCLOSURE
    )
    assert first.stop_reason == "model_requested_partial_publication"

    monkeypatch.setattr(runtime, "load_asv3_checkpoint", lambda **_kwargs: previous)
    kwargs["resume_message_id"] = kwargs["assistant_message_id"]
    llm.reset_mock()
    page.reset_mock()
    sources.lookups.clear()
    llm.invoke.side_effect = AssertionError("An exhausted repair cannot call the model")
    runtime.run_asv3_loop(**kwargs)
    resumed = harnesses[-1]
    assert resumed.context.run_id == first.context.run_id
    assert resumed.context.scope == first.context.scope
    assert resumed.request == first.request
    assert resumed.publication_gap is None
    assert resumed.stop_reason == "model_requested_partial_publication"
    assert resumed.partial_submission is not None
    assert resumed.partial_submission() == first.partial_submission()
    assert llm.invoke.call_count == 0 and page.call_count == 0
    assert sources.lookups == []
    assert len(resumed.receipts) >= len(first.receipts)
    assert [receipt.call.name for receipt in resumed.receipts].count(
        "read_named_provision"
    ) == 1
    assert checkpoints[-1]["cropped_repair_state"] == saved_repair
    assert checkpoints[-1]["publication_gap"] is None
    assert checkpoints[-1]["publication_status"] == "partial"
    assert checkpoints[-1]["asv3_workflow_variant"] == ASV3_EXP_CROPPED_VARIANT
    # Resume revalidates existing originals; it does not acquire or deliver them anew.
    assert {item.source_id for item in broker.revalidated} == {lower}
    assert (
        checkpoints[-1]["evidence"]["deliveries"] == previous["evidence"]["deliveries"]
    )
