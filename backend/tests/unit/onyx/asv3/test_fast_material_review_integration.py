"""Fast retained repair and the real runtime reviewer keep their source boundaries."""

import copy
import json
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from pydantic import JsonValue

from onyx.asv3 import runtime
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.fast_material_review import FastMaterialReviewResult
from onyx.asv3.llm_adapter import ResearchModel
from onyx.asv3.models import RunContext
from onyx.asv3.retained_answer import bind_retained_answer, resolve_retained_answer
from onyx.asv3.workflow_variant import ASV3_FAST_VARIANT
from onyx.llm.interfaces import LLM
from onyx.llm.model_response import ModelResponse
from onyx.prompts.asv3.fast import FAST_TEK_CONDITION_REVIEW
from onyx.regulatory.source_notices import SourceNotice
from onyx.tracing.flows import LLMFlow
from tests.unit.onyx.asv3.test_retained_answer import decision, tools
from tests.unit.onyx.asv3.test_runtime import response, setup_run, user_payload

BODY = (
    "  Unaffected supported result [1].\n\n"
    "Either silence or written consent suffices [1]. \n\n"
    "The separate ten-day deadline remains [2].  \n"
)
REQUEST = "Explain this procedure and its deadlines."


@pytest.mark.parametrize("defect", [None, "stale", "foreign"])
def test_fast_retained_unit_repair_keeps_unaffected_text_and_rejects_other_units(
    defect: str | None,
) -> None:
    context = RunContext(
        run_id="owned-fast-run",
        scope={"tenant": "A"},
        services={"asv3_fast": True, "research_profile": "normal"},
    )
    _, descriptor = bind_retained_answer(tools(), context, BODY, request=REQUEST)
    assert descriptor is not None
    units = cast(list[dict[str, JsonValue]], descriptor["units"])
    assert len(units) == 3
    selected = units[1]
    if defect:
        other_context = RunContext(
            run_id="foreign-run" if defect == "foreign" else context.run_id,
            scope=context.scope,
            services=context.services,
        )
        _, other = bind_retained_answer(
            tools(),
            other_context,
            BODY.replace("Either silence", "An earlier version")
            if defect == "stale"
            else BODY,
            request=REQUEST,
        )
        assert other is not None
        selected = cast(list[dict[str, JsonValue]], other["units"])[1]
    emitted = decision(
        {
            "retained_answer_edits": [
                {
                    "unit_id": selected["unit_id"],
                    "replacement": "Written consent AND no objection are required [1].",
                }
            ]
        },
        name="submit_retained_answer",
    )
    wire = emitted.model_dump_json()
    repaired = resolve_retained_answer(emitted, context, BODY, request=REQUEST)
    assert emitted.model_dump_json() == wire
    call = repaired.calls[0]
    if defect:
        assert call.argument_error and "answer" not in call.arguments
    else:
        assert call.name == "submit_answer" and call.argument_error is None
        assert call.arguments["basis"] == "originals"
        assert call.arguments["answer"] == BODY.replace(
            "Either silence or written consent suffices",
            "Written consent AND no objection are required",
        )


@pytest.mark.parametrize(
    "flag,depth,cropped,enabled",
    [
        (True, 0, False, True),
        (False, 0, False, False),
        (1, 0, False, False),
        ("true", 0, False, False),
        (True, 1, False, False),
        (True, 0, True, False),
    ],
)
def test_standard_retained_repair_requires_explicit_fast_root(
    flag: object, depth: int, cropped: bool, enabled: bool
) -> None:
    context = RunContext(
        depth=depth,
        services={"asv3_fast": flag, "asv3_exp_cropped": cropped},
    )
    _, descriptor = bind_retained_answer(tools(), context, BODY, request=REQUEST)
    assert (descriptor is not None) is enabled


@pytest.mark.usefixtures("empty_source_inventory")
@pytest.mark.parametrize("defect", [None, "missing_delivery", "provider_error"])
def test_runtime_material_reviewer_charges_shared_budget_and_isolates_writer_id(
    monkeypatch: pytest.MonkeyPatch, defect: str | None
) -> None:
    kwargs, broker, primary, checkpoints, _queue = setup_run(monkeypatch)
    kwargs.pop("test_language")
    kwargs.update(research_profile="normal", workflow_variant=ASV3_FAST_VARIANT)

    def notices(
        _queries: list[str] | None,
        _identifiers: list[str] | None,
        context: RunContext,
    ) -> list[SourceNotice]:
        context.check_active()
        return []

    monkeypatch.setattr(broker, "source_notices", notices, raising=False)
    secondary = MagicMock(spec=LLM)
    secondary.config = primary.config.model_copy()
    kwargs["research_llm"] = secondary
    adapters: list[ResearchModel] = []
    constructor = runtime.ResearchModel

    def create(*args: Any, **options: Any) -> ResearchModel:
        adapter = constructor(*args, **options)
        adapters.append(adapter)
        return adapter

    monkeypatch.setattr(runtime, "ResearchModel", create)
    primary_calls = 0
    candidate = "Tamir sonucu [1]; ayrı koşul [2]."

    def primary_response(**_arguments: Any) -> ModelResponse:
        nonlocal primary_calls
        primary_calls += 1
        if primary_calls == 1:
            return response(
                calls=[
                    ("read_source_range", {"source_id": str(s.id)})
                    for s in broker.sources
                ]
            )
        assert primary_calls == 2
        return response(
            calls=[("submit_answer", {"answer": candidate, "basis": "originals"})]
        )

    primary.invoke.side_effect = primary_response
    captured: list[dict[str, Any]] = []

    def secondary_response(**arguments: Any) -> ModelResponse:
        prompt = arguments["prompt"]
        if not str(prompt[0].content).startswith(FAST_TEK_CONDITION_REVIEW):
            return response(calls=[("request_primary_answer", {})])
        payload = user_payload(prompt[-1])
        root, checker = adapters[0], adapters[-1]
        assert root is not checker and root.context is not checker.context
        assert root.context.services is not checker.context.services
        assert root.context.budget is checker.context.budget
        writer_id = root.context.services["last_model_call_id"]
        assert "last_model_call_id" not in checker.context.services
        assert payload["candidate"] == candidate
        assert payload["required_evidence_numbers"] == [1, 2]
        assert {r["text"] for r in payload["original_evidence"]} == {
            c.text for c in broker.chunks.values()
        }
        captured.append(
            {
                "writer_id": writer_id,
                "root": root,
                "checker": checker,
                "payload": copy.deepcopy(payload),
            }
        )
        if defect == "provider_error":
            raise RuntimeError("checker unavailable")
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

    secondary.invoke.side_effect = secondary_response
    record_delivery = EvidenceLedger.record_delivery

    def delivered(
        ledger: EvidenceLedger,
        call_id: str,
        flow: str,
        records: list[dict[str, JsonValue]],
    ) -> None:
        if defect != "missing_delivery" or flow != LLMFlow.ASV3_VERIFICATION.value:
            record_delivery(ledger, call_id, flow, records)

    monkeypatch.setattr(EvidenceLedger, "record_delivery", delivered)
    invoke_text = ResearchModel.invoke_text
    budget_changes: list[int] = []

    def assess(adapter: ResearchModel, *args: Any, **options: Any) -> str:
        before = adapter.context.budget.snapshot()["decisions"]
        try:
            return invoke_text(adapter, *args, **options)
        finally:
            if options.get("response_model_override") is FastMaterialReviewResult:
                budget_changes.append(
                    adapter.context.budget.snapshot()["decisions"] - before
                )

    monkeypatch.setattr(ResearchModel, "invoke_text", assess)
    runtime.run_asv3_loop(**kwargs)
    published = kwargs["state_container"].get_answer_tokens() or ""
    if defect:
        # Harness records the stop and runtime may emit its ordinary failure notice.
        assert "Tamir sonucu" not in published
        assert (
            "material-condition review is unverified" in checkpoints[-1]["stop_reason"]
        )
    else:
        assert "Tamir sonucu" in published
    assert len(captured) == 1 and budget_changes == [1]
    snapshot = captured[0]
    root, checker = snapshot["root"], snapshot["checker"]
    assert root.context.services["last_model_call_id"] == snapshot["writer_id"]
    assert root.last_call_id == snapshot["writer_id"]
    ledger = root.context.services["evidence"]
    assert isinstance(ledger, EvidenceLedger)
    assert ledger.completely_delivered(root.last_call_id or "") == {1, 2}
    if defect is None:
        assert checker.last_call_id != root.last_call_id
        assert ledger.completely_delivered(checker.last_call_id or "") == {1, 2}
