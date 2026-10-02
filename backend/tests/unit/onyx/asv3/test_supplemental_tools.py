from pydantic import JsonValue

from onyx.asv3.harness import Harness
from onyx.asv3.models import (
    CapabilityCall,
    Decision,
    OutcomeStatus,
    RunContext,
    ToolOutcome,
    ToolSpec,
)
from onyx.asv3.progress import ProgressReporter
from onyx.asv3.registry import CapabilityRegistry
from onyx.asv3.supplemental_tools import ScenarioState, build_supplemental_specs


def test_scenario_recording_retains_counterfactuals_in_coordinator_view() -> None:
    state = ScenarioState()
    context = RunContext(services={"scenario_state": state})
    registry = CapabilityRegistry(build_supplemental_specs())
    result = registry.dispatch(
        CapabilityCall(
            name="record_scenario",
            arguments={
                "questions": ["free repair", "paid repair"],
                "facts": ["standard exchange permission absent"],
            },
        ),
        context,
    )
    assert result.status == OutcomeStatus.FOUND
    harness = Harness(
        request="repair",
        context=context,
        registry=registry,
        decide=lambda _view: Decision(answer="answer"),
    )
    assert harness.view().questions == ["free repair", "paid repair"]
    assert harness.view().facts == ["standard exchange permission absent"]


def test_question_typography_does_not_duplicate_but_changed_amount_does() -> None:
    state = ScenarioState()
    original = "Firma “başka işlem gerekmez” diyebilir mi? 100.000 TL’nin tamamı mı?"
    state.record([original], [])
    state.record(
        ["Firma 'başka işlem gerekmez' diyebilir mi? 100.000 TL'nin tamamı mı?"], []
    )
    state.record([original.replace("100.000", "500.000")], [])
    assert state.snapshot()["questions"] == [
        original,
        original.replace("100.000", "500.000"),
    ]


def test_same_action_narration_is_public_only_and_provider_ids_stay_private() -> None:
    from onyx.asv3.progress import public_action_id

    received: list[dict[str, JsonValue]] = []
    reporter = ProgressReporter("run", "tr")
    context = RunContext(run_id="run", language="tr")
    registry = CapabilityRegistry(
        [
            ToolSpec(
                name="read_source",
                description="read",
                parameters={
                    "type": "object",
                    "properties": {"source": {"type": "string"}},
                    "required": ["source"],
                    "additionalProperties": False,
                },
                handler=lambda args, _ctx: (
                    received.append(args)
                    or ToolOutcome(status=OutcomeStatus.FOUND, summary="Original read")
                ),
            )
        ]
    )
    harness = Harness(
        request="İzin gerekir mi?",
        context=context,
        registry=registry,
        decide=lambda _view: Decision(answer="done"),
        progress=reporter,
    )
    pair = [
        "İzin koşullarını inceliyorum",
        "Başvuru hükmünün şartlarını ve istisnalarını kontrol ediyorum.",
    ]
    opaque_id = "call__thought__" + "private-signature" * 1000
    call = CapabilityCall(
        name="read_source",
        call_id=opaque_id,
        arguments={"source": "law", "_public_update": pair},
    )
    harness._dispatch([call])
    assert received == [{"source": "law"}]
    assert harness.receipts[0].call.call_id == opaque_id
    lifecycle = [event for event in reporter.snapshot() if event.task_id]
    assert [event.status for event in lifecycle] == ["running", "completed"]
    assert all(
        event.task_id == public_action_id(opaque_id)
        and event.title == pair[0]
        and event.message == pair[1]
        for event in lifecycle
    )
    assert opaque_id not in reporter.export().__str__()


def test_public_narration_rejects_toolnames_but_accepts_case_findings() -> None:
    reporter = ProgressReporter("run", "tr")
    context = RunContext(services={"progress": reporter})
    registry = CapabilityRegistry(build_supplemental_specs())
    invalid = registry.dispatch(
        CapabilityCall(
            name="report_progress",
            arguments={
                "title": "Araştırma",
                "message": "read_provision aracını çağırıyorum",
            },
        ),
        context,
    )
    assert invalid.status == OutcomeStatus.INVALID
    assert reporter.snapshot() == []
    result = registry.dispatch(
        CapabilityCall(
            name="report_progress",
            arguments={
                "title": "Tamir ve değişim ayrımı",
                "message": "Yeni makine gönderilmesiyle aynı makinenin tamir edilmesi farklı şartlara bağlı; eski makinenin vergileri için ayrı yolu kontrol ediyorum.",
            },
        ),
        context,
    )
    assert result.status == OutcomeStatus.FOUND
    assert reporter.snapshot()[0].message.startswith("Yeni makine")


def test_claim_verification_calls_injected_verifier_and_skill_is_not_evidence() -> None:
    called: list[dict[str, JsonValue]] = []

    def verify(args: dict[str, JsonValue], _context: RunContext) -> ToolOutcome:
        called.append(args)
        return ToolOutcome(
            status=OutcomeStatus.PARTIAL,
            summary="Required conjunctive condition is missing",
        )

    context = RunContext(services={"verify_claim": verify})
    registry = CapabilityRegistry(build_supplemental_specs())
    result = registry.dispatch(
        CapabilityCall(
            name="verify_claim",
            arguments={"claim": "silence alone permits destruction", "citations": [1]},
        ),
        context,
    )
    assert result.status == OutcomeStatus.PARTIAL
    assert called[0]["citations"] == [1]
    skill = registry.dispatch(
        CapabilityCall(name="load_skill", arguments={"name": "legal_conditions"}),
        context,
    )
    assert skill.data["legal_authority"] is False
    assert len(str(skill.data["sha256"])) == 64
    assert skill.evidence == []


def test_public_narration_checks_dynamically_registered_capability_names() -> None:
    reporter = ProgressReporter("run", "en")
    registry = CapabilityRegistry(build_supplemental_specs())
    registry.register(
        ToolSpec(
            name="read_source_range",
            description="Read original range",
            parameters={"type": "object", "properties": {}},
            handler=lambda _args, _ctx: ToolOutcome(
                status=OutcomeStatus.FOUND, summary="Read"
            ),
        )
    )
    context = RunContext(services={"progress": reporter, "registry": registry})
    result = registry.dispatch(
        CapabilityCall(
            name="report_progress",
            arguments={
                "title": "Research",
                "message": "Calling read_source_range now.",
            },
        ),
        context,
    )
    assert result.status == OutcomeStatus.INVALID
    assert reporter.snapshot() == []
