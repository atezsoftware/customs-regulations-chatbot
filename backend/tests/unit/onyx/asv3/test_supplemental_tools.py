from pydantic import JsonValue

from onyx.asv3.harness import Harness
from onyx.asv3.models import (
    CapabilityCall,
    Decision,
    OutcomeStatus,
    RunContext,
    ToolOutcome,
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
