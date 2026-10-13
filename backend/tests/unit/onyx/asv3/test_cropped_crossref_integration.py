"""Reference depth is enforced before tool I/O and retained through real receipts."""

import pytest
from pydantic import JsonValue

from onyx.asv3.cropped_crossrefs import CroppedCrossReferencePolicy
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.harness import Harness
from onyx.asv3.models import (
    CapabilityCall,
    Decision,
    OutcomeStatus,
    RunContext,
    ToolOutcome,
    ToolSpec,
)
from onyx.asv3.registry import CapabilityRegistry
from tests.unit.onyx.asv3.test_cropped_article_bindings import (
    ANSWER,
    OPTIONS,
    cropped_named_authority_gap,
    setup,
)
from tests.unit.onyx.asv3.test_native_authority import original


@pytest.mark.parametrize("cropped", [True, False])
def test_nested_reference_is_blocked_before_handler_without_stopping_harness(
    cropped: bool,
) -> None:
    context = RunContext()
    context.services["asv3_exp_cropped"] = cropped
    ledger = EvidenceLedger()
    context.services["evidence"] = ledger
    if cropped:
        context.services["cropped_cross_references"] = CroppedCrossReferencePolicy(
            context, "Başvuru için gereken şartlar nelerdir?"
        )
    root = original(
        "Başvuru Yönetmeliği",
        "1",
        kind="yönetmelik",
        text="8917 sayılı Faaliyet Kanunu'nun 27. maddesi uygulanır.",
    )
    first = original(
        "8917 sayılı Faaliyet Kanunu",
        "27",
        text="Başvuru için 8918 sayılı İzin Kanunu'nun 8. maddesi uygulanır.",
    )
    second = original("8918 sayılı İzin Kanunu", "8")
    executed: list[str] = []

    def handler(args: dict[str, JsonValue], _owner: RunContext) -> ToolOutcome:
        target = str(args.get("target_source_id", root.source_id))
        executed.append(target)
        item = {root.source_id: root, first.source_id: first, second.source_id: second}[
            target
        ]
        return ToolOutcome(
            status=OutcomeStatus.FOUND,
            summary="Actual canonical original",
            evidence=[item],
            data={"origin_chunk_id": args["chunk_id"]} if "chunk_id" in args else {},
        )

    registry = CapabilityRegistry(
        [
            ToolSpec(
                name=name,
                description="Read an original",
                parameters={"type": "object", "additionalProperties": True},
                handler=handler,
            )
            for name in ("read_source_range", "follow_reference")
        ]
    )
    harness = Harness(
        request="Başvuru için gereken şartlar nelerdir?",
        context=context,
        registry=registry,
        evidence=ledger,
        decide=lambda _: Decision(),
    )
    harness.dispatch_repair_reads(
        [CapabilityCall(name="read_source_range", arguments={})]
    )
    first_read = harness.dispatch_repair_reads(
        [
            CapabilityCall(
                name="follow_reference",
                arguments={
                    "source_id": root.source_id,
                    "chunk_id": root.chunk_id,
                    "target_source_id": first.source_id,
                    "article": "27",
                    "depth": 0,
                },
            )
        ]
    )
    assert first_read[0].outcome.status == OutcomeStatus.FOUND
    blocked = harness.dispatch_repair_reads(
        [
            CapabilityCall(
                name="follow_reference",
                arguments={
                    "source_id": first.source_id,
                    "chunk_id": first.chunk_id,
                    "target_source_id": second.source_id,
                    "article": "8",
                    "depth": 0,
                },
            )
        ]
    )
    assert blocked[0].outcome.status == (
        OutcomeStatus.DENIED if cropped else OutcomeStatus.FOUND
    )
    assert executed == (
        [root.source_id, first.source_id]
        if cropped
        else [root.source_id, first.source_id, second.source_id]
    )
    assert len(harness.receipts) == 3
    assert ledger.completely_delivered("not-a-model-call") == set()


def test_ordered_identity_proof_stays_out_of_model_receipts() -> None:
    context, ledger, state, receipt = setup()
    context.services["evidence"] = ledger
    before = ledger.export()

    def read(_args: dict[str, JsonValue], _context: RunContext) -> ToolOutcome:
        return receipt.outcome.model_copy(deep=True)

    registry = CapabilityRegistry(
        [
            ToolSpec(
                name=receipt.call.name,
                description="Read full canonical originals in source order",
                parameters={"type": "object", "additionalProperties": True},
                handler=read,
            )
        ]
    )
    harness = Harness(
        request="Question",
        context=context,
        registry=registry,
        evidence=ledger,
        decide=lambda _: Decision(),
    )
    committed = harness.dispatch_repair_reads([receipt.call])[0]
    assert "cropped_ordered_source_read" not in committed.outcome.data
    assert state.export()["reads"]
    assert cropped_named_authority_gap(ANSWER, ledger, context, **OPTIONS) is None
    assert ledger.export() == before
