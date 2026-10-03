"""Isolate answer-comparison tests from the separately tested source inventory stage."""

from typing import cast

import pytest
from pydantic import JsonValue

from onyx.asv3 import source_conditions
from onyx.asv3.condition_memory import SourceConditionMemory
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.llm_adapter import ResearchModel
from onyx.tracing.flows import LLMFlow


@pytest.fixture
def empty_source_inventory(monkeypatch: pytest.MonkeyPatch) -> None:
    def empty_inventory(
        model: ResearchModel,
        ledger: EvidenceLedger,
        payload: dict[str, JsonValue],
        questions: list[str],
        memory: SourceConditionMemory,
        *,
        consume_budget: bool,
    ) -> str:
        del model, questions, memory, consume_budget
        originals = payload["original_evidence"]
        assert isinstance(originals, list)
        receipt = "empty-source-inventory-fixture"
        ledger.record_delivery(
            receipt,
            LLMFlow.ASV3_SOURCE_INVENTORY.value,
            cast(list[dict[str, JsonValue]], originals),
        )
        return receipt

    monkeypatch.setattr(source_conditions, "retain_source_inventory", empty_inventory)
