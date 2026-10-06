"""Resource orchestration must not hide a source action's public update field."""

from typing import cast

import jsonschema
import pytest
from pydantic import JsonValue

from onyx.asv3.corpus_tools import build_corpus_specs
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.llm_adapter import ResearchModel
from onyx.asv3.models import OutcomeStatus, RunContext
from onyx.asv3.registry import CapabilityRegistry, build_core_specs
from onyx.asv3.sandbox import build_sandbox_specs
from tests.unit.onyx.asv3.test_native_model_adapter import native_action
from tests.unit.onyx.asv3.test_tuned_focused_reads import NamedBroker


@pytest.mark.parametrize("name", ["search_corpus", "read_named_provision"])
def test_orchestrated_source_update_is_exposed_validated_and_dispatchable(
    name: str,
) -> None:
    broker = NamedBroker()
    context = RunContext(services={"lean_native_mode": True})
    registry = CapabilityRegistry(
        build_corpus_specs(broker, named_provision_reads=True)
    )
    arguments: dict[str, JsonValue] = (
        {"source_name": "Example Law", "article": "7"}
        if name == "read_named_provision"
        else {
            "query": "Example Law article 7 authorization",
            "mode": "hybrid",
            "coverage_item": "Authorization conditions",
            "evidence_target": "Example Law article 7 authorization conditions",
        }
    )
    arguments["_public_update"] = ["Hüküm inceleniyor", "İzin koşullarını inceliyorum."]
    tools = registry.definitions(context)
    decision = ResearchModel._decision(native_action(name, arguments), tools)
    outcome = registry.dispatch(decision.calls[0], context)
    assert outcome.status == OutcomeStatus.FOUND
    assert broker.acquisitions == 1
    assert [item.text for item in outcome.evidence] == [
        row.text for row in broker.items[:2]
    ]
    schema = next(
        cast(dict[str, JsonValue], tool["function"])["parameters"]
        for tool in tools
        if cast(dict[str, JsonValue], tool["function"])["name"] == name
    )
    validator = jsonschema.Draft202012Validator(schema)
    for update in (["Only one entry"], ["title", 2], "text"):
        with pytest.raises(jsonschema.ValidationError):
            validator.validate({**arguments, "_public_update": update})
    with pytest.raises(jsonschema.ValidationError):
        validator.validate({**arguments, "unknown_field": "value"})


def test_protected_catalogue_and_composition_wrapper_keep_existing_update_policy() -> (
    None
):
    context = RunContext(services={"lean_native_mode": True})
    broker = NamedBroker()
    registry = CapabilityRegistry(
        [*build_corpus_specs(broker), *build_sandbox_specs(broker)]
    )
    ledger = EvidenceLedger()
    for spec in build_core_specs(registry, ledger, lambda: {}):
        registry.register(spec)
    schemas = {
        cast(dict[str, JsonValue], tool["function"])["name"]: cast(
            dict[str, JsonValue],
            cast(dict[str, JsonValue], tool["function"])["parameters"],
        )["properties"]
        for tool in registry.definitions(context)
    }
    assert "read_named_provision" not in schemas
    assert "_public_update" in cast(dict[str, JsonValue], schemas["search_corpus"])
    assert "_public_update" not in cast(
        dict[str, JsonValue], schemas["compose_tool_calls"]
    )
