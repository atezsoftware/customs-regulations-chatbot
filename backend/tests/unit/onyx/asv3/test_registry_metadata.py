"""Check native catalogue compaction without changing callable capabilities."""

import copy
from typing import Any, cast

import jsonschema
import pytest
from pydantic import JsonValue

from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import CapabilityCall, EvidenceItem, OutcomeStatus, RunContext
from onyx.asv3.registry import CapabilityRegistry, build_core_specs
from tests.unit.onyx.asv3.test_runtime import run_independent, setup_run


def test_experimental_read_hint_preserves_schema_executor_and_other_profiles() -> None:
    ledger = EvidenceLedger()
    context = RunContext(
        services={"lean_native_mode": True, "research_profile": "normal"}
    )
    ledger.add(
        [EvidenceItem(source_id="law", chunk_id="rule", text="Full rule.")], context
    )
    registry = CapabilityRegistry()
    for spec in build_core_specs(registry, ledger, lambda: {}):
        registry.register(spec)
    spec = registry.get("read_evidence")
    assert spec is not None
    base = copy.deepcopy(spec.definition())
    definitions: dict[str, dict[str, JsonValue]] = {}
    for profile in ("normal", "deep", "experimental"):
        context.services["research_profile"] = profile
        definition = next(
            row["function"]
            for row in registry.definitions(context)
            if isinstance(row["function"], dict)
            and row["function"]["name"] == "read_evidence"
        )
        assert isinstance(definition, dict)
        definitions[profile] = definition
    assert definitions["normal"] == definitions["deep"]
    experimental = copy.deepcopy(definitions["experimental"])
    description = experimental.pop("description")
    normal = copy.deepcopy(definitions["normal"])
    normal_description = normal.pop("description")
    assert experimental == normal
    assert "absent or truncated" in str(description)
    assert "original_evidence_ranges" in str(description)
    assert "missing continuation" in str(description)
    assert normal_description == spec.description
    assert spec.definition() == base
    # Delivery is a reuse hint, not a new restriction on a requested missing range.
    reopened = registry.dispatch(
        CapabilityCall(
            name="read_evidence",
            arguments={"citation": 1, "start_char": 5, "num_chars": 2},
        ),
        context,
    )
    assert reopened.status == OutcomeStatus.TRUNCATED
    assert reopened.data["text"] == "ru"
    assert reopened.original_reads[0].start_char == 5
    assert reopened.original_reads[0].end_char == 7


def test_native_catalogue_preserves_every_capability_and_validation_without_repeating_metadata_descriptions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs, broker, llm, _checkpoints, _queue = setup_run(monkeypatch)
    definitions = CapabilityRegistry.definitions
    examined: list[int] = []

    def compare(
        registry: CapabilityRegistry, context: RunContext
    ) -> list[dict[str, JsonValue]]:
        native = definitions(registry, context)
        legacy_context = context.child()
        legacy_context.services.pop("lean_native_mode", None)
        legacy = definitions(registry, legacy_context)
        native_tools = cast(list[dict[str, Any]], native)
        legacy_tools = cast(list[dict[str, Any]], legacy)
        assert len(native) == len(legacy)
        if context.depth == 0:
            assert len(native) == 39
        assert [row["function"]["name"] for row in native_tools] == [
            row["function"]["name"] for row in legacy_tools
        ]
        for native_row, legacy_row in zip(native, legacy, strict=True):
            lean_copy = copy.deepcopy(native_row)
            legacy_copy = copy.deepcopy(legacy_row)
            lean_properties = cast(dict[str, Any], lean_copy)["function"]["parameters"][
                "properties"
            ]
            legacy_properties = cast(dict[str, Any], legacy_copy)["function"][
                "parameters"
            ]["properties"]
            for name in (
                "_language",
                "_notifications",
                "_external_requested",
                "_outcomes",
                "_coverage",
            ):
                assert name in lean_properties
                assert "description" not in lean_properties.pop(name)
            for name in ("_need_id", "_public_update"):
                if name in legacy_properties:
                    assert "description" in legacy_properties[name]
                    assert "description" not in lean_properties[name]
                    legacy_properties[name].pop("description")
            assert lean_copy == legacy_copy
        source_tool = next(
            row
            for row in native_tools
            if row["function"]["name"] == "read_source_range"
        )
        schema = source_tool["function"]["parameters"]
        validator = jsonschema.Draft202012Validator(schema)
        supplied = {
            "source_id": str(broker.sources[0].id),
            "_language": "tr",
            "_notifications": {"completed": ["Yanıt hazır", "Yanıt tamamlandı."]},
            "_external_requested": False,
            "_need_id": "optional-need",
            "_public_update": ["Hüküm inceleniyor", "Tamir koşullarını inceliyorum."],
        }
        validator.validate(supplied)
        for name, invalid in (
            ("_language", "not_a_language"),
            ("_notifications", {"completed": ["Only one entry"]}),
            ("_external_requested", "true"),
            ("_need_id", ""),
            ("_public_update", ["Only one entry"]),
            ("unexpected_field", "invented"),
        ):
            with pytest.raises(jsonschema.ValidationError):
                validator.validate({**supplied, name: invalid})
        examined.append(len(native))
        return native

    monkeypatch.setattr(CapabilityRegistry, "definitions", compare)
    run_independent(**kwargs)
    assert examined and 39 in examined
