"""Lossless Cheap storage and hostile expansion checks through the durable codec."""

import copy
import json
import random
from typing import cast

import pytest
from pydantic import JsonValue

from onyx.asv3 import exp_cheap_checkpoint as codec
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.harness import Harness
from onyx.asv3.models import Decision, ResearchTurn, RunContext
from onyx.asv3.registry import CapabilityRegistry
from onyx.asv3.workflow_variant import ASV3_EXP_CHEAP_POLICY, ASV3_EXP_CHEAP_VARIANT
from onyx.db import asv3_runs
from onyx.llm.models import AssistantMessage, FunctionCall, ToolCall, ToolMessage
from tests.unit.onyx.asv3.test_parallel_answers import original


def snapshot() -> dict[str, JsonValue]:
    literal = 'MADDE 10/A — özgün koşul, istisna ve kapsam. İğşç 東京\n"\\' * 200
    return {
        "version": 1,
        "run_id": "run-cheap",
        "sequence": 7,
        "asv3_workflow_variant": ASV3_EXP_CHEAP_VARIANT,
        "asv3_workflow_policy": ASV3_EXP_CHEAP_POLICY,
        "research_profile": "normal",
        "parallel_research": False,
        "scope": {"document_set_ids": [15], "user_id": "owner", "as_of": None},
        "evidence": {"records": [{"citation": 1, "text": literal}]},
        "last_draft": literal,
        "turns": [{"results": [{"content": literal}]}],
        "progress": [{"sequence": 1, "message": "Metni okuyorum"}],
    }


def packed() -> dict[str, JsonValue]:
    value = codec.compact_exp_cheap_checkpoint(snapshot())
    assert codec.EXP_CHEAP_CHECKPOINT_STORAGE_KEY in value
    return value


def storage(value: dict[str, JsonValue]) -> dict[str, JsonValue]:
    result = value[codec.EXP_CHEAP_CHECKPOINT_STORAGE_KEY]
    assert isinstance(result, dict)
    return result


def test_exact_unicode_json_types_and_reference_shaped_user_data() -> None:
    source = snapshot()
    source["arbitrary"] = {
        "o": [[0]],
        "a": [[1]],
        "$ref": "literal user value",
        "scalars": [None, True, False, 0, -0.0, 1e-100, [], {}, ""],
    }
    expected = copy.deepcopy(source)
    encoded = asv3_runs.encode_asv3_checkpoint(source)
    restored = asv3_runs.decode_asv3_checkpoint(encoded)
    assert source == expected == restored
    assert json.dumps(restored, ensure_ascii=False) == json.dumps(
        source, ensure_ascii=False
    )
    assert asv3_runs.checkpoint_progress_packets(encoded) == source["progress"]


def test_repeated_native_turns_over_raw_limit_remain_exact_and_independent() -> None:
    source = snapshot()
    evidence = cast(dict[str, JsonValue], source["evidence"])
    result = json.dumps({"original_evidence": evidence["records"]}, ensure_ascii=False)
    source["turns"] = [
        {"assistant": {"call_id": f"read-{index}"}, "results": [{"content": result}]}
        for index in range(900)
    ]
    raw_bytes = len(json.dumps(source, ensure_ascii=False).encode())
    assert raw_bytes > asv3_runs.MAX_DECODED_CHECKPOINT_BYTES
    compact = codec.compact_exp_cheap_checkpoint(source)
    assert len(json.dumps(compact, ensure_ascii=False).encode()) < 100_000
    encoded = asv3_runs.encode_asv3_checkpoint(source)
    assert len(encoded.encode()) < asv3_runs.MAX_CHECKPOINT_BYTES
    restored = asv3_runs.decode_asv3_checkpoint(encoded)
    assert restored == source
    turns = cast(list[dict[str, JsonValue]], restored["turns"])
    first = cast(list[dict[str, JsonValue]], turns[0]["results"])
    first[0]["content"] = "changed owned container"
    assert cast(list[dict[str, JsonValue]], turns[1]["results"])[0]["content"] == result
    assert cast(list[dict[str, JsonValue]], source["turns"])[0] != turns[0]


def test_small_snapshot_keeps_raw_storage_without_pool_overhead() -> None:
    source = snapshot()
    source["evidence"] = {"records": []}
    source["last_draft"] = "short"
    source["turns"] = []
    compact = codec.compact_exp_cheap_checkpoint(source)
    assert compact == source
    assert codec.EXP_CHEAP_CHECKPOINT_STORAGE_KEY not in compact
    assert (
        asv3_runs.decode_asv3_checkpoint(asv3_runs.encode_asv3_checkpoint(source))
        == source
    )


def harness_fixture() -> tuple[Harness, dict[str, JsonValue]]:
    context = RunContext(run_id="owned-cheap", scope={"user_id": "owner"})
    context.services["lean_native_mode"] = True
    evidence = EvidenceLedger()
    text = "Özgün hükmün bütün koşul ve istisnaları aynen saklanır.\n" * 25_000
    evidence.add([original(text)], context)
    evidence.record_delivery(
        "read-original", "asv3_coordinator", [{"citation": 1, "text": text}]
    )
    evidence.pin_delivery("read-original")
    evidence.include([1])
    harness = Harness(
        request="Koşul ve istisnaları açıkla.",
        context=context,
        registry=CapabilityRegistry(),
        decide=lambda _: Decision(answer="Koşullu sonuç [1]."),
        evidence=evidence,
    )
    body = json.dumps({"original_evidence": [{"citation": 1, "text": text}]})
    harness.turns = [
        ResearchTurn(
            assistant=AssistantMessage(
                tool_calls=[
                    ToolCall(
                        id=f"read-{index}",
                        function=FunctionCall(name="read_evidence", arguments="{}"),
                    )
                ]
            ),
            results=[ToolMessage(content=body, tool_call_id=f"read-{index}")],
        )
        for index in range(6)
    ]
    harness.last_draft = "Koşullu sonuç; istisna ayrıca değerlendirilir [1]."
    source: dict[str, JsonValue] = {
        **harness.snapshot(),
        "asv3_workflow_variant": ASV3_EXP_CHEAP_VARIANT,
        "asv3_workflow_policy": ASV3_EXP_CHEAP_POLICY,
        "research_profile": "normal",
        "parallel_research": False,
        "scope": context.scope,
    }
    return harness, source


def test_actual_harness_evidence_deliveries_and_native_turns_recover_exactly() -> None:
    harness, source = harness_fixture()
    context = harness.context
    assert len(json.dumps(source, ensure_ascii=False).encode()) > 8_000_000
    restored = asv3_runs.decode_asv3_checkpoint(
        asv3_runs.encode_asv3_checkpoint(source)
    )
    assert restored == source
    recovered_context = RunContext(run_id=context.run_id, scope=context.scope)
    recovered_context.services["lean_native_mode"] = True
    recovered = Harness(
        request=harness.request,
        context=recovered_context,
        registry=CapabilityRegistry(),
        decide=lambda _: Decision(answer="Unused"),
    )
    recovered.restore(restored)
    assert recovered.snapshot() == harness.snapshot()
    assert recovered.evidence.completely_delivered("read-original") == {1}
    assert recovered.last_draft == harness.last_draft
    foreign = Harness(
        request=harness.request,
        context=RunContext(run_id="foreign", scope=context.scope),
        registry=CapabilityRegistry(),
        decide=lambda _: Decision(answer="Unused"),
    )
    with pytest.raises(ValueError, match="run identity mismatch"):
        foreign.restore(restored)


@pytest.mark.parametrize(
    "variant",
    [
        "standard",
        "asv3_tuned",
        "asv3_guarded_experimental",
        "asv3_guardrails_v2",
        "asv3_guardrails_v3",
    ],
)
def test_other_modes_never_invoke_cheap_pooling(
    monkeypatch: pytest.MonkeyPatch, variant: str
) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("Cheap codec entered another workflow")

    monkeypatch.setattr(asv3_runs, "compact_exp_cheap_checkpoint", forbidden)
    monkeypatch.setattr(asv3_runs, "restore_exp_cheap_checkpoint", forbidden)
    source = {**snapshot(), "asv3_workflow_variant": variant}
    encoded = asv3_runs.encode_asv3_checkpoint(source)
    assert asv3_runs.decode_asv3_checkpoint(encoded) == source


@pytest.mark.parametrize(
    "field,value",
    [
        ("asv3_workflow_variant", "asv3_tuned"),
        ("asv3_workflow_policy", "tuned-copy-cache-naming-v1"),
        ("research_profile", "experimental"),
        ("parallel_research", True),
        ("parallel_research", None),
    ],
)
def test_pooled_storage_requires_exact_cheap_identity(
    field: str, value: JsonValue
) -> None:
    source = snapshot()
    source[field] = value
    with pytest.raises(ValueError, match="own normal-profile policy"):
        codec.compact_exp_cheap_checkpoint(source)
    if field.startswith("asv3_"):
        saved = packed()
        saved[field] = value
        with pytest.raises(ValueError, match="Invalid Cheap checkpoint storage"):
            codec.restore_exp_cheap_checkpoint(saved)


@pytest.mark.parametrize(
    "field,value",
    [
        ("version", 2),
        ("version", True),
        ("expanded_bytes", True),
        ("expanded_bytes", 0),
        ("expanded_bytes", 64_000_001),
    ],
)
def test_invalid_storage_version_and_expansion_claims(
    field: str, value: JsonValue
) -> None:
    saved = packed()
    storage(saved)[field] = value
    with pytest.raises(ValueError, match="Invalid Cheap checkpoint storage"):
        codec.restore_exp_cheap_checkpoint(saved)


@pytest.mark.parametrize("reference", [-1, 999999, True, "0", None])
def test_invalid_pool_references_fail_closed(reference: JsonValue) -> None:
    saved = packed()
    storage(saved)["value"] = [reference]
    with pytest.raises(ValueError, match="reference"):
        codec.restore_exp_cheap_checkpoint(saved)


def test_digest_and_unused_or_duplicate_pool_values_are_rejected() -> None:
    saved = packed()
    storage(saved)["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="Changed Cheap checkpoint contents"):
        codec.restore_exp_cheap_checkpoint(saved)
    for duplicate in (False, True):
        saved = packed()
        strings = cast(list[JsonValue], storage(saved)["strings"])
        strings.append(strings[-1] if duplicate else "zzzzz-unused")
        with pytest.raises(ValueError, match="dictionary|unused pool object"):
            codec.restore_exp_cheap_checkpoint(saved)


def test_duplicate_object_fields_and_outer_metadata_are_rejected() -> None:
    saved = packed()
    root = cast(dict[str, JsonValue], storage(saved)["value"])
    fields = cast(list[JsonValue], root["o"])
    fields.insert(1, copy.deepcopy(fields[0]))
    with pytest.raises(ValueError, match="Duplicate Cheap checkpoint field"):
        codec.restore_exp_cheap_checkpoint(saved)
    saved = packed()
    saved["ignored_owner"] = "foreign"
    with pytest.raises(ValueError, match="Invalid Cheap checkpoint storage"):
        codec.restore_exp_cheap_checkpoint(saved)


def test_declared_bytes_limit_charges_strings_and_container_structure() -> None:
    for value in ({"a": [None] * 100}, {"a": [[0]] * 100}):
        saved = packed()
        storage(saved)["expanded_bytes"] = 10
        storage(saved)["value"] = cast(JsonValue, value)
        with pytest.raises(ValueError, match="expanded storage budget"):
            codec.restore_exp_cheap_checkpoint(saved)


def test_depth_cycles_and_nonfinite_values_are_rejected() -> None:
    deep: JsonValue = None
    for _ in range(codec.MAX_EXP_CHEAP_DEPTH + 1):
        deep = [deep]
    source = snapshot()
    source["deep"] = deep
    with pytest.raises(ValueError, match="storage depth"):
        codec.compact_exp_cheap_checkpoint(source)
    source = snapshot()
    source["cycle"] = source
    with pytest.raises(ValueError, match="Cyclic"):
        codec.compact_exp_cheap_checkpoint(source)
    saved = packed()
    cyclic: dict[str, JsonValue] = {"a": []}
    cast(list[JsonValue], cyclic["a"]).append(cyclic)
    storage(saved)["value"] = cyclic
    with pytest.raises(ValueError):
        codec.restore_exp_cheap_checkpoint(saved)
    source = snapshot()
    source["invalid"] = float("nan")
    with pytest.raises(ValueError):
        codec.compact_exp_cheap_checkpoint(source)


def test_packed_durable_and_expanded_budgets_remain_independent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="expanded storage budget"):
        codec.compact_exp_cheap_checkpoint(snapshot(), max_packed_bytes=100)
    with pytest.raises(ValueError, match="expanded storage budget"):
        codec.restore_exp_cheap_checkpoint(packed(), max_packed_bytes=100)
    monkeypatch.setattr(codec, "MAX_EXP_CHEAP_EXPANDED_BYTES", 1000)
    with pytest.raises(ValueError, match="expanded storage budget"):
        codec.compact_exp_cheap_checkpoint(snapshot())
    monkeypatch.undo()
    monkeypatch.setattr(asv3_runs, "MAX_CHECKPOINT_BYTES", 500)
    source = snapshot()
    source["unique"] = random.Random(5).randbytes(10_000).hex()
    with pytest.raises(ValueError, match="durable payload budget"):
        asv3_runs.encode_asv3_checkpoint(source)


def test_pooled_decoder_enforces_durable_limit_without_changing_legacy_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoded = asv3_runs.encode_asv3_checkpoint(snapshot())
    monkeypatch.setattr(asv3_runs, "MAX_CHECKPOINT_BYTES", len(encoded.encode()) - 1)
    with pytest.raises(ValueError, match="durable payload budget"):
        asv3_runs.decode_asv3_checkpoint(encoded)
    legacy = {"version": 1, "run_id": "legacy", "literal": "x" * 3000}
    assert asv3_runs.decode_asv3_checkpoint(json.dumps(legacy)) == legacy


def test_legacy_decoded_limit_still_rejects_oversized_noncheap_snapshot() -> None:
    source = {**snapshot(), "asv3_workflow_variant": "asv3_tuned"}
    source["unpooled"] = "x" * asv3_runs.MAX_DECODED_CHECKPOINT_BYTES
    with pytest.raises(ValueError, match="decoded payload budget"):
        asv3_runs.encode_asv3_checkpoint(source)


def test_inner_snapshot_identity_and_changed_expansion_size_are_checked() -> None:
    saved = packed()
    storage(saved)["expanded_bytes"] = cast(int, storage(saved)["expanded_bytes"]) + 1
    with pytest.raises(ValueError, match="Changed Cheap checkpoint expanded size"):
        codec.restore_exp_cheap_checkpoint(saved)
    saved = packed()
    saved["version"] = 2
    with pytest.raises(ValueError, match="identity"):
        codec.restore_exp_cheap_checkpoint(saved)
