import copy
import json
import threading
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import JsonValue

from onyx.asv3.llm_adapter import ResearchModel
from onyx.asv3.models import RunContext, SharedBudget
from onyx.llm.interfaces import LLM, LLMConfig
from onyx.llm.model_response import Choice, Message, ModelResponse
from onyx.tracing.flows import LLMFlow


def test_repair_context_preserves_exact_draft_and_only_latest_gap() -> None:
    from onyx.asv3.models import (
        CapabilityCall,
        HarnessView,
        OutcomeStatus,
        ToolOutcome,
        ToolReceipt,
    )

    llm = scripted_model()
    llm.invoke.return_value = ModelResponse(
        id="repair",
        created="0",
        choice=Choice(message=Message(content="Repaired wording")),
    )
    model = ResearchModel(llm, RunContext())
    draft = "İzin gerekir [1]. Sonrasında kayıtlar karşılaştırılır [2]."
    gap: dict[str, JsonValue] = {
        "missing": ["b bendi"],
        "instruction": "Read the operative clause",
    }
    stale = ToolReceipt(
        call=CapabilityCall(name="finalization_status"),
        outcome=ToolOutcome(
            status=OutcomeStatus.PARTIAL,
            summary="STALE_DIAGNOSIS",
            data={"gaps": ["outdated requirement"]},
        ),
        elapsed_seconds=0,
    )
    model.decide(
        HarnessView(
            request="Şartlar nedir?",
            questions=["Şartlar nedir?"],
            facts=[],
            receipts=[stale],
            evidence=[],
            tools=[],
            draft_to_repair=draft,
            publication_gap=gap,
        )
    )
    content = llm.invoke.call_args.kwargs["prompt"][-1].content
    assert isinstance(content, list)
    payload = json.loads(content[0].text)
    assert payload["draft_to_repair"] == draft
    assert payload["publication_gap"] == gap
    assert payload["receipts"] == []
    assert "STALE_DIAGNOSIS" not in json.dumps(payload)
    compacted = model._compact_payload(payload, 8)
    assert compacted["draft_to_repair"] == draft and compacted["publication_gap"] == gap
    assert llm.invoke.call_count == 1


def test_real_adapter_limits_shared_provider_calls_across_worker_contexts() -> None:
    llm = MagicMock(spec=LLM)
    llm.config = LLMConfig(
        model_provider="openai",
        model_name="selected-model",
        temperature=0,
        max_input_tokens=100000,
    )
    context = RunContext(budget=SharedBudget(max_inflight_models=2))
    release = threading.Event()
    lock = threading.Lock()
    active = peak = 0
    responses: list[str] = []
    errors: list[Exception] = []

    def invoke(**_kwargs: Any) -> ModelResponse:
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
            if active == 2:
                release.set()
        assert release.wait(3)
        with lock:
            active -= 1
        return ModelResponse(
            id="response",
            created="0",
            choice=Choice(message=Message(content="original selected provider result")),
        )

    llm.invoke.side_effect = invoke

    def run() -> None:
        try:
            model = ResearchModel(llm, context.child())
            assert model.llm is llm
            responses.append(model.invoke_text("read", "task", LLMFlow.ASV3_RESEARCHER))
        except Exception as error:
            errors.append(error)

    threads = [threading.Thread(target=run) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(3)
    assert errors == []
    assert not any(thread.is_alive() for thread in threads)
    assert len(responses) == 6 and llm.invoke.call_count == 6
    assert peak == 2
    assert context.budget.snapshot()["decisions"] == 6
    assert llm.config.model_name == "selected-model"


def scripted_model(limit: int = 100000) -> MagicMock:
    llm = MagicMock(spec=LLM)
    llm.config = LLMConfig(
        model_provider="openai",
        model_name="selected-model",
        temperature=0,
        max_input_tokens=limit,
    )
    return llm


def test_evidence_metadata_excludes_vectors_and_raw_content_at_creation_and_restore() -> (
    None
):
    from onyx.asv3.evidence import EvidenceLedger
    from onyx.asv3.models import EvidenceItem

    metadata: dict[str, JsonValue] = {
        "canonical_metadata": {
            "title_vector": [0.25] * 1024,
            "content_vector": [0.75] * 1024,
            "content": "DUPLICATED_RAW_DOCUMENT",
            "doc_summary": "INDEX_SUMMARY",
            "regulatory_chunk_id": "article-143",
            "publication_revision": "rev-7",
            "query_index_uuid": "index-identity",
            "heading_path": ["Kanun", "Madde 143"],
            "document_type": "kanun",
            "title": "General operative statute",
            "document_date": "2020-01-01",
            "source_links": {"0": "https://example.test/article-143"},
        },
        "read_as_of_date": "2026-10-02",
        "validity_start": "2020-01-01",
        "source_sha256": "original-hash",
        "locator": {"page": 3, "row": 2, "normalized_box": [0.1, 0.2, 0.3, 0.4]},
    }
    item = EvidenceItem(
        source_id="law",
        chunk_id="article-143",
        text="Complete operative conditions.",
        metadata=metadata,
    )
    ledger = EvidenceLedger()
    ledger.add([item], RunContext())
    checkpoint = json.loads(json.dumps(ledger.export()))
    encoded = json.dumps(checkpoint)
    assert "vector" not in encoded and "DUPLICATED_RAW_DOCUMENT" not in encoded
    retained = checkpoint["records"][0]["item"]
    assert retained["metadata"]["canonical_metadata"]["publication_revision"] == "rev-7"
    assert retained["metadata"]["canonical_metadata"]["heading_path"] == [
        "Kanun",
        "Madde 143",
    ]
    assert retained["metadata"]["locator"]["normalized_box"] == [0.1, 0.2, 0.3, 0.4]
    assert retained["metadata"]["source_sha256"] == "original-hash"
    supplied = json.loads(ledger.serialize_records([1], required=[1]))[0]
    assert supplied["metadata"]["heading_path"] == ["Kanun", "Madde 143"]
    assert supplied["metadata"]["document_type"] == "kanun"
    assert supplied["metadata"]["title"] == "General operative statute"
    assert supplied["metadata"]["publication_revision"] == "rev-7"
    assert supplied["metadata"]["query_index_uuid"] == "index-identity"
    assert "canonical_metadata" not in supplied["metadata"]
    navigation = ledger.summaries(max_chars=6000)
    assert navigation[0]["document_type"] == "kanun"
    assert navigation[0]["title"] == "General operative statute"
    assert len(json.dumps(navigation, ensure_ascii=False)) <= 6000
    stored = ledger.get(1)
    assert stored is not None and "canonical_metadata" in stored.metadata
    # Legacy checkpoint metadata is sanitized without changing canonical identity.
    retained["metadata"] = metadata
    restored = EvidenceLedger()
    restored.restore(checkpoint, RunContext())
    restored_item = restored.get(1)
    assert restored_item is not None and restored_item.identity == item.identity
    assert "vector" not in json.dumps(restored.export())
    retained["text"] = "Changed original"
    with pytest.raises(ValueError, match="hash"):
        EvidenceLedger().restore(checkpoint, RunContext())


def test_serialized_evidence_limit_counts_provenance_and_escaping_without_clipping_rules() -> (
    None
):
    from onyx.asv3.evidence import EvidenceLedger
    from onyx.asv3.models import EvidenceItem, RunStopped

    ledger = EvidenceLedger()
    item = EvidenceItem(
        source_id="law",
        chunk_id="143",
        text='"\\' * 100,
        metadata={"publication_revision": "r" * 1000, "heading_path": ["Madde 143"]},
    )
    ledger.add([item], RunContext())
    full = ledger.serialize_records([1], required=[1])
    assert json.loads(full)[0]["text"] == item.text
    assert len(item.text) < len(full) - 1
    with pytest.raises(RunStopped, match="serialized evidence limit"):
        ledger.serialize_records([1], required=[1], max_chars=len(full) - 1)
    assert ledger.serialize_records([1], max_chars=len(full) - 1) == "[]"
    assert ledger.serialize_records([1], required=[1], max_chars=len(full)) == full
    model = ResearchModel(scripted_model(limit=800), RunContext(), token_counter=len)
    with pytest.raises(RunStopped, match="Complete cited evidence"):
        model._fit(
            "Verify",
            json.dumps({"draft": "Rule [1]", "evidence": full}),
            [],
            max_tokens=100,
        )


def test_model_receives_latest_complete_pair_and_sanitized_originals_without_audit_history() -> (
    None
):
    from onyx.asv3.models import ResearchTurn
    from onyx.llm.models import AssistantMessage, FunctionCall, ToolCall, ToolMessage

    turns = [
        ResearchTurn(
            assistant=AssistantMessage(
                content=None,
                tool_calls=[
                    ToolCall(
                        id=f"call-{n}",
                        function=FunctionCall(
                            name="read_evidence", arguments='{"citation":1}'
                        ),
                    )
                ],
            ),
            results=[
                ToolMessage(tool_call_id=f"call-{n}", content=f"HISTORICAL_PAYLOAD_{n}")
            ],
        )
        for n in range(3)
    ]
    model = ResearchModel(scripted_model(), RunContext())
    prompt, _, _ = model._fit(
        "Research",
        json.dumps({"request": "Complete scenario", "evidence": []}),
        [],
        max_tokens=1000,
        research=True,
        turns=turns,
    )
    assert prompt[1] == turns[-1].assistant and prompt[2] == turns[-1].results[0]
    assert len(prompt) == 4 and len(turns) == 3
    encoded = json.dumps([message.model_dump(mode="json") for message in prompt])
    assert "HISTORICAL_PAYLOAD_2" in encoded and "HISTORICAL_PAYLOAD_0" not in encoded
    data = {
        "request": "Complete scenario",
        "draft": "Condition [1]",
        "evidence": json.dumps(
            [
                {
                    "citation": 1,
                    "text": "ALL_OPERATIVE_CONDITIONS",
                    "metadata": {
                        "title_vector": [0.25] * 1024,
                        "publication_revision": "rev-7",
                    },
                }
            ]
        ),
        "receipts": [{"audit": "OLD_AUDIT"}],
        "updates": ["OLD_AUDIT"],
    }
    prompt, _, _ = model._fit(
        "Verify", json.dumps(data), [], max_tokens=1000, turns=turns
    )
    encoded = json.dumps([message.model_dump(mode="json") for message in prompt])
    assert (
        "vector" not in encoded
        and "OLD_AUDIT" not in encoded
        and "HISTORICAL_PAYLOAD" not in encoded
    )
    assert "ALL_OPERATIVE_CONDITIONS" in encoded and "rev-7" in encoded


def test_unlimited_run_keeps_finite_selected_provider_timeout_and_finite_deadline_checks() -> (
    None
):
    from onyx.asv3.models import RunStopped

    llm = scripted_model()
    llm.invoke.return_value = ModelResponse(
        id="result",
        created="0",
        choice=Choice(message=Message(content="Complete answer")),
    )
    assert (
        ResearchModel(llm, RunContext(timeout_seconds=float("inf"))).invoke_text(
            "Answer", "Scenario", LLMFlow.ASV3_FINAL
        )
        == "Complete answer"
    )
    assert llm.invoke.call_args.kwargs["timeout_override"] == 120
    expired = RunContext(timeout_seconds=-1)
    with pytest.raises(RunStopped):
        ResearchModel(llm, expired).invoke_text(
            "Answer", "Scenario", LLMFlow.ASV3_FINAL
        )
    assert llm.invoke.call_count == 1


def complete_language_profile() -> dict[str, Any]:
    from onyx.asv3.llm_adapter import REQUIRED_NOTIFICATION_PHASES

    return {
        "language": "tr",
        "external_requested": False,
        "requires_sources": True,
        "notifications": {
            phase: ["Araştırma", "Royalti koşullarını inceliyorum."]
            for phase in REQUIRED_NOTIFICATION_PHASES
        },
    }


def test_actual_model_input_records_full_and_partial_deliveries_without_copying_files() -> (
    None
):
    from onyx.asv3.models import HarnessView

    context, ledger, tools = original_state()
    llm = scripted_model()
    llm.invoke.return_value = ModelResponse(
        id="decision", created="0", choice=Choice(message=Message(content="Draft [1]"))
    )
    model = ResearchModel(llm, context)
    original = ledger.get(1)
    assert original is not None
    view = HarnessView(
        request="Read the operative rule",
        questions=[],
        facts=[],
        receipts=[],
        evidence=[{"citation": 1, "text": original.text[:12], "truncated": True}],
        tools=tools,
    )
    model.decide(view)
    first = ledger.inspect(1)["deliveries"]
    assert isinstance(first, list) and len(first) == 1
    record = first[0]["records"][0]
    assert record["complete"] is False and record["end_char"] == 12
    assert "text" not in record and record["text_hash"] == original.text_hash
    model.invoke_text(
        "Use supplied evidence",
        json.dumps({"evidence": json.dumps([{"citation": 1, "text": original.text}])}),
        LLMFlow.ASV3_FINAL,
    )
    delivered = ledger.inspect(1)["deliveries"]
    assert (
        isinstance(delivered, list) and delivered[-1]["records"][0]["complete"] is True
    )
    assert ledger.export()["records"][0]["item"]["text"] == original.text


def test_provider_receives_paired_assistant_tool_history_and_current_state() -> None:
    from onyx.asv3.harness import Harness
    from onyx.asv3.registry import CapabilityRegistry, build_core_specs
    from onyx.llm.models import AssistantMessage, ToolMessage

    context, ledger, _ = original_state()
    llm = scripted_model()
    llm.invoke.side_effect = [
        tool_response('{"citation":1}'),
        ModelResponse(
            id="draft",
            created="0",
            choice=Choice(message=Message(content="Supported draft [1]")),
        ),
    ]
    registry = CapabilityRegistry()
    for spec in build_core_specs(registry, ledger, lambda: {}):
        registry.register(spec)
    harness = Harness(
        request="Read original",
        context=context,
        registry=registry,
        evidence=ledger,
        decide=ResearchModel(llm, context).decide,
    )
    result = harness.run()
    assert result.answer == "Supported draft [1]"
    prompt = llm.invoke.call_args_list[-1].kwargs["prompt"]
    assistant = next(
        message for message in prompt if isinstance(message, AssistantMessage)
    )
    tool = next(message for message in prompt if isinstance(message, ToolMessage))
    assert assistant.tool_calls[0].id == tool.tool_call_id == "call-1"
    assert "The original complete operative text." in tool.content
    turns = harness.snapshot()["turns"]
    assert isinstance(turns, list) and len(turns) == 1
    assert ledger.inspect(1)["deliveries"]


def text_response(profile: dict[str, Any]) -> ModelResponse:
    return ModelResponse(
        id="language",
        created="0",
        choice=Choice(message=Message(content=json.dumps(profile, ensure_ascii=False))),
    )


def verification_profile() -> dict[str, Any]:
    return {
        "status": "supported",
        "explanation": "Özgün hüküm destekliyor.",
        "required_conditions": [],
        "missing_conditions": [],
        "evidence_numbers": [1],
    }


def tool_response(arguments: str, name: str = "read_evidence") -> ModelResponse:
    from onyx.llm.model_response import ChatCompletionMessageToolCall, FunctionCall

    return ModelResponse(
        id="result",
        created="0",
        choice=Choice(
            message=Message(
                tool_calls=[
                    ChatCompletionMessageToolCall(
                        id="call-1",
                        function=FunctionCall(name=name, arguments=arguments),
                    )
                ]
            )
        ),
    )


def test_vertex_tool_normalization_does_not_mutate_decision_validation_schema() -> None:
    from litellm.llms.vertex_ai.common_utils import _build_vertex_schema

    from onyx.asv3.models import HarnessView

    llm = scripted_model()
    tools: list[dict[str, JsonValue]] = [
        {
            "type": "function",
            "function": {
                "name": "search_corpus",
                "description": "search",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "mode": {"type": "string", "enum": ["keyword", "hybrid"]},
                    },
                    "required": ["query"],
                    "$defs": {"unused": {"type": "string"}},
                    "additionalProperties": False,
                },
            },
        }
    ]
    original = copy.deepcopy(tools)

    def invoke(**kwargs: Any) -> ModelResponse:
        supplied = kwargs["tools"][0]["function"]["parameters"]
        normalized = _build_vertex_schema(supplied)
        assert normalized["properties"]["mode"]["type"] == "string"
        assert "$defs" not in supplied
        return tool_response('{"query":"royalti","mode":"keyword"}', "search_corpus")

    llm.invoke.side_effect = invoke
    decision = ResearchModel(llm, RunContext()).decide(
        HarnessView(
            request="Royalti koşullarını araştır",
            questions=[],
            facts=[],
            receipts=[],
            evidence=[],
            tools=tools,
        )
    )
    assert decision.calls[0].arguments["mode"] == "keyword"
    assert llm.invoke.call_count == 1
    assert tools == original


def original_state() -> tuple[RunContext, Any, list[dict[str, Any]]]:
    from onyx.asv3.evidence import EvidenceLedger
    from onyx.asv3.models import EvidenceItem
    from onyx.asv3.registry import CapabilityRegistry, build_core_specs

    ledger = EvidenceLedger()
    context = RunContext(services={"evidence": ledger})
    ledger.add(
        [
            EvidenceItem(
                source_id="law",
                chunk_id="143",
                text="The original complete operative text.",
            )
        ],
        context,
    )
    registry = CapabilityRegistry()
    for spec in build_core_specs(registry, ledger, lambda: {}):
        registry.register(spec)
    return context, ledger, registry.definitions(context)


def test_tool_json_and_schema_get_one_selected_model_repair_before_dispatch() -> None:
    from onyx.asv3.models import HarnessView

    context, ledger, tools = original_state()
    llm = scripted_model()
    llm.invoke.side_effect = [
        tool_response('{"citation":"1"}'),
        tool_response('{"citation":1}'),
    ]
    model = ResearchModel(llm, context)
    view = HarnessView(
        request="Read the complete operative provision",
        questions=[],
        facts=[],
        receipts=[],
        evidence=[],
        tools=tools,
    )
    before = ledger.export()
    decision = model.decide(view)
    assert len(decision.calls) == 1 and decision.calls[0].arguments == {"citation": 1}
    assert llm.invoke.call_count == 2
    assert context.budget.snapshot()["decisions"] == 1
    assert ledger.export() == before
    correction = (
        llm.invoke.call_args_list[1].kwargs["prompt"][1].content[0].text
        if isinstance(llm.invoke.call_args_list[1].kwargs["prompt"][1].content, list)
        else llm.invoke.call_args_list[1].kwargs["prompt"][1].content
    )
    assert "format_repair" in correction and "Do not invent" in correction
    assert llm.config.model_name == "selected-model"


def test_repeated_malformed_tool_call_fails_after_one_repair_and_keeps_sources() -> (
    None
):
    import pytest

    from onyx.asv3.models import HarnessView

    context, ledger, tools = original_state()
    llm = scripted_model()
    llm.invoke.side_effect = [
        tool_response('{"citation":'),
        tool_response('{"citation":'),
    ]
    model = ResearchModel(llm, context)
    before = ledger.export()
    with pytest.raises(ValueError):
        model.decide(
            HarnessView(
                request="read",
                questions=[],
                facts=[],
                receipts=[],
                evidence=[],
                tools=tools,
            )
        )
    assert llm.invoke.call_count == 2
    assert ledger.export() == before
    assert context.budget.model_slots.acquire(blocking=False)
    context.budget.model_slots.release()


def test_language_json_repair_uses_original_request_and_same_selected_model() -> None:
    from onyx.asv3.llm_adapter import parse_json_object

    llm = scripted_model()
    llm.invoke.side_effect = [
        ModelResponse(
            id="bad", created="0", choice=Choice(message=Message(content="```{broken"))
        ),
        text_response(complete_language_profile()),
    ]
    context = RunContext()
    model = ResearchModel(llm, context)
    result = model.invoke_text(
        "Identify response language and return JSON",
        "Yanıt Türkçe olsun",
        LLMFlow.ASV3_LANGUAGE,
    )
    assert parse_json_object(result)["language"] == "tr"
    assert llm.invoke.call_count == 2
    assert context.budget.snapshot()["decisions"] == 2
    assert llm.invoke.call_args.kwargs["prompt"][1].content == "Yanıt Türkçe olsun"


@pytest.mark.parametrize(
    "defect",
    [
        "missing_phase",
        "missing_native",
        "single_value",
        "blank_message",
        "string_consent",
    ],
)
def test_valid_json_incomplete_language_profile_is_repaired_once(defect: str) -> None:
    from onyx.asv3.llm_adapter import LanguageProfile

    incomplete = complete_language_profile()
    if defect == "missing_phase":
        del incomplete["notifications"]["tools"]
    elif defect == "missing_native":
        del incomplete["notifications"]["native_citation"]
    elif defect == "single_value":
        incomplete["notifications"]["started"] = ["Araştırma"]
    elif defect == "blank_message":
        incomplete["notifications"]["final"] = ["Araştırma", "  "]
    else:
        incomplete["external_requested"] = "true"
    llm = scripted_model()
    llm.invoke.side_effect = [
        text_response(incomplete),
        text_response(complete_language_profile()),
    ]
    result = ResearchModel(llm, RunContext()).invoke_text(
        "Identify response language",
        "Royalti hakkında Türkçe yanıtla",
        LLMFlow.ASV3_LANGUAGE,
    )
    profile = LanguageProfile.model_validate_json(result)
    assert profile.language == "tr" and profile.external_requested is False
    assert llm.invoke.call_count == 2
    assert llm.config.model_name == "selected-model"
    for call in llm.invoke.call_args_list:
        response_format = call.kwargs["structured_response_format"]
        assert response_format["type"] == "json_schema"
        schema = response_format["json_schema"]["schema"]
        notification_schema = schema["properties"]["notifications"]
        assert set(notification_schema["required"]) == set(profile.notifications)
        assert notification_schema["properties"]["native_citation"]["type"] == "array"
        assert notification_schema["additionalProperties"] is False
        system = call.kwargs["prompt"][0].content
        assert isinstance(system, str)
        assert '"required": ["started", "tools"' in system
        assert '"native_citation"' in system and '"maxItems": 2' in system


def test_repeated_incomplete_language_profile_fails_closed_after_single_repair() -> (
    None
):
    profile = complete_language_profile()
    del profile["notifications"]["cancelled"]
    llm = scripted_model()
    llm.invoke.side_effect = [text_response(profile), text_response(profile)]
    context = RunContext()
    with pytest.raises(ValueError, match="cancelled"):
        ResearchModel(llm, context).invoke_text(
            "Identify language", "Türkçe yanıtla", LLMFlow.ASV3_LANGUAGE
        )
    assert llm.invoke.call_count == 2
    assert context.corpus_only is True
    assert context.budget.snapshot()["decisions"] == 2


def test_safe_review_with_unsupported_assertions_requires_consistent_repair() -> None:
    from onyx.asv3.llm_adapter import VerificationResult

    llm = scripted_model()
    contradictory = verification_profile()
    contradictory["safe_to_publish"] = True
    contradictory["unsupported_claims"] = ["The governing source has not been obtained"]
    corrected = verification_profile()
    corrected["safe_to_publish"] = True
    corrected["status"] = "incomplete"
    corrected["missing_conditions"] = ["The governing source has not been obtained"]
    llm.invoke.side_effect = [text_response(contradictory), text_response(corrected)]
    result = ResearchModel(llm, RunContext()).invoke_text(
        "Verify only original evidence",
        '{"claim":"The source gap is explicit","evidence":"original"}',
        LLMFlow.ASV3_VERIFICATION,
    )
    review = VerificationResult.model_validate_json(result)
    assert review.safe_to_publish is True and review.unsupported_claims == []
    assert review.missing_conditions == corrected["missing_conditions"]
    assert llm.invoke.call_count == 2


def test_verification_uses_typed_provider_schema_and_repairs_incomplete_json() -> None:
    from onyx.asv3.llm_adapter import VerificationResult

    llm = scripted_model()
    llm.invoke.side_effect = [
        text_response({"status": "supported"}),
        text_response(verification_profile()),
    ]
    result = ResearchModel(llm, RunContext()).invoke_text(
        "Verify the claim",
        '{"claim":"Royalti [1]","evidence":"original"}',
        LLMFlow.ASV3_VERIFICATION,
    )
    assert VerificationResult.model_validate_json(result).evidence_numbers == [1]
    assert llm.invoke.call_count == 2
    for call in llm.invoke.call_args_list:
        response_format = call.kwargs["structured_response_format"]
        schema = response_format["json_schema"]["schema"]
        assert schema["properties"]["status"]["enum"] == [
            "supported",
            "contradicted",
            "incomplete",
            "uncertain",
        ]
        assert set(schema["required"]) == set(verification_profile())
        assert json.loads(call.kwargs["prompt"][1].content) == {
            "claim": "Royalti [1]",
            "evidence": "original",
        }


def test_final_answer_remains_free_text_without_structured_schema() -> None:
    llm = scripted_model()
    llm.invoke.return_value = ModelResponse(
        id="final", created="0", choice=Choice(message=Message(content="Yanıt [1]."))
    )
    assert (
        ResearchModel(llm, RunContext()).invoke_text(
            "Write answer", "question", LLMFlow.ASV3_FINAL
        )
        == "Yanıt [1]."
    )
    assert llm.invoke.call_args.kwargs["structured_response_format"] is None


def test_wrapped_unique_structured_object_is_normalized_without_repair() -> None:
    from onyx.asv3.llm_adapter import LanguageProfile

    llm = scripted_model()
    profile = complete_language_profile()
    llm.invoke.return_value = ModelResponse(
        id="wrapped",
        created="0",
        choice=Choice(
            message=Message(
                content="İstenen çıktı:\n"
                + json.dumps(profile, ensure_ascii=False)
                + "\nBu dilde devam edeceğim."
            )
        ),
    )
    result = ResearchModel(llm, RunContext()).invoke_text(
        "Language", "Türkçe yanıtla", LLMFlow.ASV3_LANGUAGE
    )
    assert LanguageProfile.model_validate_json(result).external_requested is False
    assert json.loads(result) == profile
    assert llm.invoke.call_count == 1


@pytest.mark.parametrize("defect", ["duplicate", "nonfinite", "malformed", "ambiguous"])
def test_wrapped_structured_objects_preserve_strict_rejection(defect: str) -> None:
    from onyx.asv3.llm_adapter import LanguageProfile, normalize_structured_response

    raw = json.dumps(complete_language_profile())
    if defect == "duplicate":
        raw = raw.replace(
            '"external_requested": false',
            '"external_requested": false, "external_requested": true',
        )
    elif defect == "nonfinite":
        raw = raw.replace('"external_requested": false', '"external_requested": NaN')
    elif defect == "malformed":
        raw = '{"broken": ' + raw
    else:
        raw += "\n" + raw
    with pytest.raises(ValueError):
        normalize_structured_response("Çıktı:\n" + raw, LanguageProfile)


@pytest.mark.parametrize("error_kind", ["rate_limit", "timeout", "server"])
def test_transient_provider_retry_preserves_model_schema_and_shared_budget(
    error_kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from litellm.exceptions import InternalServerError, RateLimitError

    from onyx.asv3 import llm_adapter
    from onyx.llm.multi_llm import LLMTimeoutError

    monkeypatch.setattr(llm_adapter, "LLM_FIRST_CHUNK_RETRY_BASE_DELAY_S", 0)
    monkeypatch.setattr(llm_adapter, "LLM_FIRST_CHUNK_RETRY_MAX_DELAY_S", 0)
    error = (
        RateLimitError(
            "limited",
            llm_provider="vertex_ai",
            model="selected-model",
            headers={"Retry-After": "0"},
        )
        if error_kind == "rate_limit"
        else LLMTimeoutError("timeout")
        if error_kind == "timeout"
        else InternalServerError(
            "unavailable", llm_provider="vertex_ai", model="selected-model"
        )
    )
    llm = scripted_model()
    llm.invoke.side_effect = [error, text_response(complete_language_profile())]
    context = RunContext()
    result = ResearchModel(llm, context).invoke_text(
        "Language", "Türkçe yanıtla", LLMFlow.ASV3_LANGUAGE
    )
    assert json.loads(result)["language"] == "tr"
    assert llm.invoke.call_count == 2
    assert context.budget.snapshot()["decisions"] == 2
    assert llm.config.model_name == "selected-model"
    assert (
        llm.invoke.call_args_list[0].kwargs["structured_response_format"]
        == llm.invoke.call_args_list[1].kwargs["structured_response_format"]
    )


def test_provider_retries_stop_after_three_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from onyx.asv3 import llm_adapter
    from onyx.llm.multi_llm import LLMTimeoutError

    monkeypatch.setattr(llm_adapter, "LLM_FIRST_CHUNK_RETRY_BASE_DELAY_S", 0)
    monkeypatch.setattr(llm_adapter, "LLM_FIRST_CHUNK_RETRY_MAX_DELAY_S", 0)
    llm = scripted_model()
    llm.invoke.side_effect = LLMTimeoutError("timeout")
    context = RunContext()
    with pytest.raises(LLMTimeoutError):
        ResearchModel(llm, context).invoke_text("Final", "question", LLMFlow.ASV3_FINAL)
    assert llm.invoke.call_count == 3
    assert context.budget.snapshot()["decisions"] == 3


@pytest.mark.parametrize("kind", ["cancel", "deadline", "budget"])
def test_retry_wait_checks_cancellation_deadline_and_budget(kind: str) -> None:
    from litellm.exceptions import RateLimitError

    from onyx.asv3.models import RunStopped

    context = RunContext(
        timeout_seconds=0.1 if kind == "deadline" else 2,
        budget=SharedBudget(max_decisions=1, final_decision_reserve=0)
        if kind == "budget"
        else None,
    )
    error = RateLimitError(
        "limited",
        llm_provider="vertex_ai",
        model="selected-model",
        headers={"Retry-After": "0.2"},
    )
    llm = scripted_model()
    llm.invoke.side_effect = error
    timer = threading.Timer(0.02, context.cancel)
    if kind == "cancel":
        timer.start()
    try:
        with pytest.raises(RunStopped):
            ResearchModel(llm, context).invoke_text(
                "Final", "question", LLMFlow.ASV3_FINAL
            )
    finally:
        timer.cancel()
    assert llm.invoke.call_count == 1


@pytest.mark.parametrize("kind", ["authentication", "unsupported"])
def test_nonretryable_provider_failure_is_attempted_once(kind: str) -> None:
    from litellm.exceptions import AuthenticationError

    llm = scripted_model()
    error = (
        AuthenticationError("denied", llm_provider="vertex_ai", model="selected-model")
        if kind == "authentication"
        else ValueError("Unsupported request")
    )
    llm.invoke.side_effect = error
    with pytest.raises(type(error)):
        ResearchModel(llm, RunContext()).invoke_text(
            "Final", "question", LLMFlow.ASV3_FINAL
        )
    assert llm.invoke.call_count == 1


def test_strict_json_rejects_duplicate_keys_nonfinite_values_and_bad_fences() -> None:
    import pytest

    from onyx.asv3.llm_adapter import parse_json_object

    for text in ('{"citation":1,"citation":2}', '{"value":NaN}', "```{}", "[1]"):
        with pytest.raises(ValueError):
            parse_json_object(text)


def test_pathological_context_fits_selected_limit_and_originals_remain_reopenable() -> (
    None
):
    import json

    from onyx.asv3.models import EvidenceItem, HarnessView

    context, ledger, tools = original_state()
    original = "FULL_ORIGINAL_MARKER" + "Text " * 30000
    ledger.add(
        [EvidenceItem(source_id="another-law", chunk_id="article", text=original)],
        context,
    )
    for n in range(20):
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": f"large_schema_{n}",
                    "description": "Description " * 5000,
                    "parameters": {"type": "object"},
                },
            }
        )
    llm = scripted_model(40000)
    llm.invoke.return_value = ModelResponse(
        id="valid",
        created="0",
        choice=Choice(message=Message(content="Read source evidence next.")),
    )
    history = "EARLY_HISTORY_CONSTRAINT " + "history " * 2000
    model = ResearchModel(llm, context, token_counter=len, history=history)
    view = HarnessView(
        request="Cover all three alternatives.",
        questions=["free repair", "paid repair", "replacement"],
        facts=["no standard exchange permission"],
        receipts=[],
        evidence=[
            {"citation": 1, "source_id": "law", "text": "first source"},
            {"citation": 2, "source_id": "another-law", "text": original},
        ],
        tools=tools,
    )
    model.decide(view)
    request = llm.invoke.call_args.kwargs
    assert (
        model._input_cost(request["prompt"], request["tools"]) + request["max_tokens"]
        <= llm.config.max_input_tokens
    )
    payload = json.loads(request["prompt"][1].content[0].text)
    assert (
        payload["request"] == view.request
        and payload["questions"] == view.questions
        and payload["facts"] == view.facts
    )
    assert payload["conversation"] == history
    assert payload["evidence"][1]["truncated"] is True
    assert payload["capability_context"]["reopen"] == "discover_tools"
    assert {tool["function"]["name"] for tool in request["tools"]} >= {
        "read_evidence",
        "discover_tools",
    }
    original_item = ledger.get(2)
    assert original_item is not None and original_item.text == original
    assert view.evidence[1]["text"] == original


def test_irreducible_question_and_complete_cited_evidence_fail_before_provider() -> (
    None
):
    import json

    import pytest

    from onyx.asv3.models import RunStopped

    context, ledger, tools = original_state()
    llm = scripted_model(12000)
    model = ResearchModel(llm, context)
    from onyx.asv3.harness import Harness
    from onyx.asv3.registry import CapabilityRegistry

    question = "QUESTION " * 10000
    harness = Harness(
        request=question,
        context=context,
        registry=CapabilityRegistry(),
        decide=model.decide,
    )
    view = harness.view()
    assert view.request == question
    with pytest.raises(RunStopped, match="scenario"):
        model.decide(view)
    text = "Complete law " * 10000
    with pytest.raises(RunStopped, match="Complete cited evidence"):
        model.invoke_text(
            "Verify the cited rule",
            json.dumps(
                {
                    "claim": "Rule [1]",
                    "scenario": "facts",
                    "evidence": json.dumps([{"citation": 1, "text": text}]),
                }
            ),
            LLMFlow.ASV3_VERIFICATION,
        )
    assert llm.invoke.call_count == 0
    assert ledger.get(1) is not None


def test_final_review_removes_only_uncited_supplemental_context() -> None:
    import json

    llm = scripted_model(12000)
    llm.invoke.return_value = text_response(verification_profile())
    model = ResearchModel(llm, RunContext())
    original = "Complete operative paragraph with all exceptions."
    model.invoke_text(
        "Verify original cited text",
        json.dumps(
            {
                "claim": "Rule [1]",
                "scenario": "facts",
                "evidence": json.dumps(
                    [
                        {"citation": 1, "text": original, "truncated": False},
                        {"citation": 2, "text": "Uncited " * 10000},
                    ]
                ),
            }
        ),
        LLMFlow.ASV3_VERIFICATION,
    )
    payload = json.loads(llm.invoke.call_args.kwargs["prompt"][1].content)
    assert json.loads(payload["evidence"]) == [
        {"citation": 1, "text": original, "truncated": False}
    ]
    assert payload["supplemental_evidence_omitted"] is True


def test_schema_repair_cannot_drop_an_already_valid_parallel_action() -> None:
    import pytest

    from onyx.asv3.models import HarnessView

    context, _ledger, tools = original_state()
    llm = scripted_model()
    first = tool_response('{"citation":1}')
    invalid = tool_response('{"citation":"2"}')
    assert invalid.choice.message.tool_calls and first.choice.message.tool_calls
    invalid.choice.message.tool_calls[0].id = "call-2"
    first.choice.message.tool_calls += invalid.choice.message.tool_calls
    repair = tool_response('{"citation":2}')
    assert repair.choice.message.tool_calls
    repair.choice.message.tool_calls[0].id = "call-2"
    llm.invoke.side_effect = [first, repair]
    model = ResearchModel(llm, context)
    with pytest.raises(ValueError, match="action count"):
        model.decide(
            HarnessView(
                request="Read both",
                questions=[],
                facts=[],
                receipts=[],
                evidence=[],
                tools=tools,
            )
        )
    assert llm.invoke.call_count == 2


def test_selected_tokenizer_preserves_nonascii_cited_law_that_fits_actual_limit() -> (
    None
):
    import json

    import tiktoken

    encoding = tiktoken.get_encoding("cl100k_base")
    llm = scripted_model(12000)
    llm.invoke.return_value = text_response(verification_profile())
    model = ResearchModel(
        llm, RunContext(), token_counter=lambda text: len(encoding.encode(text))
    )
    original = "Tamir işlemi ücretsizdir. Üretim hatası kanıtlanır. " * 300
    assert len(original.encode("utf-8")) > llm.config.max_input_tokens
    model.invoke_text(
        "Verify original cited text",
        json.dumps(
            {
                "claim": "Tamir koşulları [1]",
                "scenario": "Ücretsiz tamir",
                "evidence": json.dumps(
                    [{"citation": 1, "text": original, "truncated": False}],
                    ensure_ascii=False,
                ),
            },
            ensure_ascii=False,
        ),
        LLMFlow.ASV3_VERIFICATION,
    )
    request = llm.invoke.call_args.kwargs
    assert (
        model._input_cost(request["prompt"], request["tools"] or [])
        + request["max_tokens"]
        <= llm.config.max_input_tokens
    )
    payload = json.loads(request["prompt"][1].content)
    assert json.loads(payload["evidence"])[0]["text"] == original
    assert llm.invoke.call_count == 1


def test_invalid_token_counter_fails_before_provider_and_unsupported_falls_back() -> (
    None
):
    import pytest

    llm = scripted_model()
    for value in (-1, True):
        model = ResearchModel(llm, RunContext(), token_counter=lambda _text: value)
        with pytest.raises(ValueError, match="invalid count"):
            model.invoke_text("instruction", "question", LLMFlow.ASV3_FINAL)
    assert llm.invoke.call_count == 0

    def unsupported(_text: str) -> int:
        raise NotImplementedError("No provider tokenizer")

    model = ResearchModel(llm, RunContext(), token_counter=unsupported)
    assert model._tokens("Ücretsiz") == len("Ücretsiz".encode("utf-8"))
