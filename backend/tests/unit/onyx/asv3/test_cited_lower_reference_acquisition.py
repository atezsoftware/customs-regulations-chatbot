"""Cited implementing rules cannot bypass reading their explicit governing referral."""

from dataclasses import replace
from threading import Barrier
from typing import Any

import pytest

from onyx.asv3 import runtime
from onyx.asv3.authority import (
    cited_lower_statute_gap,
    cited_lower_statute_references,
)
from onyx.asv3.corpus_tools import evidence_for_chunk
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.llm_adapter import ResearchModel
from onyx.asv3.models import (
    CapabilityCall,
    OutcomeStatus,
    RunContext,
    ToolOutcome,
    ToolReceipt,
)
from onyx.asv3.workflow_variant import ASV3_TUNED_VARIANT
from onyx.llm.model_response import ModelResponse
from tests.unit.onyx.asv3.test_native_model_adapter import adaptive_tool_view, model
from tests.unit.onyx.asv3.test_parallel_authority_navigation import opaque_original
from tests.unit.onyx.asv3.test_runtime import (
    delivered_originals,
    response,
    setup_run,
    user_payload,
)
from tests.unit.onyx.asv3.test_tuned_source_followthrough import tuned_context


def reference_ledger() -> tuple[RunContext, EvidenceLedger]:
    context, ledger, _reviews = tuned_context()
    ledger.add(
        [
            opaque_original(
                "İzin Tebliği",
                "4",
                kind="tebliğ",
                text="8917 sayılı Faaliyet Kanununun 27 inci maddesi uyarınca izin gerekir.",
            ),
            opaque_original("8917 sayılı Faaliyet Kanunu", "26"),
        ],
        context,
    )
    return context, ledger


def test_implicit_claim_keeps_literal_dependency_without_an_answer_statute_name() -> (
    None
):
    context, ledger = reference_ledger()
    refs = cited_lower_statute_references("İzin gerekir [4].", ledger)
    assert len(refs) == 1
    assert refs[0]["instrument_number"] == "8917"
    assert refs[0]["article"] == "27"
    assert refs[0]["origin_citations"] == [4]
    assert cited_lower_statute_gap("İzin gerekir [4].", ledger, {4, 5})
    assert not cited_lower_statute_references("Başka sonuç [1].", ledger)


def test_exact_original_reuse_requires_actual_delivery_not_related_articles() -> None:
    context, ledger = reference_ledger()
    number = ledger.add(
        [opaque_original("8917 sayılı Faaliyet Kanunu", "27")], context
    )[0]
    assert cited_lower_statute_gap("İzin gerekir [4].", ledger, {4, 5})
    assert cited_lower_statute_gap("İzin gerekir [4].", ledger, {4, 5, number}) is None
    assert cited_lower_statute_references("İzin gerekir [4].", ledger)[0][
        "matching_original_evidence"
    ] == [number]


@pytest.mark.parametrize("ordinal", ["23'üncü", "23’üncü", "23 üncü", "23 uncu"])
def test_turkish_ordinal_referrals_keep_literal_text_and_canonical_identity(
    ordinal: str,
) -> None:
    context, ledger = reference_ledger()
    text = f"Faaliyet Kanunu'nun {ordinal} maddesinin uygulanması mümkündür."
    citation = ledger.add(
        [opaque_original("İzin Tebliği", "9", kind="tebliğ", text=text)], context
    )[0]
    refs = cited_lower_statute_references(f"Bu sonuç mümkündür [{citation}].", ledger)
    assert len(refs) == 1 and refs[0]["article"] == "23"
    assert refs[0]["instrument_number"] == "8917"
    literal = refs[0]["reference_text"]
    assert isinstance(literal, str) and literal in text


@pytest.mark.parametrize(
    "kind", ["kanun", "court_decision", "judicial_decision", "judgment"]
)
def test_primary_original_does_not_trigger_a_blanket_referral_walk(kind: str) -> None:
    context, ledger = reference_ledger()
    citation = ledger.add(
        [
            opaque_original(
                "Court decision",
                "2",
                kind=kind,
                text="7251 sayılı Veri Kanunu m.43 gereğince belirtilen ibare iptal edilmiştir.",
            )
        ],
        context,
    )[0]
    assert not cited_lower_statute_references(f"Mahkeme sonucu [{citation}].", ledger)


@pytest.mark.parametrize(
    "excluded",
    ["uncited", "heading", "derived", "external", "untrusted", "foreign_chunk"],
)
def test_unrelated_or_untrusted_results_do_not_create_reading_dependencies(
    excluded: str,
) -> None:
    context, ledger = reference_ledger()
    answer = "İzin gerekir [4]."
    item = ledger.get(4)
    assert item is not None and item.search_doc is not None
    if excluded == "uncited":
        answer = "Başka sonuç [1]."
    elif excluded == "heading":
        answer = "## İzin [4]"
    elif excluded == "foreign_chunk":
        item.search_doc.metadata["regulatory_chunk_id"] = "foreign"
    else:
        item.metadata[excluded] = True
    originals = [ledger.get(n) for n in ledger.citation_mapping()]
    originals[3] = item
    checked = EvidenceLedger()
    checked.add([original for original in originals if original is not None], context)
    assert not cited_lower_statute_references(answer, checked)


def test_distinct_instruments_keep_bound_locators_and_deduplicate_across_citations() -> (
    None
):
    context, ledger = reference_ledger()
    text = (
        "8917 sayılı Faaliyet Kanunu m.27 uygulanır. "
        "7251 sayılı Veri Kanunu geçici m.8 farklı bir usul düzenler. "
        "Uygulama Yönetmeliği m.53 ve 2025/19 sayılı Genelge başka aşamalardır."
    )
    citations = ledger.add(
        [
            opaque_original("İzin Tebliği", str(n), kind="tebliğ", text=text)
            for n in (7, 8)
        ],
        context,
    )
    refs = cited_lower_statute_references(
        f"Usul [{citations[0]}][{citations[1]}].", ledger
    )
    assert [(r["instrument_number"], r["article"], r["qualifier"]) for r in refs] == [
        ("8917", "27", None),
        ("7251", "8", "gecici"),
    ]
    assert all(r["origin_citations"] == citations for r in refs)


@pytest.mark.parametrize(
    "barrier", ["normal", "experimental", "available", "attempted"]
)
def test_acquisition_is_scoped_reuses_text_and_never_repeats_a_failed_read(
    barrier: str,
) -> None:
    context, ledger = reference_ledger()
    known = ledger.get(5)
    assert known is not None
    view = adaptive_tool_view().model_copy(
        update={
            "draft_to_repair": "İzin gerekir [4].",
            "publication_gap": {"unread_cited_statute_references": []},
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "read_provision",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        }
    )
    if barrier in {"normal", "experimental"}:
        context.services.pop("asv3_workflow_variant")
        context.services["research_profile"] = barrier
    elif barrier == "available":
        ledger.add([opaque_original("8917 sayılı Faaliyet Kanunu", "27")], context)
    else:
        view.receipts.append(
            ToolReceipt(
                call=CapabilityCall(
                    name="read_provision",
                    arguments={"source_id": known.source_id, "article": "27"},
                ),
                outcome=ToolOutcome(
                    status=OutcomeStatus.DENIED, summary="Access denied"
                ),
                elapsed_seconds=0,
            )
        )
    assert ResearchModel(model(), context)._publication_source_acquisition(view) is None


def test_host_reads_known_provision_without_an_extra_generation_or_review() -> None:
    context, ledger = reference_ledger()
    known = ledger.get(5)
    assert known is not None
    selected = model()
    view = adaptive_tool_view().model_copy(
        update={
            "draft_to_repair": "İzin gerekir [4].",
            "publication_gap": {"unread_cited_statute_references": []},
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "read_provision",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        }
    )
    decision = ResearchModel(selected, context)._publication_source_acquisition(view)
    assert decision is not None
    assert [(c.name, c.arguments) for c in decision.calls] == [
        ("read_provision", {"source_id": known.source_id, "article": "27"})
    ]
    assert selected.invoke.call_count == 0


def test_host_preserves_formal_name_when_referred_statute_identity_is_unknown() -> None:
    context, _, _reviews = tuned_context()
    ledger = EvidenceLedger()
    ledger.add(
        [
            opaque_original(
                "İzin Tebliği",
                "4",
                kind="tebliğ",
                text="8917 sayılı Faaliyet Kanununun 27 inci maddesi uyarınca izin gerekir.",
            )
        ],
        context,
    )
    context.services["evidence"] = ledger
    selected = model()
    view = adaptive_tool_view().model_copy(
        update={
            "draft_to_repair": "İzin gerekir [1].",
            "publication_gap": {"unread_cited_statute_references": []},
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "read_named_provision",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        }
    )
    decision = ResearchModel(selected, context)._publication_source_acquisition(view)
    assert decision is not None
    assert [(c.name, c.arguments) for c in decision.calls] == [
        (
            "read_named_provision",
            {"source_name": "8917 sayılı faaliyet kanun", "article": "27"},
        )
    ]
    selected.invoke.assert_not_called()
    view.receipts.append(
        ToolReceipt(
            call=decision.calls[0],
            outcome=ToolOutcome(status=OutcomeStatus.NOT_FOUND, summary="Not found"),
            elapsed_seconds=0,
        )
    )
    assert (
        ResearchModel(selected, context)._publication_source_acquisition(view) is None
    )


@pytest.mark.parametrize("variant", [None, ASV3_TUNED_VARIANT])
def test_runtime_closes_implicit_referral_before_publication_and_preserves_normal(
    monkeypatch: pytest.MonkeyPatch,
    variant: str | None,
) -> None:
    kwargs, broker, selected, checkpoints, _queue = setup_run(monkeypatch)
    kwargs.pop("test_language")
    broker.barrier = Barrier(1)
    lower_id, law_id = (str(source.id) for source in broker.sources)
    broker.chunks[lower_id] = replace(
        broker.chunks[lower_id],
        text="8917 sayılı Faaliyet Kanununun 27 inci maddesi uyarınca izin gerekir.",
        heading_path=("İzin Tebliği", "Madde 4"),
        metadata={"document_type": "tebliğ"},
    )
    broker.chunks[law_id] = replace(
        broker.chunks[law_id],
        text="İzin, yalnız onaylı işlemler için gerekir.",
        heading_path=("8917 sayılı Faaliyet Kanunu", "Madde 27"),
        metadata={"document_type": "kanun"},
    )
    build = runtime.build_corpus_specs
    reads: list[str] = []

    def specs(*args: Any, **options: Any) -> Any:
        result = build(*args, **options)

        def read(arguments: dict[str, Any], child: RunContext) -> ToolOutcome:
            assert arguments == {
                "source_name": "8917 sayılı faaliyet kanun",
                "article": "27",
            }
            child.check_active()
            reads.append(law_id)
            return ToolOutcome(
                status=OutcomeStatus.FOUND,
                summary="Canonical referenced original",
                evidence=[evidence_for_chunk(broker.sources[1], broker.chunks[law_id])],
            )

        return [
            s.model_copy(update={"handler": read})
            if s.name == "read_named_provision"
            else s
            for s in result
        ]

    monkeypatch.setattr(runtime, "build_corpus_specs", specs)
    calls = 0

    def scripted(**arguments: Any) -> ModelResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            return response(
                calls=[
                    ("read_source_range", {"source_id": lower_id, "_language": "tr"})
                ]
            )
        if calls == 2:
            return response(
                calls=[
                    (
                        "submit_answer",
                        {"answer": "İzin gerekir [1].", "basis": "originals"},
                    )
                ]
            )
        assert calls == 3
        originals = delivered_originals(arguments)
        assert any(
            r["source_id"] == law_id and r["text"] == broker.chunks[law_id].text
            for r in originals
        )
        return response(
            calls=[
                (
                    "submit_retained_answer",
                    {
                        "retained_answer_edits": [
                            {
                                "unit_id": user_payload(arguments["prompt"][-1])[
                                    "draft_to_repair"
                                ]["units"][0]["unit_id"],
                                "replacement": "İzin yalnız onaylı işlemler için gerekir [1][2].",
                            }
                        ],
                    },
                )
            ]
        )

    selected.invoke.side_effect = scripted
    kwargs.update(research_profile="normal")
    if variant is not None:
        kwargs["workflow_variant"] = variant
    runtime.run_asv3_loop(**kwargs)
    assert calls == (3 if variant else 2)
    assert reads == ([law_id] if variant else [])
    assert checkpoints[-1]["publication_status"] == "found"
    assert kwargs["state_container"].answer_tokens
