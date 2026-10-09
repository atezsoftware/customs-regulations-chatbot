"""Explicit references trigger scoped reads; preliminary court text cannot close them."""

from copy import deepcopy
from threading import Barrier
from unittest.mock import Mock
from uuid import uuid4

import pytest
from pydantic import JsonValue, ValidationError

from onyx.asv3.corpus_tools import evidence_for_chunk
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import OutcomeStatus, RunContext, ToolOutcome, ToolSpec
from onyx.asv3.registry import CapabilityRegistry
from onyx.db.asv3_corpus import CorpusChunk, CorpusSource
from onyx.db.legal_composite_sources import SourceKind
from onyx.legal_composite.acquisition import CanonicalAcquirer, InvalidSourceAction
from onyx.legal_composite.dependencies import (
    CompositeDependencyExpander,
    DependencyExpander,
    assess_dependencies,
    dependency_required_citations,
    material_dependency_gaps,
)
from onyx.legal_composite.models import (
    AnswerReview,
    DependencyAssessment,
    DependencyWitness,
    DraftAnswer,
    MaterialDependencyRequest,
    PassageSupport,
    ResearchNeed,
    ResearchPlan,
    SourceAction,
    SourceRequirement,
    WorkflowPolicy,
)


def plan() -> ResearchPlan:
    return ResearchPlan(
        language="tr",
        requires_sources=True,
        needs=[
            ResearchNeed(
                need_id="permit",
                question="İzin şartı nedir?",
                governing_source="Faaliyet Kanunu",
                conditions_to_check=["Yetki kapsamı"],
            ),
            ResearchNeed(
                need_id="scope",
                question="Uygulama kapsamı nedir?",
                governing_source="Faaliyet Kanunu",
                conditions_to_check=["Uygulama zamanı"],
            ),
        ],
        initial_actions=[],
        missing_user_facts=[],
    )


def chunk(
    source: CorpusSource, text: str, position: int, *, title: str, article: str = "27"
) -> CorpusChunk:
    return CorpusChunk(
        f"rc_{source.id}_{position}",
        source.id,
        text,
        position,
        position,
        (title, f"MADDE {article}"),
        {"document_type": "kanun", "title": title},
        None,
        None,
        "active",
    )


def setup_expander(
    *,
    name="Faaliyet Kanunu",
    number="8917",
    article="27",
    numberless=False,
    full_context=True,
    additional_reference="",
    reference_text=None,
    expander_class=DependencyExpander,
):
    law = CorpusSource(uuid4(), f"{number} sayılı {name}.md", "law")
    lower = CorpusSource(uuid4(), "Faaliyet Tebliği.md", "lower")
    court = CorpusSource(
        uuid4(), f"Mahkeme {name} {article} maddesinin iptali.md", "court"
    )
    law_chunk = chunk(
        law,
        f"MADDE {article} - İzin için yetkili idareye başvuru gerekir.",
        0,
        title=f"{number} sayılı {name}",
        article=article,
    )
    lower_chunk = chunk(
        lower,
        reference_text
        if reference_text is not None
        else (
            f"{name}nun {article} inci maddesi uyarınca izin gerekir."
            if numberless
            else f"{number} sayılı {name}'nun {article}. maddesi uyarınca izin gerekir."
        )
        + additional_reference,
        0,
        title="Faaliyet Tebliği",
        article="3",
    )
    lower_chunk.metadata["document_type"] = "tebliğ"
    court_chunks = [
        chunk(court, text, index, title="Wrong law heading", article="999")
        for index, text in enumerate(
            [
                "ANAYASA MAHKEMESİ KARARI\nKarar tarihi bilinmiyor.",
                "## Tarafların iddiaları\nKuralın iptal edilmesini isteriz.",
                "## HÜKÜM\nKuralın yetki kapsamı daraltılmıştır.",
                "Bu karar kesinleşmiş işlemleri etkilemez.",
            ]
        )
    ]
    ledger = EvidenceLedger()
    context = RunContext(corpus_only=True)
    item = evidence_for_chunk(lower, lower_chunk)
    item.question_ids = ["permit", "scope"]
    ledger.add([item], context)
    broker = Mock()
    broker.chunk.return_value = (law, law_chunk)
    broker.related_catalog_sources.return_value = ([court], False)
    calls: list[tuple[str, dict[str, JsonValue]]] = []
    overlap = Barrier(1 if numberless else 2)

    def resolve(args, _context):
        calls.append(("resolve_source", dict(args)))
        return ToolOutcome(
            status=OutcomeStatus.FOUND,
            summary="Scoped identity candidates.",
            data={
                "sources": [{"source_id": str(law.id), "name": law.name}],
                "has_more": False,
            },
        )

    def provision(args, _context):
        calls.append(("read_provision", dict(args)))
        return ToolOutcome(
            status=OutcomeStatus.FOUND,
            summary="Own original.",
            evidence=[evidence_for_chunk(law, law_chunk)],
        )

    def search(args, _context):
        calls.append(("search_corpus", dict(args)))
        # Independent identity variants overlap in the host search, without a court-kind gate.
        overlap.wait(timeout=3)
        return ToolOutcome(
            status=OutcomeStatus.FOUND,
            summary="Candidate body, not holding.",
            evidence=[
                evidence_for_chunk(court, row)
                for row in (court_chunks if full_context else court_chunks[1:2])
            ],
        )

    def read_range(args, _context):
        calls.append(("read_source_range", dict(args)))
        start = int(args.get("start", 0))
        chosen = court_chunks[start : start + 2]
        return ToolOutcome(
            status=OutcomeStatus.PARTIAL if start == 0 else OutcomeStatus.FOUND,
            summary="Full canonical bodies in pages.",
            data={"has_more": start == 0, "next_position": start + 2},
            evidence=[evidence_for_chunk(court, row) for row in chosen],
        )

    def spec(name, handler):
        return ToolSpec(
            name=name, description=name, parameters={"type": "object"}, handler=handler
        )

    host = CapabilityRegistry(
        [
            spec("resolve_source", resolve),
            spec("search_corpus", search),
            spec("read_provision", provision),
        ]
    )
    typed = CapabilityRegistry(
        [
            spec("read_provision", provision),
            spec("search_corpus", search),
            spec("read_source_range", read_range),
        ]
    )
    expand = Mock(
        side_effect=AssertionError(
            "Host dependency calls must not fan out to all twelve lanes"
        )
    )
    acquirer = CanonicalAcquirer(
        CapabilityRegistry(),
        context,
        ledger,
        WorkflowPolicy(max_tools=100, max_search_calls=20),
        registry_for_action=lambda _action: typed,
        expand_actions=expand,
        host_registry=host,
    )
    expander = expander_class(
        broker=broker,
        acquirer=acquirer,
        ledger=ledger,
        context=context,
        source_kinds={
            str(law.id): SourceKind.STATUTE,
            str(lower.id): SourceKind.COMMUNIQUE,
            str(court.id): SourceKind.JUDICIAL_DECISION,
        },
    )
    return expander, ledger, calls, broker, court, law, expand


def test_reference_search_reuses_original_hits_without_whole_candidate_file_reads() -> (
    None
):
    expander, ledger, calls, broker, court, law, expand = setup_expander()
    edges = expander.expand(plan())
    assert len(edges) == 1
    edge = edges[0]
    assert edge.instrument_number == "8917" and edge.article == "27"
    assert edge.need_ids == ["permit", "scope"]
    assert edge.governing_citations and len(edge.candidate_citations) == 4
    assert edge.judicial_source_ids == [str(court.id)]
    assert not edge.incomplete_source_ids and not edge.discovery_gaps
    assert not any(name == "read_source_range" for name, _args in calls)
    assert sum(name == "read_provision" for name, _args in calls) == 1
    assert sum(name == "search_corpus" for name, _args in calls) == 2
    assert any(
        args["query"] == "8917 27" for name, args in calls if name == "search_corpus"
    )
    for citation in edge.candidate_citations + edge.governing_citations:
        assert set(ledger.get(citation).question_ids) == {"permit", "scope"}
    broker.related_catalog_sources.assert_called_once()
    broker.chunk.assert_not_called()
    assert len(calls) == 4
    expander.expand(plan(), frontier={origin.citation for origin in edge.origins})
    assert len(calls) == 4
    expand.assert_not_called()
    protected = dependency_required_citations(edges, ledger)
    assert set(edge.governing_citations) <= set(protected)
    assert any("HÜKÜM" in ledger.get(number).text for number in protected)
    assert not any(
        ledger.get(number).source_id == str(law.id)
        and number in edge.candidate_citations
        for number in protected
    )


def test_rejected_background_frontier_does_not_trigger_unrelated_dependencies() -> None:
    expander, _ledger, calls, broker, _court, _law, _expand = setup_expander()
    assert expander.expand(plan(), frontier=set()) == []
    assert not calls
    broker.related_catalog_sources.assert_not_called()


def review(edge, status="examined_nonmaterial", witnesses=None, disclosure=None):
    return AnswerReview(
        request_coverage_complete=True,
        material_claims_supported=True,
        counter_authority_checked=True,
        needs=[],
        defects=[],
        repair_actions=[],
        dependency_assessments=[
            DependencyAssessment(
                edge_id=edge.edge_id,
                need_ids=edge.need_ids,
                status=status,
                witnesses=witnesses or [],
                explanation="Original scope is evaluated against these facts, not the source title.",
                scope_and_date="This nonmaterial scope exclusion does not depend on event dates.",
                temporal_status="established",
                gap_disclosure=disclosure,
            )
        ],
    )


def test_dependency_gate_requires_assessment_and_exact_delivered_disposition_not_argument_or_wrong_heading() -> (
    None
):
    expander, ledger, _calls, _broker, court, _law, _expand = setup_expander()
    edge = expander.expand(plan())[0]
    gov = edge.governing_citations[0]
    disposition = next(
        number
        for number in edge.candidate_citations
        if "HÜKÜM" in ledger.get(number).text
    )
    argument = next(
        number
        for number in edge.candidate_citations
        if "iddiaları" in ledger.get(number).text
    )
    draft = DraftAnswer(answer="Supported law only.", unresolved_need_ids=[])
    delivered = set(ledger.citation_numbers())
    missing = review(edge)
    missing.dependency_assessments = None
    assert assess_dependencies([edge], plan(), draft, missing, ledger, delivered)[
        :2
    ] == (False, False)
    witnesses = [
        DependencyWitness(
            citation=gov, quotation=ledger.get(gov).text, role="governing"
        ),
        DependencyWitness(
            citation=disposition,
            quotation="Kuralın yetki kapsamı daraltılmıştır.",
            role="operative",
        ),
    ]
    valid = review(edge, witnesses=witnesses)
    assert assess_dependencies([edge], plan(), draft, valid, ledger, delivered)[:2] == (
        True,
        True,
    )
    for invalid in [
        review(
            edge,
            witnesses=[
                witnesses[0],
                DependencyWitness(
                    citation=argument,
                    quotation="Kuralın iptal edilmesini isteriz.",
                    role="operative",
                ),
            ],
        ),
        review(edge, witnesses=[witnesses[0]]),
    ]:
        assert assess_dependencies([edge], plan(), draft, invalid, ledger, delivered)[
            :2
        ] == (False, False)
    assert assess_dependencies(
        [edge], plan(), draft, valid, ledger, delivered - {disposition}
    )[:2] == (False, False)
    incomplete = edge.model_copy(update={"incomplete_source_ids": [str(court.id)]})
    assert assess_dependencies([incomplete], plan(), draft, valid, ledger, delivered)[
        :2
    ] == (False, False)
    stale = deepcopy(edge)
    stale.origins[0].text_hash = "0" * 64
    assert assess_dependencies([stale], plan(), draft, valid, ledger, delivered)[
        :2
    ] == (False, False)


@pytest.mark.parametrize("citation", [True, 1.0, "1"])
def test_dependency_witness_ids_are_strict(citation) -> None:
    with pytest.raises(ValidationError):
        DependencyWitness(citation=citation, quotation="Original", role="governing")


def test_host_large_receipt_preserves_structural_paging_and_identity_values() -> None:
    context = RunContext(corpus_only=True)
    data: dict[str, JsonValue] = {
        "sources": [{"source_id": str(uuid4()), "name": "x" * 300} for _ in range(100)],
        "has_more": True,
        "next_offset": 100,
        "next_position": 100,
        "evidence_next_position": 80,
    }
    registry = CapabilityRegistry(
        [
            ToolSpec(
                name="resolve_source",
                description="Identity",
                parameters={"type": "object"},
                handler=lambda _args, _context: ToolOutcome(
                    status=OutcomeStatus.PARTIAL, summary="Navigation", data=data
                ),
            )
        ]
    )
    acquirer = CanonicalAcquirer(registry, context, EvidenceLedger(), WorkflowPolicy())
    action = SourceAction(
        need_ids=["permit"], tool="resolve_source", arguments={"query": "Own source"}
    )
    receipt = acquirer.acquire_host_actions([action], plan())[0]
    receipt_data = receipt["data"]
    assert isinstance(receipt_data, dict)
    assert receipt_data["receipt_omitted"] is True
    for key, value in data.items():
        assert receipt_data[key] == value


def test_literal_numberless_instrument_is_navigation_without_existing_own_law_or_number() -> (
    None
):
    expander, ledger, _calls, _broker, _court, _law, _expand = setup_expander()
    origin = ledger.get(1)
    origin.text = "Gümrük Kanununun 241 inci maddesi uyarınca işlem yapılır."
    origin = origin.model_copy(update={"chunk_id": "numberless", "text_hash": ""})
    origin.search_doc.metadata["regulatory_chunk_id"] = "numberless"
    fresh = EvidenceLedger()
    fresh.add([origin], RunContext())
    expander.ledger = fresh
    expander.acquirer.ledger = fresh
    expander._collect(plan())
    assert len(expander.edges) == 1
    edge = next(iter(expander.edges.values()))
    assert edge.article == "241" and edge.instrument_number is None
    assert edge.instrument_name == "Gümrük Kanunu"
    assert not edge.governing_citations
    assert expander._matching_edge("GÜMRÜK KANUNU", "4458", "241", None) is edge
    assert expander._matching_edge("Different Kanunu", "4458", "241", None) is None


def test_verified_nonstatute_cannot_become_governing_through_wrong_law_metadata() -> (
    None
):
    expander, ledger, _calls, _broker, _court, _law, _expand = setup_expander()
    original = ledger.get(1)
    original.metadata["canonical_metadata"] = {
        "document_type": "kanun",
        "title": "Faaliyet Kanunu",
        "heading_path": ["Faaliyet Kanunu", "MADDE 27"],
    }
    expander.verified_kinds[original.source_id] = SourceKind.JUDICIAL_DECISION
    assert expander._anchor(original) is None
    expander.verified_kinds[original.source_id] = SourceKind.UNKNOWN
    assert expander._anchor(original) is None


def test_mixed_reasoning_disposition_chunk_remains_a_whole_required_original() -> None:
    expander, ledger, _calls, _broker, court, _law, _expand = setup_expander()
    edge = expander.expand(plan())[0]
    selected = next(
        ledger.get(number)
        for number in edge.candidate_citations
        if "HÜKÜM" in ledger.get(number).text
    )
    mixed = selected.model_copy(
        update={
            "chunk_id": "mixed",
            "text_hash": "",
            "text": "## Esasın incelenmesi\nReasoning alone.\n## HÜKÜM\nKuralın kapsamı daraltılmıştır.",
        }
    )
    mixed.search_doc.metadata["regulatory_chunk_id"] = "mixed"
    fresh = EvidenceLedger()
    opening = next(
        ledger.get(number)
        for number in edge.candidate_citations
        if ledger.get(number).metadata["position"] == 0
    )
    numbers = fresh.add([opening, mixed], RunContext())
    edge = edge.model_copy(
        update={
            "governing_citations": [],
            "candidate_citations": numbers,
            "judicial_source_ids": [str(court.id)],
        }
    )
    assert dependency_required_citations([edge], fresh) == numbers
    retained = fresh.get(numbers[1])
    assert retained is not None
    assert retained.text == mixed.text


def test_temporal_unknown_requires_exact_conditional_or_unresolved_user_disclosure() -> (
    None
):
    from onyx.legal_composite.models import NeedReview

    expander, ledger, _calls, _broker, _court, _law, _expand = setup_expander()
    edge = expander.expand(plan())[0]
    gov = edge.governing_citations[0]
    operative = next(
        number
        for number in edge.candidate_citations
        if "HÜKÜM" in ledger.get(number).text
    )
    witnesses = [
        DependencyWitness(
            citation=gov, quotation=ledger.get(gov).text, role="governing"
        ),
        DependencyWitness(
            citation=operative,
            quotation="Kuralın yetki kapsamı daraltılmıştır.",
            role="operative",
        ),
    ]
    examined = review(edge, witnesses=witnesses)
    row = examined.dependency_assessments[0]
    row.temporal_status = "conditional"
    draft = DraftAnswer(
        answer="İşlemin tarihi ve kesinleşme durumu uygunsa bu kapsam geçerlidir.",
        unresolved_need_ids=[],
    )
    assert assess_dependencies(
        [edge], plan(), draft, examined, ledger, set(ledger.citation_numbers())
    )[:2] == (False, False)
    row.conditional_excerpt = draft.answer
    examined.needs = [
        NeedReview(
            need_id=need,
            status="conditional",
            supports=[],
            conditions_preserved=True,
            condition_reviews=[],
            explanation="User dates and finality are missing.",
        )
        for need in edge.need_ids
    ]
    assert assess_dependencies(
        [edge], plan(), draft, examined, ledger, set(ledger.citation_numbers())
    )[:2] == (True, True)
    row.temporal_status = "unresolved"
    assert assess_dependencies(
        [edge], plan(), draft, examined, ledger, set(ledger.citation_numbers())
    )[:2] == (False, False)
    row.status = "unresolved"
    row.gap_disclosure = draft.answer
    draft.unresolved_need_ids = list(edge.need_ids)
    for need in examined.needs:
        need.status = "unresolved"
    assert assess_dependencies(
        [edge], plan(), draft, examined, ledger, set(ledger.citation_numbers())
    )[:2] == (False, True)


def test_selection_first_pass_keeps_every_receipt_and_delta_pass_only_classifies_new_originals() -> (
    None
):
    from onyx.legal_composite.engine import LegalCompositeEngine
    from onyx.legal_composite.selection import SourceIdentity, SourceSelectionResult

    expander, ledger, _calls, _broker, _court, _law, _expand = setup_expander()
    edges = expander.expand(plan())
    selector = Mock()
    requests = []

    def select(request):
        requests.append(request)
        return SourceSelectionResult(
            protected_citations=[row.citation for row in request.candidates],
            background_citations=[],
            rejected_citations=[],
            retained_citations=[row.citation for row in request.candidates],
            selection_complete=True,
            gaps=[],
            identities=[
                SourceIdentity.model_validate(
                    {
                        key: value
                        for key, value in row.model_dump().items()
                        if key in SourceIdentity.model_fields
                    }
                )
                for row in request.candidates
            ],
            receipts=[],
            call_id=None,
        )

    selector.select.side_effect = select
    engine = LegalCompositeEngine(
        gateway=Mock(),
        acquirer=expander.acquirer,
        ledger=ledger,
        policy=WorkflowPolicy(),
        check_active=lambda: None,
        research_available=lambda: True,
        selector=selector,
    )
    engine.receipts = [
        {"source_kind": "statute", "citations": edges[0].governing_citations},
        {"source_kind": "unknown", "citations": edges[0].candidate_citations},
    ]
    all_numbers = set(ledger.citation_numbers())
    first = all_numbers - {max(all_numbers)}
    engine._select_sources("Question", plan(), first)
    assert {row.citation for row in requests[0].candidates} == first
    engine._select_sources("Question", plan(), {max(all_numbers)})
    assert {row.citation for row in requests[1].candidates} == {max(all_numbers)}
    assert engine.selection is not None
    assert set(engine.selection.protected_citations) == all_numbers
    engine._select_sources("Question", plan())
    assert {row.citation for row in requests[2].candidates} == all_numbers


def test_opening_read_start_zero_can_begin_at_canonical_position_one_and_still_verify_court_kind() -> (
    None
):
    expander, ledger, _calls, _broker, _court, _law, _expand = setup_expander()
    edge = expander.expand(plan())[0]
    originals = [ledger.get(number) for number in edge.candidate_citations]
    fresh = EvidenceLedger()
    for item in originals:
        item.metadata["position"] += 1
    numbers = fresh.add(originals, RunContext())
    expander.ledger = fresh
    assert (
        expander._original_kind(originals[0].source_id) == SourceKind.JUDICIAL_DECISION
    )
    assert numbers


@pytest.mark.parametrize(
    "text",
    ["Bu Kanunun 27. maddesi uygulanır.", "İlgili Kanunun 27 nci maddesi uygulanır."],
)
def test_anaphoric_law_mentions_do_not_create_a_phantom_named_instrument(text) -> None:
    from onyx.legal_composite.dependencies import _literal_names

    assert _literal_names(text) == {}


def test_dependency_frontier_uses_per_need_material_relevance_bindings() -> None:
    expander, _ledger, _calls, _broker, _court, _law, _expand = setup_expander()
    expander._collect(plan(), frontier={1}, need_bindings={1: {"permit"}})
    assert next(iter(expander.edges.values())).need_ids == ["permit"]


def test_numberless_lower_reference_resolves_numbered_own_original_then_cross_kind_court_body() -> (
    None
):
    expander, ledger, calls, broker, court, law, _expand = setup_expander(
        name="Gümrük Kanunu", number="4458", article="241", numberless=True
    )
    assert ledger.get(1).text.startswith("Gümrük Kanununun 241 inci maddesi")
    assert len(ledger.citation_numbers()) == 1
    edge = expander.expand(plan())[0]
    assert len(expander.edges) == 1
    assert edge.instrument_number == "4458" and edge.article == "241"
    assert all(
        ledger.get(number).source_id == str(law.id)
        for number in edge.governing_citations
    )
    assert edge.judicial_source_ids == [str(court.id)]
    assert len(edge.candidate_citations) == 4
    assert not edge.discovery_gaps and not edge.incomplete_source_ids
    broker.related_catalog_sources.assert_called_once()
    assert any(
        args.get("query") == "Gümrük Kanunu 241"
        for name, args in calls
        if name == "search_corpus"
    )
    assert sum(name == "read_provision" for name, _args in calls) == 1


def test_search_argument_is_not_enough_but_only_targeted_context_is_read() -> None:
    expander, ledger, calls, _broker, court, _law, _expand = setup_expander(
        full_context=False
    )
    edge = expander.expand(plan())[0]
    assert len(edge.candidate_citations) == 1
    assert not any(name == "read_source_range" for name, _ in calls)
    assert "iddiaları" in ledger.get(edge.candidate_citations[0]).text
    # A known missing disposition is read directly; the preceding file is never paged.
    expander.acquirer.acquire_host_actions(
        [
            SourceAction(
                need_ids=edge.need_ids,
                tool="read_source_range",
                source_kind=SourceKind.JUDICIAL_DECISION,
                arguments={"source_id": str(court.id), "start": 2, "limit": 2},
            )
        ],
        plan(),
    )
    expander.synchronize()
    assert [args["start"] for name, args in calls if name == "read_source_range"] == [2]
    assert any("HÜKÜM" in ledger.get(n).text for n in edge.candidate_citations)


def setup_composite_expander(**kwargs):
    return setup_expander(expander_class=CompositeDependencyExpander, **kwargs)


def material_target(**overrides: JsonValue) -> MaterialDependencyRequest:
    return MaterialDependencyRequest.model_validate(
        {
            "need_ids": ["permit"],
            "origin_citation": 1,
            "instrument_name": "Faaliyet Kanunu",
            "instrument_number": "8917",
            "article": "27",
            "reason": "The original expressly makes this article govern the permission rule.",
            **overrides,
        }
    )


def test_empty_material_targets_never_activate_incidental_references() -> None:
    expander, _ledger, calls, broker, _court, _law, _expand = setup_composite_expander()
    assert expander.expand(plan(), material_targets=[]) == []
    assert not calls
    broker.related_catalog_sources.assert_not_called()


def test_material_expansion_uses_only_requested_reference_and_need_without_rereads() -> (
    None
):
    expander, _ledger, calls, broker, _court, _law, expand = setup_composite_expander(
        additional_reference=" 9621 sayılı Denetim Kanunu'nun 13. maddesi farklı işlemleri düzenler."
    )
    edges = expander.expand(plan(), frontier={1}, material_targets=[material_target()])
    assert len(edges) == 1
    edge = edges[0]
    assert edge.instrument_number == "8917" and edge.article == "27"
    assert edge.need_ids == ["permit"]
    assert edge.origins[0].citation == 1
    assert len(calls) == 4
    assert not any("Denetim" in str(args) or "9621" in str(args) for _, args in calls)
    expander.expand(plan(), frontier={1}, material_targets=[material_target()])
    assert len(calls) == 4
    broker.related_catalog_sources.assert_called_once()
    expand.assert_not_called()


@pytest.mark.parametrize(
    "changes",
    [
        {"instrument_name": "Invented Kanunu"},
        {"instrument_number": "9621"},
        {"article": "13"},
        {"origin_citation": 999},
        {"need_ids": ["invented"]},
        {"need_ids": ["permit", "permit"]},
        {"reason": " "},
    ],
)
def test_invented_material_anchors_fail_before_io_or_edge_mutation(changes) -> None:
    expander, _ledger, calls, broker, _court, _law, _expand = setup_composite_expander()
    with pytest.raises(InvalidSourceAction):
        expander.expand(
            plan(), frontier={1}, material_targets=[material_target(**changes)]
        )
    assert not calls and not expander.edges
    broker.related_catalog_sources.assert_not_called()


def test_material_target_rejects_stale_origin_and_undelivered_frontier() -> None:
    expander, ledger, calls, _broker, _court, _law, _expand = setup_composite_expander()
    with pytest.raises(InvalidSourceAction, match="not delivered"):
        expander.expand(plan(), frontier=set(), material_targets=[material_target()])
    ledger._items[1].text_hash = "0" * 64
    with pytest.raises(InvalidSourceAction, match="not canonical"):
        expander.expand(plan(), frontier={1}, material_targets=[material_target()])
    assert not calls and not expander.edges


def test_numberless_material_request_cannot_invent_an_instrument_number() -> None:
    expander, _ledger, calls, _broker, _court, _law, _expand = setup_composite_expander(
        numberless=True
    )
    with pytest.raises(InvalidSourceAction):
        expander.expand(plan(), material_targets=[material_target()])
    assert not calls
    edge = expander.expand(
        plan(), material_targets=[material_target(instrument_number=None)]
    )[0]
    assert edge.instrument_number == "8917"


def test_material_readiness_requires_original_binding_and_body_not_review_flags() -> (
    None
):
    expander, ledger, _calls, _broker, court, _law, _expand = setup_composite_expander()
    edge = expander.expand(plan(), material_targets=[material_target()])[0]
    delivered = set(ledger.citation_numbers())
    disposition = next(
        n for n in edge.candidate_citations if "HÜKÜM" in ledger.get(n).text
    )
    requirement = SourceRequirement(
        requirement_id="authority_scope",
        need_id="permit",
        dimension="authority",
        rule="Permission must respect the authority scope.",
        application="The narrowed authority scope must be considered.",
        supports=[
            PassageSupport(
                citation=disposition, quotation="Kuralın yetki kapsamı daraltılmıştır."
            )
        ],
    )
    assert material_dependency_gaps([edge], ledger, delivered, [requirement]) == {
        edge.edge_id: []
    }
    assert material_dependency_gaps([edge], ledger, delivered - {disposition})[
        edge.edge_id
    ] == [
        f"citation_undelivered@{disposition}",
        f"judicial_operative_body_unread@{court.id}",
    ]
    stale = deepcopy(edge)
    stale.origins[0].text_hash = "0" * 64
    assert material_dependency_gaps([stale], ledger, delivered)[edge.edge_id] == [
        "origin_binding_mismatch@1"
    ]
    ledger._items[disposition].text_hash = "0" * 64
    assert (
        f"citation_noncanonical@{disposition}"
        in material_dependency_gaps([edge], ledger, delivered)[edge.edge_id]
    )


def test_material_readiness_rejects_arguments_and_false_headings_but_keeps_window_limits() -> (
    None
):
    expander, ledger, _calls, _broker, court, _law, _expand = setup_composite_expander(
        full_context=False
    )
    edge = expander.expand(plan(), material_targets=[material_target()])[0]
    argument = ledger.get(edge.candidate_citations[0])
    ledger._items[edge.candidate_citations[0]].metadata["heading_path"] = ["HÜKÜM"]
    edge.discovery_limits = ["Bounded window cannot prove corpus absence."]
    delivered = set(ledger.citation_numbers())
    gaps = material_dependency_gaps([edge], ledger, delivered)[edge.edge_id]
    assert gaps == [f"judicial_operative_body_unread@{court.id}"]
    missing = edge.model_copy(
        update={
            "governing_citations": [],
            "incomplete_source_ids": [str(court.id)],
            "discovery_gaps": ["Canonical acquisition stopped."],
        }
    )
    gaps = material_dependency_gaps([missing], ledger, delivered)[edge.edge_id]
    assert {
        "governing_original_unread",
        "discovery_gap_unresolved",
        f"candidate_original_missing@{court.id}",
    } <= set(gaps)
    assert not any(argument.text in gap for gap in gaps)


def test_legacy_assessment_keeps_governing_original_requirement() -> None:
    expander, ledger, _calls, _broker, _court, _law, _expand = (
        setup_composite_expander()
    )
    expander._collect_material(plan(), material_targets=[material_target()])
    edge = next(iter(expander.edges.values()))
    witness = DependencyWitness(
        citation=1, quotation=ledger.get(1).text, role="nonmaterial"
    )
    draft = DraftAnswer(
        answer="The referenced rule does not govern this distinct question.",
        unresolved_need_ids=[],
    )
    nonmaterial = review(edge, witnesses=[witness])
    assert assess_dependencies([edge], plan(), draft, nonmaterial, ledger, {1})[:2] == (
        False,
        False,
    )
    applicable = review(edge, status="examined_applicable", witnesses=[witness])
    complete, safe, gaps = assess_dependencies(
        [edge], plan(), draft, applicable, ledger, {1}
    )
    assert not complete and not safe
    assert "missing, stale, unread" in gaps[0]
    assert ledger.get(1).text not in gaps[0]


@pytest.mark.parametrize(
    "name,number,article,kind,reference",
    [
        (
            "Liman Güvenliği Yönetmeliği",
            "6721",
            "9",
            SourceKind.REGULATION,
            "6721 sayılı Liman Güvenliği Yönetmeliğinin 9 uncu maddesi uygulanır.",
        ),
        (
            "Bölgesel İzin Kurulu Kararı",
            "9032",
            "6",
            SourceKind.EXECUTIVE_DECISION,
            "9032 sayılı Bölgesel İzin Kurulu Kararının 6 ncı maddesi uygulanır.",
        ),
    ],
)
def test_explicit_nonstatute_dependencies_use_actual_named_reference_and_own_body(
    name, number, article, kind, reference
) -> None:
    expander, ledger, calls, _broker, _court, own, _expand = setup_composite_expander(
        name=name, number=number, article=article, reference_text=reference
    )
    expander.source_kinds[str(own.id)] = kind
    expander.verified_kinds[str(own.id)] = kind
    target = material_target(
        instrument_name=name, instrument_number=number, article=article
    )
    edge = expander.expand(plan(), material_targets=[target])[0]
    assert edge.instrument_name == name and edge.instrument_number == number
    assert edge.article == article and edge.need_ids == ["permit"]
    assert edge.governing_citations
    assert all(
        ledger.get(citation).source_id == str(own.id)
        for citation in edge.governing_citations
    )
    assert len(calls) == 4
    expander.expand(plan(), material_targets=[target])
    assert len(calls) == 4


def test_formal_nonstatute_number_cannot_bind_to_an_unrelated_earlier_number() -> None:
    expander, _ledger, calls, _broker, _court, _own, _expand = setup_composite_expander(
        reference_text="427 kaynak incelenmiştir; Taşıma Güvenliği Genelgesinin 8 inci maddesi uygulanır."
    )
    target = material_target(
        instrument_name="Taşıma Güvenliği Genelgesi",
        instrument_number="427",
        article="8",
    )
    with pytest.raises(InvalidSourceAction):
        expander.expand(plan(), material_targets=[target])
    assert not calls and not expander.edges
    expander._collect_material(
        plan(), material_targets=[target.model_copy(update={"instrument_number": None})]
    )
    edge = next(iter(expander.edges.values()))
    assert edge.instrument_number is None and edge.article == "8"


def test_two_formal_nonstatute_titles_do_not_borrow_the_other_article_or_number() -> (
    None
):
    expander, _ledger, calls, _broker, _court, _own, _expand = setup_composite_expander(
        reference_text="7812 sayılı İzin ve Denetim Tebliği’nin 14. maddesi ve 5923 sayılı Deniz Taşıması Antlaşması'nın 22. maddesi uygulanır."
    )
    with pytest.raises(InvalidSourceAction):
        expander.expand(
            plan(),
            material_targets=[
                material_target(
                    instrument_name="İzin ve Denetim Tebliği",
                    instrument_number="5923",
                    article="22",
                )
            ],
        )
    assert not calls and not expander.edges
    expander._collect_material(
        plan(),
        material_targets=[
            material_target(
                instrument_name="Deniz Taşıması Antlaşması",
                instrument_number="5923",
                article="22",
            )
        ],
    )
    assert (
        next(iter(expander.edges.values())).instrument_name
        == "Deniz Taşıması Antlaşması"
    )


def test_unknown_wrong_nonstatute_metadata_cannot_establish_own_governing_identity() -> (
    None
):
    expander, ledger, _calls, _broker, _court, _own, _expand = (
        setup_composite_expander()
    )
    origin = ledger.get(1)
    origin.metadata["canonical_metadata"] = {
        "title": "İzin Güvenliği Yönetmeliği",
        "heading_path": ["İzin Güvenliği Yönetmeliği", "MADDE 27"],
        "document_type": "yönetmelik",
    }
    expander.verified_kinds[origin.source_id] = SourceKind.UNKNOWN
    assert expander._anchor(origin) is None


def test_generic_reference_extension_is_isolated_from_legacy_and_supersearch() -> None:
    from onyx.legal_composite.dependencies import _literal_names, _normalized_name
    from onyx.supersearch.dependencies import SupersearchDependencyExpander

    text = "6721 sayılı Liman Güvenliği Yönetmeliğinin 9 uncu maddesi uygulanır."
    legacy, _ledger, _calls, _broker, _court, _own, _expand = setup_expander(
        reference_text=text
    )
    legacy._collect(plan(), frontier={1})
    assert not legacy.edges
    assert _literal_names(text) == {}
    assert (
        _normalized_name("Liman Güvenliği Yönetmeliği") == "liman guvenligi yonetmeligi"
    )
    assert DependencyExpander in SupersearchDependencyExpander.__mro__
    assert CompositeDependencyExpander not in SupersearchDependencyExpander.__mro__
    composite, _ledger, _calls, _broker, _court, _own, _expand = (
        setup_composite_expander(reference_text=text)
    )
    composite._collect_material(
        plan(),
        material_targets=[
            material_target(
                instrument_name="Liman Güvenliği Yönetmeliği",
                instrument_number="6721",
                article="9",
            )
        ],
    )
    assert len(composite.edges) == 1


def test_unknown_norm_anchor_requires_own_formal_title_and_article_in_actual_body() -> (
    None
):
    expander, _ledger, _calls, _broker, _court, own, _expand = (
        setup_composite_expander()
    )
    expander.verified_kinds[str(own.id)] = SourceKind.UNKNOWN
    item = evidence_for_chunk(
        own,
        chunk(
            own,
            "6721 sayılı Liman Güvenliği Yönetmeliği\nMADDE 9 - Güvenlik belgesi gerekir.",
            0,
            title="6721 sayılı Liman Güvenliği Yönetmeliği",
            article="9",
        ),
    )
    anchor = expander._anchor(item)
    assert (
        anchor is not None and anchor.instrument_name == "Liman Güvenliği Yönetmeliği"
    )
    assert anchor.article_no == "9" and anchor.instrument_number == "6721"
    wrong_title = item.model_copy(deep=True)
    canonical = wrong_title.metadata["canonical_metadata"]
    assert isinstance(canonical, dict)
    canonical["title"] = "6721 sayılı Farklı Güvenlik Yönetmeliği"
    assert expander._anchor(wrong_title) is None
    without_title = evidence_for_chunk(
        own,
        chunk(
            own,
            "MADDE 9 - Güvenlik belgesi gerekir.",
            0,
            title="6721 sayılı Liman Güvenliği Yönetmeliği",
            article="9",
        ),
    )
    assert expander._anchor(without_title) is None
