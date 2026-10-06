"""Explicit source dependencies remain navigation until a matching legal attribution."""

import copy
import hashlib
from uuid import NAMESPACE_URL, uuid5

import pytest
from pydantic import JsonValue

from onyx.asv3.authority import explicit_reference_leads, native_named_authority_gap
from onyx.asv3.authority_requirements import AuthorityRequirements
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.llm_adapter import ResearchModel
from onyx.asv3.models import EvidenceItem, RunContext
from tests.unit.onyx.asv3.test_native_authority import original
from tests.unit.onyx.asv3.test_native_model_adapter import (
    adaptive_tool_view,
    last_payload,
    model,
    native_action,
)
from tests.unit.onyx.asv3.test_shared_originals import full_record

LOWER_TEXT = "Faaliyet Kanununun 27 inci maddesi uyarınca izin gerekir."
INTRO = "8917 sayılı Faaliyet Kanunu m.26 başka bir izin düzenler [2]."
CLAIM = "FK m.27 uyarınca izin gerekir [1]."


def opaque_original(
    title: str, article: str, *, text: str = "Başvuru gerekir.", kind: str = "kanun"
) -> EvidenceItem:
    item = original(title, article, kind=kind, text=text)
    source = str(uuid5(NAMESPACE_URL, title))
    chunk = "rc_" + hashlib.sha256((title + article + text).encode()).hexdigest()
    assert item.search_doc is not None
    return item.model_copy(
        update={
            "source_id": source,
            "chunk_id": chunk,
            "search_doc": item.search_doc.model_copy(
                update={
                    "document_id": source,
                    "metadata": {"regulatory_chunk_id": chunk},
                }
            ),
        }
    )


def setup(lower_text: str = LOWER_TEXT) -> tuple[RunContext, EvidenceLedger]:
    context = RunContext(
        scope={"corpus": "captured-authorized-scope"},
        services={
            "research_profile": "experimental",
            "experimental_parallel": True,
            "scenario_request": "What is the permission in this supplied scenario?",
            "independent_question": True,
            "task_id": "owned-independent-task",
        },
        depth=1,
        timeout_seconds=float("inf"),
    )
    ledger = EvidenceLedger()
    ledger.add(
        [
            opaque_original("Uygulama Tebliği", "3", text=lower_text, kind="tebliğ"),
            opaque_original("8917 SAYILI FAALİYET KANUNU", "26"),
        ],
        context,
    )
    context.services["evidence"] = ledger
    return context, ledger


def complete(ledger: EvidenceLedger, *numbers: int) -> list[dict[str, JsonValue]]:
    return [full_record(ledger, number) for number in numbers]


def test_explicit_reference_lead_is_deduplicated_and_only_complete_target_closes_it() -> (
    None
):
    context, ledger = setup()
    ledger.add(
        [opaque_original("Uygulama Tebliği", "4", text=LOWER_TEXT, kind="tebliğ")],
        context,
    )
    records = complete(ledger, 1, 2, 3)
    before = copy.deepcopy(records)
    leads = explicit_reference_leads(ledger, [*records, records[0]])
    assert len(leads) == 1
    assert leads[0] == {
        "instrument_number": "8917",
        "formal_name": "faaliyet kanun",
        "article": "27",
        "qualifier": None,
        "origin_citations": [1, 3],
        "matching_original_evidence": [],
    }
    ledger.add([opaque_original("8917 SAYILI FAALİYET KANUNU", "27")], context)
    assert explicit_reference_leads(ledger, records)[0][
        "matching_original_evidence"
    ] == [4]
    assert explicit_reference_leads(ledger, [*records, *complete(ledger, 4)]) == []
    assert records == before


@pytest.mark.parametrize(
    "changed", ["source", "chunk", "hash", "text", "range", "derived", "external"]
)
def test_reference_navigation_requires_its_exact_full_canonical_original(
    changed: str,
) -> None:
    context, ledger = setup()
    record = full_record(ledger, 1)
    if changed in {"derived", "external"}:
        item = ledger.get(1)
        statute = ledger.get(2)
        assert item is not None and statute is not None
        item.metadata[changed] = True
        ledger = EvidenceLedger()
        ledger.add([item, statute], context)
        record = full_record(ledger, 1)
    elif changed == "range":
        record.update(start_char=1, text=str(record["text"])[1:])
    else:
        field = {
            "source": "source_id",
            "chunk": "chunk_id",
            "hash": "text_hash",
            "text": "text",
        }[changed]
        record[field] = "different"
    assert explicit_reference_leads(ledger, [record]) == []


def test_ambiguous_formal_identity_is_not_guessed_but_explicit_number_is_a_lead() -> (
    None
):
    context, ledger = setup()
    ledger.add([opaque_original("8918 SAYILI FAALİYET KANUNU", "26")], context)
    assert explicit_reference_leads(ledger, complete(ledger, 1, 2, 3)) == []
    numbered = "8917 sayılı Faaliyet Kanununun 27 inci maddesi uyarınca izin gerekir."
    ledger.add(
        [opaque_original("Uygulama Tebliği", "4", text=numbered, kind="tebliğ")],
        context,
    )
    lead = explicit_reference_leads(ledger, complete(ledger, 2, 3, 4))
    assert lead[0]["instrument_number"] == "8917"
    assert lead[0]["article"] == "27"


def test_defined_statute_label_is_opt_in_and_requires_its_own_governing_original() -> (
    None
):
    context, ledger = setup()
    intro = INTRO.replace("Kanunu", "Kanunu (FK)")
    answer = intro + "\n\n" + CLAIM
    assert native_named_authority_gap(answer, ledger) is None
    gap = native_named_authority_gap(answer, ledger, resolve_defined_abbreviations=True)
    assert gap is not None
    missing = gap["named_authority_gaps"]
    assert isinstance(missing, list) and len(missing) == 1
    assert isinstance(missing[0], dict)
    assert missing[0]["instrument_number"] == "8917"
    assert missing[0]["article"] == "27"
    assert missing[0]["inline_evidence"] == [1]
    ledger.add([opaque_original("8917 SAYILI FAALİYET KANUNU", "27")], context)
    assert native_named_authority_gap(
        answer, ledger, resolve_defined_abbreviations=True
    )
    assert (
        native_named_authority_gap(
            answer.replace("[1]", "[1][3]"), ledger, resolve_defined_abbreviations=True
        )
        is None
    )


@pytest.mark.parametrize(
    "claim",
    [
        CLAIM.replace("27", "28"),
        "Uygulama Tebliği m.27 bu başvuruyu düzenler [1].",
        "TEBLİĞ m.27 bu başvuruyu düzenler [1].",
        "FK m.27 [1]",
        "# FK m.27 [1]",
        "FK m.27 özgün metni incelenemedi.",
        "8917 sayılı Faaliyet Kanunu m.27 özgün metni incelenemedi.",
    ],
)
def test_unrelated_article_instrument_and_nonoperative_or_precise_gap_are_not_dependencies(
    claim: str,
) -> None:
    _, ledger = setup()
    assert (
        native_named_authority_gap(
            INTRO + "\n\n" + claim, ledger, resolve_defined_abbreviations=True
        )
        is None
    )


def test_verified_quoted_lower_reference_is_not_an_asserted_governing_rule() -> None:
    _, ledger = setup(LOWER_TEXT + " FK m.27 uyarınca izin gerekir.")
    quote = f'Uygulama Tebliği şöyle der: "{LOWER_TEXT} FK m.27 uyarınca izin gerekir." [1].'
    assert (
        native_named_authority_gap(
            INTRO + "\n\n" + quote, ledger, resolve_defined_abbreviations=True
        )
        is None
    )


@pytest.mark.parametrize(
    "claim",
    ["FK m.27 [1]", "# FK m.27 [1]", "FK m.27 özgün metni incelenemedi."],
)
def test_defined_abbreviation_locator_and_precise_gap_are_not_legal_applications(
    claim: str,
) -> None:
    _, ledger = setup()
    answer = INTRO.replace("Kanunu", "Kanunu (FK)") + "\n\n" + claim
    assert (
        native_named_authority_gap(answer, ledger, resolve_defined_abbreviations=True)
        is None
    )


def test_literal_source_quotation_does_not_declare_an_answer_abbreviation() -> None:
    definition = "8917 sayılı Faaliyet Kanunu (BK) m.27 izin gerektirir."
    _, ledger = setup(definition)
    answer = f'Kaynak şöyle der: "{definition}" [1].\n\nBK m.27 izin gerektirir [1].'
    assert (
        native_named_authority_gap(answer, ledger, resolve_defined_abbreviations=True)
        is None
    )


@pytest.mark.parametrize(
    "claim",
    [
        CLAIM,
        "BK m.27 uyarınca izin gerekir [1].",
        "GY m.27 uyarınca izin gerekir [1].",
        "ABC m.27 uyarınca izin gerekir [1].",
        "BK m.27 bu işleme uygulanmaz [1].",
        "Uygulama Yönetmeliği m.27 izin gerektirir [1].",
    ],
)
def test_unknown_label_and_coincidental_lower_article_never_bind_a_statute(
    claim: str,
) -> None:
    _, ledger = setup()
    assert explicit_reference_leads(ledger, complete(ledger, 1, 2))
    assert (
        native_named_authority_gap(
            INTRO + "\n\n" + claim, ledger, resolve_defined_abbreviations=True
        )
        is None
    )


def test_multiple_cited_same_article_identities_do_not_invent_an_acronym_mapping() -> (
    None
):
    context, ledger = setup(
        LOWER_TEXT + " 8918 sayılı Başka Kanununun 27 inci maddesi uygulanır."
    )
    assert (
        native_named_authority_gap(
            INTRO + "\n\n" + CLAIM, ledger, resolve_defined_abbreviations=True
        )
        is None
    )
    ledger.add([opaque_original("8918 SAYILI BAŞKA KANUNU", "26")], context)
    ambiguous = (
        INTRO.replace("Kanunu", "Kanunu (FK)")
        + "\n\n8918 sayılı Başka Kanunu (FK) m.26 başvuru gerekir [3].\n\n"
        + CLAIM
    )
    assert (
        native_named_authority_gap(
            ambiguous, ledger, resolve_defined_abbreviations=True
        )
        is None
    )


def test_rejected_cited_dependency_survives_name_deletion_without_an_extra_model_stage() -> (
    None
):
    context, ledger = setup()
    ledger.record_delivery("first", "asv3_researcher", complete(ledger, 1, 2))
    state = AuthorityRequirements(context, "What is the permission?")
    answer = INTRO.replace("Kanunu", "Kanunu (FK)") + "\n\n" + CLAIM
    gap = native_named_authority_gap(answer, ledger, resolve_defined_abbreviations=True)
    assert state.publication_gap(answer, "first", context, ledger, native_gap=gap)
    assert state.publication_gap("İzin gerekir [1].", "first", context, ledger)
    retained = state.view(context)["retained_authority_requirements"]
    assert isinstance(retained, list) and len(retained) == 1


@pytest.mark.parametrize(
    "profile,parallel",
    [
        ("experimental", True),
        ("experimental", False),
        ("normal", True),
        ("deep", True),
    ],
)
def test_parallel_dispatcher_uses_ordinary_reference_payload_without_research_overrides(
    profile: str, parallel: bool
) -> None:
    context, ledger = setup()
    context.depth = 0
    context.services.update(research_profile=profile, experimental_parallel=parallel)
    selected, secondary = model(), model()
    source = ledger.get(1)
    assert source is not None
    selected.invoke.return_value = native_action(
        "read_provision", {"source_id": source.source_id}
    )
    before = ledger.export()
    adapter = ResearchModel(
        selected,
        context,
        research_llm=secondary if profile == "experimental" else None,
        lean_native_mode=True,
    )
    adapter.decide(adaptive_tool_view(original_evidence=complete(ledger, 1, 2)))
    payload = last_payload(selected)
    assert "research_navigation" not in payload
    assert selected.invoke.call_count == 1
    secondary.invoke.assert_not_called()
    assert ledger.export()["records"] == before["records"]
    assert adapter.last_call_id is not None
    assert ledger.completely_delivered(adapter.last_call_id) == {1, 2}
