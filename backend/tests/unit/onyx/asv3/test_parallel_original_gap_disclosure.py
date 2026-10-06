"""Pure source disclosures close only their explicitly owned historical dependency."""

import copy
import hashlib

import pytest
from pydantic import JsonValue

from onyx.asv3.authority import native_named_authority_gap
from onyx.asv3.authority_requirements import AuthorityRequirements
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.llm_adapter import COORDINATOR_SESSION_ACTIONS, ResearchModel
from onyx.asv3.models import RunContext
from onyx.prompts.asv3.experimental import (
    EXPERIMENTAL_COORDINATOR_PROMPT,
    EXPERIMENTAL_PARALLEL_COORDINATOR,
    EXPERIMENTAL_RESEARCHER_PROMPT,
)
from tests.unit.onyx.asv3.test_native_authority import ledger_with, original
from tests.unit.onyx.asv3.test_native_model_adapter import model
from tests.unit.onyx.asv3.test_shared_originals import full_record

CLAIMS = (
    "8917 sayılı Faaliyet Kanunu m.27 uyarınca faiz gerekir [1].\n\n"
    "7251 sayılı Veri Kanunu hükümleri oranı belirler [1].\n\n"
    "8917 sayılı Faaliyet Kanunu m.31 uyarınca iade yapılır [1]."
)
PAIR = (
    "Bu yanıt, ödemenin faiz veya ceza sonuçları hakkında bir belirleme yapmıyor; "
    "8917 sayılı Faaliyet Kanunu m.27 ve 7251 sayılı Veri Kanunu hükümlerinin "
    "özgün metinleri bu sınırlı usul incelemesinde değerlendirilmedi."
)
ALL = (
    "Bu yanıt, ödemenin ceza veya faiz sonucunu belirlemiyor; bu sınırlı usul "
    "incelemesinde 8917 sayılı Faaliyet Kanunu m.27 ve 7251 sayılı Veri Kanun’un "
    "özgün metinleri değerlendirilmedi. Faaliyet Kanunu m.31 kapsamında "
    "geri verme/kaldırma sonucu hakkında da bir belirleme yapılmıyor; "
    "bu hükmün özgün metni bu incelemede ele alınmadı."
)
GROUP = (
    "Bu yanıt 8917 sayılı Faaliyet Kanunu m.27, 7251 sayılı Veri Kanunu veya "
    "8917 sayılı Kanun m.31’in hukuki etkisi hakkında sonuç bildirmemektedir; "
    "bu hükümlerin özgün metinleri bu sınırlı incelemede değerlendirilmemiştir."
)


def setup(
    *, parallel: bool = True
) -> tuple[RunContext, EvidenceLedger, AuthorityRequirements]:
    context = RunContext(scope={"authorized": "owned"})
    context.services["task_id"] = "procedure-owner"
    ledger = ledger_with(
        original("Uygulama Tebliği", "3", kind="tebliğ", text=CLAIMS),
        original("8917 sayılı Faaliyet Kanunu", "28"),
        original("7251 sayılı Veri Kanunu", "9"),
    )
    ledger.record_delivery("first", "asv3_researcher", [full_record(ledger, 1)])
    state = AuthorityRequirements(
        context, "Explain the procedure", syntactic_reference_binding=parallel
    )
    assert state.publication_gap(CLAIMS, "first", context, ledger)
    return context, ledger, state


def remaining(gap: dict[str, JsonValue] | None) -> set[tuple[str, str | None]]:
    rows = (gap or {}).get("retained_authority_requirements", [])
    assert isinstance(rows, list)
    result: set[tuple[str, str | None]] = set()
    for row in rows:
        assert isinstance(row, dict)
        number, article = row["instrument_number"], row["article"]
        assert isinstance(number, str)
        assert article is None or isinstance(article, str)
        result.add((number, article))
    return result


@pytest.mark.parametrize(
    "notice",
    [
        PAIR,
        "8917 sayılı Faaliyet Kanunu m.27 özgün metni incelenmedi.\n\n"
        "7251 sayılı Veri Kanunu özgün metni incelenmedi.",
        "8917 sayılı Faaliyet Kanunu m.27 ile 7251 sayılı Veri Kanunu "
        "özgün metinleri ele alınmadı.",
        "Faaliyet Kanunu m.27 ve Veri Kanunu özgün metinleri incelenmedi.",
    ],
)
def test_two_disclosures_leave_undisclosed_sibling_open_without_deleting_history(
    notice: str,
) -> None:
    context, ledger, state = setup()
    retained = copy.deepcopy(state.export())
    answer = "Başvuru belgesi uygulama metninde açıklanır [1].\n\n" + notice
    gap = state.publication_gap(answer, "first", context, ledger)
    assert remaining(gap) == {("8917", "31")}
    assert not (gap or {}).get("named_authority_gaps")
    assert state.export() == retained


@pytest.mark.parametrize("notice", [ALL, GROUP])
def test_immediate_provision_antecedent_closes_three_bound_gaps_and_restores(
    notice: str,
) -> None:
    context, ledger, state = setup()
    retained = copy.deepcopy(state.export())
    answer = "Başvuru belgesi uygulama metninde açıklanır [1].\n\n" + notice
    assert state.publication_gap(answer, "first", context, ledger) is None
    assert state.export() == retained
    restored = AuthorityRequirements(
        context, "Explain the procedure", syntactic_reference_binding=True
    )
    restored.restore(retained, context, "Explain the procedure")
    assert restored.publication_gap(answer, "first", context, ledger) is None
    assert remaining(
        restored.publication_gap("İade yapılır [1].", "first", context, ledger)
    ) == {("8917", "27"), ("7251", None), ("8917", "31")}
    context.services["task_id"] = "different-owner"
    assert restored.view(context)["retained_authority_requirements"] == []


@pytest.mark.parametrize(
    "notice",
    [
        "8917 sayılı Faaliyet Kanunu'nun 27. maddesinin özgün metni incelenemedi.",
        "8917 sayılı Faaliyet Kanunu m.27 özgün metni değerlendirilmedi.",
        "Faaliyet Kanunu m.27 özgün metni bu incelemede ele alınmadı.",
        "Law No. 8917 Article 27 original text has not been examined.",
        "This review makes no determination about Law No. 8917 Article 27; "
        "this provision's original text has not been reviewed.",
    ],
)
def test_literal_identity_and_pure_predicate_use_the_same_native_and_retained_path(
    notice: str,
) -> None:
    context, ledger, state = setup()
    assert (
        native_named_authority_gap(
            notice,
            ledger,
            strict_reference_boundaries=True,
            resolve_defined_abbreviations=True,
            syntactic_reference_binding=True,
        )
        is None
    )
    gap = state.publication_gap(
        "Belge bilgisi [1].\n\n" + notice, "first", context, ledger
    )
    assert remaining(gap) == {("7251", None), ("8917", "31")}


@pytest.mark.parametrize(
    "mixed",
    [
        "İzin gerekir; " + PAIR,
        PAIR + " İade otomatik yapılır.",
        PAIR + " Faaliyet Kanunu m.31 uygulanmaz.",
        PAIR + " Başvuru 15 gün içinde yapılmalıdır.",
        PAIR + " Vergi oranı yüzde 2'dir.",
        "Bu yanıt, izin gerekir ve faiz sonucunu belirlemiyor; "
        "8917 sayılı Faaliyet Kanunu m.27 özgün metni incelenmedi.",
        "Bu yanıt, ek vergi tahakkuk eder ve faiz sonucunu belirlemiyor; "
        "8917 sayılı Faaliyet Kanunu m.27 özgün metni incelenmedi.",
        "Bu yanıt, 8917 sayılı Faaliyet Kanunu m.27 kapsamında faiz ödemek "
        "zorundasınız ve faiz/ceza sonucunu belirlemiyor; "
        "8917 sayılı Faaliyet Kanunu m.27 özgün metni incelenmedi.",
        "Bu yanıt, 8917 sayılı Faaliyet Kanunu m.27 kapsamında ceza hukuka "
        "uygundur ve faiz/ceza sonucunu belirlemiyor; "
        "8917 sayılı Faaliyet Kanunu m.27 özgün metni incelenmedi.",
        "Bu yanıt, 8917 sayılı Faaliyet Kanunu m.27 kapsamında itiraz "
        "hakkınız yoktur ve faiz/ceza sonucunu belirlemiyor; "
        "8917 sayılı Faaliyet Kanunu m.27 özgün metni incelenmedi.",
        "8917 sayılı Faaliyet Kanunu m.27 izin verir özgün metni incelenmedi.",
        "8917 sayılı Faaliyet Kanunu m.27 özgün metni incelenmedi [1].",
        "8917 sayılı Faaliyet Kanunu m.27 kuralının olmadığı belirlendi.",
        "8917 sayılı Faaliyet Kanunu m.27 özgün metni elde edilemedi; "
        "bu nedenle ceza yoktur.",
    ],
)
def test_every_clause_must_be_pure_negative_disclosure(mixed: str) -> None:
    context, ledger, state = setup()
    gap = state.publication_gap(
        "Belge bilgisi [1].\n\n" + mixed, "first", context, ledger
    )
    assert remaining(gap) == {("8917", "27"), ("7251", None), ("8917", "31")}


@pytest.mark.parametrize(
    "notice",
    [
        "Bu yanıt, faiz sonucunu belirlemiyor; bu hükmün özgün metni incelenmedi.",
        "Faaliyet Kanunu m.27 ve m.31 sonucu hakkında bir belirleme yapılmıyor; "
        "bu hükmün özgün metni incelenmedi.",
        "Faaliyet Kanunu m.27 sonucu hakkında bir belirleme yapılmıyor; "
        "Veri Kanunu sonucu hakkında bir belirleme yapılmıyor; "
        "bu hükmün özgün metni incelenmedi.",
        "8917 sayılı Faaliyet Kanunu m.27 özgün metni incelenmedi; "
        "bu hükmün özgün metni incelenmedi.",
        "Faaliyet Kanunu m.27 sonucu hakkında bir belirleme yapılmıyor; "
        "7251 sayılı Veri Kanunu özgün metni incelenmedi.",
        "8917 sayılı Faaliyet Kanunu özgün metni incelenmedi.",
        "8918 sayılı Faaliyet Kanunu m.27 özgün metni incelenmedi.",
        "8917 sayılı Faaliyet Kanunu m.28 özgün metni incelenmedi.",
        "8917 sayılı Faaliyet Kanunu geçici m.27 özgün metni incelenmedi.",
        "8917 sayılı Faaliyet Kanunu m.27, 15 gün özgün metni incelenmedi.",
        "8917 sayılı Faaliyet Kanunu m.27, yüzde 2 özgün metni incelenmedi.",
        "8917 sayılı Faaliyet Kanunu m.27, 30 Eylül 2026 özgün metni incelenmedi.",
        "FK m.27 özgün metni incelenmedi.",
        "Bu yanıt, faiz sonucunu belirlemiyor; bu hükümlerin özgün metinleri "
        "değerlendirilmemiştir.",
        "Faaliyet Kanunu m.27 hukuki etkisi hakkında sonuç bildirmemektedir; "
        "bu hükümlerin özgün metinleri değerlendirilmemiştir.",
        GROUP.replace("veya", "uyarınca ceza hukuka uygundur veya"),
    ],
)
def test_unknown_ambiguous_or_wrong_subject_cannot_close_exact_article(
    notice: str,
) -> None:
    context, ledger, state = setup()
    gap = state.publication_gap(
        "Belge bilgisi [1].\n\n" + notice, "first", context, ledger
    )
    assert ("8917", "27") in remaining(gap)


def test_explicit_current_answer_acronym_closes_only_its_actual_identity() -> None:
    context, ledger, state = setup()
    ledger.record_delivery(
        "next", "asv3_researcher", [full_record(ledger, 1), full_record(ledger, 2)]
    )
    declared = "8917 sayılı Faaliyet Kanunu (FK) düzenlemesi incelendi [2].\n\n"
    notice = "FK m.27 sonucu hakkında bir belirleme yapılmıyor; bu hükmün özgün metni ele alınmadı."
    gap = state.publication_gap(declared + notice, "next", context, ledger)
    assert remaining(gap) == {("7251", None), ("8917", "31")}
    assert state.publication_gap(notice, "next", context, ledger) is not None
    explicit_original = notice.replace(
        "bu hükmün özgün metni", "8917 sayılı Faaliyet Kanunu m.27 özgün metni"
    )
    assert remaining(
        state.publication_gap(declared + explicit_original, "next", context, ledger)
    ) == {("7251", None), ("8917", "31")}


@pytest.mark.parametrize("reference_list", ["FK m.27 ve VK m.11", "VK m.11 ve FK m.27"])
@pytest.mark.parametrize("reverse_declarations", [False, True])
def test_declared_identity_list_consumption_follows_text_position(
    reference_list: str, reverse_declarations: bool
) -> None:
    context, ledger, state = setup()
    ledger.record_delivery(
        "all", "asv3_researcher", [full_record(ledger, number) for number in (1, 2, 3)]
    )
    assert ("7251", "11") in remaining(
        state.publication_gap(
            "7251 sayılı Veri Kanunu m.11 uyarınca izin gerekir [1].",
            "all",
            context,
            ledger,
        )
    )
    declarations = [
        "8917 sayılı Faaliyet Kanunu (FK) özgün düzenlemesi incelendi [2].",
        "7251 sayılı Veri Kanunu (VK) özgün düzenlemesi incelendi [3].",
    ]
    if reverse_declarations:
        declarations.reverse()
    answer = "\n\n".join(
        declarations + [reference_list + " özgün metinleri incelenmedi."]
    )
    retained = copy.deepcopy(state.export())
    assert (
        native_named_authority_gap(
            answer,
            ledger,
            strict_reference_boundaries=True,
            resolve_defined_abbreviations=True,
            syntactic_reference_binding=True,
        )
        is None
    )
    assert remaining(state.publication_gap(answer, "all", context, ledger)) == {
        ("8917", "31")
    }
    assert state.export() == retained


def test_legacy_behavior_and_checkpoint_bytes_are_unchanged() -> None:
    context, ledger, state = setup(parallel=False)
    retained = state.export()
    assert "reference_policy" not in retained
    before = copy.deepcopy(retained)
    assert state.publication_gap(
        "Belge bilgisi [1].\n\n" + ALL, "first", context, ledger
    )
    assert state.export() == before
    parallel = AuthorityRequirements(
        context, "Explain the procedure", syntactic_reference_binding=True
    )
    with pytest.raises(ValueError, match="policy changed"):
        parallel.restore(before, context, "Explain the procedure")


def test_ambiguous_formal_identity_and_other_owner_cannot_fill_subject() -> None:
    context, ledger, state = setup()
    context.services["task_id"] = "other-owner"
    assert state.publication_gap(
        "9259 sayılı Olay Kanunu m.7 izin gerektirir [1].", "first", context, ledger
    )
    context.services["task_id"] = "procedure-owner"
    notice = "Faaliyet Kanunu m.27 ve Olay Kanunu m.7 özgün metinleri incelenmedi."
    assert ("8917", "27") in remaining(
        state.publication_gap(
            "Belge bilgisi [1].\n\n" + notice, "first", context, ledger
        )
    )
    ledger.add([original("8918 sayılı Faaliyet Kanunu", "27")], context)
    assert ("8917", "27") in remaining(
        state.publication_gap(
            "Belge bilgisi [1].\n\nFaaliyet Kanunu m.27 özgün metni incelenmedi.",
            "first",
            context,
            ledger,
        )
    )


@pytest.mark.parametrize(
    "notice",
    [
        "8917 sayılı faiz ödemek zorundasınız Faaliyet Kanunu m.27 özgün metni incelenmedi.",
        "8917 sayılı ceza hukuka uygundur Faaliyet Kanunu m.27 özgün metni incelenmedi.",
        "8917 sayılı itiraz hakkınız yoktur Faaliyet Kanunu m.27 özgün metni incelenmedi.",
        "Bu yanıt, 8917 sayılı ceza hukuka uygundur Faaliyet Kanunu m.27 "
        "sonucunu belirlemiyor; bu hükmün özgün metni incelenmedi.",
        "7251 sayılı Faaliyet Kanunu m.27 özgün metni incelenmedi.",
        "8917 sayılı Veri Kanunu m.27 özgün metni incelenmedi.",
    ],
)
def test_numbered_title_cannot_mask_prose_or_a_conflicting_canonical_identity(
    notice: str,
) -> None:
    context, ledger, state = setup()
    assert (
        native_named_authority_gap(
            notice,
            ledger,
            strict_reference_boundaries=True,
            resolve_defined_abbreviations=True,
            syntactic_reference_binding=True,
        )
        is not None
    )
    assert ("8917", "27") in remaining(
        state.publication_gap(
            "Belge bilgisi [1].\n\n" + notice, "first", context, ledger
        )
    )


@pytest.mark.parametrize(
    "notice",
    [
        "8917 sayılı İzin Gerekir Faaliyet Kanunu m.27 özgün metni incelenmedi.",
        "İzin Gerekir Faaliyet Kanunu m.27 özgün metni incelenmedi.",
        "Bu yanıt, İzin Gerekir Faaliyet Kanunu m.27 sonucu hakkında "
        "bir belirleme yapmıyor; bu hükmün özgün metni incelenmedi.",
    ],
)
def test_rejected_model_title_cannot_bootstrap_a_later_masking_alias(
    notice: str,
) -> None:
    context = RunContext(scope={"authorized": "owned"})
    context.services["task_id"] = "procedure-owner"
    ledger = ledger_with(original("Uygulama Tebliği", "3", kind="tebliğ"))
    ledger.record_delivery("first", "asv3_researcher", [full_record(ledger, 1)])
    state = AuthorityRequirements(context, "Explain", syntactic_reference_binding=True)
    rejected = "8917 sayılı İzin Gerekir Faaliyet Kanunu m.27 uygulanır [1]."
    assert remaining(state.publication_gap(rejected, "first", context, ledger)) == {
        ("8917", "27")
    }
    retained = copy.deepcopy(state.export())
    assert remaining(
        state.publication_gap(
            "Belge bilgisi [1].\n\n" + notice, "first", context, ledger
        )
    ) == {("8917", "27")}
    assert state.export() == retained
    numeric = "Belge bilgisi [1].\n\n8917 sayılı Kanun m.27 özgün metni incelenmedi."
    assert state.publication_gap(numeric, "first", context, ledger) is None
    assert state.export() == retained


def test_declared_identity_label_cannot_mask_a_predicate_inside_a_scope_subject() -> (
    None
):
    context, ledger, state = setup()
    declaration = "8917 sayılı Faaliyet Kanunu (YOKTUR) düzenlemesi incelendi [2].\n\n"
    mixed = declaration + (
        "Bu yanıt, ceza YOKTUR m.27 sonucunu belirlemiyor; "
        "bu hükmün özgün metni incelenmedi."
    )
    assert (
        native_named_authority_gap(
            mixed,
            ledger,
            strict_reference_boundaries=True,
            resolve_defined_abbreviations=True,
            syntactic_reference_binding=True,
        )
        is not None
    )
    assert ("8917", "27") in remaining(
        state.publication_gap(mixed, "first", context, ledger)
    )
    pure = declaration + "YOKTUR m.27 özgün metni incelenmedi."
    assert ("8917", "27") not in remaining(
        state.publication_gap(pure, "first", context, ledger)
    )


@pytest.mark.parametrize(
    "base,expected_sha",
    [
        (
            EXPERIMENTAL_COORDINATOR_PROMPT,
            "03537ac639eb90d21592140d522d1b370a6ef1e7324de21ef1fa0553c8a34ea1",
        ),
        (
            EXPERIMENTAL_RESEARCHER_PROMPT,
            "87a37be96b0b9e844f38ca3c047c7d28589df4e8cabc14fd03f17e32b63d59fe",
        ),
    ],
)
def test_serial_experimental_prompt_bytes_remain_the_protected_baseline(
    base: str, expected_sha: str
) -> None:
    assert hashlib.sha256(base.encode()).hexdigest() == expected_sha


@pytest.mark.parametrize("depth", [0, 1])
def test_actual_native_prompt_routing_keeps_serial_and_removes_old_parallel_child(
    depth: int,
) -> None:
    base = (
        EXPERIMENTAL_RESEARCHER_PROMPT
        if depth
        else EXPERIMENTAL_COORDINATOR_PROMPT + "\n\n" + COORDINATOR_SESSION_ACTIONS
    )
    context = RunContext(depth=depth, services={"research_profile": "experimental"})
    adapter = ResearchModel(model(), context, lean_native_mode=True)
    assert adapter._research_instruction() == base
    context.services["experimental_parallel"] = True
    if depth:
        with pytest.raises(ValueError, match="ordinary Experimental session"):
            adapter._research_instruction()
    else:
        assert adapter._research_instruction() == (
            base + "\n\n" + EXPERIMENTAL_PARALLEL_COORDINATOR
        )
    context.services["experimental_parallel"] = False
    assert adapter._research_instruction() == base
