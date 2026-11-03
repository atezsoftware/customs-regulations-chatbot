"""Captured213/0213 law identity must never select the3213 substring candidate."""

import json
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Any, cast
from uuid import UUID

import pytest

from onyx.asv3.authority import known_statute_source_ids
from onyx.asv3.cheap_authority import canonical_named_authority_gap
from onyx.asv3.corpus_tools import build_corpus_specs, evidence_for_chunk
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import OutcomeStatus, RunContext
from onyx.db import asv3_corpus
from onyx.db.asv3_corpus import CorpusChunk, CorpusSource
from onyx.db.models import User
from onyx.regulatory.law_identity import canonical_law_number, own_law_filename_number
from tests.unit.onyx.asv3.test_corpus_source_sandbox import MemoryBroker
from tests.unit.onyx.asv3.test_numbered_source_resolution import catalogue, filters

_CAPTURE = json.loads(
    (
        Path(__file__).parent / "fixtures" / "numbered_law_213_candidates_7468.json"
    ).read_text()
)
_VUK = UUID("d3c15591-f202-45e2-9f91-7fa5254adad2")
_MINING = UUID("d04fd7ab-d799-4b4b-acc6-a0826749e6da")


def chunk(row: dict[str, Any], source_id: UUID) -> CorpusChunk:
    return CorpusChunk(
        row.get("chunk_id", row.get("id")),
        source_id,
        row["text"],
        row["position"],
        row["projection_ordinal"],
        tuple(row["heading_path"]),
        row["chunk_metadata"],
        date.fromisoformat(row["validity_start_date"])
        if row["validity_start_date"]
        else None,
        date.fromisoformat(row["validity_end_date"])
        if row["validity_end_date"]
        else None,
        row.get("chunk_status", row.get("status")),
    )


def candidates() -> tuple[list[CorpusSource], dict[UUID, list[CorpusChunk]]]:
    sources: dict[UUID, CorpusSource] = {}
    heads: dict[UUID, list[CorpusChunk]] = {}
    for row in _CAPTURE["candidate_heads"]:
        identifier = UUID(row["id"])
        sources[identifier] = CorpusSource(
            identifier, row["name"], "captured-canonical"
        )
        heads.setdefault(identifier, []).append(chunk(row, identifier))
    return list(sources.values()), heads


@pytest.mark.parametrize("query", ["213 sayılı Kanun", "0213 sayılı Kanun"])
def test_actual_named_provision_reads_the_verified_law_then_keeps_its_canonical_identity(
    monkeypatch: pytest.MonkeyPatch, query: str
) -> None:
    sources, heads = candidates()
    session = catalogue(monkeypatch, [sources])
    source_filter = filters()
    user = cast(User, object())
    observed: list[str] = []

    def read_heads(**arguments: Any) -> dict[UUID, list[CorpusChunk]]:
        assert arguments["user"] is user and arguments["filters"] is source_filter
        arguments["check_active"]()
        return {source.id: heads[source.id] for source in arguments["sources"]}

    monkeypatch.setattr(asv3_corpus, "_read_identity_heads_batch", read_heads)
    broker = MemoryBroker([chunk(row, _VUK) for row in _CAPTURE["provision"]])
    broker.item = next(source for source in sources if source.id == _VUK)

    def lookup(
        name: str, context: RunContext, **kwargs: Any
    ) -> tuple[list[CorpusSource], bool]:
        observed.append(name)
        return asv3_corpus.find_sources(
            session,
            user=user,
            filters=source_filter,
            query=name,
            limit=kwargs["limit"],
            allow_canonical_filename_fallback=True,
            check_active=context.check_active,
        )

    monkeypatch.setattr(broker, "sources", lookup)
    monkeypatch.setattr(broker, "related_sources_for_provision", lambda *_args: None)
    ledger = EvidenceLedger()
    context = RunContext(services={"asv3_exp_cropped": True, "evidence": ledger})
    tool = next(
        spec
        for spec in build_corpus_specs(broker, named_provision_reads=True)
        if spec.name == "read_named_provision"
    )
    outcome = tool.handler({"source_name": query, "article": "315"}, context)
    assert outcome.status == OutcomeStatus.FOUND
    assert observed == [query]
    assert len(outcome.evidence) == 1
    assert outcome.evidence[0].source_id == str(_VUK)
    assert outcome.evidence[0].text == _CAPTURE["provision"][0]["text"]
    ledger.add(outcome.evidence, context)
    before = ledger.export()
    for name in ("213 sayılı Kanun", "0213 sayılı Vergi Usul Kanunu"):
        assert known_statute_source_ids(name, ledger) == [str(_VUK)]
    assert (
        canonical_named_authority_gap(
            "213 sayılı Vergi Usul Kanunu m.315 uygulanır [1].",
            ledger,
            strict_reference_boundaries=True,
            syntactic_reference_binding=True,
        )
        is None
    )
    assert ledger.export() == before


def test_actual_opening_proves_law_number_not_the_gazette_number() -> None:
    sources, heads = candidates()
    valid = next(source for source in sources if source.id == _VUK)
    mining = next(source for source in sources if source.id == _MINING)
    assert heads[_VUK][0].metadata["document_number"] == "10703 - 10705"
    assert asv3_corpus._source_has_own_law_number(valid, heads[_VUK], "213")
    assert not asv3_corpus._source_has_own_law_number(mining, heads[_MINING], "213")


def test_numeric_law_filename_still_requires_own_canonical_number_proof() -> None:
    sources, heads = candidates()
    source = replace(
        next(source for source in sources if source.id == _VUK),
        name="Kanunlar/0213_sayili_kanun.md",
    )
    assert asv3_corpus._source_has_own_law_number(source, heads[_VUK], "213")
    without_opening = [replace(row, text="Sadece hüküm metni.") for row in heads[_VUK]]
    assert not asv3_corpus._source_has_own_law_number(source, without_opening, "213")


@pytest.mark.parametrize(
    "mutation",
    [
        "decision",
        "khk",
        "untrusted",
        "derived",
        "external",
        "truncated",
        "wrong_source",
        "prefix",
        "wrong_title",
        "quoted_number",
        "conflicting_number",
        "zero",
    ],
)
def test_unproved_or_foreign_candidate_cannot_become_the213_original(
    mutation: str,
) -> None:
    sources, heads = candidates()
    source = next(source for source in sources if source.id == _VUK)
    originals = [replace(row, metadata=dict(row.metadata)) for row in heads[_VUK]]
    first = originals[0]
    if mutation == "decision":
        originals = [
            replace(row, metadata={**row.metadata, "document_type": "court_decision"})
            for row in originals
        ]
    elif mutation == "khk":
        originals = [
            replace(row, heading_path=("213 SAYILI KANUN HÜKMÜNDE KARARNAME",))
            for row in originals
        ]
    elif mutation in {"untrusted", "derived", "external", "truncated"}:
        originals[0] = replace(first, metadata={**first.metadata, mutation: True})
    elif mutation == "wrong_source":
        originals[0] = replace(first, source_id=_MINING)
    elif mutation == "prefix":
        source = replace(
            source, name="Diğer Kanunlar/0213_untrusted_prefix_vergi_usul_kanunu.md"
        )
    elif mutation == "wrong_title":
        originals = [replace(row, heading_path=("MADEN KANUNU",)) for row in originals]
    elif mutation == "quoted_number":
        originals[0] = replace(
            first, text="BAŞKA KARAR\nVERGİ USUL KANUNU\nKanun Numarası :213"
        )
    elif mutation == "conflicting_number":
        originals[0] = replace(
            first,
            text=first.text.replace("Kanun Numarası : 213", "Kanun Numarası :3213"),
        )
    else:
        source = replace(source, name=source.name.replace("0213_", "0000_"))
    assert not asv3_corpus._source_has_own_law_number(source, originals, "213")


@pytest.mark.parametrize("fence", ["denied", "unpublished"])
def test_candidate_identity_read_never_receives_an_unauthorized_source(
    monkeypatch: pytest.MonkeyPatch, fence: str
) -> None:
    sources, heads = candidates()
    session = catalogue(
        monkeypatch,
        [sources],
        **({"denied": {_VUK}} if fence == "denied" else {"unpublished": {_VUK}}),
    )
    seen: list[UUID] = []

    def read_heads(**arguments: Any) -> dict[UUID, list[CorpusChunk]]:
        seen.extend(source.id for source in arguments["sources"])
        assert _VUK not in seen
        return {identifier: heads[identifier] for identifier in seen}

    monkeypatch.setattr(asv3_corpus, "_read_identity_heads_batch", read_heads)
    assert asv3_corpus.find_sources(
        session,
        user=cast(User, object()),
        filters=filters(),
        query="213 sayılı Kanun",
        allow_canonical_filename_fallback=True,
    ) == ([], False)


@pytest.mark.parametrize("query", ["0 sayılı Kanun", "0000 sayılı Kanun"])
def test_zero_is_not_a_law_identity(
    monkeypatch: pytest.MonkeyPatch, query: str
) -> None:
    sources, _heads = candidates()
    session = catalogue(monkeypatch, [sources])
    assert asv3_corpus.find_sources(
        session,
        user=cast(User, object()),
        filters=filters(),
        query=query,
        allow_canonical_filename_fallback=True,
    ) == ([], False)
    assert canonical_law_number(query.split()[0]) is None


def test_other_workflows_keep_existing_candidate_discovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources, _heads = candidates()
    session = catalogue(monkeypatch, [sources])
    assert asv3_corpus.find_sources(
        session, user=cast(User, object()), filters=filters(), query="213 sayılı Kanun"
    ) == (sources, False)


def test_unseen_candidate_pages_do_not_become_a_unique_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources, heads = candidates()
    valid = next(source for source in sources if source.id == _VUK)
    other = next(source for source in sources if source.id == _MINING)
    session = catalogue(monkeypatch, [[valid, other]])
    monkeypatch.setattr(
        asv3_corpus,
        "_read_identity_heads_batch",
        lambda **kwargs: {source.id: heads[source.id] for source in kwargs["sources"]},
    )
    assert asv3_corpus.find_sources(
        session,
        user=cast(User, object()),
        filters=filters(),
        query="213 sayılı Kanun",
        limit=1,
        allow_canonical_filename_fallback=True,
    ) == ([valid], True)


def test_canonical_head_fence_failure_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources, _heads = candidates()
    session = catalogue(monkeypatch, [sources])

    def changed(**_kwargs: Any) -> dict[UUID, list[CorpusChunk]]:
        raise asv3_corpus.CorpusScopeUnavailable(
            "Publication changed during identity read"
        )

    monkeypatch.setattr(asv3_corpus, "_read_identity_heads_batch", changed)
    with pytest.raises(asv3_corpus.CorpusScopeUnavailable):
        asv3_corpus.find_sources(
            session,
            user=cast(User, object()),
            filters=filters(),
            query="213 sayılı Kanun",
            allow_canonical_filename_fallback=True,
        )


@pytest.mark.parametrize(
    "mutation",
    ["untrusted", "truncated", "decision", "khk", "prefix", "zero", "mining"],
)
def test_retained_identity_cannot_promote_a_foreign_or_untrusted_prefix(
    mutation: str,
) -> None:
    sources, _heads = candidates()
    source = next(source for source in sources if source.id == _VUK)
    row = chunk(_CAPTURE["provision"][0], _VUK)
    metadata = dict(row.metadata)
    if mutation in {"untrusted", "truncated"}:
        metadata[mutation] = True
    elif mutation == "decision":
        metadata["document_type"] = "court_decision"
    elif mutation == "khk":
        row = replace(
            row, heading_path=("213 SAYILI KANUN HÜKMÜNDE KARARNAME", "MADDE 315")
        )
    elif mutation == "prefix":
        source = replace(source, name=source.name.replace("0213_", "0213_fake_prefix_"))
    elif mutation == "zero":
        source = replace(source, name=source.name.replace("0213_", "0000_"))
    else:
        source = replace(source, name=source.name.replace("0213_", "3213_"))
    row = replace(row, metadata=metadata)
    ledger = EvidenceLedger()
    ledger.add([evidence_for_chunk(source, row)], RunContext())
    assert known_statute_source_ids("213 sayılı Kanun", ledger) == []


def test_numbered_filename_must_match_the_whole_formal_title() -> None:
    assert (
        own_law_filename_number("0213_sayili_vergi_usul_kanunu.md", "VERGİ USUL KANUNU")
        == "213"
    )
    assert (
        own_law_filename_number("0213_sayili_maden_kanunu.md", "VERGİ USUL KANUNU")
        is None
    )


@pytest.mark.parametrize(
    "field", ["213 - 3213", "213 ve 3213", "213 sayılı başka hükme atıf", "0000"]
)
@pytest.mark.parametrize("numbered_root", [False, True])
def test_malformed_own_field_cannot_be_overridden_by_a_numbered_root(
    field: str, numbered_root: bool
) -> None:
    sources, heads = candidates()
    source = next(source for source in sources if source.id == _VUK)
    originals = [
        replace(
            row,
            heading_path=(
                (
                    "213 SAYILI VERGİ USUL KANUNU"
                    if numbered_root
                    else "VERGİ USUL KANUNU"
                ),
                *row.heading_path[1:],
            ),
        )
        for row in heads[_VUK]
    ]
    originals[0] = replace(
        originals[0], text=f"VERGİ USUL KANUNU\nKanun Numarası : {field}"
    )
    assert not asv3_corpus._source_has_own_law_number(source, originals, "213")


@pytest.mark.parametrize(
    "text",
    [
        "VERGİ USUL KANUNU\nMADEN KANUNU\nKanun Numarası :213",
        "VERGİ USUL KANUNU\n“Başka düzenlemede:\nKanun Numarası :213",
        "3213 sayılı Vergi Usul Kanunu\nKanun Numarası :213",
        "VERGİ USUL KANUNU\nKanun Numarası :213\nKanun Numarası :3213",
        "VERGİ USUL KANUNU\n"
        + " " * (1600 - len("Kanun Numarası : ") - 3)
        + "Kanun Numarası : 2139",
        "VERGİ USUL KANUNU\n" + "\n" * 13 + "Kanun Numarası :213",
    ],
)
def test_complete_contiguous_opening_is_required(text: str) -> None:
    sources, heads = candidates()
    source = next(source for source in sources if source.id == _VUK)
    originals = list(heads[_VUK])
    originals[0] = replace(originals[0], text=text)
    assert not asv3_corpus._source_has_own_law_number(source, originals, "213")


@pytest.mark.parametrize(
    "name,root",
    [
        ("213 sayılı Vergi Usul Kanunu", "3213 SAYILI VERGİ USUL KANUNU"),
        ("3213_sayili_vergi_usul_kanunu.md", "0213 SAYILI VERGİ USUL KANUNU"),
        ("213 sayılı Maden Kanunu", "VERGİ USUL KANUNU"),
        ("0213_fake_prefix_vergi_usul_kanunu.md", "VERGİ USUL KANUNU"),
    ],
)
def test_conflicting_own_numbers_or_titles_never_become_retained_authority(
    name: str, root: str
) -> None:
    sources, _ = candidates()
    source = replace(next(source for source in sources if source.id == _VUK), name=name)
    row = replace(
        chunk(_CAPTURE["provision"][0], _VUK), heading_path=(root, "MADDE 315")
    )
    ledger = EvidenceLedger()
    ledger.add([evidence_for_chunk(source, row)], RunContext())
    assert known_statute_source_ids("213 sayılı Kanun", ledger) == []
    assert known_statute_source_ids("3213 sayılı Kanun", ledger) == []
    assert (
        canonical_named_authority_gap(
            "213 sayılı Vergi Usul Kanunu m.315 uygulanır [1].",
            ledger,
            strict_reference_boundaries=True,
            syntactic_reference_binding=True,
        )
        is not None
    )


def test_exact_generic_filename_keeps_supported_canonical_identity_and_rejects_conflicting_opening() -> (
    None
):
    sources, heads = candidates()
    source = replace(
        next(source for source in sources if source.id == _VUK),
        name="Kanunlar/0213_sayili_kanun.md",
    )
    provision = chunk(_CAPTURE["provision"][0], _VUK)
    ledger = EvidenceLedger()
    ledger.add([evidence_for_chunk(source, provision)], RunContext())
    assert known_statute_source_ids("213 sayılı Kanun", ledger) == [str(_VUK)]
    assert (
        canonical_named_authority_gap(
            "213 sayılı Vergi Usul Kanunu m.315 uygulanır [1].",
            ledger,
            strict_reference_boundaries=True,
            syntactic_reference_binding=True,
        )
        is None
    )
    contradictory_head = replace(
        heads[_VUK][0],
        text=heads[_VUK][0].text.replace(
            "Kanun Numarası : 213", "Kanun Numarası :3213"
        ),
    )
    ledger.add([evidence_for_chunk(source, contradictory_head)], RunContext())
    assert known_statute_source_ids("213 sayılı Kanun", ledger) == []
    assert (
        canonical_named_authority_gap(
            "213 sayılı Vergi Usul Kanunu m.315 uygulanır [1].",
            ledger,
            strict_reference_boundaries=True,
            syntactic_reference_binding=True,
        )
        is not None
    )


_INTAKES = json.loads(
    (
        Path(__file__).parent / "fixtures" / "numbered_intake_display_names.json"
    ).read_text()
)["laws"]


@pytest.mark.parametrize("capture", _INTAKES, ids=[row["number"] for row in _INTAKES])
def test_captured_intake_display_names_discover_and_pass_final_original_guard(
    monkeypatch: pytest.MonkeyPatch,
    capture: dict[str, Any],
) -> None:
    from hashlib import sha256

    identifier = UUID(capture["source_id"])
    source = CorpusSource(
        identifier, capture["display_name"], "retained-owned-original"
    )
    heads = [chunk(row, identifier) for row in capture["candidate_heads"]]
    provisions = [chunk(row, identifier) for row in capture["provision"]]
    session = catalogue(monkeypatch, [[source]])
    source_filter = filters()

    def read_heads(**arguments: Any) -> dict[UUID, list[CorpusChunk]]:
        assert arguments["filters"] is source_filter
        assert arguments["filters"].as_of_date == date(2026, 5, 18)
        arguments["check_active"]()
        return {source.id: heads}

    monkeypatch.setattr(asv3_corpus, "_read_identity_heads_batch", read_heads)
    broker = MemoryBroker(provisions)
    broker.item = source
    broker.filters = source_filter

    def lookup(
        name: str, context: RunContext, **kwargs: Any
    ) -> tuple[list[CorpusSource], bool]:
        return asv3_corpus.find_sources(
            session,
            user=cast(User, object()),
            filters=source_filter,
            query=name,
            limit=kwargs["limit"],
            allow_canonical_filename_fallback=True,
            check_active=context.check_active,
        )

    monkeypatch.setattr(broker, "sources", lookup)
    monkeypatch.setattr(broker, "related_sources_for_provision", lambda *_args: None)
    ledger = EvidenceLedger()
    context = RunContext(services={"asv3_exp_cropped": True, "evidence": ledger})
    tool = next(
        spec
        for spec in build_corpus_specs(broker, named_provision_reads=True)
        if spec.name == "read_named_provision"
    )
    outcome = tool.handler(
        {
            "source_name": f"{capture['number']} sayılı Kanun",
            "article": capture["article"],
        },
        context,
    )
    assert outcome.status == OutcomeStatus.FOUND
    assert {item.chunk_id for item in outcome.evidence} == {
        row["id"] for row in capture["provision"]
    }
    assert all(
        sha256(item.text.encode()).hexdigest()
        == next(
            row["text_sha256"]
            for row in capture["provision"]
            if row["id"] == item.chunk_id
        )
        for item in outcome.evidence
    )
    ledger.add(outcome.evidence, context)
    citation = next(row["citation"] for row in ledger.authority_metadata())
    before = ledger.export()
    assert known_statute_source_ids(f"{capture['number']} sayılı Kanun", ledger) == [
        str(identifier)
    ]
    assert (
        canonical_named_authority_gap(
            f"{capture['display_name']} m.{capture['article']} uygulanır [{citation}].",
            ledger,
            strict_reference_boundaries=True,
            syntactic_reference_binding=True,
        )
        is None
    )
    assert ledger.export() == before
    assert capture["number"] != "6001" or len(outcome.evidence) == 14


@pytest.mark.parametrize(
    "mutation",
    [
        "wrong_display_number",
        "wrong_display_title",
        "wrong_metadata_number",
        "wrong_root",
        "wrong_opening_number",
        "missing_full_opening",
    ],
)
def test_captured_fragmented_6001_title_requires_consistent_full_source_proof(
    mutation: str,
) -> None:
    capture = next(row for row in _INTAKES if row["number"] == "6001")
    identifier = UUID(capture["source_id"])
    source = CorpusSource(
        identifier, capture["display_name"], "retained-owned-original"
    )
    heads = [chunk(row, identifier) for row in capture["candidate_heads"]]
    if mutation == "wrong_display_number":
        source = replace(source, name=source.name.replace("6001", "213"))
    elif mutation == "wrong_display_title":
        source = replace(source, name="6001 sayılı Maden Kanunu")
    elif mutation == "wrong_metadata_number":
        heads = [
            replace(row, metadata={**row.metadata, "document_number": "5393"})
            for row in heads
        ]
    elif mutation == "wrong_root":
        heads = [
            replace(row, heading_path=("VERGİ USUL KANUNU", *row.heading_path[1:]))
            for row in heads
        ]
    elif mutation == "wrong_opening_number":
        heads[0] = replace(heads[0], text=heads[0].text.replace("6001", "5393"))
    elif mutation == "missing_full_opening":
        heads[0] = replace(heads[0], text="Hakkında Kanun [1]\nKanun Numarası :6001")
    assert not asv3_corpus._source_has_own_law_number(source, heads, "6001")
