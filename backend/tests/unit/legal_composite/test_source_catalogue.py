from collections.abc import Generator
from contextlib import contextmanager
from datetime import date
from threading import Barrier, Event, Lock
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import Session

from onyx.context.search.models import IndexFilters
from onyx.db import legal_composite_sources as sources
from onyx.db.asv3_corpus import CorpusChunk, CorpusScopeUnavailable, CorpusSource
from onyx.db.legal_composite_sources import SourceKind, classify_source
from onyx.db.models import User
from shared_configs.contextvars import CURRENT_TENANT_ID_CONTEXTVAR

READ_ORIGINAL_OPENING = sources._opening_texts
HEADINGS = {
    SourceKind.CONSTITUTION: "TÜRKİYE CUMHURİYETİ ANAYASASI",
    SourceKind.STATUTE: "GÜMRÜK KANUNU",
    SourceKind.TREATY: "Uluslararası Ticaret Sözleşmesi",
    SourceKind.PRESIDENTIAL_DECREE: "CUMHURBAŞKANLIĞI KARARNAMESİ",
    SourceKind.REGULATION: "GÜMRÜK YÖNETMELİĞİ",
    SourceKind.COMMUNIQUE: "GÜMRÜK GENEL TEBLİĞİ",
    SourceKind.CIRCULAR: "GENELGE 2025/1",
    SourceKind.JUDICIAL_DECISION: "ANAYASA MAHKEMESİ KARARI",
    SourceKind.EXECUTIVE_DECISION: "CUMHURBAŞKANI KARARI",
    SourceKind.PRIVATE_RULING: "ÖZELGE",
    SourceKind.OTHER: "YÖNERGE",
    SourceKind.UNKNOWN: "",
}


@pytest.fixture(autouse=True)
def original_opening_pages(monkeypatch: pytest.MonkeyPatch) -> None:
    def opening(_session: Session, **kwargs: object) -> tuple[str, ...]:
        source = cast(CorpusSource, kwargs["source"])
        kind = SourceKind.CIRCULAR if "genelge" in source.name else SourceKind.STATUTE
        return (HEADINGS[kind] + "\nMADDE 1- İlk hüküm.",)

    monkeypatch.setattr(sources, "_opening_texts", opening)


@pytest.mark.parametrize(
    "name,metadata,expected,uncertain",
    [
        ("anayasa.md", ("anayasa",), SourceKind.CONSTITUTION, False),
        ("kanun.md", ("kanun",), SourceKind.STATUTE, False),
        ("sozlesme.md", ("sozlesme",), SourceKind.TREATY, False),
        ("cumhurbaskanligi_kararnamesi.md", (), SourceKind.PRESIDENTIAL_DECREE, False),
        ("yonetmelik.md", ("yonetmelik",), SourceKind.REGULATION, False),
        ("teblig.md", ("tebliğ",), SourceKind.COMMUNIQUE, False),
        ("genelge.md", ("genelge",), SourceKind.CIRCULAR, False),
        (
            "anayasa_mahkemesi_karari.md",
            ("karar",),
            SourceKind.JUDICIAL_DECISION,
            False,
        ),
        ("danistay_karari.md", ("karar",), SourceKind.JUDICIAL_DECISION, False),
        ("cumhurbaskani_karari.md", ("karar",), SourceKind.EXECUTIVE_DECISION, False),
        ("ozelge.md", ("unknown",), SourceKind.PRIVATE_RULING, False),
        ("belge.md", ("ozelge",), SourceKind.PRIVATE_RULING, False),
        ("yonerge.md", ("yonerge",), SourceKind.OTHER, False),
        ("karar.md", ("karar",), SourceKind.UNKNOWN, True),
        ("Kanun, Tebliğ, Genelge/belge.md", ("unknown",), SourceKind.UNKNOWN, True),
    ],
)
def test_source_kinds_preserve_classification_uncertainty(
    name: str, metadata: tuple[str, ...], expected: SourceKind, uncertain: bool
) -> None:
    source = CorpusSource(uuid4(), name, "physical-file")
    record = classify_source(
        source, metadata, opening_texts=(HEADINGS[expected] + "\nMADDE 1- İlk hüküm.",)
    )
    assert record.kind is expected and record.uncertain is uncertain
    assert record.admits(expected)
    assert record.admits(SourceKind.UNKNOWN) is uncertain


@pytest.mark.parametrize(
    "metadata", [("kanun", "yonetmelik"), ("kanun", "karar"), ("unrecognized",)]
)
def test_conflicting_or_unrecognized_types_remain_unknown(
    metadata: tuple[str, ...],
) -> None:
    record = classify_source(CorpusSource(uuid4(), "kanun.md", "file"), metadata)
    assert record.kind is SourceKind.UNKNOWN and record.uncertain


def test_unknown_identity_participates_in_every_type_without_changing_its_kind() -> (
    None
):
    record = classify_source(CorpusSource(uuid4(), "yonetmelik.md", "file"), ("kanun",))
    assert record.kind is SourceKind.UNKNOWN
    assert record.admits(SourceKind.STATUTE) and record.admits(SourceKind.REGULATION)
    assert record.method == "original_identity_unverified"


def test_wrong_metadata_cannot_hide_actual_presidential_decree() -> None:
    source = CorpusSource(uuid4(), "yonetmelik.md", "file")
    record = classify_source(
        source,
        ("yonetmelik",),
        opening_texts=("CUMHURBAŞKANLIĞI KARARNAMESİ\nMADDE 1- Amaç.",),
    )
    assert record.kind is SourceKind.PRESIDENTIAL_DECREE
    assert record.admits(SourceKind.PRESIDENTIAL_DECREE) and record.admits(
        SourceKind.UNKNOWN
    )
    assert record.admits(SourceKind.REGULATION)
    assert record.opening_identity_sha256 and record.uncertain


def test_metadata_and_name_without_original_identity_stay_unknown() -> None:
    source = CorpusSource(uuid4(), "cumhurbaskanligi_kararnamesi.md", "file")
    record = classify_source(source, ("cumhurbaskanligi_kararnamesi",))
    assert record.kind is SourceKind.UNKNOWN
    assert record.admits(SourceKind.PRESIDENTIAL_DECREE)
    assert record.opening_identity_sha256 is None


@pytest.mark.parametrize(
    "text",
    [
        "MADDE 1- Uygulama.\nCUMHURBAŞKANLIĞI KARARNAMESİ",
        "Dayanak: Cumhurbaşkanlığı Kararnamesi\nMADDE 1- Uygulama.",
        "Bu metin Cumhurbaşkanlığı Kararnamesi uyarınca uygulanır.",
        "Hukuki dayanak Cumhurbaşkanlığı Kararnamesi.",
        "[MADDE 1] Uygulama.\nCUMHURBAŞKANLIĞI KARARNAMESİ",
    ],
)
def test_body_references_do_not_become_document_identity(text: str) -> None:
    record = classify_source(
        CorpusSource(uuid4(), "kararname.md", "file"),
        ("cumhurbaskanligi_kararnamesi",),
        opening_texts=(text,),
    )
    assert record.kind is SourceKind.UNKNOWN and record.opening_identity_sha256 is None


def test_body_reference_cannot_override_actual_regulation_header() -> None:
    record = classify_source(
        CorpusSource(uuid4(), "file.md", "file"),
        ("yonetmelik",),
        opening_texts=(
            "GÜMRÜK YÖNETMELİĞİ\nMADDE 1- Uygulama.\nCUMHURBAŞKANLIĞI KARARNAMESİ",
        ),
    )
    assert record.kind is SourceKind.REGULATION


def test_original_identity_reads_use_authorized_published_page_and_bounded_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = CorpusSource(uuid4(), "wrong-yonetmelik.md", "file")
    user = cast(User, SimpleNamespace(id=uuid4()))
    filters = IndexFilters(
        access_control_list=["user"],
        asv3_document_set_id=15,
        forced_document_set=["PC Külliyatı"],
        as_of_date=date(2020, 1, 1),
    )
    monkeypatch.setattr(
        sources,
        "resolve_source_query_index",
        lambda _session, identifier: (
            None if identifier == source.id else pytest.fail("Wrong source")
        ),
    )
    captured: list[dict[str, object]] = []

    def read(
        _session: Session, **kwargs: object
    ) -> tuple[CorpusSource, list[CorpusChunk], bool]:
        captured.append(kwargs)
        assert kwargs["user"] is user and kwargs["filters"] is filters
        assert kwargs["source_id"] == source.id and kwargs["start"] == 0
        assert kwargs["limit"] == 3 and kwargs["query_indexes"] == {}
        chunk = CorpusChunk(
            "chunk",
            source.id,
            "CUMHURBAŞKANLIĞI KARARNAMESİ\nMADDE 1- " + "x" * 10_000,
            0,
            0,
            (),
            {"document_type": "yonetmelik"},
            None,
            None,
            "active",
        )
        return source, [chunk], False

    monkeypatch.setattr(sources, "read_source_chunks", read)
    openings = READ_ORIGINAL_OPENING(
        cast(Session, MagicMock()),
        user=user,
        filters=filters,
        source=source,
        check_active=lambda: None,
    )
    assert len(captured) == 1 and sum(map(len, openings)) == 4_096
    record = classify_source(source, ("yonetmelik",), opening_texts=openings)
    assert record.kind is SourceKind.PRESIDENTIAL_DECREE
    assert record.uncertain and record.opening_identity_sha256


def test_inventory_pages_only_authorized_sources_and_selects_metadata_without_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user = cast(User, SimpleNamespace(id=uuid4()))
    first = CorpusSource(uuid4(), "kanun.md", "first")
    second = CorpusSource(uuid4(), "genelge.md", "second")
    denied = uuid4()
    filters = IndexFilters(
        access_control_list=["user"],
        asv3_document_set_id=15,
        forced_document_set=["PC Külliyatı"],
    )
    page_calls: list[int] = []

    def find(_session: Session, **kwargs: object) -> tuple[list[CorpusSource], bool]:
        assert kwargs["filters"] is filters and kwargs["user"] is user
        page_calls.append(cast(int, kwargs["offset"]))
        return ([first], True) if kwargs["offset"] == 0 else ([second], False)

    monkeypatch.setattr(sources, "find_source_inventory_page", find)
    session = MagicMock()
    session.execute.return_value.all.side_effect = [
        [(first.id, "kanun")],
        [(second.id, "genelge")],
    ]
    catalogue = sources.load_source_lane_catalogue(
        cast(Session, session), user=user, filters=filters, check_active=lambda: None
    )
    assert page_calls == [0, 1000]
    assert catalogue.complete and catalogue.source_ids(SourceKind.STATUTE) == (
        first.id,
    )
    assert catalogue.source_ids(SourceKind.CIRCULAR) == (second.id,)
    assert denied not in {row.source_id for row in catalogue.records}
    for call in session.execute.call_args_list:
        query = call.args[0].compile(dialect=postgresql.dialect())
        assert "regulatory_chunk.text" not in str(query)
        assert denied not in [
            item
            for value in query.params.values()
            if isinstance(value, list)
            for item in value
        ]
    session.add.assert_not_called()
    session.commit.assert_not_called()


def test_partial_inventory_reports_unsearched_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = CorpusSource(uuid4(), "kanun.md", "file")
    monkeypatch.setattr(
        sources,
        "find_source_inventory_page",
        lambda *_args, **_kwargs: ([record], True),
    )
    session = MagicMock()
    session.execute.return_value.all.return_value = [(record.id, "kanun")]
    catalogue = sources.load_source_lane_catalogue(
        cast(Session, session),
        user=cast(User, SimpleNamespace(id=uuid4())),
        filters=IndexFilters(access_control_list=[]),
        check_active=lambda: None,
        max_sources=1,
    )
    assert not catalogue.complete and catalogue.limitations
    assert len(catalogue.records) == 1


def test_parallel_opening_reads_overlap_with_owned_sessions_and_tenant_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_page = [
        CorpusSource(uuid4(), f"source-{index}.md", "file") for index in range(8)
    ]
    positions = {source.id: index for index, source in enumerate(source_page)}
    user = cast(User, SimpleNamespace(id=uuid4()))
    filters = IndexFilters(
        access_control_list=["authorized-user"],
        asv3_document_set_id=15,
        forced_document_set=["PC Külliyatı"],
        as_of_date=date(2020, 1, 1),
    )
    inventory = MagicMock(spec=Session)
    inventory.execute.return_value.all.return_value = [
        (source.id, "kanun") for source in source_page
    ]
    monkeypatch.setattr(
        sources,
        "find_source_inventory_page",
        lambda *_args, **_kwargs: (source_page, False),
    )
    sessions: list[MagicMock] = []
    closed: list[MagicMock] = []
    completion_order: list[int] = []
    lock, barrier, fourth_complete = Lock(), Barrier(4), Event()
    active, peak = [0], [0]

    @contextmanager
    def opening_session() -> Generator[Session, None, None]:
        assert CURRENT_TENANT_ID_CONTEXTVAR.get() == "opening-test-tenant"
        own_session = MagicMock(spec=Session)
        with lock:
            sessions.append(own_session)
        try:
            yield cast(Session, own_session)
        finally:
            with lock:
                closed.append(own_session)

    def read_batch(
        session: Session,
        source_ids: tuple[UUID, ...],
        captured_filters: IndexFilters,
        check_active: object,
    ) -> dict[UUID, tuple[str, ...] | None]:
        assert session is not inventory
        assert CURRENT_TENANT_ID_CONTEXTVAR.get() == "opening-test-tenant"
        assert captured_filters is filters and callable(check_active)
        assert len(source_ids) == 2
        index = positions[source_ids[0]]
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        try:
            barrier.wait(timeout=3)
            if index == 0:
                assert fourth_complete.wait(timeout=3)
            with lock:
                completion_order.append(index)
            if index == 6:
                fourth_complete.set()
            return {
                source_id: None
                if positions[source_id] == 6
                else ("ÖRNEK KANUNU\nMADDE 1- İlk hüküm.",)
                for source_id in source_ids
            }
        finally:
            with lock:
                active[0] -= 1

    def check_active() -> None:
        assert CURRENT_TENANT_ID_CONTEXTVAR.get() == "opening-test-tenant"

    monkeypatch.setattr(sources, "_opening_texts", READ_ORIGINAL_OPENING)
    monkeypatch.setattr(sources, "get_session_with_current_tenant", opening_session)
    monkeypatch.setattr(sources, "_opening_batch", read_batch)
    token = CURRENT_TENANT_ID_CONTEXTVAR.set("opening-test-tenant")
    try:
        catalogue = sources.load_source_lane_catalogue(
            cast(Session, inventory),
            user=user,
            filters=filters,
            check_active=check_active,
            opening_workers=4,
        )
    finally:
        CURRENT_TENANT_ID_CONTEXTVAR.reset(token)
    assert peak == [4] and active == [0]
    assert completion_order.index(6) < completion_order.index(0)
    assert [row.source_id for row in catalogue.records] == [
        source.id for source in source_page
    ]
    assert catalogue.records[6].kind is SourceKind.UNKNOWN
    assert not catalogue.complete and catalogue.limitations
    assert len(sessions) == len(closed) == len({id(item) for item in sessions}) == 4
    assert {id(item) for item in closed} == {id(item) for item in sessions}
    for session in sessions:
        session.add.assert_not_called()
        session.commit.assert_not_called()


@pytest.mark.parametrize("workers", [0, 5, True])
def test_invalid_opening_concurrency_fails_before_inventory(workers: int) -> None:
    session = MagicMock(spec=Session)
    with pytest.raises(ValueError, match="Opening workers"):
        sources.load_source_lane_catalogue(
            cast(Session, session),
            user=cast(User, SimpleNamespace(id=uuid4())),
            filters=IndexFilters(access_control_list=[]),
            check_active=lambda: None,
            opening_workers=workers,
        )
    session.execute.assert_not_called()


def test_metadata_type_query_uses_the_original_temporal_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = CorpusSource(uuid4(), "kanun.md", "file")
    monkeypatch.setattr(
        sources,
        "find_source_inventory_page",
        lambda *_args, **_kwargs: ([record], False),
    )
    session = MagicMock()
    session.execute.return_value.all.return_value = [(record.id, "kanun")]
    as_of = date(2020, 1, 1)
    sources.load_source_lane_catalogue(
        cast(Session, session),
        user=cast(User, SimpleNamespace(id=uuid4())),
        filters=IndexFilters(access_control_list=[], as_of_date=as_of),
        check_active=lambda: None,
    )
    query = session.execute.call_args.args[0].compile(dialect=postgresql.dialect())
    assert as_of in query.params.values()
    assert "validity_start_date" in str(query) and "validity_end_date" in str(query)


def test_classification_drift_fails_even_when_source_stays_authorized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = CorpusSource(uuid4(), "belge.md", "file")
    record = classify_source(
        source,
        ("kanun",),
        opening_texts=(HEADINGS[SourceKind.STATUTE] + "\nMADDE 1- Amaç.",),
    )
    monkeypatch.setattr(sources, "require_source", lambda *_args, **_kwargs: source)
    session = MagicMock()
    session.execute.return_value.all.return_value = [(source.id, "yonetmelik")]
    monkeypatch.setattr(
        sources,
        "_opening_texts",
        lambda *_args, **_kwargs: (
            HEADINGS[SourceKind.REGULATION] + "\nMADDE 1- Amaç.",
        ),
    )
    with pytest.raises(CorpusScopeUnavailable, match="classification changed"):
        sources.revalidate_source_classification(
            cast(Session, session),
            user=cast(User, object()),
            filters=IndexFilters(access_control_list=[]),
            recorded=record,
            check_active=lambda: None,
        )


def test_metadata_changes_alone_cannot_exclude_unchanged_original_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = CorpusSource(uuid4(), "belge.md", "file")
    original = "GÜMRÜK KANUNU\nMADDE 1- Amaç."
    record = classify_source(source, ("kanun",), opening_texts=(original,))
    monkeypatch.setattr(sources, "require_source", lambda *_args, **_kwargs: source)
    monkeypatch.setattr(
        sources, "_opening_texts", lambda *_args, **_kwargs: (original,)
    )
    session = MagicMock()
    session.execute.return_value.all.return_value = [(source.id, "yonetmelik")]
    assert (
        sources.revalidate_source_classification(
            cast(Session, session),
            user=cast(User, object()),
            filters=IndexFilters(access_control_list=[]),
            recorded=record,
            check_active=lambda: None,
        )
        == source
    )
