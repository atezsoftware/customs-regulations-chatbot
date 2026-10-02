"""Behavioral recovery, scope and execution tests for the research source tools."""

from collections.abc import Generator
from contextlib import nullcontext
from contextvars import ContextVar
from datetime import date
from decimal import Decimal
from io import BytesIO
from threading import RLock
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from pydantic import JsonValue

from onyx.asv3.corpus_tools import CorpusBroker, build_corpus_specs
from onyx.asv3.models import (
    CapabilityCall,
    OutcomeStatus,
    RunContext,
    SharedBudget,
    ToolOutcome,
    ToolSpec,
)
from onyx.asv3.registry import CapabilityRegistry
from onyx.asv3.sandbox import build_sandbox_specs, calculate, compose, run_code
from onyx.asv3.source_tools import build_source_specs, original_evidence, selected_pdf
from onyx.asv3.supplemental_tools import build_supplemental_specs
from onyx.context.search.models import IndexFilters
from onyx.db.asv3_corpus import (
    CorpusChunk,
    CorpusScopeUnavailable,
    CorpusSource,
    find_sources,
    original_source_record,
)
from onyx.db.models import User
from onyx.server.asv3_citations import native_item_from_checkpoint
from onyx.tools.tool_implementations.python.code_interpreter_client import (
    BashExecResponse,
    CreateSessionResponse,
    ExecuteResponse,
)


class MemoryBroker(CorpusBroker):
    def __init__(self, chunks: list[CorpusChunk], *, partial: bool = False) -> None:
        self.item = CorpusSource(
            chunks[0].source_id, "Gümrük Yönetmeliği", "physical-original"
        )
        self.items = chunks
        self.partial = partial
        self.filters = IndexFilters(access_control_list=[])
        self.search_adapter = None
        self.file_store = None
        self.vision_llm = None

    def chunk_siblings(
        self, seed: CorpusChunk, context: RunContext
    ) -> tuple[CorpusSource, tuple[str, ...]]:
        parent = seed.heading_path[:-1]
        rows = sorted(
            (
                row
                for row in self.items
                if (
                    bool(row.heading_path) and row.heading_path[:-1] == parent
                    if seed.heading_path
                    else row.id == seed.id
                )
            ),
            key=lambda row: (row.position, row.projection_ordinal or 0, row.id),
        )
        return self.source(str(seed.source_id), context), tuple(row.id for row in rows)

    def sibling_page(
        self, source: CorpusSource, ids: tuple[str, ...], context: RunContext
    ) -> Generator[CorpusChunk, None, None]:
        self.source(str(source.id), context)
        for row in sorted(
            self.items,
            key=lambda row: (row.position, row.projection_ordinal or 0, row.id),
        ):
            if row.id in ids:
                yield row

    def provision_start(
        self, source_id: str, article: str, qualifier: str | None, context: RunContext
    ) -> int | None:
        del article, qualifier
        self.source(source_id, context)
        return None

    def source(self, source_id: str, context: RunContext) -> CorpusSource:
        context.check_active()
        if source_id != str(self.item.id):
            raise PermissionError("denied")
        return self.item

    def scan(
        self,
        source_id: str,
        context: RunContext,
        *,
        as_of: date | None = None,
        historical_inventory: bool = False,
    ) -> tuple[CorpusSource, list[CorpusChunk], bool]:
        del as_of, historical_inventory
        return self.source(source_id, context), self.items, self.partial

    def chunk(
        self, source_id: str, chunk_id: str, context: RunContext
    ) -> tuple[CorpusSource, CorpusChunk | None]:
        return self.source(source_id, context), next(
            (chunk for chunk in self.items if chunk.id == chunk_id), None
        )

    def page(
        self,
        source_id: str,
        context: RunContext,
        *,
        start: int = 0,
        limit: int = 50,
        as_of: date | None = None,
        historical_inventory: bool = False,
    ) -> tuple[CorpusSource, list[CorpusChunk], bool]:
        del as_of, historical_inventory
        source = self.source(source_id, context)
        rows = [row for row in self.items if row.position >= start]
        return source, rows[:limit], self.partial or len(rows) > limit


def test_all_owned_tool_fields_have_provider_compatible_explicit_types(
    broker: MemoryBroker,
) -> None:
    from litellm.llms.vertex_ai.common_utils import _build_vertex_schema

    from onyx.asv3.evidence import EvidenceLedger
    from onyx.asv3.registry import build_core_specs
    from onyx.asv3.workers import WorkerPool

    def inspect(node: JsonValue) -> None:
        assert isinstance(node, dict)
        assert "type" in node or any(
            key in node for key in ("anyOf", "oneOf", "allOf", "$ref")
        )
        if "enum" in node:
            assert node["type"] == "string"
        properties = node.get("properties", {})
        assert isinstance(properties, dict)
        for child in properties.values():
            inspect(child)
        if "items" in node:
            inspect(node["items"])

    specs = (
        build_corpus_specs(broker)
        + build_source_specs(broker)
        + build_sandbox_specs(broker)
        + build_supplemental_specs()
    )
    registry = CapabilityRegistry(specs)
    pool = WorkerPool(
        RunContext(),
        lambda _task, _context, _updates: ToolOutcome(
            status=OutcomeStatus.FOUND, summary="done"
        ),
    )
    try:
        specs += pool.tool_specs() + build_core_specs(
            registry, EvidenceLedger(), lambda: {}
        )
        for spec in specs:
            inspect(spec.parameters)
            function = spec.definition()["function"]
            assert isinstance(function, dict)
            parameters = function["parameters"]
            assert isinstance(parameters, dict)
            normalized = _build_vertex_schema(parameters)
            inspect(normalized)
    finally:
        pool.close()


def test_native_citation_uses_owned_saved_evidence_without_fake_chunk(
    broker: MemoryBroker,
) -> None:
    context = RunContext()
    extracted = original_evidence(
        str(broker.item.id), "Exact native row", "a" * 64, {"page": 2}
    )
    item = broker.attach_native_citation(extracted, 3, 10, context)
    assert item.search_doc is not None
    assert item.search_doc.document_id == str(broker.item.id)
    assert item.search_doc.chunk_ind == -3
    assert "regulatory_chunk_id" not in item.search_doc.metadata
    assert (
        item.search_doc.metadata["asv3_citation_preview_url"]
        == "/api/asv3/citation/10/3"
    )
    checkpoint: dict[str, JsonValue] = {
        "scope": broker.filters.model_dump(mode="json"),
        "evidence": {
            "version": 1,
            "records": [{"citation": 3, "item": item.model_dump(mode="json")}],
        },
    }
    loaded, _ = native_item_from_checkpoint(checkpoint, 3)
    assert loaded.text == extracted.text and loaded.text_hash == extracted.text_hash
    with pytest.raises(ValueError, match="not found"):
        native_item_from_checkpoint(checkpoint, 4)
    records = cast(dict[str, JsonValue], checkpoint["evidence"])["records"]
    assert isinstance(records, list) and isinstance(records[0], dict)
    stored = records[0]["item"]
    assert isinstance(stored, dict)
    stored["text"] = "Changed native row"
    with pytest.raises(ValueError, match="hash"):
        native_item_from_checkpoint(checkpoint, 3)


def test_native_revalidation_rejects_original_hash_change(
    broker: MemoryBroker, monkeypatch: pytest.MonkeyPatch
) -> None:
    from onyx.asv3 import source_tools

    item = original_evidence(str(broker.item.id), "row", "a" * 64, {"page": 1})
    monkeypatch.setattr(
        source_tools,
        "read_verified_original",
        lambda *_args: (b"changed", "text/plain", "source"),
    )
    with pytest.raises(CorpusScopeUnavailable, match="changed"):
        broker.revalidate_evidence([item], RunContext())


@pytest.fixture
def broker() -> MemoryBroker:
    identifier = uuid4()
    rows = [
        CorpusChunk(
            "ordinary",
            identifier,
            "Ordinary article",
            0,
            5,
            ("MADDE 5",),
            {"article_no": "5"},
            None,
            None,
            "active",
        ),
        CorpusChunk(
            "temporary-head",
            identifier,
            "Prerequisites must BOTH hold.",
            1,
            6,
            ("GEÇİCİ MADDE 5",),
            {},
            None,
            None,
            "active",
        ),
        CorpusChunk(
            "temporary-tail",
            identifier,
            "Continuation contains the exception.",
            2,
            7,
            (),
            {},
            None,
            None,
            "active",
        ),
        CorpusChunk(
            "six",
            identifier,
            "Next article",
            3,
            8,
            ("MADDE 6",),
            {},
            None,
            None,
            "active",
        ),
    ]
    return MemoryBroker(rows)


def test_provision_keeps_qualifier_continuation_and_exact_citation(
    broker: MemoryBroker,
) -> None:
    registry = CapabilityRegistry(build_corpus_specs(broker))
    result = registry.dispatch(
        CapabilityCall(
            name="read_provision",
            arguments={"source_id": str(broker.item.id), "article": "GEÇİCİ 5"},
        ),
        RunContext(),
    )
    assert result.status == OutcomeStatus.FOUND
    assert [item.chunk_id for item in result.evidence] == [
        "temporary-head",
        "temporary-tail",
    ]
    assert [
        item.search_doc.chunk_ind for item in result.evidence if item.search_doc
    ] == [6, 7]
    assert all(item.source_id == str(broker.item.id) for item in result.evidence)


def test_provision_keeps_cross_referencing_clause_and_stops_at_boundary() -> None:
    identifier = uuid4()
    rows = [
        CorpusChunk(
            "intro",
            identifier,
            "Başvuru şartları",
            0,
            0,
            ("MADDE 27", "1. Başvuru şartları"),
            {"article_no": "27", "paragraph_no": "1"},
            None,
            None,
            "active",
        ),
        CorpusChunk(
            "clause",
            identifier,
            "ç) 9923 sayılı Kanunun 99 uncu maddesine göre izin gerekir.",
            1,
            1,
            (
                "MADDE 27",
                "1. Başvuru şartları",
                "ç) 9923 sayılı Kanunun 99 uncu maddesine göre izin gerekir.",
            ),
            {"article_no": "27", "clause_label": "ç"},
            None,
            None,
            "active",
        ),
        CorpusChunk(
            "next",
            identifier,
            "Sonraki madde",
            2,
            2,
            ("MADDE 28",),
            {"article_no": "28"},
            None,
            None,
            "active",
        ),
        *[
            CorpusChunk(
                f"later-{n}",
                identifier,
                "İlgisiz hüküm",
                n,
                n,
                ("MADDE 99",),
                {"article_no": "99"},
                None,
                None,
                "active",
            )
            for n in range(3, 200)
        ],
    ]
    broker = MemoryBroker(rows)
    from onyx.asv3.corpus_tools import article_identity

    assert article_identity(rows[1]) == ("27", None)
    result = CapabilityRegistry(build_corpus_specs(broker)).dispatch(
        CapabilityCall(
            name="read_provision",
            arguments={
                "source_id": str(identifier),
                "article": "27",
                "paragraph": "1",
                "clause": "ç",
            },
        ),
        RunContext(),
    )
    assert result.status == OutcomeStatus.FOUND
    assert result.data["subunit_verified"] is True
    assert result.data["scan_truncated"] is False
    assert result.data["next_position"] == 2
    assert [item.chunk_id for item in result.evidence] == ["intro", "clause"]


def test_partial_scan_is_never_proof_of_missing_article(broker: MemoryBroker) -> None:
    broker.partial = True
    result = CapabilityRegistry(build_corpus_specs(broker)).dispatch(
        CapabilityCall(
            name="read_provision",
            arguments={"source_id": str(broker.item.id), "article": "999"},
        ),
        RunContext(),
    )
    assert result.status == OutcomeStatus.PARTIAL
    assert result.data["absence_proven"] is False


def test_exact_locator_retains_chunk_identity_when_semantic_positions_repeat(
    broker: MemoryBroker, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dataclasses import replace

    from onyx.asv3 import corpus_tools

    broker.items[1] = replace(broker.items[1], position=0)
    broker.query_indexes = {}
    broker.user = User()
    broker._index_lock = RLock()
    monkeypatch.setattr(
        corpus_tools,
        "get_session_with_current_tenant",
        lambda: nullcontext(MagicMock()),
    )
    monkeypatch.setattr(corpus_tools, "resolve_source_query_index", lambda *_args: None)
    monkeypatch.setattr(
        corpus_tools,
        "iter_source_chunks_by_ids",
        lambda *_args, **kwargs: iter(
            row for row in broker.items if row.id in kwargs["chunk_ids"]
        ),
    )
    _, chunk = CorpusBroker.chunk(
        broker, str(broker.item.id), "temporary-head", RunContext()
    )
    assert chunk is not None and chunk.id == "temporary-head"


def test_source_tools_cannot_escape_captured_owner_scope(broker: MemoryBroker) -> None:
    result = CapabilityRegistry(build_corpus_specs(broker)).dispatch(
        CapabilityCall(name="read_source_range", arguments={"source_id": str(uuid4())}),
        RunContext(),
    )
    assert result.status == OutcomeStatus.DENIED
    assert not result.evidence


def test_literal_match_offsets_are_original_unicode_offsets(
    broker: MemoryBroker,
) -> None:
    broker.items[0] = CorpusChunk(
        "unicode",
        broker.item.id,
        "İstanbul istanbul",
        0,
        5,
        ("MADDE 5",),
        {},
        None,
        None,
        "active",
    )
    result = CapabilityRegistry(build_corpus_specs(broker)).dispatch(
        CapabilityCall(
            name="search_source_text",
            arguments={"source_id": str(broker.item.id), "pattern": "istanbul"},
        ),
        RunContext(),
    )
    assert result.status == OutcomeStatus.FOUND
    matches = cast(list[dict[str, JsonValue]], result.data["matches"])
    assert matches[1]["start"] == 9
    assert matches[1]["text"] == "istanbul"


def test_regex_path_really_searches_and_reports_spans(broker: MemoryBroker) -> None:
    result = CapabilityRegistry(build_corpus_specs(broker)).dispatch(
        CapabilityCall(
            name="search_source_text",
            arguments={
                "source_id": str(broker.item.id),
                "pattern": "[Ee]xception",
                "mode": "regex",
            },
        ),
        RunContext(),
    )
    assert result.status == OutcomeStatus.FOUND
    assert result.evidence[0].chunk_id == "temporary-tail"


def test_corpus_sources_acl_is_applied_after_metadata_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from onyx.db import asv3_corpus

    allowed, forbidden = uuid4(), uuid4()
    session = MagicMock()
    session.execute.return_value.all.return_value = [
        SimpleNamespace(id=allowed, name="allowed", file_id="a"),
        SimpleNamespace(id=forbidden, name="secret", file_id="b"),
    ]
    monkeypatch.setattr(
        asv3_corpus,
        "get_document_set_by_id_for_user",
        lambda *_args, **_kwargs: SimpleNamespace(
            id=73, name=asv3_corpus.PC_CORPUS_NAME
        ),
    )
    monkeypatch.setattr(
        asv3_corpus,
        "filter_document_set_names_by_user_access",
        lambda _session, names, _user: set(names),
    )
    monkeypatch.setattr(
        asv3_corpus,
        "get_access_for_user_files",
        lambda *_args: {
            str(allowed): SimpleNamespace(to_acl=lambda: {"user:x"}),
            str(forbidden): SimpleNamespace(to_acl=lambda: {"user:other"}),
        },
    )
    monkeypatch.setattr(asv3_corpus, "get_acl_for_user", lambda *_args: {"user:x"})
    monkeypatch.setattr(asv3_corpus, "observe_publication_read", lambda: object())
    monkeypatch.setattr(
        asv3_corpus,
        "filter_publication_read",
        lambda _observation, items, _getter: items,
    )
    result, _ = find_sources(
        session,
        user=cast(User, SimpleNamespace()),
        filters=IndexFilters(
            access_control_list=[],
            forced_document_set=[asv3_corpus.PC_CORPUS_NAME],
            asv3_document_set_id=73,
        ),
    )
    assert [row.id for row in result] == [allowed]


def test_unsupported_fences_fail_closed_before_database_read() -> None:
    session = MagicMock()
    with pytest.raises(CorpusScopeUnavailable):
        find_sources(
            session,
            user=cast(User, SimpleNamespace()),
            filters=IndexFilters(access_control_list=[], hierarchy_node_ids=[42]),
        )
    session.execute.assert_not_called()


def test_historical_original_is_not_a_version_recovery_bypass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from onyx.db import asv3_corpus

    source = CorpusSource(uuid4(), "original", "physical")
    monkeypatch.setattr(asv3_corpus, "require_source", lambda *_args, **_kwargs: source)
    session = MagicMock()
    with pytest.raises(CorpusScopeUnavailable, match="dated/versioned"):
        original_source_record(
            session,
            user=cast(User, SimpleNamespace()),
            filters=IndexFilters(access_control_list=[], as_of_date=date(2020, 1, 1)),
            source_id=source.id,
        )
    session.get.assert_not_called()


def test_decimal_partial_return_and_exact_following_month_deadline() -> None:
    result = calculate(
        {"operation": "percent", "values": ["500000", "20"]}, RunContext()
    )
    assert Decimal(str(result.data["result"])) == Decimal("100000")
    result = calculate(
        {"operation": "following_month_day", "date": "2026-05-18", "day": 26},
        RunContext(),
    )
    assert result.data["result"] == "2026-06-26"


def test_business_days_requires_explicit_calendar() -> None:
    result = calculate(
        {"operation": "business_days", "date": "2026-05-18", "days": 3}, RunContext()
    )
    assert result.status == OutcomeStatus.INVALID
    result = calculate(
        {
            "operation": "business_days",
            "date": "2026-05-22",
            "days": 1,
            "calendar_name": "explicit supplied holidays",
            "holidays": ["2026-05-25"],
        },
        RunContext(),
    )
    assert result.data["result"] == "2026-05-26"


def test_composition_validates_cycles_before_any_tool_runs() -> None:
    calls = []
    registry = CapabilityRegistry(
        [
            ToolSpec(
                name="read",
                description="read",
                parameters={"type": "object"},
                handler=lambda args, _ctx: (
                    calls.append(args)
                    or ToolOutcome(status=OutcomeStatus.FOUND, summary="done")
                ),
            )
        ]
    )
    with pytest.raises(ValueError, match="cycle"):
        compose(
            {
                "steps": [
                    {"id": "a", "tool": "read", "depends_on": ["b"]},
                    {"id": "b", "tool": "read", "depends_on": ["a"]},
                ]
            },
            RunContext(services={"registry": registry}),
        )
    assert not calls


def test_composition_runs_real_dependencies_and_propagates_contextvars(
    broker: MemoryBroker,
) -> None:
    tenant: ContextVar[str] = ContextVar("test_tenant", default="wrong")
    token = tenant.set("captured")
    try:

        def read(args: dict[str, JsonValue], _context: RunContext) -> ToolOutcome:
            assert tenant.get() == "captured"
            return ToolOutcome(
                status=OutcomeStatus.FOUND,
                summary="ok",
                data={"value": args.get("value", "4")},
            )

        registry = CapabilityRegistry(
            [
                ToolSpec(
                    name="read",
                    description="read",
                    parameters={"type": "object"},
                    handler=read,
                ),
                *build_sandbox_specs(broker),
            ]
        )
        context = RunContext(services={"registry": registry})
        result = registry.dispatch(
            CapabilityCall(
                name="compose_tool_calls",
                arguments={
                    "steps": [
                        {"id": "a", "tool": "read"},
                        {
                            "id": "b",
                            "tool": "read",
                            "arguments": {"value": {"$ref": "a.data.value"}},
                            "depends_on": ["a"],
                        },
                    ]
                },
            ),
            context,
        )
        assert result.status == OutcomeStatus.FOUND
        assert context.budget.snapshot()["tools"] == 3
        assert cast(dict, cast(dict, result.data["steps"])["b"])["data"]["value"] == "4"
    finally:
        tenant.reset(token)


def test_code_uses_isolated_client_and_authorized_manifest_and_cleans_inputs(
    broker: MemoryBroker,
) -> None:
    client = MagicMock()
    client.__enter__.return_value = client
    client.health.return_value.healthy = True
    client.upload_file.return_value = "input-upload"
    client.execute.return_value = ExecuteResponse(
        stdout="100000",
        stderr="",
        exit_code=0,
        timed_out=False,
        duration_ms=1,
        files=[],
    )
    context = RunContext(services={"code_interpreter_factory": lambda: client})
    result = run_code(
        broker, {"code": "print(100000)", "source_ids": [str(broker.item.id)]}, context
    )
    assert result.status == OutcomeStatus.FOUND
    manifest = client.upload_file.call_args.args[0]
    assert b"temporary-tail" in manifest
    assert b"password" not in manifest
    client.execute.assert_called_once()
    client.delete_file.assert_called_once_with("input-upload")


def test_bash_runs_only_in_disposable_service_session(broker: MemoryBroker) -> None:
    client = MagicMock()
    client.__enter__.return_value = client
    client.health.return_value.healthy = True
    client.supports.return_value = True
    client.create_session.return_value = CreateSessionResponse(
        session_id="disposable", expires_at=1
    )
    client.execute_bash_in_session.return_value = BashExecResponse(
        stdout="row", stderr="", exit_code=0, timed_out=False, duration_ms=1
    )
    context = RunContext(services={"code_interpreter_factory": lambda: client})
    result = run_code(broker, {"code": "cat source.json", "language": "bash"}, context)
    assert result.status == OutcomeStatus.FOUND
    assert result.data["language"] == "bash"
    client.execute.assert_not_called()
    client.create_session.assert_called_once_with(ttl_seconds=60, files=[])
    client.execute_bash_in_session.assert_called_once()
    client.delete_session.assert_called_once_with("disposable")
    client.supports.return_value = False
    unavailable = run_code(broker, {"code": "pwd", "language": "bash"}, context)
    assert unavailable.status == OutcomeStatus.UNAVAILABLE


def test_native_table_extraction_preserves_original_row_locations(
    broker: MemoryBroker, monkeypatch: pytest.MonkeyPatch
) -> None:
    from openpyxl import Workbook
    from openpyxl.worksheet.worksheet import Worksheet

    from onyx.asv3 import source_tools

    workbook = Workbook()
    worksheet = workbook.active
    assert isinstance(worksheet, Worksheet)
    worksheet.append(["country", "rate"])
    worksheet.append(["China", "10"])
    buffer = BytesIO()
    workbook.save(buffer)
    workbook.close()
    monkeypatch.setattr(
        source_tools,
        "read_verified_original",
        lambda *_args: (
            buffer.getvalue(),
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "rates.xlsx",
        ),
    )
    result = CapabilityRegistry(build_source_specs(broker)).dispatch(
        CapabilityCall(
            name="extract_table", arguments={"source_id": str(broker.item.id)}
        ),
        RunContext(),
    )
    assert result.status == OutcomeStatus.FOUND
    assert any(
        isinstance(item.metadata.get("locator"), dict)
        and cast(dict[str, JsonValue], item.metadata["locator"])["row"] == 2
        and "China" in item.text
        for item in result.evidence
    )
    assert all(item.metadata["source_sha256"] for item in result.evidence)


def test_selected_pdf_rejects_outside_page() -> None:
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    buffer = BytesIO()
    writer.write(buffer)
    with pytest.raises(ValueError, match="outside"):
        selected_pdf(buffer.getvalue(), (2,))


def test_source_vision_uses_shared_model_capacity_and_research_reserve(
    broker: MemoryBroker, monkeypatch: pytest.MonkeyPatch
) -> None:
    from onyx.asv3 import source_tools
    from onyx.llm.interfaces import LLM
    from onyx.regulatory.amendments.annexes.models import AnnexExtraction

    broker.vision_llm = cast(LLM, MagicMock())
    context = RunContext(budget=SharedBudget(max_inflight_models=1))
    expected = AnnexExtraction(source_sha256="a" * 64, mime_type="application/pdf")

    def extract(*_args: object, **kwargs: object) -> AnnexExtraction:
        assert not context.budget.model_slots.acquire(blocking=False)
        assert kwargs["vision_llm"] is broker.vision_llm
        assert kwargs["vision_deadline"] == context.research_deadline
        return expected

    monkeypatch.setattr(
        source_tools, "llm_generation_span", lambda *_args: nullcontext()
    )
    monkeypatch.setattr(source_tools, "extract_annex_structure", extract)
    result = source_tools.extract_source_vision(
        broker, b"verified", "application/pdf", context
    )
    assert result is expected
    assert context.budget.snapshot()["decisions"] == 1
    assert context.budget.model_slots.acquire(blocking=False)
    context.budget.model_slots.release()


@pytest.mark.parametrize(
    "scope",
    [
        {"regulatory_source_hint": "4458"},
        {"regulatory_lookup_heading": "MADDE 168"},
        {"regulatory_candidate_ids": []},
        {"regulatory_candidate_ids": ["one-candidate"]},
    ],
)
def test_live_extra_scopes_are_never_silently_relaxed(
    scope: dict[str, JsonValue],
) -> None:
    session = MagicMock()
    with pytest.raises(CorpusScopeUnavailable, match="scopes require"):
        find_sources(
            session,
            user=cast(User, object()),
            filters=IndexFilters.model_validate({"access_control_list": [], **scope}),
        )
    session.execute.assert_not_called()


def test_query_snapshot_uses_only_requested_source_receipts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from onyx.db import asv3_corpus, search_settings

    source_id = uuid4()
    session = MagicMock()
    session.execute.return_value = [("physical-index", "configured-index")]
    monkeypatch.setattr(asv3_corpus, "qualified_file_ids", lambda *_args: {source_id})
    monkeypatch.setattr(
        search_settings,
        "get_current_search_settings",
        lambda *_args: SimpleNamespace(index_name="configured-index"),
    )
    resolve = MagicMock()
    monkeypatch.setattr(asv3_corpus, "resolve_public_query_index", resolve)
    asv3_corpus.resolve_source_query_index(session, source_id)
    resolve.assert_called_once_with(
        "configured-index", "physical-index", file_ids=(source_id,)
    )


def test_parent_context_selects_all_siblings_and_excludes_descendants(
    broker: MemoryBroker,
) -> None:
    from dataclasses import replace

    rows = [
        replace(
            broker.items[0],
            id=str(n),
            position=n * 100,
            projection_ordinal=n,
            heading_path=("MADDE 27", f"Clause {n}"),
        )
        for n in range(9)
    ]
    rows[3] = replace(rows[3], position=200, projection_ordinal=3)
    rows.append(
        replace(rows[-1], id="other-parent", heading_path=("MADDE 28", "Clause 1"))
    )
    rows.append(
        replace(
            rows[-1], id="descendant", heading_path=("MADDE 27", "Clause 1", "Detail")
        )
    )
    broker.items = rows
    result = CapabilityRegistry(build_corpus_specs(broker)).dispatch(
        CapabilityCall(
            name="read_chunk_context",
            arguments={"source_id": str(broker.item.id), "chunk_id": "2"},
        ),
        RunContext(),
    )
    assert [item.chunk_id for item in result.evidence] == [str(n) for n in range(9)]
    assert result.status == OutcomeStatus.FOUND
    assert result.data["sibling_count"] == 9
    assert result.data["has_more"] is False
    assert result.data["article_closure_complete"] is False
    first = CapabilityRegistry(build_corpus_specs(broker)).dispatch(
        CapabilityCall(
            name="read_chunk_context",
            arguments={"source_id": str(broker.item.id), "chunk_id": "2", "limit": 4},
        ),
        RunContext(),
    )
    second = CapabilityRegistry(build_corpus_specs(broker)).dispatch(
        CapabilityCall(
            name="read_chunk_context",
            arguments={
                "source_id": str(broker.item.id),
                "chunk_id": "2",
                "offset": first.data["next_offset"],
            },
        ),
        RunContext(),
    )
    assert first.data["sibling_count"] == second.data["sibling_count"] == 9
    assert first.status == OutcomeStatus.PARTIAL
    assert [item.chunk_id for item in first.evidence + second.evidence] == [
        str(n) for n in range(9)
    ]
