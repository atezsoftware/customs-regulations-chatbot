"""Exercise actual nested dispatch at the Cropped closure boundary."""

import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import cast
from unittest.mock import MagicMock

import pytest
from pydantic import JsonValue

from onyx.asv3.corpus_tools import CorpusBroker, build_corpus_specs
from onyx.asv3.cropped_research_limit import CroppedResearchLimit, CroppedResearchPolicy
from onyx.asv3.harness import Harness
from onyx.asv3.models import (
    CapabilityCall,
    Decision,
    EvidenceItem,
    OutcomeStatus,
    RunContext,
    ToolOutcome,
    ToolSpec,
)
from onyx.asv3.registry import CapabilityRegistry
from onyx.asv3.sandbox import compose
from onyx.asv3.shared_reads import SharedReads
from onyx.configs.constants import DocumentSource
from onyx.context.search.models import SearchDoc


def saved_operation(snapshot: dict[str, JsonValue], article: str) -> bool:
    state = snapshot.get("state")
    rows = state.get("critical_reads") if isinstance(state, dict) else None
    if not isinstance(rows, dict):
        return False
    return any(
        isinstance(row, dict)
        and isinstance(operation := row.get("operation"), dict)
        and operation.get("article") == article.casefold()
        for row in rows.values()
    )


def force_tail(limit: CroppedResearchLimit) -> None:
    limit.force_finish("Actual input or financial boundary reached")


def setup(
    *,
    cropped: bool = True,
    continuous: bool = False,
    critical_reads: int = 8,
    legacy_closure: bool = True,
    source_text_spec: ToolSpec | None = None,
    compose_handler: Callable[
        [dict[str, JsonValue], RunContext], ToolOutcome
    ] = compose,
) -> tuple[Harness, CroppedResearchLimit, list[str], list[dict[str, JsonValue]]]:
    context = RunContext(run_id="run", scope={"owner": "onur", "document_set": 15})
    limit = CroppedResearchLimit(
        context,
        "Actual question",
        policy=CroppedResearchPolicy(
            critical_reads=critical_reads, enforce_observed_limits=legacy_closure
        ),
    )
    executed: list[str] = []
    saved: list[dict[str, JsonValue]] = []

    def read(args: dict[str, JsonValue], _context: RunContext) -> ToolOutcome:
        if limit.long_tail and cropped:
            assert saved and any(
                saved_operation(snapshot, str(args["article"])) for snapshot in saved
            )
        article = str(args["article"])
        executed.append(article)
        return ToolOutcome(
            status=OutcomeStatus.FOUND,
            summary="Exact full original acquired",
            evidence=[
                EvidenceItem(
                    source_id=str(args["source_id"]),
                    chunk_id=article,
                    text=f"Complete operative text, exception and continuation {article}.",
                )
            ],
            data={"next_article": "244", "subunit_verified": True},
        )

    def broad(_args: dict[str, JsonValue], _context: RunContext) -> ToolOutcome:
        executed.append("broad")
        return ToolOutcome(status=OutcomeStatus.FOUND, summary="Actual broad search")

    registry = CapabilityRegistry(
        [
            ToolSpec(
                name="read_provision",
                description="Exact provision",
                parameters={"type": "object"},
                handler=read,
                parallel_safe=True,
            ),
            source_text_spec
            or ToolSpec(
                name="search_source_text",
                description="Search source text",
                parameters={"type": "object"},
                handler=broad,
                parallel_safe=True,
            ),
            ToolSpec(
                name="search_corpus",
                description="Broad search",
                parameters={"type": "object"},
                handler=broad,
                parallel_safe=True,
            ),
            ToolSpec(
                name="compose_tool_calls",
                description="Compose actual capabilities",
                parameters={"type": "object"},
                handler=compose_handler,
                parallel_safe=True,
                orchestrates=True,
            ),
        ]
    )
    context.services.update(
        asv3_exp_cropped=cropped,
        cropped_research_controller=limit,
        registry=registry,
        experimental_parallel=continuous,
    )
    harness = Harness(
        request="Actual question",
        context=context,
        registry=registry,
        decide=lambda _view: Decision(answer="Done"),
        checkpoint=lambda _snapshot: saved.append(limit.export()),
    )
    return harness, limit, executed, saved


def program(steps: list[dict[str, JsonValue]]) -> CapabilityCall:
    return CapabilityCall(
        name="compose_tool_calls", arguments={"steps": steps, "max_parallel": 4}
    )


def read_step(article: str, *, key: str | None = None) -> dict[str, JsonValue]:
    return {
        "id": key or article,
        "tool": "read_provision",
        "arguments": {"source_id": "known-law", "article": article},
    }


def test_v13_explicit_finish_closes_new_nested_reads_but_retains_free_cached_original() -> (
    None
):
    harness, limit, executed, _ = setup(legacy_closure=False)
    first = harness._dispatch([program([read_step("242")])])[0]
    assert executed == ["242"]
    assert limit.progress()["critical_reads_reserved"] == 0
    force_tail(limit)
    fresh = harness._dispatch([program([read_step("24")])])[0]
    assert fresh.outcome.status == OutcomeStatus.PARTIAL
    assert fresh.outcome.data["not_read"] is True
    assert fresh.outcome.data["absence_proven"] is False
    assert executed == ["242"]
    cached = harness._dispatch(
        [
            CapabilityCall(
                name="read_provision",
                arguments={"source_id": "known-law", "article": "242"},
            )
        ]
    )[0]
    assert cached.evidence_ids == first.evidence_ids == [1]
    assert cached.outcome.data["reused_recorded_read"] is True
    original = harness.evidence.get(1)
    assert original is not None and original.text.endswith("continuation 242.")
    assert executed == ["242"]


@pytest.mark.parametrize("continuous", [False, True])
def test_mixed_actual_compose_executes_independent_exact_reads_and_valid_dependencies(
    continuous: bool,
) -> None:
    harness, limit, executed, _ = setup(continuous=continuous)
    force_tail(limit)
    steps: list[dict[str, JsonValue]] = [
        read_step("242"),
        read_step("24"),
        {
            "id": "broad",
            "tool": "search_source_text",
            "arguments": {"source_id": "known-law", "pattern": "KKDF"},
        },
        {
            **read_step("never", key="blocked-dependent"),
            "depends_on": ["broad"],
        },
        {
            "id": "valid-dependent",
            "tool": "read_provision",
            "arguments": {
                "source_id": "known-law",
                "article": {"$ref": "242.data.next_article"},
            },
            "depends_on": ["242"],
        },
    ]
    receipt = harness._dispatch([program(steps)])[0]
    assert set(executed) == {"242", "24", "244"}
    assert limit.progress()["critical_reads_reserved"] == 3
    assert receipt.outcome.status == OutcomeStatus.PARTIAL
    rows = cast(dict[str, dict[str, JsonValue]], receipt.outcome.data["steps"])
    assert rows["broad"]["status"] == "partial"
    assert rows["broad"]["data"] == {
        "blocked_by_research_limit": True,
        "not_read": True,
        "absence_proven": False,
        "admission": limit.admit_tool(
            "search_source_text", {"source_id": "known-law", "pattern": "KKDF"}
        ).model_dump(mode="json"),
    }
    assert rows["blocked-dependent"]["status"] == "partial"
    assert rows["blocked-dependent"]["data"] == {
        "not_read": True,
        "absence_proven": False,
        "unread_dependencies": ["broad"],
    }
    assert rows["valid-dependent"]["status"] == "found"
    assert len(receipt.evidence_ids) == 3
    assert {
        item.text
        for number in receipt.evidence_ids
        if (item := harness.evidence.get(number)) is not None
    } == {
        f"Complete operative text, exception and continuation {article}."
        for article in executed
    }


def test_nested_complete_cache_reuses_before_admission_and_flat_cache_also_reuses() -> (
    None
):
    harness, limit, executed, _ = setup()
    first = harness._dispatch([program([read_step("242")])])[0]
    force_tail(limit)
    nested = harness._dispatch([program([read_step("242")])])[0]
    flat = harness._dispatch(
        [
            CapabilityCall(
                name="read_provision",
                arguments={"source_id": "known-law", "article": "242"},
            )
        ]
    )[0]
    assert executed == ["242"]
    assert first.evidence_ids == nested.evidence_ids == flat.evidence_ids == [1]
    assert flat.outcome.data["reused_recorded_read"] is True
    assert limit.progress()["critical_reads_reserved"] == 0
    original = harness.evidence.get(1)
    assert original is not None
    assert original.text == "Complete operative text, exception and continuation 242."


def test_explicit_legacy_closure_headroom_allows_sixteen_exact_targets_then_reports_unread() -> (
    None
):
    assert CroppedResearchPolicy().critical_reads == 16
    harness, limit, executed, _ = setup(critical_reads=16)
    force_tail(limit)
    receipt = harness._dispatch([program([read_step(str(i)) for i in range(17)])])[0]
    assert len(executed) == 16
    assert limit.progress()["critical_reads_reserved"] == 16
    assert len(receipt.evidence_ids) == 16
    rows = cast(dict[str, dict[str, JsonValue]], receipt.outcome.data["steps"])
    unread = [row for row in rows.values() if row["status"] == "partial"]
    assert len(unread) == 1
    data = cast(dict[str, JsonValue], unread[0]["data"])
    assert data["not_read"] is True and data["absence_proven"] is False
    assert receipt.outcome.status == OutcomeStatus.PARTIAL


def test_one_global_eight_attempt_allowance_for_flat_nested_and_concurrent_dispatch() -> (
    None
):
    harness, limit, executed, saved = setup()
    force_tail(limit)
    harness._dispatch(
        [
            CapabilityCall(
                name="read_provision",
                arguments={"source_id": "known-law", "article": "first"},
            )
        ]
    )
    receipt = harness._dispatch([program([read_step(str(i)) for i in range(10)])])[0]
    assert len(executed) == len(set(executed)) == 8
    assert limit.progress()["critical_reads_reserved"] == 8
    assert len(receipt.evidence_ids) == 7
    assert receipt.outcome.status == OutcomeStatus.PARTIAL
    assert saved[-1]["state"] == limit.export()["state"]
    forbidden: list[str] = []
    harness.registry.register(
        ToolSpec(
            name="read_chunk",
            description="Exact chunk",
            parameters={"type": "object"},
            handler=lambda _a, _c: (
                forbidden.append("I/O")
                or ToolOutcome(status=OutcomeStatus.FOUND, summary="Should not execute")
            ),
            parallel_safe=True,
        )
    )
    with ThreadPoolExecutor(max_workers=4) as pool:
        outcomes = list(
            pool.map(
                lambda i: harness.registry.dispatch(
                    CapabilityCall(
                        name="read_chunk",
                        arguments={"source_id": "known-law", "chunk_id": str(i)},
                    ),
                    harness.context,
                ),
                range(12),
            )
        )
    assert not forbidden
    assert all(
        x.status == OutcomeStatus.PARTIAL and x.data["not_read"] for x in outcomes
    )
    assert limit.progress()["critical_reads_reserved"] == 8


def test_concurrent_same_exact_nested_read_runs_one_physical_handler() -> None:
    harness, limit, executed, _ = setup()
    force_tail(limit)
    with ThreadPoolExecutor(max_workers=4) as pool:
        receipts = list(
            pool.map(
                lambda _i: harness._dispatch([program([read_step("242")])])[0], range(4)
            )
        )
    assert executed == ["242"]
    assert all(x.outcome.status == OutcomeStatus.FOUND for x in receipts)
    assert limit.progress()["critical_reads_reserved"] == 1


def test_started_orchestration_rechecks_tail_at_each_nested_handler() -> None:
    started, proceed = threading.Event(), threading.Event()

    def delayed(args: dict[str, JsonValue], context: RunContext) -> ToolOutcome:
        started.set()
        assert proceed.wait(5)
        return compose(args, context)

    harness, limit, executed, _ = setup(compose_handler=delayed)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            harness._dispatch,
            [
                program(
                    [
                        read_step("242"),
                        {
                            "id": "search",
                            "tool": "search_corpus",
                            "arguments": {"query": "more"},
                        },
                    ]
                )
            ],
        )
        assert started.wait(5)
        force_tail(limit)
        proceed.set()
        receipt = future.result()[0]
    assert executed == ["242"]
    assert receipt.outcome.status == OutcomeStatus.PARTIAL
    assert limit.progress()["critical_reads_reserved"] == 1


@pytest.mark.parametrize("cropped,tail", [(False, True), (True, False)])
def test_ordinary_and_protected_compose_keep_all_existing_capabilities(
    cropped: bool, tail: bool
) -> None:
    harness, limit, executed, _ = setup(cropped=cropped)
    if tail:
        force_tail(limit)
    receipt = harness._dispatch(
        [
            program(
                [
                    read_step("242"),
                    {
                        "id": "search",
                        "tool": "search_corpus",
                        "arguments": {"query": "more"},
                    },
                ]
            )
        ]
    )[0]
    assert set(executed) == {"242", "broad"}
    assert receipt.outcome.status == OutcomeStatus.FOUND
    assert limit.progress()["critical_reads_reserved"] == 0


@pytest.mark.parametrize("cross_owner", [False, True])
def test_nested_shared_read_rechecks_source_authority_before_cached_reuse(
    cross_owner: bool,
) -> None:
    harness, limit, _, _ = setup()
    context = harness.context
    context.services["research_profile"] = "experimental"
    source = "3c3b3448-1f67-46a0-84f5-16980309c088"
    fences: list[str] = []
    physical: list[str] = []
    authorized = True

    def fence(source_id: str, _context: RunContext) -> str | ToolOutcome:
        fences.append(source_id)
        if not authorized:
            return ToolOutcome(
                status=OutcomeStatus.DENIED, summary="Source access revoked"
            )
        return "captured-index-version"

    context.services["shared_reads"] = SharedReads(
        fence=fence, producer_context=lambda caller: caller.child()
    )

    def read(_args: dict[str, JsonValue], _context: RunContext) -> ToolOutcome:
        physical.append(source)
        return ToolOutcome(
            status=OutcomeStatus.FOUND,
            summary="Actual immutable chunk",
            evidence=[
                EvidenceItem(
                    source_id=source,
                    chunk_id="canonical",
                    text="Exact condition and exception.",
                    search_doc=SearchDoc(
                        document_id=source,
                        chunk_ind=0,
                        semantic_identifier="Original",
                        blurb="Original",
                        source_type=DocumentSource.FILE,
                        boost=0,
                        hidden=False,
                        metadata={},
                        match_highlights=[],
                    ),
                )
            ],
        )

    harness.registry.register(
        ToolSpec(
            name="read_chunk",
            description="Exact chunk",
            parameters={"type": "object"},
            handler=read,
            parallel_safe=True,
        )
    )
    if not cross_owner:
        force_tail(limit)
    steps: list[dict[str, JsonValue]] = [
        {
            "id": "chunk",
            "tool": "read_chunk",
            "arguments": {"source_id": source, "chunk_id": "canonical"},
        }
    ]
    first = harness._dispatch([program(steps)])[0]
    if cross_owner:
        force_tail(limit)
        worker_context = context.child()
        worker_context.services["task_id"] = "worker"
        worker = Harness(
            request="Worker question",
            context=worker_context,
            registry=harness.registry,
            evidence=harness.evidence,
            decide=lambda _view: Decision(answer="Done"),
        )
    else:
        worker = harness
    repeated = worker._dispatch([program(steps)])[0]
    assert first.evidence_ids == repeated.evidence_ids == [1]
    assert physical == [source] and len(fences) == 2
    assert limit.progress()["critical_reads_reserved"] == (0 if cross_owner else 1)
    authorized = False
    rejected = worker._dispatch([program(steps)])[0]
    assert rejected.outcome.status == OutcomeStatus.PARTIAL
    assert rejected.evidence_ids == []
    assert physical == [source] and len(fences) == 3


def test_invalid_nested_metadata_never_consumes_critical_attempt_or_bypasses_validation() -> (
    None
):
    harness, limit, executed, _ = setup()
    force_tail(limit)
    invalid = read_step("242")
    arguments = cast(dict[str, JsonValue], invalid["arguments"])
    arguments["_outcomes"] = []
    receipt = harness._dispatch([program([invalid])])[0]
    assert receipt.outcome.status == OutcomeStatus.PARTIAL
    assert executed == [] and limit.progress()["critical_reads_reserved"] == 0
    corrected = harness._dispatch([program([read_step("242")])])[0]
    assert corrected.outcome.status == OutcomeStatus.FOUND
    assert executed == ["242"] and limit.progress()["critical_reads_reserved"] == 1


@pytest.mark.parametrize("cropped,tail", [(True, True), (True, False), (False, True)])
def test_mixed_range_pagination_and_complete_article_preserves_healthy_path(
    cropped: bool,
    tail: bool,
) -> None:
    harness, limit, executed, _ = setup(cropped=cropped)
    ranged: list[int] = []

    def read_range(args: dict[str, JsonValue], _context: RunContext) -> ToolOutcome:
        ranged.append(int(cast(int, args.get("start", 0))))
        return ToolOutcome(
            status=OutcomeStatus.FOUND,
            summary="Actual complete original range",
            evidence=[
                EvidenceItem(
                    source_id="known-law",
                    chunk_id="range",
                    text="Exact complete literal range.",
                )
            ],
        )

    harness.registry.register(
        ToolSpec(
            name="read_source_range",
            description="Source range",
            parameters={"type": "object"},
            handler=read_range,
        )
    )
    if tail:
        force_tail(limit)
    receipt = harness._dispatch(
        [
            program(
                [
                    {
                        "id": "range",
                        "tool": "read_source_range",
                        "arguments": {
                            "source_id": "known-law",
                            "start": 100,
                            "limit": 50,
                        },
                    },
                    read_step("8"),
                    {**read_step("never", key="dependent"), "depends_on": ["range"]},
                ]
            )
        ]
    )[0]
    rows = cast(dict[str, dict[str, JsonValue]], receipt.outcome.data["steps"])
    if cropped and tail:
        assert ranged == [] and executed == ["8"]
        assert rows["range"]["status"] == rows["dependent"]["status"] == "partial"
        assert cast(dict[str, JsonValue], rows["range"]["data"])["not_read"] is True
        assert (
            cast(dict[str, JsonValue], rows["range"]["data"])["absence_proven"] is False
        )
        assert limit.progress()["critical_reads_reserved"] == 1
    else:
        assert ranged == [100] and set(executed) == {"8", "never"}
        assert receipt.outcome.status == OutcomeStatus.FOUND
        assert limit.progress()["critical_reads_reserved"] == 0


def test_complete_range_cached_before_tail_reuses_without_new_range_admission() -> None:
    harness, limit, _, _ = setup()
    physical: list[int] = []

    def read_range(_args: dict[str, JsonValue], _context: RunContext) -> ToolOutcome:
        physical.append(1)
        return ToolOutcome(
            status=OutcomeStatus.FOUND,
            summary="Actual complete immutable source range",
            evidence=[
                EvidenceItem(
                    source_id="known-law",
                    chunk_id="range",
                    text="Every original literal remains unchanged.",
                )
            ],
        )

    harness.registry.register(
        ToolSpec(
            name="read_source_range",
            description="Range",
            parameters={"type": "object"},
            handler=read_range,
        )
    )
    arguments: dict[str, JsonValue] = {
        "source_id": "known-law",
        "start": 0,
        "limit": 30,
    }
    first = harness._dispatch(
        [
            program(
                [{"id": "range", "tool": "read_source_range", "arguments": arguments}]
            )
        ]
    )[0]
    force_tail(limit)
    repeated = harness._dispatch(
        [
            program(
                [{"id": "range", "tool": "read_source_range", "arguments": arguments}]
            )
        ]
    )[0]
    assert physical == [1] and first.evidence_ids == repeated.evidence_ids == [1]
    rows = cast(dict[str, dict[str, JsonValue]], repeated.outcome.data["steps"])
    assert (
        cast(dict[str, JsonValue], rows["range"]["data"])["reused_recorded_read"]
        is True
    )
    assert limit.progress()["critical_reads_reserved"] == 0


def test_actual_nested_heading_locator_follows_reserved_article_only_once() -> None:
    locators: list[str] = []

    def locate(args: dict[str, JsonValue], _context: RunContext) -> ToolOutcome:
        assert limit.progress()["critical_reads_reserved"] == 2
        locators.append(str(args["pattern"]))
        return ToolOutcome(
            status=OutcomeStatus.FOUND,
            summary="Literal source locator",
            data={"absence_proven": False},
        )

    harness, limit, executed, _ = setup(
        source_text_spec=ToolSpec(
            name="search_source_text",
            description="Locator",
            parameters={"type": "object"},
            handler=locate,
        )
    )
    force_tail(limit)
    harness._dispatch([program([read_step("61")])])
    limit.record_sources(
        [
            EvidenceItem(
                source_id="known-law",
                chunk_id="61",
                text="Previously acquired exact source original.",
            )
        ]
    )
    steps: list[dict[str, JsonValue]] = [
        {
            "id": "heading",
            "tool": "search_source_text",
            "arguments": {"source_id": "known-law", "pattern": "MADDE 61"},
        },
        {
            "id": "broad",
            "tool": "search_source_text",
            "arguments": {"source_id": "known-law", "pattern": "KKDF"},
        },
    ]
    receipt = harness._dispatch([program(steps)])[0]
    assert locators == ["MADDE 61"] and executed == ["61"]
    assert receipt.outcome.status == OutcomeStatus.PARTIAL
    harness._dispatch([program(steps)])
    assert locators == ["MADDE 61"]
    assert limit.progress()["critical_reads_reserved"] == 2


def test_heading_locator_uses_actual_corpus_handler_authority_check() -> None:
    broker = MagicMock(spec=CorpusBroker)
    broker.iter_chunks.side_effect = PermissionError("Current authority denied")
    locator = next(
        spec for spec in build_corpus_specs(broker) if spec.name == "search_source_text"
    )
    harness, limit, _, _ = setup(source_text_spec=locator)
    force_tail(limit)
    source = "3c3b3448-1f67-46a0-84f5-16980309c088"
    limit.record_sources(
        [
            EvidenceItem(
                source_id=source, chunk_id="owned", text="Exact acquired source."
            )
        ]
    )
    assert limit.admit_tool(
        "read_provision",
        {"source_id": source, "article": "61"},
        critical_target_key="actual-attempt",
    ).allowed
    result = harness.registry.dispatch(
        CapabilityCall(
            name="search_source_text",
            arguments={"source_id": source, "pattern": "MADDE 61"},
        ),
        harness.context,
    )
    assert result.status == OutcomeStatus.DENIED and result.evidence == []
    broker.iter_chunks.assert_called_once_with(source, harness.context, start=0)
    assert limit.progress()["critical_reads_reserved"] == 2
