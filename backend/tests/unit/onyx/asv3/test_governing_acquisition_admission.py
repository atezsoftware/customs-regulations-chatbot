"""Completed missing-source targets fence physical calls without approving answers."""

import copy
from collections import Counter
from collections.abc import Callable
from typing import cast
from unittest.mock import MagicMock

import pytest
from pydantic import JsonValue

from onyx.asv3.canonical_authority import canonical_named_authority_gap
from onyx.asv3.corpus_tools import CorpusBroker, build_corpus_specs
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.focused_source_target import focused_source_target
from onyx.asv3.governing_acquisitions import (
    GoverningOriginalAcquisitions,
    GoverningOriginalTarget,
    governing_original_targets,
)
from onyx.asv3.harness import Harness
from onyx.asv3.models import (
    CapabilityCall,
    Decision,
    EvidenceItem,
    OutcomeStatus,
    RunContext,
    ToolOutcome,
    ToolReceipt,
    ToolSpec,
)
from onyx.asv3.registry import CapabilityRegistry
from onyx.asv3.sandbox import compose
from onyx.asv3.workflow_variant import ASV3_STANDARD_VARIANT, ASV3_TUNED_VARIANT
from tests.unit.onyx.asv3.test_native_authority import original

SOURCE_ID = "00000000-0000-0000-0000-000000006455"
OTHER_SOURCE_ID = "00000000-0000-0000-0000-000000005326"
REQUEST = "6455 sayılı Kanun m.11 ve diğer bağımsız soruları araştır."


def source_original(title: str, article: str, source_id: str) -> EvidenceItem:
    item = original(title, article)
    assert item.search_doc is not None
    return item.model_copy(
        update={
            "source_id": source_id,
            "search_doc": item.search_doc.model_copy(update={"document_id": source_id}),
        }
    )


def search_arguments(query: str) -> dict[str, JsonValue]:
    return {
        "query": query,
        "mode": "hybrid",
        "coverage_item": "Özgün hüküm ve diğer bağımsız hukuki kaynakları ayır.",
        "evidence_target": "Sorulan özgün hüküm metni.",
    }


def target(**changes: JsonValue) -> GoverningOriginalTarget:
    row: dict[str, JsonValue] = {
        "instrument_number": "6455",
        "article": "11",
        "matching_original_evidence": [],
        **changes,
    }
    return governing_original_targets({"named_authority_gaps": [row]})[0]


class AdmissionRun:
    def __init__(self, variant: str = ASV3_TUNED_VARIANT) -> None:
        self.context = RunContext(
            scope={"document_set_id": 15, "as_of_date": "2026-10-13"},
            services={"asv3_workflow_variant": variant},
        )
        self.ledger = EvidenceLedger()
        self.ledger.add(
            [
                source_original("6455 sayılı İşlem Kanunu", "1", SOURCE_ID),
                source_original("5326 sayılı Kabahatler Kanunu", "1", OTHER_SOURCE_ID),
            ],
            self.context,
        )
        self.state = GoverningOriginalAcquisitions(self.context, REQUEST)
        self.counts: Counter[str] = Counter()
        self.observed: list[CapabilityCall] = []
        self.handler: Callable[[dict[str, JsonValue], RunContext], ToolOutcome] = (
            self.missing
        )
        specs: list[ToolSpec] = []
        for name in (
            "read_named_provision",
            "read_provision",
            "read_source_range",
            "read_chunk",
            "read_chunk_context",
            "search_source_text",
            "search_corpus",
            "resolve_source",
            "query_corpus",
            "follow_reference",
            "compare_versions",
            "run_research_code",
        ):
            specs.append(
                ToolSpec(
                    name=name,
                    description="Count actual physical source dispatches.",
                    parameters={"type": "object"},
                    handler=self.counted(name),
                )
            )
        specs.append(
            ToolSpec(
                name="compose_tool_calls",
                description="Actual nested registry dispatch.",
                parameters={"type": "object"},
                handler=compose,
                orchestrates=True,
            )
        )
        self.registry = CapabilityRegistry(specs)
        self.context.services.update(
            evidence=self.ledger,
            tuned_governing_acquisitions=self.state,
            registry=self.registry,
        )

    @staticmethod
    def missing(_arguments: dict[str, JsonValue], _context: RunContext) -> ToolOutcome:
        return ToolOutcome(
            status=OutcomeStatus.NOT_FOUND,
            summary="Actual source lookup did not obtain the original.",
            data={"sources": [], "has_more": False, "absence_proven": False},
        )

    def counted(
        self, name: str
    ) -> Callable[[dict[str, JsonValue], RunContext], ToolOutcome]:
        def execute(
            arguments: dict[str, JsonValue], context: RunContext
        ) -> ToolOutcome:
            self.counts[name] += 1
            self.observed.append(CapabilityCall(name=name, arguments=arguments))
            return self.handler(arguments, context)

        return execute

    def dispatch(
        self,
        name: str,
        arguments: dict[str, JsonValue],
        context: RunContext | None = None,
    ) -> ToolOutcome:
        return self.registry.dispatch(
            CapabilityCall(name=name, arguments=arguments), context or self.context
        )

    def unavailable(
        self, missing_target: GoverningOriginalTarget | None = None
    ) -> GoverningOriginalTarget:
        missing_target = missing_target or target()
        call = self.state.next_call(missing_target, self.context, self.ledger)
        assert call is not None
        assert self.state.reserve(missing_target, call, self.context)
        outcome = self.registry.dispatch(call, self.context)
        assert outcome.status == OutcomeStatus.NOT_FOUND
        assert self.state.record_receipt(
            ToolReceipt(call=call, outcome=outcome, elapsed_seconds=0),
            self.context,
            self.ledger,
        )
        return missing_target

    def use_actual_search_schema(self) -> None:
        specs = build_corpus_specs(
            cast(CorpusBroker, MagicMock(spec=CorpusBroker)),
            require_search_targets=True,
            source_identity_guidance=True,
            named_provision_reads=True,
        )
        spec = next(spec for spec in specs if spec.name == "search_corpus")
        self.registry = CapabilityRegistry(
            [spec.model_copy(update={"handler": self.counted("search_corpus")})]
        )
        self.context.services["registry"] = self.registry


@pytest.mark.parametrize(
    "name,arguments",
    [
        ("read_named_provision", {"source_name": "6455 sayılı Kanun", "article": "11"}),
        ("read_named_provision", {"source_name": "6455 SAYILI KANUN", "article": "11"}),
        ("read_named_provision", {"source_name": "İşlem Kanunu", "article": "11"}),
        ("read_provision", {"source_id": SOURCE_ID, "article": "11"}),
        ("search_source_text", {"source_id": SOURCE_ID, "pattern": "MADDE 11"}),
        ("search_source_text", {"source_id": SOURCE_ID, "pattern": "11. maddesi"}),
        ("search_corpus", {"query": "6455 sayılı Kanun m. 11"}),
        ("search_corpus", {"query": "6455 sayılı Kanun'un 11. maddesi"}),
        ("resolve_source", {"query": "6455 sayılı Kanun m.11"}),
        ("query_corpus", {"operation": "inventory", "query": "6455 sayılı Kanun m.11"}),
        (
            "follow_reference",
            {
                "source_id": OTHER_SOURCE_ID,
                "chunk_id": "actual-origin",
                "target_source_id": SOURCE_ID,
                "article": "11",
            },
        ),
    ],
)
def test_completed_target_denies_direct_rephrased_calls_before_handler(
    name: str, arguments: dict[str, JsonValue]
) -> None:
    run = AdmissionRun()
    run.unavailable()
    before = run.counts.copy()
    outcome = run.dispatch(name, arguments)
    assert outcome.status == OutcomeStatus.DENIED
    assert run.counts == before
    assert run.ledger.completely_delivered("unexecuted-model-call") == set()


@pytest.mark.parametrize(
    "name,arguments",
    [
        ("read_named_provision", {"source_name": "5326 sayılı Kanun", "article": "11"}),
        ("read_provision", {"source_id": SOURCE_ID, "article": "12"}),
        ("read_provision", {"source_id": SOURCE_ID, "article": "11", "paragraph": "2"}),
        ("read_provision", {"source_id": SOURCE_ID, "article": "11", "clause": "b"}),
        ("read_provision", {"source_id": SOURCE_ID, "article": "geçici 11"}),
        ("search_corpus", {"query": "5326 sayılı Kanun m.11"}),
        ("search_corpus", {"query": "ödeme koşulları ve başka belgeler"}),
        ("search_source_text", {"source_id": SOURCE_ID, "pattern": "MADDE 12"}),
        ("search_source_text", {"source_id": OTHER_SOURCE_ID, "pattern": "MADDE 11"}),
        ("search_source_text", {"source_id": SOURCE_ID, "pattern": "ödeme belgesi"}),
        ("resolve_source", {"query": "6455 sayılı Kanun"}),
        ("read_source_range", {"source_id": SOURCE_ID, "start": 0, "limit": 20}),
        ("run_research_code", {"code": "print(11)", "source_ids": [SOURCE_ID]}),
        (
            "compare_versions",
            {
                "source_id": SOURCE_ID,
                "old_date": "2024-01-01",
                "new_date": "2026-10-13",
                "article": "11",
            },
        ),
        (
            "follow_reference",
            {"source_id": SOURCE_ID, "chunk_id": "actual-origin"},
        ),
        (
            "follow_reference",
            {
                "source_id": SOURCE_ID,
                "chunk_id": "actual-origin",
                "target_source_id": OTHER_SOURCE_ID,
                "article": "11",
            },
        ),
        (
            "follow_reference",
            {
                "source_id": OTHER_SOURCE_ID,
                "chunk_id": "actual-origin",
                "target_source_id": SOURCE_ID,
                "article": "12",
            },
        ),
    ],
)
def test_an_unavailable_article_does_not_blacklist_other_targets_or_unbound_actions(
    name: str, arguments: dict[str, JsonValue]
) -> None:
    run = AdmissionRun()
    run.unavailable()
    before = run.counts[name]
    assert run.dispatch(name, arguments).status == OutcomeStatus.NOT_FOUND
    assert run.counts[name] == before + 1


def test_nested_composition_blocks_same_target_and_executes_unrelated_step() -> None:
    run = AdmissionRun()
    run.unavailable()
    before = run.counts.copy()
    outcome = run.dispatch(
        "compose_tool_calls",
        {
            "steps": [
                {
                    "id": "unavailable",
                    "tool": "read_named_provision",
                    "arguments": {"source_name": "6455 sayılı Kanun", "article": "11"},
                },
                {
                    "id": "other-law",
                    "tool": "read_named_provision",
                    "arguments": {"source_name": "5326 sayılı Kanun", "article": "11"},
                },
                {
                    "id": "sandbox",
                    "tool": "run_research_code",
                    "arguments": {
                        "code": "print(1 + 1)",
                        "source_ids": [OTHER_SOURCE_ID],
                    },
                },
            ]
        },
    )
    steps = outcome.data["steps"]
    assert isinstance(steps, dict)
    assert cast(dict[str, JsonValue], steps["unavailable"])["status"] == "denied"
    assert cast(dict[str, JsonValue], steps["other-law"])["status"] == "not_found"
    assert run.counts["read_named_provision"] == before["read_named_provision"] + 1
    assert run.counts["run_research_code"] == 1


def test_shared_worker_and_restored_same_request_cannot_reopen_completed_target() -> (
    None
):
    run = AdmissionRun()
    run.unavailable()
    worker = run.context.child()
    worker.services["task_id"] = "actual-worker"
    before = run.counts.copy()
    assert (
        run.dispatch(
            "read_provision", {"source_id": SOURCE_ID, "article": "11"}, worker
        ).status
        == OutcomeStatus.DENIED
    )
    restored = GoverningOriginalAcquisitions(run.context, REQUEST)
    restored.restore(copy.deepcopy(run.state.export()))
    run.context.services["tuned_governing_acquisitions"] = restored
    assert (
        run.dispatch(
            "read_named_provision",
            {"source_name": "6455 sayılı Kanun", "article": "11"},
        ).status
        == OutcomeStatus.DENIED
    )
    assert run.counts == before


def test_only_own_reserved_call_can_execute_during_active_episode() -> None:
    run = AdmissionRun()
    missing_target = target()
    reserved = run.state.next_call(missing_target, run.context, run.ledger)
    assert reserved is not None
    assert run.state.reserve(missing_target, reserved, run.context)
    worker = run.context.child()
    worker.services["task_id"] = "other-owner"
    assert run.registry.dispatch(reserved, worker).status == OutcomeStatus.DENIED
    assert (
        run.dispatch(
            "read_provision", {"source_id": SOURCE_ID, "article": "11"}, worker
        ).status
        == OutcomeStatus.DENIED
    )
    assert not run.counts
    assert (
        run.registry.dispatch(reserved, run.context).status == OutcomeStatus.NOT_FOUND
    )
    assert run.counts["read_provision"] == 1


def test_actual_advancing_continuation_remains_executable_in_same_episode() -> None:
    run = AdmissionRun()
    missing_target = target()
    reserved = run.state.next_call(missing_target, run.context, run.ledger)
    assert reserved is not None
    assert run.state.reserve(missing_target, reserved, run.context)

    def partial(_arguments: dict[str, JsonValue], _context: RunContext) -> ToolOutcome:
        return ToolOutcome(
            status=OutcomeStatus.PARTIAL,
            summary="The actual bounded scan has a continuation.",
            data={
                "source_id": SOURCE_ID,
                "scan_truncated": True,
                "evidence_truncated": False,
                "next_position": 2000,
                "absence_proven": False,
            },
        )

    run.handler = partial
    outcome = run.registry.dispatch(reserved, run.context)
    assert run.state.record_receipt(
        ToolReceipt(call=reserved, outcome=outcome, elapsed_seconds=0),
        run.context,
        run.ledger,
    )
    continuation = run.state.next_call(missing_target, run.context, run.ledger)
    assert continuation is not None
    assert continuation.arguments["start"] == 2000
    assert run.state.reserve(missing_target, continuation, run.context)
    run.handler = run.missing
    final = run.registry.dispatch(continuation, run.context)
    assert final.status == OutcomeStatus.NOT_FOUND
    assert run.state.record_receipt(
        ToolReceipt(call=continuation, outcome=final, elapsed_seconds=0),
        run.context,
        run.ledger,
    )
    assert run.counts["read_provision"] == 2
    assert (
        run.dispatch(
            "read_provision", {"source_id": SOURCE_ID, "article": "11", "start": 2000}
        ).status
        == OutcomeStatus.DENIED
    )


@pytest.mark.parametrize(
    "missing_identity, same_arguments, different_arguments",
    [
        (
            {"paragraph": "3"},
            {"article": "11/3"},
            {"article": "11", "paragraph": "1"},
        ),
        (
            {"clause": "a"},
            {"article": "11", "clause": "a"},
            {"article": "11", "clause": "b"},
        ),
        (
            {"article_qualifier": "geçici"},
            {"article": "GEÇİCİ 11"},
            {"article": "11"},
        ),
    ],
)
def test_subunit_and_article_qualifier_are_target_identity_not_keyword_matches(
    missing_identity: dict[str, JsonValue],
    same_arguments: dict[str, JsonValue],
    different_arguments: dict[str, JsonValue],
) -> None:
    run = AdmissionRun()
    missing_target = run.unavailable(target(**missing_identity))
    denied = run.dispatch("read_provision", {"source_id": SOURCE_ID, **same_arguments})
    assert denied.status == OutcomeStatus.DENIED
    assert denied.data["governing_original_target_key"] == missing_target.key
    assert (
        run.dispatch(
            "read_provision", {"source_id": SOURCE_ID, **different_arguments}
        ).status
        == OutcomeStatus.NOT_FOUND
    )


def test_mixed_reference_search_is_not_assumed_to_be_only_the_failed_target() -> None:
    run = AdmissionRun()
    run.unavailable()
    assert (
        run.dispatch(
            "search_corpus",
            {"query": "6455 sayılı Kanun m.11 ile 5326 sayılı Kanun m.15 ilişkisi"},
        ).status
        == OutcomeStatus.NOT_FOUND
    )


def test_new_date_scope_must_use_a_new_episode_state_and_can_research_again() -> None:
    run = AdmissionRun()
    run.unavailable()
    changed = RunContext(
        run_id=run.context.run_id,
        scope={**run.context.scope, "as_of_date": "2024-01-01"},
        services=run.context.services,
    )
    with pytest.raises(ValueError, match="scope changed"):
        run.state.admission_gap(
            CapabilityCall(
                name="read_provision",
                arguments={"source_id": SOURCE_ID, "article": "11"},
            ),
            changed,
            run.ledger,
        )
    changed.services["tuned_governing_acquisitions"] = GoverningOriginalAcquisitions(
        changed, REQUEST
    )
    assert (
        run.dispatch(
            "read_provision", {"source_id": SOURCE_ID, "article": "11"}, changed
        ).status
        == OutcomeStatus.NOT_FOUND
    )


@pytest.mark.parametrize("profile", ["normal", "experimental", "deep"])
def test_other_workflows_do_not_adopt_tuned_admission_even_if_state_is_present(
    profile: str,
) -> None:
    run = AdmissionRun(ASV3_STANDARD_VARIANT)
    run.context.services["research_profile"] = profile
    run.unavailable()
    assert (
        run.dispatch("read_provision", {"source_id": SOURCE_ID, "article": "11"}).status
        == OutcomeStatus.NOT_FOUND
    )
    assert run.counts["read_provision"] == 2


def test_actual_later_original_removes_denial_without_creating_model_delivery() -> None:
    run = AdmissionRun()
    run.unavailable()
    number = run.ledger.add(
        [source_original("6455 sayılı İşlem Kanunu", "11", SOURCE_ID)],
        run.context,
    )[0]
    assert (
        run.dispatch("read_provision", {"source_id": SOURCE_ID, "article": "11"}).status
        == OutcomeStatus.NOT_FOUND
    )
    assert run.ledger.completely_delivered("never-invoked") == set()
    assert (
        canonical_named_authority_gap(
            f"6455 sayılı Kanun m.11 uygulanır [{number}].", run.ledger
        )
        is None
    )
    assert run.state.records_for([target()])[0]["legal_approval"] is False


@pytest.mark.parametrize("with_other_target", [False, True])
def test_search_preparation_never_receives_blocked_target(
    with_other_target: bool,
) -> None:
    run = AdmissionRun()
    run.unavailable()
    run.use_actual_search_schema()
    prepared: list[list[str]] = []

    def prepare(calls: list[CapabilityCall], context: RunContext) -> dict[str, object]:
        assert context is run.context
        prepared.append([call.call_id for call in calls])
        assert all("6455" not in str(call.arguments.get("query")) for call in calls)
        return {}

    run.context.services["prepare_search_batch"] = prepare
    blocked = CapabilityCall(
        name="search_corpus", arguments=search_arguments("6455 sayılı Kanun m.11")
    )
    other = CapabilityCall(
        name="search_corpus", arguments=search_arguments("5326 sayılı Kanun m.15")
    )
    harness = Harness(
        request=REQUEST,
        context=run.context,
        registry=run.registry,
        decide=lambda _view: Decision(),
        evidence=run.ledger,
    )
    receipts = harness._dispatch([blocked, other] if with_other_target else [blocked])
    assert prepared == [[other.call_id] if with_other_target else []]
    assert receipts[0].outcome.status == OutcomeStatus.DENIED
    assert run.counts["search_corpus"] == int(with_other_target)
    assert not run.ledger.completely_delivered("unexecuted-model-call")


def test_explicit_related_authority_discovery_with_actual_schema_is_not_original_retry() -> (
    None
):
    run = AdmissionRun()
    run.unavailable()
    run.use_actual_search_schema()
    arguments: dict[str, JsonValue] = {
        **search_arguments(
            "6455 sayılı Kanun m.11 uygulamasına ilişkin Danıştay kararları"
        ),
        "coverage_item": "Başka bir yargı kararının kapsamını incele.",
        "evidence_target": "Bağımsız yargı kararı veya başka düzenlemenin özgün metni.",
        "discover_related_sources": True,
    }
    assert run.dispatch("search_corpus", arguments).status == OutcomeStatus.NOT_FOUND
    assert run.counts["search_corpus"] == 1
    assert run.state.records_for([target()])[0]["legal_approval"] is False


@pytest.mark.parametrize("discovery", [False, "true", None])
def test_original_retry_is_not_exempted_by_non_boolean_discovery_flag(
    discovery: JsonValue,
) -> None:
    run = AdmissionRun()
    run.unavailable()
    run.use_actual_search_schema()
    arguments = {
        **search_arguments("6455 sayılı Kanun m.11"),
        "discover_related_sources": discovery,
    }
    assert run.dispatch("search_corpus", arguments).status == OutcomeStatus.DENIED
    assert run.counts["search_corpus"] == 0


def test_formal_only_source_alias_requires_an_unambiguous_actual_identity() -> None:
    run = AdmissionRun()
    run.unavailable()
    run.ledger.add(
        [
            source_original(
                "5326 sayılı İşlem Kanunu", "1", "00000000-0000-0000-0000-000000005327"
            )
        ],
        run.context,
    )
    assert (
        run.dispatch(
            "read_named_provision", {"source_name": "İşlem Kanunu", "article": "11"}
        ).status
        == OutcomeStatus.NOT_FOUND
    )


@pytest.mark.parametrize(
    "evidence_target, expected_source, expected_article, expected_status",
    [
        (
            "5326 sayılı Kabahatler Kanunu 15. maddesi özgün metni",
            "5326 sayılı Kabahatler Kanunu",
            "15",
            OutcomeStatus.NOT_FOUND,
        ),
        (
            "6455 sayılı İşlem Kanunu 12. maddesi özgün metni",
            "6455 sayılı İşlem Kanunu",
            "12",
            OutcomeStatus.NOT_FOUND,
        ),
        (
            "6455 sayılı İşlem Kanunu 11. maddesi özgün metni",
            "6455 sayılı İşlem Kanunu",
            "11",
            OutcomeStatus.DENIED,
        ),
    ],
)
def test_canonical_search_redirect_is_checked_by_actual_focused_target(
    evidence_target: str,
    expected_source: str,
    expected_article: str,
    expected_status: OutcomeStatus,
) -> None:
    run = AdmissionRun()
    run.unavailable()
    run.use_actual_search_schema()
    arguments = {
        **search_arguments("6455 sayılı Kanun m.11"),
        "evidence_target": evidence_target,
    }
    assert focused_source_target(arguments) == (expected_source, expected_article)
    assert run.dispatch("search_corpus", arguments).status == expected_status
    assert run.counts["search_corpus"] == int(
        expected_status == OutcomeStatus.NOT_FOUND
    )
