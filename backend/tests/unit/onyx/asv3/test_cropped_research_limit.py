"""Cropped limits are shared, resume-safe and never evidence of legal absence."""

import copy
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from typing import cast

import pytest
from pydantic import JsonValue

from onyx.asv3.cropped_research_limit import CroppedResearchLimit, CroppedResearchPolicy
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import EvidenceItem, HarnessView, RunContext


def setup(
    *, policy: CroppedResearchPolicy | None = None
) -> tuple[RunContext, CroppedResearchLimit]:
    context = RunContext(
        run_id="one-run",
        scope={"owner": "onur", "document_set": 15, "as_of": "2026-10-12"},
    )
    return context, CroppedResearchLimit(
        context,
        "Actual question",
        policy=policy
        or CroppedResearchPolicy(critical_reads=8, enforce_observed_limits=True),
    )


def sources(count: int) -> list[EvidenceItem]:
    return [
        EvidenceItem(
            source_id="law",
            chunk_id=str(index),
            text=f"Complete condition and exception {index}",
        )
        for index in range(count)
    ]


def bind_controller(context: RunContext, state: CroppedResearchLimit) -> None:
    context.services.update(asv3_exp_cropped=True, cropped_research_controller=state)


def checkpoint_hash(snapshot: dict[str, JsonValue]) -> None:
    snapshot["state_hash"] = hashlib.sha256(
        json.dumps(
            snapshot["state"],
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def long_tail() -> tuple[RunContext, CroppedResearchLimit]:
    context, state = setup()
    state.record_search("query-one")
    state.record_search("query-two")
    state.record_search("query-three")
    for index in range(8):
        state.reserve_model("root" if index % 2 else "worker")
    state.force_finish("Explicit closure test boundary")
    assert state.long_tail
    return context, state


def test_joint_threshold_does_not_cut_ordinary_searches_or_223_originals() -> None:
    _, state = setup()
    state.record_sources(sources(223))
    for _ in range(8):
        state.reserve_model("root")
    state.record_search("one")
    assert not state.long_tail
    assert state.admit_tool(
        "search_corpus", {"query": "second focused question"}
    ).allowed
    state.record_search("two")
    assert not state.long_tail
    assert state.admit_tool(
        "search_corpus", {"query": "third focused question"}
    ).allowed
    state.record_search("three")
    assert not state.long_tail
    state.record_sources([sources(224)[-1]])
    assert state.long_tail
    admission = state.admit_tool("search_corpus", {"query": "broad fourth query"})
    assert not admission.allowed and not admission.absence_proven


def test_original_joint_threshold_preserves_every_original_and_exception() -> None:
    _, state = setup()
    rows = sources(243)
    state.record_sources(rows)
    assert state.progress()["original_records"] == 243
    assert not state.long_tail
    for _ in range(7):
        state.reserve_model("root")
    assert not state.long_tail
    state.reserve_model("worker")
    assert not state.long_tail
    for index in range(3):
        state.record_search(f"actual-query-{index}")
    assert state.long_tail
    assert state.progress()["original_records"] == 243
    assert rows[-1].text.endswith("exception 242")


def test_actual_v10_joint_envelope_still_activates_with_shared_worker_counts() -> None:
    _, state = setup()
    state.record_sources(sources(444))
    for index in range(3):
        state.record_search(f"actual-search-{index}")
    for _ in range(4):
        state.reserve_model("root")
    for _ in range(3):
        state.reserve_model("worker")
    assert not state.long_tail
    state.reserve_model("worker")
    assert state.long_tail
    assert state.progress()["research_model_calls"] == 8
    assert state.progress()["original_records"] == 444


def test_tail_schemas_keep_exact_reads_cached_ranges_and_all_publication_controls() -> (
    None
):
    context, state = setup()
    names = [
        "search_corpus",
        "query_corpus",
        "spawn_researcher",
        "run_research_code",
        "read_named_provision",
        "read_provision",
        "read_chunk",
        "read_chunk_context",
        "follow_reference",
        "search_source_text",
        "read_source_range",
        "read_evidence",
        "inspect_evidence_path",
        "inspect_research",
        "read_research_state",
        "discover_tools",
        "list_researchers",
        "wait_researcher",
        "update_research",
        "compose_tool_calls",
        "submit_answer",
        "submit_partial_answer",
        "submit_retained_answer",
        "submit_retained_partial_answer",
    ]
    tools: list[dict[str, JsonValue]] = [
        {
            "type": "function",
            "function": {"name": name, "parameters": {"type": "object"}},
        }
        for name in names
    ]
    before = copy.deepcopy(tools)
    assert state.available_tools(tools, context) is tools
    state.force_finish("actual tail")
    selected = state.available_tools(tools, context)
    assert [
        cast(dict[str, JsonValue], tool["function"])["name"] for tool in selected
    ] == names[4:]
    assert tools == before
    assert not state.admit_tool("search_corpus", {"query": "closed"}).allowed
    assert state.admit_tool(
        "read_provision",
        {"source_id": "law", "article": "8"},
        critical_target_key="real material condition",
    ).allowed
    with pytest.raises(ValueError, match="scope"):
        state.available_tools(
            tools, RunContext(run_id=context.run_id, scope={"owner": "other"})
        )


def test_read_planning_reservation_is_tail_only_once_shared_and_resumable() -> None:
    context, state = setup()
    assert not state.reserve_read_plan(context, "root")
    state.force_finish("actual tail")
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(lambda _: state.reserve_read_plan(context, "root"), range(2))
        )
    assert sorted(results) == [False, True]
    snapshot = state.export()
    resumed = CroppedResearchLimit(context, state.request, policy=state.policy)
    resumed.restore(snapshot)
    assert not resumed.reserve_read_plan(context, "root")
    assert not resumed.reserve_read_plan(context.child(), "worker")
    changed = copy.deepcopy(snapshot)
    cast(dict[str, JsonValue], changed["state"])["read_planned_owners"] = []
    with pytest.raises(ValueError, match="integrity"):
        CroppedResearchLimit(context, state.request, policy=state.policy).restore(
            changed
        )


def test_one_shared_counter_includes_workers_but_final_not_research_and_search_id_once() -> (
    None
):
    _, state = setup()
    state.record_search("first")
    state.record_search("first")
    state.record_search("second")
    state.record_sources(sources(224))
    for _ in range(7):
        state.reserve_model("worker")
    state.reserve_model("root", research=False)
    assert not state.long_tail
    state.reserve_model("root")
    assert not state.long_tail
    state.record_search("third")
    assert state.long_tail
    assert state.progress()["research_model_calls"] == 8
    assert state.progress()["final_model_calls"] == 1
    assert state.progress()["searches"] == 3


@pytest.mark.parametrize(
    "assistant_message_id,calls,searches,originals",
    [
        (5010, 7, 2, 243),
        (5082, 9, 2, 202),
        (5079, 11, 3, 161),
        (4914, 9, 1, 261),
        (4917, 9, 2, 251),
        (4908, 8, 2, 229),
        (4896, 12, 3, 134),
        (4890, 10, 3, 138),
        (5074, 5, 2, 49),
        (5004, 8, 2, 181),
        (4998, 9, 2, 157),
        (4911, 6, 2, 143),
        (4899, 7, 2, 156),
        (4834, 8, 2, 188),
        (4828, 5, 5, 140),
        (4825, 9, 2, 157),
        (4822, 7, 1, 100),
    ],
)
@pytest.mark.parametrize(
    "phase_order", ["calls_first", "sources_first", "searches_first"]
)
def test_successful_observed_run_envelopes_keep_every_prefix_ordinary(
    assistant_message_id: int,
    calls: int,
    searches: int,
    originals: int,
    phase_order: str,
) -> None:
    # Frozen cohort totals bound every monotone prefix; this is operational
    # isolation evidence, not a reconstruction or legal-quality assertion.
    context, state = setup()
    bind_controller(context, state)
    assert assistant_message_id in {
        5010,
        5082,
        5079,
        4914,
        4917,
        4908,
        4896,
        4890,
        5074,
        5004,
        4998,
        4911,
        4899,
        4834,
        4828,
        4825,
        4822,
    }
    rows = sources(originals)
    actions = {
        "calls": [
            lambda: state.reserve_model("root", context=context) for _ in range(calls)
        ],
        "sources": [lambda row=row: state.record_sources([row]) for row in rows],
        "searches": [
            lambda index=index: state.record_search(f"captured-query-{index}")
            for index in range(searches)
        ],
    }
    order = {
        "calls_first": ("calls", "searches", "sources"),
        "sources_first": ("sources", "calls", "searches"),
        "searches_first": ("searches", "sources", "calls"),
    }[phase_order]
    view = HarnessView(
        request="Actual question",
        questions=[],
        facts=[],
        receipts=[],
        evidence=[],
        tools=[],
    )
    for phase in order:
        for action in actions[phase]:
            action()
            assert not state.long_tail
            assert not state.has_delegated_research
            assert not state.budget_limited
            assert state.finish_request(view, context) is None
            assert state.admit_tool(
                "search_corpus", {"query": "ordinary discovery"}
            ).allowed
    assert state.progress()["research_model_calls"] == calls
    assert state.progress()["searches"] == searches
    assert state.progress()["original_records"] == originals


def test_context_free_worker_counter_never_infers_actual_delegation() -> None:
    context, state = setup()
    bind_controller(context, state)
    state.reserve_model("a-worker-shaped-name")
    assert state.progress()["research_model_calls"] == 1
    assert not state.has_delegated_research
    assert (
        cast(dict[str, JsonValue], state.export()["state"])["delegated_research_owners"]
        == []
    )


def test_actual_host_task_entry_is_shared_deduplicated_and_resume_safe() -> None:
    context, state = setup()
    bind_controller(context, state)
    worker = context.child()
    worker.services["task_id"] = "host-assigned-task"
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(
            pool.map(
                lambda _: state.reserve_model("host-assigned-task", context=worker),
                range(16),
            )
        )
    assert state.has_delegated_research
    assert state.progress()["research_model_calls"] == 16
    snapshot = state.export()
    assert snapshot["version"] == 3
    assert cast(dict[str, JsonValue], snapshot["state"])[
        "delegated_research_owners"
    ] == ["host-assigned-task"]
    resumed = CroppedResearchLimit(context, state.request, policy=state.policy)
    bind_controller(context, resumed)
    resumed.restore(snapshot)
    assert resumed.has_delegated_research
    assert resumed.export() == snapshot


@pytest.mark.parametrize(
    "mode",
    ["root", "final", "independent", "default", "cheap", "unowned", "reserved_root"],
)
def test_only_nonindependent_cropped_child_research_can_mark_delegation(
    mode: str,
) -> None:
    context, state = setup()
    bind_controller(context, state)
    invoked = context if mode == "root" else context.child()
    invoked.services["task_id"] = "host-assigned-task"
    if mode == "independent":
        invoked.services["independent_question"] = True
    elif mode == "default":
        invoked.services["asv3_exp_cropped"] = False
    elif mode == "cheap":
        invoked.services["asv3_exp_cheap"] = True
    elif mode == "unowned":
        invoked.services["task_id"] = None
    elif mode == "reserved_root":
        invoked.services["task_id"] = "root"
    owner = "root" if mode in {"unowned", "reserved_root"} else "host-assigned-task"
    state.reserve_model(owner, research=mode != "final", context=invoked)
    assert not state.has_delegated_research
    assert (
        cast(dict[str, JsonValue], state.export()["state"])["delegated_research_owners"]
        == []
    )
    assert state.progress()["final_model_calls"] == (1 if mode == "final" else 0)


@pytest.mark.parametrize(
    "mismatch", ["run", "scope", "controller", "task", "invalid_task"]
)
def test_delegation_rejects_foreign_fence_or_unowned_task_before_accounting(
    mismatch: str,
) -> None:
    context, state = setup()
    bind_controller(context, state)
    worker = context.child()
    worker.services["task_id"] = "host-assigned-task"
    if mismatch == "run":
        worker.run_id = "other-run"
    elif mismatch == "scope":
        worker.scope["owner"] = "other-owner"
    elif mismatch == "controller":
        worker.services["cropped_research_controller"] = CroppedResearchLimit(
            context, state.request, policy=state.policy
        )
    elif mismatch == "task":
        worker.services["task_id"] = "different-task"
    else:
        worker.services["task_id"] = 7
    before = state.export()
    with pytest.raises(ValueError, match="Cropped"):
        state.reserve_model("host-assigned-task", context=worker)
    assert state.export() == before


def test_delegation_property_checks_current_owner_binding_and_scope() -> None:
    context, state = setup()
    bind_controller(context, state)
    worker = context.child()
    worker.services["task_id"] = "actual-task"
    state.reserve_model("actual-task", context=worker)
    assert state.has_delegated_research
    context.services["cropped_research_controller"] = object()
    assert not state.has_delegated_research
    context.services["cropped_research_controller"] = state
    context.scope["owner"] = "changed-owner"
    with pytest.raises(ValueError, match="scope"):
        _ = state.has_delegated_research


@pytest.mark.parametrize(
    "tamper", ["missing_owner", "final_only", "duplicate", "blank", "reserved_root"]
)
def test_delegation_checkpoint_rejects_unbacked_markers_even_with_rehashed_state(
    tamper: str,
) -> None:
    context, state = setup()
    bind_controller(context, state)
    worker = context.child()
    worker.services["task_id"] = "actual-task"
    state.reserve_model("actual-task", context=worker)
    snapshot = state.export()
    saved_state = cast(dict[str, JsonValue], snapshot["state"])
    if tamper == "missing_owner":
        saved_state["owner_calls"] = {}
    elif tamper == "final_only":
        saved_state["owner_calls"] = {"actual-task": {"research": 0, "final": 1}}
    elif tamper == "duplicate":
        saved_state["delegated_research_owners"] = ["actual-task", "actual-task"]
    elif tamper == "reserved_root":
        saved_state["owner_calls"] = {"root": {"research": 1, "final": 0}}
        saved_state["delegated_research_owners"] = ["root"]
    else:
        saved_state["delegated_research_owners"] = [" "]
    checkpoint_hash(snapshot)
    resumed = CroppedResearchLimit(context, state.request, policy=state.policy)
    with pytest.raises(ValueError, match="delegated research owner"):
        resumed.restore(snapshot)


def test_delegation_checkpoint_version_and_unreported_marker_tamper_fail_closed() -> (
    None
):
    context, state = setup()
    bind_controller(context, state)
    state.reserve_model("counter-only-worker")
    snapshot = state.export()
    changed = copy.deepcopy(snapshot)
    changed["version"] = 1
    with pytest.raises(ValueError):
        CroppedResearchLimit(context, state.request, policy=state.policy).restore(
            changed
        )
    changed = copy.deepcopy(snapshot)
    cast(dict[str, JsonValue], changed["state"])["delegated_research_owners"] = [
        "counter-only-worker"
    ]
    with pytest.raises(ValueError, match="integrity"):
        CroppedResearchLimit(context, state.request, policy=state.policy).restore(
            changed
        )


def test_delegation_capacity_refuses_new_owner_without_partial_counter_mutation() -> (
    None
):
    context, state = setup()
    bind_controller(context, state)
    for index in range(64):
        worker = context.child()
        task_id = f"host-task-{index}"
        worker.services["task_id"] = task_id
        state.reserve_model(task_id, context=worker)
    before = state.export()
    worker = context.child()
    worker.services["task_id"] = "host-task-over-capacity"
    with pytest.raises(ValueError, match="delegated research capacity"):
        state.reserve_model("host-task-over-capacity", context=worker)
    assert state.export() == before
    resumed = CroppedResearchLimit(context, state.request, policy=state.policy)
    bind_controller(context, resumed)
    resumed.restore(before)
    assert resumed.has_delegated_research


def delegated_entries(count: int) -> tuple[RunContext, CroppedResearchLimit]:
    context, state = setup()
    bind_controller(context, state)
    worker = context.child()
    worker.services["task_id"] = "actual-task"
    state.reserve_model("actual-task", context=worker)
    for _ in range(count - 1):
        state.reserve_model("root", context=context)
    return context, state


def pre_draft_view() -> HarnessView:
    return HarnessView(
        request="Actual question",
        questions=["Assess the governing conditions and the exception"],
        facts=["The required event and its date remain unverified"],
        receipts=[],
        evidence=[],
        tools=[],
        required_evidence_numbers=[1, 2],
        research_state={"unresolved_condition": "The original is still needed"},
    )


def test_default_volume_and_delegation_signals_do_not_stop_healthy_research() -> None:
    context, state = setup(policy=CroppedResearchPolicy())
    bind_controller(context, state)
    child = context.child()
    child.services["task_id"] = "actual-worker"
    state.reserve_model("actual-worker", context=child)
    for _ in range(12):
        state.reserve_model("root", context=context)
    for index in range(4):
        state.record_search(f"query-{index}")
    rows = sources(300)
    state.record_sources(rows)
    assert state.long_tail and state.has_delegated_research
    assert not state.forced_finish and not state.budget_limited
    before = state.export()
    assert state.finish_request(pre_draft_view(), context) is None
    assert state.admit_tool(
        "search_corpus", {"query": "different material need"}
    ).allowed
    assert state.admit_tool(
        "read_provision", {"source_id": "law", "article": "48"}
    ).allowed
    assert state.admit_tool("spawn_researcher", {}).allowed
    tools: list[dict[str, JsonValue]] = [
        {"function": {"name": name}}
        for name in ("search_corpus", "read_provision", "submit_answer")
    ]
    assert state.available_tools(tools, context) is tools
    assert not state.reserve_read_plan(context, "root")
    assert state.export() == before


def test_default_explicit_finish_uses_available_evidence_without_extra_reads() -> None:
    context, state = setup(policy=CroppedResearchPolicy())
    bind_controller(context, state)
    rows = sources(3)
    state.record_sources(rows)
    state.force_finish("research cost reached ten dollars")
    assert state.forced_finish and state.budget_limited
    calls: tuple[tuple[str, dict[str, JsonValue]], ...] = (
        ("search_corpus", {"query": "new acquisition"}),
        ("read_provision", {"source_id": "law", "article": "48"}),
        ("read_source_range", {"source_id": "law", "start_char": 0, "num_chars": 100}),
        ("resolve_source", {"query": "exact law title"}),
        ("compose_tool_calls", {"program": []}),
        ("discover_tools", {"names": ["read_provision"]}),
    )
    for name, arguments in calls:
        admission = state.admit_tool(
            name, arguments, critical_target_key="required-condition"
        )
        assert not admission.allowed and not admission.absence_proven
    for name in (
        "read_evidence",
        "wait_researcher",
        "update_research",
        "submit_retained_answer",
    ):
        assert state.admit_tool(name, {}).allowed
    view = pre_draft_view()
    view.publication_gap = {
        "missing_conditions": [{"condition_id": "one", "status": "unknown"}]
    }
    snapshot = view.model_dump(mode="json")
    notice = state.finish_request(view, context)
    assert notice is not None and notice["mode"] == "finish_with_available_evidence"
    assert notice["pending_publication_gap"] == view.publication_gap
    assert notice["absence_proven"] is False
    assert view.model_dump(mode="json") == snapshot
    assert state.progress()["original_records"] == 3
    assert state.progress()["critical_reads_reserved"] == 0
    tools: list[dict[str, JsonValue]] = [
        {"function": {"name": name}}
        for name in (
            "discover_tools",
            "read_provision",
            "read_evidence",
            "submit_answer",
        )
    ]
    assert state.available_tools(tools, context) == tools[2:]
    clone = CroppedResearchLimit(context, "Actual question")
    clone.restore(state.export())
    assert clone.forced_finish and clone.export() == state.export()


def test_v13_controller_rejects_old_and_boolean_checkpoint_versions() -> None:
    context, state = setup(policy=CroppedResearchPolicy())
    for version in (2, True):
        snapshot = state.export()
        snapshot["version"] = version
        with pytest.raises(ValueError):
            CroppedResearchLimit(context, "Actual question").restore(snapshot)


@pytest.mark.parametrize("count", [6, 7, 8, 10])
def test_delegated_soft_transition_uses_shared_entries_not_source_closure(
    count: int,
) -> None:
    context, state = delegated_entries(count)
    view = pre_draft_view()
    before = state.export()
    notice = state.finish_request(view, context)
    assert state.budget_limited is (count >= 8)
    assert not state.long_tail
    if count < 8:
        assert notice is None
    else:
        assert notice is not None
        assert notice["mode"] == "fresh_source_writer_transition"
        assert notice["absence_proven"] is False
        assert notice["research_statuses_unchanged"] is True
        assert notice["required_evidence_numbers"] == [1, 2]
        assert "does not close source discovery" in cast(str, notice["instruction"])
    assert state.export() == before
    assert state.progress()["original_records"] == 0
    assert state.progress()["critical_reads_reserved"] == 0


def test_soft_transition_keeps_all_admission_schemas_and_original_obligations() -> None:
    context, state = delegated_entries(8)
    ledger = EvidenceLedger()
    rows = sources(2)
    ledger.add(rows, context)
    state.record_sources(rows)
    view = pre_draft_view()
    view.original_evidence = json.loads(
        ledger.serialize_records([1, 2], max_chars=None)
    )
    tools: list[dict[str, JsonValue]] = [
        {
            "type": "function",
            "function": {"name": name, "parameters": {"type": "object"}},
        }
        for name in [
            "read_source_range",
            "resolve_source",
            "read_provision",
            "search_corpus",
        ]
    ]
    before_view = view.model_dump(mode="json")
    before_ledger = ledger.export()
    before_state = state.export()
    assert state.budget_limited and not state.long_tail
    assert state.available_tools(tools, context) is tools
    operations: list[tuple[str, dict[str, JsonValue]]] = [
        ("read_source_range", {"source_id": "law", "start": 0, "limit": 70}),
        ("resolve_source", {"query": "actual formal source title"}),
        ("read_provision", {"source_id": "law", "article": "9"}),
        ("search_corpus", {"query": "still needed operative scope"}),
    ]
    for name, arguments in operations:
        assert state.admit_tool(name, arguments).allowed
    assert not state.reserve_read_plan(context, "root")
    notice = state.finish_request(view, context)
    assert notice is not None and notice["mode"] == "fresh_source_writer_transition"
    assert view.model_dump(mode="json") == before_view
    assert ledger.export() == before_ledger
    assert state.export() == before_state


@pytest.mark.parametrize(
    "phase", ["worker", "independent", "draft", "gap", "empty_gap"]
)
def test_soft_transition_is_root_predraft_only_and_budget_remains_limited(
    phase: str,
) -> None:
    context, state = delegated_entries(8)
    view = pre_draft_view()
    owner = context
    if phase == "worker":
        owner = context.child()
    elif phase == "independent":
        owner = RunContext(
            run_id=context.run_id,
            scope=context.scope,
            services={**context.services, "independent_question": True},
        )
    elif phase == "draft":
        view.draft_to_repair = "Exact original-dependent qualification [1]."
    elif phase == "gap":
        view.publication_gap = {"kind": "missing_original", "article": "9"}
    else:
        view.publication_gap = {}
    before = view.model_dump(mode="json")
    assert state.finish_request(view, owner) is None
    assert state.budget_limited and not state.long_tail
    assert view.model_dump(mode="json") == before


def test_soft_transition_and_budget_restore_without_a_new_paid_entry_or_marker() -> (
    None
):
    context, state = delegated_entries(8)
    snapshot = state.export()
    resumed = CroppedResearchLimit(context, state.request, policy=state.policy)
    bind_controller(context, resumed)
    resumed.restore(snapshot)
    assert resumed.budget_limited and not resumed.long_tail
    notice = resumed.finish_request(pre_draft_view(), context)
    assert notice is not None and notice["mode"] == "fresh_source_writer_transition"
    assert resumed.export() == snapshot
    resumed.force_finish("Existing joint/physical budget closure")
    assert resumed.long_tail and resumed.budget_limited
    hard_notice = resumed.finish_request(pre_draft_view(), context)
    assert (
        hard_notice is not None
        and hard_notice["mode"] == "finish_with_required_evidence"
    )


def test_final_entries_and_unowned_worker_names_do_not_request_soft_transition() -> (
    None
):
    context, state = delegated_entries(7)
    state.reserve_model("actual-task", research=False, context=context.child())
    assert not state.budget_limited
    assert state.finish_request(pre_draft_view(), context) is None
    context, state = setup()
    bind_controller(context, state)
    for _ in range(8):
        state.reserve_model("worker-shaped-name")
    assert not state.budget_limited
    assert state.finish_request(pre_draft_view(), context) is None


def test_average_search_is_not_a_threshold_hit_and_extra_search_is_joint() -> None:
    _, state = setup()
    assert state.policy.observed_root_decisions == (7, 9)
    assert sum(state.policy.observed_root_decisions) / 2 == 8
    assert state.policy.observed_searches == (2, 2)
    assert sum(state.policy.observed_original_records) / 2 == 222.5
    state.record_sources(sources(224))
    for index in range(3):
        state.record_search(str(index))
    for _ in range(7):
        state.reserve_model("root")
    assert not state.long_tail
    state.reserve_model("worker")
    assert state.long_tail


def test_same_canonical_original_deduplicates_but_versions_and_sources_stay_distinct() -> (
    None
):
    _, state = setup()
    first = sources(1)[0]
    state.record_sources([first, first.model_copy(deep=True)])
    state.record_sources([first])
    state.record_sources(
        [
            EvidenceItem(
                source_id=first.source_id,
                chunk_id=first.chunk_id,
                text="Different exact revision",
            ),
            EvidenceItem(
                source_id="other-source", chunk_id=first.chunk_id, text=first.text
            ),
        ]
    )
    assert state.progress()["original_records"] == 3
    assert state.progress()["original_source_ids"] == 2


@pytest.mark.parametrize(
    "name",
    [
        "search_corpus",
        "search_source_text",
        "spawn_researcher",
        "resolve_source",
        "query_corpus",
    ],
)
def test_broad_tools_cannot_bypass_finish_with_a_claimed_critical_label(
    name: str,
) -> None:
    _, state = long_tail()
    result = state.admit_tool(
        name,
        {"query": "more sources", "source_id": "law", "pattern": "everything"},
        critical_target_key="required-condition",
    )
    assert not result.allowed and result.long_tail and not result.absence_proven


@pytest.mark.parametrize(
    "name,args",
    [
        (
            "read_provision",
            {"source_id": "law", "article": "3", "paragraph": "9", "clause": "a"},
        ),
        ("read_named_provision", {"source_name": "6455 sayılı Kanun", "article": "11"}),
        ("read_chunk", {"source_id": "law", "chunk_id": "original-chunk"}),
        ("read_chunk_context", {"source_id": "law", "chunk_id": "original-chunk"}),
        (
            "follow_reference",
            {
                "source_id": "law",
                "chunk_id": "original-chunk",
                "target_source_id": "referenced-law",
                "article": "4",
            },
        ),
    ],
)
def test_exact_critical_read_allowed_once_without_any_whole_original_clipping(
    name: str, args: dict[str, JsonValue]
) -> None:
    _, state = long_tail()
    assert not state.admit_tool(name, args).allowed
    first = state.admit_tool(
        name, args, critical_target_key="actual-material-condition"
    )
    assert first.allowed
    changed = {
        **args,
        "_public_update": "Different prose",
        "_need_id": "different-bookkeeping",
    }
    repeat = state.admit_tool(name, changed, critical_target_key="a-different-label")
    assert (
        not repeat.allowed
        and first.critical_operation_key == repeat.critical_operation_key
    )
    state.record_sources(sources(53))
    assert state.progress()["original_records"] == 53
    assert state.admit_tool("read_evidence", {"citation": 1}).allowed


def test_critical_complete_article_continuations_are_bounded_globally() -> None:
    _, state = long_tail()
    for start in range(0, 240, 30):
        assert state.admit_tool(
            "read_provision",
            {"source_id": "law", "article": "48", "start": start},
            critical_target_key="one-required-full-context",
        ).allowed
    blocked = state.admit_tool(
        "read_provision",
        {"source_id": "law", "article": "48", "start": 240},
        critical_target_key="one-required-full-context",
    )
    assert not blocked.allowed and not blocked.absence_proven
    assert state.progress()["critical_reads_reserved"] == 8


@pytest.mark.parametrize("start,limit", [(0, 1), (0, 100), (100, 50), (100, 60)])
def test_general_source_pagination_is_not_a_critical_obligation(
    start: int, limit: int
) -> None:
    _, state = long_tail()
    result = state.admit_tool(
        "read_source_range",
        {"source_id": "law", "start": start, "limit": limit},
        critical_target_key="claimed-required-whole-source",
    )
    assert not result.allowed and not result.absence_proven
    assert state.progress()["critical_reads_reserved"] == 0
    outcome = state.blocked_tool(result)
    assert outcome.data["not_read"] is True
    assert outcome.data["absence_proven"] is False
    assert state.admit_tool("read_research_state", {}).allowed


@pytest.mark.parametrize(
    "article,pattern",
    [("61", "MADDE 61"), ("10/A", "madde10/A"), ("GEÇİCİ 1", "GEÇİCİ MADDE 1")],
)
def test_literal_heading_locator_requires_actual_same_source_article_attempt_and_once(
    article: str, pattern: str
) -> None:
    context, state = long_tail()
    state.record_sources(sources(1))
    args: dict[str, JsonValue] = {"source_id": "law", "pattern": pattern}
    assert not state.admit_tool(
        "search_source_text", args, critical_target_key="need"
    ).allowed
    assert state.admit_tool(
        "read_provision",
        {"source_id": "law", "article": article},
        critical_target_key="need",
    ).allowed
    first = state.admit_tool("search_source_text", args, critical_target_key="need")
    assert first.allowed
    repeated = state.admit_tool(
        "search_source_text",
        {**args, "pattern": pattern.lower(), "_need_id": "new-label"},
        critical_target_key="different-prose",
    )
    assert not repeated.allowed
    assert repeated.critical_operation_key == first.critical_operation_key
    assert state.progress()["critical_reads_reserved"] == 2
    restored = CroppedResearchLimit(context, "Actual question", policy=state.policy)
    restored.restore(state.export())
    assert not restored.admit_tool(
        "search_source_text", args, critical_target_key="need"
    ).allowed


@pytest.mark.parametrize(
    "changes",
    [
        {"pattern": "KKDF"},
        {"pattern": "MADDE 61 veya 62"},
        {"pattern": "MADDE 62"},
        {"pattern": "MADDE 61", "mode": "regex"},
        {"pattern": "MADDE 61", "start": 100},
        {"pattern": "MADDE 61", "start": True},
        {"source_id": "unowned-law", "pattern": "MADDE 61"},
    ],
)
def test_source_text_locator_cannot_expand_into_new_targets_or_pagination(
    changes: dict[str, JsonValue],
) -> None:
    _, state = long_tail()
    state.record_sources(sources(1))
    assert state.admit_tool(
        "read_provision",
        {"source_id": "law", "article": "61"},
        critical_target_key="need",
    ).allowed
    result = state.admit_tool(
        "search_source_text",
        {"source_id": "law", **changes},
        critical_target_key="claimed-critical",
    )
    assert not result.allowed and state.progress()["critical_reads_reserved"] == 1


def test_unowned_source_cannot_become_heading_locator_through_reserved_read_only() -> (
    None
):
    _, state = long_tail()
    assert state.admit_tool(
        "read_provision",
        {"source_id": "law", "article": "61"},
        critical_target_key="need",
    ).allowed
    assert not state.admit_tool(
        "search_source_text",
        {"source_id": "law", "pattern": "MADDE 61"},
        critical_target_key="need",
    ).allowed


def test_follow_reference_tail_requires_chosen_exact_target_article() -> None:
    _, state = long_tail()
    cases: list[dict[str, JsonValue]] = [
        {"source_id": "law", "chunk_id": "owned-origin"},
        {"source_id": "law", "chunk_id": "owned-origin", "target_source_id": "target"},
    ]
    for arguments in cases:
        assert not state.admit_tool(
            "follow_reference", arguments, critical_target_key="need"
        ).allowed
    assert state.progress()["critical_reads_reserved"] == 0


def test_shared_controller_accepts_same_run_worker_but_rejects_foreign_context() -> (
    None
):
    context, state = setup()
    state.check_context(context.child())
    with pytest.raises(ValueError, match="run, scope, request or policy"):
        state.check_context(RunContext(run_id="other", scope=context.scope))


def test_parallel_reservations_count_all_models_and_admit_single_critical_operation() -> (
    None
):
    _, state = long_tail()
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(
            pool.map(
                lambda index: state.reserve_model(str(index), research=False), range(40)
            )
        )
        results = list(
            pool.map(
                lambda _: state.admit_tool(
                    "read_provision",
                    {"source_id": "law", "article": "11"},
                    critical_target_key="required-original",
                ),
                range(40),
            )
        )
    assert sum(result.allowed for result in results) == 1
    assert state.progress()["final_model_calls"] == 40


def test_interrupted_critical_reservation_is_not_retried_on_resume() -> None:
    context, state = long_tail()
    args: dict[str, JsonValue] = {"source_id": "law", "article": "11"}
    assert state.admit_tool("read_provision", args, critical_target_key="need").allowed
    state.record_sources(sources(3))
    restored = CroppedResearchLimit(context, "Actual question", policy=state.policy)
    restored.restore(state.export())
    assert restored.progress() == state.progress()
    assert not restored.admit_tool(
        "read_provision", args, critical_target_key="renamed-need"
    ).allowed


@pytest.mark.parametrize("which", ["run", "scope", "request", "policy"])
def test_foreign_checkpoint_is_rejected(which: str) -> None:
    context, state = long_tail()
    context = RunContext(
        run_id="foreign" if which == "run" else context.run_id,
        scope={"foreign": True} if which == "scope" else context.scope,
    )
    restored = CroppedResearchLimit(
        context,
        "Foreign question" if which == "request" else "Actual question",
        policy=CroppedResearchPolicy(research_model_calls=9)
        if which == "policy"
        else None,
    )
    with pytest.raises(ValueError, match="identity or policy"):
        restored.restore(state.export())


def test_checkpoint_counter_or_original_tampering_and_live_replacement_rejected() -> (
    None
):
    context, state = long_tail()
    snapshot = state.export()
    changed = copy.deepcopy(snapshot)
    data = changed["state"]
    assert isinstance(data, dict)
    data["forced_finish_reason"] = "forged"
    restored = CroppedResearchLimit(context, "Actual question", policy=state.policy)
    with pytest.raises(ValueError, match="integrity"):
        restored.restore(changed)
    state.record_search("a-third-query")
    with pytest.raises(ValueError, match="live progress"):
        state.restore(snapshot)


def test_finish_notice_preserves_pending_conditions_and_required_sources_without_closing_them() -> (
    None
):
    context, state = long_tail()
    view = HarnessView(
        request="child assignment",
        questions=["Open condition"],
        facts=[],
        receipts=[],
        evidence=[],
        tools=[],
        required_evidence_numbers=[1, 2],
        publication_gap={
            "missing_conditions": [{"condition_id": "c1", "status": "unknown"}]
        },
    )
    original = view.model_dump(mode="json")
    notice = state.finish_request(view, context.child())
    assert notice is not None
    assert notice["pending_publication_gap"] == view.publication_gap
    assert notice["required_evidence_numbers"] == [1, 2]
    assert notice["absence_proven"] is False
    assert view.model_dump(mode="json") == original
    cast(dict[str, JsonValue], notice["pending_publication_gap"])[
        "missing_conditions"
    ] = []
    assert view.publication_gap != notice["pending_publication_gap"]


def test_cost_gate_can_force_finish_without_fabricated_source_or_research_completion() -> (
    None
):
    context, state = setup()
    state.force_finish("Research cost reserve exhausted before provider call")
    assert state.long_tail
    assert state.progress()["original_records"] == 0
    assert not state.admit_tool("spawn_researcher", {}).allowed
    state.reserve_model("root", research=False)
    assert state.progress()["final_model_calls"] == 1
    view = HarnessView(
        request="Actual question",
        questions=[],
        facts=[],
        receipts=[],
        evidence=[],
        tools=[],
    )
    assert state.finish_request(view, context) is not None


def failed_named_read(
    source_name: str = "3065 sayılı Katma Değer Vergisi Kanunu",
    *,
    article: str = "21",
    call_id: str = "actual-read-call",
) -> dict[str, JsonValue]:
    return {
        "owner_task_id": "actual-worker",
        "call_id": call_id,
        "name": "read_named_provision",
        "arguments": {"source_name": source_name, "article": article},
        "status": "not_found",
        "data": {"sources": [], "has_more": False, "absence_proven": False},
        "summary": "Actual unresolved source identity; absence is not proven.",
    }


def test_failed_named_source_exposes_resolution_only_until_one_shared_reservation() -> (
    None
):
    context, state = long_tail()
    tools: list[dict[str, JsonValue]] = [
        {"function": {"name": "resolve_source"}},
        {"function": {"name": "read_provision"}},
    ]
    assert state.available_tools(tools, context) == tools[1:]
    assert state.register_failed_named_reads([failed_named_read()], context) == 1
    assert state.available_tools(tools, context) == tools
    preview = state.admit_tool("resolve_source", {"query": "3065"}, reserve=False)
    assert preview.allowed and state.progress()["critical_reads_reserved"] == 0
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(
                lambda query: state.admit_tool("resolve_source", {"query": query}),
                ["3065", "3065 sayılı Katma Değer Vergisi Kanunu"],
            )
        )
    assert sum(row.allowed for row in results) == 1
    assert len({row.critical_operation_key for row in results}) == 1
    assert state.progress()["critical_reads_reserved"] == 1
    assert state.available_tools(tools, context) == tools[1:]
    assert not state.admit_tool(
        "resolve_source",
        {"query": "3065", "limit": 100},
        critical_target_key="renamed-topic",
    ).allowed
    admission = state.admit_tool("resolve_source", {"query": "3065"})
    assert not state.blocked_tool(admission).data["absence_proven"]


@pytest.mark.parametrize(
    "query",
    [
        "Katma Değer",
        "Kanun",
        "KDV",
        "3065 sayılı Kanun",
        "3065 madde21",
        "6183",
        "03065",
        "3065 veya6183",
    ],
)
def test_registered_source_does_not_authorize_partial_topics_or_other_numbers(
    query: str,
) -> None:
    context, state = long_tail()
    state.register_failed_named_reads([failed_named_read()], context)
    assert not state.admit_tool(
        "resolve_source", {"query": query}, critical_target_key="claimed-source-gap"
    ).allowed
    assert state.progress()["critical_reads_reserved"] == 0


@pytest.mark.parametrize(
    "changes",
    [
        {"offset": 20},
        {"offset": True},
        {"offset": -1},
        {"limit": 0},
        {"limit": 101},
        {"limit": True},
    ],
)
def test_source_resolution_cannot_start_pagination_or_invalid_page_size(
    changes: dict[str, JsonValue],
) -> None:
    context, state = long_tail()
    state.register_failed_named_reads([failed_named_read()], context)
    assert not state.admit_tool("resolve_source", {"query": "3065", **changes}).allowed


def test_exact_normalized_title_is_allowed_without_inventing_a_teblig_number_alias() -> (
    None
):
    context, state = long_tail()
    name = "Kaynak Kullanımını Destekleme Fonu Hakkında6 Sıra No.lu Tebliğ"
    state.register_failed_named_reads([failed_named_read(name, article="2")], context)
    assert not state.admit_tool("resolve_source", {"query": "6"}).allowed
    assert not state.admit_tool("resolve_source", {"query": "Kaynak Kullanımı"}).allowed
    assert state.admit_tool(
        "resolve_source",
        {"query": "KAYNAK KULLANIMINI DESTEKLEME FONU HAKKINDA6 SIRA NO LU TEBLIG"},
    ).allowed


def test_cited_law_number_in_a_regulation_title_is_not_its_number_alias() -> None:
    context, state = long_tail()
    name = "6183 sayılı Kanuna Göre Tahsil Yönetmeliği"
    state.register_failed_named_reads([failed_named_read(name)], context)
    assert not state.admit_tool("resolve_source", {"query": "6183"}).allowed
    assert state.admit_tool("resolve_source", {"query": name}).allowed


def test_failed_named_articles_and_aliases_share_one_source_resolution_identity() -> (
    None
):
    context, state = long_tail()
    first = failed_named_read(
        "6183 sayılı Amme Alacaklarının Tahsil Usulü Hakkında Kanun", article="48"
    )
    second = failed_named_read(
        "6183 sayılı Kanun", article="51", call_id="another-read"
    )
    second["parent_call_id"] = "same-real-compose-call"
    second["step_id"] = "second-step"
    assert state.register_failed_named_reads([first, second], context.child()) == 1
    admitted = state.admit_tool("resolve_source", {"query": "6183"})
    assert admitted.allowed
    assert not state.admit_tool(
        "resolve_source", {"query": "6183 sayılı Kanun"}
    ).allowed
    assert state.progress()["failed_named_source_targets"] == 1
    state.register_failed_named_reads([second], context)
    first_args = cast(dict[str, JsonValue], first["arguments"])
    assert not state.admit_tool(
        "resolve_source", {"query": first_args["source_name"]}
    ).allowed


@pytest.mark.parametrize(
    "change",
    [
        {"name": "resolve_source"},
        {"status": "found"},
        {"status": "partial"},
        {"call_id": ""},
        {"arguments": {"source_name": "Kanun", "article": "21"}},
        {"arguments": {"source_name": "3065 sayılı Kanun", "article": "all"}},
        {"data": {}},
        {"data": {"sources": [], "has_more": False, "absence_proven": True}},
        {
            "data": {
                "sources": [],
                "has_more": False,
                "absence_proven": False,
                "not_read": True,
            }
        },
        {
            "data": {
                "sources": [],
                "has_more": False,
                "absence_proven": False,
                "blocked_by_research_limit": True,
            }
        },
        {"data": {"source_id": "known-law", "article": "21", "absence_proven": False}},
        {"evidence_ids": [1]},
        {"original_reads": [{"citation": 1}]},
    ],
)
def test_only_actual_source_identity_failures_can_register(
    change: dict[str, JsonValue],
) -> None:
    context, state = long_tail()
    row = {**failed_named_read(), **change}
    assert (
        state.register_failed_named_reads([row, "model-proposed-title"], context) == 0
    )
    assert not state.admit_tool("resolve_source", {"query": "3065"}).allowed


def test_ambiguous_source_identity_remains_a_bounded_resolution_not_approval() -> None:
    context, state = long_tail()
    row = failed_named_read()
    row.update(
        status="ambiguous",
        data={
            "sources": [
                {"source_id": "first", "name": "First version"},
                {"source_id": "second", "name": "Second version"},
            ],
            "has_more": False,
            "absence_proven": False,
        },
    )
    assert state.register_failed_named_reads([row], context) == 1
    assert state.admit_tool("resolve_source", {"query": "3065"}).allowed
    assert state.progress()["original_records"] == 0


def test_source_resolution_uses_global_critical_allowance_and_is_resume_safe() -> None:
    context, state = long_tail()
    state.register_failed_named_reads([failed_named_read()], context)
    state.reserve_read_plan(context, "root")
    assert state.read_plan_reserved(context, "root")
    assert not state.read_plan_reserved(context, "different-owner")
    assert state.admit_tool("resolve_source", {"query": "3065"}).allowed
    saved = state.export()
    resumed = CroppedResearchLimit(context, state.request, policy=state.policy)
    resumed.restore(saved)
    assert resumed.export() == saved
    assert resumed.read_plan_reserved(context, "root")
    assert not resumed.admit_tool("resolve_source", {"query": "3065"}).allowed
    assert not resumed.admit_tool(
        "resolve_source", {"query": "3065 sayılı Katma Değer Vergisi Kanunu"}
    ).allowed
    for start in range(7):
        assert resumed.admit_tool(
            "read_provision",
            {"source_id": "law", "article": "21", "start": start},
            critical_target_key="actual-condition",
        ).allowed
    resumed.register_failed_named_reads(
        [failed_named_read("6183 sayılı Kanun", call_id="second-source")], context
    )
    assert not resumed.admit_tool("resolve_source", {"query": "6183"}).allowed
    assert resumed.progress()["critical_reads_reserved"] == 8


def test_registration_and_read_plan_getter_reject_foreign_context_and_checkpoint_tampering() -> (
    None
):
    context, state = long_tail()
    foreign = RunContext(run_id="old-m11-run", scope=context.scope)
    with pytest.raises(ValueError, match="run, scope"):
        state.register_failed_named_reads([failed_named_read()], foreign)
    with pytest.raises(ValueError, match="run, scope"):
        state.read_plan_reserved(foreign, "root")
    state.register_failed_named_reads([failed_named_read()], context)
    saved = state.export()
    changed = copy.deepcopy(saved)
    data = cast(dict[str, JsonValue], changed["state"])
    cast(dict[str, JsonValue], data["failed_named_sources"]).clear()
    with pytest.raises(ValueError, match="integrity"):
        CroppedResearchLimit(context, state.request, policy=state.policy).restore(
            changed
        )


def test_ordinary_research_tools_and_admission_remain_unchanged() -> None:
    context, state = setup()
    tools: list[dict[str, JsonValue]] = [{"function": {"name": "resolve_source"}}]
    before = state.export()
    assert state.available_tools(tools, context) is tools
    assert state.admit_tool(
        "resolve_source", {"query": "arbitrary ordinary query", "offset": 20}
    ).allowed
    assert state.export() == before


def alias_fixture() -> tuple[RunContext, CroppedResearchLimit, dict[str, JsonValue]]:
    context, state = long_tail()
    name = "Amme Alacaklarının Tahsil Usulü Hakkında Kanun"
    state.register_failed_named_reads([failed_named_read(name, article="48")], context)
    ledger = EvidenceLedger()
    context.services["evidence"] = ledger
    quote = "6183 sayılı Amme Alacaklarının Tahsil Usulü Hakkında Kanunun 48 inci maddesine göre tecil faizi alınır."
    citation = ledger.add(
        [
            EvidenceItem(
                source_id="actual-cross-reference",
                chunk_id="actual-original",
                text=quote,
            )
        ],
        context,
    )[0]
    return (
        context,
        state,
        {
            "target": {"instrument_number": "6183", "article": "48"},
            "exact_witnesses": [{"citation": citation, "source_quote": quote}],
        },
    )


def test_same_run_exact_original_witness_adds_one_numeric_source_alias() -> None:
    context, state, reference = alias_fixture()
    assert not state.admit_tool("resolve_source", {"query": "6183"}).allowed
    assert state.register_failed_named_source_aliases([reference], context) == 1
    assert state.register_failed_named_source_aliases([reference], context) == 0
    assert state.progress()["critical_reads_reserved"] == 0
    assert state.admit_tool("resolve_source", {"query": "6183"}).allowed
    assert not state.admit_tool(
        "resolve_source", {"query": "Amme Alacaklarının Tahsil Usulü Hakkında Kanun"}
    ).allowed
    assert not state.admit_tool("resolve_source", {"query": "6455"}).allowed
    assert state.progress()["critical_reads_reserved"] == 1


def test_witness_alias_does_not_reopen_prior_title_reservation_and_survives_resume() -> (
    None
):
    context, state, reference = alias_fixture()
    title: dict[str, JsonValue] = {
        "query": "Amme Alacaklarının Tahsil Usulü Hakkında Kanun"
    }
    assert state.admit_tool("resolve_source", title).allowed
    assert state.register_failed_named_source_aliases([reference], context) == 1
    assert not state.admit_tool("resolve_source", {"query": "6183"}).allowed
    snapshot = state.export()
    restored = CroppedResearchLimit(context, state.request, policy=state.policy)
    restored.restore(snapshot)
    assert restored.export() == snapshot
    assert not restored.admit_tool("resolve_source", {"query": "6183"}).allowed
    assert not restored.admit_tool("resolve_source", title).allowed


def test_same_witnessed_instrument_links_already_registered_source_name_variants() -> (
    None
):
    context, state, reference = alias_fixture()
    numbered = "6183 sayılı Amme Alacaklarının Tahsil Usulü Hakkında Kanun"
    state.register_failed_named_reads(
        [failed_named_read(numbered, article="51")], context
    )
    state.register_failed_named_source_aliases([reference], context)
    first = state.admit_tool("resolve_source", {"query": "6183"})
    second = state.admit_tool("resolve_source", {"query": numbered})
    assert first.allowed and not second.allowed
    assert first.critical_operation_key == second.critical_operation_key
    assert state.progress()["critical_reads_reserved"] == 1


@pytest.mark.parametrize(
    "changes",
    [
        {"target": {"instrument_number": "3065", "article": "48"}},
        {"target": {"instrument_number": "6183", "article": "51"}},
        {"target": {"instrument_number": "6183", "article": "48", "qualifier": "EK"}},
        {
            "target": {
                "instrument_number": "6183",
                "article": "48",
                "source_name": "Katma Değer Vergisi Kanunu",
            }
        },
        {
            "exact_witnesses": [
                {
                    "citation": 999,
                    "source_quote": "6183 sayılı Amme Alacaklarının Tahsil Usulü Hakkında Kanun",
                }
            ]
        },
        {
            "exact_witnesses": [
                {
                    "citation": 1,
                    "source_quote": "6183 sayılı Amme Alacaklarının Tahsil Usulü Hakkında Kanun yabancı metin",
                }
            ]
        },
        {
            "exact_witnesses": [
                {
                    "citation": True,
                    "source_quote": "6183 sayılı Amme Alacaklarının Tahsil Usulü Hakkında Kanun",
                }
            ]
        },
    ],
)
def test_unrelated_article_number_title_or_unavailable_literal_cannot_add_alias(
    changes: dict[str, JsonValue],
) -> None:
    context, state, reference = alias_fixture()
    assert (
        state.register_failed_named_source_aliases([{**reference, **changes}], context)
        == 0
    )
    assert not state.admit_tool("resolve_source", {"query": "6183"}).allowed


@pytest.mark.parametrize(
    "quote",
    [
        "6183 sayılı Tahsil Usulü Hakkında Kanunun48 inci maddesi",
        "6183 başka sayılı Amme Alacaklarının Tahsil Usulü Hakkında Kanunun48 inci maddesi",
        "3065 sayılı Amme Alacaklarının Tahsil Usulü Hakkında Kanunun48 inci maddesi",
        "6183 sayılı Amme Alacaklarının Tahsil Usulü Hakkında Kanunlarımız uygulanır",
    ],
)
def test_contiguous_original_still_requires_complete_title_and_adjacent_own_number(
    quote: str,
) -> None:
    context, state, reference = alias_fixture()
    ledger = cast(EvidenceLedger, context.services["evidence"])
    citation = ledger.add(
        [EvidenceItem(source_id="another-original", text=quote)], context
    )[0]
    reference["exact_witnesses"] = [{"citation": citation, "source_quote": quote}]
    assert state.register_failed_named_source_aliases([reference], context) == 0


def test_witnessed_alias_requires_same_run_ledger_scope_and_a_registered_failure() -> (
    None
):
    context, state, reference = alias_fixture()
    foreign = RunContext(run_id="foreign", scope=context.scope)
    with pytest.raises(ValueError, match="run, scope"):
        state.register_failed_named_source_aliases([reference], foreign)
    changed = context.child()
    changed.services["evidence"] = EvidenceLedger()
    with pytest.raises(ValueError, match="actual evidence"):
        state.register_failed_named_source_aliases([reference], changed)
    unrelated = CroppedResearchLimit(context, state.request, policy=state.policy)
    unrelated.force_finish("actual tail")
    assert unrelated.register_failed_named_source_aliases([reference], context) == 0
    assert not unrelated.admit_tool("resolve_source", {"query": "6183"}).allowed


def test_checkpoint_alias_tampering_and_missing_current_original_cannot_authorize_resolution() -> (
    None
):
    context, state, reference = alias_fixture()
    state.register_failed_named_source_aliases([reference], context)
    snapshot = state.export()
    changed = copy.deepcopy(snapshot)
    rows = cast(
        dict[str, JsonValue],
        cast(dict[str, JsonValue], changed["state"])["failed_named_sources"],
    )
    row = cast(dict[str, JsonValue], next(iter(rows.values())))
    cast(dict[str, JsonValue], cast(list[JsonValue], row["witnessed_aliases"])[0])[
        "number"
    ] = "6455"
    with pytest.raises(ValueError):
        CroppedResearchLimit(context, state.request, policy=state.policy).restore(
            changed
        )
    restored = CroppedResearchLimit(context, state.request, policy=state.policy)
    restored.restore(snapshot)
    context.services["evidence"] = EvidenceLedger()
    assert not restored.admit_tool("resolve_source", {"query": "6183"}).allowed
