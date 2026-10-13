"""Cropped repair identity, exactly-once reservation and checkpoint isolation."""

import copy
from concurrent.futures import ThreadPoolExecutor
from typing import cast

import pytest
from pydantic import JsonValue, ValidationError

from onyx.asv3.assertions import assertion_inventory
from onyx.asv3.cropped_repair_state import (
    CroppedRepairState,
    CroppedRepairTarget,
    RepairStatus,
    extract_repair_targets,
)
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import EvidenceItem, RunContext


def setup() -> tuple[RunContext, EvidenceLedger, CroppedRepairState]:
    context = RunContext(run_id="owned-run", scope={"authorized": "owned"})
    ledger = EvidenceLedger()
    ledger.add(
        [
            EvidenceItem(
                source_id="owned-law",
                chunk_id="article-11",
                text="MADDE 11 - Özgün metin.",
            )
        ],
        context,
    )
    return context, ledger, CroppedRepairState(context, "Original owned question")


def governing_gap(
    draft: str, field: str = "named_authority_gaps"
) -> dict[str, JsonValue]:
    unit_id = assertion_inventory(draft)[0]["unit_id"]
    return {
        field: [
            {
                "unit_id": unit_id,
                "instrument_number": "6455",
                "article": "11",
                "qualifier": None,
                "reference_text": "6455 sayılı Kanunun 11. maddesi",
                "matching_original_evidence": [],
            }
        ]
    }


def test_governing_defect_survives_wording_unit_and_gate_changes() -> None:
    _, ledger, state = setup()
    first = "6455 sayılı Kanunun 11. maddesi kendi aslından incelenemedi."
    second = "6455 sayılı Kanunun 11. maddesinin özgün metnine erişilemedi; sonuç ileri sürülmüyor."
    target = extract_repair_targets(governing_gap(first), first, ledger)[0]
    assert state.reserve(target)
    next_target = extract_repair_targets(
        governing_gap(second, "unread_cited_statute_references"), second, ledger
    )[0]
    assert next_target.key == target.key
    assert next_target.unit_ids != target.unit_ids
    assert not state.reserve(next_target)
    row = state.records_for([next_target])[0]
    assert row["status"] == "in_progress"
    assert row["target"] == next_target.checkpoint_value()
    assert next_target.source_name == "6455 sayılı Kanun"


def test_both_governing_gap_categories_merge_but_keep_current_units_and_raw_rows() -> (
    None
):
    _, ledger, _ = setup()
    draft = "6455 sayılı Kanunun 11. maddesi bilinmiyor.\n\nBaşka bir uygulanabilirlik sorusu açık."
    units = assertion_inventory(draft)
    gap = governing_gap(draft)
    gap["unread_cited_statute_references"] = [
        {
            "instrument_number": "6455",
            "article": "11",
            "answer_unit_ids": [units[-1]["unit_id"]],
            "origin_citations": [42],
        }
    ]
    targets = extract_repair_targets(gap, draft, ledger)
    assert len(targets) == 1
    assert targets[0].unit_ids == [units[0]["unit_id"], units[-1]["unit_id"]]
    entries = targets[0].details["gap_entries"]
    assert isinstance(entries, list) and len(entries) == 2


@pytest.mark.parametrize(
    "updates",
    [
        {"article": "12"},
        {"qualifier": "geçici"},
        {"instrument_number": "4458"},
        {"check_kind": "related_source"},
    ],
)
def test_distinct_material_source_targets_have_distinct_keys(
    updates: dict[str, str],
) -> None:
    target = CroppedRepairTarget(
        check_kind="governing_original", instrument_number="6455", article="11"
    )
    assert target.model_copy(update=updates).key != target.key


def test_bookkeeping_and_known_source_resolution_do_not_authorize_new_attempt() -> None:
    target = CroppedRepairTarget(
        check_kind="governing_original", instrument_number="6455", article="11"
    )
    assert (
        target.model_copy(
            update={
                "source_id": "newly-resolved-source",
                "anchor_source_id": "different-lower-original",
                "unit_ids": ["new-unit"],
                "evidence_numbers": [33],
                "text_hash": "new-delivery-hash",
                "reference_text": "Reworded reference",
                "formal_name": "A resolved display name",
                "details": {"revision": 50},
            }
        ).key
        == target.key
    )


def test_article_qualifier_case_and_letter_punctuation_normalize() -> None:
    first = CroppedRepairTarget(
        check_kind="governing_original",
        instrument_number="6455",
        article="03/a",
        qualifier="GEÇİCİ",
    )
    second = CroppedRepairTarget(
        check_kind="governing_original",
        instrument_number="6455",
        article="3/A",
        qualifier="gecici",
    )
    assert first.key == second.key


def test_related_sources_keep_source_and_anchor_identity() -> None:
    target = CroppedRepairTarget(
        check_kind="related_source",
        source_id="decision-one",
        anchor_source_id="law",
        article="11",
    )
    assert target.model_copy(update={"source_id": "decision-two"}).key != target.key
    assert target.model_copy(update={"anchor_source_id": "other-law"}).key != target.key
    assert (
        target.model_copy(update={"requirement_id": "new-lead-label"}).key == target.key
    )


def test_delivery_uses_canonical_original_not_renumbered_citation_or_hash() -> None:
    context, ledger, _ = setup()
    other = EvidenceLedger()
    item = ledger.get(1)
    assert item is not None
    other.add([EvidenceItem(source_id="padding", text="Unrelated")], context)
    other.add([item], context)
    first = extract_repair_targets(
        {"undelivered_citations": [1]}, "Original [1].", ledger
    )[0]
    second = extract_repair_targets(
        {"undelivered_citations": [2]}, "Original [2].", other
    )[0]
    assert first.key == second.key
    assert first.text_hash == item.text_hash
    assert first.evidence_numbers == [1] and second.evidence_numbers == [2]
    assert (
        first.model_copy(update={"text_hash": "changed-observation"}).key == first.key
    )
    assert first.model_copy(update={"chunk_id": "article-12"}).key != first.key


def test_unknown_delivery_identity_does_not_restart_for_invented_numbers() -> None:
    _, ledger, _ = setup()
    first = extract_repair_targets({"undelivered_citations": [80]}, "", ledger)[0]
    second = extract_repair_targets({"undelivered_citations": [81]}, "", ledger)[0]
    assert first.key == second.key
    assert first.source_id is None


def test_generic_fallback_omits_draft_and_defect_prose_but_preserves_requirement_id() -> (
    None
):
    _, ledger, _ = setup()
    first = extract_repair_targets(
        {"requirement_id": "condition_1", "omitted_condition": "Long text"},
        "First draft",
        ledger,
    )[0]
    second = extract_repair_targets(
        {
            "requirement_id": "condition_1",
            "omitted_condition": "Reworded text",
            "instruction": "New instructions",
        },
        "Changed draft",
        ledger,
    )[0]
    third = extract_repair_targets(
        {"requirement_id": "condition_2", "omitted_condition": "Long text"},
        "First draft",
        ledger,
    )[0]
    assert first.key == second.key
    assert first.key != third.key
    assert not extract_repair_targets({}, "", ledger)


def test_only_one_parallel_reservation_wins() -> None:
    _, _, state = setup()
    target = CroppedRepairTarget(
        check_kind="governing_original", instrument_number="6455", article="11"
    )
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: state.reserve(target), range(64)))
    assert sum(results) == 1


@pytest.mark.parametrize("status", ["found", "not_found", "failed"])
def test_completed_attempt_never_restarts_and_is_not_absence_proof(status: str) -> None:
    _, _, state = setup()
    target = CroppedRepairTarget(
        check_kind="governing_original", instrument_number="6455", article="11"
    )
    assert state.reserve(target)
    state.finish(
        target.key,
        cast(RepairStatus, status),
        [1] if status == "found" else [],
        "Only this attempt was completed.",
    )
    assert not state.reserve(target)
    row = state.records_for([target])[0]
    assert row["status"] == status
    assert row["absence_proven"] is False
    with pytest.raises(ValueError, match="finish only once"):
        state.finish(target.key, "found", [1], "Cannot overwrite")


def test_in_progress_resume_does_not_restart_or_reset_live_attempts() -> None:
    context, _, state = setup()
    target = CroppedRepairTarget(
        check_kind="governing_original", instrument_number="6455", article="11"
    )
    assert state.reserve(target)
    snapshot = state.export()
    resumed = CroppedRepairState(context, "Original owned question")
    resumed.restore(snapshot)
    assert resumed.export() == snapshot
    assert not resumed.reserve(target)
    resumed.finish(
        target.key,
        "not_found",
        [],
        "Interrupted before source delivery; absence unproven.",
    )
    with pytest.raises(ValueError, match="replace live"):
        resumed.restore(snapshot)


@pytest.mark.parametrize("foreign", ["run", "scope", "request"])
def test_foreign_checkpoint_rejected(foreign: str) -> None:
    context, _, state = setup()
    snapshot = state.export()
    other_context = RunContext(
        run_id="other" if foreign == "run" else context.run_id,
        scope={"authorized": "other"} if foreign == "scope" else context.scope,
    )
    other = CroppedRepairState(
        other_context,
        "Other question" if foreign == "request" else "Original owned question",
    )
    with pytest.raises(ValueError, match="identity or request"):
        other.restore(snapshot)


def test_checkpoint_tampering_duplicate_and_unrelated_integrity_rejected() -> None:
    context, _, state = setup()
    target = CroppedRepairTarget(
        check_kind="governing_original", instrument_number="6455", article="11"
    )
    state.reserve(target)
    snapshot = state.export()
    records = snapshot["records"]
    assert isinstance(records, list) and isinstance(records[0], dict)
    corrupted = copy.deepcopy(snapshot)
    corrupted_records = corrupted["records"]
    assert isinstance(corrupted_records, list) and isinstance(
        corrupted_records[0], dict
    )
    corrupted_records[0]["detail"] = "Forged result"
    corrupted_records[0]["status"] = "not_found"
    with pytest.raises(ValueError, match="integrity"):
        CroppedRepairState(context, "Original owned question").restore(corrupted)
    duplicated = {**snapshot, "records": [*records, *records]}
    with pytest.raises(ValueError, match="integrity"):
        CroppedRepairState(context, "Original owned question").restore(duplicated)
    extra = copy.deepcopy(snapshot)
    integrity = extra["record_integrity"]
    assert isinstance(integrity, dict)
    integrity["unrelated"] = "unused"
    with pytest.raises(ValueError, match="unrelated"):
        CroppedRepairState(context, "Original owned question").restore(extra)


def test_reservation_and_export_detach_mutable_observations() -> None:
    _, _, state = setup()
    target = CroppedRepairTarget(
        check_kind="generic", details={"example": ["original"]}
    )
    state.reserve(target)
    target.details["example"] = "mutated caller"
    snapshot = state.export()
    records = snapshot["records"]
    assert isinstance(records, list) and isinstance(records[0], dict)
    saved_target = records[0]["target"]
    assert isinstance(saved_target, dict) and saved_target["details"] == {
        "example": ["original"]
    }
    records.clear()
    assert state.export()["records"]


@pytest.mark.parametrize(
    "gap",
    [
        {"named_authority_gaps": "bad"},
        {"unread_related_sources": [None]},
        {"undelivered_citations": [True]},
        {"undelivered_citations": [0]},
    ],
)
def test_malformed_target_inputs_rejected(gap: dict[str, JsonValue]) -> None:
    _, ledger, _ = setup()
    with pytest.raises(ValueError):
        extract_repair_targets(gap, "", ledger)


def test_finish_requires_reservation_positive_evidence_and_valid_result() -> None:
    _, _, state = setup()
    target = CroppedRepairTarget(check_kind="generic")
    with pytest.raises(ValueError, match="reserved"):
        state.finish(target.key, "found", [1], "")
    state.reserve(target)
    cases: list[tuple[RepairStatus, list[int], str]] = [
        ("found", [], ""),
        ("not_found", [], ""),
        ("incomplete", [], ""),
        ("found", cast(list[int], [True]), ""),
        ("found", [1, 1], ""),
    ]
    for status, numbers, detail in cases:
        with pytest.raises((ValueError, ValidationError)):
            state.finish(target.key, status, numbers, detail)
    assert state.records_for([target])[0]["status"] == "in_progress"


def test_incomplete_original_check_persists_without_acquisition_or_integration_retry() -> (
    None
):
    context, _, state = setup()
    target = CroppedRepairTarget(
        check_kind="governing_original", instrument_number="6455", article="11"
    )
    assert state.reserve(target)
    detail = "The original was delivered, but its applicability remains unverified."
    state.finish(target.key, "incomplete", [], detail)
    record = state.records_for([target])[0]
    assert record["status"] == "incomplete"
    assert record["evidence_numbers"] == []
    assert record["detail"] == detail
    assert record["absence_proven"] is False
    assert state.reserve_integration([target], "coordinator")

    resumed = CroppedRepairState(context, state.request)
    resumed.restore(state.export())
    assert resumed.records_for([target]) == [record]
    assert resumed.close_interrupted_attempts() == 0
    assert not resumed.reserve(target)
    assert not resumed.reserve_integration([target], "coordinator")
    assert resumed.reserve_integration([target], "child-1")
    assert not resumed.reserve_integration([target], "child-1")


def test_changed_live_scope_is_rejected() -> None:
    context, _, state = setup()
    context.scope["authorized"] = "foreign"
    with pytest.raises(ValueError, match="scope changed"):
        state.export()


def test_integration_is_once_per_target_and_owner_and_persists_before_completion() -> (
    None
):
    context, _, state = setup()
    target = CroppedRepairTarget(
        check_kind="governing_original", instrument_number="6455", article="11"
    )
    state.reserve(target)
    state.finish(
        target.key, "not_found", [], "Acquisition could not obtain the original."
    )
    assert state.request == "Original owned question"
    assert state.reserve_integration([target], "child-1")
    assert not state.reserve_integration([target], "child-1")
    assert state.reserve_integration([target], "coordinator")
    resumed = CroppedRepairState(context, state.request)
    resumed.restore(state.export())
    assert not resumed.reserve_integration([target], "child-1")
    assert not resumed.reserve_integration([target], "coordinator")
    assert resumed.reserve_integration([target], "child-2")
    assert not resumed.reserve(target)


def test_overlapping_integration_batch_is_atomic() -> None:
    _, _, state = setup()
    first = CroppedRepairTarget(
        check_kind="governing_original", instrument_number="6455", article="11"
    )
    second = first.model_copy(update={"article": "12"})
    state.reserve(first)
    state.reserve(second)
    state.finish(
        first.key, "not_found", [], "This attempt did not obtain its original."
    )
    state.finish(
        second.key, "not_found", [], "This attempt did not obtain its original."
    )
    assert state.reserve_integration([first], "coordinator")
    assert not state.reserve_integration([first, second], "coordinator")
    assert state.reserve_integration([second], "coordinator")
    with pytest.raises(ValueError, match="reserved source"):
        state.reserve_integration(
            [second.model_copy(update={"article": "13"})], "coordinator"
        )
    with pytest.raises(ValueError, match="bounded identity"):
        state.reserve_integration([first], " ")


def test_parallel_integration_only_one_wins() -> None:
    _, _, state = setup()
    target = CroppedRepairTarget(check_kind="generic")
    state.reserve(target)
    state.finish(
        target.key, "not_found", [], "This attempt did not obtain its original."
    )
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(
            pool.map(
                lambda _: state.reserve_integration([target], "coordinator"), range(32)
            )
        )
    assert sum(results) == 1


def test_integration_checkpoint_tampering_rejected() -> None:
    context, _, state = setup()
    target = CroppedRepairTarget(check_kind="generic")
    state.reserve(target)
    state.finish(
        target.key, "not_found", [], "This attempt did not obtain its original."
    )
    state.reserve_integration([target], "coordinator")
    snapshot = state.export()
    integrations = snapshot["integrations"]
    assert isinstance(integrations, dict)
    integrations["coordinator"] = []
    with pytest.raises(ValueError, match="integration checkpoint integrity"):
        CroppedRepairState(context, state.request).restore(snapshot)


def test_matching_originals_are_candidates_but_inline_lower_sources_are_not() -> None:
    _, ledger, _ = setup()
    draft = "6455 sayılı Kanunun 11. maddesi açık."
    gap = governing_gap(draft)
    rows = gap["named_authority_gaps"]
    assert isinstance(rows, list) and isinstance(rows[0], dict)
    rows[0]["matching_original_evidence"] = [1, 999]
    rows[0]["inline_evidence"] = [42]
    target = extract_repair_targets(gap, draft, ledger)[0]
    assert target.evidence_numbers == [1]


def test_related_candidates_must_belong_to_requested_source() -> None:
    _, ledger, _ = setup()
    target = extract_repair_targets(
        {
            "unread_related_sources": [
                {
                    "source_id": "other-law",
                    "available_original_citations": [1],
                }
            ]
        },
        "",
        ledger,
    )[0]
    assert not target.evidence_numbers
    target = extract_repair_targets(
        {
            "unread_related_sources": [
                {
                    "source_id": "owned-law",
                    "available_original_citations": [1],
                }
            ]
        },
        "",
        ledger,
    )[0]
    assert target.evidence_numbers == [1]


def test_generic_conditions_survive_unit_churn_and_distinguish_same_source_obligations() -> (
    None
):
    _, ledger, _ = setup()
    first = extract_repair_targets(
        {
            "condition_id": "condition_1",
            "source_id": "owned-law",
            "unit_id": "au-old",
            "omitted_condition": "Old prose",
        },
        "",
        ledger,
    )[0]
    second = extract_repair_targets(
        {
            "condition_id": "condition_1",
            "source_id": "owned-law",
            "answer_unit_ids": ["au-new"],
            "omitted_condition": "New prose",
        },
        "",
        ledger,
    )[0]
    third = extract_repair_targets(
        {
            "condition_id": "condition_2",
            "source_id": "owned-law",
            "omitted_condition": "Old prose",
        },
        "",
        ledger,
    )[0]
    assert first.key == second.key and first.key != third.key
    nested = extract_repair_targets(
        {
            "missing_conditions": [
                {"condition_id": "condition_1", "source_id": "owned-law"},
                {"condition_id": "condition_2", "source_id": "owned-law"},
            ]
        },
        "",
        ledger,
    )
    assert len(nested) == 2 and nested[0].key != nested[1].key


def test_in_progress_attempt_cannot_be_integrated_as_completed_source_result() -> None:
    _, _, state = setup()
    target = CroppedRepairTarget(check_kind="generic")
    state.reserve(target)
    with pytest.raises(ValueError, match="completed source attempts"):
        state.reserve_integration([target], "coordinator")


def test_result_retains_a_large_bounded_verified_response() -> None:
    _, _, state = setup()
    target = CroppedRepairTarget(check_kind="generic")
    state.reserve(target)
    state.finish(target.key, "found", [1], "x" * 12000)
    assert len(cast(str, state.records_for([target])[0]["detail"])) == 12000


def test_restored_interrupted_attempt_can_be_disclosed_without_restarting_lookup() -> (
    None
):
    context, _, state = setup()
    interrupted = CroppedRepairTarget(
        check_kind="governing_original", instrument_number="6455", article="11"
    )
    completed = interrupted.model_copy(update={"article": "12"})
    state.reserve(interrupted)
    state.reserve(completed)
    state.finish(completed.key, "found", [1], "Verified source result retained.")
    resumed = CroppedRepairState(context, state.request)
    resumed.restore(state.export())
    assert resumed.close_interrupted_attempts() == 1
    assert resumed.close_interrupted_attempts() == 0
    assert not resumed.reserve(interrupted)
    row = resumed.records_for([interrupted])[0]
    assert row["status"] == "failed" and row["evidence_numbers"] == []
    assert (
        row["detail"]
        == "Interrupted focused attempt; no verified finding; absence not proven."
    )
    assert row["absence_proven"] is False
    assert resumed.reserve_integration([interrupted], "coordinator")
    assert resumed.records_for([completed])[0]["status"] == "found"
