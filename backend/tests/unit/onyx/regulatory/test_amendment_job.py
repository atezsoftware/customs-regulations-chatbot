from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock
from uuid import UUID

import pytest

from onyx.regulatory.amendments import drafter, job, pipeline
from onyx.regulatory.amendments.models import (
    AmendmentInstruction,
    MatchResult,
)
from onyx.regulatory.amendments.ranker import CandidateChunk

_CREATOR_ID = UUID("00000000-0000-0000-0000-000000000321")


def _patch_empty_retriever(
    monkeypatch: pytest.MonkeyPatch,
) -> MagicMock:
    retriever = MagicMock()
    retriever.search.return_value = []
    monkeypatch.setattr(
        job,
        "build_amendment_search_retriever",
        MagicMock(return_value=retriever),
    )
    return retriever


def _candidate(chunk_id: str) -> CandidateChunk:
    return CandidateChunk(
        chunk_id=chunk_id,
        user_file_id="00000000-0000-0000-0000-000000000123",
        text=f"Existing text for {chunk_id}",
    )


def _run_grouping_job(
    monkeypatch: pytest.MonkeyPatch,
    *,
    batch_id: int,
    targets: list[str | None],
    raw_date_phrases: list[str | None] | None = None,
    candidate_ids: list[list[str]] | None = None,
    processed_instruction_count: int = 0,
    processed_instruction_indices: list[int] | None = None,
    heartbeat_result: bool = True,
) -> SimpleNamespace:
    instructions = [
        AmendmentInstruction(
            instruction_text=f"Instruction {index}",
            raw_date_phrase=(raw_date_phrases or [None] * len(targets))[index],
        )
        for index in range(len(targets))
    ]
    matches = [
        MatchResult(
            old_chunk_id=target,
            confidence=0.9 - index / 10,
            rationale=f"rationale {index}",
        )
        for index, target in enumerate(targets)
    ]
    batch_values: dict[str, object] = {
        "id": batch_id,
        "document_set_id": 7,
        "created_by": _CREATOR_ID,
        "raw_text": "original",
        "user_file_ids": ["00000000-0000-0000-0000-000000000123"],
        "reference_date": None,
        "segmented_instructions": [item.model_dump() for item in instructions],
        "processed_instruction_count": processed_instruction_count,
    }
    if processed_instruction_indices is not None:
        batch_values["processed_instruction_indices"] = processed_instruction_indices
    batch = SimpleNamespace(**batch_values)
    session_depth = 0

    @contextmanager
    def _session():
        nonlocal session_depth
        session_depth += 1
        try:
            yield MagicMock()
        finally:
            session_depth -= 1

    events: list[tuple[str, object]] = []
    candidate_lists = [
        [_candidate(chunk_id) for chunk_id in ids]
        for ids in (
            candidate_ids
            or [
                [target or f"new-reference-{index}"]
                for index, target in enumerate(targets)
            ]
        )
    ]
    index_by_text = {
        instruction.instruction_text: index
        for index, instruction in enumerate(instructions)
    }

    def retrieve(
        *,
        retriever: object,
        llm: object,
        instruction: AmendmentInstruction,
    ) -> tuple[list[CandidateChunk], MatchResult]:
        del retriever, llm
        assert session_depth == 0
        instruction_index = index_by_text[instruction.instruction_text]
        events.append(("match", instruction_index))
        return candidate_lists[instruction_index], matches[instruction_index]

    def load_context(
        *_args: object,
        candidates: list[CandidateChunk],
        match: MatchResult,
        **_kwargs: object,
    ) -> SimpleNamespace:
        assert session_depth == 1
        events.append(
            (
                "load",
                (match.old_chunk_id, [candidate.chunk_id for candidate in candidates]),
            )
        )
        return SimpleNamespace(match=match, candidates=candidates)

    def draft_group(*_args: object, **kwargs: Any) -> SimpleNamespace:
        assert session_depth == 0
        events.append(("draft", list(kwargs["instruction_indices"])))
        return SimpleNamespace(
            instruction_indices=list(kwargs["instruction_indices"]),
            instruction_texts=[
                instruction.instruction_text for instruction in kwargs["instructions"]
            ],
            match_confidence=min(match.confidence for match in kwargs["matches"]),
        )

    persisted = MagicMock(return_value=True)
    heartbeat = MagicMock(return_value=heartbeat_result)
    draft_group_mock = MagicMock(side_effect=draft_group)
    monkeypatch.setattr(job, "_session", _session)
    monkeypatch.setattr(job, "get_batch", lambda *_args: batch)
    monkeypatch.setattr(
        job,
        "build_amendment_search_retriever",
        MagicMock(return_value=MagicMock()),
    )
    monkeypatch.setattr(job, "retrieve_and_confirm_instruction", retrieve)
    monkeypatch.setattr(job, "load_instruction_draft_context", load_context)
    monkeypatch.setattr(
        job, "draft_instruction_group_proposal", draft_group_mock, raising=False
    )
    monkeypatch.setattr(job, "touch_batch_heartbeat", heartbeat, raising=False)
    monkeypatch.setattr(job, "persist_proposal_checkpoint", persisted)
    monkeypatch.setattr(
        job, "persist_unmatched_checkpoint", MagicMock(return_value=True)
    )
    monkeypatch.setattr(job, "mark_batch_analyzed", MagicMock(return_value=True))
    monkeypatch.setattr(job, "get_default_llm", MagicMock(return_value=MagicMock()))
    job.run_amendment_batch(batch_id=batch_id, lease_generation=2)
    return SimpleNamespace(
        draft=draft_group_mock,
        persist=persisted,
        heartbeat=heartbeat,
        events=events,
        instructions=instructions,
    )


def test_unmatched_initial_search_gets_one_recovery_before_giving_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instruction = AmendmentInstruction(
        instruction_text="Carnets can now cover up to 8 customs offices.",
        search_query=(
            "TIR karnesi kapsamında işlem yapılabilecek gümrük idaresi sayısı kaçtır?"
        ),
        recovery_query="TIR karnesi gümrük idaresi azami sayı",
    )
    candidate = CandidateChunk(
        chunk_id="tir-chunk-6",
        user_file_id="00000000-0000-0000-0000-000000000123",
        text="Bir TIR taşıması en fazla sekiz gümrük idaresini kapsayabilir.",
    )
    retriever = MagicMock()
    retriever.search.side_effect = [[], [candidate]]
    expected_match = MatchResult(
        old_chunk_id="tir-chunk-6",
        confidence=0.93,
        rationale="The recovered chunk governs the customs-office limit.",
    )
    confirm = MagicMock(return_value=expected_match)
    monkeypatch.setattr(job, "confirm_instruction_match", confirm)

    candidates, match = job.retrieve_and_confirm_instruction(
        retriever=retriever,
        llm=MagicMock(),
        instruction=instruction,
    )

    assert candidates == [candidate]
    assert match == expected_match
    assert retriever.search.call_args_list[0].kwargs == {
        "instruction": instruction,
        "recovery": False,
    }
    assert retriever.search.call_args_list[1].kwargs == {
        "instruction": instruction,
        "recovery": True,
    }
    confirm.assert_called_once()


def test_match_rejection_gets_only_one_recovery_and_rechecks_merged_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instruction = AmendmentInstruction(
        instruction_text="Risk-inspection criteria were updated.",
        search_query="TIR işlemlerinde risk kriterlerine göre muayene nasıl yapılır?",
        recovery_query="TIR risk kriterleri muayene kontrol",
    )
    first = CandidateChunk(
        chunk_id="wrong",
        user_file_id="00000000-0000-0000-0000-000000000123",
        text="Yetkilendirilmiş yükümlü şartları.",
    )
    recovered = CandidateChunk(
        chunk_id="risk",
        user_file_id="00000000-0000-0000-0000-000000000123",
        text="Risk kriterlerine göre fiziki muayeneye sevk edilir.",
    )
    retriever = MagicMock()
    retriever.search.side_effect = [[first], [recovered]]
    expected_match = MatchResult(
        old_chunk_id="risk", confidence=0.91, rationale="Recovered target"
    )
    confirm = MagicMock(side_effect=[None, expected_match])
    monkeypatch.setattr(job, "confirm_instruction_match", confirm)

    candidates, match = job.retrieve_and_confirm_instruction(
        retriever=retriever,
        llm=MagicMock(),
        instruction=instruction,
    )

    assert candidates == [first, recovered]
    assert match == expected_match
    assert retriever.search.call_count == 2
    assert confirm.call_count == 2


def test_appendix_without_replacement_body_never_reaches_matcher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instruction = AmendmentInstruction(
        instruction_text=(
            "Gümrük Genel Tebliği (TIR İşlemleri) (Seri No: 1)'nin "
            "EK-4’ü ekteki şekilde değiştirilmiştir."
        ),
        recovery_query="TIR İşlemleri EK-4",
    )
    candidates = [
        CandidateChunk(
            chunk_id="ek-4-part-1",
            user_file_id="00000000-0000-0000-0000-000000000123",
            text="EK-4 first part",
            metadata={"appendix_label": "EK-4"},
        ),
        CandidateChunk(
            chunk_id="ek-4-part-2",
            user_file_id="00000000-0000-0000-0000-000000000123",
            text="EK-4 second part",
            metadata={"appendix_label": "EK-4"},
        ),
    ]
    retriever = MagicMock()
    retriever.search.return_value = candidates
    confirm = MagicMock()
    monkeypatch.setattr(job, "confirm_instruction_match", confirm)

    returned_candidates, match = job.retrieve_and_confirm_instruction(
        retriever=retriever,
        llm=MagicMock(),
        instruction=instruction,
    )

    assert returned_candidates == candidates
    assert match is None
    assert retriever.search.call_count == 1
    confirm.assert_not_called()


def test_recovery_only_appendix_without_body_never_reaches_matcher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instruction = AmendmentInstruction(
        instruction_text="TIR İşlemleri Tebliği'nin EK-10’u ekteki şekilde değiştirilmiştir.",
        target_source="Gümrük Genel Tebliği (TIR İşlemleri) (Seri No: 1)",
        search_query="TIR İşlemleri EK-10",
        recovery_query="TIR İşlemleri eki 10",
    )
    recovered = CandidateChunk(
        chunk_id="ek-10-part-1",
        user_file_id="00000000-0000-0000-0000-000000000123",
        text="EK-10 content",
        metadata={"appendix_label": "EK-10"},
    )
    retriever = MagicMock()
    retriever.search.side_effect = [[], [recovered]]
    confirm = MagicMock()
    monkeypatch.setattr(job, "confirm_instruction_match", confirm)

    candidates, match = job.retrieve_and_confirm_instruction(
        retriever=retriever,
        llm=MagicMock(),
        instruction=instruction,
    )

    assert candidates == [recovered]
    assert match is None
    assert retriever.search.call_count == 2
    confirm.assert_not_called()


def test_resume_reuses_segmentation_and_skips_completed_instructions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = SimpleNamespace(
        id=9,
        document_set_id=7,
        created_by=_CREATOR_ID,
        raw_text="original",
        user_file_ids=["00000000-0000-0000-0000-000000000123"],
        reference_date=None,
        segmented_instructions=[
            {"instruction_text": "MADDE 1"},
            {"instruction_text": "MADDE 2"},
        ],
        processed_instruction_count=1,
    )
    db_session = MagicMock()

    @contextmanager
    def _session():
        yield db_session

    monkeypatch.setattr(job, "_session", _session)
    monkeypatch.setattr(job, "get_batch", lambda *_args: batch)
    segmentation = MagicMock()
    monkeypatch.setattr(job, "segment_amendment_text", segmentation)
    retriever = _patch_empty_retriever(monkeypatch)
    persist_unmatched = MagicMock(return_value=True)
    monkeypatch.setattr(job, "persist_unmatched_checkpoint", persist_unmatched)
    monkeypatch.setattr(job, "mark_batch_analyzed", MagicMock(return_value=True))
    monkeypatch.setattr(job, "get_default_llm", MagicMock(return_value=MagicMock()))

    job.run_amendment_batch(batch_id=9, lease_generation=3)

    segmentation.assert_not_called()
    assert [
        item.kwargs["instruction"].instruction_text
        for item in retriever.search.call_args_list
        if item.kwargs["recovery"] is False
    ] == ["MADDE 2"]
    persist_unmatched.assert_called_once()


def test_first_run_persists_segmentation_before_instruction_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = SimpleNamespace(
        id=10,
        document_set_id=7,
        created_by=_CREATOR_ID,
        raw_text="MADDE 1",
        user_file_ids=["00000000-0000-0000-0000-000000000123"],
        reference_date=None,
        segmented_instructions=[],
        processed_instruction_count=0,
    )
    db_session = MagicMock()

    @contextmanager
    def _session():
        yield db_session

    monkeypatch.setattr(job, "_session", _session)
    monkeypatch.setattr(job, "get_batch", lambda *_args: batch)
    segment_result = SimpleNamespace(
        reference_date="2026-08-26",
        instructions=[
            SimpleNamespace(model_dump=lambda: {"instruction_text": "MADDE 1"})
        ],
    )
    monkeypatch.setattr(
        job, "segment_amendment_text", MagicMock(return_value=segment_result)
    )

    def _persist_segmentation(*_args: object, **kwargs: Any) -> bool:
        batch.segmented_instructions = list(kwargs["instructions"])
        batch.reference_date = "2026-08-26"
        return True

    persist_segmentation = MagicMock(side_effect=_persist_segmentation)
    monkeypatch.setattr(job, "persist_segmentation_checkpoint", persist_segmentation)
    _patch_empty_retriever(monkeypatch)
    monkeypatch.setattr(
        job, "persist_unmatched_checkpoint", MagicMock(return_value=True)
    )
    monkeypatch.setattr(job, "mark_batch_analyzed", MagicMock(return_value=True))
    monkeypatch.setattr(job, "get_default_llm", MagicMock(return_value=MagicMock()))

    job.run_amendment_batch(batch_id=10, lease_generation=1)

    persist_segmentation.assert_called_once()


def test_segmentation_runs_without_an_open_database_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = SimpleNamespace(
        id=11,
        document_set_id=7,
        created_by=_CREATOR_ID,
        raw_text="MADDE 1",
        user_file_ids=["00000000-0000-0000-0000-000000000123"],
        reference_date=None,
        segmented_instructions=[],
        processed_instruction_count=0,
    )
    session_depth = 0

    @contextmanager
    def _session():
        nonlocal session_depth
        session_depth += 1
        try:
            yield MagicMock()
        finally:
            session_depth -= 1

    def segment(*_args: object) -> SimpleNamespace:
        assert session_depth == 0
        return SimpleNamespace(
            reference_date=None,
            instructions=[
                SimpleNamespace(model_dump=lambda: {"instruction_text": "MADDE 1"})
            ],
        )

    monkeypatch.setattr(job, "_session", _session)
    monkeypatch.setattr(job, "get_batch", lambda *_args: batch)
    monkeypatch.setattr(job, "segment_amendment_text", segment)
    monkeypatch.setattr(
        job, "persist_segmentation_checkpoint", MagicMock(return_value=True)
    )
    _patch_empty_retriever(monkeypatch)
    monkeypatch.setattr(
        job, "persist_unmatched_checkpoint", MagicMock(return_value=True)
    )
    monkeypatch.setattr(job, "mark_batch_analyzed", MagicMock(return_value=True))
    monkeypatch.setattr(job, "get_default_llm", MagicMock(return_value=MagicMock()))

    job.run_amendment_batch(batch_id=11, lease_generation=1)


def test_empty_segmentation_fails_instead_of_checkpointing_ambiguous_empty_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = SimpleNamespace(
        id=12,
        document_set_id=7,
        created_by=_CREATOR_ID,
        raw_text="belirsiz metin",
        user_file_ids=["00000000-0000-0000-0000-000000000123"],
        reference_date=None,
        segmented_instructions=[],
        processed_instruction_count=0,
    )

    @contextmanager
    def _session():
        yield MagicMock()

    monkeypatch.setattr(job, "_session", _session)
    monkeypatch.setattr(job, "get_batch", lambda *_args: batch)
    _patch_empty_retriever(monkeypatch)
    monkeypatch.setattr(
        job,
        "segment_amendment_text",
        MagicMock(return_value=SimpleNamespace(reference_date=None, instructions=[])),
    )
    persist = MagicMock()
    monkeypatch.setattr(job, "persist_segmentation_checkpoint", persist)
    monkeypatch.setattr(job, "get_default_llm", MagicMock(return_value=MagicMock()))

    with pytest.raises(RuntimeError, match="no instructions"):
        job.run_amendment_batch(batch_id=12, lease_generation=1)

    persist.assert_not_called()


def test_match_and_draft_llm_calls_run_outside_database_sessions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instructions = [
        {"instruction_text": "MADDE 1"},
        {"instruction_text": "MADDE 1 ikinci değişiklik"},
    ]
    batch = SimpleNamespace(
        id=13,
        document_set_id=7,
        created_by=_CREATOR_ID,
        raw_text="MADDE 1",
        user_file_ids=["00000000-0000-0000-0000-000000000123"],
        reference_date=None,
        segmented_instructions=instructions,
        processed_instruction_count=0,
        processed_instruction_indices=[],
    )
    session_depth = 0

    @contextmanager
    def _session():
        nonlocal session_depth
        session_depth += 1
        try:
            yield MagicMock()
        finally:
            session_depth -= 1

    candidate = SimpleNamespace(chunk_id="chunk-1")
    match = MatchResult(old_chunk_id="chunk-1", confidence=0.9, rationale="match")
    context = SimpleNamespace()
    proposal = SimpleNamespace(instruction_index=0)
    matcher_calls = 0

    def search(**_kwargs: object) -> list[SimpleNamespace]:
        assert session_depth == 0
        return [candidate]

    def confirm(*_args: object, **_kwargs: object) -> MatchResult:
        nonlocal matcher_calls
        assert session_depth == 0
        matcher_calls += 1
        return match

    def load_context(*_args: object, **_kwargs: object) -> SimpleNamespace:
        assert session_depth == 1
        return context

    def draft(*_args: object, **_kwargs: object) -> SimpleNamespace:
        assert session_depth == 0
        return proposal

    draft_mock = MagicMock(side_effect=draft)

    monkeypatch.setattr(job, "_session", _session)
    monkeypatch.setattr(job, "get_batch", lambda *_args: batch)
    retriever = MagicMock()
    retriever.search.side_effect = search
    monkeypatch.setattr(
        job,
        "build_amendment_search_retriever",
        MagicMock(return_value=retriever),
    )
    monkeypatch.setattr(job, "confirm_instruction_match", confirm)
    monkeypatch.setattr(job, "load_instruction_draft_context", load_context)
    monkeypatch.setattr(
        job, "draft_instruction_group_proposal", draft_mock, raising=False
    )
    monkeypatch.setattr(
        job, "touch_batch_heartbeat", MagicMock(return_value=True), raising=False
    )
    monkeypatch.setattr(
        job, "persist_proposal_checkpoint", MagicMock(return_value=True)
    )
    monkeypatch.setattr(job, "mark_batch_analyzed", MagicMock(return_value=True))
    monkeypatch.setattr(job, "get_default_llm", MagicMock(return_value=MagicMock()))

    job.run_amendment_batch(batch_id=13, lease_generation=1)

    assert matcher_calls == 2
    draft_mock.assert_called_once()


def test_same_target_instructions_create_one_combined_proposal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _run_grouping_job(
        monkeypatch,
        batch_id=20,
        targets=["shared", "shared"],
        candidate_ids=[["shared", "first-only"], ["shared", "second-only"]],
        processed_instruction_indices=[],
    )

    result.draft.assert_called_once()
    assert result.draft.call_args.kwargs["instruction_indices"] == [0, 1]
    proposal = result.persist.call_args.kwargs["proposal"]
    assert proposal.instruction_indices == [0, 1]
    assert proposal.instruction_texts == ["Instruction 0", "Instruction 1"]
    assert proposal.match_confidence == 0.8
    assert result.heartbeat.call_count == 2
    assert result.events == [
        ("match", 0),
        ("match", 1),
        ("load", ("shared", ["shared", "first-only", "second-only"])),
        ("draft", [0, 1]),
    ]


def test_noncontiguous_same_target_instructions_still_consolidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _run_grouping_job(
        monkeypatch,
        batch_id=21,
        targets=["target-a", "target-b", "target-a"],
        processed_instruction_indices=[],
    )

    assert [
        call.kwargs["instruction_indices"] for call in result.draft.call_args_list
    ] == [[0, 2], [1]]
    assert [
        call.kwargs["proposal"].instruction_indices
        for call in result.persist.call_args_list
    ] == [[0, 2], [1]]
    assert result.events[:3] == [("match", 0), ("match", 1), ("match", 2)]


@pytest.mark.parametrize(
    ("targets", "batch_id"),
    [
        pytest.param(["target-a", "target-b"], 22, id="different-target-ids"),
        pytest.param([None, None], 23, id="separate-new-provisions"),
    ],
)
def test_distinct_targets_create_separate_proposals(
    monkeypatch: pytest.MonkeyPatch,
    targets: list[str | None],
    batch_id: int,
) -> None:
    result = _run_grouping_job(
        monkeypatch,
        batch_id=batch_id,
        targets=targets,
        processed_instruction_indices=[],
    )

    assert [
        call.kwargs["instruction_indices"] for call in result.draft.call_args_list
    ] == [[0], [1]]
    assert result.persist.call_count == 2


def test_resume_uses_exact_processed_indices_instead_of_a_count_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _run_grouping_job(
        monkeypatch,
        batch_id=24,
        targets=["target-a", "already-done", "target-c"],
        processed_instruction_count=1,
        processed_instruction_indices=[1],
    )

    assert [event for event in result.events if event[0] == "match"] == [
        ("match", 0),
        ("match", 2),
    ]
    assert [
        call.kwargs["instruction_indices"] for call in result.draft.call_args_list
    ] == [[0], [2]]


def test_resume_uses_legacy_count_when_exact_indices_are_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _run_grouping_job(
        monkeypatch,
        batch_id=26,
        targets=["already-done", "target-b"],
        processed_instruction_count=1,
        processed_instruction_indices=[],
    )

    assert [event for event in result.events if event[0] == "match"] == [("match", 1)]
    assert [
        call.kwargs["instruction_indices"] for call in result.draft.call_args_list
    ] == [[1]]


def test_lost_heartbeat_lease_stops_before_drafting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(RuntimeError, match="lost its lease"):
        _run_grouping_job(
            monkeypatch,
            batch_id=25,
            targets=["target-a"],
            processed_instruction_indices=[],
            heartbeat_result=False,
        )


def test_same_target_conflicting_explicit_date_phrases_fail_before_drafting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instructions = [
        AmendmentInstruction(
            instruction_text="Change the amount.",
            raw_date_phrase="Yayımı tarihinden itibaren",
        ),
        AmendmentInstruction(
            instruction_text="Change the deadline.",
            raw_date_phrase="1 Ocak 2027 tarihinde",
        ),
    ]
    matches = [
        MatchResult(old_chunk_id="shared", confidence=0.9, rationale="amount"),
        MatchResult(old_chunk_id="shared", confidence=0.8, rationale="deadline"),
    ]
    context = pipeline.InstructionDraftContext(
        match=matches[0],
        old_chunk_snapshot={"id": "shared", "text": "old"},
        target_user_file_id=UUID("00000000-0000-0000-0000-000000000123"),
        target_position=4,
        sibling_reference=None,
        base_metadata={},
        base_heading_path=[],
    )
    generate_structured = MagicMock()
    monkeypatch.setattr(drafter, "generate_structured", generate_structured)

    with pytest.raises(RuntimeError, match="effective-date phrases"):
        pipeline.draft_instruction_group_proposal(
            MagicMock(),
            instruction_indices=[0, 1],
            instructions=instructions,
            matches=matches,
            reference_date="2026-08-27",
            context=context,
        )

    generate_structured.assert_not_called()
