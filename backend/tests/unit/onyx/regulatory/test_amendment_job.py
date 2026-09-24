from collections.abc import Callable
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock
from uuid import UUID

import pytest

from onyx.regulatory.amendments import drafter, job, pipeline
from onyx.regulatory.amendments.draft_integrity import DraftIntegrityError
from onyx.regulatory.amendments.models import (
    AmendmentInstruction,
    MatchResult,
)
from onyx.regulatory.amendments.ranker import CandidateChunk

_CREATOR_ID = UUID("00000000-0000-0000-0000-000000000321")


@pytest.fixture(autouse=True)
def _checkpoint_storage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(job, "match_scope_fingerprint", lambda *_args: "scope")
    monkeypatch.setattr(
        job,
        "capture_match_evidence",
        lambda *_args: job.MatchEvidence(scope_sha256="scope", candidates={}),
    )
    saved: dict[int, object] = {}

    def persist(*_args: object, **kwargs: Any) -> bool:
        saved[kwargs["checkpoint"].instruction_index] = kwargs["checkpoint"]
        return True

    monkeypatch.setattr(job, "persist_match_checkpoint", persist)
    monkeypatch.setattr(
        job,
        "load_match_checkpoint",
        lambda *_args, **kwargs: saved.get(kwargs["instruction_index"]),
    )


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
    draft_error: Exception | None = None,
    draft_failures: dict[int, Exception] | None = None,
    instructions_override: list[AmendmentInstruction] | None = None,
    source_package_id: UUID | None = None,
    checkpoint_store: dict[int, object] | None = None,
    interrupt_match_at: int | None = None,
    before_work: Callable[[], None] | None = None,
) -> SimpleNamespace:
    instructions = instructions_override or [
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
        "source_package_id": source_package_id,
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
        amendment_context: object = None,
        trace: object = None,
        capture_evidence: object = None,
    ) -> tuple[list[CandidateChunk], MatchResult]:
        del retriever, llm, amendment_context, trace, capture_evidence
        assert session_depth == 0
        instruction_index = index_by_text[instruction.instruction_text]
        events.append(("match", instruction_index))
        if instruction_index == interrupt_match_at:
            raise RuntimeError("simulated worker interruption")
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
        return SimpleNamespace(
            match=match,
            candidates=candidates,
            target_user_file_id=UUID("00000000-0000-0000-0000-000000000123"),
        )

    def draft_group(*_args: object, **kwargs: Any) -> SimpleNamespace:
        assert session_depth == 0
        events.append(("draft", list(kwargs["instruction_indices"])))
        if draft_failures and kwargs["instruction_indices"][0] in draft_failures:
            raise draft_failures[kwargs["instruction_indices"][0]]
        if draft_error is not None:
            raise draft_error
        return SimpleNamespace(
            instruction_indices=list(kwargs["instruction_indices"]),
            instruction_texts=[
                instruction.instruction_text for instruction in kwargs["instructions"]
            ],
            match_confidence=min(match.confidence for match in kwargs["matches"]),
        )

    persisted = MagicMock(return_value=True)
    unmatched = MagicMock(return_value=True)
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
    monkeypatch.setattr(job, "draft_multi_chunk_group_proposal", draft_group_mock)
    monkeypatch.setattr(job, "touch_batch_heartbeat", heartbeat, raising=False)
    monkeypatch.setattr(job, "persist_proposal_checkpoint", persisted)
    monkeypatch.setattr(job, "persist_unmatched_checkpoint", unmatched)
    monkeypatch.setattr(job, "mark_batch_analyzed", MagicMock(return_value=True))
    monkeypatch.setattr(
        job, "get_amendment_analysis_llm", MagicMock(return_value=MagicMock())
    )
    saved_matches = checkpoint_store if checkpoint_store is not None else {}

    def save_match(*_args: object, **kwargs: Any) -> bool:
        events.append(("checkpoint", kwargs["checkpoint"].instruction_index))
        if not heartbeat_result:
            return False
        saved_matches[kwargs["checkpoint"].instruction_index] = kwargs["checkpoint"]
        return True

    monkeypatch.setattr(job, "persist_match_checkpoint", save_match, raising=False)
    monkeypatch.setattr(
        job,
        "load_match_checkpoint",
        lambda *_args, **kwargs: saved_matches.get(kwargs["instruction_index"]),
        raising=False,
    )
    job.run_amendment_batch(
        batch_id=batch_id, lease_generation=2, before_work=before_work
    )
    return SimpleNamespace(
        draft=draft_group_mock,
        persist=persisted,
        heartbeat=heartbeat,
        unmatched=unmatched,
        events=events,
        instructions=instructions,
    )


def test_pdf_explicit_annex_edits_produce_one_proposal_and_no_document_review(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from onyx.db import (
        amendment_pdf_evidence,
        regulatory_annex_changes,
        regulatory_annexes,
    )
    from onyx.db.models import RegulatoryChunk
    from onyx.regulatory.amendments.annexes import analysis, config

    file_id = UUID("00000000-0000-0000-0000-000000000123")
    rows = [
        RegulatoryChunk(
            id=f"annex-{i}",
            user_file_id=file_id,
            text=f"Row {i}",
            chunk_type="table",
            chunk_metadata={"appendix_label": "EK 2"},
        )
        for i in range(14)
    ]
    monkeypatch.setattr(config, "REGULATORY_ANNEX_UPDATES_ENABLED", True)
    monkeypatch.setattr(
        amendment_pdf_evidence, "load_batch_pdf_source", lambda *_args: None
    )
    monkeypatch.setattr(
        regulatory_annex_changes,
        "resolve_annex_instruction_file",
        lambda *_args, **_kwargs: file_id,
    )
    monkeypatch.setattr(
        regulatory_annexes, "load_legacy_annex_chunks", lambda *_args, **_kwargs: rows
    )
    reviews: list[object] = []

    def record_reviews(**kwargs: Any) -> set[int]:
        reviews.extend(kwargs["groups"])
        return set()

    monkeypatch.setattr(analysis, "run_annex_groups", record_reviews)
    result = _run_grouping_job(
        monkeypatch,
        batch_id=96,
        targets=["annex-0", "annex-0"],
        source_package_id=UUID("00000000-0000-0000-0000-000000000999"),
        instructions_override=[
            AmendmentInstruction(instruction_text=text, article_reference="Ek-2")
            for text in [
                "MADDE 16- Aynı Tebliğin Ek-2’sinde yer alan listenin 26 ncı sırası yürürlükten kaldırılmıştır.",
                "MADDE 17- Aynı Tebliğin Ek-2’sinde yer alan listeye aşağıdaki sıra eklenmiştir.\n26. 8429.11.00.00.00 Paletli olanlar Makina, Emisyon, Gürültü",
            ]
        ],
    )
    assert reviews == []
    assert ("draft", [0, 1]) in result.events
    assert len(result.persist.call_args_list) == 1
    assert result.persist.call_args.kwargs["proposal"].instruction_indices == [0, 1]
    assert len(result.draft.call_args.kwargs["contexts"]) == 14


def test_schema_failure_preserves_other_groups_as_an_attention_item(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from onyx.regulatory.structured_llm import StructuredOutputValidationError

    error = StructuredOutputValidationError("invalid draft date")
    result = _run_grouping_job(
        monkeypatch,
        batch_id=31,
        targets=["failed", "completed", "good"],
        processed_instruction_indices=[1],
        processed_instruction_count=1,
        draft_failures={0: error},
    )
    cast(MagicMock, job.persist_proposal_checkpoint).assert_called_once()
    assert cast(MagicMock, job.persist_proposal_checkpoint).call_args.kwargs[
        "proposal"
    ].instruction_indices == [2]
    assert [
        call.kwargs["instruction_indices"]
        for call in cast(MagicMock, job.draft_instruction_group_proposal).call_args_list
    ] == [[0], [2]]
    result.unmatched.assert_called_once()
    assert (
        "Draft output could not be validated"
        in result.unmatched.call_args.kwargs["instruction_text"]
    )
    cast(MagicMock, job.mark_batch_analyzed).assert_called_once()


def test_invalid_generated_draft_becomes_an_attention_item(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _run_grouping_job(
        monkeypatch,
        batch_id=27,
        targets=["target-a"],
        processed_instruction_indices=[],
        draft_error=DraftIntegrityError(
            "The generated draft does not contain the explicit replacement body."
        ),
    )

    result.persist.assert_not_called()
    result.unmatched.assert_called_once()
    assert (
        "explicit replacement body"
        in result.unmatched.call_args.kwargs["instruction_text"]
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
        source_package_id=None,
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
    monkeypatch.setattr(
        job, "get_amendment_analysis_llm", MagicMock(return_value=MagicMock())
    )

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
        source_package_id=None,
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
    monkeypatch.setattr(
        job, "get_amendment_analysis_llm", MagicMock(return_value=MagicMock())
    )

    job.run_amendment_batch(batch_id=10, lease_generation=1)

    persist_segmentation.assert_called_once()


def test_segmentation_runs_without_an_open_database_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = SimpleNamespace(
        id=11,
        document_set_id=7,
        source_package_id=None,
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
    monkeypatch.setattr(
        job, "get_amendment_analysis_llm", MagicMock(return_value=MagicMock())
    )

    job.run_amendment_batch(batch_id=11, lease_generation=1)


@pytest.mark.parametrize(
    "raw_text",
    [
        "belirsiz metin",
        "KAFE MENÜ\nFiltre kahve 100 TL",
        "Hatıra fotoğrafı: 2026 yaz tatili",
    ],
)
def test_empty_segmentation_checkpoints_and_finishes_analyzed_without_review(
    monkeypatch: pytest.MonkeyPatch,
    raw_text: str,
) -> None:
    """No update instructions is a legitimate terminal outcome, not a crash.

    The segmenter is explicitly instructed to return an empty instruction
    list for content (or context) that doesn't express update intent —
    ambiguous text, a menu, a photo caption. The batch must still reach
    `analyzed` with nothing to review, not raise, so retrying the identical
    text isn't offered as if it could ever produce a different result.
    """
    from onyx.llm.model_response import Choice, Message, ModelResponse

    batch = SimpleNamespace(
        id=12,
        document_set_id=7,
        source_package_id=None,
        created_by=_CREATOR_ID,
        raw_text=raw_text,
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
    retriever = _patch_empty_retriever(monkeypatch)
    model = MagicMock()
    model.config.model_provider = "fixture"
    model.config.model_name = "fixture"
    model.invoke.return_value = ModelResponse(
        id="empty-segmentation",
        created="2026-09-14",
        choice=Choice(
            message=Message(content='{"reference_date":null,"instructions":[]}')
        ),
    )
    persist = MagicMock(return_value=True)
    monkeypatch.setattr(job, "persist_segmentation_checkpoint", persist)
    monkeypatch.setattr(
        job, "get_amendment_analysis_llm", MagicMock(return_value=model)
    )
    draft = MagicMock()
    proposals = MagicMock()
    monkeypatch.setattr(job, "draft_instruction_group_proposal", draft)
    monkeypatch.setattr(job, "persist_proposal_checkpoint", proposals)
    mark_analyzed = MagicMock(return_value=True)
    monkeypatch.setattr(job, "mark_batch_analyzed", mark_analyzed)

    job.run_amendment_batch(batch_id=12, lease_generation=1)

    assert persist.call_args.kwargs["instructions"] == []
    retriever.search.assert_not_called()
    draft.assert_not_called()
    proposals.assert_not_called()
    mark_analyzed.assert_called_once()
    assert mark_analyzed.call_args.kwargs["batch_id"] == 12
    assert mark_analyzed.call_args.kwargs["lease_generation"] == 1


def test_contextual_replacement_table_is_accepted_without_formal_legal_phrase() -> None:
    import json

    from onyx.llm.model_response import Choice, Message, ModelResponse
    from onyx.regulatory.amendments.segmenter import segment_amendment_text

    context = "Tablonun yeni hali bu:\nÜrün | Oran\nBuğday | %17"
    model = MagicMock()
    model.config.model_provider = "fixture"
    model.config.model_name = "fixture"
    model.invoke.return_value = ModelResponse(
        id="replacement-table",
        created="2026-09-14",
        choice=Choice(
            message=Message(
                content=json.dumps(
                    {
                        "reference_date": None,
                        "instructions": [
                            {
                                "instruction_text": context,
                                "search_query": "Buğday için mevcut oran tablosu nedir?",
                                "recovery_query": "buğday oran tablosu",
                            }
                        ],
                    }
                )
            )
        ),
    )
    result = segment_amendment_text(model, context)
    assert len(result.instructions) == 1
    assert result.instructions[0].instruction_text == context
    assert result.instructions[0].article_reference is None
    assert result.instructions[0].target_source is None
    assert result.reference_date is None


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
        source_package_id=None,
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

    candidate = _candidate("chunk-1")
    match = MatchResult(old_chunk_id="chunk-1", confidence=0.9, rationale="match")
    context = SimpleNamespace()
    proposal = SimpleNamespace(instruction_index=0)
    matcher_calls = 0

    def search(**_kwargs: object) -> list[CandidateChunk]:
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
    monkeypatch.setattr(
        job, "get_amendment_analysis_llm", MagicMock(return_value=MagicMock())
    )

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
    assert result.heartbeat.call_count == 0
    assert result.events == [
        ("match", 0),
        ("checkpoint", 0),
        ("match", 1),
        ("checkpoint", 1),
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
    assert [event for event in result.events if event[0] == "match"] == [
        ("match", 0),
        ("match", 1),
        ("match", 2),
    ]


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


def test_additional_article_replacement_and_new_units_are_separate_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import json
    from pathlib import Path

    payload = json.loads(
        (Path(__file__).parent / "fixtures/7594_amendment.json").read_text()
    )
    result = _run_grouping_job(
        monkeypatch,
        batch_id=102,
        targets=["table", None, None],
        instructions_override=[
            AmendmentInstruction.model_validate(payload["instructions"][i])
            for i in [4, 5, 6]
        ],
    )
    assert [
        call.kwargs["instruction_indices"] for call in result.draft.call_args_list
    ] == [[0], [1], [2]]


def test_annex_groups_do_not_cross_source_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys
    from dataclasses import replace

    original = _candidate
    monkeypatch.setattr(
        sys.modules[__name__],
        "_candidate",
        lambda chunk_id: replace(
            original(chunk_id),
            user_file_id=(
                "00000000-0000-0000-0000-000000000124"
                if chunk_id == "second"
                else "00000000-0000-0000-0000-000000000123"
            ),
        ),
    )
    instructions = [
        AmendmentInstruction(
            instruction_text=f'MADDE {index + 1}- EK-3 içindeki "eski" ibaresi "yeni" şeklinde değiştirilmiştir.',
            annex_change_basis="explicit_amendment",
        )
        for index in range(2)
    ]
    result = _run_grouping_job(
        monkeypatch,
        batch_id=104,
        targets=["first", "second"],
        instructions_override=instructions,
    )
    assert [
        call.kwargs["instruction_indices"] for call in result.draft.call_args_list
    ] == [[0], [1]]


def test_missing_source_attention_is_preserved_without_model_call() -> None:
    retriever = MagicMock()
    retriever.search.return_value = []
    retriever.query_stats = []
    retriever.last_attention = "7082 source is absent from the captured Document Set"
    trace = job._InstructionTrace()
    candidates, match = job.retrieve_and_confirm_instruction(
        retriever=retriever,
        llm=MagicMock(),
        instruction=AmendmentInstruction(instruction_text="7082 sayılı Kanun"),
        trace=trace,
    )
    assert candidates == [] and match is None
    assert "7082" in trace.describe()
    assert retriever.search.call_count == 1


def test_quoted_annex_designator_loads_complete_canonical_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from onyx.db import regulatory_annexes

    rows = [SimpleNamespace(id=chunk_id, chunk_metadata={}) for chunk_id in ["a", "b"]]
    load = MagicMock(return_value=rows)
    monkeypatch.setattr(regulatory_annexes, "load_legacy_annex_chunks", load)
    result = _run_grouping_job(
        monkeypatch,
        batch_id=105,
        targets=["a"],
        instructions_override=[
            AmendmentInstruction(
                instruction_text='Tebliğin “EK-3” başlıklı ekinde "eski" ibaresi "yeni" olarak değiştirilmiştir.',
                article_reference="EK-3",
                annex_change_basis="explicit_amendment",
            )
        ],
    )
    load.assert_called_once()
    assert len(result.draft.call_args.kwargs["contexts"]) == 2


def test_restart_reuses_committed_matches_and_preserves_same_target_grouping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    saved: dict[int, object] = {}
    first = _run_grouping_job(
        monkeypatch,
        batch_id=97,
        targets=["same", "other", "same"],
        checkpoint_store=saved,
    )
    assert len(saved) == 3
    assert first.events.index(("checkpoint", 0)) < first.events.index(("match", 1))
    restarted = _run_grouping_job(
        monkeypatch,
        batch_id=97,
        targets=["same", "other", "same"],
        checkpoint_store=saved,
    )
    assert not any(event == "match" for event, _ in restarted.events)
    assert ("draft", [0, 2]) in restarted.events
    assert ("draft", [1]) in restarted.events


def test_partial_matching_restart_skips_saved_work_and_merges_later_same_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    saved: dict[int, object] = {}
    with pytest.raises(RuntimeError, match="simulated worker interruption"):
        _run_grouping_job(
            monkeypatch,
            batch_id=98,
            targets=["same", "other", "same"],
            checkpoint_store=saved,
            interrupt_match_at=1,
        )
    assert list(saved) == [0]
    restarted = _run_grouping_job(
        monkeypatch,
        batch_id=98,
        targets=["same", "other", "same"],
        checkpoint_store=saved,
    )
    assert [index for event, index in restarted.events if event == "match"] == [1, 2]
    assert ("draft", [0, 2]) in restarted.events


def test_evidence_is_captured_before_model_confirmation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    retriever = MagicMock()
    retriever.search.return_value = [_candidate("candidate")]
    retriever.last_attention = None
    retriever.query_stats = []
    monkeypatch.setattr(job, "deterministic_structural_candidate", lambda *_: None)

    def confirm(*_args: object, **_kwargs: object) -> MatchResult:
        events.append("confirm")
        return MatchResult(old_chunk_id="candidate", confidence=1, rationale="test")

    monkeypatch.setattr(job, "confirm_instruction_match", confirm)
    job.retrieve_and_confirm_instruction(
        retriever=retriever,
        llm=MagicMock(),
        instruction=AmendmentInstruction(instruction_text="Replace text"),
        capture_evidence=lambda _candidates: events.append("capture"),
    )
    assert events == ["capture", "confirm"]


def test_draft_does_not_start_without_reserved_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from onyx.regulatory.amendments.memory_budget import ResourcePressure

    def no_headroom() -> None:
        raise ResourcePressure("insufficient_draft_headroom")

    with pytest.raises(ResourcePressure):
        _run_grouping_job(
            monkeypatch, batch_id=500, targets=["same"], before_work=no_headroom
        )
    cast(MagicMock, job.persist_proposal_checkpoint).assert_not_called()
