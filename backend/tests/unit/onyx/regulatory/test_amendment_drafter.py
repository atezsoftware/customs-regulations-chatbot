from unittest.mock import MagicMock
from uuid import UUID

import pytest

from onyx.regulatory.amendments import drafter, pipeline
from onyx.regulatory.amendments.draft_integrity import (
    DraftIntegrityError,
    reconcile_existing_heading_path,
)
from onyx.regulatory.amendments.models import (
    AmendmentInstruction,
    ChunkFieldsDraft,
    DateResolution,
    DraftResult,
    MatchResult,
    MultiChunkDraftResult,
    MultiChunkFieldsDraft,
)


def test_draft_validates_completed_response_after_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from contextlib import nullcontext

    from onyx.llm.model_response import Choice, Message, ModelResponse
    from onyx.regulatory import structured_llm

    result = DraftResult(
        new_chunk=ChunkFieldsDraft(text="(7) Mülga.", chunk_type="paragraph"),
        dates=DateResolution(rationale="No date provided"),
    )
    llm = MagicMock()
    clock = [100.0]
    monkeypatch.setattr(structured_llm.time, "monotonic", lambda: clock[0])

    def invoke(*_args: object, **_kwargs: object) -> ModelResponse:
        clock[0] = 227.0
        return ModelResponse(
            id="draft",
            created="2026-09-24T00:00:00Z",
            choice=Choice(message=Message(content=result.model_dump_json())),
        )

    llm.invoke.side_effect = invoke
    monkeypatch.setattr(
        structured_llm, "llm_generation_span", lambda **_: nullcontext(MagicMock())
    )
    monkeypatch.setattr(structured_llm, "record_llm_response", lambda *_: None)
    draft = drafter.draft_combined_chunk(
        llm,
        instructions=[
            AmendmentInstruction(
                instruction_text="MADDE 3- Aynı Kararın 5 inci maddesinin yedinci fıkrası yürürlükten kaldırılmıştır."
            )
        ],
        old_chunk={
            "text": "(7) Eski fıkra.",
            "metadata": {"article_no": "5", "paragraph_no": "7"},
        },
        sibling_reference=None,
        reference_date=None,
    )
    assert draft == result
    assert llm.invoke.call_count == 1


@pytest.mark.parametrize("value", ["yayımı tarihinde", "2026-02-30"])
def test_model_date_schema_safely_drops_unresolved_or_invalid_dates(value: str) -> None:
    assert (
        DateResolution(
            effective_start_date=value, rationale="fixture"
        ).effective_start_date
        is None
    )
    schema = DateResolution.model_json_schema()
    assert schema["properties"]["effective_start_date"]["anyOf"][0]["pattern"]
    assert (
        DateResolution(
            effective_start_date="2026-07-04", rationale="publication"
        ).effective_end_date
        is None
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2026-9-20", "2026-09-20"),
        ("2026-09-20T00:00:00Z", "2026-09-20"),
        ("20.09.2026", "2026-09-20"),
        ("20/09/2026", "2026-09-20"),
        ("20-09-2026", "2026-09-20"),
        ("Yürürlük tarihi: 2026-09-20.", "2026-09-20"),
    ],
)
def test_date_schema_normalizes_unambiguous_model_formats(
    value: str, expected: str
) -> None:
    assert (
        DateResolution(
            effective_start_date=value, rationale="fixture"
        ).effective_start_date
        == expected
    )


def test_combined_draft_prompt_contains_each_instruction_and_returns_one_proposal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instructions = [
        AmendmentInstruction(
            instruction_text="The amount is changed to 500 lira.",
            raw_date_phrase=" Yayımı   Tarihinden İtibaren ",
        ),
        AmendmentInstruction(
            instruction_text="The filing period is changed to 30 days.",
            raw_date_phrase="yayımı tarihinden itibaren",
        ),
    ]
    matches = [
        MatchResult(old_chunk_id="shared", confidence=0.91, rationale="amount row"),
        MatchResult(old_chunk_id="shared", confidence=0.73, rationale="period row"),
    ]
    context = pipeline.InstructionDraftContext(
        match=matches[0],
        old_chunk_snapshot={
            "id": "shared",
            "text": "The amount is 100 lira and the filing period is 10 days.",
            "chunk_type": "article",
            "heading_path": ["MADDE 4"],
            "metadata": {"article_no": "4"},
        },
        target_user_file_id=UUID("00000000-0000-0000-0000-000000000123"),
        target_position=4,
        sibling_reference=None,
        base_metadata={"article_no": "4"},
        base_heading_path=["MADDE 4"],
    )
    generated = DraftResult(
        new_chunk=ChunkFieldsDraft(
            text="The amount is 500 lira and the filing period is 30 days.",
            chunk_type="article",
        ),
        dates=DateResolution(
            effective_start_date="2026-08-27",
            effective_end_date=None,
            rationale="Both changes use the publication date.",
        ),
    )
    generate_structured = MagicMock(return_value=generated)
    monkeypatch.setattr(drafter, "generate_structured", generate_structured)

    proposal = pipeline.draft_instruction_group_proposal(
        MagicMock(),
        instruction_indices=[2, 5],
        instructions=instructions,
        matches=matches,
        reference_date="2026-08-27",
        context=context,
    )

    generate_structured.assert_called_once()
    system_prompt = generate_structured.call_args.kwargs["system_prompt"].lower()
    user_prompt = generate_structured.call_args.kwargs["user_prompt"]
    assert "one full replacement chunk" in system_prompt
    assert "The amount is changed to 500 lira." in user_prompt
    assert "The filing period is changed to 30 days." in user_prompt
    assert "Yayımı   Tarihinden İtibaren" in user_prompt
    assert "yayımı tarihinden itibaren" in user_prompt
    assert proposal.instruction_index == 2
    assert proposal.instruction_text == instructions[0].instruction_text
    assert proposal.instruction_indices == [2, 5]
    assert proposal.instruction_texts == [
        instruction.instruction_text for instruction in instructions
    ]
    assert proposal.new_chunk_draft["text"] == generated.new_chunk.text
    assert proposal.match_confidence == 0.73
    assert "amount row" in (proposal.match_rationale or "")
    assert "period row" in (proposal.match_rationale or "")


@pytest.mark.parametrize("invalid", [None, "outside_scope", "index", "coverage"])
def test_multi_chunk_scope_becomes_one_atomic_editable_proposal(
    monkeypatch: pytest.MonkeyPatch,
    invalid: str | None,
) -> None:
    instructions = [
        AmendmentInstruction(instruction_text="EK-2 sıra 3 adı değiştirilmiştir."),
        AmendmentInstruction(instruction_text="EK-2 sıra 8 GTİP değiştirilmiştir."),
    ]
    matches = [
        MatchResult(old_chunk_id="first", confidence=0.9, rationale="row 3"),
        MatchResult(old_chunk_id="second", confidence=0.8, rationale="row 8"),
    ]

    def context(chunk_id: str, position: int) -> pipeline.InstructionDraftContext:
        snapshot = {
            "id": chunk_id,
            "user_file_id": "00000000-0000-0000-0000-000000000123",
            "position": position,
            "text": f"old {chunk_id}",
            "chunk_type": "table",
            "heading_path": ["EK-2"],
            "metadata": {"appendix_label": "EK-2"},
        }
        return pipeline.InstructionDraftContext(
            match=MatchResult(
                old_chunk_id=chunk_id, confidence=0.9, rationale="appendix scope"
            ),
            old_chunk_snapshot=snapshot,
            target_user_file_id=UUID("00000000-0000-0000-0000-000000000123"),
            target_position=position,
            sibling_reference=None,
            base_metadata={"appendix_label": "EK-2"},
            base_heading_path=["EK-2"],
        )

    generated = MultiChunkDraftResult(
        changes=[
            MultiChunkFieldsDraft(
                old_chunk_id="first",
                instruction_indexes=[0],
                new_chunk=ChunkFieldsDraft(text="new first", chunk_type="table"),
            ),
            MultiChunkFieldsDraft(
                old_chunk_id="second",
                instruction_indexes=[1],
                new_chunk=ChunkFieldsDraft(text="new second", chunk_type="table"),
            ),
        ],
        dates=DateResolution(
            effective_start_date="2026-09-20", rationale="publication"
        ),
    )
    monkeypatch.setattr(
        pipeline, "draft_multi_chunk_scope", MagicMock(return_value=generated)
    )

    if invalid is not None:
        if invalid == "outside_scope":
            generated.changes[0].old_chunk_id = "other-source"
        elif invalid == "index":
            generated.changes[0].instruction_indexes = [2]
        else:
            generated.changes = generated.changes[:1]
        with pytest.raises(DraftIntegrityError):
            pipeline.draft_multi_chunk_group_proposal(
                MagicMock(),
                instruction_indices=[17, 18],
                instructions=instructions,
                matches=matches,
                contexts=[context("first", 1), context("second", 2)],
                reference_date="2026-09-20",
            )
        return

    proposal = pipeline.draft_multi_chunk_group_proposal(
        MagicMock(),
        instruction_indices=[17, 18],
        instructions=instructions,
        matches=matches,
        contexts=[context("first", 1), context("second", 2)],
        reference_date="2026-09-20",
    )

    assert proposal.instruction_indices == [17, 18]
    assert [change.old_chunk_id for change in proposal.chunk_changes] == [
        "first",
        "second",
    ]
    assert [change.new_chunk_draft["text"] for change in proposal.chunk_changes] == [
        "new first",
        "new second",
    ]
    assert proposal.instruction_index == 17
    assert proposal.instruction_texts == [
        instruction.instruction_text for instruction in instructions
    ]
    assert proposal.match_confidence == 0.8


def _article_20_context(
    *, has_active_descendants: bool = False
) -> pipeline.InstructionDraftContext:
    heading_path = [
        "GÜMRÜK GENEL TEBLİĞİ (TIR İşlemleri) (Seri No: 1)",
        "ÜÇÜNCÜ BÖLÜM",
        "MADDE 20 - TIR karnesi himayesinde eşya taşıma yöntemleri",
        "(1) Bir TIR taşımasında hareket ve varış gümrük idarelerinin toplam sayısı yediyi geçemez",
    ]
    metadata = {
        "article_no": "20",
        "paragraph_no": "1",
        "heading_path": heading_path,
    }
    return pipeline.InstructionDraftContext(
        match=MatchResult(
            old_chunk_id="article-20-v2", confidence=1, rationale="exact"
        ),
        old_chunk_snapshot={
            "id": "article-20-v2",
            "text": "**MADDE 20 -** (1) Bir TIR taşımasında hareket ve varış gümrük idarelerinin toplam sayısı yediyi geçemez.",
            "chunk_type": "paragraph",
            "heading_path": heading_path,
            "metadata": metadata,
        },
        target_user_file_id=UUID("00000000-0000-0000-0000-000000000123"),
        target_position=132,
        sibling_reference=None,
        base_metadata=metadata,
        base_heading_path=heading_path,
        has_active_descendants=has_active_descendants,
    )


def test_explicit_replacement_draft_must_contain_authoritative_new_body() -> None:
    instruction = AmendmentInstruction(
        instruction_text=(
            "Aynı Tebliğin 20 nci maddesinin birinci fıkrası aşağıdaki şekilde "
            "değiştirilmiştir. “(1) Bir TIR taşımasında hareket ve varış "
            "gümrük idarelerinin toplam sayısı sekizi geçemez.”"
        )
    )
    unchanged_draft = DraftResult(
        new_chunk=ChunkFieldsDraft(
            text="**MADDE 20 -** (1) Bir TIR taşımasında hareket ve varış gümrük idarelerinin toplam sayısı yediyi geçemez."
        ),
        dates=DateResolution(rationale="publication date"),
    )

    with pytest.raises(DraftIntegrityError, match="explicit replacement"):
        pipeline._build_proposal_draft(
            instruction_indices=[0],
            instructions=[instruction],
            matches=[_article_20_context().match],
            context=_article_20_context(),
            draft=unchanged_draft,
        )


def test_existing_chunk_type_and_heading_follow_the_amended_text() -> None:
    instruction = AmendmentInstruction(
        instruction_text=(
            "Aynı Tebliğin 20 nci maddesinin birinci fıkrası aşağıdaki şekilde "
            "değiştirilmiştir. “(1) Bir TIR taşımasında hareket ve varış "
            "gümrük idarelerinin toplam sayısı sekizi geçemez.”"
        )
    )
    amended_text = (
        "**MADDE 20 -** (1) Bir TIR taşımasında hareket ve varış gümrük "
        "idarelerinin toplam sayısı sekizi geçemez."
    )
    draft = DraftResult(
        new_chunk=ChunkFieldsDraft(
            text=amended_text,
            chunk_type="clause",
            heading_path=["INVENTED DOCUMENT", "INVENTED ARTICLE"],
            metadata_changes={"clause_label": "a", "subclause_label": "i"},
        ),
        dates=DateResolution(rationale="publication date"),
    )

    proposal = pipeline._build_proposal_draft(
        instruction_indices=[0],
        instructions=[instruction],
        matches=[_article_20_context().match],
        context=_article_20_context(),
        draft=draft,
    )

    expected_heading = [
        "GÜMRÜK GENEL TEBLİĞİ (TIR İşlemleri) (Seri No: 1)",
        "ÜÇÜNCÜ BÖLÜM",
        "MADDE 20 - TIR karnesi himayesinde eşya taşıma yöntemleri",
        "(1) Bir TIR taşımasında hareket ve varış gümrük idarelerinin toplam sayısı sekizi geçemez",
    ]
    assert proposal.new_chunk_draft["chunk_type"] == "paragraph"
    assert "clause_label" not in proposal.new_chunk_draft["metadata"]
    assert "subclause_label" not in proposal.new_chunk_draft["metadata"]
    assert proposal.new_chunk_draft["heading_path"] == expected_heading
    assert proposal.new_chunk_draft["metadata"]["heading_path"] == expected_heading


def test_incomplete_full_replacement_of_parent_with_descendants_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instruction = AmendmentInstruction(
        instruction_text=(
            "20 nci maddenin üçüncü fıkrası aşağıdaki şekilde değiştirilmiştir. "
            "“(3) Yediden fazla idare için: a) Birinci yöntem... b) İkinci yöntem...”"
        )
    )
    generate_structured = MagicMock()
    monkeypatch.setattr(drafter, "generate_structured", generate_structured)

    with pytest.raises(DraftIntegrityError, match="descendant chunks"):
        pipeline.draft_instruction_group_proposal(
            MagicMock(),
            instruction_indices=[0],
            instructions=[instruction],
            matches=[_article_20_context(has_active_descendants=True).match],
            reference_date="2026-07-04",
            context=_article_20_context(has_active_descendants=True),
        )

    generate_structured.assert_not_called()


@pytest.mark.parametrize(
    ("heading_path", "text", "chunk_type", "labels", "expected"),
    [
        (
            ["KANUN", "MADDE 20 - Eski başlık"],
            "MADDE 20 - Yeni hüküm.",
            "article",
            {"article_no": "20", "article_title": "Yeni başlık"},
            ["KANUN", "MADDE 20 - Yeni başlık"],
        ),
        (
            ["KANUN", "GEÇİCİ MADDE 1 - Eski başlık"],
            "GEÇİCİ MADDE 1 - Yeni hüküm.",
            "article",
            {"article_no": "GEÇİCİ 1", "article_title": "Yeni başlık"},
            ["KANUN", "GEÇİCİ MADDE 1 - Yeni başlık"],
        ),
        (
            ["KANUN", "MÜKERRER MADDE 2"],
            "MÜKERRER MADDE 2 - Yeni hüküm.",
            "article",
            {"article_no": "MÜKERRER 2"},
            ["KANUN", "MÜKERRER MADDE 2"],
        ),
        (
            ["MADDE 20", "(3) Yöntemler", "a) Eski yöntem"],
            "a) Yeni yöntem uygulanır.",
            "clause",
            {"clause_label": "a"},
            ["MADDE 20", "(3) Yöntemler", "a) Yeni yöntem uygulanır"],
        ),
        (
            ["MADDE 20", "a) Yöntem", "(i) Eski alt bent"],
            "(i) Yeni alt bent uygulanır.",
            "subclause",
            {"clause_label": "a", "subclause_label": "i"},
            ["MADDE 20", "a) Yöntem", "(i) Yeni alt bent uygulanır"],
        ),
        (
            [],
            "(1) Yeni paragraf uygulanır.",
            "paragraph",
            {"paragraph_no": "1"},
            ["(1) Yeni paragraf uygulanır"],
        ),
    ],
)
def test_reconcile_existing_heading_path_for_structural_types(
    heading_path: list[str],
    text: str,
    chunk_type: str,
    labels: dict[str, str],
    expected: list[str],
) -> None:
    assert (
        reconcile_existing_heading_path(
            heading_path,
            amended_text=text,
            chunk_type=chunk_type,
            article_no=labels.get("article_no"),
            article_title=labels.get("article_title"),
            paragraph_no=labels.get("paragraph_no"),
            clause_label=labels.get("clause_label"),
            subclause_label=labels.get("subclause_label"),
        )
        == expected
    )


def test_multichunk_conflicting_dates_stop_before_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generate = MagicMock()
    monkeypatch.setattr(drafter, "generate_structured", generate)
    with pytest.raises(RuntimeError, match="incompatible"):
        drafter.draft_multi_chunk_scope(
            MagicMock(),
            instructions=[
                AmendmentInstruction(instruction_text="A", raw_date_phrase="1/9/2026"),
                AmendmentInstruction(
                    instruction_text="B", raw_date_phrase="yayımı tarihinde"
                ),
            ],
            old_chunks=[{"id": "a"}, {"id": "b"}],
            reference_date="2026-09-22",
        )
    generate.assert_not_called()


@pytest.mark.parametrize(
    ("metadata_article", "heading", "valid"),
    [
        ("EK 3", "EK MADDE 3", True),
        ("3", "EK MADDE 3", False),
        ("EK 3", "MADDE 3", False),
    ],
)
def test_verified_new_unit_identity_survives_generation(
    metadata_article: str, heading: str, valid: bool
) -> None:
    from onyx.regulatory.amendments.insertion_order import OrderMember

    context = pipeline.InstructionDraftContext(
        match=MatchResult(
            old_chunk_id=None, confidence=1.0, rationale="verified parent"
        ),
        old_chunk_snapshot={},
        target_user_file_id=UUID("00000000-0000-0000-0000-000000000123"),
        target_position=10,
        sibling_reference={
            "metadata": {"article_no": "EK 3"},
            "heading_path": ["EK MADDE 3"],
        },
        base_metadata={},
        base_heading_path=[],
        insertion_members=[
            OrderMember(id="parent", position=9, article_no="EK 3", paragraph_no="1")
        ],
    )
    instruction = AmendmentInstruction(
        instruction_text="3713 sayılı Kanunun ek 3 üncü maddesine aşağıdaki fıkra eklenmiştir. “Yeni metin.”"
    )
    draft = DraftResult(
        new_chunk=ChunkFieldsDraft(
            text="Yeni metin.",
            chunk_type="paragraph",
            heading_path=[heading],
            metadata_changes={"article_no": metadata_article},
        ),
        dates=DateResolution(rationale="date"),
    )
    if valid:
        proposal = pipeline._build_proposal_draft(
            instruction_indices=[0],
            instructions=[instruction],
            matches=[context.match],
            context=context,
            draft=draft,
        )
        assert proposal.new_chunk_draft["metadata"]["article_no"] == "EK 3"
    else:
        with pytest.raises(DraftIntegrityError, match="identity"):
            pipeline._build_proposal_draft(
                instruction_indices=[0],
                instructions=[instruction],
                matches=[context.match],
                context=context,
                draft=draft,
            )


def test_unnumbered_paragraph_after_an_explicit_third_paragraph_gets_fourth_identity() -> (
    None
):
    instruction = AmendmentInstruction(
        instruction_text="MADDE 4- 3713 sayılı Kanunun ek 3 üncü maddesine üçüncü fıkrasından sonra gelmek üzere aşağıdaki fıkra eklenmiştir. “Yeni hüküm.”"
    )
    result = pipeline._added_unit_structure(
        instruction,
        "Yeni hüküm.",
        {"heading_path": ["Kanun", "EK MADDE 3", "(3) Mevcut hüküm"]},
    )
    assert result is not None
    metadata, path = result
    assert metadata == {"article_no": "EK 3", "paragraph_no": "4"}
    assert path[-1].startswith("(4)")


@pytest.mark.parametrize("label", ["m", "ğ", "ı"])
def test_added_clause_uses_the_explicit_parent_and_body_identity(label: str) -> None:
    context = pipeline.InstructionDraftContext(
        match=MatchResult(old_chunk_id=None, confidence=1, rationale="verified parent"),
        old_chunk_snapshot={},
        target_user_file_id=UUID("00000000-0000-0000-0000-000000000123"),
        target_position=10,
        sibling_reference={
            "metadata": {"article_no": "8", "paragraph_no": "2"},
            "heading_path": [
                "Kaynak",
                "MADDE 8 - Tanımlar",
                "(2) Tanımlar",
                "a) Önceki",
            ],
        },
        base_metadata={},
        base_heading_path=[],
        expected_new_article_no="8",
    )
    text = f"{label}) Yeni ve tam tanım."
    instruction = AmendmentInstruction(
        instruction_text=f"MADDE 4- Aynı Kanunun 8 inci maddesinin ikinci fıkrasına aşağıdaki bent eklenmiştir. “{text}”"
    )
    draft = DraftResult(
        new_chunk=ChunkFieldsDraft(
            text=text,
            chunk_type="clause",
            heading_path=["Kaynak", "MADDE 8", "a) Yanlış"],
            metadata_changes={
                "article_no": "8",
                "paragraph_no": "1",
                "clause_label": "a",
            },
        ),
        dates=DateResolution(rationale="date"),
    )
    proposal = pipeline._build_proposal_draft(
        instruction_indices=[0],
        instructions=[instruction],
        matches=[context.match],
        context=context,
        draft=draft,
    )
    result = proposal.new_chunk_draft
    assert result["metadata"]["paragraph_no"] == "2"
    assert result["metadata"]["clause_label"] == label
    assert result["heading_path"][:3] == [
        "Kaynak",
        "MADDE 8 - Tanımlar",
        "(2) Tanımlar",
    ]
    assert result["heading_path"][-1].startswith(f"{label}) ")
    assert result["text"] == text


def test_new_unit_cannot_omit_part_of_the_supplied_addition() -> None:
    from onyx.regulatory.amendments.draft_integrity import (
        validate_explicit_replacements,
    )

    instruction = AmendmentInstruction(
        instruction_text="MADDE 4- Aynı Kanunun 8 inci maddesine aşağıdaki fıkra eklenmiştir. “(4) Başvuru yapılır ve onay beklenir.”"
    )
    with pytest.raises(DraftIntegrityError, match="body"):
        validate_explicit_replacements([instruction], "(4) Başvuru yapılır.")


@pytest.mark.parametrize("article_no", ["20", "GEÇİCİ 20"])
def test_new_temporary_article_preserves_expected_namespace(article_no: str) -> None:
    context = pipeline.InstructionDraftContext(
        match=MatchResult(old_chunk_id=None, confidence=1, rationale="new"),
        old_chunk_snapshot={},
        target_user_file_id=UUID("00000000-0000-0000-0000-000000000123"),
        target_position=11,
        sibling_reference=None,
        base_metadata={},
        base_heading_path=[],
        expected_new_article_no="GEÇİCİ 20",
    )
    instruction = AmendmentInstruction(
        instruction_text="3713 sayılı Kanuna aşağıdaki geçici madde eklenmiştir. “GEÇİCİ MADDE 20- Yeni metin.”"
    )
    draft = DraftResult(
        new_chunk=ChunkFieldsDraft(
            text="GEÇİCİ MADDE 20- Yeni metin.",
            chunk_type="article",
            heading_path=["GEÇİCİ MADDE 20"],
            metadata_changes={"article_no": article_no},
        ),
        dates=DateResolution(rationale="date"),
    )
    if article_no == "20":
        with pytest.raises(DraftIntegrityError, match="identity"):
            pipeline._build_proposal_draft(
                instruction_indices=[0],
                instructions=[instruction],
                matches=[context.match],
                context=context,
                draft=draft,
            )
    else:
        result = pipeline._build_proposal_draft(
            instruction_indices=[0],
            instructions=[instruction],
            matches=[context.match],
            context=context,
            draft=draft,
        )
        assert result.new_chunk_draft["metadata"]["article_no"] == article_no


def test_added_body_ignores_addition_verb_inside_a_quoted_heading() -> None:
    from onyx.regulatory.amendments.draft_integrity import explicit_added_body

    instruction = "MADDE 1- Aynı Kanunun 8 inci maddesinin başlığı “Yeni kayıt eklenmiştir” şeklinde değiştirilmiş ve maddeye aşağıdaki fıkra eklenmiştir. “(4) Yeni kural.”"
    assert explicit_added_body(instruction) == "(4) Yeni kural."


def test_direct_article_clause_proposal_keeps_unnumbered_parent() -> None:
    from uuid import uuid4

    from onyx.regulatory.amendments.insertion_order import OrderMember
    from onyx.regulatory.amendments.models import (
        AmendmentInstruction,
        ChunkFieldsDraft,
        DateResolution,
        DraftResult,
        MatchResult,
    )
    from onyx.regulatory.amendments.pipeline import (
        InstructionDraftContext,
        _build_proposal_draft,
    )

    instruction = AmendmentInstruction(
        instruction_text="MADDE 4- Aynı Kanunun 8 inci maddesine aşağıdaki bent eklenmiştir. “c) Yeni kural.”"
    )
    match = MatchResult(
        old_chunk_id=None, outcome="new_provision", confidence=1, rationale="new clause"
    )
    context = InstructionDraftContext(
        match=match,
        old_chunk_snapshot={},
        target_user_file_id=uuid4(),
        target_position=99,
        sibling_reference={
            "text": "MADDE 8- Tanımlar:",
            "metadata": {"article_no": "8"},
            "heading_path": ["Kaynak", "MADDE 8"],
        },
        base_metadata={},
        base_heading_path=[],
        expected_new_article_no="8",
        insertion_members=[
            OrderMember(
                id="parent", position=0, article_no="8", direct_clause_parent=True
            ),
            OrderMember(id="a", position=1, article_no="8", clause_label="a"),
            OrderMember(id="b", position=2, article_no="8", clause_label="b"),
        ],
    )
    result = _build_proposal_draft(
        instruction_indices=[0],
        instructions=[instruction],
        matches=[match],
        context=context,
        draft=DraftResult(
            new_chunk=ChunkFieldsDraft(
                text="c) Yeni kural.",
                chunk_type="clause",
                heading_path=["Kaynak", "MADDE 8", "c) Yeni kural"],
                metadata_changes={"article_no": "8", "clause_label": "c"},
            ),
            dates=DateResolution(
                effective_start_date=None, effective_end_date=None, rationale="unknown"
            ),
        ),
    )
    assert result.new_chunk_draft["metadata"].get("paragraph_no") is None
    assert result.new_chunk_draft["position"] == 3
    assert result.new_chunk_draft["insertion_order"]["paragraph_no"] is None
