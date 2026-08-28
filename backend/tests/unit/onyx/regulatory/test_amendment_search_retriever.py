from unittest.mock import MagicMock
from uuid import UUID

from onyx.configs.constants import DocumentSource
from onyx.context.search.models import SearchDoc, SearchDocsResponse
from onyx.regulatory.amendments.models import AmendmentInstruction
from onyx.regulatory.amendments.ranker import CandidateChunk
from onyx.regulatory.amendments.search_retriever import AmendmentSearchRetriever
from onyx.tools.constants import REGULATORY_MAX_SEARCH_QUERY_CHARS
from onyx.tools.models import ToolResponse

_FILE_ID = UUID("00000000-0000-0000-0000-000000000123")


def _search_doc(*, file_id: str | None, chunk_id: str) -> SearchDoc:
    return SearchDoc(
        document_id="document-1",
        chunk_ind=7,
        semantic_identifier="Gümrük Genel Tebliği (TIR İşlemleri)",
        blurb=(
            "Bir TIR taşıması en fazla sekiz hareket ve varış gümrük "
            "idaresini kapsayabilir."
        ),
        source_type=DocumentSource.FILE,
        boost=1,
        hidden=False,
        metadata={
            "regulatory_chunk_id": chunk_id,
            "regulatory_heading_path": ["MADDE 6", "TIR karnesinin kapsamı"],
        },
        match_highlights=[],
        file_id=file_id,
    )


def test_amendment_retriever_runs_real_search_tool_shape_and_keeps_exact_chunk_id() -> (
    None
):
    # Real PC Külliyatı SearchTool responses carry the exact regulatory id even
    # when the generic SearchDoc file enrichment is absent.
    in_scope = _search_doc(file_id=None, chunk_id="tir-chunk-6")
    out_of_scope = _search_doc(
        file_id="00000000-0000-0000-0000-000000000999",
        chunk_id="foreign-chunk",
    )
    search_tool = MagicMock()
    search_tool.run.return_value = ToolResponse(
        rich_response=SearchDocsResponse(
            search_docs=[in_scope, out_of_scope],
            displayed_docs=[in_scope, out_of_scope],
            citation_mapping={},
        ),
        llm_facing_response="",
    )
    retriever = AmendmentSearchRetriever(
        search_tool_factory=lambda: search_tool,
        canonical_candidate_loader=lambda _chunk_ids: {
            "tir-chunk-6": CandidateChunk(
                chunk_id="tir-chunk-6",
                user_file_id=str(_FILE_ID),
                text="Canonical PostgreSQL text for MADDE 6.",
                metadata={"heading_path": ["MADDE 6", "TIR karnesinin kapsamı"]},
            ),
            "foreign-chunk": CandidateChunk(
                chunk_id="foreign-chunk",
                user_file_id="00000000-0000-0000-0000-000000000999",
                text=out_of_scope.blurb,
            ),
        },
        allowed_user_file_ids=[_FILE_ID],
    )
    instruction = AmendmentInstruction(
        instruction_text="Carnets can now cover up to 8 customs offices.",
        target_source="Gümrük Genel Tebliği (TIR İşlemleri) Seri No: 9",
        search_query=(
            "TIR karnesi kapsamında işlem yapılabilecek hareket ve varış "
            "gümrük idaresi sayısı kaçtır?"
        ),
        recovery_query="TIR karnesi gümrük idaresi azami sayı",
    )

    candidates = retriever.search(instruction)

    assert [candidate.chunk_id for candidate in candidates] == ["tir-chunk-6"]
    assert candidates[0].user_file_id == str(_FILE_ID)
    assert candidates[0].text == "Canonical PostgreSQL text for MADDE 6."
    assert candidates[0].metadata["heading_path"] == [
        "MADDE 6",
        "TIR karnesinin kapsamı",
    ]
    call = search_tool.run.call_args
    assert call.kwargs["queries"] == [instruction.search_query]
    assert call.kwargs["search_mode"] == "hybrid"
    assert call.kwargs["source_anchors"] == [instruction.target_source]
    assert call.kwargs["override_kwargs"].skip_query_expansion is False


def test_recovery_is_one_focused_search_without_query_expansion() -> None:
    search_tool = MagicMock()
    search_tool.run.return_value = ToolResponse(
        rich_response=SearchDocsResponse(
            search_docs=[], citation_mapping={}, displayed_docs=None
        ),
        llm_facing_response="",
    )
    retriever = AmendmentSearchRetriever(
        search_tool_factory=lambda: search_tool,
        canonical_candidate_loader=lambda _chunk_ids: {},
        allowed_user_file_ids=[_FILE_ID],
    )
    instruction = AmendmentInstruction(
        instruction_text="Risk-inspection criteria were updated.",
        search_query="TIR işlemlerinde risk kriterlerine göre muayene nasıl yapılır?",
        recovery_query="TIR risk kriterleri muayene kontrol",
    )

    retriever.search(instruction, recovery=True)

    call = search_tool.run.call_args
    assert call.kwargs["queries"] == [instruction.recovery_query]
    assert call.kwargs["override_kwargs"].skip_query_expansion is True


def test_legacy_checkpoint_instruction_fallback_respects_search_query_limit() -> None:
    search_tool = MagicMock()
    search_tool.run.return_value = ToolResponse(
        rich_response=SearchDocsResponse(
            search_docs=[], citation_mapping={}, displayed_docs=None
        ),
        llm_facing_response="",
    )
    retriever = AmendmentSearchRetriever(
        search_tool_factory=lambda: search_tool,
        canonical_candidate_loader=lambda _chunk_ids: {},
        allowed_user_file_ids=[_FILE_ID],
    )
    instruction = AmendmentInstruction(
        instruction_text="MADDE 20 " + "çok uzun eski checkpoint metni " * 100,
    )

    retriever.search(instruction)

    query = search_tool.run.call_args.kwargs["queries"][0]
    assert query == instruction.instruction_text[:REGULATORY_MAX_SEARCH_QUERY_CHARS]
    assert len(query) == REGULATORY_MAX_SEARCH_QUERY_CHARS


def test_explicit_clause_candidate_is_kept_when_search_tool_omits_it() -> None:
    search_result = _search_doc(file_id=None, chunk_id="article-3-intro")
    search_tool = MagicMock()
    search_tool.run.return_value = ToolResponse(
        rich_response=SearchDocsResponse(
            search_docs=[search_result],
            displayed_docs=[search_result],
            citation_mapping={},
        ),
        llm_facing_response="",
    )
    exact_clause = CandidateChunk(
        chunk_id="article-3-clause-u",
        user_file_id=str(_FILE_ID),
        text="u) Son üç yıl içinde ...",
        source_name="Gümrük Genel Tebliği (TIR İşlemleri) (Seri No: 1)",
        metadata={"article_no": "3", "clause_label": "u"},
    )
    retriever = AmendmentSearchRetriever(
        search_tool_factory=lambda: search_tool,
        canonical_candidate_loader=lambda _chunk_ids: {
            "article-3-intro": CandidateChunk(
                chunk_id="article-3-intro",
                user_file_id=str(_FILE_ID),
                text="MADDE 3 – Bu Tebliğde geçen ...",
            )
        },
        structural_candidate_loader=lambda _instruction: [exact_clause],
        allowed_user_file_ids=[_FILE_ID],
    )
    instruction = AmendmentInstruction(
        instruction_text=(
            "Gümrük Genel Tebliği (TIR İşlemleri) (Seri No: 1)’nin "
            "3 üncü maddesinin birinci fıkrasının (u) bendinde yer alan "
            "“iki” ibaresi “dört” şeklinde değiştirilmiştir."
        ),
        article_reference="Madde 3",
        target_source="Gümrük Genel Tebliği (TIR İşlemleri) (Seri No: 1)",
        search_query="TIR İşlemleri Tebliği Madde 3 birinci fıkra u bendi nedir?",
        recovery_query="TIR İşlemleri Madde 3 bent u",
    )

    candidates = retriever.search(instruction)

    assert [candidate.chunk_id for candidate in candidates] == [
        "article-3-intro",
        "article-3-clause-u",
    ]


def test_structural_expansion_is_disabled_without_distinguishing_source_identity() -> (
    None
):
    search_tool = MagicMock()
    search_tool.run.return_value = ToolResponse(
        rich_response=SearchDocsResponse(
            search_docs=[], citation_mapping={}, displayed_docs=None
        ),
        llm_facing_response="",
    )
    structural_loader = MagicMock(return_value=[_candidate_for_wrong_source()])
    retriever = AmendmentSearchRetriever(
        search_tool_factory=lambda: search_tool,
        canonical_candidate_loader=lambda _chunk_ids: {},
        structural_candidate_loader=structural_loader,
        allowed_user_file_ids=[_FILE_ID],
    )
    for target_source in (None, "Gümrük Genel Tebliği"):
        instruction = AmendmentInstruction(
            instruction_text="3 üncü maddesinin (u) bendi değiştirilmiştir.",
            article_reference="Madde 3",
            target_source=target_source,
        )

        assert retriever.search(instruction) == []
    structural_loader.assert_not_called()


def _candidate_for_wrong_source() -> CandidateChunk:
    return CandidateChunk(
        chunk_id="wrong-instrument-article-3-u",
        user_file_id=str(_FILE_ID),
        text="Another instrument's article 3 clause u.",
        source_name="Unrelated instrument",
        metadata={"article_no": "3", "clause_label": "u"},
        structured_match=True,
    )
