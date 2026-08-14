import json
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from onyx.llm.models import ReasoningEffort
from onyx.prompts.regulatory_evidence_matrix import (
    REGULATORY_EVIDENCE_MATRIX_SYSTEM_PROMPT,
)
from onyx.regulatory.candidate_answer_review import (
    build_candidate_answer_evidence_chunk,
)
from onyx.regulatory.evidence_matrix import (
    EvidenceCoverageStatus,
    RegulatoryEvidenceMatrix,
    RegulatoryEvidenceMatrixRow,
    RegulatoryNavigationLead,
    build_regulatory_evidence_matrix,
    evidence_matrix_recovery_queries,
    format_regulatory_evidence_matrix,
    merge_regulatory_evidence_matrices,
)
from onyx.tracing.flows import LLMFlow


def _evidence(retrieval_number: int = 7):
    return build_candidate_answer_evidence_chunk(
        document_id="document-id",
        chunk_id=3,
        citation_number=None,
        retrieval_number=retrieval_number,
        chunk_identifier="chunk-id",
        heading="Instrument > Provision",
        research_target="Specific evidence target: operative prerequisite.",
        content="Exact operative text.",
    )


def test_matrix_contract_can_trace_exact_evidence_to_explicit_request_facts() -> None:
    normalized_prompt = " ".join(REGULATORY_EVIDENCE_MATRIX_SYSTEM_PROMPT.split())
    assert "expressly supplied fact, actor, or relationship" in normalized_prompt
    assert "exact text directly governs that same request element" in normalized_prompt
    for dataset_term in ("promil", "HGS", "situs", "1.500", "m.112", "m.114"):
        assert dataset_term not in REGULATORY_EVIDENCE_MATRIX_SYSTEM_PROMPT


def test_supported_matrix_row_requires_a_document() -> None:
    with pytest.raises(ValidationError):
        RegulatoryEvidenceMatrixRow(
            target="Operative prerequisite",
            status=EvidenceCoverageStatus.SUPPORTED,
            supported_proposition="The prerequisite applies.",
        )


def test_build_matrix_filters_unknown_document_and_marks_row_missing() -> None:
    llm = MagicMock()
    draft = RegulatoryEvidenceMatrix(
        rows=[
            RegulatoryEvidenceMatrixRow(
                target="Operative prerequisite",
                status=EvidenceCoverageStatus.SUPPORTED,
                supported_proposition="The prerequisite applies.",
                document_numbers=[999],
                target_ids=["T1"],
            )
        ]
    )

    with patch(
        "onyx.regulatory.evidence_matrix.generate_structured", return_value=draft
    ) as generate:
        matrix = build_regulatory_evidence_matrix(
            llm,
            user_request="How does the procedure operate?",
            coverage_contract="Target the operative prerequisite.",
            evidence_chunks=[_evidence()],
            navigation_leads=[
                RegulatoryNavigationLead(
                    document_title="Instrument",
                    article_key="part::article:9",
                    heading_label="PART II > Article 9 - Subsequent operation",
                    research_targets=["How does the procedure operate?"],
                )
            ],
        )

    assert matrix is not None
    assert matrix.rows[0].status is EvidenceCoverageStatus.MISSING
    assert matrix.rows[0].document_numbers == []
    call = generate.call_args.kwargs
    assert call["flow"] is LLMFlow.REGULATORY_EVIDENCE_MATRIX
    assert call["reasoning_effort"] is ReasoningEffort.HIGH
    payload = json.loads(call["user_prompt"])
    assert payload["evidence_chunks"][0]["document"] == 7
    assert payload["evidence_chunks"][0]["content"] == "Exact operative text."
    assert payload["evidence_chunks"][0]["target_ids"] == ["T1"]
    assert payload["research_targets"] == [
        {
            "target_id": "T1",
            "target": "Specific evidence target: operative prerequisite.",
        }
    ]
    assert payload["navigation_leads"] == [
        {
            "document_title": "Instrument",
            "article_key": "part::article:9",
            "heading_label": "PART II > Article 9 - Subsequent operation",
            "research_targets": ["How does the procedure operate?"],
        }
    ]


def test_build_matrix_recovers_provider_status_without_documents() -> None:
    llm = MagicMock()
    provider_draft = MagicMock()
    provider_draft.rows = [
        MagicMock(
            target="Operative deadline",
            target_ids=["T1"],
            status=EvidenceCoverageStatus.SUPPORTED,
            supported_proposition="A deadline applies.",
            document_numbers=[],
            missing_aspects=[],
            recovery_query=None,
        )
    ]

    with patch(
        "onyx.regulatory.evidence_matrix.generate_structured",
        return_value=provider_draft,
    ):
        matrix = build_regulatory_evidence_matrix(
            llm,
            user_request="What is the operative deadline?",
            coverage_contract=None,
            evidence_chunks=[_evidence()],
        )

    assert matrix is not None
    assert matrix.rows[0].status is EvidenceCoverageStatus.MISSING
    assert matrix.rows[0].document_numbers == []
    assert matrix.rows[0].recovery_query == "Operative deadline"


def test_recovery_queries_are_open_only_and_deduplicated() -> None:
    matrix = RegulatoryEvidenceMatrix(
        rows=[
            RegulatoryEvidenceMatrixRow(
                target="Supported row",
                status=EvidenceCoverageStatus.SUPPORTED,
                supported_proposition="Supported.",
                document_numbers=[1],
            ),
            RegulatoryEvidenceMatrixRow(
                target="First open row",
                status=EvidenceCoverageStatus.PARTIAL,
                recovery_query="Instrument transfer deadline",
            ),
            RegulatoryEvidenceMatrixRow(
                target="Duplicate open row",
                status=EvidenceCoverageStatus.MISSING,
                recovery_query="  instrument   transfer deadline  ",
            ),
            RegulatoryEvidenceMatrixRow(
                target="Second open row",
                status=EvidenceCoverageStatus.MISSING,
                recovery_query="Instrument domestic prerequisite",
            ),
        ]
    )

    assert evidence_matrix_recovery_queries(matrix, limit=10) == [
        "Instrument transfer deadline",
        "Instrument domestic prerequisite",
    ]


def test_formatted_matrix_is_explicitly_advisory() -> None:
    matrix = RegulatoryEvidenceMatrix(
        rows=[
            RegulatoryEvidenceMatrixRow(
                target="Operative prerequisite",
                status=EvidenceCoverageStatus.SUPPORTED,
                supported_proposition="The prerequisite applies.",
                document_numbers=[7],
            )
        ]
    )

    formatted = format_regulatory_evidence_matrix(matrix)

    assert formatted is not None
    assert "AI-generated evidence analysis, not legal evidence" in formatted
    assert '"document_numbers": [7]' in formatted


def test_independent_matrix_union_preserves_distinct_same_target_effects() -> None:
    primary = RegulatoryEvidenceMatrix(
        rows=[
            RegulatoryEvidenceMatrixRow(
                target="Administrative consequence",
                target_ids=["T1"],
                status=EvidenceCoverageStatus.SUPPORTED,
                supported_proposition="The text establishes an administrative effect.",
                document_numbers=[3],
            )
        ]
    )
    independent = RegulatoryEvidenceMatrix(
        rows=[
            RegulatoryEvidenceMatrixRow(
                target="Judicial consequence",
                target_ids=["T1"],
                status=EvidenceCoverageStatus.SUPPORTED,
                supported_proposition="The text establishes a distinct judicial effect.",
                document_numbers=[4],
            )
        ]
    )

    merged = merge_regulatory_evidence_matrices(primary, independent)

    assert merged is not None
    assert [row.target for row in merged.rows] == [
        "Administrative consequence",
        "Judicial consequence",
    ]
    assert [row.target_ids for row in merged.rows] == [["T1"], ["T1"]]


def test_build_matrix_preserves_distinct_effects_with_the_same_target() -> None:
    llm = MagicMock()
    provider_draft = RegulatoryEvidenceMatrix(
        rows=[
            RegulatoryEvidenceMatrixRow(
                target="Consequences for the supplied act",
                target_ids=["T1"],
                status=EvidenceCoverageStatus.SUPPORTED,
                supported_proposition="The text establishes an administrative effect.",
                document_numbers=[7],
            ),
            RegulatoryEvidenceMatrixRow(
                target="Consequences for the supplied act",
                target_ids=["T1"],
                status=EvidenceCoverageStatus.SUPPORTED,
                supported_proposition="The text establishes a distinct judicial effect.",
                document_numbers=[8],
            ),
        ]
    )

    with patch(
        "onyx.regulatory.evidence_matrix.generate_structured",
        return_value=provider_draft,
    ):
        matrix = build_regulatory_evidence_matrix(
            llm,
            user_request="What consequences follow from the supplied act?",
            coverage_contract=None,
            evidence_chunks=[_evidence(7), _evidence(8)],
        )

    assert matrix is not None
    assert [row.supported_proposition for row in matrix.rows] == [
        "The text establishes an administrative effect.",
        "The text establishes a distinct judicial effect.",
    ]
    assert [row.document_numbers for row in matrix.rows] == [[7], [8]]


def test_uncovered_retrieval_target_does_not_create_an_answer_obligation() -> None:
    llm = MagicMock()
    with patch(
        "onyx.regulatory.evidence_matrix.generate_structured",
        return_value=RegulatoryEvidenceMatrix(),
    ):
        matrix = build_regulatory_evidence_matrix(
            llm,
            user_request="How does the procedure operate?",
            coverage_contract=None,
            evidence_chunks=[_evidence()],
        )

    assert matrix is not None
    assert matrix.rows == []


def test_matrix_rejects_targets_not_derived_from_retrieved_evidence() -> None:
    with pytest.raises(TypeError):
        cast(Any, build_regulatory_evidence_matrix)(
            MagicMock(),
            user_request="What consequence follows?",
            coverage_contract=None,
            evidence_chunks=[_evidence()],
            required_targets=["Resolve the expressly requested consequence."],
        )


def test_failed_refresh_preserves_the_prior_matrix() -> None:
    llm = MagicMock()
    prior = RegulatoryEvidenceMatrix(
        rows=[
            RegulatoryEvidenceMatrixRow(
                target="Open prerequisite",
                target_ids=["T1"],
                status=EvidenceCoverageStatus.MISSING,
                recovery_query="operative prerequisite",
            )
        ]
    )
    with patch(
        "onyx.regulatory.evidence_matrix.generate_structured",
        side_effect=RuntimeError("provider output failed"),
    ):
        matrix = build_regulatory_evidence_matrix(
            llm,
            user_request="How does the procedure operate?",
            coverage_contract=None,
            evidence_chunks=[_evidence()],
            prior_matrix=prior,
        )

    assert matrix is prior


def test_refresh_merges_open_row_by_stable_target_id() -> None:
    llm = MagicMock()
    prior = RegulatoryEvidenceMatrix(
        rows=[
            RegulatoryEvidenceMatrixRow(
                target="Original open prerequisite",
                target_ids=["T1"],
                status=EvidenceCoverageStatus.MISSING,
                recovery_query="operative prerequisite",
            )
        ]
    )
    update = RegulatoryEvidenceMatrix(
        rows=[
            RegulatoryEvidenceMatrixRow(
                target="Paraphrased prerequisite",
                target_ids=["T1"],
                status=EvidenceCoverageStatus.SUPPORTED,
                supported_proposition="The exact text now closes the row.",
                document_numbers=[7],
            )
        ]
    )
    with patch(
        "onyx.regulatory.evidence_matrix.generate_structured", return_value=update
    ):
        matrix = build_regulatory_evidence_matrix(
            llm,
            user_request="How does the procedure operate?",
            coverage_contract=None,
            evidence_chunks=[_evidence()],
            prior_matrix=prior,
        )

    assert matrix is not None
    assert matrix.rows[0].status is EvidenceCoverageStatus.SUPPORTED
    assert matrix.rows[0].document_numbers == [7]


def test_refresh_preserves_prior_rows_and_appends_new_atomic_target() -> None:
    llm = MagicMock()
    prior = RegulatoryEvidenceMatrix(
        rows=[
            RegulatoryEvidenceMatrixRow(
                target="Existing supported target",
                target_ids=["T1"],
                status=EvidenceCoverageStatus.SUPPORTED,
                supported_proposition="The existing target is supported.",
                document_numbers=[7],
            )
        ]
    )
    update = RegulatoryEvidenceMatrix(
        rows=[
            RegulatoryEvidenceMatrixRow(
                target="New independently supported target",
                target_ids=["T2"],
                status=EvidenceCoverageStatus.SUPPORTED,
                supported_proposition="A distinct target is supported.",
                document_numbers=[8],
            )
        ]
    )
    second_evidence = build_candidate_answer_evidence_chunk(
        document_id="document-id",
        chunk_id=4,
        citation_number=None,
        retrieval_number=8,
        chunk_identifier="chunk-id-2",
        heading="Instrument > Other provision",
        research_target="Specific evidence target: distinct operative result.",
        content="Other exact operative text.",
    )
    with patch(
        "onyx.regulatory.evidence_matrix.generate_structured", return_value=update
    ):
        matrix = build_regulatory_evidence_matrix(
            llm,
            user_request="How do both targets operate?",
            coverage_contract=None,
            evidence_chunks=[_evidence(), second_evidence],
            prior_matrix=prior,
        )

    assert matrix is not None
    assert [row.target for row in matrix.rows] == [
        "Existing supported target",
        "New independently supported target",
    ]


def test_refresh_does_not_replace_supported_row_with_same_target_id() -> None:
    llm = MagicMock()
    prior_row = RegulatoryEvidenceMatrixRow(
        target="Existing supported effect",
        target_ids=["T1"],
        status=EvidenceCoverageStatus.SUPPORTED,
        supported_proposition="The first independent effect is supported.",
        document_numbers=[7],
    )
    update_row = RegulatoryEvidenceMatrixRow(
        target="New supported effect",
        target_ids=["T1"],
        status=EvidenceCoverageStatus.SUPPORTED,
        supported_proposition="A second independent effect is supported.",
        document_numbers=[8],
    )
    with patch(
        "onyx.regulatory.evidence_matrix.generate_structured",
        return_value=RegulatoryEvidenceMatrix(rows=[update_row]),
    ):
        matrix = build_regulatory_evidence_matrix(
            llm,
            user_request="What effects follow?",
            coverage_contract=None,
            evidence_chunks=[_evidence(7), _evidence(8)],
            prior_matrix=RegulatoryEvidenceMatrix(rows=[prior_row]),
        )

    assert matrix is not None
    assert matrix.rows == [prior_row, update_row]


def test_refresh_discards_rephrased_supported_rows_without_new_exact_evidence() -> None:
    llm = MagicMock()
    prior_row = RegulatoryEvidenceMatrixRow(
        target="Existing supported effect",
        target_ids=["T1"],
        status=EvidenceCoverageStatus.SUPPORTED,
        supported_proposition="The exact text establishes the effect.",
        document_numbers=[7],
    )
    rephrased_row = RegulatoryEvidenceMatrixRow(
        target="Restated supported effect",
        target_ids=["T1"],
        status=EvidenceCoverageStatus.SUPPORTED,
        supported_proposition="The effect is established by the exact text.",
        document_numbers=[7],
    )
    with patch(
        "onyx.regulatory.evidence_matrix.generate_structured",
        return_value=RegulatoryEvidenceMatrix(rows=[rephrased_row]),
    ):
        matrix = build_regulatory_evidence_matrix(
            llm,
            user_request="What effects follow?",
            coverage_contract=None,
            evidence_chunks=[_evidence(7)],
            prior_matrix=RegulatoryEvidenceMatrix(rows=[prior_row]),
        )

    assert matrix is not None
    assert matrix.rows == [prior_row]
