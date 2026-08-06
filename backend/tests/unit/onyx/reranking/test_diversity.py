from datetime import datetime, timezone

from onyx.configs.constants import DocumentSource
from onyx.context.search.models import InferenceChunk
from onyx.reranking.diversity import apply_soft_diversity
from onyx.reranking.payload import canonical_chunk_source


def _chunk(
    document_id: str,
    chunk_id: int,
    body: str,
    *,
    source: str | None = None,
) -> InferenceChunk:
    return InferenceChunk(
        chunk_id=chunk_id,
        blurb=body,
        content=body,
        source_links={0: source} if source is not None else None,
        image_file_id=None,
        section_continuation=False,
        document_id=document_id,
        source_type=DocumentSource.WEB,
        semantic_identifier=document_id,
        title=document_id,
        boost=0,
        score=1.0,
        hidden=False,
        metadata={},
        match_highlights=[],
        doc_summary="",
        chunk_context="",
        updated_at=datetime(2026, 8, 6, tzinfo=timezone.utc),
    )


def _scores(*items: tuple[InferenceChunk, float]) -> dict[tuple[str, int], float]:
    return {(chunk.document_id, chunk.chunk_id): score for chunk, score in items}


def test_three_complementary_chunks_from_one_document_survive() -> None:
    first = _chunk("kanun", 1, "Madde 1 kapsam ve tanımları düzenler.")
    second = _chunk("kanun", 4, "Madde 4 başvuru usulünü düzenler.")
    third = _chunk("kanun", 9, "Madde 9 istisnaları ve süreyi düzenler.")
    weak_other = _chunk("genelge", 1, "Konuya uzaktan değinen genel açıklama.")
    chunks = [first, second, third, weak_other]

    ranked = apply_soft_diversity(
        chunks=chunks,
        scores=_scores(
            (first, 0.99),
            (second, 0.97),
            (third, 0.95),
            (weak_other, 0.40),
        ),
        limit=3,
    )

    assert [(chunk.document_id, chunk.chunk_id) for chunk in ranked] == [
        ("kanun", 1),
        ("kanun", 4),
        ("kanun", 9),
    ]


def test_near_duplicate_is_penalized_without_a_source_quota() -> None:
    first = _chunk("kanun", 1, "Madde 5 başvuru süresi otuz gündür.")
    duplicate = _chunk(
        "teblig",
        2,
        "MADDE 5 — Başvuru süresi otuz gündür!",
        source="https://example.test/teblig",
    )
    complement = _chunk("kanun", 3, "Süre haklı sebep varsa eski hâle getirilir.")

    ranked = apply_soft_diversity(
        chunks=[first, duplicate, complement],
        scores=_scores((first, 0.99), (duplicate, 0.98), (complement, 0.96)),
        limit=3,
    )

    assert ranked == [first, complement, duplicate]


def test_source_bonus_never_promotes_a_noncompetitive_chunk() -> None:
    first = _chunk("kanun", 1, "Birinci güçlü ve bağımsız hüküm.")
    same_source = _chunk("kanun", 2, "İkinci güçlü ve bağımsız hüküm.")
    weak_other = _chunk(
        "genelge",
        1,
        "Zayıf başka kaynak.",
        source="https://other.test/doc",
    )

    ranked = apply_soft_diversity(
        chunks=[first, same_source, weak_other],
        scores=_scores((first, 0.99), (same_source, 0.90), (weak_other, 0.20)),
        limit=2,
    )

    assert ranked == [first, same_source]


def test_missing_scores_and_ties_keep_stable_input_order() -> None:
    chunks = [
        _chunk("first", 1, "Birinci farklı metin."),
        _chunk("second", 2, "İkinci farklı metin."),
        _chunk("third", 3, "Üçüncü farklı metin."),
    ]

    assert apply_soft_diversity(chunks=chunks, scores={}, limit=3) == chunks
    assert apply_soft_diversity(chunks=chunks, scores={}, limit=0) == []


def test_canonical_source_normalizes_lowest_nonempty_url_or_uses_document_id() -> None:
    linked = _chunk("doc", 1, "Metin")
    linked.source_links = {
        0: "",
        4: "HTTPS://MEVZUAT.EXAMPLE/kanun/5/#bolum",
        8: "https://other.test/later",
    }
    unlinked = _chunk("fallback-doc", 2, "Metin")

    assert canonical_chunk_source(linked) == "https://mevzuat.example/kanun/5"
    assert canonical_chunk_source(unlinked) == "fallback-doc"
