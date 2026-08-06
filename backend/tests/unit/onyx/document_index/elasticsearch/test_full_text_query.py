from typing import Any

import pytest

from onyx.context.search.models import IndexFilters
from onyx.document_index.elasticsearch.schema import (
    DECISION_NUMBERS_FIELD_NAME,
    HEADING_PATH_FIELD_NAME,
    LEGAL_DATES_FIELD_NAME,
    PROVISION_IDENTIFIERS_FIELD_NAME,
)
from onyx.document_index.elasticsearch.search import (
    FULL_TEXT_MINIMUM_SHOULD_MATCH,
    DocumentQuery,
)
from onyx.document_index.interfaces_new import TenantState


def _match_operators(query: dict[str, Any]) -> list[str]:
    clauses = query["bool"]["should"]
    return [
        next(iter(clause["match"].values()))["operator"]
        for clause in clauses
        if "match" in clause
    ]


def _source_boosts(query: dict[str, Any]) -> list[dict[str, Any]]:
    source_clause = next(
        clause["dis_max"]
        for clause in query["bool"]["should"]
        if "dis_max" in clause
        and all(
            "multi_match" in item.get("constant_score", {}).get("filter", {})
            for item in clause["dis_max"].get("queries", [])
        )
    )
    assert source_clause["tie_breaker"] == 0.0
    return [
        {
            **item["constant_score"]["filter"]["multi_match"],
            "boost": item["constant_score"]["boost"],
        }
        for item in source_clause["queries"]
    ]


def _required_heading_phrases(query: dict[str, Any]) -> set[str]:
    structural_options = query["bool"]["must"][0]["bool"]
    assert structural_options["minimum_should_match"] == 1
    return {
        clause["match_phrase"][HEADING_PATH_FIELD_NAME]["query"]
        for clause in structural_options["should"]
    }


def _scope_heading_boosts(query: dict[str, Any]) -> dict[str, float]:
    for clause in query["bool"]["should"]:
        raw_dis_max = clause.get("dis_max")
        if not isinstance(raw_dis_max, dict):
            continue
        raw_queries = raw_dis_max.get("queries")
        if not isinstance(raw_queries, list) or not raw_queries:
            continue
        if not all("constant_score" in item for item in raw_queries):
            continue
        return {
            item["constant_score"]["filter"]["match_phrase"][HEADING_PATH_FIELD_NAME][
                "query"
            ]: item["constant_score"]["boost"]
            for item in raw_queries
        }
    return {}


def _legal_exact_boosts(query: dict[str, Any]) -> dict[str, tuple[list[str], float]]:
    boosts: dict[str, tuple[list[str], float]] = {}
    for clause in query["bool"]["should"]:
        constant_score = clause.get("constant_score")
        if not isinstance(constant_score, dict):
            continue
        terms = constant_score.get("filter", {}).get("terms", {})
        if not isinstance(terms, dict) or len(terms) != 1:
            continue
        field_name, values = next(iter(terms.items()))
        boosts[field_name] = (values, constant_score["boost"])
    return boosts


def test_keyword_query_accepts_any_term_by_default() -> None:
    query = DocumentQuery._get_title_content_combined_keyword_search_query(
        "UND DAC global teminat"
    )

    assert _match_operators(query) == ["or", "or", "or"]
    assert any(
        HEADING_PATH_FIELD_NAME in clause.get("match", {})
        for clause in query["bool"]["should"]
    )


def test_full_text_query_requires_high_analyzed_term_coverage() -> None:
    query = DocumentQuery._get_title_content_combined_keyword_search_query(
        "UND DAC global teminat",
        high_term_coverage=True,
    )

    match_options = [
        next(iter(clause["match"].values()))
        for clause in query["bool"]["should"]
        if "match" in clause
    ]
    assert all("operator" not in options for options in match_options)
    assert {options["minimum_should_match"] for options in match_options} == {
        FULL_TEXT_MINIMUM_SHOULD_MATCH
    }
    assert sum("match_phrase" in clause for clause in query["bool"]["should"]) == 3

    cross_fields = next(
        clause["multi_match"]
        for clause in query["bool"]["should"]
        if "multi_match" in clause
    )
    assert cross_fields["type"] == "cross_fields"
    assert (
        cross_fields["minimum_should_match"]
        == FULL_TEXT_MINIMUM_SHOULD_MATCH
        == "2<75% 5<60%"
    )
    assert HEADING_PATH_FIELD_NAME in " ".join(cross_fields["fields"])


def test_regulatory_explicit_provision_adds_exact_heading_constraint() -> None:
    search_body = DocumentQuery.get_keyword_search_query(
        query_text="Basel Sözleşmesi Madde 8 tamamlanamayan hareket",
        num_hits=8,
        tenant_state=TenantState(tenant_id="public", multitenant=False),
        index_filters=IndexFilters(
            access_control_list=None,
            regulatory_chunks_only=True,
        ),
        include_hidden=False,
        high_term_coverage=True,
    )

    query = search_body["query"]
    structural_options = query["bool"]["must"][0]["bool"]
    phrases = {
        clause["match_phrase"][HEADING_PATH_FIELD_NAME]["query"]
        for clause in structural_options["should"]
    }
    assert structural_options["minimum_should_match"] == 1
    assert {"MADDE 8", "MADDE8", "8 MADDESİ", "ARTICLE 8"} <= phrases
    assert query["bool"]["minimum_should_match"] == 0
    assert any(
        clause.get("multi_match", {}).get("fuzziness") == "AUTO"
        for clause in query["bool"]["should"]
    )
    source_boost, distinctive_boost = _source_boosts(query)
    assert source_boost["query"] == "Basel Sözleşmesi"
    assert source_boost["minimum_should_match"] == FULL_TEXT_MINIMUM_SHOULD_MATCH
    assert source_boost["fuzziness"] == "AUTO"
    assert distinctive_boost["query"] == "Basel"
    assert distinctive_boost["minimum_should_match"] == "100%"
    assert distinctive_boost["fuzziness"] == "AUTO"


def test_non_regulatory_or_ambiguous_query_has_no_heading_constraint() -> None:
    tenant = TenantState(tenant_id="public", multitenant=False)
    non_regulatory = DocumentQuery.get_keyword_search_query(
        query_text="Basel Sözleşmesi Madde 8",
        num_hits=8,
        tenant_state=tenant,
        index_filters=IndexFilters(access_control_list=None),
        include_hidden=False,
        high_term_coverage=True,
    )["query"]
    ambiguous = DocumentQuery.get_keyword_search_query(
        query_text="Madde 6 ve Madde 8",
        num_hits=8,
        tenant_state=tenant,
        index_filters=IndexFilters(
            access_control_list=None,
            regulatory_chunks_only=True,
        ),
        include_hidden=False,
        high_term_coverage=True,
    )["query"]

    assert "must" not in non_regulatory["bool"]
    assert "must" not in ambiguous["bool"]


@pytest.mark.parametrize(
    ("query_text", "expected_source", "expected_distinctive"),
    [
        ("Basel Sözleşmesi Madde 8", "Basel Sözleşmesi", "Basel"),
        ("Article 8 Basel Convention", "Basel Convention", "Basel"),
    ],
)
def test_named_instrument_is_a_soft_ranking_signal_for_exact_provision(
    query_text: str,
    expected_source: str,
    expected_distinctive: str,
) -> None:
    search_body = DocumentQuery.get_keyword_search_query(
        query_text=query_text,
        num_hits=8,
        tenant_state=TenantState(tenant_id="public", multitenant=False),
        index_filters=IndexFilters(
            access_control_list=None,
            regulatory_chunks_only=True,
        ),
        include_hidden=False,
        high_term_coverage=True,
    )

    query = search_body["query"]
    assert {"MADDE 8", "ARTICLE 8"} <= _required_heading_phrases(query)
    source_boost, distinctive_boost = _source_boosts(query)
    assert source_boost["query"] == expected_source
    assert distinctive_boost["query"] == expected_distinctive
    assert source_boost["boost"] == distinctive_boost["boost"] == 8.0
    assert source_boost["fields"] == distinctive_boost["fields"]
    # Instrument identity influences ordering only. The sole hard constraint is
    # structural article identity, so a spelling or translation mismatch can
    # still return the right provision from another source.
    assert all(
        expected_distinctive.casefold()
        not in clause["match_phrase"][HEADING_PATH_FIELD_NAME]["query"].casefold()
        for clause in query["bool"]["must"][0]["bool"]["should"]
    )


def test_generic_exact_provision_does_not_assume_an_instrument() -> None:
    search_body = DocumentQuery.get_keyword_search_query(
        query_text="Madde 8",
        num_hits=8,
        tenant_state=TenantState(tenant_id="public", multitenant=False),
        index_filters=IndexFilters(
            access_control_list=None,
            regulatory_chunks_only=True,
        ),
        include_hidden=False,
        high_term_coverage=True,
    )

    query = search_body["query"]
    assert {"MADDE 8", "ARTICLE 8"} <= _required_heading_phrases(query)
    assert not any("dis_max" in clause for clause in query["bool"]["should"])


def test_compact_alphanumeric_provision_alias_builds_exact_structural_query() -> None:
    search_body = DocumentQuery.get_keyword_search_query(
        query_text="Basel md4a ihracat yasağı",
        num_hits=8,
        tenant_state=TenantState(tenant_id="public", multitenant=False),
        index_filters=IndexFilters(
            access_control_list=None,
            regulatory_chunks_only=True,
        ),
        include_hidden=False,
    )

    query = search_body["query"]
    assert {"MADDE 4A", "MADDE4A", "ARTICLE 4A"} <= _required_heading_phrases(query)
    assert _source_boosts(query)[0]["query"] == "basel"


def test_explicit_annex_scope_is_a_soft_structural_ranking_signal() -> None:
    search_body = DocumentQuery.get_keyword_search_query(
        query_text="Ortak Transit Sözleşmesi Ek IV Madde 15 zamanaşımı",
        num_hits=8,
        tenant_state=TenantState(tenant_id="public", multitenant=False),
        index_filters=IndexFilters(
            access_control_list=None,
            regulatory_chunks_only=True,
        ),
        include_hidden=False,
    )

    query = search_body["query"]
    assert {"MADDE 15", "ARTICLE 15"} <= _required_heading_phrases(query)
    assert _scope_heading_boosts(query) == {"EK IV": 32.0, "ANNEX IV": 32.0}
    # Scope does not become a hard filter because legacy chunks can omit an
    # enclosing annex label while still containing the correct provision.
    assert len(query["bool"]["must"]) == 1


def test_series_number_does_not_mask_the_requested_article() -> None:
    search_body = DocumentQuery.get_keyword_search_query(
        query_text="Transit Rejimi Tebliği Seri No 4 Madde 12 teminat",
        num_hits=8,
        tenant_state=TenantState(tenant_id="public", multitenant=False),
        index_filters=IndexFilters(
            access_control_list=None,
            regulatory_chunks_only=True,
        ),
        include_hidden=False,
    )

    query = search_body["query"]
    assert {"MADDE 12", "ARTICLE 12"} <= _required_heading_phrases(query)
    assert _scope_heading_boosts(query) == {
        "SERİ NO 4": 32.0,
        "SERI NO 4": 32.0,
        "SERIES NO 4": 32.0,
    }


def test_regulatory_concept_query_soft_boosts_named_instrument() -> None:
    search_body = DocumentQuery.get_keyword_search_query(
        query_text="Basel Sözleşmesi yasadışı trafik tanımı",
        num_hits=8,
        tenant_state=TenantState(tenant_id="public", multitenant=False),
        index_filters=IndexFilters(
            access_control_list=None,
            regulatory_chunks_only=True,
        ),
        include_hidden=False,
    )

    query = search_body["query"]
    required_lexical_options = query["bool"]["must"][0]["bool"]
    assert required_lexical_options["minimum_should_match"] == 1
    source_boost, distinctive_boost = _source_boosts(query)
    assert source_boost["query"] == "Basel Sözleşmesi"
    assert source_boost["fuzziness"] == "AUTO"
    assert distinctive_boost["query"] == "Basel"


@pytest.mark.parametrize("high_term_coverage", [False, True])
def test_source_title_match_alone_cannot_satisfy_lexical_requirement(
    high_term_coverage: bool,
) -> None:
    query = DocumentQuery._get_title_content_combined_keyword_search_query(
        "unrelated operative body terms",
        high_term_coverage=high_term_coverage,
        preferred_source_hint="Named Instrument",
        preferred_distinctive_source_hint="Instrument",
    )

    required_lexical_options = query["bool"]["must"][0]["bool"]
    assert required_lexical_options["minimum_should_match"] == 1
    assert required_lexical_options["should"]
    assert all("dis_max" not in clause for clause in required_lexical_options["should"])

    # The title/heading source hint remains an optional scoring signal. A chunk
    # whose title matches only this hint cannot satisfy the required group.
    source_boost, distinctive_boost = _source_boosts(query)
    assert source_boost["query"] == "Named Instrument"
    assert distinctive_boost["query"] == "Instrument"
    assert query["bool"]["minimum_should_match"] == 0


def test_regulatory_hybrid_query_boosts_named_instrument_in_lexical_lane() -> None:
    search_body = DocumentQuery.get_hybrid_search_query(
        query_text="Basel Sözleşmesi geri alma yükümlülüğü",
        query_vector=[0.1, 0.2],
        num_hits=8,
        tenant_state=TenantState(tenant_id="public", multitenant=False),
        index_filters=IndexFilters(
            access_control_list=None,
            regulatory_chunks_only=True,
        ),
        include_hidden=False,
    )

    lexical_lane = search_body["_onyx_hybrid_fusion"]["subqueries"][-1]
    source_boost, distinctive_boost = _source_boosts(lexical_lane)
    assert source_boost["query"] == "Basel Sözleşmesi"
    assert distinctive_boost["query"] == "Basel"


def test_hybrid_query_preserves_weighted_minmax_fusion() -> None:
    search_body = DocumentQuery.get_hybrid_search_query(
        query_text="retention schedule",
        query_vector=[0.1, 0.2],
        num_hits=10,
        tenant_state=TenantState(tenant_id="public", multitenant=False),
        index_filters=IndexFilters(access_control_list=None),
        include_hidden=False,
    )

    fusion = search_body["_onyx_hybrid_fusion"]
    assert fusion["rank_window_size"] == 500
    assert fusion["weights"] == [0.5, 0.5]
    assert fusion["normalizer"] == "minmax"
    vector_lane = fusion["subqueries"][0]["knn"]
    assert vector_lane["field"] == "content_vector"
    assert vector_lane["query_vector"] == [0.1, 0.2]
    assert vector_lane["k"] == 500
    assert vector_lane["num_candidates"] == 500


def test_semantic_query_uses_elasticsearch_8_top_level_knn() -> None:
    search_body = DocumentQuery.get_semantic_search_query(
        query_embedding=[0.1, 0.2],
        num_hits=10,
        tenant_state=TenantState(tenant_id="public", multitenant=False),
        index_filters=IndexFilters(access_control_list=None),
        include_hidden=False,
    )

    assert "query" not in search_body
    assert search_body["knn"]["field"] == "content_vector"
    assert search_body["knn"]["query_vector"] == [0.1, 0.2]
    assert search_body["knn"]["k"] == 10
    assert search_body["knn"]["num_candidates"] == 10
    assert "filter" in search_body["knn"]


def test_non_regulatory_concept_query_has_no_instrument_boost() -> None:
    search_body = DocumentQuery.get_keyword_search_query(
        query_text="Basel Sözleşmesi yasadışı trafik",
        num_hits=8,
        tenant_state=TenantState(tenant_id="public", multitenant=False),
        index_filters=IndexFilters(access_control_list=None),
        include_hidden=False,
    )

    assert not any(
        "dis_max" in clause for clause in search_body["query"]["bool"]["should"]
    )


def test_legal_exact_values_add_soft_constant_score_boosts() -> None:
    query = DocumentQuery._get_title_content_combined_keyword_search_query(
        "Geçici Madde 2, 2024/17 sayılı karar, 06.08.2026"
    )

    boosts = _legal_exact_boosts(query)
    assert boosts == {
        PROVISION_IDENTIFIERS_FIELD_NAME: (["geçici madde 2"], 24.0),
        DECISION_NUMBERS_FIELD_NAME: (["2024/17"], 20.0),
        LEGAL_DATES_FIELD_NAME: (["2026-08-06"], 12.0),
    }
    assert query["bool"].get("filter") is None


def test_query_without_legal_identifiers_has_no_exact_field_clause() -> None:
    query = DocumentQuery._get_title_content_combined_keyword_search_query(
        "teminat uygulamasının koşulları"
    )

    assert _legal_exact_boosts(query) == {}
