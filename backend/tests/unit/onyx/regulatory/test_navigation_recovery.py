import json
from unittest.mock import MagicMock, patch

from onyx.regulatory.evidence_matrix import RegulatoryNavigationLead
from onyx.regulatory.navigation_recovery import (
    select_regulatory_navigation_recovery_leads,
)
from onyx.tracing.flows import LLMFlow


def _lead(article_key: str, heading: str) -> RegulatoryNavigationLead:
    return RegulatoryNavigationLead(
        document_title="Instrument",
        article_key=article_key,
        heading_label=heading,
        research_targets=["How does the requested process operate?"],
    )


def test_navigation_recovery_returns_only_supplied_selected_ids() -> None:
    draft = MagicMock(navigation_ids=["N2", "N2", "N99", "N1"])
    with patch(
        "onyx.regulatory.navigation_recovery.generate_structured",
        return_value=draft,
    ) as generate:
        selected = select_regulatory_navigation_recovery_leads(
            MagicMock(),
            user_request="How does the requested process operate?",
            coverage_contract="Resolve the requested process.",
            navigation_leads=[
                _lead("part::article:8", "PART II > Article 8 - Initial operation"),
                _lead("part::article:9", "PART II > Article 9 - Later operation"),
            ],
        )

    assert [lead.article_key for lead in selected] == [
        "part::article:9",
        "part::article:8",
    ]
    call = generate.call_args.kwargs
    assert call["flow"] is LLMFlow.REGULATORY_NAVIGATION_RECOVERY
    payload = json.loads(call["user_prompt"])
    assert [item["navigation_id"] for item in payload["navigation_leads"]] == [
        "N1",
        "N2",
    ]
    assert "operative text" not in call["user_prompt"].casefold()


def test_navigation_recovery_skips_provider_without_request_or_leads() -> None:
    with patch("onyx.regulatory.navigation_recovery.generate_structured") as generate:
        assert (
            select_regulatory_navigation_recovery_leads(
                MagicMock(),
                user_request="",
                coverage_contract=None,
                navigation_leads=[_lead("part::article:8", "Article 8")],
            )
            == []
        )
        assert (
            select_regulatory_navigation_recovery_leads(
                MagicMock(),
                user_request="A request",
                coverage_contract=None,
                navigation_leads=[],
            )
            == []
        )

    generate.assert_not_called()


def test_navigation_recovery_keeps_full_outline_and_allows_broader_coverage() -> None:
    leads = [
        _lead(f"part::article:{index}", f"Article {index}") for index in range(1, 141)
    ]
    draft = MagicMock(navigation_ids=[f"N{index}" for index in range(125, 141)])
    with patch(
        "onyx.regulatory.navigation_recovery.generate_structured",
        return_value=draft,
    ) as generate:
        selected = select_regulatory_navigation_recovery_leads(
            MagicMock(),
            user_request="Resolve each independently requested result.",
            coverage_contract="Several independent obligations remain open.",
            navigation_leads=leads,
        )

    assert (
        len(json.loads(generate.call_args.kwargs["user_prompt"])["navigation_leads"])
        == 140
    )
    assert (
        generate.call_args.kwargs["response_model"].model_json_schema()["properties"][
            "navigation_ids"
        ]["maxItems"]
        == 16
    )
    assert [lead.article_key for lead in selected] == [
        f"part::article:{index}" for index in range(125, 141)
    ]
