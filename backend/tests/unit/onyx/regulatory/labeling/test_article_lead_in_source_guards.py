"""Source-complete lead-in authority rejects hidden, quoted and incomplete evidence."""

import pytest
from bs4 import BeautifulSoup, Tag

from onyx.regulatory.labeling.article_lead_in import build_article_lead_in_source_proof
from tests.unit.onyx.regulatory.labeling.test_source_only_article_lead_in import (
    AGGREGATE,
    ASSET,
    FILE,
    RAW,
    TARGET,
    proof,
    rows,
)


@pytest.mark.parametrize(
    "change",
    [
        "hidden_subject",
        "hidden_caption",
        "hidden_a",
        "hidden_predicate",
        "hidden_next_caption",
        "hidden_next_marker",
        "quoted_subject",
        "quoted_child",
        "quoted_predicate",
        "table_child",
        "missing_a",
        "missing_predicate",
        "reordered_conditions",
        "duplicate_condition",
        "extra_rule",
        "bare_rule",
        "image",
        "unclosed_quote",
        "quoted_target_heading",
        "foreign_number",
        "foreign_title",
        "unknown_subject",
        "conditional_subject",
        "extra_subject_rule",
        "changed_predicate",
        "missing_next_marker",
        "functional_css",
        "invisible_predicate",
        "rule_after_caption",
    ],
)
def test_full_native_source_rejects_missing_hidden_quoted_or_extra_rules(
    change: str,
) -> None:
    soup = BeautifulSoup(RAW, "html.parser")
    paragraphs = soup.find_all("p")
    if change.startswith("hidden_"):
        indexes = {
            "hidden_subject": 252,
            "hidden_caption": 251,
            "hidden_a": 253,
            "hidden_predicate": 257,
            "hidden_next_caption": 259,
            "hidden_next_marker": 260,
        }
        paragraphs[indexes[change]]["style"] = "display:none"
    elif change.startswith("quoted_") and change != "quoted_target_heading":
        indexes = {"quoted_subject": 252, "quoted_child": 254, "quoted_predicate": 257}
        paragraphs[indexes[change]].wrap(soup.new_tag("blockquote"))
    elif change == "table_child":
        paragraphs[253].wrap(soup.new_tag("table"))
    elif change in {"missing_a", "missing_predicate", "missing_next_marker"}:
        paragraphs[
            {"missing_a": 253, "missing_predicate": 257, "missing_next_marker": 260}[
                change
            ]
        ].decompose()
    elif change == "reordered_conditions":
        paragraphs[254].insert_after(paragraphs[253].extract())
    elif change == "duplicate_condition":
        paragraphs[254].insert_before(
            BeautifulSoup(str(paragraphs[253]), "html.parser")
        )
    elif change == "extra_rule":
        node = soup.new_tag("p")
        node.string = "Belediye meclisi ayrıca bağımsız karar verir."
        paragraphs[257].insert_after(node)
    elif change == "bare_rule":
        paragraphs[257].insert_after("Ek şart uygulanır.")
    elif change == "image":
        paragraphs[255].insert_after(soup.new_tag("img", src="condition.png"))
    elif change == "unclosed_quote":
        paragraphs[251].insert_before("“")
    elif change == "quoted_target_heading":
        paragraphs[254].insert_before(
            BeautifulSoup("<p><b>Madde 231-</b>alıntı</p>", "html.parser")
        )
    elif change == "foreign_number":
        node = soup.find(string=lambda text: bool(text and "5393" in text))
        assert node is not None
        node.replace_with(str(node).replace("5393", "6001"))
    elif change == "foreign_title":
        node = soup.find(string=lambda text: bool(text and "BELEDİYE KANUNU" in text))
        assert node is not None
        node.replace_with(str(node).replace("BELEDİYE KANUNU", "YABANCI KANUNU"))
    elif change in {"unknown_subject", "conditional_subject", "extra_subject_rule"}:
        node = paragraphs[252].find(
            string=lambda text: bool(text and "Belediye" in text)
        )
        assert node is not None
        replacement = {
            "unknown_subject": "Belediye encümeninin;",
            "conditional_subject": "Şart gerçekleşirse Belediye meclisinin;",
            "extra_subject_rule": "Belediye meclisinin; ek şart uygulanır.",
        }[change]
        node.replace_with(replacement)
    elif change == "changed_predicate":
        paragraphs[257].string = "Hâllerinde, belediye meclisi karar verir."
    elif change == "functional_css":
        assert isinstance(soup.head, Tag)
        style = soup.new_tag("style")
        style.string = ":is(p,:unknown-pseudo){display:none}"
        soup.head.append(style)
    elif change == "invisible_predicate":
        paragraphs[257]["style"] = "opacity:0"
    elif change == "rule_after_caption":
        paragraphs[251].insert_after("Ayrıca bağımsız şart uygulanır.")
    with pytest.raises(ValueError):
        proof(raw=str(soup).encode())


@pytest.mark.parametrize(
    "change",
    [
        "missing_child",
        "extra_child",
        "superseded_child",
        "inactive_child",
        "foreign_child",
        "wrong_path",
        "wrong_clause",
        "wrong_order",
        "missing_aggregate",
        "partial_aggregate",
        "aggregate_order",
        "aggregate_lineage",
        "foreign_aggregate",
        "duplicate_aggregate",
        "unmodeled_atom",
        "changed_threshold",
        "dropped_predicate",
        "next_caption",
        "root_with_explicit_title",
    ],
)
def test_canonical_article_requires_exact_full_children_and_existing_aggregate(
    change: str,
) -> None:
    scope = rows()
    child = next(row for row in scope if row.position == 114)
    aggregate = next(row for row in scope if row.id == AGGREGATE)
    if change == "missing_child":
        scope.remove(child)
    elif change in {"extra_child", "unmodeled_atom"}:
        scope.append(
            child.model_copy(
                update={
                    "id": "extra",
                    "position": 118,
                    "metadata": {
                        **child.metadata,
                        "chunk_variant": "atomic"
                        if change == "extra_child"
                        else "unknown",
                    },
                }
            )
        )
    elif change == "missing_aggregate":
        scope.remove(aggregate)
    elif change == "duplicate_aggregate":
        scope.append(aggregate.model_copy(update={"id": "duplicate"}))
    elif change == "next_caption":
        scope = [row for row in scope if row.metadata.get("article_no") != "32"]
    elif change == "root_with_explicit_title":
        scope = [
            row.model_copy(
                update={
                    "heading_path": ["YABANCI KANUNU", *row.heading_path[1:]],
                    "metadata": {
                        **row.metadata,
                        "document_title": "BELEDİYE KANUNU",
                        "hierarchy_root_path": [
                            "YABANCI KANUNU",
                            *row.heading_path[1:],
                        ],
                    },
                }
            )
            if row.metadata.get("article_no") == "31"
            else row
            for row in scope
        ]
    else:
        if change.startswith("aggregate_") or change in {
            "partial_aggregate",
            "foreign_aggregate",
        }:
            target = aggregate
            update = {
                "partial_aggregate": {"text": aggregate.text[:-10]},
                "aggregate_order": {
                    "metadata": {
                        **aggregate.metadata,
                        "source_chunk_orders": [114, 113, 115, 116, 117],
                    }
                },
                "aggregate_lineage": {
                    "metadata": {
                        **aggregate.metadata,
                        "source_regulatory_chunk_ids": [TARGET],
                    }
                },
                "foreign_aggregate": {
                    "metadata": {**aggregate.metadata, "document_number": "6001"}
                },
            }[change]
        else:
            target = (
                next(row for row in scope if row.position == 117)
                if change == "dropped_predicate"
                else child
            )
            update = {
                "superseded_child": {"superseded_by_chunk_id": "replacement"},
                "inactive_child": {"status": "inactive"},
                "foreign_child": {
                    "metadata": {**target.metadata, "document_number": "6001"}
                },
                "wrong_path": {
                    "heading_path": ["YABANCI KANUNU", *target.heading_path[1:]]
                },
                "wrong_clause": {"metadata": {**target.metadata, "clause_label": "z"}},
                "wrong_order": {"position": 116},
                "changed_threshold": {
                    "text": target.text.replace("feshi", "feshi ve başka şart")
                },
                "dropped_predicate": {"text": target.text.splitlines()[0]},
            }[change]
        scope = [
            row.model_copy(update=update) if row.id == target.id else row
            for row in scope
        ]
    with pytest.raises(ValueError):
        proof(scope=scope)


def test_retained_source_hash_is_required_before_structural_evidence() -> None:
    with pytest.raises(ValueError, match="bytes changed"):
        build_article_lead_in_source_proof(
            raw_html=RAW,
            source_sha256="0" * 64,
            source_asset_id=ASSET,
            user_file_id=FILE,
            chunk_id=TARGET,
            rows=rows(),
        )


@pytest.mark.parametrize(
    "change",
    [
        "details_subject",
        "details_predicate",
        "dialog_predicate",
        "offscreen",
        "clipped",
        "white_on_white",
        "single_quote",
        "curly_single_quote",
        "quote_across_blocks",
        "negative_margin",
        "important_duplicate",
        "escaped_property",
        "glyph_font",
        "deleted_condition",
        "body_margin",
        "script",
        "event_handler",
        "stylesheet_upper",
        "popover",
        "shadow_root",
        "meta_refresh",
        "style_quote_closure",
        "hidden_quote_closure",
    ],
)
def test_closed_profile_rejects_unmodeled_visibility_and_outer_quotation(
    change: str,
) -> None:
    soup = BeautifulSoup(RAW, "html.parser")
    paragraphs = soup.find_all("p")
    if change in {"details_subject", "details_predicate", "dialog_predicate"}:
        paragraphs[252 if change == "details_subject" else 257].wrap(
            soup.new_tag("dialog" if change == "dialog_predicate" else "details")
        )
    elif change in {"single_quote", "curly_single_quote", "quote_across_blocks"}:
        opening, closing = ("‘", "’") if change == "curly_single_quote" else ("'", "'")
        if change == "quote_across_blocks":
            paragraphs[250].append("A" + opening)
        else:
            paragraphs[251].insert_before(opening)
        paragraphs[260].insert_after(closing)
    elif change in {
        "offscreen",
        "clipped",
        "white_on_white",
        "negative_margin",
        "important_duplicate",
        "escaped_property",
        "glyph_font",
    }:
        paragraphs[253]["style"] = {
            "offscreen": "position:absolute;left:-10000px",
            "clipped": "height:0;overflow:hidden",
            "white_on_white": "color:#fff;background-color:#fff",
            "negative_margin": "margin-left:-10000px",
            "important_duplicate": "margin-left:-10000px!important;margin-left:0",
            "escaped_property": r"mar\67 in-left:-10000px",
            "glyph_font": "font-family:Wingdings",
        }[change]
    elif change == "deleted_condition":
        paragraphs[253].wrap(soup.new_tag("del"))
    elif change == "body_margin":
        assert soup.body is not None
        soup.body["leftmargin"] = "10000"
    elif change == "event_handler":
        assert soup.body is not None
        soup.body["onload"] = "document.body.style.display='none'"
    elif change == "stylesheet_upper":
        assert soup.head is not None
        soup.head.append(
            soup.new_tag(
                "link", rel="STYLESHEET", href="data:text/css,body%7Bdisplay:none%7D"
            )
        )
    elif change == "popover":
        paragraphs[257]["popover"] = "auto"
    elif change == "shadow_root":
        host = soup.new_tag("div")
        paragraphs[257].wrap(host)
        host.insert(0, soup.new_tag("template", shadowrootmode="open"))
    elif change == "meta_refresh":
        assert soup.head is not None
        soup.head.append(
            soup.new_tag(
                "meta", attrs={"http-equiv": "refresh", "content": "0;url=about:blank"}
            )
        )
    elif change in {"style_quote_closure", "hidden_quote_closure"}:
        paragraphs[251].insert_before("'")
        node = soup.new_tag("style" if change == "style_quote_closure" else "span")
        node.string = "/* ' */" if change == "style_quote_closure" else "'"
        if change == "hidden_quote_closure":
            node["hidden"] = ""
        paragraphs[251].insert_before(node)
        paragraphs[260].insert_after("'")
    else:
        node = soup.new_tag("script")
        node.string = "document.body.style.display='none'"
        assert soup.body is not None
        soup.body.append(node)
    with pytest.raises(ValueError):
        proof(raw=str(soup).encode())


@pytest.mark.parametrize(
    "style",
    [
        "display:/**/none",
        "display:none ! important;display:block",
        "display:none !/**/important;display:block",
        r"display:\6e one",
        "display:table-column",
        "visibility:collapse",
        r"visibility:\68 idden",
    ],
)
def test_closed_profile_rejects_ambiguous_or_nonvisible_display_cascade(
    style: str,
) -> None:
    soup = BeautifulSoup(RAW, "html.parser")
    soup.find_all("p")[257]["style"] = style
    with pytest.raises(ValueError):
        proof(raw=str(soup).encode())
