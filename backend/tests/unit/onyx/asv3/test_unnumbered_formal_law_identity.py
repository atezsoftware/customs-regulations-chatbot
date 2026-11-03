"""Natural law names require a full own masthead, never metadata alone."""

import json
import re
from dataclasses import replace
from pathlib import Path
from typing import Any, cast
from uuid import UUID

import pytest

from onyx.db import asv3_corpus
from onyx.db.asv3_corpus import CorpusChunk, CorpusSource
from onyx.db.models import User
from tests.unit.onyx.asv3.test_numbered_law_identity_213 import chunk
from tests.unit.onyx.asv3.test_numbered_source_resolution import catalogue, filters

CAPTURES = json.loads(
    (Path(__file__).parent / "fixtures/numbered_intake_display_names.json").read_text()
)["laws"]


def captured(capture: dict[str, Any]) -> tuple[CorpusSource, list[CorpusChunk]]:
    identifier = UUID(capture["source_id"])
    name = re.sub(r"^\d+\s+sayılı\s+", "", capture["display_name"])
    return CorpusSource(identifier, name, "retained-owned-original"), [
        chunk(row, identifier) for row in capture["candidate_heads"]
    ]


@pytest.mark.parametrize("capture", CAPTURES, ids=lambda row: row["number"])
@pytest.mark.parametrize("formal", [False, True])
def test_actual_natural_candidates_keep_only_the_real_own_masthead(
    monkeypatch: pytest.MonkeyPatch, capture: dict[str, Any], formal: bool
) -> None:
    source, heads = captured(capture)
    session = catalogue(monkeypatch, [[], [source]] if formal else [[source]])
    monkeypatch.setattr(
        asv3_corpus, "_read_identity_heads_batch", lambda **_: {source.id: heads}
    )
    result, more = asv3_corpus.find_sources(
        session,
        user=cast(User, object()),
        filters=filters(),
        query=capture["display_name"]
        if formal
        else capture["number"] + " sayılı Kanun",
        allow_canonical_filename_fallback=True,
    )
    assert not more and result == [source]
    assert result[0].identity_originals == (heads[0],)
    assert source.identity_originals == ()


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_opening",
        "quoted_opening",
        "code_quote",
        "wrong_number",
        "conflicting_root",
        "conflicting_title",
        "wrong_type",
        "truncated",
        "foreign",
        "missing_metadata",
        "secondary_heading",
        "wrong_full_title",
        "numbered_name",
        "filename",
        "diacritic",
    ],
)
def test_natural_name_never_bypasses_own_identity(mutation: str) -> None:
    capture = next(row for row in CAPTURES if row["number"] == "5393")
    source, heads = captured(capture)
    if mutation == "numbered_name":
        source = replace(source, name="213 sayılı " + source.name)
    elif mutation == "filename":
        source = replace(source, name=source.name + ".md")
    elif mutation == "missing_opening":
        heads = [replace(row, text="Madde 1- normal provision") for row in heads]
    elif mutation in {"quoted_opening", "code_quote", "secondary_heading"}:
        prefix = {
            "quoted_opening": "> ",
            "code_quote": "```text\n",
            "secondary_heading": "Bakanlar Kurulu Kararı\n",
        }[mutation]
        heads[0] = replace(heads[0], text=prefix + heads[0].text)
    elif mutation == "diacritic":
        source = replace(source, name=source.name.replace("Belediye", "Beledıye"))
    else:
        changes = {
            "wrong_number": {"document_number": "3213"},
            "conflicting_title": {"document_title": "213 sayılı Belediye Kanunu"},
            "wrong_type": {"document_type": "bakanlar kurulu kararı"},
            "truncated": {"truncated": True},
            "foreign": {"external": True},
            "missing_metadata": {"document_number": None},
            "wrong_full_title": {"document_title": "Maden Kanunu"},
            "conflicting_root": {},
        }[mutation]
        heads = [
            replace(
                row,
                metadata={**row.metadata, **changes},
                heading_path=("213 sayılı Belediye Kanunu", *row.heading_path[1:])
                if mutation == "conflicting_root"
                else row.heading_path,
            )
            for row in heads
        ]
    assert not asv3_corpus._source_has_unnumbered_own_law_identity(
        source, heads, "5393"
    )


@pytest.mark.parametrize("failure", ["denied", "unpublished", "nonoptin", "more"])
def test_natural_name_preserves_scope_optin_and_uncertain_pagination(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    source, heads = captured(CAPTURES[0])
    session = catalogue(
        monkeypatch,
        [[source, source]]
        if failure == "more"
        else [[]]
        if failure == "nonoptin"
        else [[source]],
        denied={source.id} if failure == "denied" else None,
        unpublished={source.id} if failure == "unpublished" else None,
    )
    monkeypatch.setattr(
        asv3_corpus,
        "_read_identity_heads_batch",
        lambda **kwargs: {row.id: heads for row in kwargs["sources"]},
    )
    result, more = asv3_corpus.find_sources(
        session,
        user=cast(User, object()),
        filters=filters(),
        query=CAPTURES[0]["number"] + " sayılı Kanun",
        limit=1,
        allow_canonical_filename_fallback=failure != "nonoptin",
    )
    if failure == "more":
        assert result == [source] and more
    else:
        assert not result and not more


@pytest.mark.parametrize("capture", CAPTURES, ids=lambda row: row["number"])
def test_already_numbered_canonical_title_keeps_full_formal_lookup(
    capture: dict[str, Any],
) -> None:
    source, heads = captured(capture)
    heads = [
        replace(
            row, metadata={**row.metadata, "document_title": capture["display_name"]}
        )
        for row in heads
    ]
    assert asv3_corpus._source_has_unnumbered_own_law_identity(
        source,
        heads,
        capture["number"],
        requested=asv3_corpus._numbered_law_title(capture["display_name"]),
    )
