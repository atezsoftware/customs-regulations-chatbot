"""A dependent subject fragment is complete only with its exact own source article."""

import hashlib
import re
from collections.abc import Sequence
from typing import Literal
from uuid import UUID

from bs4 import BeautifulSoup, Comment, NavigableString, Tag
from pydantic import BaseModel, ConfigDict, Field, model_validator

from onyx.document_index.publication_models import publication_digest
from onyx.regulatory.amendments.annexes.models import AnnexCanonicalSnapshot
from onyx.regulatory.amendments.annexes.sources import MAX_ASSET_BYTES
from onyx.regulatory.labeling.article_lead_in_visibility import (
    article_lead_in_quotes_closed,
    lead_in_prefix_text_is_visible,
    require_article_lead_in_visibility,
)
from onyx.regulatory.labeling.intake_abstention_repair import AbstentionRepairAudit
from onyx.regulatory.labeling.self_repeal import (
    NativeSourceStyles,
    PriorFailedAbstention,
    native_article_caption,
    native_closed_quotes,
    native_interval_text,
    native_own_article_path,
    native_own_header,
    native_title,
    native_visible_content,
    normalized_text,
    require_native_visibility,
)
from onyx.regulatory.law_identity import LAW_DOCUMENT_TYPES
from onyx.utils.web_content import decode_html_bytes

ARTICLE_LEAD_IN_KEY = "intake_source_only_article_lead_in"
_SHA = r"^[0-9a-f]{64}$"
_ARTICLE = re.compile(r"^MADDE\s+([1-9]\d*)\s*[-–—:]", re.I)
_LEAD_IN = re.compile(r"MADDE\s+[1-9]\d*\s*[-–—:]\s*Belediye meclisinin;", re.I)
_SUBJECT = "Belediye meclisinin;"
# This closed profile cannot admit an arbitrary semicolon or genitive sentence.
_COMMON_PREDICATE = (
    "Hâllerinde, meclis çalışabilir duruma gelinceye veya yeni meclis seçimi "
    "yapılıncaya kadar meclis görevi, belediye encümeninin memur üyeleri "
    "tarafından yürütülür."
)


class ArticleLeadInSourceProof(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    version: Literal[1] = 1
    profile: Literal["municipal_council_conditions_v1"] = (
        "municipal_council_conditions_v1"
    )
    user_file_id: UUID
    source_asset_id: UUID
    source_sha256: str = Field(pattern=_SHA)
    chunk_id: str
    document_number: str
    document_title: str
    article_number: str
    source_paragraph: int = Field(ge=0)
    source_last_paragraph: int = Field(ge=0)
    article_chunk_ids: tuple[str, ...]
    aggregate_chunk_id: str
    canonical_article_sha256: str = Field(pattern=_SHA)
    native_region_sha256: str = Field(pattern=_SHA)
    canonical_scope_sha256: str = Field(pattern=_SHA)


class SourceOnlyArticleLeadInReceipt(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    version: Literal[1] = 1
    disposition: Literal["source_only_article_lead_in"] = "source_only_article_lead_in"
    classification: Literal["abstained"] = "abstained"
    proof: ArticleLeadInSourceProof
    frozen_target_sha256: str = Field(pattern=_SHA)
    taxonomy_sha256: str = Field(pattern=_SHA)
    provider_binding_sha256: str = Field(pattern=_SHA)
    provider_abstention: AbstentionRepairAudit | None = None
    prior_failed_abstention: PriorFailedAbstention | None = None

    @model_validator(mode="after")
    def one_abstention(self) -> "SourceOnlyArticleLeadInReceipt":
        if (self.provider_abstention is None) == (self.prior_failed_abstention is None):
            raise ValueError("One unchanged provider abstention lineage is required")
        if self.provider_abstention is not None and (
            not self.provider_abstention.outcome.abstained
            or self.provider_abstention.outcome.labels
        ):
            raise ValueError("A structural lead-in preserves an empty abstention")
        return self


def article_lead_in_candidate(text: str) -> bool:
    return _LEAD_IN.fullmatch(normalized_text(text)) is not None


def _canonical_article(
    rows: Sequence[AnnexCanonicalSnapshot], target: AnnexCanonicalSnapshot
) -> tuple[list[AnnexCanonicalSnapshot], AnnexCanonicalSnapshot]:
    article = str(target.metadata.get("article_no"))
    own_path = native_own_article_path(target)
    atoms = sorted(
        (
            row
            for row in rows
            if str(row.metadata.get("article_no")) == article
            and row.metadata.get("chunk_variant") == "atomic"
        ),
        key=lambda row: (row.position, row.id),
    )
    if (
        len(atoms) != 5
        or atoms[0].id != target.id
        or target.chunk_type != "article"
        or [row.chunk_type for row in atoms[1:]] != ["clause"] * 4
        or [row.metadata.get("clause_label") for row in atoms]
        != [None, "a", "b", "c", "d"]
        or len({row.position for row in atoms}) != 5
        or any(
            row.status != "active"
            or row.superseded_by_chunk_id is not None
            or native_own_article_path(row) != own_path
            or row.metadata.get("paragraph_no") is not None
            or row.metadata.get("subclause_label") is not None
            or row.metadata.get("document_number")
            != target.metadata.get("document_number")
            or str(row.metadata.get("document_type", "")).casefold()
            not in LAW_DOCUMENT_TYPES
            or native_title(
                str(row.metadata.get("document_title") or row.heading_path[0])
            )
            != native_title(str(target.metadata.get("document_title") or own_path[0]))
            for row in atoms
        )
    ):
        raise ValueError("The structural lead-in has no complete ordered own article")
    aggregates = [
        row
        for row in rows
        if str(row.metadata.get("article_no")) == article
        and row.chunk_type == "hierarchical_aggregate"
        and tuple(row.heading_path) == own_path
    ]
    if len(aggregates) != 1:
        raise ValueError("The structural lead-in has no unique complete own aggregate")
    aggregate = aggregates[0]
    if (
        {row.id for row in rows if str(row.metadata.get("article_no")) == article}
        != {row.id for row in [*atoms, aggregate]}
        or aggregate.status != "active"
        or aggregate.superseded_by_chunk_id is not None
        or aggregate.metadata.get("chunk_variant") != "hierarchical_aggregate"
        or aggregate.metadata.get("hierarchy_root_path") != list(own_path)
        or aggregate.metadata.get("source_regulatory_chunk_ids")
        != [row.id for row in atoms]
        or aggregate.metadata.get("source_chunk_orders")
        != [row.position for row in atoms]
        or aggregate.metadata.get("document_number")
        != target.metadata.get("document_number")
        or str(aggregate.metadata.get("document_type", "")).casefold()
        not in LAW_DOCUMENT_TYPES
        or native_title(
            str(aggregate.metadata.get("document_title") or aggregate.heading_path[0])
        )
        != native_title(str(target.metadata.get("document_title") or own_path[0]))
        or aggregate.text
        != own_path[-1] + "\n\n" + "\n\n".join(row.text for row in atoms)
    ):
        raise ValueError(
            "The structural lead-in aggregate lineage or full text differs"
        )
    return atoms, aggregate


def build_article_lead_in_source_proof(
    *,
    raw_html: bytes,
    source_sha256: str,
    source_asset_id: UUID,
    user_file_id: UUID,
    chunk_id: str,
    rows: Sequence[AnnexCanonicalSnapshot],
) -> ArticleLeadInSourceProof:
    """Verify the closed subject fragment and all source conditions and predicate."""
    if (
        len(raw_html) > MAX_ASSET_BYTES
        or hashlib.sha256(raw_html).hexdigest() != source_sha256
    ):
        raise ValueError("The retained structural lead-in source bytes changed")
    by_id = {row.id: row for row in rows}
    if len(by_id) != len(rows) or any(
        row.user_file_id != str(user_file_id) for row in rows
    ):
        raise ValueError("The structural lead-in canonical scope changed")
    target = by_id.get(chunk_id)
    if target is None or not article_lead_in_candidate(target.text):
        raise ValueError("The target is not a supported subject-only lead-in")
    article = str(target.metadata.get("article_no"))
    if (match := _ARTICLE.match(normalized_text(target.text))) is None or match[
        1
    ] != article:
        raise ValueError("The structural lead-in own article number differs")
    atoms, aggregate = _canonical_article(rows, target)
    number = str(target.metadata.get("document_number", ""))
    title = str(target.metadata.get("document_title") or target.heading_path[0])
    if not number.isdecimal() or not re.fullmatch(r".+\s+kanunu?", native_title(title)):
        raise ValueError("The structural lead-in own law identity is incomplete")
    if native_title(target.heading_path[0]) != native_title(title):
        raise ValueError("The structural lead-in canonical own root differs")
    soup = BeautifulSoup(decode_html_bytes(raw_html, "text/html"), "html.parser")
    require_native_visibility(soup)
    styles = NativeSourceStyles(soup)
    paragraphs = list(soup.find_all("p"))
    texts = [normalized_text(p.get_text(" ", strip=True)) for p in paragraphs]
    markers = [
        (i, match[1]) for i, text in enumerate(texts) if (match := _ARTICLE.match(text))
    ]
    candidates = [(j, i) for j, (i, value) in enumerate(markers) if value == article]
    if len(candidates) != 1:
        raise ValueError("The native structural article is missing or duplicated")
    marker_order, index = candidates[0]
    if marker_order == 0 or marker_order + 1 == len(markers):
        raise ValueError("The native structural article has no complete own boundary")
    previous, following = markers[marker_order - 1], markers[marker_order + 1]
    if (
        [int(value) for _, value in markers[: marker_order + 2]]
        != list(range(1, int(following[1]) + 1))
        or int(previous[1]) + 1 != int(article)
        or int(article) + 1 != int(following[1])
        or any(
            not native_own_header(paragraphs[i], styles)
            for i, _ in markers[: marker_order + 2]
        )
    ):
        raise ValueError("The native hierarchy does not prove an unquoted own article")
    mastheads = [
        (i, match[1])
        for i, text in enumerate(texts[: markers[0][0]])
        if (match := re.fullmatch(r"Kanun Numarası\s*:\s*(\d+)", text, re.I))
    ]
    if (
        len(mastheads) != 1
        or mastheads[0][1] != number
        or native_title(" ".join(texts[: mastheads[0][0]])) != native_title(title)
        or any(
            not native_visible_content(p, styles)
            for p in paragraphs[: mastheads[0][0] + 1]
        )
    ):
        raise ValueError("The native structural source masthead differs")
    if soup.body is None:
        raise ValueError("The structural source has no visible body")
    prefix: list[str] = []
    quote_context: list[Tag] = []
    for node in soup.body.descendants:
        if node is paragraphs[index]:
            break
        if isinstance(node, Tag) and node.name in {
            "p",
            "div",
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
        }:
            prefix.append("\n")
        if isinstance(node, NavigableString) and not isinstance(node, Comment):
            parent = node.parent
            if isinstance(parent, Tag) and lead_in_prefix_text_is_visible(node, styles):
                prefix.append(str(node))
                if any(char in str(node) for char in "'\"‘’“”"):
                    quote_context.append(parent)
    if not native_closed_quotes(prefix) or not article_lead_in_quotes_closed(
        "".join(prefix)
    ):
        raise ValueError("The native lead-in is inside an unresolved quotation")
    caption_index = next(
        (i for i in range(index - 1, previous[0], -1) if texts[i]), None
    )
    if caption_index is not None:
        require_article_lead_in_visibility(
            soup,
            [
                *paragraphs[: mastheads[0][0] + 1],
                *quote_context,
                *(paragraphs[i] for i, _ in markers[: marker_order + 2]),
                *paragraphs[caption_index : following[0] + 1],
            ],
        )
    if (
        caption_index is None
        or texts[caption_index] != native_article_caption(target)
        or not native_visible_content(paragraphs[caption_index], styles)
        or not styles.visible_bold(paragraphs[caption_index])
        or texts[index] != normalized_text(target.text)
        or _ARTICLE.sub("", texts[index]).strip() != _SUBJECT
        or native_interval_text(paragraphs[caption_index], paragraphs[index], styles)
        != ""
    ):
        raise ValueError("The native structural article caption or subject differs")
    next_atoms = [
        row
        for row in rows
        if row.metadata.get("chunk_variant") == "atomic"
        and str(row.metadata.get("article_no")) == following[1]
    ]
    next_captions = {native_article_caption(row) for row in next_atoms}
    region = [(i, texts[i]) for i in range(index + 1, following[0]) if texts[i]]
    if (
        len(region) != 6
        or len(next_captions) != 1
        or region[-1][1] != next(iter(next_captions))
        or not styles.visible_bold(paragraphs[region[-1][0]])
        or region[-2][1] != _COMMON_PREDICATE
        or any(
            not native_visible_content(paragraphs[i], styles)
            for i in range(index + 1, following[0])
        )
    ):
        raise ValueError(
            "The native structural article list or common predicate is incomplete"
        )
    native_atoms = [
        texts[index],
        *(text for _, text in region[:3]),
        " ".join(text for _, text in region[3:5]),
    ]
    if (
        [normalized_text(row.text) for row in atoms] != native_atoms
        or any(
            not re.fullmatch(rf"{letter}\)\s+.+,", region[offset][1])
            for offset, letter in enumerate("abcd")
        )
        or not native_closed_quotes([texts[index], *(text for _, text in region)])
        or not article_lead_in_quotes_closed(
            " ".join([texts[index], *(text for _, text in region)])
        )
        or native_interval_text(paragraphs[index], paragraphs[following[0]], styles)
        != normalized_text(" ".join(text for _, text in region))
    ):
        raise ValueError("The complete native article differs from its canonical atoms")
    return ArticleLeadInSourceProof(
        user_file_id=user_file_id,
        source_asset_id=source_asset_id,
        source_sha256=source_sha256,
        chunk_id=chunk_id,
        document_number=number,
        document_title=title,
        article_number=article,
        source_paragraph=index,
        source_last_paragraph=region[-2][0],
        article_chunk_ids=tuple(row.id for row in atoms),
        aggregate_chunk_id=aggregate.id,
        canonical_article_sha256=publication_digest(
            [row.model_dump(mode="json") for row in [*atoms, aggregate]]
        ),
        native_region_sha256=publication_digest(
            [texts[index], *(text for _, text in region)]
        ),
        canonical_scope_sha256=publication_digest(
            [
                row.model_dump(mode="json")
                for row in sorted(rows, key=lambda row: (row.position, row.id))
            ]
        ),
    )
