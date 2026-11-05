"""Source-qualified insertion anchors, independent of physical index ordinals."""

import re

from pydantic import BaseModel, ConfigDict, Field

from onyx.document_index.publication_models import publication_digest
from onyx.regulatory.article_scope import ARTICLE_OPENING
from onyx.regulatory.provision_identity import article_identity, canonical_clause_label

_ALPHABET = "abcçdefgğhıijklmnoöprsştuüvyz"


class InsertionOrderError(ValueError):
    """The source cannot prove a safe insertion without further review."""


class OrderMember(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    id: str
    position: int
    article_no: str | None = None
    paragraph_no: str | None = None
    clause_label: str | None = None
    active: bool = True
    derived: bool = False
    source_sha256: str | None = None
    direct_clause_parent: bool = False
    appendix_label: str | None = Field(default=None, exclude=True)
    annex_item_no: str | None = Field(default=None, exclude=True)


class InsertionOrder(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    baseline_sha256: str
    article_no: str | None
    paragraph_no: str | None
    clause_label: str | None
    after_article_no: str | None = None
    after_chunk_id: str | None
    before_chunk_id: str | None
    after_source_sha256: str | None = None
    before_source_sha256: str | None = None
    position: int = Field(ge=0)
    appendix_label: str | None = Field(
        default=None, exclude_if=lambda value: value is None
    )


def reconcile_insertion_order(
    reviewed: InsertionOrder, current: InsertionOrder
) -> InsertionOrder:
    """Rebase an insertion only when its reviewed structural boundary survives."""
    identity_fields = (
        "article_no",
        "paragraph_no",
        "clause_label",
        "after_article_no",
        "after_chunk_id",
        "before_chunk_id",
        "appendix_label",
    )
    if any(
        getattr(reviewed, field) != getattr(current, field) for field in identity_fields
    ):
        raise ValueError(
            "Insertion source order changed after review; reanalyze before approval"
        )
    for field in ("after_source_sha256", "before_source_sha256"):
        expected = getattr(reviewed, field)
        if expected is not None and expected != getattr(current, field):
            raise ValueError(
                "Insertion source order changed after review; reanalyze before approval"
            )
    return current


def plan_insertion(
    rows: list[OrderMember],
    *,
    article_no: str | None,
    paragraph_no: str | None,
    clause_label: str | None,
    after_article_no: str | None = None,
    appendix_label: str | None = None,
) -> InsertionOrder:
    ordered = sorted(rows, key=lambda row: (row.position, row.id))
    if appendix_label is not None:
        return _plan_annex_item_insertion(
            ordered, appendix_label, article_no, paragraph_no, clause_label
        )
    if article_no is None:
        raise InsertionOrderError("Insertion article identity is missing")
    if paragraph_no is None and clause_label is None:
        return _plan_article_insertion(ordered, article_no, after_article_no)
    scope = [
        row
        for row in ordered
        if row.active
        and not row.derived
        and row.article_no == article_no
        and (clause_label is None or row.paragraph_no == paragraph_no)
    ]
    if not scope:
        raise InsertionOrderError("Insertion parent could not be verified")
    if paragraph_no is None and (
        clause_label is None
        or sum(row.direct_clause_parent for row in scope) != 1
        or any(
            row.active
            and not row.derived
            and row.article_no == article_no
            and row.paragraph_no is not None
            for row in ordered
        )
    ):
        raise InsertionOrderError(
            "Direct clause parent could not be verified without a numbered paragraph"
        )

    def rank(value: str | None) -> int:
        if value is None:
            return -1
        if clause_label is None:
            if not value.isdecimal():
                raise InsertionOrderError("Insertion paragraph identity is ambiguous")
            return int(value)
        label = canonical_clause_label(value)
        if label is None or len(label) != 1 or label not in _ALPHABET:
            raise InsertionOrderError("Insertion clause identity is ambiguous")
        return _ALPHABET.index(label)

    target = rank(clause_label if clause_label is not None else paragraph_no)
    ranks = [
        rank(row.clause_label if clause_label is not None else row.paragraph_no)
        for row in scope
    ]
    if target in ranks:
        raise InsertionOrderError(
            "Insertion identity already exists; a reviewed renumbering is required"
        )
    if ranks != sorted(ranks):
        raise InsertionOrderError(
            "Insertion parent order conflicts with its source metadata"
        )
    lower = [row for row, value in zip(scope, ranks) if value < target]
    upper = [row for row, value in zip(scope, ranks) if value > target]
    after = lower[-1] if lower else None
    before = (
        upper[0]
        if upper
        else next(
            (
                row
                for row in ordered
                if row.active and not row.derived and row.position > scope[-1].position
            ),
            None,
        )
    )
    position = before.position if before else scope[-1].position + 1
    if after is not None and after.position >= position:
        raise InsertionOrderError("Insertion has overlapping source positions")
    return InsertionOrder(
        baseline_sha256=publication_digest(
            [row.model_dump(mode="json") for row in ordered]
        ),
        article_no=article_no,
        paragraph_no=paragraph_no,
        clause_label=clause_label,
        after_chunk_id=after.id if after else None,
        before_chunk_id=before.id if before else None,
        after_source_sha256=after.source_sha256 if after else None,
        before_source_sha256=before.source_sha256 if before else None,
        position=position,
    )


def _plan_annex_item_insertion(
    ordered: list[OrderMember],
    appendix_label: str,
    article_no: str | None,
    paragraph_no: str | None,
    clause_label: str | None,
) -> InsertionOrder:
    from onyx.regulatory.amendments.structural_target import normalize_appendix_label

    if (
        article_no is not None
        or clause_label is not None
        or paragraph_no is None
        or not paragraph_no.isdecimal()
    ):
        raise InsertionOrderError(
            "Numbered annex item requires its own annex and numeric identity"
        )
    active = [r for r in ordered if r.active and not r.derived]
    scope = [
        r
        for r in active
        if r.appendix_label is not None
        and normalize_appendix_label(r.appendix_label)
        == normalize_appendix_label(appendix_label)
    ]
    if not scope or any(
        r.article_no is not None
        or r.clause_label is not None
        or (r.paragraph_no is not None and not r.paragraph_no.isdecimal())
        for r in scope
    ):
        raise InsertionOrderError("Numbered annex item parent could not be verified")
    if any(
        r not in scope and scope[0].position <= r.position <= scope[-1].position
        for r in active
    ):
        raise InsertionOrderError("Annex list contains an unrelated source boundary")
    numbered = [r for r in scope if r.paragraph_no is not None]
    if any(r.annex_item_no != r.paragraph_no for r in numbered):
        raise InsertionOrderError(
            "Annex item identity differs from its own source text"
        )
    ranks = [int(r.paragraph_no or "0") for r in numbered]
    target = int(paragraph_no)
    if target in ranks:
        raise InsertionOrderError(
            "Insertion identity already exists; a reviewed renumbering is required"
        )
    if not numbered or ranks != sorted(ranks):
        raise InsertionOrderError("Annex item order conflicts with its source metadata")
    lower = [r for r in numbered if int(r.paragraph_no or "0") < target]
    upper = [r for r in numbered if int(r.paragraph_no or "0") > target]
    after = lower[-1] if lower else scope[0]
    before = (
        upper[0]
        if upper
        else next((r for r in active if r.position > scope[-1].position), None)
    )
    position = before.position if before else scope[-1].position + 1
    if after.position >= position:
        raise InsertionOrderError("Annex insertion has overlapping source positions")
    return InsertionOrder(
        baseline_sha256=publication_digest(
            [r.model_dump(mode="json") for r in ordered]
        ),
        article_no=None,
        paragraph_no=paragraph_no,
        clause_label=None,
        appendix_label=appendix_label,
        after_chunk_id=after.id,
        before_chunk_id=before.id if before else None,
        after_source_sha256=after.source_sha256,
        before_source_sha256=before.source_sha256 if before else None,
        position=position,
    )


def _plan_article_insertion(
    ordered: list[OrderMember], article_no: str, after_article_no: str | None
) -> InsertionOrder:
    def identity(value: str) -> tuple[str, int, str]:
        match = re.fullmatch(r"(?:(EK|GEÇİCİ|MÜKERRER) )?(\d+)(/?[A-Z])?", value)
        if match is None:
            raise InsertionOrderError("Insertion article identity is ambiguous")
        return match[1] or "", int(match[2]), match[3] or ""

    namespace, number, suffix = identity(article_no)
    active = [row for row in ordered if row.active and not row.derived]
    if after_article_no is not None and not any(
        row.article_no == after_article_no for row in active
    ):
        raise InsertionOrderError("Explicit insertion neighbour could not be verified")
    scoped = [
        row
        for row in active
        if row.article_no is not None and identity(row.article_no)[0] == namespace
    ]
    ranks = [identity(row.article_no or "")[1:] for row in scoped]
    target = (number, suffix)
    if target in ranks:
        raise InsertionOrderError("Insertion article identity already exists")
    if ranks != sorted(ranks):
        raise InsertionOrderError(
            "Insertion article order conflicts with source metadata"
        )
    if not scoped:
        if not namespace:
            raise InsertionOrderError(
                "Insertion article namespace has no verified anchor"
            )
        article_rows = [row for row in active if row.article_no is not None]
        if not article_rows:
            raise InsertionOrderError("Insertion source has no verified article anchor")
        scoped = article_rows
        ranks = []
    lower = [row for row, rank in zip(scoped, ranks) if rank < target]
    upper = [row for row, rank in zip(scoped, ranks) if rank > target]
    after = lower[-1] if lower else scoped[-1] if not ranks else None
    if after_article_no is not None:
        anchor_namespace, anchor_number, anchor_suffix = identity(after_article_no)
        if (
            anchor_namespace != namespace
            or (anchor_number, anchor_suffix) >= target
            or after is None
            or identity(after.article_no or "")[:2] != (anchor_namespace, anchor_number)
        ):
            raise InsertionOrderError(
                "Insertion identity conflicts with its explicit neighbour"
            )
    before = (
        upper[0]
        if upper
        else next((row for row in active if row.position > scoped[-1].position), None)
    )
    if after_article_no is not None and after is not None and before is not None:
        immediate = next((row for row in active if row.position > after.position), None)
        if immediate is not None and immediate.position < before.position:
            if immediate.article_no is None:
                raise InsertionOrderError("Insertion article boundary is unqualified")
            # Honour the named neighbour without moving a temporary article past it.
            before = immediate
    if after and before and after.position >= before.position:
        raise InsertionOrderError("Insertion has overlapping source positions")
    if (
        after
        and before
        and any(after.position < row.position < before.position for row in active)
    ):
        raise InsertionOrderError(
            "Insertion article boundary contains unrelated source rows"
        )
    return InsertionOrder(
        baseline_sha256=publication_digest(
            [row.model_dump(mode="json") for row in ordered]
        ),
        article_no=article_no,
        paragraph_no=None,
        clause_label=None,
        after_article_no=after_article_no,
        after_chunk_id=after.id if after else None,
        before_chunk_id=before.id if before else None,
        after_source_sha256=after.source_sha256 if after else None,
        before_source_sha256=before.source_sha256 if before else None,
        position=before.position if before else scoped[-1].position + 1,
    )


def is_direct_article_opening(preview: str | None, article_no: str | None) -> bool:
    if not preview or article_no is None:
        return False
    clean = re.sub(r"(?:\*\*|__|`)", "", preview).strip()
    heading = ARTICLE_OPENING.match(clean)
    if heading is None or article_identity(heading.group()) != article_no:
        return False
    body = clean[heading.end() :].lstrip(" -–—:.\n\t")
    return not re.match(r"^\(\d+\)", body)
