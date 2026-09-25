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


class InsertionOrder(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    baseline_sha256: str
    article_no: str
    paragraph_no: str | None
    clause_label: str | None
    after_chunk_id: str | None
    before_chunk_id: str | None
    position: int = Field(ge=0)


def plan_insertion(
    rows: list[OrderMember],
    *,
    article_no: str,
    paragraph_no: str | None,
    clause_label: str | None,
) -> InsertionOrder:
    ordered = sorted(rows, key=lambda row: (row.position, row.id))
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
        position=position,
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
