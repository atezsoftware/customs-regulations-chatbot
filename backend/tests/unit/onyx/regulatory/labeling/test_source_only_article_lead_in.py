"""The actual municipal lead-in never substitutes for its conditions or predicate."""

import json
from hashlib import sha256
from pathlib import Path
from uuid import UUID

import pytest

from onyx.document_index.publication_models import publication_digest
from onyx.regulatory.amendments.annexes.models import AnnexCanonicalSnapshot
from onyx.regulatory.labeling.article_lead_in import (
    ArticleLeadInSourceProof,
    SourceOnlyArticleLeadInReceipt,
    article_lead_in_candidate,
    build_article_lead_in_source_proof,
)
from onyx.regulatory.labeling.self_repeal import SourceOnlySelfRepealReceipt
from tests.unit.onyx.regulatory.labeling.abstention_repair_test_helpers import (
    REJECTED,
    audit,
    build,
    response,
)
from tests.unit.onyx.regulatory.labeling.test_native_word_self_repeal import (
    ASSET,
    EXPECTED_SHA,
    FILE,
    RAW,
    rows,
)

TARGET = "rc_8d2a10a4a29c07d7f56fce470246e9cc32cd520f"
AGGREGATE = "rc_8448f6c46c56a66515f1e012fcaaa585ec3841ff"


def test_old6001_receipt_model_dump_and_json_are_exact_to_deployed_baseline() -> None:
    # Exact approved c8 receipt serialized with the deployed 66a8e732 model.
    golden = (
        (Path(__file__).parent / "fixtures/6001-self-repeal-receipt-66a8e732.json")
        .read_text()
        .strip()
    )
    receipt = SourceOnlySelfRepealReceipt.model_validate_json(golden)
    assert receipt.model_dump(mode="json") == json.loads(golden)
    assert receipt.model_dump_json() == golden
    python_values = json.loads(golden)
    for key in ("user_file_id", "source_asset_id"):
        python_values["proof"][key] = UUID(python_values["proof"][key])
    for key in ("run_id", "item_id"):
        python_values["prior_failed_abstention"][key] = UUID(
            python_values["prior_failed_abstention"][key]
        )
    assert receipt.model_dump() == python_values
    assert publication_digest(receipt.model_dump(mode="json")) == (
        "762dc8dc0110925cf1284477bb12f1a4cfb4e2b625e4921b945c6346eb0c1fa5"
    )


@pytest.mark.parametrize("change", ["absent", "both", "positive", "nonabstained_empty"])
def test_lead_in_receipt_requires_exactly_one_true_empty_abstention(
    change: str,
) -> None:
    old = SourceOnlySelfRepealReceipt.model_validate_json(
        (
            Path(__file__).parent / "fixtures/6001-self-repeal-receipt-66a8e732.json"
        ).read_text()
    )
    values = {
        **old.model_dump(mode="json"),
        "proof": proof().model_dump(mode="json"),
        "disposition": "source_only_article_lead_in",
        "prior_failed_abstention": None,
    }
    if change != "absent":
        provider = audit(
            build(),
            REJECTED
            if change == "both"
            else response(empty=change == "nonabstained_empty"),
        )
        values["provider_abstention"] = provider.model_dump(mode="json")
    if change == "both":
        values["prior_failed_abstention"] = old.model_dump(mode="json")[
            "prior_failed_abstention"
        ]
    with pytest.raises(ValueError):
        SourceOnlyArticleLeadInReceipt.model_validate(values)


def proof(
    raw: bytes = RAW, scope: list[AnnexCanonicalSnapshot] | None = None
) -> ArticleLeadInSourceProof:
    return build_article_lead_in_source_proof(
        raw_html=raw,
        source_sha256=sha256(raw).hexdigest(),
        source_asset_id=ASSET,
        user_file_id=FILE,
        chunk_id=TARGET,
        rows=rows() if scope is None else scope,
    )


def test_actual_source_complete_own31_preserves_all370_canonical_rows() -> None:
    scope = rows()
    before = [row.model_dump() for row in scope]
    result = proof(scope=scope)
    assert sha256(RAW).hexdigest() == EXPECTED_SHA and len(RAW) == 318898
    assert (
        result.document_number == "5393" and result.document_title == "BELEDİYE KANUNU"
    )
    assert result.article_number == "31"
    assert (result.source_paragraph, result.source_last_paragraph) == (252, 257)
    assert result.article_chunk_ids == tuple(
        row.id for row in scope if 113 <= row.position <= 117
    )
    assert result.aggregate_chunk_id == AGGREGATE
    assert len(result.article_chunk_ids) == 5 and len(scope) == 370
    assert [row.model_dump() for row in scope] == before


@pytest.mark.parametrize("document_type", ["kanun", "law", "act", "statute", "tüzük"])
def test_own_law_type_uses_the_shared_aliases_with_full_source_proof(
    document_type: str,
) -> None:
    scope = [
        row.model_copy(
            update={"metadata": {**row.metadata, "document_type": document_type}}
        )
        for row in rows()
    ]
    if document_type == "tüzük":
        with pytest.raises(ValueError):
            proof(scope=scope)
    else:
        assert proof(scope=scope).document_number == "5393"


@pytest.mark.parametrize(
    "text",
    [
        "Madde 31- Belediye meclisi bu işlemi yapamaz;",
        "Madde 31- Belediye meclisi bu işlemi yapmak zorundadır;",
        "Madde 31- Şart gerçekleşirse Belediye meclisinin;",
        "Madde 31- Belediye meclisinin; belediye encümeni karar verir.",
        "Madde 31- Belediye encümeninin;",
        "Madde 31- Belediye meclisinin",
        "Madde 31- (Mülga: 1/1/2000-1000/1 md.)",
        "GEÇİCİ MADDE 31- Belediye meclisinin;",
    ],
)
def test_unknown_or_operative_intro_is_never_a_structural_candidate(text: str) -> None:
    assert not article_lead_in_candidate(text)
    scope = [
        row.model_copy(update={"text": text}) if row.id == TARGET else row
        for row in rows()
    ]
    with pytest.raises(ValueError):
        proof(scope=scope)
