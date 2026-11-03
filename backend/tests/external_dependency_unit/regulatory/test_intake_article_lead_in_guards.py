"""Real owned sources keep structural abstention separate from child authority."""

from copy import deepcopy
from typing import cast
from uuid import uuid4

import pytest
from pydantic import JsonValue
from sqlalchemy import select
from sqlalchemy.orm import Session

from onyx.db import document_intake_edit as editing
from onyx.db import regulatory_labeling as labeling
from onyx.db import regulatory_labeling_article_lead_in as lead_in
from onyx.db import regulatory_labeling_self_repeal as self_repeal
from onyx.db.models import (
    RegulatoryChunk,
    RegulatoryDocumentIntakeFile,
    RegulatoryLabelingItem,
    RegulatoryLabelingRun,
    RegulatoryLabelTaxonomy,
    RegulatorySourceAsset,
)
from onyx.db.regulatory_labeling_review_authority import labeling_review_authority
from onyx.db.regulatory_labeling_source_only import (
    CONTINUED_ABSTENTION,
    SourceOnlyReceiptKey,
    recover_failed_source_only_items,
    require_single_source_only_kind,
)
from onyx.document_index.publication_models import publication_digest
from onyx.regulatory.labeling.article_lead_in import (
    ARTICLE_LEAD_IN_KEY,
)
from onyx.regulatory.labeling.self_repeal import (
    SOURCE_ONLY_KEY,
    SourceOnlySelfRepealReceipt,
)
from tests.external_dependency_unit.regulatory.test_intake_article_lead_in_flow import (
    recover_article_lead_in,
    seed_article_lead_in,
)
from tests.external_dependency_unit.regulatory.test_intake_source_only_self_repeal_flow import (
    _create,
    _end,
)
from tests.external_dependency_unit.regulatory.test_intake_source_only_self_repeal_flow import (
    _seed as seed_self_repeal,
)
from tests.external_dependency_unit.regulatory.test_labeling_jobs import LabelingData
from tests.external_dependency_unit.regulatory.test_labeling_jobs import (
    labeling_data as labeling_data,
)
from tests.external_dependency_unit.regulatory.test_labeling_jobs import (
    labeling_database as labeling_database,
)


@pytest.mark.parametrize(
    "drift",
    [
        "raw",
        "predicate",
        "aggregate",
        "context",
        "taxonomy",
        "receipt",
        "labels",
        "prior_error",
        "asset",
        "both_kinds",
    ],
)
def test_drift_blocks_current_search_review_and_reuse(
    labeling_data: LabelingData, monkeypatch: pytest.MonkeyPatch, drift: str
) -> None:
    with Session(labeling_data.database.engine) as session:
        fixture = seed_article_lead_in(session, labeling_data, monkeypatch)
        replacement, copied = recover_article_lead_in(session, fixture)
        if drift == "raw":
            fixture.store.raw += b" changed"
        elif drift == "predicate":
            row = session.scalar(
                select(RegulatoryChunk).where(
                    RegulatoryChunk.user_file_id == fixture.data.file_id,
                    RegulatoryChunk.position == 117,
                )
            )
            assert row is not None
            row.text += " Changed common predicate."
        elif drift == "aggregate":
            session.get_one(RegulatoryChunk, fixture.data.derived_id).text += " changed"
        elif drift == "context":
            copied.context_snapshot = (copied.context_snapshot or "") + " changed"
        elif drift == "taxonomy":
            tax = session.get_one(RegulatoryLabelTaxonomy, fixture.data.taxonomy_id)
            tax.definition = {**tax.definition, "name": "Changed frozen taxonomy"}
        elif drift == "receipt":
            value = cast(
                dict[str, object], deepcopy(copied.source_snapshot[ARTICLE_LEAD_IN_KEY])
            )
            value["taxonomy_sha256"] = "f" * 64
            copied.source_snapshot = {
                **copied.source_snapshot,
                ARTICLE_LEAD_IN_KEY: value,
            }
        elif drift == "labels":
            copied.labels = ["condition"]
        elif drift == "prior_error":
            fixture.items[
                fixture.data.first_id
            ].error = "Unverified generic provider failure"
        elif drift == "asset":
            member = session.scalar(
                select(RegulatoryDocumentIntakeFile).where(
                    RegulatoryDocumentIntakeFile.user_file_id == fixture.data.file_id
                )
            )
            assert member is not None
            asset = session.get_one(RegulatorySourceAsset, member.source_asset_id)
            changed = RegulatorySourceAsset(
                package_id=asset.package_id,
                sha256="f" * 64,
                file_id="different-retained-source",
                mime_type=asset.mime_type,
                display_name=asset.display_name,
                byte_count=asset.byte_count,
            )
            session.add(changed)
            session.flush()
            member.source_asset_id = changed.id
        else:
            copied.source_snapshot = {**copied.source_snapshot, SOURCE_ONLY_KEY: {}}
        session.flush()
        with pytest.raises(ValueError):
            lead_in.validate_item_article_lead_in_receipt(
                session, item=copied, run=replacement
            )
        assert fixture.data.first_id not in labeling.current_labeling_items_for_search(
            session, run=replacement, items=[copied]
        )
        with pytest.raises(ValueError):
            labeling_review_authority(session, replacement.id)
        _end(replacement, "completed")
        session.flush()
        following = _create(session, fixture.data)
        editing.reuse_completed_intake_labels(
            session, previous=replacement, replacement=following
        )
        next_item = session.scalar(
            select(RegulatoryLabelingItem).where(
                RegulatoryLabelingItem.run_id == following.id,
                RegulatoryLabelingItem.regulatory_chunk_id == fixture.data.first_id,
            )
        )
        assert next_item is not None and next_item.status != "completed"


def test_legacy_m5_receipt_and_authority_keep_the_frozen_wire_contract(
    labeling_data: LabelingData, monkeypatch: pytest.MonkeyPatch
) -> None:
    with Session(labeling_data.database.engine) as session:
        data = labeling_data
        old_run, _, items, store, _ = seed_self_repeal(session, data, monkeypatch)
        old = items[data.first_id]
        old.status, old.error = "failed", CONTINUED_ABSTENTION
        items[data.second_id].status = "completed"
        _end(old_run)
        session.flush()
        replacement = _create(session, data)
        assert (
            editing.reuse_completed_intake_labels(
                session, previous=old_run, replacement=replacement
            )
            == 2
        )
        copied = session.scalar(
            select(RegulatoryLabelingItem).where(
                RegulatoryLabelingItem.run_id == replacement.id,
                RegulatoryLabelingItem.regulatory_chunk_id == data.first_id,
            )
        )
        assert copied is not None
        saved = cast(dict[str, JsonValue], copied.source_snapshot[SOURCE_ONLY_KEY])
        current = self_repeal.build_source_only_self_repeal_receipt(
            session, item=copied, run=replacement, prior_failed=old
        )
        assert current.model_dump(mode="json") == saved
        assert (
            current.model_dump_json()
            == SourceOnlySelfRepealReceipt.model_validate(saved).model_dump_json()
        )
        # This field map and review payload are the frozen pre-lead-in wire contract.
        assert current.frozen_target_sha256 == publication_digest(
            {
                "chunk_id": copied.regulatory_chunk_id,
                "file_id": str(copied.user_file_id),
                "text": copied.text_snapshot,
                "canonical_text_sha256": copied.canonical_text_sha256,
                "context_sha256": copied.context_sha256,
                "context": copied.context_snapshot,
                "request_hash": copied.request_hash,
                "request_payload": copied.request_payload,
                "source_snapshot": {
                    key: value
                    for key, value in copied.source_snapshot.items()
                    if key != SOURCE_ONLY_KEY
                },
            }
        )
        expected_authority = {
            "version": 1,
            "run_id": str(replacement.id),
            "run_status": replacement.status,
            "provider_binding_sha256": publication_digest(replacement.provider_binding),
            "taxonomy_id": str(replacement.taxonomy_id),
            "model": replacement.model,
            "semantic_repairs": [],
            "source_only_dispositions": [
                {
                    "chunk_id": copied.regulatory_chunk_id,
                    "receipt_sha256": publication_digest(saved),
                }
            ],
            "structural_readers": [],
        }
        authority = labeling_review_authority(session, replacement.id)
        assert authority == expected_authority
        assert publication_digest(authority) == publication_digest(expected_authority)
        copied.source_snapshot = {**copied.source_snapshot, ARTICLE_LEAD_IN_KEY: {}}
        with pytest.raises(ValueError):
            self_repeal.validate_item_source_only_receipt(
                session, item=copied, run=replacement
            )
        copied.source_snapshot = {
            **{
                key: value
                for key, value in copied.source_snapshot.items()
                if key != SOURCE_ONLY_KEY
            },
            ARTICLE_LEAD_IN_KEY: {},
        }
        with pytest.raises(ValueError):
            self_repeal.build_source_only_self_repeal_receipt(
                session, item=copied, run=replacement, prior_failed=old
            )


def test_unknown_source_only_kind_is_rejected_at_runtime() -> None:
    with pytest.raises(ValueError, match="Unknown"):
        require_single_source_only_kind(
            {},
            prospective_key=cast(SourceOnlyReceiptKey, "unrecognized_source_only_kind"),
        )

    with Session() as session, pytest.raises(ValueError, match="Unknown"):
        recover_failed_source_only_items(
            session,
            previous=RegulatoryLabelingRun(id=uuid4()),
            replacement=RegulatoryLabelingRun(id=uuid4()),
            receipt_key=cast(SourceOnlyReceiptKey, "unrecognized_source_only_kind"),
            build_receipt=self_repeal.build_source_only_self_repeal_receipt,
        )
