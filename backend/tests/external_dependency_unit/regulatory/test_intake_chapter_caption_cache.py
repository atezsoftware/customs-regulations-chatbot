"""Owned PostgreSQL guards remain fresh while nested native proofs are reused."""

from contextlib import nullcontext
from copy import deepcopy
from dataclasses import replace
from typing import cast
from unittest.mock import Mock

import pytest
from sqlalchemy.orm import Session

from onyx.db import regulatory_labeling_chapter_caption as captions
from onyx.db import regulatory_labeling_review_authority as authority
from onyx.db.models import RegulatoryChunk, UserFile
from onyx.regulatory.document_intake.embedding import (
    FrozenEmbeddingGroup,
    encode_intake_embedding_groups,
)
from onyx.regulatory.labeling import chapter_caption as chapter
from onyx.regulatory.labeling import chapter_caption_cache as cache
from tests.external_dependency_unit.regulatory.test_intake_chapter_caption_flow import (
    apply_caption,
    seed,
)
from tests.external_dependency_unit.regulatory.test_labeling_jobs import LabelingData
from tests.external_dependency_unit.regulatory.test_labeling_jobs import (
    labeling_data as labeling_data,
)
from tests.external_dependency_unit.regulatory.test_labeling_jobs import (
    labeling_database as labeling_database,
)


def test_review_nested_contexts_read_parse_once_and_next_operation_reopens(
    labeling_data: LabelingData, monkeypatch: pytest.MonkeyPatch
) -> None:
    with Session(labeling_data.database.engine) as session:
        fixture = seed(session, labeling_data, monkeypatch)
        apply_caption(session, fixture)
        with monkeypatch.context() as uncached:
            uncached.setattr(authority, "native_caption_proof_operation", nullcontext)
            expected = authority.labeling_review_authority(session, fixture.run.id)
        read = Mock(wraps=fixture.store.read_file)
        parse = Mock(wraps=cache.build_chapter_caption_source_proof)
        validate = Mock(wraps=captions.validate_item_chapter_caption_receipt)
        owned = Mock(wraps=captions.require_owned_intake_source)
        monkeypatch.setattr(fixture.store, "read_file", read)
        monkeypatch.setattr(cache, "build_chapter_caption_source_proof", parse)
        monkeypatch.setattr(captions, "validate_item_chapter_caption_receipt", validate)
        monkeypatch.setattr(captions, "require_owned_intake_source", owned)
        for operation in (1, 2):
            assert (
                authority.labeling_review_authority(session, fixture.run.id) == expected
            )
            assert read.call_count == parse.call_count == operation
            assert validate.call_count >= 3 * operation
            assert owned.call_count == validate.call_count


@pytest.mark.parametrize("drift", [False, True])
def test_encoder_guard_nested_reviews_reuse_native_proof_but_recheck_postgres(
    labeling_data: LabelingData,
    monkeypatch: pytest.MonkeyPatch,
    drift: bool,
) -> None:
    with Session(labeling_data.database.engine) as session:
        fixture = seed(session, labeling_data, monkeypatch)
        apply_caption(session, fixture)
        expected = authority.labeling_review_authority(session, fixture.run.id)
        read = Mock(wraps=fixture.store.read_file)
        parse = Mock(wraps=cache.build_chapter_caption_source_proof)
        monkeypatch.setattr(fixture.store, "read_file", read)
        monkeypatch.setattr(cache, "build_chapter_caption_source_proof", parse)
        checks = 0

        def guard() -> None:
            nonlocal checks
            checks += 1
            if drift and checks == 4:
                row = session.get_one(RegulatoryChunk, fixture.data.derived_id)
                row.text += " changed after native cache warmed"
                session.flush()
            assert (
                authority.labeling_review_authority(session, fixture.run.id) == expected
            )

        def run() -> None:
            encode_intake_embedding_groups(
                [FrozenEmbeddingGroup(str(index), (str(index),)) for index in range(6)],
                encode=lambda texts: [[0.1, 0.2, 0.3] for _ in texts],
                expected_dimension=3,
                check_owner=guard,
            )

        with cache.native_caption_proof_operation():
            if drift:
                with pytest.raises(ValueError):
                    run()
                assert checks == 4
            else:
                run()
                assert checks >= 18
        assert read.call_count == parse.call_count == 1
        if not drift:
            authority.labeling_review_authority(session, fixture.run.id)
            assert read.call_count == parse.call_count == 2


@pytest.mark.parametrize(
    "change",
    [
        "current_fact",
        "current_text",
        "context_members",
        "source",
        "taxonomy",
        "provider",
        "request",
        "qualification",
    ],
)
def test_warm_native_proof_never_caches_current_db_or_receipt_guards(
    labeling_data: LabelingData,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    with Session(labeling_data.database.engine) as session:
        fixture = seed(session, labeling_data, monkeypatch)
        item = apply_caption(session, fixture)
        with cache.native_caption_proof_operation():
            captions.validate_item_chapter_caption_receipt(
                session, item=item, run=fixture.run
            )
            if change == "current_fact":
                own = session.get_one(RegulatoryChunk, fixture.data.derived_id)
                metadata = deepcopy(own.chunk_metadata)
                metadata["intake_legal_metadata"]["fields"]["article_title"][
                    "evidence_scope"
                ] = "heading"
                own.chunk_metadata = metadata
            elif change == "current_text":
                session.get_one(
                    RegulatoryChunk, fixture.data.derived_id
                ).text += " changed"
            elif change == "context_members":
                members = cast(list[str], item.source_snapshot["context_member_ids"])
                item.source_snapshot = {
                    **item.source_snapshot,
                    "context_member_ids": [
                        identifier
                        for identifier in members
                        if identifier != fixture.data.first_id
                    ],
                }
            elif change == "source":
                session.get_one(
                    UserFile, item.user_file_id
                ).file_id = "changed-original"
            elif change == "taxonomy":
                fixture.run.taxonomy_id = labeling_data.taxonomy_id
            elif change == "provider":
                fixture.run.provider_binding = {
                    **fixture.run.provider_binding,
                    "changed": True,
                }
            elif change == "request":
                item.request_payload = {
                    **(item.request_payload or {}),
                    "prompt": "changed",
                }
            else:
                monkeypatch.setattr(
                    chapter,
                    "_PROFILES",
                    tuple(
                        replace(profile, chapter_lines=("Changed",))
                        for profile in chapter._PROFILES
                    ),
                )
            session.flush()
            with pytest.raises(ValueError):
                captions.validate_item_chapter_caption_receipt(
                    session, item=item, run=fixture.run
                )
