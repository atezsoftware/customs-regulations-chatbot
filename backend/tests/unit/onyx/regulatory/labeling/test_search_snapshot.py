from typing import cast
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from onyx.db import regulatory_label_search
from onyx.db.models import (
    RegulatoryLabelingRun,
    RegulatoryLabelSettings,
    RegulatoryLabelTaxonomy,
)
from onyx.regulatory.labeling import search_runtime
from onyx.regulatory.labeling.defaults import load_default_taxonomy


def _run(document_set_id: int) -> RegulatoryLabelingRun:
    definition = load_default_taxonomy()
    taxonomy_id = uuid4()
    return RegulatoryLabelingRun(
        id=uuid4(),
        document_set_id=document_set_id,
        taxonomy_id=taxonomy_id,
        taxonomy=RegulatoryLabelTaxonomy(
            id=taxonomy_id,
            definition=definition.model_dump(mode="json"),
            version_hash=definition.version_hash,
        ),
        status="completed",
        stage="finished",
    )


def test_native_snapshot_selects_finalized_current_taxonomy_in_captured_scope() -> None:
    run = _run(73)
    session = MagicMock(spec=Session)
    session.scalars.side_effect = [[run.id], [run]]
    session.get.return_value = RegulatoryLabelSettings(
        id=1, taxonomy_id=run.taxonomy_id
    )

    state = regulatory_label_search.load_document_set_search_snapshot(
        cast(Session, session), tenant_id="authorized-tenant", document_set_id=73
    )

    assert state is not None
    assert state.run_ids == (run.id,)
    assert state.document_set_id == 73
    assert state.tenant_id == "authorized-tenant"
    query = (
        session.scalars.call_args_list[0].args[0].compile(dialect=postgresql.dialect())
    )
    assert query.params["document_set_id_1"] == 73
    assert set(query.params["status_1"]) == {"completed", "completed_with_errors"}
    assert query.params["stage_1"] == "finished"
    assert (
        "regulatory_label_settings.taxonomy_id = regulatory_labeling_run.taxonomy_id"
        in str(query)
    )


@pytest.mark.parametrize("stale", ["scope", "taxonomy", "unfinished"])
def test_native_snapshot_revalidates_selected_runs_before_activation(
    stale: str,
) -> None:
    run = _run(74 if stale == "scope" else 73)
    if stale == "unfinished":
        run.stage = "running"
    session = MagicMock(spec=Session)
    session.scalars.side_effect = [[run.id], [run]]
    session.get.return_value = RegulatoryLabelSettings(
        id=1, taxonomy_id=uuid4() if stale == "taxonomy" else run.taxonomy_id
    )

    assert (
        regulatory_label_search.load_document_set_search_snapshot(
            cast(Session, session), tenant_id="authorized-tenant", document_set_id=73
        )
        is None
    )


def test_no_eligible_native_labels_preserves_baseline_without_taxonomy_read() -> None:
    session = MagicMock(spec=Session)
    session.scalars.return_value = []

    assert (
        regulatory_label_search.load_document_set_search_snapshot(
            cast(Session, session), tenant_id="authorized-tenant", document_set_id=73
        )
        is None
    )
    session.get.assert_not_called()


def test_native_snapshot_database_failure_is_fail_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        search_runtime,
        "label_read_session",
        MagicMock(side_effect=OperationalError("read", {}, Exception("offline"))),
    )

    assert search_runtime.search_snapshot_for_document_set(73) is None


def test_overlay_revalidation_retains_captured_document_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from datetime import date

    from onyx.regulatory.labeling.search_models import LabelSearchSnapshot

    run = _run(73)
    state = LabelSearchSnapshot(
        tenant_id="authorized-tenant",
        run_ids=(run.id,),
        taxonomy=load_default_taxonomy(),
        mode="hybrid",
        document_set_id=73,
    )
    monkeypatch.setattr(
        regulatory_label_search, "get_current_tenant_id", lambda: "authorized-tenant"
    )
    revalidate = MagicMock(return_value=None)
    monkeypatch.setattr(regulatory_label_search, "load_search_snapshot", revalidate)

    overlay = regulatory_label_search.load_label_overlay(
        cast(Session, MagicMock(spec=Session)),
        snapshot=state,
        chunk_ids=["canonical-chunk"],
        candidate_label_ids=["SUB.TAX.VAT"],
        as_of_date=date(2026, 7, 1),
    )

    assert overlay.candidate_ids == ()
    assert revalidate.call_args.kwargs["document_set_id"] == 73
