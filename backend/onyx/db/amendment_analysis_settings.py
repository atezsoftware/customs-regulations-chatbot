"""Small immutable analysis choices saved in the batch creation transaction."""

from collections.abc import Mapping
from typing import cast

from sqlalchemy.orm import Session

from onyx.db.models import KVStore
from onyx.regulatory.amendments.model_choice import AmendmentAnalysisModel
from onyx.utils.special_types import JSON_ro


def store_analysis_model(
    session: Session, batch_id: int, model: AmendmentAnalysisModel
) -> None:
    session.add(
        KVStore(
            key=f"amendment_analysis_settings:{batch_id}", value={"model": model.value}
        )
    )


def load_analysis_model(session: Session, batch_id: int) -> AmendmentAnalysisModel:
    row = session.get(KVStore, f"amendment_analysis_settings:{batch_id}")
    if row is None:
        return AmendmentAnalysisModel.FLASH
    if not isinstance(row.value, Mapping):
        raise ValueError("Invalid saved amendment analysis model")
    selected = cast(Mapping[str, JSON_ro], row.value).get("model")
    if not isinstance(selected, str):
        raise ValueError("Invalid saved amendment analysis model")
    return AmendmentAnalysisModel(selected)


def get_batch_analysis_model(batch_id: int) -> AmendmentAnalysisModel:
    from onyx.db.engine.sql_engine import get_session_with_current_tenant

    with get_session_with_current_tenant() as session:
        return load_analysis_model(session, batch_id)
