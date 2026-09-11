"""Fixed read-only ownership lookup for the retained 861 source diagnostic."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from onyx.db.regulatory_annex_acceptance import CanaryRun


def load_source861_canary(session: "Session") -> "CanaryRun | None":
    from onyx.db.models import KVStore
    from onyx.db.regulatory_annex_acceptance import CanaryRun

    saved = session.get(
        KVStore,
        "regulatory_annex_acceptance:8612398f20e5d3d03d154fffa196c5bf951b1862",
    )
    return CanaryRun.model_validate(saved.value) if saved is not None else None
