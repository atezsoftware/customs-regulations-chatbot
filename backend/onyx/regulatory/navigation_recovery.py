"""Request-grounded selection of metadata-only regulatory navigation leads."""

from __future__ import annotations

import json
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator

from onyx.configs.chat_configs import REGULATORY_REVIEW_TIMEOUT_S
from onyx.llm.interfaces import LLM
from onyx.llm.models import ReasoningEffort
from onyx.prompts.regulatory_navigation_recovery import (
    REGULATORY_NAVIGATION_RECOVERY_SYSTEM_PROMPT,
)
from onyx.regulatory.evidence_matrix import RegulatoryNavigationLead
from onyx.regulatory.structured_llm import generate_structured
from onyx.tracing.flows import LLMFlow
from onyx.utils.logger import setup_logger

logger = setup_logger()

_MAX_NAVIGATION_SELECTIONS = 16
_MAX_NAVIGATION_INPUT_LEADS = 256
_MAX_REQUEST_CHARS = 24_000
_MAX_COVERAGE_CHARS = 24_000
_NAVIGATION_SELECTION_MAX_TOKENS = 4_000


class _RegulatoryNavigationSelectionDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    navigation_ids: list[str] = Field(
        default_factory=list,
        max_length=_MAX_NAVIGATION_SELECTIONS,
    )

    @field_validator("navigation_ids", mode="before")
    @classmethod
    def bound_navigation_ids(cls, value: object) -> object:
        if not isinstance(value, list):
            return value
        return value[:_MAX_NAVIGATION_SELECTIONS]


def _bounded(value: str, max_chars: int) -> str:
    normalized = " ".join(value.split())
    return normalized[:max_chars].rstrip()


def select_regulatory_navigation_recovery_leads(
    llm: LLM,
    *,
    user_request: str,
    coverage_contract: str | None,
    navigation_leads: Sequence[RegulatoryNavigationLead],
) -> list[RegulatoryNavigationLead]:
    """Choose only supplied outline entries for one bounded exact-text recovery."""

    bounded_request = _bounded(user_request, _MAX_REQUEST_CHARS)
    bounded_leads = list(navigation_leads[:_MAX_NAVIGATION_INPUT_LEADS])
    if not bounded_request or not bounded_leads:
        return []

    lead_by_id = {
        f"N{index}": lead for index, lead in enumerate(bounded_leads, start=1)
    }
    payload = json.dumps(
        {
            "user_request": bounded_request,
            "coverage_contract": (
                _bounded(coverage_contract, _MAX_COVERAGE_CHARS)
                if coverage_contract
                else None
            ),
            "navigation_leads": [
                {
                    "navigation_id": navigation_id,
                    **lead.model_dump(mode="json"),
                }
                for navigation_id, lead in lead_by_id.items()
            ],
        },
        ensure_ascii=False,
    )
    try:
        draft = generate_structured(
            llm,
            flow=LLMFlow.REGULATORY_NAVIGATION_RECOVERY,
            system_prompt=REGULATORY_NAVIGATION_RECOVERY_SYSTEM_PROMPT,
            user_prompt=payload,
            response_model=_RegulatoryNavigationSelectionDraft,
            timeout_override=REGULATORY_REVIEW_TIMEOUT_S,
            max_tokens=_NAVIGATION_SELECTION_MAX_TOKENS,
            reasoning_effort=ReasoningEffort.HIGH,
            max_attempts=1,
        )
    except Exception:
        logger.exception("Regulatory navigation recovery selection failed")
        return []

    selected: list[RegulatoryNavigationLead] = []
    seen_ids: set[str] = set()
    for navigation_id in draft.navigation_ids:
        normalized_id = navigation_id.strip().upper()
        lead = lead_by_id.get(normalized_id)
        if lead is None or normalized_id in seen_ids:
            continue
        seen_ids.add(normalized_id)
        selected.append(lead)
    return selected
