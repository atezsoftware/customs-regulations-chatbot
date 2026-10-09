"""Partial answers cannot waive canonical source identity or unchanged evidence checks."""

from typing import TypeVar

import pytest
from pydantic import BaseModel, JsonValue

from onyx.legal_composite.acquisition import InvalidSourceAction
from onyx.legal_composite.draft_composition import DraftComposition
from onyx.legal_composite.engine import LegalCompositeEngine
from onyx.legal_composite.models import (
    AuthorityDependency,
    DependencyOrigin,
    ReviewCheck,
    SemanticReview,
    SourceAction,
    SourceRequirement,
    WorkflowPolicy,
)
from onyx.legal_composite.models import IssueResearchPlan as ResearchPlan
from onyx.legal_composite.models import ResearchPlan as LegacyResearchPlan
from onyx.legal_composite.models import StructuredDraftAnswer as DraftAnswer
from onyx.legal_composite.requirements import RequirementLedger
from onyx.legal_composite.reviewer import ReviewQuestion, build_checks
from onyx.tracing.flows import LLMFlow
from tests.unit.legal_composite.test_semantic_reviewer import inputs
from tests.unit.onyx.legal_composite.draft_composition_fixture import composition_for

T = TypeVar("T", bound=BaseModel)


@pytest.mark.parametrize("fault", ["document", "chunk", "missing_chunk", "hash"])
def test_requirement_registration_rejects_stale_canonical_identity(fault: str) -> None:
    ledger, plan, _draft, requirements = inputs()
    item = ledger._items[1]
    assert item.search_doc is not None
    if fault == "document":
        item.search_doc.document_id = "different-source"
    elif fault == "chunk":
        item.search_doc.metadata["regulatory_chunk_id"] = "different-chunk"
    elif fault == "missing_chunk":
        item.search_doc.metadata.pop("regulatory_chunk_id")
    else:
        item.text_hash = "0" * 64
    recorded = RequirementLedger(ledger)
    with pytest.raises(InvalidSourceAction):
        recorded.update(requirements, plan, {1})
    assert not recorded.records()


@pytest.mark.parametrize("unresolved", [True, False])
def test_partial_dependency_status_never_waives_stale_origin(unresolved: bool) -> None:
    ledger, plan, draft, requirements = inputs()
    original = ledger.get(1)
    assert original is not None
    if unresolved:
        draft.unresolved_need_ids = ["n1"]
    edge = AuthorityDependency(
        edge_id="material-permission",
        need_ids=["n1"],
        instrument_name="Faaliyet Kanunu",
        instrument_number="8917",
        article="27",
        origins=[
            DependencyOrigin(
                citation=1,
                source_id=original.source_id,
                chunk_id=original.chunk_id,
                text_hash="0" * 64,
            )
        ],
        governing_citations=[1],
    )

    class Gateway:
        last_call_id: str | None = "canonical-draft"
        last_delivered_citations = {1}

        def complete(
            self,
            system: str,
            payload: dict[str, JsonValue],
            response_type: type[T],
            flow: LLMFlow,
            finalizing: bool = False,
        ) -> T:
            del system, payload, flow, finalizing
            assert response_type is DraftComposition
            return response_type.model_validate(composition_for(draft).model_dump())

    class Acquirer:
        def definitions(self) -> list[dict[str, JsonValue]]:
            return []

        def acquire(
            self, actions: list[SourceAction], plan: LegacyResearchPlan
        ) -> list[dict[str, JsonValue]]:
            del actions, plan
            raise AssertionError("Boundary rejection needs no new source work")

    class Reviewer:
        def expected_checks(
            self,
            request: str,
            plan: ResearchPlan,
            draft: DraftAnswer,
            requirements: list[SourceRequirement],
            dependencies: list[AuthorityDependency],
            delivered: set[int],
            previous: SemanticReview | None = None,
            affected_sections: set[str] | None = None,
        ) -> dict[str, ReviewQuestion]:
            del previous, affected_sections
            return build_checks(
                request,
                plan,
                draft,
                requirements,
                dependencies,
                delivered,
                ledger=ledger,
            )

        def review(
            self,
            request: str,
            plan: ResearchPlan,
            draft: DraftAnswer,
            requirements: list[SourceRequirement],
            dependencies: list[AuthorityDependency],
            delivered: set[int],
            previous: SemanticReview | None = None,
            affected_sections: set[str] | None = None,
        ) -> SemanticReview:
            del previous, affected_sections
            questions = self.expected_checks(
                request, plan, draft, requirements, dependencies, delivered
            )
            return SemanticReview(
                checks=[
                    ReviewCheck(
                        check_id=identity,
                        need_ids=question.need_ids,
                        section_ids=question.section_ids,
                        status="addressed",
                        confidence=0.99,
                    )
                    for identity, question in questions.items()
                ]
            )

    engine = LegalCompositeEngine(
        gateway=Gateway(),
        acquirer=Acquirer(),
        ledger=ledger,
        policy=WorkflowPolicy(max_reviews=1),
        check_active=lambda: None,
        research_available=lambda: False,
        reviewer=Reviewer(),
    )
    engine.plan = plan
    engine.dependencies = [edge]
    engine.requirements.update(requirements, plan, {1})
    result = engine._finalize_semantic("What permit is required?", "", None, plan)
    assert result.status == "unavailable" and result.answer is None
    assert any("origin_binding_mismatch" in gap for gap in result.gaps)
