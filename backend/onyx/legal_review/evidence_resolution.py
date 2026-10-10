"""Bind evidence dispositions to accepted questions without model-owned routing IDs."""

from types import GenericAlias
from typing import Literal

from pydantic import Field, JsonValue, create_model

from onyx.legal_review.models import (
    EvidenceResolutionDecision,
    ReadingDecision,
    ResearchDisposition,
    ReviewDiagnosisBatch,
)


class EvidenceResolutionPlan:
    def __init__(
        self,
        diagnoses: ReviewDiagnosisBatch,
        check_issues: dict[str, set[str]],
    ) -> None:
        self.diagnoses = diagnoses
        self.slots = {
            f"r{index:04d}": row
            for index, row in enumerate(
                (row for row in diagnoses.diagnoses if row.kind == "research"), 1
            )
        }
        self.issues = {
            slot: sorted(
                {issue for check in row.check_ids for issue in check_issues[check]}
            )
            for slot, row in self.slots.items()
        }

    def response_model(self) -> type[ReadingDecision]:
        disposition = (
            create_model(
                "BoundResearchDisposition",
                __base__=ResearchDisposition,
                slot=(Literal.__getitem__(tuple(self.slots)), Field()),
            )
            if self.slots
            else ResearchDisposition
        )
        return create_model(
            "BoundEvidenceResolutionDecision",
            __base__=ReadingDecision,
            research_resolutions=(
                GenericAlias(list, disposition),
                Field(min_length=len(self.slots), max_length=len(self.slots)),
            ),
        )

    def state(self, state: dict[str, JsonValue]) -> dict[str, JsonValue]:
        tasks = {row.task_id: row for row in self.diagnoses.research_tasks}
        receipts = state.get("source_operations", [])
        originals = state.get("original_evidence", [])
        assert isinstance(receipts, list) and isinstance(originals, list)
        investigations: dict[str, JsonValue] = {}
        for identity, task in tasks.items():
            evidence_ids: list[int] = []
            for receipt in receipts:
                if not isinstance(receipt, dict):
                    continue
                needs = receipt.get("research_need_ids")
                if not isinstance(needs, list) or task.existing_need_id not in needs:
                    continue
                citations = receipt.get("evidence_ids", [])
                if isinstance(citations, list):
                    for citation in citations:
                        if isinstance(citation, int) and citation not in evidence_ids:
                            evidence_ids.append(citation)
            investigations[identity] = {
                **task.model_dump(mode="json"),
                "returned_citations": evidence_ids,
            }
        source_index: dict[str, dict[str, JsonValue]] = {}
        for original in originals:
            if not isinstance(original, dict):
                continue
            source_id = original.get("source_id")
            citation = original.get("citation")
            metadata = original.get("metadata")
            if not isinstance(source_id, str) or not isinstance(citation, int):
                continue
            assert isinstance(metadata, dict)
            entry = source_index.setdefault(
                source_id,
                {"title": metadata.get("title"), "citations": []},
            )
            citations = entry["citations"]
            assert isinstance(citations, list)
            citations.append(citation)
        return {
            **{key: value for key, value in state.items() if key != "repair_contract"},
            "research_resolution_contract": {
                "questions": {
                    slot: {
                        **row.model_dump(mode="json"),
                        "issue_ids": self.issues[slot],
                    }
                    for slot, row in self.slots.items()
                },
                "investigations": investigations,
                "source_index": source_index,
                "source_index_is_navigation_only": True,
            },
        }

    def compile(self, result: ReadingDecision) -> EvidenceResolutionDecision:
        payload = result.model_dump(mode="json")
        supplied = payload["research_resolutions"]
        rows = {row["slot"]: row for row in supplied}
        if len(supplied) != len(rows) or rows.keys() != self.slots.keys():
            raise ValueError(
                "Evidence resolution must preserve every accepted question"
            )
        payload["research_resolutions"] = [
            {
                **{key: value for key, value in rows[slot].items() if key != "slot"},
                "check_ids": diagnosis.check_ids,
                "issue_ids": self.issues[slot],
            }
            for slot, diagnosis in self.slots.items()
        ]
        return EvidenceResolutionDecision.model_validate(payload)
