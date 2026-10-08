"""Resolve stored aggregate bindings before retrieving original provision closure."""

from __future__ import annotations

from collections import defaultdict
from uuid import UUID

from onyx.asv3.corpus_tools import CorpusBroker
from onyx.asv3.models import EvidenceItem, RunContext
from onyx.context.search.models import SearchDoc
from onyx.db.engine.sql_engine import get_session_with_current_tenant
from onyx.db.supersearch import resolve_supersearch_center_ids


class SupersearchCorpusBroker(CorpusBroker):
    def hydrate_search_centers(
        self, docs: list[SearchDoc], context: RunContext
    ) -> dict[tuple[str, int], list[EvidenceItem]]:
        grouped: dict[str, list[SearchDoc]] = defaultdict(list)
        for doc in docs:
            grouped[doc.document_id].append(doc)

        def hydrate(
            source_id: str, centers: list[SearchDoc]
        ) -> dict[tuple[str, int], list[EvidenceItem]]:
            context.check_active()
            center_ids = tuple(
                dict.fromkeys(
                    str(doc.metadata["regulatory_chunk_id"]) for doc in centers
                )
            )
            with get_session_with_current_tenant() as session:
                bindings = resolve_supersearch_center_ids(
                    session,
                    user=self.user,
                    filters=self.filters,
                    source_id=UUID(source_id),
                    center_ids=center_ids,
                    check_active=context.check_active,
                )
            resolved: list[SearchDoc] = []
            child_keys: dict[str, tuple[str, int]] = {}
            for atomic_id in dict.fromkeys(
                atomic for children in bindings.values() for atomic in children
            ):
                # Keys are local navigation only; canonical projection ordinals are
                # established by the existing original hydration, never by these keys.
                ordinal = len(resolved)
                resolved.append(
                    centers[0].model_copy(
                        deep=True,
                        update={
                            "chunk_ind": ordinal,
                            "metadata": {
                                **centers[0].metadata,
                                "regulatory_chunk_id": atomic_id,
                            },
                        },
                    )
                )
                child_keys[atomic_id] = source_id, ordinal
            originals = (
                super(SupersearchCorpusBroker, self).hydrate_search_results(
                    resolved, context
                )
                if resolved
                else {}
            )
            result: dict[tuple[str, int], list[EvidenceItem]] = {}
            for doc in centers:
                center_id = str(doc.metadata["regulatory_chunk_id"])
                children = bindings.get(center_id, ())
                retained: dict[tuple[str, str | None, str], EvidenceItem] = {}
                for child_id in children:
                    for item in originals.get(child_keys[child_id], []):
                        retained[item.identity] = item.model_copy(
                            update={
                                "metadata": {
                                    **item.metadata,
                                    "supersearch_retrieved_center_id": center_id,
                                    "supersearch_aggregate_resolved": children
                                    != (center_id,),
                                }
                            }
                        )
                result[doc.document_id, doc.chunk_ind] = list(retained.values())
            return result

        # Acquisition workers already bound concurrent searches. Keep each search's
        # canonical reads in its owning worker instead of multiplying DB readers.
        hydrated: dict[tuple[str, int], list[EvidenceItem]] = {}
        for source_id, centers in grouped.items():
            hydrated.update(hydrate(source_id, centers))
        return hydrated
