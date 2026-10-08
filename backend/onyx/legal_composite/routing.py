"""Assign independent canonical actions to authorized source-kind registries."""

from collections.abc import Callable

from pydantic import JsonValue

from onyx.asv3.registry import CapabilityRegistry
from onyx.db.legal_composite_sources import SourceKind, SourceLaneCatalogue
from onyx.legal_composite.acquisition import InvalidSourceAction
from onyx.legal_composite.models import ResearchPlan, SourceAction


class SourceLaneRouter:
    def __init__(
        self,
        catalogue: SourceLaneCatalogue,
        build_registry: Callable[[SourceKind], CapabilityRegistry],
    ) -> None:
        self.catalogue = catalogue
        self._build_registry = build_registry
        self._registries: dict[SourceKind, CapabilityRegistry] = {}
        self._source_kinds = {str(row.source_id): row.kind for row in catalogue.records}

    def inventory(self) -> dict[str, JsonValue]:
        return {
            **self.catalogue.provenance(),
            "kinds": {
                kind.value: len(self.catalogue.source_ids(kind)) for kind in SourceKind
            },
            "classification_is_legal_authority": False,
        }

    def registry(self, action: SourceAction) -> CapabilityRegistry:
        if action.source_kind is None:
            raise InvalidSourceAction("Canonical action has no source-kind lane")
        kind = action.source_kind
        if kind not in self._registries:
            self._registries[kind] = self._build_registry(kind)
        return self._registries[kind]

    def expand(
        self, actions: list[SourceAction], plan: ResearchPlan
    ) -> list[SourceAction]:
        expanded: list[SourceAction] = []
        planned = {need.need_id: need for need in plan.needs}
        for action in actions:
            if set(action.need_ids) - planned.keys():
                raise InvalidSourceAction("Source action refers to an unknown need")
            source_id = action.arguments.get("source_id")
            if action.tool == "search_corpus":
                # Discovery coverage is fixed by the host, never the planner.
                kinds = list(SourceKind)
            elif isinstance(source_id, str):
                observed_kind = self._source_kinds.get(source_id)
                if observed_kind is None:
                    raise InvalidSourceAction(
                        "Source identity is outside the inventory"
                    )
                kinds = [action.source_kind or observed_kind]
            elif action.source_kind is not None:
                kinds = [action.source_kind]
            else:
                kinds = list(
                    dict.fromkeys(
                        kind
                        for need_id in action.need_ids
                        for kind in planned[need_id].source_kinds
                    )
                ) or [SourceKind.STATUTE, SourceKind.REGULATION]
            if action.tool in {"resolve_source", "query_corpus"}:
                # Ambiguous identities are searchable without contaminating typed lanes.
                kinds = list(dict.fromkeys([*kinds, SourceKind.UNKNOWN]))
            expanded.extend(
                action.model_copy(update={"source_kind": kind}) for kind in kinds
            )
        return expanded
