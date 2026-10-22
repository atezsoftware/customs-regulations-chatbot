"""Presentation fields derived from one retained original, outside model context."""

from dataclasses import dataclass
from threading import RLock

from onyx.asv3.models import EvidenceItem
from onyx.asv3.progress import official_corpus_source_name
from onyx.asv3.workflow_variant import (
    ASV3_EXP_CROPPED_VARIANT,
    ASV3_STANDARD_VARIANT,
)
from onyx.regulatory.legistation_metadata import parse_legistation_metadata

CITATION_SOURCE_TITLES_SERVICE = "asv3_citation_source_titles"


class CitationSourceTitles:
    """Authorized file labels shared by child contexts, never serialized as evidence."""

    def __init__(self) -> None:
        self._titles: dict[str, str] = {}
        self._lock = RLock()

    def remember(self, source_id: str, title: str | None) -> None:
        if title and title.strip():
            with self._lock:
                self._titles[source_id] = title

    def get(self, source_id: str) -> str | None:
        with self._lock:
            return self._titles.get(source_id)


@dataclass(frozen=True)
class CitationPresentation:
    display_title: str | None
    chunk_id: str | None
    text: str


def citation_presentation_enabled(
    workflow_variant: str, research_profile: str, parallel_research: bool
) -> bool:
    return (
        workflow_variant in {ASV3_STANDARD_VARIANT, ASV3_EXP_CROPPED_VARIANT}
        and research_profile == "normal"
        and not parallel_research
    )


def evidence_citation_presentation(
    item: EvidenceItem, *, source_title: str | None = None
) -> CitationPresentation:
    """Keep the recorded passage intact and choose an optional file-level title."""
    annotation = parse_legistation_metadata(item.metadata.get("legistation_metadata"))
    title = annotation.official_title if annotation is not None else None
    if not title or not title.strip():
        title = source_title
    if not title or not title.strip():
        title = official_corpus_source_name(item)
    if not title and item.search_doc is not None:
        title = item.search_doc.semantic_identifier
    return CitationPresentation(
        display_title=title, chunk_id=item.chunk_id, text=item.text
    )
