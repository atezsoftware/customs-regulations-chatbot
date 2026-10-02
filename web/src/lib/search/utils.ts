import { Tag, ValidSources } from "../types";
import {
  CitationPreviewDocument,
  Filters,
  MinimalOnyxDocument,
  OnyxDocument,
  SourceMetadata,
} from "./interfaces";
import { StreamingCitation } from "@/app/app/services/streamingModels";
import { DateRangePickerValue } from "@/components/dateRangeSelectors/AdminDateRangeSelector";

export const buildFilters = (
  sources: SourceMetadata[],
  documentSets: string[],
  timeRange: DateRangePickerValue | null,
  tags: Tag[]
): Filters => {
  const filters = {
    source_type:
      sources.length > 0 ? sources.map((source) => source.internalName) : null,
    document_set: documentSets.length > 0 ? documentSets : null,
    updated_at_range: timeRange?.from
      ? { start: timeRange.from, end: null }
      : null,
    tags: tags,
  };

  return filters;
};

// If we have a link, open it in a new tab (including if it's a file)
// If above fails and we have a file, update the presenting document
export const openDocument = (
  document: OnyxDocument,
  updatePresentingDocument?: (document: MinimalOnyxDocument) => void
) => {
  if (document.link) {
    window.open(document.link, "_blank");
  } else if (
    document.source_type === ValidSources.File ||
    document.source_type === ValidSources.UserFile
  ) {
    updatePresentingDocument?.(document);
  }
};

/** Public internet evidence points to its original page, outside the canonical chunk index. */
export function openExternalCitation(link: string): void {
  try {
    const url = new URL(link);
    if (url.protocol !== "https:" && url.protocol !== "http:") return;
    window.open(url.href, "_blank", "noopener,noreferrer");
  } catch {
    // A malformed external target never becomes a synthetic indexed chunk.
  }
}

export const openCitation = (
  citation: StreamingCitation,
  semanticIdentifier: string,
  updatePresentingDocument: (document: MinimalOnyxDocument) => void
) => {
  const citationTarget: CitationPreviewDocument = {
    preview_type: "citation",
    citation_number: citation.citation_num,
    document_id: citation.document_id,
    semantic_identifier: semanticIdentifier,
    citation_chunk_ind: citation.chunk_ind,
    ...(citation.preview_url
      ? { citation_preview_url: citation.preview_url }
      : {}),
  };
  updatePresentingDocument(citationTarget);
};
