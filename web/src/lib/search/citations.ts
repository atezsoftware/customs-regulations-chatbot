import type { StreamingCitation } from "@/app/app/services/streamingModels";
import type { OnyxDocument } from "@/lib/search/interfaces";
import { ValidSources } from "@/lib/types";

/** Resolve exact targets; a file-only legacy reference must never choose among chunks. */
export function getCitationDocument(
  citation: StreamingCitation,
  documents: OnyxDocument[],
  fallbackLink?: string
): OnyxDocument {
  const link = citation.link ?? fallbackLink ?? "";
  const candidates = documents.filter(
    (document) =>
      document.document_id === citation.document_id &&
      (!citation.chunk_id ||
        document.metadata.regulatory_chunk_id === citation.chunk_id)
  );
  const matched =
    citation.chunk_ind == null
      ? candidates.length === 1
        ? candidates[0]
        : undefined
      : candidates.find(
          (document) => document.chunk_ind === citation.chunk_ind
        );
  return {
    document_id: citation.document_id,
    semantic_identifier:
      citation.semantic_identifier ?? `Source ${citation.citation_num}`,
    link,
    source_type: citation.source_type ?? ValidSources.NotApplicable,
    blurb: "",
    boost: 0,
    hidden: false,
    score: 0,
    chunk_ind: citation.chunk_ind ?? 0,
    match_highlights: [],
    metadata: {},
    updated_at: null,
    is_internet:
      citation.resolution_status === "external" ||
      (citation.source_type === ValidSources.Web &&
        /^https?:\/\//i.test(link) &&
        citation.resolution_status == null &&
        !citation.chunk_id &&
        !citation.preview_url &&
        citation.text == null),
    ...matched,
    ...(citation.link != null ? { link: citation.link } : {}),
  };
}

export function getCitationReference(
  citation: StreamingCitation,
  document: OnyxDocument
): string | undefined {
  const heading =
    citation.reference ?? document.metadata.regulatory_heading_path;
  const parts = Array.isArray(heading)
    ? heading
    : typeof heading === "string"
      ? heading.split(" > ")
      : (document.semantic_identifier?.split(/\s[—·]\s/).slice(1) ?? []);
  const articleIndex = parts.reduce(
    (last, part, index) => (/\bMADDE\s+\d+/i.test(part) ? index : last),
    -1
  );
  return (
    (articleIndex >= 0 ? parts.slice(articleIndex) : parts).join(" > ") ||
    undefined
  );
}
